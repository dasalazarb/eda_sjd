from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from common import ANALYTIC_DIR, REPORTS_DIR, print_kv, print_script_overview, print_step, resolve_canonical_column, setup_logger


@dataclass(frozen=True)
class AuditConfig:
    input_path: Path
    output_dir: Path
    min_obs_for_longitudinal: int


KEY_CANONICALS = {
    "subject_number",
    "patient_record_number",
    "interval_name",
    "visit_date",
    "visit_datetime",
    "source_protocol",
    "source_file",
    "row_id_raw",
}


def _parse_args() -> AuditConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Compute longitudinal plausibility metrics per variable for the collapsed visits dataset "
            "and produce an ML-readiness classification."
        )
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=ANALYTIC_DIR / "visits_long_collapsed_by_interval_codebook_corrected.parquet",
        help=(
            "Path to source table (.parquet/.csv/.xlsx). "
            "Default: data_analytic/visits_long_collapsed_by_interval_codebook_corrected.parquet"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPORTS_DIR / "longitudinal_plausibility",
        help="Directory for outputs (default: reports/longitudinal_plausibility).",
    )
    parser.add_argument(
        "--min-obs-for-longitudinal",
        type=int,
        default=2,
        help="Minimum number of observations per patient used as an expected threshold (default: 2).",
    )
    args = parser.parse_args()
    return AuditConfig(
        input_path=args.input_path,
        output_dir=args.output_dir,
        min_obs_for_longitudinal=max(1, args.min_obs_for_longitudinal),
    )


def _load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls", ".xlsm"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported input format: {path}")


def _resolve_subject_col(df: pd.DataFrame) -> str:
    try:
        return resolve_canonical_column(df, "subject_number")
    except KeyError:
        return resolve_canonical_column(df, "patient_record_number")


def _resolve_visit_time_col(df: pd.DataFrame) -> str:
    try:
        col = resolve_canonical_column(df, "visit_datetime")
        df[col] = pd.to_datetime(df[col], errors="coerce")
        return col
    except KeyError:
        col = resolve_canonical_column(df, "visit_date")
        df[col] = pd.to_datetime(df[col], errors="coerce")
        return col


def _build_base_order(df: pd.DataFrame, subject_col: str, interval_col: str, visit_time_col: str) -> pd.DataFrame:
    base = df[[subject_col, interval_col, visit_time_col]].copy()
    base[visit_time_col] = pd.to_datetime(base[visit_time_col], errors="coerce")
    base["row_order"] = np.arange(len(base), dtype=int)
    base = base.sort_values([subject_col, visit_time_col, "row_order"], kind="stable")
    base["visit_index"] = base.groupby(subject_col).cumcount() + 1
    return base


def _non_missing_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        txt = series.astype("string").str.strip()
        return txt.notna() & (txt != "")
    return series.notna()


def _variable_type(series: pd.Series, name: str) -> str:
    non_missing = series[_non_missing_mask(series)]
    if non_missing.empty:
        return "categorical"

    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    if pd.api.types.is_bool_dtype(series):
        return "boolean"

    if pd.api.types.is_numeric_dtype(series):
        return "numeric"

    lower_name = name.lower()
    invariant_tokens = ["sex", "gender", "dob", "birth", "ethnicity", "race", "blood_type"]
    if any(tok in lower_name for tok in invariant_tokens):
        return "invariant"

    txt = non_missing.astype("string").str.strip()
    txt_lower = txt.str.lower()

    # Boolean-like strings (yes/no, true/false, 1/0, etc.).
    boolean_tokens = {"true", "false", "yes", "no", "y", "n", "si", "sí", "0", "1", "t", "f"}
    unique_tokens = set(txt_lower.dropna().unique().tolist())
    if unique_tokens and unique_tokens.issubset(boolean_tokens):
        return "boolean"

    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        parsed_dates = pd.to_datetime(non_missing, errors="coerce")
        parsed_ratio = float(parsed_dates.notna().mean()) if len(non_missing) else 0.0
        if parsed_ratio >= 0.9:
            return "datetime"

    numeric_parsed = pd.to_numeric(non_missing, errors="coerce")
    numeric_ratio = float(numeric_parsed.notna().mean())
    if numeric_ratio >= 0.85:
        numeric_values = numeric_parsed.dropna()
        if not numeric_values.empty:
            near_integer = np.isclose(numeric_values, np.round(numeric_values))
            decimal_ratio = float((~near_integer).mean())
            unique_numeric = int(numeric_values.nunique(dropna=True))
            min_v, max_v = float(numeric_values.min()), float(numeric_values.max())
            range_v = max_v - min_v

            # Avoid marking tiny coded categories (e.g., 0..4 only) as numeric.
            looks_like_small_code_set = (
                unique_numeric <= 6
                and decimal_ratio < 0.05
                and min_v >= 0
                and max_v <= 10
            )

            has_continuous_signal = (decimal_ratio >= 0.05) or (unique_numeric >= 8 and range_v > 10)
            if has_continuous_signal and not looks_like_small_code_set:
                return "numeric"

    cardinality = txt.nunique(dropna=True)
    if cardinality <= 2:
        return "boolean"
    return "categorical"


def _adjacent_pair_stats(work: pd.DataFrame, subject_col: str, value_col: str) -> tuple[int, int]:
    tmp = work[[subject_col, "visit_index", value_col]].copy()
    tmp["has_value"] = _non_missing_mask(tmp[value_col])
    tmp["has_prev"] = tmp.groupby(subject_col)["has_value"].shift(1).fillna(False)
    valid_pairs = int((tmp["has_value"] & tmp["has_prev"]).sum())
    total_pairs = int(max(len(tmp) - tmp[subject_col].nunique(), 0))
    return valid_pairs, total_pairs


def _numeric_metrics(work: pd.DataFrame, subject_col: str, value_col: str, visit_time_col: str) -> dict[str, object]:
    tmp = work[[subject_col, visit_time_col, "visit_index", value_col]].copy()
    tmp["x"] = pd.to_numeric(tmp[value_col], errors="coerce")
    tmp = tmp.dropna(subset=["x", visit_time_col])
    tmp = tmp.sort_values([subject_col, "visit_index"], kind="stable")

    tmp["x_prev"] = tmp.groupby(subject_col)["x"].shift(1)
    tmp["t_prev"] = tmp.groupby(subject_col)[visit_time_col].shift(1)
    tmp["delta_abs"] = (tmp["x"] - tmp["x_prev"]).abs()
    days = (tmp[visit_time_col] - tmp["t_prev"]).dt.total_seconds() / 86400.0
    tmp["delta_annualized"] = np.where(days > 0, tmp["delta_abs"] * (365.25 / days), np.nan)

    deltas = tmp["delta_abs"].dropna()
    if deltas.empty:
        return {
            "n_consecutive_pairs": 0,
            "delta_abs_p50": np.nan,
            "delta_abs_p95": np.nan,
            "delta_annualized_p50": np.nan,
            "extreme_change_rate": np.nan,
            "delta_outlier_rate": np.nan,
            "consistency_score": np.nan,
        }

    p95 = float(deltas.quantile(0.95))
    q1, q3 = float(deltas.quantile(0.25)), float(deltas.quantile(0.75))
    iqr = q3 - q1
    high_thr = q3 + 3 * iqr
    extreme_rate = float((deltas > p95).mean())
    outlier_rate = float((deltas > high_thr).mean()) if iqr > 0 else 0.0

    return {
        "n_consecutive_pairs": int(len(deltas)),
        "delta_abs_p50": float(deltas.quantile(0.5)),
        "delta_abs_p95": p95,
        "delta_annualized_p50": float(tmp["delta_annualized"].dropna().quantile(0.5)) if tmp["delta_annualized"].notna().any() else np.nan,
        "extreme_change_rate": round(100 * extreme_rate, 2),
        "delta_outlier_rate": round(100 * outlier_rate, 2),
        "consistency_score": round(100 * (1 - outlier_rate), 2),
    }


def _categorical_metrics(work: pd.DataFrame, subject_col: str, value_col: str) -> dict[str, object]:
    tmp = work[[subject_col, "visit_index", value_col]].copy()
    tmp["v"] = tmp[value_col].astype("string").str.strip()
    tmp = tmp[(tmp["v"].notna()) & (tmp["v"] != "")].copy()
    tmp = tmp.sort_values([subject_col, "visit_index"], kind="stable")

    tmp["v_prev"] = tmp.groupby(subject_col)["v"].shift(1)
    pairs = tmp[tmp["v_prev"].notna()].copy()
    if pairs.empty:
        return {
            "n_consecutive_pairs": 0,
            "change_rate": np.nan,
            "flip_rate": np.nan,
            "reversion_rate": np.nan,
            "contradiction_rate": np.nan,
            "consistency_score": np.nan,
        }

    pairs["changed"] = pairs["v"] != pairs["v_prev"]

    # Flip = A->B->A patterns
    tmp["v_prev2"] = tmp.groupby(subject_col)["v"].shift(2)
    tmp["is_flip"] = (tmp["v_prev2"].notna()) & (tmp["v"] == tmp["v_prev2"]) & (tmp["v"] != tmp["v_prev"])

    # Contradiction heuristic for binary style values
    contradiction_tokens = {
        frozenset(["yes", "no"]),
        frozenset(["present", "absent"]),
        frozenset(["positive", "negative"]),
        frozenset(["male", "female"]),
        frozenset(["true", "false"]),
        frozenset(["1", "0"]),
    }

    def _is_contradict(a: str, b: str) -> bool:
        pair = frozenset([str(a).lower(), str(b).lower()])
        return pair in contradiction_tokens

    pairs["is_contradiction"] = [_is_contradict(a, b) for a, b in zip(pairs["v_prev"], pairs["v"])]

    return {
        "n_consecutive_pairs": int(len(pairs)),
        "change_rate": round(100 * pairs["changed"].mean(), 2),
        "flip_rate": round(100 * tmp["is_flip"].mean(), 2),
        "reversion_rate": round(100 * tmp["is_flip"].mean(), 2),
        "contradiction_rate": round(100 * pairs["is_contradiction"].mean(), 2),
        "consistency_score": round(100 * (1 - pairs["changed"].mean()), 2),
    }


def _datetime_metrics(work: pd.DataFrame, subject_col: str, value_col: str) -> dict[str, object]:
    tmp = work[[subject_col, "visit_index", value_col]].copy()
    tmp["dt"] = pd.to_datetime(tmp[value_col], errors="coerce")
    tmp = tmp.dropna(subset=["dt"]).sort_values([subject_col, "visit_index"], kind="stable")

    tmp["dt_prev"] = tmp.groupby(subject_col)["dt"].shift(1)
    delta_days = (tmp["dt"] - tmp["dt_prev"]).dt.total_seconds().abs() / 86400.0
    delta_days = delta_days.dropna()

    if delta_days.empty:
        return {
            "n_consecutive_pairs": 0,
            "delta_abs_p50": np.nan,
            "delta_abs_p95": np.nan,
            "extreme_change_rate": np.nan,
            "delta_outlier_rate": np.nan,
            "consistency_score": np.nan,
        }

    p95 = float(delta_days.quantile(0.95))
    q1, q3 = float(delta_days.quantile(0.25)), float(delta_days.quantile(0.75))
    iqr = q3 - q1
    high_thr = q3 + 3 * iqr
    extreme_rate = float((delta_days > p95).mean())
    outlier_rate = float((delta_days > high_thr).mean()) if iqr > 0 else 0.0

    return {
        "n_consecutive_pairs": int(len(delta_days)),
        "delta_abs_p50": float(delta_days.quantile(0.5)),
        "delta_abs_p95": p95,
        "extreme_change_rate": round(100 * extreme_rate, 2),
        "delta_outlier_rate": round(100 * outlier_rate, 2),
        "consistency_score": round(100 * (1 - outlier_rate), 2),
    }


def _invariant_metrics(work: pd.DataFrame, subject_col: str, value_col: str) -> dict[str, object]:
    tmp = work[[subject_col, value_col]].copy()
    tmp = tmp[_non_missing_mask(tmp[value_col])].copy()
    if tmp.empty:
        return {"intra_patient_discordance_rate": np.nan, "consistency_score": np.nan}

    per_patient = tmp.groupby(subject_col)[value_col].nunique(dropna=True)
    discordance_rate = float((per_patient > 1).mean()) if len(per_patient) else np.nan
    return {
        "intra_patient_discordance_rate": round(100 * discordance_rate, 2),
        "consistency_score": round(100 * (1 - discordance_rate), 2),
    }


def _missingness_bias(work: pd.DataFrame, interval_col: str, value_col: str) -> float:
    tmp = work[[interval_col, value_col]].copy()
    tmp["is_missing"] = ~_non_missing_mask(tmp[value_col])
    rates = tmp.groupby(interval_col)["is_missing"].mean()
    if rates.empty:
        return np.nan
    return round(100 * float(rates.max() - rates.min()), 2)


def _classify(row: pd.Series) -> str:
    has_cov = row["pct_patients_ge1"] >= 40
    has_repeat = row["pct_patients_ge2"] >= 20
    multi_visit = row["observed_interval_count"] >= 2
    low_temporal_issues = (pd.isna(row["temporal_violation_rate"]) or row["temporal_violation_rate"] <= 10)
    low_missing_bias = (pd.isna(row["missingness_bias_by_interval_pp"]) or row["missingness_bias_by_interval_pp"] <= 30)
    high_consistency = (pd.isna(row["consistency_score"]) or row["consistency_score"] >= 70)

    score = sum([has_cov, has_repeat, multi_visit, low_temporal_issues, low_missing_bias, high_consistency])
    if score >= 6:
        return "lista para longitudinal"
    if score >= 4:
        return "usable con cautela"
    if has_cov and not has_repeat:
        return "usable solo cross-sectional"
    return "no usable todavía"


def run_audit(df: pd.DataFrame, min_obs_for_longitudinal: int = 2) -> pd.DataFrame:
    subject_col = _resolve_subject_col(df)
    interval_col = resolve_canonical_column(df, "interval_name")
    visit_time_col = _resolve_visit_time_col(df)

    work = df.copy()
    work[visit_time_col] = pd.to_datetime(work[visit_time_col], errors="coerce")
    ordered = _build_base_order(work, subject_col, interval_col, visit_time_col)
    work = work.loc[ordered.index].copy()
    work["visit_index"] = ordered["visit_index"].values

    total_patients = int(work[subject_col].nunique(dropna=True))
    total_visits = int(len(work))

    candidate_vars = [c for c in df.columns if c not in KEY_CANONICALS and not str(c).startswith("ids__")]
    # Ensure deterministic order
    candidate_vars = sorted(dict.fromkeys(candidate_vars))

    rows: list[dict[str, object]] = []
    for var in candidate_vars:
        s = work[var]
        has_value = _non_missing_mask(s)
        observed = work.loc[has_value, [subject_col, interval_col, "visit_index", visit_time_col, var]].copy()

        pats_counts = observed.groupby(subject_col).size() if not observed.empty else pd.Series(dtype=int)

        ge1 = int((pats_counts >= 1).sum())
        ge2 = int((pats_counts >= 2).sum())
        ge3 = int((pats_counts >= 3).sum())

        observed_interval_count = int(observed[interval_col].nunique(dropna=True)) if not observed.empty else 0
        multi_interval_patients = int((observed.groupby(subject_col)[interval_col].nunique() >= 2).sum()) if not observed.empty else 0

        valid_pairs, total_pairs = _adjacent_pair_stats(work[[subject_col, "visit_index", var]].copy(), subject_col, var)

        temporal_violations = 0
        if not observed.empty and observed[visit_time_col].notna().any():
            check = observed[[subject_col, "visit_index", visit_time_col]].sort_values([subject_col, "visit_index"])
            prev_t = check.groupby(subject_col)[visit_time_col].shift(1)
            temporal_violations = int((check[visit_time_col] < prev_t).sum())

        var_type = _variable_type(s, var)
        if var_type == "numeric":
            extra = _numeric_metrics(work[[subject_col, "visit_index", visit_time_col, var]].copy(), subject_col, var, visit_time_col)
        elif var_type == "datetime":
            extra = _datetime_metrics(work[[subject_col, "visit_index", var]].copy(), subject_col, var)
        elif var_type in {"categorical", "boolean"}:
            extra = _categorical_metrics(work[[subject_col, "visit_index", var]].copy(), subject_col, var)
        else:
            extra = _invariant_metrics(work[[subject_col, var]].copy(), subject_col, var)

        consistency = extra.get("consistency_score", np.nan)

        row = {
            "variable": var,
            "variable_type": var_type,
            "patients_with_ge1": ge1,
            "pct_patients_ge1": round(100 * ge1 / total_patients, 2) if total_patients else np.nan,
            "patients_with_ge2": ge2,
            "pct_patients_ge2": round(100 * ge2 / total_patients, 2) if total_patients else np.nan,
            "patients_with_ge3": ge3,
            "pct_patients_ge3": round(100 * ge3 / total_patients, 2) if total_patients else np.nan,
            "visits_covered": int(has_value.sum()),
            "pct_visits_covered": round(100 * has_value.mean(), 2) if total_visits else np.nan,
            "observed_interval_count": observed_interval_count,
            "patients_in_multiple_intervals": multi_interval_patients,
            "pct_patients_in_multiple_intervals": round(100 * multi_interval_patients / total_patients, 2) if total_patients else np.nan,
            "consecutive_pairs_with_data": valid_pairs,
            "consecutive_pair_coverage": round(100 * valid_pairs / total_pairs, 2) if total_pairs else np.nan,
            "temporal_violations": temporal_violations,
            "temporal_violation_rate": round(100 * temporal_violations / max(valid_pairs, 1), 2) if valid_pairs else np.nan,
            "n_distinct_values": int(observed[var].astype("string").nunique(dropna=True)) if not observed.empty else 0,
            "intervals_with_ge3_observed_values": int(
                (observed.groupby(interval_col)[var].apply(lambda x: x.astype("string").nunique(dropna=True)) >= 3).sum()
            ) if not observed.empty else 0,
            "missingness_bias_by_interval_pp": _missingness_bias(work[[interval_col, var]].copy(), interval_col, var),
            "minimum_obs_threshold": min_obs_for_longitudinal,
            "meets_min_obs_threshold": bool((round(100 * ge2 / total_patients, 2) if total_patients else 0) > 0) if min_obs_for_longitudinal <= 2 else bool(ge3 > 0),
            "consistency_score": consistency,
        }
        row.update(extra)
        rows.append(row)

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary

    summary["ml_longitudinal_label"] = summary.apply(_classify, axis=1)
    summary = summary.sort_values(["ml_longitudinal_label", "pct_patients_ge2", "pct_visits_covered"], ascending=[True, False, False])
    return summary


def main() -> None:
    config = _parse_args()
    logger = setup_logger("13_longitudinal_plausibility_audit")

    print_script_overview(
        "13_longitudinal_plausibility_audit.py",
        "Builds variable-level longitudinal plausibility metrics and ML-readiness labels.",
    )

    print_step(1, "Load source dataset")
    df = _load_table(config.input_path)

    print_step(2, "Compute per-variable longitudinal plausibility metrics")
    summary = run_audit(df, min_obs_for_longitudinal=config.min_obs_for_longitudinal)

    print_step(3, "Persist outputs")
    config.output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = config.output_dir / "longitudinal_variable_summary.csv"
    if summary.empty:
        summary = pd.DataFrame(columns=["variable", "ml_longitudinal_label"])
    summary.to_csv(summary_path, index=False)

    label_counts = summary["ml_longitudinal_label"].value_counts(dropna=False).rename_axis("label").reset_index(name="n_variables")
    labels_path = config.output_dir / "longitudinal_variable_label_counts.csv"
    label_counts.to_csv(labels_path, index=False)

    print_kv(
        "Longitudinal plausibility audit",
        {
            "input_path": str(config.input_path),
            "n_rows": len(df),
            "n_columns": df.shape[1],
            "n_variables_audited": len(summary),
            "summary_csv": str(summary_path),
            "label_counts_csv": str(labels_path),
        },
    )
    logger.info("Longitudinal plausibility audit completed: variables=%d", len(summary))


if __name__ == "__main__":
    main()
