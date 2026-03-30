from __future__ import annotations

import argparse
from dataclasses import dataclass

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

from common import (
    ANALYTIC_DIR,
    EDA_UNIFIED_REPORT_PATH,
    MISSING_TOKENS,
    REPORTS_DIR,
    build_targeted_eda_sheets,
    merge_sheet_dicts,
    print_kv,
    print_script_overview,
    print_step,
    resolve_canonical_column,
    setup_logger,
    upsert_eda_sheets_xlsx,
)

MISSING_TOKEN_UPPER = {str(token).upper() for token in MISSING_TOKENS}


@dataclass(frozen=True)
class CollapseConfig:
    collapse: bool


def _normalize_yes_no(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"yes", "y", "true", "1"}:
        return True
    if normalized in {"no", "n", "false", "0"}:
        return False
    raise argparse.ArgumentTypeError("collapse must be one of: yes/no, true/false, 1/0")


def _parse_args() -> CollapseConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Audits repeated measures inside PATIENT_RECORD_NUMBER x INTERVAL_NAME and "
            "optionally collapses them into one row per patient-interval."
        )
    )
    parser.add_argument(
        "--collapse",
        default="yes",
        type=_normalize_yes_no,
        help="Set to yes/no to save a collapsed table (default: yes).",
    )
    args = parser.parse_args()
    return CollapseConfig(collapse=bool(args.collapse))


def _resolve_columns(visits: pd.DataFrame) -> tuple[str, str, str]:
    subject_col = resolve_canonical_column(visits, "patient_record_number")
    interval_col = resolve_canonical_column(visits, "interval_name")

    try:
        visit_date_col = resolve_canonical_column(visits, "visit_date")
    except KeyError:
        visit_datetime_col = resolve_canonical_column(visits, "visit_datetime")
        visits["visit_date"] = pd.to_datetime(visits[visit_datetime_col], errors="coerce").dt.normalize()
        visit_date_col = "visit_date"

    return subject_col, interval_col, visit_date_col


def _normalize_missing_values(series: pd.Series) -> pd.Series:
    out = series.copy()
    if pd.api.types.is_object_dtype(out) or pd.api.types.is_string_dtype(out):
        str_values = out.astype("string").str.strip()
        missing_mask = str_values.isna() | str_values.str.upper().isin(MISSING_TOKEN_UPPER)
        str_values = str_values.mask(missing_mask, pd.NA)
        return str_values
    return out


def _collapse_column(series: pd.Series):
    normalized = _normalize_missing_values(series)
    non_missing = normalized.dropna()

    if non_missing.empty:
        return pd.NA

    if is_numeric_dtype(non_missing):
        return non_missing.max()

    if is_datetime64_any_dtype(non_missing):
        return non_missing.max()

    unique_values = sorted({str(v).strip() for v in non_missing if pd.notna(v)})
    if len(unique_values) == 1:
        return unique_values[0]

    # Explicit conflict representation (e.g., "yes | no").
    return " | ".join(unique_values)


def _build_variable_audit(
    visits: pd.DataFrame,
    group_cols: list[str],
    excluded_cols: set[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_size = visits.groupby(group_cols).size().rename("n_rows_group")
    repeated_mask = group_size > 1

    for col in visits.columns:
        if col in excluded_cols:
            continue

        normalized_col = _normalize_missing_values(visits[col])
        grouped = normalized_col.groupby([visits[g] for g in group_cols])
        non_missing_count = grouped.apply(lambda s: s.notna().sum())
        distinct_non_missing = grouped.apply(lambda s: s.dropna().nunique())

        affected_groups = int((repeated_mask & (non_missing_count > 0)).sum())
        conflict_groups = int((repeated_mask & (distinct_non_missing > 1)).sum())
        complementary_groups = int((repeated_mask & (non_missing_count > 0) & (distinct_non_missing == 1)).sum())

        rows.append(
            {
                "variable": col,
                "groups_with_repeated_rows": int(repeated_mask.sum()),
                "affected_groups": affected_groups,
                "complementary_groups": complementary_groups,
                "conflict_groups": conflict_groups,
                "pct_conflict_among_affected": round(
                    (100 * conflict_groups / affected_groups) if affected_groups else 0.0,
                    2,
                ),
                "pct_complementary_among_affected": round(
                    (100 * complementary_groups / affected_groups) if affected_groups else 0.0,
                    2,
                ),
            }
        )

    audit = pd.DataFrame(rows).sort_values(
        ["conflict_groups", "complementary_groups", "affected_groups"],
        ascending=False,
    )
    return audit


def _build_conflict_examples(
    visits: pd.DataFrame,
    group_cols: list[str],
    excluded_cols: set[str],
    max_examples_per_variable: int = 25,
) -> pd.DataFrame:
    examples: list[dict[str, object]] = []

    repeated_groups = visits.groupby(group_cols).size().rename("n_rows")
    repeated_groups = repeated_groups[repeated_groups > 1]
    if repeated_groups.empty:
        return pd.DataFrame(columns=[*group_cols, "variable", "observed_values", "collapsed_value"])

    repeated_index = set(repeated_groups.index)

    for col in visits.columns:
        if col in excluded_cols:
            continue

        sample_n = 0
        for keys, group in visits.groupby(group_cols, sort=False):
            if keys not in repeated_index:
                continue

            normalized = _normalize_missing_values(group[col])
            unique_values = sorted({str(v).strip() for v in normalized.dropna() if pd.notna(v)})
            if len(unique_values) <= 1:
                continue

            record: dict[str, object] = {}
            if not isinstance(keys, tuple):
                keys = (keys,)
            for idx, key_col in enumerate(group_cols):
                record[key_col] = keys[idx]

            record["variable"] = col
            record["observed_values"] = " | ".join(unique_values)
            record["collapsed_value"] = _collapse_column(group[col])
            examples.append(record)

            sample_n += 1
            if sample_n >= max_examples_per_variable:
                break

    if not examples:
        return pd.DataFrame(columns=[*group_cols, "variable", "observed_values", "collapsed_value"])

    return pd.DataFrame(examples)


def main() -> None:
    config = _parse_args()
    logger = setup_logger("09_interval_collapse_audit")

    print_script_overview(
        "09_interval_collapse_audit.py",
        "Audits repeated measurements by interval and optionally collapses to one row per patient-interval.",
    )

    print_step(1, "Load visits_long")
    visits = pd.read_parquet(ANALYTIC_DIR / "visits_long.parquet")

    print_step(2, "Resolve canonical columns and prepare grouping keys")
    subject_col, interval_col, visit_date_col = _resolve_columns(visits)
    visits[interval_col] = visits[interval_col].astype("string").fillna("(missing)").str.strip()
    visits[visit_date_col] = pd.to_datetime(visits[visit_date_col], errors="coerce")

    group_cols = [subject_col, interval_col]
    repeated_groups = visits.groupby(group_cols).size().rename("n_rows").reset_index()
    repeated_groups = repeated_groups[repeated_groups["n_rows"] > 1].copy()

    print_step(3, "Measure temporal window distribution inside patient-interval groups")
    window_stats = (
        visits.groupby(group_cols, as_index=False)
        .agg(
            n_rows_group=(subject_col, "size"),
            min_date=(visit_date_col, "min"),
            max_date=(visit_date_col, "max"),
        )
    )
    window_stats["window_days"] = (window_stats["max_date"] - window_stats["min_date"]).dt.days

    repeated_window = window_stats[window_stats["n_rows_group"] > 1].copy()
    repeated_window_summary = pd.DataFrame(
        [
            {
                "metric": "repeated_groups",
                "value": int(len(repeated_window)),
            },
            {
                "metric": "window_days_p50",
                "value": float(repeated_window["window_days"].median()) if not repeated_window.empty else 0.0,
            },
            {
                "metric": "window_days_p90",
                "value": float(repeated_window["window_days"].quantile(0.9)) if not repeated_window.empty else 0.0,
            },
            {
                "metric": "window_days_max",
                "value": float(repeated_window["window_days"].max()) if not repeated_window.empty else 0.0,
            },
        ]
    )

    print_step(4, "Audit complementarity vs conflicts per variable")
    excluded_cols = {subject_col, interval_col}
    variable_audit = _build_variable_audit(visits, group_cols, excluded_cols)
    conflict_examples = _build_conflict_examples(visits, group_cols, excluded_cols)

    print_step(5, "Export reports")
    repeated_groups.to_csv(REPORTS_DIR / "interval_collapse_repeated_groups.csv", index=False)
    window_stats.to_csv(REPORTS_DIR / "interval_collapse_window_stats.csv", index=False)
    repeated_window_summary.to_csv(REPORTS_DIR / "interval_collapse_window_summary.csv", index=False)
    variable_audit.to_csv(REPORTS_DIR / "interval_collapse_variable_audit.csv", index=False)
    conflict_examples.to_csv(REPORTS_DIR / "interval_collapse_conflict_examples.csv", index=False)

    if config.collapse:
        print_step(6, "Collapse to one row per patient-interval and save")
        sort_cols = [subject_col, interval_col, visit_date_col]
        sort_cols = [c for c in sort_cols if c in visits.columns]
        sorted_visits = visits.sort_values(sort_cols)

        aggregated = {
            col: _collapse_column
            for col in sorted_visits.columns
            if col not in group_cols
        }
        collapsed = sorted_visits.groupby(group_cols, as_index=False).agg(aggregated)
        collapsed.to_parquet(ANALYTIC_DIR / "visits_long_collapsed_by_interval.parquet", index=False)
        collapsed.to_csv(ANALYTIC_DIR / "visits_long_collapsed_by_interval.csv", index=False)
        logger.info("Saved collapsed visits to data_analytic/visits_long_collapsed_by_interval.{parquet,csv}")
    else:
        collapsed = None

    metrics = {
        "rows_original": len(visits),
        "groups_subject_interval": visits.groupby(group_cols).ngroups,
        "groups_with_repeated_rows": len(repeated_groups),
        "collapse_requested": config.collapse,
    }
    print_kv("Interval collapse audit", metrics)
    print_step(7, "Append targeted EDA for visits/collapsed outputs to unified workbook")
    sheets = {}
    sheets = merge_sheet_dicts(
        sheets,
        build_targeted_eda_sheets(visits, "09_visits_long_output", "09_visits_long_output", consolidated=True),
    )
    if collapsed is not None:
        sheets = merge_sheet_dicts(
            sheets,
            build_targeted_eda_sheets(
                collapsed,
                "09_collapsed_by_interval_output",
                "09_collapsed_by_interval_output",
                consolidated=True,
            ),
        )
    workbook = upsert_eda_sheets_xlsx(EDA_UNIFIED_REPORT_PATH, sheets)
    logger.info("Updated unified EDA workbook: %s", workbook)

    logger.info(
        "Saved reports: interval_collapse_{repeated_groups,window_stats,window_summary,variable_audit,conflict_examples}.csv"
    )


if __name__ == "__main__":
    main()
