from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data_raw"
INTERMEDIATE_DIR = ROOT / "data_intermediate"
ANALYTIC_DIR = ROOT / "data_analytic"
REPORTS_DIR = ROOT / "reports"
EDA_UNIFIED_REPORT_PATH = REPORTS_DIR / "eda_unificado.xlsx"
LOGS_DIR = ROOT / "logs"
LOG_FILE = LOGS_DIR / "pipeline.log"

MISSING_TOKENS = {"", " ", "NA", "N/A", "NULL", "NONE", "NAN", "NaN", "na", "n/a"}


def ensure_dirs() -> None:
    for p in [RAW_DIR, INTERMEDIATE_DIR, ANALYTIC_DIR, REPORTS_DIR, LOGS_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def setup_logger(name: str) -> logging.Logger:
    ensure_dirs()
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    fh = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


def _make_unique_columns(columns: pd.Index) -> pd.Index:
    seen: dict[str, int] = {}
    unique: list[str] = []

    for col in columns.astype(str):
        n = seen.get(col, 0)
        if n == 0:
            unique.append(col)
        else:
            unique.append(f"{col}__dup{n}")
        seen[col] = n + 1

    return pd.Index(unique)


def build_group_prefixed_columns(group_row: pd.Series, variable_row: pd.Series) -> pd.Index:
    group_filled = group_row.ffill()
    out: list[str] = []

    for i, (group, variable) in enumerate(zip(group_filled, variable_row), start=1):
        group_txt = "uncategorized" if pd.isna(group) else str(group).strip()
        variable_txt = "" if pd.isna(variable) else str(variable).strip()

        if not variable_txt:
            variable_txt = f"unnamed_col_{i}"

        out.append(f"{group_txt}__{variable_txt}")

    return pd.Index(out)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    def _normalize_column_name(col: object) -> str:
        raw = str(col).strip().lower()
        parts = raw.split("__")
        normalized_parts: list[str] = []

        for idx, part in enumerate(parts, start=1):
            clean = re.sub(r"[^0-9a-zA-Z]+", "_", part).strip("_")
            if not clean:
                clean = f"unnamed_part_{idx}"
            normalized_parts.append(clean)

        return "__".join(normalized_parts)

    cols = pd.Index([_normalize_column_name(col) for col in df.columns])
    out = df.copy()
    out.columns = _make_unique_columns(cols)
    return out


def replace_empty_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_object_dtype(out[col]):
            out[col] = out[col].astype(str).str.strip()
            out[col] = out[col].replace(list(MISSING_TOKENS), np.nan)
    return out


def parse_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def _resolve_column_name(base_col: str) -> str | None:
        if base_col in out.columns:
            return base_col
        candidates = [c for c in out.columns if str(c).endswith(f"__{base_col}")]
        if not candidates:
            return None
        non_dup = [c for c in candidates if "__dup" not in str(c)]
        return non_dup[0] if non_dup else candidates[0]

    visit_date_col = _resolve_column_name("visit_date")
    time_col = _resolve_column_name("time_24_hour")

    if visit_date_col:
        out["visit_date"] = pd.to_datetime(out[visit_date_col], errors="coerce")

    if time_col:
        t = pd.to_datetime(out[time_col], errors="coerce", format="mixed")
        out["time_24_hour"] = t.dt.strftime("%H:%M:%S")
        out.loc[t.isna(), "time_24_hour"] = np.nan

    if {"visit_date", "time_24_hour"}.issubset(out.columns):
        dt = out["visit_date"].dt.strftime("%Y-%m-%d") + " " + out["time_24_hour"].fillna("00:00:00")
        out["visit_datetime"] = pd.to_datetime(dt, errors="coerce")
    elif "visit_date" in out.columns:
        out["visit_datetime"] = out["visit_date"]
    else:
        out["visit_datetime"] = pd.NaT

    return out


def save_parquet_and_csv(df: pd.DataFrame, base_path: Path, logger: logging.Logger) -> None:
    df.to_parquet(base_path.with_suffix(".parquet"), index=False)
    df.to_csv(base_path.with_suffix(".csv"), index=False)
    logger.info("Saved %s.{parquet,csv} rows=%d cols=%d", base_path, len(df), len(df.columns))


def _normalize_excel_sheet_name(sheet_name: str) -> str:
    clean = re.sub(r"[\[\]:*?/\\]+", "_", str(sheet_name).strip())
    clean = re.sub(r"\s+", " ", clean).strip()
    if not clean:
        clean = "sheet"
    return clean[:31]


def _make_unique_sheet_names(sheet_names: Iterable[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    seen: dict[str, int] = {}

    for original_name in sheet_names:
        base = _normalize_excel_sheet_name(original_name)
        idx = seen.get(base, 0)
        if idx == 0 and base not in mapping.values():
            unique = base
        else:
            while True:
                idx += 1
                suffix = f"_{idx}"
                trim_len = max(1, 31 - len(suffix))
                candidate = f"{base[:trim_len]}{suffix}"
                if candidate not in mapping.values():
                    unique = candidate
                    break
        seen[base] = idx
        mapping[str(original_name)] = unique

    return mapping


def upsert_eda_sheets_xlsx(
    workbook_path: Path | str = EDA_UNIFIED_REPORT_PATH,
    sheets_dict: dict[str, pd.DataFrame] | None = None,
) -> Path:
    ensure_dirs()
    workbook = Path(workbook_path)
    workbook.parent.mkdir(parents=True, exist_ok=True)

    if not sheets_dict:
        return workbook

    normalized_names = _make_unique_sheet_names(sheets_dict.keys())
    writer_mode = "a" if workbook.exists() else "w"
    writer_kwargs: dict[str, object] = {"engine": "openpyxl", "mode": writer_mode}
    if writer_mode == "a":
        writer_kwargs["if_sheet_exists"] = "replace"

    with pd.ExcelWriter(workbook, **writer_kwargs) as writer:
        for original_name, df in sheets_dict.items():
            sheet_name = normalized_names[str(original_name)]
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    return workbook




def drop_sensitive_name_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    drop_cols: list[str] = []

    for col in out.columns.astype(str):
        if (
            col in {"first_name", "last_name", "frist_name"}
            or col.endswith("__first_name")
            or col.endswith("__last_name")
            or col.endswith("__frist_name")
        ):
            drop_cols.append(col)

    if drop_cols:
        out = out.drop(columns=drop_cols)

    return out, sorted(drop_cols)


def print_script_overview(script_name: str, description: str) -> None:
    print("\n" + "=" * 84)
    print(f"{script_name} | {description}")
    print("=" * 84)


def print_step(step_number: int, message: str) -> None:
    print(f"\n  Step {step_number:02d} -> {message}")

def profile_dataframe(df: pd.DataFrame, name: str) -> pd.DataFrame:
    records = []
    n = len(df)

    for col in df.columns:
        s = df[col]
        missing_mask = s.isna()
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            normalized = s.astype(str).str.strip()
            missing_tokens_mask = normalized.str.upper().isin(MISSING_TOKENS)
            missing_mask = missing_mask | missing_tokens_mask

        non_missing = s[~missing_mask]
        missing = missing_mask.mean() * 100
        nunique = non_missing.nunique(dropna=True)
        top = non_missing.astype(str).value_counts().head(5).to_dict()

        records.append(
            {
                "dataset": name,
                "column": col,
                "dtype": str(s.dtype),
                "n_rows": n,
                "missing_pct": round(missing, 2),
                "cardinality": int(nunique),
                "is_constant": bool(nunique <= 1),
                "top_values": json.dumps(top, ensure_ascii=False),
            }
        )

    return pd.DataFrame(records)


def _normalize_text_missing_mask(series: pd.Series, include_missing_variants: bool) -> pd.Series:
    normalized = series.astype("string").str.strip()
    missing_mask = normalized.isna() | (normalized == "")
    if include_missing_variants:
        token_set = {str(token).strip().upper() for token in MISSING_TOKENS}
        missing_mask = missing_mask | normalized.str.upper().isin(token_set)
    return missing_mask


def _safe_resolve_column(df: pd.DataFrame, canonical_name: str) -> str | None:
    try:
        return resolve_canonical_column(df, canonical_name)
    except KeyError:
        return None


def build_targeted_eda_report(
    df: pd.DataFrame,
    dataset_name: str,
    include_missing_variants: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Build a compact EDA package with targeted outputs for core patient/visit variables.

    Returns a dictionary of DataFrames with the following keys:
    - summary_general
    - dob_age_metrics
    - distribution_demographics
    - distribution_interval
    - distribution_visit_date_by_interval
    - distribution_visits_per_subject
    - robust_missing
    """

    report: dict[str, pd.DataFrame] = {}
    working = df.copy()

    target_vars = [
        "patient_record_number",
        "subject_number",
        "dob",
        "age_at_visit",
        "race",
        "ethnicity",
        "sex",
        "interval_name",
        "visit_date",
    ]
    resolved_cols = {var: _safe_resolve_column(working, var) for var in target_vars}

    # 1) General summary
    patient_col = resolved_cols["patient_record_number"]
    subject_col = resolved_cols["subject_number"]
    report["summary_general"] = pd.DataFrame(
        [
            {
                "dataset": dataset_name,
                "n_rows": int(len(working)),
                "n_columns": int(working.shape[1]),
                "n_unique_patient_record_number": int(working[patient_col].nunique(dropna=True))
                if patient_col
                else pd.NA,
                "n_unique_subject_number": int(working[subject_col].nunique(dropna=True))
                if subject_col
                else pd.NA,
            }
        ]
    )

    # 2) DOB and age_at_visit metrics
    metric_rows: list[dict[str, object]] = []
    for var in ["dob", "age_at_visit"]:
        col = resolved_cols[var]
        if not col:
            metric_rows.append(
                {
                    "dataset": dataset_name,
                    "variable": var,
                    "resolved_column": pd.NA,
                    "missing_count": pd.NA,
                    "min": pd.NA,
                    "max": pd.NA,
                    "mean": pd.NA,
                    "median": pd.NA,
                    "std": pd.NA,
                    "unknown_unparseable_values": pd.NA,
                }
            )
            continue

        raw = working[col]
        if pd.api.types.is_object_dtype(raw) or pd.api.types.is_string_dtype(raw):
            raw_text = raw.astype("string").str.strip()
            missing_mask = _normalize_text_missing_mask(raw, include_missing_variants)
        else:
            raw_text = raw.astype("string")
            missing_mask = raw.isna()

        if var == "dob":
            parsed = pd.to_datetime(raw, errors="coerce")
        else:
            parsed = pd.to_numeric(raw, errors="coerce")

        unknown_mask = (~missing_mask) & parsed.isna()
        unknown_values = sorted({str(v) for v in raw_text[unknown_mask].dropna().unique().tolist()})
        valid = parsed.dropna()

        metric_rows.append(
            {
                "dataset": dataset_name,
                "variable": var,
                "resolved_column": col,
                "missing_count": int(missing_mask.sum()),
                "min": valid.min() if not valid.empty else pd.NaT if var == "dob" else pd.NA,
                "max": valid.max() if not valid.empty else pd.NaT if var == "dob" else pd.NA,
                "mean": valid.mean() if not valid.empty else pd.NaT if var == "dob" else pd.NA,
                "median": valid.median() if not valid.empty else pd.NaT if var == "dob" else pd.NA,
                "std": valid.std() if not valid.empty else pd.NaT if var == "dob" else pd.NA,
                "unknown_unparseable_values": json.dumps(unknown_values, ensure_ascii=False),
            }
        )
    report["dob_age_metrics"] = pd.DataFrame(metric_rows)

    # 3) Distributions
    demo_rows: list[dict[str, object]] = []
    for var in ["race", "ethnicity", "sex"]:
        col = resolved_cols[var]
        if not col:
            continue
        series = working[col].astype("string").str.strip()
        counts = series.fillna("(missing)").replace("", "(missing)").value_counts(dropna=False)
        total = max(int(counts.sum()), 1)
        for value, count in counts.items():
            demo_rows.append(
                {
                    "dataset": dataset_name,
                    "variable": var,
                    "value": value,
                    "count": int(count),
                    "pct": round(100 * count / total, 2),
                }
            )
    report["distribution_demographics"] = pd.DataFrame(demo_rows)

    interval_col = resolved_cols["interval_name"]
    if interval_col:
        interval_counts = (
            working[interval_col]
            .astype("string")
            .str.strip()
            .fillna("(missing)")
            .replace("", "(missing)")
            .value_counts(dropna=False)
            .rename_axis("interval_name")
            .reset_index(name="count")
        )
        interval_counts.insert(0, "dataset", dataset_name)
        report["distribution_interval"] = interval_counts
    else:
        report["distribution_interval"] = pd.DataFrame(columns=["dataset", "interval_name", "count"])

    visit_date_col = resolved_cols["visit_date"]
    if interval_col and visit_date_col:
        visit_dates = pd.to_datetime(working[visit_date_col], errors="coerce").dt.date
        by_interval_date = (
            pd.DataFrame(
                {
                    "interval_name": working[interval_col].astype("string").str.strip(),
                    "visit_date": visit_dates,
                }
            )
            .assign(
                interval_name=lambda d: d["interval_name"].fillna("(missing)").replace("", "(missing)"),
                visit_date=lambda d: d["visit_date"].astype("string").fillna("(unparseable/missing)"),
            )
            .value_counts(["interval_name", "visit_date"])
            .rename("count")
            .reset_index()
            .sort_values(["interval_name", "visit_date"])
        )
        by_interval_date.insert(0, "dataset", dataset_name)
        report["distribution_visit_date_by_interval"] = by_interval_date
    else:
        report["distribution_visit_date_by_interval"] = pd.DataFrame(
            columns=["dataset", "interval_name", "visit_date", "count"]
        )

    if subject_col:
        visits_per_subject = (
            working.groupby(subject_col, dropna=False)
            .size()
            .rename("n_visits")
            .reset_index()
            .assign(subject_number=lambda d: d[subject_col].astype("string").fillna("(missing)").str.strip())
        )
        distribution_visits = (
            visits_per_subject.groupby("n_visits")
            .size()
            .rename("n_subjects")
            .reset_index()
            .sort_values("n_visits")
        )
        distribution_visits.insert(0, "dataset", dataset_name)
        report["distribution_visits_per_subject"] = distribution_visits
    else:
        report["distribution_visits_per_subject"] = pd.DataFrame(columns=["dataset", "n_visits", "n_subjects"])

    # 4) Robust missingness for target variables
    missing_rows: list[dict[str, object]] = []
    for var in target_vars:
        col = resolved_cols[var]
        if not col:
            missing_rows.append(
                {
                    "dataset": dataset_name,
                    "variable": var,
                    "resolved_column": pd.NA,
                    "missing_count": pd.NA,
                    "missing_pct": pd.NA,
                    "problematic_values": pd.NA,
                }
            )
            continue

        s = working[col]
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            text = s.astype("string").str.strip()
            missing_mask = _normalize_text_missing_mask(s, include_missing_variants)
            problematic = sorted({str(v) for v in text[missing_mask & text.notna()].unique().tolist()})
        else:
            missing_mask = s.isna()
            problematic = []

        missing_rows.append(
            {
                "dataset": dataset_name,
                "variable": var,
                "resolved_column": col,
                "missing_count": int(missing_mask.sum()),
                "missing_pct": round(float(missing_mask.mean() * 100), 2),
                "problematic_values": json.dumps(problematic, ensure_ascii=False),
            }
        )

    report["robust_missing"] = pd.DataFrame(missing_rows)
    return report


def build_target_columns_status(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    target_vars = [
        "patient_record_number",
        "subject_number",
        "dob",
        "age_at_visit",
        "race",
        "ethnicity",
        "sex",
        "interval_name",
        "visit_date",
    ]
    rows: list[dict[str, object]] = []
    for var in target_vars:
        resolved = _safe_resolve_column(df, var)
        rows.append(
            {
                "dataset": dataset_name,
                "variable": var,
                "resolved_column": resolved if resolved else pd.NA,
                "status": "ok" if resolved else "columna ausente",
            }
        )
    return pd.DataFrame(rows)


def build_input_baseline_summary(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": dataset_name,
                "n_rows": int(len(df)),
                "n_columns": int(df.shape[1]),
                "column_names": json.dumps([str(c) for c in df.columns.tolist()], ensure_ascii=False),
            }
        ]
    )


def build_targeted_eda_sheets(df: pd.DataFrame, dataset_name: str, sheet_prefix: str) -> dict[str, pd.DataFrame]:
    report = build_targeted_eda_report(df=df, dataset_name=dataset_name, include_missing_variants=True)
    status = build_target_columns_status(df=df, dataset_name=dataset_name)
    sheets: dict[str, pd.DataFrame] = {
        f"{sheet_prefix}_target_columns_status": status,
    }
    sheets.update({f"{sheet_prefix}_{key}": val for key, val in report.items()})
    return sheets


def print_kv(title: str, kv: dict[str, object]) -> None:
    print(f"\n=== {title} ===")
    for k, v in kv.items():
        print(f"- {k}: {v}")


def required_columns_check(df: pd.DataFrame, cols: Iterable[str], logger: logging.Logger, dataset: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        logger.warning("%s missing required columns: %s", dataset, missing)


def resolve_canonical_column(df: pd.DataFrame, canonical_name: str) -> str:
    if canonical_name in df.columns:
        return canonical_name

    score_map: dict[str, int] = {}
    for col in df.columns.astype(str):
        if col.endswith(f"__{canonical_name}"):
            score_map[col] = max(score_map.get(col, 0), 90)
        if col.endswith(f"_{canonical_name}"):
            score_map[col] = max(score_map.get(col, 0), 80)
        if col.endswith(canonical_name):
            score_map[col] = max(score_map.get(col, 0), 70)
        if canonical_name in col:
            score_map[col] = max(score_map.get(col, 0), 10)

    if not score_map:
        raise KeyError(
            f"Could not identify canonical column '{canonical_name}'. "
            f"Available columns: {list(df.columns)}"
        )

    best_score = max(score_map.values())
    best = [c for c, s in score_map.items() if s == best_score]
    # Deterministic tie-breaker: shortest name first (usually closest to canonical), then alphabetical.
    return sorted(best, key=lambda c: (len(c), c))[0]
