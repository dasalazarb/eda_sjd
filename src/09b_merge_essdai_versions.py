from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd

from common import ANALYTIC_DIR, print_kv, print_script_overview, print_step, setup_logger

ESSDAI_PREFIX_LEGACY = "essdai-_r__"
ESSDAI_PREFIX_CANONICAL = "essdai__"
KEY_PATIENT = "ids__patient_record_number"
KEY_INTERVAL = "ids__interval_name"
KEY_VISIT_DATE = "ids__visit_date"
NH_INTERVAL = "Natural History Protocol 478 Interval"
OPT15D_PREFIX = "15D Optional Evaluation"
OPT15D_PATTERN = re.compile(rf"^{re.escape(OPT15D_PREFIX)}(?:\s*\{{?\d+\}}?)?$", re.IGNORECASE)
NATURAL_PATTERN = re.compile(r"^natural(?:\s+history.*)?$", re.IGNORECASE)
ADDITIONAL_MERGE_PAIRS = [
    ("sjogren's_syndrome_history__arthritis", "systems_review_for_physician__arthritis"),
    ("systems_review_for_physician__musculo_tndnts", "physical_examination-initial_evaluation__musculo_tndnts"),
    ("systems_review_for_physician__mouth_drynss", "physical_examination-initial_evaluation__mouth_drynss"),
]


@dataclass(frozen=True)
class MergeConfig:
    input_path: Path
    output_base: Path


def _parse_args() -> MergeConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Merge ESSDAI columns by shared suffix from legacy prefix "
            "'essdai-_r__' into canonical prefix 'essdai__', merge selected "
            "non-ESSDAI aliases, and report/drop fully empty columns."
        )
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=ANALYTIC_DIR / "visits_long_collapsed_by_interval_codebook_not_clean.parquet",
        help="Path to input table (.parquet/.csv/.xlsx).",
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=ANALYTIC_DIR / "visits_long_collapsed_by_interval_codebook_corrected",
        help="Output base path (without extension). Script writes both .parquet and .csv.",
    )
    args = parser.parse_args()
    return MergeConfig(input_path=args.input_path, output_base=args.output_base)


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


def _tokenize_cell(value: object) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    parts = [part.strip() for part in text.split("|")]
    return [part for part in parts if part]


def _merge_cell_values(values: list[object]) -> object:
    merged: list[str] = []
    seen: set[str] = set()

    for value in values:
        for token in _tokenize_cell(value):
            key = token.casefold()
            if key in seen:
                continue
            seen.add(key)
            merged.append(token)

    if not merged:
        return pd.NA
    if len(merged) == 1:
        return merged[0]
    return " | ".join(merged)


def _prefix_map(columns: list[str], prefix: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for col in columns:
        if col.startswith(prefix):
            suffix = col[len(prefix) :]
            out.setdefault(suffix, []).append(col)
    return out


def _count_non_empty(series: pd.Series) -> int:
    return int(series.map(lambda value: len(_tokenize_cell(value)) > 0).sum())


def _merge_column_group(df: pd.DataFrame, source_cols: list[str], target_col: str) -> pd.DataFrame:
    out = df.copy()
    out[target_col] = out[source_cols].apply(lambda row: _merge_cell_values(row.tolist()), axis=1)
    drop_cols = [col for col in source_cols if col != target_col]
    if drop_cols:
        out = out.drop(columns=drop_cols)
    return out


def _merge_essdai_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    out = df.copy()
    colnames = [str(c) for c in out.columns]
    legacy_map = _prefix_map(colnames, ESSDAI_PREFIX_LEGACY)
    canonical_map = _prefix_map(colnames, ESSDAI_PREFIX_CANONICAL)
    shared_suffixes = sorted(set(legacy_map).intersection(canonical_map))

    for suffix in shared_suffixes:
        source_cols = [*legacy_map[suffix], *canonical_map[suffix]]
        merged_col = f"{ESSDAI_PREFIX_CANONICAL}{suffix}"
        out = _merge_column_group(out, source_cols, merged_col)

    return out, len(shared_suffixes)


def _merge_additional_pairs(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    out = df.copy()
    merged_pairs = 0
    skipped_pairs = 0

    for left_col, right_col in ADDITIONAL_MERGE_PAIRS:
        if left_col not in out.columns or right_col not in out.columns:
            skipped_pairs += 1
            continue

        left_non_empty = _count_non_empty(out[left_col])
        right_non_empty = _count_non_empty(out[right_col])
        target_col = left_col if left_non_empty >= right_non_empty else right_col
        out = _merge_column_group(out, [left_col, right_col], target_col)
        merged_pairs += 1

    return out, merged_pairs, skipped_pairs


def _drop_fully_empty_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    empty_columns = [col for col in df.columns if _count_non_empty(df[col]) == 0]
    if not empty_columns:
        return df.copy(), []
    out = df.drop(columns=empty_columns)
    return out, empty_columns


def _normalize_string(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _patients_to_exclude(df: pd.DataFrame) -> set[str]:
    if KEY_PATIENT not in df.columns or KEY_INTERVAL not in df.columns:
        return set()

    base = pd.DataFrame(
        {
            KEY_PATIENT: df[KEY_PATIENT].map(_normalize_string),
            KEY_INTERVAL: df[KEY_INTERVAL].map(_normalize_string),
        }
    )
    base = base[base[KEY_PATIENT] != ""].copy()
    if base.empty:
        return set()

    grouped = base.groupby(KEY_PATIENT, dropna=False)[KEY_INTERVAL]

    def _is_natural_or_15d_optional(interval: str) -> bool:
        normalized_interval = " ".join(interval.split())
        if not normalized_interval:
            return False
        if normalized_interval.casefold() == NH_INTERVAL.casefold():
            return True
        if NATURAL_PATTERN.match(normalized_interval):
            return True
        return bool(OPT15D_PATTERN.match(normalized_interval))

    def _exclude_patient(intervals: pd.Series) -> bool:
        unique_non_empty = {interval for interval in intervals if interval}
        if not unique_non_empty:
            return False
        return all(_is_natural_or_15d_optional(interval) for interval in unique_non_empty)

    exclude_mask = grouped.apply(_exclude_patient)
    return set(exclude_mask[exclude_mask].index.tolist())


def _filter_patients(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if KEY_PATIENT not in df.columns or KEY_INTERVAL not in df.columns:
        return df.copy(), 0

    out = df.copy()
    out[KEY_PATIENT] = out[KEY_PATIENT].map(_normalize_string)
    excluded_patients_set = _patients_to_exclude(out)
    filtered = out[~out[KEY_PATIENT].isin(excluded_patients_set)].copy()
    excluded_patients = len(excluded_patients_set)
    return filtered, int(excluded_patients)


def _is_15d_optional(interval: object) -> bool:
    normalized_interval = " ".join(_normalize_string(interval).split())
    if not normalized_interval:
        return False
    return bool(OPT15D_PATTERN.match(normalized_interval))


def _collapse_15d_optional_same_year(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    required = {KEY_PATIENT, KEY_INTERVAL, KEY_VISIT_DATE}
    if any(column not in df.columns for column in required):
        return df.copy(), 0, 0

    work = df.copy()
    parsed_dates = pd.to_datetime(work[KEY_VISIT_DATE], errors="coerce")
    optional_mask = work[KEY_INTERVAL].map(_is_15d_optional)
    year_mask = parsed_dates.notna()
    target_mask = optional_mask & year_mask

    if not target_mask.any():
        return work, 0, 0

    targets = work.loc[target_mask].copy()
    targets["_visit_year"] = parsed_dates.loc[target_mask].dt.year.astype("Int64")
    grouped = targets.groupby([KEY_PATIENT, "_visit_year"], dropna=False, sort=False)

    collapsed_groups = 0
    collapsed_rows = 0
    collapsed_records: list[dict[str, object]] = []

    for _, group in grouped:
        if len(group) <= 1:
            continue
        collapsed_groups += 1
        collapsed_rows += int(len(group) - 1)
        merged_row = {column: _merge_cell_values(group[column].tolist()) for column in df.columns}
        collapsed_records.append(merged_row)

    if not collapsed_records:
        return work, 0, 0

    keep_mask = pd.Series(True, index=work.index)
    keep_mask.loc[targets.index] = False
    surviving = work.loc[keep_mask].copy()
    collapsed_df = pd.DataFrame(collapsed_records, columns=df.columns)
    out = pd.concat([surviving, collapsed_df], ignore_index=True)
    return out, collapsed_groups, collapsed_rows


def run(config: MergeConfig) -> None:
    logger = setup_logger("09b_merge_essdai_versions")
    print_script_overview(
        "09b_merge_essdai_versions.py",
        "Merge ESSDAI legacy/canonical columns by suffix and export both parquet+csv.",
    )

    print_step(1, "Load input")
    df = _load_table(config.input_path)

    print_step(2, "Filter patients by interval requirements")
    filtered, excluded_patients = _filter_patients(df)

    print_step(3, "Collapse 15D Optional Evaluation visits in same year by patient")
    filtered, collapsed_optional_groups, collapsed_optional_rows = _collapse_15d_optional_same_year(filtered)

    print_step(4, "Merge ESSDAI legacy/canonical columns")
    merged, merged_pairs = _merge_essdai_columns(filtered)

    print_step(5, "Merge additional requested column pairs")
    merged, additional_pairs_merged, additional_pairs_skipped = _merge_additional_pairs(merged)

    print_step(6, "Drop fully empty columns and write report")
    merged, dropped_empty_columns = _drop_fully_empty_columns(merged)
    config.output_base.parent.mkdir(parents=True, exist_ok=True)
    empty_cols_report_path = config.output_base.parent / f"{config.output_base.name}_dropped_empty_columns.csv"
    pd.DataFrame({"column_name": dropped_empty_columns}).to_csv(empty_cols_report_path, index=False)

    print_step(7, "Save outputs")
    parquet_path = config.output_base.with_suffix(".parquet")
    csv_path = config.output_base.with_suffix(".csv")
    merged.to_parquet(parquet_path, index=False)
    merged.to_csv(csv_path, index=False)

    logger.info("Saved merged dataset parquet: %s", parquet_path)
    logger.info("Saved merged dataset csv: %s", csv_path)
    logger.info("Saved empty-column drop report csv: %s", empty_cols_report_path)
    print_kv(
        "merge_summary",
        {
            "rows": len(merged),
            "columns": len(merged.columns),
            "excluded_patients_not_meeting_interval_requirements": excluded_patients,
            "collapsed_optional_15d_patient_year_groups": collapsed_optional_groups,
            "collapsed_optional_15d_reduced_rows": collapsed_optional_rows,
            "merged_suffix_pairs": merged_pairs,
            "additional_pairs_merged": additional_pairs_merged,
            "additional_pairs_skipped_missing_columns": additional_pairs_skipped,
            "dropped_fully_empty_columns": len(dropped_empty_columns),
            "parquet_path": str(parquet_path),
            "csv_path": str(csv_path),
            "empty_columns_report_path": str(empty_cols_report_path),
        },
    )


if __name__ == "__main__":
    run(_parse_args())
