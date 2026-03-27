from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from common import (
    EDA_UNIFIED_REPORT_PATH,
    INTERMEDIATE_DIR,
    build_targeted_eda_sheets,
    merge_sheet_dicts,
    print_kv,
    print_script_overview,
    print_step,
    resolve_canonical_column,
    save_parquet_and_csv,
    setup_logger,
    upsert_eda_sheets_xlsx,
)

LINKAGE_KEYS = ["subject_number", "patient_record_number", "visit_date", "time_24_hour"]


def _normalize_identifier(series: pd.Series) -> pd.Series:
    """Normalize identifiers for deterministic matching and overlap counts."""
    normalized = series.astype(str).str.strip()
    normalized = normalized.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    return normalized


def _resolve_columns(df: pd.DataFrame, canonical_names: Iterable[str]) -> dict[str, str]:
    """
    Resolve canonical names from heterogeneous schemas.

    Supports columns like:
      - subject_number
      - categoria__subject_number
      - anything_subject_number
    """
    resolved: dict[str, str] = {}
    for canonical in canonical_names:
        try:
            resolved[canonical] = resolve_canonical_column(df, canonical)
        except KeyError:
            continue
    return resolved


def overlap_table(df11: pd.DataFrame, df15: pd.DataFrame) -> pd.DataFrame:
    subject11 = _resolve_columns(df11, ["subject_number"]).get("subject_number")
    subject15 = _resolve_columns(df15, ["subject_number"]).get("subject_number")
    if not subject11 or not subject15:
        return pd.DataFrame(columns=["subject_number", "in_11d", "in_15d", "overlap_type"])

    a = set(_normalize_identifier(df11[subject11]).dropna().unique())
    b = set(_normalize_identifier(df15[subject15]).dropna().unique())

    union = sorted(a | b)
    out = pd.DataFrame({"subject_number": union})
    out["in_11d"] = out["subject_number"].isin(a)
    out["in_15d"] = out["subject_number"].isin(b)
    out["overlap_type"] = np.select(
        [
            out["in_11d"] & out["in_15d"],
            out["in_11d"] & ~out["in_15d"],
            ~out["in_11d"] & out["in_15d"],
        ],
        ["both", "only_11d", "only_15d"],
        default="unknown",
    )
    return out


def build_episode_candidates(df11: pd.DataFrame, df15: pd.DataFrame) -> pd.DataFrame:
    resolved_11 = _resolve_columns(df11, LINKAGE_KEYS)
    resolved_15 = _resolve_columns(df15, LINKAGE_KEYS)

    left_cols = [*resolved_11.values(), "row_id_raw"]
    right_cols = [*resolved_15.values(), "row_id_raw"]
    left = df11[left_cols].rename(
        columns={**{actual: canonical for canonical, actual in resolved_11.items()}, "row_id_raw": "row_id_11d"}
    )
    right = df15[right_cols].rename(
        columns={**{actual: canonical for canonical, actual in resolved_15.items()}, "row_id_raw": "row_id_15d"}
    )

    for key in ["subject_number", "patient_record_number"]:
        if key in left.columns:
            left[key] = _normalize_identifier(left[key])
        if key in right.columns:
            right[key] = _normalize_identifier(right[key])

    on_cols = [c for c in LINKAGE_KEYS if c in left.columns and c in right.columns]
    output_cols = ["subject_number", "rule_type", "row_id_11d", "row_id_15d"]
    if not on_cols:
        return pd.DataFrame(columns=output_cols)

    exact = left.merge(right, on=on_cols, how="inner")
    if exact.empty:
        return pd.DataFrame(columns=output_cols)

    exact["rule_type"] = "A_exact_same_episode"
    return exact


def main() -> None:
    logger = setup_logger("03_linkage")

    print_script_overview(
        "03_linkage.py",
        "Builds cross-protocol patient overlap and exact episode candidates between 11D and 15D.",
    )

    print_step(1, "Load enriched 11D and 15D datasets")
    df11 = pd.read_parquet(INTERMEDIATE_DIR / "11d_raw_enriched.parquet")
    df15 = pd.read_parquet(INTERMEDIATE_DIR / "15d_raw_enriched.parquet")

    print_step(2, "Generate overlap table and exact episode candidate matches")
    ov = overlap_table(df11, df15)
    ep = build_episode_candidates(df11, df15)

    print_step(3, "Save linkage outputs and print summary counts")
    save_parquet_and_csv(ov, INTERMEDIATE_DIR / "overlap_subjects", logger)
    save_parquet_and_csv(ep, INTERMEDIATE_DIR / "episode_candidates", logger)

    print_kv("Overlap summary", ov["overlap_type"].value_counts(dropna=False).to_dict())
    print_kv("Episode candidates", {"n_candidates": len(ep)})
    print_step(4, "Append targeted EDA for overlap/episode outputs to unified workbook")
    sheets = {}
    sheets = merge_sheet_dicts(
        sheets,
        build_targeted_eda_sheets(ov, "03_overlap_subjects_output", "03_overlap_subjects_output", consolidated=True),
    )
    sheets = merge_sheet_dicts(
        sheets,
        build_targeted_eda_sheets(
            ep,
            "03_episode_candidates_output",
            "03_episode_candidates_output",
            consolidated=True,
        ),
    )
    workbook = upsert_eda_sheets_xlsx(EDA_UNIFIED_REPORT_PATH, sheets)
    logger.info("Updated unified EDA workbook: %s", workbook)


if __name__ == "__main__":
    main()
