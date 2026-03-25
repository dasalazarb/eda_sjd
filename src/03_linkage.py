from __future__ import annotations

import numpy as np
import pandas as pd

from common import (
    INTERMEDIATE_DIR,
    print_kv,
    print_script_overview,
    print_step,
    resolve_canonical_column,
    save_parquet_and_csv,
    setup_logger,
)


def overlap_table(df11: pd.DataFrame, df15: pd.DataFrame) -> pd.DataFrame:
    subject11 = resolve_canonical_column(df11, "subject_number")
    subject15 = resolve_canonical_column(df15, "subject_number")
    a = set(df11[subject11].dropna().astype(str).unique())
    b = set(df15[subject15].dropna().astype(str).unique())

    union = sorted(a | b)
    out = pd.DataFrame({"subject_number": union})
    out["in_11d"] = out["subject_number"].isin(a)
    out["in_15d"] = out["subject_number"].isin(b)
    out["overlap_type"] = np.select(
        [out["in_11d"] & out["in_15d"], out["in_11d"] & ~out["in_15d"], ~out["in_11d"] & out["in_15d"]],
        ["both", "only_11d", "only_15d"],
        default="unknown",
    )
    return out


def build_episode_candidates(df11: pd.DataFrame, df15: pd.DataFrame) -> pd.DataFrame:
    canonical_to_actual_11 = {}
    canonical_to_actual_15 = {}
    for key in ["subject_number", "patient_record_number", "visit_date", "time_24_hour"]:
        try:
            canonical_to_actual_11[key] = resolve_canonical_column(df11, key)
        except KeyError:
            pass
        try:
            canonical_to_actual_15[key] = resolve_canonical_column(df15, key)
        except KeyError:
            pass

    left_cols = list(canonical_to_actual_11.values()) + ["row_id_raw"]
    right_cols = list(canonical_to_actual_15.values()) + ["row_id_raw"]
    left = df11[left_cols].rename(columns={**{v: k for k, v in canonical_to_actual_11.items()}, "row_id_raw": "row_id_11d"})
    right = df15[right_cols].rename(columns={**{v: k for k, v in canonical_to_actual_15.items()}, "row_id_raw": "row_id_15d"})

    on_cols = [c for c in ["subject_number", "patient_record_number", "visit_date", "time_24_hour"] if c in left.columns and c in right.columns]
    if not on_cols:
        return pd.DataFrame(columns=["subject_number", "rule_type", "row_id_11d", "row_id_15d"])

    exact = left.merge(right, on=on_cols, how="inner")
    if exact.empty:
        return pd.DataFrame(columns=["subject_number", "rule_type", "row_id_11d", "row_id_15d"])

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


if __name__ == "__main__":
    main()
