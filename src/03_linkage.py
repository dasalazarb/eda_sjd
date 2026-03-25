from __future__ import annotations

import numpy as np
import pandas as pd

from common import INTERMEDIATE_DIR, print_kv, save_parquet_and_csv, setup_logger


def overlap_table(df11: pd.DataFrame, df15: pd.DataFrame) -> pd.DataFrame:
    a = set(df11["subject_number"].dropna().astype(str).unique())
    b = set(df15["subject_number"].dropna().astype(str).unique())

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
    keys = ["subject_number", "patient_record_number", "visit_date", "time_24_hour"]
    left = df11[[c for c in keys if c in df11.columns] + ["row_id_raw"]].rename(columns={"row_id_raw": "row_id_11d"})
    right = df15[[c for c in keys if c in df15.columns] + ["row_id_raw"]].rename(columns={"row_id_raw": "row_id_15d"})

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

    df11 = pd.read_parquet(INTERMEDIATE_DIR / "11d_raw_enriched.parquet")
    df15 = pd.read_parquet(INTERMEDIATE_DIR / "15d_raw_enriched.parquet")

    ov = overlap_table(df11, df15)
    ep = build_episode_candidates(df11, df15)

    save_parquet_and_csv(ov, INTERMEDIATE_DIR / "overlap_subjects", logger)
    save_parquet_and_csv(ep, INTERMEDIATE_DIR / "episode_candidates", logger)

    print_kv("Overlap summary", ov["overlap_type"].value_counts(dropna=False).to_dict())
    print_kv("Episode candidates", {"n_candidates": len(ep)})


if __name__ == "__main__":
    main()
