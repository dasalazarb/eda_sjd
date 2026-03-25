from __future__ import annotations

import pandas as pd

from common import INTERMEDIATE_DIR, print_kv, save_parquet_and_csv, setup_logger


KEY_COLS = ["subject_number", "patient_record_number", "visit_datetime"]


def deduplicate_within_protocol(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    existing_keys = [c for c in KEY_COLS if c in df.columns]
    if not existing_keys:
        kept = df.copy()
        audit = pd.DataFrame({"row_id_raw": df["row_id_raw"], "decision": "kept_as_is", "reason": "missing_dedup_keys"})
        return kept, audit

    ranked = df.sort_values(existing_keys + ["row_id_raw"]).copy()
    ranked["dup_rank"] = ranked.groupby(existing_keys).cumcount() + 1

    kept = ranked[ranked["dup_rank"] == 1].drop(columns=["dup_rank"])

    audit = ranked[["row_id_raw"]].copy()
    audit["decision"] = ranked["dup_rank"].map(lambda x: "kept_as_is" if x == 1 else "consolidated_within_protocol")
    audit["reason"] = ranked["dup_rank"].map(lambda x: "first_by_sort_order" if x == 1 else "duplicate_key")

    return kept, audit


def main() -> None:
    logger = setup_logger("04_dedup_rules")

    df11 = pd.read_parquet(INTERMEDIATE_DIR / "11d_raw_enriched.parquet")
    df15 = pd.read_parquet(INTERMEDIATE_DIR / "15d_raw_enriched.parquet")

    kept11, audit11 = deduplicate_within_protocol(df11)
    kept15, audit15 = deduplicate_within_protocol(df15)

    dedup = pd.concat([kept11, kept15], ignore_index=True)
    audit = pd.concat([audit11, audit15], ignore_index=True)

    save_parquet_and_csv(dedup, INTERMEDIATE_DIR / "deduped_visits", logger)
    save_parquet_and_csv(audit, INTERMEDIATE_DIR / "conflict_log", logger)

    print_kv(
        "Dedup summary",
        {
            "input_rows": len(df11) + len(df15),
            "output_rows": len(dedup),
            "within_protocol_consolidated": int((audit["decision"] == "consolidated_within_protocol").sum()),
        },
    )


if __name__ == "__main__":
    main()
