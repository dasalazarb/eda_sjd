from __future__ import annotations

import pandas as pd

from common import (
    EDA_UNIFIED_REPORT_PATH,
    INTERMEDIATE_DIR,
    build_targeted_eda_sheets,
    print_kv,
    print_script_overview,
    print_step,
    resolve_canonical_column,
    save_parquet_and_csv,
    setup_logger,
    upsert_eda_sheets_xlsx,
)


KEY_COLS = ["subject_number", "patient_record_number", "visit_datetime"]


def _comparison_columns(df: pd.DataFrame, dedup_keys: list[str]) -> list[str]:
    excluded = set(dedup_keys) | {"row_id_raw"}
    return [c for c in df.columns if "__" in c and c not in excluded]


def _classify_against_anchor(anchor: pd.Series, current: pd.Series, columns: list[str]) -> tuple[str, str]:
    diff_cols: list[str] = []
    current_only_cols: list[str] = []
    conflict = False

    for col in columns:
        a = anchor[col]
        b = current[col]
        a_null = pd.isna(a)
        b_null = pd.isna(b)
        if a_null and b_null:
            continue
        if a_null != b_null:
            diff_cols.append(col)
            if not b_null and a_null:
                current_only_cols.append(col)
            continue
        if a != b:
            diff_cols.append(col)
            conflict = True

    if not diff_cols:
        return "exact_duplicate", ""
    if conflict:
        return "conflicting_duplicate", "|".join(diff_cols)
    return "complementary_duplicate", "|".join(current_only_cols or diff_cols)


def deduplicate_within_protocol(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    existing_keys = []
    for key in KEY_COLS:
        try:
            existing_keys.append(resolve_canonical_column(df, key))
        except KeyError:
            continue
    if not existing_keys:
        kept = df.copy()
        audit = pd.DataFrame(
            {
                "row_id_raw": df["row_id_raw"],
                "decision": "kept_as_is",
                "reason": "missing_dedup_keys",
                "duplicate_group_id": pd.NA,
                "dup_rank": 1,
                "visit_datetime_adjustment_seconds": 0,
                "comparison_type": "not_evaluated",
                "comparison_detail": "",
            }
        )
        return kept, audit

    ranked = df.sort_values(existing_keys + ["row_id_raw"]).copy().reset_index(drop=True)
    ranked["dup_rank"] = ranked.groupby(existing_keys).cumcount() + 1
    ranked["duplicate_group_size"] = ranked.groupby(existing_keys)["row_id_raw"].transform("size")
    ranked["duplicate_group_id"] = pd.factorize(
        ranked[existing_keys].apply(lambda row: tuple(row.values.tolist()), axis=1)
    )[0] + 1
    ranked["visit_datetime_adjustment_seconds"] = ranked["dup_rank"] - 1

    visit_datetime_col = resolve_canonical_column(df, "visit_datetime")
    ranked[visit_datetime_col] = pd.to_datetime(ranked[visit_datetime_col], errors="coerce")
    ranked[visit_datetime_col] = ranked[visit_datetime_col] + pd.to_timedelta(
        ranked["visit_datetime_adjustment_seconds"], unit="s"
    )

    compare_cols = _comparison_columns(ranked, existing_keys)
    ranked["comparison_type"] = "not_evaluated"
    ranked["comparison_detail"] = ""

    for _, group in ranked.groupby(existing_keys, sort=False):
        if len(group) <= 1:
            row_idx = group.index[0]
            ranked.at[row_idx, "comparison_type"] = "unique_key"
            continue
        anchor = ranked.loc[group.index[0]]
        ranked.at[group.index[0], "comparison_type"] = "anchor_duplicate_group"
        for idx in group.index[1:]:
            comparison_type, detail = _classify_against_anchor(anchor, ranked.loc[idx], compare_cols)
            ranked.at[idx, "comparison_type"] = comparison_type
            ranked.at[idx, "comparison_detail"] = detail

    kept = ranked.drop(columns=["duplicate_group_size"])

    audit = ranked[["row_id_raw"]].copy()
    try:
        patient_record_col = resolve_canonical_column(ranked, "patient_record_number")
        audit["patient_record_number"] = ranked[patient_record_col]
    except KeyError:
        audit["patient_record_number"] = pd.NA
    audit["decision"] = ranked["dup_rank"].map(lambda x: "kept_as_is" if x == 1 else "kept_with_shifted_datetime")
    audit["reason"] = ranked["dup_rank"].map(lambda x: "first_by_sort_order" if x == 1 else "duplicate_key_shifted_time")
    audit["duplicate_group_id"] = ranked["duplicate_group_id"]
    audit["dup_rank"] = ranked["dup_rank"]
    audit["visit_datetime_adjustment_seconds"] = ranked["visit_datetime_adjustment_seconds"]
    audit["comparison_type"] = ranked["comparison_type"]
    audit["comparison_detail"] = ranked["comparison_detail"]

    return kept, audit


def main() -> None:
    logger = setup_logger("04_dedup_rules")

    print_script_overview(
        "04_dedup_rules.py",
        "Applies deterministic within-protocol deduplication and writes audit decisions.",
    )

    print_step(1, "Load enriched protocol datasets")
    df11 = pd.read_parquet(INTERMEDIATE_DIR / "11d_raw_enriched.parquet")
    df15 = pd.read_parquet(INTERMEDIATE_DIR / "15d_raw_enriched.parquet")

    print_step(2, "Deduplicate rows within each protocol using key hierarchy")
    kept11, audit11 = deduplicate_within_protocol(df11)
    kept15, audit15 = deduplicate_within_protocol(df15)

    print_step(3, "Persist deduplicated visits and conflict/audit log")
    dedup = pd.concat([kept11, kept15], ignore_index=True)
    audit = pd.concat([audit11, audit15], ignore_index=True)

    save_parquet_and_csv(dedup, INTERMEDIATE_DIR / "deduped_visits", logger)
    save_parquet_and_csv(audit, INTERMEDIATE_DIR / "conflict_log", logger)

    print_kv(
        "Dedup summary",
        {
            "input_rows": len(df11) + len(df15),
            "output_rows": len(dedup),
            "rows_with_shifted_datetime": int((audit["visit_datetime_adjustment_seconds"] > 0).sum()),
            "exact_duplicates": int((audit["comparison_type"] == "exact_duplicate").sum()),
            "complementary_duplicates": int((audit["comparison_type"] == "complementary_duplicate").sum()),
            "conflicting_duplicates": int((audit["comparison_type"] == "conflicting_duplicate").sum()),
        },
    )
    print_step(4, "Append targeted EDA for dedup (+partial audit) to unified workbook")
    sheets = {}
    sheets.update(
        build_targeted_eda_sheets(dedup, "04_deduped_visits_output", "04_deduped_visits_output", consolidated=True)
    )
    sheets.update(build_targeted_eda_sheets(audit, "04_conflict_log_output", "04_conflict_log_output", consolidated=True))
    workbook = upsert_eda_sheets_xlsx(EDA_UNIFIED_REPORT_PATH, sheets)
    logger.info("Updated unified EDA workbook: %s", workbook)


if __name__ == "__main__":
    main()
