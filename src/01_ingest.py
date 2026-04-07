from __future__ import annotations

from pathlib import Path

import pandas as pd

from common import (
    EDA_UNIFIED_REPORT_PATH,
    RAW_DIR,
    INTERMEDIATE_DIR,
    build_group_prefixed_columns,
    build_targeted_eda_sheets,
    drop_sensitive_name_columns,
    parse_datetime_columns,
    print_kv,
    print_script_overview,
    print_step,
    replace_empty_with_nan,
    merge_sheet_dicts,
    save_parquet_and_csv,
    setup_logger,
    standardize_columns,
    upsert_eda_sheets_xlsx,
)


def relabel_15d_optional_evaluations(df: pd.DataFrame, logger) -> pd.DataFrame:
    """Re-label 15D interval names after the baseline visit date per patient."""
    interval_col = "ids__interval_name"
    patient_col = "ids__patient_record_number"
    visit_date_col = "ids__visit_date"
    visit_datetime_col = "visit_datetime"
    time_col = "ids__time_24_hour"

    required_cols = [interval_col, patient_col, visit_date_col]
    if any(col not in df.columns for col in required_cols):
        logger.warning(
            "Skipping 15D interval relabel because required columns are missing: %s",
            [c for c in required_cols if c not in df.columns],
        )
        return df

    out = df.copy()
    out["_visit_date_dt"] = pd.to_datetime(out[visit_date_col], errors="coerce", format="mixed")
    out["_visit_day"] = out["_visit_date_dt"].dt.normalize()
    if time_col in out.columns:
        out["_visit_time_dt"] = pd.to_datetime(out[time_col], errors="coerce", format="mixed")
    else:
        out["_visit_time_dt"] = pd.NaT
    out = out.sort_values([patient_col, "_visit_day", "_visit_time_dt"], ascending=[True, True, True], kind="stable")

    baseline_day = out.groupby(patient_col, dropna=False)["_visit_day"].transform("min")
    post_baseline_mask = out["_visit_day"].notna() & baseline_day.notna() & (out["_visit_day"] > baseline_day)

    unique_post_visit_rank = (
        out.loc[post_baseline_mask, [patient_col, "_visit_day"]]
        .drop_duplicates()
        .sort_values([patient_col, "_visit_day"], ascending=[True, True], kind="stable")
    )
    unique_post_visit_rank["evaluation_num"] = (
        unique_post_visit_rank.groupby(patient_col, dropna=False).cumcount() + 1
    )

    out = out.merge(
        unique_post_visit_rank,
        on=[patient_col, "_visit_day"],
        how="left",
        sort=False,
    )

    relabel_mask = out["evaluation_num"].notna()
    out.loc[relabel_mask, interval_col] = "15D Optional Evaluation " + out.loc[
        relabel_mask, "evaluation_num"
    ].astype(int).astype(str)
    out = out.drop(columns=["_visit_date_dt", "_visit_day", "_visit_time_dt", "evaluation_num"])

    logger.info(
        "Applied 15D interval relabeling | patient_col=%s | relabeled_rows=%d | baseline_ties_preserved=%s",
        patient_col,
        int(relabel_mask.sum()),
        True,
    )
    return out


def ingest_one(path: Path, source_protocol: str, logger) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("Reading file: %s", path)
    raw = pd.read_excel(path, header=None)
    if len(raw) < 2:
        raise ValueError(f"{path.name} must include group row + variable row in first two rows.")

    grouped_columns = build_group_prefixed_columns(raw.iloc[0], raw.iloc[1])
    df = raw.iloc[2:].reset_index(drop=True).copy()
    df.columns = grouped_columns
    input_baseline = df.copy()
    n_groups = int(raw.iloc[0].ffill().nunique(dropna=True))
    logger.info(
        "Header preprocessing complete for %s | dropped_header_rows=2 | detected_groups=%d | columns=%d",
        path.name,
        n_groups,
        len(df.columns),
    )

    df = standardize_columns(df)
    dup_after_standardization = int(df.columns.to_series().str.contains(r"__dup\d+$", regex=True).sum())
    if dup_after_standardization:
        logger.info(
            "Standardized columns created %d disambiguated duplicates in %s",
            dup_after_standardization,
            path.name,
        )
    df = replace_empty_with_nan(df)
    df = parse_datetime_columns(df)
    df, dropped_name_cols = drop_sensitive_name_columns(df)
    if dropped_name_cols:
        logger.info("Removed sensitive name columns from %s: %s", path.name, dropped_name_cols)

    df["source_protocol"] = source_protocol
    df["source_file"] = path.name
    df["row_id_raw"] = [f"{source_protocol}_{i:07d}" for i in range(len(df))]

    return df, input_baseline


def resolve_raw_path(candidates: list[str]) -> Path:
    for candidate in candidates:
        path = RAW_DIR / candidate
        if path.exists():
            return path
    joined = ", ".join(candidates)
    raise FileNotFoundError(f"None of the expected raw files were found in {RAW_DIR}: {joined}")


def main() -> None:
    logger = setup_logger("01_ingest")

    print_script_overview(
        "01_ingest.py",
        "Reads raw Excel files, creates {category}__{variable} headers, cleans fields, and saves enriched raw datasets.",
    )

    f11 = resolve_raw_path(["CTDB Data Download 11D.xlsx", "CTDB_Data_Download_11D.xlsx"])
    f15 = resolve_raw_path(["CTDB Data Download 15D.xlsx", "CTDB Data Download 15D.xslx", "CTDB_Data_Download_15D.xlsx"])

    print_step(1, "Read 11D and 15D raw files with two-row grouped headers")
    df11, input11_baseline = ingest_one(f11, "11D", logger)
    df15, input15_baseline = ingest_one(f15, "15D", logger)
    df15 = relabel_15d_optional_evaluations(df15, logger)

    print_step(2, "Save cleaned raw outputs to data_intermediate")
    save_parquet_and_csv(df11, INTERMEDIATE_DIR / "11d_raw_enriched", logger)
    save_parquet_and_csv(df15, INTERMEDIATE_DIR / "15d_raw_enriched", logger)

    print_step(3, "Print ingest metrics for traceability")
    print_kv(
        "Ingest summary",
        {
            "11D_rows": len(df11),
            "11D_cols": len(df11.columns),
            "15D_rows": len(df15),
            "15D_cols": len(df15.columns),
        },
    )
    print_step(4, "Build baseline/input and clean/output EDA and append sheets to unified workbook")
    sheets = {}
    sheets = merge_sheet_dicts(
        sheets,
        build_targeted_eda_sheets(input11_baseline, "01_11D_input", "01_11D_input", consolidated=True),
    )
    sheets = merge_sheet_dicts(sheets, build_targeted_eda_sheets(df11, "01_11D_output", "01_11D_output", consolidated=True))
    sheets = merge_sheet_dicts(
        sheets,
        build_targeted_eda_sheets(input15_baseline, "01_15D_input", "01_15D_input", consolidated=True),
    )
    sheets = merge_sheet_dicts(sheets, build_targeted_eda_sheets(df15, "01_15D_output", "01_15D_output", consolidated=True))
    workbook = upsert_eda_sheets_xlsx(EDA_UNIFIED_REPORT_PATH, sheets)
    logger.info("Updated unified EDA workbook: %s", workbook)


if __name__ == "__main__":
    main()
