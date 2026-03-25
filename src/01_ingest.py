from __future__ import annotations

from pathlib import Path

import pandas as pd

from common import (
    RAW_DIR,
    INTERMEDIATE_DIR,
    build_group_prefixed_columns,
    parse_datetime_columns,
    print_kv,
    replace_empty_with_nan,
    save_parquet_and_csv,
    setup_logger,
    standardize_columns,
)


def ingest_one(path: Path, source_protocol: str, logger) -> pd.DataFrame:
    logger.info("Reading file: %s", path)
    raw = pd.read_excel(path, header=None)
    if len(raw) < 2:
        raise ValueError(f"{path.name} must include group row + variable row in first two rows.")

    grouped_columns = build_group_prefixed_columns(raw.iloc[0], raw.iloc[1])
    df = raw.iloc[2:].reset_index(drop=True).copy()
    df.columns = grouped_columns
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

    df["source_protocol"] = source_protocol
    df["source_file"] = path.name
    df["row_id_raw"] = [f"{source_protocol}_{i:07d}" for i in range(len(df))]

    return df


def main() -> None:
    logger = setup_logger("01_ingest")

    f11 = RAW_DIR / "CTDB_Data_Download_11D.xlsx"
    f15 = RAW_DIR / "CTDB_Data_Download_15D.xlsx"

    df11 = ingest_one(f11, "11D", logger)
    df15 = ingest_one(f15, "15D", logger)

    save_parquet_and_csv(df11, INTERMEDIATE_DIR / "11d_raw_enriched", logger)
    save_parquet_and_csv(df15, INTERMEDIATE_DIR / "15d_raw_enriched", logger)

    print_kv(
        "Ingest summary",
        {
            "11D_rows": len(df11),
            "11D_cols": len(df11.columns),
            "15D_rows": len(df15),
            "15D_cols": len(df15.columns),
        },
    )


if __name__ == "__main__":
    main()
