from __future__ import annotations

import pandas as pd

from common import (
    EDA_UNIFIED_REPORT_PATH,
    INTERMEDIATE_DIR,
    REPORTS_DIR,
    build_input_baseline_summary,
    build_targeted_eda_sheets,
    print_kv,
    print_script_overview,
    print_step,
    profile_dataframe,
    resolve_canonical_column,
    setup_logger,
    upsert_eda_sheets_xlsx,
)


def summarize_ids(df: pd.DataFrame) -> dict[str, int]:
    out = {
        "rows": len(df),
        "columns": len(df.columns),
    }

    for canonical in ["subject_number", "patient_record_number"]:
        try:
            col = resolve_canonical_column(df, canonical)
            out[f"{canonical}_unique"] = int(df[col].nunique(dropna=True))
            out[f"{canonical}_column"] = col
        except KeyError:
            continue

    return out


def main() -> None:
    logger = setup_logger("02_profile_individual")

    print_script_overview(
        "02_profile_individual.py",
        "Profiles each protocol independently and exports column-level quality summaries.",
    )

    print_step(1, "Load enriched 11D and 15D datasets from data_intermediate")
    df11 = pd.read_parquet(INTERMEDIATE_DIR / "11d_raw_enriched.parquet")
    df15 = pd.read_parquet(INTERMEDIATE_DIR / "15d_raw_enriched.parquet")

    print_step(2, "Run profile_dataframe on each dataset")
    p11 = profile_dataframe(df11, "11D")
    p15 = profile_dataframe(df15, "15D")

    print_step(3, "Save reports and print dataset dimensions")
    p11.to_csv(REPORTS_DIR / "eda_11d.csv", index=False)
    p15.to_csv(REPORTS_DIR / "eda_15d.csv", index=False)

    print_kv("EDA 11D dimensions", summarize_ids(df11))
    print_kv("EDA 15D dimensions", summarize_ids(df15))
    print_step(4, "Append targeted EDA + input summaries to unified workbook")
    sheets = {}
    sheets.update(build_targeted_eda_sheets(df11, "02_input_11d", "02_input_11d"))
    sheets.update(build_targeted_eda_sheets(df15, "02_input_15d", "02_input_15d"))
    sheets["02_input_11d_baseline"] = build_input_baseline_summary(df11, "02_input_11d_baseline")
    sheets["02_input_15d_baseline"] = build_input_baseline_summary(df15, "02_input_15d_baseline")
    workbook = upsert_eda_sheets_xlsx(EDA_UNIFIED_REPORT_PATH, sheets)
    logger.info("Updated unified EDA workbook: %s", workbook)

    logger.info("Saved reports/eda_11d.csv and reports/eda_15d.csv")


if __name__ == "__main__":
    main()
