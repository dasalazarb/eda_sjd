from __future__ import annotations

import pandas as pd

from common import (
    INTERMEDIATE_DIR,
    REPORTS_DIR,
    print_kv,
    print_script_overview,
    print_step,
    profile_dataframe,
    setup_logger,
)


def summarize_ids(df: pd.DataFrame) -> dict[str, int]:
    out = {
        "rows": len(df),
        "columns": len(df.columns),
    }
    if "subject_number" in df.columns:
        out["subject_number_unique"] = int(df["subject_number"].nunique(dropna=True))
    if "patient_record_number" in df.columns:
        out["patient_record_number_unique"] = int(df["patient_record_number"].nunique(dropna=True))
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

    logger.info("Saved reports/eda_11d.csv and reports/eda_15d.csv")


if __name__ == "__main__":
    main()
