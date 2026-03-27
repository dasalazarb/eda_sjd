from __future__ import annotations

import pandas as pd

from common import (
    ANALYTIC_DIR,
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


def _resolve_optional_column(df: pd.DataFrame, canonical_name: str) -> str | None:
    """Return the best column match for a canonical name, or None when absent."""
    try:
        return resolve_canonical_column(df, canonical_name)
    except KeyError:
        return None


def build_patient_master(visits: pd.DataFrame) -> pd.DataFrame:
    """Create one row per patient with visit counts and temporal coverage."""
    subject_col = resolve_canonical_column(visits, "subject_number")
    row_id_col = resolve_canonical_column(visits, "row_id_raw")
    visit_datetime_col = resolve_canonical_column(visits, "visit_datetime")
    source_protocol_col = _resolve_optional_column(visits, "source_protocol")

    grp = visits.groupby(subject_col, dropna=False)
    master = grp.agg(
        n_visits=(row_id_col, "count"),
        first_visit=(visit_datetime_col, "min"),
        last_visit=(visit_datetime_col, "max"),
    ).reset_index()

    if subject_col != "subject_number":
        master = master.rename(columns={subject_col: "subject_number"})

    if source_protocol_col:
        sources = grp[source_protocol_col].nunique(dropna=True).reset_index(name="n_protocols")
        if subject_col != "subject_number":
            sources = sources.rename(columns={subject_col: "subject_number"})
        master = master.merge(sources, on="subject_number", how="left")

    return master


def main() -> None:
    logger = setup_logger("05_build_backbone")

    print_script_overview(
        "05_build_backbone.py",
        "Builds patient-level master table and longitudinal visits backbone from deduplicated data.",
    )

    print_step(1, "Load deduplicated visits")
    visits = pd.read_parquet(INTERMEDIATE_DIR / "deduped_visits.parquet")

    print_step(2, "Create patient_master with visit count and temporal span")
    master = build_patient_master(visits)

    print_step(3, "Save patient_master and visits_long outputs")
    save_parquet_and_csv(master, ANALYTIC_DIR / "patient_master", logger)
    save_parquet_and_csv(visits, ANALYTIC_DIR / "visits_long", logger)

    print_kv(
        "Backbone summary",
        {
            "n_patients": int(master["subject_number"].nunique(dropna=True)),
            "n_visits": len(visits),
        },
    )
    print_step(4, "Append targeted EDA for patient_master and visits to unified workbook")
    sheets = {}
    sheets.update(
        build_targeted_eda_sheets(master, "05_patient_master_output", "05_patient_master_output", consolidated=True)
    )
    sheets.update(build_targeted_eda_sheets(visits, "05_visits_long_output", "05_visits_long_output", consolidated=True))
    workbook = upsert_eda_sheets_xlsx(EDA_UNIFIED_REPORT_PATH, sheets)
    logger.info("Updated unified EDA workbook: %s", workbook)


if __name__ == "__main__":
    main()
