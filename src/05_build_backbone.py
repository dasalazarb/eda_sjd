from __future__ import annotations

import pandas as pd

from common import (
    ANALYTIC_DIR,
    INTERMEDIATE_DIR,
    print_kv,
    print_script_overview,
    print_step,
    resolve_canonical_column,
    save_parquet_and_csv,
    setup_logger,
)


def build_patient_master(visits: pd.DataFrame) -> pd.DataFrame:
    subject_col = resolve_canonical_column(visits, "subject_number")
    grp = visits.groupby(subject_col, dropna=False)
    master = grp.agg(
        n_visits=("row_id_raw", "count"),
        first_visit=("visit_datetime", "min"),
        last_visit=("visit_datetime", "max"),
    ).reset_index()

    if subject_col != "subject_number":
        master = master.rename(columns={subject_col: "subject_number"})

    if "source_protocol" in visits.columns:
        sources = grp["source_protocol"].nunique(dropna=True).reset_index(name="n_protocols")
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


if __name__ == "__main__":
    main()
