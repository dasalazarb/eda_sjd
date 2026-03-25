from __future__ import annotations

import pandas as pd

from common import ANALYTIC_DIR, INTERMEDIATE_DIR, print_kv, save_parquet_and_csv, setup_logger


def build_patient_master(visits: pd.DataFrame) -> pd.DataFrame:
    grp = visits.groupby("subject_number", dropna=False)
    master = grp.agg(
        n_visits=("row_id_raw", "count"),
        first_visit=("visit_datetime", "min"),
        last_visit=("visit_datetime", "max"),
    ).reset_index()

    if "source_protocol" in visits.columns:
        sources = grp["source_protocol"].nunique(dropna=True).reset_index(name="n_protocols")
        master = master.merge(sources, on="subject_number", how="left")

    return master


def main() -> None:
    logger = setup_logger("05_build_backbone")

    visits = pd.read_parquet(INTERMEDIATE_DIR / "deduped_visits.parquet")
    master = build_patient_master(visits)

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
