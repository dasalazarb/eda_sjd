from __future__ import annotations

import pandas as pd

from common import ANALYTIC_DIR, INTERMEDIATE_DIR, print_kv, save_parquet_and_csv, setup_logger


def _resolve_subject_column(visits: pd.DataFrame) -> str:
    if "subject_number" in visits.columns:
        return "subject_number"

    suffix_matches = [c for c in visits.columns if c.endswith("__subject_number")]
    if len(suffix_matches) == 1:
        return suffix_matches[0]

    fuzzy_matches = [c for c in visits.columns if "subject_number" in c]
    if len(fuzzy_matches) == 1:
        return fuzzy_matches[0]

    raise KeyError(
        "Could not uniquely identify subject number column. "
        f"Matches ending with '__subject_number': {suffix_matches}; "
        f"fuzzy matches containing 'subject_number': {fuzzy_matches}"
    )


def build_patient_master(visits: pd.DataFrame) -> pd.DataFrame:
    subject_col = _resolve_subject_column(visits)
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
