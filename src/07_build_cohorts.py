from __future__ import annotations

import pandas as pd

from common import ANALYTIC_DIR, REPORTS_DIR, print_kv, save_parquet_and_csv, setup_logger


def main() -> None:
    logger = setup_logger("07_build_cohorts")

    master = pd.read_parquet(ANALYTIC_DIR / "patient_master.parquet")
    visits = pd.read_parquet(ANALYTIC_DIR / "visits_long.parquet")

    baseline = visits.sort_values(["subject_number", "visit_datetime"]).groupby("subject_number", as_index=False).first()
    longitudinal = visits.groupby("subject_number").filter(lambda x: len(x) >= 2)

    time_to_event = master.copy()
    time_to_event["followup_days"] = (
        (time_to_event["last_visit"] - time_to_event["first_visit"]).dt.total_seconds() / 86400
    )

    save_parquet_and_csv(baseline, ANALYTIC_DIR / "cohort_baseline", logger)
    save_parquet_and_csv(longitudinal, ANALYTIC_DIR / "cohort_longitudinal", logger)
    save_parquet_and_csv(time_to_event, ANALYTIC_DIR / "cohort_time_to_event", logger)

    readiness = pd.DataFrame(
        [
            {"question": "unique_patients_after_dedup", "value": int(master["subject_number"].nunique(dropna=True))},
            {
                "question": "patients_with_cross_protocol_continuity",
                "value": int((master.get("n_protocols", pd.Series(dtype=float)) >= 2).sum()),
            },
            {
                "question": "patients_with_repeated_measures",
                "value": int((master["n_visits"] >= 2).sum()) if "n_visits" in master.columns else 0,
            },
            {"question": "episodes_total", "value": len(visits)},
        ]
    )
    readiness.to_csv(REPORTS_DIR / "analysis_readiness.csv", index=False)

    print_kv(
        "Cohort summary",
        {
            "baseline_n": len(baseline),
            "longitudinal_n_rows": len(longitudinal),
            "tte_n": len(time_to_event),
        },
    )


if __name__ == "__main__":
    main()
