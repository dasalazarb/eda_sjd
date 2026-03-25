from __future__ import annotations

import pandas as pd

from common import ANALYTIC_DIR, REPORTS_DIR, print_kv, print_script_overview, print_step, setup_logger


def main() -> None:
    logger = setup_logger("06_postmerge_eda")

    print_script_overview(
        "06_postmerge_eda.py",
        "Computes post-merge readiness metrics from backbone outputs and exports a compact report.",
    )

    print_step(1, "Load patient_master and visits_long tables")
    master = pd.read_parquet(ANALYTIC_DIR / "patient_master.parquet")
    visits = pd.read_parquet(ANALYTIC_DIR / "visits_long.parquet")

    print_step(2, "Compute post-merge quality metrics")
    report = pd.DataFrame(
        [
            {"metric": "n_patients", "value": int(master["subject_number"].nunique(dropna=True))},
            {"metric": "n_visits", "value": len(visits)},
            {
                "metric": "median_visits_per_patient",
                "value": float(master["n_visits"].median()) if not master.empty else 0.0,
            },
            {
                "metric": "patients_with_2plus_visits",
                "value": int((master["n_visits"] >= 2).sum()) if "n_visits" in master.columns else 0,
            },
        ]
    )

    print_step(3, "Save report and print metric summary")
    report.to_csv(REPORTS_DIR / "eda_postmerge.csv", index=False)

    print_kv("Post-merge EDA", dict(zip(report["metric"], report["value"])))
    logger.info("Saved reports/eda_postmerge.csv")


if __name__ == "__main__":
    main()
