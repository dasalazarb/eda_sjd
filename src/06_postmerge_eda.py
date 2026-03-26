from __future__ import annotations

import pandas as pd

from common import (
    ANALYTIC_DIR,
    REPORTS_DIR,
    print_kv,
    print_script_overview,
    print_step,
    resolve_canonical_column,
    setup_logger,
)


def _safe_median(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce")
    return float(numeric.median()) if numeric.notna().any() else 0.0


def main() -> None:
    logger = setup_logger("06_postmerge_eda")

    print_script_overview(
        "06_postmerge_eda.py",
        "Computes post-merge readiness metrics from backbone outputs and exports a compact report.",
    )

    print_step(1, "Load patient_master and visits_long tables")
    master = pd.read_parquet(ANALYTIC_DIR / "patient_master.parquet")
    visits = pd.read_parquet(ANALYTIC_DIR / "visits_long.parquet")

    print_step(2, "Resolve canonical columns (including prefixed variants)")
    subject_col = resolve_canonical_column(master, "subject_number")
    n_visits_col = resolve_canonical_column(master, "n_visits")
    logger.info("Resolved subject_number column: %s", subject_col)
    logger.info("Resolved n_visits column: %s", n_visits_col)

    print_step(3, "Compute post-merge quality metrics")
    n_visits_series = master[n_visits_col] if not master.empty else pd.Series(dtype=float)
    report = pd.DataFrame(
        [
            {"metric": "n_patients", "value": int(master[subject_col].nunique(dropna=True))},
            {"metric": "n_visits", "value": len(visits)},
            {
                "metric": "median_visits_per_patient",
                "value": _safe_median(n_visits_series),
            },
            {
                "metric": "patients_with_2plus_visits",
                "value": int((pd.to_numeric(n_visits_series, errors="coerce") >= 2).sum()),
            },
        ]
    )

    print_step(4, "Save report and print metric summary")
    report.to_csv(REPORTS_DIR / "eda_postmerge.csv", index=False)

    print_kv("Post-merge EDA", dict(zip(report["metric"], report["value"])))
    logger.info("Saved reports/eda_postmerge.csv")


if __name__ == "__main__":
    main()
