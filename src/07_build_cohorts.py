from __future__ import annotations

import pandas as pd

from common import (
    ANALYTIC_DIR,
    EDA_UNIFIED_REPORT_PATH,
    REPORTS_DIR,
    build_targeted_eda_sheets,
    print_kv,
    print_script_overview,
    print_step,
    resolve_canonical_column,
    save_parquet_and_csv,
    setup_logger,
    upsert_eda_sheets_xlsx,
)


def _ensure_category_prefixed_subject(df: pd.DataFrame, category: str) -> tuple[pd.DataFrame, str]:
    """
    Ensure the subject identifier column follows '{category}__subject_number'.

    Returns the updated dataframe and the resolved canonical subject column name
    before renaming (or the target name if it already exists).
    """
    out = df.copy()
    target = f"{category}__subject_number"

    if target in out.columns:
        return out, target

    source = resolve_canonical_column(out, "subject_number")
    out = out.rename(columns={source: target})
    return out, target


def _format_protocol_origin(series: pd.Series) -> str:
    """Format protocol provenance as '11D', '15D', or '11D | 15D'."""
    vals = {str(v).strip() for v in series.dropna() if str(v).strip()}
    if not vals:
        return pd.NA

    ordered = [protocol for protocol in ("11D", "15D") if protocol in vals]
    extras = sorted(v for v in vals if v not in {"11D", "15D"})
    return " | ".join(ordered + extras)


def main() -> None:
    logger = setup_logger("07_build_cohorts")

    print_script_overview(
        "07_build_cohorts.py",
        "Builds baseline, longitudinal, and time-to-event cohorts and writes analysis readiness checks.",
    )

    print_step(1, "Load patient_master and visits_long")
    master = pd.read_parquet(ANALYTIC_DIR / "patient_master.parquet")
    visits = pd.read_parquet(ANALYTIC_DIR / "visits_long.parquet")

    print_step(2, "Resolve canonical columns and derive cohort tables")
    visit_subject_col = resolve_canonical_column(visits, "subject_number")
    visit_datetime_col = resolve_canonical_column(visits, "visit_datetime")
    source_protocol_col = resolve_canonical_column(visits, "source_protocol")

    baseline = (
        visits.sort_values([visit_subject_col, visit_datetime_col])
        .groupby(visit_subject_col, as_index=False)
        .first()
    )
    longitudinal = visits.groupby(visit_subject_col).filter(lambda x: len(x) >= 2)

    master_subject_col = resolve_canonical_column(master, "subject_number")
    first_visit_col = resolve_canonical_column(master, "first_visit")
    last_visit_col = resolve_canonical_column(master, "last_visit")

    time_to_event = master.copy()
    time_to_event["followup_days"] = (
        (time_to_event[last_visit_col] - time_to_event[first_visit_col]).dt.total_seconds() / 86400
    )
    protocol_origin = (
        visits.groupby(visit_subject_col, dropna=False)[source_protocol_col]
        .apply(_format_protocol_origin)
        .rename("protocol_origin")
        .reset_index()
    )
    time_to_event = time_to_event.merge(
        protocol_origin,
        left_on=master_subject_col,
        right_on=visit_subject_col,
        how="left",
    )
    if visit_subject_col in time_to_event.columns and visit_subject_col != master_subject_col:
        time_to_event = time_to_event.drop(columns=[visit_subject_col])
    if "n_protocols" in time_to_event.columns:
        time_to_event = time_to_event.drop(columns=["n_protocols"])

    baseline, baseline_subject_col = _ensure_category_prefixed_subject(baseline, "baseline")
    longitudinal, longitudinal_subject_col = _ensure_category_prefixed_subject(longitudinal, "longitudinal")
    time_to_event, tte_subject_col = _ensure_category_prefixed_subject(time_to_event, "time_to_event")

    print_step(3, "Save cohort outputs and analysis readiness table")
    save_parquet_and_csv(baseline, ANALYTIC_DIR / "cohort_baseline", logger)
    save_parquet_and_csv(longitudinal, ANALYTIC_DIR / "cohort_longitudinal", logger)
    save_parquet_and_csv(time_to_event, ANALYTIC_DIR / "cohort_time_to_event", logger)

    readiness = pd.DataFrame(
        [
            {
                "question": "unique_patients_after_dedup",
                "value": int(master[master_subject_col].nunique(dropna=True)),
            },
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
            "baseline_subject_col": baseline_subject_col,
            "longitudinal_subject_col": longitudinal_subject_col,
            "time_to_event_subject_col": tte_subject_col,
        },
    )
    print_step(4, "Append targeted EDA for cohort outputs to unified workbook")
    sheets = {}
    sheets.update(
        build_targeted_eda_sheets(
            baseline,
            "07_cohort_baseline_output",
            "07_cohort_baseline_output",
            consolidated=True,
        )
    )
    sheets.update(
        build_targeted_eda_sheets(
            longitudinal,
            "07_cohort_longitudinal_output",
            "07_cohort_longitudinal_output",
            consolidated=True,
        )
    )
    sheets.update(
        build_targeted_eda_sheets(
            time_to_event,
            "07_cohort_time_to_event_output",
            "07_cohort_time_to_event_output",
            consolidated=True,
        )
    )
    workbook = upsert_eda_sheets_xlsx(EDA_UNIFIED_REPORT_PATH, sheets)
    logger.info("Updated unified EDA workbook: %s", workbook)


if __name__ == "__main__":
    main()
