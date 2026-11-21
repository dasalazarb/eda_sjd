"""Summarize longitudinal visit follow-up for the 11D and 15D protocols."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
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

PROTOCOLS = ("11D", "15D")
SJD_ELIGIBLE_VALUES = {1, 2, 4}
DEFAULT_INPUT_CANDIDATES = [
    ANALYTIC_DIR / "visits_long_collapsed_by_interval_codebook_corrected.parquet",
    ANALYTIC_DIR / "visits_long_collapsed_by_interval.parquet",
    ANALYTIC_DIR / "visits_long.parquet",
]
RETENTION_THRESHOLDS = {
    "6 months": 182,
    "1 year": 365,
    "2 years": 730,
    "3 years": 1095,
    "5 years": 1826,
    "10 years": 3652,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Describe longitudinal visit follow-up in 11D, 15D, and unique patients."
    )
    parser.add_argument("--input-path", type=Path, default=None)
    parser.add_argument(
        "--output-dir", type=Path, default=REPORTS_DIR / "followup_summary"
    )
    return parser.parse_args()


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xls", ".xlsx"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported input extension: {path}")


def _choose_input_path(explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        if not explicit_path.exists():
            raise FileNotFoundError(f"Input not found: {explicit_path}")
        return explicit_path
    for path in DEFAULT_INPUT_CANDIDATES:
        if path.exists():
            return path
    checked = "\n".join(f"- {path}" for path in DEFAULT_INPUT_CANDIDATES)
    raise FileNotFoundError(f"No default visit-level input found. Checked:\n{checked}")


def _resolve_optional_column(df: pd.DataFrame, names: Iterable[str]) -> str | None:
    for name in names:
        try:
            return resolve_canonical_column(df, name)
        except KeyError:
            continue
    return None


def _normalize_protocol(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.upper()


def _protocol_mask(df: pd.DataFrame, protocol_col: str, protocol: str) -> pd.Series:
    values = _normalize_protocol(df[protocol_col]).fillna("")
    return values.str.split(r"\s*\|\s*", regex=True).apply(
        lambda parts: protocol in set(parts)
    )


def _eligible_sjd_patient_set(
    visits: pd.DataFrame, patient_col: str, classification_col: str
) -> set:
    """Return patients classified as SjD (1, 2, or 4) at least once."""

    def has_eligible_value(value: object) -> bool:
        if pd.isna(value):
            return False
        components = [part.strip() for part in str(value).split("|")]
        for component in components:
            numeric = pd.to_numeric(component, errors="coerce")
            if not pd.isna(numeric) and numeric in SJD_ELIGIBLE_VALUES:
                return True
        return False

    eligible_rows = visits[classification_col].apply(has_eligible_value)
    return set(visits.loc[eligible_rows, patient_col].dropna())


def _clean_visit_date(value: object) -> tuple[pd.Timestamp | pd.NaT, bool, str]:
    """Return the earliest valid component of a possibly concatenated visit date."""
    if pd.isna(value):
        return pd.NaT, False, "missing_date"
    text = str(value).strip()
    had_multiple = " | " in text
    parts = [part.strip() for part in text.split(" | ")] if had_multiple else [text]
    parsed = [pd.to_datetime(part, errors="coerce") for part in parts]
    valid = [date for date in parsed if not pd.isna(date)]
    if not valid:
        return pd.NaT, had_multiple, "unparseable_date"
    reason = "multiple_values_earliest_used" if had_multiple else ""
    if had_multiple and len(valid) < len(parts):
        reason += ";invalid_component_discarded"
    return min(valid), had_multiple, reason


def _prepare_visits(
    df: pd.DataFrame, patient_col: str, protocol_col: str, date_col: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    visits = df.copy()
    visits["_source_row"] = np.arange(len(visits))
    visits["visit_date_original"] = visits[date_col]
    cleaned = visits[date_col].apply(_clean_visit_date)
    # Extracting Timestamp objects from tuple results leaves an object-dtype
    # Series in recent pandas versions. Convert the already-cleaned scalar
    # dates explicitly so downstream subtraction always supports ``.dt``.
    visits["visit_date"] = pd.to_datetime(
        cleaned.str[0], errors="coerce", utc=True
    ).dt.tz_localize(None)
    visits["visit_date_had_multiple_values"] = cleaned.str[1]
    visits["visit_date_issue"] = cleaned.str[2]
    visits["patient_record_number"] = visits[patient_col]
    visits["protocol_membership"] = _normalize_protocol(visits[protocol_col])

    special = visits.loc[
        visits["visit_date_had_multiple_values"] | visits["visit_date_issue"].ne(""),
        [
            "patient_record_number",
            "visit_date_original",
            "visit_date",
            "visit_date_had_multiple_values",
            "visit_date_issue",
        ],
    ].rename(columns={"visit_date_issue": "reason"})
    return visits, special


def _deduplicate_visits(visits: pd.DataFrame, include_protocol: bool) -> pd.DataFrame:
    """Remove only clear duplicates sharing patient, date, and available visit label."""
    interval_col = _resolve_optional_column(visits, ("interval_name", "interval_code"))
    valid = visits[visits["visit_date"].notna()].copy()
    missing = visits[visits["visit_date"].isna()].copy()
    keys = ["patient_record_number", "visit_date"]
    if include_protocol:
        keys.append("protocol_membership")
    if interval_col:
        keys.append(interval_col)
    valid = valid.drop_duplicates(keys, keep="first")
    return pd.concat([valid, missing], ignore_index=True)


def _build_intervisit_gaps(visits: pd.DataFrame) -> pd.DataFrame:
    dated = visits[visits["visit_date"].notna()].copy()
    dated = dated.sort_values(["patient_record_number", "visit_date", "_source_row"])
    dated["previous_visit_date"] = dated.groupby("patient_record_number")[
        "visit_date"
    ].shift()
    gaps = dated[dated["previous_visit_date"].notna()].copy()
    gaps["gap_days"] = (gaps["visit_date"] - gaps["previous_visit_date"]).dt.days
    gaps["gap_order"] = gaps.groupby("patient_record_number").cumcount() + 1
    gaps["gap_zero_days"] = gaps["gap_days"].eq(0)
    gaps["gap_negative"] = gaps["gap_days"].lt(0)
    return gaps[
        [
            "patient_record_number",
            "protocol_membership",
            "previous_visit_date",
            "visit_date",
            "gap_days",
            "gap_order",
            "gap_zero_days",
            "gap_negative",
        ]
    ]


def _build_patient_followup_metrics(
    visits: pd.DataFrame, protocol_sets: dict[str, set]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    deduped = _deduplicate_visits(visits, include_protocol=False)
    dated = deduped[deduped["visit_date"].notna()]
    dates = dated.groupby("patient_record_number")["visit_date"].agg(["min", "max"])
    counts = deduped.groupby("patient_record_number").size().rename("n_visits")
    metrics = counts.to_frame().join(
        dates.rename(columns={"min": "first_visit_date", "max": "last_visit_date"})
    )
    metrics["followup_days"] = (
        metrics["last_visit_date"] - metrics["first_visit_date"]
    ).dt.days
    metrics["followup_years"] = metrics["followup_days"] / 365.25

    gaps = _build_intervisit_gaps(deduped)
    valid_gaps = gaps.loc[~gaps["gap_negative"]]
    gap_stats = valid_gaps.groupby("patient_record_number")["gap_days"].agg(
        median_gap_days="median", max_gap_days="max"
    )
    metrics = metrics.join(gap_stats)
    metrics["visits_per_year"] = np.where(
        metrics["followup_years"] > 0,
        (metrics["n_visits"] - 1) / metrics["followup_years"],
        np.nan,
    )
    metrics["in_protocol_11d"] = metrics.index.isin(protocol_sets["11D"])
    metrics["in_protocol_15d"] = metrics.index.isin(protocol_sets["15D"])
    for label, days in RETENTION_THRESHOLDS.items():
        suffix = {
            "6 months": "6mo",
            "1 year": "1yr",
            "2 years": "2yr",
            "3 years": "3yr",
            "5 years": "5yr",
            "10 years": "10yr",
        }[label]
        metrics[f"has_followup_{suffix}"] = metrics["followup_days"].ge(days)
    for days in (180, 365, 730):
        patients = set(
            valid_gaps.loc[valid_gaps["gap_days"] > days, "patient_record_number"]
        )
        metrics[f"has_gap_over_{days}d"] = metrics.index.isin(patients)
    metrics.index.name = "patient_record_number"
    return metrics.reset_index(), gaps


def _cohort_metrics(
    visits: pd.DataFrame, patients: pd.DataFrame, gaps: pd.DataFrame
) -> dict[str, object]:
    denominator = len(patients)

    def n_pct(mask: pd.Series) -> str:
        n = int(mask.fillna(False).sum())
        return (
            f"{n:,} (NA)"
            if denominator == 0
            else f"{n:,} ({100 * n / denominator:.1f}%)"
        )

    def median_iqr(series: pd.Series, decimals: int = 1) -> str:
        clean = pd.to_numeric(series, errors="coerce").dropna()
        if clean.empty:
            return "NA"
        q1, median, q3 = clean.quantile([0.25, 0.5, 0.75])
        return f"{median:.{decimals}f} ({q1:.{decimals}f}–{q3:.{decimals}f})"

    followup = patients["followup_years"]
    valid_gaps = gaps.loc[gaps["gap_days"].notna() & ~gaps["gap_negative"], "gap_days"]
    result: dict[str, object] = {
        "Raw records": len(visits),
        "Unique patients": denominator,
        "Patients with exactly 1 visit": n_pct(patients["n_visits"].eq(1)),
        "Patients with >=2 visits": n_pct(patients["n_visits"].ge(2)),
        "Patients with >=3 visits": n_pct(patients["n_visits"].ge(3)),
        "Patients with >=5 visits": n_pct(patients["n_visits"].ge(5)),
        "Patients with >=10 visits": n_pct(patients["n_visits"].ge(10)),
        "Follow-up, median (IQR), years": median_iqr(followup),
        "Visits per patient, median (IQR)": median_iqr(patients["n_visits"], 0),
        "Visits per patient, mean": (
            round(patients["n_visits"].mean(), 1) if denominator else "NA"
        ),
        "Maximum visits per patient": (
            int(patients["n_visits"].max()) if denominator else "NA"
        ),
        "Median inter-visit gap, days": (
            round(valid_gaps.median(), 1) if not valid_gaps.empty else "NA"
        ),
        "IQR inter-visit gap, days": (
            f"{valid_gaps.quantile(0.25):.1f}–{valid_gaps.quantile(0.75):.1f}"
            if not valid_gaps.empty
            else "NA"
        ),
        "P90 inter-visit gap, days": (
            round(valid_gaps.quantile(0.90), 1) if not valid_gaps.empty else "NA"
        ),
        "Visits per follow-up year, median (IQR)": median_iqr(
            patients["visits_per_year"]
        ),
    }
    for percentile in (0.10, 0.25, 0.50, 0.75, 0.90):
        result[f"Follow-up P{int(percentile * 100)}, years"] = (
            round(followup.quantile(percentile), 1) if followup.notna().any() else "NA"
        )
    result["Maximum follow-up, years"] = (
        round(followup.max(), 1) if followup.notna().any() else "NA"
    )
    suffixes = ("6mo", "1yr", "2yr", "3yr", "5yr", "10yr")
    for (label, _), suffix in zip(RETENTION_THRESHOLDS.items(), suffixes):
        result[f"Follow-up >={label}"] = n_pct(patients[f"has_followup_{suffix}"])
    for days in (180, 365, 730):
        result[f"Patients with at least one gap >{days} days"] = n_pct(
            patients[f"has_gap_over_{days}d"]
        )
    return result


def _build_summary_table(
    cohort_data: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = {name: _cohort_metrics(*data) for name, data in cohort_data.items()}
    indicators = list(columns["Protocol 11D"])
    wide = pd.DataFrame(
        {
            "Indicator": indicators,
            **{
                name: [values[i] for i in indicators]
                for name, values in columns.items()
            },
        }
    )
    long = wide.melt(id_vars="Indicator", var_name="cohort", value_name="value")
    return wide, long


def _build_retention_table(cohorts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for label, days in RETENTION_THRESHOLDS.items():
        row: dict[str, object] = {"followup_threshold": label, "threshold_days": days}
        for output_name, patients in cohorts.items():
            n = int(patients["followup_days"].ge(days).sum())
            row[f"{output_name}_n"] = n
            row[f"{output_name}_pct"] = (
                100 * n / len(patients) if len(patients) else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _save_empty_plot(path: Path, title: str) -> None:
    _, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(title)
    ax.text(0.5, 0.5, "No data available", ha="center", va="center")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_followup_distribution(cohorts: dict[str, pd.DataFrame], path: Path) -> None:
    if not any(df["followup_years"].notna().any() for df in cohorts.values()):
        return _save_empty_plot(path, "Follow-up distribution")
    _, ax = plt.subplots(figsize=(8, 5))
    for label, df in cohorts.items():
        values = df["followup_years"].dropna()
        if not values.empty:
            ax.hist(values, bins=20, alpha=0.5, label=label)
    ax.set(
        xlabel="Follow-up (years)", ylabel="Patients", title="Follow-up distribution"
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_retention(cohorts: dict[str, pd.DataFrame], path: Path) -> None:
    if not any(df["followup_days"].notna().any() for df in cohorts.values()):
        return _save_empty_plot(path, "Descriptive follow-up retention curve")
    _, ax = plt.subplots(figsize=(8, 5))
    observed = pd.concat(
        [df["followup_years"].dropna() for df in cohorts.values()], ignore_index=True
    )
    max_years = observed.max() if not observed.empty else 0
    grid = np.linspace(0, max(1, max_years), 100)
    for label, df in cohorts.items():
        values = df["followup_years"].dropna()
        if not values.empty:
            ax.plot(
                grid,
                [100 * values.ge(year).sum() / len(df) for year in grid],
                label=label,
            )
    ax.set(
        xlabel="Years from first visit",
        ylabel="Patients with follow-up >= time (%)",
        title="Descriptive follow-up retention curve (not Kaplan-Meier)",
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_visits_per_patient(cohorts: dict[str, pd.DataFrame], path: Path) -> None:
    labels = ["1 visit", "2 visits", "3–4 visits", "5–9 visits", ">=10 visits"]
    bins = [0, 1, 2, 4, 9, np.inf]
    x = np.arange(len(labels))
    width = 0.35
    _, ax = plt.subplots(figsize=(9, 5))
    for offset, (name, df) in zip((-width / 2, width / 2), cohorts.items()):
        groups = (
            pd.cut(df["n_visits"], bins=bins, labels=labels)
            .value_counts()
            .reindex(labels, fill_value=0)
        )
        ax.bar(x + offset, groups, width, label=name)
    ax.set_xticks(x, labels)
    ax.set(ylabel="Patients", title="Visits per patient")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_followup_vs_visits(cohorts: dict[str, pd.DataFrame], path: Path) -> None:
    _, ax = plt.subplots(figsize=(8, 5))
    plotted = False
    for label, df in cohorts.items():
        clean = df.dropna(subset=["followup_years", "n_visits"])
        if not clean.empty:
            ax.scatter(
                clean["followup_years"], clean["n_visits"], alpha=0.55, label=label
            )
            plotted = True
    if not plotted:
        plt.close()
        return _save_empty_plot(path, "Follow-up vs visits")
    ax.set(
        xlabel="Follow-up (years)",
        ylabel="Number of visits",
        title="Follow-up vs visits",
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_swimmer(visits: pd.DataFrame, metrics: pd.DataFrame, path: Path) -> None:
    dated = visits[visits["visit_date"].notna()].copy()
    if dated.empty:
        return _save_empty_plot(path, "Longitudinal visit follow-up")
    order = metrics.sort_values(["n_visits", "followup_days"], ascending=False)[
        "patient_record_number"
    ].tolist()
    positions = {patient: index for index, patient in enumerate(order)}
    dated = dated[dated["patient_record_number"].isin(positions)].merge(
        metrics[["patient_record_number", "first_visit_date"]],
        on="patient_record_number",
    )
    dated["days_from_first"] = (dated["visit_date"] - dated["first_visit_date"]).dt.days
    height = min(18, max(6, len(order) * 0.08))
    _, ax = plt.subplots(figsize=(10, height))
    for patient, rows in dated.groupby("patient_record_number"):
        y = positions[patient]
        days = rows["days_from_first"]
        ax.plot([days.min(), days.max()], [y, y], color="0.7", linewidth=0.8)
        ax.scatter(days, np.full(len(days), y), s=8)
    ax.set(
        xlabel="Days from first visit",
        ylabel="Patients (ordered)",
        title="Longitudinal visit follow-up",
    )
    ax.set_yticks([])
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    args = _parse_args()
    logger = setup_logger("24_protocol_flow_table")
    print_script_overview(
        "24_protocol_flow_table.py",
        "Summarizes longitudinal visit follow-up by protocol.",
    )

    print_step(1, "Load and prepare visit-level data")
    input_path = _choose_input_path(args.input_path)
    source = _read_table(input_path)
    if source.empty:
        raise ValueError(f"Input dataset is empty: {input_path}")
    patient_col = resolve_canonical_column(source, "patient_record_number")
    protocol_col = resolve_canonical_column(source, "source_protocol")
    # Prefer the original visit-date field so concatenated values remain visible
    # to the explicit component-by-component cleaning below.
    date_col = _resolve_optional_column(source, ("visit_date", "visit_datetime"))
    if date_col is None:
        raise KeyError("Could not identify a visit_datetime or visit_date column")
    visits, special_dates = _prepare_visits(source, patient_col, protocol_col, date_col)

    classification_col = _resolve_optional_column(
        visits,
        ("visit_summary_form__sjogrens_class", "sjogrens_class"),
    )
    if classification_col is None:
        raise KeyError(
            "Could not identify the visit_summary_form__sjogrens_class column"
        )
    eligible_patients = _eligible_sjd_patient_set(
        visits, "patient_record_number", classification_col
    )
    visits = visits[visits["patient_record_number"].isin(eligible_patients)].copy()
    special_dates = special_dates[
        special_dates["patient_record_number"].isin(eligible_patients)
    ].copy()
    if visits.empty:
        raise ValueError(
            "No patients had visit_summary_form__sjogrens_class equal to 1, 2, or 4"
        )
    logger.info(
        "Restricted follow-up cohort to %d patients ever classified as 1, 2, or 4",
        len(eligible_patients),
    )
    protocol_sets = {
        protocol: set(
            visits.loc[
                _protocol_mask(visits, "protocol_membership", protocol),
                "patient_record_number",
            ].dropna()
        )
        for protocol in PROTOCOLS
    }

    print_step(2, "Calculate patient follow-up, gaps, retention, and summaries")
    patient_metrics, total_gaps = _build_patient_followup_metrics(visits, protocol_sets)
    protocol_results = {}
    protocol_patients = {}
    for protocol in PROTOCOLS:
        protocol_visits = visits[
            _protocol_mask(visits, "protocol_membership", protocol)
        ].copy()
        protocol_metrics, protocol_gaps = _build_patient_followup_metrics(
            protocol_visits, protocol_sets
        )
        protocol_patients[protocol] = protocol_metrics
        protocol_results[f"Protocol {protocol}"] = (
            protocol_visits,
            protocol_metrics,
            protocol_gaps,
        )
    protocol_results["Unique total"] = (visits, patient_metrics, total_gaps)
    summary, summary_long = _build_summary_table(protocol_results)
    retention = _build_retention_table(
        {
            "11D": protocol_patients["11D"],
            "15D": protocol_patients["15D"],
            "total": patient_metrics,
        }
    )

    print_step(3, "Save follow-up tables and figures")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "summary": args.output_dir / "followup_summary.csv",
        "summary_long": args.output_dir / "followup_summary_long.csv",
        "patient_metrics": args.output_dir / "patient_followup_metrics.csv",
        "retention": args.output_dir / "retention_by_time.csv",
        "gaps": args.output_dir / "intervisit_gaps.csv",
        "special_dates": args.output_dir / "visit_date_special_cases.csv",
        "xlsx": args.output_dir / "followup_summary.xlsx",
    }
    for table, key in (
        (summary, "summary"),
        (summary_long, "summary_long"),
        (patient_metrics, "patient_metrics"),
        (retention, "retention"),
        (total_gaps, "gaps"),
        (special_dates, "special_dates"),
    ):
        table.to_csv(outputs[key], index=False)
    with pd.ExcelWriter(outputs["xlsx"]) as writer:
        summary.to_excel(writer, sheet_name="followup_summary", index=False)
        summary_long.to_excel(writer, sheet_name="followup_summary_long", index=False)
        patient_metrics.to_excel(writer, sheet_name="patient_metrics", index=False)
        retention.to_excel(writer, sheet_name="retention", index=False)
        total_gaps.to_excel(writer, sheet_name="intervisit_gaps", index=False)
        special_dates.to_excel(writer, sheet_name="date_special_cases", index=False)

    plot_cohorts = {"11D": protocol_patients["11D"], "15D": protocol_patients["15D"]}
    _plot_followup_distribution(
        plot_cohorts, args.output_dir / "followup_distribution.png"
    )
    _plot_retention(
        {**plot_cohorts, "Unique total": patient_metrics},
        args.output_dir / "retention_curve.png",
    )
    _plot_visits_per_patient(plot_cohorts, args.output_dir / "visits_per_patient.png")
    _plot_followup_vs_visits(plot_cohorts, args.output_dir / "followup_vs_visits.png")
    _plot_swimmer(
        _deduplicate_visits(visits, False),
        patient_metrics,
        args.output_dir / "swimmer_followup.png",
    )
    logger.info("Saved follow-up outputs to %s", args.output_dir)
    print_kv("Follow-up outputs", {"input_path": input_path, **outputs})


if __name__ == "__main__":
    main()
