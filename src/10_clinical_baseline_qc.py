"""Diagnose differences between first activity and first clinical baseline.

This reporting-only step consumes the finalized 09d episode table. It does not
alter episode membership, identifiers, or any analytical/cohort dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import ANALYTIC_DIR, REPORTS_DIR, setup_logger

INPUT_PATH = (
    ANALYTIC_DIR
    / "visits_long_collapsed_by_clinical_episode_codebook_corrected.parquet"
)
OUTPUT_DIR = REPORTS_DIR / "clinical_baseline"
COMPARISON_PATH = OUTPUT_DIR / "10_clinical_baseline_comparison.csv"
SUMMARY_PATH = OUTPUT_DIR / "10_clinical_baseline_qc_summary.csv"
COMPARISON_SJD_PATH = OUTPUT_DIR / "10_clinical_baseline_comparison_sjd.csv"
SUMMARY_SJD_PATH = OUTPUT_DIR / "10_clinical_baseline_qc_summary_sjd.csv"
KEYS = ["patient_id", "clinical_episode_id"]
ESSDAI_TOTAL_COL = "essdai__essdai_total_score"
ESSPRI_COMPONENT_COLS = (
    "esspri_questionnaire__dryness",
    "esspri_questionnaire__fatigue",
    "esspri_questionnaire__pain",
)
SJD_CLASS_COL = "visit_summary_form__sjogrens_class"
SJD_TARGET_CLASSES = {"1", "2", "4"}
EMPTY_LIKE_LITERALS = {"", "nan", "none", "null", "na", "n/a"}
REQUIRED_COLUMNS = {
    *KEYS,
    "episode_start_date",
    "clinical_anchor_date",
    "visit_type",
    "clinical_visit",
    ESSDAI_TOTAL_COL,
    *ESSPRI_COMPONENT_COLS,
    SJD_CLASS_COL,
}
SHIFT_REASONS = (
    "same_date",
    "same_episode_anchor_later",
    "first_episode_research_only",
    "first_episode_ambiguous",
    "later_clinical_episode",
    "no_clinical_baseline",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line input and report paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, default=INPUT_PATH)
    parser.add_argument("--comparison-path", type=Path, default=COMPARISON_PATH)
    parser.add_argument("--summary-path", type=Path, default=SUMMARY_PATH)
    parser.add_argument(
        "--comparison-sjd-path", type=Path, default=COMPARISON_SJD_PATH
    )
    parser.add_argument("--summary-sjd-path", type=Path, default=SUMMARY_SJD_PATH)
    return parser.parse_args()


def _as_bool(series: pd.Series) -> pd.Series:
    """Convert common serialized boolean representations to strict booleans."""
    true_values = {"true", "1", "yes", "y", "t"}
    return series.fillna(False).map(
        lambda value: (
            value if isinstance(value, bool) else str(value).lower() in true_values
        )
    )


def _has_value(series: pd.Series) -> pd.Series:
    """Identify values that are neither missing nor serialized missing literals."""
    return series.notna() & ~series.astype("string").str.strip().str.casefold().isin(
        EMPTY_LIKE_LITERALS
    )


def _sjogrens_class_tokens(value: object) -> list[str]:
    """Extract normalized Sjögren class tokens using the longitudinal 09b logic."""
    if pd.isna(value):
        return []
    tokens: list[str] = []
    for raw_token in str(value).split("|"):
        token = raw_token.strip()
        if token.casefold() in EMPTY_LIKE_LITERALS:
            continue
        numeric = pd.to_numeric(token, errors="coerce")
        if pd.notna(numeric) and float(numeric).is_integer():
            token = str(int(numeric))
        tokens.append(token)
    return tokens


def _patient_sjogrens_classes(episodes: pd.DataFrame) -> pd.DataFrame:
    """Summarize all observed Sjögren classes and ever-1/2/4 status by patient."""
    rows: list[dict[str, object]] = []
    for patient_id, values in episodes.groupby("patient_id", sort=False)[SJD_CLASS_COL]:
        class_values: list[str] = []
        seen: set[str] = set()
        for value in values:
            for token in _sjogrens_class_tokens(value):
                if token not in seen:
                    seen.add(token)
                    class_values.append(token)
        rows.append(
            {
                "patient_id": patient_id,
                "sjd_ever_1_2_4": bool(seen & SJD_TARGET_CLASSES),
                "sjogrens_class_patient_values": (
                    " | ".join(class_values) if class_values else pd.NA
                ),
            }
        )
    return pd.DataFrame(rows)


def validate_input(episodes: pd.DataFrame) -> None:
    """Validate the finalized 09d table without changing its architecture."""
    missing = REQUIRED_COLUMNS.difference(episodes.columns)
    if missing:
        raise ValueError(f"09d input missing required columns: {sorted(missing)}")
    if episodes[KEYS].isna().any().any():
        raise ValueError("09d input contains missing patient or episode identifiers")
    if episodes.duplicated(KEYS).any():
        raise ValueError("09d input contains duplicate patient/episode identifiers")


def _shift_reason(row: pd.Series) -> str:
    """Assign one simple, mutually exclusive baseline-shift reason."""
    if pd.isna(row["clinical_baseline_episode_id"]):
        return "no_clinical_baseline"
    if row["first_recorded_episode_id"] == row["clinical_baseline_episode_id"]:
        if row["first_recorded_date"] == row["clinical_baseline_date"]:
            return "same_date"
        return "same_episode_anchor_later"
    if row["first_recorded_visit_type"] == "research_or_procedure_only_candidate":
        return "first_episode_research_only"
    if row["first_recorded_visit_type"] == "ambiguous":
        return "first_episode_ambiguous"
    return "later_clinical_episode"


def build_comparison(episodes: pd.DataFrame) -> pd.DataFrame:
    """Build one diagnostic baseline-comparison row per patient.

    Parameters
    ----------
    episodes : pd.DataFrame
        Finalized 09d table with one row per authoritative clinical episode.

    Returns
    -------
    pd.DataFrame
        Patient-level comparison of first recorded and first clinical episodes.
    """
    validate_input(episodes)
    work = episodes.copy()
    work["episode_start_date"] = pd.to_datetime(
        work["episode_start_date"], errors="coerce"
    ).dt.normalize()
    work["clinical_anchor_date"] = pd.to_datetime(
        work["clinical_anchor_date"], errors="coerce"
    ).dt.normalize()
    if work["episode_start_date"].isna().any():
        raise ValueError("09d input contains an episode without episode_start_date")

    work["_clinical"] = _as_bool(work["clinical_visit"])
    work["_has_essdai"] = _has_value(work[ESSDAI_TOTAL_COL])
    work["_essdai_total"] = pd.to_numeric(work[ESSDAI_TOTAL_COL], errors="coerce")
    work["_has_esspri"] = pd.concat(
        [_has_value(work[column]) for column in ESSPRI_COMPONENT_COLS], axis=1
    ).all(axis=1)
    work["_manual_review"] = _as_bool(
        work.get("manual_review_required", pd.Series(False, index=work.index))
    )
    ordered = work.sort_values(
        ["patient_id", "episode_start_date", "clinical_episode_id"]
    )

    first = ordered.groupby("patient_id", sort=False).head(1).set_index("patient_id")
    clinical = ordered.loc[
        ordered["_clinical"] & ordered["clinical_anchor_date"].notna()
    ]
    baseline = (
        clinical.groupby("patient_id", sort=False).head(1).set_index("patient_id")
    )

    comparison = pd.DataFrame(index=first.index)
    comparison["first_recorded_episode_id"] = first["clinical_episode_id"]
    comparison["first_recorded_date"] = first["episode_start_date"]
    comparison["first_recorded_visit_type"] = first["visit_type"]
    comparison["clinical_baseline_episode_id"] = baseline["clinical_episode_id"]
    comparison["clinical_baseline_date"] = baseline["clinical_anchor_date"]
    comparison["baseline_visit_type"] = baseline["visit_type"]
    comparison["days_shifted"] = (
        comparison["clinical_baseline_date"] - comparison["first_recorded_date"]
    ).dt.days.astype("Int64")
    comparison["baseline_shifted"] = comparison["days_shifted"].gt(0).astype("boolean")
    comparison.loc[comparison["days_shifted"].isna(), "baseline_shifted"] = pd.NA
    comparison["reason_for_shift"] = comparison.apply(_shift_reason, axis=1)
    comparison["baseline_has_essdai"] = (
        baseline["_has_essdai"].fillna(False).astype(bool)
    )
    comparison["baseline_has_esspri"] = (
        baseline["_has_esspri"].fillna(False).astype(bool)
    )
    baseline_essdai_total = baseline["_essdai_total"]
    comparison["baseline_pop_classifiable"] = baseline_essdai_total.ge(5) | (
        baseline_essdai_total.lt(5) & comparison["baseline_has_esspri"]
    )
    comparison["baseline_manual_review_required"] = (
        baseline["_manual_review"].fillna(False).astype(bool)
        | comparison["clinical_baseline_episode_id"].isna()
    )
    comparison.loc[
        comparison["clinical_baseline_episode_id"].isna(),
        "baseline_manual_review_required",
    ] = True
    assert comparison.loc[
        comparison["clinical_baseline_episode_id"].isna(),
        "baseline_manual_review_required",
    ].all()
    comparison = comparison.reset_index().merge(
        _patient_sjogrens_classes(work), on="patient_id", how="left", validate="one_to_one"
    )
    hard_qc(episodes, comparison)
    return comparison


def hard_qc(episodes: pd.DataFrame, comparison: pd.DataFrame) -> None:
    """Enforce patient ownership, chronology, cardinality, and ID preservation."""
    if comparison["patient_id"].duplicated().any():
        raise ValueError("hard QC: a patient has more than one clinical baseline")
    ownership = set(map(tuple, episodes[KEYS].astype(str).to_numpy()))
    selected = comparison.dropna(subset=["clinical_baseline_episode_id"])
    selected_keys = set(
        map(
            tuple,
            selected[["patient_id", "clinical_baseline_episode_id"]]
            .astype(str)
            .to_numpy(),
        )
    )
    if not selected_keys.issubset(ownership):
        raise ValueError("hard QC: a clinical baseline belongs to another patient")
    invalid_dates = selected["clinical_baseline_date"].lt(
        selected["first_recorded_date"]
    )
    if invalid_dates.any():
        raise ValueError("hard QC: clinical baseline precedes first recorded activity")
    if comparison["patient_id"].nunique() != episodes["patient_id"].nunique():
        raise ValueError("hard QC: patients were added or removed")


def build_summary(
    comparison: pd.DataFrame, cohort_comparison: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Build a one-row QC summary, including shift-reason counts."""
    report = comparison if cohort_comparison is None else cohort_comparison
    shifted_days = comparison.loc[
        comparison["baseline_shifted"].fillna(False), "days_shifted"
    ]
    if cohort_comparison is not None:
        shifted_days = report.loc[
            report["baseline_shifted"].fillna(False), "days_shifted"
        ]
    has_baseline = report["clinical_baseline_episode_id"].notna()
    both = report["baseline_has_essdai"] & report["baseline_has_esspri"]
    n_patients = len(report)
    values: dict[str, object] = {
        "n_patients": n_patients,
        "n_patients_total": len(comparison),
        "n_patients_sjd_ever_1_2_4": int(comparison["sjd_ever_1_2_4"].sum()),
        "n_patients_not_sjd_ever_1_2_4": int((~comparison["sjd_ever_1_2_4"]).sum()),
        "n_patients_without_sjogrens_class_information": int(
            comparison["sjogrens_class_patient_values"].isna().sum()
        ),
        "n_with_clinical_baseline": int(has_baseline.sum()),
        "n_same_baseline_date": int(report["days_shifted"].eq(0).sum()),
        "n_shifted_baseline": int(report["baseline_shifted"].fillna(False).sum()),
        "pct_shifted_baseline": (
            100 * report["baseline_shifted"].fillna(False).sum() / n_patients
            if n_patients
            else 0.0
        ),
        "median_shift_days": shifted_days.median(),
        "max_shift_days": shifted_days.max(),
        "n_baseline_with_essdai": int(report["baseline_has_essdai"].sum()),
        "n_baseline_with_esspri": int(report["baseline_has_esspri"].sum()),
        "n_baseline_with_both": int(both.sum()),
        "n_baseline_pop_classifiable": int(
            report["baseline_pop_classifiable"].sum()
        ),
        "n_clinical_but_not_pop_classifiable": int(
            (has_baseline & ~report["baseline_pop_classifiable"]).sum()
        ),
        "n_without_clinical_baseline": int((~has_baseline).sum()),
        "n_baseline_manual_review": int(
            report["baseline_manual_review_required"].sum()
        ),
    }
    reason_counts = report["reason_for_shift"].value_counts()
    for reason in SHIFT_REASONS:
        values[f"n_reason_{reason}"] = int(reason_counts.get(reason, 0))
    return pd.DataFrame([values])


def main() -> None:
    """Read finalized episodes and write diagnostic CSV reports only."""
    args = parse_args()
    logger = setup_logger("10_clinical_baseline_qc")
    logger.info("Reading finalized 09d episodes from %s", args.input_path)
    episodes = pd.read_parquet(args.input_path)
    logger.info("Loaded episodes rows=%d cols=%d", len(episodes), len(episodes.columns))
    comparison = build_comparison(episodes)
    summary = build_summary(comparison)
    comparison_sjd = comparison.loc[comparison["sjd_ever_1_2_4"]].copy()
    summary_sjd = build_summary(comparison, comparison_sjd)
    args.comparison_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.comparison_sjd_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_sjd_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.comparison_path, index=False)
    summary.to_csv(args.summary_path, index=False)
    comparison_sjd.to_csv(args.comparison_sjd_path, index=False)
    summary_sjd.to_csv(args.summary_sjd_path, index=False)
    logger.info("Saved %s rows=%d", args.comparison_path, len(comparison))
    logger.info("Saved %s rows=%d", args.summary_path, len(summary))
    logger.info("Saved %s rows=%d", args.comparison_sjd_path, len(comparison_sjd))
    logger.info("Saved %s rows=%d", args.summary_sjd_path, len(summary_sjd))


if __name__ == "__main__":
    main()
