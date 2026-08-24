"""Build the definitive longitudinal spine from frozen clinical episodes.

This step only attaches patient-level cohort and clinical-baseline fields from
step 10. It does not reconstruct, combine, split, or otherwise modify the
clinical episodes finalized by step 09d.

Outputs
-------
data_analytic/clinical_episode_spine_all.parquet and .csv
    Complete clinical-episode spine for every patient.
data_analytic/clinical_episode_spine_sjd.parquet and .csv
    Clinical-episode spine restricted to patients in the SjD cohort.
reports/clinical_episode_spine/11_clinical_episode_spine_qc.parquet and .csv
    One-row quality-control summary confirming episode preservation and counts.
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
BASELINE_PATH = (
    REPORTS_DIR / "clinical_baseline" / "10_clinical_baseline_comparison.csv"
)
OUTPUT_ALL_PATH = ANALYTIC_DIR / "clinical_episode_spine_all.parquet"
OUTPUT_ALL_CSV_PATH = ANALYTIC_DIR / "clinical_episode_spine_all.csv"
OUTPUT_SJD_PATH = ANALYTIC_DIR / "clinical_episode_spine_sjd.parquet"
OUTPUT_SJD_CSV_PATH = ANALYTIC_DIR / "clinical_episode_spine_sjd.csv"
QC_PATH = REPORTS_DIR / "clinical_episode_spine" / "11_clinical_episode_spine_qc.csv"
QC_PARQUET_PATH = (
    REPORTS_DIR / "clinical_episode_spine" / "11_clinical_episode_spine_qc.parquet"
)
KEYS = ["patient_id", "clinical_episode_id"]
EPISODE_STRUCTURE_COLUMNS = {
    *KEYS,
    "episode_start_date",
    "clinical_anchor_date",
    "episode_end_date",
    "visit_type",
    "clinical_visit",
}
BASELINE_COLUMNS = [
    "patient_id",
    "sjd_ever_1_2_4",
    "sjogrens_class_patient_values",
    "clinical_baseline_episode_id",
    "clinical_baseline_date",
]
TRUE_VALUES = {"true", "1", "yes", "y", "t"}
FALSE_VALUES = {"false", "0", "no", "n", "f"}


def parse_args() -> argparse.Namespace:
    """Parse input and output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, default=INPUT_PATH)
    parser.add_argument("--baseline-path", type=Path, default=BASELINE_PATH)
    parser.add_argument("--output-all-path", type=Path, default=OUTPUT_ALL_PATH)
    parser.add_argument(
        "--output-all-csv-path", type=Path, default=OUTPUT_ALL_CSV_PATH
    )
    parser.add_argument("--output-sjd-path", type=Path, default=OUTPUT_SJD_PATH)
    parser.add_argument(
        "--output-sjd-csv-path", type=Path, default=OUTPUT_SJD_CSV_PATH
    )
    parser.add_argument("--qc-path", type=Path, default=QC_PATH)
    parser.add_argument("--qc-parquet-path", type=Path, default=QC_PARQUET_PATH)
    return parser.parse_args()


def _strict_bool(series: pd.Series, column_name: str) -> pd.Series:
    """Convert common serialized booleans and reject unrecognized values."""
    normalized = series.astype("string").str.strip().str.casefold()
    invalid = series.notna() & ~normalized.isin(TRUE_VALUES | FALSE_VALUES)
    if invalid.any():
        values = sorted(series.loc[invalid].astype(str).unique())
        raise ValueError(f"{column_name} contains invalid boolean values: {values}")
    result = normalized.isin(TRUE_VALUES)
    return result.astype(bool)


def validate_inputs(episodes: pd.DataFrame, baseline: pd.DataFrame) -> None:
    """Validate episode identity and the patient-level step-10 input."""
    missing_episode = EPISODE_STRUCTURE_COLUMNS.difference(episodes.columns)
    if missing_episode:
        raise ValueError(
            f"09d input missing required columns: {sorted(missing_episode)}"
        )
    missing_baseline = set(BASELINE_COLUMNS).difference(baseline.columns)
    if missing_baseline:
        raise ValueError(
            f"step-10 input missing required columns: {sorted(missing_baseline)}"
        )
    if episodes[KEYS].isna().any().any():
        raise ValueError("09d input contains missing patient or episode identifiers")
    if episodes.duplicated(KEYS).any():
        raise ValueError("09d input contains duplicate patient/episode identifiers")
    if baseline["patient_id"].isna().any():
        raise ValueError("step-10 input contains a missing patient_id")
    if baseline["patient_id"].duplicated().any():
        raise ValueError("step-10 input must contain exactly one row per patient")


def build_spines(
    episodes: pd.DataFrame, baseline: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach step-10 patient attributes and create all-patient and SjD spines.

    Parameters
    ----------
    episodes : pd.DataFrame
        Final step-09d table, with one row per frozen clinical episode.
    baseline : pd.DataFrame
        Step-10 comparison table, with one row per patient.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        The complete episode spine and its patient-level SjD subset.
    """
    validate_inputs(episodes, baseline)
    before_ids = set(episodes["clinical_episode_id"])
    baseline_fields = baseline[BASELINE_COLUMNS].copy()
    baseline_fields["sjd_ever_1_2_4"] = _strict_bool(
        baseline_fields["sjd_ever_1_2_4"], "sjd_ever_1_2_4"
    )
    spine_all = episodes.merge(
        baseline_fields, on="patient_id", how="left", validate="many_to_one"
    )
    if len(spine_all) != len(episodes):
        raise ValueError("hard QC: episode count changed during baseline merge")
    if set(spine_all["clinical_episode_id"]) != before_ids:
        raise ValueError("hard QC: clinical episode IDs changed during baseline merge")

    spine_all["sjd_ever_1_2_4"] = spine_all["sjd_ever_1_2_4"].fillna(False)
    spine_all["is_clinical_baseline"] = (
        spine_all["clinical_baseline_episode_id"].notna()
        & spine_all["clinical_episode_id"].eq(spine_all["clinical_baseline_episode_id"])
    ).astype(bool)
    hard_qc(spine_all)

    spine_sjd = spine_all.loc[spine_all["sjd_ever_1_2_4"]].copy()
    if not spine_sjd["sjd_ever_1_2_4"].all():
        raise ValueError("hard QC: non-SjD patient found in the SjD spine")
    return spine_all, spine_sjd


def hard_qc(spine: pd.DataFrame) -> None:
    """Enforce unique episodes and at most one clinical baseline per patient."""
    if spine.duplicated(KEYS).any():
        raise ValueError("hard QC: duplicate patient/clinical_episode_id pairs")
    baselines_per_patient = spine.groupby("patient_id")["is_clinical_baseline"].sum()
    if baselines_per_patient.gt(1).any():
        raise ValueError("hard QC: a patient has more than one clinical baseline")


def build_qc(
    episodes_before: pd.DataFrame,
    spine_all: pd.DataFrame,
    spine_sjd: pd.DataFrame,
) -> pd.DataFrame:
    """Build the requested one-row episode-spine QC report."""
    sjd_patients = spine_sjd["patient_id"].nunique()
    patients_with_baseline = spine_sjd.loc[
        spine_sjd["is_clinical_baseline"], "patient_id"
    ].nunique()
    visit_type = spine_sjd["visit_type"].astype("string")
    clinical = _strict_bool(spine_sjd["clinical_visit"], "clinical_visit")
    ids_preserved = set(episodes_before["clinical_episode_id"]) == set(
        spine_all["clinical_episode_id"]
    )
    return pd.DataFrame(
        [
            {
                "patients_all": spine_all["patient_id"].nunique(),
                "episodes_all": len(spine_all),
                "patients_sjd": sjd_patients,
                "episodes_sjd": len(spine_sjd),
                "patients_sjd_with_clinical_baseline": patients_with_baseline,
                "patients_sjd_without_clinical_baseline": (
                    sjd_patients - patients_with_baseline
                ),
                "episodes_sjd_clinical": int(clinical.sum()),
                "episodes_sjd_ambiguous": int(visit_type.eq("ambiguous").sum()),
                "episodes_sjd_research_or_procedure_only": int(
                    visit_type.eq("research_or_procedure_only_candidate").sum()
                ),
                "episodes_before_merge": len(episodes_before),
                "episodes_after_merge": len(spine_all),
                "episode_count_preserved": len(episodes_before) == len(spine_all),
                "clinical_episode_id_set_preserved": ids_preserved,
            }
        ]
    )


def write_outputs(
    spine_all: pd.DataFrame,
    spine_sjd: pd.DataFrame,
    qc: pd.DataFrame,
    output_all_path: Path,
    output_all_csv_path: Path,
    output_sjd_path: Path,
    output_sjd_csv_path: Path,
    qc_path: Path,
    qc_parquet_path: Path,
) -> None:
    """Write each step-11 output in both Parquet and CSV formats.

    Parameters
    ----------
    spine_all : pd.DataFrame
        Complete clinical-episode spine.
    spine_sjd : pd.DataFrame
        SjD-only clinical-episode spine.
    qc : pd.DataFrame
        One-row quality-control summary.
    output_all_path, output_all_csv_path : Path
        Parquet and CSV destinations for the complete spine.
    output_sjd_path, output_sjd_csv_path : Path
        Parquet and CSV destinations for the SjD-only spine.
    qc_path, qc_parquet_path : Path
        CSV and Parquet destinations for the quality-control summary.
    """
    paths = (
        output_all_path,
        output_all_csv_path,
        output_sjd_path,
        output_sjd_csv_path,
        qc_path,
        qc_parquet_path,
    )
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)

    spine_all.to_parquet(output_all_path, index=False)
    spine_all.to_csv(output_all_csv_path, index=False)
    spine_sjd.to_parquet(output_sjd_path, index=False)
    spine_sjd.to_csv(output_sjd_csv_path, index=False)
    qc.to_csv(qc_path, index=False)
    qc.to_parquet(qc_parquet_path, index=False)


def main() -> None:
    """Read steps 09d and 10, validate them, and write definitive spines."""
    args = parse_args()
    logger = setup_logger("11_build_clinical_episode_spine")
    logger.info("Reading finalized 09d episodes from %s", args.input_path)
    episodes = pd.read_parquet(args.input_path)
    logger.info("Reading step-10 patient baseline data from %s", args.baseline_path)
    baseline = pd.read_csv(args.baseline_path)
    spine_all, spine_sjd = build_spines(episodes, baseline)
    qc = build_qc(episodes, spine_all, spine_sjd)

    write_outputs(
        spine_all,
        spine_sjd,
        qc,
        args.output_all_path,
        args.output_all_csv_path,
        args.output_sjd_path,
        args.output_sjd_csv_path,
        args.qc_path,
        args.qc_parquet_path,
    )

    row = qc.iloc[0]
    logger.info("patients_all=%d", row["patients_all"])
    logger.info("episodes_all=%d", row["episodes_all"])
    logger.info("patients_sjd=%d", row["patients_sjd"])
    logger.info("episodes_sjd=%d", row["episodes_sjd"])
    logger.info(
        "patients_sjd_with_clinical_baseline=%d",
        row["patients_sjd_with_clinical_baseline"],
    )
    logger.info(
        "patients_sjd_without_clinical_baseline=%d",
        row["patients_sjd_without_clinical_baseline"],
    )
    logger.info(
        "Wrote complete spine to %s and %s",
        args.output_all_path,
        args.output_all_csv_path,
    )
    logger.info(
        "Wrote SjD spine to %s and %s",
        args.output_sjd_path,
        args.output_sjd_csv_path,
    )
    logger.info("Wrote QC report to %s and %s", args.qc_path, args.qc_parquet_path)
    logger.info("Episode count and clinical_episode_id set were preserved")


if __name__ == "__main__":
    main()
