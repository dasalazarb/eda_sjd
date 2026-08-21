"""Synthetic-data tests for clinical episode construction."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

MODULE_PATH = Path(__file__).parents[1] / "src" / "08c_build_clinical_episode_map.py"
SPEC = importlib.util.spec_from_file_location("clinical_episode_map", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
EPISODES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EPISODES)


def _visits() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": [1, 1, 1, 2],
            "row_id_raw": [10, 11, 12, 20],
            "interval_name": ["V1", "V1-extra", "V2", "V1"],
            "visit_date": ["2024-01-01", "2024-01-10", "2024-03-01", "2024-02-01"],
            "essdai__domain": [0, pd.NA, pd.NA, pd.NA],
            "esspri_questionnaire__pain": [pd.NA, "No", pd.NA, pd.NA],
            "eye_examination__eye_exam_done": [pd.NA, pd.NA, "No", pd.NA],
            "salivary_flow_form__tot_sim_sal_flow": [pd.NA, pd.NA, 0, pd.NA],
            "ccgo__participate_ccgo": [pd.NA, pd.NA, pd.NA, "No"],
        }
    )


def test_flags_and_temporal_assignment_keep_valid_zero_and_no() -> None:
    prepared, _ = EPISODES.prepare_visits(_visits())
    flagged = EPISODES.add_presence_flags(prepared)
    units = EPISODES.build_daily_activity_units(flagged)
    assigned_units = EPISODES.assign_episodes(units)
    assigned = EPISODES.propagate_episode_assignments(flagged, assigned_units)
    manifest = EPISODES.build_manifest(assigned)

    first = manifest.loc[manifest["has_essdai_form"]].iloc[0]
    assert first["has_esspri_core"]
    assert first["episode_span_days"] == 9
    assert first["clinical_anchor_date"] == pd.Timestamp("2024-01-01")
    assert assigned.loc[assigned["row_id_raw"].eq(10), "clinical_episode_id"].iloc[0] == (
        assigned.loc[assigned["row_id_raw"].eq(11), "clinical_episode_id"].iloc[0]
    )


def test_objective_pair_is_clinical_and_research_only_is_not() -> None:
    prepared, _ = EPISODES.prepare_visits(_visits())
    flagged = EPISODES.add_presence_flags(prepared)
    units = EPISODES.assign_episodes(EPISODES.build_daily_activity_units(flagged))
    assigned = EPISODES.propagate_episode_assignments(flagged, units)
    manifest = EPISODES.build_manifest(assigned)

    objective = manifest.loc[manifest["intervals_involved"].eq("V2")].iloc[0]
    research = manifest.loc[
        manifest["visit_type"].eq("research_or_procedure_only_candidate")
    ].iloc[0]
    assert objective["clinical_visit"]
    assert research["patient_id"] == 2
    assert pd.isna(research["clinical_anchor_date"])


def test_prepare_visits_rejects_duplicate_raw_row_ids() -> None:
    visits = _visits()
    visits.loc[1, "row_id_raw"] = 10
    with pytest.raises(ValueError, match="row_id_raw must be complete and unique"):
        EPISODES.prepare_visits(visits)


def test_write_parquet_and_csv_creates_matching_outputs(tmp_path: Path) -> None:
    frame = pd.DataFrame({"clinical_episode_id": ["1__CE0001"]})

    parquet_path, csv_path = EPISODES.write_parquet_and_csv(
        frame, tmp_path / "episode_manifest.parquet"
    )

    assert parquet_path == tmp_path / "episode_manifest.parquet"
    assert csv_path == tmp_path / "episode_manifest.csv"
    assert parquet_path.exists()
    assert csv_path.exists()
    assert pd.read_csv(csv_path).to_dict(orient="records") == frame.to_dict(
        orient="records"
    )


def test_same_patient_date_is_one_indivisible_clinical_unit() -> None:
    visits = pd.DataFrame(
        {
            "patient_id": [7, 7, 7, 7],
            "row_id_raw": [1, 2, 3, 4],
            "interval_name": ["V1", "V1", "Screening", "V1"],
            "visit_date": ["2024-04-05"] * 4,
            "essdai__domain": [0, pd.NA, pd.NA, pd.NA],
            "esspri_questionnaire__pain": [pd.NA, "No", pd.NA, pd.NA],
            "eye_examination__eye_exam_done": [pd.NA, pd.NA, "Yes", pd.NA],
            "oral_exam_form__performed": [pd.NA, pd.NA, pd.NA, "Yes"],
        }
    )
    prepared, provenance = EPISODES.prepare_visits(visits)
    flagged = EPISODES.add_presence_flags(prepared)
    daily_units = EPISODES.build_daily_activity_units(flagged, provenance)
    assigned_units = EPISODES.assign_episodes(daily_units)
    assigned_rows = EPISODES.propagate_episode_assignments(flagged, assigned_units)
    source_intervals = EPISODES.build_source_interval_qc(prepared)
    manifest = EPISODES.build_manifest(assigned_rows, source_intervals)
    qc = EPISODES.build_qc(
        assigned_rows, assigned_units, manifest, source_intervals
    )

    assert len(daily_units) == 1
    assert set(daily_units.loc[0, "row_ids_involved"]) == {1, 2, 3, 4}
    assert set(daily_units.loc[0, "interval_names_involved"]) == {"V1", "Screening"}
    assert assigned_rows["clinical_episode_id"].nunique() == 1
    assert manifest.loc[0, "clinical_visit"]
    assert qc.loc[0, "raw_rows"] == 4
    assert qc.loc[0, "unique_patient_collection_dates"] == 1
    assert qc.loc[0, "daily_activity_units"] == 1
    assert qc.loc[0, "raw_rows_unassigned"] == 0
    assert qc.loc[0, "raw_rows_multiply_assigned"] == 0
    assert qc.loc[0, "patient_date_units_assigned_to_multiple_episodes"] == 0


def test_source_interval_qc_drives_explicit_manual_review_reasons() -> None:
    visits = pd.DataFrame(
        {
            "patient_id": [9, 9, 9, 9],
            "row_id_raw": [1, 2, 3, 4],
            "interval_name": ["A", "A", "B", "C"],
            "visit_date": ["2024-01-01", "2024-03-15", "2024-01-01", pd.NA],
            "essdai__domain": [0, 0, pd.NA, pd.NA],
            "esspri_questionnaire__pain": [pd.NA, pd.NA, "No", "No"],
        }
    )
    prepared, _ = EPISODES.prepare_visits(visits)
    source_intervals = EPISODES.build_source_interval_qc(prepared)
    flagged = EPISODES.add_presence_flags(prepared)
    units = EPISODES.assign_episodes(EPISODES.build_daily_activity_units(flagged))
    assigned = EPISODES.propagate_episode_assignments(flagged, units)
    manifest = EPISODES.build_manifest(assigned, source_intervals)
    qc = EPISODES.build_qc(assigned, units, manifest, source_intervals)
    reasons = EPISODES.manual_review_reason_distribution(manifest)

    interval_a = source_intervals.loc[source_intervals["interval_name"].eq("A")].iloc[0]
    assert interval_a["source_interval_span_days"] == 74
    assert interval_a["source_interval_span_gt30"]
    first_episode = manifest.loc[
        manifest["episode_start_date"].eq(pd.Timestamp("2024-01-01"))
    ].iloc[0]
    assert first_episode["max_source_interval_span_days"] == 74
    assert set(first_episode["manual_review_reason"].split("|")) == {
        "source_interval_span_gt30",
        "overlapping_source_interval_ranges",
    }
    assert manifest["manual_review_reason"].str.contains(
        "missing_collection_date", regex=False
    ).any()
    assert qc.loc[0, "source_intervals_span_gt30"] == 1
    assert qc.loc[0, "patients_with_source_interval_span_gt30"] == 1
    assert set(reasons["manual_review_reason"]) == {
        "missing_collection_date",
        "overlapping_source_interval_ranges",
        "source_interval_span_gt30",
    }
