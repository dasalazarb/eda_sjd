"""Tests for the read-only visit episode temporal review."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

MODULE_PATH = Path(__file__).parents[1] / "src" / "08b_visit_episode_temporal_audit.py"
SPEC = importlib.util.spec_from_file_location(
    "visit_episode_temporal_audit", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def _episodes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": [1, 1, 2],
            "interval_name": ["V1", "V2", "Baseline"],
            "episode_start_date": ["2024-01-01", "2024-01-12", "2024-03-01"],
            "episode_end_date": ["2024-01-02", "2024-02-15", "2024-03-01"],
            "episode_span_days": [1, 34, 0],
            "candidate_visit_type": [
                "clinical_candidate",
                "ambiguous",
                "research_only_candidate",
            ],
        }
    )


def _components() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": [1, 1, 1, 1],
            "interval_name": ["V1", "V1", "V2", "V2"],
            "component": [
                "essdai",
                "eye_examination",
                "esspri_questionnaire",
                "salivary_flow_form",
            ],
        }
    )


def test_build_temporal_review_finds_requested_cases() -> None:
    context, cases = AUDIT.build_temporal_review(_episodes(), _components())

    first = context.loc[context["interval_name"].eq("V1")].iloc[0]
    assert first["next_interval"] == "V2"
    assert first["gap_next_days"] == 10
    assert set(cases["case_type"]) == {
        "episode_span_gt_14",
        "episode_span_gt_30",
        "patient_without_clinical_candidate",
        "ambiguous_near_clinical",
        "essdai_esspri_split",
        "clinical_components_split",
    }


def test_build_temporal_review_handles_no_review_cases() -> None:
    episodes = _episodes().iloc[[0]].copy()
    components = _components().loc[lambda frame: frame["interval_name"].eq("V1")]

    context, cases = AUDIT.build_temporal_review(episodes, components)

    assert len(context) == 1
    assert pd.isna(context.loc[0, "previous_interval"])
    assert cases.empty
    assert list(cases.columns)[0] == "case_type"


def test_build_temporal_review_rejects_missing_episode_columns() -> None:
    with pytest.raises(ValueError, match="episode_span_days"):
        AUDIT.build_temporal_review(
            _episodes().drop(columns="episode_span_days"), _components()
        )


def test_build_composite_candidates_compares_windows_and_excludes_records() -> None:
    context = pd.DataFrame(
        {
            "patient_id": [1, 1, 1, 2, 3, 3, 4, 4],
            "interval_name": ["A", "B", "C", "LONG", "X", "Y", "E", "P"],
            "episode_start_date": [
                "2024-01-01",
                "2024-01-03",
                "2024-01-08",
                "2024-02-01",
                "2024-03-01",
                "2024-03-02",
                "2024-04-01",
                "2024-04-03",
            ],
            "episode_end_date": [
                "2024-01-01",
                "2024-01-03",
                "2024-01-08",
                "2024-03-10",
                "2024-03-03",
                "2024-03-04",
                "2024-04-01",
                "2024-04-03",
            ],
            "candidate_visit_type": [
                "ambiguous",
                "ambiguous",
                "research_only_candidate",
                "ambiguous",
                "ambiguous",
                "ambiguous",
                "clinical_candidate",
                "ambiguous",
            ],
        }
    )
    cases = pd.DataFrame(
        {
            "case_type": ["episode_span_gt_30"],
            "patient_id": [2],
            "interval_name": ["LONG"],
        }
    )
    components = pd.DataFrame(
        {
            "patient_id": [1, 1, 1, 1, 1, 4, 4],
            "interval_name": ["A", "A", "B", "B", "C", "E", "P"],
            "component": [
                "eye_examination",
                "salivary_flow_form",
                "systems_review_for_physician",
                "visit_summary_form",
                "ccgo",
                "essdai",
                "esspri_questionnaire",
            ],
        }
    )

    candidates, summary, excluded = AUDIT.build_composite_episode_candidates(
        context, cases, components
    )

    patient_one = candidates.loc[
        candidates["patient_id"].eq(1) & candidates["window_days"].eq(3)
    ].iloc[0]
    assert patient_one["intervals_involved"] == "A|B"
    assert patient_one["clinical_anchor_count_combined"] == 4
    assert bool(patient_one["ambiguous_to_clinical_candidate"])
    assert candidates.loc[
        candidates["patient_id"].eq(4), "essdai_esspri_reunited"
    ].all()
    assert set(summary["window_days"]) == {3, 7, 14}
    assert (
        summary.loc[
            summary["window_days"].eq(3),
            "patients_without_clinical_candidate_recovered",
        ].iloc[0]
        == 1
    )
    assert set(excluded["interval_name"]) == {"LONG", "X", "Y"}


def test_build_composite_candidates_returns_empty_window_summary() -> None:
    context = pd.DataFrame(
        {
            "patient_id": [1],
            "interval_name": ["A"],
            "episode_start_date": ["2024-01-01"],
            "episode_end_date": ["2024-01-01"],
            "candidate_visit_type": ["ambiguous"],
        }
    )

    candidates, summary, excluded = AUDIT.build_composite_episode_candidates(
        context, pd.DataFrame(columns=["case_type"]), pd.DataFrame()
    )

    assert candidates.empty
    assert summary["n_clusters_created"].eq(0).all()
    assert excluded.empty
