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
