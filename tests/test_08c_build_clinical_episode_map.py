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
    assigned = EPISODES.assign_episodes(EPISODES.add_presence_flags(prepared))
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
    manifest = EPISODES.build_manifest(
        EPISODES.assign_episodes(EPISODES.add_presence_flags(prepared))
    )

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
