"""Synthetic tests for the definitive clinical-episode spine step."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd
import pytest

MODULE_PATH = Path(__file__).parents[1] / "src" / "11_build_clinical_episode_spine.py"
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("clinical_episode_spine_11", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def episodes() -> pd.DataFrame:
    """Return a small frozen episode table with all visit categories."""
    return pd.DataFrame(
        {
            "patient_id": ["p1", "p1", "p2", "p3"],
            "clinical_episode_id": ["p1_e1", "p1_e2", "p2_e1", "p3_e1"],
            "episode_start_date": pd.to_datetime(
                ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"]
            ),
            "clinical_anchor_date": pd.to_datetime(
                [None, "2024-02-01", None, "2024-04-01"]
            ),
            "episode_end_date": pd.to_datetime(
                ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"]
            ),
            "visit_type": [
                "research_or_procedure_only_candidate",
                "clinical_candidate",
                "ambiguous",
                "clinical_candidate",
            ],
            "clinical_visit": [False, True, False, True],
            "clinical_measure": [10, 11, 20, 30],
        }
    )


def baselines() -> pd.DataFrame:
    """Return one synthetic step-10 row per patient."""
    return pd.DataFrame(
        {
            "patient_id": ["p1", "p2", "p3"],
            "sjd_ever_1_2_4": [True, "true", False],
            "sjogrens_class_patient_values": ["1", "2 | 3", "3"],
            "clinical_baseline_episode_id": ["p1_e2", pd.NA, "p3_e1"],
            "clinical_baseline_date": ["2024-02-01", pd.NA, "2024-04-01"],
        }
    )


def test_build_spines_preserves_every_episode_for_sjd_patients() -> None:
    spine_all, spine_sjd = module.build_spines(episodes(), baselines())

    assert (
        spine_all["clinical_episode_id"].tolist()
        == episodes()["clinical_episode_id"].tolist()
    )
    assert spine_all["clinical_measure"].tolist() == [10, 11, 20, 30]
    assert set(spine_sjd["clinical_episode_id"]) == {"p1_e1", "p1_e2", "p2_e1"}
    assert spine_all.set_index("clinical_episode_id").loc[
        "p1_e2", "is_clinical_baseline"
    ]


def test_missing_baseline_produces_no_baseline_episode() -> None:
    _, spine_sjd = module.build_spines(episodes(), baselines())
    qc = module.build_qc(episodes(), *module.build_spines(episodes(), baselines()))

    assert not spine_sjd.loc[
        spine_sjd["patient_id"].eq("p2"), "is_clinical_baseline"
    ].any()
    assert qc.loc[0, "patients_sjd_without_clinical_baseline"] == 1
    assert qc.loc[0, "episodes_sjd_research_or_procedure_only"] == 1


def test_duplicate_patient_episode_pair_is_rejected() -> None:
    duplicated = pd.concat([episodes(), episodes().iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate patient/episode"):
        module.build_spines(duplicated, baselines())


def test_multiple_baseline_matches_are_rejected() -> None:
    spine = episodes().copy()
    spine["is_clinical_baseline"] = [True, True, False, False]

    with pytest.raises(ValueError, match="more than one clinical baseline"):
        module.hard_qc(spine)
