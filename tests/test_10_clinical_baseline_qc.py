"""Synthetic tests for the diagnostic clinical-baseline QC step."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd
import pytest

MODULE_PATH = Path(__file__).parents[1] / "src" / "10_clinical_baseline_qc.py"
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("clinical_baseline_10", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def episodes() -> pd.DataFrame:
    """Return synthetic episodes covering shifted, same, and absent baselines."""
    return pd.DataFrame(
        {
            "patient_id": ["p1", "p1", "p2", "p3"],
            "clinical_episode_id": ["p1_e1", "p1_e2", "p2_e1", "p3_e1"],
            "episode_start_date": [
                "2024-01-01",
                "2024-01-05",
                "2024-02-01",
                "2024-03-01",
            ],
            "clinical_anchor_date": [pd.NaT, "2024-01-06", "2024-02-01", pd.NaT],
            "visit_type": [
                "research_or_procedure_only_candidate",
                "clinical_candidate",
                "clinical_candidate",
                "ambiguous",
            ],
            "clinical_visit": [False, True, True, False],
            "manual_review_required": [False, False, True, False],
            "essdai__essdai_total_score": [pd.NA, 4, 2, pd.NA],
            "esspri_questionnaire__pain": [pd.NA, 3, pd.NA, pd.NA],
        }
    )


def test_build_comparison_uses_first_clinical_anchor() -> None:
    result = module.build_comparison(episodes()).set_index("patient_id")

    assert result.loc["p1", "clinical_baseline_episode_id"] == "p1_e2"
    assert result.loc["p1", "clinical_baseline_date"] == pd.Timestamp("2024-01-06")
    assert result.loc["p1", "days_shifted"] == 5
    assert result.loc["p1", "reason_for_shift"] == "first_episode_research_only"
    assert result.loc["p1", "baseline_pop_classifiable"]
    assert result.loc["p2", "reason_for_shift"] == "same_date"


def test_missing_clinical_baseline_is_flagged_for_review() -> None:
    result = module.build_comparison(episodes()).set_index("patient_id")

    assert pd.isna(result.loc["p3", "clinical_baseline_episode_id"])
    assert result.loc["p3", "reason_for_shift"] == "no_clinical_baseline"
    assert result.loc["p3", "baseline_manual_review_required"]


def test_duplicate_episode_identifier_is_rejected() -> None:
    duplicated = pd.concat([episodes(), episodes().iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate patient/episode"):
        module.build_comparison(duplicated)


def test_summary_reports_requested_counts_and_reasons() -> None:
    summary = module.build_summary(module.build_comparison(episodes())).iloc[0]

    assert summary["n_patients"] == 3
    assert summary["n_shifted_baseline"] == 1
    assert summary["n_baseline_with_both"] == 1
    assert summary["n_clinical_but_not_pop_classifiable"] == 1
    assert summary["n_without_clinical_baseline"] == 1
    assert summary["n_reason_first_episode_research_only"] == 1
