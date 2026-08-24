"""Synthetic tests for clinical-episode finalization step 09d."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd
import pytest

MODULE_PATH = Path(__file__).parents[1] / "src" / "09d_finalize_clinical_episode_dataset.py"
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("finalize_09d", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def source_rows(values, dates, times):
    """Create minimal synthetic source rows for one variable."""
    return pd.DataFrame(
        {
            "measure": values,
            "_collection_date": pd.to_datetime(dates),
            "_time": pd.to_datetime(times, format="%H:%M").time,
        }
    )


def test_time_varying_conflict_prefers_anchor_then_latest_time():
    rows = source_rows([10, 20, 30], ["2024-01-01"] * 3, ["08:00", "09:00", "10:00"])
    value, status, method = module.resolve_generic(rows, "measure", "2024-01-01")
    assert (value, status, method) == (30, "resolved", "latest_timestamp_same_day")


def test_incompatible_historical_values_are_unresolved():
    rows = source_rows(["2021", "07/2022"], ["2024-01-01"] * 2, ["08:00", "09:00"])
    rows = rows.rename(columns={"measure": "diagnosis_date"})
    value, status, method = module.resolve_generic(rows, "diagnosis_date", "2024-01-01")
    assert pd.isna(value)
    assert status == "unresolved"
    assert method == "unresolved_historical_conflict"


def test_hard_qc_rejects_changed_anchor_date():
    before = pd.DataFrame(
        {"patient_id": ["p1"], "clinical_episode_id": ["p1__CE0001"],
         "episode_start_date": ["2024-01-01"], "clinical_anchor_date": ["2024-01-01"],
         "episode_end_date": ["2024-01-02"]}
    )
    after = before.copy()
    after["clinical_anchor_date"] = "2024-01-02"
    with pytest.raises(ValueError, match="architecture QC"):
        module.hard_qc(before, after)


def residual_frames(source_values):
    """Create minimal episode and source frames for residual-pipe tests."""
    output = pd.DataFrame(
        {
            "patient_id": ["p1"],
            "clinical_episode_id": ["p1__CE0001"],
            "clinical_anchor_date": [pd.Timestamp("2024-01-01")],
            "esspri_questionnaire__mental_fatigue_rank": ["2 | 5"],
        }
    )
    sources = pd.DataFrame(
        {
            "patient_id": ["p1"] * len(source_values),
            "clinical_episode_id": ["p1__CE0001"] * len(source_values),
            "esspri_questionnaire__mental_fatigue_rank": source_values,
            "_collection_date": pd.to_datetime(["2024-01-01"] * len(source_values)),
            "_time": pd.to_datetime(
                [f"0{index + 8}:00" for index in range(len(source_values))],
                format="%H:%M",
            ).time,
            "_interval": ["Baseline"] * len(source_values),
            "_protocol": ["11-D-0172"] * len(source_values),
        }
    )
    return output, sources


def test_residual_multirow_pipe_uses_existing_timestamp_rule():
    output, sources = residual_frames([2, 5])
    records = []

    result = module.resolve_residual_pipes(output, sources, records)

    assert result.loc[0, "esspri_questionnaire__mental_fatigue_rank"] == 5
    assert records[0]["resolution_method"] == "latest_timestamp_same_day"
    assert records[0]["conflict_origin"] == "09c_multirow_conflict"


def test_preexisting_source_pipe_is_missing_and_unresolved():
    output, sources = residual_frames(["2 | 5"])
    records = []

    result = module.resolve_residual_pipes(output, sources, records)

    assert pd.isna(result.loc[0, "esspri_questionnaire__mental_fatigue_rank"])
    assert records[0]["resolution_status"] == "unresolved"
    assert records[0]["resolution_method"] == "preexisting_source_pipe"
    assert records[0]["conflict_origin"] == "preexisting_source_pipe"


@pytest.mark.parametrize("column", sorted(module.PROVENANCE_PIPE_COLUMNS))
def test_provenance_pipe_columns_are_exempt_from_analytical_qc(column):
    module.validate_analytical_pipes(pd.DataFrame({column: ["left | right"]}))
