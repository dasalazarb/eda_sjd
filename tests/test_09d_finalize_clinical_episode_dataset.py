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
