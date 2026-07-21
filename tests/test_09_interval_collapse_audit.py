from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
MODULE_PATH = REPO_ROOT / "src" / "09_interval_collapse_audit.py"
spec = importlib.util.spec_from_file_location("interval_collapse_audit", MODULE_PATH)
interval_collapse_audit = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = interval_collapse_audit
assert spec.loader is not None
spec.loader.exec_module(interval_collapse_audit)


def test_merge_same_date_intervals_prefers_non_natural_history_values() -> None:
    collapsed = pd.DataFrame(
        {
            "ids__patient_record_number": ["P1", "P1"],
            "ids__interval_name": [
                "Natural History Protocol 478 Interval",
                "Year 1",
            ],
            "ids__visit_date": ["2015-05-13", "2015-05-13 | 2015-05-18"],
            "vital_signs__pulse": ["70", "80"],
            "lab__result": ["positive", "negative"],
        }
    )

    result, report = interval_collapse_audit._merge_same_date_intervals(
        collapsed,
        subject_col="ids__patient_record_number",
        interval_col="ids__interval_name",
        visit_date_col="ids__visit_date",
    )

    assert len(result) == 1
    assert result.loc[0, "ids__interval_name"] == "Year 1"
    assert result.loc[0, "vital_signs__pulse"] == "80"
    assert result.loc[0, "lab__result"] == "negative | positive"
    assert report.loc[0, "action"] == "merged_prefer_non_natural_history"


def test_merge_same_date_intervals_keeps_two_natural_history_rows() -> None:
    collapsed = pd.DataFrame(
        {
            "ids__patient_record_number": ["P1", "P1"],
            "ids__interval_name": [
                "Natural History Protocol 478 Interval",
                "Natural History Protocol 478 Interval",
            ],
            "ids__visit_date": ["2015-05-13", "05/13/2015"],
        }
    )

    result, report = interval_collapse_audit._merge_same_date_intervals(
        collapsed,
        subject_col="ids__patient_record_number",
        interval_col="ids__interval_name",
        visit_date_col="ids__visit_date",
    )

    assert len(result) == 2
    assert set(report["action"]) == {"not_merged_both_natural_history"}


def test_merge_same_date_intervals_ignores_invalid_dates() -> None:
    collapsed = pd.DataFrame(
        {
            "ids__patient_record_number": ["P1", "P1"],
            "ids__interval_name": [
                "Natural History Protocol 478 Interval",
                "Year 1",
            ],
            "ids__visit_date": ["not a date", "2015-05-13"],
        }
    )

    result, report = interval_collapse_audit._merge_same_date_intervals(
        collapsed,
        subject_col="ids__patient_record_number",
        interval_col="ids__interval_name",
        visit_date_col="ids__visit_date",
    )

    assert len(result) == 2
    assert report.empty
