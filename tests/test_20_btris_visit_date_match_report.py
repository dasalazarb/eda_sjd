from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
MODULE_PATH = REPO_ROOT / "src" / "20_btris_visit_date_match_report.py"
spec = importlib.util.spec_from_file_location("btris_visit_date_match_report", MODULE_PATH)
btris_report = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = btris_report
assert spec.loader is not None
spec.loader.exec_module(btris_report)


def test_normalize_patient_id_removes_leading_zeroes_and_separators() -> None:
    series = pd.Series(["012-345", " 000987 "])

    normalized = btris_report._normalize_patient_id(series)

    assert normalized.tolist() == ["12345", "987"]


def test_normalize_patient_id_preserves_internal_zeroes() -> None:
    series = pd.Series(["10020", "10 020"])

    normalized = btris_report._normalize_patient_id(series)

    assert normalized.tolist() == ["10020", "10020"]


def test_normalize_patient_id_marks_invalid_empty_values_missing() -> None:
    series = pd.Series(["", " 0 ", None, float("nan")])

    normalized = btris_report._normalize_patient_id(series)

    assert normalized.isna().tolist() == [True, True, True, True]
