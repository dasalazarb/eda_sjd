from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
MODULE_PATH = REPO_ROOT / "src" / "09b_merge_essdai_versions.py"
spec = importlib.util.spec_from_file_location("merge_essdai_versions", MODULE_PATH)
merge_essdai_versions = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = merge_essdai_versions
assert spec.loader is not None
spec.loader.exec_module(merge_essdai_versions)


def test_merge_essdai_columns_prefers_essdai_r_when_values_conflict() -> None:
    df = pd.DataFrame(
        {
            "essdai-_r__constitutional": ["2"],
            "essdai__constitutional": ["1"],
        }
    )

    result, pairs_merged = merge_essdai_versions._merge_essdai_columns(df)

    assert pairs_merged == 1
    assert result.loc[0, "essdai__constitutional"] == "2"
    assert "essdai-_r__constitutional" not in result.columns


def test_merge_essdai_columns_keeps_canonical_value_when_values_agree() -> None:
    df = pd.DataFrame(
        {
            "essdai-_r__constitutional": ["1"],
            "essdai__constitutional": [" 1 "],
        }
    )

    result, _ = merge_essdai_versions._merge_essdai_columns(df)

    assert result.loc[0, "essdai__constitutional"] == "1"


def test_merge_essdai_columns_uses_essdai_r_when_canonical_value_is_empty() -> None:
    df = pd.DataFrame(
        {
            "essdai-_r__constitutional": ["3"],
            "essdai__constitutional": [pd.NA],
        }
    )

    result, _ = merge_essdai_versions._merge_essdai_columns(df)

    assert result.loc[0, "essdai__constitutional"] == "3"


def test_add_sjogrens_class_patient_cohort_keeps_patient_with_any_target_class() -> None:
    df = pd.DataFrame(
        {
            "ids__patient_record_number": ["P1", "P1"],
            "visit_summary_form__sjogrens_class": ["1", "8"],
        }
    )

    result, patient_cohorts = merge_essdai_versions._add_sjogrens_class_patient_cohort(df)

    assert set(result["sjogrens_class_patient_cohort"]) == {"ever_1_2_4"}
    assert patient_cohorts.loc[0, "sjogrens_class_patient_values"] == "1|8"


def test_add_sjogrens_class_patient_cohort_marks_never_target_patients() -> None:
    df = pd.DataFrame(
        {
            "ids__patient_record_number": ["P2", "P2", "P2", "P2", "P2"],
            "visit_summary_form__sjogrens_class": ["3", "5", "6", "8", "8"],
        }
    )

    result, patient_cohorts = merge_essdai_versions._add_sjogrens_class_patient_cohort(df)

    assert set(result["sjogrens_class_patient_cohort"]) == {"never_1_2_4"}
    assert patient_cohorts.loc[0, "sjogrens_class_patient_values"] == "3|5|6|8"


def test_add_sjogrens_class_patient_cohort_handles_pipe_delimited_and_decimal_values() -> None:
    df = pd.DataFrame(
        {
            "ids__patient_record_number": ["P3", "P3"],
            "visit_summary_form__sjogrens_class": ["3 | 5.0", "2.0 | 8"],
        }
    )

    result, patient_cohorts = merge_essdai_versions._add_sjogrens_class_patient_cohort(df)

    assert set(result["sjogrens_class_patient_cohort"]) == {"ever_1_2_4"}
    assert patient_cohorts.loc[0, "sjogrens_class_patient_values"] == "3|5|2|8"
