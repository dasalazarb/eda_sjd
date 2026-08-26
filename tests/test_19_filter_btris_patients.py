from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
MODULE_PATH = REPO_ROOT / "src" / "19_filter_btris_patients.py"
spec = importlib.util.spec_from_file_location("filter_btris_patients", MODULE_PATH)
btris_filter = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = btris_filter
assert spec.loader is not None
spec.loader.exec_module(btris_filter)


def _cohort(*mrns: object) -> object:
    return btris_filter._build_patient_cohort(
        pd.DataFrame({btris_filter.MRN_COLUMN: list(mrns)})
    )


def test_mrn_normalization_equates_separators_and_leading_zeroes() -> None:
    normalized = {
        btris_filter._canonical_patient_record_number(value)
        for value in ["001-234", "001234", "1234", " 001/234 ", "001\\234"]
    }

    assert normalized == {"1234"}
    assert btris_filter._canonical_patient_record_number(None) == ""


def test_multiple_clinical_episodes_produce_one_unique_patient() -> None:
    cohort = _cohort("001-234", "001234", "1234", "001-234")

    assert cohort.patient_ids == {"1234"}
    assert len(cohort.patient_ids) == 1


def test_coverage_summary_counts_unique_found_patients() -> None:
    cohort = _cohort("001", "002", "003")
    detail = btris_filter._build_coverage_detail(
        cohort,
        Counter({"1": 4, "2": 1}),
        {"1": {"file_a.csv"}, "2": {"file_b.csv"}},
        {"11D": {"1", "2"}},
    )

    summary = btris_filter._build_coverage_summary(detail).iloc[0]

    assert summary["n_spine_patients"] == 3
    assert summary["n_btris_patients_found"] == 2
    assert summary["n_btris_patients_not_found"] == 1
    assert summary["pct_btris_patient_coverage"] == pytest.approx(200 / 3)


def test_duplicate_btris_appearances_count_patient_once_but_all_files() -> None:
    cohort = _cohort("001")
    detail = btris_filter._build_coverage_detail(
        cohort,
        Counter({"1": 5}),
        {"1": {"file_a.csv", "file_b.csv"}},
        {"11D": {"1"}, "15D": {"1"}},
    )

    summary = btris_filter._build_coverage_summary(detail).iloc[0]

    assert summary["n_btris_patients_found"] == 1
    assert detail.loc[0, "n_btris_rows"] == 5
    assert detail.loc[0, "n_btris_files"] == 2


def test_protocol_flags_describe_btris_sources() -> None:
    cohort = _cohort("001", "002", "003")
    detail = btris_filter._build_coverage_detail(
        cohort,
        Counter({"1": 1, "2": 1}),
        {"1": {"11d.csv"}, "2": {"15d.csv"}},
        {"11D": {"1"}, "15D": {"2"}},
    ).set_index("patient_record_number_normalized")

    assert bool(detail.loc["1", "found_in_11d"])
    assert not bool(detail.loc["1", "found_in_15d"])
    assert not bool(detail.loc["2", "found_in_11d"])
    assert bool(detail.loc["2", "found_in_15d"])
    assert not bool(detail.loc["3", "found_in_btris"])


def test_lab_files_are_restricted_to_allowed_order_names(tmp_path: Path) -> None:
    lab_path = tmp_path / "Lab_results.csv"
    pd.DataFrame(
        {
            "MRN": ["001-234", "001234", "999"],
            "Order Name": [" ANA ", "Unwanted", "ANA"],
            "Result": ["positive", "negative", "negative"],
        }
    ).to_csv(lab_path, index=False)

    filtered, metrics, row_counts = btris_filter._filter_single_csv(
        lab_path, {"1234"}, {"ana"}
    )

    assert filtered["Result"].tolist() == ["positive"]
    assert metrics["is_lab_file"] is True
    assert metrics["patients_identified"] == 1
    assert row_counts == Counter({"1234": 1})


def test_spine_without_mrn_column_fails_clearly() -> None:
    with pytest.raises(KeyError, match="columna MRN requerida"):
        btris_filter._build_patient_cohort(pd.DataFrame({"patient_id": ["123"]}))


def test_spine_with_only_missing_mrns_fails_clearly() -> None:
    with pytest.raises(ValueError, match="vacíos después de la normalización"):
        _cohort(None, float("nan"), " ")
