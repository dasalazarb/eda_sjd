"""Synthetic tests for exact-pair BTRIS laboratory harmonization."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

SRC = Path(__file__).parents[1] / "src"
sys.path.insert(0, str(SRC))
SPEC = importlib.util.spec_from_file_location(
    "harmonization_20c", SRC / "20c_btris_lab_rule_harmonization.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def row(family: str, **updates: object) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build one synthetic step-20 row and its exact-pair rule."""
    record = {
        "order_name_original": "Order A",
        "cluster_name_original": "Cluster A",
        "result_valid_for_analysis": True,
        "result_raw": pd.NA,
        "result_text": pd.NA,
        "result_numeric": pd.NA,
        "result_numeric_exact": pd.NA,
        "result_operator": pd.NA,
        "result_numeric_bound": pd.NA,
        "reference_low": pd.NA,
        "reference_high": pd.NA,
        "reference_operator": pd.NA,
        "reference_bound": pd.NA,
        "reported_interpretation": pd.NA,
    }
    record.update(updates)
    rules = pd.DataFrame(
        {
            "Order Name": ["Order A"],
            "Cluster Name": ["Cluster A"],
            "rule_family": [family],
        }
    )
    return pd.DataFrame([record]), rules


@pytest.mark.parametrize(
    ("value", "expected"), [(8, "LOW"), (20, "NORMAL"), (45, "HIGH")]
)
def test_continuous_range_exact(value: float, expected: str) -> None:
    """Exact values use the contemporaneous bilateral reference."""
    labs, rules = row(
        "CONTINUOUS_RANGE",
        result_numeric_exact=value,
        reference_low=10,
        reference_high=40,
    )
    result = MODULE.harmonize_labs(labs, rules).iloc[0]
    assert result["result_category"] == expected
    assert result["harmonization_status"] == "HARMONIZED"


def test_threshold_binary_censored_positive() -> None:
    """A censored result decisively above a negative threshold is positive."""
    labs, rules = row(
        "THRESHOLD_BINARY",
        result_raw=">8.0",
        result_operator=">",
        result_numeric_bound=8.0,
        reference_operator="<",
        reference_bound=1.0,
    )
    result = MODULE.harmonize_labs(labs, rules).iloc[0]
    assert result["result_category"] == "POSITIVE"
    assert result["harmonization_source"] == "reference_range_censored"


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("present", "POSITIVE"),
        ("non reactive", "NEGATIVE"),
        ("borderline", "INDETERMINATE"),
    ],
)
def test_categorical_binary(token: str, expected: str) -> None:
    """Allowed qualitative tokens map without coercing ambiguity to positive."""
    labs, rules = row("CATEGORICAL_BINARY", result_text=token)
    assert MODULE.harmonize_labs(labs, rules).iloc[0]["result_category"] == expected


def test_multiclass_normalizes_only_case_and_spaces() -> None:
    """Multiclass values retain their meaning in a controlled representation."""
    labs, rules = row("CATEGORICAL_MULTICLASS", result_raw="  Few   ")
    result = MODULE.harmonize_labs(labs, rules).iloc[0]
    assert result["result_category"] == "FEW"
    assert result["result_raw"] == "  Few   "


@pytest.mark.parametrize(
    ("family", "status"), [("SPECIAL", "SKIPPED_SPECIAL"), ("EXCLUDE", "EXCLUDED")]
)
def test_non_feature_families_are_retained(family: str, status: str) -> None:
    """SPECIAL and EXCLUDE rows remain present with no derived category."""
    labs, rules = row(family, result_raw="evidence")
    result = MODULE.harmonize_labs(labs, rules).iloc[0]
    assert result["harmonization_status"] == status
    assert pd.isna(result["result_category"])


def test_continuous_only_preserves_numeric_without_category() -> None:
    """Continuous-only records preserve numeric evidence and invent no class."""
    labs, rules = row("CONTINUOUS_ONLY", result_numeric=85)
    result = MODULE.harmonize_labs(labs, rules).iloc[0]
    assert result["result_numeric"] == 85
    assert pd.isna(result["result_category"])
    assert result["harmonization_status"] == "HARMONIZED"


def test_unmatched_exact_pair_is_no_rule_and_appears_in_qc() -> None:
    """Absent exact pairs are nonfatal and reported for rule-map maintenance."""
    labs, rules = row("CONTINUOUS_ONLY")
    rules.loc[0, "Cluster Name"] = "Different cluster"
    harmonized = MODULE.harmonize_labs(labs, rules)
    assert harmonized.iloc[0]["harmonization_status"] == "NO_RULE"
    unmatched = MODULE.build_qc(harmonized)["unmatched_rule_pairs.csv"]
    assert unmatched.iloc[0].to_dict() == {
        "Order Name": "Order A",
        "Cluster Name": "Cluster A",
        "n_records": 1,
    }


def test_invalid_result_precedes_interpretation() -> None:
    """Invalid step-20 evidence never produces a clinical category."""
    labs, rules = row(
        "CATEGORICAL_BINARY", result_text="positive", result_valid_for_analysis=False
    )
    result = MODULE.harmonize_labs(labs, rules).iloc[0]
    assert result["harmonization_status"] == "INVALID_RESULT"
    assert pd.isna(result["result_category"])


def test_loader_rejects_duplicate_exact_pairs(tmp_path: Path) -> None:
    """Duplicate rule keys fail before a many-to-one merge can occur."""
    pytest.importorskip("openpyxl")
    path = tmp_path / "rules.xlsx"
    duplicate = pd.DataFrame(
        {
            "Order Name": ["A", "A"],
            "Cluster Name": ["B", "B"],
            "Suggested Rule Family": ["EXCLUDE", "SPECIAL"],
        }
    )
    duplicate.to_excel(path, sheet_name="Lab rule map", index=False)
    with pytest.raises(ValueError, match="duplicate join keys"):
        MODULE.load_rule_map(path)
