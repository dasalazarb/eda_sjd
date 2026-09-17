"""Reusable rule-family interpretation for longitudinal BTRIS laboratories."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

POSITIVE_TOKENS = {"POSITIVE", "POS", "REACTIVE", "DETECTED", "PRESENT"}
NEGATIVE_TOKENS = {
    "NEGATIVE",
    "NEG",
    "NONREACTIVE",
    "NON REACTIVE",
    "NOT DETECTED",
    "NOT DETECTABLE",
    "ABSENT",
}
INDETERMINATE_TOKENS = {"INDETERMINATE", "EQUIVOCAL", "BORDERLINE"}


@dataclass(frozen=True)
class HarmonizedResult:
    """A nullable controlled category and its evidence source."""

    category: Any
    source: str


def normalize_qualitative(value: Any) -> str:
    """Normalize case and whitespace without discarding the original evidence."""
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip()).upper()


def _number(value: Any) -> float | None:
    """Return a finite numeric scalar or ``None``."""
    parsed = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(parsed) else float(parsed)


def _exact_result(row: Mapping[str, Any]) -> float | None:
    """Prefer step 20's exact numeric field, with its legacy numeric fallback."""
    exact = _number(row.get("result_numeric_exact"))
    return exact if exact is not None else _number(row.get("result_numeric"))


def interpret_continuous_range(row: Mapping[str, Any]) -> HarmonizedResult:
    """Classify an exact or censored numeric result against its own range."""
    value = _exact_result(row)
    low = _number(row.get("reference_low"))
    high = _number(row.get("reference_high"))
    if value is not None and low is not None and high is not None:
        category = "LOW" if value < low else "HIGH" if value > high else "NORMAL"
        return HarmonizedResult(category, "reference_range_exact_numeric")

    operator = row.get("result_operator")
    bound = _number(row.get("result_numeric_bound"))
    if bound is None or low is None or high is None:
        return HarmonizedResult(pd.NA, "rule_family")
    if operator in {"<", "<="} and (bound < low or (bound == low and operator == "<")):
        return HarmonizedResult("LOW", "reference_range_censored")
    if operator in {">", ">="} and (
        bound > high or (bound == high and operator == ">")
    ):
        return HarmonizedResult("HIGH", "reference_range_censored")
    return HarmonizedResult(pd.NA, "rule_family")


def _matches_negative_reference(value: float, operator: str, threshold: float) -> bool:
    """Evaluate a value against a reference interval labelled negative."""
    return {
        "<": value < threshold,
        "<=": value <= threshold,
        ">": value > threshold,
        ">=": value >= threshold,
    }[operator]


def interpret_threshold_binary(row: Mapping[str, Any]) -> HarmonizedResult:
    """Classify threshold assays using the contemporaneous one-sided reference."""
    operator = row.get("reference_operator")
    threshold = _number(row.get("reference_bound"))
    if operator not in {"<", "<=", ">", ">="} or threshold is None:
        return HarmonizedResult(pd.NA, "rule_family")
    value = _exact_result(row)
    if value is not None:
        category = (
            "NEGATIVE"
            if _matches_negative_reference(value, operator, threshold)
            else "POSITIVE"
        )
        return HarmonizedResult(category, "reference_range_exact_numeric")

    result_operator = row.get("result_operator")
    bound = _number(row.get("result_numeric_bound"))
    if result_operator not in {"<", "<=", ">", ">="} or bound is None:
        return HarmonizedResult(pd.NA, "rule_family")
    if operator in {"<", "<="}:
        if result_operator in {">", ">="} and (
            bound > threshold
            or (bound == threshold and result_operator == ">")
            or (bound == threshold and result_operator == ">=" and operator == "<")
        ):
            return HarmonizedResult("POSITIVE", "reference_range_censored")
        if result_operator in {"<", "<="} and (
            bound < threshold
            or (bound == threshold and result_operator == "<")
            or (bound == threshold and result_operator == "<=" and operator == "<=")
        ):
            return HarmonizedResult("NEGATIVE", "reference_range_censored")
    else:
        if result_operator in {"<", "<="} and (
            bound < threshold
            or (bound == threshold and result_operator == "<")
            or (bound == threshold and result_operator == "<=" and operator == ">")
        ):
            return HarmonizedResult("POSITIVE", "reference_range_censored")
        if result_operator in {">", ">="} and (
            bound > threshold
            or (bound == threshold and result_operator == ">")
            or (bound == threshold and result_operator == ">=" and operator == ">=")
        ):
            return HarmonizedResult("NEGATIVE", "reference_range_censored")
    return HarmonizedResult(pd.NA, "rule_family")


def interpret_categorical_binary(row: Mapping[str, Any]) -> HarmonizedResult:
    """Map explicit binary and ambiguous qualitative tokens conservatively."""
    candidates = (
        (row.get("reported_interpretation"), "reported_interpretation"),
        (row.get("result_text"), "raw_qualitative_result"),
        (row.get("result_raw"), "raw_qualitative_result"),
    )
    for value, source in candidates:
        token = normalize_qualitative(value)
        if token in POSITIVE_TOKENS:
            return HarmonizedResult("POSITIVE", source)
        if token in NEGATIVE_TOKENS:
            return HarmonizedResult("NEGATIVE", source)
        if token in INDETERMINATE_TOKENS:
            return HarmonizedResult("INDETERMINATE", source)
    return HarmonizedResult(pd.NA, "rule_family")


def interpret_categorical_multiclass(row: Mapping[str, Any]) -> HarmonizedResult:
    """Return a normalized qualitative category without binary coercion."""
    for column in ("result_text", "result_raw"):
        token = normalize_qualitative(row.get(column))
        if token:
            return HarmonizedResult(token, "raw_qualitative_result")
    return HarmonizedResult(pd.NA, "rule_family")
