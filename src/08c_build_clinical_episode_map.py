"""Build a row-level clinical episode map and an episode manifest.

This step is operational QC only: it does not alter interval labels or define the
official baseline.  Rows are joined only when their populated content is
complementary and meet the documented temporal or interval-aware rules.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

from common import (
    ANALYTIC_DIR,
    INTERMEDIATE_DIR,
    MISSING_TOKENS,
    REPORTS_DIR,
    setup_logger,
)

INPUT_PATH = ANALYTIC_DIR / "visits_long.parquet"
ROW_MAP_PATH = INTERMEDIATE_DIR / "clinical_episode_row_map.parquet"
MANIFEST_PATH = ANALYTIC_DIR / "clinical_episode_manifest.parquet"
QC_DIR = REPORTS_DIR / "clinical_episode_map"
MISSED_BACKWARD_MERGES_FILENAME = "08c_possible_missed_backward_merges.csv"
BACKWARD_SUMMARY_FILENAME = "08c_backward_reconciliation_summary.csv"
BACKWARD_LOG_FILENAME = "08c_backward_reconciliation_log.csv"
EXTENDED_VISIT_REVIEW_FILENAME = "08c_extended_visit_exception_review.csv"
INTERVAL_EXTENDED_RECONCILIATION_FILENAME = (
    "08c_interval_extended_reconciliation.csv"
)
INTERVAL_CLUSTER_RECONCILIATION_FILENAME = (
    "08c_interval_cluster_reconciliation.csv"
)
INTERVAL_CLUSTER_SUMMARY_FILENAME = "08c_interval_cluster_summary.csv"
PREVIOUS_EPISODES_PATH = REPORTS_DIR / "visit_episode_audit" / "02_episode_summary.csv"
PREVIOUS_CANDIDATES_PATH = (
    REPORTS_DIR / "visit_episode_audit" / "08b_composite_episode_candidates.csv"
)

BLOCK_PREFIXES: dict[str, tuple[str, ...]] = {
    "has_essdai_form": ("essdai", "essdai-_r"),
    "has_esspri_form": ("esspri_questionnaire",),
    "has_eye_exam": ("eye_examination",),
    "has_salivary_flow": ("salivary_flow_form",),
    "has_systems_review": ("systems_review_for_physician",),
    "has_physical_exam": (
        "physical_examination",
        "physical_examination-initial_evaluation",
    ),
    "has_visit_summary": (
        "visit_summary_form",
        "visit_summary_-_2016_classification_criteria",
    ),
    "has_oral_exam": ("oral_exam_form",),
    "has_vital_signs": ("vital_signs",),
    "has_biopsy_pathology": ("biopsy_pathology",),
}
RESEARCH_PREFIXES: dict[str, tuple[str, ...]] = {
    "has_ccgo": ("ccgo",),
    "has_buccal_swab": ("buccal_swab_form",),
    "has_ipscs_specimen": ("ipscs_specimen",),
    "has_skin_biopsy": ("skin_biopsy",),
    "has_mucosal_biopsy": ("mucosal_biopsy",),
    "has_plaque_collection": ("plaque_collection",),
    "has_oral_rinse": ("oral_rinse",),
    "has_oral_rinse_plaque_collection": ("oral_rinse/plaque_collection",),
}
CORE_FLAGS = (
    "has_essdai_form",
    "has_systems_review",
    "has_physical_exam",
    "has_visit_summary",
)
OBJECTIVE_FLAGS = ("has_eye_exam", "has_salivary_flow", "has_oral_exam")
SUPPORT_FLAGS = ("has_esspri_form", "has_vital_signs", "has_biopsy_pathology")
ANCHOR_PRIORITY = (
    "has_essdai_form",
    "has_systems_review",
    "has_physical_exam",
    "has_visit_summary",
    "has_eye_exam",
    "has_salivary_flow",
    "has_oral_exam",
)
MISSING_UPPER = {str(value).strip().upper() for value in MISSING_TOKENS}


def parse_args() -> argparse.Namespace:
    """Parse input and output paths.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, default=INPUT_PATH)
    parser.add_argument("--row-map-path", type=Path, default=ROW_MAP_PATH)
    parser.add_argument("--manifest-path", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--qc-dir", type=Path, default=QC_DIR)
    parser.add_argument(
        "--previous-episodes-path", type=Path, default=PREVIOUS_EPISODES_PATH
    )
    return parser.parse_args()


def resolve_column(
    df: pd.DataFrame, names: Iterable[str], required: bool = True
) -> str | None:
    """Resolve the first exact or uniquely group-prefixed column name."""
    for name in names:
        if name in df.columns:
            return name
        matches = [
            str(column) for column in df.columns if str(column).endswith(f"__{name}")
        ]
        if len(matches) == 1:
            return matches[0]
    if required:
        raise ValueError(f"Required column not found; tried {list(names)}")
    return None


def has_information(series: pd.Series) -> pd.Series:
    """Identify valid populated values without treating zero or negative answers as missing."""
    result = series.notna()
    if pd.api.types.is_object_dtype(series.dtype) or pd.api.types.is_string_dtype(
        series.dtype
    ):
        text = series.astype("string").str.strip()
        result &= text.notna() & ~text.str.upper().isin(MISSING_UPPER)
    return result.fillna(False)


def _columns_for_prefixes(df: pd.DataFrame, prefixes: Iterable[str]) -> list[str]:
    accepted = set(prefixes)
    return [
        str(column)
        for column in df.columns
        if "__" in str(column)
        and str(column).split("__", 1)[0].strip().lower() in accepted
    ]


def add_presence_flags(visits: pd.DataFrame) -> pd.DataFrame:
    """Add all specified clinical and research/procedure presence flags."""
    result = visits.copy()
    definitions = {**BLOCK_PREFIXES, **RESEARCH_PREFIXES}
    for flag, prefixes in definitions.items():
        columns = _columns_for_prefixes(result, prefixes)
        result[flag] = False
        for column in columns:
            result[flag] |= has_information(result[column])

    total_columns = [
        column
        for column in (
            "essdai__essdai_total_score",
            "essdai-_r__essdai_total_score",
        )
        if column in result.columns
    ]
    result["has_essdai_total"] = False
    for column in total_columns:
        result["has_essdai_total"] |= has_information(result[column])
    esspri_core = [
        f"esspri_questionnaire__{name}" for name in ("dryness", "fatigue", "pain")
    ]
    result["has_esspri_core"] = False
    for column in esspri_core:
        if column in result.columns:
            result["has_esspri_core"] |= has_information(result[column])
    return _classify_evidence(result)


def _classify_evidence(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate evidence counts and the operational row classification."""
    result = frame.copy()
    result["physician_core_count"] = result[list(CORE_FLAGS)].sum(axis=1).astype(int)
    result["objective_exam_count"] = (
        result[list(OBJECTIVE_FLAGS)].sum(axis=1).astype(int)
    )
    result["clinical_candidate"] = (
        result["has_essdai_form"]
        | (result["physician_core_count"] >= 2)
        | (
            (result["physician_core_count"] >= 1)
            & (result["objective_exam_count"] >= 1)
        )
        | (result["objective_exam_count"] >= 2)
    )
    result["has_research_component"] = result[list(RESEARCH_PREFIXES)].any(axis=1)
    result["has_any_clinical_evidence"] = result[
        list(dict.fromkeys((*CORE_FLAGS, *OBJECTIVE_FLAGS, *SUPPORT_FLAGS)))
    ].any(axis=1)
    return result


def _aggregate_rows(rows: pd.DataFrame) -> pd.Series:
    flags = (
        list(BLOCK_PREFIXES)
        + ["has_essdai_total", "has_esspri_core"]
        + list(RESEARCH_PREFIXES)
    )
    values = {flag: bool(rows[flag].any()) for flag in flags}
    return _classify_evidence(pd.DataFrame([values])).iloc[0]


def build_daily_activity_units(
    flagged_rows: pd.DataFrame, provenance_columns: Iterable[str] = ()
) -> pd.DataFrame:
    """Consolidate raw rows into indivisible patient-date activity units.

    Parameters
    ----------
    flagged_rows : pd.DataFrame
        Prepared row-level visits with presence flags.
    provenance_columns : Iterable[str]
        Existing protocol, origin, or source columns to retain.

    Returns
    -------
    pd.DataFrame
        One row per dated patient-day. Undated raw rows remain separate units so
        that missing dates never cause unrelated records to be combined.
    """
    work = flagged_rows.copy()
    dated_key = work["collection_date"].dt.strftime("%Y-%m-%d")
    undated_key = "undated-row-" + work["row_id_raw"].astype("string")
    work["_daily_key"] = dated_key.fillna(undated_key)
    flag_columns = (
        list(BLOCK_PREFIXES)
        + ["has_essdai_total", "has_esspri_core"]
        + list(RESEARCH_PREFIXES)
    )
    records: list[dict[str, object]] = []
    for (patient_id, daily_key), rows in work.groupby(
        ["patient_id", "_daily_key"], sort=False, dropna=False
    ):
        record: dict[str, object] = {
            "patient_id": patient_id,
            "collection_date": rows["collection_date"].iloc[0],
            "daily_activity_unit_id": f"{patient_id}__{daily_key}",
            "row_ids_involved": tuple(rows["row_id_raw"].tolist()),
            "interval_names_involved": tuple(
                dict.fromkeys(rows["interval_name"].dropna().astype(str))
            ),
            "_source_order": int(rows["_source_order"].min()),
        }
        for column in flag_columns:
            record[column] = bool(rows[column].any())
        for column in provenance_columns:
            values = tuple(dict.fromkeys(rows[column].dropna().astype(str)))
            record[f"{column}_involved"] = values
        records.append(record)
    units = pd.DataFrame(records)
    if units.empty:
        return units
    return _classify_evidence(units).sort_values("_source_order").reset_index(drop=True)


def _merge_rule(left: pd.DataFrame, right: pd.DataFrame) -> str | None:
    """Return the temporal assignment rule when two groups are complementary."""
    dated = pd.concat([left["collection_date"], right["collection_date"]]).dropna()
    if dated.empty:
        return None
    span = int((dated.max() - dated.min()).days)
    if span > 14:
        return None
    before = _aggregate_rows(left)
    incoming = _aggregate_rows(right)
    combined = _aggregate_rows(pd.concat([left, right]))
    evidence_flags = list(BLOCK_PREFIXES) + list(RESEARCH_PREFIXES)
    adds_content = any(
        bool(incoming[flag]) and not bool(before[flag]) for flag in evidence_flags
    )
    if not adds_content:
        return None
    joins_essdai_esspri = bool(
        combined.has_essdai_form and combined.has_esspri_form
    ) and not (
        bool(before.has_essdai_form and before.has_esspri_form)
        or bool(incoming.has_essdai_form and incoming.has_esspri_form)
    )
    becomes_clinical = bool(combined.clinical_candidate) and not bool(
        before.clinical_candidate
    )
    completes_clinical = bool(before.clinical_candidate) and bool(
        incoming.has_any_clinical_evidence
    )
    if span <= 7 and (joins_essdai_esspri or becomes_clinical or completes_clinical):
        return "complementary_within_7_days"
    if span >= 8 and (joins_essdai_esspri or becomes_clinical or completes_clinical):
        return "qualifying_complement_within_8_14_days"
    return None


def assign_episodes(daily_units: pd.DataFrame) -> pd.DataFrame:
    """Assign every daily activity unit to one patient-specific temporal episode."""
    assigned: list[pd.DataFrame] = []
    for patient_id, patient_units in daily_units.groupby(
        "patient_id", sort=False, dropna=False
    ):
        ordered = patient_units.sort_values(
            ["collection_date", "_source_order"], na_position="last"
        )
        groups: list[tuple[pd.DataFrame, str]] = []
        for index in ordered.index:
            row = ordered.loc[[index]]
            rule = _merge_rule(groups[-1][0], row) if groups else None
            if rule:
                groups[-1] = (pd.concat([groups[-1][0], row]), rule)
            else:
                groups.append((row, "standalone_record"))
        for sequence, (rows, rule) in enumerate(groups, start=1):
            rows = rows.copy()
            rows["clinical_episode_id"] = f"{patient_id}__CE{sequence:04d}"
            rows["assignment_rule"] = rule
            assigned.append(rows)
    if not assigned:
        return daily_units.copy()
    return pd.concat(assigned).sort_values("_source_order")


def _episode_visit_type(rows: pd.DataFrame) -> str:
    """Return the operational visit type for a group of activity units."""
    evidence = _aggregate_rows(rows)
    has_core_or_objective = bool(
        evidence[list(CORE_FLAGS) + list(OBJECTIVE_FLAGS)].any()
    )
    if evidence.clinical_candidate:
        return "clinical_candidate"
    if evidence.has_research_component and not has_core_or_objective:
        return "research_or_procedure_only_candidate"
    return "ambiguous"


def _episode_interval_values(rows: pd.DataFrame) -> set[str]:
    """Return the original, nonmissing interval labels carried by an episode."""
    intervals: set[str] = set()
    for involved in rows["interval_names_involved"]:
        values = involved if isinstance(involved, (list, tuple, set)) else (involved,)
        for value in values:
            if pd.isna(value):
                continue
            interval = str(value).strip()
            if interval:
                intervals.add(interval)
    return intervals


def _normalized_interval_name(interval_name: str) -> str:
    """Normalize an interval label for comparison without changing provenance."""
    return re.sub(r"\s+", " ", interval_name.strip().casefold())


def _canonical_interval_name(interval_name: str) -> str:
    """Map a real CTDB interval label to its comparison-only category."""
    normalized = _normalized_interval_name(interval_name)
    exact_categories = {
        "natural history protocol 478 interval": "natural_history",
        "phase 1: initial full evaluation": "phase1_full_1",
        "phase 1: second full evaluation": "phase1_full_2",
        "phase 1: final full (third full) evaluation": "phase1_full_3",
        "phase 2: 4th full evaluation": "phase2_full",
        "phase 2: 5th full evaluation": "phase2_full",
    }
    if normalized in exact_categories:
        return exact_categories[normalized]
    if re.fullmatch(r"optional evaluation [1-4]", normalized):
        return "optional_evaluation"
    if re.fullmatch(r"15d optional evaluation [1-5]", normalized):
        return "natural_history_optional"
    return "unrecognized"


def _interval_compatibility(left: pd.DataFrame, right: pd.DataFrame) -> str:
    """Classify exact or natural-visit-family interval compatibility."""
    left_original = _episode_interval_values(left)
    right_original = _episode_interval_values(right)
    left_normalized = {
        _normalized_interval_name(interval) for interval in left_original
    }
    right_normalized = {
        _normalized_interval_name(interval) for interval in right_original
    }
    if left_normalized & right_normalized:
        return "exact_same_interval"
    natural_categories = {"natural_history", "natural_history_optional"}
    left_categories = {_canonical_interval_name(interval) for interval in left_original}
    right_categories = {
        _canonical_interval_name(interval) for interval in right_original
    }
    if (
        left_categories
        and right_categories
        and left_categories <= natural_categories
        and right_categories <= natural_categories
    ):
        return "natural_visit_family"
    return "none"


def _backward_merge_decision(
    left: pd.DataFrame, right: pd.DataFrame
) -> tuple[str | None, dict[str, object]]:
    """Evaluate consecutive episodes under temporal and interval-aware rules."""
    left_evidence = _aggregate_rows(left)
    right_evidence = _aggregate_rows(right)
    combined = _aggregate_rows(pd.concat([left, right]))
    left_type = _episode_visit_type(left)
    right_type = _episode_visit_type(right)
    left_dates = left["collection_date"].dropna()
    right_dates = right["collection_date"].dropna()
    details: dict[str, object] = {
        "a_visit_type": left_type,
        "b_visit_type": right_type,
        "a_has_essdai": bool(left_evidence.has_essdai_form),
        "a_has_esspri": bool(left_evidence.has_esspri_form),
        "b_has_essdai": bool(right_evidence.has_essdai_form),
        "b_has_esspri": bool(right_evidence.has_esspri_form),
        "combined_has_essdai": bool(combined.has_essdai_form),
        "combined_has_esspri": bool(combined.has_esspri_form),
        "special_extended_visit_exception": False,
        "interval_compatibility_type": "none",
        "clinical_components_added": "",
        "reunited_essdai_esspri": False,
        "eligible_pair": False,
    }
    if left_dates.empty or right_dates.empty:
        return None, details
    left_end = left_dates.max()
    right_start = right_dates.min()
    gap_days = int((right_start - left_end).days)
    combined_start = min(left_dates.min(), right_dates.min())
    combined_end = max(left_dates.max(), right_dates.max())
    combined_span = int((combined_end - combined_start).days)
    details.update(
        gap_days=gap_days,
        combined_span_days=combined_span,
        combined_start_date=combined_start,
        combined_end_date=combined_end,
    )
    if gap_days < 0:
        return None, details
    # A pair is eligible only when an established clinical episode is being
    # complemented by a clinically incomplete episode. Research-only content
    # can travel with such an episode, but is deliberately absent from evidence.
    left_candidate = bool(left_evidence.clinical_candidate)
    right_candidate = bool(right_evidence.clinical_candidate)
    clinical_flags = list(
        dict.fromkeys((*CORE_FLAGS, *OBJECTIVE_FLAGS, "has_esspri_form"))
    )
    if gap_days > 14:
        compatibility = _interval_compatibility(left, right)
        details["interval_compatibility_type"] = compatibility
        details["eligible_pair"] = True
        if compatibility not in {"exact_same_interval", "natural_visit_family"}:
            has_interval_information = bool(
                _episode_interval_values(left) and _episode_interval_values(right)
            )
            details["rejection"] = (
                "interval_incompatible_gt14"
                if has_interval_information
                else "missing_interval_information"
            )
            return None, details
        if left_candidate and right_candidate:
            details["rejection"] = "both_complete_clinical_episodes"
            return None, details
        if not (left_candidate or right_candidate):
            details["rejection"] = "no_clinical_complementarity_gt14"
            return None, details

    if not (left_candidate or right_candidate) or (left_candidate and right_candidate):
        return None, details
    candidate = left_evidence if left_candidate else right_evidence
    incomplete = right_evidence if left_candidate else left_evidence
    added_flags = [
        flag
        for flag in clinical_flags
        if bool(incomplete[flag]) and not bool(candidate[flag])
    ]
    reunited = bool(
        combined.has_essdai_form
        and combined.has_esspri_form
        and not (left_evidence.has_essdai_form and left_evidence.has_esspri_form)
        and not (right_evidence.has_essdai_form and right_evidence.has_esspri_form)
    )
    details["reunited_essdai_esspri"] = reunited
    details["clinical_components_added"] = "|".join(added_flags)
    details["eligible_pair"] = True
    if gap_days > 14:
        compatibility = str(details["interval_compatibility_type"])
        if not (reunited or added_flags):
            details["rejection"] = "no_clinical_complementarity_gt14"
            return None, details
        details["special_extended_visit_exception"] = True
        complement = (
            "reunites_essdai_esspri" if reunited else "adds_clinical_component"
        )
        return f"extended_{compatibility}_{complement}", details
    if combined_span > 14:
        compatibility = _interval_compatibility(left, right)
        details["interval_compatibility_type"] = compatibility
        strong_complementarity = (
            reunited
            or (compatibility == "exact_same_interval" and len(added_flags) >= 1)
            or (compatibility == "natural_visit_family" and len(added_flags) >= 2)
        )
        if compatibility == "none":
            details["rejection"] = "span_gt14_incompatible_intervals"
            return None, details
        if not strong_complementarity:
            details["rejection"] = "span_gt14_insufficient_complementarity"
            return None, details
        details["special_extended_visit_exception"] = True
        if reunited and compatibility == "natural_visit_family":
            return "extended_natural_visit_reunites_essdai_esspri", details
        if compatibility == "exact_same_interval":
            return "extended_same_interval_clinical_episode", details
        return "extended_natural_visit_family_episode", details
    if gap_days <= 7 and (reunited or added_flags):
        reason = (
            "reunites_essdai_esspri"
            if reunited
            else "incomplete_episode_adds_clinical_component:" + "|".join(added_flags)
        )
        return reason, details
    if gap_days >= 8 and (reunited or len(added_flags) >= 2):
        reason = (
            "reunites_essdai_esspri"
            if reunited
            else "multiple_complementary_clinical_components:" + "|".join(added_flags)
        )
        return reason, details
    details["rejection"] = "no_clinical_complementarity"
    return None, details


def backward_reconciliation(
    assigned_units: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Iteratively reconcile consecutive episodes and regenerate their IDs.

    Parameters
    ----------
    assigned_units : pd.DataFrame
        Daily activity units after :func:`assign_episodes`.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Reassigned units, one-row summary, merge log, and extended-span review.
    """
    log_columns = [
        "patient_id",
        "episode_a_original",
        "episode_b_original",
        "episode_a_start",
        "episode_a_end",
        "episode_b_start",
        "episode_b_end",
        "gap_days",
        "combined_span_days",
        "a_visit_type",
        "b_visit_type",
        "a_has_essdai",
        "a_has_esspri",
        "b_has_essdai",
        "b_has_esspri",
        "combined_has_essdai",
        "combined_has_esspri",
        "special_extended_visit_exception",
        "interval_compatibility_type",
        "merge_reason",
        "reunited_essdai_esspri",
        "new_clinical_episode_id",
    ]
    extended_review_columns = [
        "patient_id",
        "episode_a_original",
        "episode_b_original",
        "episode_a_start",
        "episode_a_end",
        "episode_b_start",
        "episode_b_end",
        "intervals_a",
        "intervals_b",
        "gap_days",
        "combined_span_days",
        "interval_compatibility_type",
        "a_visit_type",
        "b_visit_type",
        "a_has_essdai",
        "a_has_esspri",
        "b_has_essdai",
        "b_has_esspri",
        "clinical_components_added",
        "reunited_essdai_esspri",
        "merge_performed",
        "merge_reason",
        "new_clinical_episode_id",
    ]
    log_records: list[dict[str, object]] = []
    extended_review_records: list[dict[str, object]] = []
    rejected_span = 0
    rejected_complement = 0
    patient_merge_counts: dict[object, int] = {}
    final_groups: list[tuple[object, list[dict[str, object]]]] = []
    for patient_id, patient_units in assigned_units.groupby(
        "patient_id", sort=False, dropna=False
    ):
        groups: list[dict[str, object]] = []
        ordered = patient_units.sort_values(
            ["collection_date", "_source_order"], na_position="last"
        )
        for original_id, rows in ordered.groupby(
            "clinical_episode_id", sort=False, dropna=False
        ):
            groups.append({"rows": rows.copy(), "original_ids": [original_id]})
        index = 1
        while index < len(groups):
            left = groups[index - 1]
            right = groups[index]
            reason, details = _backward_merge_decision(left["rows"], right["rows"])
            if details.get("gap_days", 0) > 14:
                left_rows = left["rows"]
                right_rows = right["rows"]
                extended_review_records.append(
                    {
                        "patient_id": patient_id,
                        "episode_a_original": " | ".join(
                            map(str, left["original_ids"])
                        ),
                        "episode_b_original": " | ".join(
                            map(str, right["original_ids"])
                        ),
                        "episode_a_start": left_rows["collection_date"].min(),
                        "episode_a_end": left_rows["collection_date"].max(),
                        "episode_b_start": right_rows["collection_date"].min(),
                        "episode_b_end": right_rows["collection_date"].max(),
                        "intervals_a": " | ".join(
                            sorted(_episode_interval_values(left["rows"]))
                        ),
                        "intervals_b": " | ".join(
                            sorted(_episode_interval_values(right["rows"]))
                        ),
                        "gap_days": details["gap_days"],
                        "combined_span_days": details["combined_span_days"],
                        "interval_compatibility_type": details[
                            "interval_compatibility_type"
                        ],
                        "a_visit_type": details["a_visit_type"],
                        "b_visit_type": details["b_visit_type"],
                        "a_has_essdai": details["a_has_essdai"],
                        "a_has_esspri": details["a_has_esspri"],
                        "b_has_essdai": details["b_has_essdai"],
                        "b_has_esspri": details["b_has_esspri"],
                        "clinical_components_added": details[
                            "clinical_components_added"
                        ],
                        "reunited_essdai_esspri": details["reunited_essdai_esspri"],
                        "merge_performed": reason is not None,
                        "merge_reason": reason or details.get("rejection", "rejected"),
                        "new_clinical_episode_id": None,
                    }
                )
            if reason is None:
                if str(details.get("rejection", "")).startswith("span_gt14"):
                    rejected_span += 1
                elif details.get("rejection") == "no_clinical_complementarity":
                    rejected_complement += 1
                index += 1
                continue
            left_rows = left["rows"]
            right_rows = right["rows"]
            log_records.append(
                {
                    "patient_id": patient_id,
                    "episode_a_original": " | ".join(map(str, left["original_ids"])),
                    "episode_b_original": " | ".join(map(str, right["original_ids"])),
                    "episode_a_start": left_rows["collection_date"].min(),
                    "episode_a_end": left_rows["collection_date"].max(),
                    "episode_b_start": right_rows["collection_date"].min(),
                    "episode_b_end": right_rows["collection_date"].max(),
                    "gap_days": details["gap_days"],
                    "combined_span_days": details["combined_span_days"],
                    **{
                        key: details[key]
                        for key in (
                            "a_visit_type",
                            "b_visit_type",
                            "a_has_essdai",
                            "a_has_esspri",
                            "b_has_essdai",
                            "b_has_esspri",
                            "combined_has_essdai",
                            "combined_has_esspri",
                            "special_extended_visit_exception",
                            "interval_compatibility_type",
                        )
                    },
                    "merge_reason": reason,
                    "reunited_essdai_esspri": details["reunited_essdai_esspri"],
                    "new_clinical_episode_id": None,
                }
            )
            groups[index - 1] = {
                "rows": pd.concat([left_rows, right_rows]),
                "original_ids": left["original_ids"] + right["original_ids"],
            }
            del groups[index]
            patient_merge_counts[patient_id] = (
                patient_merge_counts.get(patient_id, 0) + 1
            )
            index = max(1, index - 1)
        final_groups.append((patient_id, groups))

    reconciled: list[pd.DataFrame] = []
    original_to_final: dict[tuple[object, str], str] = {}
    for patient_id, groups in final_groups:
        for sequence, group in enumerate(groups, start=1):
            episode_id = f"{patient_id}__CE{sequence:04d}"
            rows = group["rows"].copy()
            rows["clinical_episode_id"] = episode_id
            if len(group["original_ids"]) > 1:
                rows["assignment_rule"] = "backward_reconciliation"
            reconciled.append(rows)
            for original_id in group["original_ids"]:
                original_to_final[(patient_id, str(original_id))] = episode_id
    for record in log_records:
        original = str(record["episode_a_original"]).split(" | ")[0]
        record["new_clinical_episode_id"] = original_to_final[
            (record["patient_id"], original)
        ]
    for record in extended_review_records:
        original = str(record["episode_a_original"]).split(" | ")[0]
        record["new_clinical_episode_id"] = original_to_final[
            (record["patient_id"], original)
        ]
    result = (
        pd.concat(reconciled).sort_values("_source_order")
        if reconciled
        else assigned_units.copy()
    )
    log = pd.DataFrame(log_records).reindex(columns=log_columns)
    extended_review = pd.DataFrame(extended_review_records).reindex(
        columns=extended_review_columns
    )
    gaps = log["gap_days"] if not log.empty else pd.Series(dtype="int64")
    extended_merges = log.loc[log["special_extended_visit_exception"].eq(True)]
    reviewed_gt14 = extended_review
    merged_gt14 = reviewed_gt14.loc[reviewed_gt14["merge_performed"].eq(True)]
    summary = pd.DataFrame(
        [
            {
                "episodes_before_backward_reconciliation": assigned_units[
                    "clinical_episode_id"
                ].nunique(),
                "episodes_after_backward_reconciliation": result[
                    "clinical_episode_id"
                ].nunique(),
                "episodes_merged": len(log),
                "patients_affected": len(patient_merge_counts),
                "merges_gap_le3": int(gaps.le(3).sum()),
                "merges_gap_4_7": int(gaps.between(4, 7).sum()),
                "merges_gap_8_14": int(gaps.between(8, 14).sum()),
                "merges_reuniting_essdai_esspri": int(
                    log.get("reunited_essdai_esspri", pd.Series(dtype=bool)).sum()
                ),
                "merges_with_clinical_complementarity": len(log),
                "candidate_merges_rejected_span_gt14": rejected_span,
                "candidate_merges_rejected_no_clinical_complementarity": rejected_complement,
                "extended_visit_exception_merges": len(extended_merges),
                "extended_same_interval_merges": extended_merges[
                    "interval_compatibility_type"
                ]
                .eq("exact_same_interval")
                .sum(),
                "extended_natural_visit_family_merges": extended_merges[
                    "interval_compatibility_type"
                ]
                .eq("natural_visit_family")
                .sum(),
                "extended_visit_merges_reuniting_essdai_esspri": extended_merges[
                    "reunited_essdai_esspri"
                ].sum(),
                "max_span_days_extended_merge": extended_merges[
                    "combined_span_days"
                ].max(),
                "chains_with_multiple_merges": sum(
                    len(group["original_ids"]) > 2
                    for _, groups in final_groups
                    for group in groups
                ),
                "n_pairs_gt14_reviewed": len(reviewed_gt14),
                "n_pairs_gt14_merged": len(merged_gt14),
                "n_merged_exact_same_interval": int(
                    merged_gt14["interval_compatibility_type"]
                    .eq("exact_same_interval")
                    .sum()
                ),
                "n_merged_natural_visit_family": int(
                    merged_gt14["interval_compatibility_type"]
                    .eq("natural_visit_family")
                    .sum()
                ),
                "n_rejected_both_complete_clinical": int(
                    reviewed_gt14["merge_reason"]
                    .eq("both_complete_clinical_episodes")
                    .sum()
                ),
                "n_rejected_interval_incompatible": int(
                    reviewed_gt14["merge_reason"]
                    .eq("interval_incompatible_gt14")
                    .sum()
                ),
                "n_rejected_no_complementarity": int(
                    reviewed_gt14["merge_reason"]
                    .eq("no_clinical_complementarity_gt14")
                    .sum()
                ),
                "median_gap_days_gt14_merged": merged_gt14["gap_days"].median(),
                "max_gap_days_gt14_merged": merged_gt14["gap_days"].max(),
            }
        ]
    )
    return result, summary, log, extended_review


def _append_pipe_value(existing: object, value: str) -> str:
    """Append a pipe-delimited value without discarding existing provenance."""
    parts = [] if pd.isna(existing) else str(existing).split("|")
    return "|".join(dict.fromkeys([part for part in parts if part] + [value]))


def _interval_cluster_keys(episodes: list[pd.DataFrame]) -> list[list[int]]:
    """Return connected components of interval-compatible episodes."""
    parents = list(range(len(episodes)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left in range(len(episodes)):
        for right in range(left + 1, len(episodes)):
            if _interval_compatibility(episodes[left], episodes[right]) in {
                "exact_same_interval",
                "natural_visit_family",
            }:
                union(left, right)
    components: dict[int, list[int]] = {}
    for index in range(len(episodes)):
        components.setdefault(find(index), []).append(index)
    return list(components.values())


def interval_cluster_reconciliation(
    assigned_units: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reconcile non-adjacent, interval-compatible episode fragments.

    PASS 4 considers every compatible episode for a patient together, rather
    than only consecutive pairs. For an exact non-Natural-History interval, a
    single clinical candidate absorbs every other fragment without requiring
    complementarity. Natural History families retain complementarity rules,
    and multiple clinical candidates are never automatically combined.

    Parameters
    ----------
    assigned_units : pd.DataFrame
        Daily units carrying the episode IDs produced by PASS 1--3.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Reassigned units, cluster-level reconciliation audit, and one-row
        summary.
    """
    audit_columns = [
        "patient_id",
        "cluster_id",
        "intervals_involved",
        "episode_ids_before",
        "episode_start_dates_before",
        "episode_end_dates_before",
        "n_episodes_before",
        "n_clinical_candidate_before",
        "n_ambiguous_before",
        "cluster_start_date",
        "cluster_end_date",
        "cluster_span_days",
        "combined_has_essdai",
        "combined_has_esspri",
        "clinical_components_added",
        "merge_performed",
        "merge_reason",
        "manual_review_required",
        "manual_review_reason",
        "new_clinical_episode_id",
    ]
    clinical_flags = list(
        dict.fromkeys((*CORE_FLAGS, *OBJECTIVE_FLAGS, "has_esspri_form"))
    )
    records: list[dict[str, object]] = []
    patient_groups: list[tuple[object, list[dict[str, object]]]] = []
    episodes_before = assigned_units["clinical_episode_id"].nunique()
    episodes_absorbed = 0

    for patient_id, patient_units in assigned_units.groupby(
        "patient_id", sort=False, dropna=False
    ):
        ordered = patient_units.sort_values(
            ["collection_date", "_source_order"], na_position="last"
        )
        groups = [
            {"rows": rows.copy(), "original_ids": [str(episode_id)]}
            for episode_id, rows in ordered.groupby(
                "clinical_episode_id", sort=False, dropna=False
            )
        ]
        episodes = [group["rows"] for group in groups]
        clusters = list(enumerate(_interval_cluster_keys(episodes), start=1))
        # Mutating higher-position components first keeps the original indices
        # valid for every disjoint component.
        clusters.sort(key=lambda item: max(item[1]), reverse=True)
        for cluster_number, indices in clusters:
            if len(indices) < 2:
                continue
            cluster_groups = [groups[index] for index in indices]
            cluster_rows = [group["rows"] for group in cluster_groups]
            visit_types = [_episode_visit_type(rows) for rows in cluster_rows]
            n_candidates = visit_types.count("clinical_candidate")
            n_ambiguous = visit_types.count("ambiguous")
            combined_rows = pd.concat(cluster_rows)
            combined = _aggregate_rows(combined_rows)
            dates = combined_rows["collection_date"].dropna()
            start = dates.min() if not dates.empty else pd.NaT
            end = dates.max() if not dates.empty else pd.NaT
            span = int((end - start).days) if not dates.empty else pd.NA
            normalized_intervals = [
                {
                    _normalized_interval_name(interval)
                    for interval in _episode_interval_values(rows)
                }
                for rows in cluster_rows
            ]
            shared_exact_intervals = set.intersection(*normalized_intervals)
            aggressive_exact_interval = any(
                _canonical_interval_name(interval)
                not in {"natural_history", "natural_history_optional"}
                for interval in shared_exact_intervals
            )
            selected: list[int] = []
            added_flags: list[str] = []
            reason = "no_clinical_complementarity"

            if n_candidates >= 2:
                reason = (
                    "multiple_clinical_candidates_same_interval"
                    if aggressive_exact_interval
                    else "multiple_complete_clinical_episodes_same_interval"
                )
            elif n_candidates == 1:
                anchor = indices[visit_types.index("clinical_candidate")]
                if aggressive_exact_interval:
                    selected = sorted(indices)
                    current = _aggregate_rows(combined_rows)
                    reason = "interval_cluster_absorb_all_same_interval_fragments"
                else:
                    selected = [anchor]
                    current = _aggregate_rows(groups[anchor]["rows"])
                    # Evaluate every episode in the cluster; adjacency is irrelevant.
                    for index, visit_type in zip(indices, visit_types):
                        if index == anchor or visit_type != "ambiguous":
                            continue
                        fragment = _aggregate_rows(groups[index]["rows"])
                        additions = [
                            flag
                            for flag in clinical_flags
                            if bool(fragment[flag]) and not bool(current[flag])
                        ]
                        if additions:
                            selected.append(index)
                            added_flags.extend(additions)
                            current = _aggregate_rows(
                                pd.concat([groups[item]["rows"] for item in selected])
                            )
                if len(selected) > 1:
                    selected_evidence = [
                        _aggregate_rows(groups[index]["rows"])
                        for index in selected
                    ]
                    reunited = bool(
                        current.has_essdai_form
                        and current.has_esspri_form
                        and not any(
                            item.has_essdai_form and item.has_esspri_form
                            for item in selected_evidence
                        )
                    )
                    if not aggressive_exact_interval:
                        reason = (
                            "interval_cluster_reunites_essdai_esspri"
                            if reunited
                            else "interval_cluster_adds_clinical_component"
                        )
            elif bool(combined.clinical_candidate):
                useful = [
                    index
                    for index, visit_type in zip(indices, visit_types)
                    if visit_type == "ambiguous"
                    and bool(
                        _aggregate_rows(groups[index]["rows"])[clinical_flags].any()
                    )
                ]
                if len(useful) >= 2 and bool(
                    _aggregate_rows(
                        pd.concat([groups[index]["rows"] for index in useful])
                    ).clinical_candidate
                ):
                    selected = useful
                    reason = "interval_cluster_adds_clinical_component"

            merge_performed = len(selected) > 1
            if merge_performed and n_candidates >= 2:
                raise RuntimeError(
                    "PASS 4 attempted to merge multiple complete clinical episodes"
                )
            if merge_performed and not aggressive_exact_interval and any(
                visit_types[indices.index(index)]
                == "research_or_procedure_only_candidate"
                for index in selected
            ):
                raise RuntimeError(
                    "PASS 4 attempted to absorb a research-only episode"
                )
            if (
                merge_performed
                and n_candidates == 1
                and not aggressive_exact_interval
                and not added_flags
            ):
                raise RuntimeError(
                    "PASS 4 merged an ambiguous fragment without complementarity"
                )
            merge_target_before: str | None = None
            if merge_performed:
                target = selected[0]
                merge_target_before = groups[target]["original_ids"][0]
                episodes_absorbed += len(selected) - 1
                merged_rows = pd.concat([groups[index]["rows"] for index in selected])
                merged_rows = merged_rows.copy()
                merged_rows["assignment_rule"] = merged_rows["assignment_rule"].map(
                    lambda value: _append_pipe_value(value, reason)
                )
                if pd.notna(span) and span > 90:
                    merged_rows["pass4_manual_review_reason"] = merged_rows.get(
                        "pass4_manual_review_reason", pd.Series(index=merged_rows.index)
                    ).map(
                        lambda value: _append_pipe_value(
                            value, "interval_cluster_span_gt90_days"
                        )
                    )
                groups[target] = {
                    "rows": merged_rows,
                    "original_ids": sum(
                        (groups[index]["original_ids"] for index in selected), []
                    ),
                }
                for index in sorted(selected[1:], reverse=True):
                    del groups[index]

            records.append(
                {
                    "patient_id": patient_id,
                    "cluster_id": f"{patient_id}__IC{cluster_number:04d}",
                    "intervals_involved": " | ".join(
                        sorted(_episode_interval_values(combined_rows))
                    ),
                    "episode_ids_before": " | ".join(
                        group["original_ids"][0] for group in cluster_groups
                    ),
                    "episode_start_dates_before": " | ".join(
                        str(rows["collection_date"].min().date())
                        if rows["collection_date"].notna().any()
                        else ""
                        for rows in cluster_rows
                    ),
                    "episode_end_dates_before": " | ".join(
                        str(rows["collection_date"].max().date())
                        if rows["collection_date"].notna().any()
                        else ""
                        for rows in cluster_rows
                    ),
                    "n_episodes_before": len(indices),
                    "n_clinical_candidate_before": n_candidates,
                    "n_ambiguous_before": n_ambiguous,
                    "cluster_start_date": start,
                    "cluster_end_date": end,
                    "cluster_span_days": span,
                    "combined_has_essdai": bool(combined.has_essdai_form),
                    "combined_has_esspri": bool(combined.has_esspri_form),
                    "clinical_components_added": "|".join(
                        dict.fromkeys(added_flags)
                    ),
                    "merge_performed": merge_performed,
                    "merge_reason": reason,
                    "manual_review_required": bool(
                        (aggressive_exact_interval and n_candidates >= 2)
                        or (pd.notna(span) and span > 90)
                    ),
                    "manual_review_reason": "|".join(
                        reason
                        for reason, applies in (
                            (
                                "multiple_clinical_candidates_same_interval",
                                aggressive_exact_interval and n_candidates >= 2,
                            ),
                            (
                                "interval_cluster_span_gt90_days",
                                pd.notna(span) and span > 90,
                            ),
                        )
                        if applies
                    ),
                    "new_clinical_episode_id": None,
                    "_merge_target_before": merge_target_before,
                }
            )
        patient_groups.append((patient_id, groups))

    reconciled: list[pd.DataFrame] = []
    original_to_final: dict[tuple[object, str], str] = {}
    for patient_id, groups in patient_groups:
        groups.sort(
            key=lambda group: (
                group["rows"]["collection_date"].min(),
                group["rows"]["_source_order"].min(),
            )
        )
        for sequence, group in enumerate(groups, start=1):
            episode_id = f"{patient_id}__CE{sequence:04d}"
            rows = group["rows"].copy()
            rows["clinical_episode_id"] = episode_id
            reconciled.append(rows)
            for original_id in group["original_ids"]:
                original_to_final[(patient_id, original_id)] = episode_id
    audit_work = pd.DataFrame(records)
    if not audit_work.empty:
        for index, record in audit_work.loc[
            audit_work["merge_performed"].eq(True)
        ].iterrows():
            audit_work.at[index, "new_clinical_episode_id"] = original_to_final[
                (record["patient_id"], str(record["_merge_target_before"]))
            ]
    audit = audit_work.reindex(columns=audit_columns)
    result = (
        pd.concat(reconciled).sort_values("_source_order")
        if reconciled
        else assigned_units.copy()
    )
    merged = audit.loc[audit["merge_performed"].eq(True)]
    summary = pd.DataFrame(
        [
            {
                "n_clusters_evaluated": len(audit),
                "n_clusters_merged": len(merged),
                "n_episodes_absorbed": episodes_absorbed,
                "n_patients_affected": merged["patient_id"].nunique(),
                "n_reunited_essdai_esspri": merged["merge_reason"]
                .eq("interval_cluster_reunites_essdai_esspri")
                .sum(),
                "n_rejected_multiple_clinical_candidates": audit["merge_reason"]
                .isin(
                    {
                        "multiple_clinical_candidates_same_interval",
                        "multiple_complete_clinical_episodes_same_interval",
                    }
                )
                .sum(),
                "n_rejected_no_complementarity": audit["merge_reason"]
                .eq("no_clinical_complementarity")
                .sum(),
                "n_manual_review_gt90_days": audit["manual_review_required"].sum(),
                "episodes_before_pass4": episodes_before,
                "episodes_after_pass4": result["clinical_episode_id"].nunique(),
            }
        ]
    )
    if len(result) != len(assigned_units):
        raise RuntimeError("PASS 4 changed the number of daily activity units")
    before_raw_rows = assigned_units["row_ids_involved"].explode()
    after_raw_rows = result["row_ids_involved"].explode()
    if after_raw_rows.duplicated().any():
        raise RuntimeError("PASS 4 assigned a raw row more than once")
    if sorted(map(str, before_raw_rows)) != sorted(map(str, after_raw_rows)):
        raise RuntimeError("PASS 4 changed raw-row membership")
    immutable_columns = [
        "daily_activity_unit_id",
        "collection_date",
        "interval_names_involved",
        *BLOCK_PREFIXES,
        *RESEARCH_PREFIXES,
    ]
    before_values = assigned_units[immutable_columns].sort_values(
        "daily_activity_unit_id"
    )
    after_values = result[immutable_columns].sort_values("daily_activity_unit_id")
    if not before_values.reset_index(drop=True).equals(
        after_values.reset_index(drop=True)
    ):
        raise RuntimeError("PASS 4 changed dates, intervals, or clinical values")
    if summary.at[0, "episodes_after_pass4"] > episodes_before:
        raise RuntimeError("PASS 4 unexpectedly increased the episode count")
    return result, audit, summary


def propagate_episode_assignments(
    flagged_rows: pd.DataFrame, assigned_units: pd.DataFrame
) -> pd.DataFrame:
    """Propagate each daily unit's episode assignment back to all raw rows."""
    membership = assigned_units[
        [
            "daily_activity_unit_id",
            "row_ids_involved",
            "clinical_episode_id",
            "assignment_rule",
        ]
    ].explode("row_ids_involved")
    membership = membership.rename(columns={"row_ids_involved": "row_id_raw"})
    return flagged_rows.merge(
        membership,
        on="row_id_raw",
        how="left",
        validate="one_to_one",
    ).sort_values("_source_order")


def _anchor_date(rows: pd.DataFrame) -> pd.Timestamp:
    for flag in ANCHOR_PRIORITY:
        dates = rows.loc[rows[flag], "collection_date"].dropna()
        if not dates.empty:
            return dates.min()
    return pd.NaT


def build_source_interval_qc(prepared_rows: pd.DataFrame) -> pd.DataFrame:
    """Calculate original patient-interval date ranges before episode clustering.

    Parameters
    ----------
    prepared_rows : pd.DataFrame
        Normalized raw rows containing patient, interval, and collection date.

    Returns
    -------
    pd.DataFrame
        One row per original patient-interval with its source date range and
        greater-than-30-day review flag.
    """
    source_intervals = prepared_rows.groupby(
        ["patient_id", "interval_name"], as_index=False, dropna=False
    ).agg(
        source_interval_start_date=("collection_date", "min"),
        source_interval_end_date=("collection_date", "max"),
    )
    source_intervals["source_interval_span_days"] = (
        source_intervals["source_interval_end_date"]
        - source_intervals["source_interval_start_date"]
    ).dt.days.astype("Int64")
    source_intervals["source_interval_span_gt30"] = (
        source_intervals["source_interval_span_days"].gt(30).fillna(False)
    )
    return source_intervals


def _source_interval_key(
    patient_id: object, interval_name: object
) -> tuple[object, str]:
    """Return a stable lookup key, including for a missing interval label."""
    interval_key = "<MISSING>" if pd.isna(interval_name) else str(interval_name)
    return patient_id, interval_key


def _overlapping_source_interval_keys(
    source_intervals: pd.DataFrame,
) -> set[tuple[object, str]]:
    """Find original patient-interval ranges that overlap another named interval."""
    dated = source_intervals.dropna(
        subset=[
            "interval_name",
            "source_interval_start_date",
            "source_interval_end_date",
        ]
    )
    overlapping: set[tuple[object, str]] = set()
    for patient_id, rows in dated.groupby("patient_id", sort=False, dropna=False):
        values = list(rows.itertuples(index=False))
        for position, left in enumerate(values):
            for right in values[position + 1 :]:
                if (
                    left.source_interval_start_date <= right.source_interval_end_date
                    and right.source_interval_start_date
                    <= left.source_interval_end_date
                ):
                    overlapping.add(
                        _source_interval_key(patient_id, left.interval_name)
                    )
                    overlapping.add(
                        _source_interval_key(patient_id, right.interval_name)
                    )
    return overlapping


def build_manifest(
    assigned: pd.DataFrame,
    source_intervals: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Collapse assigned source rows into the requested episode manifest."""
    if source_intervals is None:
        source_intervals = build_source_interval_qc(assigned)
    interval_lookup = {
        _source_interval_key(row.patient_id, row.interval_name): row
        for row in source_intervals.itertuples(index=False)
    }
    overlapping_keys = _overlapping_source_interval_keys(source_intervals)
    records: list[dict[str, object]] = []
    output_flags = list(BLOCK_PREFIXES) + ["has_essdai_total", "has_esspri_core"]
    for (patient_id, episode_id), rows in assigned.groupby(
        ["patient_id", "clinical_episode_id"], sort=False, dropna=False
    ):
        evidence = _aggregate_rows(rows)
        dates = rows["collection_date"].dropna()
        start = dates.min() if not dates.empty else pd.NaT
        end = dates.max() if not dates.empty else pd.NaT
        span = int((end - start).days) if pd.notna(start) and pd.notna(end) else pd.NA
        has_core_or_objective = bool(
            evidence[list(CORE_FLAGS) + list(OBJECTIVE_FLAGS)].any()
        )
        research_only = bool(
            evidence.has_research_component and not has_core_or_objective
        )
        if evidence.clinical_candidate:
            visit_type = "clinical_candidate"
        elif research_only:
            visit_type = "research_or_procedure_only_candidate"
        else:
            visit_type = "ambiguous"
        interval_keys = {
            _source_interval_key(patient_id, interval_name)
            for interval_name in rows["interval_name"].unique()
        }
        interval_qc = [
            interval_lookup[key] for key in interval_keys if key in interval_lookup
        ]
        source_spans = [
            item.source_interval_span_days
            for item in interval_qc
            if pd.notna(item.source_interval_span_days)
        ]
        max_source_span = max(source_spans) if source_spans else pd.NA
        reasons: list[str] = []
        if any(bool(item.source_interval_span_gt30) for item in interval_qc):
            reasons.append("source_interval_span_gt30")
        if interval_keys & overlapping_keys:
            reasons.append("overlapping_source_interval_ranges")
        if rows["collection_date"].isna().any():
            reasons.append("missing_collection_date")
        if "pass4_manual_review_reason" in rows:
            for value in rows["pass4_manual_review_reason"].dropna():
                reasons.extend(part for part in str(value).split("|") if part)
        reasons = list(dict.fromkeys(reasons))
        intervals = sorted({str(value) for value in rows["interval_name"].dropna()})
        record: dict[str, object] = {
            "patient_id": patient_id,
            "clinical_episode_id": episode_id,
            "intervals_involved": " | ".join(intervals),
            "episode_start_date": start,
            "clinical_anchor_date": _anchor_date(rows),
            "episode_end_date": end,
            "episode_span_days": span,
            "max_source_interval_span_days": max_source_span,
            "physician_core_count": int(evidence.physician_core_count),
            "objective_exam_count": int(evidence.objective_exam_count),
            "visit_type": visit_type,
            "clinical_visit": bool(evidence.clinical_candidate),
            "manual_review_required": bool(reasons),
            "manual_review_reason": "|".join(reasons),
        }
        record.update({flag: bool(evidence[flag]) for flag in output_flags})
        records.append(record)
    columns = [
        "patient_id",
        "clinical_episode_id",
        "intervals_involved",
        "episode_start_date",
        "clinical_anchor_date",
        "episode_end_date",
        "episode_span_days",
        "has_essdai_form",
        "has_essdai_total",
        "has_esspri_form",
        "has_esspri_core",
        "has_systems_review",
        "has_physical_exam",
        "has_visit_summary",
        "has_eye_exam",
        "has_salivary_flow",
        "has_oral_exam",
        "physician_core_count",
        "objective_exam_count",
        "visit_type",
        "clinical_visit",
        "manual_review_required",
        "manual_review_reason",
        "max_source_interval_span_days",
    ]
    return pd.DataFrame(records).reindex(columns=columns)


def prepare_visits(visits: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Normalize identifiers and dates while retaining original provenance columns."""
    patient_col = resolve_column(visits, ("patient_id", "patient_record_number"))
    interval_col = resolve_column(visits, ("interval_name",))
    date_col = resolve_column(
        visits, ("collection_date", "visit_date", "visit_datetime")
    )
    row_col = resolve_column(visits, ("row_id_raw",), required=False)
    provenance = [
        str(column)
        for column in visits.columns
        if any(term in str(column).lower() for term in ("protocol", "origin", "source"))
    ]
    result = visits.copy()
    result["_source_order"] = range(len(result))
    result["patient_id"] = result[patient_col]
    result["interval_name"] = result[interval_col]
    result["collection_date"] = pd.to_datetime(
        result[date_col], errors="coerce"
    ).dt.normalize()
    result["row_id_raw"] = result[row_col] if row_col else result["_source_order"]
    if result["patient_id"].isna().any():
        raise ValueError(
            "patient_id contains missing values; episode IDs cannot be made safely"
        )
    if result["row_id_raw"].isna().any() or result["row_id_raw"].duplicated().any():
        raise ValueError("row_id_raw must be complete and unique")
    return result, list(dict.fromkeys(provenance))


def build_qc(
    assigned_rows: pd.DataFrame,
    assigned_units: pd.DataFrame,
    manifest: pd.DataFrame,
    source_intervals: pd.DataFrame,
    missed_backward_merges: pd.DataFrame | None = None,
    backward_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create the requested one-row QC metric table."""
    if missed_backward_merges is None:
        missed_backward_merges = build_missed_backward_merge_qc(manifest)
    duplicated_assignments = assigned_rows.groupby("row_id_raw")[
        "clinical_episode_id"
    ].nunique()
    split_units = assigned_rows.groupby("daily_activity_unit_id")[
        "clinical_episode_id"
    ].nunique()
    unique_patient_dates = assigned_rows[
        ["patient_id", "collection_date"]
    ].drop_duplicates()
    reunited = manifest["has_essdai_form"] & manifest["has_esspri_form"]
    long_source_intervals = source_intervals.loc[
        source_intervals["source_interval_span_gt30"]
    ]
    has_long_source_interval = manifest["manual_review_reason"].str.contains(
        "source_interval_span_gt30", regex=False, na=False
    )
    has_source_overlap = manifest["manual_review_reason"].str.contains(
        "overlapping_source_interval_ranges", regex=False, na=False
    )
    metrics = {
        "patients": assigned_rows["patient_id"].nunique(),
        "raw_rows": len(assigned_rows),
        "unique_patient_collection_dates": len(unique_patient_dates),
        "daily_activity_units": len(assigned_units),
        "final_clinical_episode_ids": manifest["clinical_episode_id"].nunique(),
        "clinical_candidate": manifest["visit_type"].eq("clinical_candidate").sum(),
        "research_or_procedure_only_candidate": manifest["visit_type"]
        .eq("research_or_procedure_only_candidate")
        .sum(),
        "ambiguous": manifest["visit_type"].eq("ambiguous").sum(),
        "manual_review": manifest["manual_review_required"].sum(),
        "source_intervals_span_gt30": len(long_source_intervals),
        "patients_with_source_interval_span_gt30": long_source_intervals[
            "patient_id"
        ].nunique(),
        "clinical_episodes_with_source_interval_span_gt30": has_long_source_interval.sum(),
        "clinical_episodes_with_overlapping_source_interval_ranges": has_source_overlap.sum(),
        "episodes_with_multiple_intervals": manifest["intervals_involved"]
        .str.contains(" \\| ", regex=True, na=False)
        .sum(),
        "episodes_with_essdai_and_esspri": reunited.sum(),
        "raw_rows_unassigned": assigned_rows["clinical_episode_id"].isna().sum(),
        "raw_rows_multiply_assigned": (duplicated_assignments > 1).sum(),
        "patient_date_units_assigned_to_multiple_episodes": (split_units > 1).sum(),
        "possible_missed_backward_merge_pairs": len(missed_backward_merges),
        "patients_with_possible_missed_backward_merge": missed_backward_merges[
            "patient_id"
        ].nunique(),
        "possible_merges_with_essdai_esspri_reunited": missed_backward_merges[
            "would_reunite_essdai_esspri"
        ].sum(),
        "possible_merges_becoming_clinical": missed_backward_merges[
            "would_become_clinical"
        ].sum(),
    }
    if backward_summary is not None:
        backward = backward_summary.iloc[0]
        metrics.update(
            {
                "episodes_before_backward_reconciliation": backward[
                    "episodes_before_backward_reconciliation"
                ],
                "episodes_after_backward_reconciliation": backward[
                    "episodes_after_backward_reconciliation"
                ],
                "backward_merges_performed": backward["episodes_merged"],
                "patients_with_backward_merges": backward["patients_affected"],
                "essdai_esspri_pairs_recovered": backward[
                    "merges_reuniting_essdai_esspri"
                ],
                "remaining_possible_missed_backward_merges": len(
                    missed_backward_merges
                ),
            }
        )
    for label in ("gap_le_3_days", "gap_4_7_days", "gap_8_14_days"):
        metrics[f"possible_missed_backward_merge_pairs_{label}"] = (
            missed_backward_merges["gap_group"].eq(label).sum()
        )
    return pd.DataFrame([metrics])


def _combined_episode_evidence(left: pd.Series, right: pd.Series) -> pd.Series:
    """Combine manifest evidence without changing either episode assignment."""
    flags = list(BLOCK_PREFIXES) + ["has_essdai_total", "has_esspri_core"]
    values = {
        flag: bool(left.get(flag, False) or right.get(flag, False)) for flag in flags
    }
    values.update({flag: False for flag in RESEARCH_PREFIXES})
    return _classify_evidence(pd.DataFrame([values])).iloc[0]


def build_missed_backward_merge_qc(manifest: pd.DataFrame) -> pd.DataFrame:
    """Identify consecutive episodes that merit backward-merge review.

    Parameters
    ----------
    manifest : pd.DataFrame
        Episode-level manifest produced by :func:`build_manifest`.

    Returns
    -------
    pd.DataFrame
        Candidate pairs only. This audit does not mutate episode identifiers,
        assignment rules, or the manifest itself.
    """
    columns = [
        "patient_id",
        "episode_a_id",
        "episode_b_id",
        "episode_a_start_date",
        "episode_a_end_date",
        "episode_b_start_date",
        "episode_b_end_date",
        "gap_days",
        "gap_group",
        "episode_a_visit_type",
        "episode_b_visit_type",
        "episode_a_intervals",
        "episode_b_intervals",
        "a_has_essdai",
        "a_has_esspri",
        "b_has_essdai",
        "b_has_esspri",
        "combined_has_essdai",
        "combined_has_esspri",
        "combined_physician_core_count",
        "combined_objective_exam_count",
        "would_reunite_essdai_esspri",
        "would_become_clinical",
        "possible_missed_backward_merge",
        "review_reason",
    ]
    records: list[dict[str, object]] = []
    # The manifest intentionally does not retain vital-signs or pathology flags.
    # ESSPRI is the only support flag needed to identify a complementary PRO episode.
    clinical_detail_flags = (*CORE_FLAGS, *OBJECTIVE_FLAGS, "has_esspri_form")
    for patient_id, episodes in manifest.groupby(
        "patient_id", sort=False, dropna=False
    ):
        ordered = episodes.assign(
            _review_date=episodes["clinical_anchor_date"].fillna(
                episodes["episode_start_date"]
            )
        ).sort_values(["_review_date", "episode_start_date", "clinical_episode_id"])
        pairs = zip(ordered.iloc[:-1].iterrows(), ordered.iloc[1:].iterrows())
        for (_, left), (_, right) in pairs:
            if pd.isna(left["episode_end_date"]) or pd.isna(
                right["episode_start_date"]
            ):
                continue
            gap_days = int(
                (right["episode_start_date"] - left["episode_end_date"]).days
            )
            if gap_days < 0 or gap_days > 14:
                continue
            combined = _combined_episode_evidence(left, right)
            left_clinical = bool(left["clinical_visit"])
            right_clinical = bool(right["clinical_visit"])
            has_incomplete_episode = not left_clinical or not right_clinical
            both_research_only = (
                left["visit_type"]
                == right["visit_type"]
                == ("research_or_procedure_only_candidate")
            )
            reunited = bool(
                combined.has_essdai_form
                and combined.has_esspri_form
                and not (left.has_essdai_form and left.has_esspri_form)
                and not (right.has_essdai_form and right.has_esspri_form)
            )
            becomes_clinical = bool(
                combined.clinical_candidate and not left_clinical and not right_clinical
            )
            incomplete = left if not left_clinical else right
            incomplete_complements_candidate = bool(
                left_clinical != right_clinical
                and any(bool(incomplete[flag]) for flag in clinical_detail_flags)
                and any(
                    bool(left[flag]) != bool(right[flag])
                    for flag in clinical_detail_flags
                )
            )
            coherent_clinical_complement = bool(
                combined.physician_core_count + combined.objective_exam_count >= 2
                and any(
                    bool(left[flag]) and not bool(right[flag])
                    for flag in (*CORE_FLAGS, *OBJECTIVE_FLAGS)
                )
                and any(
                    bool(right[flag]) and not bool(left[flag])
                    for flag in (*CORE_FLAGS, *OBJECTIVE_FLAGS)
                )
            )
            reasons = []
            if reunited:
                reasons.append("reunites_essdai_esspri")
            if becomes_clinical:
                reasons.append("combined_evidence_becomes_clinical")
            if incomplete_complements_candidate:
                reasons.append("incomplete_episode_complements_clinical_candidate")
            if coherent_clinical_complement:
                reasons.append("coherent_clinical_components_reunited")
            possible = bool(
                has_incomplete_episode and not both_research_only and reasons
            )
            if not possible:
                continue
            gap_group = (
                "gap_le_3_days"
                if gap_days <= 3
                else "gap_4_7_days" if gap_days <= 7 else "gap_8_14_days"
            )
            records.append(
                {
                    "patient_id": patient_id,
                    "episode_a_id": left["clinical_episode_id"],
                    "episode_b_id": right["clinical_episode_id"],
                    "episode_a_start_date": left["episode_start_date"],
                    "episode_a_end_date": left["episode_end_date"],
                    "episode_b_start_date": right["episode_start_date"],
                    "episode_b_end_date": right["episode_end_date"],
                    "gap_days": gap_days,
                    "gap_group": gap_group,
                    "episode_a_visit_type": left["visit_type"],
                    "episode_b_visit_type": right["visit_type"],
                    "episode_a_intervals": left["intervals_involved"],
                    "episode_b_intervals": right["intervals_involved"],
                    "a_has_essdai": bool(left["has_essdai_form"]),
                    "a_has_esspri": bool(left["has_esspri_form"]),
                    "b_has_essdai": bool(right["has_essdai_form"]),
                    "b_has_esspri": bool(right["has_esspri_form"]),
                    "combined_has_essdai": bool(combined.has_essdai_form),
                    "combined_has_esspri": bool(combined.has_esspri_form),
                    "combined_physician_core_count": int(combined.physician_core_count),
                    "combined_objective_exam_count": int(combined.objective_exam_count),
                    "would_reunite_essdai_esspri": reunited,
                    "would_become_clinical": becomes_clinical,
                    "possible_missed_backward_merge": possible,
                    "review_reason": "|".join(reasons),
                }
            )
    return pd.DataFrame(records).reindex(columns=columns)


def manual_review_reason_distribution(manifest: pd.DataFrame) -> pd.DataFrame:
    """Count each combinable manual-review reason across clinical episodes."""
    reasons = manifest.loc[
        manifest["manual_review_reason"].ne(""),
        ["clinical_episode_id", "manual_review_reason"],
    ].copy()
    if reasons.empty:
        return pd.DataFrame(columns=["manual_review_reason", "clinical_episodes"])
    reasons["manual_review_reason"] = reasons["manual_review_reason"].str.split("|")
    return (
        reasons.explode("manual_review_reason")
        .groupby("manual_review_reason", as_index=False)
        .agg(clinical_episodes=("clinical_episode_id", "nunique"))
        .sort_values("manual_review_reason")
        .reset_index(drop=True)
    )


def compare_previous(manifest: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Compare new classifications with 08b when its episode summary is available."""
    columns = ["previous_visit_type", "visit_type", "episodes"]
    if not path.exists():
        return pd.DataFrame(columns=columns)
    previous = pd.read_csv(path)
    if {
        "window_days",
        "patient_id",
        "intervals_involved",
        "candidate_visit_types_involved",
    }.issubset(previous.columns):
        previous = previous.loc[previous["window_days"].eq(14)].copy()
        previous["interval_name"] = previous["intervals_involved"].str.split("|")
        previous = previous.explode("interval_name").rename(
            columns={"candidate_visit_types_involved": "candidate_visit_type"}
        )
    previous_type = "candidate_visit_type"
    required = {"patient_id", "interval_name", previous_type}
    if not required.issubset(previous.columns):
        return pd.DataFrame(columns=columns)
    exploded = manifest.assign(
        interval_name=manifest["intervals_involved"].str.split(" \\| ", regex=True)
    ).explode("interval_name")
    comparison = exploded.merge(
        previous[["patient_id", "interval_name", previous_type]],
        on=["patient_id", "interval_name"],
        how="left",
    )
    return (
        comparison.groupby([previous_type, "visit_type"], dropna=False)
        .size()
        .rename("episodes")
        .reset_index()
        .rename(columns={previous_type: "previous_visit_type"})
        .reindex(columns=columns)
    )


def write_parquet_and_csv(frame: pd.DataFrame, parquet_path: Path) -> tuple[Path, Path]:
    """Write an output table in both typed and human-readable formats.

    Parameters
    ----------
    frame : pd.DataFrame
        Table to persist.
    parquet_path : Path
        Parquet destination. The CSV is written beside it with the same stem.

    Returns
    -------
    tuple[Path, Path]
        Paths of the Parquet and CSV files, respectively.
    """
    parquet_path = parquet_path.with_suffix(".parquet")
    csv_path = parquet_path.with_suffix(".csv")
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(parquet_path, index=False)
    frame.to_csv(csv_path, index=False)
    return parquet_path, csv_path


def validate_final_assignments(
    input_rows: pd.DataFrame, assigned_rows: pd.DataFrame, manifest: pd.DataFrame
) -> None:
    """Enforce lossless, unique assignments and valid final episode dates."""
    if len(input_rows) != len(assigned_rows):
        raise RuntimeError("Rows were lost or added during episode assignment")
    if assigned_rows["clinical_episode_id"].isna().any():
        raise RuntimeError("At least one row_id_raw has no clinical episode")
    if assigned_rows["row_id_raw"].duplicated().any():
        raise RuntimeError("At least one row_id_raw appears in multiple episodes")
    if set(input_rows["row_id_raw"]) != set(assigned_rows["row_id_raw"]):
        raise RuntimeError("Final row_id_raw values differ from the input")
    episode_patients = assigned_rows.groupby("clinical_episode_id")[
        "patient_id"
    ].nunique(dropna=False)
    if episode_patients.gt(1).any():
        raise RuntimeError("A clinical_episode_id belongs to multiple patients")
    if manifest.duplicated(["patient_id", "clinical_episode_id"]).any():
        raise RuntimeError("Duplicate episodes exist in the final manifest")
    dated = manifest.dropna(
        subset=["episode_start_date", "clinical_anchor_date", "episode_end_date"]
    )
    invalid_anchor = ~dated["clinical_anchor_date"].between(
        dated["episode_start_date"], dated["episode_end_date"]
    )
    if invalid_anchor.any():
        raise RuntimeError("A clinical anchor date falls outside its episode")


def main() -> None:
    """Read visits, construct episodes, validate assignments, and write outputs."""
    args = parse_args()
    logger = setup_logger("08c_build_clinical_episode_map")
    logger.info("Reading %s", args.input_path)
    visits = pd.read_parquet(args.input_path)
    prepared, provenance = prepare_visits(visits)
    source_intervals = build_source_interval_qc(prepared)
    flagged_rows = add_presence_flags(prepared)
    daily_units = build_daily_activity_units(flagged_rows, provenance)
    assigned_units = assign_episodes(daily_units)
    (
        assigned_units,
        backward_summary,
        backward_log,
        extended_visit_review,
    ) = backward_reconciliation(assigned_units)
    episodes_before_pass4 = assigned_units["clinical_episode_id"].nunique()
    assigned_units, interval_cluster_audit, interval_cluster_summary = (
        interval_cluster_reconciliation(assigned_units)
    )
    assigned_rows = propagate_episode_assignments(flagged_rows, assigned_units)
    manifest = build_manifest(assigned_rows, source_intervals)
    missed_backward_merges = build_missed_backward_merge_qc(manifest)
    qc = build_qc(
        assigned_rows,
        assigned_units,
        manifest,
        source_intervals,
        missed_backward_merges,
        backward_summary,
    )
    assignment_failures = qc.loc[
        0,
        [
            "raw_rows_unassigned",
            "raw_rows_multiply_assigned",
            "patient_date_units_assigned_to_multiple_episodes",
        ],
    ]
    if assignment_failures.astype(int).any():
        raise RuntimeError("Episode assignment failed one-to-one QC")
    validate_final_assignments(flagged_rows, assigned_rows, manifest)
    if assigned_units["clinical_episode_id"].nunique() > episodes_before_pass4:
        raise RuntimeError("PASS 4 increased the number of clinical episodes")
    row_columns = [
        "patient_id",
        "row_id_raw",
        "interval_name",
        "collection_date",
        "daily_activity_unit_id",
        *provenance,
        "clinical_episode_id",
        "assignment_rule",
        "manual_review_required",
    ]
    assigned_rows = assigned_rows.merge(
        manifest[["clinical_episode_id", "manual_review_required"]],
        on="clinical_episode_id",
        how="left",
        validate="many_to_one",
    )
    args.qc_dir.mkdir(parents=True, exist_ok=True)
    row_map_paths = write_parquet_and_csv(assigned_rows[row_columns], args.row_map_path)
    manifest_paths = write_parquet_and_csv(manifest, args.manifest_path)
    qc.to_csv(args.qc_dir / "08c_qc_summary.csv", index=False)
    backward_summary.to_csv(args.qc_dir / BACKWARD_SUMMARY_FILENAME, index=False)
    backward_log.to_csv(args.qc_dir / BACKWARD_LOG_FILENAME, index=False)
    extended_visit_review.to_csv(
        args.qc_dir / EXTENDED_VISIT_REVIEW_FILENAME, index=False
    )
    extended_visit_review.to_csv(
        args.qc_dir / INTERVAL_EXTENDED_RECONCILIATION_FILENAME, index=False
    )
    interval_cluster_audit.to_csv(
        args.qc_dir / INTERVAL_CLUSTER_RECONCILIATION_FILENAME, index=False
    )
    interval_cluster_summary.to_csv(
        args.qc_dir / INTERVAL_CLUSTER_SUMMARY_FILENAME, index=False
    )
    missed_backward_merges.to_csv(
        args.qc_dir / MISSED_BACKWARD_MERGES_FILENAME, index=False
    )
    source_intervals.to_csv(args.qc_dir / "08c_source_interval_qc.csv", index=False)
    manual_review_reason_distribution(manifest).to_csv(
        args.qc_dir / "08c_manual_review_reason_distribution.csv", index=False
    )
    previous_path = args.previous_episodes_path
    if previous_path == PREVIOUS_EPISODES_PATH and not previous_path.exists():
        previous_path = PREVIOUS_CANDIDATES_PATH
    compare_previous(manifest, previous_path).to_csv(
        args.qc_dir / "08c_vs_08b_classification.csv", index=False
    )
    logger.info("QC summary: %s", qc.iloc[0].to_dict())
    logger.info(
        "Saved row map %s and manifest %s in Parquet and CSV formats",
        row_map_paths,
        manifest_paths,
    )


if __name__ == "__main__":
    main()
