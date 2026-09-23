"""Reconstruct clinical visits with interval-first, auditable merge rules.

The atomic unit in this step is a source row.  Original interval labels and
dates are immutable provenance: apparent date errors are flagged, never fixed.
Step 09c remains responsible for resolving values within an episode.
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
MERGE_AUDIT_FILENAME = "08c_merge_decision_audit.csv"
EXCEPTIONAL_MERGES_FILENAME = "08c_exceptional_merges.csv"
MERGE_SUMMARY_FILENAME = "08c_merge_summary.csv"

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
RESEARCH_PREFIXES = {
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
PRIMARY_CLINICAL_FLAGS = (
    "has_essdai_form",
    "has_essdai_total",
    "has_esspri_form",
    "has_esspri_core",
    "has_systems_review",
    "has_visit_summary",
    "has_eye_exam",
    "has_salivary_flow",
    "has_oral_exam",
)
EVIDENCE_FLAGS = tuple(
    dict.fromkeys(
        (*PRIMARY_CLINICAL_FLAGS, *CORE_FLAGS, *OBJECTIVE_FLAGS, *SUPPORT_FLAGS)
    )
)
ANCHOR_PRIORITY = (
    "has_essdai_total",
    "has_essdai_form",
    "has_systems_review",
    "has_physical_exam",
    "has_visit_summary",
    "has_eye_exam",
    "has_salivary_flow",
    "has_oral_exam",
)
MISSING_UPPER = {str(value).strip().upper() for value in MISSING_TOKENS}

SCHEDULED_VISITS = {
    "phase 1: initial full evaluation": "phase1_initial_full",
    "phase 1: second full evaluation": "phase1_second_full",
    "phase 1: final full (third full) evaluation": "phase1_third_full",
    "phase 2: 4th full evaluation": "phase2_fourth_full",
    "phase 2: 5th full evaluation": "phase2_fifth_full",
}
SCHEDULED_ORDER = {
    value: number for number, value in enumerate(SCHEDULED_VISITS.values(), 1)
}
STATIC_COLUMN_PATTERNS = ("sex", "date_of_birth", "birth_date", "dob")
AUDIT_COLUMNS = [
    "patient_id",
    "episode_or_fragment_a",
    "episode_or_fragment_b",
    "interval_a",
    "interval_b",
    "canonical_family_a",
    "canonical_family_b",
    "date_a",
    "date_b",
    "gap_days",
    "same_calendar_month",
    "same_calendar_year",
    "interval_compatibility_type",
    "a_visit_type",
    "b_visit_type",
    "clinical_components_added",
    "reunited_essdai_esspri",
    "hard_conflict",
    "duplicate_complete_assessment",
    "merge_performed",
    "merge_stage",
    "merge_reason",
    "cross_interval_merge",
    "cross_year_merge",
    "suspected_date_error",
    "interval_order_anomaly",
    "exceptional_merge",
    "manual_review_required",
    "manual_review_reason",
    "final_clinical_episode_id",
    "cluster_span_days",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, default=INPUT_PATH)
    parser.add_argument("--row-map-path", type=Path, default=ROW_MAP_PATH)
    parser.add_argument("--manifest-path", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--qc-dir", type=Path, default=QC_DIR)
    return parser.parse_args()


def resolve_column(
    df: pd.DataFrame, names: Iterable[str], required: bool = True
) -> str | None:
    """Resolve the first exact or uniquely group-prefixed column name."""
    for name in names:
        if name in df.columns:
            return name
        matches = [str(column) for column in df if str(column).endswith(f"__{name}")]
        if len(matches) == 1:
            return matches[0]
    if required:
        raise ValueError(f"Required column not found; tried {list(names)}")
    return None


def has_information(series: pd.Series) -> pd.Series:
    """Return whether values are populated, retaining zero and negative answers."""
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
        for column in df
        if "__" in str(column)
        and str(column).split("__", 1)[0].strip().lower() in accepted
    ]


def _classify_evidence(frame: pd.DataFrame) -> pd.DataFrame:
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
    result["has_any_clinical_evidence"] = result[list(EVIDENCE_FLAGS)].any(axis=1)
    return result


def add_presence_flags(visits: pd.DataFrame) -> pd.DataFrame:
    """Add clinical evidence flags, preserving form-versus-completeness semantics."""
    result = visits.copy()
    for flag, prefixes in {**BLOCK_PREFIXES, **RESEARCH_PREFIXES}.items():
        result[flag] = False
        for column in _columns_for_prefixes(result, prefixes):
            result[flag] |= has_information(result[column])
    result["has_essdai_total"] = False
    for column in ("essdai__essdai_total_score", "essdai-_r__essdai_total_score"):
        if column in result:
            result["has_essdai_total"] |= has_information(result[column])
    result["has_esspri_core"] = False
    for name in ("dryness", "fatigue", "pain"):
        column = f"esspri_questionnaire__{name}"
        if column in result:
            result["has_esspri_core"] |= has_information(result[column])
    return _classify_evidence(result)


def _aggregate_rows(rows: pd.DataFrame) -> pd.Series:
    flags = list(
        dict.fromkeys(
            (*BLOCK_PREFIXES, "has_essdai_total", "has_esspri_core", *RESEARCH_PREFIXES)
        )
    )
    values = {flag: bool(rows[flag].any()) for flag in flags}
    return _classify_evidence(pd.DataFrame([values])).iloc[0]


def _normalized_interval_name(value: object) -> str:
    return "" if pd.isna(value) else re.sub(r"\s+", " ", str(value).strip().casefold())


def interval_identity(value: object) -> tuple[str, str | None]:
    """Return comparison family and scheduled visit without changing the label."""
    normalized = _normalized_interval_name(value)
    if normalized == "natural history protocol 478 interval" or re.fullmatch(
        r"15d optional evaluation [1-5]", normalized
    ):
        return "natural_history_family", None
    if normalized in SCHEDULED_VISITS:
        return "full_evaluation_family", SCHEDULED_VISITS[normalized]
    if re.fullmatch(r"optional evaluation [1-4]", normalized):
        return "full_evaluation_family", None
    return "unrecognized", None


def prepare_visits(visits: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Normalize working identifiers/dates while retaining source provenance."""
    patient_col = resolve_column(visits, ("patient_id", "patient_record_number"))
    interval_col = resolve_column(visits, ("interval_name",))
    date_col = resolve_column(
        visits, ("collection_date", "visit_date", "visit_datetime")
    )
    row_col = resolve_column(visits, ("row_id_raw",), required=False)
    result = visits.copy()
    result["_source_order"] = range(len(result))
    result["patient_id"] = result[patient_col]
    result["interval_name"] = result[interval_col]
    result["collection_date"] = pd.to_datetime(
        result[date_col], errors="coerce"
    ).dt.normalize()
    result["row_id_raw"] = result[row_col] if row_col else result["_source_order"]
    identities = result["interval_name"].map(interval_identity)
    result["canonical_interval_family"] = identities.map(lambda value: value[0])
    result["canonical_scheduled_visit"] = identities.map(lambda value: value[1])
    if result["patient_id"].isna().any():
        raise ValueError(
            "patient_id contains missing values; episode IDs cannot be made safely"
        )
    if result["row_id_raw"].isna().any() or result["row_id_raw"].duplicated().any():
        raise ValueError("row_id_raw must be complete and unique")
    provenance = [
        str(column)
        for column in visits
        if any(term in str(column).lower() for term in ("protocol", "origin", "source"))
    ]
    return result, list(dict.fromkeys(provenance))


def build_atomic_activity_units(
    flagged_rows: pd.DataFrame, provenance_columns: Iterable[str] = ()
) -> pd.DataFrame:
    """Create one immutable activity unit per raw source row."""
    units = flagged_rows.copy()
    units["atomic_activity_unit_id"] = units["row_id_raw"].map(
        lambda value: f"row-{value}"
    )
    units["daily_activity_unit_id"] = units[
        "atomic_activity_unit_id"
    ]  # 09c compatibility
    units["row_ids_involved"] = units["row_id_raw"].map(lambda value: (value,))
    units["interval_names_involved"] = units["interval_name"].map(
        lambda value: () if pd.isna(value) else (str(value),)
    )
    for column in provenance_columns:
        units[f"{column}_involved"] = units[column].map(
            lambda value: () if pd.isna(value) else (str(value),)
        )
    return units


def build_daily_activity_units(
    flagged_rows: pd.DataFrame, provenance_columns: Iterable[str] = ()
) -> pd.DataFrame:
    """Compatibility alias; importantly, this no longer groups rows by date."""
    return build_atomic_activity_units(flagged_rows, provenance_columns)


def _episode_visit_type(rows: pd.DataFrame) -> str:
    evidence = _aggregate_rows(rows)
    if evidence.clinical_candidate:
        return "clinical_candidate"
    if evidence.has_research_component and not bool(
        evidence[list(CORE_FLAGS) + list(OBJECTIVE_FLAGS)].any()
    ):
        return "research_or_procedure_only_candidate"
    return "ambiguous"


def _episode_date(rows: pd.DataFrame) -> pd.Timestamp:
    for flag in ANCHOR_PRIORITY:
        dates = rows.loc[rows[flag], "collection_date"].dropna()
        if not dates.empty:
            return dates.min()
    dates = rows["collection_date"].dropna()
    return dates.min() if not dates.empty else pd.NaT


def _interval_compatibility(left: pd.DataFrame, right: pd.DataFrame) -> str:
    left_names = {
        _normalized_interval_name(value) for value in left["interval_name"].dropna()
    }
    right_names = {
        _normalized_interval_name(value) for value in right["interval_name"].dropna()
    }
    if left_names & right_names:
        return "exact_same_interval"
    left_families = set(left["canonical_interval_family"].dropna())
    right_families = set(right["canonical_interval_family"].dropna())
    if left_families == right_families == {"natural_history_family"}:
        return "natural_history_family_alias"
    if left_families == right_families == {"full_evaluation_family"}:
        left_optional = left["canonical_scheduled_visit"].isna().all()
        right_optional = right["canonical_scheduled_visit"].isna().all()
        if left_optional != right_optional:
            return "optional_to_full_evaluation_family"
    return "different_interval"


def _static_conflicts(left: pd.DataFrame, right: pd.DataFrame) -> list[str]:
    conflicts: list[str] = []
    for column in left.columns.intersection(right.columns):
        normalized = str(column).lower().replace(" ", "_")
        if not any(pattern in normalized for pattern in STATIC_COLUMN_PATTERNS):
            continue
        left_values = set(
            left.loc[has_information(left[column]), column]
            .astype(str)
            .str.strip()
            .str.casefold()
        )
        right_values = set(
            right.loc[has_information(right[column]), column]
            .astype(str)
            .str.strip()
            .str.casefold()
        )
        if left_values and right_values and left_values.isdisjoint(right_values):
            conflicts.append(str(column))
    return conflicts


def assess_episode_compatibility(
    left: pd.DataFrame, right: pd.DataFrame
) -> dict[str, object]:
    """Screen structural conflicts; dynamic clinical-value differences never block."""
    left_evidence, right_evidence = _aggregate_rows(left), _aggregate_rows(right)
    conflicts = _static_conflicts(left, right)
    duplicate_complete = bool(
        left_evidence.clinical_candidate and right_evidence.clinical_candidate
    )
    ignored = [
        column for column in left.columns if "vital_signs__" in str(column).lower()
    ]
    hard_conflict = bool(conflicts)
    return {
        "compatible": not hard_conflict and not duplicate_complete,
        "hard_conflict": hard_conflict,
        "duplicate_complete_assessment": duplicate_complete,
        "conflicting_components": conflicts,
        "ignored_nonblocking_differences": ignored,
        "reason": (
            "static_data_conflict"
            if hard_conflict
            else (
                "multiple_complete_clinical_episodes"
                if duplicate_complete
                else "compatible"
            )
        ),
    }


def _components_added(left: pd.DataFrame, right: pd.DataFrame) -> list[str]:
    existing, incoming = _aggregate_rows(left), _aggregate_rows(right)
    return [
        flag
        for flag in EVIDENCE_FLAGS
        if bool(incoming[flag]) and not bool(existing[flag])
    ]


def _order_anomaly(
    fragment: pd.DataFrame, target: pd.DataFrame, patient_rows: pd.DataFrame
) -> bool:
    scheduled = fragment["canonical_scheduled_visit"].dropna()
    fragment_date = _episode_date(fragment)
    if scheduled.empty or pd.isna(fragment_date):
        return False
    rank = SCHEDULED_ORDER.get(str(scheduled.iloc[0]))
    for visit, rows in patient_rows.dropna(
        subset=["canonical_scheduled_visit"]
    ).groupby("canonical_scheduled_visit"):
        other_rank = SCHEDULED_ORDER.get(str(visit))
        other_date = _episode_date(rows)
        if (
            other_rank
            and pd.notna(other_date)
            and (
                (other_rank > rank and other_date < fragment_date)
                or (other_rank < rank and other_date > fragment_date)
            )
        ):
            return True
    return False


def _decision(
    left: pd.DataFrame, right: pd.DataFrame, stage: str, patient_rows: pd.DataFrame
) -> tuple[bool, dict[str, object]]:
    date_a, date_b = _episode_date(left), _episode_date(right)
    gap = (
        abs(int((date_b - date_a).days))
        if pd.notna(date_a) and pd.notna(date_b)
        else None
    )
    same_month = bool(
        pd.notna(date_a)
        and pd.notna(date_b)
        and date_a.to_period("M") == date_b.to_period("M")
    )
    same_year = bool(
        pd.notna(date_a) and pd.notna(date_b) and date_a.year == date_b.year
    )
    compatibility_type = _interval_compatibility(left, right)
    compatibility = assess_episode_compatibility(left, right)
    added_lr, added_rl = _components_added(left, right), _components_added(right, left)
    added = added_lr or added_rl
    complementary = bool(added)
    eligible_interval = compatibility_type != "different_interval"
    if stage == "interval_same_year":
        temporal = same_year
        permitted = eligible_interval and temporal
    elif stage == "interval_outlier":
        temporal = not same_year and pd.notna(date_a) and pd.notna(date_b)
        # Exceptional recovery needs one and only one established clinical anchor.
        one_candidate = (_episode_visit_type(left) == "clinical_candidate") != (
            _episode_visit_type(right) == "clinical_candidate"
        )
        permitted = eligible_interval and temporal and one_candidate
    else:
        temporal = bool(gap is not None and gap <= 30) or same_month
        permitted = temporal
    merge = bool(permitted and complementary and compatibility["compatible"])
    cross_interval = not bool(
        {_normalized_interval_name(value) for value in left["interval_name"].dropna()}
        & {
            _normalized_interval_name(value)
            for value in right["interval_name"].dropna()
        }
    )
    cross_year = not same_year if pd.notna(date_a) and pd.notna(date_b) else False
    order_anomaly = bool(
        stage == "interval_outlier" and _order_anomaly(right, left, patient_rows)
    )
    suspected = bool(merge and stage == "interval_outlier" and cross_year)
    span_dates = pd.concat([left["collection_date"], right["collection_date"]]).dropna()
    span = (
        int((span_dates.max() - span_dates.min()).days)
        if not span_dates.empty
        else None
    )
    exceptional = bool(
        merge
        and (
            cross_year
            or cross_interval
            or (span is not None and span > 30)
            or compatibility_type == "optional_to_full_evaluation_family"
            or stage == "temporal_rescue"
            or suspected
            or order_anomaly
        )
    )
    if merge:
        if stage == "temporal_rescue":
            reason = "temporal_rescue_complementary"
        elif stage == "interval_outlier":
            reason = "same_interval_outlier_complement"
        elif compatibility_type == "natural_history_family_alias":
            reason = "natural_history_alias_merge"
        elif compatibility_type == "optional_to_full_evaluation_family":
            reason = "optional_full_family_merge"
        else:
            reason = "interval_exact_same_year"
    elif compatibility["hard_conflict"]:
        reason = "rejected_hard_conflict"
    elif compatibility["duplicate_complete_assessment"]:
        reason = "rejected_multiple_complete_clinical"
    elif not complementary:
        reason = "rejected_no_complementarity"
    else:
        reason = "not_eligible_for_stage"
    review_reason = ""
    if merge and cross_year:
        review_reason = "cross_year_same_interval_merge"
    elif (
        not merge
        and compatibility["duplicate_complete_assessment"]
        and eligible_interval
    ):
        review_reason = "multiple_complete_clinical_episodes_same_interval"
    elif not merge and compatibility["hard_conflict"]:
        review_reason = "hard_structural_conflict"
    record = {
        "patient_id": left["patient_id"].iloc[0],
        "episode_or_fragment_a": "|".join(map(str, left["row_id_raw"])),
        "episode_or_fragment_b": "|".join(map(str, right["row_id_raw"])),
        "interval_a": " | ".join(
            dict.fromkeys(left["interval_name"].dropna().astype(str))
        ),
        "interval_b": " | ".join(
            dict.fromkeys(right["interval_name"].dropna().astype(str))
        ),
        "canonical_family_a": "|".join(sorted(set(left["canonical_interval_family"]))),
        "canonical_family_b": "|".join(sorted(set(right["canonical_interval_family"]))),
        "date_a": date_a,
        "date_b": date_b,
        "gap_days": gap,
        "same_calendar_month": same_month,
        "same_calendar_year": same_year,
        "interval_compatibility_type": compatibility_type,
        "a_visit_type": _episode_visit_type(left),
        "b_visit_type": _episode_visit_type(right),
        "clinical_components_added": "|".join(added),
        "reunited_essdai_esspri": bool(
            _aggregate_rows(pd.concat([left, right])).has_essdai_form
            and _aggregate_rows(pd.concat([left, right])).has_esspri_form
            and not (
                _aggregate_rows(left).has_essdai_form
                and _aggregate_rows(left).has_esspri_form
            )
            and not (
                _aggregate_rows(right).has_essdai_form
                and _aggregate_rows(right).has_esspri_form
            )
        ),
        "hard_conflict": compatibility["hard_conflict"],
        "duplicate_complete_assessment": compatibility["duplicate_complete_assessment"],
        "merge_performed": merge,
        "merge_stage": stage,
        "merge_reason": reason,
        "cross_interval_merge": bool(merge and cross_interval),
        "cross_year_merge": bool(merge and cross_year),
        "suspected_date_error": suspected,
        "interval_order_anomaly": order_anomaly,
        "exceptional_merge": exceptional,
        "manual_review_required": bool(review_reason),
        "manual_review_reason": review_reason,
        "final_clinical_episode_id": None,
        "cluster_span_days": span,
    }
    return merge, record


def _run_stage(
    groups: list[pd.DataFrame],
    stage: str,
    patient_rows: pd.DataFrame,
    audit: list[dict[str, object]],
) -> list[pd.DataFrame]:
    changed = True
    while changed:
        changed = False
        for left_index in range(len(groups)):
            for right_index in range(left_index + 1, len(groups)):
                merge, record = _decision(
                    groups[left_index], groups[right_index], stage, patient_rows
                )
                audit.append(record)
                if record["manual_review_required"] and not merge:
                    for group_index in (left_index, right_index):
                        groups[group_index]["manual_review_required"] = True
                        groups[group_index]["manual_review_reason"] = groups[
                            group_index
                        ]["manual_review_reason"].map(
                            lambda value, reason=str(record["manual_review_reason"]): (
                                _append_pipe_value(value, reason)
                            )
                        )
                if merge:
                    merged = pd.concat(
                        [groups[left_index], groups[right_index]]
                    ).sort_values("_source_order")
                    rule = {
                        "interval_same_year": (
                            "interval_family_same_year"
                            if record["interval_compatibility_type"]
                            != "exact_same_interval"
                            else "interval_exact_same_year"
                        ),
                        "interval_outlier": "same_interval_cross_year_complement",
                        "temporal_rescue": (
                            "temporal_rescue_same_calendar_month"
                            if record["same_calendar_month"]
                            else "temporal_rescue_within_30_days"
                        ),
                    }[stage]
                    merged["assignment_rule"] = merged["assignment_rule"].map(
                        lambda value: _append_pipe_value(value, rule)
                    )
                    for column in (
                        "merge_stage",
                        "merge_reason",
                        "interval_compatibility_type",
                        "clinical_components_added",
                        "manual_review_reason",
                    ):
                        merged[column] = merged[column].map(
                            lambda value, item=str(record[column]): _append_pipe_value(
                                value, item
                            )
                        )
                    for column in (
                        "cross_interval_merge",
                        "cross_year_merge",
                        "suspected_date_error",
                        "interval_order_anomaly",
                        "exceptional_merge",
                        "manual_review_required",
                    ):
                        merged[column] = merged[column].astype(bool) | bool(
                            record[column]
                        )
                    groups[left_index] = merged
                    del groups[right_index]
                    changed = True
                    break
            if changed:
                break
    return groups


def _append_pipe_value(existing: object, value: str) -> str:
    values = (
        [] if pd.isna(existing) or str(existing) == "" else str(existing).split("|")
    )
    if value and value not in values:
        values.append(value)
    return "|".join(values)


def assign_episodes(atomic_units: pd.DataFrame) -> pd.DataFrame:
    """Run PASS 1 interval-first reconstruction and PASS 2 temporal rescue."""
    assigned: list[pd.DataFrame] = []
    audit: list[dict[str, object]] = []
    for patient_id, patient_rows in atomic_units.groupby(
        "patient_id", sort=False, dropna=False
    ):
        groups: list[pd.DataFrame] = []
        for _, row in patient_rows.sort_values("_source_order").iterrows():
            group = row.to_frame().T.infer_objects()
            group["assignment_rule"] = "standalone_record"
            group["merge_stage"] = "standalone"
            group["merge_reason"] = "standalone_record"
            group["interval_compatibility_type"] = ""
            group["clinical_components_added"] = ""
            group["cross_interval_merge"] = False
            group["cross_year_merge"] = False
            group["suspected_date_error"] = False
            group["interval_order_anomaly"] = False
            group["exceptional_merge"] = False
            group["manual_review_required"] = False
            group["manual_review_reason"] = ""
            groups.append(group)
        groups = _run_stage(groups, "interval_same_year", patient_rows, audit)
        groups = _run_stage(groups, "interval_outlier", patient_rows, audit)
        groups = _run_stage(groups, "temporal_rescue", patient_rows, audit)
        groups.sort(
            key=lambda rows: (_episode_date(rows), int(rows["_source_order"].min()))
        )
        for sequence, rows in enumerate(groups, 1):
            rows["clinical_episode_id"] = f"{patient_id}__CE{sequence:04d}"
            assigned.append(rows)
    result = (
        pd.concat(assigned).sort_values("_source_order")
        if assigned
        else atomic_units.copy()
    )
    decision_audit = pd.DataFrame(audit).reindex(columns=AUDIT_COLUMNS)
    if not decision_audit.empty:
        episode_lookup = result.set_index("row_id_raw")["clinical_episode_id"].to_dict()
        for index, record in decision_audit.iterrows():
            if bool(record["merge_performed"]):
                first_id = str(record["episode_or_fragment_a"]).split("|")[0]
                for raw_id, episode_id in episode_lookup.items():
                    if str(raw_id) == first_id:
                        decision_audit.at[index, "final_clinical_episode_id"] = (
                            episode_id
                        )
                        break
    result.attrs["merge_decision_audit"] = decision_audit
    return result


def propagate_episode_assignments(
    flagged_rows: pd.DataFrame, assigned_units: pd.DataFrame
) -> pd.DataFrame:
    """Propagate unit assignments to source rows without changing provenance."""
    qc_columns = [
        "row_id_raw",
        "clinical_episode_id",
        "assignment_rule",
        "merge_stage",
        "merge_reason",
        "interval_compatibility_type",
        "clinical_components_added",
        "cross_interval_merge",
        "cross_year_merge",
        "suspected_date_error",
        "interval_order_anomaly",
        "exceptional_merge",
        "manual_review_required",
        "manual_review_reason",
    ]
    result = flagged_rows.merge(
        assigned_units[qc_columns], on="row_id_raw", how="left", validate="one_to_one"
    )
    result.attrs.update(assigned_units.attrs)
    return result.sort_values("_source_order")


def _anchor_date(rows: pd.DataFrame) -> pd.Timestamp:
    """Select the clinical-core date; exceptional fragments cannot move the anchor."""
    reliable = rows.loc[~rows["suspected_date_error"].fillna(False)]
    return _episode_date(reliable if not reliable.empty else rows)


def build_source_interval_qc(prepared_rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize immutable source patient-interval date ranges."""
    result = prepared_rows.groupby(
        ["patient_id", "interval_name"], as_index=False, dropna=False
    ).agg(
        source_interval_start_date=("collection_date", "min"),
        source_interval_end_date=("collection_date", "max"),
    )
    result["source_interval_span_days"] = (
        result["source_interval_end_date"] - result["source_interval_start_date"]
    ).dt.days.astype("Int64")
    result["source_interval_span_gt30"] = (
        result["source_interval_span_days"].gt(30).fillna(False)
    )
    return result


def build_manifest(
    assigned: pd.DataFrame, source_intervals: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Build the 09c-compatible manifest plus interval-first QC fields."""
    records: list[dict[str, object]] = []
    output_flags = list(
        dict.fromkeys(
            (
                *PRIMARY_CLINICAL_FLAGS,
                "has_physical_exam",
                "has_vital_signs",
                "has_biopsy_pathology",
            )
        )
    )
    for (patient_id, episode_id), rows in assigned.groupby(
        ["patient_id", "clinical_episode_id"], sort=False
    ):
        evidence = _aggregate_rows(rows)
        dates = rows["collection_date"].dropna()
        start, end = (dates.min(), dates.max()) if not dates.empty else (pd.NaT, pd.NaT)
        reasons = [
            part
            for value in rows["manual_review_reason"].dropna()
            for part in str(value).split("|")
            if part
        ]
        if rows["collection_date"].isna().any():
            reasons.append("missing_collection_date")
        record: dict[str, object] = {
            "patient_id": patient_id,
            "clinical_episode_id": episode_id,
            "intervals_involved": " | ".join(
                dict.fromkeys(rows["interval_name"].dropna().astype(str))
            ),
            "episode_start_date": start,
            "clinical_anchor_date": _anchor_date(rows),
            "episode_end_date": end,
            "episode_span_days": int((end - start).days) if pd.notna(start) else pd.NA,
            "visit_type": _episode_visit_type(rows),
            "clinical_visit": bool(evidence.clinical_candidate),
            "manual_review_required": bool(reasons),
            "manual_review_reason": "|".join(dict.fromkeys(reasons)),
            "canonical_interval_family": "|".join(
                dict.fromkeys(rows["canonical_interval_family"].astype(str))
            ),
            "canonical_scheduled_visit": "|".join(
                dict.fromkeys(rows["canonical_scheduled_visit"].dropna().astype(str))
            ),
            "assignment_rule": "|".join(
                dict.fromkeys(
                    part
                    for value in rows["assignment_rule"]
                    for part in str(value).split("|")
                )
            ),
            "merge_stage": "|".join(
                dict.fromkeys(
                    part
                    for value in rows["merge_stage"]
                    for part in str(value).split("|")
                )
            ),
            "cross_interval_merge": bool(rows["cross_interval_merge"].any()),
            "cross_year_merge": bool(rows["cross_year_merge"].any()),
            "suspected_date_error": bool(rows["suspected_date_error"].any()),
            "interval_order_anomaly": bool(rows["interval_order_anomaly"].any()),
            "exceptional_merge": bool(rows["exceptional_merge"].any()),
            "physician_core_count": int(evidence.physician_core_count),
            "objective_exam_count": int(evidence.objective_exam_count),
        }
        record.update({flag: bool(evidence[flag]) for flag in output_flags})
        records.append(record)
    return pd.DataFrame(records)


def validate_final_assignments(
    source: pd.DataFrame, assigned: pd.DataFrame
) -> tuple[int, int]:
    """Assert conservation, unique assignment, and immutable source provenance."""
    counts = assigned["row_id_raw"].value_counts()
    unassigned = len(set(source["row_id_raw"]) - set(assigned["row_id_raw"]))
    multiplied = int((counts > 1).sum())
    assert len(source) == len(assigned), "Raw row conservation failed"
    assert unassigned == 0, "A raw row was not assigned"
    assert multiplied == 0, "A raw row was assigned more than once"
    columns = ["row_id_raw", "patient_id", "collection_date", "interval_name"]
    before = source[columns].sort_values("row_id_raw").reset_index(drop=True)
    after = assigned[columns].sort_values("row_id_raw").reset_index(drop=True)
    assert before.equals(after), "Patient/date/interval provenance changed"
    assert assigned["clinical_episode_id"].notna().all(), "Missing episode assignment"
    return unassigned, multiplied


def build_merge_summary(
    assigned: pd.DataFrame, manifest: pd.DataFrame, audit: pd.DataFrame
) -> pd.DataFrame:
    """Build required one-row merge and conservation metrics."""
    merged = audit.loc[audit["merge_performed"].fillna(False)]
    rejected = audit.loc[~audit["merge_performed"].fillna(False)]
    metric = lambda condition: int(condition.sum())
    return pd.DataFrame(
        [
            {
                "n_raw_rows": len(assigned),
                "n_patients": assigned["patient_id"].nunique(),
                "n_episodes_final": manifest["clinical_episode_id"].nunique(),
                "n_merges_total": len(merged),
                "n_pass1_exact_interval_merges": metric(
                    merged["interval_compatibility_type"].eq("exact_same_interval")
                    & merged["merge_stage"].str.startswith("interval")
                ),
                "n_pass1_family_alias_merges": metric(
                    merged["interval_compatibility_type"].isin(
                        [
                            "natural_history_family_alias",
                            "optional_to_full_evaluation_family",
                        ]
                    )
                    & merged["merge_stage"].str.startswith("interval")
                ),
                "n_pass1_same_year_merges": metric(
                    merged["merge_stage"].eq("interval_same_year")
                ),
                "n_pass1_span_gt30_merges": metric(merged["cluster_span_days"].gt(30)),
                "n_pass1_span_gt90_merges": metric(merged["cluster_span_days"].gt(90)),
                "n_pass1_cross_year_merges": metric(
                    merged["merge_stage"].eq("interval_outlier")
                ),
                "n_natural_history_15d_alias_merges": metric(
                    merged["interval_compatibility_type"].eq(
                        "natural_history_family_alias"
                    )
                ),
                "n_optional_full_family_merges": metric(
                    merged["interval_compatibility_type"].eq(
                        "optional_to_full_evaluation_family"
                    )
                ),
                "n_pass2_temporal_rescue_merges": metric(
                    merged["merge_stage"].eq("temporal_rescue")
                ),
                "n_pass2_cross_interval_merges": metric(
                    merged["merge_stage"].eq("temporal_rescue")
                    & merged["cross_interval_merge"]
                ),
                "n_pass2_same_month_merges": metric(
                    merged["merge_stage"].eq("temporal_rescue")
                    & merged["same_calendar_month"]
                ),
                "n_pass2_within_30_days_merges": metric(
                    merged["merge_stage"].eq("temporal_rescue")
                    & merged["gap_days"].le(30)
                ),
                "n_suspected_date_error_merges": metric(merged["suspected_date_error"]),
                "n_interval_order_anomaly_merges": metric(
                    merged["interval_order_anomaly"]
                ),
                "n_rejected_multiple_complete_clinical": metric(
                    rejected["duplicate_complete_assessment"]
                ),
                "n_rejected_hard_conflict": metric(rejected["hard_conflict"]),
                "n_rejected_no_complementarity": metric(
                    rejected["merge_reason"].eq("rejected_no_complementarity")
                ),
                "n_exceptional_merges": metric(merged["exceptional_merge"]),
                "n_manual_review_required": metric(assigned["manual_review_required"]),
                "raw_rows_unassigned": 0,
                "raw_rows_multiply_assigned": 0,
            }
        ]
    )


def build_qc(
    assigned_rows: pd.DataFrame,
    assigned_units: pd.DataFrame,
    manifest: pd.DataFrame,
    source_intervals: pd.DataFrame,
    **_: object,
) -> pd.DataFrame:
    """Compatibility wrapper returning the new merge summary."""
    audit = assigned_units.attrs.get(
        "merge_decision_audit",
        assigned_rows.attrs.get(
            "merge_decision_audit", pd.DataFrame(columns=AUDIT_COLUMNS)
        ),
    )
    result = build_merge_summary(assigned_rows, manifest, audit)
    result["raw_rows"] = result["n_raw_rows"]
    result["patients"] = result["n_patients"]
    result["final_clinical_episode_ids"] = result["n_episodes_final"]
    result["daily_activity_units"] = len(assigned_units)
    result["unique_patient_collection_dates"] = len(
        assigned_rows[["patient_id", "collection_date"]].drop_duplicates()
    )
    result["patient_date_units_assigned_to_multiple_episodes"] = (
        assigned_rows.groupby(["patient_id", "collection_date"], dropna=False)[
            "clinical_episode_id"
        ]
        .nunique()
        .gt(1)
        .sum()
    )
    return result


def write_parquet_and_csv(frame: pd.DataFrame, parquet_path: Path) -> tuple[Path, Path]:
    """Write a dataframe in machine-readable and inspectable forms."""
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = parquet_path.with_suffix(".csv")
    frame.to_parquet(parquet_path, index=False)
    frame.to_csv(csv_path, index=False)
    return parquet_path, csv_path


def main() -> None:
    """Build the row map, manifest, merge audit, exceptional report, and summary."""
    args = parse_args()
    logger = setup_logger("08c_build_clinical_episode_map")
    logger.info("Reading %s", args.input_path)
    source, provenance = prepare_visits(pd.read_parquet(args.input_path))
    flagged = add_presence_flags(source)
    units = build_atomic_activity_units(flagged, provenance)
    assigned_units = assign_episodes(units)
    audit = assigned_units.attrs["merge_decision_audit"]
    assigned = propagate_episode_assignments(flagged, assigned_units)
    validate_final_assignments(source, assigned)
    manifest = build_manifest(assigned)
    summary = build_merge_summary(assigned, manifest, audit)
    write_parquet_and_csv(assigned, args.row_map_path)
    write_parquet_and_csv(manifest, args.manifest_path)
    args.qc_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.qc_dir / MERGE_AUDIT_FILENAME, index=False)
    audit.loc[audit["exceptional_merge"].fillna(False)].to_csv(
        args.qc_dir / EXCEPTIONAL_MERGES_FILENAME, index=False
    )
    summary.to_csv(args.qc_dir / MERGE_SUMMARY_FILENAME, index=False)
    logger.info(
        "Completed: %d raw rows -> %d clinical episodes", len(assigned), len(manifest)
    )


if __name__ == "__main__":
    main()
