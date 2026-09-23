"""Reconstruct clinical visits with interval-first, auditable merge rules.

The atomic unit in this step is a source row.  Original interval labels and
dates are immutable provenance: apparent date errors are flagged, never fixed.
Step 09c remains responsible for resolving values within an episode.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from time import perf_counter
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
    "candidate_generation_rule",
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
    "components_added",
    "target_rank",
    "reunited_essdai_esspri",
    "hard_conflict",
    "independent_complete_conflict",
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

QC_DEFAULTS: dict[str, object] = {
    "assignment_rule": "standalone_record",
    "merge_stage": "standalone",
    "merge_reason": "standalone_record",
    "interval_compatibility_type": "",
    "clinical_components_added": "",
    "cross_interval_merge": False,
    "cross_year_merge": False,
    "suspected_date_error": False,
    "interval_order_anomaly": False,
    "exceptional_merge": False,
    "manual_review_required": False,
    "manual_review_reason": "",
}


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


def is_independent_complete_assessment(evidence: pd.Series) -> bool:
    """Return whether an episode has a self-contained clinical assessment.

    This intentionally is not a count-of-nine checklist.  Independence requires
    a strong disease-activity core plus corroborating encounter structure.
    """
    structural = sum(
        bool(evidence[flag])
        for flag in ("has_systems_review", "has_physical_exam", "has_visit_summary")
    )
    physician_and_objective = int(evidence["physician_core_count"]) + int(
        evidence["objective_exam_count"]
    )
    total_path = bool(evidence["has_essdai_total"]) and structural >= 2
    multi_instrument_path = bool(
        evidence["has_essdai_form"]
        and evidence["has_esspri_core"]
        and physician_and_objective >= 4
    )
    return total_path or (
        bool(evidence["has_essdai_form"])
        and structural >= 2
        and physician_and_objective >= 3
    ) or multi_instrument_path


class EpisodeSummary:
    """Cached episode rows and evidence, refreshed only after a merge."""

    def __init__(
        self,
        episode_id: int,
        rows: pd.DataFrame,
        evidence: pd.Series,
        anchor_date: pd.Timestamp,
        interval_names: frozenset[str],
        families: frozenset[str],
        scheduled_visits: frozenset[str],
        independent_complete_assessment: bool,
    ) -> None:
        self.episode_id = episode_id
        self.rows = rows
        self.evidence = evidence
        self.anchor_date = anchor_date
        self.interval_names = interval_names
        self.families = families
        self.scheduled_visits = scheduled_visits
        self.independent_complete_assessment = independent_complete_assessment


def _episode_visit_type_from_evidence(evidence: pd.Series) -> str:
    if bool(evidence.clinical_candidate):
        return "clinical_candidate"
    if bool(evidence.has_research_component) and not bool(
        evidence[list(CORE_FLAGS) + list(OBJECTIVE_FLAGS)].any()
    ):
        return "research_or_procedure_only_candidate"
    return "ambiguous"


def _episode_visit_type(rows: pd.DataFrame) -> str:
    return _episode_visit_type_from_evidence(_aggregate_rows(rows))


def _episode_date(rows: pd.DataFrame) -> pd.Timestamp:
    for flag in ANCHOR_PRIORITY:
        dates = rows.loc[rows[flag], "collection_date"].dropna()
        if not dates.empty:
            return dates.min()
    dates = rows["collection_date"].dropna()
    return dates.min() if not dates.empty else pd.NaT


def _summarize_episode(episode_id: int, rows: pd.DataFrame) -> EpisodeSummary:
    evidence = _aggregate_rows(rows)
    return EpisodeSummary(
        episode_id=episode_id,
        rows=rows,
        evidence=evidence,
        anchor_date=_episode_date(rows),
        interval_names=frozenset(
            _normalized_interval_name(value) for value in rows["interval_name"].dropna()
        ),
        families=frozenset(rows["canonical_interval_family"].dropna().astype(str)),
        scheduled_visits=frozenset(
            rows["canonical_scheduled_visit"].dropna().astype(str)
        ),
        independent_complete_assessment=is_independent_complete_assessment(evidence),
    )


def _interval_compatibility_summary(left: EpisodeSummary, right: EpisodeSummary) -> str:
    if left.scheduled_visits & right.scheduled_visits:
        return "exact_scheduled_interval"
    if left.interval_names & right.interval_names:
        return "exact_same_interval"
    if left.families == right.families == {"natural_history_family"}:
        return "natural_history_family_alias"
    if left.families == right.families == {"full_evaluation_family"}:
        if bool(left.scheduled_visits) != bool(right.scheduled_visits):
            return "optional_to_full_evaluation_family"
    return "different_interval"


def _interval_compatibility(left: pd.DataFrame, right: pd.DataFrame) -> str:
    return _interval_compatibility_summary(
        _summarize_episode(0, left), _summarize_episode(1, right)
    )


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


def _compatibility(left: EpisodeSummary, right: EpisodeSummary) -> dict[str, object]:
    conflicts = _static_conflicts(left.rows, right.rows)
    duplicate_complete = bool(
        left.independent_complete_assessment
        and right.independent_complete_assessment
    )
    return {
        "compatible": not conflicts and not duplicate_complete,
        "hard_conflict": bool(conflicts),
        "duplicate_complete_assessment": duplicate_complete,
        "conflicting_components": conflicts,
    }


def assess_episode_compatibility(
    left: pd.DataFrame, right: pd.DataFrame
) -> dict[str, object]:
    """Screen structural and independent-assessment conflicts."""
    result = _compatibility(
        _summarize_episode(0, left), _summarize_episode(1, right)
    )
    ignored = [
        column for column in left.columns if "vital_signs__" in str(column).lower()
    ]
    result["ignored_nonblocking_differences"] = ignored
    result["reason"] = (
        "static_data_conflict"
        if result["hard_conflict"]
        else (
            "multiple_complete_clinical_episodes"
            if result["duplicate_complete_assessment"]
            else "compatible"
        )
    )
    return result


def _components_added_summary(
    target: EpisodeSummary, fragment: EpisodeSummary
) -> list[str]:
    return [
        flag
        for flag in EVIDENCE_FLAGS
        if bool(fragment.evidence[flag]) and not bool(target.evidence[flag])
    ]


def _clinical_strength(episode: EpisodeSummary) -> tuple[int, int, int, int]:
    """Return an interpretable tuple used to identify main versus fragment."""
    return (
        int(episode.independent_complete_assessment),
        int(bool(episode.evidence.clinical_candidate)),
        sum(bool(episode.evidence[flag]) for flag in EVIDENCE_FLAGS),
        int(bool(episode.scheduled_visits)),
    )


def _stable_anchor_key(episode: EpisodeSummary) -> tuple[object, ...]:
    """Rank anchors clinically, using immutable row ids only as the last tie-break."""
    return (
        *_clinical_strength(episode),
        tuple(sorted(map(str, episode.rows["row_id_raw"]))),
    )


def _main_and_fragment(
    left: EpisodeSummary, right: EpisodeSummary
) -> tuple[EpisodeSummary, EpisodeSummary]:
    # Raw ids are immutable and therefore provide a stable *technical* tie-breaker;
    # input order / _source_order must never influence clinical assignment.
    left_key = (_clinical_strength(left), tuple(map(str, left.rows["row_id_raw"])))
    right_key = (_clinical_strength(right), tuple(map(str, right.rows["row_id_raw"])))
    return (left, right) if left_key >= right_key else (right, left)


def _candidate_rule(
    left: EpisodeSummary, right: EpisodeSummary, stage: str
) -> tuple[str, str] | None:
    compatibility = _interval_compatibility_summary(left, right)
    date_a, date_b = left.anchor_date, right.anchor_date
    same_year = bool(
        pd.notna(date_a) and pd.notna(date_b) and date_a.year == date_b.year
    )
    if stage == "interval_same_year" and same_year:
        rules = {
            "exact_scheduled_interval": "same_scheduled_interval_same_year",
            "exact_same_interval": "same_exact_interval_same_year",
            "natural_history_family_alias": "natural_history_family_same_year",
            "optional_to_full_evaluation_family": "optional_full_family_same_year",
        }
        rule = rules.get(compatibility)
        return (rule, compatibility) if rule else None
    if (
        stage == "interval_outlier"
        and not same_year
        and pd.notna(date_a)
        and pd.notna(date_b)
    ):
        if compatibility in {"exact_scheduled_interval", "exact_same_interval"}:
            return "same_interval_cross_year", compatibility
        if compatibility == "natural_history_family_alias":
            return "same_explicit_family_cross_year", compatibility
        return None
    if stage == "temporal_rescue" and pd.notna(date_a) and pd.notna(date_b):
        gap = abs(int((date_b - date_a).days))
        same_month = date_a.to_period("M") == date_b.to_period("M")
        if same_month:
            return "same_calendar_month", compatibility
        if gap <= 30:
            return "temporal_window_30d", compatibility
    return None


def _generate_candidates(
    episodes: dict[int, EpisodeSummary], stage: str
) -> list[tuple[int, int, str, str]]:
    """Generate only interval-keyed or time-window candidate pairs."""
    values = list(episodes.values())
    pairs: set[tuple[int, int]] = set()
    if stage.startswith("interval"):
        buckets: dict[tuple[object, ...], list[int]] = {}
        for episode in values:
            year = (
                int(episode.anchor_date.year)
                if pd.notna(episode.anchor_date)
                else None
            )
            bucket_year = year if stage == "interval_same_year" else None
            keys = [("name", value, bucket_year) for value in episode.interval_names]
            keys += [
                ("scheduled", value, bucket_year)
                for value in episode.scheduled_visits
            ]
            keys += [
                ("family", value, bucket_year)
                for value in episode.families
                if value == "natural_history_family"
                or (value == "full_evaluation_family" and stage == "interval_same_year")
            ]
            for key in keys:
                buckets.setdefault(key, []).append(episode.episode_id)
        for identifiers in buckets.values():
            members = [episodes[value] for value in sorted(set(identifiers))]
            complete = [
                episode
                for episode in members
                if episode.independent_complete_assessment
            ]
            # Adjacent complete assessments are QC-relevant, but are direct
            # KEEP-SEPARATE decisions and never enter static conflict analysis.
            ordered_complete = sorted(
                complete,
                key=lambda episode: (episode.anchor_date, _stable_anchor_key(episode)),
            )
            pairs.update(
                tuple(sorted((left.episode_id, right.episode_id)))
                for left, right in zip(ordered_complete, ordered_complete[1:])
            )
            anchors = complete or (
                [max(members, key=_stable_anchor_key)] if members else []
            )
            fragments = [episode for episode in members if episode not in anchors]
            for fragment in fragments:
                for anchor in anchors:
                    # PASS 1B is deliberately anchor + fragment only.
                    if stage == "interval_outlier" and (
                        not bool(anchor.evidence.clinical_candidate)
                        or bool(fragment.evidence.clinical_candidate)
                    ):
                        continue
                    pairs.add(
                        tuple(sorted((anchor.episode_id, fragment.episode_id)))
                    )
    else:
        dated = sorted(
            (episode.anchor_date, episode.episode_id)
            for episode in values
            if pd.notna(episode.anchor_date)
        )
        start = 0
        for end, (date, episode_id) in enumerate(dated):
            while (date - dated[start][0]).days > 30:
                start += 1
            pairs.update(
                (min(episode_id, other_id), max(episode_id, other_id))
                for _, other_id in dated[start:end]
            )
    candidates = []
    for left_id, right_id in sorted(pairs):
        generated = _candidate_rule(episodes[left_id], episodes[right_id], stage)
        if generated:
            rule, compatibility = generated
            candidates.append((left_id, right_id, rule, compatibility))
    return candidates


def _interval_priority(compatibility: str, stage: str) -> int:
    if stage == "temporal_rescue" and compatibility == "different_interval":
        return 4
    return {
        "exact_scheduled_interval": 0,
        "exact_same_interval": 1,
        "natural_history_family_alias": 2,
        "optional_to_full_evaluation_family": 3,
        "different_interval": 4,
    }[compatibility]


def _evaluate_candidate(
    left: EpisodeSummary,
    right: EpisodeSummary,
    stage: str,
    generation_rule: str,
    compatibility_type: str,
) -> tuple[tuple[object, ...], dict[str, object], int, int]:
    main, fragment = _main_and_fragment(left, right)
    date_a, date_b = left.anchor_date, right.anchor_date
    gap = abs(int((date_b - date_a).days))
    same_month = date_a.to_period("M") == date_b.to_period("M")
    same_year = date_a.year == date_b.year
    added = _components_added_summary(main, fragment)
    if not added:
        # Complementarity is symmetric; retain the direction that actually adds data.
        reverse = _components_added_summary(fragment, main)
        if reverse:
            main, fragment, added = fragment, main, reverse
    # This boolean summary is the cheap gate.  Static columns are inspected only
    # for candidates that can add clinical evidence.
    both_complete = bool(
        left.independent_complete_assessment
        and right.independent_complete_assessment
    )
    if both_complete:
        compatibility = {
            "compatible": False,
            "hard_conflict": False,
            "duplicate_complete_assessment": True,
            "conflicting_components": [],
        }
    elif added:
        compatibility = _compatibility(left, right)
    else:
        compatibility = {
            "compatible": False,
            "hard_conflict": False,
            "duplicate_complete_assessment": False,
            "conflicting_components": [],
        }
    permitted = bool(added) and bool(compatibility["compatible"])
    if stage == "interval_outlier":
        permitted &= bool(main.evidence.clinical_candidate) != bool(
            fragment.evidence.clinical_candidate
        )
    cross_interval = not bool(left.interval_names & right.interval_names)
    cross_year = not same_year
    dates = pd.concat(
        [left.rows["collection_date"], right.rows["collection_date"]]
    ).dropna()
    span = int((dates.max() - dates.min()).days) if not dates.empty else None
    suspected = bool(permitted and stage == "interval_outlier" and cross_year)
    exceptional = bool(
        permitted
        and (
            cross_year
            or cross_interval
            or (span is not None and span > 30)
            or compatibility_type == "optional_to_full_evaluation_family"
            or stage == "temporal_rescue"
        )
    )
    if permitted:
        reason = {
            "interval_outlier": "same_interval_outlier_complement",
            "temporal_rescue": "temporal_rescue_complementary",
        }.get(stage)
        reason = reason or {
            "natural_history_family_alias": "natural_history_alias_merge",
            "optional_to_full_evaluation_family": "optional_full_family_merge",
        }.get(compatibility_type, "interval_exact_same_year")
    elif compatibility["hard_conflict"]:
        reason = "rejected_hard_conflict"
    elif compatibility["duplicate_complete_assessment"]:
        reason = "rejected_multiple_complete_clinical"
    else:
        reason = "rejected_no_complementarity"
    review_reason = ""
    if permitted and cross_year:
        review_reason = "cross_year_same_interval_merge"
    elif compatibility["duplicate_complete_assessment"]:
        review_reason = "multiple_complete_clinical_episodes_same_interval"
    elif compatibility["hard_conflict"]:
        review_reason = "hard_structural_conflict"
    record = {
        "patient_id": left.rows["patient_id"].iloc[0],
        "candidate_generation_rule": generation_rule,
        "episode_or_fragment_a": "|".join(map(str, left.rows["row_id_raw"])),
        "episode_or_fragment_b": "|".join(map(str, right.rows["row_id_raw"])),
        "interval_a": " | ".join(
            dict.fromkeys(left.rows["interval_name"].dropna().astype(str))
        ),
        "interval_b": " | ".join(
            dict.fromkeys(right.rows["interval_name"].dropna().astype(str))
        ),
        "canonical_family_a": "|".join(sorted(left.families)),
        "canonical_family_b": "|".join(sorted(right.families)),
        "date_a": date_a,
        "date_b": date_b,
        "gap_days": gap,
        "same_calendar_month": same_month,
        "same_calendar_year": same_year,
        "interval_compatibility_type": compatibility_type,
        "a_visit_type": _episode_visit_type_from_evidence(left.evidence),
        "b_visit_type": _episode_visit_type_from_evidence(right.evidence),
        "clinical_components_added": "|".join(added),
        "components_added": len(added),
        "reunited_essdai_esspri": bool(
            (left.evidence.has_essdai_form or right.evidence.has_essdai_form)
            and (left.evidence.has_esspri_form or right.evidence.has_esspri_form)
            and not (left.evidence.has_essdai_form and left.evidence.has_esspri_form)
            and not (right.evidence.has_essdai_form and right.evidence.has_esspri_form)
        ),
        "hard_conflict": compatibility["hard_conflict"],
        "independent_complete_conflict": compatibility[
            "duplicate_complete_assessment"
        ],
        "duplicate_complete_assessment": compatibility["duplicate_complete_assessment"],
        "merge_performed": permitted,
        "merge_stage": stage,
        "merge_reason": reason,
        "cross_interval_merge": bool(permitted and cross_interval),
        "cross_year_merge": bool(permitted and cross_year),
        "suspected_date_error": suspected,
        "interval_order_anomaly": bool(
            stage == "interval_outlier"
            and fragment.scheduled_visits
            and main.scheduled_visits
        ),
        "exceptional_merge": exceptional,
        "manual_review_required": bool(review_reason),
        "manual_review_reason": review_reason,
        "final_clinical_episode_id": None,
        "cluster_span_days": span,
    }
    rank = (
        _interval_priority(compatibility_type, stage),
        0 if main.independent_complete_assessment else 1,
        0 if same_year else 1,
        0 if same_month else 1,
        gap,
        -len(added),
        tuple(-value for value in _clinical_strength(main)),
    )
    record["target_rank"] = repr(rank)
    return rank, record, main.episode_id, fragment.episode_id


def _mark_review(episode: EpisodeSummary, reason: str) -> None:
    episode.rows.loc[:, "manual_review_required"] = True
    episode.rows.loc[:, "manual_review_reason"] = episode.rows[
        "manual_review_reason"
    ].map(lambda value: _append_pipe_value(value, reason))


def _merge_episode(
    target: EpisodeSummary,
    fragment: EpisodeSummary,
    record: dict[str, object],
    next_id: int,
) -> EpisodeSummary:
    target_rows = target.rows.copy()
    fragment_rows = fragment.rows.copy()
    if record["suspected_date_error"]:
        fragment_rows.loc[:, "suspected_date_error"] = True
    rows = pd.concat([target_rows, fragment_rows]).sort_values("_source_order").copy()
    rule = {
        "interval_same_year": "interval_exact_same_year",
        "interval_outlier": "same_interval_cross_year_complement",
        "temporal_rescue": (
            "temporal_rescue_same_calendar_month"
            if record["same_calendar_month"]
            else "temporal_rescue_within_30_days"
        ),
    }[str(record["merge_stage"])]
    for column, value in {
        "assignment_rule": rule,
        "merge_stage": str(record["merge_stage"]),
        "merge_reason": str(record["merge_reason"]),
        "interval_compatibility_type": str(record["interval_compatibility_type"]),
        "clinical_components_added": str(record["clinical_components_added"]),
        "manual_review_reason": str(record["manual_review_reason"]),
    }.items():
        rows.loc[:, column] = rows[column].map(
            lambda existing, item=value: _append_pipe_value(existing, item)
        )
    for column in (
        "cross_interval_merge", "cross_year_merge",
        "interval_order_anomaly", "exceptional_merge", "manual_review_required",
    ):
        rows.loc[:, column] = rows[column].astype(bool) | bool(record[column])
    flag_names = list(
        dict.fromkeys(
            (
                *BLOCK_PREFIXES,
                "has_essdai_total",
                "has_esspri_core",
                *RESEARCH_PREFIXES,
            )
        )
    )
    merged_flags = {
        flag: bool(target.evidence[flag]) or bool(fragment.evidence[flag])
        for flag in flag_names
    }
    evidence = _classify_evidence(pd.DataFrame([merged_flags])).iloc[0]
    return EpisodeSummary(
        episode_id=next_id,
        rows=rows,
        evidence=evidence,
        anchor_date=_episode_date(rows),
        interval_names=target.interval_names | fragment.interval_names,
        families=target.families | fragment.families,
        scheduled_visits=target.scheduled_visits | fragment.scheduled_visits,
        independent_complete_assessment=is_independent_complete_assessment(evidence),
    )


def _run_candidate_stage(
    episodes: dict[int, EpisodeSummary],
    stage: str,
    audit: list[dict[str, object]],
    next_id: int,
) -> tuple[dict[int, EpisodeSummary], int, int, int]:
    candidates = _generate_candidates(episodes, stage)
    evaluated = []
    for left_id, right_id, rule, compatibility in candidates:
        rank, record, target_id, fragment_id = _evaluate_candidate(
            episodes[left_id], episodes[right_id], stage, rule, compatibility
        )
        evaluated.append((rank, target_id, fragment_id, record))
    by_fragment: dict[int, list[tuple[tuple[object, ...], int, dict[str, object]]]] = {}
    for rank, target_id, fragment_id, record in evaluated:
        audit.append(record)
        if record["merge_performed"]:
            by_fragment.setdefault(fragment_id, []).append((rank, target_id, record))
        elif record["manual_review_required"]:
            _mark_review(episodes[fragment_id], str(record["manual_review_reason"]))
    proposals = []
    ambiguous = 0
    for fragment_id, options in by_fragment.items():
        options.sort(key=lambda item: item[0])
        if len(options) > 1 and options[0][0] == options[1][0]:
            ambiguous += 1
            _mark_review(episodes[fragment_id], "ambiguous_multiple_candidate_targets")
            for _, _, record in options:
                record["merge_performed"] = False
                record["merge_reason"] = "rejected_ambiguous_multiple_targets"
                record["manual_review_required"] = True
                record["manual_review_reason"] = "ambiguous_multiple_candidate_targets"
            continue
        proposals.append((options[0][0], options[0][1], fragment_id, options[0][2]))
        for _, _, record in options[1:]:
            record["merge_performed"] = False
            record["merge_reason"] = "not_selected_better_candidate"
    current_target: dict[int, int] = {}
    merges = 0
    for _, target_id, fragment_id, record in sorted(
        proposals, key=lambda item: item[0]
    ):
        resolved_target = current_target.get(target_id, target_id)
        if resolved_target not in episodes or fragment_id not in episodes:
            record["merge_performed"] = False
            record["merge_reason"] = "not_selected_overlapping_merge"
            continue
        # Earlier attachments update the cached anchor.  Recheck only this selected
        # edge, rather than regenerating every patient pair.
        _, refreshed, _, _ = _evaluate_candidate(
            episodes[resolved_target],
            episodes[fragment_id],
            stage,
            str(record["candidate_generation_rule"]),
            str(record["interval_compatibility_type"]),
        )
        for key in (
            "clinical_components_added",
            "components_added",
            "hard_conflict",
            "duplicate_complete_assessment",
            "merge_performed",
            "merge_reason",
        ):
            record[key] = refreshed[key]
        if not record["merge_performed"]:
            continue
        merged = _merge_episode(
            episodes[resolved_target], episodes[fragment_id], record, next_id
        )
        del episodes[resolved_target]
        del episodes[fragment_id]
        episodes[next_id] = merged
        current_target[target_id] = next_id
        next_id += 1
        merges += 1
    return episodes, next_id, len(candidates), merges + ambiguous * 0


def _append_pipe_value(existing: object, value: str) -> str:
    values = (
        [] if pd.isna(existing) or str(existing) == "" else str(existing).split("|")
    )
    if value and value not in values:
        values.append(value)
    return "|".join(values)


def assign_episodes(atomic_units: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct episodes using bounded candidates and cached evidence."""
    started = perf_counter()
    metadata = pd.DataFrame(
        {column: value for column, value in QC_DEFAULTS.items()},
        index=atomic_units.index,
    )
    prepared = pd.concat([atomic_units, metadata], axis=1).copy()
    assigned: list[pd.DataFrame] = []
    audit: list[dict[str, object]] = []
    metrics = {
        "n_candidate_pairs_pass1_same_year": 0,
        "n_candidate_pairs_pass1_outlier": 0,
        "n_candidate_pairs_pass2": 0,
        "n_ambiguous_multiple_targets": 0,
    }
    patient_count = prepared["patient_id"].nunique(dropna=False)
    for patient_number, (patient_id, patient_rows) in enumerate(
        prepared.groupby("patient_id", sort=False, dropna=False), 1
    ):
        episodes = {
            number: _summarize_episode(number, patient_rows.loc[[index]].copy())
            for number, index in enumerate(patient_rows.index)
        }
        next_id = len(episodes)
        patient_counts = []
        for stage, metric in (
            ("interval_same_year", "n_candidate_pairs_pass1_same_year"),
            ("interval_outlier", "n_candidate_pairs_pass1_outlier"),
            ("temporal_rescue", "n_candidate_pairs_pass2"),
        ):
            episodes, next_id, stage_candidates, total_merges = _run_candidate_stage(
                episodes, stage, audit, next_id
            )
            metrics[metric] += stage_candidates
            patient_counts.append((stage_candidates, total_merges))
        if patient_number % 25 == 0 or patient_number == patient_count:
            # Stored for callers; main emits the cohort-level summary.
            metrics["last_patient_raw_rows"] = len(patient_rows)
            metrics["last_patient_pass1_candidates"] = sum(
                value[0] for value in patient_counts[:2]
            )
            metrics["last_patient_pass2_candidates"] = patient_counts[2][0]
        ordered = sorted(
            episodes.values(),
            key=lambda episode: (episode.anchor_date, episode.episode_id),
        )
        for sequence, episode in enumerate(ordered, 1):
            episode.rows.loc[:, "clinical_episode_id"] = (
                f"{patient_id}__CE{sequence:04d}"
            )
            assigned.append(episode.rows)
    result = pd.concat(assigned).sort_values("_source_order") if assigned else prepared
    decision_audit = pd.DataFrame(audit).reindex(columns=AUDIT_COLUMNS)
    if not decision_audit.empty:
        lookup = result.set_index("row_id_raw")["clinical_episode_id"].astype(str)
        merged_records = decision_audit.loc[decision_audit["merge_performed"]]
        for index, record in merged_records.iterrows():
            raw_id = str(record["episode_or_fragment_a"]).split("|")[0]
            match = lookup.loc[[key for key in lookup.index if str(key) == raw_id]]
            if not match.empty:
                decision_audit.at[index, "final_clinical_episode_id"] = match.iloc[0]
    metrics["wall_time_seconds"] = perf_counter() - started
    metrics["n_ambiguous_multiple_targets"] = int(
        decision_audit["manual_review_reason"]
        .eq("ambiguous_multiple_candidate_targets")
        .sum()
    )
    rules = decision_audit["candidate_generation_rule"]
    metrics.update(
        {
            "n_candidate_pairs_pass1_exact": int(
                rules.isin(
                    [
                        "same_scheduled_interval_same_year",
                        "same_exact_interval_same_year",
                    ]
                ).sum()
            ),
            "n_candidate_pairs_pass1_nh_family": int(
                rules.eq("natural_history_family_same_year").sum()
            ),
            "n_candidate_pairs_pass1_optional_full": int(
                rules.eq("optional_full_family_same_year").sum()
            ),
            "n_candidate_pairs_pass1_cross_year": int(
                decision_audit["merge_stage"].eq("interval_outlier").sum()
            ),
            "n_candidate_pairs_pass2_temporal": int(
                decision_audit["merge_stage"].eq("temporal_rescue").sum()
            ),
            "n_candidates_total": len(decision_audit),
            "n_merges_total": int(decision_audit["merge_performed"].sum()),
        }
    )
    result.attrs["merge_decision_audit"] = decision_audit
    result.attrs["performance_metrics"] = metrics
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
    performance = assigned.attrs.get("performance_metrics", {})
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
                "n_candidate_pairs_pass1_same_year": performance.get(
                    "n_candidate_pairs_pass1_same_year", len(audit.loc[
                        audit["merge_stage"].eq("interval_same_year")
                    ])
                ),
                "n_candidate_pairs_pass1_outlier": performance.get(
                    "n_candidate_pairs_pass1_outlier", len(audit.loc[
                        audit["merge_stage"].eq("interval_outlier")
                    ])
                ),
                "n_candidate_pairs_pass2": performance.get(
                    "n_candidate_pairs_pass2", len(audit.loc[
                        audit["merge_stage"].eq("temporal_rescue")
                    ])
                ),
                "n_candidate_pairs_pass1_exact": performance.get(
                    "n_candidate_pairs_pass1_exact", 0
                ),
                "n_candidate_pairs_pass1_nh_family": performance.get(
                    "n_candidate_pairs_pass1_nh_family", 0
                ),
                "n_candidate_pairs_pass1_optional_full": performance.get(
                    "n_candidate_pairs_pass1_optional_full", 0
                ),
                "n_candidate_pairs_pass1_cross_year": performance.get(
                    "n_candidate_pairs_pass1_cross_year", 0
                ),
                "n_candidate_pairs_pass2_temporal": performance.get(
                    "n_candidate_pairs_pass2_temporal", 0
                ),
                "n_candidates_total": performance.get(
                    "n_candidates_total", len(audit)
                ),
                "n_ambiguous_multiple_targets": performance.get(
                    "n_ambiguous_multiple_targets", 0
                ),
                "n_candidates_not_evaluated_due_to_time_window": 0,
                "episode_assignment_wall_time_seconds": performance.get(
                    "wall_time_seconds", pd.NA
                ),
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
        "Completed: %d raw rows -> %d clinical episodes; candidates "
        "PASS1A=%d PASS1B=%d PASS2=%d; merges=%d; assignment wall time=%.3fs",
        len(assigned),
        len(manifest),
        int(summary.loc[0, "n_candidate_pairs_pass1_same_year"]),
        int(summary.loc[0, "n_candidate_pairs_pass1_outlier"]),
        int(summary.loc[0, "n_candidate_pairs_pass2"]),
        int(summary.loc[0, "n_merges_total"]),
        float(summary.loc[0, "episode_assignment_wall_time_seconds"]),
    )


if __name__ == "__main__":
    main()
