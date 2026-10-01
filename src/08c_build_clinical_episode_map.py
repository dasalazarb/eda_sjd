"""Reconstruct clinical episodes without fragmenting ordinary intervals by date.

The assignment order is deliberate: ordinary intervals are consolidated first,
15-D Optional records are adjudicated against Natural History, remaining Optional
records are clustered with complete-link compatibility, and those clusters are
then compared with (but never allowed to join) principal episodes. Source dates
and values remain immutable; representative dates are navigation aids only.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import re
from pathlib import Path
from time import perf_counter
from typing import Iterable

import pandas as pd
from tqdm import tqdm

from common import ANALYTIC_DIR, INTERMEDIATE_DIR, MISSING_TOKENS, REPORTS_DIR, setup_logger

INPUT_PATH = ANALYTIC_DIR / "visits_long.parquet"
ROW_MAP_PATH = INTERMEDIATE_DIR / "clinical_episode_row_map.parquet"
MANIFEST_PATH = ANALYTIC_DIR / "clinical_episode_manifest.parquet"
QC_DIR = REPORTS_DIR / "clinical_episode_map"
MERGE_AUDIT_FILENAME = "08c_merge_decision_audit.csv"
VALUE_CONFLICTS_FILENAME = "08c_value_conflicts.csv"
MERGE_INCOMPATIBILITIES_FILENAME = "08c_merge_incompatibilities.csv"
MERGE_SUMMARY_FILENAME = "08c_merge_summary.csv"
LONG_SPANS_FILENAME = "08c_long_interval_spans.csv"
OPTIONAL_PAIRS_FILENAME = "08c_optional_pair_candidates.csv"
OPTIONAL_CLUSTERS_FILENAME = "08c_optional_clusters.csv"
OPTIONAL_ADJUDICATION_FILENAME = "08c_optional_adjudication.csv"
UNRESOLVED_FILENAME = "08c_unresolved_assignments.csv"
PROVENANCE_FILENAME = "08c_source_value_provenance.parquet"
DATE_DISCREPANCIES_FILENAME = "08c_date_discrepancies.csv"

NATURAL_HISTORY = "natural history protocol 478 interval"
OPTIONAL_NEAR_DAYS = 14
OPTIONAL_COMPATIBLE_DAYS = 180
OPTIONAL_LONG_GAP_DAYS = 365
PHASE_ORDER = {
    "Phase 1: Initial Full Evaluation": 1,
    "Phase 1: Second Full Evaluation": 2,
    "Phase 1: Final Full (Third Full) Evaluation": 3,
    "Phase 2: 4th Full Evaluation": 4,
    "Phase 2: 5th Full Evaluation": 5,
}
NORMALIZED_PHASE_ORDER = {name.casefold(): order for name, order in PHASE_ORDER.items()}
MISSING_UPPER = {str(value).strip().upper() for value in MISSING_TOKENS}
DATE_TERMS = ("date", "datetime", "time_24_hour", "timestamp")
TECHNICAL_COLUMNS = {
    "patient_id", "row_id_raw", "interval_name", "interval_normalized",
    "collection_date", "collection_year", "clinical_episode_id", "_source_order",
    "atomic_activity_unit_id", "daily_activity_unit_id", "row_ids_involved",
    "interval_names_involved", "assignment_rule", "merge_stage", "merge_rule",
    "manual_review_required", "manual_review_reason", "representative_interval",
    "representative_date", "episode_precedence", "optional_cluster_id",
    "optional_adjudication", "visit_type", "clinical_visit", "source_file",
    "source_protocol", "origin",
}
COMPONENT_PATTERNS = {
    "essdai": ("essdai",), "esspri": ("esspri",),
    "systems_review": ("systems_review", "systems review"),
    "visit_summary": ("visit_summary", "visit summary"),
    "eye_examination": ("eye_examination", "eye exam", "ocular"),
    "salivary_flow": ("salivary_flow", "salivary flow"),
    "oral_examination": ("oral_examination", "oral exam"),
    "physical_examination": ("physical_examination", "physical exam"),
}
MANIFEST_FLAG_COLUMNS = (
    "cross_interval_merge", "cross_year_merge", "long_interval_span_warning",
    "possible_date_entry_error", "optional_adjudication", "optional_cluster_ids",
    "episode_span_days", "manual_review_required", "manual_review_reason",
    "visit_type", "clinical_visit",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, default=INPUT_PATH)
    parser.add_argument("--row-map-path", type=Path, default=ROW_MAP_PATH)
    parser.add_argument("--manifest-path", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--qc-dir", type=Path, default=QC_DIR)
    return parser.parse_args()


def resolve_column(df: pd.DataFrame, names: Iterable[str], required: bool = True) -> str | None:
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


def _normalized_interval(value: object) -> str:
    """Normalize whitespace/case for identity comparison without changing source labels."""
    return "" if pd.isna(value) else re.sub(r"\s+", " ", str(value).strip().casefold())


def _is_15d_optional(value: str) -> bool:
    return value.startswith("15d optional evaluation")


def _is_optional(value: str) -> bool:
    return value.startswith("optional evaluation") or _is_15d_optional(value)


def get_episode_priority(interval_name: object) -> str:
    """Classify an interval for deterministic clinical episode precedence."""
    normalized = _normalized_interval(interval_name)
    if normalized == NATURAL_HISTORY:
        return "natural"
    if _is_15d_optional(normalized):
        return "15d_optional"
    if normalized in NORMALIZED_PHASE_ORDER:
        return f"phase_{NORMALIZED_PHASE_ORDER[normalized]}"
    if _is_optional(normalized):
        return "optional"
    return "other"


def intervals_are_compatible(interval_a: object, interval_b: object) -> bool:
    """Return compatibility without ever declaring two distinct principals identical."""
    left, right = _normalized_interval(interval_a), _normalized_interval(interval_b)
    if left == right:
        return True
    if {left, right} and ((left == NATURAL_HISTORY and _is_15d_optional(right)) or (right == NATURAL_HISTORY and _is_15d_optional(left))):
        return True
    return (_is_optional(left) and not _is_optional(right)) or (_is_optional(right) and not _is_optional(left))


def has_information(series: pd.Series) -> pd.Series:
    """Mark populated values while retaining zero, False, No, and negative values."""
    result = series.notna()
    if pd.api.types.is_object_dtype(series.dtype) or pd.api.types.is_string_dtype(series.dtype):
        text = series.astype("string").str.strip()
        result &= text.notna() & ~text.str.upper().isin(MISSING_UPPER)
    return result.fillna(False)


def _unique_values(values: Iterable[object]) -> list[object]:
    unique, seen = [], set()
    for value in values:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if not text or text.upper() in MISSING_UPPER:
            continue
        if text not in seen:
            unique.append(value); seen.add(text)
    return unique


def collapse_values(values: Iterable[object]) -> object:
    """Collapse unique nonmissing values without discarding disagreements."""
    unique = _unique_values(values)
    if not unique:
        return pd.NA
    return unique[0] if len(unique) == 1 else " | ".join(map(lambda x: str(x).strip(), unique))


def resolve_preferred_value(preferred_value: object, secondary_value: object, preferred_source: object, secondary_source: object) -> tuple[object, bool]:
    """Select the preferred populated source and report disagreement."""
    del preferred_source, secondary_source
    preferred, secondary = _unique_values([preferred_value]), _unique_values([secondary_value])
    if not preferred:
        return (secondary[0] if secondary else pd.NA), False
    if not secondary:
        return preferred[0], False
    return preferred[0], str(preferred[0]).strip() != str(secondary[0]).strip()


def prepare_visits(visits: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Create working columns while retaining every original source field."""
    patient_col = resolve_column(visits, ("patient_id", "patient_record_number"))
    interval_col = resolve_column(visits, ("interval_name",))
    date_col = resolve_column(visits, ("collection_date", "visit_date", "visit_datetime"))
    row_col = resolve_column(visits, ("row_id_raw",), required=False)
    result = visits.copy(); result["_source_order"] = range(len(result))
    result["patient_id"] = result[patient_col]; result["interval_name"] = result[interval_col]
    result["interval_normalized"] = result["interval_name"].map(_normalized_interval)
    result["collection_date"] = pd.to_datetime(result[date_col], errors="coerce").dt.normalize()
    result["collection_year"] = result["collection_date"].dt.year.astype("Int64")
    result["row_id_raw"] = result[row_col] if row_col else result["_source_order"]
    if result["patient_id"].isna().any():
        raise ValueError("patient_id contains missing values")
    if result["row_id_raw"].isna().any() or result["row_id_raw"].duplicated().any():
        raise ValueError("row_id_raw must be complete and unique")
    provenance = [str(c) for c in visits if any(t in str(c).lower() for t in ("protocol", "origin", "source"))]
    return result, list(dict.fromkeys(provenance))


def add_presence_flags(visits: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic component evidence derived from informative source values."""
    result = visits.copy()
    for component, patterns in COMPONENT_PATTERNS.items():
        columns = [c for c in result if any(pattern in str(c).casefold() for pattern in patterns)]
        result[f"has_{component}"] = pd.concat([has_information(result[c]) for c in columns], axis=1).any(axis=1) if columns else False
    return result


def build_atomic_activity_units(flagged_rows: pd.DataFrame, provenance_columns: Iterable[str] = ()) -> pd.DataFrame:
    """Create one immutable activity unit per raw source row."""
    units = flagged_rows.copy()
    units["atomic_activity_unit_id"] = units["row_id_raw"].map(lambda value: f"row-{value}")
    units["daily_activity_unit_id"] = units["atomic_activity_unit_id"]
    units["row_ids_involved"] = units["row_id_raw"].map(lambda value: (value,))
    units["interval_names_involved"] = units["interval_name"].map(lambda value: () if pd.isna(value) else (str(value),))
    for column in provenance_columns:
        units[f"{column}_involved"] = units[column].map(lambda value: () if pd.isna(value) else (str(value),))
    return units


def build_daily_activity_units(flagged_rows: pd.DataFrame, provenance_columns: Iterable[str] = ()) -> pd.DataFrame:
    """Backward-compatible alias with no same-day consolidation."""
    return build_atomic_activity_units(flagged_rows, provenance_columns)


def _data_columns(rows: pd.DataFrame) -> list[str]:
    """Return clinical data columns, excluding dates, IDs, flags, and provenance."""
    columns = []
    for column in rows:
        text, leaf = str(column).casefold(), str(column).split("__")[-1]
        if leaf in TECHNICAL_COLUMNS or str(column).startswith("has_") or str(column).endswith("_involved"):
            continue
        if any(term in text for term in DATE_TERMS) or any(term in text for term in ("source_", "protocol", "origin", "identifier", "subject_number")):
            continue
        columns.append(str(column))
    return columns


def _components(rows: pd.DataFrame) -> set[str]:
    return {name.removeprefix("has_") for name in rows if name.startswith("has_") and bool(rows[name].fillna(False).any())}


def _informative_columns(rows: pd.DataFrame) -> set[str]:
    return {column for column in _data_columns(rows) if has_information(rows[column]).any()}


def episodes_are_complementary(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    """Return whether either unit contributes an informative field absent in the other."""
    return bool(_informative_columns(left) ^ _informative_columns(right))


def find_incompatible_variables(record_a: pd.DataFrame, record_b: pd.DataFrame) -> list[dict[str, object]]:
    """Find differing populated clinical values, excluding date/provenance fields."""
    conflicts = []
    for column in sorted(set(_data_columns(record_a)) & set(_data_columns(record_b))):
        left, right = _unique_values(record_a[column]), _unique_values(record_b[column])
        if left and right and {str(v).strip() for v in left} != {str(v).strip() for v in right}:
            conflicts.append({"variable": column, "value_a": collapse_values(left), "value_b": collapse_values(right)})
    return conflicts


def _display(values: Iterable[object]) -> str:
    return " | ".join(str(value) for value in dict.fromkeys(values) if pd.notna(value))


def _stable_id(prefix: str, patient_id: object, row_ids: Iterable[object]) -> str:
    payload = f"{patient_id}|" + "|".join(sorted(map(str, row_ids)))
    return f"{prefix}{hashlib.sha1(payload.encode()).hexdigest()[:12]}"


def _documented_independent(rows: pd.DataFrame) -> bool:
    markers = ("documented_new_visit", "independent_visit", "distinct_clinical_event")
    return any(column in rows and has_information(rows[column]).any() and rows.loc[has_information(rows[column]), column].astype(str).str.casefold().isin({"1", "true", "yes", "y"}).any() for column in markers)


def _clinical_evidence(rows: pd.DataFrame) -> bool:
    components = _components(rows)
    physician = bool(components & {"essdai", "systems_review", "visit_summary", "physical_examination"})
    objective = bool(components & {"eye_examination", "salivary_flow", "oral_examination"})
    return len(components) >= 3 and physician and (objective or "esspri" in components)


def _date_stats(rows: pd.DataFrame) -> dict[str, object]:
    dates = sorted(rows["collection_date"].dropna().unique())
    start, end = (pd.Timestamp(dates[0]), pd.Timestamp(dates[-1])) if dates else (pd.NaT, pd.NaT)
    span = (end - start).days if dates else pd.NA
    years = {date.year for date in dates}
    same_month_day = len(dates) > 1 and len({(date.month, date.day) for date in dates}) < len(dates)
    return {"start": start, "end": end, "span": span, "dates": dates, "cross_year": len(years) > 1, "possible_error": bool(span is not pd.NA and span >= 365 and same_month_day)}


def _cluster_compatible(left: pd.DataFrame, right: pd.DataFrame) -> tuple[bool, str, object]:
    left_dates, right_dates = left["collection_date"].dropna(), right["collection_date"].dropna()
    days = min((abs((a - b).days) for a in left_dates for b in right_dates), default=pd.NA)
    if _documented_independent(left) or _documented_independent(right):
        return False, "documented distinct clinical event", days
    complementary = episodes_are_complementary(left, right)
    shared_components = bool(_components(left) & _components(right))
    if pd.isna(days):
        return False, "missing date and insufficient explicit linkage", days
    if days >= OPTIONAL_LONG_GAP_DAYS:
        return False, "optional gap >=365 days", days
    if days <= OPTIONAL_NEAR_DAYS and (complementary or shared_components):
        return True, "nearby complementary activity", days
    if days <= OPTIONAL_COMPATIBLE_DAYS and complementary:
        return True, "compatible complementary activity", days
    if days < OPTIONAL_LONG_GAP_DAYS and complementary and bool(_components(left) & _components(right)):
        return True, "strong component evidence across long optional gap", days
    return False, "insufficient evidence of same evaluation", days


def _complete_link_clusters(rows: pd.DataFrame, pair_records: list[dict[str, object]]) -> list[pd.DataFrame]:
    """Cluster Optional units only when every cross-pair is compatible."""
    units = [group.copy() for _, group in rows.groupby("row_id_raw", sort=True)]
    compatibility: dict[frozenset[object], bool] = {}
    for left, right in itertools.combinations(units, 2):
        compatible, reason, days = _cluster_compatible(left, right)
        left_id, right_id = left["row_id_raw"].iloc[0], right["row_id_raw"].iloc[0]
        compatibility[frozenset((left_id, right_id))] = compatible
        pair_records.append({
            "patient_id": left["patient_id"].iloc[0], "row_id_a": left_id, "row_id_b": right_id,
            "date_a": left["collection_date"].iloc[0], "date_b": right["collection_date"].iloc[0], "days_apart": days,
            "components_a": _display(sorted(_components(left))), "components_b": _display(sorted(_components(right))),
            "incremental_fields_a": _display(sorted(_informative_columns(left) - _informative_columns(right))),
            "incremental_fields_b": _display(sorted(_informative_columns(right) - _informative_columns(left))),
            "contradictions": _display(c["variable"] for c in find_incompatible_variables(left, right)),
            "independent_event_evidence": _documented_independent(left) or _documented_independent(right),
            "decision": "merge_candidate" if compatible else "keep_separate", "reason": reason, "optional_cluster_id": "",
        })
    clusters: list[list[pd.DataFrame]] = []
    for unit in units:
        candidates = []
        for index, cluster in enumerate(clusters):
            if all(compatibility.get(frozenset((unit["row_id_raw"].iloc[0], member["row_id_raw"].iloc[0])), True) for member in cluster):
                candidates.append(index)
        if candidates:
            clusters[candidates[0]].append(unit)
        else:
            clusters.append([unit])
    return [pd.concat(cluster).sort_values("row_id_raw") for cluster in clusters]


def _principal_group(rows: pd.DataFrame, rule: str) -> pd.DataFrame:
    result = rows.sort_values(["collection_date", "row_id_raw"], na_position="last").copy()
    result["assignment_rule"] = rule; result["merge_rule"] = rule; result["merge_stage"] = "ordinary_interval"
    result["representative_interval"] = result["interval_name"].iloc[0]
    result["representative_date"] = result["collection_date"].dropna().min() if result["collection_date"].notna().any() else pd.NaT
    result["episode_precedence"] = range(len(result))
    return result


def assign_episodes(atomic_units: pd.DataFrame) -> pd.DataFrame:
    """Assign episodes through ordinary, 15-D, Optional-cluster, and attachment stages."""
    started = perf_counter(); prepared = atomic_units.copy()
    defaults = {"assignment_rule": "standalone_record", "merge_stage": "standalone", "merge_rule": "standalone_record", "manual_review_required": False, "manual_review_reason": "", "optional_cluster_id": "", "optional_adjudication": "not_optional", "visit_type": "clinical_episode", "clinical_visit": True}
    for column, default in defaults.items(): prepared[column] = default
    episodes, audits, pairs, clusters_qc, adjudications, unresolved, incompatibilities = [], [], [], [], [], [], []
    for patient_id, patient in tqdm(prepared.groupby("patient_id", sort=True), desc="Building clinical episodes"):
        normalized = patient["interval_normalized"]
        principal_rows = patient.loc[~normalized.map(_is_optional)]
        optional_rows = patient.loc[normalized.map(_is_optional)]
        principals = [_principal_group(group, "same_normalized_interval_all_dates") for _, group in principal_rows.groupby("interval_normalized", sort=True, dropna=False)]
        natural = next((episode for episode in principals if episode["interval_normalized"].iloc[0] == NATURAL_HISTORY), None)
        remaining_optional = []
        for _, row in optional_rows.sort_values("row_id_raw").groupby("row_id_raw"):
            if _is_15d_optional(row["interval_normalized"].iloc[0]) and natural is not None and not _documented_independent(row):
                cluster_id = _stable_id("OC_", patient_id, row["row_id_raw"]); row = row.copy()
                contributed_fields = _informative_columns(row) - _informative_columns(natural)
                row["optional_cluster_id"] = cluster_id; row["optional_adjudication"] = "attached_to_natural_15d"; row["assignment_rule"] = "15d_optional_to_natural_default"; row["merge_rule"] = row["assignment_rule"]; row["merge_stage"] = "15d_reintegration"; row["representative_interval"] = natural["representative_interval"].iloc[0]; row["representative_date"] = natural["representative_date"].iloc[0]; row["episode_precedence"] = list(range(len(natural), len(natural) + len(row))); natural = pd.concat([natural, row]); principals = [natural if episode["interval_normalized"].iloc[0] == NATURAL_HISTORY else episode for episode in principals]
                adjudications.append({"patient_id": patient_id, "row_id_raw": row["row_id_raw"].iloc[0], "optional_cluster_id": cluster_id, "candidate_episode_ids": "Natural History", "candidate_distances_days": min((abs((a-b).days) for a in row["collection_date"].dropna() for b in natural["collection_date"].dropna()), default=pd.NA), "new_fields": _display(sorted(contributed_fields)), "decision": "attached_to_natural_15d", "justification": "15-D transversal default; no documented distinct event", "confidence": "provisional"})
            else:
                remaining_optional.append(row)
        remaining = pd.concat(remaining_optional) if remaining_optional else optional_rows.iloc[0:0]
        optional_clusters = _complete_link_clusters(remaining, pairs)
        # If Natural is absent, 15D Optional activities are reconstructed as one transversal cluster unless a distinct event is documented.
        if natural is None:
            fifteen = [c for c in optional_clusters if c["interval_normalized"].map(_is_15d_optional).all() and not _documented_independent(c)]
            if len(fifteen) > 1:
                combined = pd.concat(fifteen); optional_clusters = [c for c in optional_clusters if not c["interval_normalized"].map(_is_15d_optional).all()] + [combined]
        for cluster in optional_clusters:
            cluster = cluster.copy(); cluster_id = _stable_id("OC_", patient_id, cluster["row_id_raw"]); cluster["optional_cluster_id"] = cluster_id
            for record in pairs:
                if record["patient_id"] == patient_id and {record["row_id_a"], record["row_id_b"]}.issubset(set(cluster["row_id_raw"])): record["optional_cluster_id"] = cluster_id
            clinical = _clinical_evidence(cluster) or _documented_independent(cluster)
            candidates = []
            for index, principal in enumerate(principals):
                distances = [abs((a-b).days) for a in cluster["collection_date"].dropna() for b in principal["collection_date"].dropna()]
                if distances: candidates.append((min(distances), index, principal))
            candidates.sort(key=lambda item: (item[0], str(item[2]["interval_normalized"].iloc[0])))
            tied = len(candidates) > 1 and candidates[0][0] == candidates[1][0]
            complementary = bool(candidates and episodes_are_complementary(cluster, candidates[0][2]))
            attach = bool(candidates and not tied and not _documented_independent(cluster) and complementary and (candidates[0][0] <= OPTIONAL_COMPATIBLE_DAYS or (candidates[0][0] < OPTIONAL_LONG_GAP_DAYS and bool(_components(cluster)))))
            if attach:
                distance, index, principal = candidates[0]; cluster["optional_adjudication"] = "attached_to_main"; cluster["assignment_rule"] = "optional_cluster_to_main"; cluster["merge_rule"] = cluster["assignment_rule"]; cluster["merge_stage"] = "optional_to_main"; cluster["representative_interval"] = principal["representative_interval"].iloc[0]; cluster["representative_date"] = principal["representative_date"].iloc[0]; cluster["episode_precedence"] = list(range(len(principal), len(principal) + len(cluster))); principals[index] = pd.concat([principal, cluster]); decision, visit_type = "attached_to_main", "clinical_episode"
            else:
                decision = "independent_clinical" if clinical else ("unresolved" if tied or cluster["collection_date"].isna().all() else "partial_unattached")
                visit_type = "optional_independent_clinical" if clinical else ("optional_unresolved" if decision == "unresolved" else "optional_partial_unattached")
                cluster["optional_adjudication"] = decision; cluster["assignment_rule"] = decision; cluster["merge_rule"] = decision; cluster["merge_stage"] = "optional_residual"; cluster["representative_interval"] = cluster["interval_name"].iloc[0]; cluster["representative_date"] = cluster["collection_date"].dropna().min() if cluster["collection_date"].notna().any() else pd.NaT; cluster["episode_precedence"] = range(len(cluster)); cluster["visit_type"] = visit_type; cluster["clinical_visit"] = clinical; cluster["manual_review_required"] = not clinical; cluster["manual_review_reason"] = "ambiguous principal candidates" if tied else ("missing source date" if cluster["collection_date"].isna().all() else "partial Optional evidence")
                principals.append(cluster)
                if decision == "unresolved": unresolved.append({"patient_id": patient_id, "optional_cluster_id": cluster_id, "row_ids": _display(cluster["row_id_raw"]), "reason": cluster["manual_review_reason"].iloc[0], "candidate_intervals": _display(item[2]["representative_interval"].iloc[0] for item in candidates)})
            clusters_qc.append({"patient_id": patient_id, "optional_cluster_id": cluster_id, "row_ids": _display(cluster["row_id_raw"]), "source_dates": _display(cluster["collection_date"]), "sources": _display(cluster.get("source_file", pd.Series(dtype=object))), "components": _display(sorted(_components(cluster))), "clinical_potential": clinical, "destination": candidates[0][2]["representative_interval"].iloc[0] if attach else "", "classification": "clinical" if clinical else ("ambiguous" if decision == "unresolved" else "complementary")})
            for row_id in cluster["row_id_raw"]:
                adjudications.append({"patient_id": patient_id, "row_id_raw": row_id, "optional_cluster_id": cluster_id, "candidate_episode_ids": _display(item[2]["representative_interval"].iloc[0] for item in candidates), "candidate_distances_days": _display(item[0] for item in candidates), "new_fields": _display(sorted(_informative_columns(cluster) - (_informative_columns(candidates[0][2]) if candidates else set()))), "decision": decision, "justification": cluster["assignment_rule"].iloc[0], "confidence": "high" if clinical or attach else "review"})
        for episode in principals: episodes.append(episode)
    # Stable episode IDs depend on patient, representative identity, and immutable row IDs, not input order.
    for episode in episodes:
        episode["clinical_episode_id"] = _stable_id("EP_", episode["patient_id"].iloc[0], episode["row_id_raw"])
        stats = _date_stats(episode)
        if len(episode) > 1:
            audits.append({"patient_id": episode["patient_id"].iloc[0], "clinical_episode_id": episode["clinical_episode_id"].iloc[0], "row_ids": _display(episode["row_id_raw"]), "intervals": _display(episode["interval_name"]), "merge_rule": _display(episode["merge_rule"]), "merged": True, "episode_span_days": stats["span"], "reason": "staged reconstruction"})
    result = pd.concat(episodes).sort_values(["patient_id", "row_id_raw"]) if episodes else prepared
    result.attrs.update({"merge_decision_audit": pd.DataFrame(audits), "merge_incompatibilities": pd.DataFrame(incompatibilities), "optional_pair_candidates": pd.DataFrame(pairs), "optional_clusters": pd.DataFrame(clusters_qc), "optional_adjudication": pd.DataFrame(adjudications), "unresolved_assignments": pd.DataFrame(unresolved), "performance_metrics": {"wall_time_seconds": perf_counter()-started, "n_candidates_total": len(audits)+len(pairs), "n_merges_total": len(audits)}})
    return result


def propagate_episode_assignments(flagged_rows: pd.DataFrame, assigned_units: pd.DataFrame) -> pd.DataFrame:
    """Propagate assignment metadata without modifying original source values."""
    columns = ["row_id_raw", "clinical_episode_id", "assignment_rule", "merge_stage", "merge_rule", "manual_review_required", "manual_review_reason", "representative_interval", "representative_date", "episode_precedence", "optional_cluster_id", "optional_adjudication", "visit_type", "clinical_visit"]
    result = flagged_rows.merge(assigned_units[columns], on="row_id_raw", how="left", validate="one_to_one").sort_values("_source_order")
    result.attrs.update(assigned_units.attrs); return result


def build_manifest(assigned: pd.DataFrame, source_intervals: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build one calculated manifest record per frozen episode."""
    del source_intervals; records = []
    for (patient_id, episode_id), rows in assigned.groupby(["patient_id", "clinical_episode_id"], sort=True):
        stats = _date_stats(rows); review = bool(rows["manual_review_required"].any() or (stats["span"] is not pd.NA and stats["span"] >= 365))
        reasons = _display(rows.loc[rows["manual_review_reason"].astype(str).ne(""), "manual_review_reason"])
        if stats["span"] is not pd.NA and stats["span"] >= 365: reasons = _display([reasons, "episode span >=365 days"])
        records.append({"patient_id": patient_id, "clinical_episode_id": episode_id, "intervals_involved": _display(rows["interval_name"]), "representative_interval": rows["representative_interval"].iloc[0], "episode_start_date": stats["start"], "clinical_anchor_date": rows["representative_date"].iloc[0], "episode_end_date": stats["end"], "episode_span_days": stats["span"], "n_source_dates": len(stats["dates"]), "source_dates": _display(stats["dates"]), "n_raw_rows": len(rows), "visit_type": rows["visit_type"].iloc[0], "clinical_visit": bool(rows["clinical_visit"].any()), "manual_review_required": review, "manual_review_reason": reasons, "assignment_rule": _display(rows["assignment_rule"]), "merge_stage": _display(rows["merge_stage"]), "cross_interval_merge": rows["interval_normalized"].nunique(dropna=False) > 1, "cross_year_merge": stats["cross_year"], "long_interval_span_warning": bool(stats["span"] is not pd.NA and stats["span"] >= 365), "possible_date_entry_error": stats["possible_error"], "optional_adjudication": _display(rows["optional_adjudication"]), "optional_cluster_ids": _display(rows["optional_cluster_id"]), "interval_order_anomaly": False, "exceptional_merge": False})
    return pd.DataFrame(records)


def collapse_episode_rows(rows: pd.DataFrame) -> pd.Series:
    """Collapse an episode with principal-source precedence and conflict preservation."""
    ordered = rows.sort_values(["episode_precedence", "collection_date", "row_id_raw"], na_position="last"); representative = ordered["representative_interval"].iloc[0]
    primary = ordered.loc[ordered["interval_name"].eq(representative)]; secondary = ordered.loc[~ordered.index.isin(primary.index)]; collapsed = {}
    for column in _data_columns(rows):
        collapsed[column], _ = resolve_preferred_value(collapse_values(primary[column]), collapse_values(secondary[column]), representative, _display(secondary["interval_name"]))
    return pd.Series(collapsed)


def build_value_conflicts(assigned: pd.DataFrame) -> pd.DataFrame:
    """Record every conflicting clinical value with row/date/source provenance."""
    records = []
    for (patient_id, episode_id), rows in assigned.groupby(["patient_id", "clinical_episode_id"], sort=True):
        for column in _data_columns(rows):
            populated = rows.loc[has_information(rows[column])]
            if len(_unique_values(populated[column])) < 2: continue
            intervals = populated["interval_normalized"]; conflict_type = "same_interval_conflict" if intervals.nunique() == 1 else ("15d_natural_conflict" if intervals.eq(NATURAL_HISTORY).any() and intervals.map(_is_15d_optional).any() else ("optional_main_conflict" if intervals.map(_is_optional).any() else "temporal_value_change"))
            records.append({"patient_id": patient_id, "clinical_episode_id": episode_id, "variable": column, "source_values": _display(populated[column]), "source_dates": _display(populated["collection_date"]), "source_intervals": _display(populated["interval_name"]), "source_row_ids": _display(populated["row_id_raw"]), "source_protocols": _display(populated.get("source_protocol", pd.Series(dtype=object))), "merge_rule": _display(populated["merge_rule"]), "conflict_type": conflict_type, "selected_value": collapse_episode_rows(rows).get(column), "selection_rule": "principal_precedence_preserve_all_sources"})
    return pd.DataFrame(records)


def build_date_discrepancies(assigned: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Build temporal QC separately from clinical value conflicts."""
    records = []
    date_columns = [c for c in assigned if any(term in str(c).casefold() for term in DATE_TERMS) and c != "collection_date"]
    for _, episode in manifest.iterrows():
        rows = assigned.loc[assigned["clinical_episode_id"].eq(episode["clinical_episode_id"])]
        if episode["n_source_dates"] > 1 or any(rows[c].nunique(dropna=True) > 1 for c in date_columns):
            records.append({"patient_id": episode["patient_id"], "clinical_episode_id": episode["clinical_episode_id"], "intervals": episode["intervals_involved"], "source_dates": episode["source_dates"], "source_row_ids": _display(rows["row_id_raw"]), "episode_span_days": episode["episode_span_days"], "date_discrepancy_detected": True, "possible_date_entry_error": episode["possible_date_entry_error"], "date_error_evidence": "same month/day in different years" if episode["possible_date_entry_error"] else "multiple source dates retained", "date_field_discrepancies": _display(c for c in date_columns if rows[c].nunique(dropna=True) > 1), "manual_review_required": episode["manual_review_required"]})
    return pd.DataFrame(records)


def build_source_value_provenance(assigned: pd.DataFrame) -> pd.DataFrame:
    """Return one row per informative clinical value and its source chronology."""
    records = []
    selected = {episode_id: collapse_episode_rows(rows) for episode_id, rows in assigned.groupby("clinical_episode_id")}
    for _, row in assigned.iterrows():
        for variable in _data_columns(assigned):
            if not has_information(pd.Series([row.get(variable)])).iloc[0]: continue
            records.append({"patient_id": row["patient_id"], "clinical_episode_id": row["clinical_episode_id"], "row_id_raw": row["row_id_raw"], "variable": variable, "source_value": row[variable], "source_date": row["collection_date"], "source_interval": row["interval_name"], "optional_cluster_id": row["optional_cluster_id"], "selected_value": selected[row["clinical_episode_id"]].get(variable), "selection_rule": "principal_precedence_preserve_all_sources"})
    return pd.DataFrame(records)


def build_long_interval_spans(assigned: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Return every successfully merged episode spanning at least 365 days."""
    records = []
    for _, episode in manifest.loc[manifest["long_interval_span_warning"]].iterrows():
        rows = assigned.loc[assigned["clinical_episode_id"].eq(episode["clinical_episode_id"])]
        records.append({"patient_id": episode["patient_id"], "clinical_episode_id": episode["clinical_episode_id"], "interval": episode["representative_interval"], "min_date": episode["episode_start_date"], "max_date": episode["episode_end_date"], "span_days": episode["episode_span_days"], "years": _display(rows["collection_date"].dropna().dt.year), "source_dates": episode["source_dates"], "source_row_ids": _display(rows["row_id_raw"]), "cross_year_merge": episode["cross_year_merge"], "possible_date_entry_error": episode["possible_date_entry_error"], "date_error_evidence": "same month/day in different years" if episode["possible_date_entry_error"] else "no reproducible digit-entry indicator", "manual_review_required": True})
    return pd.DataFrame(records)


def validate_final_assignments(source: pd.DataFrame, assigned: pd.DataFrame) -> tuple[int, int]:
    """Assert exact row conservation, unique assignment, and immutable identity/date fields."""
    counts = assigned["row_id_raw"].value_counts(); unassigned = len(set(source["row_id_raw"])-set(assigned["row_id_raw"])); multiplied = int((counts > 1).sum())
    assert len(source) == len(assigned) and not unassigned and not multiplied
    columns = ["row_id_raw", "patient_id", "collection_date", "interval_name"]
    assert source[columns].sort_values("row_id_raw").reset_index(drop=True).equals(assigned[columns].sort_values("row_id_raw").reset_index(drop=True))
    assert not assigned["clinical_episode_id"].isna().any(); return unassigned, multiplied


def build_merge_summary(assigned: pd.DataFrame, manifest: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    """Build merge counts and hard conservation checks."""
    return pd.DataFrame([{ "n_raw_rows": len(assigned), "n_patients": assigned["patient_id"].nunique(), "n_episodes_before": len(assigned), "n_episodes_final": len(manifest), "n_merges_total": max(0, len(assigned)-len(manifest)), "n_same_interval_merges": int(assigned["merge_rule"].eq("same_normalized_interval_all_dates").sum()-manifest["assignment_rule"].str.contains("same_normalized_interval_all_dates").sum()), "n_15d_to_natural": int(assigned["optional_adjudication"].eq("attached_to_natural_15d").sum()), "n_optional_to_main": int(assigned["optional_adjudication"].eq("attached_to_main").sum()), "n_optional_clinical": int(manifest["visit_type"].eq("optional_independent_clinical").sum()), "n_optional_ambiguous": int(manifest["visit_type"].eq("optional_unresolved").sum()), "n_long_interval_warnings": int(manifest["long_interval_span_warning"].sum()), "raw_rows_unassigned": 0, "raw_rows_multiply_assigned": 0, "episode_assignment_wall_time_seconds": assigned.attrs.get("performance_metrics", {}).get("wall_time_seconds", pd.NA)}])


def write_parquet_and_csv(frame: pd.DataFrame, parquet_path: Path) -> tuple[Path, Path]:
    """Write a dataframe to Parquet and CSV without runtime attrs."""
    parquet_path.parent.mkdir(parents=True, exist_ok=True); csv_path = parquet_path.with_suffix(".csv"); serializable = frame.copy(deep=False); serializable.attrs = {}; serializable.to_parquet(parquet_path, index=False); serializable.to_csv(csv_path, index=False); return parquet_path, csv_path


def main() -> None:
    """Build frozen episode assignments and all source-preserving QC products."""
    args = parse_args(); logger = setup_logger("08c_build_clinical_episode_map"); logger.info("Reading %s", args.input_path)
    source, provenance_columns = prepare_visits(pd.read_parquet(args.input_path)); flagged = add_presence_flags(source); units = build_atomic_activity_units(flagged, provenance_columns); assigned_units = assign_episodes(units); assigned = propagate_episode_assignments(flagged, assigned_units); validate_final_assignments(source, assigned)
    manifest = build_manifest(assigned); conflicts = build_value_conflicts(assigned); dates = build_date_discrepancies(assigned, manifest); provenance = build_source_value_provenance(assigned); long_spans = build_long_interval_spans(assigned, manifest); audit = assigned.attrs["merge_decision_audit"]; summary = build_merge_summary(assigned, manifest, audit)
    write_parquet_and_csv(assigned, args.row_map_path); write_parquet_and_csv(manifest, args.manifest_path); args.qc_dir.mkdir(parents=True, exist_ok=True)
    outputs = {MERGE_AUDIT_FILENAME: audit, VALUE_CONFLICTS_FILENAME: conflicts, MERGE_INCOMPATIBILITIES_FILENAME: assigned.attrs["merge_incompatibilities"], MERGE_SUMMARY_FILENAME: summary, LONG_SPANS_FILENAME: long_spans, OPTIONAL_PAIRS_FILENAME: assigned.attrs["optional_pair_candidates"], OPTIONAL_CLUSTERS_FILENAME: assigned.attrs["optional_clusters"], OPTIONAL_ADJUDICATION_FILENAME: assigned.attrs["optional_adjudication"], UNRESOLVED_FILENAME: assigned.attrs["unresolved_assignments"], DATE_DISCREPANCIES_FILENAME: dates}
    for filename, frame in outputs.items(): frame.to_csv(args.qc_dir / filename, index=False)
    write_parquet_and_csv(provenance, args.qc_dir / PROVENANCE_FILENAME)
    logger.info("Completed: %d raw rows -> %d episodes; %d conflicts", len(assigned), len(manifest), len(conflicts))


if __name__ == "__main__":
    main()
