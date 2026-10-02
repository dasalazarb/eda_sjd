"""Reconstruct clinical episodes without fragmenting ordinary intervals by date.

The assignment order is deliberate: ordinary intervals are consolidated first,
15-D Optional records are adjudicated against Natural History, remaining Optional
records are clustered with complete-link compatibility, and those clusters are
then compared with (but never allowed to join) principal episodes. Source dates
and values remain immutable; representative dates are navigation aids only.

The default ``standard`` QC mode writes provenance only for exact
``(clinical_episode_id, variable)`` conflict keys and skips pair-level detail.
Use ``--qc-mode full`` for exhaustive provenance and enriched Optional pair QC,
and ``--write-core-csv`` only when wide compatibility CSVs are required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import resource
from bisect import bisect_left
from dataclasses import dataclass, field
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
PERFORMANCE_FILENAME = "08c_performance_metrics.json"

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
    parser.add_argument(
        "--qc-mode",
        choices=("standard", "full"),
        default="standard",
        help="standard writes conflict-only provenance; full writes exhaustive provenance",
    )
    parser.add_argument(
        "--export-full-provenance-csv",
        action="store_true",
        help="also export the potentially very large exhaustive provenance CSV",
    )
    parser.add_argument(
        "--write-core-csv",
        action="store_true",
        help="also write wide CSV copies of the row map and manifest",
    )
    parser.add_argument("--profile", action="store_true", help="log stage metrics")
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


@dataclass
class RowSignature:
    """Compact immutable matching metadata for one source row."""

    position: int
    patient_id: object
    row_id: object
    interval: str
    date: pd.Timestamp | None
    informative_mask: int
    component_mask: int
    independent: bool
    explicit_links: frozenset[tuple[str, str]]


@dataclass
class EpisodeState:
    """Mutable episode aggregate containing indices rather than clinical rows."""

    positions: list[int]
    informative_mask: int
    component_mask: int
    dates: list[pd.Timestamp] = field(default_factory=list)
    representative_interval: object = ""
    representative_date: object = pd.NaT

    def add(self, positions: Iterable[int], signatures: list[RowSignature]) -> None:
        """Add row positions and update compact aggregate signatures."""
        for position in positions:
            if position in self.positions:
                continue
            signature = signatures[position]
            self.positions.append(position)
            self.informative_mask |= signature.informative_mask
            self.component_mask |= signature.component_mask
            if signature.date is not None:
                self.dates.append(signature.date)
        self.dates.sort()


def _truthy_marker(value: object) -> bool:
    """Return whether a source value explicitly marks an independent visit."""
    return pd.notna(value) and str(value).strip().casefold() in {"1", "true", "yes", "y"}


def build_row_signatures(
    rows: pd.DataFrame,
) -> tuple[list[RowSignature], list[str], dict[str, int]]:
    """Precompute compact per-row clinical and linkage signatures.

    Parameters
    ----------
    rows : pd.DataFrame
        Prepared atomic source rows.

    Returns
    -------
    tuple
        Row signatures, ordered clinical columns, and component bit positions.
    """
    clinical_columns = _data_columns(rows)
    presence = {
        column: has_information(rows[column]).to_numpy(dtype=bool)
        for column in clinical_columns
    }
    component_columns = sorted(column for column in rows if column.startswith("has_"))
    component_bits = {column: index for index, column in enumerate(component_columns)}
    link_columns = [
        str(column)
        for column in rows
        if any(
            term in str(column).casefold()
            for term in ("evaluation_id", "visit_id", "event_id", "encounter_id", "accession")
        )
    ]
    marker_columns = [
        column
        for column in ("documented_new_visit", "independent_visit", "distinct_clinical_event")
        if column in rows
    ]
    signatures: list[RowSignature] = []
    for position in range(len(rows)):
        informative_mask = 0
        for bit, column in enumerate(clinical_columns):
            if presence[column][position]:
                informative_mask |= 1 << bit
        component_mask = 0
        for column, bit in component_bits.items():
            if bool(rows.iloc[position][column]):
                component_mask |= 1 << bit
        links = frozenset(
            (column, str(rows.iloc[position][column]).strip())
            for column in link_columns
            if has_information(pd.Series([rows.iloc[position][column]])).iloc[0]
        )
        raw_date = rows.iloc[position]["collection_date"]
        signatures.append(
            RowSignature(
                position=position,
                patient_id=rows.iloc[position]["patient_id"],
                row_id=rows.iloc[position]["row_id_raw"],
                interval=rows.iloc[position]["interval_normalized"],
                date=None if pd.isna(raw_date) else pd.Timestamp(raw_date),
                informative_mask=informative_mask,
                component_mask=component_mask,
                independent=any(_truthy_marker(rows.iloc[position][column]) for column in marker_columns),
                explicit_links=links,
            )
        )
    return signatures, clinical_columns, component_bits


def _signature_compatible(
    left: RowSignature, right: RowSignature
) -> tuple[bool, str, int | None]:
    """Evaluate Optional compatibility using compact signatures only."""
    days = None if left.date is None or right.date is None else abs((left.date - right.date).days)
    if left.independent or right.independent:
        return False, "documented distinct clinical event", days
    complementary = (left.informative_mask ^ right.informative_mask) != 0
    shared_component = (left.component_mask & right.component_mask) != 0
    if left.explicit_links & right.explicit_links and (complementary or shared_component):
        return True, "explicit same-evaluation linkage", days
    if days is None:
        return False, "missing date and insufficient explicit linkage", days
    if days >= OPTIONAL_LONG_GAP_DAYS:
        return False, "optional gap >=365 days", days
    if days <= OPTIONAL_NEAR_DAYS and (complementary or shared_component):
        return True, "nearby complementary activity", days
    if days <= OPTIONAL_COMPATIBLE_DAYS and complementary:
        return True, "compatible complementary activity", days
    if complementary and shared_component:
        return True, "strong component evidence across long optional gap", days
    return False, "insufficient evidence of same evaluation", days


def _candidate_signature_pairs(
    positions: list[int], signatures: list[RowSignature]
) -> tuple[list[tuple[int, int]], dict[str, int]]:
    """Generate sparse Optional candidate pairs without copying source rows."""
    dated = sorted(
        (position for position in positions if signatures[position].date is not None),
        key=lambda position: (signatures[position].date, str(signatures[position].row_id)),
    )
    candidates: list[tuple[int, int]] = []
    seen: set[frozenset[int]] = set()
    left = 0
    for right_index, right_position in enumerate(dated):
        right_date = signatures[right_position].date
        assert right_date is not None
        while left < right_index:
            left_date = signatures[dated[left]].date
            assert left_date is not None
            if (right_date - left_date).days < OPTIONAL_LONG_GAP_DAYS:
                break
            left += 1
        for prior_position in dated[left:right_index]:
            pair = frozenset((prior_position, right_position))
            seen.add(pair)
            candidates.append((prior_position, right_position))
    buckets: dict[tuple[object, ...], list[int]] = {}
    for position in positions:
        signature = signatures[position]
        if signature.date is not None:
            buckets.setdefault(("month_day", signature.date.month, signature.date.day), []).append(position)
        for link in signature.explicit_links:
            buckets.setdefault(("explicit", *link), []).append(position)
    for bucket in buckets.values():
        for index, first in enumerate(bucket):
            for second in bucket[index + 1 :]:
                pair = frozenset((first, second))
                if len(pair) < 2 or pair in seen:
                    continue
                seen.add(pair)
                candidates.append((first, second))
    theoretical = len(positions) * (len(positions) - 1) // 2
    return candidates, {
        "theoretical_pairs": theoretical,
        "candidate_pairs": len(candidates),
        "filtered_pairs": theoretical - len(candidates),
    }


def _complete_link_position_clusters(
    positions: list[int],
    signatures: list[RowSignature],
    decision_counts: dict[str, int],
    pair_records: list[dict[str, object]],
    include_pair_records: bool,
) -> tuple[list[list[int]], dict[str, int | float]]:
    """Cluster Optional row positions with cached complete-link compatibility."""
    generation_started = perf_counter()
    candidates, metrics = _candidate_signature_pairs(positions, signatures)
    metrics["optional_candidate_generation_seconds"] = perf_counter() - generation_started
    clustering_started = perf_counter()
    compatible: set[frozenset[int]] = set()
    for left, right in candidates:
        accepted, reason, days = _signature_compatible(signatures[left], signatures[right])
        decision_counts[reason] = decision_counts.get(reason, 0) + 1
        if accepted:
            compatible.add(frozenset((left, right)))
        if include_pair_records:
            pair_records.append(
                {
                "patient_id": signatures[left].patient_id,
                "row_id_a": signatures[left].row_id,
                "row_id_b": signatures[right].row_id,
                "days_apart": days,
                "decision": "merge_candidate" if accepted else "keep_separate",
                "reason": reason,
                "optional_cluster_id": "",
                }
            )
    clusters: list[list[int]] = []
    ordered = sorted(
        positions,
        key=lambda position: (
            pd.Timestamp.max if signatures[position].date is None else signatures[position].date,
            str(signatures[position].row_id),
        ),
    )
    for position in ordered:
        destination = next(
            (
                cluster
                for cluster in clusters
                if all(frozenset((position, member)) in compatible for member in cluster)
            ),
            None,
        )
        if destination is None:
            clusters.append([position])
        else:
            destination.append(position)
    row_to_cluster: dict[object, str] = {}
    for cluster in clusters:
        cluster_id = _stable_id(
            "OC_", signatures[cluster[0]].patient_id, (signatures[p].row_id for p in cluster)
        )
        for position in cluster:
            row_to_cluster[signatures[position].row_id] = cluster_id
    recent_records = pair_records[-len(candidates) :] if include_pair_records and candidates else ()
    for record in recent_records:
        if row_to_cluster.get(record["row_id_a"]) == row_to_cluster.get(record["row_id_b"]):
            record["optional_cluster_id"] = row_to_cluster[record["row_id_a"]]
    metrics.update(
        {
            "evaluated_pairs": len(candidates),
            "clusters": len(clusters),
            "optional_clustering_seconds": perf_counter() - clustering_started,
        }
    )
    return clusters, metrics


def _nearest_signature_distance(cluster: EpisodeState, principal: EpisodeState) -> int | None:
    """Return minimum distance between two sorted episode date lists."""
    if not cluster.dates or not principal.dates:
        return None
    nearest: int | None = None
    for date in cluster.dates:
        position = bisect_left(principal.dates, date)
        for index in (position - 1, position):
            if 0 <= index < len(principal.dates):
                distance = abs((date - principal.dates[index]).days)
                nearest = distance if nearest is None else min(nearest, distance)
    return nearest


def assign_episodes(
    atomic_units: pd.DataFrame, include_pair_records: bool = True
) -> pd.DataFrame:
    """Freeze episode assignment from compact row signatures, then materialize once."""
    total_started = perf_counter()
    prepared = atomic_units.reset_index(drop=True)
    signature_started = perf_counter()
    signatures, clinical_columns, component_bits = build_row_signatures(prepared)
    signature_seconds = perf_counter() - signature_started
    n_rows = len(prepared)
    metadata: dict[str, list[object]] = {
        "assignment_rule": ["standalone_record"] * n_rows,
        "merge_stage": ["standalone"] * n_rows,
        "merge_rule": ["standalone_record"] * n_rows,
        "manual_review_required": [False] * n_rows,
        "manual_review_reason": [""] * n_rows,
        "representative_interval": [""] * n_rows,
        "representative_date": [pd.NaT] * n_rows,
        "episode_precedence": [0] * n_rows,
        "optional_cluster_id": [""] * n_rows,
        "optional_adjudication": ["not_optional"] * n_rows,
        "visit_type": ["clinical_episode"] * n_rows,
        "clinical_visit": [True] * n_rows,
        "clinical_episode_id": [""] * n_rows,
    }
    patients: dict[object, list[int]] = {}
    for signature in signatures:
        patients.setdefault(signature.patient_id, []).append(signature.position)
    episodes: list[EpisodeState] = []
    pairs: list[dict[str, object]] = []
    clusters_qc: list[dict[str, object]] = []
    adjudications: list[dict[str, object]] = []
    unresolved: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    decision_counts: dict[str, int] = {}
    optional_counts: list[int] = []
    timings = {
        "principal_grouping_seconds": 0.0,
        "natural_15d_seconds": 0.0,
        "optional_candidate_generation_seconds": 0.0,
        "optional_clustering_seconds": 0.0,
        "optional_main_assignment_seconds": 0.0,
    }
    pair_metrics = {key: 0 for key in ("theoretical_pairs", "candidate_pairs", "evaluated_pairs", "filtered_pairs", "clusters")}

    for patient_id in tqdm(sorted(patients, key=str), desc="Building clinical episodes"):
        positions = patients[patient_id]
        started = perf_counter()
        principal_groups: dict[str, list[int]] = {}
        optional_positions: list[int] = []
        for position in positions:
            signature = signatures[position]
            if _is_optional(signature.interval):
                optional_positions.append(position)
            else:
                principal_groups.setdefault(signature.interval, []).append(position)
        principals: list[EpisodeState] = []
        for group_positions in principal_groups.values():
            first = prepared.iloc[group_positions[0]]
            dates = sorted(signatures[p].date for p in group_positions if signatures[p].date is not None)
            state = EpisodeState(
                positions=list(group_positions),
                informative_mask=0,
                component_mask=0,
                dates=dates,
                representative_interval=first["interval_name"],
                representative_date=dates[0] if dates else pd.NaT,
            )
            for position in group_positions:
                state.informative_mask |= signatures[position].informative_mask
                state.component_mask |= signatures[position].component_mask
                metadata["assignment_rule"][position] = "same_normalized_interval_all_dates"
                metadata["merge_rule"][position] = "same_normalized_interval_all_dates"
                metadata["merge_stage"][position] = "ordinary_interval"
            principals.append(state)
        timings["principal_grouping_seconds"] += perf_counter() - started

        started = perf_counter()
        natural = next(
            (state for state in principals if signatures[state.positions[0]].interval == NATURAL_HISTORY),
            None,
        )
        eligible_fifteen = [
            position
            for position in optional_positions
            if _is_15d_optional(signatures[position].interval) and not signatures[position].independent
        ]
        remaining = [position for position in optional_positions if position not in set(eligible_fifteen)]
        direct_fifteen: list[list[int]] = []
        if eligible_fifteen and natural is not None:
            baseline_mask = natural.informative_mask
            baseline_dates = list(natural.dates)
            natural.add(eligible_fifteen, signatures)
            for offset, position in enumerate(eligible_fifteen):
                cluster_id = _stable_id("OC_", patient_id, [signatures[position].row_id])
                metadata["optional_cluster_id"][position] = cluster_id
                metadata["optional_adjudication"][position] = "attached_to_natural_15d"
                metadata["assignment_rule"][position] = "15d_optional_to_natural_default"
                metadata["merge_rule"][position] = "15d_optional_to_natural_default"
                metadata["merge_stage"][position] = "15d_reintegration"
                adjudications.append(
                    {
                        "patient_id": patient_id,
                        "row_id_raw": signatures[position].row_id,
                        "optional_cluster_id": cluster_id,
                        "candidate_episode_ids": "Natural History",
                        "candidate_distances_days": _nearest_signature_distance(
                            EpisodeState([position], signatures[position].informative_mask, signatures[position].component_mask, [signatures[position].date] if signatures[position].date else []),
                            EpisodeState([], baseline_mask, natural.component_mask, baseline_dates),
                        ),
                        "new_fields": "",
                        "decision": "attached_to_natural_15d",
                        "justification": "15-D transversal default; no documented distinct event",
                        "confidence": "provisional",
                    }
                )
        elif eligible_fifteen:
            direct_fifteen.append(eligible_fifteen)
        timings["natural_15d_seconds"] += perf_counter() - started
        optional_counts.append(len(remaining))

        optional_clusters, cluster_metrics = _complete_link_position_clusters(
            remaining,
            signatures,
            decision_counts,
            pairs,
            include_pair_records,
        )
        optional_clusters.extend(direct_fifteen)
        for key in pair_metrics:
            pair_metrics[key] += int(cluster_metrics.get(key, 0))
        timings["optional_candidate_generation_seconds"] += float(cluster_metrics.get("optional_candidate_generation_seconds", 0.0))
        timings["optional_clustering_seconds"] += float(cluster_metrics.get("optional_clustering_seconds", 0.0))

        started = perf_counter()
        for cluster_positions in optional_clusters:
            cluster_mask = 0
            component_mask = 0
            cluster_dates: list[pd.Timestamp] = []
            independent = False
            for position in cluster_positions:
                signature = signatures[position]
                cluster_mask |= signature.informative_mask
                component_mask |= signature.component_mask
                independent |= signature.independent
                if signature.date is not None:
                    cluster_dates.append(signature.date)
            cluster_dates.sort()
            cluster = EpisodeState(cluster_positions, cluster_mask, component_mask, cluster_dates)
            cluster_id = _stable_id("OC_", patient_id, (signatures[p].row_id for p in cluster_positions))
            for position in cluster_positions:
                metadata["optional_cluster_id"][position] = cluster_id
            clinical = independent or (
                component_mask.bit_count() >= 3
                and any(
                    component_mask & (1 << component_bits.get(f"has_{name}", 10_000))
                    for name in ("essdai", "systems_review", "visit_summary", "physical_examination")
                    if f"has_{name}" in component_bits
                )
                and any(
                    component_mask & (1 << component_bits.get(f"has_{name}", 10_000))
                    for name in ("eye_examination", "salivary_flow", "oral_examination", "esspri")
                    if f"has_{name}" in component_bits
                )
            )
            candidates = []
            for index, principal in enumerate(principals):
                distance = _nearest_signature_distance(cluster, principal)
                if distance is not None:
                    candidates.append((distance, index, principal))
            candidates.sort(key=lambda item: (item[0], str(item[2].representative_interval)))
            tied = len(candidates) > 1 and candidates[0][0] == candidates[1][0]
            complementary = bool(candidates and (cluster_mask ^ candidates[0][2].informative_mask))
            attach = bool(
                candidates
                and not tied
                and not independent
                and complementary
                and (
                    candidates[0][0] <= OPTIONAL_COMPATIBLE_DAYS
                    or (candidates[0][0] < OPTIONAL_LONG_GAP_DAYS and component_mask)
                )
            )
            if attach:
                distance, _, principal = candidates[0]
                precedence = len(principal.positions)
                principal.add(cluster_positions, signatures)
                decision = "attached_to_main"
                for offset, position in enumerate(cluster_positions):
                    metadata["optional_adjudication"][position] = decision
                    metadata["assignment_rule"][position] = "optional_cluster_to_main"
                    metadata["merge_rule"][position] = "optional_cluster_to_main"
                    metadata["merge_stage"][position] = "optional_to_main"
                    metadata["episode_precedence"][position] = precedence + offset
            else:
                decision = "independent_clinical" if clinical else (
                    "unresolved" if tied or not cluster_dates else "partial_unattached"
                )
                visit_type = "optional_independent_clinical" if clinical else (
                    "optional_unresolved" if decision == "unresolved" else "optional_partial_unattached"
                )
                first = prepared.iloc[cluster_positions[0]]
                cluster.representative_interval = first["interval_name"]
                cluster.representative_date = cluster_dates[0] if cluster_dates else pd.NaT
                principals.append(cluster)
                reason = "ambiguous principal candidates" if tied else (
                    "missing source date" if not cluster_dates else "partial Optional evidence"
                )
                for offset, position in enumerate(cluster_positions):
                    metadata["optional_adjudication"][position] = decision
                    metadata["assignment_rule"][position] = decision
                    metadata["merge_rule"][position] = decision
                    metadata["merge_stage"][position] = "optional_residual"
                    metadata["episode_precedence"][position] = offset
                    metadata["visit_type"][position] = visit_type
                    metadata["clinical_visit"][position] = clinical
                    metadata["manual_review_required"][position] = not clinical
                    metadata["manual_review_reason"][position] = reason
                if decision == "unresolved":
                    unresolved.append(
                        {
                            "patient_id": patient_id,
                            "optional_cluster_id": cluster_id,
                            "row_ids": _display(signatures[p].row_id for p in cluster_positions),
                            "reason": reason,
                            "candidate_intervals": _display(item[2].representative_interval for item in candidates),
                        }
                    )
            destination = candidates[0][2].representative_interval if attach else ""
            clusters_qc.append(
                {
                    "patient_id": patient_id,
                    "optional_cluster_id": cluster_id,
                    "row_ids": _display(signatures[p].row_id for p in cluster_positions),
                    "source_dates": _display(cluster_dates),
                    "components": component_mask,
                    "clinical_potential": clinical,
                    "destination": destination,
                    "classification": "clinical" if clinical else ("ambiguous" if decision == "unresolved" else "complementary"),
                }
            )
            for position in cluster_positions:
                adjudications.append(
                    {
                        "patient_id": patient_id,
                        "row_id_raw": signatures[position].row_id,
                        "optional_cluster_id": cluster_id,
                        "candidate_episode_ids": _display(item[2].representative_interval for item in candidates),
                        "candidate_distances_days": _display(item[0] for item in candidates),
                        "new_fields": "",
                        "decision": decision,
                        "justification": metadata["assignment_rule"][position],
                        "confidence": "high" if clinical or attach else "review",
                    }
                )
        timings["optional_main_assignment_seconds"] += perf_counter() - started
        episodes.extend(principals)

    freeze_started = perf_counter()
    for episode in episodes:
        episode_id = _stable_id(
            "EP_", signatures[episode.positions[0]].patient_id, (signatures[p].row_id for p in episode.positions)
        )
        representative_interval = episode.representative_interval
        representative_date = episode.representative_date
        for precedence, position in enumerate(episode.positions):
            metadata["clinical_episode_id"][position] = episode_id
            metadata["representative_interval"][position] = representative_interval
            metadata["representative_date"][position] = representative_date
            if metadata["episode_precedence"][position] == 0:
                metadata["episode_precedence"][position] = precedence
        if len(episode.positions) > 1:
            episode_rows = prepared.iloc[episode.positions]
            stats = _date_stats(episode_rows)
            audits.append(
                {
                    "patient_id": signatures[episode.positions[0]].patient_id,
                    "clinical_episode_id": episode_id,
                    "row_ids": _display(signatures[p].row_id for p in episode.positions),
                    "intervals": _display(episode_rows["interval_name"]),
                    "merge_rule": _display(metadata["merge_rule"][p] for p in episode.positions),
                    "merged": True,
                    "episode_span_days": stats["span"],
                    "reason": "staged reconstruction",
                }
            )
    result = prepared.assign(**metadata).sort_values(["patient_id", "row_id_raw"])
    optional_series = pd.Series(optional_counts, dtype="int64")
    performance = {
        "episode_assignment_seconds": perf_counter() - total_started,
        "metadata_signatures_seconds": signature_seconds,
        "freeze_assignment_seconds": perf_counter() - freeze_started,
        "n_candidates_total": pair_metrics["candidate_pairs"],
        "n_merges_total": len(audits),
        "clinical_columns": len(clinical_columns),
        "decision_counts": decision_counts,
        **pair_metrics,
        **timings,
        "optional_per_patient_p50": float(optional_series.quantile(0.50)) if len(optional_series) else 0.0,
        "optional_per_patient_p95": float(optional_series.quantile(0.95)) if len(optional_series) else 0.0,
        "optional_per_patient_max": int(optional_series.max()) if len(optional_series) else 0,
    }
    result.attrs.update(
        {
            "merge_decision_audit": pd.DataFrame(audits),
            "merge_incompatibilities": pd.DataFrame(),
            "optional_pair_candidates": pd.DataFrame(pairs),
            "optional_clusters": pd.DataFrame(clusters_qc),
            "optional_adjudication": pd.DataFrame(adjudications),
            "unresolved_assignments": pd.DataFrame(unresolved),
            "performance_metrics": performance,
        }
    )
    return result
def propagate_episode_assignments(flagged_rows: pd.DataFrame, assigned_units: pd.DataFrame) -> pd.DataFrame:
    """Propagate assignment metadata without modifying original source values."""
    columns = ["row_id_raw", "clinical_episode_id", "assignment_rule", "merge_stage", "merge_rule", "manual_review_required", "manual_review_reason", "representative_interval", "representative_date", "episode_precedence", "optional_cluster_id", "optional_adjudication", "visit_type", "clinical_visit"]
    result = flagged_rows.merge(assigned_units[columns], on="row_id_raw", how="left", validate="one_to_one").sort_values("_source_order")
    result.attrs.update(assigned_units.attrs); return result


def build_optional_pair_qc(
    assigned: pd.DataFrame, pair_decisions: pd.DataFrame
) -> pd.DataFrame:
    """Enrich frozen Optional pair decisions without changing assignments."""
    if pair_decisions.empty:
        return pair_decisions.copy()
    rows_by_id = {
        row_id: rows
        for row_id, rows in assigned.groupby("row_id_raw", sort=False)
    }
    records: list[dict[str, object]] = []
    for decision in pair_decisions.to_dict("records"):
        left = rows_by_id[decision["row_id_a"]]
        right = rows_by_id[decision["row_id_b"]]
        record = dict(decision)
        record.update(
            {
                "date_a": left["collection_date"].iloc[0],
                "date_b": right["collection_date"].iloc[0],
                "components_a": _display(sorted(_components(left))),
                "components_b": _display(sorted(_components(right))),
                "incremental_fields_a": _display(
                    sorted(_informative_columns(left) - _informative_columns(right))
                ),
                "incremental_fields_b": _display(
                    sorted(_informative_columns(right) - _informative_columns(left))
                ),
                "contradictions": _display(
                    conflict["variable"]
                    for conflict in find_incompatible_variables(left, right)
                ),
                "independent_event_evidence": (
                    _documented_independent(left) or _documented_independent(right)
                ),
            }
        )
        records.append(record)
    return pd.DataFrame(records)


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
    output_columns = [
        "patient_id",
        "clinical_episode_id",
        "variable",
        "source_values",
        "source_dates",
        "source_intervals",
        "source_row_ids",
        "source_protocols",
        "merge_rule",
        "conflict_type",
        "selected_value",
        "selection_rule",
    ]
    records = []
    clinical_columns = _data_columns(assigned)
    multi = assigned.loc[
        assigned.duplicated(["patient_id", "clinical_episode_id"], keep=False)
    ]
    for (patient_id, episode_id), rows in multi.groupby(
        ["patient_id", "clinical_episode_id"], sort=True
    ):
        ordered = rows.sort_values(
            ["episode_precedence", "collection_date", "row_id_raw"],
            na_position="last",
        )
        populated_counts = rows[clinical_columns].notna().sum(axis=0)
        candidate_columns = populated_counts.index[populated_counts.ge(2)]
        for column in candidate_columns:
            populated = rows.loc[has_information(rows[column])]
            if len(_unique_values(populated[column])) < 2: continue
            representative = ordered["representative_interval"].iloc[0]
            primary = ordered.loc[ordered["interval_name"].eq(representative), column]
            secondary = ordered.loc[~ordered["interval_name"].eq(representative), column]
            selected_value, _ = resolve_preferred_value(
                collapse_values(primary),
                collapse_values(secondary),
                representative,
                "secondary",
            )
            intervals = populated["interval_normalized"]; conflict_type = "same_interval_conflict" if intervals.nunique() == 1 else ("15d_natural_conflict" if intervals.eq(NATURAL_HISTORY).any() and intervals.map(_is_15d_optional).any() else ("optional_main_conflict" if intervals.map(_is_optional).any() else "temporal_value_change"))
            records.append({"patient_id": patient_id, "clinical_episode_id": episode_id, "variable": column, "source_values": _display(populated[column]), "source_dates": _display(populated["collection_date"]), "source_intervals": _display(populated["interval_name"]), "source_row_ids": _display(populated["row_id_raw"]), "source_protocols": _display(populated.get("source_protocol", pd.Series(dtype=object))), "merge_rule": _display(populated["merge_rule"]), "conflict_type": conflict_type, "selected_value": selected_value, "selection_rule": "principal_precedence_preserve_all_sources"})
    return pd.DataFrame(records, columns=output_columns)


def build_date_discrepancies(assigned: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Build temporal QC separately from clinical value conflicts."""
    records = []
    date_columns = [c for c in assigned if any(term in str(c).casefold() for term in DATE_TERMS) and c != "collection_date"]
    episode_rows = {episode_id: rows for episode_id, rows in assigned.groupby("clinical_episode_id", sort=False)}
    for _, episode in manifest.iterrows():
        rows = episode_rows[episode["clinical_episode_id"]]
        if episode["n_source_dates"] > 1 or any(rows[c].nunique(dropna=True) > 1 for c in date_columns):
            records.append({"patient_id": episode["patient_id"], "clinical_episode_id": episode["clinical_episode_id"], "intervals": episode["intervals_involved"], "source_dates": episode["source_dates"], "source_row_ids": _display(rows["row_id_raw"]), "episode_span_days": episode["episode_span_days"], "date_discrepancy_detected": True, "possible_date_entry_error": episode["possible_date_entry_error"], "date_error_evidence": "same month/day in different years" if episode["possible_date_entry_error"] else "multiple source dates retained", "date_field_discrepancies": _display(c for c in date_columns if rows[c].nunique(dropna=True) > 1), "manual_review_required": episode["manual_review_required"]})
    return pd.DataFrame(records)


def build_source_value_provenance(
    assigned: pd.DataFrame,
    variables: Iterable[str] | None = None,
    conflict_keys: pd.DataFrame | Iterable[tuple[object, str]] | None = None,
) -> pd.DataFrame:
    """Return source values, optionally restricted to exact conflict keys."""
    keys: set[tuple[object, str]] | None = None
    if conflict_keys is not None:
        if isinstance(conflict_keys, pd.DataFrame):
            keys = (
                set(conflict_keys[["clinical_episode_id", "variable"]].itertuples(index=False, name=None))
                if {"clinical_episode_id", "variable"}.issubset(conflict_keys)
                else set()
            )
        else:
            keys = set(conflict_keys)
    clinical_columns = (
        sorted({variable for _, variable in keys})
        if keys is not None
        else (list(variables) if variables is not None else _data_columns(assigned))
    )
    output_columns = ["patient_id", "clinical_episode_id", "row_id_raw", "variable", "source_value", "source_date", "source_interval", "optional_cluster_id", "selected_value", "selection_rule"]
    if not clinical_columns:
        return pd.DataFrame(columns=output_columns)
    selected: dict[object, pd.Series] = {}
    selected_by_key: dict[tuple[object, str], object] = {}
    if isinstance(conflict_keys, pd.DataFrame) and {
        "clinical_episode_id", "variable", "selected_value"
    }.issubset(conflict_keys):
        selected_by_key = {
            (episode_id, variable): selected_value
            for episode_id, variable, selected_value in conflict_keys[
                ["clinical_episode_id", "variable", "selected_value"]
            ].itertuples(index=False, name=None)
        }
    if keys is None:
        selected = {
            episode_id: collapse_episode_rows(rows)
            for episode_id, rows in assigned.groupby("clinical_episode_id")
        }
    else:
        missing_selected = keys - set(selected_by_key)
        if missing_selected:
            episode_groups = {
                episode_id: rows
                for episode_id, rows in assigned.groupby("clinical_episode_id", sort=False)
            }
            for episode_id, variable in missing_selected:
                rows = episode_groups[episode_id].sort_values(
                    ["episode_precedence", "collection_date", "row_id_raw"],
                    na_position="last",
                )
                representative = rows["representative_interval"].iloc[0]
                primary = rows.loc[rows["interval_name"].eq(representative), variable]
                secondary = rows.loc[~rows["interval_name"].eq(representative), variable]
                selected_by_key[(episode_id, variable)] = resolve_preferred_value(
                    collapse_values(primary), collapse_values(secondary), representative, "secondary"
                )[0]
    working = assigned.reset_index(drop=True)
    value_parts: list[pd.DataFrame] = []
    for variable in clinical_columns:
        variable_rows = working
        if keys is not None:
            episode_ids = {episode_id for episode_id, key_variable in keys if key_variable == variable}
            variable_rows = working.loc[
                working["clinical_episode_id"].isin(episode_ids)
            ]
        informative = variable_rows.loc[has_information(variable_rows[variable])]
        if informative.empty:
            continue
        part = informative[
            ["patient_id", "clinical_episode_id", "row_id_raw", "collection_date", "interval_name", "optional_cluster_id"]
        ].copy()
        part["variable"] = variable
        part["source_value"] = informative[variable]
        value_parts.append(part)
    if not value_parts:
        return pd.DataFrame(columns=output_columns)
    # Avoid pandas comparing non-scalar DataFrame.attrs during concat.
    # These intermediate attrs are not part of the provenance table.
    for part in value_parts:
        part.attrs = {}
    result = pd.concat(value_parts, ignore_index=True)
    result = result.rename(columns={"collection_date": "source_date", "interval_name": "source_interval"})
    result["selected_value"] = [
        selected_by_key[(episode_id, variable)]
        if keys is not None
        else selected[episode_id].get(variable)
        for episode_id, variable in zip(result["clinical_episode_id"], result["variable"])
    ]
    result["selection_rule"] = "principal_precedence_preserve_all_sources"
    return result[output_columns]


def build_long_interval_spans(assigned: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Return every successfully merged episode spanning at least 365 days."""
    records = []
    episode_rows = {episode_id: rows for episode_id, rows in assigned.groupby("clinical_episode_id", sort=False)}
    for _, episode in manifest.loc[manifest["long_interval_span_warning"]].iterrows():
        rows = episode_rows[episode["clinical_episode_id"]]
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
    return pd.DataFrame([{ "n_raw_rows": len(assigned), "n_patients": assigned["patient_id"].nunique(), "n_episodes_before": len(assigned), "n_episodes_final": len(manifest), "n_merges_total": max(0, len(assigned)-len(manifest)), "n_same_interval_merges": int(assigned["merge_rule"].eq("same_normalized_interval_all_dates").sum()-manifest["assignment_rule"].str.contains("same_normalized_interval_all_dates").sum()), "n_15d_to_natural": int(assigned["optional_adjudication"].eq("attached_to_natural_15d").sum()), "n_optional_to_main": int(assigned["optional_adjudication"].eq("attached_to_main").sum()), "n_optional_clinical": int(manifest["visit_type"].eq("optional_independent_clinical").sum()), "n_optional_ambiguous": int(manifest["visit_type"].eq("optional_unresolved").sum()), "n_long_interval_warnings": int(manifest["long_interval_span_warning"].sum()), "raw_rows_unassigned": 0, "raw_rows_multiply_assigned": 0, "episode_assignment_wall_time_seconds": assigned.attrs.get("performance_metrics", {}).get("episode_assignment_seconds", pd.NA)}])


def write_parquet_and_csv(frame: pd.DataFrame, parquet_path: Path) -> tuple[Path, Path]:
    """Write a dataframe to Parquet and CSV without runtime attrs."""
    parquet_path.parent.mkdir(parents=True, exist_ok=True); csv_path = parquet_path.with_suffix(".csv"); serializable = frame.copy(deep=False); serializable.attrs = {}; serializable.to_parquet(parquet_path, index=False); serializable.to_csv(csv_path, index=False); return parquet_path, csv_path


def write_parquet(frame: pd.DataFrame, parquet_path: Path) -> Path:
    """Write a dataframe to Parquet without duplicating it as a wide CSV."""
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = frame.copy(deep=False)
    serializable.attrs = {}
    serializable.to_parquet(parquet_path, index=False)
    return parquet_path


def write_parquet_atomic(frame: pd.DataFrame, parquet_path: Path) -> Path:
    """Publish a Parquet file only after a structural read-back validation."""
    import pyarrow.parquet as pq

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = parquet_path.with_name(parquet_path.name + ".tmp.parquet")
    serializable = frame.copy(deep=False)
    serializable.attrs = {}
    try:
        serializable.to_parquet(temporary_path, index=False)
        parquet_file = pq.ParquetFile(temporary_path)
        if parquet_file.metadata.num_rows != len(serializable):
            raise RuntimeError(
                f"Parquet row count differs after serialization: {parquet_path.name}"
            )
        if parquet_file.schema_arrow.names != list(serializable.columns):
            raise RuntimeError(
                f"Parquet schema columns differ after serialization: {parquet_path.name}"
            )
        temporary_path.replace(parquet_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return parquet_path


def write_provenance_parquet(
    provenance: pd.DataFrame, provenance_path: Path
) -> pd.DataFrame:
    """Write audit values as nullable text while retaining other column dtypes."""
    serializable = provenance.copy()
    for column in ("source_value", "selected_value"):
        if column in serializable:
            serializable[column] = serializable[column].astype("string")
    write_parquet_atomic(serializable, provenance_path)
    return serializable


def write_metrics_checkpoint(
    metrics: dict[str, object], path: Path, status: str, stage: str
) -> None:
    """Atomically publish non-clinical performance and run-state metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**metrics, "status": status, "current_stage": stage}
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    temporary_path.replace(path)


def parquet_artifact_metadata(path: Path, rows: int) -> dict[str, object]:
    """Return non-clinical publication metadata for a core Parquet artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.stat()
    return {
        "rows": rows,
        "sha256": digest.hexdigest(),
        "mtime_ns": stat.st_mtime_ns,
        "size_bytes": stat.st_size,
    }


def main() -> None:
    """Build frozen episode assignments and all source-preserving QC products."""
    args = parse_args()
    logger = setup_logger("08c_build_clinical_episode_map")
    total_started = perf_counter()
    metrics_path = args.qc_dir / PERFORMANCE_FILENAME
    metrics: dict[str, object] = {
        "qc_mode": args.qc_mode,
        "standard_provenance_seconds": 0.0,
        "full_provenance_seconds": 0.0,
    }

    def checkpoint(stage: str, status: str = "running") -> None:
        metrics["elapsed_seconds"] = perf_counter() - total_started
        write_metrics_checkpoint(metrics, metrics_path, status, stage)

    try:
        checkpoint("read_input")
        started = perf_counter()
        logger.info("[08c] INPUT START | path=%s", args.input_path)
        raw = pd.read_parquet(args.input_path)
        metrics["read_seconds"] = perf_counter() - started
        logger.info("[08c] INPUT DONE | rows=%d | seconds=%.2f", len(raw), metrics["read_seconds"])

        started = perf_counter()
        logger.info("[08c] ASSIGNMENT START | matching progress follows")
        source, provenance_columns = prepare_visits(raw)
        flagged = add_presence_flags(source)
        clinical_columns = _data_columns(flagged)
        units = build_atomic_activity_units(flagged, provenance_columns)
        metrics.update({
            "presence_masks_seconds": perf_counter() - started,
            "raw_rows": len(source),
            "n_rows": len(source),
            "n_columns": len(source.columns),
            "n_patients": source["patient_id"].nunique(),
            "n_optional": int(source["interval_normalized"].map(_is_optional).sum()),
            "clinical_columns": len(clinical_columns),
        })
        assigned_units = assign_episodes(
            units, include_pair_records=args.qc_mode == "full"
        )
        assigned = propagate_episode_assignments(flagged, assigned_units)
        unassigned, multiplied = validate_final_assignments(source, assigned)
        if args.qc_mode == "full":
            assigned.attrs["optional_pair_candidates"] = build_optional_pair_qc(
                assigned, assigned.attrs["optional_pair_candidates"]
            )
        metrics.update(assigned.attrs["performance_metrics"])
        metrics.update({"unassigned": unassigned, "multiplied": multiplied})
        logger.info(
            "[08c] ASSIGNMENT DONE | patients=%d | rows=%d | seconds=%.2f",
            source["patient_id"].nunique(), len(assigned),
            metrics["episode_assignment_seconds"],
        )
        checkpoint("assignment_complete")

        started = perf_counter()
        logger.info("[08c] MANIFEST START")
        manifest = build_manifest(assigned)
        metrics.update({"manifest_seconds": perf_counter() - started, "episodes": len(manifest)})
        logger.info("[08c] MANIFEST DONE | episodes=%d | seconds=%.2f", len(manifest), metrics["manifest_seconds"])
        checkpoint("manifest_complete")

        started = perf_counter()
        logger.info("[08c] CORE PARQUET START | QC remains pending")
        write_parquet_atomic(assigned, args.row_map_path)
        write_parquet_atomic(manifest, args.manifest_path)
        metrics["core_parquet_write_seconds"] = perf_counter() - started
        metrics["core_artifacts"] = {
            "row_map": parquet_artifact_metadata(args.row_map_path, len(assigned)),
            "manifest": parquet_artifact_metadata(args.manifest_path, len(manifest)),
        }
        metrics["row_map_csv_write_seconds"] = 0.0
        logger.info(
            "[08c] CORE PARQUET DONE | rowmap=%d | manifest=%d | rowmap_sha256=%s | manifest_sha256=%s | seconds=%.2f | QC=pending",
            len(assigned),
            len(manifest),
            metrics["core_artifacts"]["row_map"]["sha256"],
            metrics["core_artifacts"]["manifest"]["sha256"],
            metrics["core_parquet_write_seconds"],
        )
        checkpoint("core_parquet_complete_qc_pending")

        started = perf_counter()
        logger.info("[08c] CONFLICT DETECTION START")
        conflicts = build_value_conflicts(assigned)
        metrics.update({
            "conflict_detection_seconds": perf_counter() - started,
            "conflict_details_seconds": 0.0,
            "conflicting_variables": conflicts["variable"].nunique(),
            "conflict_keys": len(conflicts),
        })
        logger.info("[08c] CONFLICT DETECTION DONE | keys=%d | variables=%d | seconds=%.2f", len(conflicts), metrics["conflicting_variables"], metrics["conflict_detection_seconds"])
        checkpoint("conflict_detection_complete")

        started = perf_counter()
        logger.info("[08c] TEMPORAL QC START")
        dates = build_date_discrepancies(assigned, manifest)
        long_spans = build_long_interval_spans(assigned, manifest)
        metrics.update({"date_qc_seconds": perf_counter() - started, "long_spans": len(long_spans)})
        logger.info("[08c] TEMPORAL QC DONE | discrepancies=%d | long_spans=%d | seconds=%.2f", len(dates), len(long_spans), metrics["date_qc_seconds"])
        checkpoint("temporal_qc_complete")

        started = perf_counter()
        logger.info("[08c] PROVENANCE START | qc_mode=%s | conflict_keys=%d", args.qc_mode, len(conflicts))
        provenance = (
            build_source_value_provenance(assigned)
            if args.qc_mode == "full"
            else build_source_value_provenance(assigned, conflict_keys=conflicts)
        )
        provenance_metric = "full_provenance_seconds" if args.qc_mode == "full" else "standard_provenance_seconds"
        metrics.update({provenance_metric: perf_counter() - started, "informative_cells": len(provenance), "provenance_rows": len(provenance)})
        logger.info("[08c] PROVENANCE DONE | informative_cells=%d | seconds=%.2f", len(provenance), metrics[provenance_metric])
        checkpoint("provenance_built")

        audit = assigned.attrs["merge_decision_audit"]
        summary = build_merge_summary(assigned, manifest, audit)
        if args.write_core_csv:
            started = perf_counter()
            assigned.to_csv(args.row_map_path.with_suffix(".csv"), index=False)
            manifest.to_csv(args.manifest_path.with_suffix(".csv"), index=False)
            metrics["row_map_csv_write_seconds"] = perf_counter() - started

        args.qc_dir.mkdir(parents=True, exist_ok=True)
        outputs = {
            MERGE_AUDIT_FILENAME: audit,
            VALUE_CONFLICTS_FILENAME: conflicts,
            MERGE_INCOMPATIBILITIES_FILENAME: assigned.attrs["merge_incompatibilities"],
            MERGE_SUMMARY_FILENAME: summary,
            LONG_SPANS_FILENAME: long_spans,
            OPTIONAL_PAIRS_FILENAME: assigned.attrs["optional_pair_candidates"],
            OPTIONAL_CLUSTERS_FILENAME: assigned.attrs["optional_clusters"],
            OPTIONAL_ADJUDICATION_FILENAME: assigned.attrs["optional_adjudication"],
            UNRESOLVED_FILENAME: assigned.attrs["unresolved_assignments"],
            DATE_DISCREPANCIES_FILENAME: dates,
        }
        started = perf_counter()
        logger.info("[08c] QC CSV START")
        for filename, frame in outputs.items():
            frame.to_csv(args.qc_dir / filename, index=False)
        metrics["qc_write_seconds"] = perf_counter() - started
        logger.info("[08c] QC CSV DONE | files=%d | seconds=%.2f", len(outputs), metrics["qc_write_seconds"])
        checkpoint("qc_csv_complete")

        provenance_path = args.qc_dir / PROVENANCE_FILENAME
        started = perf_counter()
        logger.info("[08c] PROVENANCE PARQUET START | value_dtype=string")
        provenance = write_provenance_parquet(provenance, provenance_path)
        metrics["provenance_parquet_write_seconds"] = perf_counter() - started
        logger.info("[08c] PROVENANCE PARQUET DONE | rows=%d | value_dtype=string | seconds=%.2f", len(provenance), metrics["provenance_parquet_write_seconds"])
        if args.export_full_provenance_csv:
            started = perf_counter()
            provenance.to_csv(provenance_path.with_suffix(".csv"), index=False)
            metrics["provenance_csv_write_seconds"] = perf_counter() - started

        produced_paths = [args.row_map_path, args.manifest_path, provenance_path]
        produced_paths.extend(args.qc_dir / filename for filename in outputs)
        if args.write_core_csv:
            produced_paths.extend((args.row_map_path.with_suffix(".csv"), args.manifest_path.with_suffix(".csv")))
        metrics["output_sizes_mb"] = {
            path.name: round(path.stat().st_size / 1_048_576, 3)
            for path in produced_paths if path.exists()
        }
        metrics["files_produced"] = sorted(metrics["output_sizes_mb"])
        metrics["total_wall_time_seconds"] = perf_counter() - total_started
        metrics["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 3)
        checkpoint("complete", status="complete")
        if args.profile:
            logger.info("Performance metrics: %s", json.dumps(metrics, default=str))
        logger.info("[08c] COMPLETE | raw_rows=%d | episodes=%d | conflicts=%d | seconds=%.2f", len(assigned), len(manifest), len(conflicts), metrics["total_wall_time_seconds"])
    except Exception as error:
        metrics["error_type"] = type(error).__name__
        metrics["error"] = str(error)
        metrics["total_wall_time_seconds"] = perf_counter() - total_started
        checkpoint("failed", status="failed")
        logger.exception("[08c] FAILED | error_type=%s", type(error).__name__)
        raise


if __name__ == "__main__":
    main()
