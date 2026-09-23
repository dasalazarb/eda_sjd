"""Build a clinical-episode row map using two simple, auditable passes.

PASS 1 joins rows from the same patient, calendar year, and exact source
interval. PASS 2 joins remaining, complementary and clinically compatible
episodes no more than 30 days apart. Calendar years are an absolute boundary.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from time import perf_counter
from typing import Iterable

import pandas as pd
from tqdm import tqdm

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
VALUE_CONFLICTS_FILENAME = "08c_value_conflicts.csv"
MERGE_INCOMPATIBILITIES_FILENAME = "08c_merge_incompatibilities.csv"
MERGE_SUMMARY_FILENAME = "08c_merge_summary.csv"

NATURAL_HISTORY = "natural history protocol 478 interval"
PHASE_ORDER = {
    "Phase 1: Initial Full Evaluation": 1,
    "Phase 1: Second Full Evaluation": 2,
    "Phase 1: Final Full (Third Full) Evaluation": 3,
    "Phase 2: 4th Full Evaluation": 4,
    "Phase 2: 5th Full Evaluation": 5,
}
NORMALIZED_PHASE_ORDER = {
    _name.casefold(): order for _name, order in PHASE_ORDER.items()
}
PHASE_INTERVALS = set(NORMALIZED_PHASE_ORDER)
AUDIT_COLUMNS = [
    "patient_id",
    "episode_a",
    "episode_b",
    "interval_a",
    "interval_b",
    "date_a",
    "date_b",
    "year_a",
    "year_b",
    "days_apart",
    "same_exact_interval",
    "merge_rule",
    "primary_interval",
    "secondary_interval",
    "merged",
    "reason",
]
CONFLICT_COLUMNS = [
    "patient_id",
    "clinical_episode_id",
    "variable",
    "preferred_value",
    "secondary_value",
    "chosen_value",
    "preferred_interval",
    "secondary_interval",
    "preferred_date",
    "secondary_date",
    "days_apart",
    "merge_rule",
    "conflict_resolution",
]
INCOMPATIBILITY_COLUMNS = [
    "patient_id",
    "episode_a",
    "episode_b",
    "interval_a",
    "interval_b",
    "date_a",
    "date_b",
    "days_apart",
    "variable",
    "value_a",
    "value_b",
    "decision",
    "reason",
]
METADATA_COLUMNS = {
    "patient_id",
    "row_id_raw",
    "interval_name",
    "collection_date",
    "collection_year",
    "clinical_episode_id",
    "_source_order",
    "atomic_activity_unit_id",
    "daily_activity_unit_id",
    "row_ids_involved",
    "interval_names_involved",
    "assignment_rule",
    "merge_stage",
    "merge_rule",
    "manual_review_required",
    "manual_review_reason",
    "representative_interval",
    "representative_date",
    "episode_precedence",
    "source_file",
}
MISSING_UPPER = {str(value).strip().upper() for value in MISSING_TOKENS}


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


def _normalized_interval(value: object) -> str:
    """Normalize an interval only for comparison, never for output."""
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip().casefold())


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


def resolve_preferred_value(
    preferred_value: object,
    secondary_value: object,
    preferred_source: object,
    secondary_source: object,
) -> tuple[object, bool]:
    """Resolve two values in favor of the clinically preferred source.

    Returns the selected value and whether two populated, unequal values conflict.
    Source arguments make call sites explicit and are intentionally not interpreted.
    """
    del preferred_source, secondary_source
    preferred = _unique_values([preferred_value])
    secondary = _unique_values([secondary_value])
    if not preferred:
        return (secondary[0] if secondary else pd.NA), False
    if not secondary:
        return preferred[0], False
    conflict = str(preferred[0]).strip() != str(secondary[0]).strip()
    return preferred[0], conflict


def intervals_are_compatible(interval_a: object, interval_b: object) -> bool:
    """Return whether two original interval labels are PASS-1 compatible.

    Exact labels match. Natural History matches 15D Optional, and Optional
    Evaluation matches one of the five named Phase full evaluations, and two
    named Phase intervals are clinically compatible (with order determining
    precedence during PASS 2).
    """
    left, right = _normalized_interval(interval_a), _normalized_interval(interval_b)
    if not left or not right:
        return left == right
    if left == right:
        return True
    if (left == NATURAL_HISTORY and _is_15d_optional(right)) or (
        right == NATURAL_HISTORY and _is_15d_optional(left)
    ):
        return True
    if left in PHASE_INTERVALS and right in PHASE_INTERVALS:
        return True
    return (_is_optional(left) and right in PHASE_INTERVALS) or (
        _is_optional(right) and left in PHASE_INTERVALS
    )


def has_information(series: pd.Series) -> pd.Series:
    """Return whether values are populated, retaining zero and negative answers."""
    result = series.notna()
    if pd.api.types.is_object_dtype(series.dtype) or pd.api.types.is_string_dtype(
        series.dtype
    ):
        text = series.astype("string").str.strip()
        result &= text.notna() & ~text.str.upper().isin(MISSING_UPPER)
    return result.fillna(False)


def _unique_values(values: Iterable[object]) -> list[object]:
    unique: list[object] = []
    seen: set[str] = set()
    for value in values:
        if pd.isna(value):
            continue
        if isinstance(value, str) and (
            not value.strip() or value.strip().upper() in MISSING_UPPER
        ):
            continue
        key = str(value).strip()
        if key not in seen:
            unique.append(value)
            seen.add(key)
    return unique


def collapse_values(values: Iterable[object]) -> object:
    """Collapse nonmissing unique values, preserving conflicts with `` | ``."""
    unique = _unique_values(values)
    if not unique:
        return pd.NA
    if len(unique) == 1:
        return unique[0]
    return " | ".join(str(value).strip() for value in unique)


def prepare_visits(visits: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Create required working columns while preserving source values."""
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
    result["collection_year"] = result["collection_date"].dt.year.astype("Int64")
    result["row_id_raw"] = result[row_col] if row_col else result["_source_order"]
    if result["patient_id"].isna().any():
        raise ValueError("patient_id contains missing values")
    if result["row_id_raw"].isna().any() or result["row_id_raw"].duplicated().any():
        raise ValueError("row_id_raw must be complete and unique")
    provenance = [
        str(column)
        for column in visits
        if any(term in str(column).lower() for term in ("protocol", "origin", "source"))
    ]
    return result, list(dict.fromkeys(provenance))


def add_presence_flags(visits: pd.DataFrame) -> pd.DataFrame:
    """Return rows unchanged; evidence is evaluated directly from source columns."""
    return visits.copy()


def build_atomic_activity_units(
    flagged_rows: pd.DataFrame, provenance_columns: Iterable[str] = ()
) -> pd.DataFrame:
    """Create exactly one immutable activity unit per raw source row."""
    units = flagged_rows.copy()
    units["atomic_activity_unit_id"] = units["row_id_raw"].map(
        lambda value: f"row-{value}"
    )
    units["daily_activity_unit_id"] = units["atomic_activity_unit_id"]
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
    """Backward-compatible alias that performs no same-day pre-consolidation."""
    return build_atomic_activity_units(flagged_rows, provenance_columns)


def _episode_date(rows: pd.DataFrame) -> pd.Timestamp:
    if "representative_date" in rows and rows["representative_date"].notna().any():
        return rows.loc[
            rows["representative_date"].notna(), "representative_date"
        ].iloc[0]
    dates = rows["collection_date"].dropna()
    return dates.min() if not dates.empty else pd.NaT


def _episode_intervals(rows: pd.DataFrame) -> list[object]:
    return list(dict.fromkeys(rows["interval_name"].tolist()))


def _episodes_compatible(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    """Require every cross-pair to match, preventing Optional from bridging Phases."""
    return all(
        intervals_are_compatible(a, b)
        for a in _episode_intervals(left)
        for b in _episode_intervals(right)
    )


def _episodes_have_same_exact_interval(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    """Return whether every row has the same literal original interval."""
    intervals = {
        "<MISSING>" if pd.isna(value) else str(value)
        for value in [*_episode_intervals(left), *_episode_intervals(right)]
    }
    return len(intervals) == 1


def _data_columns(rows: pd.DataFrame) -> list[str]:
    generated_prefixes = ("has_",)
    return [
        str(column)
        for column in rows.columns
        if str(column).split("__")[-1] not in METADATA_COLUMNS
        and not str(column).endswith("_involved")
        and not str(column).startswith(generated_prefixes)
    ]


def episodes_are_complementary(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    """Return whether either episode supplies a variable missing in the other."""
    for column in set(_data_columns(left)) & set(_data_columns(right)):
        left_has = bool(has_information(left[column]).any())
        right_has = bool(has_information(right[column]).any())
        if left_has != right_has:
            return True
    return False


def find_incompatible_variables(
    record_a: pd.DataFrame, record_b: pd.DataFrame
) -> list[dict[str, object]]:
    """Find conflicting populated clinical values between two episodes.

    Parameters
    ----------
    record_a, record_b : pd.DataFrame
        Candidate episode rows. Technical and provenance columns are excluded.

    Returns
    -------
    list of dict
        One item per shared variable whose populated values differ.
    """
    conflicts: list[dict[str, object]] = []
    columns = sorted(set(_data_columns(record_a)) & set(_data_columns(record_b)))
    for column in columns:
        values_a = _unique_values(record_a[column])
        values_b = _unique_values(record_b[column])
        if not values_a or not values_b:
            continue
        keys_a = {str(value).strip() for value in values_a}
        keys_b = {str(value).strip() for value in values_b}
        if keys_a != keys_b:
            conflicts.append(
                {
                    "variable": column,
                    "value_a": collapse_values(values_a),
                    "value_b": collapse_values(values_b),
                }
            )
    return conflicts


def _pair_rule(left: pd.DataFrame, right: pd.DataFrame) -> tuple[str, int, int] | None:
    """Return the clinical merge rule and preferred/secondary side indexes."""
    left_kind = get_episode_priority(left["representative_interval"].iloc[0])
    right_kind = get_episode_priority(right["representative_interval"].iloc[0])
    kinds = {left_kind, right_kind}
    if kinds == {"natural", "15d_optional"}:
        return (
            "natural_15d_within_30_days",
            (0 if left_kind == "natural" else 1),
            (1 if left_kind == "natural" else 0),
        )
    left_phase, right_phase = left_kind.startswith("phase_"), right_kind.startswith(
        "phase_"
    )
    if left_phase and right_phase:
        left_order, right_order = int(left_kind[6:]), int(right_kind[6:])
        primary = 0 if left_order <= right_order else 1
        return "phase_phase_within_30_days", primary, 1 - primary
    if (left_phase and right_kind == "optional") or (
        right_phase and left_kind == "optional"
    ):
        primary = 0 if left_phase else 1
        return "phase_optional_within_30_days", primary, 1 - primary
    return None


def _display(values: Iterable[object]) -> str:
    return " | ".join(str(value) for value in dict.fromkeys(values) if pd.notna(value))


def _audit_record(
    patient_id: object,
    left: pd.DataFrame,
    right: pd.DataFrame,
    rule: str,
    merged: bool,
    primary: pd.DataFrame | None = None,
    secondary: pd.DataFrame | None = None,
    reason: str = "",
) -> dict[str, object]:
    date_a, date_b = _episode_date(left), _episode_date(right)
    return {
        "patient_id": patient_id,
        "episode_a": _display(left["row_id_raw"]),
        "episode_b": _display(right["row_id_raw"]),
        "interval_a": _display(left["interval_name"]),
        "interval_b": _display(right["interval_name"]),
        "date_a": date_a,
        "date_b": date_b,
        "year_a": date_a.year if pd.notna(date_a) else pd.NA,
        "year_b": date_b.year if pd.notna(date_b) else pd.NA,
        "days_apart": (
            abs((date_b - date_a).days)
            if pd.notna(date_a) and pd.notna(date_b)
            else pd.NA
        ),
        "same_exact_interval": _episodes_have_same_exact_interval(left, right),
        "merge_rule": rule,
        "primary_interval": (
            _display(primary["interval_name"]) if primary is not None else ""
        ),
        "secondary_interval": (
            _display(secondary["interval_name"]) if secondary is not None else ""
        ),
        "merged": merged,
        "reason": reason or ("merged" if merged else "not_eligible"),
    }


def _merge(
    primary: pd.DataFrame, secondary: pd.DataFrame, stage: str, rule: str
) -> pd.DataFrame:
    """Merge row membership while retaining explicit episode precedence."""
    rows = (
        pd.concat([primary, secondary])
        .sort_values(["collection_date", "_source_order"])
        .copy()
    )
    representative_interval = primary["representative_interval"].iloc[0]
    representative_date = primary["representative_date"].iloc[0]
    rows["merge_stage"] = stage
    rows["merge_rule"] = rule
    rows["assignment_rule"] = rule
    rows["representative_interval"] = representative_interval
    rows["representative_date"] = representative_date
    # Existing primary rows remain ahead of every secondary row. This supports
    # deterministic hierarchical collapse even after multiple merges.
    primary_ids = set(primary["row_id_raw"])
    base = int(rows["episode_precedence"].max()) + 1
    rows.loc[~rows["row_id_raw"].isin(primary_ids), "episode_precedence"] += base
    return rows


def _pass_one(
    patient_id: object, year_rows: pd.DataFrame, audit: list[dict[str, object]]
) -> list[pd.DataFrame]:
    episodes: list[pd.DataFrame] = []
    for _, exact_rows in year_rows.groupby("interval_name", sort=False, dropna=False):
        exact_rows = exact_rows.sort_values(["collection_date", "_source_order"]).copy()
        primary = exact_rows.iloc[[0]].copy()
        for position in range(1, len(exact_rows)):
            secondary = exact_rows.iloc[[position]].copy()
            audit.append(
                _audit_record(
                    patient_id,
                    primary,
                    secondary,
                    "same_exact_interval_same_year",
                    True,
                    primary,
                    secondary,
                )
            )
            primary = _merge(
                primary,
                secondary,
                "interval_same_year",
                "same_exact_interval_same_year",
            )
        episodes.append(primary)
    return episodes


def _pass_two(
    patient_id: object,
    episodes: list[pd.DataFrame],
    audit: list[dict[str, object]],
    incompatibilities: list[dict[str, object]],
) -> list[pd.DataFrame]:
    changed = True
    while changed:
        changed = False
        episodes.sort(
            key=lambda rows: (_episode_date(rows), int(rows["_source_order"].min()))
        )
        for i in range(len(episodes)):
            date_i = _episode_date(episodes[i])
            if pd.isna(date_i):
                continue
            for j in range(i + 1, len(episodes)):
                date_j = _episode_date(episodes[j])
                if pd.isna(date_j):
                    continue
                days_apart = (date_j - date_i).days
                if days_apart > 30:
                    audit.append(
                        _audit_record(
                            patient_id,
                            episodes[i],
                            episodes[j],
                            "different_interval_gt30_days_no_merge",
                            False,
                            reason="different intervals more than 30 days apart",
                        )
                    )
                    break

                known = _pair_rule(episodes[i], episodes[j])
                conflicts = find_incompatible_variables(episodes[i], episodes[j])
                if known:
                    rule, primary_index, secondary_index = known
                    pair = [episodes[i], episodes[j]]
                    primary, secondary = pair[primary_index], pair[secondary_index]
                    merge = True
                    reason = "clinical priority merge"
                else:
                    complementary = episodes_are_complementary(episodes[i], episodes[j])
                    merge = complementary and not conflicts
                    rule = (
                        "other_temporal_rescue"
                        if merge
                        else "other_interval_conflict_no_merge"
                    )
                    primary, secondary = episodes[i], episodes[j]
                    reason = (
                        "complementary without incompatibilities"
                        if merge
                        else "not complementary or conflicting values"
                    )
                audit.append(
                    _audit_record(
                        patient_id,
                        episodes[i],
                        episodes[j],
                        rule,
                        merge,
                        primary,
                        secondary,
                        reason,
                    )
                )
                if not merge:
                    for conflict in conflicts:
                        incompatibilities.append(
                            {
                                "patient_id": patient_id,
                                "episode_a": _display(episodes[i]["row_id_raw"]),
                                "episode_b": _display(episodes[j]["row_id_raw"]),
                                "interval_a": _display(episodes[i]["interval_name"]),
                                "interval_b": _display(episodes[j]["interval_name"]),
                                "date_a": date_i,
                                "date_b": date_j,
                                "days_apart": days_apart,
                                **conflict,
                                "decision": "no_merge",
                                "reason": "different_interval_value_conflict",
                            }
                        )
                    continue
                episodes[i] = _merge(primary, secondary, "temporal_rescue", rule)
                episodes.pop(j)
                changed = True
                break
            if changed:
                break
    return episodes


def assign_episodes(atomic_units: pd.DataFrame) -> pd.DataFrame:
    """Assign raw rows using exact-interval then clinically prioritized passes."""
    started = perf_counter()
    prepared = atomic_units.copy()
    for column, default in {
        "assignment_rule": "standalone_record",
        "merge_stage": "standalone",
        "merge_rule": "standalone_record",
        "manual_review_required": False,
        "manual_review_reason": "",
    }.items():
        prepared[column] = default
    prepared["representative_interval"] = prepared["interval_name"]
    prepared["representative_date"] = prepared["collection_date"]
    prepared["episode_precedence"] = 0
    assigned: list[pd.DataFrame] = []
    audit: list[dict[str, object]] = []
    incompatibilities: list[dict[str, object]] = []
    grouped = prepared.groupby("patient_id", sort=False, dropna=False)
    for patient_id, patient_rows in tqdm(
        grouped,
        total=prepared["patient_id"].nunique(dropna=False),
        desc="Building clinical episodes",
    ):
        episodes: list[pd.DataFrame] = []
        for _, year_rows in patient_rows.groupby(
            "collection_year", sort=True, dropna=False
        ):
            episodes.extend(
                _pass_two(
                    patient_id,
                    _pass_one(patient_id, year_rows, audit),
                    audit,
                    incompatibilities,
                )
            )
        for i in range(len(episodes)):
            for j in range(i + 1, len(episodes)):
                left_date, right_date = _episode_date(episodes[i]), _episode_date(
                    episodes[j]
                )
                if (
                    pd.notna(left_date)
                    and pd.notna(right_date)
                    and left_date.year != right_date.year
                ):
                    audit.append(
                        _audit_record(
                            patient_id,
                            episodes[i],
                            episodes[j],
                            "different_year_no_merge",
                            False,
                            reason="calendar years differ",
                        )
                    )
        episodes.sort(
            key=lambda rows: (_episode_date(rows), int(rows["_source_order"].min()))
        )
        for sequence, episode in enumerate(episodes, 1):
            episode = episode.copy()
            episode["clinical_episode_id"] = f"{patient_id}_EP{sequence:03d}"
            assigned.append(episode)
    result = pd.concat(assigned).sort_values("_source_order") if assigned else prepared
    decision_audit = pd.DataFrame(audit, columns=AUDIT_COLUMNS)
    result.attrs["merge_decision_audit"] = decision_audit
    result.attrs["merge_incompatibilities"] = pd.DataFrame(
        incompatibilities, columns=INCOMPATIBILITY_COLUMNS
    )
    result.attrs["performance_metrics"] = {
        "wall_time_seconds": perf_counter() - started,
        "n_candidates_total": len(decision_audit),
        "n_merges_total": (
            int(decision_audit["merged"].sum()) if len(decision_audit) else 0
        ),
    }
    return result


def propagate_episode_assignments(
    flagged_rows: pd.DataFrame, assigned_units: pd.DataFrame
) -> pd.DataFrame:
    """Propagate episode metadata without modifying original source values."""
    columns = [
        "row_id_raw",
        "clinical_episode_id",
        "assignment_rule",
        "merge_stage",
        "merge_rule",
        "manual_review_required",
        "manual_review_reason",
        "representative_interval",
        "representative_date",
        "episode_precedence",
    ]
    result = flagged_rows.merge(
        assigned_units[columns], on="row_id_raw", how="left", validate="one_to_one"
    ).sort_values("_source_order")
    result.attrs.update(assigned_units.attrs)
    return result


def build_manifest(
    assigned: pd.DataFrame, source_intervals: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Build the episode manifest required by downstream collapse steps."""
    del source_intervals
    records: list[dict[str, object]] = []
    for (patient_id, episode_id), rows in assigned.groupby(
        ["patient_id", "clinical_episode_id"], sort=False
    ):
        dates = rows["collection_date"].dropna()
        start, end = (dates.min(), dates.max()) if not dates.empty else (pd.NaT, pd.NaT)
        representative_interval = rows["representative_interval"].iloc[0]
        representative_date = rows["representative_date"].iloc[0]
        records.append(
            {
                "patient_id": patient_id,
                "clinical_episode_id": episode_id,
                "intervals_involved": _display(rows["interval_name"]),
                "representative_interval": representative_interval,
                "episode_start_date": start,
                "clinical_anchor_date": representative_date,
                "episode_end_date": end,
                "episode_span_days": (end - start).days if pd.notna(start) else pd.NA,
                "visit_type": "clinical_episode",
                "clinical_visit": True,
                "manual_review_required": False,
                "manual_review_reason": "",
                "assignment_rule": _display(rows["assignment_rule"]),
                "merge_stage": _display(rows["merge_stage"]),
                "cross_interval_merge": rows["interval_name"].nunique(dropna=False) > 1,
                "cross_year_merge": False,
                "suspected_date_error": False,
                "interval_order_anomaly": False,
                "exceptional_merge": False,
            }
        )
    return pd.DataFrame(records)


def collapse_episode_rows(rows: pd.DataFrame) -> pd.Series:
    """Collapse an episode, honoring representative-interval precedence."""
    ordered = rows.sort_values(
        ["episode_precedence", "collection_date", "_source_order"]
    )
    representative = ordered["representative_interval"].iloc[0]
    primary = ordered.loc[ordered["interval_name"].eq(representative)]
    secondary = ordered.loc[~ordered.index.isin(primary.index)]
    collapsed: dict[str, object] = {}
    for column in _data_columns(rows):
        preferred = collapse_values(primary[column])
        fallback = collapse_values(secondary[column])
        collapsed[column], _ = resolve_preferred_value(
            preferred, fallback, representative, _display(secondary["interval_name"])
        )
    return pd.Series(collapsed)


def build_value_conflicts(assigned: pd.DataFrame) -> pd.DataFrame:
    """Return one traceable QC row per same-source or hierarchical conflict."""
    records: list[dict[str, object]] = []
    for (patient_id, episode_id), rows in assigned.groupby(
        ["patient_id", "clinical_episode_id"], sort=False
    ):
        if len(rows) < 2:
            continue
        representative = rows["representative_interval"].iloc[0]
        preferred_rows = rows.loc[rows["interval_name"].eq(representative)]
        secondary_rows = rows.loc[~rows.index.isin(preferred_rows.index)]
        for column in _data_columns(rows):
            preferred = collapse_values(preferred_rows[column])
            secondary = collapse_values(secondary_rows[column])
            chosen, hierarchical_conflict = resolve_preferred_value(
                preferred,
                secondary,
                representative,
                _display(secondary_rows["interval_name"]),
            )
            same_interval_conflict = len(_unique_values(preferred_rows[column])) > 1
            if not hierarchical_conflict and not same_interval_conflict:
                continue
            secondary_interval = _display(secondary_rows["interval_name"])
            preferred_date = preferred_rows["collection_date"].min()
            secondary_date = secondary_rows["collection_date"].min()
            records.append(
                {
                    "patient_id": patient_id,
                    "clinical_episode_id": episode_id,
                    "variable": column,
                    "preferred_value": preferred,
                    "secondary_value": secondary,
                    "chosen_value": chosen,
                    "preferred_interval": representative,
                    "secondary_interval": secondary_interval or representative,
                    "preferred_date": preferred_date,
                    "secondary_date": (
                        secondary_date
                        if pd.notna(secondary_date)
                        else preferred_rows["collection_date"].max()
                    ),
                    "days_apart": (
                        abs((secondary_date - preferred_date).days)
                        if pd.notna(secondary_date) and pd.notna(preferred_date)
                        else 0
                    ),
                    "merge_rule": rows["merge_rule"].iloc[0],
                    "conflict_resolution": (
                        "preserved_all_same_interval_values"
                        if same_interval_conflict and not hierarchical_conflict
                        else "preferred_primary_visit"
                    ),
                }
            )
    return pd.DataFrame(records, columns=CONFLICT_COLUMNS)


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
    assert assigned["clinical_episode_id"].notna().all()
    return unassigned, multiplied


def build_merge_summary(
    assigned: pd.DataFrame, manifest: pd.DataFrame, audit: pd.DataFrame
) -> pd.DataFrame:
    """Build a concise merge and conservation summary."""
    merged = (
        audit["merged"].fillna(False) if "merged" in audit else pd.Series(dtype=bool)
    )
    return pd.DataFrame(
        [
            {
                "n_raw_rows": len(assigned),
                "n_patients": assigned["patient_id"].nunique(),
                "n_episodes_final": manifest["clinical_episode_id"].nunique(),
                "n_candidates_total": len(audit),
                "n_merges_total": int(merged.sum()),
                "n_pass1_merges": (
                    int(
                        (
                            merged
                            & audit["merge_rule"].eq("same_exact_interval_same_year")
                        ).sum()
                    )
                    if len(audit)
                    else 0
                ),
                "n_pass2_temporal_rescue_merges": (
                    int(
                        (
                            merged
                            & ~audit["merge_rule"].eq("same_exact_interval_same_year")
                        ).sum()
                    )
                    if len(audit)
                    else 0
                ),
                "raw_rows_unassigned": 0,
                "raw_rows_multiply_assigned": 0,
                "episode_assignment_wall_time_seconds": assigned.attrs.get(
                    "performance_metrics", {}
                ).get("wall_time_seconds", pd.NA),
            }
        ]
    )


def write_parquet_and_csv(frame: pd.DataFrame, parquet_path: Path) -> tuple[Path, Path]:
    """Write a dataframe to Parquet and CSV without runtime attrs."""
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = parquet_path.with_suffix(".csv")
    serializable = frame.copy(deep=False)
    serializable.attrs = {}
    serializable.to_parquet(parquet_path, index=False)
    serializable.to_csv(csv_path, index=False)
    return parquet_path, csv_path


def main() -> None:
    """Build row map, manifest, decision audit, conflicts, and summary."""
    args = parse_args()
    logger = setup_logger("08c_build_clinical_episode_map")
    logger.info("Reading %s", args.input_path)
    source, provenance = prepare_visits(pd.read_parquet(args.input_path))
    flagged = add_presence_flags(source)
    units = build_atomic_activity_units(flagged, provenance)
    assigned_units = assign_episodes(units)
    audit = assigned_units.attrs["merge_decision_audit"]
    incompatibilities = assigned_units.attrs["merge_incompatibilities"]
    assigned = propagate_episode_assignments(flagged, assigned_units)
    validate_final_assignments(source, assigned)
    manifest = build_manifest(assigned)
    conflicts = build_value_conflicts(assigned)
    summary = build_merge_summary(assigned, manifest, audit)
    write_parquet_and_csv(assigned, args.row_map_path)
    write_parquet_and_csv(manifest, args.manifest_path)
    args.qc_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.qc_dir / MERGE_AUDIT_FILENAME, index=False)
    conflicts.to_csv(args.qc_dir / VALUE_CONFLICTS_FILENAME, index=False)
    incompatibilities.to_csv(
        args.qc_dir / MERGE_INCOMPATIBILITIES_FILENAME, index=False
    )
    summary.to_csv(args.qc_dir / MERGE_SUMMARY_FILENAME, index=False)
    logger.info(
        "Completed: %d raw rows -> %d episodes; %d merges; %d conflicts",
        len(assigned),
        len(manifest),
        int(summary.loc[0, "n_merges_total"]),
        len(conflicts),
    )


if __name__ == "__main__":
    main()
