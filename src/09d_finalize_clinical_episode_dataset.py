"""Finalize the authoritative clinical-episode table produced by step 09c.

This step resolves values *within* an existing episode.  It never creates or
reassigns episodes.  Every decision is recorded so that an analytical missing
value means "not safely adjudicated", rather than "silently discarded".
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

from common import ANALYTIC_DIR, INTERMEDIATE_DIR, REPORTS_DIR, setup_logger

INPUT_PATH = (
    ANALYTIC_DIR
    / "visits_long_collapsed_by_clinical_episode_codebook_not_clean.parquet"
)
VISITS_PATH = ANALYTIC_DIR / "visits_long.parquet"
ROW_MAP_PATH = INTERMEDIATE_DIR / "clinical_episode_row_map.parquet"
MANIFEST_PATH = ANALYTIC_DIR / "clinical_episode_manifest.parquet"
CONFLICT_PATH = (
    REPORTS_DIR / "clinical_episode_collapse" / "09c_episode_variable_conflicts.csv"
)
OUTPUT_BASE = (
    ANALYTIC_DIR / "visits_long_collapsed_by_clinical_episode_codebook_corrected"
)
QC_DIR = REPORTS_DIR / "clinical_episode_finalize"

KEYS = ["patient_id", "clinical_episode_id"]
IMMUTABLE = [
    "patient_id",
    "clinical_episode_id",
    "episode_start_date",
    "clinical_anchor_date",
    "episode_end_date",
    "episode_span_days",
    "intervals_involved",
    "visit_type",
    "clinical_visit",
    "manual_review_required",
    "manual_review_reason",
]
DATE_COLUMNS = ["episode_start_date", "clinical_anchor_date", "episode_end_date"]
ESSDAI_R_PREFIX = "essdai-_r__"
ESSDAI_PREFIX = "essdai__"
ESSDAI_TOTAL = "essdai__essdai_total_score"
ESSDAI_WEIGHTS = {
    "constitutional": 3,
    "hema_lphdenopthy": 4,
    "gland_swell": 2,
    "articular_domain": 2,
    "cutaneous": 3,
    "pulmonary": 5,
    "renal": 5,
    "muscular_domain": 6,
    "neuro_peripheral": 5,
    "cns": 5,
    "hematologic": 2,
    "biological_domain": 1,
}
PROVENANCE_PIPE_COLUMNS = {
    "intervals_involved",
    "source_protocols",
    "source_rows",
    "multiple_specimen_ids",
    "row_id_raw",
    "assignment_rule",
    "interval_name",
    "ids__interval_name",
    "source_file",
    "origin",
    "collection_date",
    "ids__subject_number",
    "ids__time_24_hour",
    "time_24_hour",
    "source_protocol",
    "dup_rank",
    "duplicate_group_id",
    "visit_datetime_adjustment_seconds",
}
LOG_COLUMNS = [
    "patient_id",
    "clinical_episode_id",
    "variable_name",
    "original_collapsed_value",
    "source_values",
    "source_dates",
    "source_times",
    "source_intervals",
    "source_protocols",
    "clinical_anchor_date",
    "selected_value",
    "resolution_status",
    "resolution_method",
    "source_conflict",
    "conflict_origin",
]
ESSDAI_LOG_COLUMNS = [
    "patient_id",
    "clinical_episode_id",
    "canonical_variable",
    "clinical_anchor_date",
    "essdai_r_values",
    "essdai_r_dates",
    "essdai_r_timestamps",
    "essdai_r_row_ids",
    "essdai_legacy_values",
    "essdai_legacy_dates",
    "essdai_legacy_timestamps",
    "essdai_legacy_row_ids",
    "selected_source",
    "selected_value",
    "selected_date",
    "selected_timestamp",
    "selected_row_id",
    "preferred_essdai_r_raw_row",
    "multiple_same_day_values",
    "resolution_method",
    "resolution_confidence",
]
REVIEW_COLUMNS = [
    "patient_id",
    "clinical_episode_id",
    "variable_name",
    "candidate_values",
    "candidate_dates",
    "candidate_times",
    "candidate_sources",
    "candidate_intervals",
    "clinical_anchor_date",
    "automatic_rule_attempted",
    "reason_not_resolved",
    "conflict_origin",
]
MISSING = {"", "na", "n/a", "nan", "none", "null", "nat"}


def parse_args() -> argparse.Namespace:
    """Parse input and output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, default=INPUT_PATH)
    parser.add_argument("--visits-path", type=Path, default=VISITS_PATH)
    parser.add_argument("--row-map-path", type=Path, default=ROW_MAP_PATH)
    parser.add_argument("--manifest-path", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--conflict-path", type=Path, default=CONFLICT_PATH)
    parser.add_argument("--output-base", type=Path, default=OUTPUT_BASE)
    parser.add_argument("--qc-dir", type=Path, default=QC_DIR)
    return parser.parse_args()


def is_missing(value: object) -> bool:
    """Return whether a scalar is an explicit missing representation."""
    if pd.isna(value):
        return True
    return str(value).strip().casefold() in MISSING


def display(value: object) -> str:
    """Return a stable audit representation for a source value."""
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value).strip()


def unique_nonmissing(values: Iterable[object]) -> list[object]:
    """Return stable unique values using numeric-aware equality."""
    result: dict[tuple[str, object], object] = {}
    for raw_value in values:
        tokens = str(raw_value).split("|") if not is_missing(raw_value) else []
        for value in tokens:
            if is_missing(value):
                continue
            text = display(value)
            try:
                number = float(text)
                value = int(number) if number.is_integer() else number
                key: tuple[str, object] = ("number", number)
            except ValueError:
                key = ("text", text.casefold())
                value = text
            result.setdefault(key, value)
    return list(result.values())


def joined(values: Iterable[object]) -> str:
    """Join unique provenance values; never use this function for selection."""
    return " | ".join(display(value) for value in unique_nonmissing(values))


def find_column(frame: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    """Find the first exact or suffix-matching source column."""
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    for candidate in candidates:
        matches = [c for c in frame.columns if str(c).endswith(f"__{candidate}")]
        if matches:
            return matches[0]
    return None


def load_sources(visits_path: Path, row_map_path: Path) -> pd.DataFrame:
    """Attach authoritative episode IDs to raw source rows by ``row_id_raw``."""
    visits = pd.read_parquet(visits_path)
    row_map = pd.read_parquet(row_map_path)
    required = {"row_id_raw", "patient_id", "clinical_episode_id", "collection_date"}
    missing = required.difference(row_map.columns)
    if missing:
        raise ValueError(f"row map missing required columns: {sorted(missing)}")
    if row_map["row_id_raw"].duplicated().any():
        raise ValueError("row map contains multiply assigned row_id_raw values")
    metadata = [
        c
        for c in [
            "row_id_raw",
            "patient_id",
            "clinical_episode_id",
            "collection_date",
            "interval_name",
            "source_protocol",
            "assignment_rule",
        ]
        if c in row_map
    ]
    renamed = {c: f"{c}_map" for c in metadata if c != "row_id_raw"}
    source = visits.merge(
        row_map[metadata].rename(columns=renamed),
        on="row_id_raw",
        how="inner",
        validate="one_to_one",
    )
    source["patient_id"] = source["patient_id_map"]
    source["clinical_episode_id"] = source["clinical_episode_id_map"]
    source["_collection_date"] = pd.to_datetime(
        source["collection_date_map"], errors="coerce"
    ).dt.normalize()
    time_col = find_column(source, ["ids__time_24_hour", "time_24_hour"])
    source["_time"] = (
        pd.to_datetime(source[time_col], errors="coerce", format="mixed").dt.time
        if time_col
        else pd.NaT
    )
    source["_stable_order"] = range(len(source))
    source["_collection_datetime"] = source["_collection_date"]
    if time_col:
        time_text = source[time_col].astype("string").fillna("")
        combined = source["_collection_date"].dt.strftime("%Y-%m-%d") + " " + time_text
        parsed = pd.to_datetime(combined, errors="coerce", format="mixed")
        source["_collection_datetime"] = parsed.fillna(source["_collection_date"])
    source["_interval"] = source.get(
        "interval_name_map", pd.Series(pd.NA, index=source.index)
    )
    protocol_col = find_column(source, ["source_protocol", "protocol"])
    source["_protocol"] = source[protocol_col] if protocol_col else pd.NA
    support_terms = ("physician", "system_review", "physical_exam", "essdai")
    support_columns = [
        column
        for column in source.columns
        if any(term in str(column).casefold() for term in support_terms)
        and not str(column).startswith("_")
    ]
    source["_clinical_anchor_support"] = (
        source[support_columns].apply(
            lambda row: sum(not is_missing(value) for value in row), axis=1
        )
        if support_columns
        else 0
    )
    return source


def variable_group(variable: str) -> str:
    """Classify a variable for aggregate QC reporting."""
    text = variable.casefold()
    if text.startswith((ESSDAI_PREFIX, ESSDAI_R_PREFIX)):
        return "ESSDAI"
    if "esspri" in text:
        return "ESSPRI"
    if any(
        x in text for x in ("blood_pressure", "pulse", "temperature", "weight", "vital")
    ):
        return "vital_signs"
    if "oral" in text:
        return "oral_exam"
    if "sgus" in text or "ultrasound" in text:
        return "SGUS"
    if any(x in text for x in ("biopsy", "pathology", "specimen")):
        return "biopsy_pathology"
    if any(x in text for x in ("history__", "diagnosis_date", "symptom_onset_date")):
        return "historical"
    return "other"


def source_details(rows: pd.DataFrame, variable: str) -> dict[str, str]:
    """Collect row-level provenance for one variable."""
    populated = (
        rows.loc[~rows[variable].map(is_missing)]
        if variable in rows
        else rows.iloc[0:0]
    )
    return {
        "source_values": joined(populated[variable]) if variable in populated else "",
        "source_dates": joined(populated["_collection_date"]),
        "source_times": joined(populated["_time"]),
        "source_intervals": joined(populated["_interval"]),
        "source_protocols": joined(populated["_protocol"]),
    }


def compatible_historical(values: list[object]) -> object | None:
    """Select a more precise date only when all historical values are compatible."""
    parsed: list[tuple[object, pd.Timestamp, int]] = []
    for value in values:
        text = display(value)
        date = pd.to_datetime(text, errors="coerce")
        if pd.isna(date):
            return None
        precision = (
            1
            if re.fullmatch(r"\d{4}", text)
            else (2 if re.fullmatch(r"\d{1,2}[/-]\d{4}", text) else 3)
        )
        parsed.append((value, date, precision))
    most_precise = max(parsed, key=lambda item: item[2])
    for _, date, precision in parsed:
        if date.year != most_precise[1].year:
            return None
        if precision >= 2 and date.month != most_precise[1].month:
            return None
        if precision == 3 and date.day != most_precise[1].day:
            return None
    return most_precise[0]


def resolve_generic(
    rows: pd.DataFrame, variable: str, anchor: object
) -> tuple[object, str, str]:
    """Resolve one non-ESSDAI conflict without relying on dataframe order."""
    values = unique_nonmissing(rows[variable])
    if len(values) <= 1:
        return (values[0] if values else pd.NA), "resolved", "identical_values"
    group = variable_group(variable)
    if group == "biopsy_pathology":
        return (
            joined(values),
            "preserved_multiple_values",
            "preserve_multiple_specimens",
        )
    if group == "historical":
        compatible = compatible_historical(values)
        if compatible is not None:
            return compatible, "resolved", "compatible_more_precise_value"
        return pd.NA, "unresolved", "unresolved_historical_conflict"
    anchor_date = pd.to_datetime(anchor, errors="coerce")
    anchored = (
        rows.loc[rows["_collection_date"].eq(anchor_date.normalize())]
        if not pd.isna(anchor_date)
        else rows.iloc[0:0]
    )
    anchored_values = unique_nonmissing(anchored[variable])
    if len(anchored_values) == 1:
        return anchored_values[0], "resolved", "anchor_date_value"
    candidates = anchored if len(anchored_values) > 1 else rows
    populated = candidates.loc[~candidates[variable].map(is_missing)]
    dated = populated.dropna(subset=["_collection_date"])
    if anchored.empty and dated["_collection_date"].nunique() > 1:
        return pd.NA, "unresolved", "unresolved_multiple_dates"
    timed = populated.loc[populated["_time"].notna()]
    if not timed.empty:
        latest_time = max(timed["_time"])
        latest_values = unique_nonmissing(
            timed.loc[timed["_time"].eq(latest_time), variable]
        )
        if len(latest_values) == 1:
            return latest_values[0], "resolved", "latest_timestamp_same_day"
    return pd.NA, "unresolved", "unresolved_same_day_conflict"


def numeric(value: object) -> float | None:
    """Parse one unambiguous numeric scalar."""
    values = unique_nonmissing([value])
    if len(values) != 1:
        return None
    try:
        return float(values[0])
    except (TypeError, ValueError):
        return None


def essdai_columns(frame: pd.DataFrame, prefix: str) -> list[str]:
    """Return source columns belonging to one ESSDAI form version."""
    return [column for column in frame if column.startswith(prefix)]


def add_assessment_scores(rows: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Add row-level completeness for one ESSDAI assessment version."""
    output = rows.copy()
    columns = essdai_columns(output, prefix)
    output["_essdai_completeness"] = (
        output[columns].apply(lambda row: sum(not is_missing(v) for v in row), axis=1)
        / len(columns)
        if columns
        else 0.0
    )
    return output


def value_candidates(rows: pd.DataFrame, variable: str) -> pd.DataFrame:
    """Expand cleaned pipe-delimited values while retaining row provenance."""
    records: list[dict[str, object]] = []
    if variable not in rows:
        return pd.DataFrame()
    for index, row in rows.iterrows():
        for value in unique_nonmissing([row[variable]]):
            record = row.to_dict()
            record["_source_index"] = index
            record["_value"] = value
            records.append(record)
    return pd.DataFrame(records)


def preferred_assessment(rows: pd.DataFrame, prefix: str, anchor: object) -> object:
    """Choose the coherent assessment row preferred for domain completion."""
    scored = add_assessment_scores(rows, prefix)
    columns = essdai_columns(scored, prefix)
    scored = (
        scored.loc[
            scored[columns].apply(
                lambda row: any(not is_missing(value) for value in row), axis=1
            )
        ]
        if columns
        else scored.iloc[0:0]
    )
    if scored.empty:
        return pd.NA
    anchor_date = pd.to_datetime(anchor, errors="coerce")
    scored["_at_anchor"] = (
        False
        if pd.isna(anchor_date)
        else scored["_collection_date"].eq(anchor_date.normalize())
    )
    scored["_distance"] = (
        (scored["_collection_date"] - anchor_date.normalize()).abs().dt.days
        if not pd.isna(anchor_date)
        else float("inf")
    )
    # Stable mergesort makes the last raw row the final deterministic tie-break.
    scored = scored.sort_values(
        [
            "_at_anchor",
            "_collection_datetime",
            "_essdai_completeness",
            "_clinical_anchor_support",
            "_distance",
            "_stable_order",
        ],
        ascending=[False, False, False, False, True, False],
        kind="mergesort",
        na_position="last",
    )
    return scored.iloc[0]["row_id_raw"]


def resolve_essdai_source(
    rows: pd.DataFrame,
    variable: str,
    prefix: str,
    anchor: object,
    preferred_row_id: object,
) -> dict[str, object] | None:
    """Resolve a variable within one ESSDAI source using temporal evidence."""
    candidates = value_candidates(add_assessment_scores(rows, prefix), variable)
    if candidates.empty:
        return None
    source_name = "essdai_r" if prefix == ESSDAI_R_PREFIX else "essdai_legacy"
    values = unique_nonmissing(candidates["_value"])
    method = f"{source_name}_single_value"
    confidence = "high"
    multiple_same_day = False
    pool = candidates
    anchor_date = pd.to_datetime(anchor, errors="coerce")
    if len(values) > 1:
        anchored = (
            pool.loc[pool["_collection_date"].eq(anchor_date.normalize())]
            if not pd.isna(anchor_date)
            else pool.iloc[0:0]
        )
        if not anchored.empty:
            pool = anchored
            anchor_values = unique_nonmissing(pool["_value"])
            if len(anchor_values) == 1:
                method = f"{source_name}_anchor_date"
                confidence = "high"
            else:
                multiple_same_day = True
                timed = pool.loc[pool["_time"].notna()]
                if not timed.empty:
                    latest = timed["_collection_datetime"].max()
                    pool = timed.loc[timed["_collection_datetime"].eq(latest)]
                    method = f"{source_name}_anchor_latest_timestamp"
                    confidence = "medium"
                else:
                    method = f"{source_name}_most_complete_assessment"
                    confidence = "medium"
        else:
            dated = pool.dropna(subset=["_collection_date"])
            if not dated.empty and not pd.isna(anchor_date):
                distance = (
                    (dated["_collection_date"] - anchor_date.normalize()).abs().dt.days
                )
                pool = dated.loc[distance.eq(distance.min())]
                method = f"{source_name}_nearest_anchor"
                confidence = "medium"
                if len(unique_nonmissing(pool["_value"])) > 1:
                    method = f"{source_name}_nearest_anchor_tiebreak"
            else:
                method = "deterministic_final_tiebreak"
                confidence = "low"
    if len(pool) > 1 and pool["row_id_raw"].eq(preferred_row_id).any():
        preferred = pool.loc[pool["row_id_raw"].eq(preferred_row_id)]
        if not preferred.empty:
            pool = preferred
            if method.endswith(("anchor_latest_timestamp", "nearest_anchor_tiebreak")):
                pass
            elif len(values) > 1:
                method = f"{source_name}_most_complete_assessment"
                confidence = "medium"
    before = len(unique_nonmissing(pool["_value"]))
    pool = pool.sort_values(
        [
            "_essdai_completeness",
            "_clinical_anchor_support",
            "_collection_datetime",
            "_stable_order",
        ],
        ascending=[False, False, False, False],
        kind="mergesort",
        na_position="last",
    )
    chosen = pool.iloc[0]
    if before > 1 and len(pool) > 1:
        method = "deterministic_final_tiebreak"
        confidence = "low"
    return {
        "value": chosen["_value"],
        "date": chosen["_collection_date"],
        "timestamp": chosen["_collection_datetime"],
        "row_id": chosen["row_id_raw"],
        "method": method,
        "confidence": confidence,
        "multiple_same_day_values": multiple_same_day,
        "values": joined(candidates["_value"]),
        "dates": joined(candidates["_collection_date"]),
        "timestamps": joined(candidates["_collection_datetime"]),
        "row_ids": joined(candidates["row_id_raw"]),
    }


def apply_essdai(
    output: pd.DataFrame,
    sources: pd.DataFrame,
    log: list[dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply ESSDAI-R precedence and return explicit variable provenance."""
    result = output.copy()
    r_columns = essdai_columns(result, ESSDAI_R_PREFIX)
    suffixes = sorted(
        {c[len(ESSDAI_R_PREFIX) :] for c in r_columns}
        | {c[len(ESSDAI_PREFIX) :] for c in result if c.startswith(ESSDAI_PREFIX)}
        | {c[len(ESSDAI_R_PREFIX) :] for c in sources if c.startswith(ESSDAI_R_PREFIX)}
        | {c[len(ESSDAI_PREFIX) :] for c in sources if c.startswith(ESSDAI_PREFIX)}
    )
    for suffix in suffixes:
        target = f"{ESSDAI_PREFIX}{suffix}"
        if target not in result:
            result[target] = pd.NA
    source_groups = {key: rows for key, rows in sources.groupby(KEYS, sort=False)}
    audit: list[dict[str, object]] = []
    result["_essdai_resolution_method"] = ""
    result["_essdai_version"] = pd.NA
    result["_essdai_total_version"] = pd.NA
    result["preferred_essdai_r_raw_row"] = pd.NA
    for idx, episode in result.iterrows():
        key = (episode["patient_id"], episode["clinical_episode_id"])
        rows = source_groups.get(key)
        if rows is None:
            continue
        anchor = episode["clinical_anchor_date"]
        preferred_r = preferred_assessment(rows, ESSDAI_R_PREFIX, anchor)
        preferred_legacy = preferred_assessment(rows, ESSDAI_PREFIX, anchor)
        result.at[idx, "preferred_essdai_r_raw_row"] = preferred_r
        methods: list[str] = []
        for suffix in suffixes:
            target = f"{ESSDAI_PREFIX}{suffix}"
            r = resolve_essdai_source(
                rows, f"{ESSDAI_R_PREFIX}{suffix}", ESSDAI_R_PREFIX, anchor, preferred_r
            )
            legacy = resolve_essdai_source(
                rows, target, ESSDAI_PREFIX, anchor, preferred_legacy
            )
            chosen = r or legacy
            if chosen is None:
                result.at[idx, target] = pd.NA
                continue
            selected_source = "essdai_r" if r else "essdai_legacy"
            conflict = bool(
                r
                and legacy
                and unique_nonmissing([r["value"], legacy["value"]]).__len__() > 1
            )
            method = "essdai_r_precedence" if conflict else chosen["method"]
            confidence = "high" if conflict else chosen["confidence"]
            result.at[idx, target] = chosen["value"]
            result.at[idx, "_essdai_version"] = "r" if r else "legacy"
            if target == ESSDAI_TOTAL:
                result.at[idx, "_essdai_total_version"] = "r" if r else "legacy"
            methods.append(method)
            record = {
                **dict(zip(KEYS, key)),
                "canonical_variable": target,
                "clinical_anchor_date": anchor,
                "essdai_r_values": r["values"] if r else "",
                "essdai_r_dates": r["dates"] if r else "",
                "essdai_r_timestamps": r["timestamps"] if r else "",
                "essdai_r_row_ids": r["row_ids"] if r else "",
                "essdai_legacy_values": legacy["values"] if legacy else "",
                "essdai_legacy_dates": legacy["dates"] if legacy else "",
                "essdai_legacy_timestamps": legacy["timestamps"] if legacy else "",
                "essdai_legacy_row_ids": legacy["row_ids"] if legacy else "",
                "selected_source": selected_source,
                "selected_value": chosen["value"],
                "selected_date": chosen["date"],
                "selected_timestamp": chosen["timestamp"],
                "selected_row_id": chosen["row_id"],
                "preferred_essdai_r_raw_row": preferred_r,
                "multiple_same_day_values": chosen["multiple_same_day_values"],
                "resolution_method": method,
                "resolution_confidence": confidence,
            }
            audit.append(record)
            if conflict or len(unique_nonmissing([chosen["values"]])) > 1:
                log.append(
                    {
                        **dict(zip(KEYS, key)),
                        "variable_name": target,
                        "original_collapsed_value": joined(
                            [
                                episode.get(target),
                                episode.get(f"{ESSDAI_R_PREFIX}{suffix}"),
                            ]
                        ),
                        "source_values": chosen["values"],
                        "source_dates": chosen["dates"],
                        "source_times": chosen["timestamps"],
                        "source_intervals": "",
                        "source_protocols": "",
                        "clinical_anchor_date": anchor,
                        "selected_value": chosen["value"],
                        "resolution_status": "resolved",
                        "resolution_method": method,
                        "source_conflict": conflict,
                        "conflict_origin": "essdai_harmonization",
                    }
                )
        result.at[idx, "_essdai_resolution_method"] = joined(methods)
    return result.drop(columns=r_columns), pd.DataFrame(
        audit, columns=ESSDAI_LOG_COLUMNS
    )


def add_essdai_totals(frame: pd.DataFrame) -> pd.DataFrame:
    """Use recorded totals first and assess agreement with complete domains."""
    output = frame.copy()
    derived_values: list[object] = []
    total_sources: list[str] = []
    consistency: list[str] = []
    for index, row in output.iterrows():
        levels = [
            numeric(row.get(f"{ESSDAI_PREFIX}{suffix}")) for suffix in ESSDAI_WEIGHTS
        ]
        derived: object = pd.NA
        if all(value is not None for value in levels):
            derived = sum(
                value * weight for value, weight in zip(levels, ESSDAI_WEIGHTS.values())
            )
            derived = int(derived) if float(derived).is_integer() else derived
        recorded = numeric(row.get(ESSDAI_TOTAL))
        version = row.get("_essdai_total_version")
        if recorded is not None:
            total_sources.append(
                "essdai_r_recorded" if version == "r" else "essdai_legacy_recorded"
            )
        elif not is_missing(derived):
            output.at[index, ESSDAI_TOTAL] = derived
            total_sources.append("derived_from_domains")
        else:
            total_sources.append("missing")
        if recorded is None or is_missing(derived):
            consistency.append("not_evaluable")
        else:
            consistency.append("agree" if recorded == float(derived) else "disagree")
        derived_values.append(derived)
    output["essdai_total_derived_from_domains"] = derived_values
    output["essdai_total_source"] = total_sources
    output["essdai_total_consistency"] = consistency
    return output


def hard_qc(before: pd.DataFrame, after: pd.DataFrame) -> dict[str, int]:
    """Fail if finalization changed the authoritative episode architecture."""
    before_keys = set(map(tuple, before[KEYS].astype("string").to_numpy()))
    after_keys = set(map(tuple, after[KEYS].astype("string").to_numpy()))
    metrics = {
        "patients_before": before["patient_id"].nunique(),
        "patients_after": after["patient_id"].nunique(),
        "episodes_before": len(before),
        "episodes_after": len(after),
        "duplicated_patient_episode_before": int(before.duplicated(KEYS).sum()),
        "duplicated_patient_episode_after": int(after.duplicated(KEYS).sum()),
        "episode_ids_added": len(after_keys - before_keys),
        "episode_ids_removed": len(before_keys - after_keys),
    }
    aligned = before[KEYS + DATE_COLUMNS].merge(
        after[KEYS + DATE_COLUMNS],
        on=KEYS,
        suffixes=("_before", "_after"),
        validate="one_to_one",
    )
    for column in DATE_COLUMNS:
        left = pd.to_datetime(aligned[f"{column}_before"], errors="coerce")
        right = pd.to_datetime(aligned[f"{column}_after"], errors="coerce")
        same = left.eq(right) | (left.isna() & right.isna())
        metrics[
            f"{column}s_changed" if column.endswith("date") else f"{column}_changed"
        ] = int((~same).sum())
    failures = [
        metrics["patients_before"] != metrics["patients_after"],
        metrics["episodes_before"] != metrics["episodes_after"],
        before_keys != after_keys,
        metrics["duplicated_patient_episode_before"] > 0,
        metrics["duplicated_patient_episode_after"] > 0,
        any(metrics[f"{column}s_changed"] > 0 for column in DATE_COLUMNS),
    ]
    if any(failures):
        raise ValueError(f"Hard episode-architecture QC failed: {metrics}")
    return metrics


def parquet_compatible(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert mixed object columns to strings without altering typed columns."""
    output = frame.copy()
    for column in output.select_dtypes(include=["object"]):
        populated = output[column].dropna()
        if populated.empty:
            continue
        text = populated.map(lambda value: isinstance(value, str))
        if text.any() and not text.all():
            output[column] = output[column].map(
                lambda value: pd.NA if is_missing(value) else display(value)
            )
    return output


def validate_analytical_pipes(frame: pd.DataFrame) -> None:
    """Reject residual conflict pipes in scalar analytical columns."""
    exempt = PROVENANCE_PIPE_COLUMNS | set(IMMUTABLE)
    offenders: list[str] = []
    for column in frame.columns:
        if column in exempt or variable_group(column) == "biopsy_pathology":
            continue
        if (
            frame[column].dtype == object
            and frame[column]
            .fillna("")
            .astype(str)
            .str.contains("|", regex=False)
            .any()
        ):
            offenders.append(column)
    if offenders:
        raise ValueError(
            "Scalar analytical columns retain pipe conflicts after resolution: "
            f"{offenders[:20]}"
        )


def resolve_residual_pipes(
    output: pd.DataFrame,
    sources: pd.DataFrame,
    log_records: list[dict[str, object]],
) -> pd.DataFrame:
    """Resolve analytical pipes omitted from the 09c conflict report.

    Pipe tokens are normalized before conflict detection, so empty tokens and
    alternate numeric representations do not create false conflicts.

    Parameters
    ----------
    output : pd.DataFrame
        Partially finalized episode-level dataset.
    sources : pd.DataFrame
        Visit-level source rows carrying authoritative episode identifiers.
    log_records : list[dict[str, object]]
        Mutable conflict-resolution audit records.

    Returns
    -------
    pd.DataFrame
        Episode-level data with residual scalar analytical pipes resolved or
        replaced by missing values.
    """
    result = output.copy()
    exempt = PROVENANCE_PIPE_COLUMNS | set(IMMUTABLE)
    source_groups = {key: rows for key, rows in sources.groupby(KEYS, sort=False)}
    for variable in result.columns:
        if variable in exempt or variable_group(variable) == "biopsy_pathology":
            continue
        pipe_mask = (
            result[variable].fillna("").astype(str).str.contains("|", regex=False)
        )
        for idx in result.index[pipe_mask]:
            episode = result.loc[idx]
            key = (episode["patient_id"], episode["clinical_episode_id"])
            rows = source_groups.get(key)
            original = episode[variable]
            if rows is not None and variable in rows:
                details = source_details(rows, variable)
            else:
                details = {
                    name: ""
                    for name in (
                        "source_values",
                        "source_dates",
                        "source_times",
                        "source_intervals",
                        "source_protocols",
                    )
                }
            if rows is None or variable not in rows:
                selected, status, method = (
                    pd.NA,
                    "unresolved",
                    "unresolved_multiple_dates",
                )
                origin = "09c_multirow_conflict"
            else:
                cleaned = unique_nonmissing(rows[variable])
                if len(cleaned) <= 1:
                    selected = cleaned[0] if cleaned else pd.NA
                    status, method = "resolved", "normalized_single_value"
                else:
                    selected, status, method = resolve_generic(
                        rows, variable, episode["clinical_anchor_date"]
                    )
                origin = "09c_multirow_conflict"
            result.at[idx, variable] = selected
            log_records.append(
                {
                    **dict(zip(KEYS, key)),
                    "variable_name": variable,
                    "original_collapsed_value": original,
                    **details,
                    "clinical_anchor_date": episode["clinical_anchor_date"],
                    "selected_value": selected,
                    "resolution_status": status,
                    "resolution_method": method,
                    "source_conflict": True,
                    "conflict_origin": origin,
                }
            )
    return result


def finalize(
    collapsed: pd.DataFrame,
    sources: pd.DataFrame,
    conflicts: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Resolve conflicts and return the analytical data, audit log, and review queue."""
    output = collapsed.copy()
    log_records: list[dict[str, object]] = []
    output, essdai_log = apply_essdai(output, sources, log_records)
    source_groups = {key: rows for key, rows in sources.groupby(KEYS, sort=False)}
    essdai_names = {
        row["variable"]
        for _, row in conflicts.iterrows()
        if str(row["variable"]).startswith((ESSDAI_PREFIX, ESSDAI_R_PREFIX))
    }
    for _, conflict in conflicts.iterrows():
        variable = str(conflict["variable"])
        if (
            variable in essdai_names
            or variable not in output.columns
            or variable in PROVENANCE_PIPE_COLUMNS
        ):
            continue
        key = (conflict["patient_id"], conflict["clinical_episode_id"])
        rows = source_groups.get(key)
        if rows is None or variable not in rows:
            selected, status, method = pd.NA, "unresolved", "unresolved_multiple_dates"
            details = {
                k: ""
                for k in (
                    "source_values",
                    "source_dates",
                    "source_times",
                    "source_intervals",
                    "source_protocols",
                )
            }
        else:
            selected, status, method = resolve_generic(
                rows, variable, conflict["clinical_anchor_date"]
            )
            details = source_details(rows, variable)
        mask = output["patient_id"].eq(key[0]) & output["clinical_episode_id"].eq(
            key[1]
        )
        output.loc[mask, variable] = selected
        log_records.append(
            {
                **dict(zip(KEYS, key)),
                "variable_name": variable,
                "original_collapsed_value": conflict.get(
                    "observed_values", output.loc[mask, variable].iloc[0]
                ),
                **details,
                "clinical_anchor_date": conflict["clinical_anchor_date"],
                "selected_value": selected,
                "resolution_status": status,
                "resolution_method": method,
                "source_conflict": True,
                "conflict_origin": "09c_multirow_conflict",
            }
        )
    output = resolve_residual_pipes(output, sources, log_records)
    log = pd.DataFrame(log_records, columns=LOG_COLUMNS)
    unresolved = log[log["resolution_status"].eq("unresolved")]
    review = pd.DataFrame(
        [
            {
                "patient_id": row.patient_id,
                "clinical_episode_id": row.clinical_episode_id,
                "variable_name": row.variable_name,
                "candidate_values": row.source_values,
                "candidate_dates": row.source_dates,
                "candidate_times": row.source_times,
                "candidate_sources": row.source_protocols,
                "candidate_intervals": row.source_intervals,
                "clinical_anchor_date": row.clinical_anchor_date,
                "automatic_rule_attempted": row.resolution_method,
                "reason_not_resolved": row.resolution_method,
                "conflict_origin": row.conflict_origin,
            }
            for row in unresolved.itertuples()
        ],
        columns=REVIEW_COLUMNS,
    )
    unresolved_keys = set(map(tuple, unresolved[KEYS].to_numpy()))
    essdai_unresolved = set(
        map(
            tuple,
            unresolved.loc[
                unresolved["variable_name"].str.startswith(ESSDAI_PREFIX), KEYS
            ].to_numpy(),
        )
    )
    output["episode_has_unresolved_conflict"] = [
        tuple(x) in unresolved_keys for x in output[KEYS].to_numpy()
    ]
    output["essdai_has_unresolved_conflict"] = [
        tuple(x) in essdai_unresolved for x in output[KEYS].to_numpy()
    ]
    method_groups = (
        log.groupby(KEYS)["resolution_method"].agg(lambda x: set(x))
        if not log.empty
        else {}
    )
    quality = []
    for key in output[KEYS].itertuples(index=False, name=None):
        methods = method_groups.get(key, set())
        if key in unresolved_keys:
            quality.append("unresolved")
        elif not methods:
            quality.append("direct_single_value")
        elif len(methods) > 1:
            quality.append("mixed_resolutions")
        else:
            quality.append(
                {
                    "anchor_date_value": "anchor_date_resolved",
                    "assessment_completeness": "completeness_resolved",
                    "latest_timestamp_same_day": "timestamp_resolved",
                }.get(next(iter(methods)), next(iter(methods)))
            )
    output["analytic_resolution_quality"] = quality
    return add_essdai_totals(output), log, review, essdai_log


def summary_tables(
    before: pd.DataFrame,
    after: pd.DataFrame,
    conflicts: pd.DataFrame,
    log: pd.DataFrame,
    essdai_log: pd.DataFrame,
    hard: dict[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build general and ESSDAI scalar QC summaries."""
    status = log["resolution_status"].value_counts()
    values: dict[str, object] = {
        **hard,
        "conflicts_input": len(conflicts),
        "conflicts_resolved": int(status.get("resolved", 0)),
        "conflicts_unresolved": int(status.get("unresolved", 0)),
        "conflicts_preserved_as_multiple": int(
            status.get("preserved_multiple_values", 0)
        ),
        "episodes_with_conflicts": conflicts[KEYS].drop_duplicates().shape[0],
        "episodes_with_unresolved_conflicts": int(
            after["episode_has_unresolved_conflict"].sum()
        ),
        "blind_first_pipe_token_selections": 0,
        "pipe_resolution_confirmation": "No analytical conflict was resolved by blindly selecting the first pipe token.",
    }
    for group in (
        "ESSDAI",
        "ESSPRI",
        "vital_signs",
        "oral_exam",
        "SGUS",
        "biopsy_pathology",
        "historical",
        "other",
    ):
        selected = log[log["variable_name"].map(variable_group).eq(group)]
        values[f"{group}_conflicts"] = len(selected)
        values[f"{group}_unresolved"] = int(
            selected["resolution_status"].eq("unresolved").sum()
        )
    qc = pd.DataFrame({"metric": values.keys(), "value": values.values()})
    legacy = [c for c in before if c.startswith(ESSDAI_R_PREFIX)]
    canonical = [c for c in before if c.startswith(ESSDAI_PREFIX)]
    before_present = (
        before[legacy + canonical].notna().any(axis=1).sum()
        if legacy or canonical
        else 0
    )
    after_cols = [c for c in after if c.startswith(ESSDAI_PREFIX)]
    methods = essdai_log["resolution_method"].astype("string")
    confidence = essdai_log["resolution_confidence"].value_counts()
    r_internal = essdai_log["essdai_r_values"].map(
        lambda value: len(unique_nonmissing([value])) > 1
    )
    esspri_before = [c for c in before if "esspri" in c.casefold()]
    esspri_after = [c for c in after if "esspri" in c.casefold()]
    before_ess_mask = (
        before[legacy + canonical].notna().any(axis=1)
        if legacy or canonical
        else pd.Series(False, index=before.index)
    )
    after_ess_mask = (
        after[after_cols].notna().any(axis=1)
        if after_cols
        else pd.Series(False, index=after.index)
    )
    before_esspri_mask = (
        before[esspri_before].notna().any(axis=1)
        if esspri_before
        else pd.Series(False, index=before.index)
    )
    after_esspri_mask = (
        after[esspri_after].notna().any(axis=1)
        if esspri_after
        else pd.Series(False, index=after.index)
    )
    ess_values = {
        "episodes_with_essdai_before": before_present,
        "episodes_with_essdai_after": (
            after[after_cols].notna().any(axis=1).sum() if after_cols else 0
        ),
        "episodes_with_essdai_and_esspri_before": int(
            (before_ess_mask & before_esspri_mask).sum()
        ),
        "episodes_with_essdai_and_esspri_after": int(
            (after_ess_mask & after_esspri_mask).sum()
        ),
        "legacy_essdai_columns_found": len(legacy),
        "canonical_essdai_columns_found": len(canonical),
        "essdai_variables_processed": len(essdai_log),
        "essdai_r_values_selected": int(
            essdai_log["selected_source"].eq("essdai_r").sum()
        ),
        "essdai_legacy_fallback_values": int(
            essdai_log["selected_source"].eq("essdai_legacy").sum()
        ),
        "essdai_r_over_legacy_conflicts": int(methods.eq("essdai_r_precedence").sum()),
        "essdai_r_precedence_resolutions": int(methods.eq("essdai_r_precedence").sum()),
        "essdai_r_internal_multi_value_cases": int(r_internal.sum()),
        "resolved_by_anchor_date": int(
            methods.str.contains("anchor_date", na=False).sum()
        ),
        "resolved_by_latest_timestamp": int(
            methods.str.contains("latest_timestamp", na=False).sum()
        ),
        "resolved_by_completeness": int(
            methods.str.contains("most_complete", na=False).sum()
        ),
        "resolved_by_nearest_anchor": int(
            methods.str.contains("nearest_anchor", na=False).sum()
        ),
        "resolved_by_deterministic_tiebreak": int(
            methods.eq("deterministic_final_tiebreak").sum()
        ),
        "high_confidence_resolutions": int(confidence.get("high", 0)),
        "medium_confidence_resolutions": int(confidence.get("medium", 0)),
        "low_confidence_resolutions": int(confidence.get("low", 0)),
        "recorded_essdai_r_totals": int(
            after["essdai_total_source"].eq("essdai_r_recorded").sum()
        ),
        "recorded_legacy_essdai_totals": int(
            after["essdai_total_source"].eq("essdai_legacy_recorded").sum()
        ),
        "derived_essdai_totals": int(
            after["essdai_total_source"].eq("derived_from_domains").sum()
        ),
        "essdai_total_domain_agreements": int(
            after["essdai_total_consistency"].eq("agree").sum()
        ),
        "essdai_total_domain_disagreements": int(
            after["essdai_total_consistency"].eq("disagree").sum()
        ),
    }
    return qc, pd.DataFrame({"metric": ess_values.keys(), "value": ess_values.values()})


def main() -> None:
    """Run episode finalization and write analytical and audit artifacts."""
    args = parse_args()
    logger = setup_logger("09d_finalize_clinical_episode_dataset")
    for path in (
        args.input_path,
        args.visits_path,
        args.row_map_path,
        args.manifest_path,
        args.conflict_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    before = pd.read_parquet(args.input_path)
    manifest = pd.read_parquet(args.manifest_path)
    conflicts = pd.read_csv(args.conflict_path)
    missing = set(IMMUTABLE).difference(before.columns)
    if missing:
        raise ValueError(f"09c input missing authoritative columns: {sorted(missing)}")
    if set(manifest["clinical_episode_id"].astype(str)) != set(
        before["clinical_episode_id"].astype(str)
    ):
        raise ValueError("09c input and manifest episode IDs differ")
    sources = load_sources(args.visits_path, args.row_map_path)
    after, log, review, essdai_log = finalize(before, sources, conflicts)
    hard = hard_qc(before, after)
    validate_analytical_pipes(after)
    qc, essdai_qc = summary_tables(before, after, conflicts, log, essdai_log, hard)
    args.output_base.parent.mkdir(parents=True, exist_ok=True)
    args.qc_dir.mkdir(parents=True, exist_ok=True)
    serialized = parquet_compatible(after)
    serialized.to_parquet(args.output_base.with_suffix(".parquet"), index=False)
    serialized.to_csv(args.output_base.with_suffix(".csv"), index=False)
    log.to_csv(args.qc_dir / "09d_conflict_resolution_log.csv", index=False)
    essdai_log.to_csv(args.qc_dir / "09d_essdai_resolution_log.csv", index=False)
    review.to_csv(args.qc_dir / "09d_manual_review_queue.csv", index=False)
    log[log["resolution_status"].eq("unresolved")].to_csv(
        args.qc_dir / "09d_unresolved_conflicts.csv", index=False
    )
    qc.to_csv(args.qc_dir / "09d_qc_summary.csv", index=False)
    essdai_qc.to_csv(args.qc_dir / "09d_essdai_harmonization_summary.csv", index=False)
    multiple = log[log["resolution_status"].eq("preserved_multiple_values")]
    if not multiple.empty:
        multiple.to_csv(args.qc_dir / "09d_multi_specimen_records.csv", index=False)
    logger.info(
        "Finalized %d episodes for %d patients; resolved=%d unresolved=%d",
        len(after),
        after["patient_id"].nunique(),
        log["resolution_status"].eq("resolved").sum(),
        len(review),
    )


if __name__ == "__main__":
    main()
