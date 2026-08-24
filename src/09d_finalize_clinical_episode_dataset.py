"""Finalize the 09c clinical-episode table without changing episode structure.

The step is deliberately conservative: every 09c conflict remains traceable, and
new ESSDAI legacy/canonical disagreements are either supported by complete source
provenance or retained as explicit analytical missing values for manual review.
"""

from __future__ import annotations

import argparse
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
DATE_COLUMNS = ["episode_start_date", "clinical_anchor_date", "episode_end_date"]
IMMUTABLE = (
    KEYS
    + DATE_COLUMNS
    + [
        "episode_span_days",
        "intervals_involved",
        "visit_type",
        "clinical_visit",
        "manual_review_required",
        "manual_review_reason",
    ]
)
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
MISSING = {"", "na", "n/a", "nan", "none", "null", "nat"}
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
SOURCE_FIELDS = ["variable", "value", "date", "time", "row_id", "interval", "protocol"]
LOG_COLUMNS = (
    KEYS
    + [
        "clinical_anchor_date",
        "conflict_origin",
        "variable_name",
        "canonical_variable",
        "original_variable",
        "original_collapsed_value",
    ]
    + [f"source_{field}_{number}" for number in (1, 2) for field in SOURCE_FIELDS]
    + [
        "source_values",
        "source_dates",
        "source_times",
        "source_rows",
        "source_intervals",
        "source_protocols",
        "selected_value",
        "resolution_status",
        "resolution_method",
        "manual_review_required",
    ]
)
REVIEW_COLUMNS = LOG_COLUMNS


def parse_args() -> argparse.Namespace:
    """Parse command-line input and output paths."""
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
    return bool(pd.isna(value)) or str(value).strip().casefold() in MISSING


def display(value: object) -> str:
    """Return a stable audit representation for a scalar."""
    return value.isoformat() if isinstance(value, pd.Timestamp) else str(value).strip()


def comparison_key(value: object) -> tuple[str, object]:
    """Return a numeric-aware equality key without changing displayed values."""
    text = display(value)
    try:
        return "number", float(text)
    except ValueError:
        return "text", text.casefold()


def pipe_tokens(value: object) -> list[object]:
    """Normalize a collapsed scalar, removing empty and equivalent pipe tokens."""
    if is_missing(value):
        return []
    tokens = str(value).split("|") if isinstance(value, str) else [value]
    unique: dict[tuple[str, object], object] = {}
    for token in tokens:
        if is_missing(token):
            continue
        unique.setdefault(
            comparison_key(token), token.strip() if isinstance(token, str) else token
        )
    return list(unique.values())


def unique_nonmissing(values: Iterable[object]) -> list[object]:
    """Return stable unique scalar or pipe-token values."""
    unique: dict[tuple[str, object], object] = {}
    for value in values:
        for token in pipe_tokens(value):
            unique.setdefault(comparison_key(token), token)
    return list(unique.values())


def joined(values: Iterable[object]) -> str:
    """Join unique values for provenance; never use this result for selection."""
    return " | ".join(display(value) for value in unique_nonmissing(values))


def find_column(frame: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    """Find the first exact or suffix-matching source column."""
    for candidate in candidates:
        if candidate in frame:
            return candidate
    for candidate in candidates:
        match = next((c for c in frame if str(c).endswith(f"__{candidate}")), None)
        if match:
            return match
    return None


def load_sources(visits_path: Path, row_map_path: Path) -> pd.DataFrame:
    """Attach authoritative episode identifiers and provenance to source rows."""
    visits, row_map = pd.read_parquet(visits_path), pd.read_parquet(row_map_path)
    required = {"row_id_raw", "patient_id", "clinical_episode_id", "collection_date"}
    if missing := required.difference(row_map):
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
        ]
        if c in row_map
    ]
    source = visits.merge(
        row_map[metadata].rename(
            columns={c: f"{c}_map" for c in metadata if c != "row_id_raw"}
        ),
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
        else pd.Series(pd.NaT, index=source.index)
    )
    source["_interval"] = source.get(
        "interval_name_map", pd.Series(pd.NA, index=source.index)
    )
    protocol_col = find_column(
        source, ["source_protocol_map", "source_protocol", "protocol"]
    )
    source["_protocol"] = source[protocol_col] if protocol_col else pd.NA
    return source


def variable_group(variable: str) -> str:
    """Classify a variable for conservative resolution and reporting."""
    text = variable.casefold()
    if text.startswith((ESSDAI_PREFIX, ESSDAI_R_PREFIX)):
        return "ESSDAI"
    if "esspri" in text:
        return "ESSPRI"
    if any(
        x in text
        for x in (
            "blood_pressure",
            "pulse",
            "respiratory_rate",
            "temperature",
            "weight",
            "vital",
        )
    ):
        return "vital_signs"
    if any(x in text for x in ("biopsy", "pathology", "specimen")):
        return "biopsy_pathology"
    if any(x in text for x in ("history__", "diagnosis_date", "symptom_onset_date")):
        return "historical"
    return "other"


def observation_rows(rows: pd.DataFrame, variable: str) -> list[dict[str, object]]:
    """Build value-level provenance observations before any column is removed."""
    observations: list[dict[str, object]] = []
    if variable not in rows:
        return observations
    for _, row in rows.loc[~rows[variable].map(is_missing)].iterrows():
        for value in pipe_tokens(row[variable]):
            observations.append(
                {
                    "variable": variable,
                    "value": value,
                    "date": row.get("_collection_date", pd.NA),
                    "time": row.get("_time", pd.NA),
                    "row_id": row.get("row_id_raw", pd.NA),
                    "interval": row.get("_interval", pd.NA),
                    "protocol": row.get("_protocol", pd.NA),
                }
            )
    return observations


def provenance_fields(observations: list[dict[str, object]]) -> dict[str, object]:
    """Serialize provenance while retaining explicit slots for the first two values."""
    by_value: dict[tuple[str, object], dict[str, object]] = {}
    for observation in observations:
        by_value.setdefault(comparison_key(observation["value"]), observation)
    selected = list(by_value.values())
    fields: dict[str, object] = {}
    for number in (1, 2):
        observation = selected[number - 1] if len(selected) >= number else {}
        for field in SOURCE_FIELDS:
            fields[f"source_{field}_{number}"] = observation.get(field, "")
    fields.update(
        {
            "source_values": joined(o["value"] for o in observations),
            "source_dates": joined(o["date"] for o in observations),
            "source_times": joined(o["time"] for o in observations),
            "source_rows": joined(o["row_id"] for o in observations),
            "source_intervals": joined(o["interval"] for o in observations),
            "source_protocols": joined(o["protocol"] for o in observations),
        }
    )
    return fields


def resolve_generic(
    rows: pd.DataFrame, variable: str, anchor: object
) -> tuple[object, str, str]:
    """Resolve only documented safe generic cases; never use timestamp precedence."""
    values = unique_nonmissing(rows[variable])
    if not values:
        return pd.NA, "missing", "empty_pipe_normalized_to_missing"
    if len(values) == 1:
        return values[0], "resolved", "identical_values"
    group = variable_group(variable)
    if group == "biopsy_pathology":
        return (
            joined(values),
            "preserved_multiple_values",
            "preserve_multiple_specimens",
        )
    if group == "vital_signs":
        anchor_date = pd.to_datetime(anchor, errors="coerce")
        anchored = (
            rows.loc[rows["_collection_date"].eq(anchor_date.normalize())]
            if not pd.isna(anchor_date)
            else rows.iloc[0:0]
        )
        anchored_values = unique_nonmissing(anchored[variable])
        if len(anchored_values) == 1:
            return anchored_values[0], "resolved", "anchor_date_value"
    same_day = (
        rows.loc[~rows[variable].map(is_missing), "_collection_date"].nunique() <= 1
    )
    method = "unresolved_same_day_conflict" if same_day else "unresolved_multiple_dates"
    if group == "historical":
        method = "unresolved_historical_conflict"
    return pd.NA, "unresolved", method


def make_record(
    episode: pd.Series,
    origin: str,
    variable: str,
    canonical: str,
    original: object,
    observations: list[dict[str, object]],
    selected: object,
    status: str,
    method: str,
) -> dict[str, object]:
    """Create one complete conflict audit record."""
    return {
        **{key: episode[key] for key in KEYS},
        "clinical_anchor_date": episode.get("clinical_anchor_date"),
        "conflict_origin": origin,
        "variable_name": variable,
        "canonical_variable": canonical,
        "original_variable": variable,
        "original_collapsed_value": original,
        **provenance_fields(observations),
        "selected_value": selected,
        "resolution_status": status,
        "resolution_method": method,
        "manual_review_required": status == "unresolved",
    }


def apply_essdai(
    output: pd.DataFrame, sources: pd.DataFrame, records: list[dict[str, object]]
) -> pd.DataFrame:
    """Harmonize ESSDAI columns without undocumented version precedence."""
    result = output.copy()
    legacy = [c for c in output if c.startswith(ESSDAI_R_PREFIX)]
    suffixes = sorted(
        {c.removeprefix(ESSDAI_R_PREFIX) for c in legacy}
        | {c.removeprefix(ESSDAI_PREFIX) for c in output if c.startswith(ESSDAI_PREFIX)}
    )
    groups = {key: rows for key, rows in sources.groupby(KEYS, sort=False)}
    for suffix in suffixes:
        target = f"{ESSDAI_PREFIX}{suffix}"
        if target not in result:
            result[target] = pd.NA
    for idx, episode in output.iterrows():
        key = tuple(episode[k] for k in KEYS)
        rows = groups.get(key, sources.iloc[0:0])
        for suffix in suffixes:
            legacy_var, target = (
                f"{ESSDAI_R_PREFIX}{suffix}",
                f"{ESSDAI_PREFIX}{suffix}",
            )
            legacy_value, canonical_value = episode.get(legacy_var, pd.NA), episode.get(
                target, pd.NA
            )
            legacy_tokens, canonical_tokens = pipe_tokens(legacy_value), pipe_tokens(
                canonical_value
            )
            legacy_obs, canonical_obs = observation_rows(
                rows, legacy_var
            ), observation_rows(rows, target)
            all_obs = legacy_obs + canonical_obs
            if len(legacy_tokens) > 1 or len(canonical_tokens) > 1:
                selected, status, method = (
                    pd.NA,
                    "unresolved",
                    "insufficient_source_provenance",
                )
            elif legacy_tokens and canonical_tokens:
                if comparison_key(legacy_tokens[0]) == comparison_key(
                    canonical_tokens[0]
                ):
                    selected, status, method = (
                        legacy_tokens[0],
                        "resolved",
                        "legacy_canonical_agree",
                    )
                else:
                    selected, status, method = (
                        pd.NA,
                        "unresolved",
                        "unresolved_legacy_canonical_conflict",
                    )
                    records.append(
                        make_record(
                            episode,
                            "09d_harmonization",
                            legacy_var,
                            target,
                            joined([legacy_value, canonical_value]),
                            all_obs,
                            selected,
                            status,
                            method,
                        )
                    )
            elif legacy_tokens or canonical_tokens:
                selected = (legacy_tokens or canonical_tokens)[0]
                status, method = "resolved", "single_nonmissing_source"
            else:
                selected, status, method = pd.NA, "missing", "both_sources_missing"
            result.at[idx, target] = selected
    return result.drop(columns=legacy, errors="ignore")


def process_09c_conflicts(
    output: pd.DataFrame,
    collapsed: pd.DataFrame,
    sources: pd.DataFrame,
    conflicts: pd.DataFrame,
    records: list[dict[str, object]],
) -> pd.DataFrame:
    """Preserve and adjudicate every inherited 09c conflict exactly once."""
    result = output.copy()
    groups = {key: rows for key, rows in sources.groupby(KEYS, sort=False)}
    for _, conflict in conflicts.iterrows():
        key = (conflict["patient_id"], conflict["clinical_episode_id"])
        mask = result[KEYS[0]].eq(key[0]) & result[KEYS[1]].eq(key[1])
        if mask.sum() != 1:
            raise ValueError(f"09c conflict references absent/non-unique episode {key}")
        episode = result.loc[mask].iloc[0]
        variable = str(conflict["variable"])
        canonical = (
            f"{ESSDAI_PREFIX}{variable.removeprefix(ESSDAI_R_PREFIX)}"
            if variable.startswith(ESSDAI_R_PREFIX)
            else variable
        )
        original = conflict.get(
            "observed_values",
            collapsed.loc[mask, variable].iloc[0] if variable in collapsed else pd.NA,
        )
        rows = groups.get(key, sources.iloc[0:0])
        observations = observation_rows(rows, variable)
        tokens = pipe_tokens(original)
        if not tokens:
            selected, status, method = (
                pd.NA,
                "missing",
                "empty_pipe_normalized_to_missing",
            )
        elif variable.startswith((ESSDAI_PREFIX, ESSDAI_R_PREFIX)):
            selected, status = pd.NA, "unresolved"
            method = (
                "insufficient_source_provenance"
                if len({comparison_key(o["value"]) for o in observations}) < len(tokens)
                else "unresolved_essdai_source_conflict"
            )
        elif variable in rows:
            selected, status, method = resolve_generic(
                rows, variable, episode["clinical_anchor_date"]
            )
        else:
            selected, status, method = (
                pd.NA,
                "unresolved",
                "insufficient_source_provenance",
            )
        if canonical in result and status != "preserved_multiple_values":
            result.loc[mask, canonical] = selected
        records.append(
            make_record(
                episode,
                "09c",
                variable,
                canonical,
                original,
                observations,
                selected,
                status,
                method,
            )
        )
    return result


def resolve_residual_pipes(
    output: pd.DataFrame, sources: pd.DataFrame, records: list[dict[str, object]]
) -> pd.DataFrame:
    """Normalize empty/equivalent pipes and audit remaining unreported conflicts."""
    result = output.copy()
    groups = {key: rows for key, rows in sources.groupby(KEYS, sort=False)}
    already = {
        (r["patient_id"], r["clinical_episode_id"], r["canonical_variable"])
        for r in records
    }
    for variable in result:
        if (
            variable in PROVENANCE_PIPE_COLUMNS
            or variable in IMMUTABLE
            or variable_group(variable) == "biopsy_pathology"
        ):
            continue
        for idx in result.index[
            result[variable].fillna("").astype(str).str.contains("|", regex=False)
        ]:
            episode, original = result.loc[idx], result.at[idx, variable]
            tokens = pipe_tokens(original)
            if len(tokens) <= 1:
                result.at[idx, variable] = tokens[0] if tokens else pd.NA
                continue
            key = tuple(episode[k] for k in KEYS)
            if (*key, variable) in already:
                result.at[idx, variable] = pd.NA
                continue
            rows = groups.get(key, sources.iloc[0:0])
            observations = observation_rows(rows, variable)
            selected, status, method = (
                resolve_generic(rows, variable, episode["clinical_anchor_date"])
                if variable in rows
                else (pd.NA, "unresolved", "insufficient_source_provenance")
            )
            result.at[idx, variable] = selected
            records.append(
                make_record(
                    episode,
                    "09d_harmonization",
                    variable,
                    variable,
                    original,
                    observations,
                    selected,
                    status,
                    method,
                )
            )
    return result


def numeric(value: object) -> float | None:
    """Parse only one unambiguous numeric value."""
    tokens = pipe_tokens(value)
    if len(tokens) != 1:
        return None
    try:
        return float(tokens[0])
    except (TypeError, ValueError):
        return None


def add_essdai_totals(frame: pd.DataFrame) -> pd.DataFrame:
    """Use valid recorded totals, or derive only when every domain is unambiguous."""
    output = frame.copy()
    derived_values, total_sources, inconsistencies = [], [], []
    for idx, row in output.iterrows():
        levels = [
            numeric(row.get(f"{ESSDAI_PREFIX}{suffix}")) for suffix in ESSDAI_WEIGHTS
        ]
        derived = (
            sum(
                value * weight for value, weight in zip(levels, ESSDAI_WEIGHTS.values())
            )
            if all(value is not None for value in levels)
            else None
        )
        recorded = numeric(row.get(ESSDAI_TOTAL))
        if recorded is not None:
            source = "recorded_valid"
        elif derived is not None:
            output.at[idx, ESSDAI_TOTAL] = (
                int(derived) if float(derived).is_integer() else derived
            )
            source = "derived_from_domains"
        elif bool(row.get("essdai_has_unresolved_conflict", False)):
            output.at[idx, ESSDAI_TOTAL] = pd.NA
            source = "unresolved"
        else:
            source = "missing"
        derived_values.append(pd.NA if derived is None else derived)
        total_sources.append(source)
        inconsistencies.append(
            recorded is not None and derived is not None and recorded != derived
        )
    output["essdai_total_derived_from_domains"] = derived_values
    output["essdai_total_source"] = total_sources
    output["essdai_internal_inconsistency"] = inconsistencies
    return output


def hard_qc(
    before: pd.DataFrame,
    after: pd.DataFrame,
    conflicts: pd.DataFrame | None = None,
    log: pd.DataFrame | None = None,
) -> dict[str, int]:
    """Fail if episode architecture changes or an inherited conflict disappears."""
    before_patients, after_patients = set(before[KEYS[0]].astype(str)), set(
        after[KEYS[0]].astype(str)
    )
    before_keys = set(map(tuple, before[KEYS].astype(str).to_numpy()))
    after_keys = set(map(tuple, after[KEYS].astype(str).to_numpy()))
    metrics = {
        "patients_before": len(before_patients),
        "patients_after": len(after_patients),
        "episodes_before": len(before),
        "episodes_after": len(after),
        "episodes_created": len(after_keys - before_keys),
        "episodes_deleted": len(before_keys - after_keys),
        "episodes_merged": 0,
        "episodes_split": 0,
        "duplicate_patient_episode": int(after.duplicated(KEYS).sum()),
    }
    aligned = before[KEYS + DATE_COLUMNS].merge(
        after[KEYS + DATE_COLUMNS],
        on=KEYS,
        suffixes=("_before", "_after"),
        validate="one_to_one",
    )
    for column in DATE_COLUMNS:
        left, right = pd.to_datetime(
            aligned[f"{column}_before"], errors="coerce"
        ), pd.to_datetime(aligned[f"{column}_after"], errors="coerce")
        metrics[f"{column}_changed"] = int(
            (~(left.eq(right) | (left.isna() & right.isna()))).sum()
        )
    failures = [
        before_patients != after_patients,
        before_keys != after_keys,
        len(before) != len(after),
        before.duplicated(KEYS).any(),
        after.duplicated(KEYS).any(),
        any(metrics[f"{column}_changed"] for column in DATE_COLUMNS),
    ]
    if conflicts is not None and log is not None:
        inherited = set(
            zip(
                conflicts[KEYS[0]].astype(str),
                conflicts[KEYS[1]].astype(str),
                conflicts["variable"].astype(str),
            )
        )
        audited = set(
            zip(
                log.loc[log["conflict_origin"].eq("09c"), KEYS[0]].astype(str),
                log.loc[log["conflict_origin"].eq("09c"), KEYS[1]].astype(str),
                log.loc[log["conflict_origin"].eq("09c"), "variable_name"].astype(str),
            )
        )
        failures.append(inherited != audited)
    if any(failures):
        raise ValueError(f"Hard episode-architecture/audit QC failed: {metrics}")
    return metrics


def validate_analytical_pipes(frame: pd.DataFrame) -> None:
    """Reject unresolved pipes in scalar analytical columns."""
    offenders = [
        c
        for c in frame
        if c not in PROVENANCE_PIPE_COLUMNS | set(IMMUTABLE)
        and variable_group(c) != "biopsy_pathology"
        and frame[c].dtype == object
        and frame[c].fillna("").astype(str).str.contains("|", regex=False).any()
    ]
    if offenders:
        raise ValueError(
            f"Scalar analytical columns retain pipe conflicts: {offenders[:20]}"
        )


def parquet_compatible(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert mixed object columns to strings for deterministic parquet output."""
    output = frame.copy()
    for column in output.select_dtypes(include=["object"]):
        populated = output[column].dropna()
        if (
            not populated.empty
            and populated.map(lambda value: isinstance(value, str)).any()
            and not populated.map(lambda value: isinstance(value, str)).all()
        ):
            output[column] = output[column].map(
                lambda value: pd.NA if is_missing(value) else display(value)
            )
    return output


def finalize(
    collapsed: pd.DataFrame, sources: pd.DataFrame, conflicts: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return final analytical data, complete audit log, and manual-review queue."""
    records: list[dict[str, object]] = []
    output = apply_essdai(collapsed, sources, records)
    output = process_09c_conflicts(output, collapsed, sources, conflicts, records)
    output = resolve_residual_pipes(output, sources, records)
    log = pd.DataFrame(records).reindex(columns=LOG_COLUMNS)
    unresolved = log[log["resolution_status"].eq("unresolved")]
    unresolved_keys = set(map(tuple, unresolved[KEYS].to_numpy()))
    essdai_keys = set(
        map(
            tuple,
            unresolved.loc[
                unresolved["canonical_variable"].str.startswith(
                    ESSDAI_PREFIX, na=False
                ),
                KEYS,
            ].to_numpy(),
        )
    )
    output["episode_has_unresolved_conflict"] = [
        tuple(x) in unresolved_keys for x in output[KEYS].to_numpy()
    ]
    output["essdai_has_unresolved_conflict"] = [
        tuple(x) in essdai_keys for x in output[KEYS].to_numpy()
    ]
    output["analytic_resolution_quality"] = [
        "unresolved" if tuple(x) in unresolved_keys else "conservative_resolution"
        for x in output[KEYS].to_numpy()
    ]
    return add_essdai_totals(output), log, unresolved.reindex(columns=REVIEW_COLUMNS)


def summary_tables(
    before: pd.DataFrame,
    after: pd.DataFrame,
    conflicts: pd.DataFrame,
    log: pd.DataFrame,
    hard: dict[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build internally consistent general and ESSDAI QC summaries."""
    status = log["resolution_status"].value_counts()
    inherited = log["conflict_origin"].eq("09c")
    harmonized = log["conflict_origin"].eq("09d_harmonization")
    values = {
        **hard,
        "conflicts_from_09c": int(inherited.sum()),
        "new_conflicts_detected_in_09d": int(harmonized.sum()),
        "total_conflicts_processed": len(log),
        "conflicts_resolved": int(status.get("resolved", 0)),
        "conflicts_unresolved": int(status.get("unresolved", 0)),
        "conflicts_preserved_as_multiple": int(
            status.get("preserved_multiple_values", 0)
        ),
        "episodes_with_unresolved_conflicts": int(
            after["episode_has_unresolved_conflict"].sum()
        ),
    }
    ess_log = log[log["canonical_variable"].str.startswith(ESSDAI_PREFIX, na=False)]
    legacy = [c for c in before if c.startswith(ESSDAI_R_PREFIX)]
    canonical = [c for c in before if c.startswith(ESSDAI_PREFIX)]
    after_cols = [c for c in after if c.startswith(ESSDAI_PREFIX)]
    ess_values = {
        "essdai_variable_conflicts_total": len(ess_log),
        "essdai_variable_conflicts_resolved": int(
            ess_log["resolution_status"].eq("resolved").sum()
        ),
        "essdai_variable_conflicts_unresolved": int(
            ess_log["resolution_status"].eq("unresolved").sum()
        ),
        "essdai_total_recorded_valid": int(
            after["essdai_total_source"].eq("recorded_valid").sum()
        ),
        "essdai_total_derived": int(
            after["essdai_total_source"].eq("derived_from_domains").sum()
        ),
        "essdai_total_unresolved": int(
            after["essdai_total_source"].eq("unresolved").sum()
        ),
        "essdai_total_missing": int(after["essdai_total_source"].eq("missing").sum()),
        "episodes_with_essdai_before": (
            int(before[legacy + canonical].notna().any(axis=1).sum())
            if legacy or canonical
            else 0
        ),
        "episodes_with_essdai_after": (
            int(after[after_cols].notna().any(axis=1).sum()) if after_cols else 0
        ),
    }
    return (
        pd.DataFrame({"metric": values.keys(), "value": values.values()}),
        pd.DataFrame({"metric": ess_values.keys(), "value": ess_values.values()}),
    )


def main() -> None:
    """Run finalization and write validated analytical and audit artifacts."""
    args, logger = parse_args(), setup_logger("09d_finalize_clinical_episode_dataset")
    for path in (
        args.input_path,
        args.visits_path,
        args.row_map_path,
        args.manifest_path,
        args.conflict_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    before, manifest, conflicts = (
        pd.read_parquet(args.input_path),
        pd.read_parquet(args.manifest_path),
        pd.read_csv(args.conflict_path),
    )
    if missing := set(IMMUTABLE).difference(before):
        raise ValueError(f"09c input missing authoritative columns: {sorted(missing)}")
    if set(manifest[KEYS[1]].astype(str)) != set(before[KEYS[1]].astype(str)):
        raise ValueError("09c input and manifest episode IDs differ")
    sources = load_sources(args.visits_path, args.row_map_path)
    after, log, review = finalize(before, sources, conflicts)
    hard = hard_qc(before, after, conflicts, log)
    validate_analytical_pipes(after)
    qc, essdai_qc = summary_tables(before, after, conflicts, log, hard)
    args.output_base.parent.mkdir(parents=True, exist_ok=True)
    args.qc_dir.mkdir(parents=True, exist_ok=True)
    serialized = parquet_compatible(after)
    serialized.to_parquet(args.output_base.with_suffix(".parquet"), index=False)
    serialized.to_csv(args.output_base.with_suffix(".csv"), index=False)
    log.to_csv(args.qc_dir / "09d_conflict_resolution_log.csv", index=False)
    log[log["resolution_status"].eq("unresolved")].to_csv(
        args.qc_dir / "09d_unresolved_conflicts.csv", index=False
    )
    review.to_csv(args.qc_dir / "09d_manual_review_queue.csv", index=False)
    qc.to_csv(args.qc_dir / "09d_qc_summary.csv", index=False)
    essdai_qc.to_csv(args.qc_dir / "09d_essdai_harmonization_summary.csv", index=False)
    multiple = log[log["resolution_status"].eq("preserved_multiple_values")]
    if not multiple.empty:
        multiple.to_csv(args.qc_dir / "09d_multi_specimen_records.csv", index=False)
    logger.info(
        "Finalized %d episodes for %d patients; conflicts=%d unresolved=%d",
        len(after),
        after[KEYS[0]].nunique(),
        len(log),
        len(review),
    )


if __name__ == "__main__":
    main()
