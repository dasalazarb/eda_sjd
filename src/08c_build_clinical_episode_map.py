"""Build a row-level clinical episode map and an episode manifest.

This step is operational QC only: it does not alter interval labels or define the
official baseline.  Rows are joined only when their populated content is
complementary and their dates meet the documented seven- or fourteen-day rule.
"""

from __future__ import annotations

import argparse
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
    parser.add_argument("--previous-episodes-path", type=Path, default=PREVIOUS_EPISODES_PATH)
    return parser.parse_args()


def resolve_column(df: pd.DataFrame, names: Iterable[str], required: bool = True) -> str | None:
    """Resolve the first exact or uniquely group-prefixed column name."""
    for name in names:
        if name in df.columns:
            return name
        matches = [str(column) for column in df.columns if str(column).endswith(f"__{name}")]
        if len(matches) == 1:
            return matches[0]
    if required:
        raise ValueError(f"Required column not found; tried {list(names)}")
    return None


def has_information(series: pd.Series) -> pd.Series:
    """Identify valid populated values without treating zero or negative answers as missing."""
    result = series.notna()
    if pd.api.types.is_object_dtype(series.dtype) or pd.api.types.is_string_dtype(series.dtype):
        text = series.astype("string").str.strip()
        result &= text.notna() & ~text.str.upper().isin(MISSING_UPPER)
    return result.fillna(False)


def _columns_for_prefixes(df: pd.DataFrame, prefixes: Iterable[str]) -> list[str]:
    accepted = set(prefixes)
    return [
        str(column)
        for column in df.columns
        if "__" in str(column) and str(column).split("__", 1)[0].strip().lower() in accepted
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
    result["objective_exam_count"] = result[list(OBJECTIVE_FLAGS)].sum(axis=1).astype(int)
    result["clinical_candidate"] = (
        result["has_essdai_form"]
        | (result["physician_core_count"] >= 2)
        | ((result["physician_core_count"] >= 1) & (result["objective_exam_count"] >= 1))
        | (result["objective_exam_count"] >= 2)
    )
    result["has_research_component"] = result[list(RESEARCH_PREFIXES)].any(axis=1)
    result["has_any_clinical_evidence"] = result[
        list(dict.fromkeys((*CORE_FLAGS, *OBJECTIVE_FLAGS, *SUPPORT_FLAGS)))
    ].any(axis=1)
    return result


def _aggregate_rows(rows: pd.DataFrame) -> pd.Series:
    flags = list(BLOCK_PREFIXES) + ["has_essdai_total", "has_esspri_core"] + list(RESEARCH_PREFIXES)
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
    joins_essdai_esspri = bool(combined.has_essdai_form and combined.has_esspri_form) and not (
        bool(before.has_essdai_form and before.has_esspri_form)
        or bool(incoming.has_essdai_form and incoming.has_esspri_form)
    )
    becomes_clinical = bool(combined.clinical_candidate) and not bool(before.clinical_candidate)
    completes_clinical = bool(before.clinical_candidate) and bool(incoming.has_any_clinical_evidence)
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
    source_intervals = (
        prepared_rows.groupby(
            ["patient_id", "interval_name"], as_index=False, dropna=False
        )
        .agg(
            source_interval_start_date=("collection_date", "min"),
            source_interval_end_date=("collection_date", "max"),
        )
    )
    source_intervals["source_interval_span_days"] = (
        source_intervals["source_interval_end_date"]
        - source_intervals["source_interval_start_date"]
    ).dt.days.astype("Int64")
    source_intervals["source_interval_span_gt30"] = source_intervals[
        "source_interval_span_days"
    ].gt(30).fillna(False)
    return source_intervals


def _source_interval_key(patient_id: object, interval_name: object) -> tuple[object, str]:
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
                    left.source_interval_start_date
                    <= right.source_interval_end_date
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
        research_only = bool(evidence.has_research_component and not has_core_or_objective)
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
        "patient_id", "clinical_episode_id", "intervals_involved", "episode_start_date",
        "clinical_anchor_date", "episode_end_date", "episode_span_days", "has_essdai_form",
        "has_essdai_total", "has_esspri_form", "has_esspri_core", "has_systems_review",
        "has_physical_exam", "has_visit_summary", "has_eye_exam", "has_salivary_flow",
        "has_oral_exam", "physician_core_count", "objective_exam_count", "visit_type",
        "clinical_visit", "manual_review_required", "manual_review_reason",
        "max_source_interval_span_days",
    ]
    return pd.DataFrame(records).reindex(columns=columns)


def prepare_visits(visits: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Normalize identifiers and dates while retaining original provenance columns."""
    patient_col = resolve_column(visits, ("patient_id", "patient_record_number"))
    interval_col = resolve_column(visits, ("interval_name",))
    date_col = resolve_column(visits, ("collection_date", "visit_date", "visit_datetime"))
    row_col = resolve_column(visits, ("row_id_raw",), required=False)
    provenance = [
        str(column) for column in visits.columns
        if any(term in str(column).lower() for term in ("protocol", "origin", "source"))
    ]
    result = visits.copy()
    result["_source_order"] = range(len(result))
    result["patient_id"] = result[patient_col]
    result["interval_name"] = result[interval_col]
    result["collection_date"] = pd.to_datetime(result[date_col], errors="coerce").dt.normalize()
    result["row_id_raw"] = result[row_col] if row_col else result["_source_order"]
    if result["patient_id"].isna().any():
        raise ValueError("patient_id contains missing values; episode IDs cannot be made safely")
    if result["row_id_raw"].isna().any() or result["row_id_raw"].duplicated().any():
        raise ValueError("row_id_raw must be complete and unique")
    return result, list(dict.fromkeys(provenance))


def build_qc(
    assigned_rows: pd.DataFrame,
    assigned_units: pd.DataFrame,
    manifest: pd.DataFrame,
    source_intervals: pd.DataFrame,
) -> pd.DataFrame:
    """Create the requested one-row QC metric table."""
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
        "research_or_procedure_only_candidate": manifest["visit_type"].eq(
            "research_or_procedure_only_candidate"
        ).sum(),
        "ambiguous": manifest["visit_type"].eq("ambiguous").sum(),
        "manual_review": manifest["manual_review_required"].sum(),
        "source_intervals_span_gt30": len(long_source_intervals),
        "patients_with_source_interval_span_gt30": long_source_intervals[
            "patient_id"
        ].nunique(),
        "clinical_episodes_with_source_interval_span_gt30": has_long_source_interval.sum(),
        "clinical_episodes_with_overlapping_source_interval_ranges": has_source_overlap.sum(),
        "episodes_with_multiple_intervals": manifest["intervals_involved"].str.contains(
            " \\| ", regex=True, na=False
        ).sum(),
        "episodes_with_essdai_and_esspri": reunited.sum(),
        "raw_rows_unassigned": assigned_rows["clinical_episode_id"].isna().sum(),
        "raw_rows_multiply_assigned": (duplicated_assignments > 1).sum(),
        "patient_date_units_assigned_to_multiple_episodes": (split_units > 1).sum(),
    }
    return pd.DataFrame([metrics])


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
        on=["patient_id", "interval_name"], how="left",
    )
    return (
        comparison.groupby([previous_type, "visit_type"], dropna=False)
        .size().rename("episodes").reset_index().rename(columns={previous_type: "previous_visit_type"})
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
    assigned_rows = propagate_episode_assignments(flagged_rows, assigned_units)
    manifest = build_manifest(assigned_rows, source_intervals)
    qc = build_qc(assigned_rows, assigned_units, manifest, source_intervals)
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
    row_columns = [
        "patient_id", "row_id_raw", "interval_name", "collection_date",
        "daily_activity_unit_id",
        *provenance, "clinical_episode_id", "assignment_rule", "manual_review_required",
    ]
    assigned_rows = assigned_rows.merge(
        manifest[["clinical_episode_id", "manual_review_required"]],
        on="clinical_episode_id", how="left", validate="many_to_one",
    )
    args.qc_dir.mkdir(parents=True, exist_ok=True)
    row_map_paths = write_parquet_and_csv(
        assigned_rows[row_columns], args.row_map_path
    )
    manifest_paths = write_parquet_and_csv(manifest, args.manifest_path)
    qc.to_csv(args.qc_dir / "08c_qc_summary.csv", index=False)
    source_intervals.to_csv(
        args.qc_dir / "08c_source_interval_qc.csv", index=False
    )
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
