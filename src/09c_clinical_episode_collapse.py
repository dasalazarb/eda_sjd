"""Collapse raw study activities within clinical episodes defined by step 08c.

This parallel pipeline step does not define visits, eligibility, or temporal
windows.  It only combines variables inside the authoritative
``patient_id x clinical_episode_id`` assignments produced by step 08c.
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
    print_kv,
    print_script_overview,
    print_step,
    save_parquet_and_csv,
    setup_logger,
)

INPUT_PATH = ANALYTIC_DIR / "visits_long.parquet"
ROW_MAP_PATH = INTERMEDIATE_DIR / "clinical_episode_row_map.parquet"
MANIFEST_PATH = ANALYTIC_DIR / "clinical_episode_manifest.parquet"
OUTPUT_BASE = (
    ANALYTIC_DIR / "visits_long_collapsed_by_clinical_episode_codebook_not_clean"
)
QC_DIR = REPORTS_DIR / "clinical_episode_collapse"
OLD_COLLAPSE_PATH = (
    ANALYTIC_DIR / "visits_long_collapsed_by_interval_codebook_not_clean.parquet"
)

ROW_MAP_REQUIRED = {
    "patient_id",
    "row_id_raw",
    "clinical_episode_id",
    "interval_name",
    "collection_date",
    "assignment_rule",
    "manual_review_required",
}
MANIFEST_REQUIRED = {
    "patient_id",
    "clinical_episode_id",
    "intervals_involved",
    "episode_start_date",
    "clinical_anchor_date",
    "episode_end_date",
    "episode_span_days",
    "visit_type",
    "clinical_visit",
    "manual_review_required",
    "manual_review_reason",
}
MANIFEST_OPTIONAL = (
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
    "clinical_candidate",
    "research_or_procedure_only_candidate",
    "ambiguous",
    "has_research_component",
)
AUTHORITATIVE_COLUMNS = tuple(MANIFEST_REQUIRED) + MANIFEST_OPTIONAL
DATE_COLUMNS = ("episode_start_date", "clinical_anchor_date", "episode_end_date")
COMPATIBILITY_DATE_COLUMNS = ("ids__visit_date", "visit_date", "visit_datetime")
NON_CONFLICT_PROVENANCE = {
    "row_id_raw",
    "assignment_rule",
    "interval_name",
    "ids__interval_name",
    "source_protocol",
    "source_file",
    "origin",
    "collection_date",
    # Row-processing metadata varies by design inside a clinical episode and
    # therefore must not inflate the clinical variable-conflict report.
    "ids__time_24_hour",
    "ids__subject_number",
    "time_24_hour",
    "duplicate_group_id",
    "dup_rank",
    "visit_datetime_adjustment_seconds",
}
ANS_PREFIX = "ans__"
AUTONOMIC_PREFIX = "autonomic_nervous_system_questionnaire__"
MISSING_UPPER = {str(value).strip().upper() for value in MISSING_TOKENS}
STRICT_NUMBER = re.compile(r"^[+-]?(?:0|[1-9]\d*)(?:\.\d+)?$")


def parse_args() -> argparse.Namespace:
    """Parse command-line paths.

    Returns
    -------
    argparse.Namespace
        Parsed paths for inputs, analytical outputs, and QC reports.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, default=INPUT_PATH)
    parser.add_argument("--row-map-path", type=Path, default=ROW_MAP_PATH)
    parser.add_argument("--manifest-path", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--output-base", type=Path, default=OUTPUT_BASE)
    parser.add_argument("--qc-dir", type=Path, default=QC_DIR)
    return parser.parse_args()


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _resolve_patient_column(visits: pd.DataFrame) -> str:
    candidates = (
        "patient_id",
        "ids__patient_record_number",
        "patient_record_number",
    )
    found = [column for column in candidates if column in visits.columns]
    if not found:
        raise ValueError(
            "visits_long has no patient identifier; expected one of "
            f"{list(candidates)}"
        )
    return found[0]


def _is_missing(value: object) -> bool:
    if pd.isna(value):
        return True
    return isinstance(value, str) and value.strip().upper() in MISSING_UPPER


def _display_value(value: object) -> str:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value).strip()


def _comparison_key(value: object, numeric_column: bool) -> tuple[str, object]:
    """Return a conservative equality key without changing displayed values."""
    if numeric_column:
        try:
            return ("number", float(value))
        except (TypeError, ValueError):
            pass
    text = _display_value(value)
    # Treat plain decimal spellings such as 1 and 1.0 as equal. Leading-zero
    # identifiers deliberately do not match this expression.
    if STRICT_NUMBER.fullmatch(text):
        return ("number", float(text))
    parsed = pd.to_datetime(text, errors="coerce")
    if not pd.isna(parsed) and any(token in text for token in ("-", "/", ":")):
        return ("date", parsed)
    return ("text", text)


def _unique_values(series: pd.Series) -> tuple[list[object], int]:
    values = [value for value in series.tolist() if not _is_missing(value)]
    numeric_column = pd.api.types.is_numeric_dtype(series.dtype)
    unique: dict[tuple[str, object], object] = {}
    for value in values:
        unique.setdefault(_comparison_key(value, numeric_column), value)
    return list(unique.values()), len(values)


def _collapse_series(series: pd.Series) -> tuple[object, list[object], int]:
    unique, nonmissing_count = _unique_values(series)
    if not unique:
        return pd.NA, unique, nonmissing_count
    if len(unique) == 1:
        return unique[0], unique, nonmissing_count
    return (
        " | ".join(_display_value(value) for value in unique),
        unique,
        nonmissing_count,
    )


def _date_series(rows: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(rows["collection_date_map"], errors="coerce").dt.normalize()


def _collapse_age(
    rows: pd.DataFrame, column: str, anchor_date: object
) -> tuple[object, list[object], int]:
    dates = _date_series(rows)
    anchor = pd.to_datetime(anchor_date, errors="coerce")
    if not pd.isna(anchor):
        anchored = rows.loc[dates.eq(anchor.normalize()), column]
        anchored_result = _collapse_series(anchored)
        if anchored_result[2] > 0:
            return anchored_result
    return _collapse_series(rows[column])


def validate_inputs(
    visits: pd.DataFrame, row_map: pd.DataFrame, manifest: pd.DataFrame
) -> tuple[str, dict[str, int]]:
    """Validate complete, one-to-one row assignment and manifest episode keys.

    Parameters
    ----------
    visits, row_map, manifest : pd.DataFrame
        Step input tables.

    Returns
    -------
    tuple[str, dict[str, int]]
        Source patient column and assignment QC counters.
    """
    _require_columns(visits, {"row_id_raw"}, "visits_long")
    _require_columns(row_map, ROW_MAP_REQUIRED, "clinical_episode_row_map")
    _require_columns(manifest, MANIFEST_REQUIRED, "clinical_episode_manifest")
    patient_column = _resolve_patient_column(visits)
    if visits["row_id_raw"].isna().any() or visits["row_id_raw"].duplicated().any():
        raise ValueError("row_id_raw must be non-missing and unique in visits_long")
    if row_map["row_id_raw"].isna().any() or row_map["row_id_raw"].duplicated().any():
        raise ValueError(
            "Each row_id_raw must appear exactly once in clinical_episode_row_map"
        )
    if row_map["clinical_episode_id"].isna().any():
        raise ValueError(
            "clinical_episode_row_map contains missing clinical_episode_id"
        )
    manifest_duplicates = manifest["clinical_episode_id"].duplicated(keep=False)
    if manifest_duplicates.any():
        raise ValueError(
            "clinical_episode_id must be globally unique in clinical_episode_manifest"
        )
    visit_ids = set(visits["row_id_raw"])
    map_counts = row_map["row_id_raw"].value_counts()
    unassigned = len(visit_ids.difference(set(row_map["row_id_raw"])))
    multiply_assigned = int(map_counts.gt(1).sum())
    extra = set(row_map["row_id_raw"]).difference(visit_ids)
    if extra:
        raise ValueError(
            f"Row map contains {len(extra)} row_id_raw values absent from visits_long"
        )
    if unassigned or multiply_assigned:
        raise ValueError(
            f"Invalid assignments: raw_rows_unassigned={unassigned}, "
            f"raw_rows_multiply_assigned={multiply_assigned}"
        )
    patient_check = visits[["row_id_raw", patient_column]].merge(
        row_map[["row_id_raw", "patient_id"]], on="row_id_raw", validate="one_to_one"
    )
    mismatch = (
        ~patient_check[patient_column]
        .astype("string")
        .eq(patient_check["patient_id"].astype("string"))
    )
    if mismatch.any():
        examples = patient_check.loc[mismatch, "row_id_raw"].head(5).tolist()
        raise ValueError(
            "patient_id in row map disagrees with visits_long for row_id_raw "
            f"examples: {examples}"
        )
    return patient_column, {
        "raw_rows_unassigned": unassigned,
        "raw_rows_multiply_assigned": multiply_assigned,
    }


def join_assignments(
    visits: pd.DataFrame, row_map: pd.DataFrame, patient_column: str
) -> pd.DataFrame:
    """Join authoritative episode assignments to source rows by row_id_raw."""
    map_columns = [
        "row_id_raw",
        "patient_id",
        "clinical_episode_id",
        "assignment_rule",
        "interval_name",
        "collection_date",
        "manual_review_required",
    ]
    renamed = row_map[map_columns].rename(
        columns={
            "patient_id": "patient_id_map",
            "interval_name": "interval_name_map",
            "collection_date": "collection_date_map",
            "manual_review_required": "manual_review_required_map",
        }
    )
    joined = visits.merge(renamed, on="row_id_raw", how="left", validate="one_to_one")
    joined["patient_id"] = joined[patient_column]
    if joined["clinical_episode_id"].isna().any():
        raise ValueError("Join left visits_long rows without a clinical_episode_id")
    return joined


def _generic_columns(joined: pd.DataFrame) -> list[str]:
    excluded = {
        "patient_id",
        "patient_id_map",
        "clinical_episode_id",
        "collection_date_map",
        "interval_name_map",
        "manual_review_required_map",
        *AUTHORITATIVE_COLUMNS,
        *COMPATIBILITY_DATE_COLUMNS,
    }
    return [column for column in joined.columns if column not in excluded]


def collapse_episodes(
    joined: pd.DataFrame, manifest: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collapse variables within existing episode assignments conservatively."""
    manifest_lookup = manifest.set_index(["patient_id", "clinical_episode_id"])
    columns = _generic_columns(joined)
    records: list[dict[str, object]] = []
    conflict_records: list[dict[str, object]] = []
    for key, rows in joined.groupby(
        ["patient_id", "clinical_episode_id"], sort=False, dropna=False
    ):
        patient_id, episode_id = key
        if key not in manifest_lookup.index:
            raise ValueError(f"Assigned episode {key} is absent from manifest")
        manifest_row = manifest_lookup.loc[key]
        record: dict[str, object] = {
            "patient_id": patient_id,
            "clinical_episode_id": episode_id,
        }
        for date_column in COMPATIBILITY_DATE_COLUMNS:
            if date_column in joined.columns:
                record[date_column] = pd.NA
        for column in columns:
            if "age_at_visit" in str(column).lower():
                value, unique, n_nonmissing = _collapse_age(
                    rows, column, manifest_row["clinical_anchor_date"]
                )
            else:
                value, unique, n_nonmissing = _collapse_series(rows[column])
            record[column] = value
            if len(unique) <= 1 or column in NON_CONFLICT_PROVENANCE:
                continue
            populated = rows.loc[~rows[column].map(_is_missing)]
            conflict_records.append(
                {
                    "patient_id": patient_id,
                    "clinical_episode_id": episode_id,
                    "clinical_anchor_date": manifest_row["clinical_anchor_date"],
                    "intervals_involved": manifest_row["intervals_involved"],
                    "variable": column,
                    "n_nonmissing_values": n_nonmissing,
                    "n_unique_values": len(unique),
                    "observed_values": " | ".join(
                        _display_value(item) for item in unique
                    ),
                    "collection_dates": _join_dates(populated["collection_date_map"]),
                    "row_ids_involved": _join_unique(populated["row_id_raw"]),
                    "visit_type": manifest_row["visit_type"],
                    "clinical_visit": manifest_row["clinical_visit"],
                    "manual_review_required": manifest_row["manual_review_required"],
                    "manual_review_reason": manifest_row["manual_review_reason"],
                }
            )
        record["n_raw_rows_in_episode"] = len(rows)
        record["n_collection_dates_in_episode"] = _date_series(rows).nunique()
        records.append(record)
    conflicts = pd.DataFrame(conflict_records, columns=_conflict_columns())
    return pd.DataFrame(records), conflicts


def _join_unique(values: Iterable[object]) -> str:
    unique = dict.fromkeys(
        _display_value(value) for value in values if not _is_missing(value)
    )
    return " | ".join(unique)


def _join_dates(values: Iterable[object]) -> str:
    dates = pd.to_datetime(pd.Series(list(values)), errors="coerce").dropna()
    return " | ".join(dict.fromkeys(dates.dt.strftime("%Y-%m-%d")))


def _conflict_columns() -> list[str]:
    return [
        "patient_id",
        "clinical_episode_id",
        "clinical_anchor_date",
        "intervals_involved",
        "variable",
        "n_nonmissing_values",
        "n_unique_values",
        "observed_values",
        "collection_dates",
        "row_ids_involved",
        "visit_type",
        "clinical_visit",
        "manual_review_required",
        "manual_review_reason",
    ]


def add_manifest(collapsed: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Attach authoritative episode metadata and anchor compatibility dates."""
    columns = [
        column
        for column in AUTHORITATIVE_COLUMNS
        if column in manifest.columns
        and column not in {"patient_id", "clinical_episode_id"}
    ]
    output = collapsed.merge(
        manifest[["patient_id", "clinical_episode_id", *columns]],
        on=["patient_id", "clinical_episode_id"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    bad = output["_merge"].ne("both")
    if bad.any():
        raise ValueError(
            f"Manifest/assigned episode mismatch for {int(bad.sum())} episodes"
        )
    output = output.drop(columns="_merge")
    for column in COMPATIBILITY_DATE_COLUMNS:
        if column in collapsed.columns:
            output[column] = output["clinical_anchor_date"]
    return output


def merge_ans_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Merge only matching ANS/autonomic suffixes, preserving legacy schema logic."""
    output = frame.copy()
    ans = {
        column[len(ANS_PREFIX) :]: column
        for column in output
        if column.startswith(ANS_PREFIX)
    }
    autonomic = {
        column[len(AUTONOMIC_PREFIX) :]: column
        for column in output
        if column.startswith(AUTONOMIC_PREFIX)
    }
    for suffix in sorted(set(ans).intersection(autonomic)):
        target, source = ans[suffix], autonomic[suffix]
        output[target] = output[[target, source]].apply(
            lambda row: _collapse_series(row)[0], axis=1
        )
        output = output.drop(columns=source)
    return output


def make_parquet_compatible(frame: pd.DataFrame) -> pd.DataFrame:
    """Make mixed object columns safe for Arrow without changing missing values.

    Conservative collapsing can legitimately put a numeric singleton in one
    episode and a pipe-delimited conflict string in another. Pandas consequently
    stores that column as ``object``, while Arrow requires one physical type per
    column. When an object column mixes text (or bytes) with another Python type,
    all populated values are serialized using their existing display form. Pure
    numeric, boolean, datetime, and text object columns are left unchanged.

    Parameters
    ----------
    frame : pd.DataFrame
        Collapsed episode table before Parquet serialization.

    Returns
    -------
    pd.DataFrame
        Copy whose heterogeneous text-like object columns have consistent values.
    """
    output = frame.copy()
    for column in output.select_dtypes(include=["object"]).columns:
        populated = output[column].dropna()
        if populated.empty:
            continue
        has_text = populated.map(lambda value: isinstance(value, str)).any()
        all_text = populated.map(lambda value: isinstance(value, str)).all()
        has_bytes = populated.map(lambda value: isinstance(value, bytes)).any()
        all_bytes = populated.map(lambda value: isinstance(value, bytes)).all()
        if (has_text and not all_text) or (has_bytes and not all_bytes):
            output[column] = output[column].map(
                lambda value: pd.NA if _is_missing(value) else _display_value(value)
            )
    return output


def hard_qc(output: pd.DataFrame, manifest: pd.DataFrame) -> int:
    """Enforce unique output keys, episode completeness, and ordered dates."""
    duplicates = int(
        output.duplicated(["patient_id", "clinical_episode_id"], keep=False).sum()
    )
    if duplicates:
        raise ValueError(f"Output has {duplicates} duplicate patient-episode rows")
    if len(output) != len(manifest):
        raise ValueError(
            f"Output episodes ({len(output)}) != manifest episodes ({len(manifest)})"
        )
    parsed = output[list(DATE_COLUMNS)].apply(pd.to_datetime, errors="coerce")
    complete = parsed.notna().all(axis=1)
    invalid = complete & ~(
        parsed["episode_start_date"].le(parsed["clinical_anchor_date"])
        & parsed["clinical_anchor_date"].le(parsed["episode_end_date"])
    )
    if invalid.any():
        raise ValueError(
            f"Manifest date ordering fails for {int(invalid.sum())} episodes"
        )
    return duplicates


def _episode_flag(frame: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    present = pd.Series(False, index=frame.index)
    for column in columns:
        if column in frame.columns:
            present |= frame[column].map(lambda value: not _is_missing(value))
    return present


def _essdai(frame: pd.DataFrame) -> pd.Series:
    if "has_essdai_form" in frame.columns:
        return frame["has_essdai_form"].fillna(False).astype(bool)
    return _episode_flag(
        frame, [c for c in frame if str(c).startswith(("essdai__", "essdai-_r__"))]
    )


def _esspri(frame: pd.DataFrame) -> pd.Series:
    if "has_esspri_form" in frame.columns:
        return frame["has_esspri_form"].fillna(False).astype(bool)
    return _episode_flag(
        frame, [c for c in frame if str(c).startswith("esspri_questionnaire__")]
    )


def build_qc(
    visits: pd.DataFrame,
    joined: pd.DataFrame,
    manifest: pd.DataFrame,
    output: pd.DataFrame,
    conflicts: pd.DataFrame,
    assignment_qc: dict[str, int],
) -> pd.DataFrame:
    """Build the required scalar QC summary."""
    essdai, esspri = _essdai(output), _esspri(output)
    conflicting_episodes = (
        conflicts[["patient_id", "clinical_episode_id"]].drop_duplicates().shape[0]
    )
    interval_multiple = (
        output["intervals_involved"]
        .fillna("")
        .astype(str)
        .str.contains(" | ", regex=False)
    )
    values = {
        "patients_input": joined["patient_id"].nunique(),
        "patients_output": output["patient_id"].nunique(),
        "raw_rows_input": len(visits),
        "raw_rows_assigned": joined["clinical_episode_id"].notna().sum(),
        "clinical_episodes_manifest": len(manifest),
        "clinical_episodes_output": len(output),
        "clinical_candidate": output["visit_type"].eq("clinical_candidate").sum(),
        "research_or_procedure_only_candidate": output["visit_type"]
        .eq("research_or_procedure_only_candidate")
        .sum(),
        "ambiguous": output["visit_type"].eq("ambiguous").sum(),
        "manual_review_required": output["manual_review_required"]
        .fillna(False)
        .astype(bool)
        .sum(),
        "episodes_single_raw_row": output["n_raw_rows_in_episode"].eq(1).sum(),
        "episodes_multiple_raw_rows": output["n_raw_rows_in_episode"].gt(1).sum(),
        "episodes_single_collection_date": output["n_collection_dates_in_episode"]
        .eq(1)
        .sum(),
        "episodes_multiple_collection_dates": output["n_collection_dates_in_episode"]
        .gt(1)
        .sum(),
        "episodes_multiple_intervals": interval_multiple.sum(),
        "episodes_with_essdai": essdai.sum(),
        "episodes_with_esspri": esspri.sum(),
        "episodes_with_essdai_and_esspri": (essdai & esspri).sum(),
        "episodes_with_any_variable_conflict": conflicting_episodes,
        "number_variable_conflicts": len(conflicts),
        **assignment_qc,
        "duplicate_patient_episode_rows_output": output.duplicated(
            ["patient_id", "clinical_episode_id"], keep=False
        ).sum(),
    }
    return pd.DataFrame({"metric": values.keys(), "value": values.values()})


def build_row_counts(output: pd.DataFrame) -> pd.DataFrame:
    """Select episode complexity and authoritative manifest fields for review."""
    columns = [
        "patient_id",
        "clinical_episode_id",
        "intervals_involved",
        "n_raw_rows_in_episode",
        "n_collection_dates_in_episode",
        "episode_start_date",
        "clinical_anchor_date",
        "episode_end_date",
        "episode_span_days",
        "visit_type",
        "clinical_visit",
        "manual_review_required",
    ]
    return output[columns].copy()


def build_before_after(
    old: pd.DataFrame, new: pd.DataFrame, conflicts: pd.DataFrame
) -> pd.DataFrame:
    """Compare aggregate structure because old and new row units differ."""
    old_patient = _resolve_patient_column(old)
    old_essdai, old_esspri = _essdai(old), _esspri(old)
    new_essdai, new_esspri = _essdai(new), _esspri(new)
    old_interval = next(
        (c for c in ("ids__interval_name", "interval_name") if c in old), None
    )
    old_multiple = (
        0
        if old_interval is None
        else old[old_interval]
        .fillna("")
        .astype(str)
        .str.contains(" | ", regex=False)
        .sum()
    )
    conflict_keys = set(
        map(tuple, conflicts[["patient_id", "clinical_episode_id"]].to_numpy())
    )
    new_conflicted = sum(
        tuple(key) in conflict_keys
        for key in new[["patient_id", "clinical_episode_id"]].to_numpy()
    )
    metrics = {
        "patients": (old[old_patient].nunique(), new["patient_id"].nunique()),
        "rows / episodes": (len(old), len(new)),
        "ESSDAI available": (old_essdai.sum(), new_essdai.sum()),
        "ESSPRI available": (old_esspri.sum(), new_esspri.sum()),
        "ESSDAI + ESSPRI available": (
            (old_essdai & old_esspri).sum(),
            (new_essdai & new_esspri).sum(),
        ),
        "number of records with multiple interval names": (
            old_multiple,
            new["intervals_involved"]
            .fillna("")
            .astype(str)
            .str.contains(" | ", regex=False)
            .sum(),
        ),
        "number of records with pipe-conflicted variables": (pd.NA, new_conflicted),
        "clinical_candidate": (pd.NA, new["visit_type"].eq("clinical_candidate").sum()),
        "research_only": (
            pd.NA,
            new["visit_type"].eq("research_or_procedure_only_candidate").sum(),
        ),
        "ambiguous": (pd.NA, new["visit_type"].eq("ambiguous").sum()),
        "manual_review": (
            pd.NA,
            new["manual_review_required"].fillna(False).astype(bool).sum(),
        ),
    }
    rows = []
    for metric, (old_value, new_value) in metrics.items():
        difference = pd.NA if pd.isna(old_value) else new_value - old_value
        rows.append(
            {
                "metric": metric,
                "old_interval_collapse": old_value,
                "new_clinical_episode_collapse": new_value,
                "difference": difference,
            }
        )
    return pd.DataFrame(rows)


def build_review_examples(
    output: pd.DataFrame, conflicts: pd.DataFrame
) -> pd.DataFrame:
    """Select a few representative episodes without changing source data."""
    candidates: list[tuple[str, pd.Series]] = []
    essdai, esspri = _essdai(output), _esspri(output)
    intervals = (
        output["intervals_involved"]
        .fillna("")
        .astype(str)
        .str.contains(" | ", regex=False)
    )
    masks = {
        "single_day_clinical": output["clinical_visit"].fillna(False).astype(bool)
        & output["n_collection_dates_in_episode"].eq(1),
        "essdai_esspri_different_days": essdai
        & esspri
        & output["n_collection_dates_in_episode"].gt(1),
        "multiple_intervals": intervals,
        "clinical_plus_research": output["clinical_visit"].fillna(False).astype(bool)
        & output.get("has_research_component", False),
        "research_only": output["visit_type"].eq(
            "research_or_procedure_only_candidate"
        ),
        "ambiguous": output["visit_type"].eq("ambiguous"),
        "manual_review_required": output["manual_review_required"]
        .fillna(False)
        .astype(bool),
    }
    conflict_keys = set(
        map(tuple, conflicts[["patient_id", "clinical_episode_id"]].to_numpy())
    )
    masks["variable_conflict"] = output.apply(
        lambda row: (row["patient_id"], row["clinical_episode_id"]) in conflict_keys,
        axis=1,
    )
    used: set[tuple[object, object]] = set()
    for reason, mask in masks.items():
        matches = output.loc[mask]
        if matches.empty:
            continue
        row = matches.iloc[0]
        key = (row["patient_id"], row["clinical_episode_id"])
        if key in used:
            continue
        used.add(key)
        candidates.append((reason, row))
    columns = list(build_row_counts(output).columns)
    records = [
        {"review_reason": reason, **row[columns].to_dict()}
        for reason, row in candidates
    ]
    return pd.DataFrame(records, columns=["review_reason", *columns])


def print_output_summary(
    output: pd.DataFrame,
    conflicts: pd.DataFrame,
    output_base: Path,
    qc_dir: Path,
    before_after_created: bool,
) -> None:
    """Print generated artifacts and headline QC counts in a readable layout.

    Parameters
    ----------
    output : pd.DataFrame
        Final clinical-episode table.
    conflicts : pd.DataFrame
        Variable-level conflict report.
    output_base : Path
        Extension-free base path for the analytic Parquet and CSV outputs.
    qc_dir : Path
        Directory containing the step 09c QC reports.
    before_after_created : bool
        Whether the optional comparison with the interval pipeline was written.
    """
    output_files: dict[str, object] = {
        "Analytic table (Parquet)": output_base.with_suffix(".parquet"),
        "Analytic table (CSV)": output_base.with_suffix(".csv"),
        "Variable conflicts": qc_dir / "09c_episode_variable_conflicts.csv",
        "QC summary": qc_dir / "09c_qc_summary.csv",
        "Episode row counts": qc_dir / "09c_episode_row_counts.csv",
        "Review examples": qc_dir / "09c_review_examples.csv",
    }
    if before_after_created:
        output_files["Old vs new comparison"] = qc_dir / "09c_before_after_summary.csv"
    else:
        output_files["Old vs new comparison"] = (
            "not created (legacy interval-collapse Parquet was not found)"
        )

    print_step(6, "Outputs created")
    print_kv("09c output files", output_files)
    print_kv(
        "09c completion summary",
        {
            "patients": int(output["patient_id"].nunique()),
            "clinical episodes": int(len(output)),
            "clinical candidates": int(
                output["visit_type"].eq("clinical_candidate").sum()
            ),
            "research/procedure-only candidates": int(
                output["visit_type"].eq("research_or_procedure_only_candidate").sum()
            ),
            "ambiguous episodes": int(output["visit_type"].eq("ambiguous").sum()),
            "manual-review episodes": int(
                output["manual_review_required"].fillna(False).astype(bool).sum()
            ),
            "variable conflicts": int(len(conflicts)),
            "hard QC": "PASSED",
        },
    )


def main() -> None:
    """Run clinical-episode collapse, hard validation, and QC report creation."""
    args = parse_args()
    logger = setup_logger("09c_clinical_episode_collapse")
    print_script_overview(
        "09c_clinical_episode_collapse.py",
        "Collapse variables inside the clinical episodes already defined by step 08c.",
    )
    print_step(1, "Load visits, clinical episode row map, and manifest")
    logger.info(
        "Loading visits=%s row_map=%s manifest=%s",
        args.input_path,
        args.row_map_path,
        args.manifest_path,
    )
    visits = pd.read_parquet(args.input_path)
    row_map = pd.read_parquet(args.row_map_path)
    manifest = pd.read_parquet(args.manifest_path)
    print_step(2, "Validate authoritative row assignments and manifest keys")
    patient_column, assignment_qc = validate_inputs(visits, row_map, manifest)
    joined = join_assignments(visits, row_map, patient_column)
    print_step(3, "Collapse variables by patient_id x clinical_episode_id")
    collapsed, conflicts = collapse_episodes(joined, manifest)
    output = make_parquet_compatible(
        merge_ans_columns(add_manifest(collapsed, manifest))
    )
    print_step(4, "Run hard QC checks")
    hard_qc(output, manifest)
    qc = build_qc(visits, joined, manifest, output, conflicts, assignment_qc)

    print_step(5, "Write analytic tables and QC reports")
    args.output_base.parent.mkdir(parents=True, exist_ok=True)
    args.qc_dir.mkdir(parents=True, exist_ok=True)
    save_parquet_and_csv(output, args.output_base, logger)
    conflicts.to_csv(args.qc_dir / "09c_episode_variable_conflicts.csv", index=False)
    qc.to_csv(args.qc_dir / "09c_qc_summary.csv", index=False)
    build_row_counts(output).to_csv(
        args.qc_dir / "09c_episode_row_counts.csv", index=False
    )
    build_review_examples(output, conflicts).to_csv(
        args.qc_dir / "09c_review_examples.csv", index=False
    )
    before_after_created = OLD_COLLAPSE_PATH.exists()
    if before_after_created:
        old = pd.read_parquet(OLD_COLLAPSE_PATH)
        build_before_after(old, output, conflicts).to_csv(
            args.qc_dir / "09c_before_after_summary.csv", index=False
        )
    print_output_summary(
        output,
        conflicts,
        args.output_base,
        args.qc_dir,
        before_after_created,
    )
    logger.info(
        "Completed rows=%d patients=%d conflicts=%d; all hard QC checks passed",
        len(output),
        output["patient_id"].nunique(),
        len(conflicts),
    )


if __name__ == "__main__":
    main()
