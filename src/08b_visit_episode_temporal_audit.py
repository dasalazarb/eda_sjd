"""Audit dates and populated components within patient visit episodes.

This script is descriptive only. Its composite-candidate review reads the
previously generated chronological context, temporal cases, and component CSVs.
It never combines episodes or changes interval names or the official baseline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import MISSING_TOKENS, REPORTS_DIR, setup_logger

DEFAULT_OUTPUT_DIR = REPORTS_DIR / "visit_episode_audit"
DEFAULT_COMPONENT_DATES = DEFAULT_OUTPUT_DIR / "01_component_dates.csv"
DEFAULT_CONTEXT = DEFAULT_OUTPUT_DIR / "06_chronological_episode_context.csv"
DEFAULT_REVIEW_CASES = DEFAULT_OUTPUT_DIR / "07_temporal_review_cases.csv"
NEARBY_DAYS = 14
COMPOSITE_WINDOWS = (3, 7, 14)

COMPONENT_PREFIXES: dict[str, tuple[str, ...]] = {
    "essdai": ("essdai", "essdai-_r"),
    "esspri": ("esspri_questionnaire", "esspri"),
    "eye_examination": ("eye_examination",),
    "salivary_flow": ("salivary_flow_form", "salivary_flow"),
    "systems_review_for_physician": ("systems_review_for_physician",),
    "visit_summary": (
        "visit_summary_form",
        "visit_summary_-_2016_classification_criteria",
    ),
    "oral_examination": ("oral_exam_form", "oral_examination"),
    "ccgo": ("ccgo",),
    "buccal_swab": ("buccal_swab_form", "buccal_swab"),
    "ipsc_specimen": ("ipscs_specimen", "ipsc_specimen"),
    "skin_biopsy": ("skin_biopsy",),
}

CLINICAL_ANCHORS = (
    "eye_examination",
    "salivary_flow",
    "systems_review_for_physician",
    "visit_summary",
    "oral_examination",
)
RESEARCH_COMPONENTS = (
    "ccgo",
    "specimen",
    "buccal_swab",
    "ipsc_specimen",
    "skin_biopsy",
)
MISSING_TOKEN_UPPER = {token.strip().upper() for token in MISSING_TOKENS}


def parse_args() -> argparse.Namespace:
    """Parse command-line paths.

    Returns
    -------
    argparse.Namespace
        Parsed episode summary, component evidence, and output paths.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--component-dates-path", type=Path, default=DEFAULT_COMPONENT_DATES
    )
    parser.add_argument("--context-path", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--review-cases-path", type=Path, default=DEFAULT_REVIEW_CASES)
    return parser.parse_args()


def resolve_column(df: pd.DataFrame, canonical_name: str) -> str:
    """Resolve an exact or group-prefixed canonical column name.

    Parameters
    ----------
    df : pd.DataFrame
        Input visit table.
    canonical_name : str
        Unprefixed canonical column name to locate.

    Returns
    -------
    str
        Matching dataframe column.

    Raises
    ------
    KeyError
        If no unambiguous matching column exists.
    """
    if canonical_name in df.columns:
        return canonical_name
    matches = [col for col in df.columns if str(col).endswith(f"__{canonical_name}")]
    if len(matches) == 1:
        return str(matches[0])
    if not matches:
        raise KeyError(f"Required column not found: {canonical_name}")
    raise KeyError(f"Ambiguous columns for {canonical_name}: {matches}")


def has_information(series: pd.Series) -> pd.Series:
    """Return a mask identifying non-empty, non-sentinel values.

    Parameters
    ----------
    series : pd.Series
        Values from one source column.

    Returns
    -------
    pd.Series
        Boolean mask aligned to ``series``.
    """
    present = series.notna()
    if pd.api.types.is_object_dtype(series.dtype) or pd.api.types.is_string_dtype(
        series.dtype
    ):
        text = series.astype("string").str.strip()
        present &= text.notna() & ~text.str.upper().isin(MISSING_TOKEN_UPPER)
    return present.fillna(False)


def _prefix(column: object) -> str | None:
    text = str(column)
    if "__" not in text:
        return None
    return text.split("__", 1)[0].strip().lower()


def component_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    """Map audit components to their matching source columns.

    Parameters
    ----------
    df : pd.DataFrame
        Wide visit table whose category is the text before ``__``.

    Returns
    -------
    dict[str, list[str]]
        Component names and matching columns. ``specimen`` is a broad audit
        marker for any category prefix containing the word ``specimen``.
    """
    prefixes = {col: _prefix(col) for col in df.columns}
    result: dict[str, list[str]] = {}
    for component, accepted_prefixes in COMPONENT_PREFIXES.items():
        result[component] = [
            str(col) for col, prefix in prefixes.items() if prefix in accepted_prefixes
        ]
    result["specimen"] = [
        str(col)
        for col, prefix in prefixes.items()
        if prefix is not None and "specimen" in prefix
    ]
    return result


def build_audit_tables(
    visits: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build component-date, episode, and baseline audit tables.

    Parameters
    ----------
    visits : pd.DataFrame
        Uncollapsed longitudinal visits table.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Component dates, episode summaries, and patient baseline comparisons.
    """
    patient_col = resolve_column(visits, "patient_record_number")
    interval_col = resolve_column(visits, "interval_name")
    try:
        date_col = resolve_column(visits, "visit_date")
    except KeyError:
        date_col = resolve_column(visits, "visit_datetime")

    work = visits.copy()
    work["_patient_id"] = work[patient_col]
    work["_interval_name"] = work[interval_col].astype("string").str.strip()
    work["_collection_date"] = pd.to_datetime(
        work[date_col], errors="coerce"
    ).dt.normalize()
    work = work.dropna(subset=["_patient_id", "_interval_name"])

    matched_columns = component_columns(work)
    for component, columns in matched_columns.items():
        flag = pd.Series(False, index=work.index)
        for column in columns:
            flag |= has_information(work[column])
        work[f"_has_{component}"] = flag

    # Include every populated category prefix in the date-level evidence table.
    prefix_to_columns: dict[str, list[str]] = {}
    excluded_columns = {patient_col, interval_col, date_col}
    for column in work.columns:
        prefix = _prefix(column)
        if prefix is not None and prefix != "ids" and column not in excluded_columns:
            prefix_to_columns.setdefault(prefix, []).append(str(column))

    component_frames: list[pd.DataFrame] = []
    for component, columns in prefix_to_columns.items():
        populated = pd.Series(False, index=work.index)
        for column in columns:
            populated |= has_information(work[column])
        rows = work.loc[
            populated & work["_collection_date"].notna(),
            ["_patient_id", "_interval_name", "_collection_date"],
        ].copy()
        rows["component"] = component
        component_frames.append(rows)

    if component_frames:
        component_dates = pd.concat(component_frames, ignore_index=True)
    else:
        component_dates = pd.DataFrame(
            columns=[
                "_patient_id",
                "_interval_name",
                "_collection_date",
                "component",
            ]
        )
    component_dates = (
        component_dates.drop_duplicates()
        .rename(
            columns={
                "_patient_id": "patient_id",
                "_interval_name": "interval_name",
                "_collection_date": "collection_date",
            }
        )
        .sort_values(["patient_id", "interval_name", "collection_date", "component"])
        .reset_index(drop=True)
    )

    keys = ["_patient_id", "_interval_name"]
    aggregation: dict[str, tuple[str, str]] = {
        "episode_start_date": ("_collection_date", "min"),
        "episode_end_date": ("_collection_date", "max"),
    }
    for component in matched_columns:
        aggregation[f"has_{component}"] = (f"_has_{component}", "max")
    episodes = work.groupby(keys, dropna=False).agg(**aggregation).reset_index()
    episodes["episode_span_days"] = (
        episodes["episode_end_date"] - episodes["episode_start_date"]
    ).dt.days
    episodes["clinical_anchor_count"] = (
        episodes[[f"has_{component}" for component in CLINICAL_ANCHORS]]
        .sum(axis=1)
        .astype(int)
    )
    has_research = episodes[
        [f"has_{component}" for component in RESEARCH_COMPONENTS]
    ].any(axis=1)
    episodes["candidate_visit_type"] = "ambiguous"
    clinical = episodes["has_essdai"] | (episodes["clinical_anchor_count"] >= 4)
    research_only = (
        ~episodes["has_essdai"]
        & (episodes["clinical_anchor_count"] <= 2)
        & has_research
    )
    episodes.loc[research_only, "candidate_visit_type"] = "research_only_candidate"
    episodes.loc[clinical, "candidate_visit_type"] = "clinical_candidate"
    episodes = episodes.rename(
        columns={"_patient_id": "patient_id", "_interval_name": "interval_name"}
    ).sort_values(["patient_id", "episode_start_date", "interval_name"])

    first_dates = (
        work.groupby("_patient_id")["_collection_date"]
        .min()
        .rename("first_recorded_date")
    )
    clinical_dates = (
        episodes.loc[episodes["candidate_visit_type"] == "clinical_candidate"]
        .groupby("patient_id")["episode_start_date"]
        .min()
        .rename("candidate_clinical_baseline_date")
    )
    baselines = first_dates.to_frame().join(clinical_dates, how="left")
    baselines["baseline_changed"] = (
        baselines["candidate_clinical_baseline_date"].notna()
        & baselines["first_recorded_date"].notna()
        & (
            baselines["candidate_clinical_baseline_date"]
            != baselines["first_recorded_date"]
        )
    )
    baselines["baseline_shift_days"] = (
        baselines["candidate_clinical_baseline_date"] - baselines["first_recorded_date"]
    ).dt.days
    baselines.index.name = "patient_id"
    return component_dates, episodes.reset_index(drop=True), baselines.reset_index()


def log_summary(
    episodes: pd.DataFrame, baselines: pd.DataFrame, logger: object
) -> None:
    """Log the requested aggregate audit summary.

    Parameters
    ----------
    episodes : pd.DataFrame
        Episode-level audit results.
    baselines : pd.DataFrame
        Patient-level baseline comparisons.
    logger : object
        Logger exposing an ``info`` method.
    """
    total = len(episodes)
    counts = episodes["candidate_visit_type"].value_counts()
    changed = baselines.loc[baselines["baseline_changed"], "baseline_shift_days"]
    logger.info("Patients: %d", baselines["patient_id"].nunique())
    logger.info("Total episodes: %d", total)
    for label in ("clinical_candidate", "research_only_candidate", "ambiguous"):
        count = int(counts.get(label, 0))
        percentage = 100 * count / total if total else 0.0
        logger.info("%s: %d (%.2f%%)", label, count, percentage)
    logger.info("Patients whose baseline would change: %d", len(changed))
    logger.info(
        "Baseline shift days among changed patients: median=%s, maximum=%s",
        changed.median() if not changed.empty else "NA",
        changed.max() if not changed.empty else "NA",
    )


def _component_flags(component_dates: pd.DataFrame) -> pd.DataFrame:
    """Return episode-level flags for components used in nearby-pair review."""
    required = {"patient_id", "interval_name", "component"}
    missing = required.difference(component_dates.columns)
    if missing:
        raise ValueError(f"Component dates missing columns: {sorted(missing)}")

    component = component_dates["component"].astype("string").str.lower()
    definitions = {
        "essdai": ("essdai",),
        "esspri": ("esspri",),
        "eye_examination": ("eye_examination", "eye_exam"),
        "salivary_flow": ("salivary_flow",),
        "physician_review": ("systems_review_for_physician", "physician_review"),
        "visit_summary": ("visit_summary",),
    }
    flags = component_dates[["patient_id", "interval_name"]].copy()
    for name, terms in definitions.items():
        flags[name] = component.apply(
            lambda value: bool(pd.notna(value) and any(term in value for term in terms))
        )
    return flags.groupby(["patient_id", "interval_name"], as_index=False).max()


def build_temporal_review(
    episodes: pd.DataFrame, component_dates: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build chronological episode context and cases requiring manual review.

    Parameters
    ----------
    episodes : pd.DataFrame
        Clean episode summary with dates, span, and candidate classification.
    component_dates : pd.DataFrame
        Component evidence, preferably from ``01_component_dates.csv``.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Chronological context and a case-level review report. No rows are merged.
    """
    required = {
        "patient_id",
        "interval_name",
        "episode_start_date",
        "episode_end_date",
        "episode_span_days",
        "candidate_visit_type",
    }
    missing = required.difference(episodes.columns)
    if missing:
        raise ValueError(f"Episode summary missing columns: {sorted(missing)}")

    work = episodes.copy()
    for column in ("episode_start_date", "episode_end_date"):
        work[column] = pd.to_datetime(work[column], errors="coerce").dt.normalize()
    work = work.sort_values(
        ["patient_id", "episode_start_date", "episode_end_date", "interval_name"],
        na_position="last",
    ).reset_index(drop=True)
    grouped = work.groupby("patient_id", sort=False, dropna=False)
    work["previous_interval"] = grouped["interval_name"].shift()
    work["next_interval"] = grouped["interval_name"].shift(-1)
    work["gap_previous_days"] = (
        work["episode_start_date"] - grouped["episode_end_date"].shift()
    ).dt.days
    work["gap_next_days"] = (
        grouped["episode_start_date"].shift(-1) - work["episode_end_date"]
    ).dt.days
    context_columns = [
        "patient_id",
        "interval_name",
        "episode_start_date",
        "episode_end_date",
        "candidate_visit_type",
        "previous_interval",
        "next_interval",
        "gap_previous_days",
        "gap_next_days",
    ]
    context = work[context_columns].copy()

    case_columns = [
        "case_type",
        "patient_id",
        "interval_name",
        "paired_interval",
        "gap_days",
        "episode_start_date",
        "episode_end_date",
        "candidate_visit_type",
        "details",
    ]
    cases: list[dict[str, object]] = []

    def add_single(row: pd.Series, case_type: str, details: str) -> None:
        cases.append(
            {
                "case_type": case_type,
                "patient_id": row["patient_id"],
                "interval_name": row["interval_name"],
                "paired_interval": pd.NA,
                "gap_days": pd.NA,
                "episode_start_date": row["episode_start_date"],
                "episode_end_date": row["episode_end_date"],
                "candidate_visit_type": row["candidate_visit_type"],
                "details": details,
            }
        )

    for _, row in work.loc[work["episode_span_days"] > 14].iterrows():
        add_single(
            row,
            "episode_span_gt_14",
            f"episode_span_days={row['episode_span_days']}",
        )
    for _, row in work.loc[work["episode_span_days"] > 30].iterrows():
        add_single(
            row,
            "episode_span_gt_30",
            f"episode_span_days={row['episode_span_days']}",
        )

    patients = work[["patient_id"]].drop_duplicates()
    clinical_patients = work.loc[
        work["candidate_visit_type"].eq("clinical_candidate"), ["patient_id"]
    ].drop_duplicates()
    no_clinical = patients.merge(
        clinical_patients, on="patient_id", how="left", indicator=True
    )
    missing_clinical = no_clinical.loc[
        no_clinical["_merge"].eq("left_only"), "patient_id"
    ]
    for patient_id in missing_clinical:
        cases.append(
            {
                "case_type": "patient_without_clinical_candidate",
                "patient_id": patient_id,
                "interval_name": pd.NA,
                "paired_interval": pd.NA,
                "gap_days": pd.NA,
                "episode_start_date": pd.NaT,
                "episode_end_date": pd.NaT,
                "candidate_visit_type": pd.NA,
                "details": "No clinical_candidate episode",
            }
        )

    flags = _component_flags(component_dates)
    paired = work.merge(flags, on=["patient_id", "interval_name"], how="left")
    flag_columns = [
        column
        for column in flags.columns
        if column not in {"patient_id", "interval_name"}
    ]
    paired[flag_columns] = paired[flag_columns].fillna(False).astype(bool)
    for _, patient_rows in paired.groupby("patient_id", sort=False, dropna=False):
        rows = list(patient_rows.iterrows())
        for left_position, (_, left) in enumerate(rows):
            for _, right in rows[left_position + 1 :]:
                gap = (right["episode_start_date"] - left["episode_end_date"]).days
                if gap > NEARBY_DAYS:
                    break
                pair_types: list[tuple[str, str]] = []
                types = {left["candidate_visit_type"], right["candidate_visit_type"]}
                if types == {"ambiguous", "clinical_candidate"}:
                    pair_types.append(
                        (
                            "ambiguous_near_clinical",
                            "ambiguous and clinical_candidate",
                        )
                    )
                if (left["essdai"] and right["esspri"]) or (
                    left["esspri"] and right["essdai"]
                ):
                    pair_types.append(
                        (
                            "essdai_esspri_split",
                            "ESSDAI and ESSPRI split across episodes",
                        )
                    )
                clinical_components = (
                    "eye_examination",
                    "salivary_flow",
                    "physician_review",
                    "visit_summary",
                )
                left_components = {
                    name for name in clinical_components if bool(left[name])
                }
                right_components = {
                    name for name in clinical_components if bool(right[name])
                }
                split_components = left_components.symmetric_difference(
                    right_components
                )
                if left_components and right_components and split_components:
                    pair_types.append(
                        (
                            "clinical_components_split",
                            ", ".join(sorted(split_components)),
                        )
                    )
                for case_type, details in pair_types:
                    cases.append(
                        {
                            "case_type": case_type,
                            "patient_id": left["patient_id"],
                            "interval_name": left["interval_name"],
                            "paired_interval": right["interval_name"],
                            "gap_days": gap,
                            "episode_start_date": left["episode_start_date"],
                            "episode_end_date": left["episode_end_date"],
                            "candidate_visit_type": left["candidate_visit_type"],
                            "details": details,
                        }
                    )

    review_cases = pd.DataFrame(cases, columns=case_columns)
    return context, review_cases


def _composite_component_flags(component_dates: pd.DataFrame) -> pd.DataFrame:
    """Summarize clinical-anchor and research evidence by patient-interval."""
    required = {"patient_id", "interval_name", "component"}
    missing = required.difference(component_dates.columns)
    if missing:
        raise ValueError(f"Component dates missing columns: {sorted(missing)}")

    component = component_dates["component"].astype("string").str.lower()
    definitions = {
        "has_essdai": ("essdai",),
        "has_esspri": ("esspri",),
        "anchor_eye_examination": ("eye_examination", "eye_exam"),
        "anchor_salivary_flow": ("salivary_flow",),
        "anchor_physician_review": (
            "systems_review_for_physician",
            "physician_review",
        ),
        "anchor_visit_summary": ("visit_summary",),
        "anchor_oral_examination": ("oral_exam", "oral_examination"),
        "has_research_only_components": (
            "ccgo",
            "specimen",
            "buccal_swab",
            "ipsc",
            "skin_biopsy",
        ),
    }
    flags = component_dates[["patient_id", "interval_name"]].copy()
    for name, terms in definitions.items():
        flags[name] = component.apply(
            lambda value: bool(pd.notna(value) and any(term in value for term in terms))
        )
    return flags.groupby(["patient_id", "interval_name"], as_index=False).max()


def _excluded_temporal_records(
    context: pd.DataFrame, review_cases: pd.DataFrame
) -> pd.DataFrame:
    """Identify long or overlapping records to keep outside candidate clusters."""
    columns = [
        "patient_id",
        "interval_name",
        "episode_start_date",
        "episode_end_date",
        "exclusion_reason",
    ]
    reasons: dict[tuple[object, object], set[str]] = {}
    if {"case_type", "patient_id", "interval_name"}.issubset(review_cases.columns):
        long_rows = review_cases.loc[
            review_cases["case_type"].eq("episode_span_gt_30"),
            ["patient_id", "interval_name"],
        ].drop_duplicates()
    else:
        long_rows = pd.DataFrame(columns=["patient_id", "interval_name"])
    for row in long_rows.itertuples(index=False):
        reasons.setdefault((row.patient_id, row.interval_name), set()).add(
            "episode_span_days_gt_30"
        )

    ordered = context.sort_values(
        ["patient_id", "episode_start_date", "episode_end_date", "interval_name"]
    )
    for _, patient_rows in ordered.groupby("patient_id", sort=False, dropna=False):
        rows = list(patient_rows.itertuples(index=False))
        for position, previous in enumerate(rows):
            for current in rows[position + 1 :]:
                if current.episode_start_date > previous.episode_end_date:
                    break
                reasons.setdefault(
                    (previous.patient_id, previous.interval_name), set()
                ).add("overlapping_date_range")
                reasons.setdefault(
                    (current.patient_id, current.interval_name), set()
                ).add("overlapping_date_range")

    excluded_rows: list[dict[str, object]] = []
    indexed = context.set_index(["patient_id", "interval_name"], drop=False)
    for key, record_reasons in reasons.items():
        if key not in indexed.index:
            continue
        row = indexed.loc[key]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        excluded_rows.append(
            {
                "patient_id": key[0],
                "interval_name": key[1],
                "episode_start_date": row["episode_start_date"],
                "episode_end_date": row["episode_end_date"],
                "exclusion_reason": ";".join(sorted(record_reasons)),
            }
        )
    return pd.DataFrame(excluded_rows, columns=columns)


def _window_clusters(records: pd.DataFrame, window_days: int) -> list[pd.DataFrame]:
    """Return multi-record clusters whose complete date range fits a window."""
    clusters: list[pd.DataFrame] = []
    for _, patient_rows in records.groupby("patient_id", sort=False, dropna=False):
        patient_rows = patient_rows.sort_values(
            ["episode_start_date", "episode_end_date", "interval_name"]
        )
        current_indices: list[object] = []
        cluster_start: pd.Timestamp | None = None
        cluster_end: pd.Timestamp | None = None
        for index, row in patient_rows.iterrows():
            proposed_start = (
                row["episode_start_date"] if cluster_start is None else cluster_start
            )
            proposed_end = (
                row["episode_end_date"]
                if cluster_end is None
                else max(cluster_end, row["episode_end_date"])
            )
            if current_indices and (proposed_end - proposed_start).days > window_days:
                if len(current_indices) > 1:
                    clusters.append(patient_rows.loc[current_indices])
                current_indices = []
                cluster_start = row["episode_start_date"]
                cluster_end = row["episode_end_date"]
            else:
                cluster_start = proposed_start
                cluster_end = proposed_end
            current_indices.append(index)
        if len(current_indices) > 1:
            clusters.append(patient_rows.loc[current_indices])
    return clusters


def build_composite_episode_candidates(
    context: pd.DataFrame,
    review_cases: pd.DataFrame,
    component_dates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate non-destructive composite clinical episode candidates.

    Parameters
    ----------
    context : pd.DataFrame
        Chronological episode context from ``06_chronological_episode_context.csv``.
    review_cases : pd.DataFrame
        Temporal cases from ``07_temporal_review_cases.csv``.
    component_dates : pd.DataFrame
        Component-level evidence. May be empty when the optional file is absent.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Candidate clusters, comparison by window, and separately excluded records.
    """
    required = {
        "patient_id",
        "interval_name",
        "episode_start_date",
        "episode_end_date",
        "candidate_visit_type",
    }
    missing = required.difference(context.columns)
    if missing:
        raise ValueError(f"Chronological context missing columns: {sorted(missing)}")

    work = context.copy()
    for column in ("episode_start_date", "episode_end_date"):
        work[column] = pd.to_datetime(work[column], errors="coerce").dt.normalize()
    work = work.dropna(
        subset=["patient_id", "interval_name", "episode_start_date", "episode_end_date"]
    )
    excluded = _excluded_temporal_records(work, review_cases)
    excluded_keys = set(
        excluded[["patient_id", "interval_name"]].itertuples(index=False, name=None)
    )
    eligible = work.loc[
        ~work.apply(
            lambda row: (row["patient_id"], row["interval_name"]) in excluded_keys,
            axis=1,
        )
    ].copy()

    if component_dates.empty:
        flags = eligible[["patient_id", "interval_name"]].drop_duplicates()
        evidence_columns = [
            "has_essdai",
            "has_esspri",
            "anchor_eye_examination",
            "anchor_salivary_flow",
            "anchor_physician_review",
            "anchor_visit_summary",
            "anchor_oral_examination",
            "has_research_only_components",
        ]
        for column in evidence_columns:
            flags[column] = False
    else:
        flags = _composite_component_flags(component_dates)
        evidence_columns = [
            column
            for column in flags.columns
            if column not in {"patient_id", "interval_name"}
        ]
    eligible = eligible.merge(flags, on=["patient_id", "interval_name"], how="left")
    eligible[evidence_columns] = eligible[evidence_columns].fillna(False).astype(bool)
    anchor_columns = [
        column for column in evidence_columns if column.startswith("anchor_")
    ]

    candidate_rows: list[dict[str, object]] = []
    for window_days in COMPOSITE_WINDOWS:
        for cluster_number, cluster in enumerate(
            _window_clusters(eligible, window_days), start=1
        ):
            has_essdai = bool(cluster["has_essdai"].any())
            has_esspri = bool(cluster["has_esspri"].any())
            anchor_count = int(cluster[anchor_columns].any(axis=0).sum())
            combined_clinical = has_essdai or anchor_count >= 4
            types = cluster["candidate_visit_type"].dropna().astype(str).unique()
            essdai_esspri_together = has_essdai and has_esspri
            essdai_esspri_reunited = essdai_esspri_together and not bool(
                (cluster["has_essdai"] & cluster["has_esspri"]).any()
            )
            candidate_rows.append(
                {
                    "window_days": window_days,
                    "cluster_id": f"w{window_days}_{cluster_number:05d}",
                    "patient_id": cluster["patient_id"].iloc[0],
                    "intervals_involved": "|".join(
                        cluster["interval_name"].astype(str)
                    ),
                    "cluster_start_date": cluster["episode_start_date"].min(),
                    "cluster_end_date": cluster["episode_end_date"].max(),
                    "cluster_span_days": (
                        cluster["episode_end_date"].max()
                        - cluster["episode_start_date"].min()
                    ).days,
                    "n_records": len(cluster),
                    "has_essdai": has_essdai,
                    "has_esspri": has_esspri,
                    "clinical_anchor_count_combined": anchor_count,
                    "has_research_only_components": bool(
                        cluster["has_research_only_components"].any()
                    ),
                    "candidate_visit_types_involved": "|".join(types),
                    "n_ambiguous_records": int(
                        cluster["candidate_visit_type"].eq("ambiguous").sum()
                    ),
                    "combines_essdai_esspri": essdai_esspri_together,
                    "meets_clinical_anchor_threshold": anchor_count >= 4,
                    "ambiguous_to_clinical_candidate": bool(
                        combined_clinical
                        and cluster["candidate_visit_type"].eq("ambiguous").any()
                    ),
                    "essdai_esspri_reunited": essdai_esspri_reunited,
                }
            )
    candidate_columns = [
        "window_days",
        "cluster_id",
        "patient_id",
        "intervals_involved",
        "cluster_start_date",
        "cluster_end_date",
        "cluster_span_days",
        "n_records",
        "has_essdai",
        "has_esspri",
        "clinical_anchor_count_combined",
        "has_research_only_components",
        "candidate_visit_types_involved",
        "n_ambiguous_records",
        "combines_essdai_esspri",
        "meets_clinical_anchor_threshold",
        "ambiguous_to_clinical_candidate",
        "essdai_esspri_reunited",
    ]
    candidates = pd.DataFrame(candidate_rows, columns=candidate_columns)

    summary_rows: list[dict[str, object]] = []
    patients_with_clinical = set(
        work.loc[work["candidate_visit_type"].eq("clinical_candidate"), "patient_id"]
    )
    for window_days in COMPOSITE_WINDOWS:
        window = candidates.loc[candidates["window_days"].eq(window_days)]
        recovered = window.loc[window["ambiguous_to_clinical_candidate"]]
        recovered_without_prior = recovered.loc[
            ~recovered["patient_id"].isin(patients_with_clinical)
        ]
        summary_rows.append(
            {
                "window_days": window_days,
                "n_clusters_created": len(window),
                "patients_affected": window["patient_id"].nunique(),
                "ambiguous_episodes_recovered": int(
                    recovered["n_ambiguous_records"].sum()
                ),
                "patients_without_clinical_candidate_recovered": recovered_without_prior[
                    "patient_id"
                ].nunique(),
                "essdai_esspri_reunited_clusters": int(
                    window["essdai_esspri_reunited"].sum()
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)
    return candidates, summary, excluded


def main() -> None:
    """Evaluate composite episode windows without changing source records."""
    args = parse_args()
    logger = setup_logger("08b_visit_episode_temporal_audit")
    logger.info("Reading chronological context from %s", args.context_path)
    context = pd.read_csv(args.context_path)
    logger.info("Reading temporal review cases from %s", args.review_cases_path)
    review_cases = pd.read_csv(args.review_cases_path)
    if args.component_dates_path.exists():
        logger.info("Reading component dates from %s", args.component_dates_path)
        component_dates = pd.read_csv(args.component_dates_path)
    else:
        logger.warning(
            "Component dates unavailable at %s; component-based indicators will be false",
            args.component_dates_path,
        )
        component_dates = pd.DataFrame()
    candidates, summary, excluded = build_composite_episode_candidates(
        context, review_cases, component_dates
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "08b_composite_episode_candidates.csv": candidates,
        "08b_window_comparison_summary.csv": summary,
        "08b_temporal_records_for_review.csv": excluded,
    }
    for filename, table in outputs.items():
        path = args.output_dir / filename
        table.to_csv(path, index=False)
        logger.info("Saved %s rows=%d", path, len(table))
    logger.info("Window comparison: %s", summary.to_dict(orient="records"))


if __name__ == "__main__":
    main()
