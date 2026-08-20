"""Audit dates and populated components within patient visit episodes.

This script is descriptive only: it reads ``visits_long.parquet`` and writes
CSV reports without changing the source table or defining an official clinical
baseline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import ANALYTIC_DIR, MISSING_TOKENS, REPORTS_DIR, setup_logger

DEFAULT_INPUT = ANALYTIC_DIR / "visits_long.parquet"
DEFAULT_OUTPUT_DIR = REPORTS_DIR / "visit_episode_audit"

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
        Parsed ``input_path`` and ``output_dir`` values.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
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
    episodes["clinical_anchor_count"] = episodes[
        [f"has_{component}" for component in CLINICAL_ANCHORS]
    ].sum(axis=1).astype(int)
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
        baselines["candidate_clinical_baseline_date"]
        - baselines["first_recorded_date"]
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


def main() -> None:
    """Run the read-only visit episode temporal audit and write five CSVs."""
    args = parse_args()
    logger = setup_logger("08b_visit_episode_temporal_audit")
    logger.info("Reading visits from %s", args.input_path)
    visits = pd.read_parquet(args.input_path)
    logger.info("Loaded rows=%d columns=%d", len(visits), len(visits.columns))

    component_dates, episodes, baselines = build_audit_tables(visits)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "01_component_dates.csv": component_dates,
        "02_episode_summary.csv": episodes,
        "03_baseline_comparison.csv": baselines,
        "04_candidate_research_only.csv": episodes.loc[
            episodes["candidate_visit_type"] == "research_only_candidate"
        ],
        "05_ambiguous_episodes.csv": episodes.loc[
            episodes["candidate_visit_type"] == "ambiguous"
        ],
    }
    for filename, table in outputs.items():
        path = args.output_dir / filename
        table.to_csv(path, index=False)
        logger.info("Saved %s rows=%d", path, len(table))
    log_summary(episodes, baselines, logger)


if __name__ == "__main__":
    main()
