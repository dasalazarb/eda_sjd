"""Audit dates and populated components within patient visit episodes.

This script is descriptive only.  Its temporal review reads the previously
generated episode and component CSVs and never combines episodes or changes the
official clinical baseline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import MISSING_TOKENS, REPORTS_DIR, setup_logger

DEFAULT_OUTPUT_DIR = REPORTS_DIR / "visit_episode_audit"
DEFAULT_EPISODE_SUMMARY = DEFAULT_OUTPUT_DIR / "02_episode_summary_clean.csv"
DEFAULT_COMPONENT_DATES = DEFAULT_OUTPUT_DIR / "01_component_dates.csv"
NEARBY_DAYS = 14

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
        "--episode-summary-path", type=Path, default=DEFAULT_EPISODE_SUMMARY
    )
    parser.add_argument(
        "--component-dates-path", type=Path, default=DEFAULT_COMPONENT_DATES
    )
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


def main() -> None:
    """Run the read-only chronological review and write two audit CSVs."""
    args = parse_args()
    logger = setup_logger("08b_visit_episode_temporal_audit")
    logger.info("Reading clean episode summary from %s", args.episode_summary_path)
    episodes = pd.read_csv(args.episode_summary_path)
    logger.info("Reading component dates from %s", args.component_dates_path)
    component_dates = pd.read_csv(args.component_dates_path)
    context, review_cases = build_temporal_review(episodes, component_dates)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "06_chronological_episode_context.csv": context,
        "07_temporal_review_cases.csv": review_cases,
    }
    for filename, table in outputs.items():
        path = args.output_dir / filename
        table.to_csv(path, index=False)
        logger.info("Saved %s rows=%d", path, len(table))
    logger.info(
        "Review cases by type: %s",
        review_cases["case_type"].value_counts().to_dict(),
    )


if __name__ == "__main__":
    main()
