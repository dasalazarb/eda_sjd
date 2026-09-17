"""Harmonize longitudinal BTRIS lab results using an external rule-family map."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from btris_lab_harmonization import (
    HarmonizedResult,
    interpret_categorical_binary,
    interpret_categorical_multiclass,
    interpret_continuous_range,
    interpret_threshold_binary,
)
from common import ANALYTIC_DIR, REPORTS_DIR, ROOT, setup_logger

LAB_INPUT = ANALYTIC_DIR / "BTRIS" / "20_btris_lab_records_long.parquet"
RULE_INPUT = ROOT / "btris_lab_rule_families.xlsx"
OUTPUT = ANALYTIC_DIR / "BTRIS" / "20c_btris_lab_records_harmonized.parquet"
QC_DIR = REPORTS_DIR / "btris_labs" / "20c"
RULE_SHEET = "Lab rule map"
RULE_COLUMNS = ["Order Name", "Cluster Name", "Suggested Rule Family"]
JOIN_COLUMNS = ["order_name_original", "cluster_name_original"]
RULE_FAMILIES = {
    "CONTINUOUS_RANGE",
    "CONTINUOUS_ONLY",
    "THRESHOLD_BINARY",
    "CATEGORICAL_BINARY",
    "CATEGORICAL_MULTICLASS",
    "SPECIAL",
    "EXCLUDE",
}


def load_rule_map(path: Path) -> pd.DataFrame:
    """Load and validate the executable columns of the Excel rule map.

    Parameters
    ----------
    path : pathlib.Path
        Workbook containing the ``Lab rule map`` worksheet.

    Returns
    -------
    pandas.DataFrame
        Unique exact join keys and normalized rule families.

    Raises
    ------
    ValueError
        If required columns, keys, uniqueness, or rule families are invalid.
    """
    rules = pd.read_excel(path, sheet_name=RULE_SHEET, usecols=RULE_COLUMNS)
    missing = set(RULE_COLUMNS).difference(rules.columns)
    if missing:
        raise ValueError(f"Rule map missing required columns: {sorted(missing)}")
    if rules[["Order Name", "Cluster Name"]].isna().any(axis=None):
        raise ValueError("Rule map contains missing Order Name or Cluster Name keys")
    duplicates = rules.duplicated(["Order Name", "Cluster Name"], keep=False)
    if duplicates.any():
        pairs = rules.loc[duplicates, ["Order Name", "Cluster Name"]]
        raise ValueError(
            f"Rule map contains duplicate join keys:\n{pairs.to_string(index=False)}"
        )
    rules = rules.rename(columns={"Suggested Rule Family": "rule_family"})
    rules["rule_family"] = rules["rule_family"].astype("string").str.strip().str.upper()
    if rules["rule_family"].isna().any() or rules["rule_family"].eq("").any():
        raise ValueError("Rule map contains missing Suggested Rule Family values")
    unknown = sorted(set(rules["rule_family"].dropna()).difference(RULE_FAMILIES))
    if unknown:
        raise ValueError(f"Rule map contains unsupported rule families: {unknown}")
    return rules


def _dispatch(row: pd.Series) -> HarmonizedResult:
    """Dispatch one valid record to its configured rule family."""
    family = row["rule_family"]
    if family == "CONTINUOUS_RANGE":
        return interpret_continuous_range(row)
    if family == "THRESHOLD_BINARY":
        return interpret_threshold_binary(row)
    if family == "CATEGORICAL_BINARY":
        return interpret_categorical_binary(row)
    if family == "CATEGORICAL_MULTICLASS":
        return interpret_categorical_multiclass(row)
    return HarmonizedResult(pd.NA, "rule_family")


def harmonize_labs(labs: pd.DataFrame, rules: pd.DataFrame) -> pd.DataFrame:
    """Attach exact-pair rules and add non-destructive harmonized fields.

    Parameters
    ----------
    labs : pandas.DataFrame
        Step-20 longitudinal laboratory records.
    rules : pandas.DataFrame
        Validated output of :func:`load_rule_map`.

    Returns
    -------
    pandas.DataFrame
        All input columns and rows plus rule/category/status/source fields.
    """
    missing = set(JOIN_COLUMNS + ["result_valid_for_analysis"]).difference(labs.columns)
    if missing:
        raise ValueError(
            f"Laboratory input missing required columns: {sorted(missing)}"
        )
    merged = labs.merge(
        rules,
        how="left",
        left_on=JOIN_COLUMNS,
        right_on=["Order Name", "Cluster Name"],
        validate="many_to_one",
    ).drop(columns=["Order Name", "Cluster Name"])
    merged["result_category"] = pd.Series(pd.NA, index=merged.index, dtype="string")
    merged["harmonization_status"] = "UNINTERPRETABLE"
    merged["harmonization_source"] = pd.Series(
        pd.NA, index=merged.index, dtype="string"
    )

    no_rule = merged["rule_family"].isna()
    invalid = ~merged["result_valid_for_analysis"].fillna(False).astype(bool)
    special = merged["rule_family"].eq("SPECIAL")
    excluded = merged["rule_family"].eq("EXCLUDE")
    continuous_only = merged["rule_family"].eq("CONTINUOUS_ONLY")
    eligible = ~(no_rule | invalid | special | excluded | continuous_only)

    interpreted = merged.loc[eligible].apply(_dispatch, axis=1)
    for index, result in interpreted.items():
        merged.at[index, "result_category"] = result.category
        merged.at[index, "harmonization_source"] = result.source
    harmonized = eligible & merged["result_category"].notna()
    merged.loc[harmonized, "harmonization_status"] = "HARMONIZED"
    valid_continuous_only = continuous_only & ~invalid
    merged.loc[valid_continuous_only, "harmonization_source"] = "rule_family"
    merged.loc[valid_continuous_only, "harmonization_status"] = "HARMONIZED"
    merged.loc[invalid, "harmonization_status"] = "INVALID_RESULT"
    merged.loc[special, "harmonization_status"] = "SKIPPED_SPECIAL"
    merged.loc[excluded, "harmonization_status"] = "EXCLUDED"
    merged.loc[no_rule & ~invalid, "harmonization_status"] = "NO_RULE"
    return merged


def build_qc(harmonized: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build rule-family, status, and unmatched exact-pair QC tables."""
    summary = (
        harmonized.assign(
            is_harmonized=harmonized["harmonization_status"].eq("HARMONIZED"),
            is_uninterpretable=harmonized["harmonization_status"].eq("UNINTERPRETABLE"),
        )
        .groupby("rule_family", dropna=False)
        .agg(
            n_records=("harmonization_status", "size"),
            n_harmonized=("is_harmonized", "sum"),
            n_uninterpretable=("is_uninterpretable", "sum"),
        )
        .reset_index()
    )
    unmatched = (
        harmonized.loc[harmonized["rule_family"].isna(), JOIN_COLUMNS]
        .value_counts(dropna=False)
        .rename("n_records")
        .reset_index()
        .rename(
            columns={
                "order_name_original": "Order Name",
                "cluster_name_original": "Cluster Name",
            }
        )
    )
    statuses = (
        harmonized["harmonization_status"]
        .value_counts(dropna=False)
        .rename_axis("harmonization_status")
        .rename("n_records")
        .reset_index()
    )
    return {
        "20c_rule_family_summary.csv": summary,
        "unmatched_rule_pairs.csv": unmatched,
        "20c_harmonization_status_summary.csv": statuses,
    }


def run(lab_path: Path, rule_path: Path, output_path: Path, qc_dir: Path) -> None:
    """Run the step-20c harmonization and write its analytical and QC outputs."""
    logger = setup_logger("20c_btris_lab_rule_harmonization")
    logger.info("Reading step-20 labs from %s", lab_path)
    labs = pd.read_parquet(lab_path)
    rules = load_rule_map(rule_path)
    logger.info("Loaded labs shape=%s and rule pairs=%d", labs.shape, len(rules))
    harmonized = harmonize_labs(labs, rules)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    harmonized.to_parquet(output_path, index=False)
    qc_dir.mkdir(parents=True, exist_ok=True)
    for filename, table in build_qc(harmonized).items():
        table.to_csv(qc_dir / filename, index=False)
    counts = harmonized["harmonization_status"].value_counts()
    logger.info(
        "Wrote %s rows=%d; SPECIAL skipped=%d EXCLUDE=%d NO_RULE=%d",
        output_path,
        len(harmonized),
        int(counts.get("SKIPPED_SPECIAL", 0)),
        int(counts.get("EXCLUDED", 0)),
        int(counts.get("NO_RULE", 0)),
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line path overrides for reproducible batch execution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labs", type=Path, default=LAB_INPUT)
    parser.add_argument("--rules", type=Path, default=RULE_INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--qc-dir", type=Path, default=QC_DIR)
    return parser.parse_args()


def main() -> None:
    """Execute step 20c from command-line arguments."""
    args = parse_args()
    run(args.labs, args.rules, args.output, args.qc_dir)


if __name__ == "__main__":
    main()
