"""Harmonize BTRIS lab files in place without changing their downstream interface."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import pandas as pd

from btris_lab_harmonization import (
    HarmonizedResult,
    interpret_categorical_binary,
    interpret_continuous_range,
    interpret_threshold_binary,
    normalize_qualitative,
)
from common import setup_logger

# Do not resolve symlinks here. On Biowulf, ``/data/...`` may resolve to a
# ``/vf/users/...`` target that is not the project path exposed to the job.
PROJECT_ROOT = Path(__file__).absolute().parents[1]
BTRIS_ROOT = PROJECT_ROOT / "data_analytic" / "BTRIS"
RULE_INPUT = PROJECT_ROOT / "btris_lab_rule_families.xlsx"
QC_DIR = PROJECT_ROOT / "reports" / "btris_labs" / "20c"
PROTOCOLS = ("11D", "15D")
RULE_SHEET = "Lab rule map"
RULE_COLUMNS = ["Order Name", "Cluster Name", "Suggested Rule Family"]
JOIN_COLUMNS = ["Order Name", "Cluster Name"]
TRACE_COLUMNS = [
    "Observation Value Original",
    "rule_family",
    "result_category",
    "harmonization_status",
    "harmonization_source",
]
RULE_FAMILIES = {
    "CONTINUOUS_RANGE",
    "CONTINUOUS_ONLY",
    "THRESHOLD_BINARY",
    "CATEGORICAL_BINARY",
    "CATEGORICAL_MULTICLASS",
    "SPECIAL",
    "EXCLUDE",
}
MULTICLASS_MAP = {"RARE": "Rare", "FEW": "Few"}
NUMBER_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"


def load_rule_map(path: Path) -> pd.DataFrame:
    """Load the unique exact-pair laboratory rule map.

    Parameters
    ----------
    path : pathlib.Path
        Workbook containing the ``Lab rule map`` worksheet.

    Returns
    -------
    pandas.DataFrame
        Exact ``Order Name``/``Cluster Name`` keys and normalized rule families.
    """
    rules = pd.read_excel(path, sheet_name=RULE_SHEET, usecols=RULE_COLUMNS)
    missing = set(RULE_COLUMNS).difference(rules.columns)
    if missing:
        raise ValueError(f"Rule map missing required columns: {sorted(missing)}")
    if rules[JOIN_COLUMNS].isna().any(axis=None):
        raise ValueError("Rule map contains missing Order Name or Cluster Name keys")
    duplicates = rules.duplicated(JOIN_COLUMNS, keep=False)
    if duplicates.any():
        pairs = rules.loc[duplicates, JOIN_COLUMNS]
        raise ValueError(
            f"Rule map contains duplicate join keys:\n{pairs.to_string(index=False)}"
        )
    rules = rules.rename(columns={"Suggested Rule Family": "rule_family"})
    rules["rule_family"] = rules["rule_family"].astype("string").str.strip().str.upper()
    if rules["rule_family"].isna().any() or rules["rule_family"].eq("").any():
        raise ValueError("Rule map contains missing Suggested Rule Family values")
    unknown = sorted(set(rules["rule_family"]).difference(RULE_FAMILIES))
    if unknown:
        raise ValueError(f"Rule map contains unsupported rule families: {unknown}")
    return rules


def _number_parts(value: Any) -> tuple[float | None, str | None]:
    """Parse an exact or one-sided numeric result without imputing a bound."""
    if pd.isna(value):
        return None, None
    match = re.fullmatch(rf"\s*(<=|>=|<|>)?\s*({NUMBER_RE})\s*", str(value))
    if not match:
        return None, None
    return float(match.group(2)), match.group(1)


def _interpretation_row(row: pd.Series) -> dict[str, Any]:
    """Build the evidence fields expected by the reusable interpreters."""
    raw = row["Observation Value Original"]
    value, operator = _number_parts(raw)
    evidence: dict[str, Any] = {
        "result_raw": raw,
        "result_text": raw,
        "result_numeric_exact": value if operator is None else pd.NA,
        "result_numeric": value if operator is None else pd.NA,
        "result_operator": operator if operator else pd.NA,
        "result_numeric_bound": value if operator else pd.NA,
        "reference_low": pd.NA,
        "reference_high": pd.NA,
        "reference_operator": pd.NA,
        "reference_bound": pd.NA,
        "reported_interpretation": row.get("Interpretation", pd.NA),
    }
    normal_range = row.get("Normal Range", pd.NA)
    if not pd.isna(normal_range):
        text = str(normal_range).strip()
        one_sided = re.match(rf"^\s*(<=|>=|<|>)\s*({NUMBER_RE})", text)
        interval = re.match(rf"^\s*({NUMBER_RE})\s*[-–]\s*({NUMBER_RE})", text)
        if one_sided:
            evidence["reference_operator"] = one_sided.group(1)
            evidence["reference_bound"] = float(one_sided.group(2))
        elif interval:
            evidence["reference_low"] = float(interval.group(1))
            evidence["reference_high"] = float(interval.group(2))
    # Prefer structured fields from step 20 whenever they are present.
    for key in evidence:
        if key in row.index and not pd.isna(row[key]):
            evidence[key] = row[key]
    return evidence


def _dispatch(row: pd.Series) -> HarmonizedResult:
    """Interpret one row according to its exact-pair rule family."""
    evidence = _interpretation_row(row)
    family = row["rule_family"]
    if family == "CONTINUOUS_RANGE":
        return interpret_continuous_range(evidence)
    if family == "THRESHOLD_BINARY":
        return interpret_threshold_binary(evidence)
    if family == "CATEGORICAL_BINARY":
        return interpret_categorical_binary(evidence)
    if family == "CATEGORICAL_MULTICLASS":
        token = normalize_qualitative(row["Observation Value Original"])
        category = MULTICLASS_MAP.get(token, pd.NA)
        return HarmonizedResult(category, "raw_qualitative_result")
    return HarmonizedResult(pd.NA, "rule_family")


def harmonize_labs(labs: pd.DataFrame, rules: pd.DataFrame) -> pd.DataFrame:
    """Harmonize ``Observation Value`` while preserving every row and source value.

    Parameters
    ----------
    labs : pandas.DataFrame
        One protocol-specific BTRIS laboratory file.
    rules : pandas.DataFrame
        Validated output of :func:`load_rule_map`.

    Returns
    -------
    pandas.DataFrame
        Input rows and columns plus trace fields. ``Observation Value`` is replaced
        only when a configured rule yields an unequivocal category.
    """
    required = set(JOIN_COLUMNS + ["Observation Value"])
    missing = required.difference(labs.columns)
    if missing:
        raise ValueError(
            f"Laboratory input missing required columns: {sorted(missing)}"
        )

    original_columns = set(labs.columns)
    working = labs.copy()
    if "Observation Value Original" not in working.columns:
        working["Observation Value Original"] = working["Observation Value"]
    # Every execution starts from the immutable source column, making reruns idempotent.
    working["Observation Value"] = working["Observation Value Original"].astype(
        "object"
    )
    working = working.drop(
        columns=[column for column in TRACE_COLUMNS[1:] if column in working.columns]
    )
    merged = working.merge(rules, how="left", on=JOIN_COLUMNS, validate="many_to_one")
    if len(merged) != len(labs):
        raise RuntimeError("Rule merge changed the number of laboratory rows")

    merged["result_category"] = pd.Series(pd.NA, index=merged.index, dtype="string")
    merged["harmonization_status"] = "UNINTERPRETABLE"
    merged["harmonization_source"] = pd.Series(
        pd.NA, index=merged.index, dtype="string"
    )
    no_rule = merged["rule_family"].isna()
    special = merged["rule_family"].eq("SPECIAL")
    excluded = merged["rule_family"].eq("EXCLUDE")
    continuous_only = merged["rule_family"].eq("CONTINUOUS_ONLY")
    if "result_valid_for_analysis" in merged.columns:
        invalid = ~merged["result_valid_for_analysis"].fillna(False).astype(bool)
    else:
        invalid = pd.Series(False, index=merged.index)
    eligible = ~(no_rule | special | excluded | continuous_only | invalid)

    for index, result in merged.loc[eligible].apply(_dispatch, axis=1).items():
        merged.at[index, "result_category"] = result.category
        merged.at[index, "harmonization_source"] = result.source

    interpreted = eligible & merged["result_category"].notna()
    replace_family = merged["rule_family"].isin(
        ["THRESHOLD_BINARY", "CATEGORICAL_BINARY", "CATEGORICAL_MULTICLASS"]
    )
    replace = interpreted & replace_family
    merged.loc[replace, "Observation Value"] = merged.loc[replace, "result_category"]
    merged.loc[interpreted, "harmonization_status"] = "HARMONIZED"
    merged.loc[continuous_only, "harmonization_status"] = "PRESERVED_CONTINUOUS"
    merged.loc[continuous_only, "harmonization_source"] = "rule_family"
    merged.loc[invalid, "harmonization_status"] = "INVALID_RESULT"
    merged.loc[special, "harmonization_status"] = "SKIPPED_SPECIAL"
    merged.loc[excluded, "harmonization_status"] = "EXCLUDED"
    merged.loc[no_rule, "harmonization_status"] = "NO_RULE"

    lost = original_columns.difference(merged.columns)
    if lost:
        raise RuntimeError(f"Harmonization lost input columns: {sorted(lost)}")
    return merged


def build_qc(harmonized: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build compact status and unmatched exact-pair QC tables."""
    statuses = (
        harmonized["harmonization_status"]
        .value_counts(dropna=False)
        .rename_axis("harmonization_status")
        .rename("n_records")
        .reset_index()
    )
    unmatched = (
        harmonized.loc[harmonized["rule_family"].isna(), JOIN_COLUMNS]
        .value_counts(dropna=False)
        .rename("n_records")
        .reset_index()
    )
    return {
        "20c_harmonization_status_summary.csv": statuses,
        "20c_unmatched_rule_pairs.csv": unmatched,
    }


def _read_lab_file(path: Path) -> pd.DataFrame:
    """Read a protocol-specific laboratory CSV or Parquet file."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(
            path,
            low_memory=False,
            dtype={"Observation Value": "object"},
        )
    if suffix == ".parquet":
        frame = pd.read_parquet(path)
        if "Observation Value" in frame.columns:
            frame["Observation Value"] = frame["Observation Value"].astype("object")
        return frame
    raise ValueError(f"Unsupported lab file format: {path}")


def _write_in_place(frame: pd.DataFrame, path: Path) -> None:
    """Atomically replace a CSV or Parquet file without changing its path."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        temporary = path.with_name(f".{path.stem}.20c.tmp.csv")
        frame.to_csv(temporary, index=False)
    elif suffix == ".parquet":
        temporary = path.with_name(f".{path.stem}.20c.tmp.parquet")
        frame.to_parquet(temporary, index=False)
    else:
        raise ValueError(f"Unsupported lab file format: {path}")
    temporary.replace(path)


def run(btris_root: Path, rule_path: Path, qc_dir: Path) -> None:
    """Harmonize each protocol-specific laboratory CSV or Parquet file in place."""
    logger = setup_logger("20c_btris_lab_rule_harmonization")
    rules = load_rule_map(rule_path)
    qc_dir.mkdir(parents=True, exist_ok=True)
    for protocol in PROTOCOLS:
        protocol_dir = btris_root / protocol
        paths = []
        paths.extend(sorted(protocol_dir.glob("Lab*.csv")))
        paths.extend(sorted(protocol_dir.glob("Lab*.parquet")))
        paths.extend(sorted(protocol_dir.glob("lab*.csv")))
        paths.extend(sorted(protocol_dir.glob("lab*.parquet")))
        if not paths:
            raise FileNotFoundError(
                f"No Lab*.csv or Lab*.parquet files found in {protocol_dir}"
            )
        logger.info(
            "%s files found:\n%s",
            protocol,
            "\n".join(f"- {path}" for path in paths),
        )
        protocol_frames = []
        for path in paths:
            labs = _read_lab_file(path)
            rows_before = len(labs)
            original_columns = set(labs.columns)
            harmonized = harmonize_labs(labs, rules)
            if len(harmonized) != rows_before:
                raise RuntimeError(f"Row-count validation failed for {path}")
            required_output_columns = {
                "Observation Value",
                "Observation Value Original",
            }
            missing_output = required_output_columns.difference(harmonized.columns)
            if missing_output:
                raise RuntimeError(
                    f"Harmonized output missing required columns for {path}: "
                    f"{sorted(missing_output)}"
                )
            lost_columns = original_columns.difference(harmonized.columns)
            if lost_columns:
                raise RuntimeError(
                    f"Harmonization lost input columns for {path}: "
                    f"{sorted(lost_columns)}"
                )
            _write_in_place(harmonized, path)
            protocol_frames.append(harmonized)
        combined = pd.concat(protocol_frames, ignore_index=True)
        for filename, table in build_qc(combined).items():
            output = qc_dir / f"20c_{protocol}_{filename.removeprefix('20c_')}"
            table.to_csv(output, index=False)
        counts = combined["harmonization_status"].value_counts()
        logger.info(
            "%s: rows=%d harmonized=%d SPECIAL skipped=%d EXCLUDE=%d NO_RULE=%d",
            protocol,
            len(combined),
            int(counts.get("HARMONIZED", 0)),
            int(counts.get("SKIPPED_SPECIAL", 0)),
            int(counts.get("EXCLUDED", 0)),
            int(counts.get("NO_RULE", 0)),
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line path overrides for batch execution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--btris-root", type=Path, default=BTRIS_ROOT)
    parser.add_argument("--rules", type=Path, default=RULE_INPUT)
    parser.add_argument("--qc-dir", type=Path, default=QC_DIR)
    return parser.parse_args()


def main() -> None:
    """Execute protocol-specific in-place harmonization."""
    args = parse_args()
    run(args.btris_root, args.rules, args.qc_dir)


if __name__ == "__main__":
    main()
