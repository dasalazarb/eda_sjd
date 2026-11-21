"""Audit completeness by patient, visit, and clinical variable prefix.

This is a read-only QC utility.  It does not impute values or modify its input
dataset. Prefixes and expected variables are discovered exclusively from the
analytical input schema; no external codebook is required.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.common import ANALYTIC_DIR, ROOT

PATIENT_COLUMN = "ids__patient_record_number"
VISIT_DATE_COLUMN = "ids__visit_date"
INTERVAL_COLUMN = "ids__interval_name"
TEXT_MISSING_TOKENS = {"", "na", "n/a", "nan", "none", "unknown", "unk"}
OUTPUT_STEM = "code_del_check_pcr_info"


def load_input(path: Path) -> pd.DataFrame:
    """Load the longitudinal analytical dataset without changing it.

    Parameters
    ----------
    path : pathlib.Path
        Parquet or CSV input path.

    Returns
    -------
    pandas.DataFrame
        The analytical table.
    """
    if not path.exists():
        raise FileNotFoundError(f"Analytical input does not exist: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported analytical input format: {path.suffix}")


def is_missing(value: object) -> bool:
    """Return whether a scalar lacks information under this audit's rules.

    Numeric sentinels such as ``-99`` are deliberately not treated as missing:
    no repository-wide numeric-sentinel rule exists in ``src/common.py``.
    """
    missing = pd.isna(value)
    if isinstance(missing, bool) and missing:
        return True
    if isinstance(value, str):
        return value.strip().casefold() in TEXT_MISSING_TOKENS
    return False


def get_prefix(variable: str) -> str:
    """Extract everything before the first double underscore.

    Parameters
    ----------
    variable : str
        Canonical variable or input-column name.

    Returns
    -------
    str
        Variable prefix.
    """
    if "__" not in variable:
        raise ValueError(f"Variable has no '__' prefix separator: {variable!r}")
    return variable.split("__", 1)[0]


def build_prefix_dictionary(
    input_columns: Sequence[object],
) -> dict[str, dict[str, object]]:
    """Build the prefix-to-variable mapping from the analytical schema."""
    clinical_columns = [
        str(column)
        for column in input_columns
        if "__" in str(column) and not str(column).startswith("ids__")
    ]
    input_by_prefix: dict[str, list[str]] = {}
    for variable in clinical_columns:
        input_by_prefix.setdefault(get_prefix(variable), []).append(variable)

    result: dict[str, dict[str, object]] = {}
    for prefix in sorted(input_by_prefix):
        input_variables = sorted(set(input_by_prefix.get(prefix, [])))
        result[prefix] = {
            "input_variables": input_variables,
            "n_input_variables": len(input_variables),
        }
    return result


def _identifier_output_columns(has_interval: bool) -> list[str]:
    columns = ["patient_record_number", "visit_date"]
    if has_interval:
        columns.append("interval_name")
    return columns


def _input_visit_key_columns(data: pd.DataFrame) -> list[str]:
    """Return the columns that uniquely identify a visit in the input.

    The analytical dataset can legitimately contain more than one interval on
    the same calendar date.  When interval metadata is available, it is part of
    the visit identity rather than evidence of a duplicate row.
    """
    columns = [PATIENT_COLUMN, VISIT_DATE_COLUMN]
    if INTERVAL_COLUMN in data.columns:
        columns.append(INTERVAL_COLUMN)
    return columns


def calculate_visit_prefix_completeness(
    data: pd.DataFrame, prefix_dictionary: dict[str, dict[str, object]]
) -> pd.DataFrame:
    """Calculate one completeness record per patient, visit, and input prefix."""
    has_interval = INTERVAL_COLUMN in data.columns
    rows: list[dict[str, object]] = []
    for _, visit in data.iterrows():
        identifiers: dict[str, object] = {
            "patient_record_number": visit[PATIENT_COLUMN],
            "visit_date": visit[VISIT_DATE_COLUMN],
        }
        if has_interval:
            identifiers["interval_name"] = visit[INTERVAL_COLUMN]
        for prefix, metadata in prefix_dictionary.items():
            variables = metadata["input_variables"]
            if not variables:
                continue
            missing_variables = [
                variable for variable in variables if is_missing(visit[variable])
            ]
            n_input = int(metadata["n_input_variables"])
            n_missing = len(missing_variables)
            rows.append(
                {
                    **identifiers,
                    "prefix": prefix,
                    "n_input_variables": n_input,
                    "n_completed": n_input - n_missing,
                    "n_missing": n_missing,
                    "completeness_pct": 100.0 * (n_input - n_missing) / n_input,
                    "missing_variables": " | ".join(missing_variables),
                }
            )
    long_output = pd.DataFrame(rows)
    sort_columns = ["patient_record_number", "visit_date", "prefix"]
    return long_output.sort_values(sort_columns, kind="stable").reset_index(drop=True)


def create_wide_matrix(long_output: pd.DataFrame) -> pd.DataFrame:
    """Create a patient-visit matrix of prefix completeness percentages."""
    identifiers = _identifier_output_columns("interval_name" in long_output.columns)
    visits = long_output[identifiers].drop_duplicates()
    percentages = long_output.pivot(
        index=identifiers,
        columns="prefix",
        values="completeness_pct",
    ).reset_index()
    matrix = visits.merge(percentages, on=identifiers, how="left")
    return matrix.sort_values(identifiers, kind="stable")


def create_prefix_summary(
    long_output: pd.DataFrame, prefix_dictionary: dict[str, dict[str, object]]
) -> pd.DataFrame:
    """Summarize visit completeness for every input-derived prefix."""
    rows: list[dict[str, object]] = []
    grouped = {
        prefix: group for prefix, group in long_output.groupby("prefix", sort=False)
    }
    for prefix, metadata in prefix_dictionary.items():
        values = grouped.get(prefix, pd.DataFrame()).get(
            "completeness_pct", pd.Series(dtype=float)
        )
        n_input = int(metadata["n_input_variables"])
        rows.append(
            {
                "prefix": prefix,
                "n_input_variables": n_input,
                "n_patient_visits": len(values),
                "mean_completeness_pct": values.mean(),
                "median_completeness_pct": values.median(),
                "q1_completeness_pct": values.quantile(0.25),
                "q3_completeness_pct": values.quantile(0.75),
                "min_completeness_pct": values.min(),
                "max_completeness_pct": values.max(),
                "pct_visits_0_complete": (
                    100.0 * values.eq(0).mean() if len(values) else pd.NA
                ),
                "pct_visits_lt25_complete": (
                    100.0 * values.lt(25).mean() if len(values) else pd.NA
                ),
                "pct_visits_lt50_complete": (
                    100.0 * values.lt(50).mean() if len(values) else pd.NA
                ),
                "pct_visits_ge80_complete": (
                    100.0 * values.ge(80).mean() if len(values) else pd.NA
                ),
                "pct_visits_100_complete": (
                    100.0 * values.eq(100).mean() if len(values) else pd.NA
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("prefix").reset_index(drop=True)


def create_visit_summary(long_output: pd.DataFrame) -> pd.DataFrame:
    """Create the secondary, variable-weighted overall visit summary."""
    identifiers = _identifier_output_columns("interval_name" in long_output.columns)
    summary = (
        long_output.groupby(identifiers, dropna=False, as_index=False)
        .agg(
            n_evaluated_prefixes=("prefix", "nunique"),
            total_available_input_variables=("n_input_variables", "sum"),
            total_completed_variables=("n_completed", "sum"),
        )
        .sort_values(["patient_record_number", "visit_date"], kind="stable")
    )
    summary["overall_completeness_pct"] = (
        100.0
        * summary["total_completed_variables"]
        / summary["total_available_input_variables"]
    )
    return summary


def run_qc_checks(
    data: pd.DataFrame,
    prefix_dictionary: dict[str, dict[str, object]] | None = None,
    long_output: pd.DataFrame | None = None,
) -> None:
    """Raise a clear error when input or calculated-output invariants fail."""
    missing_identifiers = [
        c for c in [PATIENT_COLUMN, VISIT_DATE_COLUMN] if c not in data.columns
    ]
    if missing_identifiers:
        raise ValueError(
            f"Analytical input is missing required columns: {missing_identifiers}"
        )
    separated = [str(column) for column in data.columns if "__" in str(column)]
    if not separated:
        raise ValueError("No input variables containing '__' were found.")
    clinical = [column for column in separated if not column.startswith("ids__")]
    if not clinical:
        raise ValueError(
            "No clinical variables remain after excluding ids__ metadata columns."
        )
    visit_key = _input_visit_key_columns(data)
    duplicates = data.duplicated(visit_key, keep=False)
    if duplicates.any():
        examples = data.loc[duplicates, visit_key].head(5).to_dict("records")
        raise ValueError(
            "Duplicate visit rows found using patient, date, and interval when "
            "available; rows will not be aggregated. "
            f"First duplicate keys: {examples}"
        )
    if prefix_dictionary is None or long_output is None:
        return
    assigned = [
        v for details in prefix_dictionary.values() for v in details["input_variables"]
    ]
    if len(assigned) != len(set(assigned)) or set(assigned) != set(clinical):
        raise AssertionError(
            "Each clinical input variable must map to exactly one prefix."
        )
    if any(variable.startswith("ids__") for variable in assigned):
        raise AssertionError(
            "Identifier variables were included in clinical calculations."
        )
    if not long_output["completeness_pct"].between(0, 100, inclusive="both").all():
        raise AssertionError("Completeness percentages must be between 0 and 100.")
    if not (
        long_output["n_completed"] + long_output["n_missing"]
        == long_output["n_input_variables"]
    ).all():
        raise AssertionError(
            "Completed plus missing counts must equal input variable counts."
        )
    keys = _identifier_output_columns("interval_name" in long_output.columns)
    keys.append("prefix")
    if long_output.duplicated(keys).any():
        raise AssertionError(
            "Duplicate patient × visit × prefix output rows were produced."
        )


def make_figures(
    long_output: pd.DataFrame, matrix: pd.DataFrame, output_dir: Path
) -> None:
    """Write the heatmap and prefix-distribution QC figures as PDFs."""
    identifiers = _identifier_output_columns("interval_name" in matrix.columns)
    heatmap_data = matrix.drop(columns=identifiers)
    figure_height = min(30, max(4, len(matrix) * 0.08))
    plt.figure(figsize=(max(8, len(heatmap_data.columns) * 0.45), figure_height))
    sns.heatmap(heatmap_data, vmin=0, vmax=100, cmap="viridis", yticklabels=False)
    plt.xlabel("Clinical prefix")
    plt.ylabel("Patient visits (ordered by patient and date)")
    plt.tight_layout()
    plt.savefig(output_dir / f"{OUTPUT_STEM}_heatmap.pdf")
    plt.close()

    medians = long_output.groupby("prefix")["completeness_pct"].median().sort_values()
    plt.figure(figsize=(max(9, len(medians) * 0.42), 6))
    sns.boxplot(
        data=long_output,
        x="prefix",
        y="completeness_pct",
        order=medians.index,
        color="#4C78A8",
        showfliers=False,
    )
    plt.xticks(rotation=75, ha="right")
    plt.ylim(0, 100)
    plt.xlabel("Clinical prefix (lowest median first)")
    plt.ylabel("Visit completeness (%)")
    plt.tight_layout()
    plt.savefig(output_dir / f"{OUTPUT_STEM}_prefix_distribution.pdf")
    plt.close()


def _print_console_report(
    data: pd.DataFrame, prefix_summary: pd.DataFrame, output_paths: Sequence[Path]
) -> None:
    evaluated = prefix_summary[prefix_summary["n_input_variables"] > 0]
    print("=" * 56)
    print("Patient-Visit Prefix Completeness QC")
    print("=" * 56)
    print(f"Patients: {data[PATIENT_COLUMN].nunique()}")
    print(f"Visits: {len(data)}")
    print(f"Clinical prefixes evaluated: {len(evaluated)}")
    print(f"Clinical variables evaluated: {int(evaluated['n_input_variables'].sum())}")
    print("\nLowest median-completeness prefixes:")
    lowest = evaluated.nsmallest(5, "median_completeness_pct")
    for rank, row in enumerate(lowest.itertuples(), start=1):
        print(f"{rank}. {row.prefix}: {row.median_completeness_pct:.1f}%")
    for requested_prefix in ["essdai", "esspri_questionnaire"]:
        selected = evaluated[evaluated["prefix"] == requested_prefix]
        if not selected.empty:
            row = selected.iloc[0]
            print(
                f"\n{requested_prefix}: median={row['median_completeness_pct']:.1f}%, "
                f"mean={row['mean_completeness_pct']:.1f}%, visits={int(row['n_patient_visits'])}"
            )
    print("\nOutputs:")
    for path in output_paths:
        print(path)
    print("=" * 56)


def parse_args() -> argparse.Namespace:
    """Parse input and output paths for the read-only QC run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ANALYTIC_DIR
        / "visits_long_collapsed_by_interval_codebook_corrected.parquet",
        help="Canonical longitudinal analytical Parquet (or CSV) input.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "qc",
        help="Directory for QC-only CSV and PDF outputs.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the complete patient-visit-prefix completeness audit."""
    args = parse_args()
    data = load_input(args.input)
    run_qc_checks(data)
    prefix_dictionary = build_prefix_dictionary(data.columns)
    long_output = calculate_visit_prefix_completeness(data, prefix_dictionary)
    run_qc_checks(data, prefix_dictionary, long_output)
    matrix = create_wide_matrix(long_output)
    prefix_summary = create_prefix_summary(long_output, prefix_dictionary)
    visit_summary = create_visit_summary(long_output)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = [
        args.output_dir / f"{OUTPUT_STEM}_by_visit_prefix.csv",
        args.output_dir / f"{OUTPUT_STEM}_matrix.csv",
        args.output_dir / f"{OUTPUT_STEM}_prefix_summary.csv",
        args.output_dir / f"{OUTPUT_STEM}_visit_summary.csv",
        args.output_dir / f"{OUTPUT_STEM}_heatmap.pdf",
        args.output_dir / f"{OUTPUT_STEM}_prefix_distribution.pdf",
    ]
    for frame, path in zip(
        [long_output, matrix, prefix_summary, visit_summary],
        output_paths[:4],
        strict=True,
    ):
        frame.to_csv(path, index=False)
    make_figures(long_output, matrix, args.output_dir)
    _print_console_report(data, prefix_summary, output_paths)


if __name__ == "__main__":
    main()
