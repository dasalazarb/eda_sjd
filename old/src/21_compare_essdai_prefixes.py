from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import ANALYTIC_DIR, REPORTS_DIR, print_kv, print_script_overview, print_step, setup_logger


PREFIX_A = "essdai__"
DEFAULT_PREFIX_B_CANDIDATES = ("essdai-_r__", "essdai_r__")
MISSING_TOKENS = {"", "nan", "none", "null", "na", "n/a"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare essdai__ vs essdai-r__ columns in a parquet dataset, reviewing "
            "both variable names and observed values."
        )
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=ANALYTIC_DIR / "visits_long_collapsed_by_interval_codebook_corrected.parquet",
        help="Path to input parquet file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPORTS_DIR / "essdai_prefix_compare",
        help="Directory where comparison CSV outputs will be saved.",
    )
    parser.add_argument(
        "--prefix-a",
        type=str,
        default=PREFIX_A,
        help="Primary ESSDAI prefix to compare.",
    )
    parser.add_argument(
        "--prefix-b-candidates",
        type=str,
        default=",".join(DEFAULT_PREFIX_B_CANDIDATES),
        help="Comma-separated list of alternative prefixes for the ESSDAI-R side.",
    )
    return parser.parse_args()


def _normalize_value(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in MISSING_TOKENS:
        return ""
    return text


def _value_set(series: pd.Series) -> set[str]:
    return {_normalize_value(v) for v in series.tolist()} - {""}


def _extract_suffix_columns(columns: list[str], prefix: str) -> dict[str, str]:
    return {col[len(prefix) :]: col for col in columns if col.startswith(prefix)}


def _extract_suffix_columns_multi_prefix(
    columns: list[str], prefixes: list[str]
) -> tuple[dict[str, str], dict[str, str]]:
    selected: dict[str, str] = {}
    selected_prefix: dict[str, str] = {}

    for col in columns:
        for prefix in prefixes:
            if col.startswith(prefix):
                suffix = col[len(prefix) :]
                selected[suffix] = col
                selected_prefix[suffix] = prefix
                break

    return selected, selected_prefix


def main() -> None:
    args = _parse_args()
    logger = setup_logger("21_compare_essdai_prefixes")
    prefix_b_candidates = [p.strip() for p in args.prefix_b_candidates.split(",") if p.strip()]
    if not prefix_b_candidates:
        raise ValueError("At least one value is required in --prefix-b-candidates")

    print_script_overview(
        "21_compare_essdai_prefixes.py",
        "Compara columnas essdai__ vs essdai-r__ en nombres y valores observados.",
    )

    print_step(1, "Loading dataset")
    if not args.input_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_path}")
    df = pd.read_parquet(args.input_path)
    logger.info("Loaded dataset %s rows=%d cols=%d", args.input_path, len(df), len(df.columns))
    print_kv(
        "Input summary",
        {
            "path": args.input_path,
            "rows": len(df),
            "cols": len(df.columns),
        },
    )

    print_step(2, "Collecting prefixed columns")
    all_cols = [str(c) for c in df.columns]
    map_a = _extract_suffix_columns(all_cols, args.prefix_a)
    map_b, map_b_prefix = _extract_suffix_columns_multi_prefix(all_cols, prefix_b_candidates)

    suffix_a = set(map_a.keys())
    suffix_b = set(map_b.keys())

    only_a = sorted(suffix_a - suffix_b)
    only_b = sorted(suffix_b - suffix_a)
    common = sorted(suffix_a & suffix_b)

    print_kv(
        "Prefix counts",
        {
            args.prefix_a: len(suffix_a),
            ",".join(prefix_b_candidates): len(suffix_b),
            "common_variables": len(common),
            "only_essdai": len(only_a),
            "only_essdai_r": len(only_b),
        },
    )

    print_step(3, "Comparing values for shared variables")
    value_rows: list[dict[str, object]] = []
    for suffix in common:
        col_a = map_a[suffix]
        col_b = map_b[suffix]

        values_a = _value_set(df[col_a])
        values_b = _value_set(df[col_b])

        only_values_a = sorted(values_a - values_b)
        only_values_b = sorted(values_b - values_a)

        value_rows.append(
            {
                "variable_suffix": suffix,
                "column_essdai": col_a,
                "column_essdai_r": col_b,
                "matched_prefix_essdai_r": map_b_prefix[suffix],
                "n_unique_essdai": len(values_a),
                "n_unique_essdai_r": len(values_b),
                "same_unique_values": len(only_values_a) == 0 and len(only_values_b) == 0,
                "unique_only_essdai": " | ".join(only_values_a[:50]),
                "unique_only_essdai_r": " | ".join(only_values_b[:50]),
                "n_unique_only_essdai": len(only_values_a),
                "n_unique_only_essdai_r": len(only_values_b),
            }
        )

    value_df = pd.DataFrame(value_rows)
    if value_df.empty:
        value_df = pd.DataFrame(
            columns=[
                "variable_suffix",
                "column_essdai",
                "column_essdai_r",
                "matched_prefix_essdai_r",
                "n_unique_essdai",
                "n_unique_essdai_r",
                "same_unique_values",
                "unique_only_essdai",
                "unique_only_essdai_r",
                "n_unique_only_essdai",
                "n_unique_only_essdai_r",
            ]
        )
    else:
        value_df = value_df.sort_values(by=["same_unique_values", "variable_suffix"], ascending=[True, True])

    print_step(4, "Saving outputs")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(
        [
            {
                "prefix_essdai_variables": len(suffix_a),
                "prefix_essdai_r_variables": len(suffix_b),
                "prefix_essdai": args.prefix_a,
                "prefix_essdai_r_candidates": ",".join(prefix_b_candidates),
                "common_variables": len(common),
                "only_essdai_variables": len(only_a),
                "only_essdai_r_variables": len(only_b),
                "common_with_equal_unique_values": int(value_df["same_unique_values"].sum()) if not value_df.empty else 0,
                "common_with_different_unique_values": int((~value_df["same_unique_values"]).sum()) if not value_df.empty else 0,
            }
        ]
    )

    only_a_df = pd.DataFrame({"variable_suffix": only_a})
    only_b_df = pd.DataFrame({"variable_suffix": only_b})

    summary_path = args.output_dir / "essdai_prefix_summary.csv"
    only_a_path = args.output_dir / "only_in_essdai.csv"
    only_b_path = args.output_dir / "only_in_essdai_r.csv"
    value_path = args.output_dir / "shared_variables_value_comparison.csv"

    summary_df.to_csv(summary_path, index=False)
    only_a_df.to_csv(only_a_path, index=False)
    only_b_df.to_csv(only_b_path, index=False)
    value_df.to_csv(value_path, index=False)

    logger.info("Saved: %s", summary_path)
    logger.info("Saved: %s", only_a_path)
    logger.info("Saved: %s", only_b_path)
    logger.info("Saved: %s", value_path)

    print_kv(
        "Saved files",
        {
            "summary": summary_path,
            "only_essdai": only_a_path,
            "only_essdai_r": only_b_path,
            "shared_value_comparison": value_path,
        },
    )


if __name__ == "__main__":
    main()
