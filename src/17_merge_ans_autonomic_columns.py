from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from common import ANALYTIC_DIR, REPORTS_DIR, print_kv, print_script_overview, print_step, setup_logger

ANS_PREFIX = "ans__"
AUTO_PREFIX = "autonomic_nervous_system_questionnaire__"


@dataclass(frozen=True)
class MergeConfig:
    input_path: Path
    output_path: Path
    report_path: Path


def _parse_args() -> MergeConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Merge columns with prefixes 'ans__' and "
            "'autonomic_nervous_system_questionnaire__' by shared suffix."
        )
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=ANALYTIC_DIR / "visits_long_collapsed_by_interval_codebook_corrected.parquet",
        help="Path to input table (.parquet/.csv/.xlsx).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ANALYTIC_DIR / "visits_long_collapsed_by_interval_codebook_corrected_ans_merged.parquet",
        help="Path to merged output parquet.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=REPORTS_DIR / "ans_autonomic_merge_report.xlsx",
        help="Path to Excel report with merge diagnostics.",
    )
    args = parser.parse_args()
    return MergeConfig(input_path=args.input_path, output_path=args.output_path, report_path=args.report_path)


def _load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls", ".xlsm"}:
        return pd.read_excel(path)

    raise ValueError(f"Unsupported input format: {path}")


def _save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(path, index=False)
        return
    if suffix == ".csv":
        df.to_csv(path, index=False)
        return
    raise ValueError("Output path must be .parquet or .csv")


def _tokenize(value: object) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    tokens = [token.strip() for token in text.split("|")]
    return [token for token in tokens if token]


def _merge_row_values(values: list[object]) -> tuple[object, int]:
    merged: list[str] = []
    seen: set[str] = set()

    for value in values:
        for token in _tokenize(value):
            key = token.casefold()
            if key in seen:
                continue
            seen.add(key)
            merged.append(token)

    if not merged:
        return pd.NA, 0
    return " | ".join(merged), len(merged)


def _prefix_map(columns: list[str], prefix: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for col in columns:
        if col.startswith(prefix):
            suffix = col[len(prefix) :]
            out.setdefault(suffix, []).append(col)
    return out


def _build_report_rows(stems: list[str], status: str) -> pd.DataFrame:
    if not stems:
        return pd.DataFrame(columns=["suffix", "status"])
    return pd.DataFrame({"suffix": stems, "status": status})


def run(config: MergeConfig) -> None:
    logger = setup_logger("17_merge_ans_autonomic_columns")

    print_script_overview(
        "17_merge_ans_autonomic_columns",
        (
            "Load corrected visits table, merge ans__/autonomic columns by shared suffix, "
            "and save diagnostics report."
        ),
    )

    print_step(1, "Loading input table")
    df = _load_table(config.input_path)
    print_kv("input", {"rows": len(df), "columns": len(df.columns)})

    print_step(2, "Detecting mergeable column pairs by suffix")
    all_columns = [str(col) for col in df.columns]
    ans_map = _prefix_map(all_columns, ANS_PREFIX)
    auto_map = _prefix_map(all_columns, AUTO_PREFIX)

    ans_suffixes = set(ans_map)
    auto_suffixes = set(auto_map)
    shared_suffixes = sorted(ans_suffixes.intersection(auto_suffixes))
    ans_only_suffixes = sorted(ans_suffixes.difference(auto_suffixes))
    auto_only_suffixes = sorted(auto_suffixes.difference(ans_suffixes))

    print_kv(
        "pairing_summary",
        {
            "mergeable_pairs": len(shared_suffixes),
            "ans_only": len(ans_only_suffixes),
            "autonomic_only": len(auto_only_suffixes),
        },
    )

    print_step(3, "Merging shared suffix columns")
    out = df.copy()
    values_gt_two_rows: list[dict[str, object]] = []

    for suffix in shared_suffixes:
        source_cols = [*ans_map[suffix], *auto_map[suffix]]
        merged_col = f"{ANS_PREFIX}{suffix}"

        merged_values: list[object] = []
        token_counts: list[int] = []
        for row_values in out[source_cols].itertuples(index=False, name=None):
            merged_value, token_count = _merge_row_values(list(row_values))
            merged_values.append(merged_value)
            token_counts.append(token_count)

        out[merged_col] = merged_values

        for idx, token_count in enumerate(token_counts):
            if token_count > 2:
                values_gt_two_rows.append(
                    {
                        "row_index": idx,
                        "suffix": suffix,
                        "merged_column": merged_col,
                        "token_count": token_count,
                        "merged_value": merged_values[idx],
                    }
                )

        drop_cols = [col for col in source_cols if col != merged_col]
        if drop_cols:
            out = out.drop(columns=drop_cols)

    print_kv("merge_quality", {"cells_with_more_than_two_values": len(values_gt_two_rows)})

    print_step(4, "Saving merged dataset and diagnostics report")
    _save_table(out, config.output_path)

    report_summary = pd.DataFrame(
        [
            {"metric": "input_rows", "value": len(df)},
            {"metric": "input_columns", "value": len(df.columns)},
            {"metric": "output_rows", "value": len(out)},
            {"metric": "output_columns", "value": len(out.columns)},
            {"metric": "merged_pairs", "value": len(shared_suffixes)},
            {"metric": "ans_only_suffixes", "value": len(ans_only_suffixes)},
            {"metric": "autonomic_only_suffixes", "value": len(auto_only_suffixes)},
            {"metric": "cells_with_more_than_two_values", "value": len(values_gt_two_rows)},
        ]
    )

    merged_pairs_df = pd.DataFrame({"suffix": shared_suffixes})
    ans_only_df = _build_report_rows(ans_only_suffixes, "ans_only")
    auto_only_df = _build_report_rows(auto_only_suffixes, "autonomic_only")
    gt_two_df = pd.DataFrame(values_gt_two_rows)

    config.report_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(config.report_path) as writer:
        report_summary.to_excel(writer, sheet_name="summary", index=False)
        merged_pairs_df.to_excel(writer, sheet_name="merged_pairs", index=False)
        ans_only_df.to_excel(writer, sheet_name="ans_not_merged", index=False)
        auto_only_df.to_excel(writer, sheet_name="autonomic_not_merged", index=False)
        gt_two_df.to_excel(writer, sheet_name="cells_gt_2_values", index=False)

    logger.info("Saved merged dataset: %s", config.output_path)
    logger.info("Saved report: %s", config.report_path)
    print_kv("outputs", {"output_path": str(config.output_path), "report_path": str(config.report_path)})


if __name__ == "__main__":
    run(_parse_args())
