from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from common import ANALYTIC_DIR, print_kv, print_script_overview, print_step, setup_logger

ESSDAI_PREFIX_LEGACY = "essdai-_r__"
ESSDAI_PREFIX_CURRENT = "essdai_r__"


@dataclass(frozen=True)
class MergeConfig:
    input_path: Path
    output_base: Path


def _parse_args() -> MergeConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Merge ESSDAI columns by shared suffix from legacy prefix "
            "'essdai-_r__' into current prefix 'essdai_r__'."
        )
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=ANALYTIC_DIR / "visits_long_collapsed_by_interval.parquet",
        help="Path to input table (.parquet/.csv/.xlsx).",
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=ANALYTIC_DIR / "visits_long_collapsed_by_interval_essdai_merged",
        help="Output base path (without extension). Script writes both .parquet and .csv.",
    )
    args = parser.parse_args()
    return MergeConfig(input_path=args.input_path, output_base=args.output_base)


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


def _tokenize_cell(value: object) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    parts = [part.strip() for part in text.split("|")]
    return [part for part in parts if part]


def _merge_cell_values(values: list[object]) -> object:
    merged: list[str] = []
    seen: set[str] = set()

    for value in values:
        for token in _tokenize_cell(value):
            key = token.casefold()
            if key in seen:
                continue
            seen.add(key)
            merged.append(token)

    if not merged:
        return pd.NA
    if len(merged) == 1:
        return merged[0]
    return " | ".join(merged)


def _prefix_map(columns: list[str], prefix: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for col in columns:
        if col.startswith(prefix):
            suffix = col[len(prefix) :]
            out.setdefault(suffix, []).append(col)
    return out


def _merge_essdai_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    out = df.copy()
    colnames = [str(c) for c in out.columns]
    legacy_map = _prefix_map(colnames, ESSDAI_PREFIX_LEGACY)
    current_map = _prefix_map(colnames, ESSDAI_PREFIX_CURRENT)
    shared_suffixes = sorted(set(legacy_map).intersection(current_map))

    for suffix in shared_suffixes:
        source_cols = [*legacy_map[suffix], *current_map[suffix]]
        merged_col = f"{ESSDAI_PREFIX_CURRENT}{suffix}"
        out[merged_col] = out[source_cols].apply(lambda row: _merge_cell_values(row.tolist()), axis=1)
        drop_cols = [col for col in source_cols if col != merged_col]
        if drop_cols:
            out = out.drop(columns=drop_cols)

    return out, len(shared_suffixes)


def run(config: MergeConfig) -> None:
    logger = setup_logger("09b_merge_essdai_versions")
    print_script_overview(
        "09b_merge_essdai_versions.py",
        "Merge ESSDAI legacy/current columns by suffix and export both parquet+csv.",
    )

    print_step(1, "Load input")
    df = _load_table(config.input_path)

    print_step(2, "Merge ESSDAI legacy/current columns")
    merged, merged_pairs = _merge_essdai_columns(df)

    print_step(3, "Save outputs")
    config.output_base.parent.mkdir(parents=True, exist_ok=True)
    parquet_path = config.output_base.with_suffix(".parquet")
    csv_path = config.output_base.with_suffix(".csv")
    merged.to_parquet(parquet_path, index=False)
    merged.to_csv(csv_path, index=False)

    logger.info("Saved merged dataset parquet: %s", parquet_path)
    logger.info("Saved merged dataset csv: %s", csv_path)
    print_kv(
        "merge_summary",
        {
            "rows": len(merged),
            "columns": len(merged.columns),
            "merged_suffix_pairs": merged_pairs,
            "parquet_path": str(parquet_path),
            "csv_path": str(csv_path),
        },
    )


if __name__ == "__main__":
    run(_parse_args())
