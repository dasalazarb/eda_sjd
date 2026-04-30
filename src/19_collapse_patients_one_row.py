from __future__ import annotations

import argparse

import pandas as pd

from common import ANALYTIC_DIR, MISSING_TOKENS, print_kv, print_script_overview, print_step, setup_logger

MISSING_TOKEN_UPPER = {str(token).upper() for token in MISSING_TOKENS}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collapse all rows per patient into one row per patient. "
            "When values overlap/conflict, values are joined with ' | '."
        )
    )
    parser.add_argument(
        "--input-path",
        default=str(ANALYTIC_DIR / "visits_long_collapsed_by_interval_codebook_corrected.parquet"),
        help="Input dataset path.",
    )
    parser.add_argument(
        "--output-path",
        default=str(ANALYTIC_DIR / "patients_one_row_collapsed.parquet"),
        help="Output parquet path.",
    )
    return parser.parse_args()


def _normalize_missing(values: pd.Series) -> pd.Series:
    out = values.astype("string").str.strip()
    missing_mask = out.isna() | out.str.upper().isin(MISSING_TOKEN_UPPER)
    return out.mask(missing_mask, pd.NA)


def _collapse_series(values: pd.Series):
    normalized = _normalize_missing(values)
    tokens: list[str] = []
    seen: set[str] = set()

    for value in normalized.dropna().tolist():
        for token in str(value).split("|"):
            clean = token.strip()
            if not clean:
                continue
            key = clean.casefold()
            if key in seen:
                continue
            seen.add(key)
            tokens.append(clean)

    if not tokens:
        return pd.NA
    if len(tokens) == 1:
        return tokens[0]
    return " | ".join(tokens)


def main() -> None:
    args = _parse_args()
    logger = setup_logger("collapse_patients_one_row")

    print_script_overview(
        script_name="19_collapse_patients_one_row.py",
        objective="Collapse all lines to one line per patient using ' | ' for overlapping values.",
    )

    print_step(1, "Loading source dataset")
    df = pd.read_parquet(args.input_path)
    print_kv("input_path", args.input_path)
    print_kv("rows", len(df))
    print_kv("cols", len(df.columns))

    subject_col = "subject_number" if "subject_number" in df.columns else "ids__subject_number"
    if subject_col not in df.columns:
        raise KeyError("Could not find patient identifier column: subject_number or ids__subject_number")

    print_step(2, "Collapsing all rows into one row per patient")
    agg_map = {col: _collapse_series for col in df.columns if col != subject_col}
    collapsed = (
        df.groupby(subject_col, dropna=False, sort=True)
        .agg(agg_map)
        .reset_index()
    )
    print_kv("patients", len(collapsed))

    print_step(3, "Saving output")
    output_path = args.output_path
    collapsed.to_parquet(output_path, index=False)
    collapsed.to_csv(output_path.replace(".parquet", ".csv"), index=False)
    print_kv("saved_parquet", output_path)
    print_kv("saved_csv", output_path.replace('.parquet', '.csv'))

    logger.info("Collapsed to one row per patient | input=%s output=%s rows=%d", args.input_path, output_path, len(collapsed))


if __name__ == "__main__":
    main()
