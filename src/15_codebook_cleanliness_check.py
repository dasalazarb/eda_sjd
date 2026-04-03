from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from common import ANALYTIC_DIR, REPORTS_DIR, print_kv, print_script_overview, print_step, setup_logger

BLANK_TOKENS = {"", "nan", "none", "null", "na", "n/a"}


@dataclass(frozen=True)
class CheckConfig:
    codebook_path: Path
    codebook_sheet: str
    corrected_path: Path
    output_path: Path


def _parse_args() -> CheckConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Verifies codebook cleanliness column-by-column using unique values from "
            "visits_long_collapsed_by_interval_codebook_corrected."
        )
    )
    parser.add_argument(
        "--codebook-path",
        type=Path,
        default=Path("data_raw") / "Consolidated_Codebook_11D0172_15D0051.xlsx",
        help="Path to codebook workbook.",
    )
    parser.add_argument(
        "--codebook-sheet",
        type=str,
        default="Consolidated_Codebook",
        help="Codebook sheet name.",
    )
    parser.add_argument(
        "--corrected-path",
        type=Path,
        default=ANALYTIC_DIR / "visits_long_collapsed_by_interval_codebook_corrected.parquet",
        help="Path to corrected collapsed table (.xlsx/.csv/.parquet).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=REPORTS_DIR / "codebook_cleanliness_check.xlsx",
        help="Output Excel report path.",
    )
    args = parser.parse_args()
    return CheckConfig(
        codebook_path=args.codebook_path,
        codebook_sheet=args.codebook_sheet,
        corrected_path=args.corrected_path,
        output_path=args.output_path,
    )


def _normalize_token(value: object) -> str:
    if pd.isna(value):
        return ""
    text = re.sub(r"\s{2,}", " ", str(value)).strip().lower()
    if re.fullmatch(r"[+-]?\d+\.0+", text):
        text = text.split(".", 1)[0]
    text = re.sub(r"\s+", "_", text)
    return text


def _strip_repeat_suffix(value: object) -> str:
    text = str(value).strip()
    match = re.search(r"_(\d+)$", text)
    if match and int(match.group(1)) >= 2:
        return text[: match.start()]
    return text


def _is_blank(value: object) -> bool:
    return _normalize_token(value) in BLANK_TOKENS


def _split_by_delimiters(value: object) -> list[str]:
    if _is_blank(value):
        return []
    parts = [part.strip() for part in re.split(r"[|;\n]+", str(value))]
    return [part for part in parts if not _is_blank(part)]


def _parse_numeric_value(value: object) -> float | None:
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    if re.fullmatch(r"[+-]?\d+(\.\d+)?", text):
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _parse_answer_range(answer_range_raw: object) -> tuple[float | None, float | None]:
    text = str(answer_range_raw).strip().lower()
    if not text:
        return None, None

    between = re.search(r"([+-]?\d+(?:\.\d+)?)\s*(?:-|to)\s*([+-]?\d+(?:\.\d+)?)", text)
    if between:
        low = float(between.group(1))
        high = float(between.group(2))
        return min(low, high), max(low, high)

    ge = re.search(r">=\s*([+-]?\d+(?:\.\d+)?)", text)
    le = re.search(r"<=\s*([+-]?\d+(?:\.\d+)?)", text)
    gt = re.search(r">\s*([+-]?\d+(?:\.\d+)?)", text)
    lt = re.search(r"<\s*([+-]?\d+(?:\.\d+)?)", text)

    min_val = float(ge.group(1)) if ge else (float(gt.group(1)) if gt else None)
    max_val = float(le.group(1)) if le else (float(lt.group(1)) if lt else None)
    return min_val, max_val


def _load_table(path: Path, sheet_name: str | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet_name)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)

    raise ValueError(f"Unsupported format: {path}")


def _prepare_codebook(codebook: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"FORM_NAME__QUESTION_NAME", "DISPLAY", "CODEVALUE", "ANSWER_RANGE"}
    missing = required_cols.difference(codebook.columns)
    if missing:
        raise KeyError(f"Missing required codebook columns: {sorted(missing)}")

    selected = codebook[["FORM_NAME__QUESTION_NAME", "DISPLAY", "CODEVALUE", "ANSWER_RANGE"]].copy()
    selected["FORM_NAME__QUESTION_NAME"] = selected["FORM_NAME__QUESTION_NAME"].astype(str).str.strip()
    selected = selected[selected["FORM_NAME__QUESTION_NAME"] != ""]
    selected["merge_key"] = selected["FORM_NAME__QUESTION_NAME"].map(_strip_repeat_suffix)
    selected = selected[selected["merge_key"] != ""]

    aggregated = (
        selected.groupby("merge_key", as_index=False)
        .agg(
            {
                "DISPLAY": lambda s: " | ".join(str(v) for v in s if not _is_blank(v)),
                "CODEVALUE": lambda s: " | ".join(str(v) for v in s if not _is_blank(v)),
                "ANSWER_RANGE": lambda s: " | ".join(str(v) for v in s if not _is_blank(v)),
            }
        )
    )
    return aggregated


def _allowed_map(display_raw: object, codevalue_raw: object) -> dict[str, str]:
    allowed: dict[str, str] = {}
    for token in _split_by_delimiters(display_raw):
        allowed[_normalize_token(token)] = token
    for token in _split_by_delimiters(codevalue_raw):
        normalized = _normalize_token(token)
        if normalized not in allowed:
            allowed[normalized] = token
    return allowed


def _check_cleanliness(
    codebook_prepared: pd.DataFrame,
    corrected: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    column_map: dict[str, list[str]] = {}
    for col in corrected.columns.astype(str):
        column_map.setdefault(_strip_repeat_suffix(col), []).append(col)

    summary_rows: list[dict[str, object]] = []
    invalid_rows: list[dict[str, object]] = []
    pipe_rows: list[dict[str, object]] = []

    for _, cb in codebook_prepared.iterrows():
        merge_key = cb["merge_key"]
        cols = column_map.get(merge_key, [])
        if not cols:
            summary_rows.append(
                {
                    "merge_key": merge_key,
                    "status": "no_column_in_corrected",
                    "matched_columns": 0,
                    "unique_non_blank_values": 0,
                    "invalid_unique_values": 0,
                    "pipe_unique_values": 0,
                }
            )
            continue

        allowed = _allowed_map(cb["DISPLAY"], cb["CODEVALUE"])
        min_allowed, max_allowed = _parse_answer_range(cb.get("ANSWER_RANGE", ""))
        has_range = min_allowed is not None or max_allowed is not None

        observed_values: set[str] = set()
        invalid_values: set[str] = set()
        pipe_values: set[str] = set()

        for col in cols:
            for raw in corrected[col].dropna().tolist():
                value_text = str(raw).strip()
                if not value_text or _is_blank(value_text):
                    continue
                observed_values.add(value_text)

                if "|" in value_text:
                    pipe_values.add(value_text)
                    continue

                normalized = _normalize_token(value_text)
                is_allowed_exact = normalized in allowed if allowed else False

                numeric_value = _parse_numeric_value(value_text)
                in_range = True
                if has_range and numeric_value is not None:
                    if min_allowed is not None and numeric_value < min_allowed:
                        in_range = False
                    if max_allowed is not None and numeric_value > max_allowed:
                        in_range = False
                elif has_range and numeric_value is None and not is_allowed_exact:
                    in_range = False

                if not is_allowed_exact and not (has_range and in_range):
                    invalid_values.add(value_text)

        status = "clean"
        if pipe_values:
            status = "has_pipe_conflicts"
        if invalid_values:
            status = "has_invalid_values"

        summary_rows.append(
            {
                "merge_key": merge_key,
                "status": status,
                "matched_columns": len(cols),
                "unique_non_blank_values": len(observed_values),
                "invalid_unique_values": len(invalid_values),
                "pipe_unique_values": len(pipe_values),
            }
        )

        for value in sorted(invalid_values):
            invalid_rows.append({"merge_key": merge_key, "invalid_value": value})
        for value in sorted(pipe_values):
            pipe_rows.append({"merge_key": merge_key, "pipe_value": value})

    codebook_keys = set(codebook_prepared["merge_key"].astype(str))
    unmatched_corrected_rows = []
    for base_col, original_cols in sorted(column_map.items()):
        if base_col not in codebook_keys:
            unmatched_corrected_rows.append(
                {"merge_key": base_col, "source_columns": " | ".join(original_cols), "reason": "not_found_in_codebook"}
            )

    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(invalid_rows),
        pd.DataFrame(pipe_rows),
        pd.DataFrame(unmatched_corrected_rows),
    )


def main() -> None:
    config = _parse_args()
    logger = setup_logger("15_codebook_cleanliness_check")

    print_script_overview(
        "15_codebook_cleanliness_check.py",
        "Checks whether unique values in corrected collapsed dataset are clean against the codebook.",
    )

    print_step(1, "Load codebook and corrected collapsed dataset")
    codebook = _load_table(config.codebook_path, sheet_name=config.codebook_sheet)
    corrected = _load_table(config.corrected_path)

    print_step(2, "Build codebook merge keys and allowed values")
    codebook_prepared = _prepare_codebook(codebook)

    print_step(3, "Evaluate unique corrected values variable-by-variable")
    summary, invalid_values, pipe_values, unmatched_corrected = _check_cleanliness(codebook_prepared, corrected)

    print_step(4, "Save cleanliness report")
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(config.output_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary_by_variable", index=False)
        invalid_values.to_excel(writer, sheet_name="invalid_unique_values", index=False)
        pipe_values.to_excel(writer, sheet_name="pipe_unique_values", index=False)
        unmatched_corrected.to_excel(writer, sheet_name="unmatched_corrected_cols", index=False)

    summary_csv = config.output_path.with_suffix(".csv")
    summary.to_csv(summary_csv, index=False)

    metrics = {
        "variables_in_codebook": len(codebook_prepared),
        "variables_clean": int((summary["status"] == "clean").sum()) if not summary.empty else 0,
        "variables_with_invalid_values": int((summary["invalid_unique_values"] > 0).sum()) if not summary.empty else 0,
        "variables_with_pipe_values": int((summary["pipe_unique_values"] > 0).sum()) if not summary.empty else 0,
        "unmatched_corrected_columns": len(unmatched_corrected),
        "output_xlsx": str(config.output_path),
        "output_csv": str(summary_csv),
    }
    print_kv("Codebook cleanliness check", metrics)
    logger.info("Saved report: %s", config.output_path)
    logger.info("Saved summary CSV: %s", summary_csv)


if __name__ == "__main__":
    main()
