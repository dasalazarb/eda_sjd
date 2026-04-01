from __future__ import annotations

import argparse
import difflib
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from common import ANALYTIC_DIR, REPORTS_DIR, print_kv, print_script_overview, print_step, setup_logger

BLANK_TOKENS = {"", "nan", "none", "null", "na", "n/a"}


@dataclass(frozen=True)
class AuditConfig:
    codebook_path: Path
    codebook_sheet: str
    collapsed_path: Path
    output_path: Path
    corrected_output_path: Path


def _parse_args() -> AuditConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Audits values from visits_long_collapsed_by_interval using the new codebook schema "
            "(FORM_NAME__QUESTION_NAME, DISPLAY, CODEVALUE) and writes a corrected collapsed file."
        )
    )
    parser.add_argument(
        "--codebook-path",
        type=Path,
        default=Path("data_raw") / "Consolidated_Codebook_11D0172_15D0051.xlsx",
        help="Path to codebook workbook (default: data_raw/Consolidated_Codebook_11D0172_15D0051.xlsx).",
    )
    parser.add_argument(
        "--codebook-sheet",
        type=str,
        default="Consolidated_Codebook",
        help="Codebook sheet name (default: Consolidated_Codebook).",
    )
    parser.add_argument(
        "--collapsed-path",
        type=Path,
        default=ANALYTIC_DIR / "visits_long_collapsed_by_interval.parquet",
        help=(
            "Path to collapsed visits table (.xlsx/.csv/.parquet). "
            "Default: data_analytic/visits_long_collapsed_by_interval.parquet"
        ),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=REPORTS_DIR / "codebook_value_audit.xlsx",
        help="Output Excel report path (default: reports/codebook_value_audit.xlsx).",
    )
    parser.add_argument(
        "--corrected-output-path",
        type=Path,
        default=ANALYTIC_DIR / "visits_long_collapsed_by_interval_codebook_corrected.parquet",
        help=(
            "Path to corrected collapsed table (.xlsx/.csv/.parquet). "
            "Default: data_analytic/visits_long_collapsed_by_interval_codebook_corrected.parquet"
        ),
    )
    args = parser.parse_args()
    return AuditConfig(
        codebook_path=args.codebook_path,
        codebook_sheet=args.codebook_sheet,
        collapsed_path=args.collapsed_path,
        output_path=args.output_path,
        corrected_output_path=args.corrected_output_path,
    )


def _normalize_token(value: object) -> str:
    if pd.isna(value):
        return ""
    text = re.sub(r"\s{2,}", " ", str(value)).strip().lower()
    # Normalize numeric labels from spreadsheets (e.g., "1.0" -> "1")
    # so integer code values compare consistently.
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
    token = _normalize_token(value)
    return token in BLANK_TOKENS


def _split_by_delimiters(value: object) -> list[str]:
    if _is_blank(value):
        return []
    # Delimiters for multi-select labels in the codebook are pipe/semicolon/newline.
    # Do not split by comma because many labels include explanatory commas
    # (e.g., "for example, less sweet"), and splitting on commas creates
    # invalid partial tokens that later appear as no_match.
    parts = [part.strip() for part in re.split(r"[|;\n]+", str(value))]
    return [part for part in parts if _normalize_token(part) not in BLANK_TOKENS]


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


def _save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        df.to_excel(path, index=False)
        return
    if suffix == ".csv":
        df.to_csv(path, index=False)
        return
    if suffix == ".parquet":
        df.to_parquet(path, index=False)
        return
    raise ValueError(f"Unsupported format for output: {path}")


def _csv_path_for(path: Path) -> Path:
    return path.with_suffix(".csv")


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

    # If merge key appears multiple times, combine all DISPLAY/CODEVALUE possibilities.
    agg = (
        selected.groupby("merge_key", as_index=False)
        .agg(
            {
                "DISPLAY": lambda s: " | ".join([str(v) for v in s if not _is_blank(v)]),
                "CODEVALUE": lambda s: " | ".join([str(v) for v in s if not _is_blank(v)]),
                "ANSWER_RANGE": lambda s: " | ".join([str(v) for v in s if not _is_blank(v)]),
            }
        )
    )
    return agg


def _allowed_value_maps(display_raw: object, codevalue_raw: object) -> dict[str, str]:
    display_vals = _split_by_delimiters(display_raw)
    code_vals = _split_by_delimiters(codevalue_raw)

    allowed: dict[str, str] = {}
    for token in display_vals:
        allowed[_normalize_token(token)] = token.strip()
    for token in code_vals:
        n = _normalize_token(token)
        if n not in allowed:
            allowed[n] = token.strip()

    return allowed


def _best_fuzzy_match(observed: str, allowed_norms: set[str], cutoff: float = 0.9) -> str | None:
    obs_norm = _normalize_token(observed)
    if not obs_norm or not allowed_norms:
        return None
    match = difflib.get_close_matches(obs_norm, list(allowed_norms), n=1, cutoff=cutoff)
    return match[0] if match else None


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

    # Supported examples: "0-10", "0 to 10", ">=0", "<=120", "> 2", "< 9".
    between = re.search(r"([+-]?\d+(?:\.\d+)?)\s*(?:-|to)\s*([+-]?\d+(?:\.\d+)?)", text)
    if between:
        low = float(between.group(1))
        high = float(between.group(2))
        return (min(low, high), max(low, high))

    ge = re.search(r">=\s*([+-]?\d+(?:\.\d+)?)", text)
    le = re.search(r"<=\s*([+-]?\d+(?:\.\d+)?)", text)
    gt = re.search(r">\s*([+-]?\d+(?:\.\d+)?)", text)
    lt = re.search(r"<\s*([+-]?\d+(?:\.\d+)?)", text)

    min_val = float(ge.group(1)) if ge else (float(gt.group(1)) if gt else None)
    max_val = float(le.group(1)) if le else (float(lt.group(1)) if lt else None)
    return min_val, max_val


def _audit_and_correct(
    codebook_prepared: pd.DataFrame, collapsed: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if collapsed.empty:
        empty = pd.DataFrame()
        return empty, pd.DataFrame([{"metric": "matched_variables", "value": 0}]), empty, collapsed.copy(), empty

    corrected = collapsed.copy()
    findings: list[dict[str, object]] = []
    skipped_vars: list[dict[str, object]] = []
    pipe_conflicts: list[dict[str, object]] = []
    range_findings: list[dict[str, object]] = []

    column_map: dict[str, list[str]] = {}
    for col in collapsed.columns.astype(str):
        base_col = _strip_repeat_suffix(col)
        column_map.setdefault(base_col, []).append(col)

    for _, cb in codebook_prepared.iterrows():
        variable_name = cb["merge_key"]
        variable_columns = column_map.get(variable_name, [])
        if not variable_columns:
            continue

        allowed_map = _allowed_value_maps(cb["DISPLAY"], cb["CODEVALUE"])
        min_allowed, max_allowed = _parse_answer_range(cb.get("ANSWER_RANGE", ""))
        has_range = min_allowed is not None or max_allowed is not None

        if not allowed_map and not has_range:
            skipped_vars.append(
                {
                    "merge_key": variable_name,
                    "reason": "DISPLAY/CODEVALUE y ANSWER_RANGE vacíos o no parseables; variable omitida",
                }
            )
            continue

        for source_column in variable_columns:
            for idx, raw_value in collapsed[source_column].items():
                if _is_blank(raw_value):
                    continue

                value_text = str(raw_value).strip()
                if not value_text:
                    continue

                if "|" in value_text:
                    pipe_conflicts.append(
                        {
                            "merge_key": variable_name,
                            "source_column": source_column,
                            "row_index": idx,
                            "original_value": value_text,
                            "reason": "Contiene '|' y se deja sin corregir",
                        }
                    )
                    findings.append(
                        {
                            "merge_key": variable_name,
                            "source_column": source_column,
                            "row_index": idx,
                            "original_value": value_text,
                            "final_value": value_text,
                            "is_valid": False,
                            "correction_applied": False,
                            "match_method": "pipe_conflict",
                            "DISPLAY": cb["DISPLAY"],
                            "CODEVALUE": cb["CODEVALUE"],
                        }
                    )
                    continue

                if allowed_map:
                    normalized = _normalize_token(value_text)
                    corrected_value = value_text
                    is_valid = normalized in allowed_map
                    correction_applied = False
                    method = "exact" if is_valid else "no_match"

                    if not is_valid:
                        fuzzy = _best_fuzzy_match(value_text, set(allowed_map.keys()), cutoff=0.9)
                        if fuzzy is not None:
                            corrected_value = allowed_map[fuzzy]
                            is_valid = True
                            correction_applied = True
                            method = "fuzzy_corrected"
                            corrected.at[idx, source_column] = corrected_value

                    findings.append(
                        {
                            "merge_key": variable_name,
                            "source_column": source_column,
                            "row_index": idx,
                            "original_value": value_text,
                            "final_value": corrected_value,
                            "is_valid": is_valid,
                            "correction_applied": correction_applied,
                            "match_method": method,
                            "DISPLAY": cb["DISPLAY"],
                            "CODEVALUE": cb["CODEVALUE"],
                        }
                    )

                if not has_range:
                    continue

                numeric_value = _parse_numeric_value(value_text)
                if numeric_value is None:
                    range_findings.append(
                        {
                            "merge_key": variable_name,
                            "source_column": source_column,
                            "row_index": idx,
                            "original_value": value_text,
                            "answer_range": cb.get("ANSWER_RANGE", ""),
                            "issue_type": "text_or_non_numeric",
                            "issue_detail": "Valor no numérico para variable con ANSWER_RANGE.",
                        }
                    )
                    continue

                if numeric_value < 0:
                    range_findings.append(
                        {
                            "merge_key": variable_name,
                            "source_column": source_column,
                            "row_index": idx,
                            "original_value": value_text,
                            "answer_range": cb.get("ANSWER_RANGE", ""),
                            "issue_type": "negative_value",
                            "issue_detail": "Valor negativo detectado.",
                        }
                    )

                if min_allowed is not None and numeric_value < min_allowed:
                    range_findings.append(
                        {
                            "merge_key": variable_name,
                            "source_column": source_column,
                            "row_index": idx,
                            "original_value": value_text,
                            "answer_range": cb.get("ANSWER_RANGE", ""),
                            "issue_type": "below_min_range",
                            "issue_detail": f"Valor {numeric_value} menor al mínimo permitido {min_allowed}.",
                        }
                    )
                if max_allowed is not None and numeric_value > max_allowed:
                    range_findings.append(
                        {
                            "merge_key": variable_name,
                            "source_column": source_column,
                            "row_index": idx,
                            "original_value": value_text,
                            "answer_range": cb.get("ANSWER_RANGE", ""),
                            "issue_type": "above_max_range",
                            "issue_detail": f"Valor {numeric_value} mayor al máximo permitido {max_allowed}.",
                        }
                    )

    findings_df = pd.DataFrame(findings)
    skipped_df = pd.DataFrame(skipped_vars)
    pipe_df = pd.DataFrame(pipe_conflicts)
    range_df = pd.DataFrame(range_findings)

    summary = pd.DataFrame(
        [
            {
                "metric": "matched_variables",
                "value": int(findings_df["merge_key"].nunique()) if not findings_df.empty else 0,
            },
            {"metric": "records_evaluated", "value": int(len(findings_df))},
            {"metric": "valid_records", "value": int(findings_df["is_valid"].sum()) if not findings_df.empty else 0},
            {
                "metric": "corrected_records",
                "value": int((findings_df["correction_applied"] == True).sum()) if not findings_df.empty else 0,
            },
            {
                "metric": "pipe_conflicts",
                "value": int((findings_df["match_method"] == "pipe_conflict").sum()) if not findings_df.empty else 0,
            },
            {
                "metric": "uncorrected_invalid",
                "value": int(((findings_df["is_valid"] == False) & (findings_df["match_method"] != "pipe_conflict")).sum())
                if not findings_df.empty
                else 0,
            },
            {"metric": "skipped_variables", "value": int(len(skipped_df))},
            {"metric": "answer_range_issues", "value": int(len(range_df))},
        ]
    )

    return findings_df, summary, pipe_df, corrected, range_df


def main() -> None:
    config = _parse_args()
    logger = setup_logger("11_codebook_value_audit")

    print_script_overview(
        "11_codebook_value_audit.py",
        "Audits collapsed interval values against codebook DISPLAY/CODEVALUE and writes corrected output.",
    )

    print_step(1, "Load inputs")
    codebook = _load_table(config.codebook_path, sheet_name=config.codebook_sheet)
    collapsed = _load_table(config.collapsed_path)

    print_step(2, "Prepare new codebook key FORM_NAME__QUESTION_NAME")
    codebook_prepared = _prepare_codebook(codebook)

    print_step(3, "Audit values and apply fuzzy corrections when possible")
    findings, summary, pipe_conflicts, collapsed_corrected, answer_range_issues = _audit_and_correct(codebook_prepared, collapsed)

    print_step(4, "Build unmatched-key diagnostics")
    collapsed_keys = pd.DataFrame({"source_column": collapsed.columns.astype(str)})
    collapsed_keys["merge_key"] = collapsed_keys["source_column"].map(_strip_repeat_suffix)
    collapsed_keys = collapsed_keys[["source_column", "merge_key"]].drop_duplicates()
    codebook_keys = codebook_prepared[["merge_key"]].drop_duplicates()

    unmatched_in_collapsed = collapsed_keys.merge(codebook_keys, on="merge_key", how="left", indicator=True)
    unmatched_in_collapsed = unmatched_in_collapsed[unmatched_in_collapsed["_merge"] == "left_only"].drop(columns=["_merge"])

    collapsed_base_keys = collapsed_keys[["merge_key"]].drop_duplicates()
    unmatched_in_codebook = codebook_keys.merge(collapsed_base_keys, on="merge_key", how="left", indicator=True)
    unmatched_in_codebook = unmatched_in_codebook[unmatched_in_codebook["_merge"] == "left_only"].drop(columns=["_merge"])

    print_step(5, "Save audit workbook and corrected collapsed file")
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(config.output_path, engine="openpyxl") as writer:
        findings.to_excel(writer, sheet_name="audit_findings", index=False)
        summary.to_excel(writer, sheet_name="summary", index=False)
        pipe_conflicts.to_excel(writer, sheet_name="pipe_conflicts", index=False)
        answer_range_issues.to_excel(writer, sheet_name="answer_range_issues", index=False)
        unmatched_in_collapsed.to_excel(writer, sheet_name="unmatched_collapsed", index=False)
        unmatched_in_codebook.to_excel(writer, sheet_name="unmatched_codebook", index=False)

    _save_table(collapsed_corrected, config.corrected_output_path)
    corrected_csv_path = _csv_path_for(config.corrected_output_path)
    collapsed_corrected.to_csv(corrected_csv_path, index=False)

    metrics = {
        "rows_codebook": len(codebook),
        "rows_codebook_prepared": len(codebook_prepared),
        "cols_collapsed": len(collapsed.columns),
        "records_evaluated": len(findings),
        "records_corrected": int(findings["correction_applied"].sum()) if not findings.empty else 0,
        "pipe_conflicts": len(pipe_conflicts),
        "answer_range_issues": len(answer_range_issues),
        "audit_output": str(config.output_path),
        "corrected_output": str(config.corrected_output_path),
        "corrected_output_csv": str(corrected_csv_path),
    }
    print_kv("Codebook value audit", metrics)
    logger.info("Saved report: %s", config.output_path)
    logger.info("Saved corrected collapsed file: %s", config.corrected_output_path)
    logger.info("Saved corrected collapsed CSV file: %s", corrected_csv_path)


if __name__ == "__main__":
    main()
