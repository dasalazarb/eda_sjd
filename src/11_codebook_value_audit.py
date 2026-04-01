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


def _is_blank(value: object) -> bool:
    token = _normalize_token(value)
    return token in BLANK_TOKENS


def _split_by_delimiters(value: object) -> list[str]:
    if _is_blank(value):
        return []
    parts = [part.strip() for part in re.split(r"[|;,\n]+", str(value))]
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


def _prepare_codebook(codebook: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"FORM_NAME__QUESTION_NAME", "DISPLAY", "CODEVALUE"}
    missing = required_cols.difference(codebook.columns)
    if missing:
        raise KeyError(f"Missing required codebook columns: {sorted(missing)}")

    selected = codebook[["FORM_NAME__QUESTION_NAME", "DISPLAY", "CODEVALUE"]].copy()
    selected["FORM_NAME__QUESTION_NAME"] = selected["FORM_NAME__QUESTION_NAME"].astype(str).str.strip()
    selected = selected[selected["FORM_NAME__QUESTION_NAME"] != ""]

    # If merge key appears multiple times, combine all DISPLAY/CODEVALUE possibilities.
    agg = (
        selected.groupby("FORM_NAME__QUESTION_NAME", as_index=False)
        .agg(
            {
                "DISPLAY": lambda s: " | ".join([str(v) for v in s if not _is_blank(v)]),
                "CODEVALUE": lambda s: " | ".join([str(v) for v in s if not _is_blank(v)]),
            }
        )
        .rename(columns={"FORM_NAME__QUESTION_NAME": "merge_key"})
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


def _audit_and_correct(codebook_prepared: pd.DataFrame, collapsed: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if collapsed.empty:
        empty = pd.DataFrame()
        return empty, pd.DataFrame([{"metric": "matched_variables", "value": 0}]), empty, collapsed.copy()

    corrected = collapsed.copy()
    findings: list[dict[str, object]] = []
    skipped_vars: list[dict[str, object]] = []
    pipe_conflicts: list[dict[str, object]] = []

    collapsed_cols = set(collapsed.columns.astype(str).tolist())

    for _, cb in codebook_prepared.iterrows():
        variable_name = cb["merge_key"]
        if variable_name not in collapsed_cols:
            continue

        allowed_map = _allowed_value_maps(cb["DISPLAY"], cb["CODEVALUE"])
        if not allowed_map:
            skipped_vars.append(
                {
                    "merge_key": variable_name,
                    "reason": "DISPLAY y CODEVALUE vacíos; variable omitida",
                }
            )
            continue

        for idx, raw_value in collapsed[variable_name].items():
            if _is_blank(raw_value):
                continue

            value_text = str(raw_value).strip()
            if not value_text:
                continue

            if "|" in value_text:
                pipe_conflicts.append(
                    {
                        "merge_key": variable_name,
                        "row_index": idx,
                        "original_value": value_text,
                        "reason": "Contiene '|' y se deja sin corregir",
                    }
                )
                findings.append(
                    {
                        "merge_key": variable_name,
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
                    corrected.at[idx, variable_name] = corrected_value

            findings.append(
                {
                    "merge_key": variable_name,
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

    findings_df = pd.DataFrame(findings)
    skipped_df = pd.DataFrame(skipped_vars)
    pipe_df = pd.DataFrame(pipe_conflicts)

    if findings_df.empty:
        summary = pd.DataFrame([{"metric": "matched_variables", "value": 0}])
        return findings_df, summary, pipe_df, corrected

    summary = pd.DataFrame(
        [
            {"metric": "matched_variables", "value": int(findings_df["merge_key"].nunique())},
            {"metric": "records_evaluated", "value": int(len(findings_df))},
            {"metric": "valid_records", "value": int(findings_df["is_valid"].sum())},
            {
                "metric": "corrected_records",
                "value": int((findings_df["correction_applied"] == True).sum()),
            },
            {
                "metric": "pipe_conflicts",
                "value": int((findings_df["match_method"] == "pipe_conflict").sum()),
            },
            {
                "metric": "uncorrected_invalid",
                "value": int(((findings_df["is_valid"] == False) & (findings_df["match_method"] != "pipe_conflict")).sum()),
            },
            {"metric": "skipped_variables", "value": int(len(skipped_df))},
        ]
    )

    return findings_df, summary, pipe_df, corrected


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
    findings, summary, pipe_conflicts, collapsed_corrected = _audit_and_correct(codebook_prepared, collapsed)

    print_step(4, "Build unmatched-key diagnostics")
    collapsed_keys = pd.DataFrame({"merge_key": collapsed.columns.astype(str)}).drop_duplicates()
    codebook_keys = codebook_prepared[["merge_key"]].drop_duplicates()

    unmatched_in_collapsed = collapsed_keys.merge(codebook_keys, on="merge_key", how="left", indicator=True)
    unmatched_in_collapsed = unmatched_in_collapsed[unmatched_in_collapsed["_merge"] == "left_only"].drop(columns=["_merge"])

    unmatched_in_codebook = codebook_keys.merge(collapsed_keys, on="merge_key", how="left", indicator=True)
    unmatched_in_codebook = unmatched_in_codebook[unmatched_in_codebook["_merge"] == "left_only"].drop(columns=["_merge"])

    print_step(5, "Save audit workbook and corrected collapsed file")
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(config.output_path, engine="openpyxl") as writer:
        findings.to_excel(writer, sheet_name="audit_findings", index=False)
        summary.to_excel(writer, sheet_name="summary", index=False)
        pipe_conflicts.to_excel(writer, sheet_name="pipe_conflicts", index=False)
        unmatched_in_collapsed.to_excel(writer, sheet_name="unmatched_collapsed", index=False)
        unmatched_in_codebook.to_excel(writer, sheet_name="unmatched_codebook", index=False)

    _save_table(collapsed_corrected, config.corrected_output_path)

    metrics = {
        "rows_codebook": len(codebook),
        "rows_codebook_prepared": len(codebook_prepared),
        "cols_collapsed": len(collapsed.columns),
        "records_evaluated": len(findings),
        "records_corrected": int(findings["correction_applied"].sum()) if not findings.empty else 0,
        "pipe_conflicts": len(pipe_conflicts),
        "audit_output": str(config.output_path),
        "corrected_output": str(config.corrected_output_path),
    }
    print_kv("Codebook value audit", metrics)
    logger.info("Saved report: %s", config.output_path)
    logger.info("Saved corrected collapsed file: %s", config.corrected_output_path)


if __name__ == "__main__":
    main()
