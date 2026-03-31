from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from common import ANALYTIC_DIR, REPORTS_DIR, print_kv, print_script_overview, print_step, setup_logger

SEPARATOR_PATTERN = re.compile(r"\s*\|\s*")
BLANK_TOKENS = {"", "nan", "none", "null", "na", "n/a"}


@dataclass(frozen=True)
class AuditConfig:
    codebook_path: Path
    codebook_sheet: str
    collapsed_path: Path
    output_path: Path


def _parse_args() -> AuditConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Audits values from visits_long_collapsed_by_interval against final_display_options "
            "and final_answer_range from the harmonized codebook."
        )
    )
    parser.add_argument(
        "--codebook-path",
        type=Path,
        default=Path("data_raw") / "codebook_final_harmonized_once_quince.xlsx",
        help="Path to codebook workbook (default: data_raw/codebook_final_harmonized_once_quince.xlsx).",
    )
    parser.add_argument(
        "--codebook-sheet",
        type=str,
        default="final_codebook",
        help="Codebook sheet name (default: final_codebook).",
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
    args = parser.parse_args()
    return AuditConfig(
        codebook_path=args.codebook_path,
        codebook_sheet=args.codebook_sheet,
        collapsed_path=args.collapsed_path,
        output_path=args.output_path,
    )


def _normalize_token(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"\s+", "_", text)
    return text


def _is_blank(value: object) -> bool:
    token = _normalize_token(value)
    return token in BLANK_TOKENS


def _split_variants(value: object) -> list[str]:
    if _is_blank(value):
        return []
    raw = str(value)
    parts = [part.strip() for part in SEPARATOR_PATTERN.split(raw)]
    return [part for part in parts if _normalize_token(part) not in BLANK_TOKENS]


def _canonical_key(question_name: str, form_name: str) -> str:
    q = _normalize_token(question_name)
    f = _normalize_token(form_name)
    return f"{f}__{q}"


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


def _expand_codebook(codebook: pd.DataFrame) -> pd.DataFrame:
    required_cols = {
        "QUESTION_NAME",
        "final_form_names",
        "final_display_options",
        "final_answer_range",
    }
    missing = required_cols.difference(codebook.columns)
    if missing:
        raise KeyError(f"Missing required codebook columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    for _, row in codebook.iterrows():
        form_variants = _split_variants(row["final_form_names"])
        if not form_variants:
            form_variants = [""]

        for form_name in form_variants:
            out = row.to_dict()
            out["final_form_name_variant"] = form_name
            out["merge_key"] = _canonical_key(row["QUESTION_NAME"], form_name)
            rows.append(out)

    expanded = pd.DataFrame(rows)
    return expanded.drop_duplicates(subset=["merge_key", "QUESTION_NAME", "final_form_name_variant"])


def _parse_option_tokens(raw_options: object) -> set[str]:
    if _is_blank(raw_options):
        return set()

    text = str(raw_options)
    chunks = re.split(r"[|;,\n]+", text)
    tokens: set[str] = set()

    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue

        normalized_chunk = _normalize_token(chunk)
        if normalized_chunk:
            tokens.add(normalized_chunk)

        if ":" in chunk:
            left, right = chunk.split(":", 1)
            left_n = _normalize_token(left)
            right_n = _normalize_token(right)
            if left_n:
                tokens.add(left_n)
            if right_n:
                tokens.add(right_n)
        elif "=" in chunk:
            left, right = chunk.split("=", 1)
            left_n = _normalize_token(left)
            right_n = _normalize_token(right)
            if left_n:
                tokens.add(left_n)
            if right_n:
                tokens.add(right_n)

    return tokens


def _coerce_float(value: object) -> float | None:
    if _is_blank(value):
        return None

    text = str(value).strip().replace(",", "")
    if text.endswith("%"):
        text = text[:-1].strip()

    try:
        return float(text)
    except ValueError:
        return None


def _matches_range(value: object, range_text: object) -> bool:
    if _is_blank(range_text):
        return False

    numeric_value = _coerce_float(value)
    if numeric_value is None:
        return False

    expr = str(range_text).strip()
    expr_no_space = re.sub(r"\s+", "", expr)

    comparator = re.fullmatch(r"(<=|>=|<|>)(-?\d+(?:\.\d+)?)", expr_no_space)
    if comparator:
        op, threshold_raw = comparator.groups()
        threshold = float(threshold_raw)
        if op == "<":
            return numeric_value < threshold
        if op == "<=":
            return numeric_value <= threshold
        if op == ">":
            return numeric_value > threshold
        return numeric_value >= threshold

    between = re.fullmatch(r"(-?\d+(?:\.\d+)?)\s*(?:-|to|a)\s*(-?\d+(?:\.\d+)?)", expr, flags=re.IGNORECASE)
    if between:
        low, high = between.groups()
        low_v, high_v = float(low), float(high)
        lo, hi = min(low_v, high_v), max(low_v, high_v)
        return lo <= numeric_value <= hi

    bracket = re.fullmatch(r"([\[(])\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*([\])])", expr)
    if bracket:
        left_br, low, high, right_br = bracket.groups()
        low_v, high_v = float(low), float(high)
        lower_ok = numeric_value >= low_v if left_br == "[" else numeric_value > low_v
        upper_ok = numeric_value <= high_v if right_br == "]" else numeric_value < high_v
        return lower_ok and upper_ok

    equals_number = re.fullmatch(r"-?\d+(?:\.\d+)?", expr_no_space)
    if equals_number:
        return numeric_value == float(expr_no_space)

    return False


def _audit_matches(codebook_expanded: pd.DataFrame, collapsed: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if collapsed.empty:
        return pd.DataFrame(), pd.DataFrame()

    collapsed_cols = pd.DataFrame({"merge_key": collapsed.columns.astype(str)})
    codebook_matchable = codebook_expanded.merge(collapsed_cols, on="merge_key", how="inner")

    findings: list[dict[str, object]] = []

    for _, cb in codebook_matchable.iterrows():
        variable_name = cb["merge_key"]
        observed_series = collapsed[variable_name]

        observed_values: list[str] = []
        for value in observed_series.dropna().tolist():
            observed_values.extend(_split_variants(value) or [str(value).strip()])

        normalized_observed = sorted({v for v in observed_values if _normalize_token(v) not in BLANK_TOKENS})
        if not normalized_observed:
            normalized_observed = ["(sin_valor)"]

        option_tokens = _parse_option_tokens(cb["final_display_options"])
        range_text = cb["final_answer_range"]
        option_is_blank = _is_blank(cb["final_display_options"])
        range_is_blank = _is_blank(range_text)

        for value in normalized_observed:
            value_norm = _normalize_token(value)
            option_match = value_norm in option_tokens if option_tokens else False
            range_match = _matches_range(value, range_text)

            if option_is_blank and range_is_blank:
                status = "both_empty"
                detail = "final_display_options y final_answer_range vacíos"
            elif (not option_is_blank) and (not range_is_blank):
                status = "both_present"
                if option_match and range_match:
                    detail = "coincide_con_ambas"
                elif option_match:
                    detail = "coincide_con_display_options"
                elif range_match:
                    detail = "coincide_con_answer_range"
                else:
                    detail = "sin_coincidencia"
            elif not option_is_blank:
                status = "display_only"
                detail = "coincide_con_display_options" if option_match else "sin_coincidencia"
            else:
                status = "range_only"
                detail = "coincide_con_answer_range" if range_match else "sin_coincidencia"

            findings.append(
                {
                    "merge_key": variable_name,
                    "QUESTION_NAME": cb["QUESTION_NAME"],
                    "final_form_name_variant": cb["final_form_name_variant"],
                    "observed_value": value,
                    "final_display_options": cb["final_display_options"],
                    "final_answer_range": cb["final_answer_range"],
                    "option_match": option_match,
                    "range_match": range_match,
                    "status": status,
                    "evaluation": detail,
                }
            )

    findings_df = pd.DataFrame(findings)

    if findings_df.empty:
        summary = pd.DataFrame(
            [
                {
                    "metric": "matched_variables",
                    "value": 0,
                }
            ]
        )
        return findings_df, summary

    summary_rows = [
        {"metric": "matched_variables", "value": int(findings_df["merge_key"].nunique())},
        {"metric": "records_evaluated", "value": int(len(findings_df))},
        {"metric": "both_empty_records", "value": int((findings_df["status"] == "both_empty").sum())},
        {
            "metric": "both_present_records",
            "value": int((findings_df["status"] == "both_present").sum()),
        },
        {
            "metric": "display_only_records",
            "value": int((findings_df["status"] == "display_only").sum()),
        },
        {
            "metric": "range_only_records",
            "value": int((findings_df["status"] == "range_only").sum()),
        },
        {
            "metric": "option_matches",
            "value": int(findings_df["option_match"].sum()),
        },
        {
            "metric": "range_matches",
            "value": int(findings_df["range_match"].sum()),
        },
        {
            "metric": "no_matches",
            "value": int((findings_df["evaluation"] == "sin_coincidencia").sum()),
        },
    ]

    summary = pd.DataFrame(summary_rows)
    return findings_df, summary


def main() -> None:
    config = _parse_args()
    logger = setup_logger("11_codebook_value_audit")

    print_script_overview(
        "11_codebook_value_audit.py",
        "Audits collapsed interval values against codebook display options and answer ranges.",
    )

    print_step(1, "Load inputs")
    codebook = _load_table(config.codebook_path, sheet_name=config.codebook_sheet)
    collapsed = _load_table(config.collapsed_path)

    print_step(2, "Expand codebook variants for QUESTION_NAME + final_form_names")
    codebook_expanded = _expand_codebook(codebook)

    print_step(3, "Audit values against final_display_options and final_answer_range")
    findings, summary = _audit_matches(codebook_expanded, collapsed)

    print_step(4, "Build unmatched-key diagnostics")
    collapsed_keys = pd.DataFrame({"merge_key": collapsed.columns.astype(str)}).drop_duplicates()
    codebook_keys = codebook_expanded[["merge_key", "QUESTION_NAME", "final_form_name_variant"]].drop_duplicates()

    unmatched_in_collapsed = collapsed_keys.merge(codebook_keys, on="merge_key", how="left", indicator=True)
    unmatched_in_collapsed = unmatched_in_collapsed[unmatched_in_collapsed["_merge"] == "left_only"].drop(
        columns=["_merge", "QUESTION_NAME", "final_form_name_variant"]
    )

    unmatched_in_codebook = codebook_keys.merge(collapsed_keys, on="merge_key", how="left", indicator=True)
    unmatched_in_codebook = unmatched_in_codebook[unmatched_in_codebook["_merge"] == "left_only"].drop(columns=["_merge"])

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(config.output_path, engine="openpyxl") as writer:
        findings.to_excel(writer, sheet_name="audit_findings", index=False)
        summary.to_excel(writer, sheet_name="summary", index=False)
        unmatched_in_collapsed.to_excel(writer, sheet_name="unmatched_collapsed", index=False)
        unmatched_in_codebook.to_excel(writer, sheet_name="unmatched_codebook", index=False)

    metrics = {
        "rows_codebook": len(codebook),
        "rows_codebook_expanded": len(codebook_expanded),
        "cols_collapsed": len(collapsed.columns),
        "records_evaluated": len(findings),
        "output": str(config.output_path),
    }
    print_kv("Codebook value audit", metrics)
    logger.info("Saved report: %s", config.output_path)


if __name__ == "__main__":
    main()
