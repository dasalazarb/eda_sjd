from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from common import ANALYTIC_DIR, REPORTS_DIR, print_kv, print_script_overview, print_step, setup_logger


NH_INTERVAL = "Natural History Protocol 478 Interval"
OPT15D_PREFIX = "15D Optional Evaluation"
KEY_PATIENT = "ids__subject_number"
KEY_VISIT_DATE = "ids__visit_date"
KEY_INTERVAL = "ids__interval_name"
KEY_RECORD = "ids__patient_record_number"


@dataclass(frozen=True)
class MergeConfig:
    input_path: Path
    output_condensed_path: Path
    output_original_copy_path: Path
    output_report_path: Path


def _parse_args() -> MergeConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Condensa por paciente-año los intervalos 'Natural History Protocol 478 Interval' "
            "y '15D Optional Evaluation' cuando comparten año de ids__visit_date, "
            "y genera un reporte de cruce por variable."
        )
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=ANALYTIC_DIR / "visits_long_collapsed_by_interval_codebook_corrected.parquet",
        help="Ruta del dataset de entrada (parquet/csv/xlsx).",
    )
    parser.add_argument(
        "--output-condensed-path",
        type=Path,
        default=ANALYTIC_DIR / "visits_long_nh_15d_same_year_condensed.parquet",
        help="Ruta del output final con filas condensadas (parquet/csv/xlsx).",
    )
    parser.add_argument(
        "--output-original-copy-path",
        type=Path,
        default=ANALYTIC_DIR / "visits_long_collapsed_by_interval_codebook_corrected_original_copy.parquet",
        help="Ruta para guardar copia del DF original (parquet/csv/xlsx).",
    )
    parser.add_argument(
        "--output-report-path",
        type=Path,
        default=REPORTS_DIR / "nh_15d_same_year_cross_report.csv",
        help="Ruta del reporte de cruces por variable (csv/parquet/xlsx).",
    )
    args = parser.parse_args()

    return MergeConfig(
        input_path=args.input_path,
        output_condensed_path=args.output_condensed_path,
        output_original_copy_path=args.output_original_copy_path,
        output_report_path=args.output_report_path,
    )


def _load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo de entrada: {path}")

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls", ".xlsm"}:
        return pd.read_excel(path)
    raise ValueError(f"Formato no soportado para lectura: {suffix}")


def _save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        df.to_parquet(path, index=False)
        df.to_csv(path.with_suffix(".csv"), index=False)
        return
    if suffix == ".csv":
        df.to_csv(path, index=False)
        return
    if suffix in {".xlsx", ".xls", ".xlsm"}:
        df.to_excel(path, index=False)
        return

    raise ValueError(f"Formato no soportado para salida: {suffix}")


def _normalize_string(v: object) -> str:
    if pd.isna(v):
        return ""
    return str(v).strip()


def _normalize_date_string(v: object) -> str:
    if pd.isna(v):
        return ""
    ts = pd.to_datetime(v, errors="coerce")
    if pd.isna(ts):
        return _normalize_string(v)
    return ts.strftime("%Y-%m-%d")


def _build_year_column(df: pd.DataFrame) -> pd.Series:
    parsed = pd.to_datetime(df[KEY_VISIT_DATE], errors="coerce")
    return parsed.dt.year.astype("Int64")


def _is_target_interval(interval_name: object) -> bool:
    txt = _normalize_string(interval_name).lower()
    return txt == NH_INTERVAL.lower() or txt.startswith(OPT15D_PREFIX.lower())


def _merge_values(left: object, right: object) -> object:
    left_txt = _normalize_string(left)
    right_txt = _normalize_string(right)

    if left_txt and right_txt:
        if left_txt == right_txt:
            return left_txt
        return f"{left_txt}|{right_txt}"
    if left_txt:
        return left_txt
    if right_txt:
        return right_txt
    return pd.NA


def _build_overlap_rows(nh_row: pd.Series, opt_row: pd.Series, comparable_columns: list[str]) -> list[dict[str, object]]:
    report_rows: list[dict[str, object]] = []

    for col in comparable_columns:
        nh_val = nh_row[col]
        opt_val = opt_row[col]

        nh_txt = _normalize_string(nh_val)
        opt_txt = _normalize_string(opt_val)

        if not nh_txt or not opt_txt:
            continue

        if nh_txt != opt_txt:
            report_rows.append(
                {
                    KEY_PATIENT: _normalize_string(nh_row[KEY_PATIENT]) or _normalize_string(opt_row[KEY_PATIENT]),
                    "year": int(nh_row["visit_year"]),
                    KEY_RECORD: f"{_normalize_string(nh_row[KEY_RECORD])}|{_normalize_string(opt_row[KEY_RECORD])}",
                    "variable": col,
                    "values": f"{nh_txt}|{opt_txt}",
                    KEY_VISIT_DATE: f"{_normalize_date_string(nh_row[KEY_VISIT_DATE])}|{_normalize_date_string(opt_row[KEY_VISIT_DATE])}",
                    KEY_INTERVAL: f"{_normalize_string(nh_row[KEY_INTERVAL])}|{_normalize_string(opt_row[KEY_INTERVAL])}",
                }
            )

    return report_rows


def _condense_pair(nh_row: pd.Series, opt_row: pd.Series, all_columns: list[str]) -> dict[str, object]:
    out: dict[str, object] = {}

    for col in all_columns:
        if col == "visit_year":
            continue

        if col == KEY_INTERVAL:
            out[col] = f"{_normalize_string(nh_row[col])}|{_normalize_string(opt_row[col])}"
            continue

        if col == KEY_RECORD:
            out[col] = f"{_normalize_string(nh_row[col])}|{_normalize_string(opt_row[col])}"
            continue

        if col == KEY_VISIT_DATE:
            out[col] = f"{_normalize_date_string(nh_row[col])}|{_normalize_date_string(opt_row[col])}"
            continue

        out[col] = _merge_values(nh_row[col], opt_row[col])

    out["merge_rule"] = "nh_478_plus_15d_optional_same_year"
    return out


def build_condensed_and_report(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {KEY_PATIENT, KEY_VISIT_DATE, KEY_INTERVAL, KEY_RECORD}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    work = df.copy()
    work["visit_year"] = _build_year_column(work)
    work = work[work["visit_year"].notna()].copy()
    work = work[work[KEY_INTERVAL].apply(_is_target_interval)].copy()

    all_columns = work.columns.tolist()
    comparable_columns = [c for c in all_columns if c not in {KEY_PATIENT, KEY_VISIT_DATE, KEY_INTERVAL, KEY_RECORD, "visit_year"}]

    condensed_rows: list[dict[str, object]] = []
    report_rows: list[dict[str, object]] = []

    grouped = work.groupby([KEY_PATIENT, "visit_year"], dropna=False)

    for (_, _), g in grouped:
        nh_rows = g[g[KEY_INTERVAL].astype("string").str.strip().str.lower() == NH_INTERVAL.lower()]
        opt_rows = g[g[KEY_INTERVAL].astype("string").str.strip().str.lower().str.startswith(OPT15D_PREFIX.lower())]

        if nh_rows.empty or opt_rows.empty:
            continue

        for _, nh_row in nh_rows.iterrows():
            for _, opt_row in opt_rows.iterrows():
                condensed_rows.append(_condense_pair(nh_row, opt_row, all_columns))
                report_rows.extend(_build_overlap_rows(nh_row, opt_row, comparable_columns))

    condensed_df = pd.DataFrame(condensed_rows)
    report_df = pd.DataFrame(report_rows)

    return condensed_df, report_df


def main() -> None:
    cfg = _parse_args()
    logger = setup_logger("20_merge_optional_nh_same_year")

    print_script_overview(
        "Condensa NH 478 + 15D Optional por paciente-año y reporta cruces por variable",
        [
            "Carga dataset corregido colapsado.",
            "Filtra intervalos objetivo (NH 478 y 15D Optional Evaluation).",
            "Cruza filas por paciente + mismo año de ids__visit_date.",
            "Genera reporte de variables con valores distintos en ambos lados.",
            "Guarda: reporte, filas condensadas y copia del DF original.",
        ],
    )

    print_step(1, "Cargando datos")
    df_original = _load_table(cfg.input_path)
    print_kv("input_path", cfg.input_path)
    print_kv("rows_original", len(df_original))
    print_kv("cols_original", len(df_original.columns))

    print_step(2, "Construyendo filas condensadas y reporte de cruces")
    condensed_df, report_df = build_condensed_and_report(df_original)
    print_kv("rows_condensed", len(condensed_df))
    print_kv("rows_report", len(report_df))

    print_step(3, "Guardando outputs")
    _save_table(df_original, cfg.output_original_copy_path)
    _save_table(condensed_df, cfg.output_condensed_path)
    _save_table(report_df, cfg.output_report_path)
    print_kv("output_original_copy_path", cfg.output_original_copy_path)
    print_kv("output_condensed_path", cfg.output_condensed_path)
    print_kv("output_report_path", cfg.output_report_path)

    logger.info(
        "Done merge NH+15D same-year | original=%d condensed=%d report=%d",
        len(df_original),
        len(condensed_df),
        len(report_df),
    )


if __name__ == "__main__":
    main()
