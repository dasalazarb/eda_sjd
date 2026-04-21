from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from common import INTERMEDIATE_DIR, REPORTS_DIR, print_kv, print_script_overview, print_step, setup_logger


@dataclass(frozen=True)
class MatchConfig:
    patients_path: Path
    btris_root: Path
    report_all_matches_path: Path
    report_multi_measurements_path: Path


PREFIX_TO_DATE_COLUMNS: dict[str, list[str]] = {
    "clindocdiscrete": ["Document Date", "Observation Date"],
    "diagnosisandprocedure": ["Observation Date", "Admission Date"],
    "echo": ["Result Date"],
    "ekg": ["Result Date"],
    "lab": ["Collected Date Time", "Reported Date Time"],
    "medication": ["Order Date/Start Date", "Order End Date"],
    "microbiology": ["Collected Date"],
    "pathology": ["Date"],
    "pftlab": ["Performed Date Time"],
    "radiology": ["Exam Date"],
    "vital signs": ["Observation Date"],
}

IGNORED_PREFIXES = {"demographics"}
TARGET_INTERVAL_15D = "Natural History Protocol 478 Interval"


def _parse_args() -> MatchConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Cruza ids__visit_date (sin hora) de patients_with_11d_and_15d.parquet "
            "contra fechas en CSVs BTRIS 11D/15D y reporta coincidencias y múltiples mediciones."
        )
    )
    parser.add_argument(
        "--patients-path",
        type=Path,
        default=REPORTS_DIR / "longitudinal_plausibility" / "patients_with_11d_and_15d.parquet",
        help="Ruta al archivo de pacientes con ids__patient_record_number, ids__interval_name, ids__visit_date.",
    )
    parser.add_argument(
        "--btris-root",
        type=Path,
        default=INTERMEDIATE_DIR / "BTRIS",
        help="Ruta raíz que contiene carpetas 11D y 15D con CSVs BTRIS.",
    )
    parser.add_argument(
        "--report-all-matches-path",
        type=Path,
        default=REPORTS_DIR / "btris_visit_date_all_matches.csv",
        help="Salida CSV con todas las coincidencias de fecha encontradas.",
    )
    parser.add_argument(
        "--report-multi-measurements-path",
        type=Path,
        default=REPORTS_DIR / "btris_visit_date_multi_measurements.csv",
        help="Salida CSV con pacientes/intervalos donde hubo >1 medición para una misma entidad BTRIS.",
    )
    args = parser.parse_args()

    return MatchConfig(
        patients_path=args.patients_path,
        btris_root=args.btris_root,
        report_all_matches_path=args.report_all_matches_path,
        report_multi_measurements_path=args.report_multi_measurements_path,
    )


def _normalize_column_name(name: object) -> str:
    text = str(name) if name is not None else ""
    text = text.replace("\ufeff", "").replace("\u00a0", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text.lower()


def _resolve_column_name(columns: pd.Index, expected_name: str) -> str:
    normalized_expected = _normalize_column_name(expected_name)
    for col in columns:
        if _normalize_column_name(col) == normalized_expected:
            return str(col)
    raise KeyError


def _normalize_patient_id(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return re.sub(r"[-/\\\s]", "", text)


def _split_visit_dates(value: object) -> list[pd.Timestamp]:
    if pd.isna(value):
        return []

    raw = str(value)
    if "|" not in raw:
        return []
    parts = [part.strip() for part in raw.split("|") if part and part.strip()]
    out: list[pd.Timestamp] = []
    for part in parts:
        ts = pd.to_datetime(part, errors="coerce", format="mixed")
        if pd.isna(ts):
            continue
        out.append(ts.normalize())
    return out


def _load_patients(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró patients file: {path}")

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Formato no soportado para patients file: {path.suffix}")

    required = ["ids__patient_record_number", "ids__interval_name", "ids__visit_date"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Faltan columnas requeridas en patients file: {missing}")

    rows: list[dict[str, object]] = []
    for _, row in df.iterrows():
        patient_id = _normalize_patient_id(row["ids__patient_record_number"])
        if not patient_id:
            continue

        visit_dates = _split_visit_dates(row["ids__visit_date"])
        if not visit_dates:
            continue

        for visit_date in visit_dates:
            rows.append(
                {
                    "patient_id": patient_id,
                    "ids__interval_name": row["ids__interval_name"],
                    "ids__visit_date": visit_date,
                    "expected_protocol": (
                        "15D" if str(row["ids__interval_name"]).strip() == TARGET_INTERVAL_15D else "11D"
                    ),
                }
            )

    expanded = pd.DataFrame(rows)
    if expanded.empty:
        return expanded

    return expanded.drop_duplicates().reset_index(drop=True)


def _prefix_from_file_name(file_name: str) -> str:
    stem = Path(file_name).stem.lower().strip()
    stem = re.sub(r"[_\-]+", " ", stem)
    for prefix in PREFIX_TO_DATE_COLUMNS:
        if stem.startswith(prefix):
            return prefix
    return ""


def _resolve_existing_date_columns(df: pd.DataFrame, date_candidates: list[str]) -> list[str]:
    resolved: list[str] = []
    for candidate in date_candidates:
        try:
            resolved.append(_resolve_column_name(df.columns, candidate))
        except KeyError:
            continue
    return resolved


def _build_btris_matches(patients_expanded: pd.DataFrame, btris_root: Path, logger) -> pd.DataFrame:
    if patients_expanded.empty:
        return pd.DataFrame()

    patient_date_map_by_protocol = (
        patients_expanded.groupby(["patient_id", "expected_protocol"])["ids__visit_date"]
        .apply(lambda s: set(s.tolist()))
        .to_dict()
    )

    all_csvs = sorted([p for p in btris_root.rglob("*.csv") if p.is_file()])
    if not all_csvs:
        raise FileNotFoundError(f"No se encontraron CSVs en {btris_root}")

    match_rows: list[dict[str, object]] = []

    for csv_path in all_csvs:
        prefix = _prefix_from_file_name(csv_path.name)
        if not prefix:
            continue
        if prefix in IGNORED_PREFIXES:
            continue

        df = pd.read_csv(csv_path)
        try:
            mrn_col = _resolve_column_name(df.columns, "MRN")
        except KeyError:
            logger.warning("Se omite %s porque no contiene columna MRN", csv_path)
            continue

        date_cols = _resolve_existing_date_columns(df, PREFIX_TO_DATE_COLUMNS[prefix])
        if not date_cols:
            logger.warning("Se omite %s porque no tiene columnas de fecha esperadas", csv_path)
            continue

        protocol = ""
        parts = {part.upper() for part in csv_path.parts}
        if "11D" in parts:
            protocol = "11D"
        elif "15D" in parts:
            protocol = "15D"

        for row_idx, row in df.iterrows():
            patient_id = _normalize_patient_id(row[mrn_col])
            if not patient_id:
                continue
            candidate_dates = patient_date_map_by_protocol.get((patient_id, protocol))
            if not candidate_dates:
                continue

            matched_columns: list[str] = []
            matched_dates: list[pd.Timestamp] = []
            for date_col in date_cols:
                ts = pd.to_datetime(row[date_col], errors="coerce", format="mixed")
                if pd.isna(ts):
                    continue
                date_only = ts.normalize()
                if date_only in candidate_dates:
                    matched_columns.append(date_col)
                    matched_dates.append(date_only)

            if not matched_dates:
                continue

            matched_dates_unique = sorted(set(matched_dates))
            matched_dates_str = "|".join(d.strftime("%Y-%m-%d") for d in matched_dates_unique)

            patient_intervals = patients_expanded[
                (patients_expanded["patient_id"] == patient_id)
                & (patients_expanded["expected_protocol"] == protocol)
                & (patients_expanded["ids__visit_date"].isin(matched_dates_unique))
            ][["ids__interval_name", "ids__visit_date"]].drop_duplicates()

            for _, pi in patient_intervals.iterrows():
                match_rows.append(
                    {
                        "protocol": protocol,
                        "file_name": csv_path.name,
                        "file_prefix": prefix,
                        "source_path": str(csv_path),
                        "row_index": int(row_idx),
                        "patient_id": patient_id,
                        "ids__interval_name": pi["ids__interval_name"],
                        "ids__visit_date": pi["ids__visit_date"].strftime("%Y-%m-%d"),
                        "matched_btris_dates": matched_dates_str,
                        "matched_columns": "|".join(sorted(set(matched_columns))),
                    }
                )

    if not match_rows:
        return pd.DataFrame()

    return pd.DataFrame(match_rows)


def _build_multi_measurement_report(matches_df: pd.DataFrame) -> pd.DataFrame:
    if matches_df.empty:
        return pd.DataFrame()

    grouped = (
        matches_df.groupby(["patient_id", "ids__interval_name", "file_prefix"], dropna=False)
        .agg(
            n_rows=("row_index", "count"),
            n_distinct_source_files=("file_name", "nunique"),
            protocols=("protocol", lambda s: "|".join(sorted({str(v) for v in s if str(v)}))),
            source_files=("file_name", lambda s: "|".join(sorted({str(v) for v in s if str(v)}))),
            matched_visit_dates=("ids__visit_date", lambda s: "|".join(sorted({str(v) for v in s if str(v)}))),
        )
        .reset_index()
    )

    out = grouped[grouped["n_rows"] > 1].copy()
    if out.empty:
        return out

    return out.sort_values(["file_prefix", "n_rows", "patient_id"], ascending=[True, False, True]).reset_index(drop=True)


def _patients_with_multiple_visit_dates(patients_expanded: pd.DataFrame) -> pd.DataFrame:
    if patients_expanded.empty:
        return pd.DataFrame(columns=["patient_id", "ids__interval_name", "n_ids_visit_dates"])

    grouped = (
        patients_expanded.groupby(["patient_id", "ids__interval_name"], dropna=False)
        .agg(n_ids_visit_dates=("ids__visit_date", "nunique"))
        .reset_index()
    )
    return grouped[grouped["n_ids_visit_dates"] > 1].reset_index(drop=True)


def _build_cross_file_multi_measurement_report(
    matches_df: pd.DataFrame, multi_date_patients: pd.DataFrame
) -> pd.DataFrame:
    if matches_df.empty or multi_date_patients.empty:
        return pd.DataFrame()

    filtered = matches_df.merge(
        multi_date_patients[["patient_id", "ids__interval_name", "n_ids_visit_dates"]],
        on=["patient_id", "ids__interval_name"],
        how="inner",
    )
    if filtered.empty:
        return pd.DataFrame()

    grouped = (
        filtered.groupby(["patient_id", "ids__interval_name", "file_prefix"], dropna=False)
        .agg(
            n_rows=("row_index", "count"),
            n_distinct_source_files=("file_name", "nunique"),
            protocols=("protocol", lambda s: "|".join(sorted({str(v) for v in s if str(v)}))),
            source_files=("file_name", lambda s: "|".join(sorted({str(v) for v in s if str(v)}))),
            matched_visit_dates=("ids__visit_date", lambda s: "|".join(sorted({str(v) for v in s if str(v)}))),
            n_matched_visit_dates=("ids__visit_date", "nunique"),
            n_ids_visit_dates=("n_ids_visit_dates", "max"),
        )
        .reset_index()
    )

    out = grouped[grouped["n_distinct_source_files"] > 1].copy()
    if out.empty:
        return out

    return out.sort_values(
        ["file_prefix", "n_distinct_source_files", "n_rows", "patient_id"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)


def main() -> None:
    cfg = _parse_args()
    logger = setup_logger("20_btris_visit_date_match_report")

    print_script_overview(
        "20_btris_visit_date_match_report.py",
        "Busca ids__visit_date (sin hora) en archivos BTRIS 11D/15D y reporta múltiples mediciones por intervalo.",
    )

    print_step(1, "Cargando y expandiendo ids__visit_date de patients_with_11d_and_15d")
    patients_expanded = _load_patients(cfg.patients_path)
    print_kv(
        "Pacientes",
        {
            "patients_path": str(cfg.patients_path),
            "rows_expanded_patient_date": int(len(patients_expanded)),
            "unique_patient_ids": int(patients_expanded["patient_id"].nunique()) if not patients_expanded.empty else 0,
        },
    )

    print_step(2, "Buscando coincidencias exactas de fecha en BTRIS (ignorando hora)")
    matches_df = _build_btris_matches(patients_expanded, cfg.btris_root, logger)

    print_step(3, "Filtrando pacientes con múltiples ids__visit_date por intervalo")
    multi_date_patients = _patients_with_multiple_visit_dates(patients_expanded)

    print_step(
        4,
        (
            "Construyendo reporte de múltiples mediciones en diferentes archivos para pacientes "
            "con múltiples ids__visit_date"
        ),
    )
    multi_df = _build_cross_file_multi_measurement_report(matches_df, multi_date_patients)

    cfg.report_all_matches_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.report_multi_measurements_path.parent.mkdir(parents=True, exist_ok=True)

    if matches_df.empty:
        pd.DataFrame(
            columns=[
                "protocol",
                "file_name",
                "file_prefix",
                "source_path",
                "row_index",
                "patient_id",
                "ids__interval_name",
                "ids__visit_date",
                "matched_btris_dates",
                "matched_columns",
            ]
        ).to_csv(cfg.report_all_matches_path, index=False)
    else:
        matches_df.sort_values(["file_prefix", "patient_id", "ids__interval_name"]).to_csv(
            cfg.report_all_matches_path,
            index=False,
        )

    if multi_df.empty:
        pd.DataFrame(
            columns=[
                "patient_id",
                "ids__interval_name",
                "file_prefix",
                "n_rows",
                "n_distinct_source_files",
                "protocols",
                "source_files",
                "matched_visit_dates",
                "n_matched_visit_dates",
                "n_ids_visit_dates",
            ]
        ).to_csv(cfg.report_multi_measurements_path, index=False)
    else:
        multi_df.to_csv(cfg.report_multi_measurements_path, index=False)

    summary = {
        "n_matches": int(len(matches_df)),
        "n_patient_interval_with_multiple_ids_visit_dates": int(len(multi_date_patients)),
        "n_multi_measurement_rows": int(len(multi_df)),
        "report_all_matches_path": str(cfg.report_all_matches_path),
        "report_multi_measurements_path": str(cfg.report_multi_measurements_path),
    }
    print_kv("Resumen", summary)
    logger.info("Done. %s", summary)


if __name__ == "__main__":
    main()
