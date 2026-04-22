from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from common import INTERMEDIATE_DIR, REPORTS_DIR, print_kv, print_script_overview, print_step, setup_logger


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_INTERVAL_15D = "Natural History Protocol 478 Interval"

LAB_DATE_CANDIDATES = ["Collected Date Time", "Reported Date Time"]
LAB_TEST_COLUMN = "Observation Name"

MICRO_DATE_CANDIDATES = ["Collected Date"]
MICRO_TEST_COLUMN = "Event Name"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RepeatedTestsConfig:
    patients_path: Path
    btris_root: Path
    report_path: Path


def _parse_args() -> RepeatedTestsConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Detecta tests repetidos (mismo Order Name / Event Name) en 2+ visit_dates "
            "del mismo interval_name para archivos Lab* y Microbiology* de BTRIS."
        )
    )
    parser.add_argument(
        "--patients-path",
        type=Path,
        default=REPORTS_DIR / "longitudinal_plausibility" / "patients_with_11d_and_15d.parquet",
    )
    parser.add_argument(
        "--btris-root",
        type=Path,
        default=INTERMEDIATE_DIR / "BTRIS",
        help="Raíz con subcarpetas 11D/ y 15D/ conteniendo CSVs filtrados.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=REPORTS_DIR / "btris_repeated_tests.csv",
        help="Salida CSV con tests repetidos dentro del mismo intervalo.",
    )
    args = parser.parse_args()
    return RepeatedTestsConfig(
        patients_path=args.patients_path,
        btris_root=args.btris_root,
        report_path=args.report_path,
    )


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def _normalize_column_name(name: object) -> str:
    text = str(name) if name is not None else ""
    text = text.replace("\ufeff", "").replace("\u00a0", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text.lower()


def _resolve_column(columns: pd.Index, expected: str) -> str:
    norm = _normalize_column_name(expected)
    for col in columns:
        if _normalize_column_name(col) == norm:
            return str(col)
    raise KeyError(f"Columna no encontrada: '{expected}'")


def _normalize_patient_id(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.replace(r"[-/\\\s]", "", regex=True)
        .replace("nan", "")
        .replace("", pd.NA)
    )


def _detect_protocol(csv_path: Path) -> str:
    """Detecta el protocolo (11D / 15D) buscando el componente en el path."""
    for part in csv_path.parts:
        if part.upper() == "11D":
            return "11D"
        if part.upper() == "15D":
            return "15D"
    # fallback: buscar en el string completo
    path_str = str(csv_path)
    if re.search(r"[/\\]11D[/\\]", path_str, re.IGNORECASE):
        return "11D"
    if re.search(r"[/\\]15D[/\\]", path_str, re.IGNORECASE):
        return "15D"
    return ""


def _resolve_first_date_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Devuelve la primera columna de fecha candidata que exista en el DataFrame."""
    for candidate in candidates:
        try:
            return _resolve_column(df.columns, candidate)
        except KeyError:
            continue
    return None


def _parse_dates_to_date_only(series: pd.Series) -> pd.Series:
    """Parsea una columna de fechas (con o sin hora) a solo-fecha (date, no datetime)."""
    parsed = pd.to_datetime(series, errors="coerce", dayfirst=False)
    return parsed.dt.normalize().dt.date


# ---------------------------------------------------------------------------
# Patient loader
# ---------------------------------------------------------------------------


def _load_patients_expanded(path: Path) -> pd.DataFrame:
    """
    Carga el parquet de pacientes y expande ids__visit_date (separadas por '|').
    Solo incluye pacientes con 2+ visit_dates por interval_name (objetivo del análisis).

    Columnas de salida: patient_id, ids__interval_name, expected_protocol, visit_date (date)
    """
    if not path.exists():
        raise FileNotFoundError(f"No se encontró archivo de pacientes: {path}")

    df = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)

    required = ["ids__patient_record_number", "ids__interval_name", "ids__visit_date"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Faltan columnas en patients file: {missing}")

    rows: list[dict] = []
    for _, row in df.iterrows():
        raw_id = str(row["ids__patient_record_number"]).strip()
        patient_id = re.sub(r"[-/\\\s]", "", raw_id)
        if not patient_id or patient_id == "nan":
            continue

        raw_dates = str(row["ids__visit_date"])
        if "|" not in raw_dates:
            # Solo pacientes con 2+ visit_dates
            continue

        parts = [p.strip() for p in raw_dates.split("|") if p.strip()]
        parsed_dates: list = []
        for p in parts:
            ts = pd.to_datetime(p, errors="coerce", dayfirst=False)
            if pd.isna(ts):
                continue
            parsed_dates.append(ts.normalize().date())

        if len(parsed_dates) < 2:
            continue

        interval_name = str(row["ids__interval_name"]).strip()
        protocol = "15D" if interval_name == TARGET_INTERVAL_15D else "11D"

        for vd in parsed_dates:
            rows.append(
                {
                    "patient_id": patient_id,
                    "ids__interval_name": interval_name,
                    "expected_protocol": protocol,
                    "visit_date": vd,
                }
            )

    expanded = pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)
    return expanded


# ---------------------------------------------------------------------------
# Core matching — vectorized
# ---------------------------------------------------------------------------


def _match_csv_to_visits(
    csv_path: Path,
    patients_expanded: pd.DataFrame,
    date_candidates: list[str],
    test_column: str,
    logger,
) -> pd.DataFrame | None:
    """
    Carga un CSV BTRIS y cruza vectorizadamente contra patients_expanded.
    Retorna DataFrame con columnas:
        patient_id, ids__interval_name, visit_date, test_name, file_name, protocol
    o None si el archivo debe omitirse.
    """
    protocol = _detect_protocol(csv_path)
    if not protocol:
        logger.warning("No se pudo detectar protocolo para %s — se omite", csv_path.name)
        return None

    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception as exc:
        logger.warning("Error leyendo %s: %s — se omite", csv_path.name, exc)
        return None

    # -- Resolver columna MRN --
    try:
        mrn_col = _resolve_column(df.columns, "MRN")
    except KeyError:
        logger.warning("%s no tiene columna MRN — se omite", csv_path.name)
        return None

    # -- Resolver columna de test --
    try:
        test_col = _resolve_column(df.columns, test_column)
    except KeyError:
        logger.warning("%s no tiene columna '%s' — se omite", csv_path.name, test_column)
        return None

    # -- Resolver columna de fecha --
    date_col = _resolve_first_date_column(df, date_candidates)
    if date_col is None:
        logger.warning("%s no tiene columnas de fecha esperadas %s — se omite", csv_path.name, date_candidates)
        return None

    # -- Normalizar MRN y fecha --
    working = df[[mrn_col, date_col, test_col]].copy()
    working["patient_id"] = _normalize_patient_id(df[mrn_col])
    working["btris_date"] = _parse_dates_to_date_only(df[date_col])
    working["test_name"] = df[test_col].astype(str).str.strip()

    working = working.dropna(subset=["patient_id", "btris_date"])
    working = working[working["test_name"].str.len() > 0]

    if working.empty:
        return None

    # -- Filtrar solo pacientes del protocolo correspondiente --
    patients_proto = patients_expanded[
        patients_expanded["expected_protocol"] == protocol
    ][["patient_id", "ids__interval_name", "visit_date"]].copy()

    if patients_proto.empty:
        return None

    # -- Merge vectorizado: (patient_id, btris_date) == (patient_id, visit_date) --
    merged = working.merge(
        patients_proto,
        left_on=["patient_id", "btris_date"],
        right_on=["patient_id", "visit_date"],
        how="inner",
    )

    if merged.empty:
        return None

    result = merged[["patient_id", "ids__interval_name", "visit_date", "test_name"]].copy()
    result["file_name"] = csv_path.name
    result["protocol"] = protocol
    return result.drop_duplicates().reset_index(drop=True)


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def _find_csv_files_by_prefix(btris_root: Path, prefix: str) -> list[Path]:
    return sorted(
        p for p in btris_root.rglob("*.csv")
        if p.is_file() and p.name.lower().startswith(prefix.lower())
    )


# ---------------------------------------------------------------------------
# Repeated test detection
# ---------------------------------------------------------------------------


def _find_repeated_tests(
    all_matches: pd.DataFrame,
    entity_label: str,
) -> pd.DataFrame:
    """
    Dado un DataFrame de matches con columnas
    (patient_id, ids__interval_name, visit_date, test_name, file_name, protocol),
    detecta casos donde el mismo test_name aparece en 2+ visit_dates distintas
    dentro del mismo (patient_id, ids__interval_name).

    Retorna solo los casos repetidos con detalle de fechas y archivos.
    """
    if all_matches.empty:
        return pd.DataFrame()

    grouped = (
        all_matches.groupby(["patient_id", "ids__interval_name", "test_name"], dropna=False)
        .agg(
            n_visit_dates=("visit_date", "nunique"),
            visit_dates=("visit_date", lambda s: "|".join(sorted({str(v) for v in s}))),
            protocols=("protocol", lambda s: "|".join(sorted({str(v) for v in s if str(v)}))),
            source_files=("file_name", lambda s: "|".join(sorted({str(v) for v in s if str(v)}))),
        )
        .reset_index()
    )

    repeated = grouped[grouped["n_visit_dates"] > 1].copy()
    if repeated.empty:
        return repeated

    repeated.insert(0, "entity", entity_label)
    return repeated.sort_values(
        ["patient_id", "ids__interval_name", "n_visit_dates", "test_name"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    cfg = _parse_args()
    logger = setup_logger("21_btris_repeated_tests_report")

    print_script_overview(
        "21_btris_repeated_tests_report.py",
        (
            "Detecta Order Name (Lab) y Event Name (Microbiology) repetidos "
            "en 2+ visit_dates del mismo interval_name."
        ),
    )

    # Step 1 — Cargar pacientes
    print_step(1, "Cargando pacientes con 2+ visit_dates por intervalo")
    patients_expanded = _load_patients_expanded(cfg.patients_path)

    if patients_expanded.empty:
        raise ValueError(
            "No se encontraron pacientes con 2+ visit_dates. "
            "Verificar columna ids__visit_date en el parquet."
        )

    print_kv(
        "Pacientes expandidos",
        {
            "unique_patients": int(patients_expanded["patient_id"].nunique()),
            "unique_intervals": int(patients_expanded["ids__interval_name"].nunique()),
            "total_patient_date_rows": int(len(patients_expanded)),
        },
    )

    # Step 2 — Procesar Lab
    print_step(2, "Procesando archivos Lab* — columna clave: Order Name")
    lab_files = _find_csv_files_by_prefix(cfg.btris_root, "lab")
    logger.info("Archivos Lab encontrados: %d", len(lab_files))

    lab_frames: list[pd.DataFrame] = []
    for csv_path in lab_files:
        result = _match_csv_to_visits(
            csv_path=csv_path,
            patients_expanded=patients_expanded,
            date_candidates=LAB_DATE_CANDIDATES,
            test_column=LAB_TEST_COLUMN,
            logger=logger,
        )
        if result is not None and not result.empty:
            lab_frames.append(result)

    lab_matches = pd.concat(lab_frames, ignore_index=True) if lab_frames else pd.DataFrame()
    lab_repeated = _find_repeated_tests(lab_matches, entity_label="Lab")

    print_kv(
        "Lab",
        {
            "archivos_procesados": len(lab_files),
            "archivos_con_matches": len(lab_frames),
            "matches_totales": int(len(lab_matches)),
            "tests_repetidos": int(len(lab_repeated)),
        },
    )

    # Step 3 — Procesar Microbiology
    print_step(3, "Procesando archivos Microbiology* — columna clave: Event Name")
    micro_files = _find_csv_files_by_prefix(cfg.btris_root, "microbiology")
    logger.info("Archivos Microbiology encontrados: %d", len(micro_files))

    micro_frames: list[pd.DataFrame] = []
    for csv_path in micro_files:
        result = _match_csv_to_visits(
            csv_path=csv_path,
            patients_expanded=patients_expanded,
            date_candidates=MICRO_DATE_CANDIDATES,
            test_column=MICRO_TEST_COLUMN,
            logger=logger,
        )
        if result is not None and not result.empty:
            micro_frames.append(result)

    micro_matches = pd.concat(micro_frames, ignore_index=True) if micro_frames else pd.DataFrame()
    micro_repeated = _find_repeated_tests(micro_matches, entity_label="Microbiology")

    print_kv(
        "Microbiology",
        {
            "archivos_procesados": len(micro_files),
            "archivos_con_matches": len(micro_frames),
            "matches_totales": int(len(micro_matches)),
            "tests_repetidos": int(len(micro_repeated)),
        },
    )

    # Step 4 — Consolidar y guardar
    print_step(4, "Consolidando reporte final")

    all_repeated = pd.concat(
        [df for df in [lab_repeated, micro_repeated] if not df.empty],
        ignore_index=True,
    )

    cfg.report_path.parent.mkdir(parents=True, exist_ok=True)

    output_columns = [
        "entity",
        "patient_id",
        "ids__interval_name",
        "test_name",
        "n_visit_dates",
        "visit_dates",
        "protocols",
        "source_files",
    ]

    if all_repeated.empty:
        pd.DataFrame(columns=output_columns).to_csv(cfg.report_path, index=False)
        logger.info("No se encontraron tests repetidos.")
    else:
        all_repeated[output_columns].to_csv(cfg.report_path, index=False)

    print_kv(
        "Resumen final",
        {
            "lab_repeated_tests": int(len(lab_repeated)),
            "micro_repeated_tests": int(len(micro_repeated)),
            "total_repeated_tests": int(len(all_repeated)),
            "report_path": str(cfg.report_path),
        },
    )
    logger.info("Done.")


if __name__ == "__main__":
    main()