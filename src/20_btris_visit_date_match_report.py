from __future__ import annotations

"""
20_btris_visit_date_match_report.py
====================================
Resuelve discrepancias de visit_date en pacientes con múltiples fechas separadas
por '|' en la columna ids__visit_date, aplicando las siguientes reglas:

  1. Un solo año predomina   → se elige la fecha más temprana de ese año
  2. Todos los años distintos → se elige la fecha del año más temprano
  3. Todos el mismo año      → se condensa a una única fecha (la más temprana)

Salida: tabla plana con MRN, ids__interval_name, visit_date (resuelta) y
        todas las demás variables clínicas del parquet de origen.
        Se guarda en data_analytic/BTRIS/.
"""

import argparse
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from common import INTERMEDIATE_DIR, REPORTS_DIR, print_kv, print_script_overview, print_step, setup_logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_DIR_DEFAULT = Path("data_analytic/BTRIS")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VisitDateMatchConfig:
    patients_path: Path
    output_dir: Path
    output_filename: str


def _parse_args() -> VisitDateMatchConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Resuelve fechas múltiples (separadas por '|') en ids__visit_date "
            "y genera tabla canónica MRN / ids__interval_name / visit_date."
        )
    )
    parser.add_argument(
        "--patients-path",
        type=Path,
        default=REPORTS_DIR / "longitudinal_plausibility" / "patients_with_11d_and_15d.parquet",
        help="Parquet de pacientes con columna ids__visit_date (puede tener '|').",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR_DEFAULT,
        help="Directorio de salida (se crea si no existe).",
    )
    parser.add_argument(
        "--output-filename",
        type=str,
        default="btris_visit_date_resolved.parquet",
        help="Nombre del archivo de salida (.parquet o .csv).",
    )
    args = parser.parse_args()
    return VisitDateMatchConfig(
        patients_path=args.patients_path,
        output_dir=args.output_dir,
        output_filename=args.output_filename,
    )


# ---------------------------------------------------------------------------
# Visit-date resolution logic
# ---------------------------------------------------------------------------


def _parse_date_str(s: str) -> date | None:
    """Parsea un string de fecha individual a objeto date. Retorna None si falla."""
    ts = pd.to_datetime(s.strip(), errors="coerce", dayfirst=False)
    if pd.isna(ts):
        return None
    return ts.normalize().date()


def _resolve_multi_dates(raw: str) -> date | None:
    """
    Dado un string con 1..N fechas separadas por '|', devuelve UNA fecha canónica:

    - 1 fecha                    → esa fecha
    - Todas mismo año            → condensa: fecha más temprana de ese año
    - Un año predomina (moda)    → fecha más temprana del año predominante
    - Todos años distintos       → fecha más temprana (año más antiguo)

    Retorna None si no hay fechas válidas.
    """
    parts = [p.strip() for p in str(raw).split("|") if p.strip()]
    dates: list[date] = []
    for p in parts:
        d = _parse_date_str(p)
        if d is not None:
            dates.append(d)

    if not dates:
        return None

    # Caso trivial
    if len(dates) == 1:
        return dates[0]

    year_counts = Counter(d.year for d in dates)
    unique_years = sorted(year_counts.keys())

    # Todos el mismo año → condensar
    if len(unique_years) == 1:
        return min(dates)

    # Un año predomina (moda estricta)
    most_common_year, top_count = year_counts.most_common(1)[0]
    second_count = year_counts.most_common(2)[1][1] if len(year_counts) >= 2 else 0

    if top_count > second_count:
        # Hay un año con más fechas que cualquier otro
        candidates = [d for d in dates if d.year == most_common_year]
        return min(candidates)

    # Todos los años distintos O empate → año más temprano
    earliest_year = unique_years[0]
    candidates = [d for d in dates if d.year == earliest_year]
    return min(candidates)


def _resolution_tag(raw: str) -> str:
    """Etiqueta diagnóstica de qué regla se aplicó (útil para auditoría)."""
    parts = [p.strip() for p in str(raw).split("|") if p.strip()]
    dates: list[date] = []
    for p in parts:
        d = _parse_date_str(p)
        if d is not None:
            dates.append(d)

    if not dates:
        return "no_valid_dates"
    if len(dates) == 1:
        return "single_date"

    year_counts = Counter(d.year for d in dates)
    unique_years = list(year_counts.keys())

    if len(unique_years) == 1:
        return "same_year_condensed"

    most_common_year, top_count = year_counts.most_common(1)[0]
    second_count = year_counts.most_common(2)[1][1] if len(year_counts) >= 2 else 0

    if top_count > second_count:
        return f"predominant_year_{most_common_year}"

    return "all_different_years_earliest"


# ---------------------------------------------------------------------------
# Patient loader & resolver
# ---------------------------------------------------------------------------


def _load_patients(path: Path) -> pd.DataFrame:
    """Carga el parquet (o CSV) de pacientes sin modificar columnas."""
    if not path.exists():
        raise FileNotFoundError(f"Archivo de pacientes no encontrado: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def _resolve_visit_dates(df: pd.DataFrame, logger) -> pd.DataFrame:
    """
    Aplica _resolve_multi_dates sobre la columna ids__visit_date.
    Genera columnas:
        visit_date          — fecha canónica resuelta (date)
        visit_date_raw      — valor original para auditoría
        visit_date_n_raw    — número de fechas en el valor original
        visit_date_rule     — etiqueta de la regla aplicada
    """
    required_cols = ["ids__patient_record_number", "ids__interval_name", "ids__visit_date"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Faltan columnas requeridas: {missing}")

    out = df.copy()

    raw_col = out["ids__visit_date"].astype(str)

    out["visit_date_raw"] = raw_col
    out["visit_date_n_raw"] = raw_col.apply(
        lambda x: sum(1 for p in x.split("|") if p.strip())
    )
    out["visit_date"] = raw_col.apply(_resolve_multi_dates)
    out["visit_date_rule"] = raw_col.apply(_resolution_tag)

    # Estadísticas de log
    rule_counts = out["visit_date_rule"].value_counts().to_dict()
    null_count = int(out["visit_date"].isna().sum())
    logger.info("Reglas aplicadas: %s", rule_counts)
    logger.info("Filas sin fecha resuelta: %d", null_count)

    return out


def _normalize_mrn(series: pd.Series) -> pd.Series:
    """Normaliza el MRN: strip, elimina separadores, NaN → pd.NA."""
    return (
        series.astype(str)
        .str.strip()
        .str.replace(r"[-/\\\s]", "", regex=True)
        .replace("nan", pd.NA)
        .replace("", pd.NA)
    )


# ---------------------------------------------------------------------------
# Output builder
# ---------------------------------------------------------------------------


def _build_output(df_resolved: pd.DataFrame) -> pd.DataFrame:
    """
    Construye la tabla final:
        MRN (normalizado), ids__interval_name, visit_date,
        visit_date_rule, visit_date_n_raw,
        + todas las demás columnas del parquet (excepto ids__visit_date original
          y visit_date_raw, que se guardan como auditoría pero al final).

    El orden es: columnas clave → variables clínicas → columnas de auditoría.
    """
    out = df_resolved.copy()

    # MRN normalizado
    out["MRN"] = _normalize_mrn(out["ids__patient_record_number"])

    # Eliminar filas sin fecha resuelta o sin MRN
    before = len(out)
    out = out.dropna(subset=["MRN", "visit_date"]).reset_index(drop=True)
    after = len(out)
    if before != after:
        pass  # logging se hace en main

    # Columnas clave al frente
    key_cols = ["MRN", "ids__interval_name", "visit_date", "visit_date_rule", "visit_date_n_raw"]

    # Columnas de auditoría al final
    audit_cols = ["visit_date_raw", "ids__patient_record_number", "ids__visit_date"]
    audit_cols = [c for c in audit_cols if c in out.columns]

    # Variables clínicas: todo lo demás
    skip = set(key_cols) | set(audit_cols)
    clinical_cols = [c for c in out.columns if c not in skip]

    final_cols = key_cols + clinical_cols + audit_cols
    final_cols = [c for c in final_cols if c in out.columns]

    return out[final_cols]


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def _save(df: pd.DataFrame, output_dir: Path, filename: str, logger) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename

    if filename.endswith(".parquet"):
        df.to_parquet(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)

    logger.info("Guardado: %s  (%d filas, %d columnas)", out_path, len(df), len(df.columns))
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    cfg = _parse_args()
    logger = setup_logger("20_btris_visit_date_match_report")

    print_script_overview(
        "20_btris_visit_date_match_report.py",
        (
            "Resuelve fechas múltiples (separadas por '|') en ids__visit_date usando "
            "reglas de año predominante / años distintos / mismo año. "
            f"Salida → {cfg.output_dir / cfg.output_filename}"
        ),
    )

    # -------------------------------------------------------------------------
    # Step 1 — Cargar pacientes
    # -------------------------------------------------------------------------
    print_step(1, "Cargando archivo de pacientes")

    df_raw = _load_patients(cfg.patients_path)
    print_kv(
        "Pacientes cargados",
        {
            "filas": len(df_raw),
            "columnas": len(df_raw.columns),
            "con_multifecha": int(
                df_raw["ids__visit_date"].astype(str).str.contains(r"\|", regex=True).sum()
            ),
            "fuente": str(cfg.patients_path),
        },
    )

    # -------------------------------------------------------------------------
    # Step 2 — Resolver fechas
    # -------------------------------------------------------------------------
    print_step(2, "Resolviendo visit_dates múltiples")

    df_resolved = _resolve_visit_dates(df_raw, logger)

    rule_summary = df_resolved["visit_date_rule"].value_counts().to_dict()
    print_kv("Reglas aplicadas", rule_summary)

    null_dates = int(df_resolved["visit_date"].isna().sum())
    if null_dates:
        logger.warning("%d fila(s) sin fecha resuelta — serán excluidas.", null_dates)
    print_kv("Fechas no resueltas (excluidas)", null_dates)

    # -------------------------------------------------------------------------
    # Step 3 — Construir tabla final
    # -------------------------------------------------------------------------
    print_step(3, "Construyendo tabla final")

    df_output = _build_output(df_resolved)

    rows_before = len(df_resolved)
    rows_after = len(df_output)
    dropped = rows_before - rows_after

    print_kv(
        "Tabla final",
        {
            "filas": rows_after,
            "columnas": len(df_output.columns),
            "filas_eliminadas_sin_MRN_o_fecha": dropped,
            "columnas_clave": ["MRN", "ids__interval_name", "visit_date"],
        },
    )

    # Muestra rápida de fechas resueltas
    multi_only = df_output[df_output["visit_date_n_raw"] > 1]
    if not multi_only.empty:
        print_kv(
            "Muestra de resolución (primeros 5 casos multi-fecha)",
            multi_only[["MRN", "ids__interval_name", "visit_date", "visit_date_rule", "visit_date_n_raw"]]
            .head(5)
            .to_dict(orient="records"),
        )

    # -------------------------------------------------------------------------
    # Step 4 — Guardar
    # -------------------------------------------------------------------------
    print_step(4, f"Guardando en {cfg.output_dir}")

    out_path = _save(df_output, cfg.output_dir, cfg.output_filename, logger)

    print_kv(
        "Resumen final",
        {
            "filas_guardadas": len(df_output),
            "columnas_guardadas": len(df_output.columns),
            "output_path": str(out_path),
        },
    )

    logger.info("Done.")


if __name__ == "__main__":
    main()