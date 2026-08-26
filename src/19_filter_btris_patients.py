"""Filter raw BTRIS extracts to definitive SjD cohort patients and report coverage."""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from common import (
    ANALYTIC_DIR,
    INTERMEDIATE_DIR,
    RAW_DIR,
    REPORTS_DIR,
    print_kv,
    print_script_overview,
    print_step,
    setup_logger,
)

MRN_COLUMN = "ids__patient_record_number"


@dataclass(frozen=True)
class FilterConfig:
    """Paths used by the BTRIS patient-filtering step."""

    input_dirs: list[Path]
    patients_path: Path
    unique_ordersets_path: Path
    output_root: Path
    report_path: Path
    coverage_qc_path: Path
    coverage_detail_path: Path


@dataclass(frozen=True)
class PatientCohort:
    """Unique normalized spine MRNs and one display value for each MRN."""

    patient_ids: set[str]
    display_ids: dict[str, object]


def _parse_args() -> FilterConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Filtra CSVs de BTRIS (11D/15D) usando los MRN del cohort spine "
            "definitivo de SjD, con una regla adicional para archivos Lab*."
        )
    )
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        type=Path,
        default=[RAW_DIR / "BTRIS" / "11D", RAW_DIR / "BTRIS" / "15D"],
        help="Directorios de entrada con CSVs a filtrar.",
    )
    parser.add_argument(
        "--patients-path",
        type=Path,
        default=ANALYTIC_DIR / "clinical_episode_spine_sjd.parquet",
        help=f"Ruta al spine SjD con la columna {MRN_COLUMN}.",
    )
    parser.add_argument(
        "--unique-ordersets-path",
        type=Path,
        default=RAW_DIR / "unique_OrderSets.csv",
        help="Ruta a unique_OrderSets (.csv/.xlsx) con la columna 'Order Name'.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=INTERMEDIATE_DIR / "BTRIS",
        help="Directorio raíz de salida para los CSV filtrados.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=REPORTS_DIR / "btris_patient_filter_report.csv",
        help="Ruta del reporte por archivo.",
    )
    parser.add_argument(
        "--coverage-qc-path",
        type=Path,
        default=REPORTS_DIR / "btris_patient_coverage_qc.csv",
        help="Ruta del QC resumen de cobertura de pacientes.",
    )
    parser.add_argument(
        "--coverage-detail-path",
        type=Path,
        default=REPORTS_DIR / "btris_patient_coverage_detail.csv",
        help="Ruta del QC de cobertura a nivel de paciente.",
    )
    args = parser.parse_args()

    return FilterConfig(
        input_dirs=args.input_dirs,
        patients_path=args.patients_path,
        unique_ordersets_path=args.unique_ordersets_path,
        output_root=args.output_root,
        report_path=args.report_path,
        coverage_qc_path=args.coverage_qc_path,
        coverage_detail_path=args.coverage_detail_path,
    )


def _resolve_patients_path(preferred_path: Path) -> Path:
    if preferred_path.exists():
        return preferred_path
    raise FileNotFoundError(
        f"No existe el cohort spine de pacientes SjD: {preferred_path}"
    )


def _load_patients_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Formato no soportado para patients-path: {suffix}")


def _normalize_id(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    # Se eliminan separadores frecuentes en los MRN.
    return re.sub(r"[-/\\\s]", "", text)


def _normalize_patient_record_number(value: object) -> set[str]:
    """Return matching variants, retaining the established MRN normalization."""
    normalized = _normalize_id(value)
    if not normalized:
        return set()

    normalized_variants = {normalized}
    if normalized.startswith("0"):
        normalized_variants.add(normalized.lstrip("0") or "0")
    return normalized_variants


def _canonical_patient_record_number(value: object) -> str:
    """Normalize an MRN to one key so leading-zero variants count once."""
    normalized = _normalize_id(value)
    if not normalized:
        return ""
    return normalized.lstrip("0") or "0"


def _normalize_column_name(name: object) -> str:
    text = str(name) if name is not None else ""
    text = text.replace("\ufeff", "").replace("\u00a0", " ").strip()
    return re.sub(r"\s+", " ", text).lower()


def _resolve_column_name(columns: pd.Index, expected_name: str) -> str:
    normalized_expected = _normalize_column_name(expected_name)
    for col in columns:
        if _normalize_column_name(col) == normalized_expected:
            return str(col)
    raise KeyError


def _build_patient_cohort(df: pd.DataFrame) -> PatientCohort:
    """Build the unique patient-level cohort from clinical episode rows."""
    try:
        source_col = _resolve_column_name(df.columns, MRN_COLUMN)
    except KeyError as exc:
        raise KeyError(
            f"El cohort spine no contiene la columna MRN requerida '{MRN_COLUMN}'."
        ) from exc

    if df.empty:
        raise ValueError("El cohort spine SjD está vacío; n_spine_patients == 0.")

    display_ids: dict[str, object] = {}
    for value in df[source_col].tolist():
        normalized = _canonical_patient_record_number(value)
        if normalized and normalized not in display_ids:
            display_ids[normalized] = value

    if not display_ids:
        raise ValueError(
            "Todos los MRN del cohort spine están vacíos después de la normalización."
        )
    return PatientCohort(patient_ids=set(display_ids), display_ids=display_ids)


def _build_patient_id_set(df: pd.DataFrame) -> set[str]:
    """Return unique canonical MRNs from the spine (compatibility helper)."""
    return _build_patient_cohort(df).patient_ids


def _load_allowed_order_names(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(f"No existe unique_OrderSets: {path}")
    if path.suffix.lower() == ".csv":
        orders_df = pd.read_csv(path)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        orders_df = pd.read_excel(path)
    else:
        raise ValueError(f"Formato no soportado para unique_OrderSets: {path.suffix}")

    try:
        order_col = _resolve_column_name(orders_df.columns, "Order Name")
    except KeyError as exc:
        raise KeyError("unique_OrderSets no tiene la columna 'Order Name'.") from exc
    return {
        str(value).strip().lower()
        for value in orders_df[order_col].dropna().tolist()
        if str(value).strip()
    }


def _is_lab_file(file_path: Path) -> bool:
    return file_path.name.lower().startswith("lab")


def _filter_single_csv(
    file_path: Path,
    patient_ids: set[str],
    allowed_orders: set[str],
) -> tuple[pd.DataFrame, dict[str, object], Counter[str]]:
    """Filter one BTRIS CSV and return file metrics and retained rows per MRN."""
    df = pd.read_csv(file_path)
    try:
        mrn_col = _resolve_column_name(df.columns, "MRN")
    except KeyError as exc:
        raise KeyError(f"El archivo {file_path} no contiene la columna MRN.") from exc

    working = df.copy()
    working["_mrn_normalized"] = working[mrn_col].map(_canonical_patient_record_number)
    filtered = working.loc[working["_mrn_normalized"].isin(patient_ids)].copy()

    if _is_lab_file(file_path):
        try:
            order_col = _resolve_column_name(filtered.columns, "Order Name")
        except KeyError as exc:
            raise KeyError(
                f"El archivo Lab {file_path} no contiene la columna 'Order Name'."
            ) from exc
        normalized_orders = filtered[order_col].astype("string").str.strip().str.lower()
        filtered = filtered.loc[normalized_orders.isin(allowed_orders)].copy()

    row_counts = Counter(filtered["_mrn_normalized"].tolist())
    metrics: dict[str, object] = {
        "file_name": file_path.name,
        "source_path": str(file_path),
        "is_lab_file": _is_lab_file(file_path),
        "patients_identified": len(row_counts),
        "rows_output": len(filtered),
    }
    return filtered.drop(columns=["_mrn_normalized"]), metrics, row_counts


def _iter_csv_files(input_dirs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for input_dir in input_dirs:
        if input_dir.exists():
            files.extend(sorted(input_dir.rglob("*.csv")))
    return files


def _protocol_for(source_file: Path, input_dirs: list[Path]) -> str:
    for base in input_dirs:
        try:
            source_file.relative_to(base)
            return base.name.upper()
        except ValueError:
            continue
    raise ValueError(f"No se pudo determinar el protocolo para {source_file}.")


def _output_path_for(
    source_file: Path, input_dirs: list[Path], output_root: Path
) -> Path:
    for base in input_dirs:
        try:
            return output_root / base.name / source_file.relative_to(base)
        except ValueError:
            continue
    return output_root / source_file.name


def _build_coverage_detail(
    cohort: PatientCohort,
    row_counts: Counter[str],
    patient_files: dict[str, set[str]],
    protocol_patients: dict[str, set[str]],
) -> pd.DataFrame:
    """Create one coverage row per unique spine patient."""
    rows = []
    for patient_id in sorted(cohort.patient_ids):
        n_rows = int(row_counts[patient_id])
        rows.append(
            {
                "patient_record_number": cohort.display_ids[patient_id],
                "patient_record_number_normalized": patient_id,
                "found_in_btris": n_rows > 0,
                "n_btris_rows": n_rows,
                "n_btris_files": len(patient_files.get(patient_id, set())),
                "found_in_11d": patient_id in protocol_patients.get("11D", set()),
                "found_in_15d": patient_id in protocol_patients.get("15D", set()),
            }
        )
    return pd.DataFrame(rows)


def _build_coverage_summary(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Create and validate the unique-patient BTRIS coverage summary."""
    n_spine = int(len(detail_df))
    n_found = int(detail_df["found_in_btris"].sum())
    n_not_found = int((~detail_df["found_in_btris"]).sum())
    if n_spine != n_found + n_not_found:
        raise RuntimeError(
            "QC inconsistente: n_spine_patients != encontrados + no encontrados."
        )
    return pd.DataFrame(
        [
            {
                "n_spine_patients": n_spine,
                "n_btris_patients_found": n_found,
                "n_btris_patients_not_found": n_not_found,
                "pct_btris_patient_coverage": n_found / n_spine * 100,
            }
        ]
    )


def main() -> None:
    cfg = _parse_args()
    logger = setup_logger("19_filter_btris_patients")
    print_script_overview(
        "19_filter_btris_patients.py",
        "Filtra BTRIS a pacientes del spine SjD y genera QC de cobertura.",
    )

    print_step(1, "Cargando spine SjD y catálogo de Order Sets")
    patients_path = _resolve_patients_path(cfg.patients_path)
    cohort = _build_patient_cohort(_load_patients_table(patients_path))
    allowed_orders = _load_allowed_order_names(cfg.unique_ordersets_path)
    print_kv(
        "Insumos",
        {
            "patients_path": patients_path,
            "n_spine_patients": len(cohort.patient_ids),
            "unique_ordersets_path": cfg.unique_ordersets_path,
            "n_allowed_order_names": len(allowed_orders),
        },
    )

    print_step(2, "Recorriendo y filtrando CSVs de BTRIS")
    csv_files = _iter_csv_files(cfg.input_dirs)
    if not csv_files:
        raise FileNotFoundError(
            "No se encontraron CSV en los directorios BTRIS indicados."
        )

    metrics_rows: list[dict[str, object]] = []
    all_found_patient_ids: set[str] = set()
    all_row_counts: Counter[str] = Counter()
    patient_files: dict[str, set[str]] = defaultdict(set)
    protocol_patients: dict[str, set[str]] = defaultdict(set)

    for file_path in csv_files:
        protocol = _protocol_for(file_path, cfg.input_dirs)
        filtered_df, metrics, file_row_counts = _filter_single_csv(
            file_path, cohort.patient_ids, allowed_orders
        )
        output_path = _output_path_for(file_path, cfg.input_dirs, cfg.output_root)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        filtered_df.to_csv(output_path, index=False)

        for patient_id, count in file_row_counts.items():
            all_found_patient_ids.add(patient_id)
            all_row_counts[patient_id] += count
            patient_files[patient_id].add(str(file_path))
            protocol_patients[protocol].add(patient_id)
        metrics.update({"protocol": protocol, "output_path": str(output_path)})
        metrics_rows.append(metrics)
        logger.info(
            "Processed %s | protocol=%s | patients=%d | rows=%d | output=%s",
            file_path,
            protocol,
            metrics["patients_identified"],
            metrics["rows_output"],
            output_path,
        )

    print_step(3, "Guardando reportes por archivo y QC de cobertura")
    report_columns = [
        "file_name",
        "source_path",
        "protocol",
        "is_lab_file",
        "patients_identified",
        "rows_output",
        "output_path",
    ]
    report_df = pd.DataFrame(metrics_rows)[report_columns].sort_values(
        ["protocol", "is_lab_file", "file_name"]
    )
    detail_df = _build_coverage_detail(
        cohort, all_row_counts, patient_files, protocol_patients
    )
    coverage_df = _build_coverage_summary(detail_df)
    if int(coverage_df.loc[0, "n_btris_patients_found"]) != len(all_found_patient_ids):
        raise RuntimeError(
            "QC inconsistente: la cobertura no coincide con la unión de MRN BTRIS."
        )

    for path in (cfg.report_path, cfg.coverage_qc_path, cfg.coverage_detail_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(cfg.report_path, index=False)
    coverage_df.to_csv(cfg.coverage_qc_path, index=False)
    detail_df.to_csv(cfg.coverage_detail_path, index=False)

    coverage = coverage_df.iloc[0]
    summary = {
        "Spine patients": int(coverage["n_spine_patients"]),
        "Patients found in BTRIS": int(coverage["n_btris_patients_found"]),
        "Patients not found in BTRIS": int(coverage["n_btris_patients_not_found"]),
        "Patient coverage": f"{coverage['pct_btris_patient_coverage']:.1f} %",
        "Filtered BTRIS rows": int(report_df["rows_output"].sum()),
        "BTRIS files processed": len(report_df),
    }
    print_kv("BTRIS patient filtering summary", summary)
    logger.info("Done. %s", summary)


if __name__ == "__main__":
    main()
