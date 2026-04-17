from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from common import INTERMEDIATE_DIR, RAW_DIR, REPORTS_DIR, print_kv, print_script_overview, print_step, setup_logger


@dataclass(frozen=True)
class FilterConfig:
    input_dirs: list[Path]
    patients_path: Path
    unique_ordersets_path: Path
    output_root: Path
    report_path: Path


def _parse_args() -> FilterConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Filtra CSVs de BTRIS (11D/15D) usando IDs de patients_with_11d_and_15d.parquet "
            "(ids_patient_record_number) contra columna MRN, con regla adicional para archivos Lab*."
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
        default=REPORTS_DIR / "longitudinal_plausibility" / "patients_with_11d_and_15d.parquet",
        help="Ruta al parquet/csv con la columna ids_patient_record_number.",
    )
    parser.add_argument(
        "--unique-ordersets-path",
        type=Path,
        default=RAW_DIR / "unique_OrderSets.csv",
        help="Ruta al archivo unique_OrderSets.xlsx (columna 'Order Name').",
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
        help="Ruta del reporte resumen de pacientes/rows identificados por archivo.",
    )
    args = parser.parse_args()

    return FilterConfig(
        input_dirs=args.input_dirs,
        patients_path=args.patients_path,
        unique_ordersets_path=args.unique_ordersets_path,
        output_root=args.output_root,
        report_path=args.report_path,
    )


def _resolve_patients_path(preferred_path: Path) -> Path:
    if preferred_path.exists():
        return preferred_path
    raise FileNotFoundError(f"No existe el archivo de pacientes: {preferred_path}")


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
    text = re.sub(r"[-/\\\s]", "", text)
    return text


def _normalize_column_name(name: object) -> str:
    text = str(name) if name is not None else ""
    # Limpieza de BOM y espacios frecuentes invisibles en encabezados de Excel/CSV.
    text = text.replace("\ufeff", "").replace("\u00a0", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text.lower()


def _resolve_column_name(columns: pd.Index, expected_name: str) -> str:
    normalized_expected = _normalize_column_name(expected_name)
    for col in columns:
        if _normalize_column_name(col) == normalized_expected:
            return str(col)
    raise KeyError


def _build_patient_id_set(df: pd.DataFrame) -> set[str]:
    required_col = "ids__patient_record_number"
    try:
        source_col = _resolve_column_name(df.columns, required_col)
    except KeyError:
        raise KeyError(
            "No se encontró la columna requerida 'ids__patient_record_number' en el archivo de pacientes."
        )

    patient_ids = df[source_col].map(_normalize_id)
    patient_ids = patient_ids[patient_ids != ""]
    return set(patient_ids.tolist())


def _load_allowed_order_names(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(f"No existe unique_OrderSets.xlsx: {path}")

    orders_df = pd.read_excel(path)
    try:
        order_col = _resolve_column_name(orders_df.columns, "Order Name")
    except KeyError:
        raise KeyError("El archivo unique_OrderSets.xlsx no tiene la columna 'Order Name'.")

    return {
        str(v).strip().lower()
        for v in orders_df[order_col].dropna().tolist()
        if str(v).strip()
    }


def _is_lab_file(file_path: Path) -> bool:
    return file_path.name.lower().startswith("lab")


def _filter_single_csv(
    file_path: Path,
    patient_ids: set[str],
    allowed_orders: set[str],
) -> tuple[pd.DataFrame, dict[str, object]]:
    df = pd.read_csv(file_path)

    try:
        mrn_col = _resolve_column_name(df.columns, "MRN")
    except KeyError:
        raise KeyError(f"El archivo {file_path} no contiene la columna MRN.")

    working = df.copy()
    working["_mrn_normalized"] = working[mrn_col].map(_normalize_id)

    patient_mask = working["_mrn_normalized"].isin(patient_ids)
    filtered = working.loc[patient_mask].copy()

    lab_order_filter_applied = False
    if _is_lab_file(file_path):
        lab_order_filter_applied = True
        try:
            order_col = _resolve_column_name(filtered.columns, "Order Name")
        except KeyError:
            raise KeyError(f"El archivo Lab {file_path} no contiene la columna 'Order Name'.")

        order_names_normalized = filtered[order_col].astype("string").str.strip().str.lower()
        filtered = filtered.loc[order_names_normalized.isin(allowed_orders)].copy()

    patient_count = int(filtered["_mrn_normalized"].nunique(dropna=True))
    row_count = int(len(filtered))

    filtered = filtered.drop(columns=["_mrn_normalized"])

    metrics = {
        "file_name": file_path.name,
        "source_path": str(file_path),
        "is_lab_file": lab_order_filter_applied,
        "patients_identified": patient_count,
        "rows_output": row_count,
    }
    return filtered, metrics


def _iter_csv_files(input_dirs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for input_dir in input_dirs:
        if not input_dir.exists():
            continue
        files.extend(sorted(input_dir.rglob("*.csv")))
    return files


def _output_path_for(source_file: Path, input_dirs: list[Path], output_root: Path) -> Path:
    for base in input_dirs:
        try:
            rel = source_file.relative_to(base)
            return output_root / base.name / rel
        except ValueError:
            continue
    return output_root / source_file.name


def main() -> None:
    cfg = _parse_args()
    logger = setup_logger("19_filter_btris_patients")

    print_script_overview(
        "19_filter_btris_patients.py",
        "Filtra CSVs BTRIS por IDs de pacientes y aplica filtro adicional en archivos Lab*."
    )

    print_step(1, "Cargando IDs de pacientes y catálogo de Order Sets")
    patients_path = _resolve_patients_path(cfg.patients_path)
    patients_df = _load_patients_table(patients_path)
    patient_ids = _build_patient_id_set(patients_df)
    allowed_orders = _load_allowed_order_names(cfg.unique_ordersets_path)

    print_kv(
        "Insumos",
        {
            "patients_path": patients_path,
            "n_patient_ids_normalized": len(patient_ids),
            "unique_ordersets_path": cfg.unique_ordersets_path,
            "n_allowed_order_names": len(allowed_orders),
        },
    )

    print_step(2, "Recorriendo y filtrando CSVs de BTRIS")
    csv_files = _iter_csv_files(cfg.input_dirs)
    if not csv_files:
        raise FileNotFoundError(
            "No se encontraron archivos CSV en los directorios de entrada indicados."
        )

    metrics_rows: list[dict[str, object]] = []

    for file_path in csv_files:
        filtered_df, metrics = _filter_single_csv(
            file_path=file_path,
            patient_ids=patient_ids,
            allowed_orders=allowed_orders,
        )

        output_path = _output_path_for(file_path, cfg.input_dirs, cfg.output_root)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        filtered_df.to_csv(output_path, index=False)

        metrics["output_path"] = str(output_path)
        metrics_rows.append(metrics)

        logger.info(
            "Processed %s | patients_identified=%d | rows_output=%d | output=%s",
            file_path,
            metrics["patients_identified"],
            metrics["rows_output"],
            output_path,
        )

    print_step(3, "Guardando reporte consolidado por archivo")
    report_df = pd.DataFrame(metrics_rows).sort_values(["is_lab_file", "file_name"]).reset_index(drop=True)

    cfg.report_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(cfg.report_path, index=False)

    summary = {
        "n_csv_processed": int(len(report_df)),
        "total_patients_identified_sum": int(report_df["patients_identified"].sum()),
        "total_rows_output_sum": int(report_df["rows_output"].sum()),
        "report_path": str(cfg.report_path),
        "output_root": str(cfg.output_root),
    }

    print_kv("Resumen final", summary)
    logger.info("Done. %s", summary)


if __name__ == "__main__":
    main()
