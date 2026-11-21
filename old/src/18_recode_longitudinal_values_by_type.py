from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from common import ANALYTIC_DIR, REPORTS_DIR, print_kv, print_script_overview, print_step, setup_logger


@dataclass(frozen=True)
class RecodeConfig:
    input_path: Path
    variable_summary_path: Path
    output_path: Path


def _parse_args() -> RecodeConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Recodifica valores reales del longitudinal colapsado usando variable_type "
            "(longitudinal_variable_summary.csv)."
        )
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=REPORTS_DIR / "longitudinal_plausibility" / "patients_with_11d_and_15d.csv",
        help="Ruta de entrada del dataset longitudinal (parquet/csv/xlsx).",
    )
    parser.add_argument(
        "--variable-summary-path",
        type=Path,
        default=REPORTS_DIR / "longitudinal_plausibility" / "longitudinal_variable_summary.csv",
        help="CSV con el resumen de variables y su columna variable_type.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ANALYTIC_DIR / "visits_long_collapsed_by_interval_codebook_type_recode.parquet",
        help="Ruta de salida del dataset recodificado (parquet/csv/xlsx).",
    )
    args = parser.parse_args()
    return RecodeConfig(
        input_path=args.input_path,
        variable_summary_path=args.variable_summary_path,
        output_path=args.output_path,
    )


def _load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo: {path}")

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls", ".xlsm"}:
        return pd.read_excel(path)
    raise ValueError(f"Formato no soportado: {suffix}")


def _save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    csv_path = path.with_suffix(".csv")

    if suffix == ".parquet":
        df.to_parquet(path, index=False)
        df.to_csv(csv_path, index=False)
    elif suffix == ".csv":
        df.to_csv(path, index=False)
        parquet_path = path.with_suffix(".parquet")
        df.to_parquet(parquet_path, index=False)
    elif suffix in {".xlsx", ".xls", ".xlsm"}:
        df.to_excel(path, index=False)
    else:
        raise ValueError(f"Formato de salida no soportado: {suffix}")

    # Siempre exportar también a CSV para facilitar inspección/intercambio.
    if suffix != ".csv":
        df.to_csv(csv_path, index=False)


def _resolve_special_column(df: pd.DataFrame, preferred_names: list[str], suffixes: list[str]) -> str | None:
    cols = [str(c) for c in df.columns]
    lower_to_original = {c.lower(): c for c in cols}

    for name in preferred_names:
        c = lower_to_original.get(name.lower())
        if c:
            return c

    for col in cols:
        low = col.lower()
        if any(low.endswith(sfx.lower()) for sfx in suffixes):
            return col

    return None


def _build_variable_type_map(summary_df: pd.DataFrame) -> dict[str, str]:
    if "variable_type" not in summary_df.columns:
        raise ValueError("El archivo de resumen no contiene la columna 'variable_type'.")

    variable_col = None
    for candidate in ["column", "variable", "variable_name", "name"]:
        if candidate in summary_df.columns:
            variable_col = candidate
            break

    if variable_col is None:
        raise ValueError(
            "No se encontró columna de nombre de variable en el resumen. "
            "Se esperaba una de: column, variable, variable_name, name."
        )

    mapping: dict[str, str] = {}
    for _, row in summary_df[[variable_col, "variable_type"]].dropna(subset=[variable_col]).iterrows():
        col_name = str(row[variable_col]).strip()
        if not col_name:
            continue
        vtype = "unknown" if pd.isna(row["variable_type"]) else str(row["variable_type"]).strip()
        mapping[col_name] = vtype

    return mapping


def _placeholder_for_variable_type(variable_type: str) -> str:
    t = str(variable_type).strip().lower()
    if not t:
        return "unknown"

    if any(k in t for k in ["string", "text", "object", "categor", "nominal", "ordinal"]):
        return "string"
    if "bool" in t or t in {"si/no", "yes/no", "binary", "binaria", "binario"}:
        return "boolean"
    if any(k in t for k in ["int", "integer", "entero"]):
        return "integer"
    if any(k in t for k in ["float", "double", "decimal", "numeric", "number", "numero", "contin"]):
        return "numeric"
    if any(k in t for k in ["date", "datetime", "timestamp", "time", "fecha"]):
        return "date"
    return t.replace(" ", "_")


def _non_missing_mask(series: pd.Series) -> pd.Series:
    mask = ~series.isna()
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        cleaned = series.astype("string").str.strip()
        mask = mask & cleaned.ne("")
    return mask


def _recode_patient_ids(df: pd.DataFrame, patient_col: str) -> int:
    s = df[patient_col]
    non_missing = _non_missing_mask(s)

    unique_in_order = s[non_missing].drop_duplicates().tolist()
    mapping = {raw: f"pt{i}" for i, raw in enumerate(unique_in_order, start=1)}

    df[patient_col] = s.map(mapping)
    return len(mapping)


def _normalize_interval_text(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _recode_interval_name_value(value: object) -> object:
    if pd.isna(value):
        return pd.NA

    text = _normalize_interval_text(value)
    if not text:
        return pd.NA

    if re.fullmatch(r"v[1-6]", text):
        return text

    fixed_map = {
        "natural history protocol 478 interval": "v1",
        "phase 1: initial full evaluation": "v2",
        "phase 1: second full evaluation": "v3",
        "phase 1: final full (third full) evaluation": "v4",
        "phase 2: 4th full evaluation": "v5",
        "phase 2: 5th full evaluation": "v6",
    }
    if text in fixed_map:
        return fixed_map[text]

    m_15d = re.fullmatch(r"15d optional evaluation\s*(\d+)", text)
    if m_15d:
        return f"v1 opt {m_15d.group(1)}"

    m_opt = re.fullmatch(r"optional evaluation\s*(\d+)", text)
    if m_opt:
        return f"opt {m_opt.group(1)}"

    return text


def _recode_interval_names(df: pd.DataFrame, interval_col: str) -> int:
    before = df[interval_col].copy()
    df[interval_col] = before.apply(_recode_interval_name_value)
    return int((before.astype("string") != df[interval_col].astype("string")).fillna(False).sum())


def _apply_variable_type_recode(df: pd.DataFrame, variable_type_map: dict[str, str], excluded_columns: set[str]) -> int:
    recoded_columns = 0

    for col in df.columns.astype(str):
        if col in excluded_columns:
            continue
        if col not in variable_type_map:
            continue

        placeholder = _placeholder_for_variable_type(variable_type_map[col])
        mask = _non_missing_mask(df[col])
        if not bool(mask.any()):
            continue

        df.loc[mask, col] = placeholder
        recoded_columns += 1

    return recoded_columns


def _is_integer_token(text: str) -> bool:
    return bool(re.fullmatch(r"[+-]?\d+", text))


def _is_numeric_token(text: str) -> bool:
    return bool(re.fullmatch(r"[+-]?\d+(\.\d+)?", text))


def _normalize_boolean_token(text: str) -> str:
    return text.strip().lower().replace("_", "").replace("-", "")


def _is_boolean_token(text: str) -> bool:
    token = _normalize_boolean_token(text)
    valid = {
        "0",
        "1",
        "true",
        "false",
        "t",
        "f",
        "yes",
        "no",
        "y",
        "n",
        "si",
        "s",
    }
    return token in valid


def _infer_unmapped_structure(series: pd.Series) -> str:
    non_missing = series[_non_missing_mask(series)]
    if non_missing.empty:
        return "unknown"

    as_text = non_missing.astype("string").str.strip()
    if bool(as_text.map(_is_boolean_token).all()):
        return "boolean"
    if bool(as_text.map(_is_integer_token).all()):
        return "integer"
    if bool(as_text.map(_is_numeric_token).all()):
        return "numeric"

    parsed_dates = pd.to_datetime(as_text, errors="coerce", utc=False)
    if float(parsed_dates.notna().mean()) >= 0.95:
        return "date"

    return "string"


def _replace_invalid_unmapped_values(series: pd.Series, inferred_structure: str) -> tuple[pd.Series, int]:
    out = series.copy()
    non_missing = _non_missing_mask(series)
    if not bool(non_missing.any()):
        return out, 0

    values = series[non_missing].astype("string").str.strip()
    invalid_mask = pd.Series(False, index=series.index)

    if inferred_structure == "boolean":
        invalid_mask.loc[values.index] = ~values.map(_is_boolean_token)
    elif inferred_structure == "integer":
        invalid_mask.loc[values.index] = ~values.map(_is_integer_token)
    elif inferred_structure == "numeric":
        invalid_mask.loc[values.index] = ~values.map(_is_numeric_token)
    elif inferred_structure == "date":
        parsed_dates = pd.to_datetime(values, errors="coerce", utc=False)
        invalid_mask.loc[values.index] = parsed_dates.isna()
    else:
        # Para texto/unknown no invalidamos por patrón: solo limpieza de blancos.
        invalid_mask.loc[values.index] = False

    replaced = int(invalid_mask.sum())
    if replaced > 0:
        out.loc[invalid_mask] = pd.NA
    return out, replaced


def _clean_unmapped_columns(df: pd.DataFrame, variable_type_map: dict[str, str], excluded_columns: set[str]) -> tuple[int, int]:
    cleaned_columns = 0
    replaced_values = 0

    for col in df.columns.astype(str):
        if col in excluded_columns or col in variable_type_map:
            continue

        inferred = _infer_unmapped_structure(df[col])
        cleaned, replaced = _replace_invalid_unmapped_values(df[col], inferred)
        df[col] = cleaned

        if replaced > 0:
            cleaned_columns += 1
            replaced_values += replaced

    return cleaned_columns, replaced_values


def main() -> None:
    cfg = _parse_args()
    logger = setup_logger("18_recode_longitudinal_values_by_type")

    print_script_overview(
        "18_recode_longitudinal_values_by_type.py",
        "Recodificación de valores reales según variable_type y reglas especiales de IDs/intervalos",
    )

    print_step(1, "Cargando dataset longitudinal y resumen de variables")
    df = _load_table(cfg.input_path)
    summary_df = pd.read_csv(cfg.variable_summary_path)
    variable_type_map = _build_variable_type_map(summary_df)

    print_kv(
        "Input",
        {
            "dataset_path": cfg.input_path,
            "rows": len(df),
            "columns": len(df.columns),
            "variable_summary_path": cfg.variable_summary_path,
            "variables_with_type": len(variable_type_map),
        },
    )

    print_step(2, "Resolviendo columnas especiales (_patient_record_number y _interval_name)")
    patient_col = _resolve_special_column(
        df,
        preferred_names=["_patient_record_number", "ids__patient_record_number", "patient_record_number"],
        suffixes=["__patient_record_number", "_patient_record_number"],
    )
    interval_col = _resolve_special_column(
        df,
        preferred_names=["_interval_name", "ids__interval_name", "interval_name"],
        suffixes=["__interval_name", "_interval_name"],
    )

    if patient_col is None:
        raise KeyError("No se encontró columna para _patient_record_number.")
    if interval_col is None:
        raise KeyError("No se encontró columna para _interval_name.")

    print_kv(
        "Columnas especiales",
        {
            "patient_col": patient_col,
            "interval_col": interval_col,
        },
    )

    print_step(3, "Recodificando _patient_record_number e _interval_name")
    n_patients = _recode_patient_ids(df, patient_col)
    n_interval_changes = _recode_interval_names(df, interval_col)

    print_step(4, "Recodificando el resto de variables según variable_type")
    recoded_columns = _apply_variable_type_recode(df, variable_type_map, excluded_columns={patient_col, interval_col})

    print_step(5, "Evaluando estructura de variables no mapeadas y reemplazando valores no válidos")
    cleaned_unmapped_cols, replaced_unmapped_values = _clean_unmapped_columns(
        df,
        variable_type_map,
        excluded_columns={patient_col, interval_col},
    )

    print_step(6, "Guardando dataset recodificado")
    _save_table(df, cfg.output_path)

    print_kv(
        "Resumen recodificación",
        {
            "patient_ids_created": n_patients,
            "interval_values_changed": n_interval_changes,
            "columns_recoded_by_variable_type": recoded_columns,
            "unmapped_columns_cleaned": cleaned_unmapped_cols,
            "unmapped_invalid_values_replaced": replaced_unmapped_values,
            "output_path": cfg.output_path,
            "output_csv_path": cfg.output_path.with_suffix(".csv"),
        },
    )

    logger.info(
        (
            "Recodificación completada | input=%s | output=%s | n_patients=%d | "
            "interval_changes=%d | cols_by_type=%d | unmapped_cols_cleaned=%d | "
            "unmapped_values_replaced=%d"
        ),
        cfg.input_path,
        cfg.output_path,
        n_patients,
        n_interval_changes,
        recoded_columns,
        cleaned_unmapped_cols,
        replaced_unmapped_values,
    )


if __name__ == "__main__":
    main()
