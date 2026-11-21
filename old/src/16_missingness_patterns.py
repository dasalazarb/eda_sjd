from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from common import ANALYTIC_DIR, REPORTS_DIR, print_kv, print_script_overview, print_step, setup_logger

TEXT_MISSING_TOKENS = {"", " ", "na", "n/a", "nan", "none", "null"}


@dataclass(frozen=True)
class MissingnessConfig:
    input_path: Path
    output_dir: Path
    top_n: int


def _parse_args() -> MissingnessConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Evalúa missingness por variable y categoría en el dataset "
            "visits_long_collapsed_by_interval_codebook_corrected."
        )
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=ANALYTIC_DIR / "visits_long_collapsed_by_interval_codebook_corrected.parquet",
        help="Ruta al dataset (.parquet/.csv/.xlsx).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPORTS_DIR / "missingness",
        help="Carpeta de salida para reportes de missingness.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Número de variables top/bottom a exportar por missingness.",
    )
    args = parser.parse_args()
    return MissingnessConfig(input_path=args.input_path, output_dir=args.output_dir, top_n=max(1, args.top_n))


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


def _split_category_and_variable(column_name: str) -> tuple[str, str]:
    text = str(column_name)
    if "__" in text:
        category, variable = text.split("__", 1)
        return category.strip() or "sin_categoria", variable.strip() or text
    if "_" in text:
        category, variable = text.split("_", 1)
        return category.strip() or "sin_categoria", variable.strip() or text
    return "sin_categoria", text


def _missing_mask(series: pd.Series) -> pd.Series:
    base_mask = series.isna()
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        cleaned = series.astype("string").str.strip().str.lower()
        return base_mask | cleaned.isin(TEXT_MISSING_TOKENS)
    return base_mask


def _build_variable_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_rows = len(df)
    rows: list[dict[str, object]] = []
    mask_by_col: dict[str, pd.Series] = {}

    for column in df.columns.astype(str):
        mask = _missing_mask(df[column])
        mask_by_col[column] = mask
        category, variable = _split_category_and_variable(column)
        missing_count = int(mask.sum())
        rows.append(
            {
                "column": column,
                "category": category,
                "variable": variable,
                "n_rows": n_rows,
                "missing_count": missing_count,
                "non_missing_count": int(n_rows - missing_count),
                "missing_pct": round((missing_count / n_rows) * 100, 4) if n_rows else 0.0,
                "dtype": str(df[column].dtype),
            }
        )

    return pd.DataFrame(rows).sort_values(["missing_pct", "column"], ascending=[False, True]), pd.DataFrame(mask_by_col)


def _build_category_summary(variable_summary: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        variable_summary.groupby("category", as_index=False)
        .agg(
            n_variables=("column", "count"),
            total_missing_cells=("missing_count", "sum"),
            total_cells=("n_rows", lambda s: int((s.iloc[0] if len(s) else 0) * len(s))),
            mean_missing_pct=("missing_pct", "mean"),
            median_missing_pct=("missing_pct", "median"),
            max_missing_pct=("missing_pct", "max"),
            min_missing_pct=("missing_pct", "min"),
        )
        .copy()
    )

    grouped["category_missing_pct"] = (
        (grouped["total_missing_cells"] / grouped["total_cells"].replace(0, pd.NA)) * 100
    ).fillna(0.0)
    grouped["category_missing_pct"] = grouped["category_missing_pct"].round(4)
    grouped["mean_missing_pct"] = grouped["mean_missing_pct"].round(4)
    grouped["median_missing_pct"] = grouped["median_missing_pct"].round(4)
    grouped["max_missing_pct"] = grouped["max_missing_pct"].round(4)
    grouped["min_missing_pct"] = grouped["min_missing_pct"].round(4)

    return grouped.sort_values(["category_missing_pct", "category"], ascending=[False, True])


def _build_patterns(mask_df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if mask_df.empty:
        return pd.DataFrame(columns=["pattern", "rows", "pct_rows", "missing_columns", "n_missing_columns"])

    pattern_key = mask_df.apply(lambda row: "|".join(mask_df.columns[row.to_numpy()]), axis=1)
    pattern_key = pattern_key.replace("", "<sin_missing>")

    counts = pattern_key.value_counts(dropna=False).rename_axis("pattern").reset_index(name="rows")
    counts["pct_rows"] = (counts["rows"] / len(mask_df) * 100).round(4)
    counts["missing_columns"] = counts["pattern"].replace("<sin_missing>", "")
    counts["n_missing_columns"] = counts["missing_columns"].apply(
        lambda x: 0 if not x else len([v for v in str(x).split("|") if v])
    )
    return counts.head(top_n)


def _build_patterns_by_category(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=["category", "pattern", "rows", "pct_rows_category", "missing_variables", "n_missing_variables"]
        )

    rows: list[dict[str, object]] = []
    n_total = len(df)

    category_to_cols: dict[str, list[str]] = {}
    for col in df.columns.astype(str):
        category, _ = _split_category_and_variable(col)
        category_to_cols.setdefault(category, []).append(col)

    for category, cols in category_to_cols.items():
        sub = df[cols].copy()
        missing_sub = pd.DataFrame({col: _missing_mask(sub[col]) for col in cols})
        pattern = missing_sub.apply(lambda row: "|".join([c for c in cols if bool(row[c])]), axis=1)
        pattern = pattern.replace("", "<sin_missing>")

        counts = pattern.value_counts().head(top_n)
        for p, ct in counts.items():
            missing_variables = "" if p == "<sin_missing>" else p
            rows.append(
                {
                    "category": category,
                    "pattern": p,
                    "rows": int(ct),
                    "pct_rows_category": round((ct / n_total) * 100, 4) if n_total else 0.0,
                    "missing_variables": missing_variables,
                    "n_missing_variables": 0 if p == "<sin_missing>" else len(missing_variables.split("|")),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["category", "rows"], ascending=[True, False]).reset_index(drop=True)


def _write_additional_recommendations(path: Path) -> None:
    suggestions = [
        "1) Matriz de co-missingness (phi/correlación entre máscaras de faltantes) para detectar bloques de captura.",
        "2) Test MCAR de Little (si aplica) para evaluar si faltantes son completamente al azar.",
        "3) Missingness por INTERVAL_NAME / VISIT_DATE para detectar sesgo temporal.",
        "4) Modelo de propensión a missing (logístico) usando variables disponibles para identificar MAR.",
        "5) Semáforo de imputación por variable: baja/media/alta complejidad según missing_pct y tipo de dato.",
        "6) Monitoreo por versión del pipeline para comparar si el missingness mejora tras correcciones.",
    ]
    path.write_text("\n".join(suggestions) + "\n", encoding="utf-8")


def main() -> None:
    cfg = _parse_args()
    logger = setup_logger("16_missingness_patterns")

    print_script_overview(
        "16_missingness_patterns.py",
        "Análisis de missingness por variable, categoría y patrones de faltantes",
    )

    print_step(1, "Cargando dataset corregido")
    df = _load_table(cfg.input_path)
    print_kv(
        "Input",
        {
            "path": cfg.input_path,
            "rows": len(df),
            "columns": len(df.columns),
        },
    )

    print_step(2, "Calculando missingness por variable")
    variable_summary, mask_df = _build_variable_summary(df)

    print_step(3, "Calculando missingness agregado por categoría")
    category_summary = _build_category_summary(variable_summary)

    print_step(4, "Identificando patrones globales y por categoría")
    overall_patterns = _build_patterns(mask_df, top_n=cfg.top_n)
    by_category_patterns = _build_patterns_by_category(df, top_n=cfg.top_n)

    print_step(5, "Construyendo tablas de variables con más y menos faltantes")
    top_most_missing = variable_summary.head(cfg.top_n).copy()
    top_least_missing = variable_summary.sort_values(["missing_pct", "column"], ascending=[True, True]).head(cfg.top_n)

    print_step(6, "Guardando reportes")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    variable_summary.to_csv(cfg.output_dir / "missingness_by_variable.csv", index=False)
    category_summary.to_csv(cfg.output_dir / "missingness_by_category.csv", index=False)
    overall_patterns.to_csv(cfg.output_dir / "missingness_patterns_overall_top.csv", index=False)
    by_category_patterns.to_csv(cfg.output_dir / "missingness_patterns_by_category_top.csv", index=False)
    top_most_missing.to_csv(cfg.output_dir / "variables_most_missing_top.csv", index=False)
    top_least_missing.to_csv(cfg.output_dir / "variables_least_missing_top.csv", index=False)
    _write_additional_recommendations(cfg.output_dir / "missingness_additional_recommendations.txt")

    with pd.ExcelWriter(cfg.output_dir / "missingness_report.xlsx", engine="openpyxl") as writer:
        variable_summary.to_excel(writer, sheet_name="missing_by_variable", index=False)
        category_summary.to_excel(writer, sheet_name="missing_by_category", index=False)
        overall_patterns.to_excel(writer, sheet_name="patterns_overall_top", index=False)
        by_category_patterns.to_excel(writer, sheet_name="patterns_by_category", index=False)
        top_most_missing.to_excel(writer, sheet_name="top_most_missing", index=False)
        top_least_missing.to_excel(writer, sheet_name="top_least_missing", index=False)

    logger.info("Missingness report generated at %s", cfg.output_dir)
    print_kv(
        "Outputs",
        {
            "output_dir": cfg.output_dir,
            "files": 7,
            "top_n": cfg.top_n,
        },
    )


if __name__ == "__main__":
    main()
