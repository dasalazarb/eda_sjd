from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from common import (
    ANALYTIC_DIR,
    INTERMEDIATE_DIR,
    REPORTS_DIR,
    print_kv,
    print_script_overview,
    print_step,
    resolve_canonical_column,
    setup_logger,
)

PROTOCOL_A = "11D"
PROTOCOL_B = "15D"
PROTOCOLS = [PROTOCOL_A, PROTOCOL_B]
DEFAULT_INPUT_CANDIDATES = [
    ANALYTIC_DIR / "visits_long_collapsed_by_interval_codebook_corrected.parquet",
    ANALYTIC_DIR / "visits_long_collapsed_by_interval.parquet",
    ANALYTIC_DIR / "visits_long.parquet",
]
TYPE_PLACEHOLDER_VALUES = {"numeric", "datetime", "date", "string", "boolean"}

ESSDAI_COLUMNS = [
    "essdai__essdai_total_score",
    "essdai-_r__essdai_total_score",
]
ESSPRI_COLUMNS = [
    "esspri_questionnaire__esspri_total_score",
    "esspri_questionnaire__dryness",
    "esspri_questionnaire__fatigue",
    "esspri_questionnaire__pain",
    "esspri_questionnaire__limb_pain",
]
SJD_CLASSIFICATION_COLUMNS = [
    "visit_summary_form__sjogrens_class",
    "visit_summary_form__sjögrens_class",
]
SJD_CLASSIFICATION_ELIGIBLE_VALUES = {1, 2, 4}
SECONDS_PER_YEAR = 365.25 * 24 * 60 * 60


@dataclass(frozen=True)
class MetricResult:
    raw_value: float | int | str | None
    display_value: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Builds a protocol-by-protocol analytic flow table with unique totals for "
            "SjD eligibility, longitudinal availability, ESSDAI/ESSPRI coverage, follow-up, and events."
        )
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=None,
        help="Visit-level analytic parquet/csv/xlsx input. Defaults to the richest available pipeline output.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPORTS_DIR / "protocol_flow",
        help="Directory for protocol_flow_table.csv/.xlsx and protocol_flow_table_long.csv.",
    )
    return parser.parse_args()


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".xls", ".xlsx"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported input extension for {path}")


def _type_placeholder_columns(df: pd.DataFrame) -> list[str]:
    clinical_columns = [
        col
        for col in [*ESSDAI_COLUMNS, *ESSPRI_COLUMNS, *SJD_CLASSIFICATION_COLUMNS]
        if col in df.columns
    ]
    clinical_columns.extend(
        col
        for col in sorted(
            set(_candidate_columns(df, ["acr"]))
            | set(_candidate_columns(df, ["eular"]))
            | set(_candidate_columns(df, ["aecg"]))
            | set(_candidate_columns(df, ["classification", "criteria"]))
        )
        if col not in clinical_columns
    )

    placeholder_columns = []
    for col in clinical_columns:
        values = df[col].dropna().astype("string").str.strip().str.lower()
        if values.empty:
            continue
        if set(values.unique()).issubset(TYPE_PLACEHOLDER_VALUES):
            placeholder_columns.append(col)
    return placeholder_columns


def _validate_value_preserving_input(df: pd.DataFrame, path: Path) -> None:
    placeholder_columns = _type_placeholder_columns(df)
    if not placeholder_columns:
        return

    examples = ", ".join(placeholder_columns[:8])
    suffix = "" if len(placeholder_columns) <= 8 else ", ..."
    raise ValueError(
        "Protocol flow table requires value-preserving clinical data, but "
        f"{path} appears to contain codebook type placeholders in clinical columns "
        f"({examples}{suffix}). Use "
        "data_analytic/visits_long_collapsed_by_interval_codebook_corrected.parquet "
        "or another value-preserving visit-level file instead of the type-recoded output."
    )


def _choose_input_path(explicit_path: Path | None) -> Path:
    if explicit_path:
        if not explicit_path.exists():
            raise FileNotFoundError(f"Input not found: {explicit_path}")
        return explicit_path

    for candidate in DEFAULT_INPUT_CANDIDATES:
        if candidate.exists():
            return candidate

    candidates = "\n".join(f"- {path}" for path in DEFAULT_INPUT_CANDIDATES)
    raise FileNotFoundError(f"No default visit-level input found. Checked:\n{candidates}")


def _resolve_optional_column(df: pd.DataFrame, canonical_name: str) -> str | None:
    try:
        return resolve_canonical_column(df, canonical_name)
    except KeyError:
        return None


def _first_present(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    present = set(columns)
    for candidate in candidates:
        if candidate in present:
            return candidate
    return None


def _candidate_columns(df: pd.DataFrame, tokens: Iterable[str]) -> list[str]:
    token_list = [token.lower() for token in tokens]
    return [
        str(col)
        for col in df.columns
        if all(token in str(col).lower() for token in token_list)
    ]


def _normalize_protocol(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.upper()


def _protocol_mask(df: pd.DataFrame, protocol_col: str, protocol: str) -> pd.Series:
    protocol_series = _normalize_protocol(df[protocol_col]).fillna("")
    return protocol_series.str.split(r"\s*\|\s*", regex=True).apply(lambda parts: protocol in set(parts))


def _protocol_patient_sets(df: pd.DataFrame, patient_col: str, protocol_col: str) -> dict[str, set]:
    return {
        protocol: set(df.loc[_protocol_mask(df, protocol_col, protocol), patient_col].dropna().unique())
        for protocol in PROTOCOLS
    }


def _coerce_truthy(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip().str.lower()
    truthy_strings = {
        "1",
        "1.0",
        "true",
        "yes",
        "y",
        "si",
        "sí",
        "positive",
        "pos",
        "eligible",
        "met",
        "meets",
    }
    numeric = pd.to_numeric(series, errors="coerce")
    return normalized.isin(truthy_strings) | (numeric > 0)


def _eligible_patient_set(df: pd.DataFrame, patient_col: str) -> set:
    eligible_mask = pd.Series(False, index=df.index)

    sjd_class_col = _first_present(df.columns, SJD_CLASSIFICATION_COLUMNS)
    if sjd_class_col:
        numeric_class = pd.to_numeric(df[sjd_class_col], errors="coerce")
        eligible_mask = eligible_mask | numeric_class.isin(SJD_CLASSIFICATION_ELIGIBLE_VALUES)

    criteria_cols = sorted(
        set(_candidate_columns(df, ["acr"]))
        | set(_candidate_columns(df, ["eular"]))
        | set(_candidate_columns(df, ["aecg"]))
        | set(_candidate_columns(df, ["classification", "criteria"]))
    )
    for col in criteria_cols:
        eligible_mask = eligible_mask | _coerce_truthy(df[col])

    if not eligible_mask.any():
        return set(df[patient_col].dropna().unique())
    return set(df.loc[eligible_mask, patient_col].dropna().unique())


def _metric_n(n: int | float | None) -> MetricResult:
    if n is None or pd.isna(n):
        return MetricResult(None, "NA")
    return MetricResult(int(n), f"{int(n):,}")


def _metric_n_pct(n: int, denominator: int) -> MetricResult:
    pct = np.nan if denominator == 0 else 100 * n / denominator
    display = f"{n:,} (NA)" if pd.isna(pct) else f"{n:,} ({pct:.1f}%)"
    return MetricResult(n, display)


def _format_median_iqr(values: pd.Series, decimals: int) -> MetricResult:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return MetricResult(np.nan, "NA")
    q1 = clean.quantile(0.25)
    median = clean.median()
    q3 = clean.quantile(0.75)
    fmt = f"{{:.{decimals}f}}"
    return MetricResult(float(median), f"{fmt.format(median)} ({fmt.format(q1)}–{fmt.format(q3)})")


def _patient_visit_stats(
    df: pd.DataFrame,
    patient_col: str,
    visit_date_col: str | None,
    patient_set: set,
) -> tuple[pd.Series, pd.Series]:
    subset = df[df[patient_col].isin(patient_set)].copy()
    visit_counts = subset.groupby(patient_col).size()
    if not visit_date_col:
        return pd.Series(dtype=float), visit_counts

    subset[visit_date_col] = pd.to_datetime(subset[visit_date_col], errors="coerce")
    spans = subset.groupby(patient_col)[visit_date_col].agg(["min", "max"])
    followup_years = (spans["max"] - spans["min"]).dt.total_seconds() / SECONDS_PER_YEAR
    return followup_years.dropna(), visit_counts


def _patients_with_min_nonmissing_visits(
    df: pd.DataFrame,
    patient_col: str,
    columns: list[str],
    min_visits: int,
    patient_set: set,
) -> set:
    present = [col for col in columns if col in df.columns]
    if not present:
        return set()

    subset = df[df[patient_col].isin(patient_set)].copy()
    if subset.empty:
        return set()

    per_measure_sets = []
    for col in present:
        counts = subset.loc[subset[col].notna()].groupby(patient_col).size()
        per_measure_sets.append(set(counts[counts >= min_visits].index))
    return set().union(*per_measure_sets)


def _index_low_and_later_event(
    df: pd.DataFrame,
    patient_col: str,
    visit_date_col: str | None,
    essdai_col: str | None,
    patient_set: set,
) -> tuple[set, set]:
    if not essdai_col:
        return set(), set()

    subset = df[df[patient_col].isin(patient_set)].copy()
    subset["_essdai_numeric"] = pd.to_numeric(subset[essdai_col], errors="coerce")
    subset = subset[subset["_essdai_numeric"].notna()]
    if subset.empty:
        return set(), set()

    sort_cols = [patient_col]
    if visit_date_col:
        subset[visit_date_col] = pd.to_datetime(subset[visit_date_col], errors="coerce")
        sort_cols.append(visit_date_col)
    subset = subset.sort_values(sort_cols, na_position="last")

    index_rows = subset.groupby(patient_col, as_index=False).first()
    low_index = set(index_rows.loc[index_rows["_essdai_numeric"] < 5, patient_col].dropna().unique())
    later_event = set()
    for patient, patient_rows in subset[subset[patient_col].isin(low_index)].groupby(patient_col):
        if len(patient_rows) <= 1:
            continue
        later_rows = patient_rows.iloc[1:]
        if (later_rows["_essdai_numeric"] >= 5).any():
            later_event.add(patient)
    return low_index, later_event


def _format_detail_value(series: pd.Series, patient: object) -> float | int | None:
    if patient not in series.index:
        return None
    value = series.loc[patient]
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return value


def _build_calculation_tables(
    all_patients: set,
    protocol_sets: dict[str, set],
    eligible_total: set,
    eligible_by_protocol: dict[str, set],
    followup_by_protocol: dict[str, set],
    essdai_by_protocol: dict[str, set],
    esspri_by_protocol: dict[str, set],
    low_index_by_protocol: dict[str, set],
    event_by_protocol: dict[str, set],
    followup_years_by_protocol: dict[str, pd.Series],
    visit_counts_by_protocol: dict[str, pd.Series],
    total_followup: set,
    total_essdai: set,
    total_esspri: set,
    total_low_index: set,
    total_event: set,
    total_followup_years: pd.Series,
    total_visit_counts: pd.Series,
) -> dict[str, pd.DataFrame]:
    """Build auditable patient-level tables used to calculate the summary metrics."""
    patient_rows = []
    for patient in sorted(all_patients, key=lambda value: str(value)):
        row = {
            "patient_record_number": patient,
            "in_protocol_11d": patient in protocol_sets[PROTOCOL_A],
            "in_protocol_15d": patient in protocol_sets[PROTOCOL_B],
            "sjd_eligible_total_unique": patient in eligible_total,
            "has_baseline_plus_followup_total_unique": patient in total_followup,
            "has_2plus_essdai_total_unique": patient in total_essdai,
            "has_2plus_esspri_total_unique": patient in total_esspri,
            "essdai_lt5_at_index_total_unique": patient in total_low_index,
            "later_essdai_ge5_event_total_unique": patient in total_event,
            "total_unique_visit_count": _format_detail_value(total_visit_counts, patient),
            "total_unique_followup_years": _format_detail_value(total_followup_years, patient),
        }
        for protocol in PROTOCOLS:
            suffix = protocol.lower()
            row.update(
                {
                    f"sjd_eligible_{suffix}": patient in eligible_by_protocol[protocol],
                    f"has_baseline_plus_followup_{suffix}": patient in followup_by_protocol[protocol],
                    f"has_2plus_essdai_{suffix}": patient in essdai_by_protocol[protocol],
                    f"has_2plus_esspri_{suffix}": patient in esspri_by_protocol[protocol],
                    f"essdai_lt5_at_index_{suffix}": patient in low_index_by_protocol[protocol],
                    f"later_essdai_ge5_event_{suffix}": patient in event_by_protocol[protocol],
                    f"visit_count_{suffix}": _format_detail_value(visit_counts_by_protocol[protocol], patient),
                    f"followup_years_{suffix}": _format_detail_value(followup_years_by_protocol[protocol], patient),
                }
            )
        patient_rows.append(row)

    patient_flags = pd.DataFrame(patient_rows)
    protocol_membership = patient_flags[
        [
            "patient_record_number",
            "in_protocol_11d",
            "in_protocol_15d",
            "sjd_eligible_11d",
            "sjd_eligible_15d",
            "sjd_eligible_total_unique",
        ]
    ].copy()
    followup_stats = patient_flags[
        [
            "patient_record_number",
            "visit_count_11d",
            "followup_years_11d",
            "visit_count_15d",
            "followup_years_15d",
            "total_unique_visit_count",
            "total_unique_followup_years",
        ]
    ].copy()
    return {
        "calculation_patient_flags": patient_flags,
        "calculation_protocol_membership": protocol_membership,
        "calculation_followup_stats": followup_stats,
    }


def _build_metric_rows(
    df: pd.DataFrame,
    raw11: pd.DataFrame,
    raw15: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    patient_col = resolve_canonical_column(df, "patient_record_number")
    protocol_col = resolve_canonical_column(df, "source_protocol")
    visit_date_col = _resolve_optional_column(df, "visit_datetime") or _resolve_optional_column(df, "visit_date")
    essdai_col = _first_present(df.columns, ESSDAI_COLUMNS)
    esspri_cols = [col for col in ESSPRI_COLUMNS if col in df.columns]

    protocol_sets = _protocol_patient_sets(df, patient_col, protocol_col)
    total_patient_set = set(df[patient_col].dropna().unique())
    eligible_total = _eligible_patient_set(df, patient_col)
    eligible_by_protocol = {
        protocol: patients & eligible_total
        for protocol, patients in protocol_sets.items()
    }

    followup_by_protocol = {}
    essdai_by_protocol = {}
    esspri_by_protocol = {}
    low_index_by_protocol = {}
    event_by_protocol = {}
    followup_years_by_protocol = {}
    visit_counts_by_protocol = {}

    for protocol in PROTOCOLS:
        protocol_df = df[_protocol_mask(df, protocol_col, protocol)].copy()
        eligible_patients = eligible_by_protocol[protocol]
        followup_years, visit_counts = _patient_visit_stats(
            protocol_df,
            patient_col,
            visit_date_col,
            eligible_patients,
        )
        followup_patients = set(visit_counts[visit_counts >= 2].index)
        essdai_patients = _patients_with_min_nonmissing_visits(
            protocol_df,
            patient_col,
            [col for col in ESSDAI_COLUMNS if col in protocol_df.columns],
            2,
            eligible_patients,
        )
        esspri_patients = _patients_with_min_nonmissing_visits(
            protocol_df,
            patient_col,
            esspri_cols,
            2,
            eligible_patients,
        )
        low_index, later_event = _index_low_and_later_event(
            protocol_df,
            patient_col,
            visit_date_col,
            essdai_col if essdai_col in protocol_df.columns else None,
            eligible_patients,
        )

        followup_by_protocol[protocol] = followup_patients
        essdai_by_protocol[protocol] = essdai_patients
        esspri_by_protocol[protocol] = esspri_patients
        low_index_by_protocol[protocol] = low_index
        event_by_protocol[protocol] = later_event
        followup_years_by_protocol[protocol] = followup_years
        visit_counts_by_protocol[protocol] = visit_counts

    total_followup_years, total_visit_counts = _patient_visit_stats(
        df,
        patient_col,
        visit_date_col,
        eligible_total,
    )
    total_followup = set(total_visit_counts[total_visit_counts >= 2].index)
    total_essdai = _patients_with_min_nonmissing_visits(
        df,
        patient_col,
        [col for col in ESSDAI_COLUMNS if col in df.columns],
        2,
        eligible_total,
    )
    total_esspri = _patients_with_min_nonmissing_visits(df, patient_col, esspri_cols, 2, eligible_total)
    total_low_index, total_event = _index_low_and_later_event(
        df,
        patient_col,
        visit_date_col,
        essdai_col,
        eligible_total,
    )

    metrics = [
        (
            "Registros brutos",
            _metric_n(len(raw11)),
            _metric_n(len(raw15)),
            _metric_n(len(raw11) + len(raw15)),
        ),
        (
            "Pacientes únicos tras linkage/deduplicación",
            _metric_n(len(protocol_sets[PROTOCOL_A])),
            _metric_n(len(protocol_sets[PROTOCOL_B])),
            _metric_n(len(total_patient_set)),
        ),
        (
            "SjD elegible según ACR/EULAR y/o AECG",
            _metric_n(len(eligible_by_protocol[PROTOCOL_A])),
            _metric_n(len(eligible_by_protocol[PROTOCOL_B])),
            _metric_n(len(eligible_total)),
        ),
        (
            "Con basal + ≥1 seguimiento",
            _metric_n(len(followup_by_protocol[PROTOCOL_A])),
            _metric_n(len(followup_by_protocol[PROTOCOL_B])),
            _metric_n(len(total_followup)),
        ),
        (
            "Con ≥2 ESSDAI",
            _metric_n_pct(len(essdai_by_protocol[PROTOCOL_A]), len(eligible_by_protocol[PROTOCOL_A])),
            _metric_n_pct(len(essdai_by_protocol[PROTOCOL_B]), len(eligible_by_protocol[PROTOCOL_B])),
            _metric_n_pct(len(total_essdai), len(eligible_total)),
        ),
        (
            "Con ≥2 ESSPRI/PRO comparables",
            _metric_n_pct(len(esspri_by_protocol[PROTOCOL_A]), len(eligible_by_protocol[PROTOCOL_A])),
            _metric_n_pct(len(esspri_by_protocol[PROTOCOL_B]), len(eligible_by_protocol[PROTOCOL_B])),
            _metric_n_pct(len(total_esspri), len(eligible_total)),
        ),
        (
            "Seguimiento, mediana (IQR), años",
            _format_median_iqr(followup_years_by_protocol[PROTOCOL_A], 1),
            _format_median_iqr(followup_years_by_protocol[PROTOCOL_B], 1),
            _format_median_iqr(total_followup_years, 1),
        ),
        (
            "Visitas por paciente, mediana (IQR)",
            _format_median_iqr(visit_counts_by_protocol[PROTOCOL_A], 0),
            _format_median_iqr(visit_counts_by_protocol[PROTOCOL_B], 0),
            _format_median_iqr(total_visit_counts, 0),
        ),
        (
            "ESSDAI <5 al índice",
            _metric_n(len(low_index_by_protocol[PROTOCOL_A])),
            _metric_n(len(low_index_by_protocol[PROTOCOL_B])),
            _metric_n(len(total_low_index)),
        ),
        (
            "Evento posterior ESSDAI ≥5",
            _metric_n(len(event_by_protocol[PROTOCOL_A])),
            _metric_n(len(event_by_protocol[PROTOCOL_B])),
            _metric_n(len(total_event)),
        ),
    ]

    wide = pd.DataFrame(
        [
            {
                "Indicador": label,
                "Protocolo A": a.display_value,
                "Protocolo B": b.display_value,
                "Total único": total.display_value,
            }
            for label, a, b, total in metrics
        ]
    )
    long = pd.DataFrame(
        [
            {
                "indicador": label,
                "columna": column,
                "valor_crudo": metric.raw_value,
                "valor_formateado": metric.display_value,
            }
            for label, a, b, total in metrics
            for column, metric in [
                ("Protocolo A", a),
                ("Protocolo B", b),
                ("Total único", total),
            ]
        ]
    )
    calculation_tables = _build_calculation_tables(
        total_patient_set,
        protocol_sets,
        eligible_total,
        eligible_by_protocol,
        followup_by_protocol,
        essdai_by_protocol,
        esspri_by_protocol,
        low_index_by_protocol,
        event_by_protocol,
        followup_years_by_protocol,
        visit_counts_by_protocol,
        total_followup,
        total_essdai,
        total_esspri,
        total_low_index,
        total_event,
        total_followup_years,
        total_visit_counts,
    )
    return wide, long, calculation_tables


def _load_raw_protocol(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def main() -> None:
    args = _parse_args()
    logger = setup_logger("24_protocol_flow_table")

    print_script_overview(
        "24_protocol_flow_table.py",
        "Builds the requested protocol A/B/unique-total SjD analytic flow table.",
    )

    print_step(1, "Load visit-level analytic table and raw protocol tables")
    input_path = _choose_input_path(args.input_path)
    visits = _read_table(input_path)
    _validate_value_preserving_input(visits, input_path)
    raw11 = _load_raw_protocol(INTERMEDIATE_DIR / "11d_raw_enriched.parquet")
    raw15 = _load_raw_protocol(INTERMEDIATE_DIR / "15d_raw_enriched.parquet")

    print_step(2, "Compute protocol flow metrics")
    wide, long, calculation_tables = _build_metric_rows(visits, raw11, raw15)

    print_step(3, "Save protocol flow summary and calculation tables")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    wide_csv = args.output_dir / "protocol_flow_table.csv"
    long_csv = args.output_dir / "protocol_flow_table_long.csv"
    xlsx_path = args.output_dir / "protocol_flow_table.xlsx"
    wide.to_csv(wide_csv, index=False)
    long.to_csv(long_csv, index=False)
    calculation_paths = {}
    for table_name, table in calculation_tables.items():
        table_path = args.output_dir / f"{table_name}.csv"
        table.to_csv(table_path, index=False)
        calculation_paths[table_name] = table_path

    with pd.ExcelWriter(xlsx_path) as writer:
        wide.to_excel(writer, sheet_name="protocol_flow_table", index=False)
        long.to_excel(writer, sheet_name="protocol_flow_long", index=False)
        for table_name, table in calculation_tables.items():
            table.to_excel(writer, sheet_name=table_name[:31], index=False)

    logger.info("Saved protocol flow table: %s", wide_csv)
    logger.info("Saved protocol flow long table: %s", long_csv)
    for table_name, table_path in calculation_paths.items():
        logger.info("Saved protocol flow calculation table %s: %s", table_name, table_path)
    logger.info("Saved protocol flow workbook: %s", xlsx_path)
    print_kv(
        "Protocol flow outputs",
        {
            "input_path": input_path,
            "wide_csv": wide_csv,
            "long_csv": long_csv,
            **calculation_paths,
            "xlsx": xlsx_path,
        },
    )


if __name__ == "__main__":
    main()
