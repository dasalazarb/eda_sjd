from __future__ import annotations

"""
20_btris_visit_date_match_report.py  (v4)
==========================================
1. Determina qualifying dates por paciente (clustering por año, 7 reglas).
2. Extrae registros BTRIS que coinciden con esas fechas.
3. Guarda DFs separados por protocolo y prefijo en:
       data_analytic/BTRIS/11D/{prefix}_records.{parquet,csv}
       data_analytic/BTRIS/15D/{prefix}_records.{parquet,csv}
4. Analiza tests repetidos (Lab, Microbiology) dentro del mismo
   ids__interval_name → reporta si el mismo test aparece en >1 visit_date
   para la misma visita.
5. Escribe reporte de auditoría en data_analytic/BTRIS/report.txt
"""

import argparse
import datetime
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

from common import ANALYTIC_DIR, INTERMEDIATE_DIR, REPORTS_DIR, print_kv, print_script_overview, print_step, setup_logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_INTERVAL_15D = "Natural History Protocol 478 Interval"
OUTPUT_DIR_DEFAULT  = Path("data_analytic/BTRIS")

FILE_PREFIX_DATE_CONFIG: dict[str, list[str]] = {
    "lab":                   ["Collected Date Time", "Reported Date Time"],
    "microbiology":          ["Collected Date"],
    "clindocdiscrete":       ["Document Date", "Observation Date"],
    "diagnosisandprocedure": ["Observation Date", "Admission Date"],
    "echo":                  ["Result Date"],
    "ekg":                   ["Result Date"],
    "medication":            ["Order Date/Start Date", "Order Date"],
    "pathology":             ["Date"],
    "pftlab":                ["Performed Date Time"],
    "radiology":             ["Exam Date"],
    "vitalsigns":            ["Observation Date"],
    "vital signs":           ["Observation Date"],
}

# Columnas para identificar el nombre del test en análisis de repetición
LAB_TEST_COL_CANDIDATES   = ["Observation Name", "Order Name", "Test Name"]
MICRO_TEST_COL_CANDIDATES = ["Event Name", "Test Name"]

OBS_VALUE_CANDIDATES: list[str] = ["Observation Value", "Result Value", "Value", "Numeric Value"]

SKIP_PREFIXES: set[str] = {"demographics"}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VisitDateMatchConfig:
    patients_path: Path
    btris_root: Path
    output_dir: Path


def _parse_args() -> VisitDateMatchConfig:
    parser = argparse.ArgumentParser(
        description="Extrae registros BTRIS para las qualifying dates de cada paciente."
    )
    parser.add_argument(
        "--patients-path", type=Path,
        default=ANALYTIC_DIR / "visits_long_collapsed_by_interval_codebook_corrected.parquet",
    )
    parser.add_argument(
        "--btris-root", type=Path,
        default=INTERMEDIATE_DIR / "BTRIS",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR_DEFAULT)
    args = parser.parse_args()
    return VisitDateMatchConfig(
        patients_path=args.patients_path,
        btris_root=args.btris_root,
        output_dir=args.output_dir,
    )


# ---------------------------------------------------------------------------
# Column utilities
# ---------------------------------------------------------------------------


def _norm_col(name: object) -> str:
    text = str(name) if name is not None else ""
    return re.sub(r"\s+", " ", text.replace("\ufeff", "").replace("\u00a0", " ").strip()).lower()


def _resolve_column(columns: pd.Index, expected: str) -> str:
    norm = _norm_col(expected)
    for col in columns:
        if _norm_col(col) == norm:
            return str(col)
    raise KeyError(f"Columna no encontrada: '{expected}'")


def _resolve_first(columns: pd.Index, candidates: list[str]) -> Optional[str]:
    for cand in candidates:
        try:
            return _resolve_column(columns, cand)
        except KeyError:
            pass
    return None


def _normalize_patient_id(series: pd.Series) -> pd.Series:
    return (
        series.astype(str).str.strip()
        .str.replace(r"[-/\\\s]", "", regex=True)
        .replace("nan", pd.NA).replace("", pd.NA)
    )


def _parse_dates_to_date_only(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", dayfirst=False).dt.normalize().dt.date


def _detect_protocol(csv_path: Path) -> str:
    for part in csv_path.parts:
        if part.upper() == "11D":
            return "11D"
        if part.upper() == "15D":
            return "15D"
    path_str = str(csv_path)
    if re.search(r"[/\\]11D[/\\]", path_str, re.IGNORECASE):
        return "11D"
    if re.search(r"[/\\]15D[/\\]", path_str, re.IGNORECASE):
        return "15D"
    return ""


# ---------------------------------------------------------------------------
# Date-selection logic
# ---------------------------------------------------------------------------


def _cluster_years(unique_years: list[int]) -> list[list[int]]:
    clusters: list[list[int]] = []
    current = [unique_years[0]]
    for y in unique_years[1:]:
        if y - current[-1] <= 1:
            current.append(y)
        else:
            clusters.append(current)
            current = [y]
    clusters.append(current)
    return clusters


def _select_qualifying_dates(raw: str) -> tuple[list[date], str]:
    """
    Retorna (lista de fechas calificadoras, etiqueta de regla).

    Reglas:
      · 1 fecha única            → esa fecha,  'single_date'
      · todos mismo año          → todas,       'same_year_all'
      · span ≤ 1 año             → todas,       'consecutive_years_all'
      · span ≥ 2 → clustering:
          - 1 cluster dominante  → solo ese,    'dominant_cluster_YYYY'
          - empate               → más antiguo, 'tie_earliest_cluster_YYYY'
    """
    parts = [p.strip() for p in str(raw).split("|") if p.strip()]
    dates: list[date] = []
    for p in parts:
        ts = pd.to_datetime(p, errors="coerce", dayfirst=False)
        if not pd.isna(ts):
            dates.append(ts.normalize().date())
    dates = sorted(set(dates))

    if not dates:
        return [], "no_valid_dates"
    if len(dates) == 1:
        return dates, "single_date"

    year_counts   = Counter(d.year for d in dates)
    unique_years  = sorted(year_counts.keys())

    if len(unique_years) == 1:
        return dates, "same_year_all"

    if unique_years[-1] - unique_years[0] <= 1:
        return dates, "consecutive_years_all"

    # gap ≥ 2: clustering
    clusters = _cluster_years(unique_years)
    cluster_counts = [(cl, sum(year_counts[y] for y in cl)) for cl in clusters]
    max_count  = max(c for _, c in cluster_counts)
    dominant   = [cl for cl, c in cluster_counts if c == max_count]

    if len(dominant) == 1:
        winning_years = set(dominant[0])
        y_label = (str(dominant[0][0]) if len(dominant[0]) == 1
                   else f"{dominant[0][0]}-{dominant[0][-1]}")
        rule = f"dominant_cluster_{y_label}"
    else:
        winning_years = set(dominant[0])
        rule = f"tie_earliest_cluster_{dominant[0][0]}"

    return [d for d in dates if d.year in winning_years], rule


# ---------------------------------------------------------------------------
# Patient loader → expanded qualifying-date table
# ---------------------------------------------------------------------------


def _build_patient_qual_table(path: Path) -> pd.DataFrame:
    """
    Carga parquet y expande ids__visit_date en qualifying dates.
    Una fila por (patient_id, ids__interval_name, qualifying_date).
    """
    if not path.exists():
        raise FileNotFoundError(f"No encontrado: {path}")
    df = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)

    required = ["ids__patient_record_number", "ids__interval_name", "ids__visit_date"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Faltan columnas: {missing}")

    rows: list[dict] = []
    for _, row in df.iterrows():
        raw_mrn = str(row["ids__patient_record_number"]).strip()
        patient_id = re.sub(r"[-/\\\s]", "", raw_mrn)
        if not patient_id or patient_id == "nan":
            continue

        interval_name = str(row["ids__interval_name"]).strip()
        protocol      = "15D" if interval_name == TARGET_INTERVAL_15D else "11D"
        raw_dates     = str(row["ids__visit_date"])

        qual_dates, rule = _select_qualifying_dates(raw_dates)
        for qd in qual_dates:
            rows.append({
                "patient_id":                 patient_id,
                "ids__patient_record_number": raw_mrn,
                "ids__interval_name":         interval_name,
                "ids__visit_date_raw":        raw_dates,
                "qualifying_date":            qd,
                "resolution_rule":            rule,
                "expected_protocol":          protocol,
            })

    return pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)


# ---------------------------------------------------------------------------
# Prefix detection
# ---------------------------------------------------------------------------


def _get_prefix(filename: str) -> Optional[str]:
    """Retorna la clave de FILE_PREFIX_DATE_CONFIG o None si debe omitirse."""
    name_lower = filename.lower()
    for skip in SKIP_PREFIXES:
        if name_lower.startswith(skip):
            return None
    for prefix in sorted(FILE_PREFIX_DATE_CONFIG.keys(), key=len, reverse=True):
        if name_lower.startswith(prefix.replace(" ", "")):
            return prefix
    return None


def _prefix_to_filename(prefix: str) -> str:
    """Normaliza prefix para usar como nombre de archivo."""
    return re.sub(r"\s+", "", prefix.lower())


# ---------------------------------------------------------------------------
# Core matcher — one CSV file
# ---------------------------------------------------------------------------


def _match_file(
    csv_path: Path,
    patients_expanded: pd.DataFrame,
    logger,
) -> Optional[pd.DataFrame]:
    """
    Retorna todos los registros BTRIS que coincidan con
    (patient_id, qualifying_date) para el protocolo del archivo.
    """
    prefix = _get_prefix(csv_path.name)
    if prefix is None:
        return None

    protocol = _detect_protocol(csv_path)
    if not protocol:
        logger.warning("Sin protocolo detectado: %s", csv_path.name)
        return None

    patients_proto = patients_expanded[
        patients_expanded["expected_protocol"] == protocol
    ].copy()
    if patients_proto.empty:
        return None

    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception as exc:
        logger.warning("Error leyendo %s: %s", csv_path.name, exc)
        return None

    try:
        mrn_col = _resolve_column(df.columns, "MRN")
    except KeyError:
        logger.warning("%s sin columna MRN", csv_path.name)
        return None

    df["_patient_id"] = _normalize_patient_id(df[mrn_col])

    # Fallback: usar la primera columna de fecha que exista en el archivo
    resolved_col: Optional[str] = None
    for dc in FILE_PREFIX_DATE_CONFIG[prefix]:
        try:
            resolved_col = _resolve_column(df.columns, dc)
            break
        except KeyError:
            pass

    if resolved_col is None:
        logger.warning("%s sin columnas de fecha %s — se omite",
                       csv_path.name, FILE_PREFIX_DATE_CONFIG[prefix])
        return None

    lookup = (
        patients_proto[[
            "patient_id", "ids__patient_record_number", "ids__interval_name",
            "ids__visit_date_raw", "qualifying_date", "resolution_rule",
        ]]
        .drop_duplicates()
    )

    working = df.copy()
    working["_btris_date"] = _parse_dates_to_date_only(working[resolved_col])
    working = working.dropna(subset=["_patient_id", "_btris_date"])
    if working.empty:
        return None

    combined = working.merge(
        lookup,
        left_on=["_patient_id", "_btris_date"],
        right_on=["patient_id", "qualifying_date"],
        how="inner",
    )
    if combined.empty:
        return None

    combined["ids__btris_date_col"] = resolved_col

    meta_cols    = [
        "ids__patient_record_number", "ids__interval_name",
        "ids__visit_date_raw", "qualifying_date", "resolution_rule",
        "ids__btris_date_col",
    ]
    drop_internal = {"_patient_id", "_orig_idx", "_btris_date", "patient_id"}
    btris_cols    = [c for c in combined.columns if c not in set(meta_cols) | drop_internal]

    result = combined[meta_cols + btris_cols].copy()
    result = result.rename(columns={"qualifying_date": "ids__visit_date"})
    result.insert(5, "ids__source_file", csv_path.name)
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Repeated-tests analysis — merge rules
# ---------------------------------------------------------------------------

# Keywords that identify a failed / invalid specimen result
_ERROR_KWS = ("see note", "not performed", "clotted", "specimen", "see below")

# Abbreviation synonym map: normalize to a canonical form (all lower-case keys)
_SYNONYMS: dict[str, str] = {
    "neg": "Negative", "negative": "Negative",
    "pos": "Positive", "positive": "Positive",
    "rare": "Rare", "trace": "Trace",
    "clear": "Clear", "cloudy": "Cloudy",
    "normal": "Normal", "abnormal": "Abnormal",
}

# Maximum gap (days) within which serial numeric / categorical measurements
# are collapsed to the earliest date.  Gaps above this threshold are kept
# as separate records (they represent genuinely distinct visits).
_SERIAL_GAP_MAX_DAYS = 180


def _is_error_value(val: str) -> bool:
    """True if the value string represents a failed / invalid specimen."""
    v = val.lower()
    return any(kw in v for kw in _ERROR_KWS) or v.strip() in ("note:", "")


def _normalize_val(val: str) -> str:
    """
    Normalize a single observation value string:
      • strip whitespace
      • map known synonyms to canonical form
      • leave everything else unchanged
    """
    v = val.strip()
    return _SYNONYMS.get(v.lower(), v)


def _try_float(val: str) -> Optional[float]:
    """Return float if val is purely numeric (handles '< 0.2' → None)."""
    try:
        return float(val)
    except ValueError:
        return None


def _date_gap_days(dates_str: str) -> int:
    """Given a '|'-joined date string return max - min gap in days."""
    parts = [p.strip() for p in dates_str.split("|") if p.strip()]
    parsed = [pd.to_datetime(p, errors="coerce") for p in parts]
    valid  = [d for d in parsed if not pd.isna(d)]
    if len(valid) < 2:
        return 0
    return int((max(valid) - min(valid)).days)


def _apply_merge_rules(
    group_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    """
    Given all rows for one (patient, interval, test_name) group with >1 visit_date,
    apply the 8 merge rules and return:
        (rows_to_keep, merge_rule_label)

    Rules (applied in priority order):
      R1  error_specimen   — drop failed-specimen rows, keep valid ones
      R2  case_synonym     — values differ only by case/abbreviation → keep earliest
      R3  wildcard         — one value is '*' placeholder → keep informative
      R4  numeric ≤180d    — serial numeric draws, gap ≤ 180d → keep earliest date
      R5  categorical ≤30d — categorical differ, gap ≤ 30d → keep earliest date
      R6  wide_gap >180d   — genuinely separate visits → keep ALL (no merge)
      R7  categorical wide — categorical differ, gap > 30d → keep ALL (no merge)
      R8  3+_values        — more values than dates → dedup within-date first,
                             then apply rules above
    """
    rows = group_rows.copy()
    rows = rows.sort_values("ids__visit_date").reset_index(drop=True)

    vals        = rows["_obs_val"].tolist()
    dates_str   = "|".join(rows["_vdate_str"].tolist())
    gap         = _date_gap_days(dates_str)

    # ── R1: error / failed specimen ──────────────────────────────────────────
    error_mask = [_is_error_value(v) for v in vals]
    valid_rows = rows[[not e for e in error_mask]]
    if any(error_mask):
        if valid_rows.empty:
            # all rows are errors — flag but keep first row as placeholder
            rows.loc[0:0, "_merge_rule"] = "R1_all_invalid"
            return rows.iloc[[0]], "R1_all_invalid"
        rows = valid_rows.reset_index(drop=True)
        vals = rows["_obs_val"].tolist()
        if len(rows) == 1:
            rows["_merge_rule"] = "R1_error_dropped"
            return rows, "R1_error_dropped"
        # re-evaluate gap after dropping errors
        dates_str = "|".join(rows["_vdate_str"].tolist())
        gap = _date_gap_days(dates_str)

    # ── R8: more values than rows (sub-visit duplicates) — dedup within date ─
    # Group by date, keep last value per date (most recent run), then continue
    if rows["_vdate_str"].duplicated().any():
        rows = (
            rows.sort_values("_vdate_str")
            .drop_duplicates(subset=["_vdate_str"], keep="last")
            .reset_index(drop=True)
        )
        vals = rows["_obs_val"].tolist()
        if len(rows) == 1:
            rows["_merge_rule"] = "R8_subvisit_dedup"
            return rows, "R8_subvisit_dedup"

    # ── R2: case / abbreviation synonym ──────────────────────────────────────
    norm_vals = [_normalize_val(v) for v in vals]
    if len(set(norm_vals)) == 1:
        kept = rows.iloc[[0]].copy()
        kept["_obs_val"] = norm_vals[0]
        kept["_merge_rule"] = "R2_case_synonym"
        return kept, "R2_case_synonym"

    # ── R3: wildcard placeholder (*) ─────────────────────────────────────────
    non_wild = rows[rows["_obs_val"].str.strip() != "*"].copy()
    if len(non_wild) < len(rows):
        if non_wild.empty:
            rows["_merge_rule"] = "R3_all_wildcard"
            return rows.iloc[[0]], "R3_all_wildcard"
        rows = non_wild.reset_index(drop=True)
        vals = rows["_obs_val"].tolist()
        if len(rows) == 1:
            rows["_merge_rule"] = "R3_wildcard_dropped"
            return rows, "R3_wildcard_dropped"
        dates_str = "|".join(rows["_vdate_str"].tolist())
        gap = _date_gap_days(dates_str)

    # ── R4: numeric serial, gap ≤ 180d ───────────────────────────────────────
    floats = [_try_float(v) for v in vals]
    all_numeric = all(f is not None for f in floats)
    if all_numeric and gap <= _SERIAL_GAP_MAX_DAYS:
        kept = rows.iloc[[0]].copy()
        kept["_merge_rule"] = f"R4_numeric_serial_earliest_gap{gap}d"
        return kept, f"R4_numeric_serial_earliest_gap{gap}d"

    # ── R5: categorical, gap ≤ 30d ────────────────────────────────────────────
    if gap <= 30:
        kept = rows.iloc[[0]].copy()
        kept["_merge_rule"] = f"R5_categorical_earliest_gap{gap}d"
        return kept, f"R5_categorical_earliest_gap{gap}d"

    # ── R6: numeric, gap > 180d — keep earliest ──────────────────────────────
    if all_numeric and gap > _SERIAL_GAP_MAX_DAYS:
        kept = rows.iloc[[0]].copy()
        kept["_merge_rule"] = f"R6_wide_gap_earliest_{gap}d"
        return kept, f"R6_wide_gap_earliest_{gap}d"

    # ── R7: categorical, gap > 30d — keep earliest ───────────────────────────
    kept = rows.iloc[[0]].copy()
    kept["_merge_rule"] = f"R7_categorical_wide_gap_earliest_{gap}d"
    return kept, f"R7_categorical_wide_gap_earliest_{gap}d"


def _classify_repeated_tests(
    df: pd.DataFrame,
    test_col_candidates: list[str],
    entity_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Para cada grupo (patient, interval_name, test_name) con >1 visit_date,
    aplica las 8 reglas de merge y clasifica el resultado en:

      · merged_summary  — grupos que fueron colapsados (R1–R5, R8)
      · kept_separate   — grupos que se mantienen como filas separadas (R6, R7)

    Retorna:
        df_deduped       — df con duplicados resueltos
        merged_summary   — resumen de grupos colapsados
        kept_separate    — detalle de grupos no mergeados (discrepancias reales)
    """
    if df.empty:
        return df.copy(), pd.DataFrame(), pd.DataFrame()

    test_col = _resolve_first(df.columns, test_col_candidates)
    if test_col is None:
        return df.copy(), pd.DataFrame(), pd.DataFrame()

    obs_col = _resolve_first(df.columns, OBS_VALUE_CANDIDATES)

    tmp = df.copy()
    tmp["_test_name"] = tmp[test_col].astype(str).str.strip()
    tmp["_obs_val"]   = tmp[obs_col].astype(str).str.strip() if obs_col else ""
    tmp["_vdate_str"] = tmp["ids__visit_date"].astype(str)
    tmp["_merge_rule"] = ""

    GROUP = ["ids__patient_record_number", "ids__interval_name", "_test_name"]

    # Split single vs repeated
    date_nunique = tmp.groupby(GROUP)["_vdate_str"].transform("nunique")
    df_single = tmp[date_nunique == 1].copy()
    df_rep    = tmp[date_nunique  > 1].copy()

    if df_rep.empty:
        return (
            df.copy(),
            pd.DataFrame(),
            pd.DataFrame(),
        )

    # Apply merge rules group by group
    result_frames:  list[pd.DataFrame] = []
    merged_records: list[dict]         = []
    kept_records:   list[dict]         = []

    for (pid, intv, tname), grp in df_rep.groupby(GROUP, sort=False):
        kept_rows, rule = _apply_merge_rules(grp)
        result_frames.append(kept_rows)

        # Original values and dates for the summary
        orig_vals  = "|".join(sorted(set(grp["_obs_val"].tolist())))
        orig_dates = "|".join(sorted(set(grp["_vdate_str"].tolist())))
        src_files  = "|".join(sorted({str(v) for v in grp["ids__source_file"]}))
        n_orig     = grp["_vdate_str"].nunique()

        record = {
            "entity":                     entity_label,
            "ids__patient_record_number": pid,
            "ids__interval_name":         intv,
            "test_name":                  tname,
            "n_visit_dates_original":     n_orig,
            "n_rows_kept":                len(kept_rows),
            "visit_dates_original":       orig_dates,
            "observation_values_original": orig_vals,
            "merge_rule":                 rule,
            "source_files":               src_files,
        }
        merged_records.append(record)

    # Reconstruct deduped df
    internal = ["_test_name", "_obs_val", "_vdate_str", "_merge_rule"]
    all_frames = [df_single] + result_frames
    df_deduped = (
        pd.concat(all_frames, ignore_index=True)
        .drop(columns=[c for c in internal if c in
                        pd.concat(all_frames, ignore_index=True).columns])
    )

    merged_summary = (
        pd.DataFrame(merged_records)
        .sort_values(["ids__patient_record_number", "ids__interval_name"], ignore_index=True)
        if merged_records else pd.DataFrame()
    )
    kept_separate = (
        pd.DataFrame(kept_records)
        .sort_values(["ids__patient_record_number", "ids__interval_name"], ignore_index=True)
        if kept_records else pd.DataFrame()
    )

    return df_deduped, merged_summary, kept_separate


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------


def _save_df(df: pd.DataFrame, folder: Path, stem: str, logger) -> tuple[Path, Path]:
    folder.mkdir(parents=True, exist_ok=True)
    pq  = folder / f"{stem}.parquet"
    csv = folder / f"{stem}.csv"
    df.to_parquet(pq,  index=False)
    df.to_csv(csv,     index=False)
    logger.info("Guardado: %s (%d filas)", pq, len(df))
    return pq, csv


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------


def _write_report(
    report_path: Path,
    patients_expanded: pd.DataFrame,
    results_by_proto_prefix: dict[tuple[str, str], pd.DataFrame],
    merged_summary: pd.DataFrame,
    kept_separate: pd.DataFrame,
    file_stats: list[dict],
    logger,
) -> None:
    sep  = "=" * 90
    sep2 = "-" * 90
    lines: list[str] = []

    def h(title: str) -> None:
        lines.extend(["", sep, f"  {title}", sep])

    def kv(k: str, v) -> None:
        lines.append(f"  {k:<55} {v}")

    lines += [sep,
              "  REPORTE — 20_btris_visit_date_match_report.py",
              f"  Generado: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
              sep]

    # 1. Pacientes y reglas
    h("1. PACIENTES Y QUALIFYING DATES")
    kv("Pacientes únicos:", patients_expanded["patient_id"].nunique())
    kv("Filas expandidas (patient × qualifying_date):", len(patients_expanded))
    for proto in ["11D", "15D"]:
        n = int((patients_expanded["expected_protocol"] == proto).sum())
        kv(f"  Qualifying dates protocolo {proto}:", n)

    h("2. REGLAS DE RESOLUCIÓN APLICADAS")
    rule_counts = (
        patients_expanded
        .drop_duplicates(subset=["patient_id", "ids__interval_name"])
        ["resolution_rule"].value_counts()
    )
    for rule, cnt in rule_counts.items():
        kv(f"  {rule}:", f"{cnt:>5}  ({100*cnt/rule_counts.sum():.1f}%)")

    # Detalle de casos no triviales
    nontrivial = patients_expanded[
        ~patients_expanded["resolution_rule"].isin(["single_date", "same_year_all"])
    ].drop_duplicates(subset=["patient_id", "ids__interval_name"])
    if not nontrivial.empty:
        lines += ["", f"  {'MRN':<15} {'interval':<48} {'regla':<35} n_qual"]
        lines.append("  " + sep2)
        for _, r in nontrivial.iterrows():
            n_q = int(patients_expanded[
                (patients_expanded["patient_id"] == r["patient_id"]) &
                (patients_expanded["ids__interval_name"] == r["ids__interval_name"])
            ].shape[0])
            lines.append(
                f"  {str(r['patient_id']):<15} "
                f"{str(r['ids__interval_name'])[:46]:<48} "
                f"{r['resolution_rule']:<35} {n_q}"
            )

    # 3. Archivos BTRIS procesados
    h("3. ARCHIVOS BTRIS PROCESADOS")
    lines.append(f"  {'archivo':<55} {'prot':<5} {'prefix':<20} {'rows':<10} {'pacientes'}")
    lines.append("  " + sep2)
    for s in file_stats:
        lines.append(
            f"  {s['file']:<55} {s['protocol']:<5} {s['prefix']:<20} "
            f"{s['rows_matched']:<10} {s['patients_matched']}"
        )
    kv("Total archivos procesados:", len(file_stats))
    kv("Con matches:", sum(1 for s in file_stats if s["rows_matched"] > 0))
    kv("Sin matches:", sum(1 for s in file_stats if s["rows_matched"] == 0))

    # 4. Registros extraídos por protocolo y prefijo
    h("4. REGISTROS EXTRAÍDOS POR PROTOCOLO / PREFIJO")
    lines.append(f"  {'protocolo':<8} {'prefijo':<25} {'registros':<12} {'pacientes':<12} {'archivos_fuente'}")
    lines.append("  " + sep2)
    for (proto, prefix), df in sorted(results_by_proto_prefix.items()):
        n_src = int(df["ids__source_file"].nunique()) if not df.empty else 0
        lines.append(
            f"  {proto:<8} {prefix:<25} {len(df):<12} "
            f"{int(df['ids__patient_record_number'].nunique()) if not df.empty else 0:<12} {n_src}"
        )

    # 5. Tests mergeados (R1–R5, R8)
    h("5. TESTS MERGEADOS — REGLAS R1–R5 / R8 (colapsados a 1 fila por grupo)")
    if merged_summary.empty:
        lines.append("  (ninguno)")
    else:
        kv("Grupos mergeados:", len(merged_summary))
        rule_dist = merged_summary["merge_rule"].str.extract(r"^(R\d+)")[0].value_counts().to_dict()
        for rk, cnt in sorted(rule_dist.items()):
            kv(f"  {rk}:", cnt)
        lines += ["", f"  {'entity':<18} {'MRN':<15} {'interval':<38} {'test_name':<35} {'rule':<35} {'orig_vals'}"]
        lines.append("  " + sep2)
        for _, r in merged_summary.iterrows():
            lines.append(
                f"  {str(r['entity']):<18} "
                f"{str(r['ids__patient_record_number']):<15} "
                f"{str(r['ids__interval_name'])[:36]:<38} "
                f"{str(r['test_name'])[:33]:<35} "
                f"{str(r['merge_rule'])[:33]:<35} "
                f"{r['observation_values_original']}"
            )

    # 6. Tests R6/R7 — gap amplio, se conservó la fecha más temprana
    h("6. TESTS R6/R7 — GAP AMPLIO (>180d / >30d categórico) → fecha más temprana conservada)")
    r67 = merged_summary[merged_summary["merge_rule"].str.startswith(("R6","R7"))] if not merged_summary.empty else pd.DataFrame()
    if r67.empty:
        lines.append("  (ninguno)")
    else:
        kv("Grupos R6/R7 mergeados a fecha más temprana:", len(r67))
        lines += ["", f"  {'entity':<18} {'MRN':<15} {'interval':<38} {'test_name':<35} {'rule':<35} {'orig_vals'}"]
        lines.append("  " + sep2)
        for _, r in r67.iterrows():
            lines.append(
                f"  {str(r['entity']):<18} "
                f"{str(r['ids__patient_record_number']):<15} "
                f"{str(r['ids__interval_name'])[:36]:<38} "
                f"{str(r['test_name'])[:33]:<35} "
                f"{str(r['merge_rule'])[:33]:<35} "
                f"{r['observation_values_original']}"
            )

    # 7. Qualifying dates sin registros
    h("7. QUALIFYING DATES SIN REGISTROS EN NINGÚN ARCHIVO BTRIS")
    all_df = (
        pd.concat(list(results_by_proto_prefix.values()), ignore_index=True)
        if results_by_proto_prefix else pd.DataFrame()
    )
    if not all_df.empty:
        matched_pid_date = set(zip(
            all_df["ids__patient_record_number"].astype(str),
            all_df["ids__visit_date"].astype(str),
        ))
    else:
        matched_pid_date = set()

    missing: list[tuple] = []
    for _, row in patients_expanded.iterrows():
        key = (str(row["ids__patient_record_number"]), str(row["qualifying_date"]))
        if key not in matched_pid_date:
            missing.append((
                str(row["ids__patient_record_number"]),
                str(row["qualifying_date"]),
                row["ids__interval_name"],
                row["expected_protocol"],
            ))

    kv("Qualifying dates sin ningún registro:", len(missing))
    if missing:
        lines += ["", f"  {'MRN':<15} {'date':<14} {'protocol':<8} interval"]
        lines.append("  " + sep2)
        for pid, qd, intv, proto in missing[:120]:
            lines.append(f"  {pid:<15} {qd:<14} {proto:<8} {intv[:55]}")
        if len(missing) > 120:
            lines.append(f"  ... ({len(missing) - 120} más)")

    # 8. Archivos generados
    h("8. ARCHIVOS GENERADOS")
    for (proto, prefix), _ in sorted(results_by_proto_prefix.items()):
        stem = _prefix_to_filename(prefix)
        kv(f"  {proto}/{stem}.parquet + .csv", "✓")
    if not merged_summary.empty:
        kv("  repeated_tests_merged.csv", "✓")
    if not kept_separate.empty:
        kv("  repeated_tests_kept_separate.csv", "✓")
    kv("  report.txt", "✓")

    lines += ["", sep]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Reporte escrito: %s", report_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    cfg    = _parse_args()
    logger = setup_logger("20_btris_visit_date_match_report")

    print_script_overview(
        "20_btris_visit_date_match_report.py",
        (
            "Qualifying dates por clustering → extrae registros BTRIS → "
            "guarda por protocolo/prefijo → analiza tests repetidos por interval_name."
        ),
    )

    # ── Step 1: Qualifying dates ──────────────────────────────────────────────
    print_step(1, "Cargando pacientes y computando qualifying dates")

    patients_expanded = _build_patient_qual_table(cfg.patients_path)
    if patients_expanded.empty:
        raise ValueError("Sin qualifying dates. Revisar parquet de entrada.")

    print_kv("Pacientes expandidos", {
        "pacientes_unicos": int(patients_expanded["patient_id"].nunique()),
        "filas_expandidas": int(len(patients_expanded)),
        "11D": int((patients_expanded["expected_protocol"] == "11D").sum()),
        "15D": int((patients_expanded["expected_protocol"] == "15D").sum()),
    })
    print_kv("Reglas de resolución",
        patients_expanded.drop_duplicates(subset=["patient_id", "ids__interval_name"])
        ["resolution_rule"].value_counts().to_dict()
    )

    # ── Step 2: Descubrir archivos ────────────────────────────────────────────
    print_step(2, "Descubriendo archivos BTRIS")

    all_files    = sorted(p for p in cfg.btris_root.rglob("*.csv") if p.is_file())
    processable  = [f for f in all_files if _get_prefix(f.name) is not None]
    skipped      = [f for f in all_files if _get_prefix(f.name) is None]

    print_kv("Archivos", {
        "total": len(all_files),
        "a_procesar": len(processable),
        "omitidos": len(skipped),
        "omitidos_nombres": str([f.name for f in skipped[:10]]),
    })

    # ── Step 3: Procesar y acumular por (protocol, prefix) ───────────────────
    print_step(3, f"Procesando {len(processable)} archivos BTRIS")

    # Acumuladores: {(protocol, prefix): [DataFrame, ...]}
    accum: dict[tuple[str, str], list[pd.DataFrame]] = {}
    file_stats: list[dict] = []

    for i, csv_path in enumerate(processable, 1):
        prefix   = _get_prefix(csv_path.name)   # guaranteed not None
        protocol = _detect_protocol(csv_path)
        logger.info("[%d/%d] %s  prefix=%s  proto=%s",
                    i, len(processable), csv_path.name, prefix, protocol)

        result = _match_file(csv_path, patients_expanded, logger)

        rows_matched     = len(result)     if result is not None else 0
        patients_matched = (int(result["ids__patient_record_number"].nunique())
                            if result is not None and not result.empty else 0)

        file_stats.append({
            "file": csv_path.name, "protocol": protocol, "prefix": prefix or "",
            "rows_matched": rows_matched, "patients_matched": patients_matched,
        })

        if result is not None and not result.empty and protocol:
            key = (protocol, prefix or "unknown")
            accum.setdefault(key, []).append(result)
            logger.info("  → %d registros, %d pacientes", rows_matched, patients_matched)
        else:
            logger.info("  → sin matches")

    print_kv("Procesamiento", {
        "archivos_procesados":  len(processable),
        "archivos_con_matches": sum(1 for s in file_stats if s["rows_matched"] > 0),
        "total_rows":           sum(s["rows_matched"] for s in file_stats),
    })

    # ── Step 4: Consolidar por (protocol, prefix) ─────────────────────────────
    print_step(4, "Consolidando resultados por protocolo/prefijo")

    results_by_proto_prefix: dict[tuple[str, str], pd.DataFrame] = {}
    for (proto, prefix), frames in accum.items():
        results_by_proto_prefix[(proto, prefix)] = pd.concat(frames, ignore_index=True)

    # ── Step 5: Clasificar repetidos en Lab y Microbiology ────────────────────
    print_step(5, "Clasificando tests repetidos (mismo valor → colapsar / distinto → discrepancia)")

    all_merged:   list[pd.DataFrame] = []
    all_separate: list[pd.DataFrame] = []

    for proto in ["11D", "15D"]:
        for pfx, test_cands, label in [
            ("lab",          LAB_TEST_COL_CANDIDATES,   "Lab"),
            ("microbiology", MICRO_TEST_COL_CANDIDATES, "Microbiology"),
        ]:
            key = (proto, pfx)
            if key not in results_by_proto_prefix:
                continue

            df_orig = results_by_proto_prefix[key]
            df_deduped, merged_sum, kept_sep = _classify_repeated_tests(
                df_orig, test_cands, f"{label} {proto}"
            )
            results_by_proto_prefix[key] = df_deduped   # versión deduplicada

            rows_removed = len(df_orig) - len(df_deduped)
            print_kv(f"  {proto}/{pfx}", {
                "filas_originales":       len(df_orig),
                "grupos_mergeados":       int(len(merged_sum)) if not merged_sum.empty else 0,
                "filas_eliminadas":       rows_removed,
                "grupos_kept_separate":   int(len(kept_sep))   if not kept_sep.empty  else 0,
            })

            if not merged_sum.empty:
                all_merged.append(merged_sum)
            if not kept_sep.empty:
                all_separate.append(kept_sep)

    merged_summary_all = (
        pd.concat(all_merged,   ignore_index=True) if all_merged   else pd.DataFrame()
    )
    kept_separate_all  = (
        pd.concat(all_separate, ignore_index=True) if all_separate else pd.DataFrame()
    )

    # Guardar ambos CSVs
    if not merged_summary_all.empty:
        mp = cfg.output_dir / "repeated_tests_merged.csv"
        merged_summary_all.to_csv(mp, index=False)
        logger.info("Merged summary: %s", mp)
        print_kv("Guardado", {"repeated_tests_merged.csv": str(mp)})

    if not kept_separate_all.empty:
        kp = cfg.output_dir / "repeated_tests_kept_separate.csv"
        kept_separate_all.to_csv(kp, index=False)
        logger.info("Kept separate: %s", kp)
        print_kv("Guardado", {"repeated_tests_kept_separate.csv": str(kp)})

    # ── Step 6: Guardar DFs finales (post-deduplicación) ─────────────────────
    print_step(6, "Guardando archivos finales por protocolo/prefijo")

    for (proto, prefix), df_final in sorted(results_by_proto_prefix.items()):
        stem   = _prefix_to_filename(prefix)
        folder = cfg.output_dir / proto
        _save_df(df_final, folder, f"{stem}_records", logger)
        print_kv(f"  {proto}/{stem}", {
            "filas":           len(df_final),
            "pacientes":       int(df_final["ids__patient_record_number"].nunique()),
            "archivos_fuente": int(df_final["ids__source_file"].nunique()),
        })

    # ── Step 7: Reporte ───────────────────────────────────────────────────────
    print_step(7, "Generando reporte de auditoría")

    report_path = cfg.output_dir / "report.txt"
    _write_report(
        report_path=report_path,
        patients_expanded=patients_expanded,
        results_by_proto_prefix=results_by_proto_prefix,
        merged_summary=merged_summary_all,
        kept_separate=kept_separate_all,
        file_stats=file_stats,
        logger=logger,
    )

    print_kv("Reporte", {"path": str(report_path)})
    logger.info("Done.")


if __name__ == "__main__":
    main()