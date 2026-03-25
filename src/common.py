from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data_raw"
INTERMEDIATE_DIR = ROOT / "data_intermediate"
ANALYTIC_DIR = ROOT / "data_analytic"
REPORTS_DIR = ROOT / "reports"
LOGS_DIR = ROOT / "logs"
LOG_FILE = LOGS_DIR / "pipeline.log"

MISSING_TOKENS = {"", " ", "NA", "N/A", "NULL", "NONE", "NAN", "NaN", "na", "n/a"}


def ensure_dirs() -> None:
    for p in [RAW_DIR, INTERMEDIATE_DIR, ANALYTIC_DIR, REPORTS_DIR, LOGS_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def setup_logger(name: str) -> logging.Logger:
    ensure_dirs()
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    fh = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


def _make_unique_columns(columns: pd.Index) -> pd.Index:
    seen: dict[str, int] = {}
    unique: list[str] = []

    for col in columns.astype(str):
        n = seen.get(col, 0)
        if n == 0:
            unique.append(col)
        else:
            unique.append(f"{col}__dup{n}")
        seen[col] = n + 1

    return pd.Index(unique)


def build_group_prefixed_columns(group_row: pd.Series, variable_row: pd.Series) -> pd.Index:
    group_filled = group_row.ffill()
    out: list[str] = []

    for i, (group, variable) in enumerate(zip(group_filled, variable_row), start=1):
        group_txt = "uncategorized" if pd.isna(group) else str(group).strip()
        variable_txt = "" if pd.isna(variable) else str(variable).strip()

        if not variable_txt:
            variable_txt = f"unnamed_col_{i}"

        out.append(f"{group_txt}__{variable_txt}")

    return pd.Index(out)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    def _normalize_column_name(col: object) -> str:
        raw = str(col).strip().lower()
        parts = raw.split("__")
        normalized_parts: list[str] = []

        for idx, part in enumerate(parts, start=1):
            clean = re.sub(r"[^0-9a-zA-Z]+", "_", part).strip("_")
            if not clean:
                clean = f"unnamed_part_{idx}"
            normalized_parts.append(clean)

        return "__".join(normalized_parts)

    cols = pd.Index([_normalize_column_name(col) for col in df.columns])
    out = df.copy()
    out.columns = _make_unique_columns(cols)
    return out


def replace_empty_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_object_dtype(out[col]):
            out[col] = out[col].astype(str).str.strip()
            out[col] = out[col].replace(list(MISSING_TOKENS), np.nan)
    return out


def parse_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def _resolve_column_name(base_col: str) -> str | None:
        if base_col in out.columns:
            return base_col
        candidates = [c for c in out.columns if str(c).endswith(f"__{base_col}")]
        if not candidates:
            return None
        non_dup = [c for c in candidates if "__dup" not in str(c)]
        return non_dup[0] if non_dup else candidates[0]

    visit_date_col = _resolve_column_name("visit_date")
    time_col = _resolve_column_name("time_24_hour")

    if visit_date_col:
        out["visit_date"] = pd.to_datetime(out[visit_date_col], errors="coerce")

    if time_col:
        t = pd.to_datetime(out[time_col], errors="coerce", format="mixed")
        out["time_24_hour"] = t.dt.strftime("%H:%M:%S")
        out.loc[t.isna(), "time_24_hour"] = np.nan

    if {"visit_date", "time_24_hour"}.issubset(out.columns):
        dt = out["visit_date"].dt.strftime("%Y-%m-%d") + " " + out["time_24_hour"].fillna("00:00:00")
        out["visit_datetime"] = pd.to_datetime(dt, errors="coerce")
    elif "visit_date" in out.columns:
        out["visit_datetime"] = out["visit_date"]
    else:
        out["visit_datetime"] = pd.NaT

    return out


def save_parquet_and_csv(df: pd.DataFrame, base_path: Path, logger: logging.Logger) -> None:
    df.to_parquet(base_path.with_suffix(".parquet"), index=False)
    df.to_csv(base_path.with_suffix(".csv"), index=False)
    logger.info("Saved %s.{parquet,csv} rows=%d cols=%d", base_path, len(df), len(df.columns))




def drop_sensitive_name_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    drop_cols: list[str] = []

    for col in out.columns.astype(str):
        if (
            col in {"first_name", "last_name", "frist_name"}
            or col.endswith("__first_name")
            or col.endswith("__last_name")
            or col.endswith("__frist_name")
        ):
            drop_cols.append(col)

    if drop_cols:
        out = out.drop(columns=drop_cols)

    return out, sorted(drop_cols)


def print_script_overview(script_name: str, description: str) -> None:
    print("\n" + "=" * 84)
    print(f"{script_name} | {description}")
    print("=" * 84)


def print_step(step_number: int, message: str) -> None:
    print(f"\n  Step {step_number:02d} -> {message}")

def profile_dataframe(df: pd.DataFrame, name: str) -> pd.DataFrame:
    records = []
    n = len(df)

    for col in df.columns:
        s = df[col]
        missing = s.isna().mean() * 100
        nunique = s.nunique(dropna=True)
        top = s.dropna().astype(str).value_counts().head(5).to_dict()

        records.append(
            {
                "dataset": name,
                "column": col,
                "dtype": str(s.dtype),
                "n_rows": n,
                "missing_pct": round(missing, 2),
                "cardinality": int(nunique),
                "is_constant": bool(nunique <= 1),
                "top_values": json.dumps(top, ensure_ascii=False),
            }
        )

    return pd.DataFrame(records)


def print_kv(title: str, kv: dict[str, object]) -> None:
    print(f"\n=== {title} ===")
    for k, v in kv.items():
        print(f"- {k}: {v}")


def required_columns_check(df: pd.DataFrame, cols: Iterable[str], logger: logging.Logger, dataset: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        logger.warning("%s missing required columns: %s", dataset, missing)


def resolve_canonical_column(df: pd.DataFrame, canonical_name: str) -> str:
    if canonical_name in df.columns:
        return canonical_name

    score_map: dict[str, int] = {}
    for col in df.columns.astype(str):
        if col.endswith(f"__{canonical_name}"):
            score_map[col] = max(score_map.get(col, 0), 90)
        if col.endswith(f"_{canonical_name}"):
            score_map[col] = max(score_map.get(col, 0), 80)
        if col.endswith(canonical_name):
            score_map[col] = max(score_map.get(col, 0), 70)
        if canonical_name in col:
            score_map[col] = max(score_map.get(col, 0), 10)

    if not score_map:
        raise KeyError(
            f"Could not identify canonical column '{canonical_name}'. "
            f"Available columns: {list(df.columns)}"
        )

    best_score = max(score_map.values())
    best = [c for c, s in score_map.items() if s == best_score]
    # Deterministic tie-breaker: shortest name first (usually closest to canonical), then alphabetical.
    return sorted(best, key=lambda c: (len(c), c))[0]
