from __future__ import annotations

import json
import logging
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


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^0-9a-zA-Z]+", "_", regex=True)
        .str.strip("_")
    )
    out = df.copy()
    out.columns = cols
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

    if "visit_date" in out.columns:
        out["visit_date"] = pd.to_datetime(out["visit_date"], errors="coerce")

    if "time_24_hour" in out.columns:
        t = pd.to_datetime(out["time_24_hour"], errors="coerce")
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
