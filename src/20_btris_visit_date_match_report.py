"""Build the preserved longitudinal BTRIS laboratory record table.

Clinical episodes and the clinical baseline are authoritative inputs.  Laboratory
dates are used only to annotate records with temporal distances and an optional
episode association; they never create or redefine clinical visits.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from common import ANALYTIC_DIR, INTERMEDIATE_DIR, REPORTS_DIR, setup_logger

EXPECTED_PAIR_COUNT = 256
ANCHOR_MATCH_DAYS = 10
OUTPUT_COLUMNS = [
    "patient_id",
    "has_clinical_baseline",
    "clinical_baseline_episode_id",
    "clinical_baseline_date",
    "days_from_clinical_baseline",
    "order_name_original",
    "order_name_canonical",
    "cluster_name_original",
    "cluster_name_canonical",
    "mapping_status",
    "canonical_analyte",
    "lab_family",
    "analytic_role",
    "lab_date",
    "result_raw",
    "result_numeric",
    "result_text",
    "unit",
    "reference_low",
    "reference_high",
    "reported_interpretation",
    "result_status",
    "result_valid_for_analysis",
    "invalid_result_reason",
    "matched_clinical_episode_id",
    "matched_clinical_anchor_date",
    "days_from_clinical_anchor",
    "episode_match_method",
    "episode_match_ambiguous",
    "unexpected_cluster_name",
    "source_protocol",
    "source_file",
]

ORDER_NAME_ALIASES = {
    "AMYLASE": "Amylase",
    "ANA HEp-2 Substrate, IgG": "ANA Hep-2 Substrate, IgG",
    "Anti-CCP Ab": "Anti-CCP AB",
    "CRYOGLOBULINS": "Cryoglobulins",
    "RHEUMATOID FACTOR": "Rheumatoid Factor",
    "URIC ACID": "Uric Acid",
}

# Semantic assignments are deliberately exact pair mappings.  The authoritative
# 256-row pair inventory is supplied separately so additions never require fuzzy
# matching or silently change analyte meaning.
SEMANTIC_OVERRIDES = {
    ("Anti-Nuclear Antibody", "Antinuclear Antibody (ANA) (Blood)"): (
        "ana_status",
        "stable_autoimmune",
        "core",
    ),
    (
        "ANA Hep-2 Substrate, IgG",
        "Antinuclear Antibody (ANA) HEp-2 Substrate (Blood)",
    ): ("ana_hep2_status", "stable_autoimmune", "core"),
    (
        "ANA Hep-2 Substrate, IgG",
        "Antinuclear Antibody (ANA) HEp-2 Substrate Titer (Blood)",
    ): ("ana_hep2_titer", "stable_autoimmune", "core"),
    (
        "ANA Hep-2 Substrate, IgG",
        "Antinuclear Antibody (ANA) HEp-2 Substrate Pattern (Blood)",
    ): ("ana_hep2_pattern", "stable_autoimmune", "core"),
    (
        "ANA Hep-2 Substrate, IgG",
        "Antinuclear Antibody (ANA) HEp-2 Cytoplasmic Pattern (Blood)",
    ): ("ana_hep2_cytoplasmic_pattern", "stable_autoimmune", "supporting"),
    ("ENA Evaluation", "SS-A/Ro Ab, IgG (Blood)"): (
        "anti_ro_ssa",
        "stable_autoimmune",
        "core",
    ),
    ("ENA Evaluation", "SS-B/La Ab, IgG (Blood)"): (
        "anti_la_ssb",
        "stable_autoimmune",
        "core",
    ),
    ("C3/C4", "Complement C4 (Blood)"): (
        "complement_c4",
        "dynamic_immunologic",
        "core",
    ),
    ("C3/C4", "Complement C3 (Blood)"): (
        "complement_c3",
        "dynamic_immunologic",
        "supporting",
    ),
    ("CBC + Diff", "WBC (Blood)"): ("wbc", "dynamic_hematologic", "core"),
    ("Rheumatoid Factor", "Rheumatoid Factor (Blood)"): (
        "rheumatoid_factor",
        "stable_autoimmune",
        "core",
    ),
    ("Cryoglobulins", "Cryoglobulins (Blood)"): (
        "cryoglobulins",
        "dynamic_immunologic",
        "core",
    ),
    ("Ro52 & Ro60 Antibodies, IgG", "SS-Ro52 Ab, IgG (Blood)"): (
        "anti_ro52",
        "stable_autoimmune",
        "supporting",
    ),
    ("Ro52 & Ro60 Antibodies, IgG", "SS-Ro60 Ab, IgG (Blood)"): (
        "anti_ro60",
        "stable_autoimmune",
        "supporting",
    ),
}

# Exact cluster semantics may be shared by several approved order/source pairs.  They
# are applied only after an exact reference-pair match; they never make a pair valid.
CLUSTER_SEMANTICS = {
    "Neutrophil Abs (Blood)": ("anc", "dynamic_hematologic", "supporting"),
    "Lymphocytes Abs (Blood)": (
        "lymphocyte_count",
        "dynamic_hematologic",
        "supporting",
    ),
    "Hemoglobin (Blood)": ("hemoglobin", "dynamic_hematologic", "supporting"),
    "Platelet Count (Blood)": ("platelet_count", "dynamic_hematologic", "supporting"),
    "Hematocrit (Blood)": ("hematocrit", "dynamic_hematologic", "supporting"),
    "RBC (Blood)": ("rbc", "dynamic_hematologic", "supporting"),
    "Creatinine (Blood)": ("creatinine", "dynamic_renal_metabolic", "supporting"),
    "eGFR CKD-EPI 2021 Creatinine-Based (Blood)": (
        "egfr_ckd_epi_2021",
        "dynamic_renal_metabolic",
        "supporting",
    ),
    "eGFR (African-American) (Blood)": (
        "egfr_legacy_african_american",
        "dynamic_renal_metabolic",
        "context",
    ),
    "eGFR (non-African-American) (Blood)": (
        "egfr_legacy_non_african_american",
        "dynamic_renal_metabolic",
        "context",
    ),
    "BUN (Blood)": ("bun", "dynamic_renal_metabolic", "supporting"),
    "Glucose (Blood)": ("glucose", "dynamic_renal_metabolic", "context"),
    "Protein, Qualitative (Urinalysis)": (
        "urine_protein_qualitative",
        "dynamic_renal_urinary",
        "supporting",
    ),
    "RBC (Urinalysis)": ("urine_rbc", "dynamic_renal_urinary", "supporting"),
    "WBC (Urinalysis)": ("urine_wbc", "dynamic_renal_urinary", "supporting"),
    "Hemoglobin (Urinalysis)": (
        "urine_hemoglobin",
        "dynamic_renal_urinary",
        "supporting",
    ),
    "Specific Gravity (Urinalysis)": (
        "urine_specific_gravity",
        "dynamic_renal_urinary",
        "supporting",
    ),
    "WBC Casts (Urinalysis)": (
        "urine_wbc_casts",
        "dynamic_renal_urinary",
        "supporting",
    ),
    "Granular Casts (Urinalysis)": (
        "urine_granular_casts",
        "dynamic_renal_urinary",
        "supporting",
    ),
    "IgG (Blood)": ("igg", "dynamic_immunologic", "supporting"),
    "IgA (Blood)": ("iga", "dynamic_immunologic", "supporting"),
    "IgM (Blood)": ("igm", "dynamic_immunologic", "supporting"),
    "Cryoglobulins, IFE (Blood)": (
        "cryoglobulins_ife",
        "dynamic_immunologic",
        "supporting",
    ),
}

COLUMN_ALIASES = {
    "patient_id": ["patient_id", "ids__patient_record_number", "MRN", "Patient ID"],
    "order_name": ["Order Name", "order_name"],
    "cluster_name": ["Cluster Name", "cluster_name", "Observation Name"],
    "lab_date": [
        "Collected Date Time",
        "Collected Date",
        "Specimen Date Time",
        "Result Date",
        "Reported Date Time",
        "lab_date",
    ],
    "result_raw": [
        "Observation Value",
        "Result Value",
        "Result",
        "Value",
        "result_raw",
    ],
    "unit": ["Unit", "Units", "Result Unit"],
    "reference_low": ["Reference Low", "Low Reference Range", "Reference Range Low"],
    "reference_high": [
        "Reference High",
        "High Reference Range",
        "Reference Range High",
    ],
    "reported_interpretation": [
        "Interpretation",
        "Result Interpretation",
        "Abnormal Flag",
    ],
    "result_status": ["Result Status", "Status"],
}
PRESERVED_METADATA = {
    "specimen_datetime": ["Specimen Date Time", "Collected Date Time"],
    "specimen_type": ["Specimen Type", "Specimen"],
    "order_identifier": ["Order ID", "Order Identifier", "Order Number"],
    "assay": ["Assay", "Method"],
    "observation_identifier": ["Observation ID", "Result ID"],
}


@dataclass(frozen=True)
class LabConfig:
    """Filesystem inputs and traceable outputs for step 20."""

    spine_path: Path
    btris_root: Path
    reference_path: Path
    output_path: Path
    report_dir: Path


def _norm(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).replace("\ufeff", " ").strip()).casefold()


def _resolve(columns: Iterable[object], candidates: Iterable[str]) -> str | None:
    by_normalized = {_norm(column): str(column) for column in columns}
    return next(
        (
            by_normalized[_norm(name)]
            for name in candidates
            if _norm(name) in by_normalized
        ),
        None,
    )


def _normalize_patient_id(series: pd.Series) -> pd.Series:
    normalized = (
        series.astype("string")
        .str.strip()
        .str.replace(r"[-/\\\s]", "", regex=True)
        .str.replace(r"^0+", "", regex=True)
    )
    return normalized.mask(normalized.isin(["", "nan", "None"]))


def load_reference(path: Path, require_complete: bool = True) -> pd.DataFrame:
    """Load and validate the authoritative exact Order Name/Cluster Name inventory."""
    if not path.exists():
        raise FileNotFoundError(f"Authoritative laboratory reference not found: {path}")
    reference = (
        pd.read_excel(path)
        if path.suffix.lower() in {".xlsx", ".xls"}
        else pd.read_csv(path)
    )
    rename = {}
    for target in [
        "order_name",
        "cluster_name",
        "canonical_analyte",
        "lab_family",
        "analytic_role",
    ]:
        source = _resolve(reference.columns, [target, target.replace("_", " ")])
        if source:
            rename[source] = target
    reference = reference.rename(columns=rename)
    required = {"order_name", "cluster_name"}
    if not required.issubset(reference.columns):
        raise KeyError(
            f"Reference is missing columns: {sorted(required - set(reference.columns))}"
        )
    reference = reference.dropna(subset=["order_name", "cluster_name"]).copy()
    if reference.duplicated(["order_name", "cluster_name"]).any():
        raise ValueError(
            "Reference contains duplicate exact Order Name/Cluster Name pairs"
        )
    if require_complete and len(reference) != EXPECTED_PAIR_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_PAIR_COUNT} exact pairs; found {len(reference)}"
        )
    for column, default in [
        ("canonical_analyte", pd.NA),
        ("lab_family", "other"),
        ("analytic_role", "currently_unused"),
    ]:
        if column not in reference:
            reference[column] = default
    for index, row in reference.iterrows():
        pair = (row["order_name"], row["cluster_name"])
        override = SEMANTIC_OVERRIDES.get(pair) or CLUSTER_SEMANTICS.get(
            row["cluster_name"]
        )
        if override:
            reference.loc[
                index, ["canonical_analyte", "lab_family", "analytic_role"]
            ] = override
    return reference


def normalize_lab_records(
    raw: pd.DataFrame, source_file: str = "", source_protocol: str = ""
) -> pd.DataFrame:
    """Normalize a BTRIS Lab extract without discarding raw observations."""
    out = pd.DataFrame(index=raw.index)
    for target, aliases in COLUMN_ALIASES.items():
        source = _resolve(raw.columns, aliases)
        out[target] = raw[source] if source else pd.NA
    missing = [
        name
        for name in ("patient_id", "order_name", "lab_date")
        if out[name].isna().all()
    ]
    if missing:
        raise KeyError(f"Lab extract lacks required fields: {missing}")
    out["patient_id"] = _normalize_patient_id(out["patient_id"])
    out["lab_date"] = pd.to_datetime(out["lab_date"], errors="coerce").dt.normalize()
    out["result_raw"] = out["result_raw"].astype("string")
    out["result_numeric"] = pd.to_numeric(out["result_raw"], errors="coerce")
    out["result_text"] = out["result_raw"].where(out["result_numeric"].isna())
    invalid_text = (
        out["result_raw"]
        .fillna("")
        .str.cat(out["result_status"].astype("string").fillna(""), sep=" ")
        .str.casefold()
    )
    reasons = pd.Series(pd.NA, index=out.index, dtype="string")
    patterns = [
        ("administrative duplicate", "administrative_duplicate"),
        ("quantity insufficient|qns", "quantity_insufficient"),
        ("invalid specimen", "invalid_specimen"),
        ("not performed", "not_performed"),
        ("cancel", "cancelled"),
    ]
    for pattern, reason in patterns:
        reasons = reasons.mask(
            reasons.isna() & invalid_text.str.contains(pattern, regex=True), reason
        )
    out["invalid_result_reason"] = reasons
    out["result_valid_for_analysis"] = reasons.isna()
    out["source_protocol"] = source_protocol
    out["source_file"] = source_file
    for target, aliases in PRESERVED_METADATA.items():
        source = _resolve(raw.columns, aliases)
        if source:
            out[target] = raw[source]
    return out


def annotate_expected_pairs(
    labs: pd.DataFrame, reference: pd.DataFrame
) -> pd.DataFrame:
    """Attach exact/explicit-alias metadata while preserving source names."""
    out = labs.copy()
    out["order_name_original"] = out["order_name"]
    out["cluster_name_original"] = out["cluster_name"]
    out["order_name_canonical"] = out["order_name_original"].replace(ORDER_NAME_ALIASES)
    out["cluster_name_canonical"] = out["cluster_name_original"]
    metadata = reference[
        [
            "order_name",
            "cluster_name",
            "canonical_analyte",
            "lab_family",
            "analytic_role",
        ]
    ].rename(
        columns={
            "order_name": "order_name_canonical",
            "cluster_name": "cluster_name_canonical",
        }
    )
    for index, row in metadata.iterrows():
        override = SEMANTIC_OVERRIDES.get(
            (row["order_name_canonical"], row["cluster_name_canonical"])
        ) or CLUSTER_SEMANTICS.get(row["cluster_name_canonical"])
        if override:
            metadata.loc[
                index, ["canonical_analyte", "lab_family", "analytic_role"]
            ] = override
    out = out.merge(
        metadata,
        how="left",
        on=["order_name_canonical", "cluster_name_canonical"],
        indicator=True,
    )
    matched = out.pop("_merge").eq("both")
    alias_used = out["order_name_original"].ne(out["order_name_canonical"])
    out["mapping_status"] = "unexpected_unmapped"
    out.loc[matched & ~alias_used, "mapping_status"] = "exact_reference"
    out.loc[matched & alias_used, "mapping_status"] = "explicit_alias"
    out["unexpected_cluster_name"] = ~matched
    out.loc[~matched, ["canonical_analyte", "lab_family", "analytic_role"]] = pd.NA
    return out


def attach_clinical_context(
    labs: pd.DataFrame, spine: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach authoritative baseline fields and optional episode associations."""
    episodes = spine.copy()
    patient_source = _resolve(
        episodes.columns, ["patient_id", "ids__patient_record_number"]
    )
    if patient_source is None:
        raise KeyError("Clinical episode spine lacks patient_id")
    episodes["patient_id"] = _normalize_patient_id(episodes[patient_source])
    date_columns = [
        "episode_start_date",
        "clinical_anchor_date",
        "episode_end_date",
        "clinical_baseline_date",
    ]
    required = {"clinical_episode_id", "clinical_baseline_episode_id", *date_columns}
    if not required.issubset(episodes.columns):
        raise KeyError(
            f"Clinical episode spine missing: {sorted(required - set(episodes.columns))}"
        )
    for column in date_columns:
        episodes[column] = pd.to_datetime(
            episodes[column], errors="coerce"
        ).dt.normalize()
    baseline = episodes[
        ["patient_id", "clinical_baseline_episode_id", "clinical_baseline_date"]
    ].drop_duplicates()
    if baseline.duplicated("patient_id").any():
        raise ValueError("A patient has conflicting authoritative clinical baselines")
    out = labs.merge(baseline, how="left", on="patient_id", validate="many_to_one")
    out["has_clinical_baseline"] = (
        out["clinical_baseline_episode_id"].notna()
        & out["clinical_baseline_date"].notna()
    )
    out["days_from_clinical_baseline"] = (
        out["lab_date"] - out["clinical_baseline_date"]
    ).dt.days
    ambiguous_rows: list[dict[str, object]] = []
    matches: list[dict[str, object]] = []
    by_patient = {
        key: group for key, group in episodes.groupby("patient_id", sort=False)
    }
    for _, lab in out.iterrows():
        candidates = by_patient.get(lab["patient_id"], episodes.iloc[0:0]).copy()
        inside = candidates[
            (candidates["episode_start_date"] <= lab["lab_date"])
            & (lab["lab_date"] <= candidates["episode_end_date"])
        ].copy()
        minimum = pd.NA

        if not inside.empty:
            # Being inside the authoritative episode window has priority over
            # anchor proximity. A single containing episode remains a valid
            # match even when its optional anchor is missing.
            inside["_distance"] = (
                inside["clinical_anchor_date"] - lab["lab_date"]
            ).dt.days.abs()
            if len(inside) == 1:
                winners = inside.copy()
            else:
                valid_anchor = inside[inside["_distance"].notna()].copy()
                if valid_anchor.empty:
                    # With overlapping windows and no usable anchors there is
                    # no objective basis for selecting one episode.
                    winners = inside.copy()
                else:
                    minimum = valid_anchor["_distance"].min()
                    winners = valid_anchor[valid_anchor["_distance"].eq(minimum)].copy()
            eligible = True
        else:
            # Outside all windows, only a valid anchor within ±10 days can
            # provide the secondary episode association.
            pool = candidates.copy()
            pool["_distance"] = (
                pool["clinical_anchor_date"] - lab["lab_date"]
            ).dt.days.abs()
            valid_anchor = pool[pool["_distance"].notna()].copy()
            if valid_anchor.empty:
                winners = valid_anchor
                eligible = False
            else:
                minimum = valid_anchor["_distance"].min()
                winners = valid_anchor[valid_anchor["_distance"].eq(minimum)].copy()
                eligible = minimum <= ANCHOR_MATCH_DAYS

        ambiguous = eligible and len(winners) > 1
        if ambiguous:
            record = {
                "patient_id": lab["patient_id"],
                "lab_date": lab["lab_date"],
                "order_name": lab["order_name"],
                "cluster_name": lab["cluster_name"],
            }
            for number, (_, candidate) in enumerate(winners.head(2).iterrows(), 1):
                record[f"candidate_episode_id_{number}"] = candidate[
                    "clinical_episode_id"
                ]
                record[f"candidate_anchor_date_{number}"] = candidate[
                    "clinical_anchor_date"
                ]
                record[f"candidate_distance_{number}"] = candidate["_distance"]
            ambiguous_rows.append(record)
        winner = (
            winners.iloc[0]
            if eligible and not ambiguous and not winners.empty
            else None
        )
        method = (
            "ambiguous"
            if ambiguous
            else (
                "inside_episode_window"
                if winner is not None and not inside.empty
                else (
                    "closest_anchor_le10d" if winner is not None else "no_episode_match"
                )
            )
        )
        matches.append(
            {
                "matched_clinical_episode_id": (
                    winner["clinical_episode_id"] if winner is not None else pd.NA
                ),
                "matched_clinical_anchor_date": (
                    winner["clinical_anchor_date"] if winner is not None else pd.NaT
                ),
                "days_from_clinical_anchor": (
                    (lab["lab_date"] - winner["clinical_anchor_date"]).days
                    if winner is not None and pd.notna(winner["clinical_anchor_date"])
                    else pd.NA
                ),
                "episode_match_method": method,
                "episode_match_ambiguous": ambiguous,
            }
        )
    return pd.concat(
        [out.reset_index(drop=True), pd.DataFrame(matches)], axis=1
    ), pd.DataFrame(ambiguous_rows)


def build_cluster_coverage(labs: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Build expected-pair coverage plus observed unexpected-pair rows."""
    order_column = (
        "order_name_canonical" if "order_name_canonical" in labs else "order_name"
    )
    cluster_column = (
        "cluster_name_canonical" if "cluster_name_canonical" in labs else "cluster_name"
    )
    stats = (
        labs.groupby([order_column, cluster_column], dropna=False)
        .agg(
            n_rows=("patient_id", "size"),
            n_patients=("patient_id", "nunique"),
            min_date=("lab_date", "min"),
            max_date=("lab_date", "max"),
        )
        .reset_index()
        .rename(columns={order_column: "order_name", cluster_column: "cluster_name"})
    )
    expected = reference.rename(
        columns={
            "order_name": "order_name_expected",
            "cluster_name": "cluster_name_expected",
        }
    )
    expected = expected.merge(
        stats,
        how="left",
        left_on=["order_name_expected", "cluster_name_expected"],
        right_on=["order_name", "cluster_name"],
    )
    expected["found_in_input"] = expected["n_rows"].fillna(0).gt(0)
    expected["expected_cluster_not_found"] = ~expected["found_in_input"]
    unexpected = stats.merge(
        reference[["order_name", "cluster_name"]],
        how="left",
        on=["order_name", "cluster_name"],
        indicator=True,
    )
    unexpected = unexpected[unexpected["_merge"].eq("left_only")].copy()
    unexpected["order_name_expected"] = unexpected["order_name"]
    unexpected["cluster_name_expected"] = unexpected["cluster_name"]
    unexpected["found_in_input"] = True
    unexpected["expected_cluster_not_found"] = False
    return pd.concat([expected, unexpected], ignore_index=True, sort=False)


def build_alias_qc(labs: pd.DataFrame) -> pd.DataFrame:
    """Summarize exact alias decisions with original-name provenance."""
    columns = [
        "order_name_original",
        "order_name_canonical",
        "cluster_name_original",
        "cluster_name_canonical",
        "mapping_status",
    ]
    return (
        labs.groupby(columns, dropna=False)
        .agg(
            n_rows=("patient_id", "size"),
            n_patients=("patient_id", "nunique"),
            min_date=("lab_date", "min"),
            max_date=("lab_date", "max"),
        )
        .reset_index()
    )


def build_semantic_mapping_qc(labs: pd.DataFrame) -> pd.DataFrame:
    """Summarize semantic completeness for each observed canonical pair."""
    columns = [
        "order_name_canonical",
        "cluster_name_canonical",
        "canonical_analyte",
        "lab_family",
        "analytic_role",
    ]
    qc = (
        labs.groupby(columns, dropna=False)
        .agg(n_rows=("patient_id", "size"), n_patients=("patient_id", "nunique"))
        .reset_index()
    )
    qc["semantic_mapping_complete"] = (
        qc[["canonical_analyte", "lab_family", "analytic_role"]].notna().all(axis=1)
    )
    return qc


CORE_ANALYTES = [
    "anti_ro_ssa",
    "anti_la_ssb",
    "ana_status",
    "ana_hep2_status",
    "ana_hep2_titer",
    "ana_hep2_pattern",
    "rheumatoid_factor",
    "cryoglobulins",
    "complement_c4",
    "wbc",
]


def build_core_mapping_qc(labs: pd.DataFrame) -> pd.DataFrame:
    """Report whether every required core analyte is present and complete."""
    rows = []
    for analyte in CORE_ANALYTES:
        found = labs[labs["canonical_analyte"].eq(analyte)]
        rows.append(
            {
                "canonical_analyte": analyte,
                "found_in_input": not found.empty,
                "n_rows": len(found),
                "n_patients": found["patient_id"].nunique(),
                "min_date": found["lab_date"].min(),
                "max_date": found["lab_date"].max(),
                "n_unexpected_alias_rows": int(
                    found["mapping_status"].eq("unexpected_unmapped").sum()
                ),
                "semantic_mapping_complete": bool(
                    found.empty
                    or found[["canonical_analyte", "lab_family", "analytic_role"]]
                    .notna()
                    .all(axis=None)
                ),
            }
        )
    return pd.DataFrame(rows)


def build_baseline_context_qc(labs: pd.DataFrame) -> pd.DataFrame:
    """Summarize baseline timing without representing absent baselines as zero."""
    qc = (
        labs.groupby(
            [
                "patient_id",
                "has_clinical_baseline",
                "clinical_baseline_episode_id",
                "clinical_baseline_date",
            ],
            dropna=False,
        )
        .agg(
            n_total_lab_records=("patient_id", "size"),
            n_prebaseline_lab_records=(
                "days_from_clinical_baseline",
                lambda x: x.lt(0).sum(),
            ),
            n_same_day_lab_records=(
                "days_from_clinical_baseline",
                lambda x: x.eq(0).sum(),
            ),
            n_postbaseline_lab_records=(
                "days_from_clinical_baseline",
                lambda x: x.gt(0).sum(),
            ),
            min_days_from_baseline=("days_from_clinical_baseline", "min"),
            max_days_from_baseline=("days_from_clinical_baseline", "max"),
        )
        .reset_index()
    )
    timing = [
        "n_prebaseline_lab_records",
        "n_same_day_lab_records",
        "n_postbaseline_lab_records",
        "min_days_from_baseline",
        "max_days_from_baseline",
    ]
    qc.loc[~qc["has_clinical_baseline"], timing] = pd.NA
    return qc


def _read_lab_files(root: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(root.rglob("Lab*.csv")):
        protocol = next(
            (part.upper() for part in path.parts if part.upper() in {"11D", "15D"}), ""
        )
        frames.append(
            normalize_lab_records(
                pd.read_csv(path, low_memory=False), str(path), protocol
            )
        )
    if not frames:
        raise FileNotFoundError(f"No filtered Lab*.csv files found below {root}")
    return pd.concat(frames, ignore_index=True)


def _parse_args() -> LabConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spine-path",
        type=Path,
        default=ANALYTIC_DIR / "clinical_episode_spine_sjd.parquet",
    )
    parser.add_argument("--btris-root", type=Path, default=INTERMEDIATE_DIR / "BTRIS")
    parser.add_argument(
        "--reference-path",
        type=Path,
        default=INTERMEDIATE_DIR / "BTRIS" / "20_lab_order_cluster_reference.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ANALYTIC_DIR / "BTRIS" / "20_btris_lab_records_long.parquet",
    )
    parser.add_argument("--report-dir", type=Path, default=REPORTS_DIR / "btris_labs")
    args = parser.parse_args()
    return LabConfig(
        args.spine_path,
        args.btris_root,
        args.reference_path,
        args.output_path,
        args.report_dir,
    )


def main() -> None:
    """Run the laboratory preservation and QC pipeline."""
    config = _parse_args()
    logger = setup_logger("20_btris_lab_records_long")
    reference = load_reference(config.reference_path)
    spine = pd.read_parquet(config.spine_path)
    labs_input = _read_lab_files(config.btris_root)
    annotated = annotate_expected_pairs(labs_input, reference)
    labs, ambiguous = attach_clinical_context(annotated, spine)
    coverage = build_cluster_coverage(labs, reference)
    alias_qc = build_alias_qc(labs)
    semantic_qc = build_semantic_mapping_qc(labs)
    core_qc = build_core_mapping_qc(labs)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.report_dir.mkdir(parents=True, exist_ok=True)
    labs[
        [column for column in OUTPUT_COLUMNS if column in labs.columns]
        + [column for column in PRESERVED_METADATA if column in labs.columns]
    ].to_parquet(config.output_path, index=False)
    coverage.to_csv(config.report_dir / "20_lab_cluster_coverage.csv", index=False)
    alias_qc.to_csv(config.report_dir / "20_lab_alias_mapping_qc.csv", index=False)
    semantic_qc.to_csv(
        config.report_dir / "20_lab_semantic_mapping_qc.csv", index=False
    )
    core_qc.to_csv(config.report_dir / "20_core_lab_mapping_qc.csv", index=False)
    ambiguous.to_csv(config.report_dir / "20_lab_episode_ambiguous.csv", index=False)
    invalid = labs[~labs["result_valid_for_analysis"]]
    invalid[
        [
            "patient_id",
            "lab_date",
            "order_name_original",
            "cluster_name_original",
            "result_raw",
            "result_status",
            "invalid_result_reason",
            "source_file",
        ]
    ].to_csv(config.report_dir / "20_lab_invalid_results.csv", index=False)
    counts = (
        labs["episode_match_method"]
        .value_counts()
        .reindex(
            [
                "inside_episode_window",
                "closest_anchor_le10d",
                "no_episode_match",
                "ambiguous",
            ],
            fill_value=0,
        )
    )
    pd.DataFrame(
        {
            "episode_match_method": counts.index,
            "n": counts.values,
            "percent": counts.values / len(labs) * 100 if len(labs) else 0,
        }
    ).to_csv(config.report_dir / "20_lab_episode_match_summary.csv", index=False)
    baseline_qc = build_baseline_context_qc(labs)
    baseline_qc.to_csv(
        config.report_dir / "20_lab_baseline_context_qc.csv", index=False
    )
    expected_rows = coverage.iloc[: len(reference)]
    summary = pd.DataFrame(
        [
            {
                "n_patients_spine": _normalize_patient_id(
                    spine[
                        _resolve(
                            spine.columns, ["patient_id", "ids__patient_record_number"]
                        )
                    ]
                ).nunique(),
                "n_patients_with_labs": labs["patient_id"].nunique(),
                "n_patients_with_clinical_baseline": labs.loc[
                    labs["has_clinical_baseline"], "patient_id"
                ].nunique(),
                "n_patients_without_clinical_baseline": labs.loc[
                    ~labs["has_clinical_baseline"], "patient_id"
                ].nunique(),
                "n_lab_rows_input": len(labs_input),
                "n_lab_rows_valid": int(labs["result_valid_for_analysis"].sum()),
                "n_lab_rows_invalid": int((~labs["result_valid_for_analysis"]).sum()),
                "n_expected_order_cluster_pairs_found": int(
                    expected_rows["found_in_input"].sum()
                ),
                "n_expected_order_cluster_pairs_missing": int(
                    expected_rows["expected_cluster_not_found"].sum()
                ),
                "n_unexpected_order_cluster_pairs": len(coverage) - len(reference),
                "n_rows_exact_reference": int(
                    labs["mapping_status"].eq("exact_reference").sum()
                ),
                "n_rows_explicit_alias": int(
                    labs["mapping_status"].eq("explicit_alias").sum()
                ),
                "n_rows_unexpected_unmapped": int(
                    labs["mapping_status"].eq("unexpected_unmapped").sum()
                ),
                "n_semantic_pairs_complete": int(
                    semantic_qc["semantic_mapping_complete"].sum()
                ),
                "n_semantic_pairs_incomplete": int(
                    (~semantic_qc["semantic_mapping_complete"]).sum()
                ),
                "min_lab_date": labs["lab_date"].min(),
                "max_lab_date": labs["lab_date"].max(),
            }
        ]
    )
    summary.to_csv(config.report_dir / "20_lab_record_summary.csv", index=False)
    logger.info(
        "Saved %d preserved laboratory records to %s", len(labs), config.output_path
    )


if __name__ == "__main__":
    main()
