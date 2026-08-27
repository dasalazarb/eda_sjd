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
    "semantic_mapping_status",
    "lab_date",
    "result_raw",
    "result_numeric",
    "result_numeric_exact",
    "result_operator",
    "result_numeric_bound",
    "result_text",
    "unit",
    "reference_range_raw",
    "reference_low",
    "reference_high",
    "reference_operator",
    "reference_bound",
    "reference_range_parse_status",
    "observation_comment",
    "observation_note",
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

# Clinically useful non-core analytes. Pair-specific entries take precedence over
# cluster entries where the same observation label has a different assay meaning.
PAIR_SEMANTICS = {
    ("ESR", "ESR (Blood)"): ("esr", "dynamic_inflammatory", "exploratory"),
    (
        "CRP, High Sensitivity, Comprehensive",
        "C-Reactive Protein, High Sensitivity (Blood)",
    ): ("crp_high_sensitivity", "dynamic_inflammatory", "exploratory"),
    ("Mineral Panel", "Albumin (Blood)"): (
        "albumin",
        "dynamic_renal_metabolic",
        "supporting",
    ),
    ("Protein, Total", "Protein Total (Blood)"): (
        "protein_total",
        "dynamic_metabolic",
        "context",
    ),
    ("Immunofixation Electrophoresis, Serum", "Protein Total (Blood)"): (
        "protein_total",
        "dynamic_immunologic",
        "supporting",
    ),
    ("Anti-Endomysial IgA Antibody", "Endomysial IgA Ab, serum (Blood)"): (
        "anti_endomysial_iga",
        "stable_autoimmune",
        "context",
    ),
    ("Angiotensin Converting Enzyme", "Angiotensin Conv.Enzyme (Blood)"): (
        "angiotensin_converting_enzyme",
        "dynamic_diagnostic_context",
        "exploratory",
    ),
    ("Thyroid Stimulating Hormone", "TSH (Blood)"): (
        "tsh",
        "chronic_endocrine",
        "context",
    ),
    ("Vitamin D, 25 Hydroxy, Total", "Vitamin D 25-Hydroxy Total (Blood)"): (
        "vitamin_d_25oh_total",
        "chronic_nutritional",
        "context",
    ),
}

CLUSTER_SEMANTICS.update(
    {
        "ALT (Blood)": ("alt", "dynamic_hepatic", "supporting"),
        "AST (Blood)": ("ast", "dynamic_hepatic", "supporting"),
        "Alkaline Phosphatase (Blood)": (
            "alkaline_phosphatase",
            "dynamic_hepatic",
            "supporting",
        ),
        "Bilirubin Direct (Blood)": (
            "bilirubin_direct",
            "dynamic_hepatic",
            "supporting",
        ),
        "Bilirubin Total (Blood)": ("bilirubin_total", "dynamic_hepatic", "supporting"),
        "Calcium (Blood)": ("calcium", "dynamic_renal_metabolic", "supporting"),
        "Magnesium (Blood)": ("magnesium", "dynamic_renal_metabolic", "supporting"),
        "Phosphorus (Blood)": ("phosphorus", "dynamic_renal_metabolic", "supporting"),
        "Hemoglobin A1C (Blood)": ("hemoglobin_a1c", "chronic_metabolic", "context"),
        "Est. Avg. Glucose (Blood)": (
            "estimated_average_glucose",
            "chronic_metabolic",
            "supporting",
        ),
        "Cholesterol (Blood)": ("total_cholesterol", "chronic_metabolic", "context"),
        "HDL Cholesterol (Blood)": ("hdl_cholesterol", "chronic_metabolic", "context"),
        "LDL Cholesterol Calculated (Blood)": (
            "ldl_cholesterol_calculated",
            "chronic_metabolic",
            "context",
        ),
        "LDL Cholesterol Direct (Blood)": (
            "ldl_cholesterol_direct",
            "chronic_metabolic",
            "context",
        ),
        "Triglycerides (Blood)": ("triglycerides", "chronic_metabolic", "context"),
        "Creatine Kinase (Blood)": ("creatine_kinase", "dynamic_muscle", "supporting"),
        "Lactate Dehydrogenase (LDH) (Blood)": (
            "lactate_dehydrogenase",
            "dynamic_hematologic",
            "context",
        ),
        "Amylase (Blood)": ("amylase", "dynamic_organ_specific", "exploratory"),
        "Uric Acid (Blood)": ("uric_acid", "dynamic_metabolic", "context"),
        "Ferritin (Blood)": ("ferritin", "chronic_nutritional", "context"),
        "Iron % Saturation (Blood)": (
            "iron_saturation",
            "chronic_nutritional",
            "context",
        ),
        "Iron (Blood)": ("serum_iron", "chronic_nutritional", "context"),
        "Transferrin (Blood)": ("transferrin", "chronic_nutritional", "context"),
        "IgG Total (Blood)": ("igg_total", "dynamic_immunologic", "supporting"),
        **{
            f"IgG Subclass {n} (Blood)": (
                f"igg_subclass_{n}",
                "dynamic_immunologic",
                "supporting",
            )
            for n in range(1, 5)
        },
        "Albumin [Ord, Pro or IFE] (Blood)": (
            "ife_albumin",
            "dynamic_immunologic",
            "supporting",
        ),
        "Alpha 1 Globulin (Blood)": (
            "alpha1_globulin",
            "dynamic_immunologic",
            "supporting",
        ),
        "Alpha 2 Globulin (Blood)": (
            "alpha2_globulin",
            "dynamic_immunologic",
            "supporting",
        ),
        "Beta 1 (Blood)": ("beta1_globulin", "dynamic_immunologic", "supporting"),
        "Beta 2 (Blood)": ("beta2_globulin", "dynamic_immunologic", "supporting"),
        "Beta Globulin (Blood)": ("beta_globulin", "dynamic_immunologic", "supporting"),
        "Gamma Globulin (Blood)": (
            "gamma_globulin",
            "dynamic_immunologic",
            "supporting",
        ),
        "Immunofixation Electrophoresis Serum (Blood)": (
            "immunofixation_serum_interpretation",
            "dynamic_immunologic",
            "supporting",
        ),
        "Monoclonal Band (M-Spike) Serum Electrophoresis (Blood)": (
            "monoclonal_band_m_spike",
            "dynamic_immunologic",
            "supporting",
        ),
        "Protein Total [Ord:  Electroph., Serum] (Blood)": (
            "protein_total_electrophoresis",
            "dynamic_immunologic",
            "supporting",
        ),
        "Anti-CCP Ab (Blood)": ("anti_ccp", "stable_autoimmune", "context"),
        "DNA Double-Stranded Ab (Blood)": (
            "anti_dsdna",
            "stable_autoimmune",
            "context",
        ),
        "DNA Double-Stranded Ab, IgG (Blood)": (
            "anti_dsdna_igg",
            "stable_autoimmune",
            "context",
        ),
        "Anti-Cardiolipin IgG Quant (Blood)": (
            "anticardiolipin_igg",
            "stable_autoimmune",
            "context",
        ),
        "Anti-Cardiolipin IgM Quant (Blood)": (
            "anticardiolipin_igm",
            "stable_autoimmune",
            "context",
        ),
        "Anti-Thyroglobulin (Blood)": (
            "anti_thyroglobulin",
            "stable_autoimmune",
            "context",
        ),
        "Anti-Thyroglobulin Index (Blood)": (
            "anti_thyroglobulin_index",
            "stable_autoimmune",
            "context",
        ),
        "Anti-Thyroid Peroxidase (Blood)": ("anti_tpo", "stable_autoimmune", "context"),
        "Jo 1 Ab, IgG (Blood)": ("anti_jo1", "stable_autoimmune", "context"),
        "RNP Ab, IgG (Blood)": ("anti_rnp", "stable_autoimmune", "context"),
        "Scl 70 Ab, IgG (Blood)": ("anti_scl70", "stable_autoimmune", "context"),
        "Sm Ab, IgG (Blood)": ("anti_sm", "stable_autoimmune", "context"),
        "HLA-A* (Blood)": ("hla_a", "fixed_genetic", "exploratory"),
        "HLA-B* (Blood)": ("hla_b", "fixed_genetic", "exploratory"),
        "HLA-Cw* (Blood)": ("hla_c", "fixed_genetic", "exploratory"),
        "HLA-DQB1* / DQ* (Blood)": ("hla_dqb1", "fixed_genetic", "exploratory"),
        "HLA-DRB1* (Blood)": ("hla_drb1", "fixed_genetic", "exploratory"),
        "HLA-DRB_* (Blood)": ("hla_drb", "fixed_genetic", "exploratory"),
        "HLA-DPB* (Sequenced Based) (Blood)": (
            "hla_dpb",
            "fixed_genetic",
            "exploratory",
        ),
        "HBc (HepB core) Ab (Blood)": (
            "hepatitis_b_core_antibody",
            "infection_screening",
            "context",
        ),
        "HBs (HepB surface) Ab (Blood)": (
            "hepatitis_b_surface_antibody",
            "infection_screening",
            "context",
        ),
        "HBs (HepB surface) Ab Not Reported (Blood)": (
            "hepatitis_b_surface_antibody_not_reported",
            "infection_screening",
            "context",
        ),
        "HBs (HepB surface) Ag (Blood)": (
            "hepatitis_b_surface_antigen",
            "infection_screening",
            "context",
        ),
        "HCV (HepC) Ab (Blood)": (
            "hepatitis_c_antibody",
            "infection_screening",
            "context",
        ),
        "HTLV-I/II Ab (Blood)": ("htlv_1_2_antibody", "infection_screening", "context"),
        "Coronavirus SARS-CoV-2 Anti-N Antibody (Blood)": (
            "sars_cov2_anti_n",
            "infection_screening",
            "context",
        ),
        "VZV Ab IgG Numeric (Blood)": (
            "varicella_zoster_igg",
            "infection_screening",
            "context",
        ),
        "PT (Blood)": ("prothrombin_time", "procedural", "context"),
        "PT INR (Blood)": ("inr", "procedural", "context"),
        "PTT (Blood)": ("partial_thromboplastin_time", "procedural", "context"),
        "PTT Comment (Blood)": ("ptt_comment", "procedural", "currently_unused"),
        "Beta HCG Pregnancy (Blood)": (
            "beta_hcg_pregnancy",
            "pregnancy_time_specific",
            "context",
        ),
        "Pregnancy Test (Urine)": (
            "urine_pregnancy_test",
            "pregnancy_time_specific",
            "context",
        ),
        "Protein Comment (Urine)": (
            "pregnancy_test_protein_comment",
            "pregnancy_time_specific",
            "currently_unused",
        ),
    }
)

for cluster, analyte in {
    "Mycobacterium Tuberculosis Antigen 1 minus Nil, GoldPlus (Blood)": "quantiferon_tb1_minus_nil",
    "Mycobacterium Tuberculosis Antigen 2 minus Nil, GoldPlus (Blood)": "quantiferon_tb2_minus_nil",
    "Mycobacterium Tuberculosis Mitogen minus Nil, GoldPlus (Blood)": "quantiferon_mitogen_minus_nil",
    "Mycobacterium Tuberculosis Nil Value, GoldPlus (Blood)": "quantiferon_nil",
    "Mycobacterium Tuberculosis QuantiFERON, GoldPlus (Blood)": "quantiferon_final_interpretation",
}.items():
    CLUSTER_SEMANTICS[cluster] = (analyte, "infection_screening", "context")

for cluster, analyte in {
    "Basophil % (Blood)": "basophil_percent",
    "Basophil Abs (Blood)": "basophil_absolute",
    "Eosinophil % (Blood)": "eosinophil_percent",
    "Eosinophil Abs (Blood)": "eosinophil_absolute",
    "Immature Granulocytes % (Blood)": "immature_granulocyte_percent",
    "Immature Granulocytes Abs (Blood)": "immature_granulocyte_absolute",
    "Lymphocytes % (Blood)": "lymphocyte_percent",
    "MCH (Blood)": "mch",
    "MCHC (Blood)": "mchc",
    "MCV (Blood)": "mcv",
    "MPV (Blood)": "mpv",
    "Monocytes % (Blood)": "monocyte_percent",
    "Monocytes Abs (Blood)": "monocyte_absolute",
    "Neutrophil % (Blood)": "neutrophil_percent",
    "Nucleated RBC % (Blood)": "nucleated_rbc_percent",
    "Nucleated RBC Abs (Blood)": "nucleated_rbc_absolute",
    "RDW (Blood)": "rdw",
}.items():
    CLUSTER_SEMANTICS[cluster] = (analyte, "dynamic_hematologic", "supporting")

for cluster, analyte in {
    "Appearance (Urinalysis)": "urine_appearance",
    "Bacteria (Urinalysis)": "urine_bacteria",
    "Bilirubin (Urinalysis)": "urine_bilirubin",
    "Color (Urinalysis)": "urine_color",
    "Glucose, Qualitative (Urinalysis)": "urine_glucose_qualitative",
    "Ketones (Urinalysis)": "urine_ketones",
    "Leukocyte Esterase (Urinalysis)": "urine_leukocyte_esterase",
    "Microscopic Exam (Urinalysis)": "urine_microscopic_exam",
    "Nitrite (Urinalysis)": "urine_nitrite",
    "pH (Urinalysis)": "urine_ph",
    "Squamous Cells (Urinalysis)": "urine_squamous_cells",
    "Urobilinogen (Urinalysis)": "urine_urobilinogen",
}.items():
    CLUSTER_SEMANTICS[cluster] = (analyte, "dynamic_renal_urinary", "supporting")


def _semantic_assignment(order_name: object, cluster_name: object):
    """Return the exact pair assignment before an approved cluster fallback."""
    pair = (order_name, cluster_name)
    return (
        SEMANTIC_OVERRIDES.get(pair)
        or PAIR_SEMANTICS.get(pair)
        or CLUSTER_SEMANTICS.get(cluster_name)
    )


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
    "unit": ["Unit", "Units", "Result Unit", "Unit of Measure"],
    "reference_range_raw": [
        "Normal Range",
        "Reference Range",
        "Reference Interval",
    ],
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
    "observation_comment": ["Observation Comment"],
    "observation_note": ["Observation Note"],
}
PRESERVED_METADATA = {
    "specimen_datetime": ["Specimen Date Time", "Collected Date Time"],
    "specimen_type": ["Specimen Type", "Specimen"],
    "order_identifier": ["Order ID", "Order Identifier", "Order Number"],
    "assay": ["Assay", "Method"],
    "observation_identifier": ["Observation ID", "Result ID"],
}
FIELD_RESOLUTION_TARGETS = [
    "patient_id",
    "order_name",
    "cluster_name",
    "lab_date",
    "result_raw",
    "unit",
    "reference_range_raw",
    "reference_low",
    "reference_high",
    "reported_interpretation",
    "result_status",
    "observation_comment",
    "observation_note",
    "specimen_datetime",
    "specimen_type",
    "order_identifier",
    "assay",
    "observation_identifier",
]


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


@dataclass(frozen=True)
class ParsedNumericResult:
    """Structured numeric evidence from one raw laboratory result."""

    exact: float | None
    operator: str | None
    bound: float | None


_NUMBER_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_CENSORED_RESULT_PATTERN = re.compile(rf"^\s*(<=|>=|<|>|=)\s*({_NUMBER_PATTERN})\s*$")
_EXACT_RESULT_PATTERN = re.compile(rf"^\s*({_NUMBER_PATTERN})\s*$")
_LOW_HIGH_RANGE_PATTERN = re.compile(
    rf"^\s*({_NUMBER_PATTERN})\s*-\s*({_NUMBER_PATTERN})\s*$"
)
_ONE_SIDED_RANGE_PATTERN = re.compile(
    rf"^\s*(<=|>=|<|>)\s*({_NUMBER_PATTERN})(?:\s+\([^)]*\))?\s*$",
    re.IGNORECASE,
)
_TITER_REFERENCE_PATTERN = re.compile(r"^\s*(?:<=|>=|<|>)?\s*\d+\s*:\s*\d+")
_QUALITATIVE_REFERENCE_TOKENS = {
    "negative",
    "positive",
    "nonreactive",
    "reactive",
    "not detected",
    "not detectable",
}


def parse_numeric_result(value: object) -> ParsedNumericResult:
    """Parse exact or explicitly censored numeric laboratory evidence.

    Parameters
    ----------
    value : object
        Raw observation value. Mixed qualitative/numeric text is not parsed.

    Returns
    -------
    ParsedNumericResult
        Exact values and censored bounds are kept in mutually exclusive fields.
    """
    if pd.isna(value):
        return ParsedNumericResult(None, None, None)
    text = str(value)
    censored = _CENSORED_RESULT_PATTERN.fullmatch(text)
    if censored:
        return ParsedNumericResult(None, censored.group(1), float(censored.group(2)))
    exact = _EXACT_RESULT_PATTERN.fullmatch(text)
    if exact:
        return ParsedNumericResult(float(exact.group(1)), None, None)
    return ParsedNumericResult(None, None, None)


@dataclass(frozen=True)
class ParsedReferenceRange:
    """Structural fields parsed from one contemporaneous reference range."""

    low: float | None
    high: float | None
    operator: str | None
    bound: float | None
    status: str


def parse_reference_range(value: object) -> ParsedReferenceRange:
    """Parse reference-range structure without making a clinical interpretation."""
    empty = (None, None, None, None)
    if pd.isna(value) or not str(value).strip():
        return ParsedReferenceRange(*empty, "missing")
    text = str(value).strip()
    normalized = re.sub(r"\s+", " ", text).casefold()
    if _TITER_REFERENCE_PATTERN.match(text):
        return ParsedReferenceRange(*empty, "titer_reference")
    bilateral = _LOW_HIGH_RANGE_PATTERN.fullmatch(text)
    if bilateral:
        low, high = float(bilateral.group(1)), float(bilateral.group(2))
        if low <= high:
            return ParsedReferenceRange(low, high, None, None, "parsed_low_high")
        return ParsedReferenceRange(*empty, "ambiguous")
    one_sided = _ONE_SIDED_RANGE_PATTERN.fullmatch(text)
    if one_sided:
        return ParsedReferenceRange(
            None,
            None,
            one_sided.group(1),
            float(one_sided.group(2)),
            "parsed_one_sided",
        )
    if normalized in _QUALITATIVE_REFERENCE_TOKENS:
        return ParsedReferenceRange(*empty, "qualitative_reference")
    if not re.search(r"\d", text):
        return ParsedReferenceRange(*empty, "non_numeric_text")
    return ParsedReferenceRange(*empty, "ambiguous")


def classify_reference_range(value: object) -> str:
    """Return the closed-vocabulary structural parse status for a raw range."""
    return parse_reference_range(value).status


def build_source_schema_qc(raw: pd.DataFrame, source_file: str) -> pd.DataFrame:
    """Summarize the structure and completeness of one raw BTRIS extract.

    Parameters
    ----------
    raw : pd.DataFrame
        Raw extract before field normalization.
    source_file : str
        Source-file provenance label.

    Returns
    -------
    pd.DataFrame
        One structural summary row per raw column, without source values.
    """
    n_rows = len(raw)
    return pd.DataFrame(
        [
            {
                "source_file": source_file,
                "raw_column_name": str(column),
                "dtype": str(raw[column].dtype),
                "n_rows": n_rows,
                "n_nonmissing": int(raw[column].notna().sum()),
                "pct_nonmissing": (
                    float(raw[column].notna().mean() * 100) if n_rows else 0.0
                ),
            }
            for column in raw.columns
        ]
    )


def build_field_resolution_qc(
    raw: pd.DataFrame,
    source_file: str,
    column_aliases: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Audit exact alias resolution for required normalized fields.

    Parameters
    ----------
    raw : pd.DataFrame
        Raw extract before field normalization.
    source_file : str
        Source-file provenance label.
    column_aliases : dict[str, list[str]], optional
        Alias mapping to audit. Defaults to the production mappings.

    Returns
    -------
    pd.DataFrame
        One resolution result per required target field.
    """
    aliases = {**COLUMN_ALIASES, **PRESERVED_METADATA}
    if column_aliases is not None:
        aliases.update(column_aliases)
    rows = []
    for target in FIELD_RESOLUTION_TARGETS:
        source = _resolve(raw.columns, aliases.get(target, []))
        n_nonmissing = int(raw[source].notna().sum()) if source else 0
        rows.append(
            {
                "source_file": source_file,
                "target_field": target,
                "resolved_raw_column": source if source else pd.NA,
                "resolved": source is not None,
                "n_rows": len(raw),
                "n_nonmissing_resolved_column": n_nonmissing,
                "pct_nonmissing_resolved_column": (
                    n_nonmissing / len(raw) * 100 if len(raw) else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


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
        override = _semantic_assignment(*pair)
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
    parsed_results = out["result_raw"].map(parse_numeric_result)
    out["result_numeric_exact"] = pd.to_numeric(
        parsed_results.map(lambda result: result.exact), errors="coerce"
    )
    out["result_operator"] = parsed_results.map(lambda result: result.operator).astype(
        "string"
    )
    out["result_numeric_bound"] = pd.to_numeric(
        parsed_results.map(lambda result: result.bound), errors="coerce"
    )
    # Compatibility contract: censored bounds are never exact measurements.
    out["result_numeric"] = out["result_numeric_exact"]
    out["result_text"] = out["result_raw"].where(out["result_numeric_exact"].isna())
    parsed_ranges = out["reference_range_raw"].map(parse_reference_range)
    out["reference_low"] = pd.to_numeric(
        parsed_ranges.map(lambda parsed: parsed.low), errors="coerce"
    )
    out["reference_high"] = pd.to_numeric(
        parsed_ranges.map(lambda parsed: parsed.high), errors="coerce"
    )
    out["reference_operator"] = parsed_ranges.map(
        lambda parsed: parsed.operator
    ).astype("string")
    out["reference_bound"] = pd.to_numeric(
        parsed_ranges.map(lambda parsed: parsed.bound), errors="coerce"
    )
    out["reference_range_parse_status"] = parsed_ranges.map(
        lambda parsed: parsed.status
    ).astype("string")
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
        override = _semantic_assignment(
            row["order_name_canonical"], row["cluster_name_canonical"]
        )
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
    # Remaining rows are members of the reviewed 256-pair inventory rather than
    # accidental discoveries. Classify their provenance explicitly even when no
    # current analysis consumes the textual/rare descriptive result.
    reviewed_unused = matched & out["analytic_role"].eq("currently_unused")
    hematology_unused = reviewed_unused & out["order_name_canonical"].str.contains(
        "CBC|Diff", case=False, na=False
    )
    urinary_unused = reviewed_unused & out["cluster_name_canonical"].str.contains(
        "Urine|Urinalysis", case=False, na=False
    )
    out.loc[reviewed_unused, "lab_family"] = "documentary_support"
    out.loc[hematology_unused, "lab_family"] = "dynamic_hematologic"
    out.loc[urinary_unused, "lab_family"] = "dynamic_renal_urinary"
    active_roles = {"core", "supporting", "context", "exploratory"}
    out["semantic_mapping_status"] = "deliberately_unused"
    for role in active_roles:
        out.loc[matched & out["analytic_role"].eq(role), "semantic_mapping_status"] = (
            f"mapped_{role}"
        )
    out.loc[~matched, "semantic_mapping_status"] = "unexpected_unmapped"
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
        "semantic_mapping_status",
    ]
    qc = (
        labs.groupby(columns, dropna=False)
        .agg(n_rows=("patient_id", "size"), n_patients=("patient_id", "nunique"))
        .reset_index()
    )
    used = (
        qc["canonical_analyte"].notna()
        & qc["lab_family"].ne("other")
        & qc["analytic_role"].isin({"core", "supporting", "context", "exploratory"})
    )
    deliberately_unused = qc["semantic_mapping_status"].eq("deliberately_unused") & qc[
        "lab_family"
    ].ne("other")
    qc["semantic_mapping_complete"] = used | deliberately_unused
    return qc


def build_semantic_status_summary(labs: pd.DataFrame) -> pd.DataFrame:
    """Summarize observed pairs, rows, and patients by semantic intent."""
    pair_columns = ["order_name_canonical", "cluster_name_canonical"]
    return (
        labs.groupby("semantic_mapping_status", dropna=False)
        .agg(
            n_pairs=(pair_columns[0], lambda values: 0),
            n_rows=("patient_id", "size"),
            n_patients=("patient_id", "nunique"),
        )
        .drop(columns="n_pairs")
        .join(
            labs.groupby("semantic_mapping_status", dropna=False)[pair_columns]
            .apply(lambda group: len(group.drop_duplicates()))
            .rename("n_pairs")
        )[["n_pairs", "n_rows", "n_patients"]]
        .reset_index()
    )


def build_semantic_unresolved_qc(labs: pd.DataFrame) -> pd.DataFrame:
    """Return only observed pairs whose semantic meaning remains unresolved."""
    completeness = build_semantic_mapping_qc(labs)[
        ["order_name_canonical", "cluster_name_canonical", "semantic_mapping_complete"]
    ]
    annotated = labs.merge(
        completeness,
        on=["order_name_canonical", "cluster_name_canonical"],
        how="left",
        validate="many_to_one",
    )
    columns = [
        "order_name_original",
        "order_name_canonical",
        "cluster_name_original",
        "cluster_name_canonical",
        "mapping_status",
        "canonical_analyte",
        "lab_family",
        "analytic_role",
    ]
    return (
        annotated.loc[
            ~annotated["semantic_mapping_complete"],
            columns + ["patient_id", "lab_date"],
        ]
        .groupby(columns, dropna=False)
        .agg(
            n_rows=("patient_id", "size"),
            n_patients=("patient_id", "nunique"),
            min_date=("lab_date", "min"),
            max_date=("lab_date", "max"),
        )
        .reset_index()
    )


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
INTERPRETATION_EVIDENCE_ANALYTES = [
    "anti_ro_ssa",
    "anti_la_ssb",
    "ana_status",
    "ana_hep2_status",
    "rheumatoid_factor",
    "cryoglobulins",
    "complement_c4",
    "wbc",
    "cryoglobulins_ife",
]
QUALITATIVE_QC_ANALYTES = INTERPRETATION_EVIDENCE_ANALYTES[:6]


def normalize_qualitative_token(value: object) -> object:
    """Normalize result text for aggregation without clinically classifying it.

    Parameters
    ----------
    value : object
        Raw textual result value.

    Returns
    -------
    object
        Case-folded, edge-stripped text or ``pd.NA`` for a missing value.
    """
    if pd.isna(value):
        return pd.NA
    return str(value).strip().casefold()


def normalize_normal_range(value: object) -> object:
    """Normalize whitespace and case for range-token QC without losing syntax."""
    if pd.isna(value):
        return pd.NA
    normalized = re.sub(r"\s+", " ", str(value).strip()).casefold()
    return normalized if normalized else pd.NA


def build_core_interpretation_evidence_qc(labs: pd.DataFrame) -> pd.DataFrame:
    """Summarize available interpretation evidence for core analytes.

    Parameters
    ----------
    labs : pd.DataFrame
        Normalized laboratory records with exact semantic annotations.

    Returns
    -------
    pd.DataFrame
        Completeness counts, percentages, and an informational evidence warning.
    """
    rows = []
    for analyte in INTERPRETATION_EVIDENCE_ANALYTES:
        found = labs[labs["canonical_analyte"].eq(analyte)]
        n_rows = len(found)

        def count(column: str) -> int:
            return int(found[column].notna().sum())

        def percent(n_nonmissing: int) -> float:
            return n_nonmissing / n_rows * 100 if n_rows else 0.0

        counts = {
            "result_raw": count("result_raw"),
            "result_numeric": count("result_numeric"),
            "result_operator": count("result_operator"),
            "result_numeric_bound": count("result_numeric_bound"),
            "result_text": count("result_text"),
            "unit": count("unit"),
            "reported_interpretation": count("reported_interpretation"),
            "reference_range_raw": count("reference_range_raw"),
            "reference_low": count("reference_low"),
            "reference_high": count("reference_high"),
        }
        rows.append(
            {
                "canonical_analyte": analyte,
                "n_rows": n_rows,
                "n_patients": found["patient_id"].nunique(),
                **{f"n_{name}_nonmissing": value for name, value in counts.items()},
                **{
                    f"pct_{name}_nonmissing": percent(counts[name])
                    for name in [
                        "result_numeric",
                        "result_operator",
                        "result_numeric_bound",
                        "result_text",
                        "unit",
                        "reported_interpretation",
                        "reference_range_raw",
                        "reference_low",
                        "reference_high",
                    ]
                },
                "core_reference_evidence_missing": bool(
                    counts["result_numeric"] > 0
                    and counts["reference_low"] == 0
                    and counts["reference_high"] == 0
                    and counts["reported_interpretation"] == 0
                ),
            }
        )
    return pd.DataFrame(rows)


def build_core_qualitative_token_qc(labs: pd.DataFrame) -> pd.DataFrame:
    """Aggregate normalized textual result tokens without patient-level details.

    Parameters
    ----------
    labs : pd.DataFrame
        Normalized laboratory records with exact semantic annotations.

    Returns
    -------
    pd.DataFrame
        Counts and within-analyte percentages for each normalized text token.
    """
    selected = labs[labs["canonical_analyte"].isin(QUALITATIVE_QC_ANALYTES)].copy()
    selected["normalized_result_text"] = selected["result_text"].map(
        normalize_qualitative_token
    )
    selected = selected.dropna(subset=["normalized_result_text"])
    counts = (
        selected.groupby(["canonical_analyte", "normalized_result_text"], dropna=False)
        .size()
        .rename("n_rows")
        .reset_index()
    )
    if counts.empty:
        return pd.DataFrame(
            columns=[
                "canonical_analyte",
                "normalized_result_text",
                "n_rows",
                "pct_within_analyte",
            ]
        )
    counts["pct_within_analyte"] = (
        counts["n_rows"]
        / counts.groupby("canonical_analyte")["n_rows"].transform("sum")
        * 100
    )
    return counts


def build_core_normal_range_token_qc(labs: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw normal-range formats without patient or result details."""
    selected = labs[
        labs["canonical_analyte"].isin(INTERPRETATION_EVIDENCE_ANALYTES[:8])
    ].copy()
    selected["normalized_normal_range"] = selected["reference_range_raw"].map(
        normalize_normal_range
    )
    selected = selected.dropna(subset=["normalized_normal_range"])
    counts = (
        selected.groupby(["canonical_analyte", "normalized_normal_range"], dropna=False)
        .size()
        .rename("n_rows")
        .reset_index()
    )
    columns = [
        "canonical_analyte",
        "normalized_normal_range",
        "n_rows",
        "pct_within_analyte",
    ]
    if counts.empty:
        return pd.DataFrame(columns=columns)
    counts["pct_within_analyte"] = (
        counts["n_rows"]
        / counts.groupby("canonical_analyte")["n_rows"].transform("sum")
        * 100
    )
    return counts[columns]


def build_core_reference_range_parse_qc(labs: pd.DataFrame) -> pd.DataFrame:
    """Count closed-vocabulary reference parse outcomes for core analytes."""
    selected = labs[
        labs["canonical_analyte"].isin(INTERPRETATION_EVIDENCE_ANALYTES[:8])
    ]
    counts = (
        selected.groupby(
            ["canonical_analyte", "reference_range_parse_status"], dropna=False
        )
        .size()
        .rename("n_rows")
        .reset_index()
    )
    counts["pct_within_analyte"] = (
        counts["n_rows"]
        / counts.groupby("canonical_analyte")["n_rows"].transform("sum")
        * 100
    )
    return counts


def build_core_result_range_relation_qc(labs: pd.DataFrame) -> pd.DataFrame:
    """Aggregate result and range forms without exposing patient identifiers."""
    selected = labs[
        labs["canonical_analyte"].isin(INTERPRETATION_EVIDENCE_ANALYTES[:8])
    ].copy()
    operator_forms = {"<": "lt", "<=": "lte", ">": "gt", ">=": "gte"}

    def result_form(row: pd.Series) -> str:
        operator = row.get("result_operator")
        if pd.notna(operator) and operator in operator_forms:
            return f"censored_{operator_forms[operator]}"
        if pd.notna(row.get("result_numeric")):
            return "exact_numeric"
        if pd.notna(row.get("result_text")) and str(row["result_text"]).strip():
            return "qualitative"
        return "missing"

    def range_form(row: pd.Series) -> str:
        status = row.get("reference_range_parse_status")
        if status == "parsed_low_high":
            return "low_high"
        if status == "parsed_one_sided":
            suffix = operator_forms.get(row.get("reference_operator"))
            return f"one_sided_{suffix}" if suffix else "missing"
        return {
            "titer_reference": "titer",
            "qualitative_reference": "qualitative",
        }.get(status, "missing")

    selected["result_form"] = selected.apply(result_form, axis=1)
    selected["range_form"] = selected.apply(range_form, axis=1)
    return (
        selected.groupby(
            ["canonical_analyte", "result_form", "range_form"], dropna=False
        )
        .size()
        .rename("n_rows")
        .reset_index()
    )


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


def _read_lab_files(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames = []
    schema_frames = []
    resolution_frames = []
    for path in sorted(root.rglob("Lab*.csv")):
        protocol = next(
            (part.upper() for part in path.parts if part.upper() in {"11D", "15D"}), ""
        )
        raw = pd.read_csv(path, low_memory=False)
        source_file = str(path)
        schema_frames.append(build_source_schema_qc(raw, source_file))
        resolution_frames.append(build_field_resolution_qc(raw, source_file))
        frames.append(normalize_lab_records(raw, source_file, protocol))
    if not frames:
        raise FileNotFoundError(f"No filtered Lab*.csv files found below {root}")
    return (
        pd.concat(frames, ignore_index=True),
        pd.concat(schema_frames, ignore_index=True),
        pd.concat(resolution_frames, ignore_index=True),
    )


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
    labs_input, schema_qc, field_resolution_qc = _read_lab_files(config.btris_root)
    annotated = annotate_expected_pairs(labs_input, reference)
    labs, ambiguous = attach_clinical_context(annotated, spine)
    coverage = build_cluster_coverage(labs, reference)
    alias_qc = build_alias_qc(labs)
    semantic_qc = build_semantic_mapping_qc(labs)
    semantic_status_summary = build_semantic_status_summary(labs)
    semantic_unresolved = build_semantic_unresolved_qc(labs)
    core_qc = build_core_mapping_qc(labs)
    interpretation_evidence_qc = build_core_interpretation_evidence_qc(labs)
    qualitative_token_qc = build_core_qualitative_token_qc(labs)
    normal_range_token_qc = build_core_normal_range_token_qc(labs)
    reference_parse_qc = build_core_reference_range_parse_qc(labs)
    result_range_qc = build_core_result_range_relation_qc(labs)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.report_dir.mkdir(parents=True, exist_ok=True)
    labs[
        [column for column in OUTPUT_COLUMNS if column in labs.columns]
        + [column for column in PRESERVED_METADATA if column in labs.columns]
    ].to_parquet(config.output_path, index=False)
    coverage.to_csv(config.report_dir / "20_lab_cluster_coverage.csv", index=False)
    schema_qc.to_csv(config.report_dir / "20_btris_source_schema_qc.csv", index=False)
    field_resolution_qc.to_csv(
        config.report_dir / "20_btris_field_resolution_qc.csv", index=False
    )
    interpretation_evidence_qc.to_csv(
        config.report_dir / "20_core_interpretation_evidence_qc.csv", index=False
    )
    qualitative_token_qc.to_csv(
        config.report_dir / "20_core_qualitative_token_qc.csv", index=False
    )
    normal_range_token_qc.to_csv(
        config.report_dir / "20_core_normal_range_token_qc.csv", index=False
    )
    reference_parse_qc.to_csv(
        config.report_dir / "20_core_reference_range_parse_qc.csv", index=False
    )
    result_range_qc.to_csv(
        config.report_dir / "20_core_result_range_relation_qc.csv", index=False
    )
    alias_qc.to_csv(config.report_dir / "20_lab_alias_mapping_qc.csv", index=False)
    semantic_qc.to_csv(
        config.report_dir / "20_lab_semantic_mapping_qc.csv", index=False
    )
    semantic_status_summary.to_csv(
        config.report_dir / "20_lab_semantic_mapping_status_summary.csv", index=False
    )
    semantic_unresolved.to_csv(
        config.report_dir / "20_lab_semantic_unresolved_qc.csv", index=False
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
