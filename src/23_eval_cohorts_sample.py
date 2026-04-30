"""
=============================================================================
SJÖGREN'S NATURAL HISTORY PROJECT
Cohort Viability Analysis — C0 through C12
=============================================================================

INPUT:  visits_long_collapsed_by_interval_codebook_type_recode.csv
        (Codebook format: rows = visits; non-null cells = recorded value;
         CSV row 0 contains type descriptors and is removed.)

OUTPUT: cohort_viability_results.csv   — cohort summary table
        cohort_essdai_distribution.csv — ESSDAI visit distribution
        cohort_esspri_distribution.csv — ESSPRI visit distribution

BIOWULF USAGE:
    module load python
    python cohort_viability_analysis.py --input <path_to_csv>

Dependencies: pandas, numpy
=============================================================================
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np

from common import ANALYTIC_DIR, REPORTS_DIR

# ---------------------------------------------------------------------------
# CONFIG: key codebook columns
# Update if data cleaning changes column names.
# ---------------------------------------------------------------------------

COL_PATIENT    = "ids__patient_record_number"
COL_INTERVAL   = "ids__interval_name"
COL_VISIT_DATE = "ids__visit_date"

# Disease classification
COL_SJD_CLASS  = "visit_summary_form__sjogrens_class"
COL_IE_PRIM_SEC = "inclusion/exclusion_criteria__ic_sjogrens_prim_sec"
COL_IE_HV       = "inclusion/exclusion_criteria__ic_hv"

# ESSDAI / ESSPRI
COL_ESSDAI     = "essdai__essdai_total_score"
COL_ESSPRI_DRY = "esspri_questionnaire__dryness"       # ESSPRI completeness proxy
COL_ESSPRI_FAT = "esspri_questionnaire__fatigue"

# Labs
COL_LABS       = "cris_lab_form__labs_done"

# Salivary flow (glandular)
COL_SAL_FLOW   = "salivary_flow_form__flow_whole_unstim"

# Eye exam
COL_EYE        = "eye_examination__eye_exam_done"

# Comorbidities
COL_COMORBID   = "rheumatological_comorbidities__comorbid_none"
COL_PMH_CARDIO = "past_medical_history__cardio_none"

# Autoimmune overlap (para C8)
COLS_OVERLAP   = [
    "rheumatological_comorbidities__integ_raynds",
    "rheumatological_comorbidities__ra",
    "rheumatological_comorbidities__sle1",
    "rheumatological_comorbidities__systemic_sclerosis",
    "rheumatological_comorbidities__polymyositis",
    "rheumatological_comorbidities__dermatomyositis",
    "rheumatological_comorbidities__mixed_connective_tissue_disease",
    "rheumatological_comorbidities__antiphospholipid_syndrome",
    "rheumatological_comorbidities__cryoglobulinemia",
    "rheumatological_comorbidities__fibromyalgia1",
    "rheumatological_comorbidities__osteoporosis1",
    "rheumatological_comorbidities__osteopenia",
    "rheumatological_comorbidities__osteoarthritis",
    "rheumatological_comorbidities__sarcoidosis",
]

# Medications (for C9)
COLS_MEDS      = [f"medications__rx_{i}_name" for i in range(1, 11)]

# Individual ESSDAI domains (for C5 — EGM proxy)
COLS_ESSDAI_DOMAINS = [
    "essdai__constitutional", "essdai__hema_lphdenopthy",
    "essdai__gland_swell",    "essdai__articular_domain",
    "essdai__cutaneous",      "essdai__pulmonary",
    "essdai__renal",          "essdai__muscular_domain",
    "essdai__neuro_peripheral","essdai__cns",
    "essdai__hematologic",    "essdai__biological_domain",
]

# Inclusion/exclusion criteria (for C0)
COLS_IE = []   # dynamically detected

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    """Load CSV and drop the type-descriptor row (CSV row 0)."""
    df = pd.read_csv(path)
    # The first data row contains type codes ("string","numeric","boolean","date")
    # when ids__subject_number == "numeric" → this is the descriptor row → remove
    mask_type_row = df[COL_PATIENT].isna() | (df.iloc[:, 2].astype(str) == "numeric")
    # Detect descriptor row: ids__subject_number == "numeric"
    if "ids__subject_number" in df.columns:
        type_rows = df[df["ids__subject_number"].astype(str).str.lower() == "numeric"].index
        df = df.drop(index=type_rows).reset_index(drop=True)
    return df


def pts_with_data(df: pd.DataFrame, col: str, min_visits: int = 1) -> set:
    """Patients with >= min_visits non-null observations in `col`."""
    if col not in df.columns:
        return set()
    tmp = df[df[col].notna()].groupby(COL_PATIENT).size()
    return set(tmp[tmp >= min_visits].index)


def pts_with_values(df: pd.DataFrame, col: str, values: set) -> set:
    """Patients with at least one visit where `col` is in `values`."""
    if col not in df.columns:
        return set()
    numeric_col = pd.to_numeric(df[col], errors="coerce")
    valid_values = {v for v in values if pd.notna(v)}
    mask = numeric_col.isin(valid_values)
    return set(df.loc[mask, COL_PATIENT].unique())


def pts_any_col(df: pd.DataFrame, cols: list) -> set:
    """Patients with at least ONE non-null cell in any column from `cols`."""
    present = [c for c in cols if c in df.columns]
    if not present:
        return set()
    mask = df[present].notna().any(axis=1)
    return set(df[mask][COL_PATIENT].unique())


def pts_all_cols_same_visit(df: pd.DataFrame, cols: list) -> set:
    """Patients with at least ONE visit where ALL columns are non-null."""
    present = [c for c in cols if c in df.columns]
    if not present:
        return set()
    mask = df[present].notna().all(axis=1)
    return set(df[mask][COL_PATIENT].unique())


def viability_flag(n: int) -> str:
    if n >= 50:
        return "VIABLE"
    elif n >= 20:
        return "MARGINAL"
    else:
        return "INSUFFICIENT"


# ---------------------------------------------------------------------------
# MAIN ANALYSIS
# ---------------------------------------------------------------------------

def run_analysis(df: pd.DataFrame, c0_df: pd.DataFrame | None = None) -> dict:
    c0_source = c0_df if c0_df is not None and COL_PATIENT in c0_df.columns else df
    total_pts   = c0_source[COL_PATIENT].nunique()
    total_visits = len(df)

    # Detect IE columns
    ie_cols = [c for c in df.columns
               if "inclusion" in c.lower() or "exclusion" in c.lower()]

    results = {}

    # -----------------------------------------------------------------------
    # C0: Source screening cohort
    # -----------------------------------------------------------------------
    c0 = set(c0_source[COL_PATIENT].unique())
    results["C0"] = dict(
        description="Source screening (15-D + 11-D)",
        objective="Primary Objective 1 — referral context",
        inclusion_criteria="All 15-D participants (total patients)",
        time_zero_criteria="First evaluable NIH SjD visit",
        key_variables=", ".join(ie_cols[:4]) + "..." if ie_cols else "N/A",
        n=len(c0),
        pts=c0,
    )

    # -----------------------------------------------------------------------
    # C1: Core SjD master cohort
    # -----------------------------------------------------------------------
    c1 = pts_with_values(df, COL_SJD_CLASS, {1, 2, 4})
    results["C1"] = dict(
        description="Core SjD master (1: primary + 2: secondary + 4: incomplete)",
        objective="Backbone — base for all analyses",
        inclusion_criteria="SjD classification in {1, 2, 4} (unique patients)",
        time_zero_criteria="First evaluable NIH visit; collapse 15-D and 11-D if ≤30 days",
        key_variables=COL_SJD_CLASS,
        n=len(c1),
        pts=c1,
    )

    # -----------------------------------------------------------------------
    # C2: Longitudinal ESSDAI cohort
    # -----------------------------------------------------------------------
    df_c1 = df[df[COL_PATIENT].isin(c1)]
    essdai_any = pts_with_data(df_c1, COL_ESSDAI, min_visits=1)
    c2         = pts_with_data(df_c1, COL_ESSDAI, min_visits=2)
    c2_3v      = pts_with_data(df_c1, COL_ESSDAI, min_visits=3)

    essdai_visits = df_c1[df_c1[COL_ESSDAI].notna()].groupby(COL_PATIENT).size() \
                    if COL_ESSDAI in df.columns else pd.Series(dtype=int)

    results["C2"] = dict(
        description="Longitudinal ESSDAI (≥2 measurements)",
        objective="Primary Objective 2 — disease progression",
        inclusion_criteria="Primary/secondary SjD + ≥2 evaluable ESSDAI",
        time_zero_criteria="First visit with recorded ESSDAI",
        key_variables=COL_ESSDAI,
        n=len(c2),
        pts=c2,
        n_any=len(essdai_any),
        n_3v=len(c2_3v),
        median_visits=essdai_visits.median() if len(essdai_visits) else 0,
    )

    # -----------------------------------------------------------------------
    # C3: Incident severe-risk cohort
    # -----------------------------------------------------------------------
    c3 = set()
    if COL_ESSDAI in df.columns and c2:
        c2_rows = df[df[COL_PATIENT].isin(c2)].copy()
        essdai_numeric = pd.to_numeric(c2_rows[COL_ESSDAI], errors="coerce")
        c3 = set(c2_rows.loc[essdai_numeric < 5, COL_PATIENT].unique())
    results["C3"] = dict(
        description="Incident severe-risk (baseline ESSDAI < 5)",
        objective="Primary Objective 4 — time to severe disease",
        inclusion_criteria="C2 subset with total ESSDAI < 5",
        time_zero_criteria="First visit with documented ESSDAI < 5",
        key_variables=COL_ESSDAI,
        n=len(c3),
        pts=c3,
    )

    # -----------------------------------------------------------------------
    # C4: Domain/risk-factor cohort
    # -----------------------------------------------------------------------
    c4 = None
    results["C4"] = dict(
        description="Domain/risk-factor (C2 + labs)",
        objective="Primary Objective 3 — clinical and serologic risk factors",
        inclusion_criteria="C2 + baseline covariates + linked labs",
        time_zero_criteria="First visit with ESSDAI domain data + baseline predictors",
        key_variables=f"{COL_ESSDAI}, {COL_LABS}",
        n=c4,
        pts=c4,
        note="Pendiente por integración de datos desde BTRIS.",
    )

    # -----------------------------------------------------------------------
    # C5: Glandular-EGM overlap cohort
    # -----------------------------------------------------------------------
    gland_pts = pts_with_data(df, COL_SAL_FLOW)
    eye_pts   = pts_with_data(df, COL_EYE)
    egm_cols_present = [c for c in COLS_ESSDAI_DOMAINS if c in df.columns]
    egm_pts   = pts_any_col(df, egm_cols_present)
    c5        = gland_pts & egm_pts
    results["C5"] = dict(
        description="Glandular-EGM overlap",
        objective="Primary Objective 5 — glandular / extra-glandular co-occurrence",
        inclusion_criteria="SjD with ascertainable glandular phenotype + EGM data",
        time_zero_criteria="First visit with both measurable components",
        key_variables=f"{COL_SAL_FLOW}, ESSDAI domains",
        n=len(c5),
        pts=c5,
        n_gland=len(gland_pts),
        n_eye=len(eye_pts),
        n_egm=len(egm_pts),
    )

    # -----------------------------------------------------------------------
    # C6: Comorbidity cohort
    # -----------------------------------------------------------------------
    c6 = None
    results["C6"] = dict(
        description="Comorbidities (baseline + follow-up)",
        objective="Secondary Objective 1 — comorbidity prevalence and incidence",
        inclusion_criteria="SjD with documented comorbidities",
        time_zero_criteria="First evaluable SjD visit",
        key_variables=f"{COL_COMORBID}, {COL_PMH_CARDIO}",
        n=c6,
        pts=c6,
        note="Pendiente por integración de datos desde BTRIS.",
    )

    # -----------------------------------------------------------------------
    # C7: Phenotype population cohort (Pop 1–3)
    # -----------------------------------------------------------------------
    esspri_mask = pd.Series(False, index=df.index)
    if COL_ESSPRI_DRY in df.columns:
        esspri_mask = esspri_mask | df[COL_ESSPRI_DRY].notna()
    if COL_ESSPRI_FAT in df.columns:
        esspri_mask = esspri_mask | df[COL_ESSPRI_FAT].notna()
    esspri_any = set(df.loc[esspri_mask, COL_PATIENT].unique())

    paired_df = df[df[COL_ESSDAI].notna() & esspri_mask] \
        if COL_ESSDAI in df.columns else pd.DataFrame()
    c7 = set(paired_df[COL_PATIENT].unique()) if not paired_df.empty else set()
    c7 = c7 & c1
    paired_visits = paired_df.groupby(COL_PATIENT).size() if not paired_df.empty else pd.Series(dtype=int)
    results["C7"] = dict(
        description="Phenotype Pop 1-3 (paired ESSDAI+ESSPRI)",
        objective="Secondary Objective 2 — proportions in Pop 1/2/3",
        inclusion_criteria="C1 + ≥1 visit with ESSDAI and ESSPRI (fatigue o dryness) within ±30 days",
        time_zero_criteria="First paired ESSDAI-ESSPRI assessment",
        key_variables=f"{COL_ESSDAI}, {COL_ESSPRI_FAT}, {COL_ESSPRI_DRY}",
        n=len(c7),
        pts=c7,
        n_essdai_any=len(essdai_any),
        n_esspri_any=len(esspri_any),
        median_paired=paired_visits.median() if len(paired_visits) else 0,
    )

    # -----------------------------------------------------------------------
    # C8: Secondary/overlap autoimmune subgroup
    # -----------------------------------------------------------------------
    overlap_pts = pts_any_col(df, COLS_OVERLAP)
    c8 = c1 & overlap_pts
    results["C8"] = dict(
        description="Autoimmune overlap (RA/SLE/SSc/others)",
        objective="Secondary Objective 3 — progression in SjD + overlap",
        inclusion_criteria="C1 subset with coexisting autoimmune disease",
        time_zero_criteria="Same index as parent cohort (C1)",
        key_variables=", ".join(COLS_OVERLAP[:3]) + "...",
        n=len(c8),
        pts=c8,
    )

    # -----------------------------------------------------------------------
    # C9: Treatment-response cohort
    # -----------------------------------------------------------------------
    meds_present = [c for c in COLS_MEDS if c in df.columns]
    c9 = None
    results["C9"] = dict(
        description="Treatment-response (biologics vs non-biologics)",
        objective="Secondary Objective 4 — treatment effect on systemic activity",
        inclusion_criteria="Treated SjD with treatment start date + ≥2 ESSDAI (pre/post)",
        time_zero_criteria="Treatment start date",
        key_variables=", ".join(meds_present[:3]) + "..." if meds_present else "N/A",
        n=c9,
        pts=c9,
        note="Pendiente por integración de datos desde BTRIS.",
    )

    # -----------------------------------------------------------------------
    # C10: Prospective PRO cohort
    # -----------------------------------------------------------------------
    c10 = pts_with_data(df, COL_ESSPRI_DRY, min_visits=2) & c1
    results["C10"] = dict(
        description="Prospective PRO (≥2 ESSPRI ≥6 months)",
        objective="Secondary Objective 5 — symptom burden and quality of life",
        inclusion_criteria="11-D with ≥2 PRO assessments separated by ≥6 months",
        time_zero_criteria="First visit with recorded PRO",
        key_variables=COL_ESSPRI_DRY,
        n=len(c10),
        pts=c10,
        note="Temporal separation ≥6 months cannot be verified without true dates.",
    )

    # -----------------------------------------------------------------------
    # C11: ML development cohort
    # -----------------------------------------------------------------------
    c11 = c2 & c1   # minimum base; enrich with C4/C7/C10 per endpoint
    results["C11"] = dict(
        description="ML development (prediction, transitions)",
        objective="Exploratory Objective — AI/ML prediction models",
        inclusion_criteria="Longitudinal dataset derived from C2/C7/C10 per endpoint",
        time_zero_criteria="Prediction-endpoint specific",
        key_variables=f"{COL_ESSDAI}, {COL_ESSPRI_DRY}, baseline covariates",
        n=len(c11),
        pts=c11,
        note="Train/val split must be done at patient level to avoid leakage.",
    )

    # -----------------------------------------------------------------------
    # C12: Comparator cohort (healthy volunteers)
    # -----------------------------------------------------------------------
    c12 = pts_with_values(c0_source, COL_SJD_CLASS, {5})
    results["C12"] = dict(
        description="Comparator — healthy volunteers (HV)",
        objective="Selected descriptive/comparative context",
        inclusion_criteria="C0 subset with Sjögren's class = 5",
        time_zero_criteria="First evaluable 15-D visit",
        key_variables=COL_SJD_CLASS,
        n=len(c12),
        pts=c12,
    )

    return results, total_pts, total_visits


# ---------------------------------------------------------------------------
# REPORTS
# ---------------------------------------------------------------------------

def build_summary_table(results: dict, total_pts: int, total_visits: int) -> pd.DataFrame:
    rows = []
    for cohort_id, r in results.items():
        n = r["n"]
        if n is None:
            flag = "TBD"
        else:
            flag = viability_flag(n)
        rows.append({
            "Cohort":            cohort_id,
            "Description":       r["description"],
            "Objective":         r["objective"],
            "n_patients":        n if n is not None else "TBD",
            "pct_total":         f"{100*n/total_pts:.1f}%" if n is not None else "TBD",
            "Viability":         flag,
            "T0_Criteria":       r["time_zero_criteria"],
            "Key_Variables":     r["key_variables"],
            "Notes":             r.get("note", ""),
        })
    return pd.DataFrame(rows)


def build_essdai_table(df: pd.DataFrame) -> pd.DataFrame:
    if COL_ESSDAI not in df.columns:
        return pd.DataFrame()
    essdai_v = df[df[COL_ESSDAI].notna()].groupby(COL_PATIENT).size()
    dist = essdai_v.value_counts().sort_index().reset_index()
    dist.columns = ["n_ESSDAI_visits", "n_patients"]
    dist["pct"] = (dist["n_patients"] / dist["n_patients"].sum() * 100).round(1)
    dist["cumulative_n"] = dist["n_patients"][::-1].cumsum()[::-1]  # >= n visits
    return dist


def build_esspri_table(df: pd.DataFrame) -> pd.DataFrame:
    if COL_ESSPRI_DRY not in df.columns:
        return pd.DataFrame()
    esspri_v = df[df[COL_ESSPRI_DRY].notna()].groupby(COL_PATIENT).size()
    dist = esspri_v.value_counts().sort_index().reset_index()
    dist.columns = ["n_ESSPRI_visits", "n_patients"]
    dist["pct"] = (dist["n_patients"] / dist["n_patients"].sum() * 100).round(1)
    dist["cumulative_n"] = dist["n_patients"][::-1].cumsum()[::-1]
    return dist




def export_cohort_subject_ids(df: pd.DataFrame, results: dict, output_dir: str):
    """Export ids__patient_record_number lists per cohort (c0, c1, ...)."""
    subject_col = "ids__patient_record_number"
    if subject_col not in df.columns:
        print(f"  WARNING: column '{subject_col}' not found; cohort subject ID exports skipped.")
        return

    cohort_ids_dir = os.path.join(output_dir, "cohort_ids")
    os.makedirs(cohort_ids_dir, exist_ok=True)

    for cohort_id, cohort_data in results.items():
        pts = cohort_data.get("pts", set())
        if pts is None:
            pts = set()

        cohort_subjects = (
            df[df[COL_PATIENT].isin(pts)][subject_col]
            .dropna()
            .drop_duplicates()
            .sort_values()
            .rename(subject_col)
        )

        file_name = f"{cohort_id.lower()}__ids__patient_record_number.csv"
        out_path = os.path.join(cohort_ids_dir, file_name)
        cohort_subjects.to_frame().to_csv(out_path, index=False)
        print(f"  Cohort IDs saved ({cohort_id}): {out_path}")


def print_summary(results: dict, total_pts: int, total_visits: int):
    print("\n" + "="*70)
    print("  SJÖGREN NATURAL HISTORY PROJECT — COHORT VIABILITY ANALYSIS")
    print("="*70)
    print(f"  Total patients: {total_pts}   |   Total visits: {total_visits}")
    print("-"*70)
    header = f"{'Cohort':<6} {'n':>6} {'%Total':>7}  {'Viability':<15}  Description"
    print(header)
    print("-"*70)
    for cid, r in results.items():
        n = r["n"]
        flag = viability_flag(n) if n is not None else "TBD"
        pct  = f"{100*n/total_pts:.0f}%" if n is not None else "TBD"
        n_str = str(n) if n is not None else "TBD"
        print(f"  {cid:<5} {n_str:>6} {pct:>7}  {flag:<15}  {r['description']}")
    print("="*70 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Sjögren Cohort Viability Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--input", "-i",
        default=str(ANALYTIC_DIR / "visits_long_collapsed_by_interval_codebook_corrected.csv"),
        help="Path to the codebook CSV (default: data_analytic/visits_long_collapsed_by_interval_codebook_corrected.csv)",
    )
    p.add_argument(
        "--output-dir", "-o",
        default=str(REPORTS_DIR / "cohort_viability"),
        help="Output directory for result CSV files (default: reports/cohort_viability)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.input):
        sys.exit(f"ERROR: File not found: {args.input}")

    print(f"\nLoading data from: {args.input}")
    df = load_data(args.input)
    print(f"  → {df[COL_PATIENT].nunique()} patients, {len(df)} visits, {df.shape[1]} columns")

    c0_input = ANALYTIC_DIR / "visits_long_collapsed_by_interval_codebook_not_clean.parquet"
    c0_df = None
    if c0_input.exists():
        print(f"Loading C0 source from: {c0_input}")
        c0_df = pd.read_parquet(c0_input)
    else:
        print(f"WARNING: C0 source not found at {c0_input}; using --input for C0.")

    results, total_pts, total_visits = run_analysis(df, c0_df=c0_df)

    # Print console summary
    print_summary(results, total_pts, total_visits)

    # Save CSV tables
    os.makedirs(args.output_dir, exist_ok=True)

    summary = build_summary_table(results, total_pts, total_visits)
    out_summary = os.path.join(args.output_dir, "cohort_viability_results.csv")
    summary.to_csv(out_summary, index=False)
    print(f"  Summary table saved: {out_summary}")

    essdai_dist = build_essdai_table(df)
    if not essdai_dist.empty:
        out_essdai = os.path.join(args.output_dir, "cohort_essdai_distribution.csv")
        essdai_dist.to_csv(out_essdai, index=False)
        print(f"  ESSDAI distribution saved: {out_essdai}")

    esspri_dist = build_esspri_table(df)
    if not esspri_dist.empty:
        out_esspri = os.path.join(args.output_dir, "cohort_esspri_distribution.csv")
        esspri_dist.to_csv(out_esspri, index=False)
        print(f"  ESSPRI distribution saved: {out_esspri}")


    export_cohort_subject_ids(df, results, args.output_dir)

    print("\nAnalysis completed.\n")


if __name__ == "__main__":
    main()
