"""
=============================================================================
SJÖGREN'S NATURAL HISTORY PROJECT
Cohort Viability Analysis — C0 through C12
=============================================================================

INPUT:  visits_long_collapsed_by_interval_codebook_type_recode.csv
        (Formato codebook: filas = visitas; celdas no-nulas = dato registrado;
         la fila 0 del CSV contiene descriptores de tipo y se omite.)

OUTPUT: cohort_viability_results.csv   — tabla resumen por cohorte
        cohort_essdai_distribution.csv — distribución de visitas ESSDAI
        cohort_esspri_distribution.csv — distribución de visitas ESSPRI

USO EN BIOWULF:
    module load python
    python cohort_viability_analysis.py --input <ruta_al_csv>

Dependencias: pandas, numpy
=============================================================================
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# CONFIG: columnas clave del codebook
# Ajustar si la limpieza de datos cambia los nombres de columnas.
# ---------------------------------------------------------------------------

COL_PATIENT    = "ids__patient_record_number"
COL_INTERVAL   = "ids__interval_name"
COL_VISIT_DATE = "ids__visit_date"

# Clasificación de enfermedad
COL_SJD_CLASS  = "visit_summary_form__sjogrens_class"
COL_IE_PRIM_SEC = "inclusion/exclusion_criteria__ic_sjogrens_prim_sec"
COL_IE_HV       = "inclusion/exclusion_criteria__ic_hv"

# ESSDAI / ESSPRI
COL_ESSDAI     = "essdai-_r__essdai_total_score"
COL_ESSPRI_DRY = "esspri_questionnaire__dryness"       # proxy de completitud ESSPRI

# Labs
COL_LABS       = "cris_lab_form__labs_done"

# Flujo salival (glandular)
COL_SAL_FLOW   = "salivary_flow_form__flow_whole_unstim"

# Examen ocular
COL_EYE        = "eye_examination__eye_exam_done"

# Comorbilidades
COL_COMORBID   = "rheumatological_comorbidities__comorbid_none"
COL_PMH_CARDIO = "past_medical_history__cardio_none"

# Autoimmune overlap (para C8)
COLS_OVERLAP   = [
    "rheumatological_comorbidities__ra",
    "rheumatological_comorbidities__sle1",
    "rheumatological_comorbidities__systemic_sclerosis",
    "rheumatological_comorbidities__polymyositis",
    "rheumatological_comorbidities__dermatomyositis",
]

# Medicamentos (para C9)
COLS_MEDS      = [f"medications__rx_{i}_name" for i in range(1, 11)]

# Dominios ESSDAI individuales (para C5 — proxy EGM)
COLS_ESSDAI_DOMAINS = [
    "essdai__constitutional", "essdai__hema_lphdenopthy",
    "essdai__gland_swell",    "essdai__articular_domain",
    "essdai__cutaneous",      "essdai__pulmonary",
    "essdai__renal",          "essdai__muscular_domain",
    "essdai__neuro_peripheral","essdai__cns",
    "essdai__hematologic",    "essdai__biological_domain",
]

# Criterios de inclusión/exclusión (para C0)
COLS_IE = []   # se detectan dinámicamente

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    """Carga el CSV omitiendo la fila de descriptores de tipo (fila 0 del CSV)."""
    df = pd.read_csv(path)
    # La primera fila de datos contiene type-codes ("string","numeric","boolean","date")
    # cuando ids__subject_number == "numeric" → es la fila de descriptores → eliminar
    mask_type_row = df[COL_PATIENT].isna() | (df.iloc[:, 2].astype(str) == "numeric")
    # Detectar fila tipo: la columna ids__subject_number == "numeric"
    if "ids__subject_number" in df.columns:
        type_rows = df[df["ids__subject_number"].astype(str).str.lower() == "numeric"].index
        df = df.drop(index=type_rows).reset_index(drop=True)
    return df


def pts_with_data(df: pd.DataFrame, col: str, min_visits: int = 1) -> set:
    """Pacientes con >=min_visits observaciones no-nulas en 'col'."""
    if col not in df.columns:
        return set()
    tmp = df[df[col].notna()].groupby(COL_PATIENT).size()
    return set(tmp[tmp >= min_visits].index)


def pts_any_col(df: pd.DataFrame, cols: list) -> set:
    """Pacientes con al menos UNA celda no-nula en cualquiera de 'cols'."""
    present = [c for c in cols if c in df.columns]
    if not present:
        return set()
    mask = df[present].notna().any(axis=1)
    return set(df[mask][COL_PATIENT].unique())


def pts_all_cols_same_visit(df: pd.DataFrame, cols: list) -> set:
    """Pacientes con al menos UNA visita donde TODAS las columnas son no-nulas."""
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
        return "INSUFICIENTE"


# ---------------------------------------------------------------------------
# MAIN ANALYSIS
# ---------------------------------------------------------------------------

def run_analysis(df: pd.DataFrame) -> dict:
    total_pts   = df[COL_PATIENT].nunique()
    total_visits = len(df)

    # Detectar columnas IE
    ie_cols = [c for c in df.columns
               if "inclusion" in c.lower() or "exclusion" in c.lower()]

    results = {}

    # -----------------------------------------------------------------------
    # C0: Source screening cohort
    # -----------------------------------------------------------------------
    c0 = set(df[df[ie_cols].notna().any(axis=1)][COL_PATIENT].unique()) \
         if ie_cols else set(df[COL_PATIENT].unique())
    results["C0"] = dict(
        descripcion="Source screening (15-D)",
        objetivo="Obj. Primario 1 — contexto de referral",
        criterio_inclusion="Todo participante 15-D con criterios IE registrados",
        criterio_tiempo_cero="Primera visita evaluable NIH SjD",
        variables_clave=", ".join(ie_cols[:4]) + "..." if ie_cols else "N/A",
        n=len(c0),
        pts=c0,
    )

    # -----------------------------------------------------------------------
    # C1: Core SjD master cohort
    # -----------------------------------------------------------------------
    c1 = pts_with_data(df, COL_SJD_CLASS)
    results["C1"] = dict(
        descripcion="Core SjD master (primary + secondary)",
        objetivo="Backbone — base de todos los análisis",
        criterio_inclusion="Clasificación SjD registrada (primaria o secundaria)",
        criterio_tiempo_cero="Primera visita NIH evaluable; colapsar 15-D y 11-D si ≤30 días",
        variables_clave=COL_SJD_CLASS,
        n=len(c1),
        pts=c1,
    )

    # -----------------------------------------------------------------------
    # C2: Longitudinal ESSDAI cohort
    # -----------------------------------------------------------------------
    essdai_any = pts_with_data(df, COL_ESSDAI, min_visits=1)
    c2         = pts_with_data(df, COL_ESSDAI, min_visits=2)
    c2_3v      = pts_with_data(df, COL_ESSDAI, min_visits=3)

    essdai_visits = df[df[COL_ESSDAI].notna()].groupby(COL_PATIENT).size() \
                    if COL_ESSDAI in df.columns else pd.Series(dtype=int)

    results["C2"] = dict(
        descripcion="Longitudinal ESSDAI (≥2 mediciones)",
        objetivo="Obj. Primario 2 — progresión de la enfermedad",
        criterio_inclusion="SjD primaria/secundaria + ≥2 ESSDAI evaluables",
        criterio_tiempo_cero="Primera visita con ESSDAI registrado",
        variables_clave=COL_ESSDAI,
        n=len(c2),
        pts=c2,
        n_any=len(essdai_any),
        n_3v=len(c2_3v),
        median_visits=essdai_visits.median() if len(essdai_visits) else 0,
    )

    # -----------------------------------------------------------------------
    # C3: Incident severe-risk cohort
    # -----------------------------------------------------------------------
    # Requiere valores ESSDAI reales (< 5 al baseline) — no disponibles en este codebook
    results["C3"] = dict(
        descripcion="Incident severe-risk (ESSDAI < 5 basal)",
        objetivo="Obj. Primario 4 — tiempo a enfermedad severa",
        criterio_inclusion="Subconjunto de C2 con ESSDAI < 5 en visita ancla",
        criterio_tiempo_cero="Primera visita con ESSDAI < 5 documentado",
        variables_clave=COL_ESSDAI + " (valor real requerido)",
        n=None,   # No determinable sin valores ESSDAI reales
        pts=None,
        nota="REQUIERE valores ESSDAI reales del CRIS/CTDB. Techo teórico = n(C2).",
    )

    # -----------------------------------------------------------------------
    # C4: Domain/risk-factor cohort
    # -----------------------------------------------------------------------
    lab_pts = pts_with_data(df, COL_LABS)
    c4 = c2 & lab_pts
    results["C4"] = dict(
        descripcion="Domain/risk-factor (C2 + labs)",
        objetivo="Obj. Primario 3 — factores de riesgo clínicos y serológicos",
        criterio_inclusion="C2 + covariables basales + labs enlazados",
        criterio_tiempo_cero="Primera visita con datos de dominio ESSDAI + predictores basales",
        variables_clave=f"{COL_ESSDAI}, {COL_LABS}",
        n=len(c4),
        pts=c4,
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
        descripcion="Glandular-EGM overlap",
        objetivo="Obj. Primario 5 — co-ocurrencia glandular / extra-glandular",
        criterio_inclusion="SjD con fenotipo glandular ascertable + datos EGM",
        criterio_tiempo_cero="Primera visita con ambos componentes medibles",
        variables_clave=f"{COL_SAL_FLOW}, dominios ESSDAI",
        n=len(c5),
        pts=c5,
        n_gland=len(gland_pts),
        n_eye=len(eye_pts),
        n_egm=len(egm_pts),
    )

    # -----------------------------------------------------------------------
    # C6: Comorbidity cohort
    # -----------------------------------------------------------------------
    c6 = pts_with_data(df, COL_COMORBID) | pts_with_data(df, COL_PMH_CARDIO)
    results["C6"] = dict(
        descripcion="Comorbilidades (baseline + follow-up)",
        objetivo="Obj. Secundario 1 — prevalencia e incidencia de comorbilidades",
        criterio_inclusion="SjD con documentación de comorbilidades",
        criterio_tiempo_cero="Primera visita SjD evaluable",
        variables_clave=f"{COL_COMORBID}, {COL_PMH_CARDIO}",
        n=len(c6),
        pts=c6,
    )

    # -----------------------------------------------------------------------
    # C7: Phenotype population cohort (Pop 1–3)
    # -----------------------------------------------------------------------
    esspri_any = pts_with_data(df, COL_ESSPRI_DRY)
    paired_df  = df[df[COL_ESSDAI].notna() & df[COL_ESSPRI_DRY].notna()] \
                 if (COL_ESSDAI in df.columns and COL_ESSPRI_DRY in df.columns) \
                 else pd.DataFrame()
    c7 = set(paired_df[COL_PATIENT].unique()) if not paired_df.empty else set()
    paired_visits = paired_df.groupby(COL_PATIENT).size() if not paired_df.empty else pd.Series(dtype=int)
    results["C7"] = dict(
        descripcion="Phenotype Pop 1-3 (ESSDAI+ESSPRI pareados)",
        objetivo="Obj. Secundario 2 — proporciones en Pop 1/2/3",
        criterio_inclusion="≥1 visita con ESSDAI y ESSPRI en ventana ±30 días",
        criterio_tiempo_cero="Primera evaluación pareada ESSDAI-ESSPRI",
        variables_clave=f"{COL_ESSDAI}, {COL_ESSPRI_DRY}",
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
        descripcion="Overlap autoinmune (RA/SLE/SSc/otros)",
        objetivo="Obj. Secundario 3 — progresión en SjD + overlap",
        criterio_inclusion="Subconjunto de C1 con enfermedad autoinmune coexistente",
        criterio_tiempo_cero="Mismo índice que cohorte madre (C1)",
        variables_clave=", ".join(COLS_OVERLAP[:3]) + "...",
        n=len(c8),
        pts=c8,
    )

    # -----------------------------------------------------------------------
    # C9: Treatment-response cohort
    # -----------------------------------------------------------------------
    meds_present = [c for c in COLS_MEDS if c in df.columns]
    treated_pts  = pts_any_col(df, meds_present)
    c9 = treated_pts & c2
    results["C9"] = dict(
        descripcion="Treatment-response (biológicos vs no-biológicos)",
        objetivo="Obj. Secundario 4 — efecto del tratamiento sobre actividad sistémica",
        criterio_inclusion="SjD tratada con fecha de inicio + ≥2 ESSDAI (pre y post)",
        criterio_tiempo_cero="Fecha de inicio del tratamiento",
        variables_clave=", ".join(meds_present[:3]) + "..." if meds_present else "N/A",
        n=len(c9),
        pts=c9,
        nota="n muy bajo — revisar completitud de datos de medicación.",
    )

    # -----------------------------------------------------------------------
    # C10: Prospective PRO cohort
    # -----------------------------------------------------------------------
    c10 = pts_with_data(df, COL_ESSPRI_DRY, min_visits=2)
    results["C10"] = dict(
        descripcion="PRO prospectivo (≥2 ESSPRI ≥6 meses)",
        objetivo="Obj. Secundario 5 — carga sintomática y calidad de vida",
        criterio_inclusion="11-D con ≥2 evaluaciones PRO separadas ≥6 meses",
        criterio_tiempo_cero="Primera visita con PRO registrado",
        variables_clave=COL_ESSPRI_DRY,
        n=len(c10),
        pts=c10,
        nota="Separación temporal ≥6 meses no verificable sin fechas reales.",
    )

    # -----------------------------------------------------------------------
    # C11: ML development cohort
    # -----------------------------------------------------------------------
    c11 = c2   # base mínima; se enriquece con C4/C7/C10 según endpoint
    results["C11"] = dict(
        descripcion="ML development (predicción, transiciones)",
        objetivo="Obj. Exploratorio — modelos de predicción AI/ML",
        criterio_inclusion="Dataset longitudinal derivado de C2/C7/C10 según endpoint",
        criterio_tiempo_cero="Específico por endpoint de predicción",
        variables_clave=f"{COL_ESSDAI}, {COL_ESSPRI_DRY}, covariables basales",
        n=len(c11),
        pts=c11,
        nota="Split train/val debe hacerse a nivel paciente para evitar leakage.",
    )

    # -----------------------------------------------------------------------
    # C12: Comparator cohort (healthy volunteers)
    # -----------------------------------------------------------------------
    c12 = pts_with_data(df, COL_IE_HV)
    results["C12"] = dict(
        descripcion="Comparador — voluntarios sanos (HV)",
        objetivo="Contexto descriptivo / comparativo seleccionado",
        criterio_inclusion="HV y no-SjD de 15-D",
        criterio_tiempo_cero="Primera visita evaluable 15-D",
        variables_clave=COL_IE_HV,
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
            "Cohorte":           cohort_id,
            "Descripcion":       r["descripcion"],
            "Objetivo":          r["objetivo"],
            "n_pacientes":       n if n is not None else "TBD",
            "pct_total":         f"{100*n/total_pts:.1f}%" if n is not None else "TBD",
            "Viabilidad":        flag,
            "Criterio_T0":       r["criterio_tiempo_cero"],
            "Variables_clave":   r["variables_clave"],
            "Notas":             r.get("nota", ""),
        })
    return pd.DataFrame(rows)


def build_essdai_table(df: pd.DataFrame) -> pd.DataFrame:
    if COL_ESSDAI not in df.columns:
        return pd.DataFrame()
    essdai_v = df[df[COL_ESSDAI].notna()].groupby(COL_PATIENT).size()
    dist = essdai_v.value_counts().sort_index().reset_index()
    dist.columns = ["n_visitas_ESSDAI", "n_pacientes"]
    dist["pct"] = (dist["n_pacientes"] / dist["n_pacientes"].sum() * 100).round(1)
    dist["acumulado_n"] = dist["n_pacientes"][::-1].cumsum()[::-1]  # >= n visitas
    return dist


def build_esspri_table(df: pd.DataFrame) -> pd.DataFrame:
    if COL_ESSPRI_DRY not in df.columns:
        return pd.DataFrame()
    esspri_v = df[df[COL_ESSPRI_DRY].notna()].groupby(COL_PATIENT).size()
    dist = esspri_v.value_counts().sort_index().reset_index()
    dist.columns = ["n_visitas_ESSPRI", "n_pacientes"]
    dist["pct"] = (dist["n_pacientes"] / dist["n_pacientes"].sum() * 100).round(1)
    dist["acumulado_n"] = dist["n_pacientes"][::-1].cumsum()[::-1]
    return dist


def print_summary(results: dict, total_pts: int, total_visits: int):
    print("\n" + "="*70)
    print("  SJÖGREN NATURAL HISTORY PROJECT — COHORT VIABILITY ANALYSIS")
    print("="*70)
    print(f"  Total pacientes: {total_pts}   |   Total visitas: {total_visits}")
    print("-"*70)
    header = f"{'Cohorte':<6} {'n':>6} {'%Total':>7}  {'Viabilidad':<15}  Descripcion"
    print(header)
    print("-"*70)
    for cid, r in results.items():
        n = r["n"]
        flag = viability_flag(n) if n is not None else "TBD"
        pct  = f"{100*n/total_pts:.0f}%" if n is not None else "TBD"
        n_str = str(n) if n is not None else "TBD"
        print(f"  {cid:<5} {n_str:>6} {pct:>7}  {flag:<15}  {r['descripcion']}")
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
        default="visits_long_collapsed_by_interval_codebook_corrected.csv",
        help="Ruta al CSV del codebook (default: busca en directorio actual)",
    )
    p.add_argument(
        "--output-dir", "-o",
        default=".",
        help="Directorio de salida para los CSV de resultados (default: .)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.input):
        sys.exit(f"ERROR: No se encontró el archivo: {args.input}")

    print(f"\nCargando datos desde: {args.input}")
    df = load_data(args.input)
    print(f"  → {df[COL_PATIENT].nunique()} pacientes, {len(df)} visitas, {df.shape[1]} columnas")

    results, total_pts, total_visits = run_analysis(df)

    # Imprimir resumen en consola
    print_summary(results, total_pts, total_visits)

    # Guardar tablas CSV
    os.makedirs(args.output_dir, exist_ok=True)

    summary = build_summary_table(results, total_pts, total_visits)
    out_summary = os.path.join(args.output_dir, "cohort_viability_results.csv")
    summary.to_csv(out_summary, index=False)
    print(f"  Tabla resumen guardada: {out_summary}")

    essdai_dist = build_essdai_table(df)
    if not essdai_dist.empty:
        out_essdai = os.path.join(args.output_dir, "cohort_essdai_distribution.csv")
        essdai_dist.to_csv(out_essdai, index=False)
        print(f"  Distribución ESSDAI guardada: {out_essdai}")

    esspri_dist = build_esspri_table(df)
    if not esspri_dist.empty:
        out_esspri = os.path.join(args.output_dir, "cohort_esspri_distribution.csv")
        esspri_dist.to_csv(out_esspri, index=False)
        print(f"  Distribución ESSPRI guardada: {out_esspri}")

    print("\nAnálisis completado.\n")


if __name__ == "__main__":
    main()