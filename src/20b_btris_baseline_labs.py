"""Derive auditable clinical-baseline laboratory features from step 20.

The clinical baseline is authoritative and is never recalculated here.  Stable
serologies (SSA, SSB, ANA and RF) use ever-positive status on/before baseline;
their closest assay remains separate provenance.  Cryoglobulins, C4 and WBC are
dynamic: the closest result in [-365, 0] is preferred, with [1, 30] used only as
a rescue.  Missing or uninterpretable evidence always remains missing.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from common import ANALYTIC_DIR, REPORTS_DIR, setup_logger

LAB_INPUT = ANALYTIC_DIR / "BTRIS" / "20_btris_lab_records_long.parquet"
SPINE_INPUT = ANALYTIC_DIR / "clinical_episode_spine_sjd.parquet"
LONG_OUTPUT = ANALYTIC_DIR / "BTRIS" / "20b_baseline_lab_selected_long.parquet"
WIDE_OUTPUT = ANALYTIC_DIR / "BTRIS" / "20b_baseline_labs_patient_level.parquet"
QC_DIR = REPORTS_DIR / "btris_labs" / "20b"

REQUIRED_LAB_COLUMNS = {
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
    "result_text",
    "unit",
    "reference_low",
    "reference_high",
    "reported_interpretation",
    "result_status",
    "result_valid_for_analysis",
    "invalid_result_reason",
    "source_protocol",
    "source_file",
}
OPTIONAL_PROVENANCE = [
    "specimen_datetime",
    "specimen_type",
    "order_identifier",
    "assay",
    "observation_identifier",
    "reference_range_raw",
    "reference_range_parse_status",
    "result_numeric_exact",
    "result_operator",
    "result_numeric_bound",
    "observation_comment",
    "observation_note",
]
FEATURES = (
    "anti_ro_ssa",
    "anti_la_ssb",
    "ana",
    "rheumatoid_factor",
    "cryoglobulinemia",
    "low_c4",
    "leukopenia",
)
STABLE_SOURCES = {
    "anti_ro_ssa": ("anti_ro_ssa",),
    "anti_la_ssb": ("anti_la_ssb",),
    "ana": ("ana_status", "ana_hep2_status"),
    "rheumatoid_factor": ("rheumatoid_factor",),
}
DYNAMIC_SOURCES = {
    "cryoglobulinemia": "cryoglobulins",
    "low_c4": "complement_c4",
    "leukopenia": "wbc",
}
CORE_INPUT_EVIDENCE_MODES = {
    "anti_ro_ssa": "positive",
    "anti_la_ssb": "positive",
    "ana_status": "positive",
    "ana_hep2_status": "positive",
    "rheumatoid_factor": "positive",
    "cryoglobulins": "positive",
    "complement_c4": "low",
    "wbc": "low",
}
WIDE_NAMES = {
    "anti_ro_ssa": "baseline_anti_ro_ssa",
    "anti_la_ssb": "baseline_anti_la_ssb",
    "ana": "baseline_ana",
    "rheumatoid_factor": "baseline_rf",
    "cryoglobulinemia": "baseline_cryoglobulinemia",
    "low_c4": "baseline_low_c4",
    "leukopenia": "baseline_leukopenia",
}
POSITIVE = {"positive", "pos", "detected", "reactive"}
NEGATIVE = {
    "negative",
    "neg",
    "not detected",
    "not detectable",
    "nonreactive",
    "non reactive",
}
LOW = {"low", "below normal", "decreased"}
NORMAL_HIGH = {"normal", "high", "within normal limits", "within range"}
FINAL = {"final", "verified"}


@dataclass(frozen=True)
class Interpretation:
    """A conservative interpretation and its auditable source."""

    interpreted_status: Any
    interpretation_source: str
    interpretation_qc: str


def _norm(value: Any) -> str:
    """Normalize an explicitly reported qualitative token."""
    if pd.isna(value):
        return ""
    return re.sub(r"[\s_-]+", " ", str(value).strip().lower()).strip(" .:;")


def _normalize_patient_id(series: pd.Series) -> pd.Series:
    """Apply the step-20 patient identifier contract to a series.

    Step 20 writes normalized identifiers after removing common MRN separators
    and leading zeroes.  The episode spine remains authoritative but stores its
    original identifier representation, so its patient view must use the same
    key before it can be compared with the step-20 analytical table.
    """
    normalized = (
        series.astype("string")
        .str.strip()
        .str.replace(r"[-/\\\s]", "", regex=True)
        .str.replace(r"^0+", "", regex=True)
    )
    return normalized.mask(normalized.isin(["", "nan", "None"]))


def interpret_result(
    row: Mapping[str, Any], kind: str, prespecified_cutoff: float | None = None
) -> Interpretation:
    """Interpret one valid result without inventing clinical thresholds.

    Parameters
    ----------
    row : Mapping[str, Any]
        Laboratory record.
    kind : str
        ``positive`` for serology/cryoglobulins or ``low`` for C4/WBC.
    prespecified_cutoff : float, optional
        Explicit project cutoff. The production configuration supplies none.

    Returns
    -------
    Interpretation
        Nullable status, evidence source, and QC explanation.
    """
    reported = _norm(row.get("reported_interpretation"))
    raw_tokens = [_norm(row.get("result_text")), _norm(row.get("result_raw"))]
    if kind == "positive":
        for token, source in [
            (reported, "reported_interpretation"),
            *[(token, "raw_qualitative_result") for token in raw_tokens],
        ]:
            if token in POSITIVE:
                return Interpretation(True, source, "interpretable")
            if token in NEGATIVE:
                return Interpretation(False, source, "interpretable")
            # ANA status is sometimes reported alongside its titer. Only the
            # explicit standalone status word is interpreted; numeric/operator
            # evidence remains untouched and assay-agnostic.
            if re.search(r"\bpositive\b", token):
                return Interpretation(True, source, "interpretable")
            if re.search(r"\bnegative\b", token):
                return Interpretation(False, source, "interpretable")
        numeric = pd.to_numeric(row.get("result_numeric"), errors="coerce")
        reference_high = pd.to_numeric(row.get("reference_high"), errors="coerce")
        if pd.notna(numeric) and pd.notna(reference_high):
            return Interpretation(
                bool(numeric > reference_high), "reference_range", "interpretable"
            )
        if pd.notna(numeric) and prespecified_cutoff is not None:
            return Interpretation(
                bool(numeric > prespecified_cutoff),
                "prespecified_cutoff",
                "interpretable",
            )
    elif kind == "low":
        if reported in LOW:
            return Interpretation(True, "reported_interpretation", "interpretable")
        if reported in NORMAL_HIGH:
            return Interpretation(False, "reported_interpretation", "interpretable")
        numeric = pd.to_numeric(row.get("result_numeric"), errors="coerce")
        reference_low = pd.to_numeric(row.get("reference_low"), errors="coerce")
        if pd.notna(numeric) and pd.notna(reference_low):
            return Interpretation(
                bool(numeric < reference_low), "reference_range", "interpretable"
            )
        if pd.notna(numeric) and prespecified_cutoff is not None:
            return Interpretation(
                bool(numeric < prespecified_cutoff),
                "prespecified_cutoff",
                "interpretable",
            )
    else:
        raise ValueError(f"Unknown interpretation kind: {kind}")
    return Interpretation(pd.NA, "uninterpretable", "insufficient_explicit_evidence")


def temporal_window(days: Any, stable: bool = False) -> str:
    """Return the inclusive methodological time-window category."""
    if pd.isna(days):
        return "missing"
    day = int(days)
    if day == 0:
        return "same_day"
    if -30 <= day <= -1:
        return "1_30d_pre"
    if -90 <= day <= -31:
        return "31_90d_pre"
    if -365 <= day <= -91:
        return "91_365d_pre"
    if stable and day < -365:
        return "gt365d_pre"
    if 1 <= day <= 30:
        return "1_30d_post_rescue"
    if stable and 31 <= day <= 90:
        return "1_90d_post_sensitivity"
    return "missing"


def _same_identity(left: pd.Series, right: pd.Series) -> bool:
    """Return whether two rows have strong evidence of one logical observation.

    Specimen type is deliberately excluded as identity evidence: values such as
    ``Blood`` are shared by unrelated draws and assays.
    """

    def same_nonmissing(key: str) -> bool:
        return (
            key in left.index
            and key in right.index
            and pd.notna(left[key])
            and pd.notna(right[key])
            and str(left[key]).strip() != ""
            and str(right[key]).strip() != ""
            and str(left[key]) == str(right[key])
        )

    if same_nonmissing("order_identifier"):
        return True
    if not same_nonmissing("specimen_datetime"):
        return False
    if same_nonmissing("assay"):
        return True
    return same_nonmissing("order_name_original") and same_nonmissing(
        "cluster_name_original"
    )


def _all_same_identity(records: pd.DataFrame) -> bool:
    """Return whether every row is strongly linked to the first row."""
    if len(records) < 2:
        return True
    first = records.iloc[0]
    return all(_same_identity(first, row) for _, row in records.iloc[1:].iterrows())


def _result_signature(row: pd.Series, kind: str) -> tuple[Any, ...]:
    interpreted = interpret_result(row, kind).interpreted_status
    return (
        None if pd.isna(interpreted) else bool(interpreted),
        _norm(row.get("result_raw")),
        row.get("result_numeric") if pd.notna(row.get("result_numeric")) else None,
        _norm(row.get("result_text")),
        _norm(row.get("unit")),
    )


def resolve_day(records: pd.DataFrame, kind: str, ana: bool = False) -> dict[str, Any]:
    """Resolve equivalent, preliminary/final, or conflicting same-day rows."""
    if records.empty:
        return {
            "row": None,
            "conflict": False,
            "deduplicated": False,
            "selected_final": False,
            "source_priority": False,
            "provenance": "[]",
        }
    work = records.copy()
    provenance = sorted(
        work.get("observation_identifier", work.index.to_series()).astype(str).tolist()
    )
    signatures = work.apply(lambda row: _result_signature(row, kind), axis=1)
    if signatures.nunique(dropna=False) == 1:
        sort_columns = [
            column
            for column in [
                "order_identifier",
                "specimen_datetime",
                "assay",
                "order_name_original",
                "cluster_name_original",
                "observation_identifier",
            ]
            if column in work
        ]
        chosen = (
            work.sort_values(by=sort_columns, kind="stable") if sort_columns else work
        )
        return {
            "row": chosen.iloc[0],
            "conflict": False,
            "deduplicated": len(work) > 1 and _all_same_identity(work),
            "selected_final": False,
            "source_priority": False,
            "provenance": json.dumps(provenance),
        }
    statuses = work["result_status"].map(_norm)
    final = work[statuses.isin(FINAL)]
    if len(final) == 1 and all(
        _same_identity(final.iloc[0], row)
        for _, row in work[~work.index.isin(final.index)].iterrows()
    ):
        return {
            "row": final.iloc[0],
            "conflict": False,
            "deduplicated": False,
            "selected_final": True,
            "source_priority": False,
            "provenance": json.dumps(provenance),
        }
    if (
        ana
        and len(work) == 2
        and set(work["canonical_analyte"]) == {"ana_status", "ana_hep2_status"}
    ):
        a, b = work.iloc[0], work.iloc[1]
        if _same_identity(a, b):
            selected = work[work["canonical_analyte"] == "ana_hep2_status"].iloc[0]
            return {
                "row": selected,
                "conflict": False,
                "deduplicated": True,
                "selected_final": False,
                "source_priority": True,
                "provenance": json.dumps(provenance),
            }
    return {
        "row": None,
        "conflict": True,
        "deduplicated": False,
        "selected_final": False,
        "source_priority": False,
        "provenance": json.dumps(provenance),
    }


def _blank(patient: pd.Series, feature: str, has_labs: bool) -> dict[str, Any]:
    eligible = pd.notna(patient["clinical_baseline_date"])
    return {
        "patient_id": patient["patient_id"],
        "clinical_baseline_episode_id": patient["clinical_baseline_episode_id"],
        "clinical_baseline_date": patient["clinical_baseline_date"],
        "lab_baseline_eligible": bool(eligible),
        "lab_baseline_exclusion_reason": pd.NA if eligible else "no_clinical_baseline",
        "has_any_btris_lab": bool(has_labs),
        "baseline_feature": feature,
        "source_canonical_analyte": pd.NA,
        "source_order_name": pd.NA,
        "source_cluster_name": pd.NA,
        "primary_baseline_status": pd.NA,
        "selected_result_raw": pd.NA,
        "selected_result_numeric": pd.NA,
        "selected_result_text": pd.NA,
        "selected_unit": pd.NA,
        "selected_lab_date": pd.NaT,
        "days_from_clinical_baseline": pd.NA,
        "temporal_window_category": "missing",
        "selection_rule": "none",
        "interpretation_source": "uninterpretable",
        "result_interpretation": pd.NA,
        "interpretation_qc": "not_tested_or_not_available",
        "reference_low": pd.NA,
        "reference_high": pd.NA,
        "n_prebaseline_measurements": 0,
        "n_eligible_prebaseline_measurements": 0,
        "n_interpretable_prebaseline_measurements": 0,
        "n_postbaseline_measurements": 0,
        "n_total_measurements": 0,
        "n_positive_prebaseline": 0,
        "n_negative_prebaseline": 0,
        "ever_positive_prebaseline": pd.NA,
        "ever_abnormal_prebaseline": pd.NA,
        "same_day_conflict": False,
        "longitudinal_discordance": False,
        "postbaseline_rescue": False,
        "deduplicate_same_specimen": False,
        "selected_from_final_verified": False,
        "source_row_ids": "[]",
        "closest_prebaseline_status": pd.NA,
        "closest_prebaseline_value": pd.NA,
        "closest_prebaseline_date": pd.NaT,
        "postbaseline_90d_sensitivity_status": pd.NA,
        "postbaseline_90d_sensitivity_date": pd.NaT,
        "qc_status": "not_tested_or_not_available",
    }


def _populate_selected(
    out: dict[str, Any], resolved: dict[str, Any], kind: str, stable: bool
) -> None:
    row = resolved["row"]
    out["same_day_conflict"] = resolved["conflict"]
    out["deduplicate_same_specimen"] = resolved["deduplicated"]
    out["selected_from_final_verified"] = resolved["selected_final"]
    out["source_row_ids"] = resolved["provenance"]
    out["ana_source_priority_used"] = resolved["source_priority"]
    if row is None:
        if resolved["conflict"]:
            out["qc_status"] = "same_day_unresolved_conflict"
        return
    interpreted = interpret_result(row, kind)
    out.update(
        {
            "source_canonical_analyte": row["canonical_analyte"],
            "source_order_name": row["order_name_original"],
            "source_cluster_name": row["cluster_name_original"],
            "selected_result_raw": row["result_raw"],
            "selected_result_numeric": row["result_numeric"],
            "selected_result_text": row["result_text"],
            "selected_unit": row["unit"],
            "selected_lab_date": row["lab_date"],
            "days_from_clinical_baseline": row["days_from_clinical_baseline"],
            "temporal_window_category": temporal_window(
                row["days_from_clinical_baseline"], stable
            ),
            "interpretation_source": interpreted.interpretation_source,
            "result_interpretation": interpreted.interpreted_status,
            "interpretation_qc": interpreted.interpretation_qc,
            "reference_low": row["reference_low"],
            "reference_high": row["reference_high"],
            "qc_status": (
                "selected_interpretable"
                if pd.notna(interpreted.interpreted_status)
                else "selected_but_uninterpretable"
            ),
        }
    )


def derive_stable_feature(
    patient: pd.Series, records: pd.DataFrame, feature: str, has_labs: bool
) -> dict[str, Any]:
    """Derive historical and closest stable-serology concepts separately."""
    out = _blank(patient, feature, has_labs)
    if not out["lab_baseline_eligible"]:
        out["qc_status"] = "no_clinical_baseline"
        return out
    source = records[records["canonical_analyte"].isin(STABLE_SOURCES[feature])].copy()
    pre = source[source["days_from_clinical_baseline"] <= 0]
    post = source[source["days_from_clinical_baseline"].between(1, 90)]
    pre_interpretations = pre.apply(
        lambda r: interpret_result(r, "positive").interpreted_status, axis=1
    )
    interpretable = pre_interpretations.notna()
    positives = int((pre_interpretations == True).sum())  # noqa: E712
    negatives = int((pre_interpretations == False).sum())  # noqa: E712
    out.update(
        {
            "n_prebaseline_measurements": len(pre),
            "n_interpretable_prebaseline_measurements": int(interpretable.sum()),
            "n_postbaseline_measurements": len(
                source[source["days_from_clinical_baseline"] > 0]
            ),
            "n_total_measurements": len(source),
            "n_positive_prebaseline": positives,
            "n_negative_prebaseline": negatives,
            "ever_positive_prebaseline": (
                True if positives else (False if negatives else pd.NA)
            ),
            "longitudinal_discordance": bool(positives and negatives),
            "selection_rule": "ever_positive_on_or_before_baseline",
        }
    )
    out["primary_baseline_status"] = out["ever_positive_prebaseline"]
    if not pre.empty:
        nearest_day = pre["days_from_clinical_baseline"].max()
        resolved = resolve_day(
            pre[pre["days_from_clinical_baseline"] == nearest_day],
            "positive",
            feature == "ana",
        )
        _populate_selected(out, resolved, "positive", True)
        if resolved["row"] is not None:
            out["closest_prebaseline_status"] = out["result_interpretation"]
            out["closest_prebaseline_value"] = out["selected_result_numeric"]
            out["closest_prebaseline_date"] = out["selected_lab_date"]
            if (
                feature == "ana"
                and resolved["row"]["canonical_analyte"] == "ana_hep2_status"
            ):
                _attach_ana_components(out, resolved["row"], records)
    if not interpretable.any() and not post.empty:
        nearest_day = post["days_from_clinical_baseline"].min()
        sensitivity = resolve_day(
            post[post["days_from_clinical_baseline"] == nearest_day],
            "positive",
            feature == "ana",
        )
        if sensitivity["row"] is not None:
            value = interpret_result(sensitivity["row"], "positive").interpreted_status
            out["postbaseline_90d_sensitivity_status"] = value
            out["postbaseline_90d_sensitivity_date"] = sensitivity["row"]["lab_date"]
    if source.empty:
        out["qc_status"] = "not_tested_or_not_available"
    return out


def _attach_ana_components(
    out: dict[str, Any], status_row: pd.Series, records: pd.DataFrame
) -> None:
    """Attach unique HEp-2 components without using them to infer ANA status."""
    component_names = {
        "ana_hep2_titer": "ana_selected_titer",
        "ana_hep2_pattern": "ana_selected_pattern",
        "ana_hep2_cytoplasmic_pattern": "ana_selected_cytoplasmic_pattern",
    }
    out["ana_component_ambiguity"] = False
    for analyte, output_name in component_names.items():
        candidates = records[
            (records["canonical_analyte"] == analyte)
            & (records["lab_date"] == status_row["lab_date"])
        ]
        identity_matches = candidates[
            candidates.apply(lambda row: _same_identity(status_row, row), axis=1)
        ]
        if len(identity_matches) == 1:
            candidates = identity_matches
        elif identity_matches.empty:
            candidates = candidates[
                candidates["order_name_original"] == status_row["order_name_original"]
            ]
        if len(candidates) == 1:
            component = candidates.iloc[0]
            value = component["result_text"]
            if pd.isna(value):
                value = component["result_raw"]
            if pd.isna(value):
                value = component["result_numeric"]
            out[output_name] = value
        elif len(candidates) > 1:
            out["ana_component_ambiguity"] = True


def derive_dynamic_feature(
    patient: pd.Series,
    records: pd.DataFrame,
    feature: str,
    has_labs: bool,
    pre_days: int = 365,
    allow_rescue: bool = True,
) -> dict[str, Any]:
    """Select a directional dynamic result and interpret it conservatively."""
    out = _blank(patient, feature, has_labs)
    if not out["lab_baseline_eligible"]:
        out["qc_status"] = "no_clinical_baseline"
        return out
    source = records[records["canonical_analyte"] == DYNAMIC_SOURCES[feature]].copy()
    kind = "positive" if feature == "cryoglobulinemia" else "low"
    all_pre = source[source["days_from_clinical_baseline"] <= 0]
    eligible_pre = source[source["days_from_clinical_baseline"].between(-pre_days, 0)]
    rescue = source[source["days_from_clinical_baseline"].between(1, 30)]
    interpreted_pre = all_pre.apply(
        lambda r: interpret_result(r, kind).interpreted_status, axis=1
    )
    out.update(
        {
            "n_prebaseline_measurements": len(all_pre),
            "n_eligible_prebaseline_measurements": len(eligible_pre),
            "n_interpretable_prebaseline_measurements": int(
                interpreted_pre.notna().sum()
            ),
            "n_postbaseline_measurements": len(
                source[source["days_from_clinical_baseline"] > 0]
            ),
            "n_total_measurements": len(source),
            "selection_rule": f"closest_prebaseline_within_{pre_days}d_then_30d_rescue",
        }
    )
    if feature == "cryoglobulinemia":
        pos = int((interpreted_pre == True).sum())  # noqa: E712
        neg = int((interpreted_pre == False).sum())  # noqa: E712
        out["ever_positive_prebaseline"] = True if pos else (False if neg else pd.NA)
        out["cryoglobulin_tested_prebaseline"] = bool(len(all_pre))
        out["cryoglobulin_tested_in_primary_window"] = bool(len(eligible_pre))
        out["cryoglobulins_ife_available"] = bool(
            (records["canonical_analyte"] == "cryoglobulins_ife").any()
        )
    else:
        low = int((interpreted_pre == True).sum())  # noqa: E712
        normal = int((interpreted_pre == False).sum())  # noqa: E712
        out["ever_abnormal_prebaseline"] = True if low else (False if normal else pd.NA)
    candidates = eligible_pre
    if candidates.empty and allow_rescue:
        candidates = rescue
        out["postbaseline_rescue"] = not candidates.empty
    if not candidates.empty:
        selected_day = (
            candidates["days_from_clinical_baseline"].max()
            if not eligible_pre.empty
            else candidates["days_from_clinical_baseline"].min()
        )
        resolved = resolve_day(
            candidates[candidates["days_from_clinical_baseline"] == selected_day], kind
        )
        _populate_selected(out, resolved, kind, False)
        out["primary_baseline_status"] = out["result_interpretation"]
    elif source.empty:
        out["qc_status"] = "not_tested_or_not_available"
    return out


def build_patient_baseline_frame(episode_spine: pd.DataFrame) -> pd.DataFrame:
    """Collapse an episode-level spine to coherent patient baseline metadata.

    Parameters
    ----------
    episode_spine : pd.DataFrame
        Clinical spine with one row per patient and clinical episode.

    Returns
    -------
    pd.DataFrame
        Exactly one row per patient after removing only fully equivalent
        patient-level baseline metadata.

    Raises
    ------
    ValueError
        If required columns are absent or one patient has conflicting baseline
        episode/date combinations.
    """
    required_spine = [
        "patient_id",
        "clinical_baseline_episode_id",
        "clinical_baseline_date",
    ]
    if missing_spine := set(required_spine) - set(episode_spine.columns):
        raise ValueError(f"Spine missing required columns: {sorted(missing_spine)}")

    patient_spine = episode_spine[required_spine].copy()
    patient_spine["patient_id"] = _normalize_patient_id(patient_spine["patient_id"])
    if patient_spine["patient_id"].isna().any():
        raise ValueError("Clinical episode spine contains missing patient identifiers")
    patient_spine["clinical_baseline_date"] = pd.to_datetime(
        patient_spine["clinical_baseline_date"], errors="coerce"
    )
    patient_spine = patient_spine.drop_duplicates()
    conflicts = patient_spine[patient_spine["patient_id"].duplicated(keep=False)]
    if not conflicts.empty:
        raise ValueError(
            "Clinical episode spine contains conflicting patient-level "
            "clinical baseline metadata"
        )
    return patient_spine.reset_index(drop=True)


def validate_inputs(labs: pd.DataFrame, patient_spine: pd.DataFrame) -> int:
    """Validate the lab schema and agreement with patient-level baselines."""
    missing = REQUIRED_LAB_COLUMNS - set(labs.columns)
    if missing:
        raise ValueError(f"Step 20 input missing required columns: {sorted(missing)}")
    required_spine = {
        "patient_id",
        "clinical_baseline_episode_id",
        "clinical_baseline_date",
    }
    if missing_spine := required_spine - set(patient_spine.columns):
        raise ValueError(f"Spine missing required columns: {sorted(missing_spine)}")
    if patient_spine["patient_id"].duplicated().any():
        raise ValueError("Patient baseline frame must contain one row per patient_id")
    lab_baselines = labs[
        ["patient_id", "clinical_baseline_episode_id", "clinical_baseline_date"]
    ].copy()
    lab_baselines["patient_id"] = _normalize_patient_id(lab_baselines["patient_id"])
    lab_baselines["clinical_baseline_date"] = pd.to_datetime(
        lab_baselines["clinical_baseline_date"], errors="coerce"
    )
    lab_baselines = lab_baselines.drop_duplicates()
    conflicting_labs = lab_baselines[lab_baselines["patient_id"].duplicated(keep=False)]
    if not conflicting_labs.empty:
        return int(conflicting_labs["patient_id"].nunique(dropna=False))

    check = lab_baselines.merge(
        patient_spine[list(required_spine)],
        on="patient_id",
        how="left",
        suffixes=("_lab", "_spine"),
    )
    dates_equal = pd.to_datetime(check["clinical_baseline_date_lab"]).eq(
        pd.to_datetime(check["clinical_baseline_date_spine"])
    ) | (
        check["clinical_baseline_date_lab"].isna()
        & check["clinical_baseline_date_spine"].isna()
    )
    ids_equal = check["clinical_baseline_episode_id_lab"].astype("string").eq(
        check["clinical_baseline_episode_id_spine"].astype("string")
    ) | (
        check["clinical_baseline_episode_id_lab"].isna()
        & check["clinical_baseline_episode_id_spine"].isna()
    )
    return int(check.loc[~dates_equal | ~ids_equal, "patient_id"].nunique(dropna=False))


def derive_long(labs: pd.DataFrame, spine: pd.DataFrame) -> pd.DataFrame:
    """Build one explicit row per patient and core feature."""
    work = labs.copy()
    work["patient_id"] = _normalize_patient_id(work["patient_id"])
    for column in ["lab_date", "clinical_baseline_date", "specimen_datetime"]:
        if column in work:
            work[column] = pd.to_datetime(work[column], errors="coerce")
    valid = work[work["result_valid_for_analysis"].fillna(False).astype(bool)]
    rows: list[dict[str, Any]] = []
    patients_with_labs = set(work["patient_id"])
    for _, patient in spine.iterrows():
        patient_records = valid[valid["patient_id"] == patient["patient_id"]]
        has_labs = patient["patient_id"] in patients_with_labs
        for feature in FEATURES:
            if feature in STABLE_SOURCES:
                rows.append(
                    derive_stable_feature(patient, patient_records, feature, has_labs)
                )
            else:
                rows.append(
                    derive_dynamic_feature(patient, patient_records, feature, has_labs)
                )
    result = pd.DataFrame(rows)
    for column in [
        "primary_baseline_status",
        "ever_positive_prebaseline",
        "ever_abnormal_prebaseline",
        "closest_prebaseline_status",
        "postbaseline_90d_sensitivity_status",
    ]:
        result[column] = result[column].astype("boolean")
    return result


def derive_wide(long: pd.DataFrame, spine: pd.DataFrame) -> pd.DataFrame:
    """Build the patient-level core table with required provenance."""
    base_columns = [
        "patient_id",
        "clinical_baseline_episode_id",
        "clinical_baseline_date",
        "lab_baseline_eligible",
        "lab_baseline_exclusion_reason",
        "has_any_btris_lab",
    ]
    wide = long[long["baseline_feature"] == FEATURES[0]][base_columns].copy()
    provenance = {
        "selected_lab_date": "measurement_date",
        "days_from_clinical_baseline": "days_from_clinical_baseline",
        "temporal_window_category": "temporal_window_category",
        "selection_rule": "selection_rule",
        "interpretation_source": "interpretation_source",
        "n_prebaseline_measurements": "n_prebaseline_measurements",
        "n_total_measurements": "n_total_measurements",
        "same_day_conflict": "conflict_flag",
        "postbaseline_rescue": "postbaseline_rescue",
    }
    for feature in FEATURES:
        prefix = {
            "anti_ro_ssa": "ssa",
            "anti_la_ssb": "ssb",
            "rheumatoid_factor": "rf",
        }.get(feature, feature)
        subset = long[long["baseline_feature"] == feature].set_index("patient_id")
        wide[WIDE_NAMES[feature]] = wide["patient_id"].map(
            subset["primary_baseline_status"]
        )
        for source, suffix in provenance.items():
            wide[f"{prefix}_{suffix}"] = wide["patient_id"].map(subset[source])
        if feature in STABLE_SOURCES:
            wide[f"{prefix}_ever_positive_prebaseline"] = wide["patient_id"].map(
                subset["ever_positive_prebaseline"]
            )
            wide[f"{prefix}_closest_prebaseline_status"] = wide["patient_id"].map(
                subset["closest_prebaseline_status"]
            )
            wide[f"{prefix}_longitudinal_discordance"] = wide["patient_id"].map(
                subset["longitudinal_discordance"]
            )
    if (
        len(wide) != spine["patient_id"].nunique()
        or wide["patient_id"].duplicated().any()
    ):
        raise ValueError(
            "Wide output does not contain exactly one row per spine patient"
        )
    return wide


def _status_changed(left: pd.Series, right: pd.Series) -> pd.Series:
    return ~(left.astype("string").fillna("NA") == right.astype("string").fillna("NA"))


def _nonmissing(series: pd.Series) -> pd.Series:
    """Identify populated values, treating blank strings as missing."""
    return series.notna() & series.astype("string").str.strip().ne("").fillna(False)


def build_core_input_evidence_qc(labs: pd.DataFrame) -> pd.DataFrame:
    """Summarize evidence available to interpret core laboratory analytes.

    Parameters
    ----------
    labs : pd.DataFrame
        Step-20 long laboratory records. Patient-level values and dates are used
        only to calculate aggregate counts and are never included in the output.

    Returns
    -------
    pd.DataFrame
        One aggregate row per core canonical analyte, including availability
        counts, row percentages, and expected interpretation mode.
    """
    rows: list[dict[str, Any]] = []
    evidence_columns = [
        "result_raw",
        "result_numeric",
        "result_text",
        "reported_interpretation",
        "reference_low",
        "reference_high",
        "unit",
    ]
    for analyte, mode in CORE_INPUT_EVIDENCE_MODES.items():
        group = labs[labs["canonical_analyte"] == analyte]
        n_rows = len(group)
        populated = {column: _nonmissing(group[column]) for column in evidence_columns}
        counts = {column: int(mask.sum()) for column, mask in populated.items()}
        any_interpretation = (
            populated["reported_interpretation"]
            | populated["result_text"]
            | populated["reference_low"]
            | populated["reference_high"]
        )
        if mode == "positive":
            numeric_without_evidence = (
                populated["result_numeric"]
                & ~populated["reference_high"]
                & ~populated["reported_interpretation"]
                & ~populated["result_text"]
            )
        else:
            numeric_without_evidence = (
                populated["result_numeric"]
                & ~populated["reference_low"]
                & ~populated["reported_interpretation"]
            )
        row: dict[str, Any] = {
            "canonical_analyte": analyte,
            "expected_interpretation_mode": mode,
            "n_rows": n_rows,
            "n_patients": int(group["patient_id"].nunique(dropna=True)),
        }
        for column in evidence_columns:
            row[f"n_{column}_nonmissing"] = counts[column]
            row[f"pct_{column}_nonmissing"] = (
                counts[column] / n_rows * 100 if n_rows else 0.0
            )
        row.update(
            {
                "n_rows_with_any_interpretation_evidence": int(
                    any_interpretation.sum()
                ),
                "pct_rows_with_any_interpretation_evidence": (
                    any_interpretation.mean() * 100 if n_rows else 0.0
                ),
                "n_rows_numeric_but_no_reference_or_interpretation": int(
                    numeric_without_evidence.sum()
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _build_hard_qc(
    long: pd.DataFrame, wide: pd.DataFrame, mismatches: int
) -> pd.DataFrame:
    """Build invariant checks for derived baseline laboratory statuses."""
    dynamic = long[long["baseline_feature"].isin(DYNAMIC_SOURCES)]
    stable = long[long["baseline_feature"].isin(STABLE_SOURCES)]
    hard = {
        "more_than_one_primary_per_patient_feature": int(
            long.duplicated(["patient_id", "baseline_feature"]).sum()
        ),
        "ineligible_patient_with_selected_primary": int(
            (
                ~long["lab_baseline_eligible"] & long["primary_baseline_status"].notna()
            ).sum()
        ),
        "baseline_mismatch_vs_spine": mismatches,
        "stable_primary_from_postbaseline_only": int(
            (
                long["baseline_feature"].isin(STABLE_SOURCES)
                & long["primary_baseline_status"].notna()
                & (long["n_interpretable_prebaseline_measurements"] == 0)
            ).sum()
        ),
        "stable_primary_positive_without_positive_prebaseline": int(
            (
                stable["primary_baseline_status"].eq(True)
                & stable["n_positive_prebaseline"].eq(0)
            ).sum()
        ),
        "stable_primary_negative_without_negative_prebaseline": int(
            (
                stable["primary_baseline_status"].eq(False)
                & stable["n_negative_prebaseline"].eq(0)
            ).sum()
        ),
        "stable_primary_negative_despite_positive_prebaseline": int(
            (
                stable["primary_baseline_status"].eq(False)
                & stable["n_positive_prebaseline"].gt(0)
            ).sum()
        ),
        "dynamic_prebaseline_older_than_365d": int(
            (
                (dynamic["days_from_clinical_baseline"] < -365)
                & dynamic["selected_lab_date"].notna()
            ).sum()
        ),
        "dynamic_rescue_after_30d": int(
            (
                dynamic["postbaseline_rescue"]
                & (dynamic["days_from_clinical_baseline"] > 30)
            ).sum()
        ),
        "dynamic_rescue_with_eligible_prebaseline": int(
            (
                dynamic["postbaseline_rescue"]
                & (dynamic["n_eligible_prebaseline_measurements"] > 0)
            ).sum()
        ),
        "conflict_with_nonmissing_primary": int(
            (long["same_day_conflict"] & long["result_interpretation"].notna()).sum()
        ),
        "ssa_from_ro52_ro60": int(
            (
                (long["baseline_feature"] == "anti_ro_ssa")
                & long["source_canonical_analyte"].isin(["anti_ro52", "anti_ro60"])
            ).sum()
        ),
        "ana_from_titer_pattern_only": int(
            (
                (long["baseline_feature"] == "ana")
                & long["source_canonical_analyte"].isin(
                    [
                        "ana_hep2_titer",
                        "ana_hep2_pattern",
                        "ana_hep2_cytoplasmic_pattern",
                    ]
                )
            ).sum()
        ),
        "cryoglobulinemia_from_ife_only": int(
            (
                (long["baseline_feature"] == "cryoglobulinemia")
                & (long["source_canonical_analyte"] == "cryoglobulins_ife")
            ).sum()
        ),
        "low_c4_without_evidence": int(
            (
                (long["baseline_feature"] == "low_c4")
                & long["primary_baseline_status"].notna()
                & ~long["interpretation_source"].isin(
                    [
                        "reported_interpretation",
                        "reference_range",
                        "prespecified_cutoff",
                    ]
                )
            ).sum()
        ),
        "leukopenia_without_evidence": int(
            (
                (long["baseline_feature"] == "leukopenia")
                & long["primary_baseline_status"].notna()
                & ~long["interpretation_source"].isin(
                    [
                        "reported_interpretation",
                        "reference_range",
                        "prespecified_cutoff",
                    ]
                )
            ).sum()
        ),
        "missing_uninterpretable_converted_to_negative": int(
            (
                long["baseline_feature"].isin(DYNAMIC_SOURCES)
                & long["interpretation_source"].eq("uninterpretable")
                & long["primary_baseline_status"].eq(False)
            ).sum()
        ),
        "duplicate_patient_rows_wide": int(wide["patient_id"].duplicated().sum()),
    }
    return pd.DataFrame(
        [
            {
                "qc_rule": key,
                "n_violations": value,
                "status": "PASS" if value == 0 else "FAIL",
            }
            for key, value in hard.items()
        ]
    )


def build_qc(
    long: pd.DataFrame,
    wide: pd.DataFrame,
    labs: pd.DataFrame,
    spine: pd.DataFrame,
    mismatches: int,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Build selection, missingness, sensitivity, and hard-QC tables."""
    labs = labs.copy()
    labs["patient_id"] = _normalize_patient_id(labs["patient_id"])
    summaries, temporal, missingness, sensitivity = [], [], [], []
    for feature, group in long.groupby("baseline_feature", sort=False):
        selected = group["primary_baseline_status"].notna()
        measured = group["n_total_measurements"] > 0
        days = pd.to_numeric(
            group["days_from_clinical_baseline"], errors="coerce"
        ).dropna()
        summaries.append(
            {
                "baseline_feature": feature,
                "n_patients_total": len(group),
                "n_patients_eligible": int(group["lab_baseline_eligible"].sum()),
                "n_with_any_measurement": int(measured.sum()),
                "n_with_interpretable_measurement": int(
                    (group["n_interpretable_prebaseline_measurements"] > 0).sum()
                ),
                "n_with_primary_status": int(selected.sum()),
                "coverage_pct": selected.mean() * 100,
                **{
                    f"n_{cat}": int((group["temporal_window_category"] == cat).sum())
                    for cat in [
                        "same_day",
                        "1_30d_pre",
                        "31_90d_pre",
                        "91_365d_pre",
                        "gt365d_pre",
                    ]
                },
                "n_postbaseline_rescue": int(group["postbaseline_rescue"].sum()),
                "median_days_from_baseline": days.median(),
                "q1_days_from_baseline": days.quantile(0.25),
                "q3_days_from_baseline": days.quantile(0.75),
                "n_same_day_conflicts": int(group["same_day_conflict"].sum()),
                "n_longitudinal_discordance": int(
                    group["longitudinal_discordance"].sum()
                ),
                "n_missing": int((~selected).sum()),
            }
        )
        counts = group["temporal_window_category"].value_counts(dropna=False)
        temporal.extend(
            {
                "baseline_feature": feature,
                "temporal_window_category": key,
                "n": int(value),
                "pct": value / len(group) * 100,
            }
            for key, value in counts.items()
        )
        eligible = group[group["lab_baseline_eligible"]]
        missingness.append(
            {
                "baseline_feature": feature,
                "n_eligible_patients": len(eligible),
                "n_tested": int((eligible["n_total_measurements"] > 0).sum()),
                "n_not_tested_or_not_available": int(
                    (eligible["n_total_measurements"] == 0).sum()
                ),
                "n_tested_but_uninterpretable": int(
                    (
                        (eligible["n_total_measurements"] > 0)
                        & eligible["primary_baseline_status"].isna()
                    ).sum()
                ),
                "n_selected": int(eligible["primary_baseline_status"].notna().sum()),
                "pct_selected": (
                    eligible["primary_baseline_status"].notna().mean() * 100
                    if len(eligible)
                    else 0
                ),
            }
        )
    valid = labs[labs["result_valid_for_analysis"].fillna(False).astype(bool)]
    for feature in DYNAMIC_SOURCES:
        alternative_rows = []
        no_rescue_rows = []
        for _, patient in spine.iterrows():
            patient_records = valid[valid["patient_id"] == patient["patient_id"]]
            has_labs = bool((labs["patient_id"] == patient["patient_id"]).any())
            alternative_rows.append(
                derive_dynamic_feature(
                    patient, patient_records, feature, has_labs, 90, True
                )
            )
            no_rescue_rows.append(
                derive_dynamic_feature(
                    patient, patient_records, feature, has_labs, 365, False
                )
            )
        primary = long[long["baseline_feature"] == feature][
            "primary_baseline_status"
        ].reset_index(drop=True)
        for analysis, rows in [
            ("dynamic_90d", alternative_rows),
            ("prebaseline_only", no_rescue_rows),
        ]:
            alt = pd.Series(
                [row["primary_baseline_status"] for row in rows], dtype="boolean"
            )
            sensitivity.append(
                {
                    "baseline_feature": feature,
                    "analysis": analysis,
                    "primary_n_available": int(primary.notna().sum()),
                    "sensitivity_n_available": int(alt.notna().sum()),
                    "n_status_changed": int(_status_changed(primary, alt).sum()),
                    "n_became_missing": int((primary.notna() & alt.isna()).sum()),
                }
            )
    for feature in STABLE_SOURCES:
        group = long[long["baseline_feature"] == feature]
        ever, closest = (
            group["primary_baseline_status"],
            group["closest_prebaseline_status"],
        )
        sensitivity.append(
            {
                "baseline_feature": feature,
                "analysis": "ever_vs_closest",
                "n_concordant": int(
                    (
                        ever.notna() & closest.notna() & ~_status_changed(ever, closest)
                    ).sum()
                ),
                "n_ever_positive_but_closest_negative": int(
                    (ever.fillna(False) & ~closest.fillna(True)).sum()
                ),
                "n_closest_positive": int(closest.fillna(False).sum()),
                "n_uninterpretable": int(closest.isna().sum()),
            }
        )
    hard_qc = _build_hard_qc(long, wide, mismatches)
    core_input_evidence_qc = build_core_input_evidence_qc(labs)
    interpretation = (
        long.groupby(["baseline_feature", "interpretation_source"], dropna=False)
        .size()
        .rename("n")
        .reset_index()
    )
    interpretation["pct"] = (
        interpretation["n"]
        / interpretation.groupby("baseline_feature")["n"].transform("sum")
        * 100
    )
    audit_columns = [
        "patient_id",
        "lab_baseline_eligible",
        "has_any_btris_lab",
        *WIDE_NAMES.values(),
    ]
    audit = wide[audit_columns].copy()
    audit["n_core_features_available"] = (
        audit[list(WIDE_NAMES.values())].notna().sum(axis=1)
    )
    audit["n_core_features_missing"] = (
        len(FEATURES) - audit["n_core_features_available"]
    )
    return {
        "20b_baseline_lab_selection_summary.csv": pd.DataFrame(summaries),
        "20b_baseline_lab_temporal_distribution.csv": pd.DataFrame(temporal),
        "20b_baseline_lab_conflicts.csv": long[long["same_day_conflict"]],
        "20b_baseline_lab_postbaseline_rescue.csv": long[long["postbaseline_rescue"]][
            [
                "patient_id",
                "baseline_feature",
                "clinical_baseline_date",
                "selected_lab_date",
                "days_from_clinical_baseline",
                "selection_rule",
            ]
        ],
        "20b_baseline_lab_discordant_serology.csv": long[
            long["longitudinal_discordance"]
        ][
            [
                "patient_id",
                "baseline_feature",
                "ever_positive_prebaseline",
                "closest_prebaseline_status",
                "n_positive_prebaseline",
                "n_negative_prebaseline",
            ]
        ],
        "20b_baseline_lab_interpretation_qc.csv": interpretation,
        "20b_baseline_lab_patient_audit.csv": audit,
        "20b_baseline_lab_missingness.csv": pd.DataFrame(missingness),
        "20b_baseline_lab_sensitivity_summary.csv": pd.DataFrame(sensitivity),
        "20b_baseline_lab_hard_qc.csv": hard_qc,
        "20b_core_input_evidence_qc.csv": core_input_evidence_qc,
    }, hard_qc


def run(
    lab_path: Path = LAB_INPUT,
    spine_path: Path = SPINE_INPUT,
    long_path: Path = LONG_OUTPUT,
    wide_path: Path = WIDE_OUTPUT,
    qc_dir: Path = QC_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run baseline-lab derivation and write outputs before failing hard QC."""
    logger = setup_logger("20b_btris_baseline_labs")
    labs = pd.read_parquet(lab_path)
    episode_spine = pd.read_parquet(spine_path)
    patient_spine = build_patient_baseline_frame(episode_spine)
    mismatches = validate_inputs(labs, patient_spine)
    long = derive_long(labs, patient_spine)
    wide = derive_wide(long, patient_spine)
    reports, hard_qc = build_qc(long, wide, labs, patient_spine, mismatches)
    long_path.parent.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)
    long.to_parquet(long_path, index=False)
    wide.to_parquet(wide_path, index=False)
    for filename, report in reports.items():
        report.to_csv(qc_dir / filename, index=False)
    summary = reports["20b_baseline_lab_selection_summary.csv"]
    logger.info(
        "Patients spine=%d eligible=%d any_BTRIS=%d",
        len(patient_spine),
        wide["lab_baseline_eligible"].sum(),
        wide["has_any_btris_lab"].sum(),
    )
    for row in summary.itertuples():
        logger.info(
            "Feature %s coverage=%.1f%% rescue=%d conflicts=%d discordance=%d",
            row.baseline_feature,
            row.coverage_pct,
            row.n_postbaseline_rescue,
            row.n_same_day_conflicts,
            row.n_longitudinal_discordance,
        )
    evidence = reports["20b_core_input_evidence_qc.csv"].set_index("canonical_analyte")
    logger.info(
        "Core input interpretation evidence: SSA ref_high=%.1f%%; "
        "SSB ref_high=%.1f%%; RF ref_high=%.1f%%; C4 ref_low=%.1f%%; "
        "WBC ref_low=%.1f%%",
        evidence.at["anti_ro_ssa", "pct_reference_high_nonmissing"],
        evidence.at["anti_la_ssb", "pct_reference_high_nonmissing"],
        evidence.at["rheumatoid_factor", "pct_reference_high_nonmissing"],
        evidence.at["complement_c4", "pct_reference_low_nonmissing"],
        evidence.at["wbc", "pct_reference_low_nonmissing"],
    )
    violations = int(hard_qc["n_violations"].sum())
    logger.info("Hard QC violations=%d", violations)
    if violations:
        raise RuntimeError(
            f"20b hard QC failed with {violations} violations; QC files were written"
        )
    return long, wide


def main() -> None:
    """Parse CLI paths and run the derivation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labs", type=Path, default=LAB_INPUT)
    parser.add_argument("--spine", type=Path, default=SPINE_INPUT)
    parser.add_argument("--long-output", type=Path, default=LONG_OUTPUT)
    parser.add_argument("--wide-output", type=Path, default=WIDE_OUTPUT)
    parser.add_argument("--qc-dir", type=Path, default=QC_DIR)
    args = parser.parse_args()
    run(args.labs, args.spine, args.long_output, args.wide_output, args.qc_dir)


if __name__ == "__main__":
    main()
