"""Episode and BTRIS observation inventory with exact, typed mapping identities."""
from __future__ import annotations

from collections import defaultdict
from datetime import date
from decimal import Decimal
from pathlib import Path
import json
import re
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .settings import KEYS, digest, serialize

LAB = re.compile(r"lab__(.+)__(text|value)")
HLA = re.compile(r"(?:lab|sero)__hla_([a-z0-9]+)__(?:text|asof_value|patient_consensus_value)")
CONTEXT_FIELDS = (
    "source_system", "assay_id", "method", "unit", "reference_low", "reference_high",
    "reference_text", "reference_status", "specimen", "relevant_protocol_version", "assay_status",
    "observation_comment", "observation_note",
)
SOURCE_CONTEXT = {
    "assay_id": ("order_name_original", "cluster_name_original"), "method": "assay", "unit": "unit",
    "reference_low": "reference_low", "reference_high": "reference_high", "reference_text": "reference_range_raw",
    "reference_status": "reported_interpretation", "specimen": "specimen_type", "relevant_protocol_version": "source_protocol",
    "assay_status": "result_status", "observation_comment": "observation_comment", "observation_note": "observation_note",
}
WIDE_CONTEXT = {
    "source_system": "source_system", "assay_id": "assay_id", "method": "method", "unit": "unit",
    "reference_low": "reference_low", "reference_high": "reference_high", "reference_text": "reference_range",
    "reference_status": "reference_status", "specimen": "specimen", "relevant_protocol_version": "protocol_version",
    "observation_comment": "comment", "observation_note": "note",
}


def clean(value: Any) -> Any:
    """Convert source scalars to lossless JSON-compatible primitives."""
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(item) for item in value]
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def kind(value: Any, arrow_type: str | None = None) -> str:
    """Distinguish null, strings, bool, numeric and compound source values."""
    if clean(value) is None:
        return "null"
    if isinstance(value, np.generic):
        value = value.item()
    return type(value).__name__ if arrow_type is None else f"{type(value).__name__}:{arrow_type}"


def protected(name: str, cfg: dict) -> str | None:
    """Identify excluded columns without reading their values."""
    policy = cfg["concepts"]
    if name.startswith(tuple(policy["protected_prefixes"])) or name in policy["protected_names"]:
        return "protected"
    if "__" not in name or set(re.split(r"[^a-z0-9]+", name.lower())) & set(policy["identifier_tokens"]):
        return "structural_or_identifier"
    if re.search(r"(?:^|__)\w*(?:anchor|baseline|episode)_(?:date|datetime)$", name):
        return "structural_date"
    return None


def route(variable: str, cfg: dict) -> tuple[str, str]:
    """Match exact aliases or explicit namespace grammars."""
    rules = cfg["concepts"]
    hla = HLA.fullmatch(variable)
    if hla and hla[1].upper() in rules["hla_loci"]:
        return f"hla_{hla[1]}", "hla"
    if variable in rules["medications"]:
        return variable.removeprefix("medication__"), "medication"
    lab = LAB.fullmatch(variable)
    if not lab:
        family = "derived" if variable.startswith(("sero__", "pop__", "ovl__", "ext__", "essdai__", "esspri__", "pro__")) else "other"
        return variable, family
    concept, representation = lab.groups()
    if concept in rules["ana"] and representation == "text":
        return concept, "ana"
    for family, concepts in rules["narrative"].items():
        if concept in concepts and representation == "text":
            return concept, family
    if representation == "text":
        for family in ("urine_nominal", "urine_microscopy", "urine_graded", "qualitative"):
            if concept in rules[family]:
                return concept, family
        if concept.startswith(("urine_", "urinalysis_includes_microscopic")):
            return concept, "urinalysis"
    return concept, "lab_text" if representation == "text" else "lab_numeric"


def check_keys(frame: pd.DataFrame, label: str) -> pd.MultiIndex:
    """Validate one row per patient and clinical episode."""
    if not set(KEYS).issubset(frame):
        raise ValueError(f"{label} is missing episode keys")
    keys = frame[KEYS]
    if keys.isna().any().any() or keys.apply(lambda col: col.astype(str).str.strip().eq("").any()).any() or keys.duplicated().any():
        raise ValueError(f"{label} contains null, empty or duplicate episode keys")
    return pd.MultiIndex.from_frame(keys)


def source_concept(row: dict) -> str | None:
    """Use the same canonical-or-order/cluster fallback as obj1 Step 01."""
    canonical = clean(row.get("canonical_analyte"))
    if canonical:
        return canonical
    def slug(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    order = slug(clean(row.get("order_name_original")) or "")
    cluster = slug(clean(row.get("cluster_name_original")) or "")
    return f"{order}__{cluster}" if order and cluster else None


def inventory(master: pd.DataFrame, context: pd.DataFrame | None, source: pd.DataFrame | None,
              hashes: dict, cfg: dict) -> tuple[list[dict], list[dict], list[dict]]:
    """Inventory every eligible episode value and each source observation."""
    index = check_keys(master, "Step 10")
    if context is not None:
        ctx_index = check_keys(context, "Step 10 context")
        if not index.difference(ctx_index).empty or not ctx_index.difference(index).empty:
            raise ValueError("Step 10 and context episode keys differ")
        context = context.set_index(KEYS).reindex(index).reset_index(drop=True)
    types = pq.read_schema(Path(cfg["paths"]["integrated"]))
    counts, lineages, variables = {}, [], []

    def record(variable: str, raw: Any, context_fields: dict, lineage: dict, arrow_type: str | None) -> None:
        concept, family = route(variable, cfg)
        raw_type, raw_value = kind(raw, arrow_type), clean(raw)
        mapping_id = digest([variable, raw_type, raw_value, context_fields])
        if mapping_id not in counts:
            counts[mapping_id] = dict(mapping_id=mapping_id, variable=variable, concept=concept, family=family,
                                      raw_type=raw_type, raw_json=serialize(raw_value), context_json=serialize(context_fields),
                                      episodes=set(), patients=set(), baseline=0, observations=0, laboratory_source_events=0)
        counter = counts[mapping_id]
        if lineage["patient_id"] is not None:
            counter["patients"].add(lineage["patient_id"])
        if lineage["clinical_episode_id"] is not None:
            counter["episodes"].add((lineage["patient_id"], lineage["clinical_episode_id"]))
        counter["observations"] += 1
        counter["baseline"] += lineage["baseline"] is True
        counter["laboratory_source_events"] += lineage["source_grain"] == "btris_observation"
        lineages.append(dict(lineage, mapping_id=mapping_id))

    def side(name: str) -> pd.Series:
        if name in master:
            return master[name]
        if context is not None and name in context:
            return context[name]
        return pd.Series(None, index=master.index)

    base = side("is_clinical_baseline").map(clean).tolist()
    for variable in master:
        reason = protected(variable, cfg)
        if reason:
            variables.append(dict(variable=variable, policy=reason, dtype=str(master[variable].dtype), family=None,
                                  episodes=len(master), nonnull=None, nonempty=None, distinct=None))
            continue
        concept, family = route(variable, cfg)
        lab = LAB.fullmatch(variable)
        prefix = f"lab__{concept}__" if lab else None
        context_cols = {key: side(prefix + suffix).tolist() for key, suffix in WIDE_CONTEXT.items()} if prefix else {}
        measurement = side(prefix + "measurement_date").map(clean).tolist() if prefix else [None] * len(master)
        availability = side(prefix + "result_available_date").map(clean).tolist() if prefix else [None] * len(master)
        measurements = side(prefix + "n_measurements").map(clean).tolist() if prefix else [None] * len(master)
        paired = side(prefix + "value").tolist() if lab and lab[2] == "text" else [None] * len(master)
        nonnull = nonempty = 0
        distinct = set()
        for i, (key, value) in enumerate(zip(index, master[variable], strict=True)):
            raw = clean(value)
            nonnull += raw is not None
            nonempty += raw is not None and (not isinstance(raw, str) or bool(raw.strip()))
            if raw is not None:
                distinct.add((kind(value), serialize(raw)))
            metadata = {key: clean(context_cols[key][i]) if key in context_cols else None for key in CONTEXT_FIELDS}
            record(variable, value, metadata, dict(
                observation_id=digest(["episode", hashes["integrated"], i, variable]), patient_id=str(key[0]),
                clinical_episode_id=str(key[1]), source_grain="episode_variable", source_row=i,
                source_observation_id=None, source_hash=hashes["integrated"], measurement_date=measurement[i],
                result_available_date=availability[i], paired_numeric_json=serialize(clean(paired[i])),
                n_measurements=measurements[i], source_reference_status=metadata["reference_status"],
                result_valid=None, baseline=base[i]), str(types.field(variable).type))
        variables.append(dict(variable=variable, family=family, policy="eligible", dtype=str(master[variable].dtype),
                              episodes=len(master), nonnull=nonnull, nonempty=nonempty, distinct=len(distinct)))

    if source is not None:
        required = {"patient_id", "result_raw", "matched_clinical_episode_id", "lab_date", "canonical_analyte"}
        if not required.issubset(source):
            raise ValueError(f"BTRIS source is missing contracted columns: {sorted(required.difference(source))}")
        for i, row in enumerate(source.to_dict("records")):
            concept = source_concept(row)
            variable = f"lab__{concept or 'unmapped_source_observation'}__text"
            metadata = {key: clean(row.get(field)) if isinstance(field, str) else [clean(row.get(item)) for item in field]
                        for key, field in SOURCE_CONTEXT.items()}
            metadata["source_system"] = "BTRIS"
            pid, episode = clean(row["patient_id"]), clean(row["matched_clinical_episode_id"])
            record(variable, row["result_raw"], metadata, dict(
                observation_id=digest(["btris", hashes["source_events"], i]), patient_id=None if pid is None else str(pid),
                clinical_episode_id=None if episode is None else str(episode), source_grain="btris_observation",
                source_row=i, source_observation_id=clean(row.get("observation_identifier")),
                source_hash=hashes["source_events"], measurement_date=clean(row["lab_date"]),
                result_available_date=clean(row.get("result_available_date")),
                paired_numeric_json=serialize(clean(row.get("result_numeric_exact"))), n_measurements=1,
                source_reference_status=metadata["reference_status"], result_valid=clean(row.get("result_valid_for_analysis")),
                baseline=None), None)

    for row in counts.values():
        row.update(episodes=len(row["episodes"]), patients=len(row["patients"]))
    return list(counts.values()), lineages, variables
