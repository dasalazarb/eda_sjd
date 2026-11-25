"""Strict clinical result, child records, and Arrow storage contracts."""
from __future__ import annotations

from typing import Literal

import pyarrow as pa
from pydantic import BaseModel, ConfigDict, Field


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Finding(Strict):
    entity: Literal["anomalous_immunoglobulins", "immunoglobulin", "light_chain", "cryoglobulin", "cryoprecipitate"]
    assertion: Literal["present", "absent", "uncertain"]
    temporality: Literal["current", "historical", "planned", "unspecified"]
    immunoglobulin_class: Literal["IgG", "IgM", "IgA", "IgD", "IgE"] | None
    light_chain: Literal["kappa", "lambda"] | None
    pattern: Literal["monoclonal", "oligoclonal", "polyclonal"] | None
    intensity_descriptor: Literal["faint", "trace"] | None
    incubation_hours: int | None
    percentage: str | None
    comparator: Literal["EQ", "LT", "LE", "GT", "GE"] | None
    quote: str = Field(min_length=1)
    document_ids: list[str]


class Extraction(Strict):
    findings: list[Finding]


class Allele(Strict):
    locus: str
    reported: str
    allele: str | None
    resolution_fields: int | None
    suffix: str | None
    source_order: int
    ipd_match: str | None


class Medication(Strict):
    reported: str
    name: str
    dose: str | None = None
    unit: str | None = None
    route: str | None = None
    frequency: str | None = None
    rxcui: str | None = None
    source: str | None
    documentary_evidence: str
    window: str | None
    temporal_availability: Literal["unknown", "retrospective"]
    quote: str


class Evidence(Strict):
    field: str
    quote: str
    source: str
    rule: str
    strategy: str


class Semantic(Strict):
    qualitative_result: Literal["positive", "negative", "equivocal", "indeterminate", "borderline", "reactive", "nonreactive", "normal", "trace"] | None = None
    exact_value: str | None = None
    limit_value: str | None = None
    comparator: Literal["EQ", "LT", "LE", "GT", "GE", "RANGE"] | None = None
    range_low: str | None = None
    range_high: str | None = None
    raw_unit: str | None = None
    ucum_unit: str | None = None
    titer_numerator: int | None = None
    titer_denominator: int | None = None
    titer_comparator: Literal["EQ", "LT", "LE", "GT", "GE"] | None = None
    pattern_terms: list[str] = Field(default_factory=list)
    category: str | None = None
    ordinal_grade: str | None = None
    ordinal_score: int | None = None
    descriptor: str | None = None
    specimen_state: Literal["acceptable", "hemolyzed", "clotted", "rejected", "insufficient", "other"] | None = None
    result_availability: Literal["reported", "not_performed", "pending", "not_interpretable"] | None = None
    findings: list[Finding] = Field(default_factory=list)
    alleles: list[Allele] = Field(default_factory=list)
    medications: list[Medication] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    strategy: str = "unchanged"
    interpretation_note: str | None = None


STR, INT, BOOL = pa.string(), pa.int64(), pa.bool_()
FINDING = pa.schema([
    (name, INT if name == "incubation_hours" else pa.list_(STR) if name == "document_ids" else STR)
    for name in Finding.model_fields
])
ALLELE = pa.schema([(name, INT if name in {"resolution_fields", "source_order"} else STR) for name in Allele.model_fields])
MEDICATION = pa.schema([(name, STR) for name in Medication.model_fields])
EVIDENCE = pa.struct([(name, STR) for name in Evidence.model_fields])
NESTED = {"findings": pa.list_(pa.struct(FINDING)), "alleles": pa.list_(pa.struct(ALLELE)),
          "medications": pa.list_(pa.struct(MEDICATION)), "evidence": pa.list_(EVIDENCE),
          "pattern_terms": pa.list_(STR), "titer_numerator": INT, "titer_denominator": INT, "ordinal_score": INT}
SEMANTIC = pa.schema([(name, NESTED.get(name, STR)) for name in Semantic.model_fields])
IDENTITY = pa.schema([(name, STR) for name in ("mapping_id", "variable", "concept", "family", "raw_type", "raw_json", "context_json")])
INVENTORY = pa.schema([*IDENTITY, ("episodes", INT), ("patients", INT), ("observations", INT), ("baseline", INT),
                       ("laboratory_source_events", INT)])
DICTIONARY = pa.schema([*INVENTORY, *SEMANTIC, ("knowledge_doc_ids", pa.list_(STR))])
LINEAGE = pa.schema([
    ("observation_id", STR), ("mapping_id", STR), ("patient_id", STR), ("clinical_episode_id", STR),
    ("source_grain", STR), ("source_row", INT), ("source_observation_id", STR), ("source_hash", STR),
    ("measurement_date", STR), ("result_available_date", STR), ("paired_numeric_json", STR),
    ("n_measurements", INT), ("source_reference_status", STR), ("result_valid", BOOL), ("baseline", BOOL),
])
OBSERVATIONS = pa.schema([*LINEAGE, *[f for f in IDENTITY if f.name != "mapping_id"], *SEMANTIC])
CHILD_LINK = pa.schema([("observation_id", STR), ("mapping_id", STR), ("ordinal", INT)])
TRACE = pa.schema([("mapping_id", STR), ("doc_id", STR), ("rank", INT), ("lexical_rank", INT),
                   ("dense_rank", INT), ("exact", BOOL), ("snippet", STR), ("source", STR), ("source_version", STR)])
EXTRACTIONS = pa.schema([("mapping_id", STR), ("status", STR), ("findings", pa.list_(pa.struct(FINDING)))])
FEATURES = pa.schema([(name, STR) for name in ("name", "source", "kind", "dtype", "units", "temporal_availability",
                                               "allowed_use", "lineage", "analytic_eligibility_rule")])
