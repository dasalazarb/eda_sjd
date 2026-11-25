"""Apply the semantic dictionary without replacing source columns."""
from __future__ import annotations

from collections import Counter, defaultdict
from decimal import Decimal
import json
import math
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .schemas import ALLELE, CHILD_LINK, FEATURES, FINDING, MEDICATION, OBSERVATIONS, SEMANTIC
from .settings import KEYS, serialize

WIDE_FIELDS = {
    "qualitative_result": pa.string(), "exact_value": pa.float64(), "limit_value": pa.float64(),
    "comparator": pa.string(), "range_low": pa.float64(), "range_high": pa.float64(),
    "raw_unit": pa.string(), "ucum_unit": pa.string(), "titer_numerator": pa.int64(),
    "titer_denominator": pa.int64(), "titer_comparator": pa.string(), "category": pa.string(),
    "ordinal_grade": pa.string(), "ordinal_score": pa.int64(), "descriptor": pa.string(), "specimen_state": pa.string(),
    "result_availability": pa.string(),
}
SEMANTIC_FACTS = [*WIDE_FIELDS, "pattern_terms", "findings", "alleles", "medications"]


def write_parquet(path: Path, rows: list[dict], schema: pa.Schema) -> None:
    """Write a Parquet artifact with an explicit typed schema."""
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


def add_column(table: pa.Table, name: str, values: list, dtype: pa.DataType) -> pa.Table:
    """Append a feature while preventing collision with original columns."""
    if name in table.column_names:
        raise ValueError(f"Harmonized feature collides with an input column: {name}")
    return table.append_column(name, pa.array(values, type=dtype))


def project_float(value: str | None) -> float | None:
    """Project a decimal to wide only if its float round-trip is equivalent."""
    if value is None:
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) and Decimal(str(numeric)) == Decimal(value) else None


def feature(name: str, source: str, field: str, dtype: pa.DataType, availability: str = "unknown") -> dict:
    """Record the use boundary and provenance of a candidate feature."""
    return dict(name=name, source=source, kind=field, dtype=str(dtype), units="see same-variable unit fields"
                if pa.types.is_floating(dtype) else None, temporal_availability=availability,
                allowed_use="as_of" if availability == "verified_result_date" else "descriptive",
                lineage="harmonized_observations.parquet",
                analytic_eligibility_rule="Candidate field; requires study-specific admissibility")


def apply(input_table: pa.Table, source: pa.Table | None, lineage: list[dict], dictionary: list[dict],
          analytic: Path) -> tuple[pa.Table, list[dict], list[dict]]:
    """Join every observation m:1 and materialize the Arrow outputs."""
    lookup = {row["mapping_id"]: row for row in dictionary}
    if len(lookup) != len(dictionary) or set(lookup) != {row["mapping_id"] for row in lineage}:
        raise ValueError("Semantic dictionary failed the m:1 mapping contract")
    if len({row["observation_id"] for row in lineage}) != len(lineage):
        raise ValueError("Observation lineage keys must be unique")
    observations, source_rows, episode_variables = [], [], defaultdict(list)
    children = {field: [] for field in ("findings", "alleles", "medications")}
    for line in lineage:
        mapping = lookup[line["mapping_id"]]
        values = {name: mapping[name] for name in SEMANTIC.names}
        observation = dict(line, **values, **{name: mapping[name] for name in
                                             ("variable", "concept", "family", "raw_type", "raw_json", "context_json")})
        observations.append(observation)
        if line["source_grain"] == "btris_observation":
            source_rows.append(observation)
        else:
            episode_variables[mapping["variable"]].append(observation)
        for field, output in children.items():
            output.extend(dict(observation_id=line["observation_id"], mapping_id=line["mapping_id"],
                               ordinal=i, **item) for i, item in enumerate(values[field]))
    write_parquet(analytic / "harmonized_observations.parquet", observations, OBSERVATIONS)
    for field, filename, schema in (
        ("findings", "semantic_findings", FINDING), ("alleles", "hla_alleles", ALLELE),
        ("medications", "medication_entities", MEDICATION),
    ):
        write_parquet(analytic / f"{filename}.parquet", children[field], pa.schema([*CHILD_LINK, *schema]))

    if source is not None:
        if [row["source_row"] for row in source_rows] != list(range(source.num_rows)):
            raise ValueError("BTRIS source order or grain changed")
        source_candidate = add_column(source, "harm_mapping_id", [row["mapping_id"] for row in source_rows], pa.string())
        for item in SEMANTIC:
            source_candidate = add_column(source_candidate, "harm_" + item.name, [row[item.name] for row in source_rows], item.type)
        target = analytic / "20_btris_lab_records_long_harmonized.parquet"
        pq.write_table(source_candidate, target)
        validate_original(source, pq.read_table(target))

    wide, features = input_table, []
    for variable, rows in episode_variables.items():
        if [row["source_row"] for row in rows] != list(range(wide.num_rows)):
            raise ValueError(f"Episode inventory alignment failed for {variable}")
        if rows and rows[0]["family"] in {"other", "derived"}:
            continue
        for field, dtype in WIDE_FIELDS.items():
            values = [row[field] for row in rows]
            if pa.types.is_floating(dtype):
                values = [project_float(value) for value in values]
            if all(value is None for value in values):
                continue
            name = f"harm__{variable}__{field}"
            wide = add_column(wide, name, values, dtype)
            features.append(feature(name, variable, field, dtype))
    return wide, features, observations


def validate_original(before: pa.Table, after: pa.Table) -> None:
    """Verify row count, order, types, metadata and every original value."""
    names = before.column_names
    if before.num_rows != after.num_rows or after.column_names[:len(names)] != names:
        raise ValueError("Original column order or row count changed")
    after_original = after.select(names)
    if before.schema != after_original.schema or before.schema.metadata != after_original.schema.metadata:
        raise ValueError("Original Arrow schema or metadata changed")
    # Arrow equality treats NaN as unequal. Keep NaN comparisons and other types exact.
    for name in names:
        a, b = before[name], after_original[name]
        if a.equals(b):
            continue
        if not a.is_null().equals(b.is_null()):
            raise ValueError(f"Original null positions changed for {name}")
        pd.testing.assert_series_equal(a.to_pandas(), b.to_pandas(), check_exact=True, obj=name)


def timestamp(value):
    """Parse an explicit source timestamp without substituting an anchor date."""
    return pd.to_datetime(value, errors="coerce", utc=True)


def rebuild(master: pd.DataFrame, wide: pa.Table, observations: list[dict],
            features: list[dict]) -> tuple[pa.Table, list[dict]]:
    """Recompute supported longitudinal candidates from dated source evidence."""
    anchor = master["clinical_anchor_date"].map(timestamp) if "clinical_anchor_date" in master else pd.Series(pd.NaT, index=master.index)
    staged = defaultdict(list)
    for row in observations:
        if row["source_grain"] != "btris_observation" or row["result_valid"] is not True:
            continue
        date = timestamp(row["result_available_date"])
        measured = timestamp(row["measurement_date"])
        if pd.isna(date) or pd.isna(measured) or date < measured:
            continue
        staged[(row["patient_id"], row["concept"])].append((date, row))

    output = []
    for col in master:
        if not col.startswith(("sero__", "essdai__", "esspri__", "ext__", "ovl__", "pop__", "pro__")):
            continue
        after, evaluated = None, 0
        before = master[col]
        rule = "Existing producer value retained; required lineage or source components unavailable."
        is_ever = col.startswith("sero__") and col.endswith("__ever_positive_through_episode")
        is_hla = col.startswith("sero__hla_") and col.endswith("__asof_value")
        if is_ever or is_hla:
            concept = col.split("__")[1]
            values = []
            for pid, at in zip(master[KEYS[0]], anchor, strict=True):
                known = [(date, row) for date, row in staged[(str(pid), concept)] if not pd.isna(at) and date <= at]
                assay_keys = {serialize({field: json.loads(row["context_json"]).get(field)
                                         for field in ("assay_id", "method", "unit")}) for _, row in known}
                if len(assay_keys) > 1:
                    values.append(None)
                    continue
                if is_ever:
                    states = {row["qualitative_result"] for _, row in known
                              if row["specimen_state"] is None and row["result_availability"] == "reported"}
                    values.append(True if "positive" in states else False if "negative" in states else None)
                else:
                    dated = [(date, json.loads(row["raw_json"])) for date, row in known
                             if any(item["allele"] is not None for item in row["alleles"])]
                    latest = max((date for date, _ in dated), default=None)
                    alternatives = {raw for date, raw in dated if date == latest}
                    values.append(next(iter(alternatives)) if len(alternatives) == 1 else None)
            if any(value is not None for value in values):
                dtype = pa.bool_() if is_ever else pa.string()
                name = "harm__" + col
                wide = add_column(wide, name, values, dtype)
                after = pd.Series(values, dtype="boolean" if is_ever else "string")
                evaluated = int(after.notna().sum())
                scope = "Tri-state cumulative explicit serology" if is_ever else "Most recent compatible HLA report"
                rule = scope + " through each anchor using verified result availability."
                features.append(feature(name, col, "derived_recompute", dtype, "verified_result_date"))
        candidate = before if after is None else after
        states = lambda values: ["null" if pd.isna(value) else "true" if value is True or str(value) == "True"
                                 else "false" if value is False or str(value) == "False" else "observed" for value in values]
        a, b = states(before), states(candidate)
        transitions = {f"{left} → {right}": n for (left, right), n in Counter(zip(a, b)).items()}
        output.append(dict(variable=col, recomputed=after is not None, evaluated_episodes=evaluated,
                           before=serialize(dict(Counter(a))), after=serialize(dict(Counter(b))),
                           transitions=serialize(transitions), rule=rule, before_distinct=int(before.nunique()),
                           after_distinct=int(candidate.nunique())))
    return wide, output


def write_candidate(analytic: Path, wide: pa.Table, features: list[dict]) -> Path:
    """Write the standalone candidate and its feature manifest."""
    path = analytic / "10_integrated_longitudinal_clinical_episode_harmonized.parquet"
    pq.write_table(wide, path)
    write_parquet(analytic / "feature_manifest.parquet", features, FEATURES)
    return path
