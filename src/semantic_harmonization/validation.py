"""Reconciliation, exact evidence checks, and downstream compatibility."""
from __future__ import annotations

from collections import Counter, defaultdict
from decimal import Decimal
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

from .application import SEMANTIC_FACTS
from .inventory import protected
from .settings import KEYS, serialize


def validate_mappings(dictionary: list[dict], lineage: list[dict], variables: list[dict], episodes: int) -> None:
    """Reconcile dictionary coverage, populations, evidence and numeric states."""
    mapping_counts = Counter(row["mapping_id"] for row in lineage)
    episode_counts = Counter()
    patients, keys, baseline, source_events = defaultdict(set), defaultdict(set), Counter(), Counter()
    lookup = {row["mapping_id"]: row for row in dictionary}
    if mapping_counts.keys() != lookup.keys():
        raise ValueError("Lineage and dictionary contain different mapping keys")
    for line in lineage:
        mapping_id = line["mapping_id"]
        row = lookup[mapping_id]
        if line["source_grain"] == "episode_variable":
            episode_counts[row["variable"]] += 1
        if line["patient_id"] is not None:
            patients[mapping_id].add(line["patient_id"])
        if line["clinical_episode_id"] is not None:
            keys[mapping_id].add((line["patient_id"], line["clinical_episode_id"]))
        baseline[mapping_id] += line["baseline"] is True
        source_events[mapping_id] += line["source_grain"] == "btris_observation"
    if any(episode_counts[entry["variable"]] != episodes for entry in variables if entry["policy"] == "eligible"):
        raise ValueError("Inventory lost episode-variable observations")

    for row in dictionary:
        mapping_id = row["mapping_id"]
        counts = (mapping_counts[mapping_id], len(patients[mapping_id]), len(keys[mapping_id]),
                  baseline[mapping_id], source_events[mapping_id])
        if counts != tuple(row[f] for f in ("observations", "patients", "episodes", "baseline", "laboratory_source_events")):
            raise ValueError("Semantic mapping frequency did not reconcile")
        fields = (row["exact_value"], row["limit_value"], row["range_low"], row["range_high"])
        operator = row["comparator"]
        required = {"EQ": (True, False, False, False), "RANGE": (False, False, True, True),
                    None: (False, False, False, False)}.get(operator, (False, True, False, False))
        if tuple(value is not None for value in fields) != required:
            raise ValueError("Numeric comparator does not match its measurement fields")
        if any(value is not None and not Decimal(value).is_finite() for value in fields):
            raise ValueError("Nonfinite numeric result")
        raw = json.loads(row["raw_json"])
        context = json.loads(row["context_json"])
        text_sources = {"raw_text": str(raw), **{f"context.{k}": v for k, v in context.items()}}
        evidence_fields = set()
        for evidence in row["evidence"]:
            exact = text_sources.get(evidence["source"])
            if not isinstance(exact, str) or evidence["quote"] not in exact:
                raise ValueError("Semantic evidence is not a verbatim source span")
            evidence_fields.add(evidence["field"])
        for field in SEMANTIC_FACTS:
            if row[field] not in (None, []) and field not in evidence_fields:
                raise ValueError(f"Semantic field lacks evidence: {field}")
        for finding in row["findings"]:
            if not any(finding["quote"] in text for text in text_sources.values() if isinstance(text, str)):
                raise ValueError("Narrative finding lacks source evidence")


def comparison(inventory: list[dict], dictionary: list[dict]) -> list[dict]:
    """Calculate episode-grain coverage and occurrence-weighted comparisons."""
    mappings = defaultdict(list)
    for row in dictionary:
        if row["observations"] > row["laboratory_source_events"]:
            mappings[row["variable"]].append(row)
    output = []
    for variable in inventory:
        if variable["policy"] != "eligible":
            continue
        name = variable["variable"]
        rows = mappings[name]
        states, strategies, fields = Counter(), Counter(), Counter()
        rules = set()
        after = set()
        for row in rows:
            n = row["observations"] - row["laboratory_source_events"]
            fields_supported = {field: row[field] for field in SEMANTIC_FACTS if row[field] not in (None, [])}
            if fields_supported:
                after.add(serialize(fields_supported))
            states["interpreted"] += n if fields_supported else 0
            states["raw_null"] += n if row["raw_type"] == "null" else 0
            raw = json.loads(row["raw_json"])
            states["empty"] += n if isinstance(raw, str) and not raw.strip() else 0
            strategies[row["strategy"]] += n
            rules.update(item["rule"] for item in row["evidence"])
            fields.update({field: n for field in fields_supported})
        output.append(dict(variable, dtype_after=variable["dtype"], n_interpreted=states["interpreted"],
                           n_semantic_null=variable["episodes"] - states["interpreted"],
                           n_raw_null=states["raw_null"], n_raw_empty=states["empty"], distinct_semantic=len(after),
                           strategy_counts=serialize(dict(strategies)), field_occurrences=serialize(dict(fields)),
                           rule_sources=serialize(sorted(rules))))
    return output


def downstream(master: pd.DataFrame, candidate: pd.DataFrame, feature_rows: list[dict],
               cfg: dict, output: Path, staged_candidate: Path) -> list[dict]:
    """Compare covered features and call the existing Pharma eligibility rules."""
    rows = []
    for col in master:
        if protected(col, cfg):
            continue
        n = int(master[col].notna().sum())
        rows.append(dict(study="Step 10", feature=col, observed_before=n, observed_after=n,
                         eligibility_before=None, eligibility_after=None, status="original retained"))
    for feature in feature_rows:
        rows.append(dict(study="Step 10", feature=feature["name"], observed_before=0,
                         observed_after=int(candidate[feature["name"]].notna().sum()),
                         eligibility_before=None, eligibility_after=feature["allowed_use"], status="new candidate"))
    if cfg["validation"].get("profile_pharma", True):
        path = cfg["paths"]
        script = Path(path["obj1_root"]) / "src/studies/pharma/00_profile_labs.py"
        program = """
import importlib.util, pathlib, sys
import pandas as pd
script, original, candidate, context, registry, output, graph_output = sys.argv[1:]
sys.path.insert(0, str(pathlib.Path(script).parents[3]))
spec = importlib.util.spec_from_file_location("sjd_pharma_profile", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
ctx = pd.read_parquet(context) if pathlib.Path(context).is_file() else None
rules = module.load_registry(pathlib.Path(registry))
profiles, graph = [], []
for source in (original, candidate):
    frame = pd.read_parquet(source)
    profile = module.profile_labs(frame, ctx, registry=rules)
    if profile.empty:
        selection = pd.DataFrame(columns=["feature", "n_nonmissing_baseline", "included", "exclusion_reason"])
    else:
        selection = module.exploratory_lab_catalog(frame, ctx, profile, rules)
    profiles.append(profile)
    graph.append(selection[["feature", "n_nonmissing_baseline", "included", "exclusion_reason"]])

def write_comparison(before, after, key, filename):
    before = before.add_suffix("_before")
    after = after.add_suffix("_after")
    before.merge(after, left_on=key + "_before", right_on=key + "_after", how="outer",
                 validate="one_to_one").to_csv(filename, index=False)

write_comparison(*profiles, "lab", output)
write_comparison(*graph, "feature", graph_output)
"""
        result = subprocess.run([sys.executable, "-c", program, str(script), path["integrated"], str(staged_candidate),
                                 path["context"] or "", path["pharma_registry"], str(output / "pharma_eligibility.csv"),
                                 str(output / "graph_eligibility.csv")],
                                cwd=path["obj1_root"], capture_output=True)
        if result.returncode:
            raise RuntimeError("Existing Pharma profile failed on the staging inputs")
        pharma = pd.read_csv(output / "pharma_eligibility.csv")
        for row in pharma.to_dict("records"):
            name = row["lab_before"] if pd.notna(row["lab_before"]) else row["lab_after"]
            rows.append(dict(study="Pharma eligibility profile", feature=name,
                             observed_before=row.get("n_episodes_interpretable_before"),
                             observed_after=row.get("n_episodes_interpretable_after"),
                             eligibility_before=row.get("recommended_use_before"),
                             eligibility_after=row.get("recommended_use_after"), status="profile evaluated"))
        for row in pd.read_csv(output / "graph_eligibility.csv").to_dict("records"):
            name = row["feature_before"] if pd.notna(row["feature_before"]) else row["feature_after"]
            rows.append(dict(study="Graph exploratory eligibility", feature=name,
                             observed_before=row["n_nonmissing_baseline_before"],
                             observed_after=row["n_nonmissing_baseline_after"],
                             eligibility_before="included" if row["included_before"] is True else row["exclusion_reason_before"],
                             eligibility_after="included" if row["included_after"] is True else row["exclusion_reason_after"],
                             status="catalog evaluated"))
    else:
        rows.extend(dict(study=study, feature=None, status="not run")
                    for study in ("Pharma eligibility profile", "Graph exploratory eligibility"))
    rows.extend(dict(study=study, feature=None, status="not run") for study in ("Pharma models", "Graph models", "Economic"))
    return rows


def summarize(master: pd.DataFrame, variables: list[dict], dictionary: list[dict],
              observations: list[dict], extractions: list[dict], features: list[dict]) -> dict:
    """Return denominators and disjoint primary strategies from validated rows."""
    families, strategies, field_strategies = Counter(), Counter(), Counter()
    field_occurrences = Counter()
    for row in dictionary:
        strategies[row["strategy"]] += 1
        for item in {(e["field"], e["strategy"]) for e in row["evidence"]}:
            field_strategies[" / ".join(item)] += 1
        for field in SEMANTIC_FACTS:
            if row[field] not in (None, []):
                field_occurrences[field] += row["observations"]
    for row in observations:
        families[row["source_grain"]] += 1
    episode_rows = [row for row in dictionary if row["observations"] > row["laboratory_source_events"]]
    return dict(patients=int(master[KEYS[0]].nunique()), episodes=len(master), original_columns=len(master.columns),
                variables_processed=sum(row["policy"] == "eligible" for row in variables),
                variables_protected=sum(row["policy"] != "eligible" for row in variables),
                unique_raw_pairs=len({(row["variable"], row["raw_type"], row["raw_json"]) for row in episode_rows}),
                unique_mappings=len(dictionary), episode_context_keys=len(episode_rows), observations_by_grain=dict(families),
                laboratory_nonempty_text_occurrences=sum(row["observations"] - row["laboratory_source_events"] for row in episode_rows
                    if row["variable"].startswith("lab__") and row["variable"].endswith("__text") and
                    isinstance(raw := json.loads(row["raw_json"]), str) and raw.strip()),
                primary_strategy_mappings=dict(strategies), field_strategy_mappings=dict(field_strategies),
                field_observations=dict(field_occurrences), new_candidate_features=len(features),
                extraction_status=dict(Counter(row["status"] for row in extractions)),
                clinical_evaluation="not performed", technical_invariants="passed")
