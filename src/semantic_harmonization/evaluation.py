"""Optional independent, grouped holdout evaluation of extraction strategies."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import random

from .inference import extract_batch
from .schemas import Extraction, Semantic


def facts(model: Semantic) -> dict:
    """Read clinical fields without their evidence and processing metadata."""
    result = model.model_dump(exclude={"evidence", "strategy", "interpretation_note"}, exclude_none=True)
    result["findings"] = [item.model_dump(exclude={"quote", "document_ids"}, exclude_none=True) for item in model.findings]
    return {name: value for name, value in result.items() if value != []}


def evaluate(path: Path | None, rows: list[dict], m1: dict, m4: dict, retrieved: dict, model, cfg: dict) -> dict:
    """Evaluate held-out grouped annotations only when independently available."""
    if path is None:
        return {"status": "not performed", "reason": "No independent annotations configured"}
    with path.open() as stream:
        gold = [json.loads(line) for line in stream if line.strip()]
    by_family = defaultdict(lambda: defaultdict(list))
    index = {row["mapping_id"]: row for row in rows}
    seen = set()
    for item in gold:
        key = item["mapping_id"]
        if key in seen or key not in index or index[key]["family"] != item["family"] or not item["group"]:
            raise ValueError("Gold standard has an invalid or duplicate mapping, family or group")
        seen.add(key)
        by_family[item["family"]][item["group"]].append(item)
    rng = random.Random(cfg["validation"]["seed"])
    holdout = []
    for groups in by_family.values():
        names = sorted(groups)
        if len(names) < 2:
            continue
        count = max(1, min(len(names) - 1, round(len(names) * cfg["validation"]["test_fraction"])))
        holdout.extend(item for name in rng.sample(names, count) for item in groups[name])
    if not holdout:
        return {"status": "not performed", "reason": "Independent annotations lack two separate groups within a family"}

    from .inference import apply_extraction, needs_extraction

    m3 = {item["mapping_id"]: m1[item["mapping_id"]].model_copy(deep=True) for item in holdout}
    query = [index[k] for k in m3 if needs_extraction(index[k], m3[k])]
    no_context = {row["mapping_id"]: [] for row in query}
    batch = cfg["model"]["batch_size"]
    for start in range(0, len(query), batch):
        for row in extract_batch(model, query[start:start + batch], no_context, cfg["model"]["max_validation_retries"]):
            k = row["mapping_id"]
            apply_extraction(m3[k], Extraction.model_validate({"findings": row["findings"]}), index[k], cfg["prompts"]["version"])
    scores = defaultdict(Counter)
    for item in holdout:
        key = item["mapping_id"]
        labeled = facts(Semantic.model_validate(item["expected"]))
        for name, predictions in (("M1", m1), ("M3", m3), ("M4", m4)):
            observed = facts(predictions[key])
            for scope in ("all", item["family"]):
                counts = scores[f"{name}/{scope}"]
                weight = index[key]["observations"]
                counts.update(pairs=1, correct=int(labeled == observed), with_fields=int(bool(observed)),
                              occurrences=weight, correct_occurrences=weight * int(labeled == observed))
                for field in (labeled.keys() | observed.keys()) - {"findings"}:
                    counts[f"{field}/n"] += 1
                    counts[f"{field}/match"] += int(labeled.get(field) == observed.get(field))
                canon = lambda fields: Counter(json.dumps(f, sort_keys=True) for f in fields.get("findings", []))
                expected, actual = canon(labeled), canon(observed)
                counts["finding_tp"] += sum((actual & expected).values())
                counts["finding_fp"] += sum((actual - expected).values())
                counts["finding_fn"] += sum((expected - actual).values())
        if "expected_document_ids" in item:
            actual = {hit["doc_id"] for hit in retrieved.get(key, [])}
            expected = set(item["expected_document_ids"])
            scores["M2/retrieval"].update(pairs=1, known_documents=len(expected), retrieved_documents=len(actual),
                                          retrieved_expected=len(actual & expected))
    metrics = {}
    for key, counts in scores.items():
        values = dict(counts)
        if key.startswith("M2/"):
            values["retrieval_recall"] = (counts["retrieved_expected"] / counts["known_documents"]
                                          if counts["known_documents"] else None)
        else:
            values["exact_match"] = counts["correct"] / counts["pairs"]
            values["occurrence_weighted_exact_match"] = counts["correct_occurrences"] / counts["occurrences"]
            values["coverage"] = counts["with_fields"] / counts["pairs"]
            tp, fp, fn = (counts[k] for k in ("finding_tp", "finding_fp", "finding_fn"))
            values.update(finding_precision=tp / (tp + fp) if tp + fp else None,
                          finding_recall=tp / (tp + fn) if tp + fn else None,
                          finding_f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None)
        metrics[key] = values
    return {"status": "performed", "holdout_pairs": len(holdout), "metrics": metrics}
