"""Conservative, assay-routed deterministic extraction."""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
import json
import re
import unicodedata

from .schemas import Allele, Evidence, Finding, Medication, Semantic

NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?"
UNIT = r"(?:[%]|/?[A-Za-zµμ\[][^\s,;()<>:]*)"
OP = r"<=|>=|<|>|=|≤|≥"
OPERATORS = {"=": "EQ", "<": "LT", "<=": "LE", "≤": "LE", ">": "GT", ">=": "GE", "≥": "GE"}
SINGLE = re.compile(rf"(?P<operator>{OP})?\s*(?P<number>{NUMBER})\s*(?P<unit>{UNIT})?")
RANGE = re.compile(rf"(?P<low>{NUMBER})\s*(?:-|–|—|\bto\b)\s*(?P<high>{NUMBER})\s*(?P<unit>{UNIT})?")
TITER = re.compile(rf"(?:(?P<left>positive|negative)\s+)?(?P<op>{OP})?\s*(?P<num>\d{{1,9}})\s*:\s*(?P<den>\d{{1,9}})(?:\s*\((?P<right>positive|negative)\))?")
GRADE = re.compile(rf"(?:(?P<descriptor>non[- ]hemolyzed|hemolyzed)\s+)?(?P<grade>trace|[1-4]\+|\+[1-4])\s*(?:\((?P<amount>[^()]*)?(?P<close>\))?)?")
ALLELE = re.compile(r"(?:(?P<locus>[A-Z][A-Z0-9]*)\*)?(?P<fields>\d{2,3}(?::\d{2,3}){0,3})(?P<suffix>[GPNLSQCA])?")
READING = re.compile(r"(?P<status>negative|positive)\s+at\s+(?P<duration>\d{1,4})\s*(?P<unit>days?|d\b|hours?|hrs?|h\b)", re.I)
PLANNED = re.compile(r"(?:reviewed again|recheck|repeat)\s+(?:at|in)\s+(\d{1,4})\s*(days?|d\b|hours?|hrs?|h\b)", re.I)
PERCENT = re.compile(rf"(?P<op>{OP})?\s*(?P<value>{NUMBER})\s*%\s+of\s+cryoprecipitate", re.I)
NARRATIVE_PARTS = re.compile(r"[,;\n]+|\s+\+\s+|\s+and\s+", re.I)
CHAIN = re.compile(r"\b(kappa|lambda)\b", re.I)
IG = re.compile(r"\bIg([GMADE])\b", re.I)


def normalize(text: str) -> str:
    """Normalize lookup text without altering source evidence."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def valid_ucum(unit: str, units: set[str]) -> bool:
    """Recognize exact atoms and simple ratios/products of UCUM atoms."""
    if unit in units:
        return True
    factors = re.split(r"[./]", unit.removeprefix("/"))
    return len(factors) > 1 and all(factor in units for factor in factors) or (
        unit.startswith("/") and len(factors) == 1 and factors[0] in units
    )


def numeric(text: str, ucum: set[str]) -> dict:
    """Parse anchored results and separate exact values, bounds and ranges."""
    if match := RANGE.fullmatch(text):
        try:
            low, high = Decimal(match["low"]), Decimal(match["high"])
        except InvalidOperation:
            return {}
        if low > high:
            return {}
        out = dict(comparator="RANGE", range_low=str(low), range_high=str(high))
    elif match := SINGLE.fullmatch(text):
        operator = OPERATORS[match["operator"] or "="]
        try:
            value = str(Decimal(match["number"]))
        except InvalidOperation:
            return {}
        out = {"comparator": operator, "exact_value" if operator == "EQ" else "limit_value": value}
    else:
        return {}
    if unit := match["unit"]:
        # A suffix must be an exact unit or a structured unit token.
        if not valid_ucum(unit, ucum) and not re.search(r"[%/µμ\[\]]|\d", unit):
            return {}
        out["raw_unit"] = unit
        if valid_ucum(unit, ucum):
            out["ucum_unit"] = unit
    return out



def threshold_result(result: Semantic, op: str, value: str, otherwise: str | None) -> str | None:
    """Return a category only when the entire measured interval supports it."""
    threshold = Decimal(value)
    if not threshold.is_finite() or op not in {"LT", "LE", "GT", "GE"}:
        raise ValueError("Protocol threshold requires a finite Decimal and a directional comparator")
    if op in {"LT", "LE"}:
        # Negating coordinates simplifies comparison against a lower threshold.
        direction = -1
    else:
        direction = 1
    exact, bound, lower, upper = [Decimal(v) if v is not None else None for v in
                                   (result.exact_value, result.limit_value, result.range_low, result.range_high)]
    interval = {
        "EQ": (exact, True, exact, True), "RANGE": (lower, True, upper, True),
        "LT": (None, False, bound, False), "LE": (None, False, bound, True),
        "GT": (bound, False, None, False), "GE": (bound, True, None, False),
    }[result.comparator]
    low, low_closed, high, high_closed = interval if direction == 1 else (interval[2], interval[3], interval[0], interval[1])
    if direction == -1:
        low = None if low is None else low.copy_negate()
        high = None if high is None else high.copy_negate()
        threshold = threshold.copy_negate()
    inclusive = op in {"GE", "LE"}
    if low is not None and (low > threshold or low == threshold and (inclusive or not low_closed)):
        return "positive"
    if high is not None and (high < threshold or high == threshold and (not inclusive or not high_closed)):
        return otherwise
    return None


def applicable_protocol(vocab: dict, concept: str, context: dict, unit: str | None) -> dict | None:
    """Select an unambiguous assay-specific protocol by exact context."""
    candidates = [rule for rule in vocab.get("protocol_rules", []) if rule["concept_id"] == concept
                  and rule["assay_key"] == context["assay_id"] and rule["method"] == context["method"]
                  and rule.get("protocol_version") == context["relevant_protocol_version"]
                  and (not rule.get("units") or unit in rule["units"])]
    return candidates[0] if len(candidates) == 1 else None

def new_finding(entity: str, quote: str, **fields) -> Finding:
    """Construct a finding with unsupported fields left null."""
    payload = dict.fromkeys(Finding.model_fields)
    payload.update(entity=entity, quote=quote, assertion="present", temporality="unspecified", document_ids=[])
    payload.update(fields)
    return Finding.model_validate(payload)


def narrative(text: str, family: str) -> list[Finding]:
    """Extract anchored readings and explicitly named laboratory findings."""
    results = []
    if family == "cryoglobulin":
        for match in READING.finditer(text):
            prefix = text[: match.start()]
            if re.search(r"\b(?:not|no)\s+$", prefix, re.I):
                continue
            duration = int(match["duration"]) * (24 if match["unit"].lower().startswith("d") else 1)
            historical = re.search(r"\bpreviously\b[^.;,]*$", prefix, re.I)
            results.append(new_finding("cryoglobulin", match[0], incubation_hours=duration,
                                       assertion="present" if match["status"].lower() == "positive" else "absent",
                                       temporality="historical" if historical else "current"))
        for match in PLANNED.finditer(text):
            duration = int(match[1]) * (24 if match[2].lower().startswith("d") else 1)
            results.append(new_finding("cryoglobulin", match[0], incubation_hours=duration,
                                       assertion="uncertain", temporality="planned"))
        for match in PERCENT.finditer(text):
            operator = OPERATORS[match["op"] or "="]
            explicit_presence = operator == "EQ" and Decimal(match["value"]) > 0
            results.append(new_finding("cryoprecipitate", match[0], percentage=str(Decimal(match["value"])),
                                       comparator=operator, assertion="present" if explicit_presence else "uncertain"))
    absence = re.fullmatch(r"no anomalous immunoglobulins[.]?", text.strip(), re.I)
    if absence:
        results.append(new_finding("anomalous_immunoglobulins", absence[0], assertion="absent"))
    historical = False
    blocked_scope = False
    for part in NARRATIVE_PARTS.split(text):
        clause = part.strip()
        norm = normalize(clause)
        if re.search(r"\b(?:previously|historical|prior)\b", norm):
            historical = True
        if re.search(r"\b(?:currently|now)\b", norm):
            historical = False
            blocked_scope = False
        if re.search(r"\b(?:no|not|without|cannot exclude|possible|suspected)\b", norm):
            blocked_scope = True
        # The grammar is deliberately closed. Negation and speculation outside it
        # require surrounding context and are handled by the evidence verifier.
        explicit = re.fullmatch(
            r"(?:(?:previously identified as|currently|now)\s+)?"
            r"(?:(?:faint|trace)\s+)?(?:(?:increased|increase in)\s+)?"
            r"(?:(?:monoclonal|oligoclonal|polyclonal)\s+)?(?:(?:free)\s+)?"
            r"(?:ig[gmade](?:\s+(?:kappa|lambda))?|kappa|lambda)(?:\s+(?:light chains?|band))?[.]?",
            norm,
        )
        if explicit and not blocked_scope:
            ig, chain = IG.search(clause), CHAIN.search(clause)
            pattern = re.search(r"\b(monoclonal|oligoclonal|polyclonal)\b", norm)
            intensity = re.search(r"\b(faint|trace)\b", norm)
            # Carry historical scope through the same sentence and retain its exact text.
            quote = text if historical and "previously" not in norm and "prior" not in norm else clause
            results.append(new_finding("immunoglobulin" if ig else "light_chain", quote,
                                       immunoglobulin_class=f"Ig{ig[1].upper()}" if ig else None,
                                       light_chain=chain[1].lower() if chain else None,
                                       pattern=pattern[1] if pattern else None,
                                       intensity_descriptor=intensity[1] if intensity else None,
                                       temporality="historical" if historical else "unspecified"))
    return results


def hla(text: str, locus: str, ipd: dict[str, set[str]]) -> list[Allele]:
    """Retain reported resolution, ambiguity and missing alleles."""
    result = []
    for i, token in enumerate(re.split(r"\s*[,;/|]\s*", text)):
        token = token.strip()
        matched = ALLELE.fullmatch(token.upper())
        if token != "*" and (not matched or matched["locus"] not in (None, locus)):
            return []
        allele = f"{locus}*{matched['fields']}{matched['suffix'] or ''}" if matched else None
        count = len(matched["fields"].split(":")) if matched else None
        # A bare two-digit group conveys only what was reported.
        lookup = allele if count and count > 1 else None
        validated = next((kind for kind, names in ipd.items() if lookup in names), None)
        result.append(Allele(locus=locus, reported=token, allele=allele if lookup else None,
                             resolution_fields=count, suffix=matched["suffix"] if matched else None,
                             source_order=i, ipd_match=validated))
    return result


def medications(raw: str, variable: str, cfg: dict, rxnorm: dict[str, str]) -> list[Medication]:
    """Split the Step 08 name list while preserving drug combinations."""
    source, evidence, window = cfg["concepts"]["medications"][variable]
    entities = []
    for token in raw.split(cfg["concepts"]["medication_delimiter"]):
        if not (name := token.strip()):
            continue
        dose = re.fullmatch(rf"(?P<name>.+?)\s+(?P<dose>{NUMBER})\s*(?P<unit>mg|mcg|µg|g|mL|IU)(?:\s+(?P<route>oral|PO|IV|IM|SC|subcutaneous))?(?:\s+(?P<freq>daily|weekly|BID|TID|QD|QHS))?", name, re.I)
        local_name = normalize(dose["name"] if dose else name)
        entities.append(Medication(reported=token, name=local_name, source=source, documentary_evidence=evidence,
                                   window=window, quote=name, dose=dose["dose"] if dose else None,
                                   unit=dose["unit"] if dose else None, route=dose["route"] if dose else None,
                                   frequency=dose["freq"] if dose else None, rxcui=rxnorm.get(local_name),
                                   temporal_availability="retrospective" if window in {"after_episode_1y", "nearby_1y"} else "unknown"))
    return entities


def parse(row: dict, cfg: dict, vocab: dict) -> Semantic:
    """Extract fields that have exact source evidence and an applicable route."""
    raw, family, concept = json.loads(row["raw_json"]), row["family"], row["concept"]
    context = json.loads(row["context_json"])
    rules = cfg["normalization"]
    ucum = vocab.get("ucum_atoms") or vocab["ucum"]
    result = Semantic()
    if raw is None:
        return Semantic(strategy="identity", interpretation_note="Source null.")
    if isinstance(raw, str) and not raw.strip():
        return Semantic(strategy="whitespace_to_null", interpretation_note="Whitespace-only source string retained.")
    text = str(raw).strip()
    norm = normalize(text)

    def add(values: dict, strategy: str, quote: str = text, source: str = "raw_text",
            rule: str = rules["version"]) -> None:
        if not values:
            return
        if result.strategy == "unchanged":
            result.strategy = strategy
        for key, value in values.items():
            if getattr(result, key) not in (None, []):
                continue
            setattr(result, key, value)
            result.evidence.append(Evidence(field=key, quote=quote, source=source, strategy=strategy,
                                            rule=rule))

    lab_family = family not in {"derived", "medication", "other"}
    if lab_family and isinstance(raw, str):
        if norm in rules["availability"]:
            add({"result_availability": rules["availability"][norm]}, "exact_dictionary")
        if norm in rules["specimen"]:
            add({"specimen_state": rules["specimen"][norm]}, "exact_dictionary")
    if family == "hla" and result.result_availability is None and result.specimen_state is None:
        alleles = hla(text, concept.removeprefix("hla_").upper(), vocab["ipd"])
        if alleles:
            add({"alleles": alleles}, "hla_parser")
    if family == "medication" and isinstance(raw, str):
        if entities := medications(raw, row["variable"], cfg, vocab["rxnorm"]):
            add({"medications": entities}, "medication_entity_linking")
    if family == "ana":
        if titer := TITER.fullmatch(norm):
            if int(titer["num"]) > 0 and int(titer["den"]) > 0:
                add({"titer_numerator": int(titer["num"]), "titer_denominator": int(titer["den"]),
                     "titer_comparator": OPERATORS[titer["op"] or "="]}, "titer_parser")
                if (qual := titer["left"] or titer["right"]) and not (titer["left"] and titer["right"] and titer["left"] != titer["right"]):
                    add({"qualitative_result": qual}, "titer_parser")
        patterns = [item.strip() for item in re.split(r"[,;/]", norm)]
        if concept.endswith("_pattern") and set(patterns).issubset(rules["ana_patterns"]):
            add({"pattern_terms": patterns}, "exact_dictionary")
    if family in {"ana", "qualitative", "urine_graded", "cryoglobulin"} and norm in rules["qualitative"]:
        add({"qualitative_result": rules["qualitative"][norm]}, "exact_dictionary")
    if family == "urine_nominal":
        color = normalize(text.replace("-", " "))
        if color in cfg["concepts"]["urine_nominal"][concept]:
            add({"category": color}, "exact_dictionary")
    if family == "urine_microscopy" and norm in rules["microscopy"]:
        add({"category": rules["microscopy"][norm]}, "exact_dictionary")
    if family == "urine_graded" and (grade := GRADE.fullmatch(norm)):
        value = grade["grade"]
        value = value[1] + "+" if value.startswith("+") else value
        add({"ordinal_grade": value}, "ordinal_parser")
        if value == "trace":
            add({"qualitative_result": "trace"}, "ordinal_parser")
        if grade["descriptor"]:
            add({"descriptor": grade["descriptor"].replace("-", " ")}, "ordinal_parser")
        if grade["amount"] and grade["close"]:
            original = text[text.index("(") + 1:text.rindex(")")].strip()
            add(numeric(original, ucum), "numeric_parser", original)
        elif grade["amount"]:
            result.interpretation_note = "Incomplete concentration expression retained."
    if lab_family and family != "hla" and result.strategy == "unchanged":
        add(numeric(text, ucum), "numeric_parser")
    if lab_family and family != "hla" and result.comparator and not result.raw_unit and context["unit"]:
        unit = str(context["unit"])
        add({"raw_unit": unit}, "numeric_parser", unit, "context.unit")
        if valid_ucum(unit, ucum):
            add({"ucum_unit": unit}, "numeric_parser", unit, "context.unit")
    protocol = applicable_protocol(vocab, concept, context, result.raw_unit or context["unit"]) if lab_family else None
    if protocol:
        if result.comparator and result.qualitative_result is None and (cutoff := protocol.get("positive_threshold")):
            if label := threshold_result(result, cutoff["operator"], cutoff["value"], cutoff.get("otherwise")):
                add({"qualitative_result": label}, "exact_dictionary", rule=protocol["rule_id"])
        if scale := protocol.get("ordinal_scale"):
            grade = result.ordinal_grade or result.qualitative_result
            if grade in scale:
                add({"ordinal_score": scale[grade]}, "ordinal_parser", rule=protocol["rule_id"])
    if family in {"cryoglobulin", "immunofixation"}:
        findings = narrative(text, family)
        if findings:
            add({"findings": findings}, "exact_dictionary")
        for field in ("observation_comment", "observation_note"):
            comment = context.get(field)
            if not isinstance(comment, str) or not comment.strip():
                continue
            for finding in narrative(comment.strip(), family):
                if finding not in result.findings:
                    if result.strategy == "unchanged":
                        result.strategy = "exact_dictionary"
                    result.findings.append(finding)
                    result.evidence.append(Evidence(field="findings", quote=finding.quote,
                                                    source=f"context.{field}", strategy="exact_dictionary",
                                                    rule=rules["version"]))
    if result.result_availability is None and any(
        getattr(result, field) for field in ("comparator", "qualitative_result", "titer_denominator",
                                             "pattern_terms", "ordinal_grade", "category", "alleles", "findings")
    ) and result.specimen_state is None:
        add({"result_availability": "reported"}, "exact_dictionary")
    if not result.interpretation_note:
        if result.comparator in {"LT", "LE", "GT", "GE", "RANGE"}:
            result.interpretation_note = ("Bound or interval retained." if result.qualitative_result else
                                          "Bound or interval retained. No assay rule determined a qualitative result.")
        elif result.strategy == "unchanged":
            result.interpretation_note = "No supported interpretation for this source value."
    return Semantic.model_validate(result.model_dump())
