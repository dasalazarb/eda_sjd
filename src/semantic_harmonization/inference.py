"""Offline structured GPT-OSS extraction with source-span verification."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re

from pydantic import ValidationError

from .parsers import narrative, new_finding, normalize
from .schemas import Evidence, Extraction, Finding, Semantic
from .settings import serialize

NARRATIVE = {"cryoglobulin", "immunofixation"}
TEMPORAL_QUAL = re.compile(r"\b(previously|formerly|historical)\s+(positive|negative)\s*[,;]?\s*(?:and\s+)?\b(now|currently)\s+(positive|negative)\b", re.I)
ENTITY_SPAN = re.compile(r"\b(?:(?P<intensity>faint|trace)\s+)?(?:(?P<pattern>monoclonal|oligoclonal|polyclonal)\s+)?(?:(?:free)\s+)?(?:(?P<ig>ig[gmade])(?:\s+(?P<chain>kappa|lambda))?|(?P<light>kappa|lambda))\b", re.I)
NEGATION = re.compile(r"\b(?:no|not|without|absent|negative for)\b", re.I)
UNCERTAINTY = re.compile(r"\b(?:possible|suspected|questionable|cannot exclude|may represent)\b", re.I)


def force_offline() -> None:
    """Disable remote model fetching and runtime telemetry before imports."""
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                      VLLM_NO_USAGE_STATS="1", DO_NOT_TRACK="1", WANDB_MODE="disabled")


def needs_extraction(row: dict, semantic: Semantic) -> bool:
    """Route narrative lab values and comments while skipping trivial tokens."""
    raw = json.loads(row["raw_json"])
    context = json.loads(row["context_json"])
    comments = [context.get(field) for field in ("observation_comment", "observation_note")]
    if row["family"] not in NARRATIVE or not isinstance(raw, str):
        return False
    if any(isinstance(comment, str) and len(comment.split()) > 2 for comment in comments):
        return True
    return len(raw.split()) > 2 and semantic.result_availability not in {"not_performed", "pending"}


def segments(row: dict) -> list[tuple[str, str]]:
    """Return exact source segments with stable field provenance."""
    context = json.loads(row["context_json"])
    values = [("raw_text", json.loads(row["raw_json"]))]
    values.extend((f"context.{field}", context.get(field)) for field in ("observation_comment", "observation_note"))
    return [(source, value) for source, value in values if isinstance(value, str) and value]


def supports(finding: Finding, text: str, family: str) -> bool:
    """Verify polarity, entity attributes and scope against the surrounding text."""
    if finding.quote not in text or text.count(finding.quote) != 1:
        return False
    deterministic = narrative(text, family)
    for match in TEMPORAL_QUAL.finditer(text) if family == "cryoglobulin" else ():
        for position, temporal in ((2, "historical"), (4, "current")):
            deterministic.append(new_finding("cryoglobulin", match[0], temporality=temporal,
                                             assertion="present" if match[position].lower() == "positive" else "absent"))
    clinical = finding.model_dump(exclude={"document_ids", "quote"})
    if any(finding.quote in verified.quote and clinical == verified.model_dump(exclude={"document_ids", "quote"})
           for verified in deterministic):
        return True
    if finding.entity not in {"immunoglobulin", "light_chain"}:
        return False
    start = text.index(finding.quote)
    before = re.split(r"[.;\n]", text[:start])[-1]
    after = re.split(r"[.;\n]", text[start + len(finding.quote):], maxsplit=1)[0]
    clause = before + finding.quote + after
    candidates = list(ENTITY_SPAN.finditer(finding.quote))
    if len(candidates) != 1:
        return False
    match = candidates[0]
    norm = normalize(clause)
    timeline = re.findall(r"\b(previously|historical|prior|currently|now|planned)\b", norm)
    if len(set(timeline)) > 1 or re.search(r"\b(?:but|however|versus|compared)\b", norm):
        return False
    if not timeline and re.search(r"\b(?:previously|historical|prior|currently|now|planned)\b", text, re.I):
        return False
    temporality = {"previously": "historical", "historical": "historical", "prior": "historical",
                   "currently": "current", "now": "current", "planned": "planned"}.get(timeline[0] if timeline else "", "unspecified")
    assertion = "uncertain" if UNCERTAINTY.search(norm) else "absent" if NEGATION.search(norm) else "present"
    if len(list(ENTITY_SPAN.finditer(clause))) != 1:
        return False
    expected = new_finding(
        "immunoglobulin" if match["ig"] else "light_chain", finding.quote,
        assertion=assertion, temporality=temporality,
        immunoglobulin_class="Ig" + match["ig"][-1].upper() if match["ig"] else None,
        light_chain=(match["chain"] or match["light"] or "").lower() or None,
        pattern=match["pattern"].lower() if match["pattern"] else None,
        intensity_descriptor=match["intensity"].lower() if match["intensity"] else None,
    )
    return expected.model_dump(exclude={"document_ids", "quote"}) == clinical


def validate_extraction(payload: str, row: dict, snippets: list[dict]) -> Extraction:
    """Accept only final structured claims with compatible source spans."""
    extracted = Extraction.model_validate_json(payload)
    allowed = {item["doc_id"] for item in snippets}
    for finding in extracted.findings:
        if not set(finding.document_ids).issubset(allowed):
            raise ValueError("Unknown evidence document")
        if not any(supports(finding, text, row["family"]) for _, text in segments(row)):
            raise ValueError("Finding lacked supported assertion, temporal scope or source span")
    return extracted


def apply_extraction(semantic: Semantic, extracted: Extraction, row: dict, version: str) -> None:
    """Apply verified extraction fields without replacing deterministic evidence."""
    for finding in extracted.findings:
        if any(item.model_dump(exclude={"quote", "document_ids"}) == finding.model_dump(exclude={"quote", "document_ids"})
               and (finding.quote in item.quote or item.quote in finding.quote) for item in semantic.findings):
            continue
        source = next(source for source, text in segments(row) if supports(finding, text, row["family"]))
        if semantic.strategy == "unchanged":
            semantic.strategy = "llm_rag"
        semantic.findings.append(finding)
        semantic.evidence.append(Evidence(field="findings", quote=finding.quote, source=source,
                                          rule=version, strategy="llm_rag"))
        semantic.interpretation_note = None


class LocalModel:
    """Run the local GPT-OSS vLLM backend with schema-constrained final output."""

    def __init__(self, cfg: dict):
        force_offline()
        os.environ["TIKTOKEN_ENCODINGS_BASE"] = str(Path(cfg["paths"]["knowledge_dir"]) / "harmony")
        from openai_harmony import HarmonyEncodingName, Role, load_harmony_encoding
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        self.cfg = cfg
        self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self.role = Role.ASSISTANT
        path = cfg["paths"]["gpt_oss"]
        model = cfg["model"]
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        self.engine = LLM(model=path, tokenizer=path, max_model_len=model["max_context_tokens"],
                          gpu_memory_utilization=model["gpu_memory_utilization"], trust_remote_code=False,
                          seed=model["seed"], disable_log_stats=True)
        self.params = SamplingParams(temperature=model["temperature"], max_tokens=model["max_output_tokens"],
                                     seed=model["seed"], skip_special_tokens=False, structured_outputs=StructuredOutputsParams(json=Extraction.model_json_schema()))

    def messages(self, row: dict, retrieved: list[dict], retry: bool = False) -> list[dict]:
        """Construct a bounded, local-only dialogue from assay context and text."""
        context = json.loads(row["context_json"])
        payload = dict(concept=row["concept"], variable_name=row["variable"], raw_segments=dict(segments(row)),
                       assay_method=context.get("method"), raw_unit=context.get("unit"),
                       reference_information={k: v for k, v in context.items() if k.startswith("reference_")},
                       specimen=context.get("specimen"), schema_version=self.cfg["schema_version"],
                       requested_fields=list(Finding.model_fields),
                       retrieved_documents=[json.loads(item["snippet"]) for item in retrieved])
        if retry:
            payload["validation_reminder"] = "Keep only literal findings with supported assertion, temporal scope and exact source span. Return an empty list if unsupported."
        return [{"role": "system", "content": self.cfg["prompts"]["system"]},
                {"role": "user", "content": serialize(payload)}]

    def chat(self, messages: list[list[dict]]) -> list[str | None]:
        """Pass only fitted dialogues to Harmony and preserve batch alignment."""
        model = self.cfg["model"]
        available = model["max_context_tokens"] - model["max_output_tokens"]
        fitting = [i for i, prompt in enumerate(messages)
                   if len(self.tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, return_dict=False)) <= available]
        result = [None] * len(messages)
        if not fitting:
            return result
        try:
            outputs = self.engine.chat([messages[i] for i in fitting], sampling_params=self.params, use_tqdm=False)
        except RuntimeError as error:
            if "out of memory" not in str(error).lower() or len(messages) == 1:
                raise
            print(f"Reducing local GPT-OSS batch after GPU memory exhaustion ({len(messages)} items).")
            split = len(messages) // 2
            return self.chat(messages[:split]) + self.chat(messages[split:])
        for i, output in zip(fitting, outputs, strict=True):
            final = output.outputs[0]
            if final.finish_reason != "stop":
                result[i] = ""
                continue
            result[i] = self.final_channel(final.token_ids)
        return result

    def final_channel(self, ids: list[int]) -> str:
        """Use the official Harmony parser and retain only one assistant final."""
        from openai_harmony import HarmonyError

        try:
            messages = self.encoding.parse_messages_from_completion_tokens(ids, self.role, strict=False)
        except HarmonyError:
            return ""
        final = [message for message in messages if message.channel == "final"]
        if len(final) != 1:
            return ""
        return "".join(part.text for part in final[0].content if hasattr(part, "text"))

    def smoke(self) -> None:
        """Exercise actual GPU inference, schema, negation and timed readings."""
        examples = [
            ("immunofixation", "No anomalous immunoglobulins.", "absent"),
            ("cryoglobulin", "negative at 24h, positive at 7d", "present"),
            ("cryoglobulin", "not performed", None),
        ]
        items = [(dict(concept=family, variable="lab__" + family + "__text", family=family,
                       raw_json=serialize(raw), context_json="{}"), expected) for family, raw, expected in examples]
        reply = self.chat([self.messages(row, []) for row, _ in items])
        for (row, expected), answer in zip(items, reply, strict=True):
            extracted = validate_extraction(answer or "", row, [])
            if expected is None and extracted.findings or expected and not any(item.assertion == expected for item in extracted.findings):
                raise RuntimeError("GPT-OSS GPU smoke failed an expected clinical assertion")


def extract_batch(model: LocalModel, rows: list[dict], retrieved: dict, retries: int) -> list[dict]:
    """Retry schema or evidence failures once and retain unresolved mappings."""
    pending = rows
    output = {}
    for round_number in range(retries + 1):
        next_pending = []
        messages = [model.messages(row, retrieved[row["mapping_id"]], bool(round_number)) for row in pending]
        for row, answer in zip(pending, model.chat(messages), strict=True):
            mapping_id = row["mapping_id"]
            if answer is None:
                output[mapping_id] = dict(mapping_id=mapping_id, status="context_limit", findings=[])
                continue
            try:
                extracted = validate_extraction(answer, row, retrieved[mapping_id])
            except (ValidationError, ValueError):
                output[mapping_id] = dict(mapping_id=mapping_id, status="unsupported_or_invalid", findings=[])
                next_pending.append(row)
            else:
                output[mapping_id] = dict(mapping_id=mapping_id, status="validated",
                                          findings=[item.model_dump() for item in extracted.findings])
        pending = next_pending
        if not pending:
            break
    return [output[row["mapping_id"]] for row in rows]
