"""Versioned local knowledge corpus and compatible hybrid retrieval."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import re
import tempfile

import numpy as np
from openpyxl import load_workbook
from rank_bm25 import BM25Okapi
from rapidfuzz.fuzz import ratio

from .parsers import normalize
from .settings import digest, file_hash, read_yaml, serialize, write_json
from .inventory import clean

TOKENS = re.compile(r"[a-z0-9_]+")
CODEBOOK_FIELDS = (
    "SHEET_NAME", "PROTOCOLNUMBER", "FORM_NAME", "FORM_NAME__QUESTION_NAME", "QUESTIONID", "VERSION",
    "QUESTION_NAME", "QUESTION_ALIAS", "QUESTION_TEXT", "DISPLAY", "CODEVALUE", "DEFAULTVALUE",
    "UNANSWEREDVALUE", "ANSWER_RANGE", "ANSWER_FORMAT", "SOURCE_STUDY", "SELECTION_RULE",
)


def tokens(text: str) -> list[str]:
    """Tokenize canonical lookup text."""
    return TOKENS.findall(normalize(text).replace("__", " ").replace("_", " "))


def document(source: str, version: str, locator: str, title: str, aliases: list[str], text: str,
             **metadata) -> dict:
    """Create a content-addressed document with explicit applicability."""
    doc = dict(source=source, source_version=version, source_locator=locator, concept_id=None, variable_names=[],
               assay_key=None, method=None, title=title, aliases=aliases, text=text, allowed_values=[],
               units=[], rule_ids=[], family=None, approved=False)
    doc.update(metadata)
    checksum = digest(doc)
    return dict(doc, doc_id=checksum, sha256=checksum)


def codebook_documents(path: Path, version: str) -> list[dict]:
    """Keep code and display text together by exact form/question revision."""
    workbook = load_workbook(path, read_only=True, data_only=True)
    groups = defaultdict(list)
    try:
        rows = iter(workbook["Consolidated_Codebook"].values)
        headers = next(rows)
        if not set(CODEBOOK_FIELDS).issubset(headers):
            raise ValueError("Codebook is missing required metadata columns")
        for ordinal, values in enumerate(rows, start=2):
            row = {name: clean(values[headers.index(name)]) for name in CODEBOOK_FIELDS}
            if not any(value is not None for value in row.values()):
                continue
            key = tuple(row[name] for name in ("PROTOCOLNUMBER", "FORM_NAME", "QUESTIONID", "VERSION"))
            if key[2] is None:
                key += (row["QUESTION_NAME"], ordinal if row["QUESTION_NAME"] is None else None)
            groups[key].append(row)
    finally:
        workbook.close()
    docs = []
    for key, rows in groups.items():
        aliases = sorted({str(row[name]) for row in rows
                          for name in ("QUESTION_ALIAS", "QUESTION_NAME", "FORM_NAME__QUESTION_NAME") if row[name]})
        title = str(rows[0]["QUESTION_TEXT"] or rows[0]["QUESTION_NAME"] or "Codebook question")
        allowed = [{name: row[name] for name in ("DISPLAY", "CODEVALUE", "UNANSWEREDVALUE", "ANSWER_RANGE")}
                   for row in rows]
        docs.append(document("codebook", version, serialize(key), title, aliases, serialize(rows), allowed_values=allowed))
    if not docs:
        raise ValueError("Codebook sheet has no usable rows")
    return docs


def documents(cfg: dict, manifest: dict) -> list[dict]:
    """Assemble metadata, naming context and approved extraction grammar."""
    paths, resources = cfg["paths"], manifest["resources"]
    docs = codebook_documents(Path(paths["codebook"]), resources["codebook"]["sha256"])
    registry = read_yaml(Path(paths["pharma_registry"]))
    for name, item in registry["concepts"].items():
        stem = item.get("preferred_lab_stem")
        if not stem:
            continue
        related = item.get("related_not_equivalent", [])
        aliases = [stem, name]
        docs.append(document("pharma_registry", registry["registry_version"], name, name, aliases,
                             serialize(dict(note=item.get("note"), related_not_equivalent=related,
                                            review_status=item.get("clinical_equivalence_status"))),
                             concept_id=stem.removeprefix("lab__"),
                             variable_names=[stem + "__text", stem + "__value"]))
    version = cfg["normalization"]["version"]
    for family, concepts in cfg["concepts"]["narrative"].items():
        text = {
            "immunofixation": "Record only literal findings and polarity. Keep immunoglobulin class, light chain, pattern and historical scope separate. A faint light chain does not establish monoclonality.",
            "cryoglobulin": "Keep individual reading time, result and historical type separate. A planned reading has no result. Preserve the percent operator. A censored amount does not establish positivity.",
        }[family]
        docs.append(document("local_rules", version, f"narrative/{family}", family, concepts, text,
                             family=family, approved=True, rule_ids=[f"{version}:{family}"]))
    protocol = resources.get("protocols", {})
    if protocol.get("status") == "available":
        path = Path(protocol["path"])
        files = sorted(path.rglob("*.yaml")) if path.is_dir() else [path]
        for file in files:
            for entry in read_yaml(file)["documents"]:
                docs.append(document("local_protocol", protocol["version"], entry["rule_id"], entry["title"],
                                     entry["aliases"], entry["text"], concept_id=entry["concept_id"], approved=True,
                                     assay_key=entry["assay_key"], method=entry.get("method"),
                                     units=entry.get("units", []), rule_ids=[entry["rule_id"]]))
    return sorted({doc["doc_id"]: doc for doc in docs}.values(), key=lambda row: row["doc_id"])


class SapBERT:
    """Encode short aliases locally using the full-text checkpoint's CLS vector."""

    def __init__(self, cfg: dict):
        import torch
        from transformers import AutoModel, AutoTokenizer

        path = cfg["paths"]["embedding_model"]
        self.torch = torch
        self.params = cfg["retrieval"]
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        self.model = AutoModel.from_pretrained(path, local_files_only=True, trust_remote_code=False).eval()

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return unit-normalized float32 embeddings."""
        output = []
        batch = self.params["embedding_batch_size"]
        with self.torch.inference_mode():
            for start in range(0, len(texts), batch):
                encoded = self.tokenizer(texts[start:start + batch], padding=True, truncation=True,
                                         max_length=self.params["sapbert_max_tokens"], return_tensors="pt")
                vectors = self.model(**encoded).last_hidden_state[:, 0, :]
                output.append(self.torch.nn.functional.normalize(vectors).cpu().numpy())
        return np.concatenate(output).astype("float32")

    def count(self, text: str) -> int:
        """Count tokens for the retrieval context budget."""
        return len(self.tokenizer.encode(text, add_special_tokens=False, verbose=False))


class KnowledgeIndex:
    """Search a persisted local corpus using BM25, SapBERT, and RRF."""

    def __init__(self, cfg: dict, manifest: dict, docs: list[dict], directory: Path, encoder=None):
        self.cfg = cfg["retrieval"]
        self.docs = docs
        self.encoder = SapBERT(cfg) if encoder is None else encoder
        self.cache = {}
        self.queries = {}
        self.aliases = [alias for doc in docs for alias in dict.fromkeys(doc["aliases"] or [doc["title"]])]
        self.doc_numbers = np.array([i for i, doc in enumerate(docs) for _ in dict.fromkeys(doc["aliases"] or [doc["title"]])])
        corpus_hash = digest(docs)
        identity = dict(corpus=corpus_hash, encoder=manifest["resources"]["sapbert"]["sha256"],
                        rules=cfg["normalization"], config=self.cfg)
        identity_key = digest(identity)
        directory.mkdir(parents=True, exist_ok=True)
        store = directory / identity_key
        if not store.exists():
            with tempfile.TemporaryDirectory(dir=directory, prefix=".index-") as temp:
                staging = Path(temp) / "index"
                staging.mkdir()
                lexical = [tokens(alias) or ["_"] for alias in self.aliases]
                np.save(staging / "dense.npy", self.encoder.encode(self.aliases), allow_pickle=False)
                write_json(staging / "lexical.json", lexical)
                write_json(staging / "manifest.json", dict(identity, doc_ids=[row["doc_id"] for row in docs],
                    doc_numbers=self.doc_numbers.tolist(),
                    hashes={name: file_hash(staging / name)
                            for name in ("dense.npy", "lexical.json")}))
                try:
                    staging.rename(store)
                except OSError:
                    if not store.is_dir():
                        raise
        cached = json.loads((store / "manifest.json").read_text())
        if (cached["doc_ids"] != [row["doc_id"] for row in docs] or cached["doc_numbers"] != self.doc_numbers.tolist()
                or not all(file_hash(store / name) == sha for name, sha in cached["hashes"].items())):
            raise ValueError("Knowledge index failed content verification")
        self.dense = np.load(store / "dense.npy", allow_pickle=False)
        self.bm25 = BM25Okapi(json.loads((store / "lexical.json").read_text()))
        self.path = store

    def measure(self, texts: list[str]) -> None:
        """Encode query terms in batches without embedding patient narratives."""
        keys = sorted(set(texts) - self.queries.keys())
        if keys:
            self.queries.update(zip(keys, self.encoder.encode(keys), strict=True))

    def retrieve(self, row: dict) -> list[dict]:
        """Rank compatible candidates and reserve room for exact documents."""
        context = json.loads(row["context_json"])
        variable, concept = row["variable"], row["concept"]
        query = concept.replace("_", " ")
        cache_key = (variable, concept, row["family"], serialize({name: context[name] for name in ("assay_id", "method", "unit")}))
        if cache_key in self.cache:
            return self.cache[cache_key]
        allowed, exact = set(), set()
        equivalents = {variable, concept, "lab__" + concept}
        for i, doc in enumerate(self.docs):
            if doc["assay_key"] is not None and doc["assay_key"] != context["assay_id"]:
                continue
            if doc["method"] is not None and doc["method"] != context["method"]:
                continue
            if doc["units"] and context["unit"] not in doc["units"]:
                continue
            match = doc["concept_id"] == concept or variable in doc["variable_names"] or bool(equivalents & set(doc["aliases"]))
            family = doc["source"] == "local_rules" and doc["family"] == row["family"]
            if match or family:
                allowed.add(i)
                if match:
                    exact.add(i)
        if not allowed:
            return []
        self.measure([query])
        lex, dense = self.bm25.get_scores(tokens(query)), self.dense @ self.queries[query]
        best = defaultdict(lambda: [-float("inf"), -float("inf")])
        for j, i in enumerate(self.doc_numbers):
            if i in allowed:
                best[i][0] = max(best[i][0], lex[j])
                best[i][1] = max(best[i][1], float(dense[j]))
        rankings = []
        for axis, name in enumerate(("lexical_top_k", "dense_top_k")):
            ranked = sorted(best, key=lambda i: (-best[i][axis], self.docs[i]["doc_id"]))[:self.cfg[name]]
            rankings.append({i: j for j, i in enumerate(ranked, start=1)})
        lexical, semantic = rankings

        def rank(i):
            rrf = sum(1 / (self.cfg["rrf_k"] + r[i]) for r in rankings if i in r)
            fuzzy = max(ratio(query, normalize(alias).replace("_", " ")) for alias in self.docs[i]["aliases"] or [""])
            return i not in exact, -rrf, -fuzzy, self.docs[i]["doc_id"]

        selected = []
        for i in sorted(set(lexical) | set(semantic) | exact, key=rank):
            doc = self.docs[i]
            snippet = serialize(dict(id=doc["doc_id"], text=doc["text"], approved=doc["approved"]))
            if self.encoder.count(" ".join(item["snippet"] for item in selected) + " " + snippet) > self.cfg["max_context_tokens"]:
                continue
            selected.append(dict(doc_id=doc["doc_id"], rank=len(selected) + 1,
                                 lexical_rank=lexical.get(i), dense_rank=semantic.get(i), exact=i in exact,
                                 snippet=snippet, source=doc["source"], source_version=doc["source_version"]))
            if len(selected) == self.cfg["merged_top_k"]:
                break
        self.cache[cache_key] = selected
        return selected
