"""Prepare pinned local resources and verify their provenance offline."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from urllib.request import urlopen
import xml.etree.ElementTree as ET

from .parsers import normalize
from .settings import digest, file_hash, read_yaml, write_json


def fingerprint(path: Path) -> dict:
    """Record immutable file hashes, relative names, and size."""
    if not path.exists():
        raise FileNotFoundError(path)
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file() and ".cache" not in p.parts)
    if not files:
        raise ValueError(f"Resource is empty: {path}")
    items = [{"name": p.name if path.is_file() else p.relative_to(path).as_posix(), "size": p.stat().st_size,
              "sha256": file_hash(p)} for p in files]
    return {"sha256": digest(items), "size": sum(item["size"] for item in items), "files": items}


def model_info(path: Path, *, gpt: bool) -> dict:
    """Validate model identity, weights, index and tokenizer before loading."""
    config = json.loads((path / "config.json").read_text())
    if gpt and (config.get("model_type") != "gpt_oss" or "GptOssForCausalLM" not in config.get("architectures", [])):
        raise ValueError("Configured GPT-OSS path is not a GPT-OSS causal model")
    if not gpt and config.get("model_type") != "bert":
        raise ValueError("Configured SapBERT path does not contain a BERT encoder")
    if not any((path / name).is_file() for name in ("tokenizer.json", "vocab.txt")):
        raise ValueError(f"Tokenizer assets unavailable in {path}")
    if gpt and not (path / "tokenizer_config.json").is_file():
        raise ValueError("GPT-OSS tokenizer configuration is missing")
    if not list(path.glob("*.safetensors")) and not list(path.glob("pytorch_model*.bin")):
        raise ValueError(f"Model weights unavailable in {path}")
    for index in path.glob("*.index.json"):
        mapping = json.loads(index.read_text()).get("weight_map", {})
        if any(not (path / name).is_file() for name in mapping.values()):
            raise ValueError(f"Model checkpoint is missing a shard listed in {index.name}")
    return dict(model_type=config["model_type"], architectures=config.get("architectures", []),
                quantization=config.get("quantization_config"), revision=config.get("_commit_hash"),
                format="safetensors" if list(path.glob("*.safetensors")) else "pytorch")


def attempt(operation):
    """Retry intermittent public resource operations up to three times."""
    for i in range(3):
        try:
            return operation()
        except Exception:
            if i == 2:
                raise


def git_public(spec: dict, dest: Path, revision: str | None, pins: dict, name: str, pin_path: Path) -> str:
    """Fetch configured public files at one pinned Git revision."""
    url = "https://github.com/" + spec["repository"] + ".git"

    def git(directory: Path, *args: str) -> bytes:
        return subprocess.run(["git", "-C", str(directory), *args], check=True, timeout=180,
                              capture_output=True).stdout

    if not revision:
        revision = attempt(lambda: git(dest.parent, "ls-remote", url, f"refs/heads/{spec['branch']}")).decode().split()[0]
        pins[name] = revision
        write_json(pin_path, pins)
    with tempfile.TemporaryDirectory(dir=dest.parent, prefix=f".{name}-") as staging:
        repo = Path(staging) / "git"
        files = Path(staging) / "files"
        repo.mkdir()
        files.mkdir()
        git(repo, "init", "--quiet")
        attempt(lambda: git(repo, "fetch", "--quiet", "--depth=1", url, revision))
        for relative in spec["files"]:
            target = files / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(git(repo, "show", f"{revision}:{relative}"))
        files.replace(dest)
    return revision


def sapbert(spec: dict, dest: Path, revision: str | None, pins: dict, name: str, pin_path: Path) -> str:
    """Download the pinned SapBERT checkpoint into an atomic directory."""
    from huggingface_hub import HfApi, snapshot_download
    if not revision:
        revision = attempt(lambda: HfApi().model_info(spec["repository"], timeout=30).sha)
        pins[name] = revision
        write_json(pin_path, pins)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent, prefix=".sapbert-") as temporary:
        stage = Path(temporary) / "model"
        attempt(lambda: snapshot_download(repo_id=spec["repository"], revision=revision, local_dir=stage,
                                          allow_patterns=["*.json", "*.txt", "*.md", "*.safetensors", "pytorch_model*.bin"],
                                          etag_timeout=30))
        model_info(stage, gpt=False)
        stage.replace(dest)
    return revision


def prepare(cfg: dict) -> dict:
    """Prepare public resources without reading any patient datasets."""
    paths = cfg["paths"]
    knowledge = Path(paths["knowledge_dir"])
    knowledge.mkdir(parents=True, exist_ok=True, mode=0o700)
    pin_path = knowledge / "resource_revisions.json"
    manifest_path = knowledge / "resources_manifest.json"
    prior = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    pins = json.loads(pin_path.read_text()) if pin_path.is_file() else {}
    resources = {}
    receipts = knowledge / ".receipts"
    receipts.mkdir(exist_ok=True)

    def local(name: str, path: Path, **metadata) -> None:
        resources[name] = dict(path=str(path), status="available", **fingerprint(path), **metadata)

    gpt = Path(paths["gpt_oss"])
    local("gpt_oss", gpt, license="Apache-2.0", url="https://github.com/openai/gpt-oss",
          parser="vLLM Harmony", **model_info(gpt, gpt=True))
    local("codebook", Path(paths["codebook"]), license="NIH authorized local use", parser="openpyxl")
    local("pharma_registry", Path(paths["pharma_registry"]), license="Repository source", parser="PyYAML")
    for name, fetch, dest in (
        ("sapbert", sapbert, Path(paths["embedding_model"])),
        ("ipd", git_public, knowledge / "ipd"),
        ("ucum", git_public, knowledge / "ucum"),
    ):
        spec = cfg["resources"][name]
        receipt = receipts / f"{name}.json"
        prior_item = (json.loads(receipt.read_text()) if receipt.is_file()
                      else prior.get("resources", {}).get(name, {}))
        url = ("https://huggingface.co/" if name == "sapbert" else "https://github.com/") + spec["repository"]
        try:
            revision = pins.get(name)
            if dest.exists():
                if prior_item.get("sha256") != fingerprint(dest)["sha256"] or prior_item.get("revision") != revision or prior_item.get("path") != str(dest):
                    # Hugging Face local_dir contains the commit in each cached download record.
                    snapshots = list((dest / ".cache" / "huggingface" / "download").rglob("*.metadata"))
                    revisions = {p.read_text().splitlines()[0] for p in snapshots if p.stat().st_size}
                    if name != "sapbert" or not prior_item and len(revisions) != 1:
                        raise ValueError(f"Unverified resource at {dest}; point configuration to a verified snapshot or a new destination")
                    if prior_item:
                        raise ValueError(f"Resource contents changed at {dest}; restore verified files or choose a new destination")
                    revision = revisions.pop()
                    pins[name] = revision
                    write_json(pin_path, pins)
            else:
                revision = fetch(spec, dest, revision, pins, name, pin_path)
            version = (dest / "release_version.txt").read_text().strip() if name == "ipd" else revision
            metadata = model_info(dest, gpt=False) if name == "sapbert" else {}
            local(name, dest, url=url, license=spec["license"], parser=name,
                  **(metadata | {"revision": revision, "version": version}))
            write_json(receipt, resources[name])
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError, IndexError) as error:
            if name == "sapbert":
                raise
            resources[name] = dict(path=str(dest), status="unavailable", url=url, license=spec["license"],
                                   reason=type(error).__name__)
    harmony = cfg["resources"]["harmony"]
    harmony_path = knowledge / "harmony" / "o200k_base.tiktoken"
    harmony_path.parent.mkdir(exist_ok=True)
    if not harmony_path.is_file():
        def download_harmony():
            with urlopen(harmony["url"], timeout=30) as reply:
                return reply.read(5_000_000)

        payload = attempt(download_harmony)
        if hashlib.sha256(payload).hexdigest() != harmony["sha256"]:
            raise ValueError("Public Harmony tokenizer vocabulary did not match its official hash")
        with tempfile.NamedTemporaryFile(dir=harmony_path.parent, delete=False) as temporary:
            temporary.write(payload)
        Path(temporary.name).replace(harmony_path)
    if file_hash(harmony_path) != harmony["sha256"]:
        raise ValueError("Local Harmony vocabulary does not match its official hash")
    local("harmony", harmony_path, url=harmony["url"], license=harmony["license"], parser="openai-harmony",
          revision=harmony["sha256"])
    for name, spec in cfg["resources"]["optional"].items():
        metadata = {key: spec[key] for key in ("url", "license", "version")}
        if spec["path"]:
            if not spec["version"]:
                raise ValueError(f"Set the release/version for optional local resource {name}")
            local(name, Path(spec["path"]), parser=name, administrator=spec.get("administrator"), **metadata)
        else:
            resources[name] = dict(status="unavailable", **metadata)
    output = dict(resources=resources, schema=cfg["resources"]["version"],
                  prepared_at=datetime.now(timezone.utc).isoformat())
    write_json(manifest_path, output)
    return output


def verify(cfg: dict) -> dict:
    """Reject stale local resources and any missing core resource."""
    manifest = json.loads((Path(cfg["paths"]["knowledge_dir"]) / "resources_manifest.json").read_text())
    resources = manifest["resources"]
    required = {"gpt_oss": "gpt_oss", "sapbert": "embedding_model", "codebook": "codebook",
                "pharma_registry": "pharma_registry"}
    for name, config_key in required.items():
        item = resources.get(name, {})
        if item.get("status") != "available" or item.get("path") != cfg["paths"][config_key]:
            raise ValueError(f"Prepare the configured {name} before clinical execution")
    harmony = Path(cfg["paths"]["knowledge_dir"]) / "harmony" / "o200k_base.tiktoken"
    if not harmony.is_file() or file_hash(harmony) != cfg["resources"]["harmony"]["sha256"]:
        raise ValueError("Prepare the verified local Harmony vocabulary before running inference")
    for name, item in resources.items():
        if item["status"] == "available" and fingerprint(Path(item["path"]))["sha256"] != item["sha256"]:
            raise ValueError(f"Resource fingerprint changed: {name}")
    for name, spec in cfg["resources"]["optional"].items():
        item = resources.get(name, {})
        if spec["path"] != item.get("path") or spec["version"] != item.get("version"):
            raise ValueError(f"Prepare the configured local resource: {name}")
    return manifest


def vocabularies(manifest: dict) -> dict:
    """Load UCUM codes, IPD exact identities, and unambiguous local RxNorm terms."""
    resources = manifest["resources"]
    vocab = {"ucum": set(), "ucum_atoms": set(), "ipd": {}, "rxnorm": {}}
    if resources["ucum"]["status"] == "available":
        root = ET.parse(Path(resources["ucum"]["path"]) / "ucum-essence.xml").getroot()
        unit_entries = [entry for entry in root if entry.tag.split("}")[-1] in {"base-unit", "unit"}]
        prefixes = [entry.attrib["Code"] for entry in root if entry.tag.endswith("}prefix")]
        vocab["ucum"] = {entry.attrib["Code"] for entry in unit_entries}
        vocab["ucum_atoms"] = vocab["ucum"] | {
            prefix + entry.attrib["Code"] for entry in unit_entries for prefix in prefixes
            if entry.tag.endswith("}base-unit") or entry.get("isMetric") == "yes"
        }
    if resources["ipd"]["status"] == "available":
        root = Path(resources["ipd"]["path"])
        vocab["ipd"]["allele"] = {line.split(",")[1].strip() for line in (root / "Allelelist.txt").read_text().splitlines()
                                  if line and not line.startswith("#") and "," in line}
        for suffix in ("g", "p"):
            elements = [line.split(";") for line in (root / "wmda" / f"hla_nom_{suffix}.txt").read_text().splitlines()
                        if line and not line.startswith("#")]
            vocab["ipd"][f"{suffix}_group"] = {parts[0] + parts[2] for parts in elements if len(parts) > 2 and parts[2]}
    if resources["rxnorm"]["status"] == "available":
        root = Path(resources["rxnorm"]["path"])
        options = list(root.rglob("RXNCONSO.RRF")) if root.is_dir() else [root]
        if len(options) != 1:
            raise ValueError("Expected one RXNCONSO.RRF in the configured RxNorm release")
        matches = defaultdict(set)
        with options[0].open() as stream:
            for line in stream:
                fields = line.rstrip("\n").split("|")
                if len(fields) < 17:
                    raise ValueError("Invalid RXNCONSO.RRF row")
                if fields[1] == "ENG" and fields[11] == "RXNORM" and fields[12] in {"IN", "PIN", "MIN", "BN", "SCD", "SBD"} and fields[16] == "N":
                    matches[normalize(fields[14])].add(fields[0])
        vocab["rxnorm"] = {term: next(iter(codes)) for term, codes in matches.items() if len(codes) == 1}
    protocol = resources.get("protocols", {})
    if protocol.get("status") == "available":
        root = Path(protocol["path"])
        files = sorted(root.rglob("*.yaml")) if root.is_dir() else [root]
        vocab["protocol_rules"] = []
        for source in files:
            for rule in read_yaml(source)["documents"]:
                if not (rule.get("positive_threshold") or rule.get("ordinal_scale")):
                    continue
                if not all(rule.get(key) for key in ("assay_key", "method", "protocol_version", "rule_id", "concept_id")):
                    raise ValueError("Threshold and ordinal rules require exact assay, method and protocol provenance")
                if "positive_threshold" in rule:
                    cutoff = rule["positive_threshold"]
                    if not rule.get("units") or cutoff.get("otherwise") not in {None, "negative"}:
                        raise ValueError("Threshold rules require explicit units and a documented negative complement")
                if "ordinal_scale" in rule and any(type(value) is not int for value in rule["ordinal_scale"].values()):
                    raise ValueError("Ordinal scale entries must be integers")
                vocab["protocol_rules"].append(rule)
    return vocab
