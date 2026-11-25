"""Build the restricted offline report from reconciled stage files."""
from pathlib import Path
import json
import os

import pandas as pd
import pyarrow.parquet as pq
from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import Markup

from .application import SEMANTIC_FACTS
from .inventory import clean
from .settings import MODULE, serialize


def report(output: Path, cfg: dict, manifest: dict) -> None:
    """Render a single local HTML document without episode identifiers."""
    variables = pd.read_csv(output / "variable_comparison.csv").to_dict("records")
    downstream = pd.read_csv(output / "downstream_impact.csv").to_dict("records")
    dictionary = pq.read_table(output / "semantic_dictionary.parquet").to_pylist()
    derived = pq.read_table(output / "derived_comparison.parquet").to_pylist()
    summary = json.loads((output / "validation_summary.json").read_text())
    resources = json.loads((output / "resources_manifest.json").read_text())["resources"]
    keys = ("source_system", "assay_id", "method", "unit", "reference_low", "reference_high",
            "reference_status", "specimen", "relevant_protocol_version")
    details = []
    for row in dictionary:
        context = json.loads(row["context_json"])
        structured = {name: row[name] for name in SEMANTIC_FACTS if row[name] not in (None, [])}
        details.append(dict(variable=row["variable"], family=row["family"], raw=json.loads(row["raw_json"]),
                            raw_type=row["raw_type"], episodes=row["episodes"], patients=row["patients"],
                            observations=row["observations"], source_events=row["laboratory_source_events"],
                            baseline=row["baseline"], context={k: context[k] for k in keys if context.get(k) is not None},
                            structured=structured, strategy=row["strategy"], evidence=row["evidence"],
                            knowledge_document_ids=row["knowledge_doc_ids"], note=row["interpretation_note"]))
    payload = clean(dict(variables=variables, downstream=downstream, details=details, derived=derived, summary=summary,
                         run={key: manifest[key] for key in ("run_id", "started_at", "commit", "schema_version", "inputs", "model", "prompt")},
                         resources={name: {key: item.get(key) for key in ("status", "version", "revision", "sha256", "license")}
                                    for name, item in resources.items()},
                         page_size=cfg["report"]["page_size"]))
    safe = serialize(payload).replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    environment = Environment(loader=FileSystemLoader(MODULE / "templates"), autoescape=select_autoescape(["html"]))
    path = output / "semantic_harmonization_final.html"
    path.write_text(environment.get_template("report.html").render(payload=Markup(safe)))
    os.chmod(path, 0o600)
