"""Synthetic regression tests for the semantic candidate and local report."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from openpyxl import Workbook

from src.semantic_harmonization.inference import supports, validate_extraction
from src.semantic_harmonization.inventory import CONTEXT_FIELDS, route
from src.semantic_harmonization.parsers import new_finding, parse
from src.semantic_harmonization.rag import CODEBOOK_FIELDS
from src.semantic_harmonization.run_pipeline import run
from src.semantic_harmonization.settings import MODULE, digest, read_yaml, serialize

VOCAB = {"ucum": {"%", "mg", "dL"}, "ipd": {"g_group": {"DRB1*04:01:01G"}}, "rxnorm": {}}
KEYS = ["patient_id", "clinical_episode_id"]


@pytest.fixture
def cfg():
    configs = MODULE / "configs"
    return read_yaml(configs / "pipeline.yaml") | {
        name: read_yaml(configs / f"{name}.yaml")
        for name in ("concepts", "normalization", "prompts", "resources")
    }


def parsed(cfg, variable, raw, **context):
    concept, family = route(variable, cfg)
    return parse(dict(variable=variable, concept=concept, family=family, raw_json=serialize(raw),
                      context_json=serialize(dict.fromkeys(CONTEXT_FIELDS) | context)), cfg, VOCAB)


@pytest.mark.parametrize("text, comparator, expected, bound", [
    ("<10", "LT", None, "10"), (">1300", "GT", None, "1300"), ("< 0.2", "LT", None, "0.2"),
    ("25.4", "EQ", "25.4", None), ("≤ 2.5e-2", "LE", None, "0.025"),
    ("<3%", "LT", None, "3"), ("0", "EQ", "0", None), ("2025-01-03", None, None, None),
    ("10 reviewed", None, None, None),
])
def test_numeric_grammar(cfg, text, comparator, expected, bound):
    result = parsed(cfg, "lab__anti_ro60__text", text)
    assert (result.comparator, result.exact_value, result.limit_value, result.qualitative_result) == (
        comparator, expected, bound, None
    )


def test_ana_urine_status_and_metadata(cfg):
    ana = parsed(cfg, "lab__ana_hep2_status__text", "<1:80 (Negative)")
    assert (ana.titer_numerator, ana.titer_denominator, ana.titer_comparator, ana.qualitative_result) == (1, 80, "LT", "negative")
    assert parsed(cfg, "lab__ana_status__text", "Positive 1:640").titer_denominator == 640
    assert parsed(cfg, "lab__ana_status__text", "not negative").qualitative_result is None
    assert parsed(cfg, "lab__rheumatoid_factor__text", "not performed").result_availability == "not_performed"
    assert parsed(cfg, "lab__rheumatoid_factor__text", "HEMOLYZED").specimen_state == "hemolyzed"
    assert parsed(cfg, "lab__urine_protein__text", "NORMAL").qualitative_result == "normal"
    assert parsed(cfg, "lab__urine_protein__text", "NEGATIVE").qualitative_result == "negative"
    assert parsed(cfg, "lab__urine_color__text", "Light-Yellow").category == "light yellow"
    assert parsed(cfg, "lab__urine_bacteria__text", "Occassional").category == "occasional"
    assert parsed(cfg, "lab__unknown__text", "Occassional").category is None
    result = parsed(cfg, "lab__urine_protein__text", "1+(30 mg/dL)")
    assert (result.ordinal_grade, result.exact_value, result.raw_unit, result.ucum_unit) == ("1+", "30", "mg/dL", "mg/dL")
    result = parsed(cfg, "lab__urine_protein__text", "+1(30-70)")
    assert (result.ordinal_grade, result.range_low, result.range_high, result.raw_unit) == ("1+", "30", "70", None)
    result = parsed(cfg, "lab__urine_protein__text", "Trace(10-20")
    assert (result.ordinal_grade, result.comparator) == ("trace", None)
    result = parsed(cfg, "lab__urine_blood__text", "Non Hemolyzed Trace")
    assert (result.ordinal_grade, result.descriptor, result.specimen_state) == ("trace", "non hemolyzed", None)


def test_narratives_hla_and_medications(cfg):
    summary = parsed(cfg, "lab__cryoglobulins__text", "negative at 24h, positive at 7d")
    assert [(f.incubation_hours, f.assertion) for f in summary.findings] == [(24, "absent"), (168, "present")]
    summary = parsed(cfg, "lab__cryoglobulins__text", "Negative at 24 hours, reviewed again at 7 days")
    assert [(f.incubation_hours, f.temporality) for f in summary.findings] == [(24, "current"), (168, "planned")]
    summary = parsed(cfg, "lab__cryoglobulins__text", "3% of Cryoprecipitate; Previously identified as IgM Kappa and Polyclonal IgG")
    assert [(f.entity, f.temporality) for f in summary.findings] == [
        ("cryoprecipitate", "unspecified"), ("immunoglobulin", "historical"), ("immunoglobulin", "historical")]
    assert parsed(cfg, "lab__cryoglobulins__text", "<3%").qualitative_result is None
    assert parsed(cfg, "lab__immunofixation__text", "No anomalous immunoglobulins").findings[0].assertion == "absent"
    assert parsed(cfg, "lab__immunofixation__text", "Faint Free Lambda").findings[0].pattern is None
    assert not parsed(cfg, "lab__immunofixation__text", "No IgG and IgM").findings
    carried = parsed(cfg, "lab__immunofixation__text", "SEE BELOW", observation_comment="Faint Free Lambda")
    assert carried.findings[0].light_chain == "lambda"
    assert any(e.source == "context.observation_comment" for e in carried.evidence)
    result = parsed(cfg, "lab__hla_drb1__text", "04:01:01G, *")
    assert [(a.allele, a.suffix, a.ipd_match) for a in result.alleles] == [("DRB1*04:01:01G", "G", "g_group"), (None, None, None)]
    assert len(parsed(cfg, "lab__hla_a__text", "*,*").alleles) == 2
    assert parsed(cfg, "lab__hla_drb__text", "DRB1*04:01:01G").alleles == []
    assert parsed(cfg, "lab__hla_a__text", "07,17").alleles[0].allele is None
    result = parsed(cfg, "medication__btris_order_nearby_1y_names", "drug a / drug b; drug c 10 mg PO daily")
    assert [m.name for m in result.medications] == ["drug a / drug b", "drug c"]
    assert all(m.temporal_availability == "retrospective" and m.documentary_evidence == "order" for m in result.medications)
    assert (result.medications[1].dose, result.medications[1].route) == ("10", "PO")


def test_model_evidence_respects_opposite_polarity_and_temporality():
    text = "previously positive, now negative"
    assert supports(new_finding("cryoglobulin", text, assertion="present", temporality="historical"), text, "cryoglobulin")
    assert supports(new_finding("cryoglobulin", text, assertion="absent", temporality="current"), text, "cryoglobulin")
    assert not supports(new_finding("cryoglobulin", text, assertion="present", temporality="current"), text, "cryoglobulin")
    assert not supports(new_finding("immunoglobulin", "IgG kappa", immunoglobulin_class="IgG", light_chain="kappa"),
                        "No evidence of IgG kappa.", "immunofixation")
    row = dict(family="immunofixation", raw_json='"No anomalous immunoglobulins."', context_json="{}")
    unsupported = new_finding("anomalous_immunoglobulins", "No anomalous immunoglobulins.").model_dump()
    with pytest.raises(ValueError, match="source span"):
        validate_extraction(serialize({"findings": [unsupported]}), row, [])


class Encoder:
    """Deterministic test embedding with the production interface."""
    batches = 0

    def encode(self, terms):
        self.batches += 1
        return np.array([[1.0, 0.0] for _ in terms], dtype="float32")

    def count(self, text):
        return len(text.split())


class Model:
    """Inference double returning only a supported explicit observation."""
    calls = 0

    def __init__(self, cfg):
        self.cfg = cfg

    def smoke(self):
        self.__class__.calls += 1

    def messages(self, row, retrieved, retry):
        return row, retrieved

    def chat(self, messages):
        results = []
        for row, retrieved in messages:
            if row["family"] == "immunofixation":
                assert any(item["source"] == "local_rules" for item in retrieved)
                finding = new_finding("light_chain", "faint lambda", light_chain="lambda", intensity_descriptor="faint")
                results.append(serialize({"findings": [finding.model_dump()]}))
            else:
                results.append('{"findings":[]}')
        return results


def synthetic(tmp_path, cfg):
    paths = cfg["paths"]
    names = {"integrated": "step10.parquet", "context": "context.parquet", "source_events": "events.parquet",
             "codebook": "codebook.xlsx", "knowledge_dir": "knowledge", "pharma_registry": "pharma.yaml",
             "output": "report", "analytic": "analytic", "obj1_root": "obj1"}
    paths.update({name: str(tmp_path / file) for name, file in names.items()})
    cfg["validation"]["profile_pharma"] = False
    cfg["report"]["page_size"] = 1
    master = pd.DataFrame(dict(patient_id=["synthetic-A", "synthetic-A", "synthetic-B"],
                               clinical_episode_id=["e0", "e1", "e2"],
                               clinical_anchor_date=pd.to_datetime(["2024-01-01", "2024-02-01", "2024-01-01"]),
                               is_clinical_baseline=[True, False, True]))
    additions = {
        "lab__anti_ro60__text": [">1300", ">1300", " "],
        "lab__anti_ro60__value": pd.Series([1300.0, 1300.0, None], dtype="Float64"),
        "lab__immunofixation__text": ["A faint lambda band was detected.", None, None],
        "lab__ana_status__text": ["Positive 1:640", "not performed", "negative"],
        "sero__ana_status__ever_positive_through_episode": pd.Series([True, True, None], dtype="boolean"),
        "lab__other__text": ['</script><img src=x onerror=alert(2)> &', "", None],
        "spine__sjogrens_class_patient_values": ["4; 2", "1; 4", "2; 1"],
        "demo__name": ["private-synthetic-a", "private-synthetic-a", "private-synthetic-b"],
        "ext__source_identifier": ["hidden1", "hidden2", "hidden3"],
        "ext__score": [0, 10, 13],
    }
    master = master.assign(**additions)
    master.to_parquet(paths["integrated"], index=False)
    master[KEYS].assign(**{"lab__anti_ro60__unit": ["mg/dL", "IU/L", None]}).iloc[::-1].to_parquet(paths["context"], index=False)
    pd.DataFrame(dict(
        patient_id=["synthetic-A", "synthetic-B", "synthetic-B"], matched_clinical_episode_id=["e0", "e2", None],
        canonical_analyte=["ana_status"] * 3, lab_date=["2023-12-31"] * 3,
        result_available_date=["2024-01-15", "2023-12-31", None],
        result_raw=["positive", "negative", "positive"], result_valid_for_analysis=[True] * 3,
    )).to_parquet(paths["source_events"], index=False)
    wb = Workbook()
    ws = wb.active
    ws.title = "Consolidated_Codebook"
    ws.append(CODEBOOK_FIELDS)
    ws.append(["Sheet", "P", "Form", "F_Q", "Q", "1", "Q", "ana_status", "Generic metadata", "Yes", "01"])
    wb.save(paths["codebook"])
    Path(paths["pharma_registry"]).write_text("registry_version: test\nconcepts: {}\n")
    resources = {"codebook": {"status": "available", "sha256": "book"}, "gpt_oss": {"status": "available", "sha256": "gpt"},
                 "sapbert": {"status": "available", "sha256": "encoder"}}
    resources.update({name: {"status": "unavailable"} for name in ("ipd", "ucum", "rxnorm", "protocols")})
    return master, {"resources": resources}


def test_pipeline_integration_cache_and_offline_report(tmp_path, cfg):
    original, manifest = synthetic(tmp_path, cfg)
    Model.calls = 0
    options = dict(encoder=Encoder(), model_factory=Model, resource_verify=lambda cfg: manifest)
    analytic, output = run(cfg, **options)
    candidate = pd.read_parquet(analytic / "10_integrated_longitudinal_clinical_episode_harmonized.parquet")
    pd.testing.assert_frame_equal(original, candidate[original.columns])
    assert candidate["harm__lab__anti_ro60__text__limit_value"].iloc[:2].tolist() == [1300.0, 1300.0]
    assert "harm__lab__anti_ro60__text__exact_value" not in candidate
    assert candidate["harm__sero__ana_status__ever_positive_through_episode"].iloc[1] == True
    assert pd.isna(candidate["harm__sero__ana_status__ever_positive_through_episode"].iloc[0])
    assert candidate["harm__sero__ana_status__ever_positive_through_episode"].iloc[2] == False
    source = pq.read_table(analytic / "20_btris_lab_records_long_harmonized.parquet")
    assert source.num_rows == 3
    mappings = pq.read_table(output / "semantic_dictionary.parquet").to_pylist()
    censor = [row for row in mappings if row["variable"] == "lab__anti_ro60__text" and row["comparator"] == "GT"]
    assert len(censor) == 2 and len({row["mapping_id"] for row in censor}) == 2
    assert not {row["variable"] for row in mappings} & {"spine__sjogrens_class_patient_values", "demo__name", "ext__source_identifier"}
    html = output / "semantic_harmonization_final.html"
    content = html.read_text()
    assert "</script><img" not in content and "synthetic-A" not in content and "private-synthetic-a" not in content
    assert "\\u003c/script\\u003e" in content and "\\u0026" in content and "connect-src 'none'" in content
    browser_js = Path(__file__).with_name("semantic_report_browser.cjs")
    node, browser = shutil.which("node"), shutil.which("chromium")
    if node and browser and subprocess.run([node, "-p", "require.resolve('playwright')"],
                                           capture_output=True, cwd=browser_js.parent).returncode == 0:
        env = os.environ | {"XDG_CONFIG_HOME": str(tmp_path / "browser-config"),
                            "XDG_CACHE_HOME": str(tmp_path / "browser-cache")}
        checked = subprocess.run([node, str(browser_js), str(html), browser], env=env,
                                 capture_output=True, timeout=30, text=True)
        assert checked.returncode == 0, checked.stderr
    assert stat_mode(html) == 0o600
    assert run(cfg, **options) == (analytic, output)
    assert Model.calls == 1
    cfg["prompts"]["version"] = "revised-fixture-v2"
    next_analytic, _ = run(cfg, **options)
    assert next_analytic != analytic and Model.calls == 2


def stat_mode(path: Path):
    return os.stat(path).st_mode & 0o777


def test_context_cardinality_failure_publishes_no_candidate(tmp_path, cfg):
    _, manifest = synthetic(tmp_path, cfg)
    path = Path(cfg["paths"]["context"])
    frame = pd.read_parquet(path)
    pd.concat([frame, frame.iloc[[0]]]).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="duplicate episode keys"):
        run(cfg, encoder=Encoder(), model_factory=Model, resource_verify=lambda cfg: manifest)
    assert not list(Path(cfg["paths"]["analytic"]).glob("*"))


def test_protocol_threshold_uses_the_whole_interval(cfg):
    def measure(raw, unit="mg/dL"):
        row = dict(variable="lab__anti_ro60__text", concept="anti_ro60", family="qualitative",
                   raw_json=serialize(raw), context_json=serialize(dict.fromkeys(CONTEXT_FIELDS) |
                       dict(assay_id="synthetic-assay", method="synthetic-method", unit=unit, relevant_protocol_version="s1")))
        vocab = VOCAB | {"protocol_rules": [
            dict(rule_id="synthetic-rule", concept_id="anti_ro60", assay_key="synthetic-assay",
                 method="synthetic-method", protocol_version="s1", units=["mg/dL"],
                 positive_threshold=dict(operator="GE", value="10", otherwise="negative"))
        ]}
        return parse(row, cfg, vocab)

    assert measure("<10").qualitative_result == "negative"
    assert measure("<=10").qualitative_result is None
    assert measure(">=10").qualitative_result == "positive"
    assert measure(">8").qualitative_result is None
    assert measure(">10", unit="IU/L").qualitative_result is None
