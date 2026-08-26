"""Synthetic tests for the step-20 longitudinal BTRIS laboratory table."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
MODULE_PATH = REPO_ROOT / "src" / "20_btris_visit_date_match_report.py"
spec = importlib.util.spec_from_file_location("btris_lab_records", MODULE_PATH)
btris = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = btris
assert spec.loader is not None
spec.loader.exec_module(btris)


def _reference(*pairs: tuple[str, str]) -> pd.DataFrame:
    return pd.DataFrame(pairs, columns=["order_name", "cluster_name"]).assign(
        canonical_analyte=pd.NA, lab_family="other", analytic_role="currently_unused"
    )


def _raw(
    dates: list[str],
    order: str = "C3/C4",
    cluster: str = "Complement C4 (Blood)",
    values: list[str] | None = None,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "MRN": ["001-23"] * len(dates),
            "Order Name": [order] * len(dates),
            "Cluster Name": [cluster] * len(dates),
            "Collected Date Time": dates,
            "Observation Value": values or ["7.4"] * len(dates),
            "Result Status": ["Final"] * len(dates),
        }
    )


def _spine() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": ["123", "123"],
            "clinical_episode_id": ["e1", "e2"],
            "episode_start_date": ["2019-12-28", "2020-12-28"],
            "clinical_anchor_date": ["2020-01-01", "2021-01-01"],
            "episode_end_date": ["2020-01-05", "2021-01-05"],
            "clinical_baseline_episode_id": ["e1", "e1"],
            "clinical_baseline_date": ["2020-01-01", "2020-01-01"],
        }
    )


def test_source_schema_qc_is_structural_and_has_one_row_per_column() -> None:
    raw = pd.DataFrame({"Foo": ["secret"], "Bar": ["2020-01-01"], "UnitX": ["mg/dL"]})

    qc = btris.build_source_schema_qc(raw, "Lab.csv")

    assert qc["raw_column_name"].tolist() == ["Foo", "Bar", "UnitX"]
    assert list(qc.columns) == [
        "source_file",
        "raw_column_name",
        "dtype",
        "n_rows",
        "n_nonmissing",
        "pct_nonmissing",
    ]
    assert not qc.astype("string").eq("secret").any(axis=None)


def test_field_resolution_uses_only_confirmed_aliases_and_exposes_no_values() -> None:
    raw = pd.DataFrame({"UnitX": ["secret-unit"], "Result": ["secret-result"]})

    unresolved = btris.build_field_resolution_qc(raw, "Lab.csv")
    confirmed = btris.build_field_resolution_qc(
        raw, "Lab.csv", column_aliases={"unit": ["UnitX"]}
    )

    unresolved_unit = unresolved[unresolved["target_field"].eq("unit")].iloc[0]
    confirmed_unit = confirmed[confirmed["target_field"].eq("unit")].iloc[0]
    assert not bool(unresolved_unit["resolved"])
    assert pd.isna(unresolved_unit["resolved_raw_column"])
    assert bool(confirmed_unit["resolved"])
    assert confirmed_unit["resolved_raw_column"] == "UnitX"
    assert (
        not confirmed.astype("string")
        .isin(["secret-unit", "secret-result"])
        .any(axis=None)
    )


def test_qualitative_token_qc_normalizes_but_does_not_classify() -> None:
    labs = pd.DataFrame(
        {
            "canonical_analyte": ["anti_ro_ssa"] * 3,
            "result_text": ["Positive", " positive ", "<1.0"],
        }
    )

    qc = btris.build_core_qualitative_token_qc(labs)

    assert qc.set_index("normalized_result_text")["n_rows"].to_dict() == {
        "positive": 2,
        "<1.0": 1,
    }


@pytest.mark.parametrize(
    "date,expected",
    [
        ("2020-01-01", 0),
        ("2019-03-07", -300),
        ("2017-01-01", -1095),
        ("2020-01-16", 15),
        ("2025-01-01", 1827),
    ],
)
def test_all_temporal_histories_are_preserved(date: str, expected: int) -> None:
    labs = btris.normalize_lab_records(_raw([date]))
    output, _ = btris.attach_clinical_context(labs, _spine())
    assert len(output) == 1
    assert output.loc[0, "days_from_clinical_baseline"] == expected


def test_episode_matching_priorities_and_unmatched_retention() -> None:
    labs = btris.normalize_lab_records(_raw(["2020-01-03", "2021-01-10", "2022-06-01"]))
    output, _ = btris.attach_clinical_context(labs, _spine())
    assert output["episode_match_method"].tolist() == [
        "inside_episode_window",
        "closest_anchor_le10d",
        "no_episode_match",
    ]
    assert pd.isna(output.iloc[2]["matched_clinical_episode_id"])


def test_inside_episode_with_missing_anchor_is_retained() -> None:
    spine = _spine().iloc[[0]].copy()
    spine.loc[:, "clinical_anchor_date"] = pd.NaT
    labs = btris.normalize_lab_records(_raw(["2020-01-03"]))

    output, ambiguous = btris.attach_clinical_context(labs, spine)

    assert len(output) == 1
    assert output.loc[0, "matched_clinical_episode_id"] == "e1"
    assert output.loc[0, "episode_match_method"] == "inside_episode_window"
    assert pd.isna(output.loc[0, "matched_clinical_anchor_date"])
    assert pd.isna(output.loc[0, "days_from_clinical_anchor"])
    assert not bool(output.loc[0, "episode_match_ambiguous"])
    assert ambiguous.empty


def test_overlapping_episodes_with_missing_anchors_are_ambiguous() -> None:
    spine = _spine().iloc[[0]].copy()
    second = spine.copy()
    second.loc[:, "clinical_episode_id"] = "e2"
    spine.loc[:, "clinical_anchor_date"] = pd.NaT
    second.loc[:, "clinical_anchor_date"] = pd.NaT
    episodes = pd.concat([spine, second], ignore_index=True)
    labs = btris.normalize_lab_records(_raw(["2020-01-03"]))

    output, ambiguous = btris.attach_clinical_context(labs, episodes)

    assert len(output) == 1
    assert output.loc[0, "episode_match_method"] == "ambiguous"
    assert bool(output.loc[0, "episode_match_ambiguous"])
    assert pd.isna(output.loc[0, "matched_clinical_episode_id"])
    assert pd.isna(output.loc[0, "matched_clinical_anchor_date"])
    assert pd.isna(output.loc[0, "days_from_clinical_anchor"])
    assert len(ambiguous) == 1
    assert {
        ambiguous.loc[0, "candidate_episode_id_1"],
        ambiguous.loc[0, "candidate_episode_id_2"],
    } == {"e1", "e2"}


def test_exact_tie_is_ambiguous_and_not_assigned() -> None:
    spine = _spine().iloc[[0]].copy()
    second = spine.copy()
    spine.loc[:, "clinical_episode_id"] = "left"
    spine.loc[:, ["episode_start_date", "episode_end_date", "clinical_anchor_date"]] = [
        "2019-12-01",
        "2019-12-02",
        "2020-01-01",
    ]
    second.loc[:, "clinical_episode_id"] = "right"
    second.loc[
        :, ["episode_start_date", "episode_end_date", "clinical_anchor_date"]
    ] = ["2020-01-20", "2020-01-21", "2020-01-03"]
    output, ambiguous = btris.attach_clinical_context(
        btris.normalize_lab_records(_raw(["2020-01-02"])), pd.concat([spine, second])
    )
    assert bool(output.loc[0, "episode_match_ambiguous"])
    assert pd.isna(output.loc[0, "matched_clinical_episode_id"])
    assert {
        ambiguous.loc[0, "candidate_episode_id_1"],
        ambiguous.loc[0, "candidate_episode_id_2"],
    } == {"left", "right"}


def test_expected_missing_and_unexpected_cluster_qc() -> None:
    reference = _reference(
        ("C3/C4", "Complement C4 (Blood)"), ("CBC + Diff", "WBC (Blood)")
    )
    annotated = btris.annotate_expected_pairs(
        btris.normalize_lab_records(
            pd.concat(
                [_raw(["2020-01-01"]), _raw(["2020-01-02"], "Novel", "Novel cluster")]
            )
        ),
        reference,
    )
    coverage = btris.build_cluster_coverage(annotated, reference)
    assert annotated["unexpected_cluster_name"].tolist() == [False, True]
    expected = coverage.iloc[:2]
    assert expected["found_in_input"].tolist() == [True, False]
    assert expected["expected_cluster_not_found"].tolist() == [False, True]


def test_raw_result_is_preserved_and_not_reported_is_not_negative() -> None:
    labs = btris.normalize_lab_records(
        _raw(["2020-01-01", "2020-01-02"], values=["7.4", "Not Reported"])
    )
    assert labs["result_raw"].tolist() == ["7.4", "Not Reported"]
    assert labs.loc[0, "result_numeric"] == 7.4
    assert pd.isna(labs.loc[1, "result_numeric"])
    assert labs.loc[1, "result_text"] == "Not Reported"


@pytest.mark.parametrize(
    ("raw_value", "exact", "operator", "bound"),
    [
        ("7.4", 7.4, None, None),
        (">8.0", None, ">", 8.0),
        ("<15", None, "<", 15.0),
        ("> 8.0", None, ">", 8.0),
        ("Positive 1:320", None, None, None),
    ],
)
def test_numeric_result_parser_separates_exact_and_censored_evidence(
    raw_value: str,
    exact: float | None,
    operator: str | None,
    bound: float | None,
) -> None:
    parsed = btris.parse_numeric_result(raw_value)

    assert parsed.exact == exact
    assert parsed.operator == operator
    assert parsed.bound == bound
    normalized = btris.normalize_lab_records(_raw(["2020-01-01"], values=[raw_value]))
    assert normalized.loc[0, "result_operator"] == operator or (
        operator is None and pd.isna(normalized.loc[0, "result_operator"])
    )
    if exact is None:
        assert pd.isna(normalized.loc[0, "result_numeric"])
        assert pd.isna(normalized.loc[0, "result_numeric_exact"])
    else:
        assert normalized.loc[0, "result_numeric"] == exact
        assert normalized.loc[0, "result_numeric_exact"] == exact
    if bound is None:
        assert pd.isna(normalized.loc[0, "result_numeric_bound"])
    else:
        assert normalized.loc[0, "result_numeric_bound"] == bound


def test_btris_evidence_aliases_and_normal_range_qc() -> None:
    raw = (
        _raw(["2020-01-01"])
        .drop(columns="Result Status")
        .assign(
            **{
                "Unit of Measure": " mg/dL ",
                "Normal Range": " 4.0   -  10.0 ",
                "Observation Comment": "assay documentation",
                "Observation Note": "source note",
                "Status": "Final",
                "Order ID": "ORDER-1",
            }
        )
    )
    normalized = btris.normalize_lab_records(raw)
    annotated = normalized.assign(canonical_analyte="complement_c4")

    assert normalized.loc[0, "unit"] == " mg/dL "
    assert normalized.loc[0, "reference_range_raw"] == " 4.0   -  10.0 "
    assert normalized.loc[0, "reference_range_parse_status"] == "ambiguous"
    assert pd.isna(normalized.loc[0, "reference_low"])
    assert pd.isna(normalized.loc[0, "reference_high"])
    assert normalized.loc[0, "observation_comment"] == "assay documentation"
    assert normalized.loc[0, "observation_note"] == "source note"
    assert normalized.loc[0, "result_status"] == "Final"
    assert normalized.loc[0, "order_identifier"] == "ORDER-1"
    qc = btris.build_core_normal_range_token_qc(annotated)
    assert qc.loc[0, "normalized_normal_range"] == "4.0 - 10.0"
    assert qc.loc[0, "n_rows"] == 1
    assert qc.loc[0, "pct_within_analyte"] == 100.0


def test_ro52_ro60_and_ssa_remain_distinct() -> None:
    pairs = [
        ("ENA Evaluation", "SS-A/Ro Ab, IgG (Blood)"),
        ("Ro52 & Ro60 Antibodies, IgG", "SS-Ro52 Ab, IgG (Blood)"),
        ("Ro52 & Ro60 Antibodies, IgG", "SS-Ro60 Ab, IgG (Blood)"),
    ]
    reference = _reference(*pairs)
    for index, pair in enumerate(pairs):
        reference.loc[index, ["canonical_analyte", "lab_family", "analytic_role"]] = (
            btris.SEMANTIC_OVERRIDES[pair]
        )
    rows = pd.concat([_raw(["2020-01-01"], *pair) for pair in pairs], ignore_index=True)
    output = btris.annotate_expected_pairs(btris.normalize_lab_records(rows), reference)
    assert output["canonical_analyte"].tolist() == [
        "anti_ro_ssa",
        "anti_ro52",
        "anti_ro60",
    ]


def test_ana_components_are_not_collapsed() -> None:
    clusters = [
        "Antinuclear Antibody (ANA) HEp-2 Substrate (Blood)",
        "Antinuclear Antibody (ANA) HEp-2 Substrate Titer (Blood)",
        "Antinuclear Antibody (ANA) HEp-2 Substrate Pattern (Blood)",
    ]
    pairs = [("ANA", cluster) for cluster in clusters]
    rows = pd.concat([_raw(["2020-01-01"], *pair) for pair in pairs])
    output = btris.annotate_expected_pairs(
        btris.normalize_lab_records(rows), _reference(*pairs)
    )
    assert len(output) == 3
    assert output["cluster_name"].tolist() == clusters


def test_reference_requires_all_256_exact_pairs(tmp_path: Path) -> None:
    path = tmp_path / "reference.csv"
    _reference(("only", "one")).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Expected 256 exact pairs"):
        btris.load_reference(path)


def test_patient_without_baseline_is_retained() -> None:
    spine = _spine().copy()
    spine[["clinical_baseline_episode_id", "clinical_baseline_date"]] = pd.NA
    output, _ = btris.attach_clinical_context(
        btris.normalize_lab_records(_raw(["2020-01-01"])), spine
    )
    assert len(output) == 1
    assert not bool(output.loc[0, "has_clinical_baseline"])
    assert pd.isna(output.loc[0, "days_from_clinical_baseline"])


@pytest.mark.parametrize(
    "original,canonical,cluster,analyte",
    [
        (
            "RHEUMATOID FACTOR",
            "Rheumatoid Factor",
            "Rheumatoid Factor (Blood)",
            "rheumatoid_factor",
        ),
        ("CRYOGLOBULINS", "Cryoglobulins", "Cryoglobulins (Blood)", "cryoglobulins"),
    ],
)
def test_explicit_historical_aliases_preserve_original(
    original: str, canonical: str, cluster: str, analyte: str
) -> None:
    output = btris.annotate_expected_pairs(
        btris.normalize_lab_records(_raw(["2020-01-01"], original, cluster)),
        _reference((canonical, cluster)),
    )
    assert output.loc[0, "order_name_original"] == original
    assert output.loc[0, "order_name_canonical"] == canonical
    assert output.loc[0, "mapping_status"] == "explicit_alias"
    assert output.loc[0, "canonical_analyte"] == analyte


def test_similar_unapproved_order_is_not_aliased() -> None:
    output = btris.annotate_expected_pairs(
        btris.normalize_lab_records(
            _raw(["2020-01-01"], "RHEUMATOID FACTOR TEST", "Rheumatoid Factor (Blood)")
        ),
        _reference(("Rheumatoid Factor", "Rheumatoid Factor (Blood)")),
    )
    assert output.loc[0, "order_name_canonical"] == "RHEUMATOID FACTOR TEST"
    assert output.loc[0, "mapping_status"] == "unexpected_unmapped"
    assert pd.isna(output.loc[0, "canonical_analyte"])


def test_missing_cluster_is_unexpected_and_not_inferred() -> None:
    raw = _raw(["2020-01-01"], "C3/C4", "placeholder")
    raw.loc[0, "Cluster Name"] = pd.NA
    output = btris.annotate_expected_pairs(
        btris.normalize_lab_records(raw),
        _reference(("C3/C4", "Complement C4 (Blood)")),
    )
    assert output.loc[0, "mapping_status"] == "unexpected_unmapped"
    assert bool(output.loc[0, "unexpected_cluster_name"])
    assert pd.isna(output.loc[0, "canonical_analyte"])


def test_required_semantic_components_remain_distinct() -> None:
    pairs = [
        ("Anti-Nuclear Antibody", "Antinuclear Antibody (ANA) (Blood)"),
        (
            "ANA Hep-2 Substrate, IgG",
            "Antinuclear Antibody (ANA) HEp-2 Substrate (Blood)",
        ),
        (
            "ANA Hep-2 Substrate, IgG",
            "Antinuclear Antibody (ANA) HEp-2 Substrate Titer (Blood)",
        ),
        (
            "ANA Hep-2 Substrate, IgG",
            "Antinuclear Antibody (ANA) HEp-2 Substrate Pattern (Blood)",
        ),
        (
            "ANA Hep-2 Substrate, IgG",
            "Antinuclear Antibody (ANA) HEp-2 Cytoplasmic Pattern (Blood)",
        ),
        ("C3/C4", "Complement C3 (Blood)"),
        ("C3/C4", "Complement C4 (Blood)"),
        ("CBC + Diff", "WBC (Blood)"),
        ("CBC + Diff", "Neutrophil Abs (Blood)"),
        ("CBC + Diff", "Lymphocytes Abs (Blood)"),
        ("CBC + Diff", "Hemoglobin (Blood)"),
        ("CBC + Diff", "Platelet Count (Blood)"),
    ]
    rows = pd.concat([_raw(["2020-01-01"], *pair) for pair in pairs])
    output = btris.annotate_expected_pairs(
        btris.normalize_lab_records(rows), _reference(*pairs)
    )
    assert output["canonical_analyte"].tolist() == [
        "ana_status",
        "ana_hep2_status",
        "ana_hep2_titer",
        "ana_hep2_pattern",
        "ana_hep2_cytoplasmic_pattern",
        "complement_c3",
        "complement_c4",
        "wbc",
        "anc",
        "lymphocyte_count",
        "hemoglobin",
        "platelet_count",
    ]
    assert output["canonical_analyte"].is_unique


@pytest.mark.parametrize(
    "order,cluster,analyte,family,role",
    [
        ("ESR", "ESR (Blood)", "esr", "dynamic_inflammatory", "exploratory"),
        (
            "CRP, High Sensitivity, Comprehensive",
            "C-Reactive Protein, High Sensitivity (Blood)",
            "crp_high_sensitivity",
            "dynamic_inflammatory",
            "exploratory",
        ),
        ("Hepatic Panel", "ALT (Blood)", "alt", "dynamic_hepatic", "supporting"),
        ("Hepatic Panel", "AST (Blood)", "ast", "dynamic_hepatic", "supporting"),
        (
            "Hemoglobin A1C",
            "Hemoglobin A1C (Blood)",
            "hemoglobin_a1c",
            "chronic_metabolic",
            "context",
        ),
        (
            "Hemoglobin A1C",
            "Est. Avg. Glucose (Blood)",
            "estimated_average_glucose",
            "chronic_metabolic",
            "supporting",
        ),
        (
            "Lipid Panel",
            "LDL Cholesterol Calculated (Blood)",
            "ldl_cholesterol_calculated",
            "chronic_metabolic",
            "context",
        ),
        (
            "Lipid Panel",
            "LDL Cholesterol Direct (Blood)",
            "ldl_cholesterol_direct",
            "chronic_metabolic",
            "context",
        ),
        (
            "Thyroid Stimulating Hormone",
            "TSH (Blood)",
            "tsh",
            "chronic_endocrine",
            "context",
        ),
        ("HLA", "HLA-A* (Blood)", "hla_a", "fixed_genetic", "exploratory"),
        (
            "Screening",
            "HCV (HepC) Ab (Blood)",
            "hepatitis_c_antibody",
            "infection_screening",
            "context",
        ),
        (
            "PT/PTT",
            "PTT (Blood)",
            "partial_thromboplastin_time",
            "procedural",
            "context",
        ),
        (
            "Pregnancy",
            "Pregnancy Test (Urine)",
            "urine_pregnancy_test",
            "pregnancy_time_specific",
            "context",
        ),
    ],
)
def test_extended_clinical_semantics(order, cluster, analyte, family, role) -> None:
    output = btris.annotate_expected_pairs(
        btris.normalize_lab_records(_raw(["2020-01-01"], order, cluster)),
        _reference((order, cluster)),
    )
    assert output.loc[
        0, ["canonical_analyte", "lab_family", "analytic_role"]
    ].tolist() == [analyte, family, role]


def test_semantic_intent_distinguishes_reviewed_unused_from_unexpected() -> None:
    reference = _reference(("CBC + Diff", "Schistocytes (Blood)"))
    rows = pd.concat(
        [
            _raw(["2020-01-01"], "CBC + Diff", "Schistocytes (Blood)"),
            _raw(["2020-01-02"], "Acute Care Panel", pd.NA),
            _raw(["2020-01-03"], "Anti-N SARS-CoV-2 Antibodies", pd.NA),
        ]
    )
    output = btris.annotate_expected_pairs(btris.normalize_lab_records(rows), reference)
    qc = btris.build_semantic_mapping_qc(output)
    assert output["semantic_mapping_status"].tolist() == [
        "deliberately_unused",
        "unexpected_unmapped",
        "unexpected_unmapped",
    ]
    assert qc["semantic_mapping_complete"].tolist() == [True, False, False]
    assert len(btris.build_semantic_unresolved_qc(output)) == 2


def test_semantic_annotation_does_not_change_episode_matching_counts() -> None:
    normalized = btris.normalize_lab_records(_raw(["2020-01-03", "2021-01-10"]))
    before, _ = btris.attach_clinical_context(normalized, _spine())
    annotated = btris.annotate_expected_pairs(
        normalized, _reference(("C3/C4", "Complement C4 (Blood)"))
    )
    after, _ = btris.attach_clinical_context(annotated, _spine())
    assert (
        before["episode_match_method"].value_counts().to_dict()
        == after["episode_match_method"].value_counts().to_dict()
    )
