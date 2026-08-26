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
