"""Synthetic tests for conservative step-20b baseline-lab derivation."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

MODULE_PATH = Path(__file__).parents[1] / "src" / "20b_btris_baseline_labs.py"
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("baseline_labs_20b", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def patient(patient_id: str = "P1", baseline: str | None = "2020-01-01") -> pd.Series:
    """Return a synthetic authoritative spine row."""
    return pd.Series(
        {
            "patient_id": patient_id,
            "clinical_baseline_episode_id": "E1" if baseline else pd.NA,
            "clinical_baseline_date": pd.Timestamp(baseline) if baseline else pd.NaT,
        }
    )


def record(
    analyte: str, days: int, result: str = "positive", **updates: object
) -> dict:
    """Return a complete valid synthetic step-20 record."""
    row = {
        "patient_id": "P1",
        "canonical_analyte": analyte,
        "days_from_clinical_baseline": days,
        "lab_date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=days),
        "result_raw": result,
        "result_text": result,
        "result_numeric": pd.NA,
        "unit": pd.NA,
        "reference_low": pd.NA,
        "reference_high": pd.NA,
        "reported_interpretation": pd.NA,
        "result_status": "final",
        "result_valid_for_analysis": True,
        "order_name_original": "order",
        "cluster_name_original": "cluster",
        "observation_identifier": f"{analyte}-{days}-{result}",
    }
    row.update(updates)
    return row


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("positive", True),
        ("negative", False),
        ("nonreactive", False),
        ("not detected", False),
        ("equivocal", pd.NA),
        ("not reported", pd.NA),
    ],
)
def test_qualitative_parser_is_exact(token: str, expected: object) -> None:
    """Explicit tokens parse without dangerous substring matching."""
    result = MODULE.interpret_result({"result_text": token}, "positive")
    if pd.isna(expected):
        assert pd.isna(result.interpreted_status)
    else:
        assert result.interpreted_status is expected


def test_stable_historical_status_and_discordance() -> None:
    """Ever-positive and closest status retain different clinical concepts."""
    rows = pd.DataFrame(
        [
            record("anti_ro_ssa", -1000, "positive"),
            record("anti_ro_ssa", -10, "negative"),
        ]
    )
    result = MODULE.derive_stable_feature(patient(), rows, "anti_ro_ssa", True)
    assert result["ever_positive_prebaseline"] is True
    assert result["closest_prebaseline_status"] is False
    assert result["longitudinal_discordance"] is True


def test_stable_negative_no_test_and_postbaseline_sensitivity() -> None:
    """Not tested remains NA, while postbaseline evidence never backfills primary."""
    negative = MODULE.derive_stable_feature(
        patient(),
        pd.DataFrame([record("anti_la_ssb", -1, "negative")]),
        "anti_la_ssb",
        True,
    )
    assert negative["primary_baseline_status"] is False
    empty = MODULE.derive_stable_feature(
        patient(),
        pd.DataFrame(columns=pd.DataFrame([record("anti_la_ssb", 1)]).columns),
        "anti_la_ssb",
        False,
    )
    assert pd.isna(empty["primary_baseline_status"])
    post = MODULE.derive_stable_feature(
        patient(),
        pd.DataFrame([record("anti_la_ssb", 20, "positive")]),
        "anti_la_ssb",
        True,
    )
    assert pd.isna(post["primary_baseline_status"])
    assert post["postbaseline_90d_sensitivity_status"] is True


def test_ro52_and_ro60_do_not_substitute_for_ssa() -> None:
    """Supporting Ro components cannot populate the core SSA feature."""
    rows = pd.DataFrame([record("anti_ro52", -2), record("anti_ro60", -1)])
    result = MODULE.derive_stable_feature(patient(), rows, "anti_ro_ssa", True)
    assert pd.isna(result["primary_baseline_status"])


@pytest.mark.parametrize(
    ("days", "selected"),
    [(-365, True), (-366, False), (0, True), (1, True), (30, True), (31, False)],
)
def test_dynamic_boundaries_are_inclusive(days: int, selected: bool) -> None:
    """The exact -365/0/+1/+30 boundaries implement the protocol."""
    rows = pd.DataFrame(
        [record("complement_c4", days, "", result_numeric=5, reference_low=10)]
    )
    result = MODULE.derive_dynamic_feature(patient(), rows, "low_c4", True)
    assert (
        result["selected_lab_date"] is not pd.NaT
        and pd.notna(result["selected_lab_date"])
    ) is selected


def test_dynamic_directionality_and_rescue() -> None:
    """An eligible prebaseline result wins over a closer postbaseline result."""
    rows = pd.DataFrame(
        [
            record("wbc", -200, "", result_numeric=3, reference_low=4),
            record("wbc", 1, "", result_numeric=6, reference_low=4),
        ]
    )
    result = MODULE.derive_dynamic_feature(patient(), rows, "leukopenia", True)
    assert result["days_from_clinical_baseline"] == -200
    assert result["postbaseline_rescue"] is False
    rescue = MODULE.derive_dynamic_feature(
        patient(),
        pd.DataFrame([record("wbc", 20, "", result_numeric=3, reference_low=4)]),
        "leukopenia",
        True,
    )
    assert rescue["days_from_clinical_baseline"] == 20
    assert rescue["postbaseline_rescue"] is True


@pytest.mark.parametrize(
    ("updates", "expected", "source"),
    [
        ({"reported_interpretation": "low"}, True, "reported_interpretation"),
        ({"reported_interpretation": "normal"}, False, "reported_interpretation"),
        ({"result_numeric": 3, "reference_low": 4}, True, "reference_range"),
        ({"result_numeric": 4, "reference_low": 4}, False, "reference_range"),
        ({"result_numeric": 3}, pd.NA, "uninterpretable"),
    ],
)
def test_low_interpretation_never_invents_cutoff(
    updates: dict, expected: object, source: str
) -> None:
    """C4/WBC low status requires reported or contemporaneous evidence."""
    result = MODULE.interpret_result(updates, "low")
    assert result.interpretation_source == source
    if pd.isna(expected):
        assert pd.isna(result.interpreted_status)
    else:
        assert result.interpreted_status is expected


@pytest.mark.parametrize(
    ("numeric", "reference_high", "expected"),
    [(11, 10, True), (10, 10, False), (9, 10, False)],
)
def test_numeric_serology_uses_contemporaneous_reference_high(
    numeric: float, reference_high: float, expected: bool
) -> None:
    """Quantitative serology is interpreted only against available assay range."""
    result = MODULE.interpret_result(
        {"result_numeric": numeric, "reference_high": reference_high}, "positive"
    )
    assert result.interpreted_status is expected
    assert result.interpretation_source == "reference_range"


def test_numeric_serology_without_range_or_cutoff_is_missing() -> None:
    """A quantitative serology value alone does not imply positivity."""
    result = MODULE.interpret_result({"result_numeric": 100}, "positive")
    assert pd.isna(result.interpreted_status)
    assert result.interpretation_source == "uninterpretable"


def test_cryoglobulin_direct_only_and_historical_status() -> None:
    """IFE does not substitute, and old direct positivity remains historical."""
    ife = MODULE.derive_dynamic_feature(
        patient(),
        pd.DataFrame([record("cryoglobulins_ife", -1)]),
        "cryoglobulinemia",
        True,
    )
    assert pd.isna(ife["primary_baseline_status"])
    assert ife["cryoglobulins_ife_available"] is True
    rows = pd.DataFrame(
        [
            record("cryoglobulins", -500, "positive"),
            record("cryoglobulins", -10, "negative"),
        ]
    )
    direct = MODULE.derive_dynamic_feature(patient(), rows, "cryoglobulinemia", True)
    assert direct["ever_positive_prebaseline"] is True
    assert direct["primary_baseline_status"] is False


def test_equivalent_final_and_conflicting_same_day_resolution() -> None:
    """Equivalent duplicates collapse, final wins, unresolved conflict stays NA."""
    equivalent = pd.DataFrame(
        [
            record(
                "wbc",
                0,
                "",
                result_numeric=3,
                reference_low=4,
                observation_identifier="a",
                order_identifier="order-1",
            ),
            record(
                "wbc",
                0,
                "",
                result_numeric=3,
                reference_low=4,
                observation_identifier="b",
                order_identifier="order-1",
            ),
        ]
    )
    dedup = MODULE.derive_dynamic_feature(patient(), equivalent, "leukopenia", True)
    assert dedup["deduplicate_same_specimen"] is True
    preliminary = equivalent.copy()
    preliminary.loc[0, ["result_numeric", "result_status"]] = [8, "preliminary"]
    final = MODULE.derive_dynamic_feature(patient(), preliminary, "leukopenia", True)
    assert final["selected_from_final_verified"] is True
    conflict_rows = equivalent.copy()
    conflict_rows.loc[1, "result_numeric"] = 8
    conflict_rows["result_status"] = "unknown"
    conflict = MODULE.derive_dynamic_feature(
        patient(), conflict_rows, "leukopenia", True
    )
    assert conflict["same_day_conflict"] is True
    assert pd.isna(conflict["primary_baseline_status"])


def test_identical_results_without_strong_identity_are_not_specimen_deduplicated() -> (
    None
):
    """Same day/result and generic specimen type do not prove one specimen."""
    rows = pd.DataFrame(
        [
            record(
                "wbc",
                0,
                "",
                result_numeric=3,
                reference_low=4,
                observation_identifier="a",
                specimen_type="Blood",
                assay="assay-a",
            ),
            record(
                "wbc",
                0,
                "",
                result_numeric=3,
                reference_low=4,
                observation_identifier="b",
                specimen_type="Blood",
                assay="assay-b",
            ),
        ]
    )
    result = MODULE.derive_dynamic_feature(patient(), rows, "leukopenia", True)
    assert result["primary_baseline_status"] is True
    assert result["deduplicate_same_specimen"] is False


def test_preliminary_final_preference_requires_strong_identity() -> None:
    """Final status cannot resolve discordant observations from distinct assays."""
    rows = pd.DataFrame(
        [
            record(
                "wbc",
                0,
                "",
                result_numeric=3,
                reference_low=4,
                result_status="preliminary",
                assay="assay-a",
            ),
            record(
                "wbc",
                0,
                "",
                result_numeric=8,
                reference_low=4,
                result_status="final",
                assay="assay-b",
            ),
        ]
    )
    result = MODULE.derive_dynamic_feature(patient(), rows, "leukopenia", True)
    assert result["same_day_conflict"] is True
    assert pd.isna(result["primary_baseline_status"])


def test_patient_universe_and_ineligible_patient() -> None:
    """Every spine patient gets seven rows and ineligible patients select nothing."""
    columns = list(MODULE.REQUIRED_LAB_COLUMNS | {"observation_identifier"})
    labs = pd.DataFrame(columns=columns)
    spine = pd.DataFrame([patient("P1"), patient("P2", None)])
    long = MODULE.derive_long(labs, spine)
    wide = MODULE.derive_wide(long, spine)
    assert len(long) == 14
    assert len(wide) == 2
    assert wide[list(MODULE.WIDE_NAMES.values())].isna().all().all()
    assert not wide.loc[wide["patient_id"] == "P2", "lab_baseline_eligible"].item()


def test_episode_level_spine_collapses_to_one_patient() -> None:
    """Equivalent baseline metadata across clinical episodes collapses safely."""
    spine = pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "clinical_episode_id": "E1",
                "clinical_baseline_episode_id": "E1",
                "clinical_baseline_date": pd.Timestamp("2020-01-01"),
            },
            {
                "patient_id": "P1",
                "clinical_episode_id": "E2",
                "clinical_baseline_episode_id": "E1",
                "clinical_baseline_date": pd.Timestamp("2020-01-01"),
            },
        ]
    )
    patient_spine = MODULE.build_patient_baseline_frame(spine)
    assert len(patient_spine) == 1
    assert patient_spine.loc[0, "patient_id"] == "P1"
    assert patient_spine.loc[0, "clinical_baseline_episode_id"] == "E1"


def test_episode_spine_uses_step20_patient_id_normalization() -> None:
    """Spine MRN formatting differences do not erase all BTRIS coverage."""
    spine = pd.DataFrame(
        [
            {
                "patient_id": " 001-234 ",
                "clinical_baseline_episode_id": "E1",
                "clinical_baseline_date": pd.Timestamp("2020-01-01"),
            },
            {
                "patient_id": "001/234",
                "clinical_baseline_episode_id": "E1",
                "clinical_baseline_date": pd.Timestamp("2020-01-01"),
            },
        ]
    )
    patient_spine = MODULE.build_patient_baseline_frame(spine)
    assert patient_spine["patient_id"].tolist() == ["1234"]


def test_derive_long_matches_normalized_lab_and_spine_identifiers() -> None:
    """Patient membership is based on the shared normalized step-20 key."""
    lab_row = {column: pd.NA for column in MODULE.REQUIRED_LAB_COLUMNS}
    lab_row.update(
        {
            "patient_id": "001-234",
            "clinical_baseline_episode_id": "E1",
            "clinical_baseline_date": pd.Timestamp("2020-01-01"),
            "canonical_analyte": "anti_ro_ssa",
            "days_from_clinical_baseline": 0,
            "lab_date": pd.Timestamp("2020-01-01"),
            "result_raw": "positive",
            "result_text": "positive",
            "result_valid_for_analysis": True,
            "order_name_original": "ENA Evaluation",
            "cluster_name_original": "SS-A/Ro Ab, IgG (Blood)",
        }
    )
    patient_spine = pd.DataFrame([patient("1234")])
    long = MODULE.derive_long(pd.DataFrame([lab_row]), patient_spine)
    ssa = long[long["baseline_feature"] == "anti_ro_ssa"].iloc[0]
    assert bool(ssa["has_any_btris_lab"])
    assert bool(ssa["primary_baseline_status"])


def test_baseline_mismatch_counts_patients_not_source_rows() -> None:
    """Hard-QC mismatch magnitude is patient-level rather than lab-row-level."""
    lab_rows = []
    for _ in range(3):
        row = {column: pd.NA for column in MODULE.REQUIRED_LAB_COLUMNS}
        row.update(
            {
                "patient_id": "001-234",
                "clinical_baseline_episode_id": "wrong",
                "clinical_baseline_date": pd.Timestamp("2021-01-01"),
            }
        )
        lab_rows.append(row)
    patient_spine = pd.DataFrame([patient("1234")])
    assert MODULE.validate_inputs(pd.DataFrame(lab_rows), patient_spine) == 1


def test_conflicting_patient_baseline_hard_fails() -> None:
    """Distinct baseline metadata for one patient fails rather than choosing one."""
    spine = pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "clinical_baseline_episode_id": "E1",
                "clinical_baseline_date": pd.Timestamp("2020-01-01"),
            },
            {
                "patient_id": "P1",
                "clinical_baseline_episode_id": "E2",
                "clinical_baseline_date": pd.Timestamp("2021-01-01"),
            },
        ]
    )
    with pytest.raises(ValueError, match="conflicting"):
        MODULE.build_patient_baseline_frame(spine)
