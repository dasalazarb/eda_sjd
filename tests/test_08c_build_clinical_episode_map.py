"""Synthetic tests for deterministic two-pass clinical episode reconstruction."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

MODULE_PATH = Path(__file__).parents[1] / "src" / "08c_build_clinical_episode_map.py"
SPEC = importlib.util.spec_from_file_location("clinical_episode_map", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
EPISODES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EPISODES)

NATURAL = "Natural History Protocol 478 Interval"
INITIAL = "Phase 1: Initial Full Evaluation"
SECOND = "Phase 1: Second Full Evaluation"
THIRD = "Phase 1: Final Full (Third Full) Evaluation"


def _row(
    row_id: int, interval: str, date: str, patient_id: str = "P001", **values: object
) -> dict[str, object]:
    row: dict[str, object] = {
        "patient_id": patient_id,
        "row_id_raw": row_id,
        "interval_name": interval,
        "collection_date": date,
        "essdai": pd.NA,
        "esspri": pd.NA,
        "crp": pd.NA,
    }
    row.update(values)
    return row


def _run(
    rows: list[dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source, provenance = EPISODES.prepare_visits(pd.DataFrame(rows))
    units = EPISODES.build_atomic_activity_units(source, provenance)
    assigned_units = EPISODES.assign_episodes(units)
    assigned = EPISODES.propagate_episode_assignments(source, assigned_units)
    return (
        assigned,
        EPISODES.build_manifest(assigned),
        assigned_units.attrs["merge_decision_audit"],
        EPISODES.build_value_conflicts(assigned),
    )


def _collapsed(assigned: pd.DataFrame) -> pd.Series:
    assert assigned["clinical_episode_id"].nunique() == 1
    return EPISODES.collapse_episode_rows(assigned)


def test_same_interval_same_year_merges_without_day_limit() -> None:
    assigned, _, audit, conflicts = _run(
        [
            _row(1, "Interval A", "2024-01-01", essdai=4),
            _row(2, "Interval A", "2024-08-01", essdai=5),
        ]
    )
    assert _collapsed(assigned)["essdai"] == "4 | 5"
    assert audit.loc[0, "merge_rule"] == "same_exact_interval_same_year"
    assert (
        conflicts.loc[0, "conflict_resolution"] == "preserved_all_same_interval_values"
    )


def test_same_interval_different_years_never_merges() -> None:
    assigned, _, audit, _ = _run(
        [_row(1, "Interval A", "2024-12-29"), _row(2, "Interval A", "2025-01-03")]
    )
    assert assigned["clinical_episode_id"].nunique() == 2
    assert "different_year_no_merge" in set(audit["merge_rule"])


def test_natural_15d_complementary_uses_natural_identity_and_date() -> None:
    assigned, manifest, audit, _ = _run(
        [
            _row(1, NATURAL, "2024-03-01", essdai=4),
            _row(2, "15D Optional Evaluation 1", "2024-03-10", essdai=4, crp=3.2),
        ]
    )
    result = _collapsed(assigned)
    assert (result["essdai"], result["crp"]) == (4, 3.2)
    assert manifest.loc[0, "representative_interval"] == NATURAL
    assert manifest.loc[0, "clinical_anchor_date"] == pd.Timestamp("2024-03-01")
    assert (
        audit.loc[audit["merged"], "merge_rule"].iloc[0] == "natural_15d_within_30_days"
    )


def test_natural_15d_conflict_keeps_natural_and_records_qc() -> None:
    assigned, _, _, conflicts = _run(
        [
            _row(1, NATURAL, "2024-03-01", essdai=4),
            _row(2, "15D Optional Evaluation 1", "2024-03-10", essdai=7),
        ]
    )
    assert _collapsed(assigned)["essdai"] == 4
    conflict = conflicts.loc[conflicts["variable"].eq("essdai")].iloc[0]
    assert (
        conflict["preferred_value"],
        conflict["secondary_value"],
        conflict["chosen_value"],
    ) == (4, 7, 4)
    assert conflict["conflict_resolution"] == "preferred_primary_visit"


def test_natural_15d_beyond_30_days_does_not_merge() -> None:
    assigned, _, audit, _ = _run(
        [
            _row(1, NATURAL, "2024-03-01"),
            _row(2, "15D Optional Evaluation 1", "2024-04-02"),
        ]
    )
    assert assigned["clinical_episode_id"].nunique() == 2
    assert audit.loc[0, "merge_rule"] == "different_interval_gt30_days_no_merge"


def test_phase_optional_complementary_fills_missing_value() -> None:
    assigned, manifest, _, _ = _run(
        [
            _row(1, INITIAL, "2024-04-01"),
            _row(2, "Optional Evaluation 1", "2024-04-12", esspri=6),
        ]
    )
    assert _collapsed(assigned)["esspri"] == 6
    assert manifest.loc[0, "representative_interval"] == INITIAL


def test_phase_optional_conflict_keeps_phase() -> None:
    assigned, _, _, conflicts = _run(
        [
            _row(1, INITIAL, "2024-04-01", essdai=4),
            _row(2, "Optional Evaluation 1", "2024-04-12", essdai=6),
        ]
    )
    assert _collapsed(assigned)["essdai"] == 4
    assert (
        conflicts.loc[conflicts["variable"].eq("essdai"), "chosen_value"].iloc[0] == 4
    )


def test_initial_phase_beats_second_phase_on_same_day() -> None:
    assigned, manifest, audit, _ = _run(
        [
            _row(1, SECOND, "2024-05-01", essdai=7),
            _row(2, INITIAL, "2024-05-01", essdai=4),
        ]
    )
    assert _collapsed(assigned)["essdai"] == 4
    assert manifest.loc[0, "representative_interval"] == INITIAL
    assert (
        audit.loc[audit["merged"], "merge_rule"].iloc[0] == "phase_phase_within_30_days"
    )


def test_second_phase_beats_nearby_third_phase() -> None:
    assigned, manifest, _, _ = _run(
        [
            _row(1, THIRD, "2024-06-10", essdai=7),
            _row(2, SECOND, "2024-06-01", essdai=4),
        ]
    )
    assert _collapsed(assigned)["essdai"] == 4
    assert manifest.loc[0, "representative_interval"] == SECOND


def test_other_intervals_with_conflict_do_not_merge() -> None:
    assigned, _, audit, _ = _run(
        [
            _row(1, "Interval X", "2024-03-01", essdai=4),
            _row(2, "Interval Y", "2024-03-11", essdai=7),
        ]
    )
    assert assigned["clinical_episode_id"].nunique() == 2
    assert audit.loc[0, "merge_rule"] == "other_interval_conflict_no_merge"


def test_other_complementary_intervals_merge() -> None:
    assigned, _, audit, _ = _run(
        [
            _row(1, "Interval X", "2024-03-01", essdai=4),
            _row(2, "Interval Y", "2024-03-11", essdai=4, esspri=6),
        ]
    )
    result = _collapsed(assigned)
    assert (result["essdai"], result["esspri"]) == (4, 6)
    assert audit.loc[audit["merged"], "merge_rule"].iloc[0] == "other_temporal_rescue"


def test_different_intervals_beyond_30_days_do_not_merge() -> None:
    assigned, _, audit, _ = _run(
        [
            _row(1, "Interval X", "2024-01-01", essdai=4),
            _row(2, "Interval Y", "2024-02-01", esspri=6),
        ]
    )
    assert assigned["clinical_episode_id"].nunique() == 2
    assert audit.loc[0, "merge_rule"] == "different_interval_gt30_days_no_merge"


def test_row_assignment_conservation() -> None:
    rows = [
        _row(1, "Interval A", "2024-01-01"),
        _row(2, "Interval A", "2024-08-01"),
        _row(3, "Interval A", "2025-01-01"),
    ]
    source, _ = EPISODES.prepare_visits(pd.DataFrame(rows))
    assigned, _, _, _ = _run(rows)
    assert EPISODES.validate_final_assignments(source, assigned) == (0, 0)
    assert len(assigned) == len(source) == assigned["row_id_raw"].nunique()


def test_priority_classifier_and_invalid_raw_ids() -> None:
    assert EPISODES.get_episode_priority(NATURAL) == "natural"
    assert EPISODES.get_episode_priority(THIRD) == "phase_3"
    assert EPISODES.get_episode_priority("Optional Evaluation 2") == "optional"
    with pytest.raises(ValueError, match="row_id_raw must be complete and unique"):
        EPISODES.prepare_visits(
            pd.DataFrame([_row(1, "A", "2024-01-01"), _row(1, "B", "2024-01-02")])
        )
