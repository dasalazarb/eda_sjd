"""Synthetic tests for the two-pass clinical episode reconstruction."""

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


def _row(
    row_id: int,
    interval: str,
    date: str,
    **values: object,
) -> dict[str, object]:
    row: dict[str, object] = {
        "patient_id": "P001",
        "row_id_raw": row_id,
        "interval_name": interval,
        "collection_date": date,
        "essdai": pd.NA,
        "esspri": pd.NA,
        "variable_x": pd.NA,
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
    manifest = EPISODES.build_manifest(assigned)
    conflicts = EPISODES.build_value_conflicts(assigned)
    return (
        assigned,
        manifest,
        assigned_units.attrs["merge_decision_audit"],
        conflicts,
    )


def test_same_interval_same_year_merges_without_day_limit() -> None:
    assigned, _, audit, _ = _run(
        [
            _row(1, "Interval A", "2024-01-01", essdai=4),
            _row(2, "Interval A", "2024-09-01", esspri=6),
        ]
    )

    assert assigned["clinical_episode_id"].nunique() == 1
    assert audit.loc[audit["merged"], "merge_rule"].tolist() == [
        "same_interval_same_year"
    ]


def test_same_interval_different_years_never_merges() -> None:
    assigned, _, audit, _ = _run(
        [
            _row(1, "Interval A", "2024-12-29", essdai=4),
            _row(2, "Interval A", "2025-01-03", esspri=6),
        ]
    )

    assert assigned["clinical_episode_id"].nunique() == 2
    boundary = audit.loc[audit["merge_rule"].eq("different_year_no_merge")]
    assert len(boundary) == 1
    assert not boundary.iloc[0]["merged"]


def test_natural_history_and_15d_optional_merge_in_same_year() -> None:
    assigned, _, audit, _ = _run(
        [
            _row(1, NATURAL, "2024-02-01"),
            _row(2, "15D Optional Evaluation 1", "2024-10-01"),
        ]
    )

    assert assigned["clinical_episode_id"].nunique() == 1
    assert audit.iloc[0]["merge_rule"] == "natural_15d_same_year"


def test_optional_and_named_phase_merge_in_same_year() -> None:
    assigned, _, audit, _ = _run(
        [
            _row(1, "Optional Evaluation 1", "2024-02-01"),
            _row(2, INITIAL, "2024-10-01"),
        ]
    )

    assert assigned["clinical_episode_id"].nunique() == 1
    assert audit.iloc[0]["merge_rule"] == "optional_phase_same_year"


def test_distinct_phase_intervals_do_not_merge_by_family() -> None:
    assigned, _, _, _ = _run(
        [
            _row(1, INITIAL, "2024-02-01"),
            _row(2, SECOND, "2024-10-01"),
        ]
    )

    assert assigned["clinical_episode_id"].nunique() == 2
    assert not EPISODES.intervals_are_compatible(INITIAL, SECOND)


def test_temporal_rescue_merges_complementary_episodes_within_30_days() -> None:
    assigned, _, audit, _ = _run(
        [
            _row(1, "Interval A", "2024-03-01", essdai=4),
            _row(2, "Interval B", "2024-03-20", esspri=6),
        ]
    )

    assert assigned["clinical_episode_id"].nunique() == 1
    merged = audit.loc[audit["merged"]].iloc[0]
    assert merged["merge_rule"] == "temporal_rescue_within_30_days"
    assert merged["days_apart"] == 19


def test_temporal_rescue_does_not_merge_beyond_30_days() -> None:
    assigned, _, audit, _ = _run(
        [
            _row(1, "Interval A", "2024-03-01", essdai=4),
            _row(2, "Interval B", "2024-04-15", esspri=6),
        ]
    )

    assert assigned["clinical_episode_id"].nunique() == 2
    assert audit.empty


def test_conflicting_values_are_preserved_and_reported() -> None:
    assigned, _, _, conflicts = _run(
        [
            _row(1, "Interval A", "2024-03-01", variable_x=2, essdai=4),
            _row(2, "Interval B", "2024-03-20", variable_x=3, esspri=6),
        ]
    )

    collapsed = EPISODES.collapse_episode_rows(assigned)
    assert collapsed["variable_x"] == "2 | 3"
    conflict = conflicts.loc[conflicts["variable"].eq("variable_x")].iloc[0]
    assert conflict["values_found"] == "2 | 3"
    assert conflict["n_distinct_values"] == 2
    assert conflict["merge_stage"] == "temporal_rescue"
    assert conflict["merge_rule"] == "temporal_rescue_within_30_days"


def test_equal_values_collapse_once_without_conflict() -> None:
    assigned, _, _, conflicts = _run(
        [
            _row(1, "Interval A", "2024-03-01", variable_x=2, essdai=4),
            _row(2, "Interval B", "2024-03-20", variable_x=2, esspri=6),
        ]
    )

    collapsed = EPISODES.collapse_episode_rows(assigned)
    assert collapsed["variable_x"] == 2
    assert not conflicts["variable"].eq("variable_x").any()


def test_row_assignment_conservation() -> None:
    rows = [
        _row(1, "Interval A", "2024-01-01", essdai=4),
        _row(2, "Interval A", "2024-09-01", esspri=6),
        _row(3, "Interval A", "2025-01-02", essdai=5),
    ]
    source, _ = EPISODES.prepare_visits(pd.DataFrame(rows))
    assigned, _, _, _ = _run(rows)

    assert EPISODES.validate_final_assignments(source, assigned) == (0, 0)
    assert len(assigned) == len(source)
    assert assigned["row_id_raw"].is_unique


def test_not_complementary_is_audited_without_merging() -> None:
    assigned, _, audit, _ = _run(
        [
            _row(1, "Interval A", "2024-03-01", essdai=4),
            _row(2, "Interval B", "2024-03-20", essdai=5),
        ]
    )

    assert assigned["clinical_episode_id"].nunique() == 2
    assert audit.iloc[0]["merge_rule"] == "not_complementary"
    assert not audit.iloc[0]["merged"]


def test_prepare_visits_rejects_duplicate_raw_ids() -> None:
    rows = [
        _row(1, "Interval A", "2024-01-01"),
        _row(1, "Interval B", "2024-01-02"),
    ]
    with pytest.raises(ValueError, match="row_id_raw must be complete and unique"):
        EPISODES.prepare_visits(pd.DataFrame(rows))


def test_optional_does_not_bridge_two_distinct_phase_intervals() -> None:
    assigned, _, _, _ = _run(
        [
            _row(1, INITIAL, "2024-01-01"),
            _row(2, "Optional Evaluation 1", "2024-06-01"),
            _row(3, SECOND, "2024-12-01"),
        ]
    )

    assert assigned["clinical_episode_id"].nunique() == 2


def test_collapse_values_drops_missing_and_deduplicates() -> None:
    assert EPISODES.collapse_values([pd.NA, "Yes", "Yes", "No"]) == "Yes | No"
    assert pd.isna(EPISODES.collapse_values([pd.NA, "NA", ""]))
