"""Synthetic tests for staged, source-preserving clinical episode reconstruction."""
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


def _row(row_id: int, interval: str, date: str | None, patient_id: str = "P001", **values: object) -> dict[str, object]:
    row = {"patient_id": patient_id, "row_id_raw": row_id, "interval_name": interval, "collection_date": date, "essdai": pd.NA, "esspri": pd.NA, "eye_examination": pd.NA, "systems_review_for_physician": pd.NA, "ids__visit_date": pd.NA}
    row.update(values)
    return row


def _run(rows: list[dict[str, object]]):
    source, provenance = EPISODES.prepare_visits(pd.DataFrame(rows))
    flagged = EPISODES.add_presence_flags(source)
    units = EPISODES.build_atomic_activity_units(flagged, provenance)
    assigned_units = EPISODES.assign_episodes(units)
    assigned = EPISODES.propagate_episode_assignments(flagged, assigned_units)
    return source, assigned, EPISODES.build_manifest(assigned), assigned_units


def test_same_interval_cross_year_is_one_episode_and_warned() -> None:
    _, assigned, manifest, _ = _run([_row(1, INITIAL, "2023-01-01", essdai=4), _row(2, INITIAL, "2025-01-01", eye_examination="done")])
    assert assigned["clinical_episode_id"].nunique() == 1
    assert manifest.loc[0, "cross_year_merge"]
    assert manifest.loc[0, "long_interval_span_warning"]
    assert set(assigned["row_id_raw"]) == {1, 2}


def test_short_new_year_crossing_is_not_long_span() -> None:
    _, _, manifest, _ = _run([_row(1, INITIAL, "2024-12-29"), _row(2, INITIAL, "2025-01-03")])
    assert manifest.loc[0, "cross_year_merge"]
    assert not manifest.loc[0, "long_interval_span_warning"]


def test_same_month_day_year_difference_is_qc_hypothesis_only() -> None:
    _, assigned, manifest, _ = _run([_row(1, INITIAL, "2024-06-20"), _row(2, INITIAL, "2025-06-20")])
    assert manifest.loc[0, "possible_date_entry_error"]
    assert list(assigned["collection_date"]) == [pd.Timestamp("2024-06-20"), pd.Timestamp("2025-06-20")]


def test_patient_and_principal_phase_identity_are_never_bridged() -> None:
    _, assigned, _, _ = _run([_row(1, INITIAL, "2024-01-01"), _row(2, INITIAL, "2024-01-01", patient_id="P002"), _row(3, SECOND, "2024-01-02")])
    assert assigned["clinical_episode_id"].nunique() == 3


def test_15d_optional_attaches_to_natural_even_across_year() -> None:
    _, assigned, manifest, units = _run([_row(1, NATURAL, "2024-09-10", essdai=4), _row(2, "15D Optional Evaluation 1", "2025-09-16", esspri=5)])
    assert assigned["clinical_episode_id"].nunique() == 1
    assert set(assigned["optional_adjudication"]) == {"not_optional", "attached_to_natural_15d"}
    assert manifest.loc[0, "manual_review_required"]
    assert units.attrs["optional_adjudication"].loc[0, "decision"] == "attached_to_natural_15d"


def test_15d_optional_documented_distinct_event_stays_independent() -> None:
    _, assigned, manifest, _ = _run([_row(1, NATURAL, "2024-01-01"), _row(2, "15D Optional Evaluation 2", "2024-02-01", essdai=7, esspri=6, eye_examination="done", documented_new_visit=True)])
    assert assigned["clinical_episode_id"].nunique() == 2
    assert "optional_independent_clinical" in set(manifest["visit_type"])


def test_optional_optional_complementary_cluster_and_promotion() -> None:
    _, assigned, manifest, units = _run([_row(1, "Optional Evaluation A", "2024-05-01", esspri=5), _row(2, "Optional Evaluation B", "2024-05-04", eye_examination="done"), _row(3, "Optional Evaluation C", "2024-05-05", essdai=3, systems_review_for_physician="done")])
    assert assigned["optional_cluster_id"].nunique() == 1
    assert manifest.loc[0, "clinical_visit"]
    assert manifest.loc[0, "visit_type"] == "optional_independent_clinical"
    assert len(units.attrs["optional_pair_candidates"]) == 3


def test_optional_cluster_attaches_to_best_complementary_main() -> None:
    _, assigned, _, _ = _run([_row(1, INITIAL, "2024-04-01", essdai=4), _row(2, "Optional Evaluation A", "2024-04-05", esspri=6), _row(3, "Optional Evaluation B", "2024-04-06", eye_examination="done")])
    assert assigned["clinical_episode_id"].nunique() == 1
    assert set(assigned.loc[assigned["row_id_raw"].isin([2, 3]), "optional_adjudication"]) == {"attached_to_main"}


def test_complete_link_prevents_transitive_optional_bridge() -> None:
    _, assigned, _, _ = _run([_row(1, "Optional Evaluation A", "2024-01-01", esspri=1), _row(2, "Optional Evaluation B", "2024-06-01", eye_examination="done"), _row(3, "Optional Evaluation C", "2024-11-01", essdai=4)])
    assert assigned["optional_cluster_id"].nunique() == 2


def test_missing_date_optional_is_preserved_for_review() -> None:
    _, assigned, manifest, units = _run([_row(1, "Optional Evaluation A", None, esspri=5)])
    assert len(assigned) == 1
    assert manifest.loc[0, "visit_type"] == "optional_unresolved"
    assert len(units.attrs["unresolved_assignments"]) == 1


def test_valid_falsey_values_are_information() -> None:
    series = pd.Series([0, False, "No", -1, "NA", None])
    assert EPISODES.has_information(series).tolist() == [True, True, True, True, False, False]


def test_date_metadata_is_not_a_clinical_conflict() -> None:
    _, assigned, manifest, _ = _run([_row(1, INITIAL, "2024-01-01", ids__visit_date="2024-01-01"), _row(2, INITIAL, "2024-01-01", ids__visit_date="2024-01-02")])
    assert EPISODES.build_value_conflicts(assigned).empty
    assert not EPISODES.build_date_discrepancies(assigned, manifest).empty


def test_row_order_does_not_change_stable_membership_or_ids() -> None:
    rows = [_row(7, INITIAL, "2024-01-01"), _row(2, INITIAL, "2025-01-01"), _row(9, SECOND, "2024-06-01")]
    _, first, _, _ = _run(rows); _, second, _, _ = _run(list(reversed(rows)))
    first_map = first.set_index("row_id_raw")["clinical_episode_id"].to_dict(); second_map = second.set_index("row_id_raw")["clinical_episode_id"].to_dict()
    assert first_map == second_map


def test_conservation_and_unique_manifest_contract() -> None:
    rows = [_row(1, INITIAL, "2024-01-01"), _row(2, INITIAL, "2025-01-01"), _row(3, "Optional Evaluation 1", None)]
    source, assigned, manifest, _ = _run(rows)
    assert EPISODES.validate_final_assignments(source, assigned) == (0, 0)
    assert set(source["row_id_raw"]) == set(assigned["row_id_raw"])
    assert not assigned["row_id_raw"].duplicated().any()
    assert not manifest.duplicated(["patient_id", "clinical_episode_id"]).any()


def test_invalid_duplicate_raw_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="row_id_raw must be complete and unique"):
        EPISODES.prepare_visits(pd.DataFrame([_row(1, "A", "2024-01-01"), _row(1, "B", "2024-01-02")]))


def test_sparse_candidates_filter_distant_optional_pairs() -> None:
    rows = [
        _row(
            index,
            f"Optional Evaluation {index}",
            (
                pd.Timestamp("2000-01-01")
                + pd.DateOffset(years=index // 50)
                + pd.Timedelta(days=index % 50)
            ).strftime("%Y-%m-%d"),
            esspri=index,
        )
        for index in range(500)
    ]
    _, assigned, _, units = _run(rows)
    metrics = units.attrs["performance_metrics"]
    assert len(assigned) == 500
    assert metrics["theoretical_pairs"] == 500 * 499 // 2
    assert metrics["evaluated_pairs"] < 20_000
    assert metrics["filtered_pairs"] == metrics["theoretical_pairs"] - metrics["candidate_pairs"]


def test_explicit_link_is_exception_to_temporal_window() -> None:
    rows = [
        _row(1, "Optional Evaluation A", None, esspri=4, evaluation_id="E-10"),
        _row(2, "Optional Evaluation B", "2028-01-01", eye_examination="done", evaluation_id="E-10"),
    ]
    _, assigned, _, units = _run(rows)
    assert assigned["optional_cluster_id"].nunique() == 1
    pair = units.attrs["optional_pair_candidates"].iloc[0]
    assert pair["reason"] == "explicit same-evaluation linkage"


def test_15d_without_natural_is_built_directly() -> None:
    rows = [
        _row(1, "15D Optional Evaluation A", "2021-01-01", esspri=4),
        _row(2, "15D Optional Evaluation B", "2025-01-01", eye_examination="done"),
    ]
    _, assigned, _, units = _run(rows)
    assert assigned["clinical_episode_id"].nunique() == 1
    assert units.attrs["performance_metrics"]["evaluated_pairs"] == 0


def test_standard_conflict_provenance_preserves_all_conflicting_values() -> None:
    _, assigned, manifest, _ = _run(
        [_row(1, INITIAL, "2024-01-01", essdai=3), _row(2, INITIAL, "2024-01-02", essdai=7)]
    )
    conflicts = EPISODES.build_value_conflicts(assigned)
    full = EPISODES.build_source_value_provenance(assigned)
    standard = EPISODES.build_source_value_provenance(assigned, conflicts["variable"])
    assert set(standard["source_value"]) == {3, 7}
    assert standard.equals(full.loc[full["variable"].eq("essdai")].reset_index(drop=True))
    assert EPISODES.build_date_discrepancies(assigned, manifest).shape[0] == 1


def test_standard_provenance_is_restricted_by_episode_and_variable() -> None:
    """A conflict in one episode must not expand provenance in another episode."""
    _, assigned, _, _ = _run(
        [
            _row(1, INITIAL, "2024-01-01", essdai=3),
            _row(2, INITIAL, "2024-01-02", essdai=7),
            _row(3, SECOND, "2024-06-01", essdai=9),
        ]
    )
    conflicts = EPISODES.build_value_conflicts(assigned)
    standard = EPISODES.build_source_value_provenance(
        assigned, conflict_keys=conflicts
    )
    conflict_episode = conflicts.loc[0, "clinical_episode_id"]
    assert set(standard["clinical_episode_id"]) == {conflict_episode}
    assert set(standard["row_id_raw"]) == {1, 2}


def test_matching_does_not_scan_conflicting_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """Detailed contradictions belong to QC and cannot run in matching."""
    monkeypatch.setattr(
        EPISODES,
        "find_incompatible_variables",
        lambda *_args, **_kwargs: pytest.fail("conflict scan entered matching"),
    )
    _, assigned, _, _ = _run(
        [
            _row(1, "Optional Evaluation A", "2024-05-01", esspri=5),
            _row(2, "Optional Evaluation B", "2024-05-02", eye_examination="done"),
        ]
    )
    assert assigned["optional_cluster_id"].nunique() == 1


def test_signatures_preserve_falsey_information_and_exclude_patient_link() -> None:
    source, provenance = EPISODES.prepare_visits(
        pd.DataFrame(
            [
                _row(1, "Optional Evaluation A", "2024-01-01", essdai=0),
                _row(2, "Optional Evaluation B", "2024-01-02", essdai=False),
            ]
        )
    )
    units = EPISODES.build_atomic_activity_units(
        EPISODES.add_presence_flags(source), provenance
    )
    signatures, _, _ = EPISODES.build_row_signatures(units)
    assert all(signature.informative_mask for signature in signatures)
    assert all(not signature.explicit_links for signature in signatures)


def test_nearest_date_distance_uses_all_source_dates() -> None:
    left = pd.Series(pd.to_datetime(["2024-01-10", "2026-01-01"]))
    right = pd.Series(pd.to_datetime(["2020-01-01", "2024-01-12", "2030-01-01"]))
    assert EPISODES._nearest_date_distance(left, right) == 2
