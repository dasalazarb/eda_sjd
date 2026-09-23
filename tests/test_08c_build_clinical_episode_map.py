"""Synthetic tests for interval-first clinical episode reconstruction."""

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

FULL = "Phase 1: Second Full Evaluation"
NH = "Natural History Protocol 478 Interval"


def _row(
    row_id: int, interval: str, date: str, component: str, **extra: object
) -> dict[str, object]:
    row: dict[str, object] = {
        "patient_id": 1,
        "row_id_raw": row_id,
        "interval_name": interval,
        "visit_date": date,
        "essdai__domain": pd.NA,
        "essdai__essdai_total_score": pd.NA,
        "esspri_questionnaire__pain": pd.NA,
        "systems_review_for_physician__done": pd.NA,
        "physical_examination__done": pd.NA,
        "visit_summary_form__done": pd.NA,
        "eye_examination__done": pd.NA,
        "salivary_flow_form__value": pd.NA,
        "oral_exam_form__done": pd.NA,
    }
    columns = {
        "essdai": "essdai__domain",
        "essdai_total": "essdai__essdai_total_score",
        "esspri": "esspri_questionnaire__pain",
        "systems": "systems_review_for_physician__done",
        "physical": "physical_examination__done",
        "summary": "visit_summary_form__done",
        "eye": "eye_examination__done",
        "salivary": "salivary_flow_form__value",
        "oral": "oral_exam_form__done",
    }
    for item in component.split("+"):
        row[columns[item]] = 1
    row.update(extra)
    return row


def _run(
    rows: list[dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source = pd.DataFrame(rows)
    prepared, provenance = EPISODES.prepare_visits(source)
    flagged = EPISODES.add_presence_flags(prepared)
    units = EPISODES.build_atomic_activity_units(flagged, provenance)
    assigned_units = EPISODES.assign_episodes(units)
    assigned = EPISODES.propagate_episode_assignments(flagged, assigned_units)
    manifest = EPISODES.build_manifest(assigned)
    return assigned, manifest, assigned_units.attrs["merge_decision_audit"]


def test_write_parquet_and_csv_omits_runtime_dataframe_attrs(tmp_path: Path) -> None:
    frame = pd.DataFrame({"patient_id": [1], "clinical_episode_id": ["episode-1"]})
    audit = pd.DataFrame({"merge_performed": [True]})
    frame.attrs["merge_decision_audit"] = audit
    parquet_path = tmp_path / "08c_row_map.parquet"

    written_parquet, written_csv = EPISODES.write_parquet_and_csv(frame, parquet_path)

    assert written_parquet == parquet_path
    assert written_csv == parquet_path.with_suffix(".csv")
    parquet_frame = pd.read_parquet(written_parquet)
    pd.testing.assert_frame_equal(parquet_frame, frame)
    assert parquet_frame.attrs == {}
    pd.testing.assert_frame_equal(pd.read_csv(written_csv), frame)
    assert frame.attrs["merge_decision_audit"] is audit


def test_same_date_different_complete_assessments_stay_separate() -> None:
    assigned, _, _ = _run(
        [
            _row(1, FULL, "2024-04-05", "essdai+systems+physical"),
            _row(2, NH, "2024-04-05", "essdai+systems+physical"),
        ]
    )
    assert assigned["clinical_episode_id"].nunique() == 2


def test_exact_interval_complementary_rows_merge_without_30_day_limit() -> None:
    assigned, manifest, _ = _run(
        [
            _row(1, FULL, "2023-01-01", "essdai+systems"),
            _row(2, FULL, "2023-05-01", "esspri+eye+salivary"),
        ]
    )
    assert assigned["clinical_episode_id"].nunique() == 1
    assert manifest.loc[0, "episode_span_days"] == 120
    assert manifest.loc[0, "exceptional_merge"]


def test_exact_interval_two_complete_assessments_stay_separate() -> None:
    assigned, _, audit = _run(
        [
            _row(1, FULL, "2023-01-01", "essdai+systems+physical"),
            _row(2, FULL, "2023-10-01", "essdai+systems+physical"),
        ]
    )
    assert assigned["clinical_episode_id"].nunique() == 2
    assert audit["duplicate_complete_assessment"].any()


def test_natural_history_and_15d_optional_are_family_aliases() -> None:
    assigned, _, audit = _run(
        [
            _row(1, NH, "2023-05-01", "essdai+systems"),
            _row(2, "15D Optional Evaluation 2", "2023-05-12", "esspri+eye"),
        ]
    )
    assert assigned["clinical_episode_id"].nunique() == 1
    merged = audit.loc[audit["merge_performed"]].iloc[0]
    assert merged["interval_compatibility_type"] == "natural_history_family_alias"


def test_optional_can_complement_any_full_without_ordinal_mapping() -> None:
    assigned, _, audit = _run(
        [
            _row(1, FULL, "2023-05-01", "essdai+systems"),
            _row(2, "Optional Evaluation 3", "2023-05-12", "esspri+eye"),
        ]
    )
    assert assigned["clinical_episode_id"].nunique() == 1
    assert (
        audit.loc[audit["merge_performed"], "merge_reason"]
        .eq("optional_full_family_merge")
        .any()
    )


def test_cross_year_incomplete_fragment_merges_with_strict_qc() -> None:
    assigned, manifest, _ = _run(
        [
            _row(1, FULL, "2023-05-01", "essdai+systems+physical"),
            _row(2, FULL, "2025-05-02", "eye"),
        ]
    )
    assert assigned["clinical_episode_id"].nunique() == 1
    assert manifest.loc[
        0,
        [
            "cross_year_merge",
            "suspected_date_error",
            "exceptional_merge",
            "manual_review_required",
        ],
    ].all()
    assert manifest.loc[0, "clinical_anchor_date"] == pd.Timestamp("2023-05-01")


def test_cross_year_complete_assessments_do_not_merge() -> None:
    assigned, _, _ = _run(
        [
            _row(1, FULL, "2023-05-01", "essdai+systems+physical"),
            _row(2, FULL, "2025-05-02", "essdai+systems+physical"),
        ]
    )
    assert assigned["clinical_episode_id"].nunique() == 2


def test_temporal_rescue_different_intervals_within_30_days() -> None:
    assigned, _, audit = _run(
        [
            _row(1, "Unscheduled A", "2023-05-01", "essdai+systems"),
            _row(2, "Unscheduled B", "2023-05-21", "esspri+eye"),
        ]
    )
    assert assigned["clinical_episode_id"].nunique() == 1
    merged = audit.loc[audit["merge_performed"]].iloc[0]
    assert merged["merge_stage"] == "temporal_rescue"
    assert merged["cross_interval_merge"]


def test_temporal_rescue_does_not_cross_30_days() -> None:
    assigned, _, _ = _run(
        [
            _row(1, "Unscheduled A", "2023-05-01", "essdai+systems"),
            _row(2, "Unscheduled B", "2023-06-02", "esspri+eye"),
        ]
    )
    assert assigned["clinical_episode_id"].nunique() == 2


def test_same_calendar_month_is_recorded() -> None:
    _, _, audit = _run(
        [
            _row(1, "Unscheduled A", "2023-08-01", "essdai+systems"),
            _row(2, "Unscheduled B", "2023-08-30", "esspri+eye"),
        ]
    )
    merged = audit.loc[audit["merge_performed"]].iloc[0]
    assert merged["same_calendar_month"]


def test_nearby_complete_assessments_do_not_merge() -> None:
    assigned, _, _ = _run(
        [
            _row(1, "Unscheduled A", "2023-08-01", "essdai+systems+physical"),
            _row(2, "Unscheduled B", "2023-08-06", "essdai+systems+physical"),
        ]
    )
    assert assigned["clinical_episode_id"].nunique() == 2


def test_static_conflict_blocks_apparently_complementary_merge() -> None:
    assigned, _, audit = _run(
        [
            _row(1, FULL, "2023-08-01", "essdai+systems", sex="F"),
            _row(2, FULL, "2023-08-06", "esspri+eye", sex="M"),
        ]
    )
    assert assigned["clinical_episode_id"].nunique() == 2
    assert audit["hard_conflict"].any()


def test_vital_sign_differences_are_not_hard_conflicts() -> None:
    rows = [
        _row(1, FULL, "2023-08-01", "essdai+systems", vital_signs__pulse=70),
        _row(2, FULL, "2023-08-06", "esspri+eye", vital_signs__pulse=100),
    ]
    assigned, _, audit = _run(rows)
    assert assigned["clinical_episode_id"].nunique() == 1
    assert not audit.loc[audit["merge_performed"], "hard_conflict"].any()


def test_provenance_and_conservation_invariants() -> None:
    rows = [
        _row(1, FULL, "2023-08-01", "essdai+systems"),
        _row(2, FULL, "2023-08-06", "esspri+eye"),
    ]
    source, _ = EPISODES.prepare_visits(pd.DataFrame(rows))
    assigned, _, _ = _run(rows)
    assert EPISODES.validate_final_assignments(source, assigned) == (0, 0)
    assert assigned["row_id_raw"].is_unique
    assert (
        assigned.sort_values("row_id_raw")["collection_date"].tolist()
        == source.sort_values("row_id_raw")["collection_date"].tolist()
    )
    assert (
        assigned.sort_values("row_id_raw")["interval_name"].tolist()
        == source.sort_values("row_id_raw")["interval_name"].tolist()
    )


def test_prepare_visits_rejects_duplicate_raw_ids() -> None:
    rows = [_row(1, FULL, "2023-01-01", "eye"), _row(1, FULL, "2023-01-02", "oral")]
    with pytest.raises(ValueError, match="row_id_raw must be complete and unique"):
        EPISODES.prepare_visits(pd.DataFrame(rows))


def test_two_clinical_candidates_do_not_imply_two_complete_assessments() -> None:
    assigned, _, audit = _run(
        [
            _row(1, FULL, "2024-04-05", "essdai"),
            _row(2, FULL, "2024-04-06", "essdai+essdai_total+summary"),
        ]
    )
    assert assigned["clinical_episode_id"].nunique() == 1
    assert not audit.loc[
        audit["merge_performed"], "duplicate_complete_assessment"
    ].any()


def test_temporal_candidate_generation_excludes_distant_pairs() -> None:
    _, _, audit = _run(
        [
            _row(1, "Unscheduled A", "2023-01-01", "essdai+systems"),
            _row(2, "Unscheduled B", "2023-01-16", "esspri+eye"),
            _row(3, "Unscheduled C", "2023-10-28", "oral"),
            _row(4, "Unscheduled D", "2024-08-23", "salivary"),
        ]
    )
    temporal = audit.loc[audit["merge_stage"].eq("temporal_rescue")]
    assert len(temporal) == 1
    assert temporal.iloc[0]["candidate_generation_rule"] == "temporal_window_30d"
    assert temporal.iloc[0]["gap_days"] == 15


def test_best_target_selection_is_independent_of_source_order() -> None:
    rows = [
        _row(10, FULL, "2024-01-01", "essdai+systems"),
        _row(20, FULL, "2024-06-01", "essdai+systems"),
        _row(30, FULL, "2024-06-02", "eye"),
    ]
    first, _, _ = _run(rows)
    second, _, _ = _run([rows[2], rows[0], rows[1]])

    def merged_raw_ids(frame: pd.DataFrame) -> set[int]:
        fragment_episode = frame.loc[
            frame["row_id_raw"].eq(30), "clinical_episode_id"
        ].iloc[0]
        return set(
            frame.loc[
                frame["clinical_episode_id"].eq(fragment_episode), "row_id_raw"
            ]
        )

    assert merged_raw_ids(first) == {20, 30}
    assert merged_raw_ids(second) == {20, 30}


def test_audit_excludes_irrelevant_interval_pairs() -> None:
    _, _, audit = _run(
        [
            _row(1, "Unscheduled A", "2020-01-01", "eye"),
            _row(2, "Unscheduled B", "2021-01-01", "oral"),
            _row(3, "Unscheduled C", "2022-01-01", "salivary"),
        ]
    )
    assert audit.empty


def test_ambiguous_multiple_targets_require_manual_review() -> None:
    assigned, _, audit = _run(
        [
            _row(1, FULL, "2024-05-01", "essdai+systems"),
            _row(2, FULL, "2024-05-01", "essdai+systems"),
            _row(3, FULL, "2024-05-01", "eye"),
        ]
    )
    assert assigned["clinical_episode_id"].nunique() == 3
    assert assigned.loc[
        assigned["row_id_raw"].eq(3), "manual_review_required"
    ].all()
    assert audit["manual_review_reason"].eq(
        "ambiguous_multiple_candidate_targets"
    ).any()


def test_bucketed_generation_is_far_below_all_pairs_and_order_invariant() -> None:
    rows = []
    components = [
        "essdai+systems",
        "eye",
        "salivary",
        "oral",
        "esspri",
        "summary",
        "physical",
        "eye",
        "salivary",
        "oral",
    ]
    for bucket in range(10):
        interval = f"Synthetic interval {bucket}"
        for fragment, component in enumerate(components):
            rows.append(
                _row(
                    bucket * 10 + fragment,
                    interval,
                    f"2023-{bucket + 1:02d}-15",
                    component,
                )
            )

    first, _, first_audit = _run(rows)
    shuffled, _, shuffled_audit = _run(list(reversed(rows)))

    assert len(first_audit) < 495
    assert len(shuffled_audit) == len(first_audit)
    first_groups = first.groupby("clinical_episode_id")["row_id_raw"].apply(set)
    shuffled_groups = shuffled.groupby("clinical_episode_id")["row_id_raw"].apply(set)
    assert {frozenset(group) for group in first_groups} == {
        frozenset(group) for group in shuffled_groups
    }


def test_merge_summary_reports_desaturated_candidate_counts() -> None:
    assigned, manifest, audit = _run(
        [
            _row(1, FULL, "2023-05-01", "essdai+systems"),
            _row(2, FULL, "2023-05-02", "eye"),
        ]
    )
    summary = EPISODES.build_merge_summary(assigned, manifest, audit)
    expected = {
        "n_candidate_pairs_pass1_exact",
        "n_candidate_pairs_pass1_nh_family",
        "n_candidate_pairs_pass1_optional_full",
        "n_candidate_pairs_pass1_cross_year",
        "n_candidate_pairs_pass2_temporal",
        "n_candidates_total",
        "n_merges_total",
    }
    assert expected.issubset(summary.columns)
    assert summary.loc[0, "n_candidates_total"] == len(audit)
