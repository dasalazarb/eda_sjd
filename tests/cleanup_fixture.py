"""Prepare and compare synthetic data for the cleanup acceptance checks."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def write_fixture(root: Path) -> None:
    """Create only synthetic CTDB, BTRIS, and rule inputs."""
    raw = root / "data_raw"
    intermediate = root / "data_intermediate" / "BTRIS"
    raw.mkdir()
    intermediate.mkdir(parents=True)
    order, cluster = "Rheumatoid Factor", "Rheumatoid Factor (Blood)"
    protocols = {"11D": ("P1001", "Medication288444-1.csv"),
                 "15D": ("P2002", "Medication288476-1.csv")}
    header = [
        ["ids"] * 5 + ["visit_summary_form", "essdai"]
        + ["esspri_questionnaire"] * 3 + ["medications"],
        ["subject_number", "patient_record_number", "visit_date", "time_24_hour",
         "interval_name", "sjogrens_class", "essdai_total_score", "dryness",
         "fatigue", "pain", "rx_1_name"],
    ]
    initial = "Phase 1: Initial Full Evaluation"
    followup = "Phase 1: Second Full Evaluation"
    natural = "Natural History Protocol 478 Interval"
    for protocol, (patient, medication_filename) in protocols.items():
        records = [
            [patient, patient, "2020-01-01", "10:00", initial,
             1, 3, 5, 4, 3, "hydroxychloroquine"],
            [patient, patient, "2021-01-01", "10:00", followup,
             1, 5, 5, 6, 3, None],
            [patient, patient, "2021-06-01", "10:00", natural, *([None] * 6)],
        ]
        if protocol == "11D":
            records.extend([
                [patient, patient, "2023-06-01", "10:00", "Optional Evaluation 1",
                 *([None] * 6)],
                ["P3003", "P3003", "2020-01-02", "10:00", initial,
                 3, 2, 2, 2, 2, None],
            ])
        pd.DataFrame(header + records).to_excel(
            raw / f"CTDB Data Download {protocol}.xlsx", header=False, index=False
        )

        directory = raw / "BTRIS" / protocol
        directory.mkdir(parents=True)
        pd.DataFrame({
            "MRN": [patient, patient],
            "Order Name": [order, order],
            "Cluster Name": [cluster, cluster],
            "Collected Date Time": ["2020-01-01", "2021-01-01"],
            "Observation Value": ["6", "21"],
            "Reference Range": ["0-14", "0-14"],
            "Units": ["IU/mL", "IU/mL"],
            "Interpretation": ["Negative", "Positive"],
        }).to_csv(directory / "Lab Synthetic.csv", index=False)
        pd.DataFrame({
            "MRN": [patient],
            "Medication Name": ["Plaquenil"],
            "Drug Ingredient": ["hydroxychloroquine"],
            "Order Date/Start Date": ["2020-01-01"],
            "Order End Date": ["2021-01-01"],
            "Initial Consent Date": ["2019-01-01"],
        }).to_csv(directory / medication_filename, index=False)

    pd.DataFrame({"Order Name": [order]}).to_csv(raw / "unique_OrderSets.csv", index=False)
    pd.DataFrame({
        "Order Name": [order],
        "Cluster Name": [cluster],
        "Suggested Rule Family": ["CONTINUOUS_RANGE"],
    }).to_excel(
        root / "btris_lab_rule_families.xlsx", sheet_name="Lab rule map", index=False
    )
    pd.DataFrame({
        "order_name": [order] + [f"Synthetic unused order {i}" for i in range(255)],
        "cluster_name": [cluster] + [f"Synthetic unused cluster {i}" for i in range(255)],
    }).to_csv(intermediate / "20_lab_order_cluster_reference.csv", index=False)


EPISODE_KEY = ("patient_id", "clinical_episode_id")
ARTIFACTS = {
    "data_analytic/visits_long.parquet": ("row_id_raw",),
    "data_intermediate/clinical_episode_row_map.parquet": ("row_id_raw",),
    "data_analytic/clinical_episode_manifest.parquet": EPISODE_KEY,
    "data_analytic/visits_long_collapsed_by_clinical_episode_codebook_not_clean.parquet": EPISODE_KEY,
    "data_analytic/visits_long_collapsed_by_clinical_episode_codebook_not_clean.csv": EPISODE_KEY,
    "reports/clinical_episode_collapse/09c_episode_variable_conflicts.csv": (
        *EPISODE_KEY, "variable",
    ),
    "data_analytic/visits_long_collapsed_by_clinical_episode_codebook_corrected.parquet": EPISODE_KEY,
    "reports/clinical_baseline/10_clinical_baseline_comparison.csv": ("patient_id",),
    "data_analytic/clinical_episode_spine_all.parquet": EPISODE_KEY,
    "data_analytic/clinical_episode_spine_all.csv": EPISODE_KEY,
    "data_analytic/clinical_episode_spine_sjd.parquet": EPISODE_KEY,
    "data_analytic/clinical_episode_spine_sjd.csv": EPISODE_KEY,
    "data_analytic/visits_long_collapsed_by_interval_codebook_corrected.parquet": (),
    "data_analytic/visits_long_collapsed_by_interval_codebook_corrected.csv": (),
    "data_intermediate/BTRIS/11D/Lab Synthetic.csv": (),
    "data_intermediate/BTRIS/15D/Lab Synthetic.csv": (),
    "data_intermediate/BTRIS/11D/Medication288444-1.csv": (),
    "data_intermediate/BTRIS/15D/Medication288476-1.csv": (),
    "data_analytic/BTRIS/20_btris_lab_records_long.parquet": (),
    "data_analytic/BTRIS/20b_baseline_lab_selected_long.parquet": (),
    "data_analytic/BTRIS/20b_baseline_labs_patient_level.parquet": ("patient_id",),
}


def compare(baseline: Path, candidate: Path) -> None:
    """Report full schema and null counts, and fail on any content difference."""
    report = {}
    for path, keys in ARTIFACTS.items():
        read = pd.read_parquet if path.endswith(".parquet") else pd.read_csv
        original, updated = (read(root / path) for root in (baseline, candidate))
        hashes = [hashlib.sha256((root / path).read_bytes()).hexdigest()
                  for root in (baseline, candidate)]
        normalized = []
        if path == "data_analytic/BTRIS/20_btris_lab_records_long.parquet":
            for frame, root in ((original, baseline), (updated, candidate)):
                prefix = f"{root.resolve()}/"
                source = frame["source_file"].astype("string")
                assert source.str.startswith(prefix).all()
                frame["source_file"] = source.str.removeprefix(prefix)
            normalized.append("source_file (raíz absoluta de la copia)")
        pd.testing.assert_frame_equal(original, updated, check_exact=True)
        if keys:
            assert not updated[list(keys)].isna().any(axis=None), path
            assert not updated.duplicated(list(keys)).any(), path
        patient_column = next(
            (key for key in ("patient_id", "ids__patient_record_number", "MRN")
             if key in updated), None
        )
        report[path] = {
            "hash_sha256": {"baseline": hashes[0], "candidate": hashes[1]},
            "rows": len(updated),
            "patients": updated[patient_column].nunique() if patient_column else None,
            "unique_key": keys,
            "schema": updated.dtypes.astype(str).to_dict(),
            "nulls_per_column": updated.isna().sum().to_dict(),
            "value_differences": 0,
            "normalized": normalized,
        }
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    fixture = commands.add_parser("fixture")
    fixture.add_argument("root", type=Path)
    comparison = commands.add_parser("compare")
    comparison.add_argument("baseline", type=Path)
    comparison.add_argument("candidate", type=Path)
    args = parser.parse_args()
    if args.action == "fixture":
        fixture_root = args.root.resolve()
        if any((fixture_root / directory).exists() for directory in (
            "data_raw", "data_intermediate", "data_analytic",
            "btris_lab_rule_families.xlsx",
        )):
            parser.error("fixture requires an isolated checkout without data or rules")
        write_fixture(fixture_root)
    else:
        compare(args.baseline, args.candidate)


if __name__ == "__main__":
    main()
