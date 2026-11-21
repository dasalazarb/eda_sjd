"""Protect the public pipeline plan without running the clinical producers."""

from pathlib import Path
import shlex
import shutil
import subprocess
import sys


def test_list_steps_is_read_only_and_respects_dependency_order(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    shutil.copytree(root / "src", tmp_path / "src")
    result = subprocess.run(
        [sys.executable, "src/00_super_main.py", "--list-steps"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    commands = [shlex.split(line) for line in result.stdout.splitlines()]
    assert [Path(command[1]).stem for command in commands] == [
        "01_ingest",
        "04_dedup_rules",
        "05_build_backbone",
        "09_interval_collapse_audit",
        "09b_merge_essdai_versions",
        "08c_build_clinical_episode_map",
        "09c_clinical_episode_collapse",
        "09d_finalize_clinical_episode_dataset",
        "10_clinical_baseline_qc",
        "11_build_clinical_episode_spine",
        "19_filter_btris_patients",
        "20c_btris_lab_rule_harmonization",
        "20_btris_visit_date_match_report",
        "20b_btris_baseline_labs",
    ]
    assert all(command[0] == sys.executable for command in commands)
    assert commands[3][2:] == ["--collapse", "yes"]
    assert all((tmp_path / command[1]).is_file() for command in commands)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["src"]
    assert not (tmp_path / "src" / "__pycache__").exists()
