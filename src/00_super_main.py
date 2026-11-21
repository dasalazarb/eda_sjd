"""Build the CTDB and BTRIS inputs consumed by obj1_sjd."""

from __future__ import annotations

import argparse
import logging
import shlex
import subprocess
import sys
import time
from pathlib import Path


def _build_pipeline_commands() -> list[list[str]]:
    """Return the producer commands in dependency order."""
    steps = [
        ["src/01_ingest.py"],
        ["src/04_dedup_rules.py"],
        ["src/05_build_backbone.py"],
        ["src/09_interval_collapse_audit.py", "--collapse", "yes"],
        ["src/09b_merge_essdai_versions.py"],
        ["src/08c_build_clinical_episode_map.py"],
        ["src/09c_clinical_episode_collapse.py"],
        ["src/09d_finalize_clinical_episode_dataset.py"],
        ["src/10_clinical_baseline_qc.py"],
        ["src/11_build_clinical_episode_spine.py"],
        ["src/19_filter_btris_patients.py"],
        ["src/20c_btris_lab_rule_harmonization.py"],
        ["src/20_btris_visit_date_match_report.py"],
        ["src/20b_btris_baseline_labs.py"],
    ]
    return [[sys.executable, *step] for step in steps]


def _run_one(command: list[str], root: Path, logger: logging.Logger) -> None:
    """Run one producer from the repository root and propagate failures."""
    label = shlex.join(command)
    logger.info("Starting command: %s", label)
    started = time.perf_counter()
    subprocess.run(command, cwd=root, check=True)
    logger.info("Completed in %.2fs: %s", time.perf_counter() - started, label)


def main() -> None:
    """List the pipeline or run it until the first failed step."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list-steps",
        action="store_true",
        help="List the commands in execution order without creating files.",
    )
    args = parser.parse_args()
    commands = _build_pipeline_commands()
    if args.list_steps:
        for command in commands:
            print(shlex.join(command))
        return

    from common import ROOT, setup_logger

    logger = setup_logger("00_super_main")
    for command in commands:
        _run_one(command, ROOT, logger)
    logger.info("Completed all %d pipeline steps", len(commands))


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(f"Pipeline stopped after exit {exc.returncode}: {shlex.join(exc.cmd)}")
