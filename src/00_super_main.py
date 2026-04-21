from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from common import ROOT, print_kv, print_script_overview, print_step, setup_logger


@dataclass(frozen=True)
class SuperMainConfig:
    collapse: str
    include_plots_10: bool


def _parse_args() -> SuperMainConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Orchestrates the full CTDB pipeline by running all numbered scripts in order."
        )
    )
    parser.add_argument(
        "--collapse",
        default="yes",
        choices=["yes", "no"],
        help="Value forwarded to 09_interval_collapse_audit.py --collapse (default: yes).",
    )
    parser.add_argument(
        "--skip-10",
        action="store_true",
        help="Skip 10_interval_collapse_plots.py.",
    )
    args = parser.parse_args()
    return SuperMainConfig(collapse=args.collapse, include_plots_10=not args.skip_10)


def _build_pipeline_commands(config: SuperMainConfig) -> list[list[str]]:
    scripts = [
        ["python", "src/01_ingest.py"],
        ["python", "src/02_profile_individual.py"],
        ["python", "src/03_linkage.py"],
        ["python", "src/04_dedup_rules.py"],
        ["python", "src/05_build_backbone.py"],
        ["python", "src/06_postmerge_eda.py"],
        ["python", "src/07_build_cohorts.py"],
        ["python", "src/08_visit_patterns.py"],
        ["python", "src/09_interval_collapse_audit.py", "--collapse", config.collapse],
    ]
    if config.collapse == "yes":
        scripts.append(["python", "src/09b_merge_essdai_versions.py"])
    if config.include_plots_10:
        scripts.append(["python", "src/10_interval_collapse_plots.py"])
    return scripts


def _run_one(command: list[str], logger) -> float:
    label = " ".join(command)
    logger.info("Starting command: %s", label)
    started = time.perf_counter()
    result = subprocess.run(command, cwd=ROOT)
    elapsed = time.perf_counter() - started
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {label}")
    logger.info("Completed command in %.2fs: %s", elapsed, label)
    return elapsed


def main() -> None:
    config = _parse_args()
    logger = setup_logger("00_super_main")

    print_script_overview(
        "00_super_main.py",
        "Runs all numbered pipeline scripts in order and stops immediately if any script fails.",
    )

    commands = _build_pipeline_commands(config)
    total = len(commands)
    timings: list[tuple[str, float]] = []

    print_step(1, "Build ordered command list")
    print_kv(
        "Pipeline options",
        {
            "collapse_for_script_09": config.collapse,
            "include_script_10": config.include_plots_10,
            "total_commands": total,
        },
    )

    for idx, cmd in enumerate(commands, start=1):
        label = " ".join(cmd)
        print_step(idx + 1, f"Run: {label}")
        elapsed = _run_one(cmd, logger)
        timings.append((Path(cmd[1]).name, elapsed))

    print_step(total + 2, "Pipeline completed successfully")
    print_kv(
        "Execution summary (seconds)",
        {name: round(sec, 2) for name, sec in timings},
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
