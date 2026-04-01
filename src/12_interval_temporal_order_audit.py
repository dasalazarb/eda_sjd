from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import ANALYTIC_DIR, REPORTS_DIR, print_kv, print_script_overview, print_step, resolve_canonical_column, setup_logger

PLOTS_DIR = REPORTS_DIR / "visit_patterns"

EXPECTED_INTERVAL_ORDER: list[str] = [
    "Natural History Protocol 478 Interval",
    "Phase 1: Initial Full Evaluation",
    "Phase 1: Second Full Evaluation",
    "Phase 1: Final Full (Third Full) Evaluation",
    "Phase 2: 4th Full Evaluation",
    "Phase 2: 5th Full Evaluation",
]


def _resolve_visit_date(visits: pd.DataFrame) -> str:
    try:
        return resolve_canonical_column(visits, "visit_date")
    except KeyError:
        dt_col = resolve_canonical_column(visits, "visit_datetime")
        visits["visit_date"] = pd.to_datetime(visits[dt_col], errors="coerce").dt.normalize()
        return "visit_date"


def _resolve_subject(visits: pd.DataFrame) -> str:
    try:
        return resolve_canonical_column(visits, "subject_number")
    except KeyError:
        return resolve_canonical_column(visits, "patient_record_number")


def _prepare_base(visits: pd.DataFrame) -> tuple[pd.DataFrame, str, str, str]:
    subject_col = _resolve_subject(visits)
    interval_col = resolve_canonical_column(visits, "interval_name")
    visit_date_col = _resolve_visit_date(visits)

    base = visits[[subject_col, interval_col, visit_date_col]].copy()
    base[interval_col] = base[interval_col].astype("string").str.strip()
    base[visit_date_col] = pd.to_datetime(base[visit_date_col], errors="coerce")
    base = base.dropna(subset=[subject_col, interval_col, visit_date_col])
    base = base[base[interval_col].isin(EXPECTED_INTERVAL_ORDER)].copy()

    rank_map = {name: idx + 1 for idx, name in enumerate(EXPECTED_INTERVAL_ORDER)}
    base["interval_rank"] = base[interval_col].map(rank_map)
    return base, subject_col, interval_col, visit_date_col


def _build_order_violations(
    base: pd.DataFrame,
    subject_col: str,
    interval_col: str,
    visit_date_col: str,
) -> pd.DataFrame:
    ordered = base.sort_values([subject_col, visit_date_col, "interval_rank"]).copy()
    ordered["prev_interval"] = ordered.groupby(subject_col)[interval_col].shift(1)
    ordered["prev_interval_rank"] = ordered.groupby(subject_col)["interval_rank"].shift(1)
    ordered["prev_visit_date"] = ordered.groupby(subject_col)[visit_date_col].shift(1)

    violations = ordered[
        ordered["prev_interval_rank"].notna() & (ordered["interval_rank"] < ordered["prev_interval_rank"])
    ].copy()
    violations = violations.rename(
        columns={
            interval_col: "current_interval",
            visit_date_col: "current_visit_date",
        }
    )
    return violations[
        [
            subject_col,
            "prev_interval",
            "prev_visit_date",
            "current_interval",
            "current_visit_date",
            "prev_interval_rank",
            "interval_rank",
        ]
    ]


def _build_summary(
    base: pd.DataFrame,
    subject_col: str,
    interval_col: str,
    visit_date_col: str,
    violations: pd.DataFrame,
) -> pd.DataFrame:
    total_subjects = int(base[subject_col].nunique())
    subjects_with_violation = int(violations[subject_col].nunique()) if not violations.empty else 0

    summary_rows: list[dict[str, object]] = [
        {"metric": "total_rows_analyzed", "value": int(len(base))},
        {"metric": "total_subjects_analyzed", "value": total_subjects},
        {"metric": "subjects_with_temporal_order_violations", "value": subjects_with_violation},
        {
            "metric": "pct_subjects_with_violations",
            "value": round((100 * subjects_with_violation / total_subjects), 2) if total_subjects else 0.0,
        },
        {"metric": "total_violation_events", "value": int(len(violations))},
    ]

    first_dates = (
        base.groupby([subject_col, interval_col], as_index=False)[visit_date_col]
        .min()
        .rename(columns={visit_date_col: "first_date_in_interval"})
    )

    rank_map = {name: idx + 1 for idx, name in enumerate(EXPECTED_INTERVAL_ORDER)}
    for prev_name, curr_name in zip(EXPECTED_INTERVAL_ORDER[:-1], EXPECTED_INTERVAL_ORDER[1:]):
        prev_df = first_dates[first_dates[interval_col] == prev_name][[subject_col, "first_date_in_interval"]].rename(
            columns={"first_date_in_interval": "prev_first_date"}
        )
        curr_df = first_dates[first_dates[interval_col] == curr_name][[subject_col, "first_date_in_interval"]].rename(
            columns={"first_date_in_interval": "curr_first_date"}
        )
        pair = prev_df.merge(curr_df, on=subject_col, how="inner")
        if pair.empty:
            inversion_n = 0
            pair_n = 0
        else:
            inversion_n = int((pair["curr_first_date"] < pair["prev_first_date"]).sum())
            pair_n = int(len(pair))

        summary_rows.append(
            {
                "metric": f"pair_inversion_count_r{rank_map[prev_name]}_to_r{rank_map[curr_name]}",
                "value": inversion_n,
                "details": f"{prev_name} -> {curr_name}",
            }
        )
        summary_rows.append(
            {
                "metric": f"pair_subjects_with_both_intervals_r{rank_map[prev_name]}_to_r{rank_map[curr_name]}",
                "value": pair_n,
                "details": f"{prev_name} -> {curr_name}",
            }
        )

    return pd.DataFrame(summary_rows)


def _build_interval_distribution(base: pd.DataFrame, interval_col: str, visit_date_col: str) -> pd.DataFrame:
    dist = (
        base.groupby(interval_col)[visit_date_col]
        .agg(["count", "min", "median", "max"])
        .reset_index()
        .rename(columns={"count": "n_visits", "min": "date_min", "median": "date_p50", "max": "date_max"})
    )
    dist["interval_rank"] = dist[interval_col].map({name: i + 1 for i, name in enumerate(EXPECTED_INTERVAL_ORDER)})
    return dist.sort_values("interval_rank")


def _plot_interval_distribution(base: pd.DataFrame, interval_col: str, visit_date_col: str, output_path: Path) -> None:
    plot_df = base.copy()
    plot_df["visit_date_num"] = mdates.date2num(plot_df[visit_date_col])

    plt.figure(figsize=(13, 7))
    ax = sns.boxplot(
        data=plot_df,
        x=interval_col,
        y="visit_date_num",
        order=EXPECTED_INTERVAL_ORDER,
        color="#9ecae1",
        fliersize=1.5,
    )
    sns.stripplot(
        data=plot_df,
        x=interval_col,
        y="visit_date_num",
        order=EXPECTED_INTERVAL_ORDER,
        color="#08519c",
        alpha=0.22,
        size=2,
        ax=ax,
    )

    ax.set_title("Distribución de fechas por INTERVAL_NAME (orden esperado V1→V6)")
    ax.set_xlabel("INTERVAL_NAME")
    ax.set_ylabel("ids__visit_date")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")

    ax.yaxis_date()
    ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> None:
    logger = setup_logger("12_interval_temporal_order_audit")
    print_script_overview(
        "12_interval_temporal_order_audit.py",
        "Validates temporal ordering for the expected V1→V6 intervals and exports a visual date-distribution audit.",
    )

    print_step(1, "Load visits_long")
    visits = pd.read_parquet(ANALYTIC_DIR / "visits_long.parquet")

    print_step(2, "Resolve canonical columns and filter expected intervals")
    base, subject_col, interval_col, visit_date_col = _prepare_base(visits)

    print_step(3, "Compute temporal-order violations")
    violations = _build_order_violations(base, subject_col, interval_col, visit_date_col)

    print_step(4, "Build summary tables")
    summary = _build_summary(base, subject_col, interval_col, visit_date_col, violations)
    distribution = _build_interval_distribution(base, interval_col, visit_date_col)

    print_step(5, "Persist outputs and chart")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = PLOTS_DIR / "interval_temporal_order_summary.csv"
    violations_path = PLOTS_DIR / "interval_temporal_order_violations.csv"
    dist_path = PLOTS_DIR / "interval_date_distribution_by_interval.csv"
    plot_path = PLOTS_DIR / "interval_date_distribution_boxplot.png"

    summary.to_csv(summary_path, index=False)
    violations.to_csv(violations_path, index=False)
    distribution.to_csv(dist_path, index=False)
    _plot_interval_distribution(base, interval_col, visit_date_col, plot_path)

    print_kv(
        "Temporal-order audit",
        {
            "rows_analyzed": len(base),
            "subjects_analyzed": base[subject_col].nunique(),
            "violation_events": len(violations),
            "subjects_with_violations": violations[subject_col].nunique() if not violations.empty else 0,
            "summary_csv": str(summary_path),
            "violations_csv": str(violations_path),
            "distribution_csv": str(dist_path),
            "plot": str(plot_path),
        },
    )
    logger.info("Temporal order audit completed: rows=%d violations=%d", len(base), len(violations))


if __name__ == "__main__":
    main()
