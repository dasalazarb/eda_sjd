from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from common import ANALYTIC_DIR, REPORTS_DIR, print_kv, print_script_overview, print_step, resolve_canonical_column, setup_logger

PLOTS_DIR = REPORTS_DIR / "visit_patterns"


def _resolve_visit_date(visits: pd.DataFrame) -> str:
    """Resolve visit date column; fallback to canonical visit_datetime normalized to date."""
    try:
        return resolve_canonical_column(visits, "visit_date")
    except KeyError:
        visit_datetime_col = resolve_canonical_column(visits, "visit_datetime")
        visits["visit_date"] = pd.to_datetime(visits[visit_datetime_col], errors="coerce").dt.normalize()
        return "visit_date"


def _build_visit_sequence(visits: pd.DataFrame, subject_col: str, visit_date_col: str, interval_col: str) -> pd.DataFrame:
    seq = visits[[subject_col, visit_date_col, interval_col]].copy()
    seq[visit_date_col] = pd.to_datetime(seq[visit_date_col], errors="coerce")
    seq = seq.dropna(subset=[subject_col, visit_date_col]).sort_values([subject_col, visit_date_col])

    seq["visit_order"] = seq.groupby(subject_col).cumcount() + 1
    seq["first_visit_date"] = seq.groupby(subject_col)[visit_date_col].transform("min")
    seq["day_from_first"] = (seq[visit_date_col] - seq["first_visit_date"]).dt.days
    seq["interval_name"] = seq[interval_col].astype("string").fillna("(missing)")

    return seq


def _build_transition_gaps(seq: pd.DataFrame, subject_col: str, visit_date_col: str) -> pd.DataFrame:
    gap = seq.copy()
    gap["prev_visit_date"] = gap.groupby(subject_col)[visit_date_col].shift(1)
    gap["prev_visit_order"] = gap.groupby(subject_col)["visit_order"].shift(1)
    gap["prev_interval_name"] = gap.groupby(subject_col)["interval_name"].shift(1)
    gap["gap_days"] = (gap[visit_date_col] - gap["prev_visit_date"]).dt.days

    gap = gap[gap["prev_visit_date"].notna()].copy()
    gap["transition_order"] = (
        "V"
        + gap["prev_visit_order"].astype(int).astype(str)
        + "→V"
        + gap["visit_order"].astype(int).astype(str)
    )
    gap["transition_interval"] = gap["prev_interval_name"].astype(str) + "→" + gap["interval_name"].astype(str)
    return gap


def _plot_swimmer(seq: pd.DataFrame, subject_col: str, output_path: Path) -> None:
    patient_summary = (
        seq.groupby(subject_col, as_index=False)
        .agg(max_day=("day_from_first", "max"), n_visits=("visit_order", "max"))
        .sort_values(["n_visits", "max_day"], ascending=[False, False])
        .reset_index(drop=True)
    )
    patient_summary["patient_rank"] = np.arange(len(patient_summary))

    swim_points = seq.merge(patient_summary[[subject_col, "patient_rank"]], on=subject_col, how="left")

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.hlines(patient_summary["patient_rank"], xmin=0, xmax=patient_summary["max_day"], color="#d9d9d9", linewidth=1)
    scatter = ax.scatter(
        swim_points["day_from_first"],
        swim_points["patient_rank"],
        c=swim_points["visit_order"],
        cmap="viridis",
        s=18,
        alpha=0.85,
    )
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Visit order")

    ax.set_title("Swimmer plot: timeline de visitas por paciente")
    ax.set_xlabel("Días desde la primera visita")
    ax.set_ylabel("Pacientes (ordenados por número de visitas)")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_violin(gaps: pd.DataFrame, output_path: Path) -> None:
    violin_data = gaps[gaps["gap_days"].notna() & (gaps["gap_days"] >= 0)].copy()
    top_transitions = violin_data["transition_order"].value_counts().head(8).index
    violin_data = violin_data[violin_data["transition_order"].isin(top_transitions)]

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.violinplot(
        data=violin_data,
        x="transition_order",
        y="gap_days",
        inner="quartile",
        cut=0,
        ax=ax,
        color="#97d5c9",
    )
    ax.set_title("Distribución de días entre visitas por transición")
    ax.set_xlabel("Transición")
    ax.set_ylabel("Días entre visitas")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_kde_hist(gaps: pd.DataFrame, output_path: Path) -> None:
    dist = gaps[gaps["gap_days"].notna() & (gaps["gap_days"] > 0)].copy()

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(data=dist, x="gap_days", bins=40, kde=True, stat="count", ax=ax, color="#d17c58", alpha=0.45)
    ax.set_xscale("log")
    ax.set_title("KDE + histograma de intervalos entre visitas (escala log)")
    ax.set_xlabel("Días entre visitas (log)")
    ax.set_ylabel("Conteo")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_heatmap(seq: pd.DataFrame, subject_col: str, output_path: Path) -> pd.DataFrame:
    patient_rank = (
        seq.groupby(subject_col, as_index=False)["visit_order"]
        .max()
        .rename(columns={"visit_order": "n_visits"})
        .sort_values("n_visits", ascending=False)
        .reset_index(drop=True)
    )
    patient_rank["patient_rank"] = np.arange(len(patient_rank))

    heat_data = seq.merge(patient_rank[[subject_col, "patient_rank"]], on=subject_col, how="left")
    heat_data["time_bin_30d"] = (heat_data["day_from_first"] // 30).astype(int)

    matrix = (
        heat_data.groupby(["patient_rank", "time_bin_30d"], as_index=False)
        .size()
        .pivot(index="patient_rank", columns="time_bin_30d", values="size")
        .fillna(0)
    )

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(matrix, cmap="Blues", ax=ax, cbar_kws={"label": "N° visitas"})
    ax.set_title("Heatmap paciente × tiempo (bins de 30 días desde baseline)")
    ax.set_xlabel("Bin de tiempo (30 días)")
    ax.set_ylabel("Paciente (ordenado por n_visits)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    matrix_out = matrix.reset_index()
    return matrix_out


def main() -> None:
    logger = setup_logger("08_visit_patterns")

    print_script_overview(
        "08_visit_patterns.py",
        "Analyzes patient visit dynamics and exports complementary datasets + plots for visit patterns.",
    )

    print_step(1, "Load visits_long and resolve canonical columns")
    visits = pd.read_parquet(ANALYTIC_DIR / "visits_long.parquet")
    subject_col = resolve_canonical_column(visits, "subject_number")
    visit_date_col = _resolve_visit_date(visits)
    interval_col = resolve_canonical_column(visits, "interval_name")

    print_step(2, "Build visit sequence and general interval map")
    seq = _build_visit_sequence(visits, subject_col, visit_date_col, interval_col)
    interval_map = (
        seq.groupby("interval_name", dropna=False, as_index=False)
        .agg(n_visits=("interval_name", "size"), n_patients=(subject_col, "nunique"))
        .sort_values("n_visits", ascending=False)
    )

    print_step(3, "Build transition gaps and plot datasets")
    gaps = _build_transition_gaps(seq, subject_col, visit_date_col)

    plot_swimmer_data = seq[[subject_col, "visit_order", "interval_name", visit_date_col, "day_from_first"]].copy()
    plot_violin_data = gaps[[subject_col, "transition_order", "transition_interval", "gap_days"]].copy()
    plot_dist_data = gaps[[subject_col, "gap_days", "transition_order"]].copy()

    print_step(4, "Save plot-ready datasets")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    interval_map.to_csv(PLOTS_DIR / "interval_name_map.csv", index=False)
    plot_swimmer_data.to_csv(PLOTS_DIR / "plot_data_swimmer.csv", index=False)
    plot_violin_data.to_csv(PLOTS_DIR / "plot_data_violin.csv", index=False)
    plot_dist_data.to_csv(PLOTS_DIR / "plot_data_kde_hist.csv", index=False)

    print_step(5, "Render and save plots")
    _plot_swimmer(seq, subject_col, PLOTS_DIR / "swimmer_plot.png")
    _plot_violin(gaps, PLOTS_DIR / "violin_transition_plot.png")
    _plot_kde_hist(gaps, PLOTS_DIR / "kde_hist_gapdays_plot.png")
    heatmap_matrix = _plot_heatmap(seq, subject_col, PLOTS_DIR / "heatmap_patient_time.png")
    heatmap_matrix.to_csv(PLOTS_DIR / "plot_data_heatmap_matrix.csv", index=False)

    print_kv(
        "Visit patterns summary",
        {
            "n_patients": int(seq[subject_col].nunique(dropna=True)),
            "n_visits": int(len(seq)),
            "n_transitions": int(len(gaps)),
            "n_interval_names": int(interval_map["interval_name"].nunique(dropna=False)),
            "output_dir": str(PLOTS_DIR),
        },
    )
    logger.info("Saved visit pattern outputs in %s", PLOTS_DIR)


if __name__ == "__main__":
    main()
