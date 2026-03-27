from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde

from common import (
    ANALYTIC_DIR,
    EDA_UNIFIED_REPORT_PATH,
    REPORTS_DIR,
    build_targeted_eda_sheets,
    print_kv,
    print_script_overview,
    print_step,
    resolve_canonical_column,
    setup_logger,
    upsert_eda_sheets_xlsx,
)

PLOTS_DIR = REPORTS_DIR / "visit_patterns"

# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────

# Set to True to include Optional Evaluations in main plots
INCLUDE_OPTIONAL: bool = False

# Canonical phase order → mapped to V1, V2, ... in plots
# nan and Natural History Protocol are "pre-baseline" (V1)
PHASE_ORDER: list[str] = [
    "Natural History Protocol 478 Interval",       # V1
    "(missing)",                                    # V1 (NaN rows)
    "Phase 1: Initial Full Evaluation",             # V2
    "Phase 1: Second Full Evaluation",              # V3
    "Phase 1: Final Full (Third Full) Evaluation",  # V4
    "Phase 2: 4th Full Evaluation",                 # V5
    "Phase 2: 5th Full Evaluation",                 # V6
]

OPTIONAL_PHASES: list[str] = [
    "Optional Evaluation 1",
    "Optional Evaluation 2",
    "Optional Evaluation 3",
    "Optional Evaluation 4",
]

PHASE_LABELS: dict[str, str] = {
    "Natural History Protocol 478 Interval":        "V1 (Nat. Hist.)",
    "(missing)":                                    "V1 (missing)",
    "Phase 1: Initial Full Evaluation":             "V2 (Initial)",
    "Phase 1: Second Full Evaluation":              "V3 (Second)",
    "Phase 1: Final Full (Third Full) Evaluation":  "V4 (Final/3rd)",
    "Phase 2: 4th Full Evaluation":                 "V5 (4th)",
    "Phase 2: 5th Full Evaluation":                 "V6 (5th)",
}

# Canonical inter-phase transitions — ONLY these are shown in kde_by_interval
# Same-phase transitions (V2→V2, etc.) are explicitly excluded
CANONICAL_TRANSITIONS: dict[str, str] = {
    "Natural History Protocol 478 Interval → Phase 1: Initial Full Evaluation":
        "V1 (Nat.Hist.) → V2 (Initial)",
    "(missing) → Phase 1: Initial Full Evaluation":
        "V1 (missing) → V2 (Initial)",
    "Phase 1: Initial Full Evaluation → Phase 1: Second Full Evaluation":
        "V2 (Initial) → V3 (Second)",
    "Phase 1: Second Full Evaluation → Phase 1: Final Full (Third Full) Evaluation":
        "V3 (Second) → V4 (Final/3rd)",
    "Phase 1: Final Full (Third Full) Evaluation → Phase 2: 4th Full Evaluation":
        "V4 (Final/3rd) → V5 (4th)",
    "Phase 2: 4th Full Evaluation → Phase 2: 5th Full Evaluation":
        "V5 (4th) → V6 (5th)",
}

# Reference lines: label → days
TIME_REFS: dict[str, int] = {
    "1 wk":  7,
    "1 mo":  30,
    "6 mo":  182,
    "2 yr":  730,
    "4 yr":  1460,
    "6 yr":  2190,
    "8 yr":  2920,
    "10 yr": 3650,
}

# One distinct color per reference line
REF_COLORS: list[str] = [
    "#e67e22",  # 1 wk   — orange
    "#f1c40f",  # 1 mo   — yellow
    "#2ecc71",  # 6 mo   — light green
    "#2980b9",  # 2 yr   — blue
    "#27ae60",  # 4 yr   — green
    "#8e44ad",  # 6 yr   — purple
    "#c0392b",  # 8 yr   — red
    "#16a085",  # 10 yr  — teal
]

# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _resolve_visit_date(visits: pd.DataFrame) -> str:
    try:
        return resolve_canonical_column(visits, "visit_date")
    except KeyError:
        col = resolve_canonical_column(visits, "visit_datetime")
        visits["visit_date"] = (
            pd.to_datetime(visits[col], errors="coerce").dt.normalize()
        )
        return "visit_date"


def _normalize_interval(s: pd.Series) -> pd.Series:
    return s.astype("string").fillna("(missing)").str.strip()


def _phase_rank(name: str) -> int:
    if name in OPTIONAL_PHASES:
        return 99
    try:
        return PHASE_ORDER.index(name)
    except ValueError:
        return 98


def _build_visit_sequence(
    visits: pd.DataFrame,
    subject_col: str,
    visit_date_col: str,
    interval_col: str,
    include_optional: bool = INCLUDE_OPTIONAL,
) -> pd.DataFrame:
    seq = visits[[subject_col, visit_date_col, interval_col]].copy()
    seq[visit_date_col] = pd.to_datetime(seq[visit_date_col], errors="coerce")
    seq["interval_name"] = _normalize_interval(seq[interval_col])
    seq = seq.dropna(subset=[subject_col, visit_date_col]).sort_values(
        [subject_col, visit_date_col]
    )
    if not include_optional:
        seq = seq[~seq["interval_name"].isin(OPTIONAL_PHASES)].copy()

    seq["visit_order"] = seq.groupby(subject_col).cumcount() + 1
    seq["first_visit_date"] = seq.groupby(subject_col)[visit_date_col].transform("min")
    seq["day_from_first"] = (seq[visit_date_col] - seq["first_visit_date"]).dt.days
    seq["phase_rank"] = seq["interval_name"].apply(_phase_rank)
    seq["phase_label"] = seq["interval_name"].map(PHASE_LABELS).fillna(seq["interval_name"])
    return seq


def _build_transition_gaps(
    seq: pd.DataFrame,
    subject_col: str,
    visit_date_col: str,
) -> pd.DataFrame:
    gap = seq.copy()
    gap["prev_visit_date"]    = gap.groupby(subject_col)[visit_date_col].shift(1)
    gap["prev_visit_order"]   = gap.groupby(subject_col)["visit_order"].shift(1)
    gap["prev_interval_name"] = gap.groupby(subject_col)["interval_name"].shift(1)
    gap["gap_days"] = (gap[visit_date_col] - gap["prev_visit_date"]).dt.days
    gap = gap[gap["prev_visit_date"].notna()].copy()
    gap["transition_order"] = (
        "V" + gap["prev_visit_order"].astype(int).astype(str)
        + "→V" + gap["visit_order"].astype(int).astype(str)
    )
    gap["transition_interval"] = (
        gap["prev_interval_name"].astype(str)
        + " → " + gap["interval_name"].astype(str)
    )
    return gap


def _add_vref_lines(
    ax: plt.Axes,
    xlim: int | None = None,
    lw: float = 1.8,
    alpha: float = 0.9,
    label_ypos_frac: float = 0.97,
) -> list:
    """Vertical colored reference lines. Returns legend handles."""
    handles = []
    for (label, days), color in zip(TIME_REFS.items(), REF_COLORS):
        if xlim is not None and days > xlim:
            continue
        ax.axvline(days, color=color, lw=lw, ls="--", alpha=alpha, zorder=0)
        ymax = ax.get_ylim()[1]
        ax.text(
            days + 18, ymax * label_ypos_frac, label,
            fontsize=8, color=color, va="top", fontweight="bold",
        )
        handles.append(
            mlines.Line2D([], [], color=color, lw=lw, ls="--", alpha=alpha, label=label)
        )
    return handles


def _add_href_lines(ax: plt.Axes) -> list:
    """Thin, translucent horizontal reference lines. Returns legend handles."""
    handles = []
    for (label, days), color in zip(TIME_REFS.items(), REF_COLORS):
        ax.axhline(days, color=color, lw=0.9, ls="--", alpha=0.45, zorder=0)
        handles.append(
            mlines.Line2D([], [], color=color, lw=0.9, ls="--", alpha=0.7, label=label)
        )
    return handles


def _legend_below(
    ax: plt.Axes,
    handles: list,
    labels: list | None = None,
    ncol: int = 4,
    fontsize: int = 8,
    title: str | None = None,
    y_offset: float = -0.18,
) -> None:
    """Legend centered below the x-axis."""
    kw: dict = dict(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, y_offset),
        ncol=ncol,
        fontsize=fontsize,
        frameon=True,
        framealpha=0.9,
    )
    if labels is not None:
        kw["labels"] = labels
    if title:
        kw["title"] = title
        kw["title_fontsize"] = fontsize
    ax.legend(**kw)


# ─────────────────────────────────────────────────────────────────────
# Plot 1 — Swimmer (baseline = earliest record)
# ─────────────────────────────────────────────────────────────────────

def _plot_swimmer(
    seq: pd.DataFrame,
    subject_col: str,
    output_path: Path,
    xlim_days: int = 3650,
) -> None:
    """
    Each row = one patient, sorted by total number of visits (descending).
    Color = visit order (purple=early visits, yellow=late visits).
    X-axis = days from the patient's very first record.
    Patients beyond xlim_days are noted in text but not plotted beyond the limit.
    """
    patient_summary = (
        seq.groupby(subject_col, as_index=False)
        .agg(max_day=("day_from_first", "max"), n_visits=("visit_order", "max"))
        .sort_values(["n_visits", "max_day"], ascending=[False, False])
        .reset_index(drop=True)
    )
    patient_summary["patient_rank"] = np.arange(len(patient_summary))
    swim = seq.merge(
        patient_summary[[subject_col, "patient_rank"]], on=subject_col, how="left"
    )

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.subplots_adjust(bottom=0.18)

    ax.hlines(
        patient_summary["patient_rank"],
        xmin=0,
        xmax=patient_summary["max_day"].clip(upper=xlim_days),
        color="#aab4be", linewidth=0.6, zorder=1,
    )
    scatter = ax.scatter(
        swim["day_from_first"].clip(upper=xlim_days),
        swim["patient_rank"],
        c=swim["visit_order"], cmap="viridis",
        s=14, alpha=0.85, zorder=2,
    )
    cbar = fig.colorbar(scatter, ax=ax, pad=0.01)
    cbar.set_label(
        "Visit order\n(1 = first visit, higher = later visit per patient)",
        fontsize=8,
    )

    n_out = (patient_summary["max_day"] > xlim_days).sum()
    if n_out > 0:
        ax.text(
            xlim_days * 0.99, patient_summary["patient_rank"].max() * 0.97,
            f"{n_out} patients exceed {xlim_days} days →",
            ha="right", va="top", fontsize=8, color="#555",
        )

    ax.set_xlim(0, xlim_days)
    ref_handles = _add_vref_lines(ax, xlim=xlim_days)

    ax.set_title(
        "Swimmer plot — visit timeline per patient\n(baseline = first record)",
        fontsize=12,
    )
    ax.set_xlabel("Days from first record", labelpad=45)
    ax.set_ylabel(f"Patients (n={patient_summary.shape[0]}), sorted by N visits")
    ax.grid(axis="x", linestyle=":", alpha=0.3)
    _legend_below(ax, handles=ref_handles, ncol=8, title="Time references")

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Plot 2 — Swimmer (baseline = Phase 1: Initial Full Evaluation)
# ─────────────────────────────────────────────────────────────────────

def _plot_swimmer_phase1_baseline(
    seq: pd.DataFrame,
    subject_col: str,
    visit_date_col: str,
    output_path: Path,
    xlim_days: int = 3650,
) -> None:
    """
    Same as Swimmer 1 but day 0 = first occurrence of
    'Phase 1: Initial Full Evaluation' per patient.
    Patients without that phase are excluded.
    Pre-baseline records (V1: Natural History / missing) are excluded.

    Visit order color: 1 = first visit after Phase 1 baseline,
    higher numbers = later visits in that patient's follow-up.
    """
    phase1 = "Phase 1: Initial Full Evaluation"

    baseline = (
        seq[seq["interval_name"] == phase1]
        .groupby(subject_col)[visit_date_col].min()
        .rename("baseline_date").reset_index()
    )
    if baseline.empty:
        print("  [!] No Phase 1 Initial records found — skipping swimmer_phase1.")
        return

    swim = seq.merge(baseline, on=subject_col, how="inner")
    swim["day_from_phase1"] = (swim[visit_date_col] - swim["baseline_date"]).dt.days
    swim = swim[swim["day_from_phase1"] >= 0].copy()

    patient_summary = (
        swim.groupby(subject_col, as_index=False)
        .agg(max_day=("day_from_phase1", "max"), n_visits=("visit_order", "max"))
        .sort_values(["n_visits", "max_day"], ascending=[False, False])
        .reset_index(drop=True)
    )
    patient_summary["patient_rank"] = np.arange(len(patient_summary))
    swim = swim.merge(
        patient_summary[[subject_col, "patient_rank"]], on=subject_col, how="left"
    )

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.subplots_adjust(bottom=0.18)

    ax.hlines(
        patient_summary["patient_rank"],
        xmin=0,
        xmax=patient_summary["max_day"].clip(upper=xlim_days),
        color="#aab4be", linewidth=0.6, zorder=1,
    )
    scatter = ax.scatter(
        swim["day_from_phase1"].clip(upper=xlim_days),
        swim["patient_rank"],
        c=swim["visit_order"], cmap="viridis",
        s=14, alpha=0.85, zorder=2,
    )
    cbar = fig.colorbar(scatter, ax=ax, pad=0.01)
    cbar.set_label(
        "Visit order\n(1 = first visit, higher = later visit per patient)",
        fontsize=8,
    )

    n_out = (patient_summary["max_day"] > xlim_days).sum()
    if n_out > 0:
        ax.text(
            xlim_days * 0.99, patient_summary["patient_rank"].max() * 0.97,
            f"{n_out} patients exceed {xlim_days} days →",
            ha="right", va="top", fontsize=8, color="#555",
        )

    ax.set_xlim(0, xlim_days)
    ref_handles = _add_vref_lines(ax, xlim=xlim_days)

    ax.set_title(
        "Swimmer plot — visit timeline per patient\n"
        "(baseline = Phase 1: Initial Full Evaluation)",
        fontsize=12,
    )
    ax.set_xlabel("Days from Phase 1: Initial Full Evaluation", labelpad=45)
    ax.set_ylabel(f"Patients (n={patient_summary.shape[0]}), sorted by N visits")
    ax.grid(axis="x", linestyle=":", alpha=0.3)
    _legend_below(ax, handles=ref_handles, ncol=8, title="Time references")

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Plot 3 — Violin by transition
# ─────────────────────────────────────────────────────────────────────

def _plot_violin(gaps: pd.DataFrame, output_path: Path) -> None:
    """
    Violin per transition on log Y-axis.
    Shows the first 8 transitions in chronological order.
    Reference lines are thin and translucent so the violin shapes dominate.
    """
    violin_data = gaps[gaps["gap_days"].notna() & (gaps["gap_days"] > 0)].copy()

    all_trans = sorted(
        violin_data["transition_order"].unique(),
        key=lambda x: int(x.split("→")[1][1:]),
    )
    top_trans = all_trans[:8]
    violin_data = violin_data[violin_data["transition_order"].isin(top_trans)]

    fig, ax = plt.subplots(figsize=(13, 7))
    fig.subplots_adjust(bottom=0.25)

    sns.violinplot(
        data=violin_data,
        x="transition_order", y="gap_days",
        order=top_trans, inner="quartile", cut=0,
        ax=ax, color="#97d5c9",
    )
    ax.set_yscale("log")
    ax.set_ylim(bottom=1)

    # Thin, toned-down horizontal reference lines
    ref_handles = _add_href_lines(ax)

    for i, t in enumerate(top_trans):
        sub = violin_data[violin_data["transition_order"] == t]["gap_days"]
        ax.text(
            i, 1.3, f"n={len(sub)}\nmd={int(sub.median())}d",
            ha="center", va="bottom", fontsize=7, color="#333",
        )

    ax.set_title("Gap distribution between visits by transition (log scale)", fontsize=12)
    ax.set_xlabel("Transition", labelpad=55)
    ax.set_ylabel("Days between visits (log)")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", linestyle=":", alpha=0.3)

    _legend_below(ax, handles=ref_handles, ncol=8, title="Time references")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Plot 4 — KDE + histogram (log scale)
# ─────────────────────────────────────────────────────────────────────

def _plot_kde_hist(gaps: pd.DataFrame, output_path: Path) -> None:
    """
    Histogram with explicit log-spaced bins + KDE estimated in log10 space.

    KDE (Kernel Density Estimation): a smoothed continuous curve that
    estimates the probability distribution of the data. Each observation
    contributes a small bell-shaped Gaussian kernel; their sum forms
    the line. More robust than a histogram because it does not depend
    on bin choice.
    """
    dist = gaps[gaps["gap_days"].notna() & (gaps["gap_days"] > 0)].copy()
    vals = dist["gap_days"]
    log_bins = np.logspace(np.log10(vals.min()), np.log10(vals.max()), 55)

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.subplots_adjust(bottom=0.30)

    ax.hist(
        vals, bins=log_bins, color="#d17c58", alpha=0.55,
        edgecolor="white", linewidth=0.3,
    )
    ax.set_xscale("log")

    # KDE in log10 space → projected back to original axis
    log_vals = np.log10(vals)
    kde = gaussian_kde(log_vals, bw_method=0.25)
    xr = np.linspace(log_vals.min(), log_vals.max(), 500)
    ax2 = ax.twinx()
    ax2.plot(10**xr, kde(xr), color="#7c3d20", lw=2.5)
    ax2.set_yticks([])

    # Percentile lines
    pct_colors = {25: "#0F6E56", 50: "#D85A30", 75: "#993C1D"}
    pct_handles = []
    for p, c in pct_colors.items():
        v = np.percentile(vals, p)
        ax.axvline(v, color=c, lw=1.8, ls=":")
        ax.text(
            v * 1.06, ax.get_ylim()[1] * 0.65,
            f"P{p}\n{int(v)}d",
            fontsize=7.5, color=c, va="top", fontweight="bold",
        )
        pct_handles.append(
            mlines.Line2D([], [], color=c, lw=1.8, ls=":",
                          label=f"P{p} = {int(v)} days")
        )

    # Time reference lines
    ref_handles = []
    for (label, days), color in zip(TIME_REFS.items(), REF_COLORS):
        if vals.min() < days < vals.max():
            ax.axvline(days, color=color, lw=1.8, ls="--", alpha=0.9)
            ax.text(
                days * 1.05, ax.get_ylim()[1] * 0.93,
                label, fontsize=8, color=color, va="top", fontweight="bold",
            )
            ref_handles.append(
                mlines.Line2D([], [], color=color, lw=1.8, ls="--", label=label)
            )

    ax.set_xlabel("Days between visits (log scale)", labelpad=10)
    ax.set_ylabel("Number of intervals")
    ax.set_title("KDE + histogram of inter-visit gaps (log scale)", fontsize=12)
    ax.grid(axis="x", linestyle=":", alpha=0.3)

    h_hist = mpatches.Patch(color="#d17c58", alpha=0.6, label="Histogram")
    h_kde  = mlines.Line2D([], [], color="#7c3d20", lw=2, label="KDE (log-space)")
    _legend_below(
        ax,
        handles=[h_hist, h_kde] + pct_handles + ref_handles,
        ncol=4,
        title="Legend",
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Plot 5 — KDE by canonical inter-phase transition
# ─────────────────────────────────────────────────────────────────────

def _plot_kde_by_interval(gaps: pd.DataFrame, output_path: Path) -> None:
    """
    One KDE curve per canonical inter-phase transition:
        V1→V2, V2→V3, V3→V4, V4→V5, V5→V6.

    Same-phase transitions (e.g. V2→V2, two records of the same phase
    on consecutive days) are explicitly excluded by filtering against
    CANONICAL_TRANSITIONS.
    """
    dist = gaps[gaps["gap_days"].notna() & (gaps["gap_days"] > 0)].copy()

    # Keep ONLY defined canonical inter-phase pairs
    dist = dist[dist["transition_interval"].isin(CANONICAL_TRANSITIONS.keys())].copy()

    if dist.empty:
        print("  [!] No canonical inter-phase transitions found — skipping kde_by_interval.")
        return

    # Plot in canonical order
    present = [t for t in CANONICAL_TRANSITIONS.keys()
               if t in dist["transition_interval"].unique()]

    palette = sns.color_palette("tab10", len(present))

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.subplots_adjust(bottom=0.38)

    line_handles = []
    for trans, color in zip(present, palette):
        sub = dist[dist["transition_interval"] == trans]["gap_days"]
        if len(sub) < 5:
            continue
        log_sub = np.log10(sub.clip(lower=0.5))
        xr = np.linspace(log_sub.min(), log_sub.max(), 400)
        kde = gaussian_kde(log_sub, bw_method=0.3)
        label = CANONICAL_TRANSITIONS[trans]
        ax.plot(10**xr, kde(xr), lw=2.2, color=color)
        line_handles.append(
            mlines.Line2D([], [], color=color, lw=2.2,
                          label=f"{label}  (n={len(sub)})")
        )

    ax.set_xscale("log")

    # Thin, translucent vertical reference lines
    ref_handles = []
    for (label, days), color in zip(TIME_REFS.items(), REF_COLORS):
        ax.axvline(days, color=color, lw=1.0, ls="--", alpha=0.55, zorder=0)
        ref_handles.append(
            mlines.Line2D([], [], color=color, lw=1.0, ls="--",
                          alpha=0.7, label=label)
        )

    ax.set_xlabel("Days between visits (log scale)", labelpad=10)
    ax.set_ylabel("KDE density")
    ax.set_title(
        "Gap distribution by canonical inter-phase transition\n"
        "(V1→V2, V2→V3, V3→V4, V4→V5, V5→V6)",
        fontsize=12,
    )
    ax.grid(axis="x", linestyle=":", alpha=0.25)

    # Time refs: small legend inside upper-right corner
    ax.legend(
        handles=ref_handles,
        loc="upper right",
        fontsize=7,
        title="Time refs",
        title_fontsize=7.5,
        frameon=True,
        framealpha=0.85,
        ncol=2,
    )

    # Transition lines: below x-axis via figure legend
    fig.legend(
        handles=line_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=2,
        fontsize=8,
        title="Canonical transition",
        title_fontsize=8.5,
        frameon=True,
        framealpha=0.9,
    )

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Plot 6 — Heatmap patient × time
# ─────────────────────────────────────────────────────────────────────

def _plot_heatmap(
    seq: pd.DataFrame,
    subject_col: str,
    output_path: Path,
) -> pd.DataFrame:
    """
    Heatmap: patients (rows) × 30-day bins since baseline (columns).
    Columns beyond last recorded activity are trimmed.
    Reference lines at 6 mo, 2, 4, 6, 8, 10 years.
    """
    patient_rank = (
        seq.groupby(subject_col, as_index=False)["visit_order"]
        .max().rename(columns={"visit_order": "n_visits"})
        .sort_values("n_visits", ascending=False)
        .reset_index(drop=True)
    )
    patient_rank["patient_rank"] = np.arange(len(patient_rank))

    heat_data = seq.merge(
        patient_rank[[subject_col, "patient_rank"]], on=subject_col, how="left"
    )
    heat_data["time_bin_30d"] = (heat_data["day_from_first"] // 30).astype(int)

    matrix = (
        heat_data.groupby(["patient_rank", "time_bin_30d"], as_index=False)
        .size()
        .pivot(index="patient_rank", columns="time_bin_30d", values="size")
        .fillna(0)
    )
    active_cols = matrix.columns[matrix.sum(axis=0) > 0]
    matrix = matrix[active_cols]

    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(bottom=0.22)

    sns.heatmap(
        matrix, cmap="Blues", ax=ax,
        linewidths=0, linecolor="none",
        cbar_kws={"label": "N visits", "shrink": 0.5},
        yticklabels=False,
    )

    col_list = list(matrix.columns)
    ref_handles = []
    for (label, days), color in zip(TIME_REFS.items(), REF_COLORS):
        bin_num = days // 30
        if bin_num in col_list:
            col_pos = col_list.index(bin_num) + 0.5
            ax.axvline(col_pos, color=color, lw=2.0, ls="--", alpha=0.9)
            ax.text(
                col_pos + 0.3, matrix.shape[0] * 0.03,
                label, fontsize=8.5, color=color,
                va="bottom", fontweight="bold",
            )
            ref_handles.append(
                mlines.Line2D([], [], color=color, lw=2.0, ls="--", label=label)
            )

    ax.set_title(
        "Patient activity heatmap (30-day bins from baseline)", fontsize=12
    )
    ax.set_xlabel("Time bin (30 days from first record)", labelpad=45)
    ax.set_ylabel(f"Patients (n={matrix.shape[0]}), sorted by N visits")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=6)

    _legend_below(ax, handles=ref_handles, ncol=8, title="Time references")

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return matrix.reset_index()


# ─────────────────────────────────────────────────────────────────────
# Plot 7 — Optional evaluations: position in canonical sequence
# ─────────────────────────────────────────────────────────────────────

def _plot_optional_position(
    visits: pd.DataFrame,
    subject_col: str,
    visit_date_col: str,
    interval_col: str,
    output_path: Path,
) -> None:
    """
    For each optional visit, identify which canonical phase was the last
    one seen for that patient at or before that date. Shows a strip plot:
        x = last canonical phase context
        y = optional evaluation type (labeled on y-axis; no color legend needed)

    Always runs on the full dataset regardless of INCLUDE_OPTIONAL.
    """
    all_seq = visits[[subject_col, visit_date_col, interval_col]].copy()
    all_seq[visit_date_col] = pd.to_datetime(all_seq[visit_date_col], errors="coerce")
    all_seq["interval_name"] = _normalize_interval(all_seq[interval_col])
    all_seq = all_seq.dropna(subset=[subject_col, visit_date_col]).sort_values(
        [subject_col, visit_date_col]
    )

    canon = all_seq[all_seq["interval_name"].isin(PHASE_ORDER)].copy()
    opts  = all_seq[all_seq["interval_name"].isin(OPTIONAL_PHASES)].copy()

    if opts.empty:
        print("  [!] No optional evaluation visits found — skipping plot.")
        return

    results = []
    for _, row in opts.iterrows():
        pid      = row[subject_col]
        opt_date = row[visit_date_col]
        opt_name = row["interval_name"]

        before = canon[
            (canon[subject_col] == pid) & (canon[visit_date_col] <= opt_date)
        ]
        if before.empty:
            phase_ctx = "Before V2 (Initial)"
        else:
            last = before.sort_values(visit_date_col).iloc[-1]
            phase_ctx = PHASE_LABELS.get(last["interval_name"], last["interval_name"])

        results.append({"optional_type": opt_name, "phase_context": phase_ctx})

    result_df = pd.DataFrame(results)

    phase_order_labels = ["Before V2 (Initial)"] + [
        PHASE_LABELS[p] for p in PHASE_ORDER if p in PHASE_LABELS
    ]
    present_ctx = [p for p in phase_order_labels
                   if p in result_df["phase_context"].values]
    opt_order   = [o for o in OPTIONAL_PHASES
                   if o in result_df["optional_type"].values]

    palette = sns.color_palette("Set2", len(OPTIONAL_PHASES))
    opt_color_map = {opt: palette[i] for i, opt in enumerate(OPTIONAL_PHASES)}

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.subplots_adjust(bottom=0.28)

    sns.stripplot(
        data=result_df,
        x="phase_context",
        y="optional_type",
        order=present_ctx,
        hue="optional_type",
        hue_order=opt_order,
        palette=[opt_color_map[o] for o in opt_order],
        jitter=0.25,
        size=7,
        alpha=0.75,
        ax=ax,
        legend=False,   # y-axis labels each row — no color legend needed
    )

    ax.set_title(
        "Optional evaluations: position relative to canonical phase sequence",
        fontsize=12,
    )
    ax.set_xlabel("Last canonical phase before optional visit", labelpad=12)
    ax.set_ylabel("Optional evaluation type")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="x", linestyle=":", alpha=0.3)

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main() -> None:
    logger = setup_logger("08_visit_patterns")

    print_script_overview(
        "08_visit_patterns.py",
        "Analyzes patient visit dynamics and exports plot datasets + figures.\n"
        f"  INCLUDE_OPTIONAL = {INCLUDE_OPTIONAL}",
    )

    print_step(1, "Load visits_long and resolve canonical columns")
    visits = pd.read_parquet(ANALYTIC_DIR / "visits_long.parquet")
    subject_col    = resolve_canonical_column(visits, "subject_number")
    visit_date_col = _resolve_visit_date(visits)
    interval_col   = resolve_canonical_column(visits, "interval_name")

    print_step(2, "Build visit sequence and interval map")
    seq = _build_visit_sequence(
        visits, subject_col, visit_date_col, interval_col,
        include_optional=INCLUDE_OPTIONAL,
    )
    interval_map = (
        seq.groupby("interval_name", dropna=False, as_index=False)
        .agg(
            n_visits=("interval_name", "size"),
            n_patients=(subject_col, "nunique"),
        )
        .sort_values("n_visits", ascending=False)
    )

    print_step(3, "Build transition gaps")
    gaps = _build_transition_gaps(seq, subject_col, visit_date_col)

    print_step(4, "Save plot-ready datasets")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    interval_map.to_csv(PLOTS_DIR / "interval_name_map.csv", index=False)
    seq[
        [subject_col, "visit_order", "interval_name", visit_date_col, "day_from_first"]
    ].to_csv(PLOTS_DIR / "plot_data_swimmer.csv", index=False)
    gaps[
        [subject_col, "transition_order", "transition_interval", "gap_days"]
    ].to_csv(PLOTS_DIR / "plot_data_violin.csv", index=False)
    gaps[
        [subject_col, "gap_days", "transition_order"]
    ].to_csv(PLOTS_DIR / "plot_data_kde_hist.csv", index=False)

    print_step(5, "Render plots")

    _plot_swimmer(
        seq, subject_col,
        PLOTS_DIR / "swimmer_all_records.png",
        xlim_days=3650,
    )

    _plot_swimmer_phase1_baseline(
        seq, subject_col, visit_date_col,
        PLOTS_DIR / "swimmer_phase1_baseline.png",
        xlim_days=3650,
    )

    _plot_violin(gaps, PLOTS_DIR / "violin_transition_plot.png")

    _plot_kde_hist(gaps, PLOTS_DIR / "kde_hist_gapdays_plot.png")

    _plot_kde_by_interval(gaps, PLOTS_DIR / "kde_by_interval_plot.png")

    hm = _plot_heatmap(seq, subject_col, PLOTS_DIR / "heatmap_patient_time.png")
    hm.to_csv(PLOTS_DIR / "plot_data_heatmap_matrix.csv", index=False)

    # Always uses full data regardless of INCLUDE_OPTIONAL
    _plot_optional_position(
        visits, subject_col, visit_date_col, interval_col,
        PLOTS_DIR / "optional_eval_position.png",
    )

    print_kv(
        "Visit patterns summary",
        {
            "n_patients":       int(seq[subject_col].nunique(dropna=True)),
            "n_visits":         int(len(seq)),
            "n_transitions":    int(len(gaps)),
            "n_interval_names": int(interval_map["interval_name"].nunique(dropna=False)),
            "gap_days_median":  int(gaps["gap_days"].median()),
            "gap_days_p25":     int(gaps["gap_days"].quantile(0.25)),
            "gap_days_p75":     int(gaps["gap_days"].quantile(0.75)),
            "include_optional": INCLUDE_OPTIONAL,
            "output_dir":       str(PLOTS_DIR),
        },
    )
    print_step(6, "Append targeted EDA for transformed visits dataset to unified workbook")
    sheets = build_targeted_eda_sheets(
        seq,
        "08_visits_transformed_patterns_output",
        "08_visits_transformed_patterns_output",
    )
    workbook = upsert_eda_sheets_xlsx(EDA_UNIFIED_REPORT_PATH, sheets)
    logger.info("Updated unified EDA workbook: %s", workbook)
    logger.info("Saved visit pattern outputs in %s", PLOTS_DIR)


if __name__ == "__main__":
    main()
