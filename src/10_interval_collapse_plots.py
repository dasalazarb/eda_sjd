from __future__ import annotations

from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde

from common import REPORTS_DIR, print_kv, print_script_overview, print_step, setup_logger

PLOTS_DIR = REPORTS_DIR / "interval_collapse_plots"

# ─────────────────────────────────────────────────────────────────────
# Input paths (outputs from 09_interval_collapse_audit.py)
# ─────────────────────────────────────────────────────────────────────

PATH_VARIABLE_AUDIT    = REPORTS_DIR / "interval_collapse_variable_audit.csv"
PATH_WINDOW_STATS      = REPORTS_DIR / "interval_collapse_window_stats.csv"
PATH_WINDOW_SUMMARY    = REPORTS_DIR / "interval_collapse_window_summary.csv"
PATH_REPEATED_GROUPS   = REPORTS_DIR / "interval_collapse_repeated_groups.csv"
PATH_CONFLICT_EXAMPLES = REPORTS_DIR / "interval_collapse_conflict_examples.csv"

# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────

# Max variables to show in bar chart (sorted by total affected groups)
TOP_N_VARIABLES: int = 30

# Window reference lines: label → days
WINDOW_REFS: dict[str, int] = {
    "1 day":  1,
    "1 wk":   7,
    "1 mo":   30,
    "3 mo":   90,
    "6 mo":   182,
    "1 yr":   365,
}
WINDOW_REF_COLORS: list[str] = [
    "#e74c3c",  # 1 day  — red
    "#e67e22",  # 1 wk   — orange
    "#f1c40f",  # 1 mo   — yellow
    "#27ae60",  # 3 mo   — green
    "#2980b9",  # 6 mo   — blue
    "#8e44ad",  # 1 yr   — purple
]

# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _legend_below(
    ax: plt.Axes,
    handles: list,
    ncol: int = 4,
    fontsize: int = 8,
    title: str | None = None,
    y_offset: float = -0.18,
) -> None:
    kw: dict = dict(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, y_offset),
        ncol=ncol,
        fontsize=fontsize,
        frameon=True,
        framealpha=0.9,
    )
    if title:
        kw["title"] = title
        kw["title_fontsize"] = fontsize
    ax.legend(**kw)


def _load(path: Path, label: str) -> pd.DataFrame | None:
    if not path.exists():
        print(f"  [!] {label} not found at {path} — skipping related plots.")
        return None
    df = pd.read_csv(path)
    print(f"  Loaded {label}: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


# ─────────────────────────────────────────────────────────────────────
# Plot 1 — Stacked bar: complementary vs conflict groups per variable
# ─────────────────────────────────────────────────────────────────────

def _plot_variable_audit_bar(audit: pd.DataFrame, output_path: Path) -> None:
    """
    Horizontal stacked bar chart: one bar per variable (top N by affected_groups).
    Green segment = complementary_groups (safe to collapse).
    Red segment   = conflict_groups (values disagree across repeated rows).

    Answers: which variables can be collapsed safely vs which need review?
    """
    df = audit.copy()
    df = df[df["affected_groups"] > 0].copy()
    df = df.nlargest(TOP_N_VARIABLES, "affected_groups").sort_values(
        "affected_groups", ascending=True
    )

    if df.empty:
        print("  [!] No affected groups in variable_audit — skipping bar chart.")
        return

    y = np.arange(len(df))
    h = max(6, len(df) * 0.32)

    fig, ax = plt.subplots(figsize=(12, h))
    fig.subplots_adjust(left=0.38, right=0.92, bottom=0.14)

    ax.barh(y, df["complementary_groups"], color="#27ae60", alpha=0.85,
            label="Complementary groups\n(values agree — safe to collapse)")
    ax.barh(y, df["conflict_groups"], left=df["complementary_groups"],
            color="#e74c3c", alpha=0.85,
            label="Conflict groups\n(values disagree — needs review)")

    # Annotate conflict % on the right
    for i, row in enumerate(df.itertuples()):
        pct = row.pct_conflict_among_affected
        if pct > 0:
            ax.text(
                row.affected_groups + 0.3, i,
                f"{pct:.0f}% conflict",
                va="center", fontsize=7, color="#c0392b",
            )

    ax.set_yticks(y)
    ax.set_yticklabels(df["variable"], fontsize=8)
    ax.set_xlabel("Number of patient-interval groups", labelpad=10)
    ax.set_title(
        f"Variable audit: complementary vs conflict groups\n"
        f"(top {len(df)} variables by affected groups)",
        fontsize=12,
    )
    ax.grid(axis="x", linestyle=":", alpha=0.4)

    handles = [
        mpatches.Patch(color="#27ae60", alpha=0.85,
                       label="Complementary (safe to collapse)"),
        mpatches.Patch(color="#e74c3c", alpha=0.85,
                       label="Conflict (values disagree)"),
    ]
    _legend_below(ax, handles=handles, ncol=2, y_offset=-0.12)

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


# ─────────────────────────────────────────────────────────────────────
# Plot 2 — Scatter: % conflict vs % complementary per variable
# ─────────────────────────────────────────────────────────────────────

def _plot_variable_audit_scatter(audit: pd.DataFrame, output_path: Path) -> None:
    """
    Each point = one variable.
    X = % conflict among affected groups.
    Y = % complementary among affected groups.
    Size = affected_groups (absolute count).

    Quadrant interpretation:
        Bottom-left  → mostly empty / not informative
        Top-left     → ideal: records complement each other, no disagreement
        Bottom-right → problematic: records disagree consistently
        Top-right    → mixed: some complement, some conflict

    Answers: are conflicts systematic (same variables always conflict)
             or random noise?
    """
    df = audit[audit["affected_groups"] > 0].copy()
    if df.empty:
        print("  [!] No affected groups — skipping scatter.")
        return

    size_scale = 300 * df["affected_groups"] / df["affected_groups"].max()

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.subplots_adjust(bottom=0.14)

    sc = ax.scatter(
        df["pct_conflict_among_affected"],
        df["pct_complementary_among_affected"],
        s=size_scale.clip(lower=20),
        c=df["conflict_groups"],
        cmap="RdYlGn_r",
        alpha=0.75,
        edgecolors="white",
        linewidths=0.4,
    )

    cbar = fig.colorbar(sc, ax=ax, pad=0.01)
    cbar.set_label("N conflict groups (absolute)", fontsize=9)

    # Quadrant lines
    ax.axvline(50, color="#aaa", lw=0.8, ls="--", alpha=0.6)
    ax.axhline(50, color="#aaa", lw=0.8, ls="--", alpha=0.6)

    # Quadrant labels
    for (tx, ty, label) in [
        (2,  97, "Ideal\n(safe to collapse)"),
        (52, 97, "Mixed"),
        (2,  3,  "Low info"),
        (52, 3,  "Problematic\n(needs review)"),
    ]:
        ax.text(tx, ty, label, fontsize=7.5, color="#555",
                va="top" if ty > 50 else "bottom",
                ha="left" if tx < 50 else "left",
                style="italic")

    # Label top conflict variables
    top_conflict = df.nlargest(8, "conflict_groups")
    for _, row in top_conflict.iterrows():
        ax.annotate(
            row["variable"],
            xy=(row["pct_conflict_among_affected"],
                row["pct_complementary_among_affected"]),
            xytext=(6, 2), textcoords="offset points",
            fontsize=6.5, color="#333",
        )

    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)
    ax.set_xlabel("% conflict among affected groups", labelpad=10)
    ax.set_ylabel("% complementary among affected groups", labelpad=10)
    ax.set_title(
        "Variable conflict vs complementarity\n"
        "(point size ∝ N affected groups)",
        fontsize=12,
    )
    ax.grid(linestyle=":", alpha=0.3)

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


# ─────────────────────────────────────────────────────────────────────
# Plot 3 — KDE + histogram: window_days distribution (log scale)
# ─────────────────────────────────────────────────────────────────────

def _plot_window_days_kde(window_stats: pd.DataFrame, output_path: Path) -> None:
    """
    Distribution of the temporal window (max_date - min_date) inside each
    repeated patient-interval group.

    window_days = 0 → all repeated records share the same date (trivial duplicates)
    window_days > 30 → records span months (different clinical encounters)

    Answers: are duplicates same-day noise or genuine multi-visit data?
    """
    repeated = window_stats[window_stats["n_rows_group"] > 1].copy()
    vals = repeated["window_days"].dropna()

    if vals.empty:
        print("  [!] No repeated groups in window_stats — skipping KDE.")
        return

    # Separate same-day (0) from multi-day (>0)
    n_same_day = (vals == 0).sum()
    vals_pos   = vals[vals > 0]

    fig, axes = plt.subplots(
        1, 2,
        figsize=(14, 5),
        gridspec_kw={"width_ratios": [1, 2.5]},
    )
    fig.subplots_adjust(bottom=0.22, wspace=0.35)

    # ── Left panel: same-day vs multi-day pie ───────────────────────
    ax0 = axes[0]
    n_multi = len(vals_pos)
    wedges, texts, autotexts = ax0.pie(
        [n_same_day, n_multi],
        labels=["Same day\n(window = 0)", "Multi-day\n(window > 0)"],
        colors=["#2980b9", "#e67e22"],
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 9},
    )
    ax0.set_title("Repeated groups:\nsame-day vs multi-day", fontsize=10)

    # ── Right panel: KDE + hist for window > 0 ─────────────────────
    ax1 = axes[1]

    if len(vals_pos) >= 2:
        log_bins = np.logspace(
            np.log10(max(vals_pos.min(), 0.5)),
            np.log10(vals_pos.max()),
            50,
        )
        ax1.hist(
            vals_pos, bins=log_bins,
            color="#e67e22", alpha=0.55,
            edgecolor="white", linewidth=0.3,
        )
        ax1.set_xscale("log")

        log_vals = np.log10(vals_pos.clip(lower=0.5))
        kde = gaussian_kde(log_vals, bw_method=0.3)
        xr  = np.linspace(log_vals.min(), log_vals.max(), 400)
        ax1b = ax1.twinx()
        ax1b.plot(10**xr, kde(xr), color="#7c3d20", lw=2.5)
        ax1b.set_yticks([])

        # Reference lines
        ref_handles = []
        for (label, days), color in zip(WINDOW_REFS.items(), WINDOW_REF_COLORS):
            if days == 1:
                continue   # 1-day line only relevant in pie
            if vals_pos.min() <= days <= vals_pos.max():
                ax1.axvline(days, color=color, lw=1.6, ls="--", alpha=0.85)
                ax1.text(
                    days * 1.06, ax1.get_ylim()[1] * 0.93,
                    label, fontsize=7.5, color=color,
                    va="top", fontweight="bold",
                )
                ref_handles.append(
                    mlines.Line2D([], [], color=color, lw=1.6, ls="--", label=label)
                )

        # Percentiles
        for p, c in [(50, "#c0392b"), (90, "#8e44ad")]:
            v = np.percentile(vals_pos, p)
            ax1.axvline(v, color=c, lw=1.4, ls=":")
            ax1.text(
                v * 1.06, ax1.get_ylim()[1] * 0.60,
                f"P{p}={int(v)}d",
                fontsize=7.5, color=c, va="top", fontweight="bold",
            )
            ref_handles.append(
                mlines.Line2D([], [], color=c, lw=1.4, ls=":",
                              label=f"P{p} = {int(v)} days")
            )

        ax1.set_xlabel("Window days (log scale)", labelpad=10)
        ax1.set_ylabel("N groups")
        ax1.set_title(
            f"Distribution of temporal window\ninside repeated groups (n={len(vals_pos):,})",
            fontsize=10,
        )
        ax1.grid(axis="x", linestyle=":", alpha=0.3)

        h_hist = mpatches.Patch(color="#e67e22", alpha=0.6, label="Histogram")
        h_kde  = mlines.Line2D([], [], color="#7c3d20", lw=2, label="KDE (log-space)")
        _legend_below(ax1, handles=[h_hist, h_kde] + ref_handles,
                      ncol=4, y_offset=-0.20)

    else:
        ax1.text(0.5, 0.5, "Insufficient multi-day data",
                 ha="center", va="center", transform=ax1.transAxes, fontsize=10)

    fig.suptitle(
        "Temporal window inside repeated patient-interval groups",
        fontsize=12, y=1.01,
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


# ─────────────────────────────────────────────────────────────────────
# Plot 4 — Heatmap: interval_name × window_days_bin
# ─────────────────────────────────────────────────────────────────────

def _plot_window_heatmap(window_stats: pd.DataFrame, output_path: Path) -> None:
    """
    Heatmap: rows = interval_name, columns = 7-day window bins.
    Color = number of groups in that cell.

    Answers: do certain evaluation phases systematically produce
             same-day duplicates vs multi-day repeated records?
    """
    # Detect interval column (second column after subject)
    non_numeric = [c for c in window_stats.columns
                   if window_stats[c].dtype == object and c not in ("min_date", "max_date")]
    if len(non_numeric) < 2:
        print("  [!] Cannot identify interval_name column — skipping window heatmap.")
        return

    # Heuristic: interval column is the one with phase-like values
    interval_col_candidates = [c for c in non_numeric if "interval" in c.lower()]
    interval_col = interval_col_candidates[0] if interval_col_candidates else non_numeric[-1]

    repeated = window_stats[window_stats["n_rows_group"] > 1].copy()
    if repeated.empty:
        print("  [!] No repeated groups — skipping window heatmap.")
        return

    repeated["window_bin"] = (repeated["window_days"] // 7).astype(int)

    matrix = (
        repeated.groupby([interval_col, "window_bin"], as_index=False)
        .size()
        .pivot(index=interval_col, columns="window_bin", values="size")
        .fillna(0)
    )

    # Trim trailing empty columns
    active_cols = matrix.columns[matrix.sum(axis=0) > 0]
    matrix = matrix[active_cols]

    # Short labels for interval names
    matrix.index = [
        s[:40] + "…" if len(str(s)) > 40 else str(s)
        for s in matrix.index
    ]

    h = max(4, len(matrix) * 0.55)
    w = max(10, len(matrix.columns) * 0.45)

    fig, ax = plt.subplots(figsize=(w, h))
    fig.subplots_adjust(bottom=0.20)

    sns.heatmap(
        matrix, cmap="YlOrRd", ax=ax,
        linewidths=0.4, linecolor="#f0f0f0",
        cbar_kws={"label": "N repeated groups", "shrink": 0.6},
        annot=len(matrix.columns) <= 20,
        fmt=".0f",
        annot_kws={"fontsize": 7},
    )

    # Mark bin corresponding to 30 days (1 month)
    bin_1mo = 30 // 7  # bin 4
    if bin_1mo in list(matrix.columns):
        col_pos = list(matrix.columns).index(bin_1mo) + 0.5
        ax.axvline(col_pos, color="#2980b9", lw=2.0, ls="--", alpha=0.9)
        ax.text(col_pos + 0.2, -0.3, "1 mo",
                fontsize=8, color="#2980b9", fontweight="bold")

    ax.set_title(
        "Repeated groups by interval phase and window size\n"
        "(columns = 7-day bins of temporal window)",
        fontsize=12,
    )
    ax.set_xlabel("Window bin (7-day intervals since first record in group)", labelpad=10)
    ax.set_ylabel("Interval / evaluation phase")
    ax.set_xticklabels(
        [f"{int(c)*7}–{int(c)*7+6}d" for c in matrix.columns],
        rotation=45, ha="right", fontsize=7,
    )
    ax.tick_params(axis="y", labelsize=8)

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


# ─────────────────────────────────────────────────────────────────────
# Plot 5 — Histogram: N rows per repeated group
# ─────────────────────────────────────────────────────────────────────

def _plot_rows_per_group(repeated_groups: pd.DataFrame, output_path: Path) -> None:
    """
    Distribution of how many rows exist per repeated patient-interval group.

    n_rows = 2 → most common case (one extra record)
    n_rows = 10+ → systematic multi-record issue

    Answers: is duplication a minor issue (mostly 2 rows) or
             are there groups with many redundant records?
    """
    if "n_rows" not in repeated_groups.columns:
        print("  [!] n_rows column not found — skipping rows-per-group histogram.")
        return

    vals = repeated_groups["n_rows"]
    counts = vals.value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.subplots_adjust(bottom=0.16)

    bars = ax.bar(
        counts.index.astype(str),
        counts.values,
        color="#2980b9", alpha=0.75,
        edgecolor="white", linewidth=0.4,
    )

    # Annotate bars
    for bar, val in zip(bars, counts.values):
        pct = 100 * val / len(vals)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{val:,}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=7.5, color="#333",
        )

    ax.set_xlabel("Number of rows in group", labelpad=10)
    ax.set_ylabel("Number of patient-interval groups")
    ax.set_title(
        "Distribution of row count per repeated patient-interval group\n"
        f"(total repeated groups: {len(vals):,})",
        fontsize=12,
    )
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    # Summary stats annotation
    summary = (
        f"Median: {vals.median():.0f}   "
        f"P90: {vals.quantile(0.9):.0f}   "
        f"Max: {vals.max()}"
    )
    ax.text(
        0.99, 0.97, summary,
        transform=ax.transAxes,
        ha="right", va="top", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#ccc", alpha=0.9),
    )

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


# ─────────────────────────────────────────────────────────────────────
# Plot 6 — Heatmap: variable × interval_name (conflict count)
# ─────────────────────────────────────────────────────────────────────

def _plot_conflict_heatmap(
    conflict_examples: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Heatmap: rows = variable, columns = interval_name.
    Color = number of conflict examples in that cell.

    Answers: do conflicts cluster in specific phases or specific variables?
             (e.g., 'race' conflicts only in Phase 1 Initial, or everywhere?)
    """
    if conflict_examples.empty:
        print("  [!] No conflict examples — skipping conflict heatmap.")
        return

    # Detect interval column
    non_meta = [c for c in conflict_examples.columns
                if c not in ("variable", "observed_values", "collapsed_value")]
    interval_candidates = [c for c in non_meta if "interval" in c.lower()]
    interval_col = interval_candidates[0] if interval_candidates else non_meta[-1]

    if "variable" not in conflict_examples.columns or interval_col not in conflict_examples.columns:
        print("  [!] Required columns missing — skipping conflict heatmap.")
        return

    matrix = (
        conflict_examples.groupby(["variable", interval_col])
        .size()
        .unstack(fill_value=0)
    )

    # Keep only variables with at least one conflict
    matrix = matrix[matrix.sum(axis=1) > 0]

    if matrix.empty:
        print("  [!] Empty conflict matrix — skipping conflict heatmap.")
        return

    # Sort by total conflicts descending, cap at top 30
    matrix = matrix.loc[matrix.sum(axis=1).nlargest(30).index]

    # Short labels
    matrix.index = [
        s[:35] + "…" if len(str(s)) > 35 else str(s)
        for s in matrix.index
    ]
    matrix.columns = [
        s[:25] + "…" if len(str(s)) > 25 else str(s)
        for s in matrix.columns
    ]

    h = max(5, len(matrix) * 0.38)
    w = max(8, len(matrix.columns) * 1.4)

    fig, ax = plt.subplots(figsize=(w, h))
    fig.subplots_adjust(bottom=0.30)

    sns.heatmap(
        matrix, cmap="Reds", ax=ax,
        linewidths=0.4, linecolor="#f0f0f0",
        cbar_kws={"label": "N conflict examples", "shrink": 0.6},
        annot=True, fmt=".0f",
        annot_kws={"fontsize": 7},
    )

    ax.set_title(
        "Conflict examples: variable × evaluation phase\n"
        f"(top {len(matrix)} variables by total conflicts)",
        fontsize=12,
    )
    ax.set_xlabel("Evaluation phase (interval_name)", labelpad=10)
    ax.set_ylabel("Variable")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=7.5)
    ax.tick_params(axis="y", labelsize=7.5)

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main() -> None:
    logger = setup_logger("09_interval_collapse_plots")

    print_script_overview(
        "09_interval_collapse_plots.py",
        "Generates diagnostic plots for interval_collapse_audit outputs.\n"
        "Inputs: CSVs produced by 09_interval_collapse_audit.py\n"
        "Outputs: PNG plots saved to reports/interval_collapse_plots/",
    )

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print_step(1, "Load audit outputs")
    variable_audit    = _load(PATH_VARIABLE_AUDIT,    "variable_audit")
    window_stats      = _load(PATH_WINDOW_STATS,      "window_stats")
    repeated_groups   = _load(PATH_REPEATED_GROUPS,   "repeated_groups")
    conflict_examples = _load(PATH_CONFLICT_EXAMPLES, "conflict_examples")

    print_step(2, "Plot 1 — Variable audit: stacked bar (complementary vs conflict)")
    if variable_audit is not None:
        _plot_variable_audit_bar(
            variable_audit,
            PLOTS_DIR / "01_variable_audit_bar.png",
        )

    print_step(3, "Plot 2 — Variable audit: scatter (% conflict vs % complementary)")
    if variable_audit is not None:
        _plot_variable_audit_scatter(
            variable_audit,
            PLOTS_DIR / "02_variable_audit_scatter.png",
        )

    print_step(4, "Plot 3 — Window days: KDE + histogram (log scale)")
    if window_stats is not None:
        _plot_window_days_kde(
            window_stats,
            PLOTS_DIR / "03_window_days_kde.png",
        )

    print_step(5, "Plot 4 — Window days: heatmap by interval phase")
    if window_stats is not None:
        _plot_window_heatmap(
            window_stats,
            PLOTS_DIR / "04_window_heatmap_by_interval.png",
        )

    print_step(6, "Plot 5 — Rows per group: histogram")
    if repeated_groups is not None:
        _plot_rows_per_group(
            repeated_groups,
            PLOTS_DIR / "05_rows_per_group_histogram.png",
        )

    print_step(7, "Plot 6 — Conflict examples: heatmap variable × interval")
    if conflict_examples is not None:
        _plot_conflict_heatmap(
            conflict_examples,
            PLOTS_DIR / "06_conflict_heatmap_variable_x_interval.png",
        )

    saved = list(PLOTS_DIR.glob("*.png"))
    print_kv(
        "Interval collapse plots summary",
        {
            "plots_saved":  len(saved),
            "output_dir":   str(PLOTS_DIR),
            "plot_names":   [p.name for p in sorted(saved)],
        },
    )
    logger.info("Saved %d plots to %s", len(saved), PLOTS_DIR)


if __name__ == "__main__":
    main()