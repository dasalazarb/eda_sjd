from __future__ import annotations

from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text          # pip install adjustText
from scipy.stats import gaussian_kde

from common import REPORTS_DIR, print_kv, print_script_overview, print_step, setup_logger

PLOTS_DIR = REPORTS_DIR / "interval_collapse_plots"

# ─────────────────────────────────────────────────────────────────────
# Input paths  (outputs from 09_interval_collapse_audit.py)
# ─────────────────────────────────────────────────────────────────────

PATH_VARIABLE_AUDIT    = REPORTS_DIR / "interval_collapse_variable_audit.csv"
PATH_WINDOW_STATS      = REPORTS_DIR / "interval_collapse_window_stats.csv"
PATH_REPEATED_GROUPS   = REPORTS_DIR / "interval_collapse_repeated_groups.csv"
PATH_CONFLICT_EXAMPLES = REPORTS_DIR / "interval_collapse_conflict_examples.csv"

# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────

# Non-clinical metadata columns — excluded from all plots
NON_CLINICAL: set[str] = {
    "row_id_raw",
    "visit_datetime",
    "duplicate_group_id",
    "row_id",
    "unnamed: 0",
    "index",
    "record_id",
}

# Top-N variables shown in bar chart
TOP_N_VARIABLES: int = 30

# Conflict threshold above which a variable is shown in the focused scatter
CONFLICT_PCT_THRESHOLD: float = 40.0

# Window reference lines
WINDOW_REFS: dict[str, int] = {
    "1 day": 1, "1 wk": 7, "1 mo": 30,
    "3 mo": 90, "6 mo": 182, "1 yr": 365,
}
WINDOW_REF_COLORS: list[str] = [
    "#e74c3c", "#e67e22", "#f1c40f",
    "#27ae60", "#2980b9", "#8e44ad",
]

# Interval → marker shape mapping (add/edit as needed)
INTERVAL_MARKERS: dict[str, str] = {
    "Natural History Protocol 478 Interval": "^",   # triangle up
    "Phase 1: Initial Full Evaluation":       "o",   # circle
    "Phase 1: Second Full Evaluation":        "s",   # square
    "Phase 1: Final Full (Third Full) Evaluation": "D",  # diamond
    "Phase 2: 4th Full Evaluation":           "P",   # plus-filled
    "Phase 2: 5th Full Evaluation":           "X",   # x-filled
    "(missing)":                              "v",   # triangle down
    "__other__":                              "h",   # hexagon fallback
}

# Interval → color for scatter (distinct palette)
INTERVAL_COLORS: dict[str, str] = {
    "Natural History Protocol 478 Interval": "#e74c3c",
    "Phase 1: Initial Full Evaluation":       "#2980b9",
    "Phase 1: Second Full Evaluation":        "#27ae60",
    "Phase 1: Final Full (Third Full) Evaluation": "#8e44ad",
    "Phase 2: 4th Full Evaluation":           "#e67e22",
    "Phase 2: 5th Full Evaluation":           "#16a085",
    "(missing)":                              "#7f8c8d",
    "__other__":                              "#bdc3c7",
}

# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _is_non_clinical(name: str) -> bool:
    return name.strip().lower() in NON_CLINICAL


def _filter_clinical(df: pd.DataFrame, var_col: str = "variable") -> pd.DataFrame:
    if var_col in df.columns:
        return df[~df[var_col].apply(_is_non_clinical)].copy()
    return df.copy()


def _load(path: Path, label: str) -> pd.DataFrame | None:
    if not path.exists():
        print(f"  [!] {label} not found at {path} — skipping.")
        return None
    df = pd.read_csv(path)
    print(f"  Loaded {label}: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


def _legend_below(
    ax: plt.Axes,
    handles: list,
    ncol: int = 4,
    fontsize: int = 8,
    title: str | None = None,
    y_offset: float = -0.18,
) -> None:
    kw: dict = dict(
        handles=handles, loc="upper center",
        bbox_to_anchor=(0.5, y_offset),
        ncol=ncol, fontsize=fontsize,
        frameon=True, framealpha=0.92,
    )
    if title:
        kw["title"] = title
        kw["title_fontsize"] = fontsize + 1
    ax.legend(**kw)


def _short(s: str, n: int = 38) -> str:
    return s if len(s) <= n else s[:n] + "…"


def _interval_marker(name: str) -> str:
    return INTERVAL_MARKERS.get(name, INTERVAL_MARKERS["__other__"])


def _interval_color(name: str) -> str:
    return INTERVAL_COLORS.get(name, INTERVAL_COLORS["__other__"])


# ─────────────────────────────────────────────────────────────────────
# Plot 1 — Stacked bar: complementary vs conflict per variable
# ─────────────────────────────────────────────────────────────────────

def _plot_variable_audit_bar(audit: pd.DataFrame, output_path: Path) -> None:
    """
    Horizontal stacked bar — top N clinical variables by affected_groups.
    Green = complementary (safe to collapse).
    Red   = conflict (values disagree — needs review).
    Variables are sorted so the most conflicted appear at the top.
    """
    df = _filter_clinical(audit)
    df = df[df["affected_groups"] > 0].copy()
    df = df.nlargest(TOP_N_VARIABLES, "affected_groups")
    # Sort: most conflict % at top
    df = df.sort_values("pct_conflict_among_affected", ascending=True)

    if df.empty:
        print("  [!] No clinical affected groups — skipping bar.")
        return

    y  = np.arange(len(df))
    h  = max(7, len(df) * 0.36)

    fig, ax = plt.subplots(figsize=(13, h))
    fig.subplots_adjust(left=0.40, right=0.90, bottom=0.12)

    bars_c = ax.barh(y, df["complementary_groups"],
                     color="#27ae60", alpha=0.85, label="Complementary (safe)")
    bars_x = ax.barh(y, df["conflict_groups"],
                     left=df["complementary_groups"],
                     color="#e74c3c", alpha=0.85, label="Conflict (needs review)")

    # Conflict % annotation
    for i, row in enumerate(df.itertuples()):
        pct = row.pct_conflict_among_affected
        color = "#7f0000" if pct >= 90 else "#c0392b"
        if pct > 0:
            ax.text(
                row.affected_groups + 2, i,
                f"{pct:.0f}%",
                va="center", fontsize=7.5, color=color, fontweight="bold",
            )

    # Vertical threshold line at 50%
    mid = df["affected_groups"].max() / 2
    ax.axvline(mid, color="#aaa", lw=0.8, ls=":", alpha=0.6)

    ax.set_yticks(y)
    ax.set_yticklabels(df["variable"], fontsize=8)
    ax.set_xlabel("Number of patient-interval groups", labelpad=10)
    ax.set_title(
        f"Variable audit: complementary vs conflict groups\n"
        f"(top {len(df)} clinical variables by affected groups — "
        f"non-clinical metadata excluded)",
        fontsize=12,
    )
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)

    handles = [
        mpatches.Patch(color="#27ae60", alpha=0.85, label="Complementary (safe to collapse)"),
        mpatches.Patch(color="#e74c3c", alpha=0.85, label="Conflict (values disagree)"),
    ]
    _legend_below(ax, handles=handles, ncol=2, y_offset=-0.10)

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


# ─────────────────────────────────────────────────────────────────────
# Plot 2 — Focused scatter: conflict region only, shaped by interval
# ─────────────────────────────────────────────────────────────────────

def _plot_variable_audit_scatter(
    audit: pd.DataFrame,
    conflict_examples: pd.DataFrame | None,
    output_path: Path,
) -> None:
    """
    Two-panel figure:

    Left  — Overview scatter (all clinical variables).
            Highlights the problematic quadrant with a shaded box.
            Points in the safe zone are small and grey.

    Right — Zoom into the conflict region (pct_conflict >= threshold).
            Each point is shaped and colored by the interval_name that
            contributes most conflicts for that variable.
            Labels are spread with adjustText so they do not overlap.
            X-axis in log scale to separate clustered high-conflict points.
    """
    df = _filter_clinical(audit)
    df = df[df["affected_groups"] > 0].copy()

    if df.empty:
        print("  [!] No data for scatter — skipping.")
        return

    # Attach dominant interval per variable from conflict_examples
    dominant_interval: dict[str, str] = {}
    if conflict_examples is not None and not conflict_examples.empty:
        ce = _filter_clinical(conflict_examples)
        non_meta = [c for c in ce.columns
                    if c not in ("variable", "observed_values", "collapsed_value")]
        interval_candidates = [c for c in non_meta if "interval" in c.lower()]
        if interval_candidates:
            icol = interval_candidates[0]
            dom = (
                ce.groupby(["variable", icol])
                .size()
                .reset_index(name="n")
                .sort_values("n", ascending=False)
                .drop_duplicates("variable")
                .set_index("variable")[icol]
            )
            dominant_interval = dom.to_dict()

    df["interval"] = df["variable"].map(dominant_interval).fillna("__other__")
    df["marker"]   = df["interval"].apply(_interval_marker)
    df["color"]    = df["interval"].apply(_interval_color)

    # ── Figure: two panels ──────────────────────────────────────────
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(18, 8),
        gridspec_kw={"width_ratios": [1, 1.6]},
    )
    fig.subplots_adjust(bottom=0.24, wspace=0.35)

    # ── Left: overview ───────────────────────────────────────────────
    safe_mask = df["pct_conflict_among_affected"] < CONFLICT_PCT_THRESHOLD

    # Safe points — small, grey, no border
    ax_left.scatter(
        df.loc[safe_mask, "pct_conflict_among_affected"],
        df.loc[safe_mask, "pct_complementary_among_affected"],
        s=25, color="#d5d8dc", alpha=0.55, linewidths=0, zorder=2,
    )

    # Problematic points — colored, larger
    prob = df[~safe_mask]
    size_scale = 60 + 300 * prob["affected_groups"] / df["affected_groups"].max()
    for _, row in prob.iterrows():
        ax_left.scatter(
            row["pct_conflict_among_affected"],
            row["pct_complementary_among_affected"],
            s=size_scale.loc[row.name],
            c=row["color"], marker=row["marker"],
            alpha=0.88, edgecolors="white", linewidths=0.5, zorder=3,
        )

    # Shaded problematic quadrant
    ax_left.axvspan(CONFLICT_PCT_THRESHOLD, 102, alpha=0.07, color="#e74c3c", zorder=0)
    ax_left.axvline(CONFLICT_PCT_THRESHOLD, color="#e74c3c", lw=1.0, ls="--", alpha=0.5)
    ax_left.axvline(50, color="#aaa", lw=0.7, ls=":", alpha=0.5)
    ax_left.axhline(50, color="#aaa", lw=0.7, ls=":", alpha=0.5)

    ax_left.text(CONFLICT_PCT_THRESHOLD + 1, 98, "← zoomed in right panel →",
                 fontsize=7.5, color="#e74c3c", va="top", style="italic")

    for tx, ty, label in [
        (2, 97, "Ideal\n(safe to collapse)"),
        (52, 3, "Problematic\n(needs review)"),
    ]:
        ax_left.text(tx, ty, label, fontsize=7.5, color="#555", style="italic",
                     va="top" if ty > 50 else "bottom")

    ax_left.set_xlim(-2, 102)
    ax_left.set_ylim(-2, 102)
    ax_left.set_xlabel("% conflict among affected groups")
    ax_left.set_ylabel("% complementary among affected groups")
    ax_left.set_title("Overview — all clinical variables\n"
                      f"(grey = safe, colored = conflict ≥ {CONFLICT_PCT_THRESHOLD:.0f}%)",
                      fontsize=10)
    ax_left.grid(linestyle=":", alpha=0.3)
    ax_left.spines[["top", "right"]].set_visible(False)

    # ── Right: zoomed conflict region ────────────────────────────────
    prob_r = df[~safe_mask].copy()

    if prob_r.empty:
        ax_right.text(0.5, 0.5, "No variables exceed conflict threshold",
                      ha="center", va="center", transform=ax_right.transAxes)
    else:
        # Jitter on Y so points at same pct_complementary don't stack
        rng = np.random.default_rng(42)
        prob_r["y_jitter"] = (
            prob_r["pct_complementary_among_affected"]
            + rng.uniform(-1.5, 1.5, len(prob_r))
        )

        size_r = 60 + 400 * prob_r["affected_groups"] / df["affected_groups"].max()

        texts = []
        for _, row in prob_r.iterrows():
            x_val = max(row["pct_conflict_among_affected"], 0.1)
            ax_right.scatter(
                x_val, row["y_jitter"],
                s=size_r.loc[row.name],
                c=row["color"], marker=row["marker"],
                alpha=0.88, edgecolors="white", linewidths=0.6, zorder=3,
            )
            texts.append(
                ax_right.text(
                    x_val, row["y_jitter"],
                    _short(row["variable"], 28),
                    fontsize=6.5, color="#1a1a1a",
                )
            )

        # Auto-spread labels to avoid overlap
        try:
            adjust_text(
                texts, ax=ax_right,
                arrowprops=dict(arrowstyle="-", color="#aaa", lw=0.5),
                expand_text=(1.15, 1.4),
                force_text=(0.4, 0.6),
            )
        except Exception:
            pass   # adjustText optional — graceful fallback

        ax_right.set_xscale("log")
        ax_right.xaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"{x:.0f}%"
        ))

        # Reference lines
        for val, lbl in [(50, "50%"), (90, "90%"), (100, "100%")]:
            ax_right.axvline(val, color="#e74c3c", lw=0.9, ls="--", alpha=0.45)
            ax_right.text(val * 1.02, ax_right.get_ylim()[1] * 0.97,
                          lbl, fontsize=7, color="#e74c3c", va="top")

        ax_right.set_xlabel("% conflict (log scale)", labelpad=10)
        ax_right.set_ylabel("% complementary (with jitter)")
        ax_right.set_title(
            f"Conflict zone — variables with ≥{CONFLICT_PCT_THRESHOLD:.0f}% conflict\n"
            "Shape = dominant interval · Size = N affected groups",
            fontsize=10,
        )
        ax_right.grid(linestyle=":", alpha=0.3)
        ax_right.spines[["top", "right"]].set_visible(False)

    # ── Shared legend: interval shapes ───────────────────────────────
    present_intervals = df.loc[~safe_mask, "interval"].unique() if not prob_r.empty else []
    shape_handles = [
        mlines.Line2D(
            [], [],
            marker=_interval_marker(iv),
            color=_interval_color(iv),
            linewidth=0, markersize=8,
            label=_short(iv, 45),
        )
        for iv in INTERVAL_MARKERS
        if iv != "__other__" and iv in present_intervals
    ]
    shape_handles.append(
        mlines.Line2D([], [], marker="h", color="#bdc3c7",
                      linewidth=0, markersize=7, label="Other / unknown interval")
    )
    size_handles = [
        mlines.Line2D([], [], marker="o", color="#888", linewidth=0,
                      markersize=s, label=lbl, alpha=0.7)
        for s, lbl in [(5, "Few groups"), (10, "Many groups")]
    ]

    fig.legend(
        handles=shape_handles + size_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=4, fontsize=7.5,
        title="Interval (shape) · Point size ∝ N affected groups",
        title_fontsize=8,
        frameon=True, framealpha=0.92,
    )

    fig.suptitle(
        "Variable conflict vs complementarity — clinical variables only\n"
        "(non-clinical metadata excluded: row_id_raw, visit_datetime, duplicate_group_id)",
        fontsize=12, y=1.01,
    )

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


# ─────────────────────────────────────────────────────────────────────
# Plot 3 — KDE + histogram: window_days (log scale)
# ─────────────────────────────────────────────────────────────────────

def _plot_window_days_kde(window_stats: pd.DataFrame, output_path: Path) -> None:
    """
    Left  — Pie: same-day vs multi-day groups.
    Right — KDE + histogram of temporal window for multi-day groups (log scale).

    window_days = 0 → trivial same-day duplicates.
    window_days > 30 → different clinical encounters sharing an interval_name.
    """
    repeated = window_stats[window_stats["n_rows_group"] > 1].copy()
    vals     = repeated["window_days"].dropna()
    if vals.empty:
        print("  [!] No repeated groups — skipping window KDE.")
        return

    n_same  = (vals == 0).sum()
    vals_pos = vals[vals > 0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                             gridspec_kw={"width_ratios": [1, 2.5]})
    fig.subplots_adjust(bottom=0.24, wspace=0.35)

    # ── Pie ──────────────────────────────────────────────────────────
    ax0 = axes[0]
    wedges, _, autotexts = ax0.pie(
        [n_same, len(vals_pos)],
        labels=[f"Same day\n(window = 0)\nn={n_same:,}",
                f"Multi-day\n(window > 0)\nn={len(vals_pos):,}"],
        colors=["#2980b9", "#e67e22"],
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 9},
        wedgeprops={"linewidth": 0.8, "edgecolor": "white"},
    )
    for at in autotexts:
        at.set_fontweight("bold")
    ax0.set_title("Repeated groups:\nsame-day vs multi-day", fontsize=10)

    # ── KDE + hist ───────────────────────────────────────────────────
    ax1 = axes[1]
    if len(vals_pos) >= 2:
        log_bins = np.logspace(
            np.log10(max(vals_pos.min(), 0.5)),
            np.log10(vals_pos.max()),
            55,
        )
        ax1.hist(vals_pos, bins=log_bins, color="#e67e22", alpha=0.50,
                 edgecolor="white", linewidth=0.3)
        ax1.set_xscale("log")

        log_v = np.log10(vals_pos.clip(lower=0.5))
        kde   = gaussian_kde(log_v, bw_method=0.28)
        xr    = np.linspace(log_v.min(), log_v.max(), 500)
        ax1b  = ax1.twinx()
        ax1b.plot(10**xr, kde(xr), color="#7c3d20", lw=2.5)
        ax1b.set_yticks([])

        ref_handles = []
        for (lbl, days), color in zip(WINDOW_REFS.items(), WINDOW_REF_COLORS):
            if days == 1:
                continue
            if vals_pos.min() <= days <= vals_pos.max():
                ax1.axvline(days, color=color, lw=1.6, ls="--", alpha=0.85)
                ax1.text(days * 1.06, ax1.get_ylim()[1] * 0.93,
                         lbl, fontsize=7.5, color=color, va="top", fontweight="bold")
                ref_handles.append(
                    mlines.Line2D([], [], color=color, lw=1.6, ls="--", label=lbl)
                )

        pct_handles = []
        for p, c in [(50, "#c0392b"), (90, "#6c3483")]:
            v = int(np.percentile(vals_pos, p))
            ax1.axvline(v, color=c, lw=1.4, ls=":")
            ax1.text(v * 1.06, ax1.get_ylim()[1] * 0.60,
                     f"P{p}={v}d", fontsize=7.5, color=c, va="top", fontweight="bold")
            pct_handles.append(
                mlines.Line2D([], [], color=c, lw=1.4, ls=":",
                              label=f"P{p} = {v} days")
            )

        ax1.set_xlabel("Window days (log scale)", labelpad=10)
        ax1.set_ylabel("N groups")
        ax1.set_title(
            f"Temporal window — multi-day repeated groups (n={len(vals_pos):,})",
            fontsize=10,
        )
        ax1.grid(axis="x", linestyle=":", alpha=0.3)
        ax1.spines[["top", "right"]].set_visible(False)

        h_hist = mpatches.Patch(color="#e67e22", alpha=0.55, label="Histogram")
        h_kde  = mlines.Line2D([], [], color="#7c3d20", lw=2.5, label="KDE (log-space)")
        _legend_below(ax1, handles=[h_hist, h_kde] + pct_handles + ref_handles,
                      ncol=4, y_offset=-0.22)

    fig.suptitle("Temporal window inside repeated patient-interval groups",
                 fontsize=12, y=1.01)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


# ─────────────────────────────────────────────────────────────────────
# Plot 4 — Heatmap: interval_name × window_days_bin
# ─────────────────────────────────────────────────────────────────────

def _plot_window_heatmap(window_stats: pd.DataFrame, output_path: Path) -> None:
    """
    Rows = interval_name · Columns = 7-day window bins.
    Trimmed to active columns only.
    Annotated with cell counts where matrix is small enough.
    """
    non_num = [c for c in window_stats.columns
               if window_stats[c].dtype == object
               and c not in ("min_date", "max_date")]
    interval_cands = [c for c in non_num if "interval" in c.lower()]
    interval_col   = interval_cands[0] if interval_cands else (non_num[-1] if non_num else None)

    if interval_col is None:
        print("  [!] Interval column not found — skipping window heatmap.")
        return

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
    active_cols = matrix.columns[matrix.sum(axis=0) > 0]
    matrix      = matrix[active_cols]

    matrix.index = [_short(str(s), 42) for s in matrix.index]

    do_annot = len(matrix.columns) <= 25
    h = max(4, len(matrix) * 0.60)
    w = max(10, len(matrix.columns) * 0.48)

    fig, ax = plt.subplots(figsize=(w, h))
    fig.subplots_adjust(bottom=0.22)

    sns.heatmap(
        matrix, cmap="YlOrRd", ax=ax,
        linewidths=0.4, linecolor="#f0f0f0",
        cbar_kws={"label": "N repeated groups", "shrink": 0.55},
        annot=do_annot, fmt=".0f",
        annot_kws={"fontsize": 7},
    )

    # 1-month reference
    bin_1mo = 30 // 7
    if bin_1mo in list(matrix.columns):
        cp = list(matrix.columns).index(bin_1mo) + 0.5
        ax.axvline(cp, color="#2980b9", lw=2.0, ls="--", alpha=0.9)
        ax.text(cp + 0.2, -0.25, "1 mo", fontsize=8, color="#2980b9", fontweight="bold")

    ax.set_title(
        "Repeated groups by interval phase and window size\n"
        "(columns = 7-day bins of temporal window inside group)",
        fontsize=12,
    )
    ax.set_xlabel("Window bin (7-day intervals)", labelpad=10)
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
# Plot 5 — Histogram + CDF: rows per repeated group
# ─────────────────────────────────────────────────────────────────────

def _plot_rows_per_group(repeated_groups: pd.DataFrame, output_path: Path) -> None:
    """
    Left  — Bar chart of N groups by row count (with % annotations).
    Right — Empirical CDF so you can read "X% of groups have ≤ N rows".

    Answers: is duplication a minor 2-row issue or are groups systematically
             large (8–10 rows = all sub-measurements of a form)?
    """
    if "n_rows" not in repeated_groups.columns:
        print("  [!] n_rows missing — skipping.")
        return

    vals   = repeated_groups["n_rows"]
    counts = vals.value_counts().sort_index()

    fig, (ax_bar, ax_cdf) = plt.subplots(1, 2, figsize=(14, 5))
    fig.subplots_adjust(bottom=0.14, wspace=0.35)

    # ── Bar ──────────────────────────────────────────────────────────
    bars = ax_bar.bar(
        counts.index.astype(str), counts.values,
        color="#2980b9", alpha=0.75,
        edgecolor="white", linewidth=0.4,
    )
    for bar, v in zip(bars, counts.values):
        pct = 100 * v / len(vals)
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{v:,}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=7, color="#222",
        )

    ax_bar.set_xlabel("Number of rows in group", labelpad=10)
    ax_bar.set_ylabel("Number of patient-interval groups")
    ax_bar.set_title(
        f"Row count per repeated group\n(total: {len(vals):,} groups)", fontsize=11
    )
    ax_bar.grid(axis="y", linestyle=":", alpha=0.35)
    ax_bar.spines[["top", "right"]].set_visible(False)
    ax_bar.text(
        0.99, 0.97,
        f"Median: {vals.median():.0f}   P90: {vals.quantile(0.9):.0f}   Max: {vals.max()}",
        transform=ax_bar.transAxes, ha="right", va="top", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#ccc", alpha=0.92),
    )

    # ── CDF ──────────────────────────────────────────────────────────
    sorted_v = np.sort(vals)
    cdf      = np.arange(1, len(sorted_v) + 1) / len(sorted_v) * 100

    ax_cdf.plot(sorted_v, cdf, color="#2980b9", lw=2.2)
    ax_cdf.fill_between(sorted_v, cdf, alpha=0.12, color="#2980b9")

    for pct_mark, color in [(50, "#e67e22"), (80, "#e74c3c"), (95, "#8e44ad")]:
        x_val = np.percentile(vals, pct_mark)
        ax_cdf.axhline(pct_mark, color=color, lw=0.9, ls="--", alpha=0.7)
        ax_cdf.axvline(x_val,   color=color, lw=0.9, ls="--", alpha=0.7)
        ax_cdf.text(x_val + 0.1, pct_mark + 1, f"{pct_mark}% ≤ {x_val:.0f} rows",
                    fontsize=7.5, color=color, va="bottom", fontweight="bold")

    ax_cdf.set_xlabel("Number of rows in group", labelpad=10)
    ax_cdf.set_ylabel("Cumulative % of groups")
    ax_cdf.set_title("Empirical CDF — row count per group", fontsize=11)
    ax_cdf.set_ylim(0, 102)
    ax_cdf.grid(linestyle=":", alpha=0.35)
    ax_cdf.spines[["top", "right"]].set_visible(False)

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
    Rows = variable (top 30 by total conflicts, clinical only).
    Columns = interval_name.
    Color = number of conflict examples in that cell.

    Answers: do conflicts cluster in specific phases or variables?
    """
    if conflict_examples.empty:
        print("  [!] No conflict examples — skipping heatmap.")
        return

    ce = _filter_clinical(conflict_examples)

    non_meta   = [c for c in ce.columns
                  if c not in ("variable", "observed_values", "collapsed_value")]
    iv_cands   = [c for c in non_meta if "interval" in c.lower()]
    interval_col = iv_cands[0] if iv_cands else (non_meta[-1] if non_meta else None)

    if interval_col is None or "variable" not in ce.columns:
        print("  [!] Required columns missing — skipping conflict heatmap.")
        return

    matrix = (
        ce.groupby(["variable", interval_col])
        .size()
        .unstack(fill_value=0)
    )
    matrix = matrix[matrix.sum(axis=1) > 0]
    matrix = matrix.loc[matrix.sum(axis=1).nlargest(30).index]

    matrix.index   = [_short(str(s), 38) for s in matrix.index]
    matrix.columns = [_short(str(s), 28) for s in matrix.columns]

    h = max(5, len(matrix) * 0.40)
    w = max(8, len(matrix.columns) * 1.6)

    fig, ax = plt.subplots(figsize=(w, h))
    fig.subplots_adjust(bottom=0.30)

    sns.heatmap(
        matrix, cmap="Reds", ax=ax,
        linewidths=0.4, linecolor="#f5f5f5",
        cbar_kws={"label": "N conflict examples", "shrink": 0.55},
        annot=True, fmt=".0f",
        annot_kws={"fontsize": 7},
    )

    ax.set_title(
        "Conflict examples: variable × evaluation phase\n"
        f"(top {len(matrix)} clinical variables — non-clinical metadata excluded)",
        fontsize=12,
    )
    ax.set_xlabel("Evaluation phase (interval_name)", labelpad=10)
    ax.set_ylabel("Variable")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=8)
    ax.tick_params(axis="y", labelsize=8)

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


# ─────────────────────────────────────────────────────────────────────
# Plot 7 — Variable family summary (new)
# ─────────────────────────────────────────────────────────────────────

def _plot_family_summary(audit: pd.DataFrame, output_path: Path) -> None:
    """
    Groups variables by prefix family (e.g. salivary_flow_form, vital_signs)
    and shows a dot plot: median conflict % per family, sized by N variables.

    Answers: which instrument / form drives the most conflict?
    """
    df = _filter_clinical(audit)
    df = df[df["affected_groups"] > 0].copy()

    # Extract family = first two underscore-separated tokens
    df["family"] = df["variable"].str.split("__").str[0].str.split("_").str[:3].str.join("_")

    family_agg = (
        df.groupby("family")
        .agg(
            median_conflict=("pct_conflict_among_affected", "median"),
            n_vars=("variable", "count"),
            total_affected=("affected_groups", "sum"),
        )
        .reset_index()
        .sort_values("median_conflict", ascending=True)
    )

    if family_agg.empty:
        print("  [!] No family data — skipping family summary.")
        return

    fig, ax = plt.subplots(figsize=(10, max(5, len(family_agg) * 0.45)))
    fig.subplots_adjust(left=0.32, right=0.92, bottom=0.14)

    y       = np.arange(len(family_agg))
    sizes   = 40 + 200 * family_agg["n_vars"] / family_agg["n_vars"].max()
    colors  = [
        "#e74c3c" if v >= 80 else "#e67e22" if v >= 50 else "#27ae60"
        for v in family_agg["median_conflict"]
    ]

    ax.scatter(family_agg["median_conflict"], y,
               s=sizes, c=colors, alpha=0.82,
               edgecolors="white", linewidths=0.5, zorder=3)

    # Horizontal reference lines
    for v in [50, 80, 100]:
        ax.axvline(v, color="#ddd", lw=0.8, ls="--", zorder=0)
        ax.text(v + 0.5, len(family_agg) - 0.5, f"{v}%",
                fontsize=7, color="#aaa", va="top")

    ax.set_yticks(y)
    ax.set_yticklabels(family_agg["family"], fontsize=8)
    ax.set_xlabel("Median % conflict among affected groups", labelpad=10)
    ax.set_xlim(-2, 105)
    ax.set_title(
        "Variable family conflict summary\n"
        "(dot size ∝ N variables in family · color = severity)",
        fontsize=12,
    )
    ax.grid(axis="x", linestyle=":", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    # Annotate N vars
    for i, row in enumerate(family_agg.itertuples()):
        ax.text(row.median_conflict + 1.5, i,
                f"n={row.n_vars}", va="center", fontsize=7, color="#555")

    handles = [
        mpatches.Patch(color="#27ae60", label="< 50% conflict — safe"),
        mpatches.Patch(color="#e67e22", label="50–80% — review"),
        mpatches.Patch(color="#e74c3c", label="≥ 80% — problematic"),
    ]
    _legend_below(ax, handles=handles, ncol=3, y_offset=-0.12)

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
        "Diagnostic plots for 09_interval_collapse_audit.py outputs.\n"
        f"  Non-clinical vars excluded: {sorted(NON_CLINICAL)}\n"
        f"  Conflict threshold for scatter zoom: {CONFLICT_PCT_THRESHOLD}%",
    )

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print_step(1, "Load audit outputs")
    variable_audit    = _load(PATH_VARIABLE_AUDIT,    "variable_audit")
    window_stats      = _load(PATH_WINDOW_STATS,      "window_stats")
    repeated_groups   = _load(PATH_REPEATED_GROUPS,   "repeated_groups")
    conflict_examples = _load(PATH_CONFLICT_EXAMPLES, "conflict_examples")

    print_step(2, "Plot 1 — Variable audit stacked bar")
    if variable_audit is not None:
        _plot_variable_audit_bar(
            variable_audit,
            PLOTS_DIR / "01_variable_audit_bar.png",
        )

    print_step(3, "Plot 2 — Conflict scatter (overview + zoom, shaped by interval)")
    if variable_audit is not None:
        _plot_variable_audit_scatter(
            variable_audit, conflict_examples,
            PLOTS_DIR / "02_variable_audit_scatter.png",
        )

    print_step(4, "Plot 3 — Window days KDE + pie")
    if window_stats is not None:
        _plot_window_days_kde(
            window_stats,
            PLOTS_DIR / "03_window_days_kde.png",
        )

    print_step(5, "Plot 4 — Window heatmap by interval phase")
    if window_stats is not None:
        _plot_window_heatmap(
            window_stats,
            PLOTS_DIR / "04_window_heatmap_by_interval.png",
        )

    print_step(6, "Plot 5 — Rows per group: bar + CDF")
    if repeated_groups is not None:
        _plot_rows_per_group(
            repeated_groups,
            PLOTS_DIR / "05_rows_per_group_histogram.png",
        )

    print_step(7, "Plot 6 — Conflict heatmap variable × interval")
    if conflict_examples is not None:
        _plot_conflict_heatmap(
            conflict_examples,
            PLOTS_DIR / "06_conflict_heatmap_variable_x_interval.png",
        )

    print_step(8, "Plot 7 — Variable family conflict summary (new)")
    if variable_audit is not None:
        _plot_family_summary(
            variable_audit,
            PLOTS_DIR / "07_family_conflict_summary.png",
        )

    saved = sorted(PLOTS_DIR.glob("*.png"))
    print_kv(
        "Interval collapse plots summary",
        {
            "plots_saved": len(saved),
            "output_dir":  str(PLOTS_DIR),
            "plot_names":  [p.name for p in saved],
        },
    )
    logger.info("Saved %d plots to %s", len(saved), PLOTS_DIR)


if __name__ == "__main__":
    main()