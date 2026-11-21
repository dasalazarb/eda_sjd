from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from common import REPORTS_DIR, print_kv, print_script_overview, print_step, setup_logger

sns.set_theme(style="whitegrid", context="talk")

METRIC_EXPLANATIONS: dict[str, str] = {
    "pct_patients_ge1": "share of patients with ≥1 measurement (higher is better coverage)",
    "pct_patients_ge2": "share of patients with ≥2 measurements for longitudinal analysis (higher is better)",
    "pct_patients_ge3": "share of patients with ≥3 measurements (higher supports trend modeling)",
    "pct_visits_covered": "share of visits with non-missing values (higher is better)",
    "consecutive_pair_coverage": "share of adjacent visit pairs observed (higher is better temporal continuity)",
    "consistency_score": "temporal smoothness/plausibility score (higher indicates more stable trajectories)",
    "change_rate": "frequency of value changes between visits (context dependent; extreme values may be unstable)",
    "flip_rate": "for categorical/boolean variables, back-and-forth flips across visits (lower is better)",
    "reversion_rate": "rate of returning to a previous state after a change (lower is usually better)",
    "contradiction_rate": "rate of clinically implausible transitions (lower is better)",
    "delta_outlier_rate": "rate of implausibly large visit-to-visit jumps (lower is better)",
    "missingness_bias_by_interval_pp": "absolute missingness variation across time intervals in percentage points (lower is better)",
}


def _humanize_metric(metric: str) -> str:
    return metric.replace("_", " ")


def _add_metric_footnote(fig: plt.Figure, metrics: list[str]) -> None:
    notes = []
    for metric in metrics:
        if metric in METRIC_EXPLANATIONS:
            notes.append(f"{metric}: {METRIC_EXPLANATIONS[metric]}")
    if not notes:
        return
    text = "Metric note — " + " | ".join(notes)
    fig.text(0.01, 0.01, text, ha="left", va="bottom", fontsize=9, wrap=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Builds exploratory visualizations from longitudinal_variable_summary.csv "
            "to support variable selection for longitudinal modeling."
        )
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=REPORTS_DIR / "longitudinal_plausibility" / "longitudinal_variable_summary.csv",
        help="Path to longitudinal_variable_summary.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPORTS_DIR / "longitudinal_plausibility" / "plots",
        help="Directory to store plots.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Top-N variables to display for ranking/heatmap plots.",
    )
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=20.0,
        help="Threshold for pct_patients_ge2 in frontier plot.",
    )
    parser.add_argument(
        "--consistency-threshold",
        type=float,
        default=70.0,
        help="Threshold for consistency_score in frontier plot.",
    )
    return parser.parse_args()


def _load(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return pd.read_csv(path)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _present_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [c for c in columns if c in df.columns]


def plot_coverage_histograms(df: pd.DataFrame, out_dir: Path) -> int:
    metrics = _present_columns(
        df,
        [
            "pct_patients_ge1",
            "pct_patients_ge2",
            "pct_patients_ge3",
            "pct_visits_covered",
            "consecutive_pair_coverage",
        ],
    )
    if not metrics:
        return 0

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
    axes = axes.flatten()
    for i, metric in enumerate(metrics):
        ax = axes[i]
        sns.histplot(df[metric].dropna(), bins=20, kde=True, color="#2c7fb8", ax=ax)
        ax.set_title(f"Distribution of {_humanize_metric(metric)} across variables")
        ax.set_xlabel("Percentage value")

    for j in range(len(metrics), len(axes)):
        axes[j].axis("off")

    fig.suptitle("Coverage metric distributions across longitudinal variables", y=1.03, fontsize=16)
    _add_metric_footnote(fig, metrics)
    fig.tight_layout(rect=[0, 0.08, 1, 0.97])
    _save(fig, out_dir / "01_hist_cobertura_variables.png")
    return 1


def plot_variable_type_bars(df: pd.DataFrame, out_dir: Path) -> int:
    if "variable_type" not in df.columns:
        return 0

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    vc = df["variable_type"].value_counts(dropna=False)
    sns.barplot(x=vc.index.astype(str), y=vc.values, ax=axes[0, 0], color="#5e3c99")
    axes[0, 0].set_title("Number of variables by variable type")
    axes[0, 0].set_xlabel("Variable type")
    axes[0, 0].set_ylabel("Number of variables")
    axes[0, 0].tick_params(axis="x", rotation=25)

    if "ml_longitudinal_label" in df.columns:
        tmp = pd.crosstab(df["variable_type"], df["ml_longitudinal_label"], normalize="index")
        tmp.plot(kind="bar", stacked=True, ax=axes[0, 1], colormap="viridis")
        axes[0, 1].set_title("Longitudinal ML label composition by variable type")
        axes[0, 1].set_ylabel("Proportion")
        axes[0, 1].tick_params(axis="x", rotation=25)
        axes[0, 1].legend(title="ml_longitudinal_label", bbox_to_anchor=(1.02, 1), loc="upper left")
    else:
        axes[0, 1].axis("off")

    if "pct_visits_covered" in df.columns:
        tmp_cov = (
            df.groupby("variable_type", dropna=False)["pct_visits_covered"].mean().sort_values(ascending=False)
        )
        sns.barplot(x=tmp_cov.index.astype(str), y=tmp_cov.values, ax=axes[1, 0], color="#1b9e77")
        axes[1, 0].set_title("Average visit coverage by variable type")
        axes[1, 0].set_ylabel("Mean pct_visits_covered")
        axes[1, 0].tick_params(axis="x", rotation=25)
    else:
        axes[1, 0].axis("off")

    if "consistency_score" in df.columns:
        tmp_cons = (
            df.groupby("variable_type", dropna=False)["consistency_score"].mean().sort_values(ascending=False)
        )
        sns.barplot(x=tmp_cons.index.astype(str), y=tmp_cons.values, ax=axes[1, 1], color="#d95f02")
        axes[1, 1].set_title("Average temporal consistency by variable type")
        axes[1, 1].set_ylabel("Mean consistency_score")
        axes[1, 1].tick_params(axis="x", rotation=25)
    else:
        axes[1, 1].axis("off")

    fig.suptitle("Variable-type summary for longitudinal modeling readiness", y=1.03, fontsize=16)
    _add_metric_footnote(fig, ["pct_visits_covered", "consistency_score"])
    fig.tight_layout(rect=[0, 0.08, 1, 0.97])
    _save(fig, out_dir / "02_barras_variable_type.png")
    return 1


def plot_label_bars(df: pd.DataFrame, out_dir: Path) -> int:
    if "ml_longitudinal_label" not in df.columns:
        return 0

    counts = df["ml_longitudinal_label"].value_counts(dropna=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax, palette="Set2")
    ax.set_title("Final ML longitudinal utility class distribution")
    ax.set_xlabel("ML longitudinal label")
    ax.set_ylabel("Number of variables")
    ax.tick_params(axis="x", rotation=20)

    for i, v in enumerate(counts.values):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=10)

    _save(fig, out_dir / "03_barras_ml_longitudinal_label.png")
    return 1


def plot_coverage_vs_consistency(df: pd.DataFrame, out_dir: Path) -> int:
    y = "consistency_score"
    if y not in df.columns:
        return 0

    x_cols = _present_columns(df, ["pct_patients_ge2", "consecutive_pair_coverage", "pct_visits_covered"])
    if not x_cols:
        return 0

    fig, axes = plt.subplots(1, len(x_cols), figsize=(7 * len(x_cols), 6), sharey=True)
    if len(x_cols) == 1:
        axes = [axes]

    hue = "variable_type" if "variable_type" in df.columns else None
    for ax, x in zip(axes, x_cols):
        sns.scatterplot(data=df, x=x, y=y, hue=hue, alpha=0.75, ax=ax)
        ax.set_title(f"{_humanize_metric(y)} versus {_humanize_metric(x)}")
        if hue:
            ax.legend(loc="best", fontsize=8)

    fig.suptitle("Relationship between coverage metrics and temporal consistency", y=1.03, fontsize=16)
    _add_metric_footnote(fig, x_cols + [y])
    fig.tight_layout(rect=[0, 0.08, 1, 0.97])
    _save(fig, out_dir / "04_scatter_cobertura_vs_consistencia.png")
    return 1


def plot_coverage_vs_change(df: pd.DataFrame, out_dir: Path) -> int:
    pairs = [
        ("pct_patients_ge2", "change_rate"),
        ("consecutive_pair_coverage", "flip_rate"),
        ("pct_visits_covered", "reversion_rate"),
        ("pct_visits_covered", "contradiction_rate"),
    ]
    valid_pairs = [(x, y) for x, y in pairs if x in df.columns and y in df.columns]
    if not valid_pairs:
        return 0

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes_flat = axes.flatten()
    hue = "ml_longitudinal_label" if "ml_longitudinal_label" in df.columns else None

    for i, (x, y) in enumerate(valid_pairs):
        ax = axes_flat[i]
        plot_df = df.copy()
        if y in {"flip_rate", "reversion_rate", "contradiction_rate"} and "variable_type" in plot_df.columns:
            plot_df = plot_df[plot_df["variable_type"].isin(["categorical", "boolean"])]
        sns.scatterplot(data=plot_df, x=x, y=y, hue=hue, alpha=0.75, ax=ax)
        ax.set_title(f"{_humanize_metric(y)} versus {_humanize_metric(x)}")
        if hue:
            ax.legend(loc="best", fontsize=7)

    for j in range(len(valid_pairs), 4):
        axes_flat[j].axis("off")

    fig.suptitle("Coverage metrics versus longitudinal instability metrics", y=1.03, fontsize=16)
    _add_metric_footnote(fig, [x for x, _ in valid_pairs] + [y for _, y in valid_pairs])
    fig.tight_layout(rect=[0, 0.08, 1, 0.97])
    _save(fig, out_dir / "05_scatter_cobertura_vs_cambio.png")
    return 1


def plot_boxplots_by_type(df: pd.DataFrame, out_dir: Path) -> int:
    if "variable_type" not in df.columns:
        return 0

    y_cols = _present_columns(
        df,
        ["consistency_score", "pct_visits_covered", "change_rate", "missingness_bias_by_interval_pp"],
    )
    if not y_cols:
        return 0

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes_flat = axes.flatten()

    for i, y in enumerate(y_cols):
        ax = axes_flat[i]
        sns.boxplot(data=df, x="variable_type", y=y, ax=ax)
        ax.set_title(f"Distribution of {_humanize_metric(y)} by variable type")
        ax.tick_params(axis="x", rotation=25)

    for j in range(len(y_cols), 4):
        axes_flat[j].axis("off")

    fig.suptitle("Metric distributions stratified by variable type", y=1.03, fontsize=16)
    _add_metric_footnote(fig, y_cols)
    fig.tight_layout(rect=[0, 0.08, 1, 0.97])
    _save(fig, out_dir / "06_boxplots_por_tipo.png")
    return 1


def _plot_topn_barh(df: pd.DataFrame, metric: str, out_path: Path, n: int = 20, ascending: bool = False) -> bool:
    if metric not in df.columns or "variable" not in df.columns:
        return False

    plot_df = df[["variable", metric]].dropna().sort_values(metric, ascending=ascending).head(n)
    if plot_df.empty:
        return False

    fig, ax = plt.subplots(figsize=(11, 8))
    sns.barplot(data=plot_df, y="variable", x=metric, ax=ax, palette="Blues_r" if not ascending else "Reds")
    rank_label = "Top" if not ascending else "Lowest"
    ax.set_title(f"{rank_label} {len(plot_df)} variables by {_humanize_metric(metric)}")
    ax.set_xlabel(_humanize_metric(metric))
    ax.set_ylabel("Variable")
    _add_metric_footnote(fig, [metric])
    fig.tight_layout(rect=[0, 0.08, 1, 0.97])
    _save(fig, out_path)
    return True


def plot_topn_rankings(df: pd.DataFrame, out_dir: Path, n: int = 20) -> int:
    made = 0
    for metric in ["pct_patients_ge2", "consecutive_pair_coverage", "consistency_score"]:
        made += int(_plot_topn_barh(df, metric, out_dir / f"07_top_{metric}.png", n=n, ascending=False))

    made += int(
        _plot_topn_barh(
            df,
            "missingness_bias_by_interval_pp",
            out_dir / "07_top_menor_missingness_bias_by_interval_pp.png",
            n=n,
            ascending=True,
        )
    )

    for metric in ["flip_rate", "delta_outlier_rate"]:
        made += int(_plot_topn_barh(df, metric, out_dir / f"07_peores_{metric}.png", n=n, ascending=False))
    return made


def plot_metric_heatmap(df: pd.DataFrame, out_dir: Path, n: int = 50) -> int:
    metrics = _present_columns(
        df,
        [
            "pct_patients_ge2",
            "pct_visits_covered",
            "consecutive_pair_coverage",
            "consistency_score",
            "change_rate",
            "flip_rate",
            "reversion_rate",
            "contradiction_rate",
            "delta_outlier_rate",
            "missingness_bias_by_interval_pp",
        ],
    )
    if not metrics or "variable" not in df.columns:
        return 0

    subset = df.copy()
    if "ml_longitudinal_label" in subset.columns:
        cautious = subset[subset["ml_longitudinal_label"].eq("usable con cautela")]
        if not cautious.empty:
            subset = cautious

    sort_cols = [c for c in ["consistency_score", "pct_patients_ge2"] if c in subset.columns]
    if sort_cols:
        subset = subset.sort_values(sort_cols, ascending=False)
    subset = subset.head(n)

    hm = subset.set_index("variable")[metrics]
    if hm.empty:
        return 0

    fig_h = max(8, 0.32 * len(hm))
    fig, ax = plt.subplots(figsize=(14, fig_h))
    sns.heatmap(hm, cmap="RdYlGn", ax=ax, cbar_kws={"label": "value"})
    ax.set_title(f"Variable-by-metric heatmap for top {len(hm)} variables")
    ax.set_xlabel("Metrics")
    ax.set_ylabel("Variables")
    _add_metric_footnote(fig, metrics)
    fig.tight_layout(rect=[0, 0.08, 1, 0.97])
    _save(fig, out_dir / "08_heatmap_variables_vs_metricas.png")
    return 1


def plot_metric_correlation(df: pd.DataFrame, out_dir: Path) -> int:
    cols = _present_columns(
        df,
        [
            "pct_patients_ge1",
            "pct_patients_ge2",
            "pct_visits_covered",
            "consecutive_pair_coverage",
            "consistency_score",
            "change_rate",
            "flip_rate",
            "delta_outlier_rate",
            "missingness_bias_by_interval_pp",
        ],
    )
    if len(cols) < 2:
        return 0

    corr = df[cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=True, fmt=".2f", ax=ax)
    ax.set_title("Correlation matrix of longitudinal data quality and stability metrics")
    _add_metric_footnote(fig, cols)
    fig.tight_layout(rect=[0, 0.08, 1, 0.97])
    _save(fig, out_dir / "09_matriz_correlacion_metricas.png")
    return 1


def plot_selection_frontier(
    df: pd.DataFrame,
    out_dir: Path,
    coverage_threshold: float,
    consistency_threshold: float,
) -> int:
    x, y = "pct_patients_ge2", "consistency_score"
    if x not in df.columns or y not in df.columns:
        return 0

    fig, ax = plt.subplots(figsize=(10, 8))
    hue = "ml_longitudinal_label" if "ml_longitudinal_label" in df.columns else None
    sns.scatterplot(data=df, x=x, y=y, hue=hue, alpha=0.8, ax=ax)

    ax.axvline(coverage_threshold, color="black", linestyle="--", linewidth=1.5)
    ax.axhline(consistency_threshold, color="black", linestyle="--", linewidth=1.5)

    ax.text(coverage_threshold + 1, consistency_threshold + 1, "Strong candidates", fontsize=10)
    ax.text(coverage_threshold + 1, max(df[y].min(), 0) + 2, "Good coverage, review stability", fontsize=9)
    ax.text(max(df[x].min(), 0) + 2, consistency_threshold + 1, "Stable but low coverage", fontsize=9)
    ax.text(max(df[x].min(), 0) + 2, max(df[y].min(), 0) + 2, "Discard/low priority", fontsize=9)

    ax.set_title("Selection frontier for longitudinal feature candidates")
    ax.set_xlabel(_humanize_metric(x))
    ax.set_ylabel(_humanize_metric(y))
    _add_metric_footnote(fig, [x, y])
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    _save(fig, out_dir / "10_frontera_seleccion.png")
    return 1


def plot_major_category_label_heatmap(df: pd.DataFrame, out_dir: Path) -> int:
    if "major_category" not in df.columns or "ml_longitudinal_label" not in df.columns:
        return 0

    ct = pd.crosstab(df["major_category"], df["ml_longitudinal_label"])
    if ct.empty:
        return 0

    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    sns.heatmap(ct, annot=True, fmt="d", cmap="YlOrRd", ax=axes[0])
    axes[0].set_title("Absolute count: major clinical domain × ML longitudinal label")
    axes[0].set_xlabel("ML longitudinal label")
    axes[0].set_ylabel("Major clinical domain")
    axes[0].tick_params(axis="x", rotation=30)

    ct_norm = ct.div(ct.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    sns.heatmap(ct_norm, annot=True, fmt=".0%", cmap="RdYlGn", ax=axes[1])
    axes[1].set_title("Row proportion: within-domain ML longitudinal label composition")
    axes[1].set_xlabel("ML longitudinal label")
    axes[1].set_ylabel("Major clinical domain")
    axes[1].tick_params(axis="x", rotation=30)

    fig.suptitle("ML longitudinal label distribution by major clinical domain", y=1.03, fontsize=15)
    fig.tight_layout()
    _save(fig, out_dir / "11_heatmap_major_category_vs_label.png")
    return 1


def plot_metrics_by_major_category(df: pd.DataFrame, out_dir: Path) -> int:
    if "major_category" not in df.columns:
        return 0

    y_cols = _present_columns(
        df,
        [
            "pct_patients_ge2",
            "consecutive_pair_coverage",
            "consistency_score",
            "missingness_bias_by_interval_pp",
        ],
    )
    if not y_cols:
        return 0

    fig, axes = plt.subplots(2, 2, figsize=(22, 15))
    axes_flat = axes.flatten()
    palette = sns.color_palette("Set2", n_colors=df["major_category"].nunique())

    for i, metric in enumerate(y_cols):
        ax = axes_flat[i]
        order = df.groupby("major_category")[metric].median().sort_values(ascending=False).index
        sns.boxplot(data=df, x="major_category", y=metric, order=order, palette=palette, ax=ax)
        sns.stripplot(data=df, x="major_category", y=metric, order=order, color="black", alpha=0.25, size=3, ax=ax)
        ax.set_title(f"{_humanize_metric(metric)} distribution by major clinical domain")
        ax.tick_params(axis="x", rotation=35)
        ax.set_xlabel("")

    for j in range(len(y_cols), 4):
        axes_flat[j].axis("off")

    fig.suptitle("Metric distributions by major clinical domain", y=1.03, fontsize=15)
    _add_metric_footnote(fig, y_cols)
    fig.tight_layout(rect=[0, 0.08, 1, 0.97])
    _save(fig, out_dir / "12_boxplots_major_category.png")
    return 1


def plot_frontier_by_major_category(
    df: pd.DataFrame, out_dir: Path, coverage_threshold: float, consistency_threshold: float
) -> int:
    x, y = "pct_patients_ge2", "consistency_score"
    if x not in df.columns or y not in df.columns or "major_category" not in df.columns:
        return 0

    fig, ax = plt.subplots(figsize=(13, 10))
    palette = sns.color_palette("tab10", n_colors=df["major_category"].nunique())
    style_col = "variable_type" if "variable_type" in df.columns else None
    sns.scatterplot(
        data=df,
        x=x,
        y=y,
        hue="major_category",
        style=style_col,
        alpha=0.8,
        s=60,
        palette=palette,
        ax=ax,
    )
    ax.axvline(coverage_threshold, color="black", linestyle="--", lw=1.5)
    ax.axhline(consistency_threshold, color="black", linestyle="--", lw=1.5)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.set_title("Coverage-consistency selection frontier colored by major clinical domain")
    ax.set_xlabel(_humanize_metric(x))
    ax.set_ylabel(_humanize_metric(y))
    _add_metric_footnote(fig, [x, y])
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    _save(fig, out_dir / "13_frontier_major_category.png")
    return 1


def plot_domain_radar(df: pd.DataFrame, out_dir: Path) -> int:
    if "major_category" not in df.columns:
        return 0

    metrics = _present_columns(
        df,
        [
            "pct_patients_ge2",
            "consecutive_pair_coverage",
            "consistency_score",
            "pct_visits_covered",
            "pct_patients_in_multiple_intervals",
        ],
    )
    if len(metrics) < 3:
        return 0

    domain_means = df.groupby("major_category")[metrics].mean().dropna(how="all")
    if domain_means.empty:
        return 0

    normed = (domain_means - domain_means.min()) / (domain_means.max() - domain_means.min() + 1e-9)
    n_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(11, 10), subplot_kw={"polar": True})
    colors = sns.color_palette("tab10", n_colors=len(normed))
    for (domain, row), color in zip(normed.iterrows(), colors):
        values = row.tolist() + row.tolist()[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=domain, color=color)
        ax.fill(angles, values, alpha=0.08, color=color)

    ax.set_thetagrids(np.degrees(angles[:-1]), [_humanize_metric(m) for m in metrics], fontsize=9)
    ax.set_title("Normalized longitudinal profile by major clinical domain", y=1.12, fontsize=14)
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.15), fontsize=9)
    _add_metric_footnote(fig, metrics)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    _save(fig, out_dir / "14_radar_domain_profiles.png")
    return 1


def plot_category_composition_by_domain(df: pd.DataFrame, out_dir: Path) -> int:
    if "major_category" not in df.columns or "category" not in df.columns or "pct_patients_ge2" not in df.columns:
        return 0

    domains = sorted(df["major_category"].dropna().unique())
    if not domains:
        return 0
    nrows = (len(domains) + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(20, 5 * nrows))
    axes_flat = np.array(axes).reshape(-1)

    for i, domain in enumerate(domains):
        ax = axes_flat[i]
        sub = (
            df[df["major_category"] == domain]
            .groupby("category", dropna=False)["pct_patients_ge2"]
            .agg(["mean", "count"])
            .sort_values("mean", ascending=False)
            .reset_index()
        )
        bars = ax.barh(sub["category"].astype(str), sub["mean"], color="#2c7fb8")
        for bar, n_vars in zip(bars, sub["count"]):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2, f"n={n_vars}", va="center", fontsize=8)
        ax.set_title(domain.replace("_", " ").title())
        ax.set_xlabel("Mean pct_patients_ge2")
        ax.set_xlim(0, 105)

    for j in range(len(domains), len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle("Sub-category coverage drill-down within each major clinical domain", y=1.02, fontsize=14)
    _add_metric_footnote(fig, ["pct_patients_ge2"])
    fig.tight_layout(rect=[0, 0.08, 1, 0.97])
    _save(fig, out_dir / "15_drilldown_category_by_domain.png")
    return 1


def main() -> None:
    args = _parse_args()
    logger = setup_logger("14_longitudinal_plausibility_plots")

    print_script_overview(
        "14_longitudinal_plausibility_plots.py",
        "Generates a visualization suite to prioritize longitudinal variables.",
    )

    print_step(1, "Load longitudinal variable summary")
    df = _load(args.input_path)

    print_step(2, "Generate plot suite")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    generated = 0
    generated += plot_coverage_histograms(df, args.output_dir)
    generated += plot_variable_type_bars(df, args.output_dir)
    generated += plot_label_bars(df, args.output_dir)
    generated += plot_coverage_vs_consistency(df, args.output_dir)
    generated += plot_coverage_vs_change(df, args.output_dir)
    generated += plot_boxplots_by_type(df, args.output_dir)
    generated += plot_topn_rankings(df, args.output_dir, n=max(1, args.top_n))
    generated += plot_metric_heatmap(df, args.output_dir, n=max(1, max(args.top_n, 50)))
    generated += plot_metric_correlation(df, args.output_dir)
    generated += plot_selection_frontier(
        df,
        args.output_dir,
        coverage_threshold=args.coverage_threshold,
        consistency_threshold=args.consistency_threshold,
    )
    generated += plot_major_category_label_heatmap(df, args.output_dir)
    generated += plot_metrics_by_major_category(df, args.output_dir)
    generated += plot_frontier_by_major_category(
        df,
        args.output_dir,
        coverage_threshold=args.coverage_threshold,
        consistency_threshold=args.consistency_threshold,
    )
    generated += plot_domain_radar(df, args.output_dir)
    generated += plot_category_composition_by_domain(df, args.output_dir)

    print_step(3, "Report outputs")
    print_kv(
        "Longitudinal plot generation",
        {
            "input_path": str(args.input_path),
            "output_dir": str(args.output_dir),
            "n_variables": int(len(df)),
            "n_plots_generated": generated,
        },
    )
    logger.info("Longitudinal plots created: n=%d output=%s", generated, args.output_dir)


if __name__ == "__main__":
    main()
