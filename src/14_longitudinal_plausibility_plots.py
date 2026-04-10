from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from common import REPORTS_DIR, print_kv, print_script_overview, print_step, setup_logger

sns.set_theme(style="whitegrid", context="talk")


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
        ax.set_title(metric)
        ax.set_xlabel("Percentage")

    for j in range(len(metrics), len(axes)):
        axes[j].axis("off")

    fig.suptitle("Variable coverage histogram", y=1.02, fontsize=16)
    _save(fig, out_dir / "01_hist_cobertura_variables.png")
    return 1


def plot_variable_type_bars(df: pd.DataFrame, out_dir: Path) -> int:
    if "variable_type" not in df.columns:
        return 0

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    vc = df["variable_type"].value_counts(dropna=False)
    sns.barplot(x=vc.index.astype(str), y=vc.values, ax=axes[0, 0], color="#5e3c99")
    axes[0, 0].set_title("Number of variables by type")
    axes[0, 0].set_xlabel("variable_type")
    axes[0, 0].set_ylabel("n_variables")
    axes[0, 0].tick_params(axis="x", rotation=25)

    if "ml_longitudinal_label" in df.columns:
        tmp = pd.crosstab(df["variable_type"], df["ml_longitudinal_label"], normalize="index")
        tmp.plot(kind="bar", stacked=True, ax=axes[0, 1], colormap="viridis")
        axes[0, 1].set_title("Label proportion by variable_type")
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
        axes[1, 0].set_title("Average coverage by type")
        axes[1, 0].set_ylabel("pct_visits_covered (mean)")
        axes[1, 0].tick_params(axis="x", rotation=25)
    else:
        axes[1, 0].axis("off")

    if "consistency_score" in df.columns:
        tmp_cons = (
            df.groupby("variable_type", dropna=False)["consistency_score"].mean().sort_values(ascending=False)
        )
        sns.barplot(x=tmp_cons.index.astype(str), y=tmp_cons.values, ax=axes[1, 1], color="#d95f02")
        axes[1, 1].set_title("Average consistency by type")
        axes[1, 1].set_ylabel("consistency_score (mean)")
        axes[1, 1].tick_params(axis="x", rotation=25)
    else:
        axes[1, 1].axis("off")

    fig.suptitle("Summary by variable_type", y=1.02, fontsize=16)
    fig.tight_layout()
    _save(fig, out_dir / "02_barras_variable_type.png")
    return 1


def plot_label_bars(df: pd.DataFrame, out_dir: Path) -> int:
    if "ml_longitudinal_label" not in df.columns:
        return 0

    counts = df["ml_longitudinal_label"].value_counts(dropna=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax, palette="Set2")
    ax.set_title("Final longitudinal utility classification")
    ax.set_xlabel("ml_longitudinal_label")
    ax.set_ylabel("n_variables")
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
        ax.set_title(f"{x} vs {y}")
        if hue:
            ax.legend(loc="best", fontsize=8)

    fig.suptitle("Coverage vs consistency", y=1.02, fontsize=16)
    fig.tight_layout()
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
        ax.set_title(f"{x} vs {y}")
        if hue:
            ax.legend(loc="best", fontsize=7)

    for j in range(len(valid_pairs), 4):
        axes_flat[j].axis("off")

    fig.suptitle("Coverage vs change/instability", y=1.02, fontsize=16)
    fig.tight_layout()
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
        ax.set_title(f"{y} by variable_type")
        ax.tick_params(axis="x", rotation=25)

    for j in range(len(y_cols), 4):
        axes_flat[j].axis("off")

    fig.suptitle("Boxplots by variable type", y=1.02, fontsize=16)
    fig.tight_layout()
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
    ax.set_title(f"{'Top' if not ascending else 'Worst'} {len(plot_df)} by {metric}")
    ax.set_xlabel(metric)
    ax.set_ylabel("variable")
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
    ax.set_title(f"Variable vs metric heatmap (top {len(hm)})")
    ax.set_xlabel("Metrics")
    ax.set_ylabel("Variables")
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
    ax.set_title("Correlation matrix of longitudinal metrics")
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

    ax.set_title("Selection frontier: coverage vs consistency")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    _save(fig, out_dir / "10_frontera_seleccion.png")
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
