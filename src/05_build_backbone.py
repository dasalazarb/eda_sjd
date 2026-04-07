from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import (
    ANALYTIC_DIR,
    EDA_UNIFIED_REPORT_PATH,
    INTERMEDIATE_DIR,
    REPORTS_DIR,
    build_targeted_eda_sheets,
    merge_sheet_dicts,
    print_kv,
    print_script_overview,
    print_step,
    resolve_canonical_column,
    save_parquet_and_csv,
    setup_logger,
    upsert_eda_sheets_xlsx,
)


def _resolve_optional_column(df: pd.DataFrame, canonical_name: str) -> str | None:
    """Return the best column match for a canonical name, or None when absent."""
    try:
        return resolve_canonical_column(df, canonical_name)
    except KeyError:
        return None


def build_n_visits_boxplot(patient_master: pd.DataFrame) -> str | None:
    """Build a professional boxplot for patient-level visit counts."""
    n_visits = pd.to_numeric(patient_master.get("n_visits"), errors="coerce").dropna()
    if n_visits.empty:
        return None

    q1 = float(n_visits.quantile(0.25))
    q3 = float(n_visits.quantile(0.75))
    iqr = q3 - q1
    median = float(n_visits.median())
    mean = float(n_visits.mean())
    min_v = float(n_visits.min())
    max_v = float(n_visits.max())

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(11, 4.8))

    boxprops = dict(facecolor="#5DADE2", edgecolor="#1B4F72", alpha=0.35, linewidth=1.8)
    whiskerprops = dict(color="#1B4F72", linewidth=1.6)
    capprops = dict(color="#1B4F72", linewidth=1.6)
    medianprops = dict(color="#C0392B", linewidth=2.6)
    flierprops = dict(marker="o", markerfacecolor="#154360", markeredgecolor="#154360", alpha=0.4, markersize=4)

    sns.boxplot(
        x=n_visits,
        ax=ax,
        width=0.35,
        boxprops=boxprops,
        whiskerprops=whiskerprops,
        capprops=capprops,
        medianprops=medianprops,
        flierprops=flierprops,
    )

    ax.axvline(mean, color="#117A65", linestyle="--", linewidth=2.0, alpha=0.9, label=f"Mean: {mean:.2f}")
    ax.axvline(median, color="#C0392B", linestyle="-", linewidth=2.2, alpha=0.95, label=f"Median: {median:.2f}")
    ax.axvspan(q1, q3, color="#85C1E9", alpha=0.15, label=f"IQR (Q1-Q3): {q1:.2f} - {q3:.2f}")
    ax.axvline(min_v, color="#7D3C98", linestyle=":", linewidth=1.6, alpha=0.65, label=f"Min: {min_v:.0f}")
    ax.axvline(max_v, color="#7D3C98", linestyle=":", linewidth=1.6, alpha=0.65, label=f"Max: {max_v:.0f}")

    ax.set_title("Distribution of n_visits in patient_master", pad=16, fontweight="bold")
    ax.set_xlabel("Number of visits per patient")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.22)
    ax.grid(axis="y", visible=False)

    summary_text = (
        f"n={len(n_visits):,}  |  Min={min_v:.0f}  Max={max_v:.0f}  "
        f"Mean={mean:.2f}  Median={median:.2f}  IQR={iqr:.2f}"
    )
    fig.text(0.5, 0.06, summary_text, ha="center", va="center", fontsize=10, color="#1C2833")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=True, fontsize=9)

    plot_dir = REPORTS_DIR / "05_backbone"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / "patient_master_n_visits_boxplot.png"
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(plot_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(plot_path)


def build_patient_master(visits: pd.DataFrame) -> pd.DataFrame:
    """Create one row per patient with visit counts and temporal coverage."""
    patient_col = resolve_canonical_column(visits, "patient_record_number")
    subject_col = _resolve_optional_column(visits, "subject_number")
    row_id_col = resolve_canonical_column(visits, "row_id_raw")
    visit_datetime_col = resolve_canonical_column(visits, "visit_datetime")
    source_protocol_col = _resolve_optional_column(visits, "source_protocol")

    grp = visits.groupby(patient_col, dropna=False)
    master = grp.agg(
        n_visits=(row_id_col, "count"),
        first_visit=(visit_datetime_col, "min"),
        last_visit=(visit_datetime_col, "max"),
    ).reset_index()

    if patient_col != "patient_record_number":
        master = master.rename(columns={patient_col: "patient_record_number"})

    if subject_col:
        subject_map = (
            visits[[patient_col, subject_col]]
            .dropna(subset=[subject_col])
            .drop_duplicates(subset=[patient_col], keep="first")
            .rename(columns={patient_col: "patient_record_number", subject_col: "subject_number"})
        )
        master = master.merge(subject_map, on="patient_record_number", how="left")

    if source_protocol_col:
        sources = grp[source_protocol_col].nunique(dropna=True).reset_index(name="n_protocols").rename(
            columns={patient_col: "patient_record_number"}
        )
        master = master.merge(sources, on="patient_record_number", how="left")

    return master


def main() -> None:
    logger = setup_logger("05_build_backbone")

    print_script_overview(
        "05_build_backbone.py",
        "Builds patient-level master table and longitudinal visits backbone from deduplicated data.",
    )

    print_step(1, "Load deduplicated visits")
    visits = pd.read_parquet(INTERMEDIATE_DIR / "deduped_visits.parquet")

    print_step(2, "Create patient_master with visit count and temporal span")
    master = build_patient_master(visits)

    print_step(3, "Save patient_master and visits_long outputs")
    save_parquet_and_csv(master, ANALYTIC_DIR / "patient_master", logger)
    save_parquet_and_csv(visits, ANALYTIC_DIR / "visits_long", logger)

    print_step(4, "Generate professional boxplot for patient_master.n_visits")
    plot_path = build_n_visits_boxplot(master)
    if plot_path:
        logger.info("Saved n_visits boxplot: %s", plot_path)
    else:
        logger.warning("Could not build n_visits boxplot (no numeric values found).")

    print_kv(
        "Backbone summary",
        {
            "n_patients": int(master["patient_record_number"].nunique(dropna=True)),
            "n_visits": len(visits),
        },
    )
    print_step(5, "Append targeted EDA for patient_master and visits to unified workbook")
    sheets = {}
    sheets = merge_sheet_dicts(
        sheets,
        build_targeted_eda_sheets(master, "05_patient_master_output", "05_patient_master_output", consolidated=True),
    )
    sheets = merge_sheet_dicts(
        sheets,
        build_targeted_eda_sheets(visits, "05_visits_long_output", "05_visits_long_output", consolidated=True),
    )
    workbook = upsert_eda_sheets_xlsx(EDA_UNIFIED_REPORT_PATH, sheets)
    logger.info("Updated unified EDA workbook: %s", workbook)


if __name__ == "__main__":
    main()
