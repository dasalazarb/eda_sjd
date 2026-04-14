"""Tier 1 longitudinal analysis pipeline for salivary and eye outcomes."""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Gaussian, Poisson

assert sys.version_info >= (3, 10), "Python 3.10+ required"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)

ID_COLS = ["ids__patient_record_number", "ids__interval_name"]
DEMO_COLS = ["ids__dob", "ids__race", "ids__ethnicity", "ids__sex", "ids__age_at_visit"]
SAL_VARS = [
    "salivary_flow_form__flow_p1l",
    "salivary_flow_form__flow_p2l",
    "salivary_flow_form__flow_p2r",
    "salivary_flow_form__flow_pir",
    "salivary_flow_form__flow_sm1",
    "salivary_flow_form__flow_sm2",
    "salivary_flow_form__flow_whole_unstim",
    "salivary_flow_form__tot_sim_sal_flow",
    "salivary_flow_form__tot_unsim_sal_flow",
]
FLOOR_VARS = ["salivary_flow_form__flow_p1l", "salivary_flow_form__flow_pir"]
EYE_VARS = [
    "eye_examination__sch_l",
    "eye_examination__sch_r",
    "eye_examination__schwa_l",
    "eye_examination__schwa_r",
    "eye_examination__tbut_l",
    "eye_examination__tbut_r",
    "eye_examination__oxford_l",
    "eye_examination__oxford_r",
    "eye_examination__vanb_l",
]
BILATERAL_PAIRS = [
    ("eye_examination__sch_l", "eye_examination__sch_r"),
    ("eye_examination__schwa_l", "eye_examination__schwa_r"),
    ("eye_examination__tbut_l", "eye_examination__tbut_r"),
    ("eye_examination__oxford_l", "eye_examination__oxford_r"),
]


def _extract_visit_order(interval: Any) -> float:
    match = re.search(r"(\d+)", str(interval))
    if match:
        return float(match.group(1))
    if str(interval).lower() == "baseline":
        return 0.0
    return np.nan


def build_tier1_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Build Tier 1 longitudinal panel with time variables.

    Args:
        df: Raw visit-level dataframe.

    Returns:
        Filtered dataframe with visit_order and time_years.
    """
    tier_vars = SAL_VARS + EYE_VARS
    panel = df.loc[df[tier_vars].notna().any(axis=1)].copy()
    panel["visit_order"] = panel["ids__interval_name"].map(_extract_visit_order)

    def _fill_order(group: pd.DataFrame) -> pd.DataFrame:
        if group["visit_order"].isna().any():
            group = group.sort_values("ids__interval_name").copy()
            ranks = pd.Series(np.arange(group.shape[0]), index=group.index, dtype=float)
            group["visit_order"] = group["visit_order"].fillna(ranks)
        return group

    panel = panel.groupby("ids__patient_record_number", group_keys=False).apply(_fill_order)
    panel["visit_order"] = panel["visit_order"].astype(int)

    if "ids__dob" in panel.columns and "ids__age_at_visit" in panel.columns:
        panel["ids__dob"] = pd.to_datetime(panel["ids__dob"], errors="coerce")
        panel["visit_date_proxy"] = panel["ids__dob"] + pd.to_timedelta(panel["ids__age_at_visit"] * 365.25, unit="D")
        base_date = panel.groupby("ids__patient_record_number")["visit_date_proxy"].transform("min")
        panel["time_years"] = (panel["visit_date_proxy"] - base_date).dt.days / 365.25
    else:
        panel["time_years"] = np.nan

    global_gap = panel.groupby("ids__patient_record_number")["visit_order"].diff().median()
    if pd.isna(global_gap) or global_gap == 0:
        global_gap = 1.0
    fallback = (
        panel["visit_order"] - panel.groupby("ids__patient_record_number")["visit_order"].transform("min")
    ) * float(global_gap)
    panel["time_years"] = panel["time_years"].fillna(fallback / 12.0)

    for col in DEMO_COLS:
        if col in panel.columns:
            panel[col] = panel.groupby("ids__patient_record_number")[col].ffill().bfill()

    panel = panel.sort_values(["ids__patient_record_number", "time_years"]).reset_index(drop=True)
    assert (panel["time_years"] >= 0).all(), "time_years must be non-negative"
    assert not panel.duplicated(["ids__patient_record_number", "visit_order"]).any(), "Duplicate patient/visit_order"
    return panel


def describe_tier1(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Create longitudinal descriptive summaries for Tier 1 variables.

    Args:
        panel: Time-enriched panel dataframe.

    Returns:
        Dictionary containing salivary and eye descriptive tables.
    """
    total_patients = panel["ids__patient_record_number"].nunique()

    def _summarize_var(var: str) -> dict[str, Any]:
        sub = panel[["ids__patient_record_number", "time_years", var]].dropna(subset=[var]).copy()
        if sub.empty:
            return {"variable": var}
        n_obs = sub.groupby("ids__patient_record_number")[var].count()
        first_vals = sub.sort_values("time_years").groupby("ids__patient_record_number", as_index=False).first()[var]
        deltas = (
            sub.sort_values(["ids__patient_record_number", "time_years"]) 
            .groupby("ids__patient_record_number")[var]
            .diff()
            .abs()
            .dropna()
        )
        pct_zeros = float((first_vals == 0).mean() * 100) if len(first_vals) else np.nan
        return {
            "variable": var,
            "n_patients_observed": int(n_obs.index.nunique()),
            "n_patients_ge2": int((n_obs >= 2).sum()),
            "pct_patients_ge2": float((n_obs >= 2).sum() / total_patients * 100),
            "mean_baseline": float(first_vals.mean()),
            "median_baseline": float(first_vals.median()),
            "std_baseline": float(first_vals.std()),
            "pct_zeros_at_baseline": pct_zeros,
            "inter_visit_delta_p50": float(deltas.median()) if not deltas.empty else np.nan,
            "inter_visit_delta_p95": float(deltas.quantile(0.95)) if not deltas.empty else np.nan,
            "floor_effect_flag": bool(pct_zeros > 20),
        }

    sal_df = pd.DataFrame([_summarize_var(v) for v in SAL_VARS])
    eye_df = pd.DataFrame([_summarize_var(v) for v in EYE_VARS])
    LOGGER.info("Salivary descriptive summary:\n%s", sal_df.to_string(index=False))
    LOGGER.info("Eye descriptive summary:\n%s", eye_df.to_string(index=False))
    return {"salivary": sal_df, "eye": eye_df}


def run_lmm_salivary(
    panel: pd.DataFrame,
    outcome: str = "salivary_flow_form__flow_whole_unstim",
    covariates: list[str] = ["ids__age_at_visit", "ids__sex"],
) -> dict[str, Any] | None:
    """Run salivary linear mixed model with random slope and intercept.

    Args:
        panel: Time-enriched panel.
        outcome: Salivary variable to model.
        covariates: Fixed-effect covariates.

    Returns:
        Dictionary with model outputs, or None on fitting failure.
    """
    data = panel.dropna(subset=[outcome]).copy()
    if data.empty:
        return None
    first_vals = data.sort_values("time_years").groupby("ids__patient_record_number", as_index=False).first()[outcome]
    pct_zeros = float((first_vals == 0).mean() * 100)
    if pct_zeros >= 20:
        LOGGER.warning("Floor effect detected in %s. Use Tobit model (TASK 2b). Skipping LMM.", outcome)
        return None

    transform = "none"
    model_outcome = outcome
    if pct_zeros < 5:
        transform = "log"
        model_outcome = f"{outcome}__log"
        data[model_outcome] = np.log(data[outcome] + 1e-6)

    formula = f"{model_outcome} ~ time_years"
    if covariates:
        formula += " + " + " + ".join(covariates)

    try:
        model = smf.mixedlm(formula, data=data, groups=data["ids__patient_record_number"], re_formula="~time_years")
        result = model.fit(method="lbfgs")
    except Exception as exc:
        LOGGER.warning("LMM failed for %s: %s", outcome, exc)
        return None

    re_rows: list[dict[str, Any]] = []
    for pid, re in result.random_effects.items():
        re_rows.append(
            {
                "patient_id": pid,
                "re_intercept": float(re.get("Group", np.nan)) if hasattr(re, "get") else float(re.iloc[0]),
                "re_slope_time_years": float(re.get("time_years", np.nan)) if hasattr(re, "get") else float(re.iloc[-1]),
            }
        )
    re_df = pd.DataFrame(re_rows)

    conf = result.conf_int()
    fixed_df = pd.DataFrame(
        {
            "covariate": result.fe_params.index,
            "estimate": result.fe_params.values,
            "ci_low": conf.loc[result.fe_params.index, 0].values,
            "ci_high": conf.loc[result.fe_params.index, 1].values,
            "pvalue": result.pvalues[result.fe_params.index].values,
        }
    )
    return {
        "model_result": result,
        "random_effects_df": re_df,
        "fixed_effects_summary": fixed_df,
        "transform_applied": transform,
        "n_patients": int(data["ids__patient_record_number"].nunique()),
        "n_observations": int(data.shape[0]),
    }


def run_tobit_salivary(
    panel: pd.DataFrame,
    outcome: str,
    left_censoring_value: float = 0.0,
) -> dict[str, Any] | None:
    """Approximate Tobit via a two-part model for floor-effect outcomes.

    Args:
        panel: Time-enriched panel.
        outcome: Outcome expected in FLOOR_VARS.
        left_censoring_value: Censoring threshold.

    Returns:
        Dictionary with binary and continuous model outputs, or None on failure.
    """
    if outcome not in FLOOR_VARS:
        LOGGER.warning("Outcome %s is not listed in FLOOR_VARS.", outcome)
    data = panel.dropna(subset=[outcome]).copy()
    if data.empty:
        return None
    data["outcome_binary"] = (data[outcome] > left_censoring_value).astype(int)
    first_vals = data.sort_values("time_years").groupby("ids__patient_record_number", as_index=False).first()[outcome]
    pct_zeros = float((first_vals <= left_censoring_value).mean() * 100)

    try:
        binary_model = smf.logit("outcome_binary ~ time_years + ids__age_at_visit + ids__sex", data=data).fit(disp=False)
    except Exception as exc:
        LOGGER.warning("Binary Tobit part failed for %s: %s", outcome, exc)
        return None

    continuous_model = None
    positive = data.loc[data[outcome] > left_censoring_value].copy()
    if positive.shape[0] >= 30:
        try:
            formula = f"{outcome} ~ time_years + ids__age_at_visit + ids__sex"
            continuous_model = smf.mixedlm(
                formula,
                data=positive,
                groups=positive["ids__patient_record_number"],
                re_formula="~time_years",
            ).fit(method="lbfgs")
        except Exception as exc:
            LOGGER.warning("Continuous Tobit part failed for %s: %s", outcome, exc)

    return {
        "binary_model": binary_model,
        "continuous_model": continuous_model,
        "pct_zeros": pct_zeros,
        "interpretation": f"{pct_zeros:.1f}% of patients show zero {outcome} at baseline",
    }


def run_gee_eye(
    panel: pd.DataFrame,
    outcome_pair: tuple[str, str],
    covariates: list[str] = ["ids__age_at_visit", "ids__sex"],
) -> dict[str, Any] | None:
    """Fit bilateral-eye GEE with patient-level clustering.

    Args:
        panel: Time-enriched panel.
        outcome_pair: Left/right eye outcome pair.
        covariates: Additional fixed effects.

    Returns:
        Dictionary with GEE outputs, or None on failure.
    """
    left_col, right_col = outcome_pair
    long_df = panel[
        ["ids__patient_record_number", "visit_order", "time_years", *covariates, left_col, right_col]
    ].melt(
        id_vars=["ids__patient_record_number", "visit_order", "time_years", *covariates],
        value_vars=[left_col, right_col],
        var_name="eye_var",
        value_name="outcome_value",
    )
    long_df = long_df.dropna(subset=["outcome_value"])
    long_df["eye_side"] = (long_df["eye_var"] == right_col).astype(int)

    family = Poisson() if any(k in left_col for k in ["oxford", "vanb"]) else Gaussian()
    formula = "outcome_value ~ time_years + eye_side"
    if covariates:
        formula += " + " + " + ".join(covariates)

    try:
        model = smf.gee(
            formula,
            groups="ids__patient_record_number",
            cov_struct=Exchangeable(),
            data=long_df,
            family=family,
        )
        result = model.fit()
    except Exception as exc:
        LOGGER.warning("GEE failed for %s/%s: %s", left_col, right_col, exc)
        return None

    ci = result.conf_int().loc["time_years"].tolist()
    return {
        "gee_result": result,
        "eye_asymmetry_coef": float(result.params.get("eye_side", np.nan)),
        "eye_asymmetry_pval": float(result.pvalues.get("eye_side", np.nan)),
        "time_slope_coef": float(result.params.get("time_years", np.nan)),
        "time_slope_pval": float(result.pvalues.get("time_years", np.nan)),
        "time_slope_ci": (float(ci[0]), float(ci[1])),
        "n_patients": int(long_df["ids__patient_record_number"].nunique()),
        "n_eye_observations": int(long_df.shape[0]),
    }


def compute_derived_variables(panel: pd.DataFrame) -> pd.DataFrame:
    """Compute clinical derived variables used in Tier 1 analyses.

    Args:
        panel: Time-enriched panel.

    Returns:
        Panel with seven added derived columns.
    """
    panel = panel.copy()
    panel["reflex_tear_l"] = panel["eye_examination__sch_l"] - panel["eye_examination__schwa_l"]
    panel["reflex_tear_r"] = panel["eye_examination__sch_r"] - panel["eye_examination__schwa_r"]

    panel["tbut_mean"] = panel[["eye_examination__tbut_l", "eye_examination__tbut_r"]].mean(axis=1)
    panel["sch_mean"] = panel[["eye_examination__sch_l", "eye_examination__sch_r"]].mean(axis=1)
    panel["dry_eye_type"] = np.select(
        [
            (panel["sch_mean"] < 5) & (panel["tbut_mean"] >= 5),
            (panel["tbut_mean"] < 5) & (panel["sch_mean"] >= 5),
            (panel["sch_mean"] < 5) & (panel["tbut_mean"] < 5),
            (panel["sch_mean"] >= 5) & (panel["tbut_mean"] >= 5),
        ],
        ["aqueous_deficient", "evaporative", "mixed", "normal"],
        default=np.nan,
    )

    p2_total = panel[["salivary_flow_form__flow_p2l", "salivary_flow_form__flow_p2r"]].sum(axis=1, min_count=1)
    sm_total = panel[["salivary_flow_form__flow_sm1", "salivary_flow_form__flow_sm2"]].sum(axis=1, min_count=1)
    panel["parotid_dominance_ratio"] = p2_total / (p2_total + sm_total)

    z_vars = [
        "salivary_flow_form__flow_whole_unstim",
        "salivary_flow_form__tot_sim_sal_flow",
        "eye_examination__sch_l",
        "eye_examination__sch_r",
        "eye_examination__tbut_l",
        "eye_examination__tbut_r",
    ]
    baseline = panel.sort_values("time_years").groupby("ids__patient_record_number", as_index=False).first()
    for col in z_vars:
        mu = baseline[col].mean()
        sd = baseline[col].std()
        panel[f"z_{col}"] = (panel[col] - mu) / sd if sd and not np.isnan(sd) else np.nan
    z_cols = [f"z_{v}" for v in z_vars]
    valid_count = panel[z_cols].notna().sum(axis=1)
    panel["exocrine_composite_z"] = panel[z_cols].mean(axis=1)
    panel.loc[valid_count < 3, "exocrine_composite_z"] = np.nan

    denom = panel["eye_examination__sch_l"] + panel["eye_examination__sch_r"]
    panel["bilateral_sch_asymmetry"] = (panel["eye_examination__sch_l"] - panel["eye_examination__sch_r"]) / denom
    panel["bilateral_sch_asymmetry"] = panel["bilateral_sch_asymmetry"].clip(-1, 1)

    panel = panel.drop(columns=["tbut_mean", "sch_mean", *z_cols], errors="ignore")
    return panel


def cluster_salivary_trajectories(random_effects_df: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    """Cluster salivary trajectories using random effects.

    Args:
        random_effects_df: Dataframe with patient random intercept and slope.
        n_clusters: Number of k-means clusters.

    Returns:
        Cluster-labeled dataframe.
    """
    df = random_effects_df.copy()
    features = df[["re_intercept", "re_slope_time_years"]].values
    scaled = StandardScaler().fit_transform(features)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    df["cluster_id"] = km.fit_predict(scaled)

    for k in [2, 3, 4, 5]:
        if len(df) > k:
            sc = silhouette_score(scaled, KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(scaled))
            LOGGER.info("Silhouette score k=%d: %.3f", k, sc)

    cents = df.groupby("cluster_id")[["re_intercept", "re_slope_time_years"]].mean()
    hi_thr = cents["re_intercept"].median()
    labels: dict[int, str] = {}
    for cid, row in cents.iterrows():
        if row["re_slope_time_years"] > 0.05:
            labels[cid] = "improving"
        elif row["re_intercept"] >= hi_thr and row["re_slope_time_years"] < -0.05:
            labels[cid] = "declining"
        elif row["re_intercept"] >= hi_thr:
            labels[cid] = "stable_high"
        else:
            labels[cid] = "stable_low"
    df["cluster_label"] = df["cluster_id"].map(labels)
    return df[["patient_id", "re_intercept", "re_slope_time_years", "cluster_label", "cluster_id"]]


def run_km_time_to_event(panel: pd.DataFrame, event_definitions: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """Fit Kaplan-Meier time-to-event curves for dry mouth and eye events.

    Args:
        panel: Time-enriched panel dataframe.
        event_definitions: Optional event-definition mapping.

    Returns:
        Dictionary of KM fit artifacts keyed by event name.
    """
    if event_definitions is None:
        event_definitions = {
            "hyposalivation": {"col": "salivary_flow_form__flow_whole_unstim", "threshold": 0.1, "direction": "below"},
            "severe_dry_eye_sch": {"col": "eye_examination__sch_l", "threshold": 5.0, "direction": "below"},
            "severe_tbut": {"col": "eye_examination__tbut_l", "threshold": 5.0, "direction": "below"},
        }

    out: dict[str, Any] = {}
    for name, cfg in event_definitions.items():
        rows: list[dict[str, Any]] = []
        for pid, grp in panel.groupby("ids__patient_record_number"):
            g = grp.sort_values("time_years")
            obs = g.dropna(subset=[cfg["col"]])
            if obs.empty:
                continue
            if cfg["direction"] == "below":
                hit = obs[obs[cfg["col"]] < cfg["threshold"]]
            else:
                hit = obs[obs[cfg["col"]] > cfg["threshold"]]
            if hit.empty:
                rows.append({"patient_id": pid, "duration": obs["time_years"].max(), "event": 0})
            else:
                rows.append({"patient_id": pid, "duration": hit["time_years"].iloc[0], "event": 1})
        ev_df = pd.DataFrame(rows)
        if ev_df.empty:
            continue
        kmf = KaplanMeierFitter(label=name)
        kmf.fit(durations=ev_df["duration"], event_observed=ev_df["event"])
        ci = kmf.confidence_interval_survival_function_.copy()
        out[name] = {
            "kmf": kmf,
            "median_time_years": float(kmf.median_survival_time_),
            "median_ci": (float(ci.iloc[:, 0].min()), float(ci.iloc[:, 1].max())),
            "n_events": int(ev_df["event"].sum()),
            "n_censored": int((1 - ev_df["event"]).sum()),
        }
    return out


def plot_tier1_results(
    panel: pd.DataFrame,
    lmm_results: dict[str, Any],
    gee_results: dict[str, Any],
    cluster_df: pd.DataFrame,
    km_results: dict[str, Any],
    output_dir: str = "./figures",
) -> None:
    """Create and save Tier 1 visualizations.

    Args:
        panel: Derived panel.
        lmm_results: LMM output for whole unstimulated flow.
        gee_results: GEE output for Schirmer pair.
        cluster_df: Cluster assignments.
        km_results: KM outputs.
        output_dir: Figure output folder.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    merged = panel.merge(cluster_df[["patient_id", "cluster_label"]], left_on="ids__patient_record_number", right_on="patient_id", how="left")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=merged, x="time_years", y="salivary_flow_form__flow_whole_unstim", hue="cluster_label", units="ids__patient_record_number", estimator=None, alpha=0.3, legend=False, ax=ax)
    ax.axhline(0.1, color="red", linestyle="--", label="Hyposalivation threshold (0.1 mL/5min)")
    ax.legend()
    fig.tight_layout(); fig.savefig(out / "salivary_spaghetti.png", dpi=300); plt.close(fig)

    clusters = sorted(cluster_df["cluster_label"].dropna().unique())[:4]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for ax, c in zip(axes.flat, clusters):
        pids = cluster_df.loc[cluster_df["cluster_label"] == c, "patient_id"]
        sub = panel[panel["ids__patient_record_number"].isin(pids)]
        sns.lineplot(data=sub, x="time_years", y="salivary_flow_form__flow_whole_unstim", units="ids__patient_record_number", estimator=None, alpha=0.25, color="gray", ax=ax)
        sns.lineplot(data=sub, x="time_years", y="salivary_flow_form__flow_whole_unstim", estimator="mean", ci=95, color="blue", ax=ax)
        ax.set_title(f"{c} (n={pids.nunique()})")
    fig.tight_layout(); fig.savefig(out / "trajectory_clusters.png", dpi=300); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    vars4 = ["eye_examination__sch_l", "eye_examination__sch_r", "eye_examination__tbut_l", "eye_examination__tbut_r"]
    for ax, var in zip(axes.flat, vars4):
        sns.lineplot(data=panel, x="time_years", y=var, units="ids__patient_record_number", estimator=None, alpha=0.2, color="gray", ax=ax)
        sns.regplot(data=panel, x="time_years", y=var, scatter=False, color="blue", ax=ax)
        ax.text(0.02, 0.95, f"slope={gee_results.get('time_slope_coef', np.nan):.3f}\np={gee_results.get('time_slope_pval', np.nan):.3g}", transform=ax.transAxes, va="top")
    fig.tight_layout(); fig.savefig(out / "eye_bilateral_trajectories.png", dpi=300); plt.close(fig)

    dry = panel.groupby(["visit_order", "dry_eye_type"]).size().unstack(fill_value=0)
    dry = dry[dry.sum(axis=1) >= 10]
    prop = dry.div(dry.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(10, 6)); prop.plot(kind="bar", stacked=True, ax=ax)
    fig.tight_layout(); fig.savefig(out / "dry_eye_phenotype_over_time.png", dpi=300); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    kmfs = []
    for v in km_results.values():
        kmf = v["kmf"]
        kmf.plot(ci_show=True, ax=ax)
        kmfs.append(kmf)
    if kmfs:
        add_at_risk_counts(*kmfs, ax=ax)
    fig.tight_layout(); fig.savefig(out / "km_curves.png", dpi=300); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    x = cluster_df["re_slope_time_years"]
    y = cluster_df["re_slope_time_years"]
    sns.scatterplot(x=x, y=y, hue=cluster_df["cluster_label"], ax=ax)
    rho, pval = spearmanr(x, y, nan_policy="omit")
    ax.text(0.05, 0.95, f"Spearman r={rho:.3f}, p={pval:.3g}", transform=ax.transAxes, va="top")
    ax.set_xlabel("Salivary slope"); ax.set_ylabel("Schirmer slope")
    fig.tight_layout(); fig.savefig(out / "salivary_eye_correlation.png", dpi=300); plt.close(fig)


def main(input_path: str, output_dir: str = "./tier1_output") -> None:
    """Run full Tier 1 longitudinal workflow and save outputs.

    Args:
        input_path: Path to input CSV/parquet dataset.
        output_dir: Output directory for tables and figures.
    """
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    path = Path(input_path)
    LOGGER.info("Loading data from %s", path)
    df = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)

    panel = build_tier1_panel(df)
    desc = describe_tier1(panel)
    pd.concat([desc["salivary"], desc["eye"]], ignore_index=True).to_csv(outdir / "tier1_descriptive.csv", index=False)

    panel = compute_derived_variables(panel)

    sal_outcomes = [
        "salivary_flow_form__flow_whole_unstim",
        "salivary_flow_form__tot_sim_sal_flow",
        "salivary_flow_form__tot_unsim_sal_flow",
        "salivary_flow_form__flow_p2l",
        "salivary_flow_form__flow_p2r",
        "salivary_flow_form__flow_sm1",
        "salivary_flow_form__flow_sm2",
    ]
    lmm_all: dict[str, Any] = {}
    tobit_all: dict[str, Any] = {}
    fixed_rows: list[pd.DataFrame] = []
    for outcome in sal_outcomes + FLOOR_VARS:
        if outcome in FLOOR_VARS:
            tobit_all[outcome] = run_tobit_salivary(panel, outcome)
            continue
        res = run_lmm_salivary(panel, outcome=outcome)
        lmm_all[outcome] = res
        if res is not None:
            fx = res["fixed_effects_summary"].copy()
            fx.insert(0, "outcome", outcome)
            fixed_rows.append(fx)

    gee_all: dict[tuple[str, str], Any] = {}
    for pair in BILATERAL_PAIRS:
        gee_all[pair] = run_gee_eye(panel, pair)

    primary_lmm = lmm_all.get("salivary_flow_form__flow_whole_unstim")
    if primary_lmm is None:
        LOGGER.warning("Primary salivary LMM unavailable; skipping clustering and plots.")
        return
    cluster_df = cluster_salivary_trajectories(primary_lmm["random_effects_df"], n_clusters=4)

    km_results = run_km_time_to_event(panel)

    sch_gee = gee_all.get(("eye_examination__sch_l", "eye_examination__sch_r")) or {}
    plot_tier1_results(panel, primary_lmm, sch_gee, cluster_df, km_results, output_dir=str(outdir / "figures"))

    if fixed_rows:
        pd.concat(fixed_rows, ignore_index=True).to_csv(outdir / "lmm_fixed_effects_all.csv", index=False)
    LOGGER.info("Pipeline complete. Outputs in %s", outdir)


if __name__ == "__main__":
    DEFAULT_INPUT = "data_analytic/visits_long_collapsed_by_interval_codebook_corrected.parquet"
    DEFAULT_OUTPUT = "./tier1_output"
    main(DEFAULT_INPUT, DEFAULT_OUTPUT)
