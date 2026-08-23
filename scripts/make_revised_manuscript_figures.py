#!/usr/bin/env python3
"""Create revised central and robustness figures for the Paper 2 manuscript."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


MERGED = Path("results/dnn_summaries/geometry_learning_merged.csv")
ROBUST = Path("results/robustness")
OUT = Path("build/figures")
GEOMETRY = Path("results/geometry")

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
GRAY = "#5A5A5A"
LIGHT = "#D9EAF4"

LABELS = {
    "multilayer_8": "8L",
    "multilayer_10": "10L",
    "multilayer_12": "12L",
    "multilayer_14": "14L",
    "multilayer_16": "16L",
    "multilayer_18": "18L",
    "multilayer_20": "20L",
    "2x2": "2×2",
    "grating": "Grating",
}

SHORT = {
    "Design dimension": "Design dim.",
    "PCA99 dimension": "PCA99",
    "Participation ratio": "Participation ratio",
    "Effective rank": "Effective rank",
    "TwoNN local ID": "TwoNN local ID",
    "PCA99 / TwoNN gap": "PCA99 / TwoNN",
    "Effective-rank / TwoNN gap": "E-rank / TwoNN",
}


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.6,
            "axes.labelsize": 10.5,
            "axes.titlesize": 11.2,
            "legend.fontsize": 8.4,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.20,
            "grid.linewidth": 0.7,
            "savefig.bbox": "tight",
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.png", dpi=360, facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", facecolor="white")
    plt.close(fig)


def regression_band(ax: plt.Axes, x: np.ndarray, y: np.ndarray, color: str) -> None:
    slope, intercept, _, _, _ = stats.linregress(x, y)
    xx = np.linspace(x.min(), x.max(), 250)
    yy = intercept + slope * xx
    fitted = intercept + slope * x
    resid = y - fitted
    dof = len(x) - 2
    se = np.sqrt(np.sum(resid**2) / dof)
    xbar = x.mean()
    sxx = np.sum((x - xbar) ** 2)
    tcrit = stats.t.ppf(0.975, dof)
    band = tcrit * se * np.sqrt(1 / len(x) + (xx - xbar) ** 2 / sxx)
    ax.fill_between(xx, yy - band, yy + band, color=color, alpha=0.13, linewidth=0)
    ax.plot(xx, yy, color=color, lw=1.7)


def annotate_cases(ax: plt.Axes, frame: pd.DataFrame, xcol: str) -> None:
    offsets = {
        "multilayer_8": (4, 4),
        "multilayer_10": (4, 6),
        "multilayer_12": (4, 5),
        "multilayer_14": (4, 4),
        "multilayer_16": (4, 3),
        "multilayer_18": (4, 2),
        "multilayer_20": (4, -10),
        "2x2": (-22, 4),
        "grating": (-42, -12),
    }
    for _, row in frame.iterrows():
        dx, dy = offsets[row["case"]]
        ax.annotate(
            LABELS[row["case"]],
            (row[xcol], row["learning_difficulty_1_minus_auc"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8.0,
            color="#303030",
        )


def geometry_figure(frame: pd.DataFrame) -> None:
    y = frame["learning_difficulty_1_minus_auc"].to_numpy(float)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.25), constrained_layout=True)

    ax = axes[0]
    x = frame["common_pca99"].to_numpy(float)
    regression_band(ax, x, y, BLUE)
    ax.scatter(x, y, s=45, color=BLUE, edgecolor="white", linewidth=0.7, zorder=3)
    annotate_cases(ax, frame, "common_pca99")
    pearson = stats.pearsonr(x, y)
    ax.text(
        0.04,
        0.95,
        f"Pearson r = {pearson.statistic:.3f}\np = {pearson.pvalue:.4f}\nLOOCV $Q^2$ = 0.580",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#C7D8E5", "alpha": 0.94},
    )
    ax.set_xlabel("Global spectral complexity (common-grid PCA99)")
    ax.set_ylabel("Learning difficulty (1 − log-sample AUC)")
    ax.set_title("(a) Global complexity predicts difficulty")

    ax = axes[1]
    x = frame["pca99_twonn_ratio"].to_numpy(float)
    order = np.argsort(x)
    ax.plot(x[order], y[order], color=ORANGE, lw=1.3, alpha=0.65, zorder=1)
    ax.scatter(x, y, s=45, color=ORANGE, edgecolor="white", linewidth=0.7, zorder=3)
    annotate_cases(ax, frame, "pca99_twonn_ratio")
    rho = stats.spearmanr(x, y).statistic
    ax.text(
        0.04,
        0.95,
        f"Spearman ρ = {rho:.3f}\nexact p = 4.96 × 10⁻⁵\nLOO ρ range: 0.976–1.000",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#EACDBF", "alpha": 0.94},
    )
    ax.set_xlabel("Global-to-local gap (PCA99 / TwoNN)")
    ax.set_ylabel("Learning difficulty (1 − log-sample AUC)")
    ax.set_title("(b) The gap stabilizes rank ordering")
    save(fig, "figure5_geometry_learning_robustness")


def robustness_figure(frame: pd.DataFrame) -> None:
    loocv = pd.read_csv(ROBUST / "loocv_prediction_summary.csv")
    loo = pd.read_csv(ROBUST / "leave_one_out_ranges.csv")
    all_corr = pd.read_csv(ROBUST / "subset_correlations.csv")
    all_corr = all_corr[all_corr["subset"] == "all_nine"][["predictor", "spearman_rho"]]
    loo = loo.merge(all_corr, on="predictor", how="left")

    order = loocv.sort_values("loocv_rmse", ascending=True)["predictor"].tolist()
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), constrained_layout=True)

    ax = axes[0]
    q = loocv.set_index("predictor").loc[order].reset_index()
    ypos = np.arange(len(q))
    colors = [BLUE if p == "PCA99 dimension" else ORANGE if "gap" in p else GRAY for p in q["predictor"]]
    ax.barh(ypos, q["loocv_rmse"], color=colors, alpha=0.90)
    ax.set_yticks(ypos, [SHORT[p] for p in q["predictor"]])
    ax.invert_yaxis()
    ax.set_xlabel("Leave-one-case-out RMSE (lower is better)")
    ax.set_title("(a) Quantitative out-of-sample prediction")
    ax.grid(axis="y", visible=False)
    for i, value in enumerate(q["loocv_rmse"]):
        ax.text(value + 0.0012, i, f"{value:.3f}", va="center", fontsize=8.0)
    ax.set_xlim(0, q["loocv_rmse"].max() * 1.18)

    ax = axes[1]
    r = loo.set_index("predictor").loc[order].reset_index()
    ypos = np.arange(len(r))
    for i, row in r.iterrows():
        color = BLUE if row["predictor"] == "PCA99 dimension" else ORANGE if "gap" in row["predictor"] else GRAY
        ax.plot([row["spearman_min"], row["spearman_max"]], [i, i], color=color, lw=2.4, solid_capstyle="round")
        ax.scatter(row["spearman_rho"], i, s=34, color=color, edgecolor="white", linewidth=0.6, zorder=3)
    ax.axvline(0, color="#999999", lw=0.8, ls="--")
    ax.set_yticks(ypos, [SHORT[p] for p in r["predictor"]])
    ax.invert_yaxis()
    ax.set_xlim(-0.2, 1.04)
    ax.set_xlabel("Spearman ρ (point: all cases; line: LOO range)")
    ax.set_title("(b) Sensitivity to any one architecture")
    ax.grid(axis="y", visible=False)
    save(fig, "figureS1_predictor_robustness")


def stability_figure() -> None:
    nid = pd.read_csv(GEOMETRY / "nonlinear_id_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.35), constrained_layout=False)
    ax = axes[0]
    cmap = plt.get_cmap("viridis")
    for j, layer in enumerate((8, 10, 12, 14, 16, 18, 20)):
        q = nid[nid["case"] == f"multilayer_{layer}"].sort_values("n_samples")
        ax.plot(q["n_samples"], q["twonn_mle_mean"], marker="o", lw=1.7,
                color=cmap(j / 6), label=f"{layer} layers")
    ax.set_xticks([1000, 2500, 5000], ["1,000", "2,500", "5,000"])
    ax.set_xlabel("Samples used")
    ax.set_ylabel("TwoNN intrinsic dimension")
    ax.set_title("(a) Multilayer sample-size stability")
    ax.legend(frameon=False, ncol=2, fontsize=8.2, loc="upper left")

    ax = axes[1]
    for case, label, color in [
        ("multilayer_20", "20-layer", BLUE),
        ("2x2", "2×2", ORANGE),
        ("grating", "Grating", GREEN),
    ]:
        q = nid[nid["case"] == case].sort_values("n_samples")
        ax.plot(q["n_samples"], q["twonn_mle_mean"], marker="o", lw=2.0, color=color, label=label)
        ax.fill_between(q["n_samples"], q["twonn_mle_ci_low"], q["twonn_mle_ci_high"],
                        color=color, alpha=0.14, linewidth=0)
    ax.set_xticks([1000, 2500, 5000], ["1,000", "2,500", "5,000"])
    ax.set_xlabel("Samples used")
    ax.set_ylabel("TwoNN intrinsic dimension")
    ax.set_title("(b) Cross-architecture stability")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout(pad=1.5)
    fig.subplots_adjust(bottom=0.22)
    save(fig, "figure3_intrinsic_dimension_stability_revised")


def main() -> None:
    global MERGED, ROBUST, GEOMETRY, OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged-csv", type=Path, default=MERGED)
    parser.add_argument("--robustness-dir", type=Path, default=ROBUST)
    parser.add_argument("--geometry-dir", type=Path, default=GEOMETRY)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    MERGED = args.merged_csv
    ROBUST = args.robustness_dir
    GEOMETRY = args.geometry_dir
    OUT = args.output_dir
    set_style()
    frame = pd.read_csv(MERGED)
    stability_figure()
    geometry_figure(frame)
    robustness_figure(frame)


if __name__ == "__main__":
    main()
