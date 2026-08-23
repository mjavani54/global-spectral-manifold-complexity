#!/usr/bin/env python3
"""Create publication-quality geometry figures from Paper 2 CSV results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "input": "#4C4C4C",
    "pca99": "#0072B2",
    "erank": "#009E73",
    "twonn": "#D55E00",
    "lb": "#CC79A7",
}


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.labelsize": 11.5,
        "axes.titlesize": 12,
        "legend.fontsize": 9.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.7,
        "savefig.bbox": "tight",
    })


def save(fig: plt.Figure, out: Path, stem: str) -> None:
    fig.savefig(out / f"{stem}.png", dpi=320, facecolor="white")
    fig.savefig(out / f"{stem}.pdf", facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    style()

    full = pd.read_csv(args.results_dir / "global_complexity_full.csv")
    nid = pd.read_csv(args.results_dir / "nonlinear_id_summary.csv")
    nid5 = nid[(nid["n_samples"] == 5000) & (~nid["smoothed"])].copy()
    merged = full.merge(nid5, on=["case", "family", "design_dim"])
    merged["pca99_twonn_ratio"] = merged["common_pca99"] / merged["twonn_mle_mean"]
    merged["erank_twonn_ratio"] = merged["common_effective_rank"] / merged["twonn_mle_mean"]

    # Figure 1: multilayer scaling and widening global/local gap.
    ml = merged[merged["family"] == "multilayer"].sort_values("design_dim")
    x = ml["design_dim"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.25), constrained_layout=True)
    ax = axes[0]
    ax.plot(x, ml["common_pca99"], "o-", lw=2.1, ms=5.5, color=COLORS["pca99"], label="Global PCA99")
    ax.plot(x, ml["common_effective_rank"], "s-", lw=2.1, ms=5.0, color=COLORS["erank"], label="Effective rank")
    ax.plot(x, ml["twonn_mle_mean"], "^-", lw=2.1, ms=5.5, color=COLORS["twonn"], label="Local TwoNN ID")
    ax.fill_between(x, ml["twonn_mle_ci_low"], ml["twonn_mle_ci_high"], color=COLORS["twonn"], alpha=0.16, linewidth=0)
    ax.plot(x, x, "--", lw=1.5, color=COLORS["input"], label="Design dimension")
    ax.set_xlabel("Number of independently variable layers")
    ax.set_ylabel("Dimension estimate")
    ax.set_title("(a) Global and local complexity scale differently")
    ax.set_xticks(x)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, loc="upper left")

    ax = axes[1]
    ax.plot(x, ml["pca99_twonn_ratio"], "o-", lw=2.2, ms=5.5, color=COLORS["pca99"], label="PCA99 / TwoNN")
    ax.plot(x, ml["erank_twonn_ratio"], "s-", lw=2.2, ms=5.0, color=COLORS["erank"], label="Effective rank / TwoNN")
    ax.axhline(1.0, ls="--", lw=1.2, color=COLORS["input"])
    ax.set_xlabel("Number of independently variable layers")
    ax.set_ylabel("Global-to-local dimension ratio")
    ax.set_title("(b) The global-to-local geometry gap widens")
    ax.set_xticks(x)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, loc="upper left")
    save(fig, args.output_dir, "figure1_multilayer_geometry_scaling")

    # Figure 2: controlled cross-architecture comparison.
    order = [
        "multilayer_8", "multilayer_10", "multilayer_12", "multilayer_14",
        "multilayer_16", "multilayer_18", "multilayer_20", "2x2", "grating",
    ]
    d = merged.set_index("case").loc[order].reset_index()
    labels = ["8L", "10L", "12L", "14L", "16L", "18L", "20L", "2×2", "Grating"]
    pos = np.arange(len(d))
    width = 0.24
    fig, ax = plt.subplots(figsize=(11.2, 4.7), constrained_layout=True)
    ax.bar(pos - width, d["design_dim"], width, color=COLORS["input"], label="Design dimension")
    ax.bar(pos, d["twonn_mle_mean"], width, color=COLORS["twonn"], label="Local TwoNN ID")
    ax.bar(pos + width, d["common_pca99"], width, color=COLORS["pca99"], label="Global PCA99")
    ax.errorbar(pos, d["twonn_mle_mean"],
                yerr=[d["twonn_mle_mean"] - d["twonn_mle_ci_low"], d["twonn_mle_ci_high"] - d["twonn_mle_mean"]],
                fmt="none", ecolor="black", elinewidth=0.9, capsize=2.5)
    ax.set_xticks(pos, labels)
    ax.set_ylabel("Dimension estimate")
    ax.set_xlabel("Representative nanophotonic architecture")
    ax.set_title("Resolution-controlled global and local spectral complexity")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.set_ylim(bottom=0)
    save(fig, args.output_dir, "figure2_cross_architecture_dimensions")

    # Figure 3: estimator stability with sample size.
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.4), constrained_layout=False)
    ax = axes[0]
    cmap = plt.get_cmap("viridis")
    for j, case in enumerate([f"multilayer_{v}" for v in (8, 10, 12, 14, 16, 18, 20)]):
        q = nid[nid["case"] == case].sort_values("n_samples")
        layer = int(case.split("_")[-1])
        ax.plot(q["n_samples"], q["twonn_mle_mean"], marker="o", lw=1.8,
                color=cmap(j / 6), label=f"{layer} layers")
    ax.set_xticks([1000, 2500, 5000], ["1,000", "2,500", "5,000"])
    ax.set_xlabel("Samples used")
    ax.set_ylabel("TwoNN intrinsic dimension")
    ax.set_title("(a) Multilayer sample-size stability")
    ax.legend(frameon=False, ncol=2, fontsize=8.5)

    ax = axes[1]
    for case, label, color in [
        ("multilayer_20", "20-layer", COLORS["pca99"]),
        ("2x2", "2×2", COLORS["twonn"]),
        ("grating", "Grating", COLORS["erank"]),
    ]:
        q = nid[nid["case"] == case].sort_values("n_samples")
        ax.plot(q["n_samples"], q["twonn_mle_mean"], marker="o", lw=2.1, color=color, label=label)
        ax.fill_between(q["n_samples"], q["twonn_mle_ci_low"], q["twonn_mle_ci_high"], color=color, alpha=0.14, linewidth=0)
    ax.set_xticks([1000, 2500, 5000], ["1,000", "2,500", "5,000"])
    ax.set_xlabel("Samples used")
    ax.set_ylabel("TwoNN intrinsic dimension")
    ax.set_title("(b) Representative architecture stability")
    ax.legend(frameon=False)
    fig.tight_layout(pad=1.6)
    save(fig, args.output_dir, "figure3_intrinsic_dimension_stability")


if __name__ == "__main__":
    main()
