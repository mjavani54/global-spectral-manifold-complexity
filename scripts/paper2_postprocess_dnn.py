#!/usr/bin/env python3
"""Summarize Paper 2 DNN experiments and relate them to geometry metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


CASE_ORDER = [
    "multilayer_8", "multilayer_10", "multilayer_12", "multilayer_14",
    "multilayer_16", "multilayer_18", "multilayer_20", "2x2", "grating",
]


def mean_summary(data: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    metrics = [
        "test_r2_global", "test_r2_spectral_mean", "test_rmse", "test_nrmse_std",
        "test_mae", "test_spectral_angle_deg", "test_sample_max_abs_error_p95",
        "train_seconds", "best_epoch", "parameter_count",
    ]
    grouped = data.groupby(keys, as_index=False)
    mean = grouped[metrics].mean().rename(columns={m: f"{m}_mean" for m in metrics})
    sd = grouped[metrics].std().fillna(0).rename(columns={m: f"{m}_sd" for m in metrics})
    count = grouped.size().rename(columns={"size": "seeds_completed"})
    return mean.merge(sd, on=keys).merge(count, on=keys)


def learning_difficulty(summary: pd.DataFrame, target: float) -> pd.DataFrame:
    rows = []
    for case, group in summary.groupby("case"):
        g = group.sort_values("train_size")
        resolved = g[g["test_r2_global_mean"] >= target]
        threshold = int(resolved.iloc[0]["train_size"]) if not resolved.empty else np.nan
        status = "resolved" if not resolved.empty else f">{int(g['train_size'].max())}"
        x = np.log(g["train_size"].to_numpy(float))
        y = g["test_r2_global_mean"].to_numpy(float)
        if len(g) > 1:
            aulc = float(np.trapezoid(y, x) / (x[-1] - x[0]))
        else:
            aulc = float(y[0])
        rows.append({
            "case": case,
            "target_r2": target,
            "sample_threshold": threshold,
            "sample_threshold_status": status,
            "max_test_r2": float(y.max()),
            "learning_curve_auc_log_n": aulc,
            "learning_difficulty_1_minus_auc": 1.0 - aulc,
        })
    return pd.DataFrame(rows)


def capacity_thresholds(summary: pd.DataFrame, target: float) -> pd.DataFrame:
    rows = []
    for case, group in summary.groupby("case"):
        g = group.sort_values("parameter_count_mean")
        resolved = g[g["test_r2_global_mean"] >= target]
        threshold = float(resolved.iloc[0]["parameter_count_mean"]) if not resolved.empty else np.nan
        rows.append({
            "case": case,
            "target_r2": target,
            "parameter_threshold": threshold,
            "parameter_threshold_status": "resolved" if not resolved.empty else f">{int(g['parameter_count_mean'].max())}",
        })
    return pd.DataFrame(rows)


def correlations(data: pd.DataFrame, outcome: str, predictors: list[str]) -> list[dict]:
    rows = []
    for predictor in predictors:
        clean = data[[predictor, outcome]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(clean) < 4 or clean[predictor].nunique() < 2 or clean[outcome].nunique() < 2:
            continue
        pr, pp = pearsonr(clean[predictor], clean[outcome])
        sr, sp = spearmanr(clean[predictor], clean[outcome])
        rows.append({
            "outcome": outcome,
            "predictor": predictor,
            "n_cases": len(clean),
            "pearson_r": float(pr),
            "pearson_p": float(pp),
            "spearman_rho": float(sr),
            "spearman_p": float(sp),
        })
    return rows


def make_figures(learning: pd.DataFrame, capacity: pd.DataFrame, merged: pd.DataFrame, out: Path) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10.5, "axes.labelsize": 11.5,
        "axes.titlesize": 12, "legend.fontsize": 8.5, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.22,
        "savefig.bbox": "tight",
    })
    colors = plt.get_cmap("tab10")
    labels = {f"multilayer_{v}": f"{v}L" for v in (8, 10, 12, 14, 16, 18, 20)} | {"2x2": "2×2", "grating": "Grating"}

    fig, ax = plt.subplots(figsize=(8.2, 5.1))
    for i, case in enumerate(CASE_ORDER):
        g = learning[learning["case"] == case].sort_values("train_size")
        if g.empty:
            continue
        ax.errorbar(g["train_size"], g["test_r2_global_mean"], yerr=g["test_r2_global_sd"],
                    marker="o", lw=1.8, capsize=2.5, color=colors(i % 10), label=labels[case])
    ax.set_xscale("log")
    ax.set_xlabel("Training samples")
    ax.set_ylabel("Test $R^2$")
    ax.set_title("Controlled cross-architecture learning curves")
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(out / "figure4_dnn_learning_curves.png", dpi=320, facecolor="white")
    fig.savefig(out / "figure4_dnn_learning_curves.pdf", facecolor="white")
    plt.close(fig)

    if not capacity.empty:
        fig, ax = plt.subplots(figsize=(8.2, 5.1))
        for i, case in enumerate(CASE_ORDER):
            g = capacity[capacity["case"] == case].sort_values("parameter_count_mean")
            if g.empty:
                continue
            ax.errorbar(g["parameter_count_mean"], g["test_r2_global_mean"], yerr=g["test_r2_global_sd"],
                        marker="o", lw=1.8, capsize=2.5, color=colors(i % 10), label=labels[case])
        ax.set_xscale("log")
        ax.set_xlabel("Trainable parameters")
        ax.set_ylabel("Test $R^2$")
        ax.set_title("Controlled cross-architecture capacity curves")
        ax.legend(frameon=False, ncol=3)
        fig.tight_layout()
        fig.savefig(out / "figure6_dnn_capacity_curves.png", dpi=320, facecolor="white")
        fig.savefig(out / "figure6_dnn_capacity_curves.pdf", facecolor="white")
        plt.close(fig)

    if not merged.empty:
        fig, ax = plt.subplots(figsize=(7.1, 5.2))
        ax.scatter(merged["pca99_twonn_ratio"], merged["learning_difficulty_1_minus_auc"], s=58, color="#0072B2")
        for _, row in merged.iterrows():
            ax.annotate(labels.get(row["case"], row["case"]),
                        (row["pca99_twonn_ratio"], row["learning_difficulty_1_minus_auc"]),
                        xytext=(4, 4), textcoords="offset points", fontsize=8.5)
        ax.set_xlabel("Global-to-local dimension ratio (PCA99 / TwoNN)")
        ax.set_ylabel("Learning difficulty (1 − log-sample AUC)")
        ax.set_title("Does manifold folding predict surrogate-learning burden?")
        fig.tight_layout()
        fig.savefig(out / "diagnostic_geometry_vs_learning_difficulty.png", dpi=320, facecolor="white")
        fig.savefig(out / "diagnostic_geometry_vs_learning_difficulty.pdf", facecolor="white")
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dnn-results", type=Path, nargs="+", required=True)
    ap.add_argument("--geometry-results-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--target-r2", type=float, default=0.99)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.concat([pd.read_csv(p) for p in args.dnn_results], ignore_index=True)
    key = ["case", "mode", "seed", "train_size", "width", "hidden_layers"]
    raw = raw.drop_duplicates(key, keep="last")
    learning_raw = raw[raw["mode"] == "learning"].copy()
    capacity_raw = raw[raw["mode"] == "capacity"].copy()
    learning = mean_summary(learning_raw, ["case", "train_size", "width", "hidden_layers"])
    capacity = mean_summary(capacity_raw, ["case", "train_size", "width", "hidden_layers"]) if not capacity_raw.empty else pd.DataFrame()
    difficulty = learning_difficulty(learning, args.target_r2)
    capacity_t = capacity_thresholds(capacity, args.target_r2) if not capacity.empty else pd.DataFrame()

    full = pd.read_csv(args.geometry_results_dir / "global_complexity_full.csv")
    nid = pd.read_csv(args.geometry_results_dir / "nonlinear_id_summary.csv")
    nid = nid[(nid["n_samples"] == 5000) & (~nid["smoothed"])].copy()
    geometry = full.merge(nid, on=["case", "family", "design_dim"])
    geometry["pca99_twonn_ratio"] = geometry["common_pca99"] / geometry["twonn_mle_mean"]
    geometry["erank_twonn_ratio"] = geometry["common_effective_rank"] / geometry["twonn_mle_mean"]
    merged = geometry.merge(difficulty, on="case", how="inner")
    if not capacity_t.empty:
        merged = merged.merge(capacity_t, on=["case", "target_r2"], how="left")

    predictors = [
        "design_dim", "common_participation_ratio", "common_effective_rank", "common_pca99",
        "twonn_mle_mean", "lb20_harmonic_mean", "pca99_twonn_ratio", "erank_twonn_ratio",
    ]
    corr = correlations(merged, "learning_difficulty_1_minus_auc", predictors)
    corr += correlations(merged, "sample_threshold", predictors)
    if "parameter_threshold" in merged:
        corr += correlations(merged, "parameter_threshold", predictors)

    learning.to_csv(args.output_dir / "learning_curve_summary.csv", index=False)
    if not capacity.empty:
        capacity.to_csv(args.output_dir / "capacity_curve_summary.csv", index=False)
    difficulty.to_csv(args.output_dir / "learning_difficulty_summary.csv", index=False)
    if not capacity_t.empty:
        capacity_t.to_csv(args.output_dir / "capacity_threshold_summary.csv", index=False)
    merged.to_csv(args.output_dir / "geometry_learning_merged.csv", index=False)
    pd.DataFrame(corr).to_csv(args.output_dir / "geometry_learning_correlations.csv", index=False)
    make_figures(learning, capacity, merged, args.output_dir)
    print(json.dumps({"models": len(raw), "cases": sorted(raw["case"].unique()), "correlations": corr}, indent=2))


if __name__ == "__main__":
    main()
