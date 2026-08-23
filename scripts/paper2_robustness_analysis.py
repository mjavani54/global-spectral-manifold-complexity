#!/usr/bin/env python3
"""Robustness checks for Paper 2 geometry-learning claims."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


OUTCOME = "learning_difficulty_1_minus_auc"
PREDICTORS = {
    "Design dimension": "design_dim",
    "PCA99 dimension": "common_pca99",
    "Participation ratio": "common_participation_ratio",
    "Effective rank": "common_effective_rank",
    "TwoNN local ID": "twonn_mle_mean",
    "PCA99 / TwoNN gap": "pca99_twonn_ratio",
    "Effective-rank / TwoNN gap": "erank_twonn_ratio",
}


def corr_table(frame: pd.DataFrame, subset: str) -> list[dict]:
    rows: list[dict] = []
    y = frame[OUTCOME].to_numpy(float)
    for label, column in PREDICTORS.items():
        x = frame[column].to_numpy(float)
        pearson = stats.pearsonr(x, y)
        spearman = stats.spearmanr(x, y)
        rows.append(
            {
                "subset": subset,
                "n": len(frame),
                "predictor": label,
                "column": column,
                "pearson_r": pearson.statistic,
                "pearson_p": pearson.pvalue,
                "spearman_rho": spearman.statistic,
                "spearman_p_asymptotic": spearman.pvalue,
            }
        )
    return rows


def exact_spearman_pvalues(frame: pd.DataFrame) -> pd.DataFrame:
    """Exact two-sided permutation p-values for n=9 Spearman correlations."""
    y_rank = stats.rankdata(frame[OUTCOME].to_numpy(float))
    y_center = y_rank - y_rank.mean()
    denom_y = np.sqrt(np.sum(y_center**2))
    permuted = np.asarray(list(itertools.permutations(y_center)), dtype=np.float32)
    rows = []
    for label, column in PREDICTORS.items():
        x_rank = stats.rankdata(frame[column].to_numpy(float))
        x_center = x_rank - x_rank.mean()
        denom_x = np.sqrt(np.sum(x_center**2))
        obs = float(np.dot(x_center, y_center) / (denom_x * denom_y))
        perm_corr = permuted @ x_center / (denom_x * denom_y)
        p_exact = float(np.mean(np.abs(perm_corr) >= abs(obs) - 1e-12))
        rows.append(
            {
                "predictor": label,
                "column": column,
                "spearman_rho": obs,
                "exact_two_sided_p": p_exact,
            }
        )
    exact = pd.DataFrame(rows).sort_values("exact_two_sided_p").reset_index(drop=True)
    # Benjamini-Hochberg correction across the seven prespecified comparisons.
    m = len(exact)
    adjusted = exact["exact_two_sided_p"].to_numpy() * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    exact["bh_q"] = np.minimum(adjusted, 1.0)
    return exact


def partial_correlations(frame: pd.DataFrame) -> pd.DataFrame:
    """Partial Pearson and rank correlations controlling design dimension."""
    z = frame["design_dim"].to_numpy(float)
    z_design = np.column_stack([np.ones(len(frame)), z])
    z_rank = stats.rankdata(z)
    z_rank_design = np.column_stack([np.ones(len(frame)), z_rank])
    y = frame[OUTCOME].to_numpy(float)
    y_resid = y - z_design @ np.linalg.lstsq(z_design, y, rcond=None)[0]
    y_rank = stats.rankdata(y)
    y_rank_resid = y_rank - z_rank_design @ np.linalg.lstsq(
        z_rank_design, y_rank, rcond=None
    )[0]
    rows = []
    for label, column in PREDICTORS.items():
        if column == "design_dim":
            continue
        x = frame[column].to_numpy(float)
        x_resid = x - z_design @ np.linalg.lstsq(z_design, x, rcond=None)[0]
        x_rank = stats.rankdata(x)
        x_rank_resid = x_rank - z_rank_design @ np.linalg.lstsq(
            z_rank_design, x_rank, rcond=None
        )[0]
        pearson = stats.pearsonr(x_resid, y_resid)
        rank = stats.pearsonr(x_rank_resid, y_rank_resid)
        rows.append(
            {
                "predictor": label,
                "partial_pearson_r": pearson.statistic,
                "partial_pearson_p": pearson.pvalue,
                "partial_rank_r": rank.statistic,
                "partial_rank_p": rank.pvalue,
            }
        )
    return pd.DataFrame(rows)


def leave_one_out_correlations(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dropped in frame["case"]:
        sub = frame[frame["case"] != dropped]
        y = sub[OUTCOME].to_numpy(float)
        for label, column in PREDICTORS.items():
            x = sub[column].to_numpy(float)
            rows.append(
                {
                    "dropped_case": dropped,
                    "predictor": label,
                    "pearson_r": stats.pearsonr(x, y).statistic,
                    "spearman_rho": stats.spearmanr(x, y).statistic,
                }
            )
    return pd.DataFrame(rows)


def loocv_prediction(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One-predictor linear models fitted on eight cases and tested on the ninth."""
    pred_rows = []
    y_all = frame[OUTCOME].to_numpy(float)
    for label, column in PREDICTORS.items():
        predictions = np.empty(len(frame), dtype=float)
        for i in range(len(frame)):
            train = np.arange(len(frame)) != i
            x_train = frame.loc[train, column].to_numpy(float)
            y_train = y_all[train]
            design = np.column_stack([np.ones(train.sum()), x_train])
            beta = np.linalg.lstsq(design, y_train, rcond=None)[0]
            predictions[i] = beta[0] + beta[1] * float(frame.loc[i, column])
            pred_rows.append(
                {
                    "case": frame.loc[i, "case"],
                    "predictor": label,
                    "observed": y_all[i],
                    "predicted": predictions[i],
                    "error": predictions[i] - y_all[i],
                }
            )
        residuals = predictions - y_all
        sse = float(np.sum(residuals**2))
        sst = float(np.sum((y_all - y_all.mean()) ** 2))
        yield_row = {
            "predictor": label,
            "loocv_mae": float(np.mean(np.abs(residuals))),
            "loocv_rmse": float(np.sqrt(np.mean(residuals**2))),
            "loocv_q2": 1.0 - sse / sst,
        }
        if "summary_rows" not in locals():
            summary_rows = []
        summary_rows.append(yield_row)
    return pd.DataFrame(summary_rows).sort_values("loocv_rmse"), pd.DataFrame(pred_rows)


def bootstrap_spearman(
    frame: pd.DataFrame,
    column: str,
    seed: int = 20260819,
    replicates: int = 50_000,
) -> dict:
    rng = np.random.default_rng(seed)
    x = frame[column].to_numpy(float)
    y = frame[OUTCOME].to_numpy(float)
    values = []
    for _ in range(replicates):
        idx = rng.integers(0, len(frame), size=len(frame))
        if np.unique(x[idx]).size < 2 or np.unique(y[idx]).size < 2:
            continue
        values.append(stats.spearmanr(x[idx], y[idx]).statistic)
    arr = np.asarray(values)
    return {
        "predictor": column,
        "replicates_retained": int(len(arr)),
        "rho_median": float(np.median(arr)),
        "rho_ci95_low": float(np.quantile(arr, 0.025)),
        "rho_ci95_high": float(np.quantile(arr, 0.975)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/dnn_summaries/geometry_learning_merged.csv"),
        help="Merged nine-case geometry/learning CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("build/robustness"),
        help="Directory for robustness tables.",
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=50_000,
        help="Number of case-resampling bootstrap draws per ratio.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.input).reset_index(drop=True)

    subsets = {
        "all_nine": frame,
        "multilayers_only": frame[frame["family"] == "multilayer"],
        "exclude_2x2": frame[frame["case"] != "2x2"],
        "exclude_grating": frame[frame["case"] != "grating"],
        "exclude_both_cross_family": frame[frame["family"] == "multilayer"],
    }
    correlations = pd.DataFrame(
        [row for name, sub in subsets.items() for row in corr_table(sub, name)]
    )
    correlations.to_csv(args.output_dir / "subset_correlations.csv", index=False)

    exact = exact_spearman_pvalues(frame)
    exact.to_csv(args.output_dir / "exact_spearman_tests.csv", index=False)

    partial = partial_correlations(frame)
    partial.to_csv(args.output_dir / "partial_correlations_design_dim.csv", index=False)

    loo = leave_one_out_correlations(frame)
    loo.to_csv(args.output_dir / "leave_one_out_correlations.csv", index=False)
    loo_summary = (
        loo.groupby("predictor")
        .agg(
            pearson_min=("pearson_r", "min"),
            pearson_max=("pearson_r", "max"),
            spearman_min=("spearman_rho", "min"),
            spearman_max=("spearman_rho", "max"),
        )
        .reset_index()
    )
    loo_summary.to_csv(args.output_dir / "leave_one_out_ranges.csv", index=False)

    loocv, predictions = loocv_prediction(frame)
    loocv.to_csv(args.output_dir / "loocv_prediction_summary.csv", index=False)
    predictions.to_csv(args.output_dir / "loocv_predictions.csv", index=False)

    bootstrap = {
        "pca99_twonn": bootstrap_spearman(
            frame, "pca99_twonn_ratio", replicates=args.bootstrap_replicates
        ),
        "erank_twonn": bootstrap_spearman(
            frame,
            "erank_twonn_ratio",
            seed=20260820,
            replicates=args.bootstrap_replicates,
        ),
    }
    (args.output_dir / "bootstrap_spearman.json").write_text(
        json.dumps(bootstrap, indent=2) + "\n", encoding="utf-8"
    )

    print("Exact Spearman tests")
    print(exact.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    print("\nLeave-one-out ranges")
    print(loo_summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nLOOCV prediction")
    print(loocv.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\nPartial correlations controlling design dimension")
    print(partial.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nBootstrap")
    print(json.dumps(bootstrap, indent=2))


if __name__ == "__main__":
    main()
