#!/usr/bin/env python3
"""Resolution- and sample-controlled geometry analysis for Paper 2.

The script reads CSV members directly from the ZIP archive, preserving the source
files. It separates global linear/effective spectral complexity from local
nonlinear intrinsic-dimension estimators.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
import zipfile
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


CASES = [
    ("multilayer_8", "multilayer", 8, "Datasets/SiO2_TiO2_8_layer_50K_200_points_RS.csv"),
    ("multilayer_10", "multilayer", 10, "Datasets/SiO2_TiO2_10_layer_50K_200_points_RS.csv"),
    ("multilayer_12", "multilayer", 12, "Datasets/SiO2_TiO2_12_layer_50K_200_points_RS.csv"),
    ("multilayer_14", "multilayer", 14, "Datasets/SiO2_TiO2_14_layer_50K_200_points_RS.csv"),
    ("multilayer_16", "multilayer", 16, "Datasets/SiO2_TiO2_16_layer_50K_200_points_RS.csv"),
    ("multilayer_18", "multilayer", 18, "Datasets/SiO2_TiO2_18_layer_50K_200_points_RS.csv"),
    ("multilayer_20", "multilayer", 20, "Datasets/SiO2_TiO2_20_layer_200K_200_points_RS.csv"),
    ("2x2", "2x2", 9, "Datasets/2X2_RS.csv"),
    ("grating", "grating", 4, "Datasets/Grating_RS.csv"),
]


def load_csv(zf: zipfile.ZipFile, member: str) -> np.ndarray:
    with zf.open(member, "r") as f:
        return np.loadtxt(f, delimiter=",", dtype=np.float64, encoding="utf-8-sig")


def resample_normalized_axis(y: np.ndarray, output_dim: int) -> np.ndarray:
    """Linearly resample spectra along a normalized [0, 1] coordinate."""
    if y.shape[0] == output_dim:
        return y
    old = np.linspace(0.0, 1.0, y.shape[0])
    new = np.linspace(0.0, 1.0, output_dim)
    right = np.searchsorted(old, new, side="left")
    right = np.clip(right, 1, y.shape[0] - 1)
    left = right - 1
    alpha = ((new - old[left]) / (old[right] - old[left]))[:, None]
    return y[left, :] * (1.0 - alpha) + y[right, :] * alpha


def global_metrics(samples_by_features: np.ndarray) -> dict[str, float | int]:
    x = samples_by_features - samples_by_features.mean(axis=0, keepdims=True)
    cov = x.T @ x / (x.shape[0] - 1)
    eig = np.linalg.eigvalsh(cov)[::-1]
    eig = np.maximum(eig, 0.0)
    p = eig / eig.sum()
    nz = p[p > 0]
    cumulative = np.cumsum(p)
    return {
        "participation_ratio": float(1.0 / np.sum(p * p)),
        "effective_rank": float(np.exp(-np.sum(nz * np.log(nz)))),
        "pca90": int(np.searchsorted(cumulative, 0.90) + 1),
        "pca95": int(np.searchsorted(cumulative, 0.95) + 1),
        "pca99": int(np.searchsorted(cumulative, 0.99) + 1),
        "pca995": int(np.searchsorted(cumulative, 0.995) + 1),
        "pca999": int(np.searchsorted(cumulative, 0.999) + 1),
        "pc1_fraction": float(p[0]),
    }


def twonn(distances: np.ndarray, trim_quantile: float = 0.90) -> tuple[float, float, int]:
    r1 = distances[:, 1]
    r2 = distances[:, 2]
    good = (r1 > 0) & np.isfinite(r1) & np.isfinite(r2) & (r2 >= r1)
    mu = r2[good] / r1[good]
    log_mu = np.log(mu)
    positive = log_mu > 0
    mu = mu[positive]
    log_mu = log_mu[positive]
    mle = float(mu.size / np.sum(log_mu))

    order = np.argsort(mu)
    x = np.log(mu[order])
    f = (np.arange(1, mu.size + 1) - 0.5) / mu.size
    y = -np.log1p(-f)
    keep = f <= trim_quantile
    slope = float(np.dot(x[keep], y[keep]) / np.dot(x[keep], x[keep]))
    return mle, slope, int(mu.size)


def levina_bickel(distances: np.ndarray, k: int) -> tuple[float, float, float, float]:
    d = distances[:, 1 : k + 1]
    tk = d[:, -1]
    good = (d[:, 0] > 0) & np.isfinite(tk) & (tk > 0)
    d = d[good]
    tk = tk[good]
    denom = np.sum(np.log(tk[:, None] / d[:, :-1]), axis=1)
    local = (k - 2) / denom
    local = local[np.isfinite(local) & (local > 0)]
    arithmetic = float(np.mean(local))
    harmonic = float(1.0 / np.mean(1.0 / local))
    return arithmetic, harmonic, float(np.median(local)), float(np.subtract(*np.percentile(local, [75, 25])))


def local_pca_dimensions(
    x: np.ndarray,
    neighbor_indices: np.ndarray,
    rng: np.random.Generator,
    points: int,
) -> tuple[float, float, float, float]:
    chosen = rng.choice(x.shape[0], size=min(points, x.shape[0]), replace=False)
    dims95 = []
    dims99 = []
    for i in chosen:
        local = x[neighbor_indices[i, 1:]]
        local = local - local.mean(axis=0, keepdims=True)
        s = np.linalg.svd(local, full_matrices=False, compute_uv=False)
        variance = s * s
        cumulative = np.cumsum(variance) / variance.sum()
        dims95.append(int(np.searchsorted(cumulative, 0.95) + 1))
        dims99.append(int(np.searchsorted(cumulative, 0.99) + 1))
    return (
        float(np.median(dims95)),
        float(np.subtract(*np.percentile(dims95, [75, 25]))),
        float(np.median(dims99)),
        float(np.subtract(*np.percentile(dims99, [75, 25]))),
    )


def nonlinear_metrics(
    x: np.ndarray,
    rng: np.random.Generator,
    max_k: int,
    local_pca_neighbors: int,
    local_pca_points: int,
) -> dict[str, float | int]:
    # Orthogonal PCA projection removes only 0.01% of variance and accelerates
    # exact neighbor search while keeping the retained Euclidean geometry.
    pca = PCA(n_components=0.9999, svd_solver="full")
    projected = pca.fit_transform(x)
    k_required = max(max_k, local_pca_neighbors)
    nbrs = NearestNeighbors(n_neighbors=k_required + 1, algorithm="brute", n_jobs=-1)
    nbrs.fit(projected)
    distances, indices = nbrs.kneighbors(projected, return_distance=True)
    twonn_mle, twonn_slope, n_valid = twonn(distances)
    out: dict[str, float | int] = {
        "projection_components_9999": int(projected.shape[1]),
        "projection_variance": float(pca.explained_variance_ratio_.sum()),
        "twonn_mle": twonn_mle,
        "twonn_slope_trim90": twonn_slope,
        "twonn_valid_points": n_valid,
    }
    for k in (10, 20, max_k):
        if k > max_k:
            continue
        arithmetic, harmonic, median, iqr = levina_bickel(distances, k)
        out[f"lb{k}_mean"] = arithmetic
        out[f"lb{k}_harmonic"] = harmonic
        out[f"lb{k}_median"] = median
        out[f"lb{k}_iqr"] = iqr
    lp95, lp95_iqr, lp99, lp99_iqr = local_pca_dimensions(
        projected, indices[:, : local_pca_neighbors + 1], rng, local_pca_points
    )
    out["local_pca95_median"] = lp95
    out["local_pca95_iqr"] = lp95_iqr
    out["local_pca99_median"] = lp99
    out["local_pca99_iqr"] = lp99_iqr
    return out


def summarize(rows: list[dict], group_keys: list[str], value_keys: list[str]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        groups.setdefault(key, []).append(row)
    out = []
    for key, members in groups.items():
        record = dict(zip(group_keys, key))
        record["repeats"] = len(members)
        for value_key in value_keys:
            values = np.asarray([float(m[value_key]) for m in members], dtype=float)
            record[f"{value_key}_mean"] = float(values.mean())
            record[f"{value_key}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            record[f"{value_key}_ci_low"] = float(np.percentile(values, 2.5))
            record[f"{value_key}_ci_high"] = float(np.percentile(values, 97.5))
        out.append(record)
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def synthetic_self_test() -> dict[str, float]:
    rng = np.random.default_rng(909)
    x = rng.normal(size=(2500, 5))
    nbrs = NearestNeighbors(n_neighbors=31, algorithm="brute", n_jobs=-1).fit(x)
    distances, _ = nbrs.kneighbors(x)
    t_mle, t_slope, _ = twonn(distances)
    _, lb20, _, _ = levina_bickel(distances, 20)
    return {"true_dimension": 5.0, "twonn_mle": t_mle, "twonn_slope": t_slope, "lb20_harmonic": lb20}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets-zip", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--common-dim", type=int, default=200)
    ap.add_argument("--sample-sizes", type=int, nargs="+", default=[1000, 2500, 5000])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--global-bootstrap", type=int, default=30)
    ap.add_argument("--max-k", type=int, default=30)
    ap.add_argument("--local-pca-neighbors", type=int, default=100)
    ap.add_argument("--local-pca-points", type=int, default=200)
    ap.add_argument("--base-seed", type=int, default=240819)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    full_rows: list[dict] = []
    global_boot_rows: list[dict] = []
    nonlinear_rows: list[dict] = []
    smoothing_rows: list[dict] = []

    self_test = synthetic_self_test()
    print(f"Synthetic estimator self-test: {self_test}", flush=True)

    with zipfile.ZipFile(args.datasets_zip) as zf:
        for case_index, (case, family, design_dim, member) in enumerate(CASES):
            print(f"[{case_index + 1}/{len(CASES)}] Loading {case}", flush=True)
            y_native = load_csv(zf, member)
            y_common = resample_normalized_axis(y_native, args.common_dim)
            native_metrics = global_metrics(y_native.T)
            common_metrics = global_metrics(y_common.T)
            full_rows.append({
                "case": case,
                "family": family,
                "design_dim": design_dim,
                "n_total": int(y_native.shape[1]),
                "native_response_dim": int(y_native.shape[0]),
                "common_response_dim": args.common_dim,
                **{f"native_{k}": v for k, v in native_metrics.items()},
                **{f"common_{k}": v for k, v in common_metrics.items()},
            })

            # Equal-N global-complexity sensitivity.
            n_boot = min(5000, y_common.shape[1])
            for repeat in range(args.global_bootstrap):
                rng = np.random.default_rng(args.base_seed + case_index * 10_000 + repeat)
                idx = rng.choice(y_common.shape[1], size=n_boot, replace=False)
                metrics = global_metrics(y_common[:, idx].T)
                global_boot_rows.append({
                    "case": case,
                    "family": family,
                    "design_dim": design_dim,
                    "n_samples": n_boot,
                    "repeat": repeat,
                    **metrics,
                })

            # Local/nonlinear estimates at controlled sample sizes and seeds.
            for n_samples in args.sample_sizes:
                if n_samples > y_common.shape[1]:
                    continue
                for repeat in range(args.repeats):
                    rng = np.random.default_rng(
                        args.base_seed + case_index * 100_000 + n_samples * 10 + repeat
                    )
                    idx = rng.choice(y_common.shape[1], size=n_samples, replace=False)
                    x = y_common[:, idx].T
                    metrics = nonlinear_metrics(
                        x,
                        rng,
                        max_k=args.max_k,
                        local_pca_neighbors=min(args.local_pca_neighbors, n_samples - 1),
                        local_pca_points=args.local_pca_points,
                    )
                    nonlinear_rows.append({
                        "case": case,
                        "family": family,
                        "design_dim": design_dim,
                        "n_samples": n_samples,
                        "repeat": repeat,
                        "smoothed": False,
                        **metrics,
                    })

            # Quantify sensitivity to a deliberately mild smoothing operation.
            n_smooth = min(5000, y_common.shape[1])
            rng = np.random.default_rng(args.base_seed + case_index * 1_000_000 + 777)
            idx = rng.choice(y_common.shape[1], size=n_smooth, replace=False)
            raw = y_common[:, idx].T
            smooth = savgol_filter(raw, window_length=7, polyorder=3, axis=1, mode="interp")
            for label, x in ((False, raw), (True, smooth)):
                metrics = nonlinear_metrics(
                    x,
                    rng,
                    max_k=args.max_k,
                    local_pca_neighbors=min(args.local_pca_neighbors, n_smooth - 1),
                    local_pca_points=args.local_pca_points,
                )
                smoothing_rows.append({
                    "case": case,
                    "family": family,
                    "design_dim": design_dim,
                    "n_samples": n_smooth,
                    "smoothed": label,
                    **metrics,
                })
            del y_native, y_common

    global_value_keys = [
        "participation_ratio", "effective_rank", "pca90", "pca95", "pca99", "pca995", "pca999", "pc1_fraction"
    ]
    nonlinear_value_keys = [
        "projection_components_9999", "twonn_mle", "twonn_slope_trim90",
        "lb10_harmonic", "lb20_harmonic", "lb30_harmonic",
        "local_pca95_median", "local_pca99_median",
    ]
    global_summary = summarize(global_boot_rows, ["case", "family", "design_dim", "n_samples"], global_value_keys)
    nonlinear_summary = summarize(nonlinear_rows, ["case", "family", "design_dim", "n_samples", "smoothed"], nonlinear_value_keys)

    write_csv(args.output_dir / "global_complexity_full.csv", full_rows)
    write_csv(args.output_dir / "global_complexity_equal_n_repeats.csv", global_boot_rows)
    write_csv(args.output_dir / "global_complexity_equal_n_summary.csv", global_summary)
    write_csv(args.output_dir / "nonlinear_id_repeats.csv", nonlinear_rows)
    write_csv(args.output_dir / "nonlinear_id_summary.csv", nonlinear_summary)
    write_csv(args.output_dir / "smoothing_sensitivity.csv", smoothing_rows)

    metadata = {
        "method_note": "Cross-family spectra are resampled along a normalized [0,1] coordinate only to control grid resolution; physical spectral ranges differ and are not aligned.",
        "common_response_dim": args.common_dim,
        "sample_sizes": args.sample_sizes,
        "repeats": args.repeats,
        "global_bootstrap": args.global_bootstrap,
        "max_k": args.max_k,
        "local_pca_neighbors": args.local_pca_neighbors,
        "local_pca_points": args.local_pca_points,
        "base_seed": args.base_seed,
        "synthetic_self_test": self_test,
        "runtime_seconds": time.time() - start,
    }
    (args.output_dir / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
