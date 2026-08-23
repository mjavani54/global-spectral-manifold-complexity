#!/usr/bin/env python3
"""Read-only scientific audit of the Paper 2 dataset and master-output archives."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import zipfile
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


CASES = [
    ("multilayer_8", "Datasets/SiO2_TiO2_8_layer_50K_200_points_DS.csv", "Datasets/SiO2_TiO2_8_layer_50K_200_points_RS.csv", 8, 200, 50_000),
    ("multilayer_10", "Datasets/SiO2_TiO2_10_layer_50K_200_points_DS.csv", "Datasets/SiO2_TiO2_10_layer_50K_200_points_RS.csv", 10, 200, 50_000),
    ("multilayer_12", "Datasets/SiO2_TiO2_12_layer_50K_200_points_DS.csv", "Datasets/SiO2_TiO2_12_layer_50K_200_points_RS.csv", 12, 200, 50_000),
    ("multilayer_14", "Datasets/SiO2_TiO2_14_layer_50K_200_points_DS.csv", "Datasets/SiO2_TiO2_14_layer_50K_200_points_RS.csv", 14, 200, 50_000),
    ("multilayer_16", "Datasets/SiO2_TiO2_16_layer_50K_200_points_DS.csv", "Datasets/SiO2_TiO2_16_layer_50K_200_points_RS.csv", 16, 200, 50_000),
    ("multilayer_18", "Datasets/SiO2_TiO2_18_layer_50K_200_points_DS.csv", "Datasets/SiO2_TiO2_18_layer_50K_200_points_RS.csv", 18, 200, 50_000),
    ("multilayer_20", "Datasets/SiO2_TiO2_20_layer_200K_200_points_DS.csv", "Datasets/SiO2_TiO2_20_layer_200K_200_points_RS.csv", 20, 200, 200_000),
    ("2x2", "Datasets/2X2_DS.csv", "Datasets/2X2_RS.csv", 9, 551, 6_000),
    ("grating", "Datasets/Grating_DS.csv", "Datasets/Grating_RS.csv", 4, 400, 12_000),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_csv(zf: zipfile.ZipFile, member: str) -> np.ndarray:
    with zf.open(member, "r") as f:
        return np.loadtxt(f, delimiter=",", dtype=np.float64, encoding="utf-8-sig")


def covariance_metrics(y: np.ndarray) -> dict[str, float | int]:
    centered = y - y.mean(axis=1, keepdims=True)
    cov = centered @ centered.T / (y.shape[1] - 1)
    eig = np.linalg.eigvalsh(cov)[::-1]
    eig = np.maximum(eig, 0.0)
    total = float(eig.sum())
    p = eig / total
    c = np.cumsum(p)
    nonzero = p[p > 0]
    return {
        "participation_ratio": float(1.0 / np.sum(p * p)),
        "effective_rank": float(np.exp(-np.sum(nonzero * np.log(nonzero)))),
        "pca90": int(np.searchsorted(c, 0.90) + 1),
        "pca95": int(np.searchsorted(c, 0.95) + 1),
        "pca99": int(np.searchsorted(c, 0.99) + 1),
        "pca995": int(np.searchsorted(c, 0.995) + 1),
        "pca999": int(np.searchsorted(c, 0.999) + 1),
        "pc1_fraction": float(p[0]),
    }


def sampled_duplicate_count(a: np.ndarray, max_samples: int = 20_000) -> int:
    n = a.shape[1]
    if n > max_samples:
        idx = np.linspace(0, n - 1, max_samples, dtype=int)
        sample = np.ascontiguousarray(a[:, idx].T)
    else:
        sample = np.ascontiguousarray(a.T)
    return int(sample.shape[0] - np.unique(sample, axis=0).shape[0])


def pairing_sanity(x: np.ndarray, y: np.ndarray, seed: int = 1729) -> dict[str, float | int]:
    """Test whether design/response pairing contains learnable signal.

    Responses are compressed to PCA scores. ExtraTrees is deliberately used only
    as a mismatch detector, not as a publishable surrogate benchmark.
    """
    rng = np.random.default_rng(seed)
    n = x.shape[1]
    idx = rng.choice(n, size=min(n, 6_000), replace=False)
    xx = x[:, idx].T
    yy = y[:, idx].T
    n_components = min(20, yy.shape[1], yy.shape[0] - 1)
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
    scores = pca.fit_transform(yy)
    x_train, x_test, y_train, y_test = train_test_split(
        xx, scores, test_size=0.2, random_state=seed
    )
    scaler = StandardScaler().fit(x_train)
    model = ExtraTreesRegressor(
        n_estimators=160,
        min_samples_leaf=2,
        max_features=1.0,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(scaler.transform(x_train), y_train)
    paired = float(r2_score(y_test, model.predict(scaler.transform(x_test)), multioutput="variance_weighted"))

    shuffled = scores[rng.permutation(scores.shape[0])]
    xs_train, xs_test, ys_train, ys_test = train_test_split(
        xx, shuffled, test_size=0.2, random_state=seed
    )
    scaler_s = StandardScaler().fit(xs_train)
    model_s = ExtraTreesRegressor(
        n_estimators=160,
        min_samples_leaf=2,
        max_features=1.0,
        random_state=seed,
        n_jobs=-1,
    )
    model_s.fit(scaler_s.transform(xs_train), ys_train)
    permuted = float(r2_score(ys_test, model_s.predict(scaler_s.transform(xs_test)), multioutput="variance_weighted"))
    return {
        "n_used": int(idx.size),
        "pca_components": int(n_components),
        "pca_explained_fraction": float(pca.explained_variance_ratio_.sum()),
        "paired_test_r2": paired,
        "permuted_test_r2": permuted,
        "pairing_signal_gap": paired - permuted,
    }


def compare_master(computed: dict, master_row: dict) -> dict[str, float]:
    keys = ["participation_ratio", "effective_rank", "pca90", "pca95", "pca99", "pca995", "pca999", "pc1_fraction"]
    return {k: float(computed[k]) - float(master_row[k]) for k in keys}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets-zip", type=Path, required=True)
    ap.add_argument("--master-zip", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(args.master_zip) as mz:
        with mz.open("master_analysis_output/intrinsic_dimension/id_summary.csv") as f:
            text = (line.decode("utf-8") for line in f)
            master_rows = {row["case"]: row for row in csv.DictReader(text)}

    archive_meta = {
        "datasets_zip": str(args.datasets_zip),
        "datasets_sha256": sha256(args.datasets_zip),
        "master_zip": str(args.master_zip),
        "master_sha256": sha256(args.master_zip),
    }
    results = []
    pairing = {}
    with zipfile.ZipFile(args.datasets_zip) as zf:
        bad_crc = zf.testzip()
        archive_meta["zip_crc_failure"] = bad_crc
        for case, ds_name, rs_name, d_expected, r_expected, n_expected in CASES:
            print(f"Auditing {case}", flush=True)
            x = load_csv(zf, ds_name)
            y = load_csv(zf, rs_name)
            metrics = covariance_metrics(y)
            row = {
                "case": case,
                "design_rows": int(x.shape[0]),
                "design_columns": int(x.shape[1]),
                "response_rows": int(y.shape[0]),
                "response_columns": int(y.shape[1]),
                "shape_pass": bool(x.shape == (d_expected, n_expected) and y.shape == (r_expected, n_expected)),
                "finite_design": bool(np.isfinite(x).all()),
                "finite_response": bool(np.isfinite(y).all()),
                "design_min": float(np.nanmin(x)),
                "design_max": float(np.nanmax(x)),
                "response_min": float(np.nanmin(y)),
                "response_max": float(np.nanmax(y)),
                "response_below_zero_count": int(np.count_nonzero(y < 0)),
                "response_above_one_count": int(np.count_nonzero(y > 1)),
                "response_above_one_sample_count": int(np.count_nonzero(np.any(y > 1, axis=0))),
                "design_duplicate_count_sampled": sampled_duplicate_count(x),
                "response_duplicate_count_sampled": sampled_duplicate_count(y),
                "design_unique_levels_min": int(min(np.unique(x[i]).size for i in range(x.shape[0]))),
                "design_unique_levels_max": int(max(np.unique(x[i]).size for i in range(x.shape[0]))),
                **metrics,
            }
            if case in master_rows:
                row["master_differences"] = compare_master(metrics, master_rows[case])
                row["master_reproduced"] = bool(max(abs(v) for v in row["master_differences"].values()) < 1e-9)
            results.append(row)

            if case in {"2x2", "grating"}:
                pairing[case] = pairing_sanity(x, y)
            del x, y

    report = {"archive": archive_meta, "cases": results, "pairing_sanity": pairing}
    (args.output_dir / "audit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    flat_keys = [k for k, v in results[0].items() if not isinstance(v, dict)]
    with (args.output_dir / "audit_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=flat_keys)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row[k] for k in flat_keys})

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
