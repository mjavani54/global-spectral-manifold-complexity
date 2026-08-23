#!/usr/bin/env python3
"""Controlled DNN learning-curve and capacity experiments for Paper 2.

This script requires PyTorch and is intended for the user's CUDA workstation.
It reads the original ZIP directly and writes restartable CSV results after every
completed model.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:  # pragma: no cover - executed on the run workstation
    raise SystemExit(
        "PyTorch is required. Install a CUDA build from https://pytorch.org/get-started/locally/"
    ) from exc


CASES = {
    "multilayer_8": ("Datasets/SiO2_TiO2_8_layer_50K_200_points_DS.csv", "Datasets/SiO2_TiO2_8_layer_50K_200_points_RS.csv", 8),
    "multilayer_10": ("Datasets/SiO2_TiO2_10_layer_50K_200_points_DS.csv", "Datasets/SiO2_TiO2_10_layer_50K_200_points_RS.csv", 10),
    "multilayer_12": ("Datasets/SiO2_TiO2_12_layer_50K_200_points_DS.csv", "Datasets/SiO2_TiO2_12_layer_50K_200_points_RS.csv", 12),
    "multilayer_14": ("Datasets/SiO2_TiO2_14_layer_50K_200_points_DS.csv", "Datasets/SiO2_TiO2_14_layer_50K_200_points_RS.csv", 14),
    "multilayer_16": ("Datasets/SiO2_TiO2_16_layer_50K_200_points_DS.csv", "Datasets/SiO2_TiO2_16_layer_50K_200_points_RS.csv", 16),
    "multilayer_18": ("Datasets/SiO2_TiO2_18_layer_50K_200_points_DS.csv", "Datasets/SiO2_TiO2_18_layer_50K_200_points_RS.csv", 18),
    "multilayer_20": ("Datasets/SiO2_TiO2_20_layer_200K_200_points_DS.csv", "Datasets/SiO2_TiO2_20_layer_200K_200_points_RS.csv", 20),
    "2x2": ("Datasets/2X2_DS.csv", "Datasets/2X2_RS.csv", 9),
    "grating": ("Datasets/Grating_DS.csv", "Datasets/Grating_RS.csv", 4),
}


@dataclass
class RunResult:
    case: str
    mode: str
    seed: int
    train_size: int
    val_size: int
    test_size: int
    common_response_dim: int
    width: int
    hidden_layers: int
    parameter_count: int
    best_epoch: int
    epochs_trained: int
    train_seconds: float
    train_mse_scaled: float
    val_mse_scaled: float
    test_r2_global: float
    test_r2_spectral_mean: float
    test_rmse: float
    test_nrmse_std: float
    test_mae: float
    test_spectral_angle_deg: float
    test_sample_max_abs_error_p95: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_csv(zf: zipfile.ZipFile, member: str) -> np.ndarray:
    with zf.open(member, "r") as f:
        return np.loadtxt(f, delimiter=",", dtype=np.float32, encoding="utf-8-sig")


def resample_normalized_axis(y: np.ndarray, output_dim: int) -> np.ndarray:
    if y.shape[0] == output_dim:
        return y
    old = np.linspace(0.0, 1.0, y.shape[0])
    new = np.linspace(0.0, 1.0, output_dim)
    right = np.searchsorted(old, new, side="left")
    right = np.clip(right, 1, y.shape[0] - 1)
    left = right - 1
    alpha = ((new - old[left]) / (old[right] - old[left])).astype(np.float32)[:, None]
    return y[left, :] * (1.0 - alpha) + y[right, :] * alpha


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, width: int, hidden_layers: int):
        super().__init__()
        layers: list[nn.Module] = []
        current = input_dim
        for _ in range(hidden_layers):
            layers.extend([nn.Linear(current, width), nn.SiLU()])
            current = width
        layers.append(nn.Linear(current, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y.astype(np.float32)))
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def loader_mse(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    count = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        pred = model(xb)
        total += float(torch.sum((pred - yb) ** 2).item())
        count += int(yb.numel())
    return total / count


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    out = []
    loader = DataLoader(torch.from_numpy(x.astype(np.float32)), batch_size=batch_size, shuffle=False)
    for xb in loader:
        out.append(model(xb.to(device)).cpu().numpy())
    return np.concatenate(out, axis=0)


def evaluation_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residual = y_true - y_pred
    sse = float(np.sum(residual * residual))
    centered = y_true - y_true.mean()
    sst = float(np.sum(centered * centered))
    r2_global = 1.0 - sse / sst

    channel_sse = np.sum(residual * residual, axis=0)
    channel_centered = y_true - y_true.mean(axis=0, keepdims=True)
    channel_sst = np.sum(channel_centered * channel_centered, axis=0)
    valid = channel_sst > 1e-14
    r2_spectral = float(np.mean(1.0 - channel_sse[valid] / channel_sst[valid]))

    rmse = float(np.sqrt(np.mean(residual * residual)))
    nrmse = rmse / float(np.std(y_true))
    mae = float(np.mean(np.abs(residual)))

    numerator = np.sum(y_true * y_pred, axis=1)
    denominator = np.linalg.norm(y_true, axis=1) * np.linalg.norm(y_pred, axis=1)
    cosine = np.clip(numerator / np.maximum(denominator, 1e-14), -1.0, 1.0)
    spectral_angle = float(np.mean(np.degrees(np.arccos(cosine))))
    p95_max_abs = float(np.percentile(np.max(np.abs(residual), axis=1), 95))
    return {
        "test_r2_global": r2_global,
        "test_r2_spectral_mean": r2_spectral,
        "test_rmse": rmse,
        "test_nrmse_std": nrmse,
        "test_mae": mae,
        "test_spectral_angle_deg": spectral_angle,
        "test_sample_max_abs_error_p95": p95_max_abs,
    }


def fit_one(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    width: int,
    hidden_layers: int,
    seed: int,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[dict, np.ndarray]:
    set_seed(seed)
    x_mean = x_train.mean(axis=0, keepdims=True)
    x_std = x_train.std(axis=0, keepdims=True)
    x_std[x_std < 1e-12] = 1.0
    y_mean = y_train.mean(axis=0, keepdims=True)
    y_std = y_train.std(axis=0, keepdims=True)
    y_std[y_std < 1e-8] = 1.0

    x_train_s = (x_train - x_mean) / x_std
    x_val_s = (x_val - x_mean) / x_std
    x_test_s = (x_test - x_mean) / x_std
    y_train_s = (y_train - y_mean) / y_std
    y_val_s = (y_val - y_mean) / y_std

    train_loader = make_loader(x_train_s, y_train_s, args.batch_size, True, seed)
    val_loader = make_loader(x_val_s, y_val_s, args.batch_size, False, seed)
    model = MLP(x_train.shape[1], y_train.shape[1], width, hidden_layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()
    best_loss = math.inf
    best_epoch = 0
    best_state = None
    no_improvement = 0
    started = time.time()
    final_epoch = 0

    for epoch in range(1, args.max_epochs + 1):
        final_epoch = epoch
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
        val_loss = loader_mse(model, val_loader, device)
        if val_loss < best_loss - args.min_delta:
            best_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            no_improvement = 0
        else:
            no_improvement += 1
        if no_improvement >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("Training failed to produce a finite validation checkpoint")
    model.load_state_dict(best_state)
    train_mse_scaled = loader_mse(model, make_loader(x_train_s, y_train_s, args.batch_size, False, seed), device)
    pred_scaled = predict(model, x_test_s, device, args.batch_size)
    y_pred = pred_scaled * y_std + y_mean
    metrics = evaluation_metrics(y_test, y_pred)
    metrics.update({
        "parameter_count": count_parameters(model),
        "best_epoch": best_epoch,
        "epochs_trained": final_epoch,
        "train_seconds": time.time() - started,
        "train_mse_scaled": train_mse_scaled,
        "val_mse_scaled": best_loss,
    })
    return metrics, y_pred


def completed_keys(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as f:
        return {
            (row["case"], row["mode"], int(row["seed"]), int(row["train_size"]), int(row["width"]), int(row["hidden_layers"]))
            for row in csv.DictReader(f)
        }


def append_result(path: Path, result: RunResult) -> None:
    row = asdict(result)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def split_indices(n_total: int, seed: int, val_size: int, test_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_total)
    test = order[:test_size]
    val = order[test_size : test_size + val_size]
    train_pool = order[test_size + val_size :]
    return train_pool, val, test


def run_case(case: str, args: argparse.Namespace, device: torch.device, results_path: Path) -> None:
    ds_member, rs_member, design_dim = CASES[case]
    with zipfile.ZipFile(args.datasets_zip) as zf:
        x = load_csv(zf, ds_member).T
        y = resample_normalized_axis(load_csv(zf, rs_member), args.common_dim).T
    if x.shape[1] != design_dim or x.shape[0] != y.shape[0]:
        raise RuntimeError(f"Unexpected shapes for {case}: X={x.shape}, Y={y.shape}")

    done = completed_keys(results_path) if args.resume else set()
    for seed in args.seeds:
        split_seed = args.split_seed_base + seed
        train_pool, val_idx, test_idx = split_indices(x.shape[0], split_seed, args.val_size, args.test_size)
        if args.mode in {"all", "learning", "pilot"}:
            sizes = [min(args.train_sizes)] if args.mode == "pilot" else args.train_sizes
            for train_size in sizes:
                if train_size > train_pool.size:
                    print(f"Skipping {case} N={train_size}: only {train_pool.size} training samples available", flush=True)
                    continue
                key = (case, "learning", seed, train_size, args.fixed_width, args.hidden_layers)
                if key in done:
                    print(f"Already complete: {key}", flush=True)
                    continue
                print(f"Training {key} on {device}", flush=True)
                train_idx = train_pool[:train_size]
                metrics, pred = fit_one(
                    x[train_idx], y[train_idx], x[val_idx], y[val_idx], x[test_idx], y[test_idx],
                    args.fixed_width, args.hidden_layers, seed, device, args,
                )
                result = RunResult(
                    case=case, mode="learning", seed=seed, train_size=train_size,
                    val_size=args.val_size, test_size=args.test_size,
                    common_response_dim=args.common_dim, width=args.fixed_width,
                    hidden_layers=args.hidden_layers, **metrics,
                )
                append_result(results_path, result)
                if train_size == max(s for s in args.train_sizes if s <= train_pool.size) and seed == args.seeds[0]:
                    np.savez_compressed(
                        args.output_dir / f"prediction_example_{case}.npz",
                        y_true=y[test_idx][: min(40, args.test_size)],
                        y_pred=pred[: min(40, args.test_size)],
                    )

        if args.mode in {"all", "capacity"}:
            train_size = min(args.capacity_train_size, train_pool.size)
            for width in args.capacity_widths:
                key = (case, "capacity", seed, train_size, width, args.hidden_layers)
                if key in done:
                    print(f"Already complete: {key}", flush=True)
                    continue
                print(f"Training {key} on {device}", flush=True)
                train_idx = train_pool[:train_size]
                metrics, _ = fit_one(
                    x[train_idx], y[train_idx], x[val_idx], y[val_idx], x[test_idx], y[test_idx],
                    width, args.hidden_layers, seed, device, args,
                )
                append_result(results_path, RunResult(
                    case=case, mode="capacity", seed=seed, train_size=train_size,
                    val_size=args.val_size, test_size=args.test_size,
                    common_response_dim=args.common_dim, width=width,
                    hidden_layers=args.hidden_layers, **metrics,
                ))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets-zip", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=Path("paper2_dnn_output"))
    ap.add_argument("--mode", choices=["pilot", "learning", "capacity", "all"], default="pilot")
    ap.add_argument("--cases", nargs="+", choices=list(CASES), default=list(CASES))
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--train-sizes", type=int, nargs="+", default=[500, 1000, 2000, 3000, 4000, 4800])
    ap.add_argument("--capacity-train-size", type=int, default=4800)
    ap.add_argument("--capacity-widths", type=int, nargs="+", default=[32, 64, 128, 256, 512])
    ap.add_argument("--fixed-width", type=int, default=512)
    ap.add_argument("--hidden-layers", type=int, default=4)
    ap.add_argument("--common-dim", type=int, default=200)
    ap.add_argument("--val-size", type=int, default=600)
    ap.add_argument("--test-size", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-epochs", type=int, default=600)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--min-delta", type=float, default=1e-5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-6)
    ap.add_argument("--split-seed-base", type=int, default=991_000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    metadata = {
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "method_note": "All response vectors are linearly resampled to 200 points on their own normalized spectral coordinate; this controls output resolution but does not align physical wavelength/frequency ranges.",
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)
    results_path = args.output_dir / "dnn_results.csv"
    for case in args.cases:
        run_case(case, args, device, results_path)
    print(f"Complete. Results: {results_path}", flush=True)


if __name__ == "__main__":
    main()
