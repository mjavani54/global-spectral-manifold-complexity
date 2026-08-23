#!/usr/bin/env python3
"""Regenerate DNN learning and capacity figures from frozen aggregate CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from paper2_postprocess_dnn import make_figures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=Path("results/dnn_summaries"),
        help="Directory containing frozen DNN summary CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("build/figures"),
        help="Destination for regenerated figures.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    learning = pd.read_csv(args.summary_dir / "learning_curve_summary.csv")
    capacity = pd.read_csv(args.summary_dir / "capacity_curve_summary.csv")
    merged = pd.read_csv(args.summary_dir / "geometry_learning_merged.csv")
    make_figures(learning, capacity, merged, args.output_dir)


if __name__ == "__main__":
    main()

