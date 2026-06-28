#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot training loss curve from history.csv.

Usage:
    python plot_loss_curve.py
    python plot_loss_curve.py --csv history.csv --out loss_curve.png

Requirements:
    pip install pandas matplotlib
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def plot_loss_curve(csv_path: str, output_path: str, loss_col: str = "loss", x_col: str = "epoch") -> None:
    csv_file = Path(csv_path)

    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file.resolve()}")

    df = pd.read_csv(csv_file)

    if loss_col not in df.columns:
        raise ValueError(
            f"Column '{loss_col}' was not found. Available columns: {list(df.columns)}"
        )

    if x_col in df.columns:
        x = df[x_col]
        x_label = x_col
    else:
        x = range(1, len(df) + 1)
        x_label = "epoch"

    y = df[loss_col]

    plt.figure(figsize=(4, 3), dpi=300)
    plt.plot(x, y, marker="o", linewidth=2, markersize=1, color = '#BC6F45')
    plt.xlabel(x_label, fontsize = 8)
    plt.ylabel(loss_col, fontsize = 8)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    # plt.title("Training Loss Curve")
    plt.tight_layout()

    output_file = Path(output_path)
    plt.savefig(output_file, dpi=300)
    plt.grid(False)
    plt.show()

    print(f"Loss curve saved to: {output_file.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot loss curve from a training history CSV file.")
    parser.add_argument("--csv", default="path")
    parser.add_argument("--out", default="path")
    parser.add_argument("--loss-col", default="loss", help="Loss column name. Default: loss")
    parser.add_argument("--x-col", default="epoch", help="X-axis column name. Default: epoch")
    args = parser.parse_args()

    plot_loss_curve(
        csv_path=args.csv,
        output_path=args.out,
        loss_col=args.loss_col,
        x_col=args.x_col,
    )


if __name__ == "__main__":
    main()
