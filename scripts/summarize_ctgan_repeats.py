#!/usr/bin/env python3
"""Summarize repeated CTGAN runs by mean and best AUC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from analyze_reproduction_results import PAPER_TABLE1_AUC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize repeated CTGAN results.")
    parser.add_argument(
        "--downstream-root",
        type=Path,
        default=Path("outputs/ctgan_repeats_downstream"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/analysis"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for path in sorted(args.downstream_root.glob("*/*/*/*/*/metrics.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        row["metrics_path"] = str(path)
        rows.append(row)
    metrics = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    long_path = args.output_dir / "ctgan_repeats_long.csv"
    metrics.to_csv(long_path, index=False)
    print(f"Wrote {long_path} ({len(metrics)} rows)")
    if metrics.empty:
        return

    grouped = (
        metrics.groupby(["mode", "downstream_model"], as_index=False)
        .agg(
            mean_auc=("auc", "mean"),
            std_auc=("auc", "std"),
            best_auc=("auc", "max"),
            worst_auc=("auc", "min"),
            n_runs=("auc", "size"),
        )
    )
    grouped["paper_auc"] = grouped.apply(
        lambda row: PAPER_TABLE1_AUC.get((row["mode"], row["downstream_model"], "ctgan")),
        axis=1,
    )
    grouped["mean_delta_vs_paper"] = grouped["mean_auc"] - grouped["paper_auc"]
    grouped["best_delta_vs_paper"] = grouped["best_auc"] - grouped["paper_auc"]
    summary_path = args.output_dir / "ctgan_repeats_summary.csv"
    grouped.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path} ({len(grouped)} rows)")


if __name__ == "__main__":
    main()
