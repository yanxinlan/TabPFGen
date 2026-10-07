#!/usr/bin/env python3
"""Evaluate one CTGAN repeat."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from evaluate_table1_downstream import auc_score, make_model, make_train_frame, predict_proba


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one repeated CTGAN run.")
    parser.add_argument("--repeat-root", type=Path, default=Path("outputs/ctgan_repeats"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/ctgan_repeats_downstream"))
    parser.add_argument("--dataset-id", type=int, required=True)
    parser.add_argument("--split-seed", type=int, required=True)
    parser.add_argument("--ctgan-seed", type=int, required=True)
    parser.add_argument("--mode", choices=("augmentation", "replacement"), required=True)
    parser.add_argument("--downstream-model", choices=("xgb", "rf", "lr", "tabpfn"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = (
        args.repeat_root
        / str(args.dataset_id)
        / f"split_seed_{args.split_seed}"
        / f"ctgan_seed_{args.ctgan_seed}"
        / "ctgan"
    )
    out_dir = (
        args.output_root
        / str(args.dataset_id)
        / f"split_seed_{args.split_seed}"
        / f"ctgan_seed_{args.ctgan_seed}"
        / args.mode
        / args.downstream_model
    )
    result_path = out_dir / "metrics.json"
    if result_path.exists() and not args.overwrite:
        print(f"Skipping existing result: {result_path}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(base / "train.csv")
    test_df = pd.read_csv(base / "test.csv")
    synthetic = pd.read_csv(base / "synthetic.csv")
    train_used = make_train_frame(train_df, synthetic, "ctgan", args.mode)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_used["target"])
    y_test = label_encoder.transform(test_df["target"])
    X_train = train_used.drop(columns=["target"]).to_numpy(dtype=np.float32)
    X_test = test_df.drop(columns=["target"]).to_numpy(dtype=np.float32)

    model_seed = args.split_seed * 1000 + args.ctgan_seed
    model = make_model(args.downstream_model, model_seed, args.device)
    model.fit(X_train, y_train)
    auc = auc_score(y_test, predict_proba(model, X_test), len(label_encoder.classes_))
    metrics = {
        "dataset_id": args.dataset_id,
        "split_seed": args.split_seed,
        "ctgan_seed": args.ctgan_seed,
        "generator": "ctgan",
        "mode": args.mode,
        "downstream_model": args.downstream_model,
        "auc": auc,
        "n_train_real": int(len(train_df)),
        "n_train_used": int(len(train_used)),
        "n_test": int(len(test_df)),
        "classes": [str(c) for c in label_encoder.classes_],
        "result_path": str(result_path),
    }
    result_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
