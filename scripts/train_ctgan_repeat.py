#!/usr/bin/env python3
"""Train one CTGAN repeat for stability analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from train_table1_generators import (
    SYNTHCITY_HYPERPARAMS,
    load_dataset,
    save_generator,
    split_train_test,
    train_synthcity_generator,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one repeated CTGAN run.")
    parser.add_argument("--data-root", type=Path, default=Path("data/openml_cc18_tabpfgen"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/ctgan_repeats"))
    parser.add_argument("--dataset-id", type=int, required=True)
    parser.add_argument("--split-seed", type=int, required=True)
    parser.add_argument("--ctgan-seed", type=int, required=True)
    parser.add_argument("--test-size", type=float, default=0.5)
    parser.add_argument("--n-iter", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = (
        args.output_root
        / str(args.dataset_id)
        / f"split_seed_{args.split_seed}"
        / f"ctgan_seed_{args.ctgan_seed}"
        / "ctgan"
    )
    synthetic_path = out_dir / "synthetic.csv"
    metadata_path = out_dir / "metadata.json"
    if synthetic_path.exists() and metadata_path.exists() and not args.overwrite:
        print(f"Skipping existing repeat: {out_dir}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_dataset(args.data_root, args.dataset_id)
    train_df, test_df = split_train_test(df, args.split_seed, args.test_size)
    generator, synthetic, resolved_name = train_synthcity_generator(
        train_df,
        "ctgan",
        args.ctgan_seed,
        args.n_iter,
        args.device,
    )

    train_path = out_dir / "train.csv"
    test_path = out_dir / "test.csv"
    model_path = save_generator(generator, out_dir)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    synthetic.to_csv(synthetic_path, index=False)

    metadata = {
        "dataset_id": args.dataset_id,
        "generator": "ctgan",
        "resolved_generator": resolved_name,
        "split_seed": args.split_seed,
        "ctgan_seed": args.ctgan_seed,
        "test_size": args.test_size,
        "n_iter": args.n_iter,
        "device": args.device,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_synthetic": int(len(synthetic)),
        "hyperparameters": {
            **SYNTHCITY_HYPERPARAMS["ctgan"],
            "n_iter": args.n_iter,
            "random_state": args.ctgan_seed,
            "device": args.device,
        },
        "paths": {
            "train": str(train_path),
            "test": str(test_path),
            "synthetic": str(synthetic_path),
            "generator": model_path,
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote CTGAN repeat: {synthetic_path}")


if __name__ == "__main__":
    main()
