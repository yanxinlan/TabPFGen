#!/usr/bin/env python3
"""Train the Table 1 synthetic-data generators.

This script expects datasets downloaded by download_openml_cc18_tabpfgen.py.
It trains one generator for one dataset/seed at a time, which makes it suitable
for Slurm array jobs. Outputs are written under:

    outputs/table1_generators/<dataset_id>/<seed>/<generator>/

The generated sample size equals the real training-set size, matching Table 1:
synthetic data is either appended to the real training split (augmentation) or
used alone (replacement) by a later downstream-evaluation script.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.exists():
    sys.path.insert(0, str(SRC_ROOT))


SYNTHCITY_HYPERPARAMS: dict[str, dict[str, Any]] = {
    "tvae": {
        "n_units_embedding": 500,
        "lr": 5e-4,
        "weight_decay": 1e-5,
        "batch_size": 1000,
        "decoder_n_layers_hidden": 2,
        "decoder_n_units_hidden": 256,
        "decoder_nonlin": "leaky_relu",
        "decoder_dropout": 0.1,
        "encoder_n_layers_hidden": 3,
        "encoder_n_units_hidden": 256,
        "encoder_nonlin": "leaky_relu",
        "encoder_dropout": 0.1,
        "loss_factor": 1,
        "data_encoder_max_clusters": 10,
        "clipping_value": 1,
        "sampling_patience": 500,
    },
    "rtvae": {
        "n_units_embedding": 500,
        "lr": 0.001,
        "weight_decay": 1e-5,
        "batch_size": 200,
        "decoder_n_layers_hidden": 3,
        "decoder_n_units_hidden": 500,
        "decoder_nonlin": "leaky_relu",
        "decoder_dropout": 0,
        "encoder_n_layers_hidden": 3,
        "encoder_n_units_hidden": 500,
        "encoder_nonlin": "leaky_relu",
        "encoder_dropout": 0.1,
        "data_encoder_max_clusters": 10,
        "robust_divergence_beta": 2,
        "sampling_patience": 500,
    },
    "ctgan": {
        "generator_n_layers_hidden": 2,
        "generator_n_units_hidden": 256,
        "generator_nonlin": "relu",
        "generator_dropout": 0.1,
        "generator_opt_betas": (0.9, 0.999),
        "discriminator_n_layers_hidden": 2,
        "discriminator_n_units_hidden": 256,
        "discriminator_nonlin": "leaky_relu",
        "discriminator_n_iter": 1,
        "discriminator_dropout": 0.1,
        "discriminator_opt_betas": (0.9, 0.999),
        "lr": 5e-4,
        "weight_decay": 1e-3,
        "batch_size": 1000,
        "clipping_value": 1,
        "lambda_gradient_penalty": 10,
        "encoder_max_clusters": 10,
        "sampling_patience": 500,
    },
    "nflow": {
        "n_layers_hidden": 2,
        "n_units_hidden": 256,
        "batch_size": 1000,
        "num_transform_blocks": 1,
        "dropout": 0.1,
        "batch_norm": False,
        "num_bins": 8,
        "tail_bound": 3,
        "lr": 5e-4,
        "apply_unconditional_transform": True,
        "base_distribution": "standard_normal",
        "linear_transform_type": "permutation",
        "base_transform_type": "rq-autoregressive",
        "encoder_max_clusters": 10,
        "n_iter_min": 100,
        "sampling_patience": 500,
    },
    # SynthCity versions have used different names for the TabDDPM plugin.
    # The script resolves this alias at runtime against installed plugins.
    "tabddpm": {
        "is_classification": True,
        "lr": 0.002,
        "weight_decay": 0.0001,
        "batch_size": 1024,
        "gaussian_loss_type": "mse",
        "scheduler": "cosine",
        "model_type": "mlp",
        "dim_embed": 128,
        "continuous_encoder": "quantile",
        "sampling_patience": 500,
    },
}


PLUGIN_ALIASES: dict[str, tuple[str, ...]] = {
    "ctgan": ("ctgan",),
    "tvae": ("tvae",),
    "rtvae": ("rtvae",),
    "nflow": ("nflow", "nf"),
    "tabddpm": ("tabddpm", "ddpm"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one Table 1 generator on one OpenML-CC18 dataset split."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/openml_cc18_tabpfgen"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/table1_generators"))
    parser.add_argument("--dataset-id", type=int, required=True)
    parser.add_argument(
        "--generator",
        choices=("smote", "ctgan", "tvae", "nflow", "rtvae", "tabddpm", "tabpfgen"),
        required=True,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-size", type=float, default=0.5)
    parser.add_argument(
        "--n-iter",
        type=int,
        default=1000,
        help="Maximum training iterations for neural SynthCity generators.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tabpfgen-steps", type=int, default=1000)
    parser.add_argument("--tabpfgen-step-size", type=float, default=0.01)
    parser.add_argument("--tabpfgen-noise-scale", type=float, default=0.01)
    parser.add_argument("--tabpfgen-init-noise-std", type=float, default=0.01)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def dataset_dir(data_root: Path, dataset_id: int) -> Path:
    matches = sorted(data_root.glob(f"{dataset_id}_*"))
    if not matches:
        raise FileNotFoundError(
            f"No downloaded dataset directory matching {dataset_id}_* under {data_root}"
        )
    if len(matches) > 1:
        raise RuntimeError(f"Multiple dataset dirs for {dataset_id}: {matches}")
    return matches[0]


def read_table(path: Path) -> pd.DataFrame:
    if path.with_suffix(".csv").exists():
        return pd.read_csv(path.with_suffix(".csv"))
    if path.with_suffix(".parquet").exists():
        return pd.read_parquet(path.with_suffix(".parquet"))
    raise FileNotFoundError(f"Missing {path.with_suffix('.csv')} or {path.with_suffix('.parquet')}")


def load_dataset(data_root: Path, dataset_id: int) -> pd.DataFrame:
    ddir = dataset_dir(data_root, dataset_id)
    combined = read_table(ddir / "data")
    if "target" not in combined.columns:
        raise ValueError(f"{ddir} data file does not contain a 'target' column")
    return combined


def split_train_test(df: pd.DataFrame, seed: int, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["target"],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def dataframe_from_generated(generated: Any) -> pd.DataFrame:
    if hasattr(generated, "dataframe"):
        return generated.dataframe()
    if isinstance(generated, pd.DataFrame):
        return generated
    raise TypeError(f"Unsupported generated object type: {type(generated)!r}")


def train_smote(train_df: pd.DataFrame, seed: int) -> tuple[None, pd.DataFrame]:
    from imblearn.over_sampling import SMOTE

    X = train_df.drop(columns=["target"])
    y = train_df["target"]
    rng = np.random.default_rng(seed)
    target_counts = y.value_counts()
    desired = target_counts.to_dict()
    for cls, count in target_counts.items():
        desired[cls] = int(count * 2)

    k_neighbors = max(1, min(5, int(target_counts.min()) - 1))
    smote = SMOTE(random_state=seed, sampling_strategy=desired, k_neighbors=k_neighbors)
    X_res, y_res = smote.fit_resample(X, y)

    synthetic = pd.concat(
        [
            pd.DataFrame(X_res, columns=X.columns).iloc[len(train_df) :],
            pd.Series(y_res, name="target").iloc[len(train_df) :],
        ],
        axis=1,
    ).reset_index(drop=True)
    if len(synthetic) != len(train_df):
        synthetic = synthetic.sample(
            n=len(train_df),
            replace=len(synthetic) < len(train_df),
            random_state=seed,
        )
    synthetic = synthetic.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1)))
    return None, synthetic.reset_index(drop=True)


def resolve_plugin_name(requested: str) -> str:
    from synthcity.plugins import Plugins

    plugins = Plugins(categories=["generic"])
    available = set(plugins.list())
    for alias in PLUGIN_ALIASES[requested]:
        if alias in available:
            return alias
    raise RuntimeError(
        f"Could not find a SynthCity plugin for {requested}. "
        f"Tried {PLUGIN_ALIASES[requested]}; available plugins include: {sorted(available)}"
    )


def train_synthcity_generator(
    train_df: pd.DataFrame,
    generator_name: str,
    seed: int,
    n_iter: int,
    device: str,
) -> tuple[Any, pd.DataFrame, str]:
    from synthcity.plugins import Plugins

    plugin_name = resolve_plugin_name(generator_name)
    params = dict(SYNTHCITY_HYPERPARAMS[generator_name])
    params["random_state"] = seed
    if generator_name == "tabddpm":
        import torch

        params["device"] = torch.device(device)
    else:
        params["device"] = device
    if "n_iter" not in params:
        params["n_iter"] = n_iter

    plugin = Plugins(categories=["generic"]).get(plugin_name, **params)
    plugin.fit(train_df)
    synthetic = dataframe_from_generated(
        plugin.generate(count=len(train_df), random_state=seed)
    )
    return plugin, synthetic.reset_index(drop=True), plugin_name


def train_tabpfgen_generator(
    train_df: pd.DataFrame,
    seed: int,
    device: str,
    n_steps: int,
    step_size: float,
    noise_scale: float,
    init_noise_std: float,
) -> tuple[None, pd.DataFrame]:
    from tabpfgen import TabPFGen

    X = train_df.drop(columns=["target"]).to_numpy(dtype=np.float32)
    y = train_df["target"].to_numpy()
    rng = np.random.default_rng(seed)
    y_synth = rng.choice(y, size=len(y), replace=True)

    generator = TabPFGen(
        n_sgld_steps=n_steps,
        sgld_step_size=step_size,
        sgld_noise_scale=noise_scale,
        init_noise_std=init_noise_std,
        device=device,
        random_state=seed,
    )
    X_synth, y_synth = generator.generate_classification(
        X,
        y,
        n_samples=len(train_df),
        balance_classes=False,
        y_synth=y_synth,
    )
    synthetic = pd.DataFrame(X_synth, columns=train_df.drop(columns=["target"]).columns)
    synthetic["target"] = y_synth
    return None, synthetic.reset_index(drop=True)


def save_generator(generator: Any, out_dir: Path) -> str | None:
    if generator is None:
        return None
    model_path = out_dir / "generator.pkl"
    try:
        if hasattr(generator, "save_to_file"):
            generator.save_to_file(model_path)
        else:
            with model_path.open("wb") as f:
                pickle.dump(generator, f)
        return str(model_path)
    except Exception as exc:
        warning_path = out_dir / "generator_save_error.txt"
        warning_path.write_text(str(exc), encoding="utf-8")
        return None


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    out_dir = args.output_root / str(args.dataset_id) / f"seed_{args.seed}" / args.generator
    synthetic_path = out_dir / "synthetic.csv"
    train_path = out_dir / "train.csv"
    test_path = out_dir / "test.csv"
    metadata_path = out_dir / "metadata.json"
    if synthetic_path.exists() and metadata_path.exists() and not args.overwrite:
        print(f"Skipping existing output: {out_dir}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_dataset(args.data_root, args.dataset_id)
    train_df, test_df = split_train_test(df, args.seed, args.test_size)

    if args.generator == "smote":
        generator, synthetic = train_smote(train_df, args.seed)
        resolved_name = "smote"
    elif args.generator == "tabpfgen":
        generator, synthetic = train_tabpfgen_generator(
            train_df,
            args.seed,
            args.device,
            args.tabpfgen_steps,
            args.tabpfgen_step_size,
            args.tabpfgen_noise_scale,
            args.tabpfgen_init_noise_std,
        )
        resolved_name = "tabpfgen"
    else:
        generator, synthetic, resolved_name = train_synthcity_generator(
            train_df,
            args.generator,
            args.seed,
            args.n_iter,
            args.device,
        )

    model_path = save_generator(generator, out_dir)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    synthetic.to_csv(synthetic_path, index=False)

    if args.generator == "tabpfgen":
        hyperparameters = {
            "n_sgld_steps": args.tabpfgen_steps,
            "sgld_step_size": args.tabpfgen_step_size,
            "sgld_noise_scale": args.tabpfgen_noise_scale,
            "init_noise_std": args.tabpfgen_init_noise_std,
        }
    elif args.generator == "smote":
        hyperparameters = {}
    else:
        hyperparameters = {
            **SYNTHCITY_HYPERPARAMS[args.generator],
            "n_iter": args.n_iter,
            "random_state": args.seed,
            "device": args.device,
        }

    metadata = {
        "dataset_id": args.dataset_id,
        "generator": args.generator,
        "resolved_generator": resolved_name,
        "seed": args.seed,
        "test_size": args.test_size,
        "n_iter": args.n_iter,
        "device": args.device,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_synthetic": int(len(synthetic)),
        "target_counts_train": train_df["target"].value_counts().to_dict(),
        "target_counts_synthetic": synthetic["target"].value_counts().to_dict()
        if "target" in synthetic.columns
        else {},
        "paths": {
            "train": str(train_path),
            "test": str(test_path),
            "synthetic": str(synthetic_path),
            "generator": model_path,
        },
        "hyperparameters": hyperparameters,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote synthetic data: {synthetic_path}")
    print(f"Wrote metadata: {metadata_path}")


if __name__ == "__main__":
    main()
