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

# SynthCity 0.2.12 does not expose these through the plugin constructor. They
# are defaults of the internal model and are verified after fitting.
SYNTHCITY_INTERNAL_HYPERPARAMS: dict[str, dict[str, Any]] = {
    "rtvae": {"clipping_value": 1},
}


def synthcity_effective_hyperparams(generator_name: str) -> dict[str, Any]:
    return {
        **SYNTHCITY_HYPERPARAMS[generator_name],
        **SYNTHCITY_INTERNAL_HYPERPARAMS.get(generator_name, {}),
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
    parser.add_argument(
        "--data-root", type=Path, default=Path("data/openml_cc18_tabpfgen")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/table1_generators")
    )
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
    parser.add_argument(
        "--tabpfgen-init-strategy",
        choices=("paired", "class_matched", "class_agnostic"),
        default="paired",
    )
    parser.add_argument(
        "--tabpfgen-selection-metric",
        choices=("energy", "train_auc"),
        default="train_auc",
    )
    parser.add_argument("--tabpfgen-selection-interval", type=int, default=1)
    parser.add_argument(
        "--tabpfgen-label-strategy",
        choices=("exact_train_ratio", "bootstrap_train_ratio", "balanced"),
        default="exact_train_ratio",
        help="How to manually define y_synth for TabPFGen.",
    )
    parser.add_argument("--tabpfgen-swapped-energy-weight", type=float, default=1.0)
    parser.add_argument("--tabpfgen-n-ensembles", type=int, default=3)
    parser.add_argument("--tabpfgen-model-path", type=Path, default=None)
    parser.add_argument("--tabpfgen-source-root", type=Path, default=None)
    parser.add_argument(
        "--tabpfgen-scale",
        dest="tabpfgen_scale_features",
        action="store_true",
        help="Apply an extra StandardScaler before TabPFN (off for paper reproduction).",
    )
    parser.add_argument(
        "--tabpfgen-no-scale",
        dest="tabpfgen_scale_features",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(tabpfgen_scale_features=False)
    parser.add_argument(
        "--tabpfgen-stability-retries",
        action="store_true",
        help="Retry with altered SGLD parameters; off for exact paper settings.",
    )
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
    raise FileNotFoundError(
        f"Missing {path.with_suffix('.csv')} or {path.with_suffix('.parquet')}"
    )


def load_dataset(data_root: Path, dataset_id: int) -> pd.DataFrame:
    ddir = dataset_dir(data_root, dataset_id)
    combined = read_table(ddir / "data")
    if "target" not in combined.columns:
        raise ValueError(f"{ddir} data file does not contain a 'target' column")
    return combined


def split_train_test(
    df: pd.DataFrame, seed: int, test_size: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
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


def sample_exact_training_class_counts(
    batches: list[pd.DataFrame],
    train_df: pd.DataFrame,
    seed: int,
) -> pd.DataFrame | None:
    """Return exactly one training-set-sized synthetic sample, if possible."""
    if not batches:
        return None
    candidates = pd.concat(batches, ignore_index=True)
    if "target" not in candidates.columns:
        raise ValueError("Generated data does not contain the target column")

    sampled_parts: list[pd.DataFrame] = []
    target_counts = train_df["target"].value_counts(sort=False, dropna=False)
    rng = np.random.default_rng(seed)
    for label, required_count in target_counts.items():
        available = candidates[candidates["target"] == label]
        if len(available) < int(required_count):
            return None
        sampled_parts.append(
            available.sample(
                n=int(required_count),
                replace=False,
                random_state=int(rng.integers(0, 2**31 - 1)),
            )
        )

    synthetic = pd.concat(sampled_parts, ignore_index=True)
    synthetic = synthetic.sample(
        frac=1.0,
        random_state=int(rng.integers(0, 2**31 - 1)),
    ).reset_index(drop=True)
    synthetic["target"] = synthetic["target"].astype(train_df["target"].dtype)
    return synthetic.loc[:, train_df.columns]


def validate_training_sample_profile(
    synthetic: pd.DataFrame, train_df: pd.DataFrame
) -> None:
    expected = train_df["target"].value_counts(dropna=False).to_dict()
    observed = synthetic["target"].value_counts(dropna=False).to_dict()
    if len(synthetic) != len(train_df) or observed != expected:
        raise RuntimeError(
            "Synthetic data must match the training sample size and class counts; "
            f"expected n={len(train_df)}, counts={expected}, "
            f"observed n={len(synthetic)}, counts={observed}"
        )


def train_smote(train_df: pd.DataFrame, seed: int) -> tuple[None, pd.DataFrame]:
    from imblearn.over_sampling import SMOTE

    X = train_df.drop(columns=["target"])
    y = train_df["target"]
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
    synthetic = sample_exact_training_class_counts([synthetic], train_df, seed)
    if synthetic is None:
        raise RuntimeError("SMOTE did not produce the requested per-class sample counts")
    return None, synthetic


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
    if generator_name == "rtvae":
        clipping_value = getattr(
            getattr(getattr(plugin, "model", None), "model", None),
            "clipping_value",
            None,
        )
        if clipping_value != 1:
            raise RuntimeError(
                "Expected SynthCity RTVAE internal clipping_value=1, "
                f"found {clipping_value!r}"
            )
    batches: list[pd.DataFrame] = []
    synthetic = None
    for batch_index in range(10):
        batches.append(
            dataframe_from_generated(
                plugin.generate(
                    count=len(train_df),
                    random_state=seed + batch_index,
                )
            )
        )
        synthetic = sample_exact_training_class_counts(batches, train_df, seed)
        if synthetic is not None:
            break
    if synthetic is None:
        observed = pd.concat(batches, ignore_index=True)["target"].value_counts().to_dict()
        required = train_df["target"].value_counts().to_dict()
        raise RuntimeError(
            "SynthCity could not generate enough rows for every training class after "
            f"10 batches; required={required}, observed={observed}"
        )
    return plugin, synthetic, plugin_name


def train_tabpfgen_generator(
    train_df: pd.DataFrame,
    seed: int,
    device: str,
    n_steps: int,
    step_size: float,
    noise_scale: float,
    init_noise_std: float,
    init_strategy: str,
    selection_metric: str,
    selection_interval: int,
    label_strategy: str,
    swapped_energy_weight: float,
    scale_features: bool,
    n_ensembles: int,
    model_path: Path | None,
    source_root: Path | None,
    stability_retries: bool,
) -> tuple[None, pd.DataFrame, dict[str, Any]]:
    from tabpfgen import TabPFGen
    from tabpfgen.paper_tabpfn import PAPER_CHECKPOINT_SHA256, PAPER_TABPFN_COMMIT

    X = train_df.drop(columns=["target"]).to_numpy(dtype=np.float32)
    y = train_df["target"].to_numpy()
    rng = np.random.default_rng(seed)
    if label_strategy == "bootstrap_train_ratio":
        y_synth = rng.choice(y, size=len(y), replace=True)
    elif label_strategy == "balanced":
        labels, counts = np.unique(y, return_counts=True)
        base = len(y) // len(labels)
        rem = len(y) - base * len(labels)
        y_synth = np.concatenate(
            [np.repeat(label, base + (idx < rem)) for idx, label in enumerate(labels)]
        )
        rng.shuffle(y_synth)
    else:
        y_synth = y.copy()

    attempts = [
        {
            "n_sgld_steps": n_steps,
            "sgld_step_size": step_size,
            "sgld_noise_scale": noise_scale,
            "init_noise_std": init_noise_std,
        },
    ]
    if stability_retries:
        attempts.extend(
            [
                {
                    "n_sgld_steps": n_steps,
                    "sgld_step_size": step_size * 0.5,
                    "sgld_noise_scale": noise_scale * 0.5,
                    "init_noise_std": init_noise_std,
                },
                {
                    "n_sgld_steps": n_steps,
                    "sgld_step_size": step_size * 0.1,
                    "sgld_noise_scale": noise_scale * 0.1,
                    "init_noise_std": init_noise_std,
                },
            ]
        )
    errors: list[str] = []
    for attempt_idx, params in enumerate(attempts, start=1):
        try:
            generator = TabPFGen(
                **params,
                device=device,
                random_state=seed,
                init_strategy=init_strategy,
                selection_metric=selection_metric,
                selection_interval=selection_interval,
                swapped_energy_weight=swapped_energy_weight,
                scale_features=scale_features,
                n_estimators=n_ensembles,
                model_path=None if model_path is None else str(model_path),
                paper_tabpfn_root=None if source_root is None else str(source_root),
            )
            X_synth, y_synth_out = generator.generate_classification(
                X,
                y,
                n_samples=len(train_df),
                balance_classes=False,
                y_synth=y_synth,
            )
            if not np.isfinite(X_synth).all():
                raise FloatingPointError(
                    "TabPFGen returned non-finite synthetic features"
                )
            metadata = {
                **params,
                "attempt": attempt_idx,
                "retry_errors": errors,
                "fallback_used": attempt_idx > 1,
                "init_strategy": init_strategy,
                "selection_metric": selection_metric,
                "selection_interval": selection_interval,
                "label_strategy": label_strategy,
                "swapped_energy_weight": swapped_energy_weight,
                "scale_features": scale_features,
                "n_ensembles": n_ensembles,
                "model_path": None if model_path is None else str(model_path),
                "source_root": None if source_root is None else str(source_root),
                "paper_tabpfn_commit": PAPER_TABPFN_COMMIT,
                "paper_checkpoint_sha256": PAPER_CHECKPOINT_SHA256,
                "stability_retries": stability_retries,
                "best_step": (
                    generator.last_trace_.best_step
                    if generator.last_trace_ is not None
                    else None
                ),
                "best_score": (
                    generator.last_trace_.best_score
                    if generator.last_trace_ is not None
                    else None
                ),
            }
            break
        except FloatingPointError as exc:
            errors.append(f"attempt {attempt_idx}: {exc}")
    else:
        raise FloatingPointError(
            "TabPFGen failed all numerical-stability attempts: " + " | ".join(errors)
        )

    synthetic = pd.DataFrame(X_synth, columns=train_df.drop(columns=["target"]).columns)
    synthetic["target"] = y_synth_out
    return None, synthetic.reset_index(drop=True), metadata


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

    out_dir = (
        args.output_root / str(args.dataset_id) / f"seed_{args.seed}" / args.generator
    )
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
        generator, synthetic, tabpfgen_metadata = train_tabpfgen_generator(
            train_df,
            args.seed,
            args.device,
            args.tabpfgen_steps,
            args.tabpfgen_step_size,
            args.tabpfgen_noise_scale,
            args.tabpfgen_init_noise_std,
            args.tabpfgen_init_strategy,
            args.tabpfgen_selection_metric,
            args.tabpfgen_selection_interval,
            args.tabpfgen_label_strategy,
            args.tabpfgen_swapped_energy_weight,
            args.tabpfgen_scale_features,
            args.tabpfgen_n_ensembles,
            args.tabpfgen_model_path,
            args.tabpfgen_source_root,
            args.tabpfgen_stability_retries,
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

    validate_training_sample_profile(synthetic, train_df)

    model_path = save_generator(generator, out_dir)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    synthetic.to_csv(synthetic_path, index=False)

    if args.generator == "tabpfgen":
        hyperparameters = tabpfgen_metadata
    elif args.generator == "smote":
        hyperparameters = {}
    else:
        hyperparameters = {
            **synthcity_effective_hyperparams(args.generator),
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
        "target_counts_synthetic": (
            synthetic["target"].value_counts().to_dict()
            if "target" in synthetic.columns
            else {}
        ),
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
