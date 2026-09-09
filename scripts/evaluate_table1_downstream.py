#!/usr/bin/env python3
"""Evaluate one Table 1 downstream classifier job."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


GENERATORS = ("original", "smote", "ctgan", "tvae", "nflow", "rtvae", "tabddpm", "tabpfgen")
MODELS = ("xgb", "rf", "lr", "tabpfn")
MODES = ("augmentation", "replacement")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one Table 1 downstream run.")
    parser.add_argument("--data-root", type=Path, default=Path("data/openml_cc18_tabpfgen"))
    parser.add_argument("--generator-root", type=Path, default=Path("outputs/table1_generators"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/table1_downstream"))
    parser.add_argument("--dataset-id", type=int, required=True)
    parser.add_argument("--generator", choices=GENERATORS, required=True)
    parser.add_argument("--downstream-model", choices=MODELS, required=True)
    parser.add_argument("--mode", choices=MODES, default="augmentation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-size", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
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
        raise FileNotFoundError(f"No dataset directory matching {dataset_id}_* under {data_root}")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple dataset directories for {dataset_id}: {matches}")
    return matches[0]


def load_original_split(data_root: Path, dataset_id: int, seed: int, test_size: float):
    df = pd.read_csv(dataset_dir(data_root, dataset_id) / "data.csv")
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["target"],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def load_generated_split(generator_root: Path, dataset_id: int, seed: int, generator: str):
    base = generator_root / str(dataset_id) / f"seed_{seed}" / generator
    train_path = base / "train.csv"
    test_path = base / "test.csv"
    synthetic_path = base / "synthetic.csv"
    missing = [path for path in (train_path, test_path, synthetic_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing generator outputs: {missing}")
    return (
        pd.read_csv(train_path),
        pd.read_csv(test_path),
        pd.read_csv(synthetic_path),
    )


def make_train_frame(
    real_train: pd.DataFrame,
    synthetic: pd.DataFrame | None,
    generator: str,
    mode: str,
) -> pd.DataFrame:
    if generator == "original":
        return real_train.copy()
    if synthetic is None:
        raise ValueError("synthetic must be provided for non-original generators")
    if mode == "augmentation":
        return pd.concat([real_train, synthetic], ignore_index=True)
    return synthetic.copy()


def make_model(name: str, seed: int, device: str):
    if name == "xgb":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=1.0,
            colsample_bytree=1.0,
            eval_metric="logloss",
            random_state=seed,
            tree_method="hist",
            device="cuda" if device.startswith("cuda") else "cpu",
        )
    if name == "rf":
        return RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    if name == "lr":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, random_state=seed, n_jobs=-1),
        )
    if name == "tabpfn":
        from tabpfn import TabPFNClassifier

        return TabPFNClassifier(device=device, random_state=seed, n_estimators=1)
    raise ValueError(name)


def predict_proba(model, X_test: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)
    raise TypeError(f"Model {type(model)!r} does not expose predict_proba")


def auc_score(y_test_encoded: np.ndarray, proba: np.ndarray, n_classes: int) -> float:
    if n_classes == 2:
        return float(roc_auc_score(y_test_encoded, proba[:, 1]))
    return float(
        roc_auc_score(
            y_test_encoded,
            proba,
            multi_class="ovr",
            average="macro",
        )
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    mode_for_path = "original" if args.generator == "original" else args.mode
    out_dir = (
        args.output_root
        / str(args.dataset_id)
        / f"seed_{args.seed}"
        / args.generator
        / mode_for_path
        / args.downstream_model
    )
    result_path = out_dir / "metrics.json"
    if result_path.exists() and not args.overwrite:
        print(f"Skipping existing result: {result_path}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.generator == "original":
        real_train, test_df = load_original_split(
            args.data_root, args.dataset_id, args.seed, args.test_size
        )
        synthetic = None
    else:
        real_train, test_df, synthetic = load_generated_split(
            args.generator_root, args.dataset_id, args.seed, args.generator
        )

    train_df = make_train_frame(real_train, synthetic, args.generator, args.mode)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df["target"])
    y_test = label_encoder.transform(test_df["target"])
    X_train = train_df.drop(columns=["target"]).to_numpy(dtype=np.float32)
    X_test = test_df.drop(columns=["target"]).to_numpy(dtype=np.float32)

    model = make_model(args.downstream_model, args.seed, args.device)
    model.fit(X_train, y_train)
    proba = predict_proba(model, X_test)
    auc = auc_score(y_test, proba, len(label_encoder.classes_))

    metrics = {
        "dataset_id": args.dataset_id,
        "seed": args.seed,
        "generator": args.generator,
        "mode": mode_for_path,
        "downstream_model": args.downstream_model,
        "auc": auc,
        "n_train_real": int(len(real_train)),
        "n_train_used": int(len(train_df)),
        "n_test": int(len(test_df)),
        "classes": [str(c) for c in label_encoder.classes_],
        "result_path": str(result_path),
    }
    result_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
