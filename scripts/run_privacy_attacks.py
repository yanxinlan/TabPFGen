#!/usr/bin/env python3
"""Privacy attacks/evaluators for synthetic tabular data.

Outputs distance-based membership risk, RF membership attack AUC, and attribute
inference scores. This trains attack/evaluator models only; it does not train
or overwrite synthetic generators.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler

from analyze_reproduction_results import DATASET_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run privacy attacks on synthetic data.")
    parser.add_argument("--generator-root", type=Path, default=Path("outputs/table1_generators"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/privacy_attacks"))
    parser.add_argument(
        "--generators",
        nargs="*",
        default=("smote", "ctgan", "tvae", "nflow", "rtvae", "tabddpm", "tabpfgen"),
    )
    parser.add_argument("--max-rows", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--n-attribute-features", type=int, default=10)
    return parser.parse_args()


def numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["target"]).apply(pd.to_numeric, errors="coerce")


def clean_with_train(train_x: pd.DataFrame, *frames: pd.DataFrame):
    med = train_x.replace([np.inf, -np.inf], np.nan).median(numeric_only=True)
    out = []
    for frame in frames:
        common = train_x.columns.intersection(frame.columns)
        cleaned = frame[common].replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)
        out.append(cleaned)
    return out


def distance_membership(train_x: pd.DataFrame, test_x: pd.DataFrame, synth_x: pd.DataFrame):
    train_x, test_x, synth_x = clean_with_train(train_x, train_x, test_x, synth_x)
    if len(train_x) < 2 or len(test_x) < 2 or len(synth_x) < 2 or train_x.shape[1] == 0:
        return {}
    scaler = StandardScaler().fit(pd.concat([train_x, test_x], ignore_index=True))
    train = scaler.transform(train_x)
    test = scaler.transform(test_x)
    synth = scaler.transform(synth_x)
    nbrs = NearestNeighbors(n_neighbors=1).fit(synth)
    d_train = nbrs.kneighbors(train, return_distance=True)[0][:, 0]
    d_test = nbrs.kneighbors(test, return_distance=True)[0][:, 0]
    y_true = np.concatenate([np.ones_like(d_train), np.zeros_like(d_test)])
    # Smaller distance means more likely member, so negate distances.
    scores = -np.concatenate([d_train, d_test])
    return {
        "dcr_membership_auc": float(roc_auc_score(y_true, scores)),
        "member_nn_distance_median": float(np.median(d_train)),
        "nonmember_nn_distance_median": float(np.median(d_test)),
        "member_nn_distance_p01": float(np.quantile(d_train, 0.01)),
        "nonmember_nn_distance_p01": float(np.quantile(d_test, 0.01)),
        "member_exact_fraction": float(np.mean(d_train < 1e-8)),
        "nonmember_exact_fraction": float(np.mean(d_test < 1e-8)),
    }


def rf_membership_attack(train_x: pd.DataFrame, test_x: pd.DataFrame, synth_x: pd.DataFrame, seed: int):
    train_x, test_x, synth_x = clean_with_train(train_x, train_x, test_x, synth_x)
    n = min(len(train_x), len(test_x), len(synth_x))
    if n < 20 or train_x.shape[1] == 0:
        return {}
    rng = np.random.default_rng(seed)
    train_x = train_x.iloc[rng.choice(len(train_x), n, replace=False)]
    test_x = test_x.iloc[rng.choice(len(test_x), n, replace=False)]
    synth_x = synth_x.iloc[rng.choice(len(synth_x), n, replace=False)]
    # Attack features are distances to synthetic nearest neighbors plus raw features.
    scaler = StandardScaler().fit(pd.concat([train_x, test_x], ignore_index=True))
    train = scaler.transform(train_x)
    test = scaler.transform(test_x)
    synth = scaler.transform(synth_x)
    nbrs = NearestNeighbors(n_neighbors=min(5, len(synth))).fit(synth)
    d_train = nbrs.kneighbors(train, return_distance=True)[0]
    d_test = nbrs.kneighbors(test, return_distance=True)[0]
    X = np.vstack([np.hstack([train, d_train]), np.hstack([test, d_test])])
    y = np.concatenate([np.ones(n), np.zeros(n)])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.4, random_state=seed, stratify=y)
    clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)[:, 1]
    return {
        "rf_membership_auc": float(roc_auc_score(y_te, proba)),
        "rf_membership_accuracy": float(accuracy_score(y_te, proba >= 0.5)),
    }


def attribute_inference(train: pd.DataFrame, synth: pd.DataFrame, seed: int, n_features: int):
    real_x = numeric_features(train)
    synth_x = numeric_features(synth)
    real_x, synth_x = clean_with_train(real_x, real_x, synth_x)
    if real_x.shape[1] < 2 or len(synth_x) < 20:
        return {}
    variances = real_x.var().sort_values(ascending=False)
    selected = list(variances.head(n_features).index)
    rows = []
    for feature in selected:
        predictors = [c for c in real_x.columns if c != feature]
        X_attack = synth_x[predictors].to_numpy(dtype=np.float32)
        y_attack = synth_x[feature]
        X_eval = real_x[predictors].to_numpy(dtype=np.float32)
        y_eval = real_x[feature]
        unique = y_attack.nunique(dropna=True)
        if unique <= min(10, max(2, len(y_attack) // 20)):
            enc = LabelEncoder()
            enc.fit(pd.concat([y_attack.astype(str), y_eval.astype(str)], ignore_index=True))
            y_attack_enc = enc.transform(y_attack.astype(str))
            y_eval_enc = enc.transform(y_eval.astype(str))
            model = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
            model.fit(X_attack, y_attack_enc)
            score = accuracy_score(y_eval_enc, model.predict(X_eval))
            metric = "accuracy"
        else:
            model = RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1)
            model.fit(X_attack, y_attack.to_numpy(dtype=np.float32))
            score = mean_absolute_error(y_eval.to_numpy(dtype=np.float32), model.predict(X_eval))
            metric = "mae"
        rows.append({"feature": feature, "metric": metric, "score": float(score)})
    if not rows:
        return {}
    out = pd.DataFrame(rows)
    return {
        "attribute_inference_accuracy_mean": float(
            out.loc[out["metric"] == "accuracy", "score"].mean()
        ) if (out["metric"] == "accuracy").any() else np.nan,
        "attribute_inference_mae_mean": float(
            out.loc[out["metric"] == "mae", "score"].mean()
        ) if (out["metric"] == "mae").any() else np.nan,
        "attribute_inference_n_features": int(len(out)),
    }


def evaluate_one(base: Path, dataset_id: int, seed: int, generator: str, args):
    train = pd.read_csv(base / "train.csv")
    test = pd.read_csv(base / "test.csv")
    synth = pd.read_csv(base / "synthetic.csv")
    if len(train) > args.max_rows:
        train = train.sample(args.max_rows, random_state=seed)
    if len(test) > args.max_rows:
        test = test.sample(args.max_rows, random_state=seed)
    if len(synth) > args.max_rows:
        synth = synth.sample(args.max_rows, random_state=seed)
    train_x = numeric_features(train)
    test_x = numeric_features(test)
    synth_x = numeric_features(synth)
    row = {
        "dataset_id": dataset_id,
        "dataset_name": DATASET_NAMES.get(dataset_id, ""),
        "seed": seed,
        "generator": generator,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_synthetic": int(len(synth)),
    }
    row.update(distance_membership(train_x, test_x, synth_x))
    row.update(rf_membership_attack(train_x, test_x, synth_x, seed))
    row.update(attribute_inference(train, synth, seed, args.n_attribute_features))
    return row


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    wanted = set(args.generators)
    rows = []
    for base in sorted(args.generator_root.glob("*/*/*")):
        if not base.is_dir() or base.name not in wanted:
            continue
        required = [base / "train.csv", base / "test.csv", base / "synthetic.csv"]
        if not all(path.exists() for path in required):
            continue
        try:
            dataset_id = int(base.parents[1].name)
            seed = int(base.parent.name.removeprefix("seed_"))
        except ValueError:
            continue
        rows.append(evaluate_one(base, dataset_id, seed, base.name, args))
    long = pd.DataFrame(rows)
    long_path = args.output_dir / "privacy_attacks_long.csv"
    long.to_csv(long_path, index=False)
    print(f"Wrote {long_path} ({len(long)} rows)")
    if not long.empty:
        summary = long.groupby("generator", as_index=False).agg(
            {
                "dcr_membership_auc": ["mean", "median"],
                "rf_membership_auc": ["mean", "median"],
                "rf_membership_accuracy": ["mean", "median"],
                "member_exact_fraction": ["mean", "median"],
                "attribute_inference_accuracy_mean": ["mean", "median"],
                "attribute_inference_mae_mean": ["mean", "median"],
            }
        )
        summary.columns = [
            "_".join(str(part) for part in col if part) if isinstance(col, tuple) else col
            for col in summary.columns
        ]
        summary_path = args.output_dir / "privacy_attacks_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"Wrote {summary_path} ({len(summary)} rows)")


if __name__ == "__main__":
    main()
