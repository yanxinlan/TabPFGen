#!/usr/bin/env python3
"""Extra evaluation-only diagnostics for generated synthetic data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler

from analyze_reproduction_results import DATASET_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run extra synthetic-data diagnostics.")
    parser.add_argument("--generator-root", type=Path, default=Path("outputs/table1_generators"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/analysis_extra"))
    parser.add_argument(
        "--generators",
        nargs="*",
        default=("smote", "ctgan", "tvae", "nflow", "rtvae", "tabddpm", "tabpfgen"),
    )
    parser.add_argument("--max-rows", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=0)
    return parser.parse_args()


def numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["target"]).apply(pd.to_numeric, errors="coerce")


def clean_pair(real_x: pd.DataFrame, synth_x: pd.DataFrame):
    common = real_x.columns.intersection(synth_x.columns)
    real = real_x[common].replace([np.inf, -np.inf], np.nan)
    synth = synth_x[common].replace([np.inf, -np.inf], np.nan)
    med = real.median(numeric_only=True)
    real = real.fillna(med).fillna(0.0)
    synth = synth.fillna(med).fillna(0.0)
    return real, synth


def conditional_ks(train: pd.DataFrame, synth: pd.DataFrame) -> float:
    scores = []
    real_x = numeric_features(train)
    synth_x = numeric_features(synth)
    for label in sorted(set(train["target"].astype(str)) & set(synth["target"].astype(str))):
        real_mask = train["target"].astype(str) == label
        synth_mask = synth["target"].astype(str) == label
        r, s = clean_pair(real_x.loc[real_mask], synth_x.loc[synth_mask])
        if len(r) == 0 or len(s) == 0:
            continue
        for col in r.columns:
            scores.append(ks_2samp(r[col].to_numpy(), s[col].to_numpy()).statistic)
    return float(np.mean(scores)) if scores else np.nan


def pca_spectrum_l1(real_x: pd.DataFrame, synth_x: pd.DataFrame) -> float:
    r, s = clean_pair(real_x, synth_x)
    if r.shape[1] < 2 or len(r) < 3 or len(s) < 3:
        return np.nan
    scaler = StandardScaler().fit(r)
    r_scaled = scaler.transform(r)
    s_scaled = scaler.transform(s)
    n_components = min(r.shape[1], len(r) - 1, len(s) - 1)
    real_ev = PCA(n_components=n_components).fit(r_scaled).explained_variance_ratio_
    synth_ev = PCA(n_components=n_components).fit(s_scaled).explained_variance_ratio_
    return float(np.abs(real_ev - synth_ev).sum())


def partial_correlation_mae(real_x: pd.DataFrame, synth_x: pd.DataFrame) -> float:
    r, s = clean_pair(real_x, synth_x)
    if r.shape[1] < 3:
        return np.nan
    def partial_corr(x):
        corr = np.corrcoef(x, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)
        precision = np.linalg.pinv(corr)
        diag = np.sqrt(np.clip(np.diag(precision), 1e-12, None))
        pcorr = -precision / np.outer(diag, diag)
        np.fill_diagonal(pcorr, 1.0)
        return pcorr
    diff = partial_corr(r.to_numpy()) - partial_corr(s.to_numpy())
    mask = ~np.eye(diff.shape[0], dtype=bool)
    return float(np.mean(np.abs(diff[mask])))


def mutual_information_mae(train: pd.DataFrame, synth: pd.DataFrame, seed: int) -> float:
    real_x = numeric_features(train)
    synth_x = numeric_features(synth)
    r, s = clean_pair(real_x, synth_x)
    if r.shape[1] < 2:
        return np.nan
    label_encoder = LabelEncoder()
    all_labels = pd.concat(
        [train["target"].astype(str), synth["target"].astype(str)], ignore_index=True
    )
    label_encoder.fit(all_labels)
    y_real = label_encoder.transform(train["target"].astype(str))
    y_synth = label_encoder.transform(synth["target"].astype(str))
    mi_real = mutual_info_classif(r.to_numpy(), y_real, random_state=seed)
    mi_synth = mutual_info_classif(s.to_numpy(), y_synth, random_state=seed)
    return float(np.mean(np.abs(mi_real - mi_synth)))


def manifold_metrics(real_x: pd.DataFrame, synth_x: pd.DataFrame) -> dict[str, float]:
    r, s = clean_pair(real_x, synth_x)
    if len(r) < 2 or len(s) < 2 or r.shape[1] == 0:
        return {"precision": np.nan, "coverage": np.nan, "density_ratio": np.nan}
    scaler = StandardScaler().fit(r)
    r_scaled = scaler.transform(r)
    s_scaled = scaler.transform(s)
    real_nn = NearestNeighbors(n_neighbors=2).fit(r_scaled)
    real_radius = np.median(real_nn.kneighbors(r_scaled, return_distance=True)[0][:, 1])
    synth_to_real = NearestNeighbors(n_neighbors=1).fit(r_scaled).kneighbors(
        s_scaled, return_distance=True
    )[0][:, 0]
    real_to_synth = NearestNeighbors(n_neighbors=1).fit(s_scaled).kneighbors(
        r_scaled, return_distance=True
    )[0][:, 0]
    return {
        "precision": float(np.mean(synth_to_real <= real_radius)),
        "coverage": float(np.mean(real_to_synth <= real_radius)),
        "density_ratio": float(np.median(synth_to_real) / (real_radius + 1e-12)),
    }


def minority_label(train: pd.DataFrame) -> str:
    return str(train["target"].astype(str).value_counts().idxmin())


def minority_coverage(train: pd.DataFrame, synth: pd.DataFrame) -> float:
    label = minority_label(train)
    real_mask = train["target"].astype(str) == label
    synth_mask = synth["target"].astype(str) == label
    real_x, synth_x = clean_pair(numeric_features(train.loc[real_mask]), numeric_features(synth.loc[synth_mask]))
    if len(real_x) < 2 or len(synth_x) == 0:
        return np.nan
    scaler = StandardScaler().fit(real_x)
    r = scaler.transform(real_x)
    s = scaler.transform(synth_x)
    radius = np.median(NearestNeighbors(n_neighbors=2).fit(r).kneighbors(r)[0][:, 1])
    d = NearestNeighbors(n_neighbors=1).fit(s).kneighbors(r)[0][:, 0]
    return float(np.mean(d <= radius))


def evaluate_one(base: Path, dataset_id: int, seed: int, generator: str, max_rows: int) -> dict:
    train = pd.read_csv(base / "train.csv")
    synth = pd.read_csv(base / "synthetic.csv")
    if len(train) > max_rows:
        train = train.sample(max_rows, random_state=seed).reset_index(drop=True)
    if len(synth) > max_rows:
        synth = synth.sample(max_rows, random_state=seed).reset_index(drop=True)
    real_x = numeric_features(train)
    synth_x = numeric_features(synth)
    row = {
        "dataset_id": dataset_id,
        "dataset_name": DATASET_NAMES.get(dataset_id, ""),
        "seed": seed,
        "generator": generator,
        "conditional_ks": conditional_ks(train, synth),
        "pca_spectrum_l1": pca_spectrum_l1(real_x, synth_x),
        "partial_corr_mae": partial_correlation_mae(real_x, synth_x),
        "feature_target_mi_mae": mutual_information_mae(train, synth, seed),
        "minority_coverage": minority_coverage(train, synth),
    }
    row.update(manifold_metrics(real_x, synth_x))
    return row


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    wanted = set(args.generators)
    for base in sorted(args.generator_root.glob("*/*/*")):
        if not base.is_dir() or base.name not in wanted:
            continue
        required = [base / "train.csv", base / "synthetic.csv"]
        if not all(path.exists() for path in required):
            continue
        try:
            dataset_id = int(base.parents[1].name)
            seed = int(base.parent.name.removeprefix("seed_"))
        except ValueError:
            continue
        rows.append(evaluate_one(base, dataset_id, seed, base.name, args.max_rows))
    long = pd.DataFrame(rows)
    long_path = args.output_dir / "extra_synthetic_diagnostics_long.csv"
    long.to_csv(long_path, index=False)
    print(f"Wrote {long_path} ({len(long)} rows)")
    if not long.empty:
        summary = long.groupby("generator", as_index=False).agg(
            {
                "conditional_ks": ["mean", "median"],
                "pca_spectrum_l1": ["mean", "median"],
                "partial_corr_mae": ["mean", "median"],
                "feature_target_mi_mae": ["mean", "median"],
                "precision": ["mean", "median"],
                "coverage": ["mean", "median"],
                "minority_coverage": ["mean", "median"],
            }
        )
        summary.columns = [
            "_".join(str(part) for part in col if part) if isinstance(col, tuple) else col
            for col in summary.columns
        ]
        summary_path = args.output_dir / "extra_synthetic_diagnostics_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"Wrote {summary_path} ({len(summary)} rows)")


if __name__ == "__main__":
    main()
