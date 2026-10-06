#!/usr/bin/env python3
"""Analyze TabPFGen reproduction outputs.

This script reads the generated files produced by train_table1_generators.py and
evaluate_table1_downstream.py. It writes compact CSV reports for:

1. Table 1 reproduction vs. paper values.
2. Dataset/model/method slices that identify where reproduction diverges.
3. Synthetic-data diagnostics: marginal fidelity, dependency preservation, and
   nearest-neighbor privacy risk.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler


DATASET_NAMES = {
    11: "balance-scale",
    14: "mfeat-fourier",
    16: "mfeat-karhunen",
    18: "mfeat-morphological",
    22: "mfeat-zernike",
    37: "diabetes",
    54: "vehicle",
    458: "analcatdata_authorship",
    1049: "pc4",
    1050: "pc3",
    1063: "kc2",
    1068: "pc1",
    1462: "banknote-authentication",
    1464: "blood-transfusion-service-center",
    1494: "qsar-biodeg",
    1510: "wdbc",
    40982: "steel-plates-fault",
    40994: "climate-model-simulation-crashes",
}

PAPER_TABLE1_AUC = {
    ("augmentation", "xgb", "original"): 0.924,
    ("augmentation", "xgb", "smote"): 0.926,
    ("augmentation", "xgb", "ctgan"): 0.912,
    ("augmentation", "xgb", "tvae"): 0.914,
    ("augmentation", "xgb", "nflow"): 0.912,
    ("augmentation", "xgb", "rtvae"): 0.917,
    ("augmentation", "xgb", "tabddpm"): 0.927,
    ("augmentation", "xgb", "tabpfgen"): 0.934,
    ("augmentation", "rf", "original"): 0.906,
    ("augmentation", "rf", "smote"): 0.906,
    ("augmentation", "rf", "ctgan"): 0.898,
    ("augmentation", "rf", "tvae"): 0.904,
    ("augmentation", "rf", "nflow"): 0.894,
    ("augmentation", "rf", "rtvae"): 0.907,
    ("augmentation", "rf", "tabddpm"): 0.911,
    ("augmentation", "rf", "tabpfgen"): 0.912,
    ("augmentation", "lr", "original"): 0.920,
    ("augmentation", "lr", "smote"): 0.914,
    ("augmentation", "lr", "ctgan"): 0.904,
    ("augmentation", "lr", "tvae"): 0.909,
    ("augmentation", "lr", "nflow"): 0.901,
    ("augmentation", "lr", "rtvae"): 0.906,
    ("augmentation", "lr", "tabddpm"): 0.885,
    ("augmentation", "lr", "tabpfgen"): 0.921,
    ("augmentation", "tabpfn", "original"): 0.934,
    ("augmentation", "tabpfn", "smote"): 0.927,
    ("augmentation", "tabpfn", "ctgan"): 0.930,
    ("augmentation", "tabpfn", "tvae"): 0.931,
    ("augmentation", "tabpfn", "nflow"): 0.928,
    ("augmentation", "tabpfn", "rtvae"): 0.932,
    ("augmentation", "tabpfn", "tabddpm"): 0.929,
    ("augmentation", "tabpfn", "tabpfgen"): 0.935,
    ("replacement", "xgb", "smote"): 0.907,
    ("replacement", "xgb", "ctgan"): 0.842,
    ("replacement", "xgb", "tvae"): 0.858,
    ("replacement", "xgb", "nflow"): 0.700,
    ("replacement", "xgb", "rtvae"): 0.795,
    ("replacement", "xgb", "tabddpm"): 0.812,
    ("replacement", "xgb", "tabpfgen"): 0.927,
    ("replacement", "rf", "smote"): 0.894,
    ("replacement", "rf", "ctgan"): 0.837,
    ("replacement", "rf", "tvae"): 0.844,
    ("replacement", "rf", "nflow"): 0.676,
    ("replacement", "rf", "rtvae"): 0.774,
    ("replacement", "rf", "tabddpm"): 0.814,
    ("replacement", "rf", "tabpfgen"): 0.906,
    ("replacement", "lr", "smote"): 0.893,
    ("replacement", "lr", "ctgan"): 0.843,
    ("replacement", "lr", "tvae"): 0.873,
    ("replacement", "lr", "nflow"): 0.722,
    ("replacement", "lr", "rtvae"): 0.854,
    ("replacement", "lr", "tabddpm"): 0.876,
    ("replacement", "lr", "tabpfgen"): 0.920,
    ("replacement", "tabpfn", "smote"): 0.920,
    ("replacement", "tabpfn", "ctgan"): 0.888,
    ("replacement", "tabpfn", "tvae"): 0.887,
    ("replacement", "tabpfn", "nflow"): 0.705,
    ("replacement", "tabpfn", "rtvae"): 0.862,
    ("replacement", "tabpfn", "tabddpm"): 0.894,
    ("replacement", "tabpfn", "tabpfgen"): 0.934,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Table 1 reproduction outputs.")
    parser.add_argument("--generator-root", type=Path, default=Path("outputs/table1_generators"))
    parser.add_argument("--downstream-root", type=Path, default=Path("outputs/table1_downstream"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/analysis"))
    parser.add_argument(
        "--diagnostic-generators",
        nargs="*",
        default=("smote", "ctgan", "tvae", "nflow", "rtvae", "tabddpm", "tabpfgen"),
        help="Generators to include in synthetic-data diagnostics.",
    )
    parser.add_argument(
        "--skip-c2st",
        action="store_true",
        help="Skip classifier two-sample tests for faster analysis.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_metrics_files(root: Path) -> Iterable[Path]:
    yield from sorted(root.glob("*/*/*/*/*/metrics.json"))


def load_downstream_metrics(root: Path) -> pd.DataFrame:
    rows = []
    for path in iter_metrics_files(root):
        row = read_json(path)
        row["metrics_path"] = str(path)
        row["dataset_name"] = DATASET_NAMES.get(int(row["dataset_id"]), "")
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def summarize_table1(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if metrics.empty:
        return pd.DataFrame(), pd.DataFrame()

    grouped = (
        metrics.groupby(["mode", "downstream_model", "generator"], as_index=False)
        .agg(
            reproduced_auc_mean=("auc", "mean"),
            reproduced_auc_std=("auc", "std"),
            n_runs=("auc", "size"),
        )
        .sort_values(["mode", "downstream_model", "generator"])
    )
    paper_rows = [
        {
            "mode": mode,
            "downstream_model": model,
            "generator": generator,
            "paper_auc": paper_auc,
        }
        for (mode, model, generator), paper_auc in PAPER_TABLE1_AUC.items()
    ]
    paper = pd.DataFrame(paper_rows)
    comparison = paper.merge(grouped, how="left", on=["mode", "downstream_model", "generator"])
    comparison["delta_vs_paper"] = (
        comparison["reproduced_auc_mean"] - comparison["paper_auc"]
    )
    comparison["abs_delta_vs_paper"] = comparison["delta_vs_paper"].abs()
    comparison = comparison.sort_values(
        ["mode", "downstream_model", "generator"], ignore_index=True
    )

    slices = metrics.copy()
    original = slices[slices["generator"] == "original"][
        ["dataset_id", "seed", "downstream_model", "auc"]
    ].rename(columns={"auc": "original_auc"})
    synthetic = slices[slices["generator"] != "original"].merge(
        original,
        on=["dataset_id", "seed", "downstream_model"],
        how="left",
    )
    synthetic["delta_vs_original"] = synthetic["auc"] - synthetic["original_auc"]
    return comparison, synthetic


def numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["target"]).apply(pd.to_numeric, errors="coerce")


def class_distribution_tvd(real: pd.Series, synth: pd.Series) -> float:
    labels = sorted(set(real.astype(str)) | set(synth.astype(str)))
    real_p = real.astype(str).value_counts(normalize=True).reindex(labels, fill_value=0.0)
    synth_p = synth.astype(str).value_counts(normalize=True).reindex(labels, fill_value=0.0)
    return float(0.5 * np.abs(real_p.to_numpy() - synth_p.to_numpy()).sum())


def mean_ks(real_x: pd.DataFrame, synth_x: pd.DataFrame) -> float:
    scores = []
    for column in real_x.columns:
        real_col = real_x[column].dropna().to_numpy()
        synth_col = synth_x[column].dropna().to_numpy()
        if real_col.size == 0 or synth_col.size == 0:
            continue
        scores.append(ks_2samp(real_col, synth_col).statistic)
    return float(np.mean(scores)) if scores else np.nan


def correlation_mae(real_x: pd.DataFrame, synth_x: pd.DataFrame) -> float:
    real_corr = real_x.corr(numeric_only=True).fillna(0.0)
    synth_corr = synth_x.corr(numeric_only=True).fillna(0.0)
    common = real_corr.index.intersection(synth_corr.index)
    if len(common) < 2:
        return np.nan
    diff = real_corr.loc[common, common].to_numpy() - synth_corr.loc[common, common].to_numpy()
    mask = ~np.eye(len(common), dtype=bool)
    return float(np.mean(np.abs(diff[mask])))


def nearest_neighbor_privacy(real_x: pd.DataFrame, synth_x: pd.DataFrame) -> dict[str, float]:
    common = real_x.columns.intersection(synth_x.columns)
    real = real_x[common].replace([np.inf, -np.inf], np.nan).dropna(axis=1)
    synth = synth_x[real.columns].replace([np.inf, -np.inf], np.nan).dropna(axis=1)
    common = real.columns.intersection(synth.columns)
    if len(common) == 0 or len(real) < 2 or len(synth) == 0:
        return {
            "nn_min_distance": np.nan,
            "nn_median_distance": np.nan,
            "nn_distance_p01": np.nan,
            "exact_duplicate_fraction": np.nan,
        }
    real = real[common].to_numpy(dtype=np.float64)
    synth = synth[common].to_numpy(dtype=np.float64)
    scaler = StandardScaler().fit(real)
    real_scaled = scaler.transform(real)
    synth_scaled = scaler.transform(synth)
    nbrs = NearestNeighbors(n_neighbors=1).fit(real_scaled)
    distances = nbrs.kneighbors(synth_scaled, return_distance=True)[0][:, 0]
    return {
        "nn_min_distance": float(np.min(distances)),
        "nn_median_distance": float(np.median(distances)),
        "nn_distance_p01": float(np.quantile(distances, 0.01)),
        "exact_duplicate_fraction": float(np.mean(distances < 1e-8)),
    }


def c2st_auc(real_x: pd.DataFrame, synth_x: pd.DataFrame, seed: int) -> float:
    common = real_x.columns.intersection(synth_x.columns)
    if len(common) == 0 or len(real_x) < 4 or len(synth_x) < 4:
        return np.nan
    real = real_x[common].copy()
    synth = synth_x[common].copy()
    n = min(len(real), len(synth))
    real = real.sample(n=n, random_state=seed)
    synth = synth.sample(n=n, random_state=seed)
    X = pd.concat([real, synth], ignore_index=True).replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    y = np.concatenate([np.zeros(n), np.ones(n)])
    X_train, X_test, y_train, y_test = train_test_split(
        X.to_numpy(dtype=np.float32),
        y,
        test_size=0.4,
        random_state=seed,
        stratify=y,
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    clf.fit(X_train, y_train)
    return float(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))


def diagnostic_one(base: Path, dataset_id: int, seed: int, generator: str, skip_c2st: bool) -> dict:
    train = pd.read_csv(base / "train.csv")
    test = pd.read_csv(base / "test.csv")
    synth = pd.read_csv(base / "synthetic.csv")
    real_x = numeric_features(train)
    synth_x = numeric_features(synth)
    privacy = nearest_neighbor_privacy(real_x, synth_x)
    row = {
        "dataset_id": dataset_id,
        "dataset_name": DATASET_NAMES.get(dataset_id, ""),
        "seed": seed,
        "generator": generator,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_synthetic": int(len(synth)),
        "label_tvd": class_distribution_tvd(train["target"], synth["target"]),
        "mean_feature_ks": mean_ks(real_x, synth_x),
        "correlation_mae": correlation_mae(real_x, synth_x),
        **privacy,
    }
    row["c2st_auc"] = np.nan if skip_c2st else c2st_auc(real_x, synth_x, seed)
    return row


def load_diagnostics(
    generator_root: Path, generators: Iterable[str], skip_c2st: bool
) -> pd.DataFrame:
    rows = []
    wanted = set(generators)
    for base in sorted(generator_root.glob("*/*/*")):
        if not base.is_dir():
            continue
        generator = base.name
        if generator not in wanted:
            continue
        try:
            dataset_id = int(base.parents[1].name)
            seed = int(base.parent.name.removeprefix("seed_"))
        except ValueError:
            continue
        required = [base / "train.csv", base / "test.csv", base / "synthetic.csv"]
        if not all(path.exists() for path in required):
            continue
        rows.append(diagnostic_one(base, dataset_id, seed, generator, skip_c2st))
    return pd.DataFrame(rows)


def summarize_diagnostics(diagnostics: pd.DataFrame) -> pd.DataFrame:
    if diagnostics.empty:
        return pd.DataFrame()
    numeric_cols = [
        "label_tvd",
        "mean_feature_ks",
        "correlation_mae",
        "nn_min_distance",
        "nn_median_distance",
        "nn_distance_p01",
        "exact_duplicate_fraction",
        "c2st_auc",
    ]
    summary = (
        diagnostics.groupby("generator", as_index=False)[numeric_cols]
        .agg(["mean", "std", "median"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in col if part) if isinstance(col, tuple) else col
        for col in summary.columns
    ]
    return summary


def write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Wrote {path} ({len(df)} rows)")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_downstream_metrics(args.downstream_root)
    write(metrics, args.output_dir / "downstream_metrics_long.csv")

    table1, dataset_slices = summarize_table1(metrics)
    write(table1, args.output_dir / "table1_vs_paper.csv")
    write(dataset_slices, args.output_dir / "dataset_model_slices.csv")

    if not dataset_slices.empty:
        worst = dataset_slices.sort_values("delta_vs_original").head(50)
        best = dataset_slices.sort_values("delta_vs_original", ascending=False).head(50)
        write(worst, args.output_dir / "worst_delta_vs_original.csv")
        write(best, args.output_dir / "best_delta_vs_original.csv")

    diagnostics = load_diagnostics(
        args.generator_root, args.diagnostic_generators, args.skip_c2st
    )
    write(diagnostics, args.output_dir / "synthetic_diagnostics_long.csv")
    write(summarize_diagnostics(diagnostics), args.output_dir / "synthetic_diagnostics_summary.csv")


if __name__ == "__main__":
    main()
