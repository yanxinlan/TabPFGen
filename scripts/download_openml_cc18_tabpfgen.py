#!/usr/bin/env python3
"""Download the OpenML-CC18 datasets used in the TabPFGen paper.

The script is intended for batch/Slurm use. It downloads the OpenML datasets by
ID, stores features and labels separately, and writes a manifest with the final
paths and dataset metadata. Re-running the script skips datasets that already
have the expected output files unless --overwrite is passed.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class PaperDataset:
    name: str
    openml_id: int
    n_features: int
    n_instances: int
    n_classes: int
    minor_class_size: int


TABPFGEN_OPENML_CC18_DATASETS: tuple[PaperDataset, ...] = (
    PaperDataset("balance-scale", 11, 5, 625, 3, 49),
    PaperDataset("mfeat-fourier", 14, 77, 2000, 10, 200),
    PaperDataset("mfeat-karhunen", 16, 65, 2000, 10, 200),
    PaperDataset("mfeat-morphological", 18, 7, 2000, 10, 200),
    PaperDataset("mfeat-zernike", 22, 48, 2000, 10, 200),
    PaperDataset("diabetes", 37, 9, 768, 2, 268),
    PaperDataset("vehicle", 54, 19, 846, 4, 199),
    PaperDataset("analcatdata_authorship", 458, 71, 841, 4, 55),
    PaperDataset("pc4", 1049, 38, 1458, 2, 178),
    PaperDataset("pc3", 1050, 38, 1563, 2, 160),
    PaperDataset("kc2", 1063, 22, 522, 2, 107),
    PaperDataset("pc1", 1068, 22, 1109, 2, 77),
    PaperDataset("banknote-authentication", 1462, 5, 1372, 2, 610),
    PaperDataset("blood-transfusion-service-center", 1464, 5, 748, 2, 178),
    PaperDataset("qsar-biodeg", 1494, 42, 1055, 2, 356),
    PaperDataset("wdbc", 1510, 31, 569, 2, 212),
    PaperDataset("steel-plates-fault", 40982, 28, 1941, 7, 55),
    PaperDataset("climate-model-simulation-crashes", 40994, 21, 540, 2, 46),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the TabPFGen paper's OpenML-CC18 datasets."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/openml_cc18_tabpfgen"),
        help="Directory where datasets and manifest files will be written.",
    )
    parser.add_argument(
        "--ids",
        type=int,
        nargs="*",
        default=None,
        help="Optional subset of OpenML dataset IDs to download.",
    )
    parser.add_argument(
        "--format",
        choices=("csv", "parquet"),
        default="csv",
        help="Feature/label file format. CSV avoids optional parquet engines.",
    )
    parser.add_argument(
        "--openml-cache-dir",
        type=Path,
        default=None,
        help="Optional OpenML cache directory, useful on Slurm scratch storage.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download/rewrite outputs even if files already exist.",
    )
    return parser.parse_args()


def selected_datasets(ids: Iterable[int] | None) -> list[PaperDataset]:
    if ids is None:
        return list(TABPFGEN_OPENML_CC18_DATASETS)
    requested = set(ids)
    known = {dataset.openml_id: dataset for dataset in TABPFGEN_OPENML_CC18_DATASETS}
    unknown = sorted(requested - set(known))
    if unknown:
        raise ValueError(f"IDs not in the TabPFGen paper table: {unknown}")
    return [known[openml_id] for openml_id in sorted(requested)]


def safe_name(dataset: PaperDataset) -> str:
    return f"{dataset.openml_id}_{dataset.name}".replace("/", "_")


def write_frame(df: pd.DataFrame, path: Path, fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        df.to_csv(path, index=False)
    else:
        df.to_parquet(path, index=False)


def output_paths(base_dir: Path, dataset: PaperDataset, fmt: str) -> dict[str, Path]:
    suffix = "csv" if fmt == "csv" else "parquet"
    dataset_dir = base_dir / safe_name(dataset)
    return {
        "dataset_dir": dataset_dir,
        "X": dataset_dir / f"X.{suffix}",
        "y": dataset_dir / f"y.{suffix}",
        "combined": dataset_dir / f"data.{suffix}",
        "metadata": dataset_dir / "metadata.json",
    }


def download_one(
    paper_dataset: PaperDataset,
    output_dir: Path,
    fmt: str,
    overwrite: bool,
) -> dict[str, object]:
    import openml

    paths = output_paths(output_dir, paper_dataset, fmt)
    expected_files = [paths["X"], paths["y"], paths["combined"], paths["metadata"]]
    if not overwrite and all(path.exists() for path in expected_files):
        return {
            **asdict(paper_dataset),
            "status": "skipped",
            "reason": "outputs already exist",
            "dataset_dir": str(paths["dataset_dir"]),
            "X_path": str(paths["X"]),
            "y_path": str(paths["y"]),
            "combined_path": str(paths["combined"]),
            "metadata_path": str(paths["metadata"]),
        }

    dataset = openml.datasets.get_dataset(
        paper_dataset.openml_id,
        download_data=True,
        download_qualities=True,
        download_features_meta_data=True,
    )
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe",
        target=dataset.default_target_attribute,
    )
    if y is None:
        raise ValueError(
            f"Dataset {paper_dataset.openml_id} has no default target attribute"
        )

    y_frame = pd.DataFrame({"target": y})
    combined = X.copy()
    combined["target"] = y

    write_frame(X, paths["X"], fmt)
    write_frame(y_frame, paths["y"], fmt)
    write_frame(combined, paths["combined"], fmt)

    metadata = {
        "paper_table": asdict(paper_dataset),
        "openml": {
            "dataset_id": dataset.dataset_id,
            "name": dataset.name,
            "version": dataset.version,
            "default_target_attribute": dataset.default_target_attribute,
            "url": dataset.url,
            "md5_checksum": dataset.md5_checksum,
            "format": dataset.format,
            "licence": getattr(dataset, "licence", None),
        },
        "downloaded_shape": {
            "n_instances": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "n_target_values": int(y_frame.shape[0]),
        },
        "columns": {
            "features": list(attribute_names),
            "categorical_indicator": list(map(bool, categorical_indicator)),
            "target": "target",
        },
        "paths": {
            "X": str(paths["X"]),
            "y": str(paths["y"]),
            "combined": str(paths["combined"]),
        },
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        **asdict(paper_dataset),
        "status": "downloaded",
        "openml_name": dataset.name,
        "dataset_dir": str(paths["dataset_dir"]),
        "X_path": str(paths["X"]),
        "y_path": str(paths["y"]),
        "combined_path": str(paths["combined"]),
        "metadata_path": str(paths["metadata"]),
        "downloaded_n_instances": int(X.shape[0]),
        "downloaded_n_features": int(X.shape[1]),
    }


def main() -> None:
    args = parse_args()
    if args.openml_cache_dir is not None:
        import openml

        args.openml_cache_dir.mkdir(parents=True, exist_ok=True)
        openml.config.set_root_cache_directory(str(args.openml_cache_dir))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset in selected_datasets(args.ids):
        print(f"[{dataset.openml_id}] {dataset.name}", flush=True)
        row = download_one(dataset, args.output_dir, args.format, args.overwrite)
        rows.append(row)
        print(f"  -> {row['status']}: {row['dataset_dir']}", flush=True)

    manifest = pd.DataFrame(rows)
    manifest_path = args.output_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    print(f"Wrote manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
