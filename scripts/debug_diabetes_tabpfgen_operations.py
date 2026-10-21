#!/usr/bin/env python3
"""Trace every differentiable operation in one diabetes TabPFGen SGLD run."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.exists():
    sys.path.insert(0, str(SRC_ROOT))

from tabpfgen import TabPFGen
from tabpfgen.paper_tabpfn import PAPER_CHECKPOINT_SHA256, PAPER_TABPFN_COMMIT
from train_table1_generators import load_dataset, split_train_test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Operation-level trace for the diabetes TabPFGen failure."
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("data/openml_cc18_tabpfgen")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/diabetes_operation_trace")
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-size", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--step-size", type=float, default=0.01)
    parser.add_argument("--noise-scale", type=float, default=0.01)
    parser.add_argument("--init-noise-std", type=float, default=0.01)
    parser.add_argument("--swapped-energy-weight", type=float, default=1.0)
    parser.add_argument("--n-ensembles", type=int, default=3)
    parser.add_argument("--tabpfn-source-root", type=Path, default=None)
    parser.add_argument("--tabpfn-model-path", type=Path, default=None)
    parser.add_argument(
        "--save-mode",
        choices=("full", "nonfinite", "summary"),
        default="full",
        help="full saves every tensor; nonfinite saves only invalid tensors; summary saves statistics.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tensor_stats(value: torch.Tensor) -> dict[str, Any]:
    detached = value.detach()
    if detached.dtype == torch.bool:
        numeric = detached.to(torch.float32)
    elif detached.is_floating_point() or detached.is_complex():
        numeric = detached.real.to(torch.float32)
    else:
        numeric = detached.to(torch.float32)
    finite = torch.isfinite(numeric)
    finite_values = numeric[finite]
    return {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "requires_grad": bool(value.requires_grad),
        "numel": int(detached.numel()),
        "nan_count": int(torch.isnan(numeric).sum().item()),
        "posinf_count": int(torch.isposinf(numeric).sum().item()),
        "neginf_count": int(torch.isneginf(numeric).sum().item()),
        "finite": bool(finite.all().item()),
        "min": float(finite_values.min().item()) if finite_values.numel() else None,
        "max": float(finite_values.max().item()) if finite_values.numel() else None,
        "mean": float(finite_values.mean().item()) if finite_values.numel() else None,
        "std": (
            float(finite_values.std(unbiased=False).item())
            if finite_values.numel()
            else None
        ),
    }


class OperationRecorder:
    def __init__(self, step_dir: Path, save_mode: str) -> None:
        self.step_dir = step_dir
        self.tensor_dir = step_dir / "tensors"
        self.tensor_dir.mkdir(parents=True, exist_ok=True)
        self.save_mode = save_mode
        self.records: list[dict[str, Any]] = []
        self.gradient_count = 0

    @staticmethod
    def _safe_name(name: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)

    def _save(self, sequence: int, name: str, kind: str, value: torch.Tensor) -> str:
        filename = f"{sequence:05d}_{self._safe_name(name)}_{kind}.pt"
        path = self.tensor_dir / filename
        torch.save(value.detach().cpu(), path)
        return str(path)

    def _should_save(self, stats: dict[str, Any]) -> bool:
        return self.save_mode == "full" or (
            self.save_mode == "nonfinite" and not stats["finite"]
        )

    def __call__(self, name: str, value: torch.Tensor) -> None:
        sequence = len(self.records)
        stats = tensor_stats(value)
        record = {"sequence": sequence, "operation": name, **stats}
        record["value_path"] = (
            self._save(sequence, name, "value", value)
            if self._should_save(stats)
            else None
        )
        self.records.append(record)

        if value.requires_grad:

            def save_gradient(gradient: torch.Tensor, record=record) -> torch.Tensor:
                grad_stats = tensor_stats(gradient)
                record["gradient_observed_order"] = self.gradient_count
                self.gradient_count += 1
                record.update(
                    {f"gradient_{key}": val for key, val in grad_stats.items()}
                )
                record["gradient_path"] = (
                    self._save(sequence, name, "gradient", gradient)
                    if self._should_save(grad_stats)
                    else None
                )
                return gradient

            value.register_hook(save_gradient)

    def write(self, error: str | None = None) -> dict[str, Any]:
        frame = pd.DataFrame(self.records)
        frame.to_csv(self.step_dir / "operations.csv", index=False)
        (self.step_dir / "operations.json").write_text(
            json.dumps(self.records, indent=2), encoding="utf-8"
        )
        bad_forward = [row for row in self.records if row.get("finite") is False]
        bad_gradient = [
            row for row in self.records if row.get("gradient_finite") is False
        ]
        bad_gradient.sort(key=lambda row: row["gradient_observed_order"])
        summary = {
            "error": error,
            "operation_count": len(self.records),
            "first_nonfinite_forward": bad_forward[0] if bad_forward else None,
            "first_nonfinite_gradient": bad_gradient[0] if bad_gradient else None,
        }
        (self.step_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        return summary


def iter_tensors(value: Any, prefix: str):
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from iter_tensors(item, f"{prefix}.{index}")
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from iter_tensors(item, f"{prefix}.{key}")


def install_leaf_hooks(
    model: torch.nn.Module, recorder: OperationRecorder, prefix: str
):
    handles = []
    for name, module in model.named_modules():
        if name and not any(module.children()):

            def hook(_module, _inputs, output, name=name):
                for tensor_name, tensor in iter_tensors(
                    output, f"{prefix}.module.{name}.output"
                ):
                    recorder(tensor_name, tensor)

            handles.append(module.register_forward_hook(hook))
    return handles


def remove_hooks(handles) -> None:
    for handle in handles:
        handle.remove()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    run_dir = args.output_root / "37_diabetes" / f"seed_{args.seed}"
    if run_dir.exists() and any(run_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Trace already exists at {run_dir}; use --overwrite or a different output root"
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    data = load_dataset(args.data_root, 37)
    train, _ = split_train_test(data, args.seed, args.test_size)
    x = train.drop(columns=["target"]).to_numpy(dtype=np.float32)
    y = train["target"].to_numpy()

    generator = TabPFGen(
        n_sgld_steps=args.steps,
        sgld_step_size=args.step_size,
        sgld_noise_scale=args.noise_scale,
        init_noise_std=args.init_noise_std,
        device=args.device,
        scale_features=False,
        init_strategy="paired",
        n_estimators=args.n_ensembles,
        model_path=(
            None if args.tabpfn_model_path is None else str(args.tabpfn_model_path)
        ),
        paper_tabpfn_root=(
            None if args.tabpfn_source_root is None else str(args.tabpfn_source_root)
        ),
        random_state=args.seed,
        swapped_energy_weight=args.swapped_energy_weight,
    )
    generator.fit(x, y)
    x_train = generator._x_train_
    y_train = generator._y_train_
    y_synth = y_train.detach().clone()
    x_synth = generator._init_from_training_rows(x_train, y_train, y_synth)

    run_summary: list[dict[str, Any]] = []
    for step in range(args.steps):
        step_dir = run_dir / f"step_{step + 1:04d}"
        recorder = OperationRecorder(step_dir, args.save_mode)
        x_req = x_synth.detach().clone().requires_grad_(True)
        recorder("sgld.x_input", x_req)
        error = None
        try:
            with torch.autograd.detect_anomaly(check_nan=True):
                primary = generator.energy_model_
                primary.classifier.set_observer(recorder, "primary")
                handles = install_leaf_hooks(
                    primary.classifier.model, recorder, "primary"
                )
                primary_logits = primary.logits(x_req)
                remove_hooks(handles)
                rows = torch.arange(x_req.shape[0], device=x_req.device)
                primary_energy = -primary_logits[rows, y_synth]
                recorder("primary.target_energy", primary_energy)

                swapped = generator._make_energy_model(x_req, y_synth)
                swapped.classifier.set_observer(recorder, "swapped")
                handles = install_leaf_hooks(
                    swapped.classifier.model, recorder, "swapped"
                )
                swapped_logits = swapped.logits(x_train)
                remove_hooks(handles)
                train_rows = torch.arange(x_train.shape[0], device=x_train.device)
                swapped_energy = -swapped_logits[train_rows, y_train].mean()
                recorder("swapped.mean_target_energy", swapped_energy)

                total_energy = (
                    primary_energy
                    + args.swapped_energy_weight * swapped_energy
                )
                recorder("sgld.total_energy", total_energy)
                gradient = torch.autograd.grad(total_energy.sum(), x_req)[0]
            recorder("sgld.input_gradient", gradient)

            noise = torch.randn_like(x_req) * args.noise_scale
            recorder("sgld.noise", noise)
            gradient_update = args.step_size * gradient
            recorder("sgld.gradient_update", gradient_update)
            x_synth = (x_req - gradient_update + noise).detach()
            recorder("sgld.x_next", x_synth)
        except Exception as exc:
            error = repr(exc)
        finally:
            if "handles" in locals():
                remove_hooks(handles)
            if "primary" in locals():
                primary.classifier.set_observer(None)
            if "swapped" in locals():
                swapped.classifier.set_observer(None)

        summary = recorder.write(error)
        summary["step"] = step + 1
        run_summary.append(summary)
        if (
            error
            or summary["first_nonfinite_forward"]
            or summary["first_nonfinite_gradient"]
        ):
            break

    manifest = {
        "dataset_id": 37,
        "dataset_name": "diabetes",
        "seed": args.seed,
        "train_shape": list(x.shape),
        "scale_features": False,
        "init_strategy": "paired",
        "paper_tabpfn_commit": PAPER_TABPFN_COMMIT,
        "paper_checkpoint_sha256": PAPER_CHECKPOINT_SHA256,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "steps": run_summary,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
