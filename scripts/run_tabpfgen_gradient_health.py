#!/usr/bin/env python3
"""Run TabPFGen with gradient-health logging in a separate output tree."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.exists():
    sys.path.insert(0, str(SRC_ROOT))

from tabpfgen import TabPFGen
from train_table1_generators import load_dataset, split_train_test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TabPFGen gradient-health logging.")
    parser.add_argument("--data-root", type=Path, default=Path("data/openml_cc18_tabpfgen"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/gradient_health"))
    parser.add_argument("--dataset-id", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--test-size", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-samples", type=int, default=256)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--step-size", type=float, default=0.01)
    parser.add_argument("--noise-scale", type=float, default=0.01)
    parser.add_argument("--init-noise-std", type=float, default=0.01)
    parser.add_argument("--gradient-clip", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = args.output_root / str(args.dataset_id) / f"seed_{args.seed}"
    trace_path = out_dir / "gradient_trace.csv"
    summary_path = out_dir / "summary.json"
    if trace_path.exists() and summary_path.exists() and not args.overwrite:
        print(f"Skipping existing gradient-health output: {out_dir}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.data_root, args.dataset_id)
    train_df, _ = split_train_test(df, args.seed, args.test_size)
    X = train_df.drop(columns=["target"]).to_numpy(dtype=np.float32)
    y = train_df["target"].to_numpy()
    rng = np.random.default_rng(args.seed)
    n_samples = min(args.n_samples, len(y))
    y_synth = rng.choice(y, size=n_samples, replace=True)

    gen = TabPFGen(
        n_sgld_steps=args.steps,
        sgld_step_size=args.step_size,
        sgld_noise_scale=args.noise_scale,
        init_noise_std=args.init_noise_std,
        device=args.device,
        random_state=args.seed,
    )
    gen.fit(X, y)
    y_synth_encoded = gen.label_encoder.transform(y_synth).astype(np.int64)
    y_synth_t = torch.as_tensor(y_synth_encoded, dtype=torch.long, device=gen.device)
    x_train = gen._x_train_
    y_train = gen._y_train_
    x_synth = gen._init_from_training_rows(x_train, y_train, y_synth_t)

    rows = []
    failure = None
    for step in range(args.steps):
        x_req = x_synth.detach().clone().requires_grad_(True)
        try:
            energy = gen._compute_energy(x_req, y_synth_t, x_train, y_train)
            grad = torch.autograd.grad(energy.sum(), x_req)[0]
            finite_energy = bool(torch.isfinite(energy).all().item())
            finite_grad = bool(torch.isfinite(grad).all().item())
            grad_norm = torch.linalg.vector_norm(
                torch.nan_to_num(grad.detach(), nan=0.0, posinf=0.0, neginf=0.0),
                dim=1,
            )
            row = {
                "step": step + 1,
                "mean_energy": float(torch.nan_to_num(energy.detach()).mean().item()),
                "finite_energy": finite_energy,
                "finite_grad": finite_grad,
                "grad_norm_mean": float(grad_norm.mean().item()),
                "grad_norm_median": float(grad_norm.median().item()),
                "grad_norm_p95": float(torch.quantile(grad_norm, 0.95).item()),
                "grad_norm_max": float(grad_norm.max().item()),
                "x_abs_max": float(torch.nan_to_num(x_req.detach()).abs().max().item()),
            }
            rows.append(row)
            if not finite_energy or not finite_grad:
                failure = row
                break
            if args.gradient_clip > 0:
                norms = torch.linalg.vector_norm(grad, dim=1, keepdim=True).clamp_min(1e-12)
                scale = torch.clamp(args.gradient_clip / norms, max=1.0)
                grad = grad * scale
            noise = torch.randn_like(x_req) * args.noise_scale
            x_synth = (x_req - args.step_size * grad + noise).detach()
            if not torch.isfinite(x_synth).all():
                failure = {**row, "finite_sample": False}
                break
        except Exception as exc:
            failure = {"step": step + 1, "error": repr(exc)}
            rows.append(failure)
            break

    trace = pd.DataFrame(rows)
    trace.to_csv(trace_path, index=False)
    summary = {
        "dataset_id": args.dataset_id,
        "seed": args.seed,
        "n_samples": n_samples,
        "steps_requested": args.steps,
        "steps_completed": int(len(trace)),
        "failed": failure is not None,
        "failure": failure,
        "hyperparameters": vars(args),
        "paths": {"trace": str(trace_path)},
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
