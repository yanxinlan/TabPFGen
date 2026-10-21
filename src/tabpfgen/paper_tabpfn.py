from __future__ import annotations

import hashlib
import importlib
import itertools
import os
import random
import sys
import warnings
from functools import partial
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder, PowerTransformer


PAPER_TABPFN_COMMIT = "e640ef535e29c8f1b8b148c187744429f847e6d4"
PAPER_CHECKPOINT_NAME = "prior_diff_real_checkpoint_n_0_epoch_42.cpkt"
PAPER_CHECKPOINT_SHA256 = (
    "3c9aadaeddbf51462af8c0ee4b3ca3c697890f77e92318abbb0821b75261c392"
)

TensorObserver = Callable[[str, torch.Tensor], None]

_MODEL_CACHE: dict[tuple[str, str], tuple[torch.nn.Module, dict]] = {}


def _install_legacy_torch_transformer_exports() -> None:
    """Restore names exported by torch.nn.modules.transformer in older Torch."""
    transformer_module = importlib.import_module("torch.nn.modules.transformer")
    compatibility_exports = {
        "Module": torch.nn.Module,
        "Tensor": torch.Tensor,
        "Optional": Optional,
    }
    for name, value in compatibility_exports.items():
        if not hasattr(transformer_module, name):
            setattr(transformer_module, name, value)


def default_paper_tabpfn_root() -> Path:
    configured = os.environ.get("TABPFN_PAPER_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "third_party" / "tabpfn_paper"


def default_paper_checkpoint(root: Path | None = None) -> Path:
    root = default_paper_tabpfn_root() if root is None else Path(root)
    return root / "tabpfn" / "models_diff" / PAPER_CHECKPOINT_NAME


def _verify_paper_assets(root: Path, checkpoint: Path) -> None:
    required = [
        root / "tabpfn" / "encoders.py",
        root / "tabpfn" / "transformer.py",
        root / "tabpfn" / "layer.py",
        root / "tabpfn" / "utils.py",
        checkpoint,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "The paper TabPFN assets are missing. Run "
            "scripts/setup_paper_tabpfn.sh first. Missing: " + ", ".join(missing)
        )

    marker = root / ".tabpfgen-paper-commit"
    if (
        marker.is_file()
        and marker.read_text(encoding="utf-8").strip() != PAPER_TABPFN_COMMIT
    ):
        raise RuntimeError(f"Unexpected TabPFN source revision recorded in {marker}")

    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != PAPER_CHECKPOINT_SHA256:
        raise RuntimeError(
            f"Checkpoint checksum mismatch for {checkpoint}: {digest}; "
            f"expected {PAPER_CHECKPOINT_SHA256}"
        )


def _import_legacy_architecture(root: Path):
    _install_legacy_torch_transformer_exports()
    root_text = str(root)
    loaded = sys.modules.get("tabpfn")
    if loaded is not None:
        loaded_path = Path(getattr(loaded, "__file__", "")).resolve()
        if root not in loaded_path.parents:
            raise RuntimeError(
                "A different tabpfn package was imported before the paper runtime. "
                "Run the reproduction in a fresh process without importing the modern "
                "tabpfn package."
            )
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

    encoders = importlib.import_module("tabpfn.encoders")
    transformer = importlib.import_module("tabpfn.transformer")
    return encoders, transformer


def _load_paper_model(
    root: Path, checkpoint: Path, device: torch.device
) -> tuple[torch.nn.Module, dict]:
    key = (str(checkpoint.resolve()), str(device))
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    _verify_paper_assets(root, checkpoint)
    encoders, transformer = _import_legacy_architecture(root)
    try:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        saved = torch.load(checkpoint, map_location="cpu")
    model_state, _, config = saved

    if (
        config.get("nan_prob_no_reason", 0.0) > 0.0
        or config.get("nan_prob_a_reason", 0.0) > 0.0
        or config.get("nan_prob_unknown_reason", 0.0) > 0.0
    ):
        encoder_factory = encoders.NanHandlingEncoder
    else:
        encoder_factory = partial(encoders.Linear, replace_nan_by_zero=True)

    encoder = encoder_factory(config["num_features"], config["emsize"])
    y_encoder_factory = (
        encoders.get_Canonical(config["max_num_classes"])
        if config.get("canonical_y_encoder", False)
        else encoders.Linear
    )
    hidden = config["emsize"] * config["nhid_factor"]
    model = transformer.TransformerModel(
        encoder,
        config["max_num_classes"],
        config["emsize"],
        config["nhead"],
        hidden,
        config["nlayers"],
        y_encoder=y_encoder_factory(1, config["emsize"]),
        dropout=config["dropout"],
        efficient_eval_masking=config["efficient_eval_masking"],
    )
    model.criterion = torch.nn.CrossEntropyLoss(
        reduction="none", weight=torch.ones(int(config["max_num_classes"]))
    )
    clean_state = {
        key.replace("module.", ""): value for key, value in model_state.items()
    }
    model.load_state_dict(clean_state)
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    _MODEL_CACHE[key] = (model, config)
    return model, config


def _observe(
    observer: TensorObserver | None, name: str, value: torch.Tensor
) -> torch.Tensor:
    if observer is not None:
        observer(name, value)
    return value


def _masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int = 0) -> torch.Tensor:
    count = torch.where(mask, torch.ones_like(x), torch.zeros_like(x)).sum(dim=dim)
    value = torch.where(mask, x, torch.zeros_like(x)).sum(dim=dim)
    return value / count


def _masked_std(x: torch.Tensor, mask: torch.Tensor, dim: int = 0) -> torch.Tensor:
    count = torch.where(mask, torch.ones_like(x), torch.zeros_like(x)).sum(dim=dim)
    value = torch.where(mask, x, torch.zeros_like(x)).sum(dim=dim)
    mean = value / count
    expanded = torch.repeat_interleave(mean.unsqueeze(dim), x.shape[dim], dim=dim)
    squared = torch.square(torch.where(mask, expanded - x, torch.zeros_like(x)))
    return torch.sqrt(squared.sum(dim=dim) / (count - 1))


def _normalize_data(
    data: torch.Tensor,
    eval_position: int,
    observer: TensorObserver | None,
    prefix: str,
) -> torch.Tensor:
    context = data[:eval_position]
    mask = ~torch.isnan(context)
    mean = _observe(observer, f"{prefix}.mean", _masked_mean(context, mask))
    std = _observe(observer, f"{prefix}.std", _masked_std(context, mask) + 0.000001)
    centered = _observe(observer, f"{prefix}.centered", data - mean)
    divided = _observe(observer, f"{prefix}.divided", centered / std)
    return _observe(observer, f"{prefix}.clipped", torch.clip(divided, -100, 100))


def _remove_outliers(
    x: torch.Tensor,
    eval_position: int,
    observer: TensorObserver | None,
    prefix: str,
    n_sigma: float = 4.0,
) -> torch.Tensor:
    context = x[:eval_position]
    valid = ~torch.isnan(context)
    mean1 = _observe(observer, f"{prefix}.pass1_mean", _masked_mean(context, valid))
    std1 = _observe(observer, f"{prefix}.pass1_std", _masked_std(context, valid))
    lower1 = _observe(observer, f"{prefix}.pass1_lower", mean1 - std1 * n_sigma)
    upper1 = _observe(observer, f"{prefix}.pass1_upper", mean1 + std1 * n_sigma)
    inlier_mask = _observe(
        observer,
        f"{prefix}.inlier_mask",
        (context <= upper1) & (context >= lower1) & valid,
    )
    mean2 = _observe(
        observer, f"{prefix}.pass2_mean", _masked_mean(context, inlier_mask)
    )
    std2 = _observe(observer, f"{prefix}.pass2_std", _masked_std(context, inlier_mask))
    lower2 = _observe(observer, f"{prefix}.pass2_lower", mean2 - std2 * n_sigma)
    upper2 = _observe(observer, f"{prefix}.pass2_upper", mean2 + std2 * n_sigma)
    lower_transform = _observe(
        observer, f"{prefix}.lower_transform", -torch.log1p(torch.abs(x)) + lower2
    )
    after_lower = _observe(
        observer, f"{prefix}.after_lower", torch.maximum(lower_transform, x)
    )
    upper_transform = _observe(
        observer, f"{prefix}.upper_transform", torch.log1p(torch.abs(x)) + upper2
    )
    return _observe(
        observer, f"{prefix}.output", torch.minimum(upper_transform, after_lower)
    )


class PaperTabPFNLogitEnergy:
    """Legacy TabPFN v1 checkpoint runtime used by the TabPFGen paper."""

    def __init__(
        self,
        x_context: torch.Tensor,
        y_context: torch.Tensor,
        *,
        device: torch.device,
        source_root: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        n_ensembles: int = 3,
        random_state: int = 0,
    ) -> None:
        self.source_root = (
            default_paper_tabpfn_root()
            if source_root is None
            else Path(source_root).expanduser().resolve()
        )
        self.checkpoint_path = (
            default_paper_checkpoint(self.source_root)
            if checkpoint_path is None
            else Path(checkpoint_path).expanduser().resolve()
        )
        self.device = device
        self.model, self.config = _load_paper_model(
            self.source_root, self.checkpoint_path, device
        )
        self.x_context = x_context
        self.y_context = y_context.long()
        self.n_ensembles = int(n_ensembles)
        self.random_state = int(random_state)
        self.preprocess_mode = "none"
        self.observer: TensorObserver | None = None
        self.trace_prefix = "tabpfn"

    def set_observer(
        self, observer: TensorObserver | None, prefix: str = "tabpfn"
    ) -> None:
        self.observer = observer
        self.trace_prefix = prefix

    def _preprocess(
        self, x: torch.Tensor, eval_position: int, transform: str = "none"
    ) -> torch.Tensor:
        prefix = f"{self.trace_prefix}.preprocess"
        x = _observe(self.observer, f"{prefix}.input", x)
        x = _normalize_data(x, eval_position, self.observer, f"{prefix}.normalize")

        flat = _observe(self.observer, f"{prefix}.batch_squeezed", x[:, 0, :])
        context = flat[:eval_position]
        keep = torch.tensor(
            [torch.unique(context[:, col]).numel() > 1 for col in range(flat.shape[1])],
            dtype=torch.bool,
            device=flat.device,
        )
        keep = _observe(self.observer, f"{prefix}.nonconstant_mask", keep)
        flat = _observe(self.observer, f"{prefix}.nonconstant_features", flat[:, keep])
        if transform == "power_all":
            transformed = flat.detach().cpu().numpy().copy()
            power = PowerTransformer(standardize=True)
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                for column in range(transformed.shape[1]):
                    try:
                        power.fit(transformed[:eval_position, column : column + 1])
                        transformed[:, column : column + 1] = power.transform(
                            transformed[:, column : column + 1]
                        )
                    except Exception:
                        pass
            flat = torch.as_tensor(transformed, dtype=x.dtype, device=x.device)
            flat = _observe(self.observer, f"{prefix}.power_all", flat)
        elif transform != "none":
            raise ValueError(f"Unsupported legacy preprocessing transform: {transform}")
        x = _observe(self.observer, f"{prefix}.batch_restored", flat.unsqueeze(1))
        x = _remove_outliers(
            x, eval_position, self.observer, f"{prefix}.remove_outliers"
        )
        max_features = int(self.config["num_features"])
        used_features = x.shape[-1]
        normalize_with_sqrt = bool(self.config.get("normalize_with_sqrt", False))
        factor = (used_features / max_features) ** (0.5 if normalize_with_sqrt else 1.0)
        return _observe(self.observer, f"{prefix}.feature_rescale", x / factor)

    def _ensemble_configurations(
        self, num_features: int, num_classes: int
    ) -> list[tuple[tuple[torch.Tensor, torch.Tensor], str]]:
        # Keep the ensemble transformations reproducible without resetting the
        # process-wide RNG used for the fresh Gaussian noise in each SGLD step.
        ensemble_rng = torch.Generator(device="cpu")
        ensemble_rng.manual_seed(self.random_state)
        feature_shifts = torch.randperm(num_features, generator=ensemble_rng)
        class_shifts = torch.randperm(num_classes, generator=ensemble_rng)
        base_configurations = list(itertools.product(class_shifts, feature_shifts))
        random.Random(self.random_state).shuffle(base_configurations)
        transforms = (
            ["none", "power_all"] if self.preprocess_mode == "mix" else ["none"]
        )
        configurations = list(itertools.product(base_configurations, transforms))
        return configurations[: self.n_ensembles]

    def _ensemble_logits(self, x_query: torch.Tensor) -> torch.Tensor:
        if not isinstance(x_query, torch.Tensor):
            raise TypeError(
                "Paper TabPFN differentiable inference requires torch tensors"
            )
        eval_position = self.x_context.shape[0]
        x_full = torch.cat((self.x_context, x_query), dim=0).float().unsqueeze(1)
        x_full = _observe(self.observer, f"{self.trace_prefix}.x_full", x_full)
        y_context = self.y_context.float().unsqueeze(1)
        num_classes = int(torch.unique(self.y_context).numel())
        configurations = self._ensemble_configurations(
            x_full.shape[2], num_classes
        )

        outputs: list[torch.Tensor] = []
        max_features = int(self.config["num_features"])
        transformed_cache: dict[str, torch.Tensor] = {}
        for index, ((class_shift, feature_shift), transform) in enumerate(
            configurations
        ):
            prefix = f"{self.trace_prefix}.ensemble_{index}"
            if transform not in transformed_cache:
                transformed_cache[transform] = self._preprocess(
                    x_full.clone(), eval_position, transform
                )
            x_processed = transformed_cache[transform].clone()
            shifted_y = _observe(
                self.observer,
                f"{prefix}.shifted_labels",
                (y_context + class_shift) % num_classes,
            )
            shifted_x = torch.cat(
                [x_processed[..., feature_shift:], x_processed[..., :feature_shift]],
                dim=-1,
            )
            shifted_x = _observe(self.observer, f"{prefix}.shifted_features", shifted_x)
            if shifted_x.shape[-1] < max_features:
                padding = torch.zeros(
                    (*shifted_x.shape[:-1], max_features - shifted_x.shape[-1]),
                    device=shifted_x.device,
                    dtype=shifted_x.dtype,
                )
                shifted_x = torch.cat([shifted_x, padding], dim=-1)
            shifted_x = _observe(self.observer, f"{prefix}.padded_features", shifted_x)

            output = self.model(
                (None, shifted_x, shifted_y), single_eval_pos=eval_position
            )[..., :num_classes]
            output = _observe(self.observer, f"{prefix}.model_output", output / 0.8)
            unshifted = torch.cat(
                [output[..., class_shift:], output[..., :class_shift]], dim=-1
            )
            outputs.append(
                _observe(self.observer, f"{prefix}.unshifted_logits", unshifted)
            )

        stacked = torch.stack(outputs)
        _observe(self.observer, f"{self.trace_prefix}.ensemble_logits", stacked)
        return stacked

    def logits(self, x_query: torch.Tensor) -> torch.Tensor:
        averaged = self._ensemble_logits(x_query).mean(dim=0)
        averaged = torch.transpose(averaged, 0, 1).squeeze(0)
        return _observe(self.observer, f"{self.trace_prefix}.logits", averaged)

    def legacy_probabilities(self, x_query: torch.Tensor) -> torch.Tensor:
        """Match the legacy classifier's probability aggregation exactly."""
        ensemble_probabilities = torch.softmax(
            self._ensemble_logits(x_query), dim=-1
        )
        averaged = ensemble_probabilities.mean(dim=0)
        probabilities = torch.softmax(averaged, dim=-1)
        return torch.transpose(probabilities, 0, 1).squeeze(0)


class PaperTabPFNClassifier:
    """Small sklearn-style wrapper around the paper's pinned TabPFN runtime."""

    def __init__(
        self,
        *,
        device: str = "cpu",
        source_root: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        n_ensembles: int = 3,
        random_state: int = 0,
    ) -> None:
        self.device = torch.device(device)
        self.source_root = source_root
        self.checkpoint_path = checkpoint_path
        self.n_ensembles = n_ensembles
        self.random_state = random_state
        self.label_encoder = LabelEncoder()
        self.energy: PaperTabPFNLogitEnergy | None = None

    def fit(self, x: np.ndarray, y: np.ndarray):
        encoded = self.label_encoder.fit_transform(np.asarray(y)).astype(np.int64)
        x_context = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        y_context = torch.as_tensor(encoded, dtype=torch.long, device=self.device)
        self.energy = PaperTabPFNLogitEnergy(
            x_context,
            y_context,
            device=self.device,
            source_root=self.source_root,
            checkpoint_path=self.checkpoint_path,
            n_ensembles=self.n_ensembles,
            random_state=self.random_state,
        )
        self.energy.preprocess_mode = "mix"
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.energy is None:
            raise RuntimeError("Call fit() before predict_proba()")
        query = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            probabilities = self.energy.legacy_probabilities(query)
        return probabilities.cpu().numpy()
