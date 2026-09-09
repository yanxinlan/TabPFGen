from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple
import warnings

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings(
    "ignore", category=UserWarning, module="sklearn.preprocessing._encoders"
)


@dataclass
class TabPFGenSampleTrace:
    """Diagnostics from one TabPFGen sampling run."""

    mean_energy: list[float]
    best_step: int
    best_mean_energy: float


class _TabPFNLogitEnergy:
    """Expose the paper's TabPFN logit energy.

    The TabPFGen paper defines E(x | y) = -f(x)[y], where f is the frozen
    TabPFN classifier conditioned on the real training set. This wrapper
    intentionally requires differentiable logits; predict_proba would reproduce
    a different algorithm because it does not provide the paper's logit energy.
    """

    def __init__(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        *,
        device: torch.device,
        n_estimators: int = 1,
        model_path: str = "auto",
        random_state: int | None = 0,
    ) -> None:
        from tabpfn import TabPFNClassifier

        kwargs = {
            "device": str(device),
            "n_estimators": n_estimators,
            "model_path": model_path,
            "balance_probabilities": False,
            "average_before_softmax": True,
            "softmax_temperature": 1.0,
            "random_state": random_state,
            "differentiable_input": True,
        }
        try:
            self.classifier = TabPFNClassifier(**kwargs)
        except TypeError:
            kwargs.pop("differentiable_input")
            self.classifier = TabPFNClassifier(**kwargs)

        if hasattr(self.classifier, "fit_with_differentiable_input"):
            self.classifier.fit_with_differentiable_input(x_train, y_train)
        else:
            self.classifier.fit(x_train, y_train)

    def logits(self, x_query: torch.Tensor) -> torch.Tensor:
        if hasattr(self.classifier, "_raw_predict"):
            logits = self.classifier._raw_predict(x_query, return_logits=True)
        elif hasattr(self.classifier, "predict_logits"):
            logits = self.classifier.predict_logits(x_query)
        else:
            raise RuntimeError(
                "Paper-faithful TabPFGen requires a TabPFNClassifier with "
                "differentiable raw-logit support."
            )
        if not isinstance(logits, torch.Tensor):
            logits = torch.as_tensor(logits, device=x_query.device, dtype=x_query.dtype)

        if x_query.requires_grad and not logits.requires_grad:
            raise RuntimeError(
                "TabPFN logits are detached from x_synth. Install/use a TabPFN "
                "version that supports differentiable_input=True; otherwise SGLD "
                "cannot backpropagate the paper's energy."
            )

        return logits.to(device=x_query.device, dtype=x_query.dtype)


class TabPFGen:
    """Paper-faithful TabPFGen for numerical classification data.

    This class implements Algorithm 1 from Ma et al., "TabPFGen - Tabular Data
    Generation with TabPFN": initialize synthetic samples from noisy training
    rows, compute the class-conditional energy E(x | y) = -f_TabPFN(x)[y], and
    update x with SGLD while keeping TabPFN frozen.
    """

    def __init__(
        self,
        n_sgld_steps: int = 1000,
        sgld_step_size: float = 0.01,
        sgld_noise_scale: float = 0.01,
        init_noise_std: float = 0.01,
        device: str | torch.device | None = "auto",
        scale_features: bool = True,
        keep_best: bool = True,
        n_estimators: int = 1,
        model_path: str = "auto",
        random_state: int | None = 0,
        swapped_energy_weight: float = 0.0,
        verbose: bool = False,
        energy_model_factory: Optional[
            Callable[[torch.Tensor, torch.Tensor, torch.device], object]
        ] = None,
    ):
        self.n_sgld_steps = int(n_sgld_steps)
        self.sgld_step_size = float(sgld_step_size)
        self.sgld_noise_scale = float(sgld_noise_scale)
        self.init_noise_std = float(init_noise_std)
        self.scale_features = bool(scale_features)
        self.keep_best = bool(keep_best)
        self.n_estimators = int(n_estimators)
        self.model_path = model_path
        self.random_state = random_state
        self.swapped_energy_weight = float(swapped_energy_weight)
        self.verbose = bool(verbose)
        self.device = self._infer_device(device)

        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.energy_model_factory = energy_model_factory

        self.energy_model_: object | None = None
        self.classes_: np.ndarray | None = None
        self.last_trace_: TabPFGenSampleTrace | None = None

    def _infer_device(self, device: str | torch.device | None) -> torch.device:
        if device is None or (isinstance(device, str) and device == "auto"):
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(device, str):
            return torch.device(device)
        if isinstance(device, torch.device):
            return device
        raise ValueError(f"Invalid device: {device}")

    def _validate_classification_input(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X_train, dtype=np.float32)
        y = np.asarray(y_train)
        if X.ndim != 2:
            raise ValueError("X_train must be a 2D array")
        if y.ndim != 1:
            raise ValueError("y_train must be a 1D array")
        if len(X) != len(y):
            raise ValueError("X_train and y_train must have the same number of samples")
        if len(X) < 2:
            raise ValueError("TabPFGen requires at least two training samples")
        if np.unique(y).size < 2:
            raise ValueError("TabPFGen requires at least two classes")
        if not np.isfinite(X).all():
            raise ValueError("TabPFGen currently supports numerical data without NaNs")
        return X, y

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "TabPFGen":
        X, y = self._validate_classification_input(X_train, y_train)
        self.classes_ = np.unique(y)
        y_encoded = self.label_encoder.fit_transform(y).astype(np.int64)

        if self.scale_features:
            X_model = self.scaler.fit_transform(X).astype(np.float32)
        else:
            X_model = X.astype(np.float32, copy=True)

        x_train = torch.as_tensor(X_model, dtype=torch.float32, device=self.device)
        y_train_t = torch.as_tensor(y_encoded, dtype=torch.long, device=self.device)

        self.energy_model_ = self._make_energy_model(x_train, y_train_t)

        self._x_train_ = x_train
        self._y_train_ = y_train_t
        return self

    def _make_energy_model(self, x_context: torch.Tensor, y_context: torch.Tensor):
        if self.energy_model_factory is not None:
            return self.energy_model_factory(x_context, y_context, self.device)
        return _TabPFNLogitEnergy(
            x_context,
            y_context,
            device=self.device,
            n_estimators=self.n_estimators,
            model_path=self.model_path,
            random_state=self.random_state,
        )

    def _logits_from_model(self, model: object, x_query: torch.Tensor) -> torch.Tensor:
        if hasattr(model, "logits"):
            return model.logits(x_query)
        if callable(model):
            return model(x_query)
        raise TypeError("energy_model must expose logits(x) or be callable")

    def _tabpfn_logits(self, x_synth: torch.Tensor) -> torch.Tensor:
        if self.energy_model_ is None:
            raise RuntimeError("Call fit() before sampling")
        return self._logits_from_model(self.energy_model_, x_synth)

    def _compute_energy(
        self,
        x_synth: torch.Tensor,
        y_synth: torch.Tensor,
        x_train: torch.Tensor | None = None,
        y_train: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute E(x_synth | y_synth) = -f_TabPFN(x_synth)[y_synth]."""
        logits = self._tabpfn_logits(x_synth)
        if logits.ndim != 2:
            raise ValueError("TabPFN logits must have shape (n_samples, n_classes)")
        rows = torch.arange(x_synth.shape[0], device=x_synth.device)
        energy = -logits[rows, y_synth.long()]

        if self.swapped_energy_weight <= 0:
            return energy
        if x_train is None or y_train is None:
            raise ValueError("x_train and y_train are required for swapped energy")
        if torch.unique(y_synth).numel() < 2:
            raise ValueError(
                "swapped_energy_weight requires y_synth to contain at least two "
                "classes so TabPFN can use the synthetic batch as context"
            )

        swapped_model = self._make_energy_model(x_synth, y_synth.long())
        swapped_logits = self._logits_from_model(swapped_model, x_train)
        train_rows = torch.arange(x_train.shape[0], device=x_train.device)
        swapped_energy = -swapped_logits[train_rows, y_train.long()].mean()
        return energy + self.swapped_energy_weight * swapped_energy

    def _sgld_step(
        self,
        x_synth: torch.Tensor,
        y_synth: torch.Tensor,
        x_train: torch.Tensor | None = None,
        y_train: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_synth = x_synth.detach().clone().requires_grad_(True)
        energy = self._compute_energy(x_synth, y_synth, x_train, y_train)
        grad = torch.autograd.grad(energy.sum(), x_synth)[0]
        noise = torch.randn_like(x_synth) * self.sgld_noise_scale
        x_next = x_synth - self.sgld_step_size * grad + noise
        return x_next.detach(), energy.detach()

    def _init_from_training_rows(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        y_synth: torch.Tensor,
    ) -> torch.Tensor:
        x_parts: list[torch.Tensor] = []
        for label in y_synth:
            matching = torch.where(y_train == label)[0]
            if matching.numel() == 0:
                raise ValueError(f"No training rows found for encoded class {label}")
            idx = matching[torch.randint(0, matching.numel(), (1,), device=self.device)]
            x_parts.append(x_train[idx])
        x_init = torch.cat(x_parts, dim=0)
        if self.init_noise_std > 0:
            x_init = x_init + torch.randn_like(x_init) * self.init_noise_std
        return x_init

    def _target_labels(
        self,
        y_train: torch.Tensor,
        n_samples: int,
        balance_classes: bool,
    ) -> torch.Tensor:
        classes = torch.unique(y_train, sorted=True)
        if balance_classes:
            per_class = n_samples // classes.numel()
            remainder = n_samples - per_class * classes.numel()
            labels = [
                cls.repeat(per_class + (1 if i < remainder else 0))
                for i, cls in enumerate(classes)
            ]
            y_synth = torch.cat(labels)
            perm = torch.randperm(y_synth.numel(), device=self.device)
            return y_synth[perm]

        idx = torch.randint(0, y_train.numel(), (n_samples,), device=self.device)
        return y_train[idx]

    def _sample_encoded(
        self,
        n_samples: int,
        balance_classes: bool = True,
        y_synth: np.ndarray | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.energy_model_ is None:
            raise RuntimeError("Call fit() before sampling")
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")

        x_train = self._x_train_
        y_train = self._y_train_
        if y_synth is None:
            y_synth_t = self._target_labels(y_train, n_samples, balance_classes)
        else:
            y_synth_arr = np.asarray(y_synth)
            y_synth_encoded = self.label_encoder.transform(y_synth_arr).astype(np.int64)
            y_synth_t = torch.as_tensor(
                y_synth_encoded, dtype=torch.long, device=self.device
            )
            n_samples = int(y_synth_t.numel())

        x_synth = self._init_from_training_rows(x_train, y_train, y_synth_t)
        best_x = x_synth.detach().clone()
        best_energy = float("inf")
        best_step = 0
        mean_energy_history: list[float] = []

        for step in range(self.n_sgld_steps):
            x_synth, energy = self._sgld_step(x_synth, y_synth_t, x_train, y_train)
            mean_energy = float(energy.mean().item())
            mean_energy_history.append(mean_energy)

            if mean_energy < best_energy:
                best_energy = mean_energy
                best_step = step + 1
                best_x = x_synth.detach().clone()

            if self.verbose and (step == 0 or (step + 1) % 100 == 0):
                print(
                    f"Step {step + 1}/{self.n_sgld_steps}: "
                    f"mean energy={mean_energy:.6f}"
                )

        self.last_trace_ = TabPFGenSampleTrace(
            mean_energy=mean_energy_history,
            best_step=best_step,
            best_mean_energy=best_energy,
        )
        return (best_x if self.keep_best else x_synth), y_synth_t

    def generate_classification(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_samples: int,
        balance_classes: bool = True,
        y_synth: np.ndarray | torch.Tensor | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic classification data with TabPFN-logit SGLD.

        If y_synth is provided, those labels are used as the manually defined
        synthetic labels from Algorithm 1. Otherwise labels are sampled from a
        balanced or empirical class distribution.
        """
        self.fit(X_train, y_train)
        x_synth, y_synth_t = self._sample_encoded(
            n_samples=n_samples,
            balance_classes=balance_classes,
            y_synth=y_synth,
        )

        X_synth_model = x_synth.detach().cpu().numpy()
        if self.scale_features:
            X_synth = self.scaler.inverse_transform(X_synth_model)
        else:
            X_synth = X_synth_model
        y_synth_out = self.label_encoder.inverse_transform(
            y_synth_t.detach().cpu().numpy()
        )
        return X_synth, y_synth_out

    def balance_dataset(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        target_per_class: Optional[int] = None,
        min_class_size: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate minority-class samples until each eligible class is balanced."""
        X, y = self._validate_classification_input(X_train, y_train)
        classes, counts = np.unique(y, return_counts=True)
        target = int(counts.max() if target_per_class is None else target_per_class)
        if target < counts.max():
            raise ValueError(
                f"target_per_class ({target}) must be >= largest class size "
                f"({counts.max()})"
            )

        y_targets: list[np.ndarray] = []
        for cls, count in zip(classes, counts):
            if count < min_class_size:
                continue
            needed = max(0, target - int(count))
            if needed:
                y_targets.append(np.full(needed, cls, dtype=y.dtype))

        if not y_targets:
            return (
                np.empty((0, X.shape[1]), dtype=X.dtype),
                np.empty((0,), dtype=y.dtype),
                X.copy(),
                y.copy(),
            )

        requested_labels = np.concatenate(y_targets)
        X_synth, y_synth = self.generate_classification(
            X,
            y,
            n_samples=len(requested_labels),
            y_synth=requested_labels,
        )
        X_combined = np.vstack([X, X_synth])
        y_combined = np.concatenate([y, y_synth])
        return X_synth, y_synth, X_combined, y_combined

    def generate_regression(self, *args, **kwargs):
        """Regression generation is not part of the TabPFGen paper."""
        raise NotImplementedError(
            "Paper-faithful TabPFGen is a classification generator. "
            "The previous regression helper assigned continuous targets post-hoc and "
            "did not implement the paper's energy model."
        )
