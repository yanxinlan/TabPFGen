import unittest

import numpy as np
import torch

from tabpfgen import TabPFGen


class QuadraticLogitEnergy:
    """Tiny differentiable stand-in for TabPFN logits."""

    def __init__(self, x_train, y_train, device):
        del x_train, y_train, device

    def logits(self, x):
        class0 = -(x[:, 0] + 1.0).pow(2)
        class1 = -(x[:, 0] - 1.0).pow(2)
        return torch.stack([class0, class1], dim=1)


def energy_factory(x_train, y_train, device):
    return QuadraticLogitEnergy(x_train, y_train, device)


class TestTabPFGenPaperAlgorithm(unittest.TestCase):
    def setUp(self):
        self.X = np.array(
            [
                [-1.2, 0.0],
                [-0.8, 0.1],
                [0.8, 0.0],
                [1.2, 0.1],
            ],
            dtype=np.float32,
        )
        self.y = np.array(["negative", "negative", "positive", "positive"])
        self.generator = TabPFGen(
            n_sgld_steps=5,
            sgld_step_size=0.1,
            sgld_noise_scale=0.0,
            init_noise_std=0.0,
            device="cpu",
            scale_features=False,
            energy_model_factory=energy_factory,
        )

    def test_energy_is_negative_target_logit(self):
        self.generator.fit(self.X, self.y)
        x = torch.tensor([[-1.0, 0.0], [1.0, 0.0]], requires_grad=True)
        y = torch.tensor([0, 1])

        energy = self.generator._compute_energy(x, y)

        torch.testing.assert_close(energy, torch.tensor([0.0, 0.0]))
        wrong_target_energy = self.generator._compute_energy(x, 1 - y)
        torch.testing.assert_close(wrong_target_energy, torch.tensor([4.0, 4.0]))

    def test_sgld_moves_toward_requested_class_energy(self):
        self.generator.fit(self.X, self.y)
        x = torch.tensor([[0.0, 0.0]])
        y = torch.tensor([1])

        x_next, energy = self.generator._sgld_step(x, y)

        self.assertEqual(float(energy.item()), 1.0)
        self.assertGreater(float(x_next[0, 0]), 0.0)

    def test_generate_uses_user_supplied_synthetic_labels(self):
        np.random.seed(0)
        torch.manual_seed(0)
        requested = np.array(["positive", "negative", "positive"])

        X_synth, y_synth = self.generator.generate_classification(
            self.X,
            self.y,
            n_samples=len(requested),
            y_synth=requested,
        )

        self.assertEqual(X_synth.shape, (3, 2))
        np.testing.assert_array_equal(y_synth, requested)
        self.assertIsNotNone(self.generator.last_trace_)

    def test_balance_dataset_generates_exact_missing_class_labels(self):
        X = np.array([[-1.0], [-0.9], [-0.8], [1.0]], dtype=np.float32)
        y = np.array([0, 0, 0, 1])
        generator = TabPFGen(
            n_sgld_steps=1,
            sgld_step_size=0.1,
            sgld_noise_scale=0.0,
            init_noise_std=0.0,
            device="cpu",
            scale_features=False,
            energy_model_factory=energy_factory,
        )

        X_syn, y_syn, X_comb, y_comb = generator.balance_dataset(X, y)

        self.assertEqual(X_syn.shape, (2, 1))
        np.testing.assert_array_equal(y_syn, np.array([1, 1]))
        _, counts = np.unique(y_comb, return_counts=True)
        np.testing.assert_array_equal(counts, np.array([3, 3]))
        self.assertEqual(X_comb.shape, (6, 1))

    def test_regression_is_out_of_scope(self):
        with self.assertRaises(NotImplementedError):
            self.generator.generate_regression(self.X, np.array([0.1, 0.2]), 2)


if __name__ == "__main__":
    unittest.main()
