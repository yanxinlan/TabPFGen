import unittest
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_reproduction_results import summarize_table1
from train_table1_generators import (
    SYNTHCITY_HYPERPARAMS,
    sample_exact_training_class_counts,
    validate_training_sample_profile,
)
from tabpfgen.paper_tabpfn import PaperTabPFNLogitEnergy


class TestReproductionPipeline(unittest.TestCase):
    def test_ensemble_configuration_does_not_reset_sampler_rng(self):
        runtime = object.__new__(PaperTabPFNLogitEnergy)
        runtime.random_state = 7
        runtime.preprocess_mode = "none"
        runtime.n_ensembles = 3

        torch.manual_seed(123)
        expected = torch.randn(8)
        torch.manual_seed(123)
        runtime._ensemble_configurations(num_features=5, num_classes=3)
        actual = torch.randn(8)

        torch.testing.assert_close(actual, expected)

    def test_table1_error_bars_are_over_run_level_dataset_means(self):
        metrics = pd.DataFrame(
            [
                {
                    "mode": "original",
                    "downstream_model": "xgb",
                    "generator": "original",
                    "seed": 0,
                    "dataset_id": 11,
                    "auc": 0.8,
                },
                {
                    "mode": "original",
                    "downstream_model": "xgb",
                    "generator": "original",
                    "seed": 0,
                    "dataset_id": 14,
                    "auc": 1.0,
                },
                {
                    "mode": "original",
                    "downstream_model": "xgb",
                    "generator": "original",
                    "seed": 1,
                    "dataset_id": 11,
                    "auc": 0.6,
                },
                {
                    "mode": "original",
                    "downstream_model": "xgb",
                    "generator": "original",
                    "seed": 1,
                    "dataset_id": 14,
                    "auc": 0.8,
                },
            ]
        )

        comparison, _ = summarize_table1(metrics)
        row = comparison[
            (comparison["mode"] == "augmentation")
            & (comparison["downstream_model"] == "xgb")
            & (comparison["generator"] == "original")
        ].iloc[0]

        self.assertAlmostEqual(row["reproduced_auc_mean"], 0.8)
        self.assertAlmostEqual(row["reproduced_auc_std"], np.std([0.9, 0.7], ddof=1))
        self.assertEqual(row["n_runs"], 2)
        self.assertEqual(row["min_datasets_per_run"], 2)
        self.assertEqual(row["max_datasets_per_run"], 2)

    def test_synthetic_sample_matches_training_size_and_class_counts(self):
        train = pd.DataFrame(
            {"feature": np.arange(5), "target": ["a", "a", "a", "b", "b"]}
        )
        batches = [
            pd.DataFrame({"feature": [10, 11, 12], "target": ["a", "a", "b"]}),
            pd.DataFrame({"feature": [13, 14, 15], "target": ["a", "b", "b"]}),
        ]

        synthetic = sample_exact_training_class_counts(batches, train, seed=0)

        self.assertIsNotNone(synthetic)
        self.assertEqual(len(synthetic), len(train))
        self.assertEqual(
            synthetic["target"].value_counts().to_dict(),
            train["target"].value_counts().to_dict(),
        )
        validate_training_sample_profile(synthetic, train)

        with self.assertRaises(RuntimeError):
            validate_training_sample_profile(synthetic.iloc[:-1], train)

    def test_rtvae_uses_paper_clipping_value(self):
        self.assertEqual(SYNTHCITY_HYPERPARAMS["rtvae"]["clipping_value"], 1)


if __name__ == "__main__":
    unittest.main()
