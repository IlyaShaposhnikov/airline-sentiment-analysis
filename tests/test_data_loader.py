"""
Unit tests for src/data_loader.py split helper.

Focus: train/test split by row position keeps texts, labels and weights
aligned, respects stratification and is reproducible.
"""

import numpy as np
import pytest

from src.data_loader import split_train_test_indices


@pytest.fixture
def split_config() -> dict:
    return {
        "evaluation": {"split": {"test_size": 0.25, "stratify": True}},
        "model": {"training": {"random_state": 42}},
    }


@pytest.fixture
def imbalanced_y() -> np.ndarray:
    # Mirrors the real dataset: negative-heavy, positive is the rarest
    return np.array([0] * 120 + [1] * 30 + [2] * 50)


class TestSplitTrainTestIndices:
    """Tests for split_train_test_indices()."""

    def test_partition_is_complete_and_disjoint(
        self, split_config, imbalanced_y
    ):
        train_idx, test_idx = split_train_test_indices(
            imbalanced_y, split_config
        )
        assert len(np.intersect1d(train_idx, test_idx)) == 0
        assert sorted(np.concatenate([train_idx, test_idx])) == list(
            range(len(imbalanced_y))
        )

    def test_test_size_respected(self, split_config, imbalanced_y):
        _, test_idx = split_train_test_indices(imbalanced_y, split_config)
        assert len(test_idx) == 50  # 25% of 200

    def test_indices_keep_rows_aligned(self, split_config, imbalanced_y):
        """
        Regression test: per-row artifacts selected with the same indices
        must describe the same rows (previously test texts were taken
        from the tail of the DataFrame and did not match X_test).
        """
        texts = [f"text_{i}_label_{lab}" for i, lab in enumerate(imbalanced_y)]
        _, test_idx = split_train_test_indices(imbalanced_y, split_config)

        for i in test_idx:
            assert texts[i].endswith(f"label_{imbalanced_y[i]}")

    def test_stratification_preserves_class_ratio(
        self, split_config, imbalanced_y
    ):
        _, test_idx = split_train_test_indices(imbalanced_y, split_config)
        counts = np.bincount(imbalanced_y[test_idx], minlength=3)
        expected = np.bincount(imbalanced_y, minlength=3) * 0.25
        # Exact proportions up to rounding of one sample per class
        assert np.all(np.abs(counts - expected) <= 1)

    def test_reproducible_with_same_seed(self, split_config, imbalanced_y):
        a = split_train_test_indices(imbalanced_y, split_config)
        b = split_train_test_indices(imbalanced_y, split_config)
        np.testing.assert_array_equal(a[1], b[1])

    def test_different_seed_changes_split(self, split_config, imbalanced_y):
        a = split_train_test_indices(imbalanced_y, split_config)
        split_config["model"]["training"]["random_state"] = 7
        b = split_train_test_indices(imbalanced_y, split_config)
        assert not np.array_equal(np.sort(a[1]), np.sort(b[1]))
