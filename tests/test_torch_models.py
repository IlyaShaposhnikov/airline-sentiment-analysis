"""
Tests for src/torch_models.py (TorchMLPClassifier) and its integration.

Tests marked ``requires_torch`` train real (tiny) networks and are skipped
when PyTorch is not installed; the rest (config dispatch, explanation
gating, dependency error) run everywhere.
"""
import importlib.util
import sys

import joblib
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from src.interpretability import (
    explain_prediction,
    supports_word_explanations,
)
from src.models import (
    create_model,
    load_model,
    predict_sentiment,
    save_model,
    train_model,
)
from src.preprocessing import create_vectorizer
from src.torch_models import TorchMLPClassifier

requires_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="PyTorch not installed (pip install -r requirements-dl.txt)",
)

# Small, fast settings for unit tests
FAST = dict(hidden_dims=(16,), max_epochs=60, batch_size=16,
            learning_rate=0.01, random_state=0)


@pytest.fixture
def blobs() -> tuple[np.ndarray, np.ndarray]:
    """Three well-separated Gaussian blobs in 5 dimensions."""
    rng = np.random.default_rng(0)
    centers = np.array([[3, 0, 0, 0, 0], [0, 3, 0, 0, 0], [0, 0, 3, 0, 0]])
    X = np.vstack([c + rng.normal(0, 0.5, size=(40, 5)) for c in centers])
    y = np.repeat([0, 1, 2], 40)
    return X.astype(np.float32), y


@pytest.fixture
def mlp_config() -> dict:
    return {
        "preprocessing": {
            "vectorizer": {
                "type": "tfidf", "max_features": 50, "ngram_range": [1, 1]
            },
        },
        "model": {
            "type": "mlp",
            "training": {
                "class_weight": "balanced", "random_state": 0,
                "use_confidence_weights": True,
            },
            "mlp": {
                "hidden_dims": [16], "dropout": 0.1,
                "learning_rate": "1e-2",  # string on purpose (YAML quirk)
                "batch_size": 16, "max_epochs": 40, "patience": 5,
            },
        },
    }


# ============================================================================
# Config dispatch and gating (no torch needed)
# ============================================================================


class TestCreateMLP:

    def test_create_model_dispatches_to_mlp(self, mlp_config):
        model = create_model(mlp_config)

        assert isinstance(model, TorchMLPClassifier)
        assert model.hidden_dims == (16,)
        assert model.learning_rate == pytest.approx(0.01)
        assert model.class_weight == "balanced"
        assert model.random_state == 0

    def test_mlp_does_not_require_logreg_keys(self):
        """max_iter/regularization are logistic-regression settings."""
        model = create_model({"model": {"type": "mlp"}})
        assert isinstance(model, TorchMLPClassifier)

    def test_class_weight_none_string(self, mlp_config):
        mlp_config["model"]["training"]["class_weight"] = "none"
        assert create_model(mlp_config).class_weight is None

    def test_unsupported_type_lists_supported(self):
        with pytest.raises(ValueError, match="mlp"):
            create_model({
                "model": {"type": "svm", "training": {"max_iter": 10}}
            })

    def test_word_explanations_disabled_for_mlp(self):
        vec = TfidfVectorizer()
        assert supports_word_explanations(vec)
        assert supports_word_explanations(vec, model=object())
        assert not supports_word_explanations(vec, TorchMLPClassifier())

    def test_missing_torch_gives_install_hint(self, monkeypatch, blobs):
        monkeypatch.setitem(sys.modules, "torch", None)
        X, y = blobs
        with pytest.raises(ImportError, match="pip install torch"):
            TorchMLPClassifier(**FAST).fit(X, y)


# ============================================================================
# Training behaviour (requires torch)
# ============================================================================


@requires_torch
class TestTorchMLPClassifier:

    def test_learns_separable_data(self, blobs):
        X, y = blobs
        model = TorchMLPClassifier(**FAST).fit(X, y)
        assert (model.predict(X) == y).mean() == 1.0

    def test_predict_proba_is_distribution(self, blobs):
        X, y = blobs
        proba = TorchMLPClassifier(**FAST).fit(X, y).predict_proba(X)

        assert proba.shape == (len(y), 3)
        assert proba.dtype == np.float64
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_preserves_original_labels(self, blobs):
        """classes_ and predictions use the caller's labels (0/1/2 mapping
        of the project is not contiguous in binary mode)."""
        X, y = blobs
        labels = np.array([2, 0, 1])[y]
        model = TorchMLPClassifier(**FAST).fit(X, labels)

        assert model.classes_.tolist() == [0, 1, 2]
        assert (model.predict(X) == labels).mean() == 1.0

    def test_sparse_input_matches_dense(self, blobs):
        X, y = blobs
        dense = TorchMLPClassifier(**FAST).fit(X, y).predict_proba(X)
        sparse = TorchMLPClassifier(**FAST).fit(
            csr_matrix(X), y
        ).predict_proba(csr_matrix(X))
        np.testing.assert_allclose(dense, sparse, atol=1e-5)

    def test_reproducible_with_same_seed(self, blobs):
        X, y = blobs
        a = TorchMLPClassifier(**FAST).fit(X, y).predict_proba(X)
        b = TorchMLPClassifier(**FAST).fit(X, y).predict_proba(X)
        np.testing.assert_allclose(a, b)

    def test_sample_weight_is_respected(self):
        """
        Identical inputs with conflicting labels: the label carrying the
        (much) larger confidence weight must win.
        """
        X = np.ones((40, 3), dtype=np.float32)
        y = np.array([0, 1] * 20)
        weights = np.where(y == 0, 1.0, 0.01)
        model = TorchMLPClassifier(
            **{**FAST, "class_weight": None, "validation_fraction": 0}
        ).fit(X, y, sample_weight=weights)

        assert model.predict(X[:1])[0] == 0

    def test_balanced_class_weights_formula(self):
        model = TorchMLPClassifier(class_weight="balanced")
        model.classes_ = np.array([0, 1])
        weights = model._compute_class_weights(np.array([0, 0, 0, 1]))
        np.testing.assert_allclose(weights, [4 / 6, 4 / 2])

    def test_early_stopping_restores_best_epoch(self):
        """Pure noise: validation loss stops improving quickly."""
        rng = np.random.default_rng(1)
        X = rng.normal(size=(200, 20)).astype(np.float32)
        y = rng.integers(0, 2, size=200)
        model = TorchMLPClassifier(
            **{**FAST, "max_epochs": 200, "patience": 3,
               "hidden_dims": (64,)}
        ).fit(X, y)

        assert model.n_epochs_ < 200
        assert model.best_epoch_ <= model.n_epochs_
        assert model.n_epochs_ - model.best_epoch_ == 3
        assert len(model.history_) == model.n_epochs_
        assert all(h["val_loss"] is not None for h in model.history_)

    def test_no_validation_trains_all_epochs(self, blobs):
        X, y = blobs
        model = TorchMLPClassifier(
            **{**FAST, "max_epochs": 7, "validation_fraction": 0}
        ).fit(X, y)
        assert model.n_epochs_ == 7
        assert model.best_val_loss_ is None

    def test_pickle_stores_numpy_weights(self, blobs, tmp_path):
        X, y = blobs
        model = TorchMLPClassifier(**FAST).fit(X, y)
        expected = model.predict_proba(X)

        joblib.dump(model, tmp_path / "mlp.joblib")
        loaded = joblib.load(tmp_path / "mlp.joblib")

        assert "module_" not in loaded.__dict__
        assert all(
            isinstance(v, np.ndarray) for v in loaded._module_state.values()
        )
        np.testing.assert_allclose(loaded.predict_proba(X), expected,
                                   atol=1e-6)

    def test_feature_count_mismatch_raises(self, blobs):
        X, y = blobs
        model = TorchMLPClassifier(**FAST).fit(X, y)
        with pytest.raises(ValueError, match="features"):
            model.predict_proba(X[:, :3])

    def test_empty_input(self, blobs):
        X, y = blobs
        model = TorchMLPClassifier(**FAST).fit(X, y)
        assert model.predict_proba(X[:0]).shape == (0, 3)


# ============================================================================
# Pipeline integration: TF-IDF + MLP through the standard bundle
# ============================================================================


@requires_torch
class TestMLPPipelineIntegration:

    TEXTS = (
        ["great flight love thanks"] * 20
        + ["awful delay lost bag"] * 20
        + ["gate change schedule"] * 20
    )
    LABELS = np.array([1] * 20 + [0] * 20 + [2] * 20)

    def test_tfidf_mlp_train_save_load_predict(self, mlp_config, tmp_path):
        vec = create_vectorizer(mlp_config)
        X = vec.fit_transform(self.TEXTS)
        model = train_model(
            X, self.LABELS, mlp_config,
            sample_weights=np.full(len(self.LABELS), 0.9),
        )
        save_model(model, vec, tmp_path, filename="mlp.joblib")

        loaded_model, loaded_vec, _, _ = load_model(tmp_path / "mlp.joblib")
        preds = predict_sentiment(
            loaded_model, loaded_vec,
            ["@united great flight, thanks!", "lost my bag, awful delay"],
        )
        assert preds.tolist() == [1, 0]

    def test_explain_prediction_degrades_gracefully(self, mlp_config):
        vec = create_vectorizer(mlp_config)
        model = train_model(
            vec.fit_transform(self.TEXTS), self.LABELS, mlp_config
        )
        result = explain_prediction(model, vec, "great flight")
        assert result["method"] == "none"
        assert result["top_contributors"] == []


@requires_torch
def test_model_service_with_mlp_returns_no_explanation(mlp_config):
    """explain=true with an MLP bundle → prediction, explanation=None."""
    pytest.importorskip("fastapi")
    from unittest.mock import patch

    from src.api.services import ModelService
    from src.constants import TARGET_MAPPING, TARGET_MAPPING_INV

    texts = TestMLPPipelineIntegration.TEXTS
    labels = TestMLPPipelineIntegration.LABELS
    vec = create_vectorizer(mlp_config)
    model = train_model(vec.fit_transform(texts), labels, mlp_config)

    with patch("src.api.services.load_model") as mock_load, \
         patch("src.api.services.is_model_available") as mock_avail:
        mock_load.return_value = (
            model, vec, TARGET_MAPPING, TARGET_MAPPING_INV
        )
        mock_avail.return_value = True
        service = ModelService(model_path="unused.joblib")
        response = service.predict_single_sync(
            "great flight, thanks!", explain=True, n_explain=5
        )

    assert response.predicted_class == "positive"
    assert response.explanation is None
