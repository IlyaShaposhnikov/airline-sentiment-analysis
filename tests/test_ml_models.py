"""
Unit tests for src/models.py ML functions.

Focus: Model creation, training with sample weights, evaluation metrics,
prediction logic, and persistence (save/load).

Note: These tests use synthetic data and mocks to isolate ML logic
from actual model training and file I/O.
"""

import json
import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

from src.models import (
    _validate_solver_penalty,
    create_model,
    train_model,
    evaluate_model,
    prepare_sample_weights,
    save_model,
    load_model,
    predict_sentiment,
    decode_predictions,
    save_evaluation_results,
)
from src.constants import TARGET_MAPPING, TARGET_MAPPING_INV


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_config():
    """Minimal valid config for model creation."""
    return {
        "model": {
            "type": "logistic_regression",
            "training": {
                "max_iter": 100,
                "class_weight": "balanced",
                "random_state": 42,
                "use_confidence_weights": True,
            },
            "regularization": {
                "solver": "lbfgs",
                "penalty": "l2",
                "C": 1.0,
            },
        },
        "evaluation": {
            "metrics": {"primary": ["accuracy", "f1_macro"]},
            "reporting": {"include_confusion_matrix": False},
        },
    }


@pytest.fixture
def synthetic_data_binary():
    """Synthetic binary classification data."""
    np.random.seed(42)
    texts = ["good great excellent"] * 50 + ["bad terrible awful"] * 50
    y = np.array([1] * 50 + [0] * 50)  # 1=positive, 0=negative
    conf = np.random.uniform(0.7, 1.0, size=100)
    return texts, y, conf


@pytest.fixture
def synthetic_data_multiclass():
    """Synthetic multiclass classification data."""
    np.random.seed(42)
    texts = (
        ["good great excellent"] * 40 +
        ["bad terrible awful"] * 40 +
        ["okay meh neutral"] * 40
    )
    y = np.array([1] * 40 + [0] * 40 + [2] * 40)  # 0=neg, 1=pos, 2=neu
    conf = np.random.uniform(0.7, 1.0, size=120)
    return texts, y, conf


@pytest.fixture
def mock_vectorizer():
    """Mock vectorizer for prediction tests."""
    vec = MagicMock(spec=TfidfVectorizer)
    vec.transform.return_value = csr_matrix(np.random.rand(1, 10))
    return vec


@pytest.fixture
def mock_trained_model():
    """Mock a trained LogisticRegression model."""
    model = MagicMock(spec=LogisticRegression)
    model.classes_ = np.array([0, 1, 2])
    model.predict.return_value = np.array([1])
    model.predict_proba.return_value = np.array([[0.1, 0.8, 0.1]])
    return model


# ============================================================================
# _validate_solver_penalty tests
# ============================================================================


class TestValidateSolverPenalty:
    """Tests for solver/penalty combination validation."""

    def test_valid_combinations(self):
        """Valid solver/penalty pairs should not raise."""
        _validate_solver_penalty("lbfgs", "l2")
        _validate_solver_penalty("lbfgs", "none")
        _validate_solver_penalty("liblinear", "l1")
        _validate_solver_penalty("liblinear", "l2")
        _validate_solver_penalty("saga", "l1")
        _validate_solver_penalty("saga", "elasticnet", l1_ratio=0.5)

    def test_invalid_solver(self):
        """Unknown solver should raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported solver"):
            _validate_solver_penalty("unknown_solver", "l2")

    def test_invalid_penalty_for_solver(self):
        """Invalid penalty for given solver should raise."""
        with pytest.raises(ValueError, match="Invalid combination"):
            _validate_solver_penalty("lbfgs", "l1")  # lbfgs doesn't support l1

    def test_elasticnet_requires_saga(self):
        """elasticnet penalty requires saga solver."""
        with pytest.raises(ValueError, match="elasticnet"):
            _validate_solver_penalty("lbfgs", "elasticnet")

    def test_l1_ratio_range_validation(self):
        """l1_ratio must be in [0.0, 1.0]."""
        # Valid values
        _validate_solver_penalty("saga", "elasticnet", l1_ratio=0.0)
        _validate_solver_penalty("saga", "elasticnet", l1_ratio=1.0)
        _validate_solver_penalty("saga", "elasticnet", l1_ratio=0.5)

        # Invalid values
        with pytest.raises(ValueError, match="l1_ratio must be in"):
            _validate_solver_penalty("saga", "elasticnet", l1_ratio=-0.1)
        with pytest.raises(ValueError, match="l1_ratio must be in"):
            _validate_solver_penalty("saga", "elasticnet", l1_ratio=1.5)


# ============================================================================
# create_model tests
# ============================================================================


class TestCreateModel:
    """Tests for model creation from config."""

    def test_minimal_valid_config(self, sample_config):
        """Should create LogisticRegression with default params."""
        model = create_model(sample_config)
        assert isinstance(model, LogisticRegression)
        assert model.max_iter == 100
        assert model.class_weight == "balanced"
        assert model.solver == "lbfgs"
        assert model.C == 1.0

    def test_missing_required_keys(self):
        """Should raise ValueError for missing required config keys."""
        # Missing model.type
        config = {"model": {"training": {"max_iter": 100}}}
        with pytest.raises(ValueError, match="model.type"):
            create_model(config)

        # Missing model.training.max_iter
        config = {"model": {"type": "logistic_regression"}}
        with pytest.raises(ValueError, match="model.training.max_iter"):
            create_model(config)

    def test_unsupported_model_type(self):
        """Should reject unsupported model types."""
        config = {
            "model": {
                "type": "random_forest",  # Not supported
                "training": {"max_iter": 100},
            }
        }
        with pytest.raises(ValueError, match="Unsupported model_type"):
            create_model(config)

    def test_class_weight_none_handling(self, sample_config):
        """class_weight='none' should be converted to None."""
        sample_config["model"]["training"]["class_weight"] = "none"
        model = create_model(sample_config)
        assert model.class_weight is None

    def test_penalty_none_sets_C_inf(self, sample_config):
        """penalty='none' should set C=np.inf and l1_ratio=None."""
        sample_config["model"]["regularization"]["penalty"] = "none"
        model = create_model(sample_config)
        assert model.C == np.inf

    def test_elasticnet_sets_l1_ratio(self, sample_config):
        """elasticnet penalty should use configured l1_ratio."""
        sample_config["model"]["regularization"].update({
            "penalty": "elasticnet",
            "solver": "saga",
            "l1_ratio": 0.7,
        })
        model = create_model(sample_config)
        assert model.solver == "saga"
        assert model.l1_ratio == 0.7
        assert model.C == 1.0

    def test_random_state_propagation(self, sample_config):
        """random_state from config should be passed to model."""
        sample_config["model"]["training"]["random_state"] = 123
        model = create_model(sample_config)
        assert model.random_state == 123

    def test_l1_penalty_sets_l1_ratio(self, sample_config):
        """penalty='l1' should set l1_ratio=1.0 explicitly."""
        sample_config["model"]["regularization"].update({
            "penalty": "l1",
            "solver": "liblinear",
        })
        model = create_model(sample_config)
        assert model.solver == "liblinear"
        assert getattr(model, "l1_ratio", None) in [None, 1.0]

    def test_non_elasticnet_does_not_set_l1_ratio(self, sample_config):
        """
        l1_ratio should not be set (or be 0.0) for non-elasticnet penalties.
        """
        # L2 penalty with lbfgs
        sample_config["model"]["regularization"].update({
            "penalty": "l2",
            "solver": "lbfgs",
        })
        model = create_model(sample_config)
        assert model.solver == "lbfgs"
        assert getattr(model, "l1_ratio", None) in [None, 0.0]


# ============================================================================
# train_model tests
# ============================================================================


class TestTrainModel:
    """Tests for model training with sample weights."""

    def test_basic_training(self, sample_config, synthetic_data_binary):
        """Should train model on synthetic data."""
        texts, y, conf = synthetic_data_binary

        # Vectorize
        vec = TfidfVectorizer(max_features=50)
        X = vec.fit_transform(texts)

        # Train
        model = train_model(X, y, sample_config, sample_weights=None)

        assert isinstance(model, LogisticRegression)
        assert model.classes_ is not None

    def test_training_with_sample_weights(
            self, sample_config, synthetic_data_binary
    ):
        """Should accept and use sample_weights."""
        texts, y, conf = synthetic_data_binary
        vec = TfidfVectorizer(max_features=50)
        X = vec.fit_transform(texts)

        # Train with weights
        model = train_model(X, y, sample_config, sample_weights=conf)

        assert isinstance(model, LogisticRegression)
        assert hasattr(model, "coef_"), (
            "Model should have coef_ attribute after fit"
        )
        assert model.coef_ is not None, "Model coefficients should not be None"

    def test_input_validation_none_inputs(self, sample_config):
        """Should raise ValueError for None inputs."""
        with pytest.raises(ValueError, match="cannot be None"):
            train_model(None, np.array([1]), sample_config)
        with pytest.raises(ValueError, match="cannot be None"):
            train_model(np.array([[1]]), None, sample_config)

    def test_input_validation_shape_mismatch(self, sample_config):
        """Should raise ValueError for shape mismatch."""
        X = np.random.rand(10, 5)
        y = np.array([1, 2, 3])  # Wrong length
        with pytest.raises(ValueError, match="Shape mismatch"):
            train_model(X, y, sample_config)

    def test_sample_weights_shape_validation(self, sample_config):
        """Should validate sample_weights length matches y."""
        X = np.random.rand(10, 5)
        y = np.array([1] * 10)
        weights = np.array([0.9] * 5)  # Wrong length
        with pytest.raises(ValueError, match="sample_weights length"):
            train_model(X, y, sample_config, sample_weights=weights)

    def test_sparse_matrix_support(self, sample_config, synthetic_data_binary):
        """Should handle scipy sparse matrices."""
        texts, y, _ = synthetic_data_binary
        vec = TfidfVectorizer(max_features=50)
        X = vec.fit_transform(texts)  # Returns sparse matrix

        # Should not raise
        model = train_model(X, y, sample_config)
        assert isinstance(model, LogisticRegression)


# ============================================================================
# evaluate_model tests
# ============================================================================


class TestEvaluateModel:
    """Tests for model evaluation metrics."""

    def test_basic_evaluation_binary(
            self, sample_config, synthetic_data_binary
    ):
        """Should compute metrics for binary classification."""
        texts, y, _ = synthetic_data_binary
        vec = TfidfVectorizer(max_features=50)
        X = vec.fit_transform(texts)

        # Train simple model
        model = LogisticRegression(max_iter=100, random_state=42)
        model.fit(X, y)

        # Evaluate
        results = evaluate_model(model, X, y, sample_config)

        assert "accuracy" in results, (
            f"Expected 'accuracy' in results, got keys: {list(results.keys())}"
        )
        assert "f1_macro" in results, (
            f"Expected 'f1_macro' in results, got keys: {list(results.keys())}"
        )
        assert 0.0 <= results["accuracy"] <= 1.0, (
            f"Expected accuracy in [0,1], got {results['accuracy']}"
        )

    def test_multiclass_metrics(
            self, sample_config, synthetic_data_multiclass
    ):
        """Should compute multiclass metrics correctly."""
        texts, y, _ = synthetic_data_multiclass
        vec = TfidfVectorizer(max_features=50)
        X = vec.fit_transform(texts)

        model = LogisticRegression(max_iter=100, random_state=42)
        model.fit(X, y)

        sample_config["evaluation"]["metrics"]["primary"].append("f1_weighted")

        results = evaluate_model(model, X, y, sample_config)

        assert "accuracy" in results, (
            f"Expected 'accuracy' in results, got keys: {list(results.keys())}"
        )
        assert "f1_macro" in results, (
            f"Expected 'f1_macro' in results, got keys: {list(results.keys())}"
        )
        assert "f1_weighted" in results, (
            "Expected 'f1_weighted' in results, "
            f"got keys: {list(results.keys())}"
        )

    def test_empty_dataset_handling(self, sample_config):
        """Should return empty dict for empty dataset."""
        model = MagicMock(spec=LogisticRegression)
        results = evaluate_model(
            model, np.array([]), np.array([]), sample_config
        )
        assert results == {}

    def test_input_validation(self, sample_config):
        """Should validate X and y_true are not None."""
        model = MagicMock()
        with pytest.raises(ValueError, match="cannot be None"):
            evaluate_model(model, None, np.array([1]), sample_config)
        with pytest.raises(ValueError, match="cannot be None"):
            evaluate_model(model, np.array([[1]]), None, sample_config)

    def test_shape_mismatch_validation(self, sample_config):
        """Should validate X and y_true have same length."""
        model = MagicMock()
        X = np.random.rand(10, 5)
        y = np.array([1, 2, 3])  # Wrong length
        with pytest.raises(ValueError, match="Shape mismatch"):
            evaluate_model(model, X, y, sample_config)

    def test_empty_proba_skips_auc(self, sample_config, caplog):
        """Should handle empty dataset gracefully."""
        model = MagicMock()
        model.predict_proba.return_value = np.array([]).reshape(0, 3)
        model.predict.return_value = np.array([])

        with caplog.at_level(logging.WARNING, logger="src.models"):
            results = evaluate_model(
                model, np.array([]), np.array([]), sample_config
            )

        assert results == {}
        assert any(
            "empty dataset" in r.message.lower() for r in caplog.records
        ), (
            "Expected 'empty dataset' warning, "
            f"got: {[r.message for r in caplog.records]}"
        )


# ============================================================================
# prepare_sample_weights tests
# ============================================================================


class TestPrepareSampleWeights:
    """Tests for confidence-based sample weight preparation."""

    def test_basic_extraction(self):
        """Should extract confidence column values."""
        df = pd.DataFrame({
            "sentiment_confidence": [0.8, 0.9, 0.7, 1.0],
            "other_col": ["a", "b", "c", "d"]
        })
        weights = prepare_sample_weights(df, "sentiment_confidence")

        np.testing.assert_array_equal(weights, [0.8, 0.9, 0.7, 1.0])

    def test_missing_column_raises_error(self):
        """Should raise ValueError if confidence column not found."""
        df = pd.DataFrame({"other_col": [1, 2, 3]})
        with pytest.raises(ValueError, match="not found in DataFrame"):
            prepare_sample_weights(df, "missing_column")

    def test_missing_values_filled_with_default(self, caplog):
        """Should fill NaN values with 0.5."""
        df = pd.DataFrame({
            "sentiment_confidence": [0.8, np.nan, 0.9, np.nan]
        })
        with caplog.at_level("WARNING"):
            weights = prepare_sample_weights(df, "sentiment_confidence")

        assert np.isnan(weights).sum() == 0
        assert 0.5 in weights
        assert any("missing values" in r.message for r in caplog.records)

    def test_normalization_when_needed(self):
        """Should normalize if values outside [0, 1]."""
        df = pd.DataFrame({"conf": [10, 20, 30]})
        weights = prepare_sample_weights(df, "conf", normalize=True)

        # Should be normalized to [0, 1]
        assert weights.min() >= 0.0
        assert weights.max() <= 1.0

    def test_no_normalization_when_in_range(self):
        """Should not normalize if values already in [0, 1]."""
        df = pd.DataFrame({"conf": [0.7, 0.8, 0.9]})
        weights = prepare_sample_weights(df, "conf", normalize=True)

        # Should remain unchanged
        np.testing.assert_array_almost_equal(weights, [0.7, 0.8, 0.9])

    def test_normalization_exact_formula(self):
        """Normalization should use (x - min) / (max - min)."""
        df = pd.DataFrame({"conf": [10, 20, 30]})  # min=10, max=30
        weights = prepare_sample_weights(df, "conf", normalize=True)

        # Expected: (10-10)/(30-10)=0, (20-10)/20=0.5, (30-10)/20=1
        np.testing.assert_array_almost_equal(
            weights, [0.0, 0.5, 1.0],
            err_msg=f"Expected [0.0, 0.5, 1.0], got: {weights}"
        )


# ============================================================================
# save_model / load_model tests
# ============================================================================


class TestModelPersistence:
    """Tests for model save/load functionality."""

    def test_save_model_creates_file(
            self, tmp_path, mock_trained_model, mock_vectorizer
    ):
        """Should save model bundle to disk."""
        with patch("src.models.joblib.dump") as mock_dump:
            mock_dump.return_value = None

            path = save_model(
                mock_trained_model,
                mock_vectorizer,
                tmp_path,
                filename="test_bundle.joblib"
            )

            mock_dump.assert_called_once()
            call_args = mock_dump.call_args[0]
            assert call_args[1] == tmp_path / "test_bundle.joblib"
            assert path == tmp_path / "test_bundle.joblib"

    def test_load_model_restores_bundle(
            self, tmp_path, mock_trained_model, mock_vectorizer
    ):
        """Should load model bundle and return all components."""
        bundle_path = tmp_path / "bundle.joblib"

        with patch("src.models.joblib.load") as mock_load:
            mock_load.return_value = {
                "model": mock_trained_model,
                "vectorizer": mock_vectorizer,
                "target_mapping": TARGET_MAPPING,
                "target_mapping_inv": TARGET_MAPPING_INV,
            }

            model, vec, mapping, mapping_inv = load_model(bundle_path)

            assert model is mock_trained_model
            assert vec is mock_vectorizer
            assert mapping == TARGET_MAPPING
            assert mapping_inv == TARGET_MAPPING_INV
            mock_load.assert_called_once_with(bundle_path)

    def test_load_model_warns_on_legacy_vectorizer(
            self, tmp_path, mock_trained_model, caplog
    ):
        """
        Bundles trained before cleaning was embedded into the vectorizer
        skip cleaning at inference — loading one must warn loudly.
        """
        legacy_vectorizer = TfidfVectorizer()  # preprocessor=None
        with patch("src.models.joblib.load") as mock_load:
            mock_load.return_value = {
                "model": mock_trained_model,
                "vectorizer": legacy_vectorizer,
                "target_mapping": TARGET_MAPPING,
                "target_mapping_inv": TARGET_MAPPING_INV,
            }
            with caplog.at_level(logging.WARNING, logger="src.models"):
                load_model(tmp_path / "legacy.joblib")

        assert any(
            "no embedded preprocessor" in r.message for r in caplog.records
        )

    def test_load_model_no_warning_with_embedded_preprocessor(
            self, tmp_path, mock_trained_model, caplog
    ):
        """Current bundles (preprocessor set) load without the warning."""
        from src.preprocessing import create_vectorizer

        vectorizer = create_vectorizer({
            "preprocessing": {
                "vectorizer": {
                    "type": "tfidf", "max_features": 10, "ngram_range": [1, 1]
                },
            }
        })
        with patch("src.models.joblib.load") as mock_load:
            mock_load.return_value = {
                "model": mock_trained_model,
                "vectorizer": vectorizer,
                "target_mapping": TARGET_MAPPING,
                "target_mapping_inv": TARGET_MAPPING_INV,
            }
            with caplog.at_level(logging.WARNING, logger="src.models"):
                load_model(tmp_path / "current.joblib")

        assert not any(
            "no embedded preprocessor" in r.message for r in caplog.records
        )

    def test_save_model_creates_parent_dirs(
            self, tmp_path, mock_trained_model, mock_vectorizer
    ):
        """Should create parent directories if they don't exist."""
        nested_path = tmp_path / "deep" / "nested" / "path"

        with patch("src.models.joblib.dump") as mock_dump:
            path = save_model(
                mock_trained_model,
                mock_vectorizer,
                nested_path,
                filename="model.joblib"
            )

            assert path.parent.exists(), (
                f"Expected parent dir at {path.parent}"
            )
            mock_dump.assert_called_once()
            assert mock_dump.call_args[0][1] == nested_path / "model.joblib"


# ============================================================================
# predict_sentiment tests
# ============================================================================


class TestPredictSentiment:
    """Tests for prediction logic."""

    def test_single_text_prediction(self, mock_trained_model, mock_vectorizer):
        """Should predict single text and return label."""
        result = predict_sentiment(
            mock_trained_model,
            mock_vectorizer,
            "Great flight!",
            return_proba=False
        )

        assert isinstance(result, np.ndarray)
        assert result[0] == 1  # Mock returns class 1

    def test_single_text_with_proba(self, mock_trained_model, mock_vectorizer):
        """Should return both label and probabilities when requested."""
        labels, proba = predict_sentiment(
            mock_trained_model,
            mock_vectorizer,
            "Great flight!",
            return_proba=True
        )

        assert isinstance(labels, np.ndarray)
        assert isinstance(proba, np.ndarray)
        assert proba.shape == (1, 3)  # 1 sample, 3 classes

    def test_batch_prediction(self, mock_trained_model, mock_vectorizer):
        """Should handle list of texts."""
        texts = ["text1", "text2", "text3"]

        mock_trained_model.predict.return_value = np.array([1, 0, 1])
        mock_trained_model.predict_proba.return_value = np.array([
            [0.1, 0.8, 0.1],
            [0.7, 0.2, 0.1],
            [0.2, 0.7, 0.1],
        ])

        result = predict_sentiment(
            mock_trained_model,
            mock_vectorizer,
            texts,
            return_proba=False
        )

        assert len(result) == 3
        assert all(isinstance(p, (int, np.integer)) for p in result)

    def test_empty_input_handling(
            self, mock_trained_model, mock_vectorizer, caplog
    ):
        """Should handle empty input gracefully."""
        with caplog.at_level("WARNING"):
            result = predict_sentiment(
                mock_trained_model,
                mock_vectorizer,
                [],
                return_proba=False
            )

        assert len(result) == 0
        assert any("empty text list" in r.message for r in caplog.records)

    def test_empty_input_with_proba(self, mock_trained_model, mock_vectorizer):
        """Should return empty arrays with correct shape for proba."""
        labels, proba = predict_sentiment(
            mock_trained_model,
            mock_vectorizer,
            [],
            return_proba=True
        )

        assert len(labels) == 0
        assert proba.shape == (0, 3)  # 0 samples, 3 classes (mock default)

    def test_sparse_matrix_vectorization(self, mock_trained_model):
        """Should work with sparse matrix output from vectorizer."""
        # Mock vectorizer to return sparse matrix
        mock_vec = MagicMock()
        mock_vec.transform.return_value = csr_matrix(np.random.rand(1, 10))
        mock_trained_model.predict.return_value = np.array([1])

        result = predict_sentiment(mock_trained_model, mock_vec, ["test"])

        assert len(result) == 1


# ============================================================================
# decode_predictions tests
# ============================================================================


class TestDecodePredictions:
    """Tests for label decoding."""

    def test_decode_known_labels(self):
        """Should decode known integer labels to strings."""
        predictions = np.array([0, 1, 2, 1, 0])
        labels = decode_predictions(predictions)

        assert labels == [
            "negative", "positive", "neutral", "positive", "negative"
        ]

    def test_decode_with_custom_mapping(self):
        """Should use provided mapping_inv if given."""
        custom_mapping = {0: "bad", 1: "good"}
        predictions = np.array([0, 1, 0])
        labels = decode_predictions(predictions, mapping_inv=custom_mapping)

        assert labels == ["bad", "good", "bad"]

    def test_unknown_label_non_strict(self, caplog):
        """Should return 'unknown' for unmapped labels in non-strict mode."""
        predictions = np.array([0, 99, 1])  # 99 not in mapping
        with caplog.at_level("WARNING"):
            labels = decode_predictions(predictions, strict=False)

        assert "unknown" in labels
        assert any(
            "Unknown prediction label" in r.message for r in caplog.records
        )

    def test_unknown_label_strict_raises(self):
        """Should raise ValueError for unmapped labels in strict mode."""
        predictions = np.array([0, 99, 1])
        with pytest.raises(ValueError, match="Unknown prediction label"):
            decode_predictions(predictions, strict=True)

    def test_decode_empty_array(self):
        """Should handle empty predictions array gracefully."""
        predictions = np.array([])
        labels = decode_predictions(predictions)
        assert labels == [], f"Expected empty list, got: {labels}"


# ============================================================================
# save_evaluation_results tests
# ============================================================================


class TestSaveEvaluationResults:
    """Tests for metrics export functionality."""

    def test_save_metrics_to_json(self, tmp_path):
        """Should save metrics dict to JSON file."""
        metrics = {
            "accuracy": 0.85,
            "f1_macro": 0.78,
            "confusion_matrix": [[10, 2], [3, 15]],
        }

        path = save_evaluation_results(
            metrics, tmp_path, filename="metrics.json"
        )

        assert path.exists(), f"Expected file at {path}"
        assert path.suffix == ".json"
        assert path.name == "metrics.json"

        # Verify content
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)

        assert loaded["accuracy"] == 0.85, (
            f"Expected accuracy=0.85, got {loaded.get('accuracy')}"
        )
        assert loaded["confusion_matrix"] == [[10, 2], [3, 15]]

    def test_numpy_serialization(self, tmp_path):
        """Should convert numpy types to JSON-serializable formats."""
        metrics = {
            "accuracy": np.float64(0.85),
            "count": np.int32(100),
            "matrix": np.array([[1, 2], [3, 4]]),
        }

        path = save_evaluation_results(metrics, tmp_path)

        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)

        # Should be native Python types
        assert isinstance(loaded["accuracy"], float), (
            f"Expected float, got {type(loaded['accuracy'])}"
        )
        assert isinstance(loaded["count"], (int, float)), (
            f"Expected int/float, got {type(loaded['count'])}"
        )
        assert isinstance(loaded["matrix"], list), (
            f"Expected list, got {type(loaded['matrix'])}"
        )

    def test_creates_parent_directories(self, tmp_path):
        """Should create parent directories for output path."""
        metrics = {"accuracy": 0.9}

        nested = tmp_path / "reports" / "eval"
        path = save_evaluation_results(metrics, nested)

        assert path.parent.exists(), f"Expected parent dir at {path.parent}"
        assert path.exists(), f"Expected file at {path}"
