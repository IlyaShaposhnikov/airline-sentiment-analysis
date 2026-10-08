"""
Integration tests for the full Airline Sentiment Analysis pipeline.

Tests end-to-end flow: config → data → preprocessing → training →
evaluation → persistence → prediction → API service.

Uses synthetic data and minimal mocks to ensure deterministic, fast execution.
"""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml

from src.constants import TARGET_MAPPING, TARGET_MAPPING_INV
from src.data_loader import (
    load_config,
    load_and_prepare_data,
    split_train_test_indices,
)
from src.preprocessing import preprocess_texts, create_vectorizer
from src.models import (
    train_model, evaluate_model, prepare_sample_weights,
    save_model, load_model, predict_sentiment, decode_predictions,
    save_evaluation_results,
)
from src.api.services import ModelService
from src.api.models import PredictionRequest, PredictionResponse


# ============================================================================
# Fixtures for synthetic pipeline
# ============================================================================


@pytest.fixture
def synthetic_pipeline_data(tmp_path: Path):
    """
    Generate synthetic dataset CSV and corresponding config.

    Returns:
        Tuple of (tmp_path, config_path, csv_path)
    """
    # 1. Create clear, separable synthetic data
    np.random.seed(42)
    n_samples = 300
    texts = (
        ["great flight excellent service happy"] * 100 +
        ["terrible delay awful experience never"] * 100 +
        ["okay flight neutral nothing special"] * 100
    )
    sentiments = ["positive"] * 100 + ["negative"] * 100 + ["neutral"] * 100
    confidences = np.random.uniform(0.85, 0.99, n_samples)

    df = pd.DataFrame({
        "airline_sentiment": sentiments,
        "text": texts,
        "airline_sentiment_confidence": confidences,
        "negativereason_confidence": np.random.uniform(0.5, 0.9, n_samples),
    })

    # Save to CSV (absolute path to avoid resolution issues)
    csv_path = tmp_path / "synthetic_data.csv"
    df.to_csv(csv_path, index=False)

    # 2. Create minimal valid config matching project structure
    config = {
        "data": {
            "path": str(csv_path),  # Absolute path
            "target_column": "airline_sentiment",
            "text_column": "text",
            "confidence_columns": {
                "sentiment": "airline_sentiment_confidence",
                "reason": "negativereason_confidence",
            },
            "confidence_threshold": 0.7,
        },
        "preprocessing": {
            "vectorizer": {
                "type": "tfidf",
                "max_features": 50,
                "ngram_range": [1, 2],
                "lowercase": True,
            },
            "cleaning": {
                "lowercase": True,
                "remove_urls": True,
                "remove_mentions": False,
                "remove_special_chars": True,
                "remove_extra_whitespace": True
            },
            "nlp": {"lemmatize": False, "remove_stopwords": False},
        },
        "model": {
            "type": "logistic_regression",
            "training": {
                "max_iter": 200, "class_weight": "balanced",
                "random_state": 42, "use_confidence_weights": True
            },
            "regularization": {"solver": "lbfgs", "penalty": "l2", "C": 1.0},
        },
        "evaluation": {
            "split": {"test_size": 0.2, "stratify": True},
            "metrics": {"primary": ["accuracy", "f1_macro", "f1_weighted"]},
            "reporting": {
                "include_confusion_matrix": False,
                "include_classification_report": False,
                "export_formats": ["json"],
            },
        },
    }

    # Save config YAML
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False)

    return tmp_path, config_path, csv_path


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.mark.integration
class TestFullPipelineIntegration:
    """End-to-end integration tests for the ML & API pipeline."""

    @pytest.mark.asyncio
    async def test_end_to_end_pipeline(self, synthetic_pipeline_data):
        """
        Test complete flow:
        config → data → preprocess → train → evaluate →
        save → load → predict → API service
        """
        tmp_path, config_path, _ = synthetic_pipeline_data

        # =========================================================================
        # 1. Load config & data
        # =========================================================================
        config = load_config(config_path)
        df = load_and_prepare_data(config_path=config_path, base_dir=tmp_path)

        assert len(df) > 0
        assert "target" in df.columns
        assert "sentiment_confidence" in df.columns
        assert df["target"].notna().all()
        assert set(df["target"].unique()).issubset(TARGET_MAPPING.values())

        # =========================================================================
        # 2. Preprocessing sanity check
        # =========================================================================
        texts = preprocess_texts(df["text"].tolist(), config)
        assert len(texts) == len(df)
        assert all(isinstance(t, str) and len(t) > 0 for t in texts)

        # =========================================================================
        # 3. Split by position, then fit vectorizer on train only
        #    (same order as scripts/train.py; raw texts — cleaning is
        #    embedded in the vectorizer)
        # =========================================================================
        y = df["target"].to_numpy()
        train_idx, test_idx = split_train_test_indices(y, config)
        raw_texts = df["text"].tolist()

        vectorizer = create_vectorizer(config)
        X_train = vectorizer.fit_transform([raw_texts[i] for i in train_idx])
        X_test = vectorizer.transform([raw_texts[i] for i in test_idx])
        y_train, y_test = y[train_idx], y[test_idx]
        w_train = df["sentiment_confidence"].to_numpy()[train_idx]
        assert X_train.shape[0] == len(y_train)
        assert X_test.shape[0] == len(y_test)

        sample_weights = prepare_sample_weights(
            pd.DataFrame({"conf": w_train}), "conf", normalize=False
        )
        model = train_model(
            X_train, y_train, config, sample_weights=sample_weights
        )
        assert model is not None and hasattr(model, "coef_")

        # =========================================================================
        # 4. Evaluation
        # =========================================================================
        eval_results = evaluate_model(model, X_test, y_test, config)
        assert "accuracy" in eval_results
        assert 0.8 <= eval_results["accuracy"] <= 1.0, (
            "Expected high accuracy on synthetic data, "
            f"got {eval_results['accuracy']:.3f}"
        )
        assert "f1_macro" in eval_results

        # =========================================================================
        # 5. Persistence (Save/Load)
        # =========================================================================
        model_path = tmp_path / "test_bundle.joblib"
        save_model(model, vectorizer, tmp_path, filename="test_bundle.joblib")
        assert model_path.exists()

        loaded_model, loaded_vec, mapping, mapping_inv = load_model(model_path)
        assert loaded_model.classes_ is not None
        assert mapping == TARGET_MAPPING
        assert mapping_inv == TARGET_MAPPING_INV

        # =========================================================================
        # 6. Prediction & Decoding
        # =========================================================================
        new_texts = [
            "great excellent flight",
            "terrible awful delay",
            # "okay experience" was ambiguous: "experience" only occurs in
            # negative samples, so the old (vacuous) assert hid a ~50/50 call
            "okay nothing special",
        ]
        preds, probas = predict_sentiment(
            loaded_model, loaded_vec, new_texts, return_proba=True
        )

        assert len(preds) == 3
        assert probas.shape == (3, len(loaded_model.classes_))
        assert all(0.0 <= p <= 1.0 for row in probas for p in row)

        labels = decode_predictions(preds, mapping_inv)
        # Synthetic data is perfectly separable: predictions must be exact
        assert labels == ["positive", "negative", "neutral"]

        # =========================================================================
        # 7. Metrics Export
        # =========================================================================
        metrics_path = save_evaluation_results(
            eval_results, tmp_path, filename="test_metrics.json"
        )
        assert metrics_path.exists()
        with open(metrics_path, "r", encoding="utf-8") as f:
            saved_metrics = json.load(f)
        assert "accuracy" in saved_metrics
        assert isinstance(saved_metrics["accuracy"], float)

        # =========================================================================
        # 8. API Service Integration (with minimal mocks)
        # =========================================================================
        with patch("src.api.services.load_model") as mock_load, \
             patch("src.api.services.explain_prediction") as mock_exp, \
             patch("src.api.services.is_model_available") as mock_avail:

            # Mock returns the ACTUAL trained model from this test
            mock_load.return_value = (
                loaded_model, loaded_vec, mapping, mapping_inv
            )
            mock_exp.return_value = {
                "method": "weights",
                "top_contributors": [("great", 1.2), ("excellent", 0.9)]
            }
            mock_avail.return_value = True

            service = ModelService(model_path=str(model_path))
            service.load()  # Uses mock
            assert service.is_loaded is True
            assert set(service._class_names) == {
                "negative", "positive", "neutral"
            }
            assert len(service._class_names) == 3

            # Single prediction via service
            request = PredictionRequest(
                text="great excellent flight", explain=True, n_explain=5
            )
            response = await service.predict_single(request)

            assert isinstance(response, PredictionResponse)
            assert response.predicted_class == "positive"
            assert response.confidence > 0.7
            assert response.explanation is not None
            assert response.explanation.method == "weights"
            assert len(response.explanation.top_contributors) == 2

            # Batch prediction via service
            batch_results = await service.predict_batch(
                texts=["terrible delay", "okay flight"],
                explain=False, n_explain=5
            )
            assert len(batch_results) == 2
            assert batch_results[0].predicted_class == "negative"
            assert batch_results[1].predicted_class == "neutral"
