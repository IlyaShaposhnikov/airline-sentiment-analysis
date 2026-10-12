"""
Model metadata exposed by the API: ModelService.model_info /
explanations_supported and the extra /health fields.
"""
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.models import HealthResponse
from src.api.services import ModelService
from src.constants import TARGET_MAPPING, TARGET_MAPPING_INV
from src.models import train_model
from src.preprocessing import create_vectorizer

TEXTS = ["great love thanks"] * 20 + ["awful delay rude"] * 20
LABELS = np.array([1] * 20 + [0] * 20)

LR_CONFIG = {
    "model": {
        "type": "logistic_regression",
        "training": {"max_iter": 200, "random_state": 0},
        "regularization": {"solver": "lbfgs", "penalty": "l2", "C": 1.0},
    },
}


def _bundle(vectorizer_type: str):
    config = {
        **LR_CONFIG,
        "preprocessing": {
            "vectorizer": {
                "type": vectorizer_type, "max_features": 50,
                "ngram_range": [1, 1],
            },
            "embedding": {"model_name": "fake/model"},
        },
    }
    vec = create_vectorizer(config)
    model = train_model(vec.fit_transform(TEXTS), LABELS, config)
    return model, vec, TARGET_MAPPING, TARGET_MAPPING_INV


def _loaded_service(bundle) -> ModelService:
    with patch("src.api.services.load_model") as mock_load, \
         patch("src.api.services.is_model_available") as mock_avail:
        mock_load.return_value = bundle
        mock_avail.return_value = True
        service = ModelService(model_path="unused.joblib")
        service.load()
    return service


class TestModelServiceInfo:

    def test_info_empty_before_load(self):
        service = ModelService(model_path="unused.joblib")
        assert service.model_info == {
            "model_type": None, "vectorizer_type": None,
        }

    def test_tfidf_lr_supports_explanations(self):
        service = _loaded_service(_bundle("tfidf"))
        assert service.model_info == {
            "model_type": "LogisticRegression",
            "vectorizer_type": "TfidfVectorizer",
        }
        assert service.explanations_supported is True

    def test_embedding_bundle_has_no_explanations(self, fake_encoder):
        service = _loaded_service(_bundle("sentence_embedding"))
        assert service.model_info["vectorizer_type"] == (
            "SentenceEmbeddingVectorizer"
        )
        assert service.explanations_supported is False


class TestHealthModelFields:

    def test_new_fields_optional(self):
        health = HealthResponse(model_loaded=False, shap_available=False)
        assert health.model_type is None
        assert health.vectorizer_type is None
        assert health.explanations_supported is None

    @pytest.mark.parametrize(
        "vectorizer_type, expected_vec, expected_explain",
        [
            ("tfidf", "TfidfVectorizer", True),
            ("sentence_embedding", "SentenceEmbeddingVectorizer", False),
        ],
    )
    def test_health_reports_loaded_model(
        self, fake_encoder, vectorizer_type, expected_vec, expected_explain
    ):
        import src.api.main as main

        service = _loaded_service(_bundle(vectorizer_type))
        with patch.object(main, "model_service", service):
            # No context manager: startup (real model preload) is not run
            response = TestClient(main.app).get("/health")

        assert response.status_code == 200
        body = response.json()
        assert body["model_loaded"] is True
        assert body["model_type"] == "LogisticRegression"
        assert body["vectorizer_type"] == expected_vec
        assert body["explanations_supported"] is expected_explain

    def test_health_without_model(self):
        import src.api.main as main

        service = ModelService(model_path="unused.joblib")
        with patch.object(main, "model_service", service):
            body = TestClient(main.app).get("/health").json()

        assert body["model_loaded"] is False
        assert body["model_type"] is None
        assert body["explanations_supported"] is None
