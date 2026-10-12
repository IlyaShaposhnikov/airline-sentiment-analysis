"""
Model metadata exposed by the API: ModelService.model_info /
explanations_supported and the extra /health fields.

The /health handler is called directly (no TestClient): the HTTP test client
depends on a transport package whose name differs between Starlette
versions, and the HTTP layer adds nothing to what is tested here.
"""
from unittest.mock import patch

import numpy as np
import pytest

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
    async def test_health_reports_loaded_model(
        self, fake_encoder, vectorizer_type, expected_vec, expected_explain
    ):
        import src.api.main as main

        service = _loaded_service(_bundle(vectorizer_type))
        with patch.object(main, "model_service", service):
            health = await main.health_check()

        body = health.model_dump(mode="json")
        assert body["model_loaded"] is True
        assert body["model_type"] == "LogisticRegression"
        assert body["vectorizer_type"] == expected_vec
        assert body["explanations_supported"] is expected_explain

    async def test_health_without_model(self):
        import src.api.main as main

        service = ModelService(model_path="unused.joblib")
        with patch.object(main, "model_service", service):
            body = (await main.health_check()).model_dump(mode="json")

        assert body["model_loaded"] is False
        assert body["model_type"] is None
        assert body["explanations_supported"] is None

    def test_app_uses_lifespan_not_on_event(self):
        """Startup/shutdown are wired via lifespan (on_event is deprecated)."""
        import src.api.main as main

        router = main.app.router
        assert not getattr(router, "on_startup", [])
        assert not getattr(router, "on_shutdown", [])
