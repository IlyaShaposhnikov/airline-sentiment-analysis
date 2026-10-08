"""
Tests for src/embeddings.py and the sentence-embedding pipeline path.

Unit tests replace the Sentence-Transformers loader with a deterministic
fake encoder (``fake_encoder`` fixture in conftest.py), so they run
without torch and without downloading a model.
One smoke test uses the real model and is skipped when the optional
deep-learning dependencies are not installed.
"""
import sys

import joblib
import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from src.embeddings import SentenceEmbeddingVectorizer
from src.interpretability import (
    explain_prediction,
    supports_word_explanations,
)
from src.models import (
    load_model,
    predict_sentiment,
    save_model,
    train_model,
)
from src.preprocessing import (
    EMBEDDING_CLEANING_DEFAULTS,
    build_text_preprocessor,
    create_vectorizer,
)

@pytest.fixture
def embedding_config() -> dict:
    return {
        "preprocessing": {
            "vectorizer": {"type": "sentence_embedding"},
            "embedding": {
                "model_name": "fake/model",
                "batch_size": 16,
                "normalize": True,
                "device": "cpu",
            },
        },
        "model": {
            "type": "logistic_regression",
            "training": {
                "max_iter": 300, "class_weight": "balanced",
                "random_state": 42, "use_confidence_weights": True,
            },
            "regularization": {"solver": "lbfgs", "penalty": "l2", "C": 1.0},
        },
    }


# ============================================================================
# SentenceEmbeddingVectorizer
# ============================================================================


class TestSentenceEmbeddingVectorizer:

    def test_transform_returns_dense_float32_matrix(self, fake_encoder):
        vec = SentenceEmbeddingVectorizer().fit(["a", "b"])
        X = vec.transform(["great flight", "awful delay", "gate"])

        assert isinstance(X, np.ndarray)
        assert X.dtype == np.float32
        assert X.shape == (3, fake_encoder.dim)

    def test_fit_records_embedding_dim(self, fake_encoder):
        vec = SentenceEmbeddingVectorizer().fit(["anything"])
        assert vec.embedding_dim_ == fake_encoder.dim
        assert vec.n_features_in_ == fake_encoder.dim

    def test_fit_transform_matches_transform(self, fake_encoder):
        texts = ["great flight", "awful delay"]
        vec = SentenceEmbeddingVectorizer()
        np.testing.assert_allclose(
            vec.fit_transform(texts), vec.transform(texts)
        )

    def test_settings_passed_to_encoder(self, fake_encoder):
        vec = SentenceEmbeddingVectorizer(
            batch_size=7, normalize_embeddings=False
        )
        vec.transform(["great"])
        assert fake_encoder.calls[-1]["batch_size"] == 7
        assert fake_encoder.calls[-1]["normalize_embeddings"] is False

    def test_preprocessor_applied_before_encoding(self, fake_encoder):
        preprocessor = build_text_preprocessor(
            {"preprocessing": {"cleaning": EMBEDDING_CLEANING_DEFAULTS}},
            lowercase=False,
        )
        vec = SentenceEmbeddingVectorizer(preprocessor=preprocessor)
        vec.transform(["@united Great flight!! 😍 https://t.co/abc"])

        # Light profile: mention + URL removed; casing, "!!", emoji kept
        assert fake_encoder.calls[-1]["texts"] == ["Great flight!! 😍"]

    def test_encoder_loaded_lazily_and_once(self, fake_encoder):
        vec = SentenceEmbeddingVectorizer()
        assert fake_encoder.loads["count"] == 0

        vec.transform(["one"])
        vec.transform(["two"])
        assert fake_encoder.loads["count"] == 1

    def test_warmup_loads_encoder(self, fake_encoder):
        vec = SentenceEmbeddingVectorizer()
        vec.warmup()
        assert fake_encoder.loads["count"] == 1

    def test_empty_input_returns_empty_matrix(self, fake_encoder):
        vec = SentenceEmbeddingVectorizer().fit(["x"])
        X = vec.transform([])
        assert X.shape == (0, fake_encoder.dim)

    def test_single_string_rejected(self, fake_encoder):
        """A bare string would be encoded character by character."""
        with pytest.raises(TypeError, match="single string"):
            SentenceEmbeddingVectorizer().transform("great flight")

    def test_none_text_treated_as_empty(self, fake_encoder):
        vec = SentenceEmbeddingVectorizer()
        assert vec.transform([None]).shape == (1, fake_encoder.dim)

    def test_pickle_excludes_encoder(self, fake_encoder, tmp_path):
        """Bundle must store settings only, not the heavy encoder."""
        vec = SentenceEmbeddingVectorizer(model_name="fake/model").fit(["x"])
        assert getattr(vec, "_encoder", None) is not None

        path = tmp_path / "vec.joblib"
        joblib.dump(vec, path)
        loaded = joblib.load(path)

        assert getattr(loaded, "_encoder", None) is None
        assert loaded.model_name == "fake/model"
        assert loaded.embedding_dim_ == fake_encoder.dim
        # Encoder is re-loaded on first use after unpickling
        assert loaded.transform(["great"]).shape == (1, fake_encoder.dim)
        assert fake_encoder.loads["count"] == 2

    def test_missing_dependency_gives_install_hint(self, monkeypatch):
        """Without sentence-transformers the error must say what to do."""
        monkeypatch.setitem(sys.modules, "sentence_transformers", None)
        vec = SentenceEmbeddingVectorizer()
        with pytest.raises(ImportError, match="requirements-dl.txt"):
            vec.transform(["great"])


# ============================================================================
# Factory, interpretability gating, full model lifecycle
# ============================================================================


class TestEmbeddingPipelineIntegration:

    def test_factory_builds_embedding_vectorizer(self, embedding_config):
        vec = create_vectorizer(embedding_config)

        assert isinstance(vec, SentenceEmbeddingVectorizer)
        assert vec.model_name == "fake/model"
        assert vec.batch_size == 16
        assert vec.preprocessor is not None

    def test_factory_does_not_require_bow_keys(self):
        """max_features/ngram_range are TF-IDF settings, not required here."""
        vec = create_vectorizer(
            {"preprocessing": {"vectorizer": {"type": "sentence_embedding"}}}
        )
        assert isinstance(vec, SentenceEmbeddingVectorizer)

    def test_factory_uses_light_cleaning_not_tfidf_cleaning(
        self, embedding_config, fake_encoder
    ):
        """Aggressive TF-IDF cleaning must not leak into the encoder path."""
        embedding_config["preprocessing"]["cleaning"] = {
            "lowercase": True, "remove_special_chars": True,
        }
        embedding_config["preprocessing"]["nlp"] = {"lemmatize": True}
        vec = create_vectorizer(embedding_config)
        vec.transform(["Great flights!!"])

        assert fake_encoder.calls[-1]["texts"] == ["Great flights!!"]

    def test_embedding_cleaning_overridable(
        self, embedding_config, fake_encoder
    ):
        embedding_config["preprocessing"]["embedding"]["cleaning"] = {
            "lowercase": True
        }
        create_vectorizer(embedding_config).transform(["GREAT"])
        assert fake_encoder.calls[-1]["texts"] == ["great"]

    def test_word_explanations_disabled_for_embeddings(self):
        assert supports_word_explanations(TfidfVectorizer())
        assert not supports_word_explanations(SentenceEmbeddingVectorizer())

    def test_explain_prediction_degrades_gracefully(
        self, embedding_config, fake_encoder
    ):
        vec = create_vectorizer(embedding_config)
        texts = ["great love", "awful delay"] * 10
        y = np.array([1, 0] * 10)
        model = train_model(vec.fit_transform(texts), y, embedding_config)

        result = explain_prediction(model, vec, "great flight")
        assert result["method"] == "none"
        assert result["top_contributors"] == []

    def test_train_save_load_predict(
        self, embedding_config, fake_encoder, tmp_path
    ):
        """Embeddings + LogisticRegression through the standard bundle."""
        texts = (
            ["great love thanks"] * 20 + ["awful delay rude"] * 20
        )
        y = np.array([1] * 20 + [0] * 20)

        vec = create_vectorizer(embedding_config)
        model = train_model(vec.fit_transform(texts), y, embedding_config)
        save_model(model, vec, tmp_path, filename="emb.joblib")

        loaded_model, loaded_vec, _, _ = load_model(tmp_path / "emb.joblib")
        preds, proba = predict_sentiment(
            loaded_model, loaded_vec,
            ["@united awesome, thanks!", "lost my bag, rude staff"],
            return_proba=True,
        )
        assert preds.tolist() == [1, 0]
        assert proba.shape == (2, 2)


# ============================================================================
# API service with an embedding bundle
# ============================================================================


class TestModelServiceWithEmbeddings:

    @pytest.fixture
    def embedding_service(self, embedding_config, fake_encoder):
        """ModelService loaded with a real (fake-encoder) embedding bundle."""
        pytest.importorskip("fastapi")
        from unittest.mock import patch

        from src.api.services import ModelService
        from src.constants import TARGET_MAPPING, TARGET_MAPPING_INV

        texts = ["great love thanks"] * 20 + ["awful delay rude"] * 20
        y = np.array([1] * 20 + [0] * 20)
        vec = create_vectorizer(embedding_config)
        model = train_model(vec.fit_transform(texts), y, embedding_config)

        # Simulate a freshly loaded bundle: encoder not loaded yet
        vec.__dict__.pop("_encoder", None)
        fake_encoder.loads["count"] = 0

        with patch("src.api.services.load_model") as mock_load, \
             patch("src.api.services.is_model_available") as mock_avail:
            mock_load.return_value = (
                model, vec, TARGET_MAPPING, TARGET_MAPPING_INV
            )
            mock_avail.return_value = True
            service = ModelService(model_path="unused.joblib")
            service.load()
            yield service

    def test_load_warms_up_encoder(self, embedding_service, fake_encoder):
        assert embedding_service.is_loaded
        assert fake_encoder.loads["count"] == 1

    def test_explain_request_returns_prediction_without_explanation(
        self, embedding_service
    ):
        """explain=true must not turn into a 500 for embedding models."""
        response = embedding_service.predict_single_sync(
            "@united great flight, thanks!", explain=True, n_explain=5
        )
        assert response.predicted_class == "positive"
        assert response.explanation is None


# ============================================================================
# Real model smoke test (optional dependencies)
# ============================================================================


@pytest.mark.integration
def test_real_sentence_transformer_smoke():
    """
    Encode with the real default model: shape, normalization, and a basic
    semantic sanity check. Downloads ~90 MB on first run.
    """
    pytest.importorskip("sentence_transformers")

    vec = SentenceEmbeddingVectorizer().fit(["probe"])
    X = vec.transform([
        "The flight was great, thank you!",
        "Wonderful crew, loved the flight",
        "My bag was lost and nobody helped",
    ])

    assert X.shape == (3, 384)
    np.testing.assert_allclose(np.linalg.norm(X, axis=1), 1.0, atol=1e-5)
    positive_pair = float(X[0] @ X[1])
    mixed_pair = float(X[0] @ X[2])
    assert positive_pair > mixed_pair
