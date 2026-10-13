"""
Model selection support for the dashboard: bundle discovery from artifacts/
(metadata only, no bundle loading) and one shared sentence-embedding encoder
for several loaded models.
"""
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import yaml

from src.model_registry import (
    BundleInfo,
    default_bundle_index,
    discover_bundles,
)


def _make_run(run_dir: Path, *, model=None, vectorizer=None, metrics=None):
    """Create a fake training run: bundle placeholder + metadata files."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "model_bundle.joblib").write_bytes(b"")
    if model or vectorizer:
        config = {
            "model": {"type": model},
            "preprocessing": {"vectorizer": {"type": vectorizer}},
        }
        (run_dir / "config_used.yaml").write_text(
            yaml.safe_dump(config), encoding="utf-8"
        )
    if metrics is not None:
        (run_dir / "metrics.json").write_text(
            json.dumps(metrics), encoding="utf-8"
        )
    return run_dir / "model_bundle.joblib"


@pytest.fixture
def artifacts(tmp_path: Path) -> Path:
    root = tmp_path / "artifacts"
    _make_run(
        root, model="logistic_regression", vectorizer="tfidf",
        metrics={"task_type": "multiclass", "accuracy": 0.84,
                 "f1_macro": 0.785},
    )
    _make_run(
        root / "embedding_mlp", model="mlp",
        vectorizer="sentence_embedding",
        metrics={"task_type": "multiclass", "f1_macro": 0.804},
    )
    _make_run(
        root / "binary", model="logistic_regression", vectorizer="tfidf",
        metrics={"task_type": "binary", "f1_macro": 0.888},
    )
    _make_run(root / "legacy")  # bundle without metadata files
    (root / "compare").mkdir()  # results folder, no bundle
    (root / "compare" / "summary.csv").write_text("x", encoding="utf-8")
    return root


class TestDiscoverBundles:

    def test_default_first_then_runs_alphabetically(self, artifacts):
        names = [b.name for b in discover_bundles(artifacts)]
        assert names == ["default", "binary", "embedding_mlp", "legacy"]

    def test_labels_describe_setup_task_and_score(self, artifacts):
        labels = {b.name: b.label for b in discover_bundles(artifacts)}
        assert labels["default"] == (
            "default: TF-IDF + LogReg · 3 classes · F1 0.785"
        )
        assert labels["binary"] == (
            "binary: TF-IDF + LogReg · binary · F1 0.888"
        )
        assert labels["embedding_mlp"] == (
            "embedding_mlp: Embeddings + MLP · 3 classes · F1 0.804"
        )

    def test_bundle_without_metadata_still_listed(self, artifacts):
        legacy = {b.name: b for b in discover_bundles(artifacts)}["legacy"]
        assert legacy.label == "legacy: details unknown"
        assert legacy.f1_macro is None
        assert not legacy.needs_deep_learning

    def test_corrupt_metadata_ignored(self, artifacts):
        run = artifacts / "binary"
        (run / "metrics.json").write_text("{not json", encoding="utf-8")
        (run / "config_used.yaml").write_text("- [", encoding="utf-8")
        binary = {b.name: b for b in discover_bundles(artifacts)}["binary"]
        assert binary.label == "binary: details unknown"

    def test_deep_learning_flag(self, artifacts):
        flags = {
            b.name: b.needs_deep_learning for b in discover_bundles(artifacts)
        }
        assert flags["embedding_mlp"] is True
        assert flags["default"] is False

    def test_missing_artifacts_dir(self, tmp_path):
        assert discover_bundles(tmp_path / "nope") == []

    def test_default_index_follows_model_path(self, artifacts):
        bundles = discover_bundles(artifacts)
        preferred = artifacts / "embedding_mlp" / "model_bundle.joblib"
        assert bundles[default_bundle_index(bundles, preferred)].name == (
            "embedding_mlp"
        )
        assert default_bundle_index(bundles, artifacts / "missing") == 0

    def test_label_without_metrics(self):
        info = BundleInfo(
            path=Path("x"), name="run", model_type="mlp",
            vectorizer_type="tfidf",
        )
        assert info.label == "run: TF-IDF + MLP"


# ============================================================================
# Several services, one encoder
# ============================================================================


def _embedding_bundle(fake_encoder):
    from src.constants import TARGET_MAPPING, TARGET_MAPPING_INV
    from src.models import train_model
    from src.preprocessing import create_vectorizer

    config = {
        "preprocessing": {
            "vectorizer": {"type": "sentence_embedding"},
            "embedding": {"model_name": "fake/model"},
        },
        "model": {
            "type": "logistic_regression",
            "training": {"max_iter": 200, "random_state": 0},
            "regularization": {"solver": "lbfgs", "penalty": "l2", "C": 1.0},
        },
    }
    texts = ["great love thanks"] * 10 + ["awful delay rude"] * 10
    vec = create_vectorizer(config)
    model = train_model(
        vec.fit_transform(texts), np.array([1] * 10 + [0] * 10), config
    )
    vec.__dict__.pop("_encoder", None)  # as if freshly unpickled
    return model, vec, TARGET_MAPPING, TARGET_MAPPING_INV


def _load(bundle, encoder_cache):
    from src.api.services import ModelService

    with patch("src.api.services.load_model", return_value=bundle), \
         patch("src.api.services.is_model_available", return_value=True):
        service = ModelService(
            model_path="unused.joblib", encoder_cache=encoder_cache
        )
        service.load()
    return service


class TestSharedEncoder:

    @pytest.fixture(autouse=True)
    def _needs_fastapi(self):
        pytest.importorskip("fastapi")

    def test_two_embedding_bundles_share_one_encoder(self, fake_encoder):
        first_bundle = _embedding_bundle(fake_encoder)
        second_bundle = _embedding_bundle(fake_encoder)
        fake_encoder.loads["count"] = 0
        cache = {}

        first = _load(first_bundle, cache)
        second = _load(second_bundle, cache)

        assert fake_encoder.loads["count"] == 1
        assert second._vectorizer._encoder is first._vectorizer._encoder
        assert second.predict_single_sync("great thanks").predicted_class == (
            "positive"
        )

    def test_without_cache_each_service_loads_its_own(self, fake_encoder):
        first_bundle = _embedding_bundle(fake_encoder)
        second_bundle = _embedding_bundle(fake_encoder)
        fake_encoder.loads["count"] = 0

        _load(first_bundle, None)
        _load(second_bundle, None)

        assert fake_encoder.loads["count"] == 2
