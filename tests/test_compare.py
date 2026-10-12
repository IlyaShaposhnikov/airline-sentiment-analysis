"""
Tests for scripts/compare.py (fake sentence encoder: no torch needed for
the logistic-regression configurations; MLP ones run when torch exists).
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml


def _load_compare(project_root: Path):
    spec = importlib.util.spec_from_file_location(
        "compare_script", project_root / "scripts" / "compare.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def dataset_config(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    vocab = {
        "positive": ["great", "love", "thanks", "awesome"],
        "negative": ["delay", "lost", "awful", "rude"],
        "neutral": ["flight", "when", "gate", "schedule"],
    }
    labels = list(vocab)
    rows = []
    for i in range(240):
        label = labels[i % 3]
        rows.append({
            "airline_sentiment": label,
            "text": "@united " + " ".join(rng.choice(vocab[label], 3)),
            "airline_sentiment_confidence": rng.uniform(0.75, 1.0),
            "negativereason_confidence": 0.5,
        })
    csv_path = tmp_path / "tweets.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    config = {
        "data": {
            "path": str(csv_path), "target_column": "airline_sentiment",
            "text_column": "text",
            "confidence_columns": {
                "sentiment": "airline_sentiment_confidence",
                "reason": "negativereason_confidence",
            },
            "confidence_threshold": 0.7,
        },
        "preprocessing": {
            "vectorizer": {
                "type": "tfidf", "max_features": 100, "ngram_range": [1, 1]
            },
            "embedding": {"model_name": "fake/model"},
        },
        "model": {
            "type": "logistic_regression",
            "training": {
                "max_iter": 200, "class_weight": "balanced",
                "random_state": 42, "use_confidence_weights": True,
            },
            "regularization": {"solver": "lbfgs", "penalty": "l2", "C": 1.0},
            "mlp": {"hidden_dims": [8], "max_epochs": 5},
        },
        "evaluation": {"split": {"test_size": 0.25, "stratify": True}},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


@pytest.mark.integration
class TestCompareScript:

    def _run(self, project_root, dataset_config, tmp_path, *extra):
        compare = _load_compare(project_root)
        args = compare.parse_args([
            "--config", str(dataset_config),
            "--output-dir", str(tmp_path / "out"),
            "--cache-dir", str(tmp_path / "cache"),
            "--seeds", "1", "2",
            "--latency-samples", "10",
            *extra,
        ])
        return compare.main(args), tmp_path / "out"

    def test_runs_lr_grid_and_writes_outputs(
        self, project_root, dataset_config, tmp_path, fake_encoder
    ):
        code, out = self._run(
            project_root, dataset_config, tmp_path,
            "--configs", "tfidf_lr", "embedding_lr",
        )
        assert code == 0
        results = pd.read_csv(out / "results.csv")
        assert set(results["config"]) == {"tfidf_lr", "embedding_lr"}
        assert len(results) == 4  # 2 configs x 2 seeds
        assert results["accuracy"].between(0, 1).all()

        summary = pd.read_csv(out / "summary.csv")
        assert set(summary["n_seeds"]) == {2}
        md = (out / "summary.md").read_text(encoding="utf-8")
        assert "TF-IDF + LogReg" in md and "Embeddings + LogReg" in md

    def test_embeddings_encoded_once_and_cached(
        self, project_root, dataset_config, tmp_path, fake_encoder
    ):
        self._run(project_root, dataset_config, tmp_path,
                  "--configs", "embedding_lr")
        full_encodes = [c for c in fake_encoder.calls if len(c["texts"]) > 50]
        assert len(full_encodes) == 1
        assert list((tmp_path / "cache").glob("emb_*.npy"))

        self._run(project_root, dataset_config, tmp_path,
                  "--configs", "embedding_lr")
        full_encodes = [c for c in fake_encoder.calls if len(c["texts"]) > 50]
        assert len(full_encodes) == 1  # second run used the cache

    def test_binary_mode(
        self, project_root, dataset_config, tmp_path, fake_encoder
    ):
        code, out = self._run(
            project_root, dataset_config, tmp_path,
            "--configs", "tfidf_lr", "embedding_lr", "--binary-mode",
        )
        assert code == 0
        assert "binary" in (out / "summary.md").read_text(encoding="utf-8")

    @pytest.mark.skipif(
        importlib.util.find_spec("torch") is None,
        reason="PyTorch not installed",
    )
    def test_mlp_configs(
        self, project_root, dataset_config, tmp_path, fake_encoder
    ):
        code, out = self._run(
            project_root, dataset_config, tmp_path,
            "--configs", "tfidf_mlp", "embedding_mlp",
        )
        assert code == 0
        assert len(pd.read_csv(out / "results.csv")) == 4
