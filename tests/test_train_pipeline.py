"""
End-to-end regression tests for scripts/train.py.

Runs the real training CLI on a small noisy synthetic dataset and checks
artifact integrity:
- misclassified examples report the text that actually belongs to each row
- the vectorizer vocabulary is fitted on training rows only (no leakage)
- the saved bundle cleans raw input itself (no train/serve skew)

Does not import the API layer, so it runs without FastAPI installed.
"""

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml


def _load_train_module(project_root: Path):
    """Import scripts/train.py as a module (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location(
        "train_script", project_root / "scripts" / "train.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def noisy_dataset(tmp_path: Path) -> tuple[Path, pd.DataFrame]:
    """
    Separable vocabulary + 20% label noise, so the model makes some
    confident mistakes. Every tweet carries a unique ``idN`` token, which
    lets the test detect test-set tokens leaking into the vocabulary.
    """
    rng = np.random.default_rng(0)
    vocab = {
        "positive": ["great", "love", "thanks", "awesome"],
        "negative": ["delay", "lost", "awful", "rude"],
        # "united" also appears as a plain word (as in real tweets:
        # "United lost my bag"), so it is in the vocabulary — otherwise
        # an un-stripped "@united" at inference would be silently ignored
        # and the skew test below could not detect anything
        "neutral": ["flight", "when", "gate", "united"],
    }
    labels = list(vocab)
    rows = []
    for i in range(300):
        label = labels[i % 3]
        words = " ".join(rng.choice(vocab[label], 3))
        noisy = rng.random() < 0.2
        rows.append({
            "airline_sentiment": rng.choice(labels) if noisy else label,
            "text": f"@united {words} id{i} https://t.co/x{i}",
            "airline_sentiment_confidence": rng.uniform(0.75, 1.0),
            "negativereason_confidence": 0.5,
        })
    df = pd.DataFrame(rows)
    csv_path = tmp_path / "tweets.csv"
    df.to_csv(csv_path, index=False)

    config = {
        "data": {
            "path": str(csv_path),
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
                "type": "tfidf", "max_features": 5000, "ngram_range": [1, 1]
            },
            "cleaning": {
                "lowercase": True, "remove_urls": True,
                "remove_mentions": True, "remove_special_chars": True,
                "remove_extra_whitespace": True,
            },
            "nlp": {"lemmatize": False, "remove_stopwords": False},
        },
        "model": {
            "type": "logistic_regression",
            "training": {
                "max_iter": 300, "class_weight": "balanced",
                "random_state": 42, "use_confidence_weights": True,
            },
            "regularization": {"solver": "lbfgs", "penalty": "l2", "C": 1.0},
        },
        "evaluation": {
            "split": {"test_size": 0.3, "stratify": True},
            "metrics": {"primary": ["accuracy", "f1_macro"]},
            "reporting": {
                "export_formats": ["json"],
                "misclassified_examples": {"n_top": 50, "include_text": True},
            },
        },
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)
    return config_path, df


def _run_train(project_root: Path, config_path: Path, output_dir: Path,
               **overrides) -> int:
    """Call scripts/train.py main() with default CLI args + overrides."""
    train = _load_train_module(project_root)
    args = dict(
        config=str(config_path), output_dir=str(output_dir),
        binary_mode=False, no_plots=True, explain=False,
        n_explain=5, use_shap=False, seed=None, vectorizer=None,
    )
    args.update(overrides)
    return train.main(argparse.Namespace(**args))


@pytest.fixture
def trained_artifacts(project_root, noisy_dataset, tmp_path):
    """Run scripts/train.py main() once and return (output_dir, raw_df)."""
    config_path, df = noisy_dataset
    output_dir = tmp_path / "artifacts"
    assert _run_train(project_root, config_path, output_dir) == 0
    return output_dir, df


@pytest.mark.integration
class TestTrainScriptArtifacts:

    def test_misclassified_texts_match_their_true_labels(
        self, trained_artifacts
    ):
        """
        Regression: test texts used to be taken from the tail of the
        DataFrame, so reported texts did not belong to reported labels.
        """
        output_dir, df = trained_artifacts
        report = pd.read_csv(output_dir / "misclassified_examples.csv")
        assert not report.empty, "Noisy data should produce some errors"

        label_of_text = df.set_index("text")["airline_sentiment"]
        actual = report["text"].map(label_of_text)
        assert (report["true_label_name"] == actual).all()

    def test_vocabulary_fitted_on_train_rows_only(self, trained_artifacts):
        """Regression: vectorizer used to be fitted before the split."""
        from src.models import load_model

        output_dir, df = trained_artifacts
        _, vectorizer, _, _ = load_model(output_dir / "model_bundle.joblib")
        id_tokens = {
            t for t in vectorizer.vocabulary_ if t.startswith("id")
        }

        # 300 rows, test_size=0.3 → exactly 210 training ids may be present
        assert len(id_tokens) == 210

    def test_saved_bundle_cleans_raw_input(self, trained_artifacts):
        """Regression: inference used to skip training-time cleaning."""
        from src.models import load_model, predict_sentiment

        output_dir, _ = trained_artifacts
        model, vectorizer, _, _ = load_model(
            output_dir / "model_bundle.joblib"
        )
        raw = "@united great love https://t.co/zzz"
        clean = "great love"

        _, proba_raw = predict_sentiment(
            model, vectorizer, raw, return_proba=True
        )
        _, proba_clean = predict_sentiment(
            model, vectorizer, clean, return_proba=True
        )
        np.testing.assert_allclose(proba_raw, proba_clean)


@pytest.mark.integration
class TestTrainScriptWithSentenceEmbeddings:
    """
    --vectorizer sentence_embedding (fake encoder: no torch, no download).
    Plots and --explain are enabled on purpose: word-level artifacts must be
    skipped gracefully instead of crashing on a vocabulary-less vectorizer.
    """

    @pytest.fixture
    def embedding_artifacts(
        self, project_root, noisy_dataset, tmp_path, fake_encoder
    ):
        config_path, df = noisy_dataset
        output_dir = tmp_path / "emb_artifacts"
        exit_code = _run_train(
            project_root, config_path, output_dir,
            vectorizer="sentence_embedding", no_plots=False, explain=True,
        )
        return exit_code, output_dir

    def test_training_succeeds(self, embedding_artifacts):
        exit_code, output_dir = embedding_artifacts
        assert exit_code == 0
        assert (output_dir / "model_bundle.joblib").exists()
        assert (output_dir / "metrics.json").exists()

    def test_word_level_artifacts_skipped(self, embedding_artifacts):
        _, output_dir = embedding_artifacts
        assert (output_dir / "confusion_matrix.png").exists()
        assert not list(output_dir.glob("feature_importance_*.png"))
        assert not (output_dir / "explanations.json").exists()

    def test_bundle_contains_embedding_vectorizer(self, embedding_artifacts):
        from src.embeddings import SentenceEmbeddingVectorizer
        from src.models import load_model

        _, output_dir = embedding_artifacts
        _, vectorizer, _, _ = load_model(output_dir / "model_bundle.joblib")
        assert isinstance(vectorizer, SentenceEmbeddingVectorizer)

    def test_override_recorded_in_config_backup(self, embedding_artifacts):
        _, output_dir = embedding_artifacts
        with open(output_dir / "config_used.yaml", encoding="utf-8") as f:
            used = yaml.safe_load(f)
        assert used["preprocessing"]["vectorizer"]["type"] == (
            "sentence_embedding"
        )
