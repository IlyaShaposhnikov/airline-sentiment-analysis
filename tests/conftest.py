import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the project root directory."""
    return PROJECT_ROOT


# ============================================================================
# Preprocessing test cases (SYNCED: input + expected in one fixture)
# ============================================================================


@pytest.fixture(scope="session")
def tweet_test_cases() -> List[Tuple[str, str]]:
    """
    Paired test cases: (raw_tweet, expected_cleaned_tweet).

    Ensures input/expected outputs stay synchronized.

    Returns:
        List of tuples: (input_text, expected_cleaned_text)

    Note: Expected outputs assume default cleaning params:
    - remove_special_chars=True removes @, commas, etc.
    - remove_extra_whitespace=True collapses spaces
    """
    return [
        (
            "@VirginAmerica Great flight, excellent service!",
            "virginamerica great flight excellent service!",
        ),
        (
            "Terrible delay, never flying again 😠",
            "terrible delay never flying again",
        ),
        (
            "Meh, okay experience, nothing special",
            "meh okay experience nothing special",
        ),
        (
            "https://t.co/abc Check this out!!!",
            "check this out!!!",
        ),
        ("", ""),
        ("a" * 1500, "a" * 1500),
    ]


@pytest.fixture(scope="session")
def edge_case_tweets() -> List[str]:
    """
    Additional edge cases for robustness testing.

    These test boundary conditions that may not have specific expected outputs,
    but should not cause crashes or unexpected behavior.
    """
    return [
        "!!!",  # Only punctuation
        "12345",  # Only numbers
        "🎉🎊🎈",  # Only emojis
        "café naïve résumé",  # Unicode with accents
        "test\nwith\tnewlines",  # Whitespace characters
        "a" * 10_000,  # Very long text (stress test)
        "@user@user@user",  # Multiple mentions
        "http://example.com https://t.co/abc www.test.com",  # Multiple URLs
    ]


# ============================================================================
# Integration test fixtures
# ============================================================================


@pytest.fixture(scope="session")
def model_bundle_path(project_root: Path) -> Path:
    """Path to the trained model bundle."""
    return project_root / "artifacts" / "model_bundle.joblib"


@pytest.fixture(scope="session")
def trained_model(model_bundle_path: Path):
    """
    Load trained model for integration tests.

    Skips test if model bundle not found (useful for CI environments).
    """
    from src.models import load_model

    if not model_bundle_path.exists():
        pytest.skip(f"Model bundle not found: {model_bundle_path}")

    return load_model(model_bundle_path)


@pytest.fixture
def clean_text_func():
    """Return the clean_text function for direct testing."""
    from src.preprocessing import clean_text
    return clean_text


# ============================================================================
# API model test data fixtures (shared)
# ============================================================================


@pytest.fixture
def api_valid_prediction_request_data() -> dict:
    """
    Minimal valid data for PredictionRequest.
    """
    return {"text": "Great flight!"}


@pytest.fixture
def api_valid_batch_request_data() -> dict:
    """
    Minimal valid data for BatchPredictionRequest.
    """
    return {"texts": ["Great flight!", "Terrible service"]}


@pytest.fixture
def api_valid_prediction_response_data() -> dict:
    """
    Minimal valid data for PredictionResponse (without auto-generated fields).
    """
    return {
        "text": "Great flight!",
        "predicted_class": "positive",
        "predicted_class_idx": 1,
        "probabilities": {"negative": 0.1, "positive": 0.8, "neutral": 0.1},
        "confidence": 0.8,
    }


# ============================================================================
# Sentence-embedding fixtures (no torch / no model download)
# ============================================================================


class FakeSentenceEncoder:
    """
    Deterministic stand-in for SentenceTransformer.

    Encodes sentiment-bearing words into fixed dimensions, so a linear
    classifier can learn from it. Records every call for assertions.
    """

    POSITIVE = {"great", "love", "thanks", "awesome"}
    NEGATIVE = {"delay", "lost", "awful", "rude"}

    dim = 8

    def __init__(self):
        self.calls: list[dict] = []
        self.loads = {"count": 0}

    def encode(
        self,
        texts,
        batch_size=32,
        normalize_embeddings=False,
        show_progress_bar=False,
        convert_to_numpy=True,
    ):
        self.calls.append({
            "texts": list(texts),
            "batch_size": batch_size,
            "normalize_embeddings": normalize_embeddings,
        })
        out = np.zeros((len(texts), self.dim), dtype=np.float64)
        for i, text in enumerate(texts):
            words = text.lower().replace("!", " ").split()
            out[i, 0] = sum(w in self.POSITIVE for w in words)
            out[i, 1] = sum(w in self.NEGATIVE for w in words)
            out[i, 2] = len(words)
            out[i, 3] = 1.0  # bias-like constant, keeps rows non-zero
            out[i, 4 + hash(text) % 4] += 0.01
        if normalize_embeddings:
            out /= np.linalg.norm(out, axis=1, keepdims=True)
        return out


@pytest.fixture
def fake_encoder(monkeypatch) -> FakeSentenceEncoder:
    """
    Patch the Sentence-Transformers loader with a fake encoder:
    no torch, no model download. Counts loads in ``encoder.loads``.
    """
    import src.embeddings as embeddings_module

    encoder = FakeSentenceEncoder()

    def fake_loader(model_name, device):
        encoder.loads["count"] += 1
        return encoder

    monkeypatch.setattr(
        embeddings_module, "_load_sentence_transformer", fake_loader
    )
    return encoder
