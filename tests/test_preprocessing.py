"""
Unit tests for src/preprocessing.py text cleaning functions.

Focus: Edge cases, boundary conditions, and sentiment-preserving behavior.
"""

import logging

import pytest
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from src.preprocessing import (
    preprocess_text,
    preprocess_texts,
    create_vectorizer,
    _ensure_nltk_resources,
)


class TestCleanText:
    """Tests for the clean_text() function."""

    def test_lowercase_conversion(self, clean_text_func):
        """Text should be lowercased when lowercase=True (default)."""
        assert clean_text_func("GREAT Flight!") == "great flight!"

    def test_url_removal_http_https(self, clean_text_func):
        """HTTP and HTTPS URLs should be removed."""
        assert "http" not in clean_text_func("Check https://t.co/abc")
        assert "www" not in clean_text_func("Visit www.example.com now")

    def test_url_removal_edge_cases(self, clean_text_func):
        """URLs with various formats should be handled."""
        result1 = clean_text_func("Go to http://test.com/path?x=1")
        assert "go" in result1 and "to" in result1
        assert "http" not in result1 and "test.com" not in result1

        result2 = clean_text_func("Link: www.site.org")
        assert "link" in result2
        assert "www" not in result2 and "site.org" not in result2

    def test_mentions_preserved_by_default(self, clean_text_func):
        """
        @mentions should be kept when remove_mentions=False
        AND remove_special_chars=False.
        """
        result = clean_text_func(
            "@VirginAmerica thanks!",
            remove_mentions=False,
            remove_special_chars=False  # ← Preserve @ symbol
        )
        assert "@virginamerica" in result

    def test_mentions_removed_when_requested(self, clean_text_func):
        """@mentions should be removed when remove_mentions=True."""
        result = clean_text_func("@User hello @Another", remove_mentions=True)
        assert "@" not in result

    def test_sentiment_punctuation_preserved(self, clean_text_func):
        """Exclamation/question marks should be kept for sentiment context."""
        result = clean_text_func(
            "Great!!! Really? Yes.", remove_special_chars=True
        )
        assert "!!!" in result
        assert "?" in result
        assert "." in result

    def test_other_special_chars_removed(self, clean_text_func):
        """Non-sentiment punctuation should be removed."""
        result = clean_text_func(
            "Price: $100 #deal", remove_special_chars=True
        )
        assert "$" not in result
        assert "#" not in result

    def test_whitespace_normalization(self, clean_text_func):
        """Multiple spaces/tabs/newlines should collapse to single space."""
        assert clean_text_func(
            "  multiple   spaces\t\nhere  "
        ) == "multiple spaces here"

    def test_empty_string_handling(self, clean_text_func):
        """Empty or whitespace-only input should return empty string."""
        assert clean_text_func("") == ""
        assert clean_text_func("   \t\n  ") == ""

    def test_non_string_input_handling(self, clean_text_func):
        """Non-string input should return empty string gracefully."""
        assert clean_text_func(None) == ""
        assert clean_text_func(123) == ""
        assert clean_text_func(["list"]) == ""

    def test_emoji_removal(self, clean_text_func):
        """Emoji characters should be removed as special chars."""
        result = clean_text_func("Love it! 😍✈️🎉", remove_special_chars=True)

        for emoji in ["😍", "✈️", "🎉"]:
            assert emoji not in result, f"Emoji {emoji} not removed"

        assert "love" in result
        assert "it" in result

    def test_long_text_not_truncated(self, clean_text_func):
        """clean_text() should not truncate long texts."""
        long_text = "word " * 500
        result = clean_text_func(long_text)
        assert len(result) > 1000

    def test_combined_operations(self, clean_text_func):
        """All cleaning steps should work together correctly."""
        text = "  CHECK OUT https://t.co/abc @User GREAT!!! 😊  "
        result = clean_text_func(
            text,
            lowercase=True,
            remove_urls=True,
            remove_mentions=True,
            remove_special_chars=True,
            remove_extra_whitespace=True,
        )
        assert "check" in result
        assert "out" in result
        assert "great!!!" in result
        assert "http" not in result
        assert "@" not in result
        assert "😊" not in result
        assert result == result.strip()


class TestPreprocessText:
    """Tests for the preprocess_text() pipeline function."""

    def test_preprocess_with_default_config(self, clean_text_func):
        """Default config should apply standard cleaning."""
        config = {
            "preprocessing": {
                "cleaning": {
                    "remove_special_chars": False,
                },
                "nlp": {"lemmatize": False},
            }
        }
        result = preprocess_text("  GREAT https://t.co/abc!!!  ", config)

        assert "great" in result
        assert "http" not in result
        assert "!!!" in result

    def test_preprocess_with_custom_cleaning_config(self, clean_text_func):
        """Custom cleaning params should override defaults."""
        config = {
            "preprocessing": {
                "cleaning": {
                    "lowercase": False,
                    "remove_urls": False,
                    "remove_mentions": True,
                    "remove_special_chars": False,
                },
                "nlp": {"lemmatize": False},
            }
        }
        result = preprocess_text("@User GREAT https://t.co/abc", config)

        # assert "@" not in result or "@user" not in result.lower()
        if config["preprocessing"]["cleaning"].get("remove_mentions", False):
            assert "@" not in result, (
                "@ should be removed when remove_mentions=True"
            )
        assert "GREAT" in result
        assert "https" in result or "http" in result

    def test_preprocess_empty_input(self, clean_text_func):
        """Empty input should return empty string through pipeline."""
        config = {
            "preprocessing": {"cleaning": {}, "nlp": {"lemmatize": False}}
        }
        assert preprocess_text("", config) == ""
        assert preprocess_text("   ", config) == ""


class TestPreprocessTexts:
    """Tests for the preprocess_texts() batch function."""

    def test_preprocess_texts_batch(self, tweet_test_cases):
        """Batch preprocessing should apply cleaning to each text."""
        input_texts = [raw for raw, _ in tweet_test_cases]
        expected_outputs = [expected for _, expected in tweet_test_cases]
        config = {
            "preprocessing": {
                "cleaning": {"remove_mentions": False},
                "nlp": {"lemmatize": False},
            }
        }
        result = preprocess_texts(input_texts, config)

        assert len(result) == len(input_texts)
        assert all(isinstance(t, str) for t in result)

        for i, (inp, expected) in enumerate(
            zip(input_texts, expected_outputs)
        ):
            assert result[i] == expected, (
                f"Mismatch at index {i}: input={inp!r}, "
                f"expected={expected!r}, actual={result[i]!r}"
            )

    def test_preprocess_texts_warns_on_empty(self, caplog):
        """Should log warning when cleaning produces empty strings."""
        config = {
            "preprocessing": {
                "cleaning": {
                    "remove_special_chars": True, "remove_urls": True
                },
                "nlp": {"lemmatize": False},
            }
        }
        texts = ["@#$%^&*()", "https://t.co/abc", "normal text"]
        with caplog.at_level(logging.WARNING, logger="src.preprocessing"):
            _ = preprocess_texts(texts, config)

        # Check that warning was logged with expected keywords
        assert caplog.records, "Expected at least one warning log record"
        warning_messages = [r.message.lower() for r in caplog.records]

        assert any("empty" in msg for msg in warning_messages), (
            f"Expected 'empty' in warning, got: {warning_messages}"
        )
        assert any("adjust" in msg for msg in warning_messages), (
            f"Expected 'adjust' in warning, got: {warning_messages}"
        )


class TestCreateVectorizer:
    """Tests for the create_vectorizer() factory function."""

    def test_create_tfidf_vectorizer(self):
        """Should create TfidfVectorizer with correct params."""
        config = {
            "preprocessing": {
                "vectorizer": {
                    "type": "tfidf",
                    "max_features": 500,
                    "ngram_range": [1, 2],
                },
                "nlp": {"remove_stopwords": False},
            }
        }
        vec = create_vectorizer(config)
        assert isinstance(vec, TfidfVectorizer)
        assert vec.max_features == 500
        assert vec.ngram_range == (1, 2)

    def test_create_count_vectorizer(self):
        """Should create CountVectorizer when type='count'."""
        config = {
            "preprocessing": {
                "vectorizer": {
                    "type": "count",
                    "max_features": 100,
                    "ngram_range": [1, 1],
                },
                "nlp": {"remove_stopwords": False},
            }
        }
        vec = create_vectorizer(config)
        assert isinstance(vec, CountVectorizer)

    def test_create_vectorizer_missing_keys(self):
        """Should raise ValueError if required keys are missing."""
        config = {
            "preprocessing": {
                "vectorizer": {
                    "type": "tfidf",
                },
                "nlp": {},
            }
        }
        with pytest.raises(
            ValueError, match="Missing required vectorizer config keys"
        ):
            create_vectorizer(config)

    def test_create_vectorizer_unknown_type(self):
        """Should raise ValueError for unknown vectorizer type."""
        config = {
            "preprocessing": {
                "vectorizer": {
                    "type": "unknown_type",
                    "max_features": 100,
                    "ngram_range": [1, 2],
                },
                "nlp": {},
            }
        }
        with pytest.raises(ValueError, match="Unknown vectorizer type"):
            create_vectorizer(config)


class TestNLTKInitialization:
    """Tests for NLTK resource handling."""

    def test_ensure_nltk_resources_idempotent(self):
        """_ensure_nltk_resources() should be safe to call multiple times."""
        _ensure_nltk_resources()
        _ensure_nltk_resources()
        assert True

    def test_ensure_nltk_resources_does_not_crash(self):
        """Should handle download failures gracefully (warning, not crash)."""
        try:
            _ensure_nltk_resources()
            assert True
        except Exception as e:
            pytest.fail(
                f"_ensure_nltk_resources() raised unexpected error: {e}"
            )


class TestLemmatizeText:
    """Tests for the lemmatize_text() function."""
    def test_lemmatize_basic(self):
        """Basic lemmatization should work."""
        from src.preprocessing import lemmatize_text

        result = lemmatize_text("cats dogs running", remove_stopwords=False)
        assert "cat" in result  # cats → cat
        assert "dog" in result  # dogs → dog

    def test_lemmatize_with_stopwords_removal(self):
        """Lemmatization with stopword removal."""
        from src.preprocessing import lemmatize_text

        result = lemmatize_text("the cats are running", remove_stopwords=True)
        # Stopwords removed: the, are
        assert "the" not in result
        assert "are" not in result
        assert "cat" in result


class TestVectorizerEmbeddedPreprocessing:
    """
    The vectorizer must apply the training-time cleaning pipeline itself,
    so raw texts at inference are processed exactly like training texts
    (regression tests for train/serve skew).
    """

    @staticmethod
    def _config(**cleaning) -> dict:
        base_cleaning = {
            "lowercase": True,
            "remove_urls": True,
            "remove_mentions": True,
            "remove_special_chars": True,
            "remove_extra_whitespace": True,
        }
        base_cleaning.update(cleaning)
        return {
            "preprocessing": {
                "vectorizer": {
                    "type": "tfidf",
                    "max_features": 100,
                    "ngram_range": [1, 1],
                },
                "cleaning": base_cleaning,
                "nlp": {"lemmatize": False, "remove_stopwords": False},
            }
        }

    @pytest.mark.parametrize("vec_type", ["tfidf", "count"])
    def test_raw_text_is_cleaned_by_vectorizer(self, vec_type):
        """Mentions and URLs removed in training must not leak at inference."""
        config = self._config()
        config["preprocessing"]["vectorizer"]["type"] = vec_type
        vec = create_vectorizer(config)

        analyzer = vec.build_analyzer()
        tokens = analyzer("@united GREAT flight https://t.co/abc123")

        assert tokens == ["great", "flight"]

    def test_transform_raw_equals_transform_cleaned(self):
        """transform(raw) must equal transform(preprocess(raw))."""
        config = self._config()
        vec = create_vectorizer(config)
        vec.fit(["great flight", "awful delay", "united airlines gate"])

        raw = "@United Great flight!!! https://t.co/xyz"
        cleaned = preprocess_text(raw, config)

        diff = vec.transform([raw]) - vec.transform([cleaned])
        assert diff.nnz == 0

    def test_mention_kept_when_cleaning_disabled(self):
        """Embedded hook must follow config, not hardcode cleaning."""
        vec = create_vectorizer(
            self._config(remove_mentions=False, remove_special_chars=False)
        )
        assert "united" in vec.build_analyzer()("@united thanks")

    def test_vectorizer_lowercase_honoured_with_preprocessor(self):
        """
        sklearn skips its own lowercasing when preprocessor is set,
        so the hook must apply vectorizer.lowercase itself.
        """
        vec = create_vectorizer(self._config(lowercase=False))
        assert vec.build_analyzer()("GREAT Flight") == ["great", "flight"]

    def test_preprocessor_survives_joblib_roundtrip(self, tmp_path):
        """Cleaning must travel inside the model bundle."""
        import joblib

        vec = create_vectorizer(self._config())
        vec.fit(["great flight", "awful delay"])
        path = tmp_path / "vec.joblib"
        joblib.dump(vec, path)
        loaded = joblib.load(path)

        assert loaded.preprocessor is not None
        assert loaded.build_analyzer()("@united great https://t.co/a") == [
            "great"
        ]

    def test_config_mutation_does_not_affect_vectorizer(self):
        """Captured cleaning config is a snapshot, not a live reference."""
        config = self._config(remove_mentions=True)
        vec = create_vectorizer(config)
        config["preprocessing"]["cleaning"]["remove_mentions"] = False

        assert "united" not in vec.build_analyzer()("@united great")
