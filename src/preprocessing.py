import logging
import re
from typing import Union

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from .utils.logging_config import setup_logger

logger = setup_logger(__name__)

_NLTK_INITIALIZED = False


def clean_text(
    text: str,
    lowercase: bool = True,
    remove_urls: bool = True,
    remove_mentions: bool = False,  # Keep @airline mentions
    remove_special_chars: bool = True,
    remove_extra_whitespace: bool = True,
) -> str:
    """Clean raw tweet text for NLP processing."""
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    if lowercase:
        text = text.lower()

    # Remove URLs (http/https)
    if remove_urls:
        text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # Remove mentions (@username)
    if remove_mentions:
        text = re.sub(r"@\w+", "", text)

    # Remove special characters and punctuation
    if remove_special_chars:
        # Keep letters, digits, spaces, !, ?, . for sentiment context
        text = re.sub(r"[^a-z0-9\s!?.]", "", text)

    # Remove extra whitespace
    if remove_extra_whitespace:
        text = re.sub(r"\s+", " ", text).strip()

    return text


def _ensure_nltk_resources():
    """Download required NLTK resources if not already present."""
    global _NLTK_INITIALIZED

    if _NLTK_INITIALIZED:
        return

    import nltk
    resources = ['punkt', 'wordnet', 'omw-1.4']
    for name in resources:
        try:
            nltk.download(name, quiet=True, raise_on_error=True)
            logger.debug(f"NLTK resource ready: {name}")
        except Exception as e:
            logger.warning(f"Could not ensure NLTK '{name}': {e}")

    _NLTK_INITIALIZED = True


def lemmatize_text(
        text: str,
        remove_stopwords: bool = False
) -> str:
    """Lemmatize tokenized text using NLTK WordNetLemmatizer."""
    # Lazy import to avoid overhead if not used
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    from nltk import pos_tag

    _ensure_nltk_resources()
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)

    # Optional: remove stopwords
    if remove_stopwords:
        stop_words = set(stopwords.words("english"))
        tokens = [t for t in tokens if t not in stop_words]

    # Map POS tags to WordNet format
    def get_wordnet_pos(treebank_tag: str) -> str:
        if treebank_tag.startswith('J'):
            return 'a'  # adjective
        elif treebank_tag.startswith('V'):
            return 'v'  # verb
        elif treebank_tag.startswith('N'):
            return 'n'  # noun
        elif treebank_tag.startswith('R'):
            return 'r'  # adverb
        else:
            return 'n'  # default to noun

    lemmatized = [
        lemmatizer.lemmatize(t, get_wordnet_pos(pos))
        for t, pos in pos_tag(tokens)
        if t and not t.isspace()
    ]
    return " ".join(lemmatized)


def preprocess_text(
    text: str,
    config: dict,
) -> str:
    """Apply full preprocessing pipeline to a single text sample."""
    cleaning_cfg = config.get("preprocessing", {}).get("cleaning", {})
    nlp_cfg = config.get("preprocessing", {}).get("nlp", {})
    # Step 1: Basic cleaning
    cleaned = clean_text(
        text,
        lowercase=cleaning_cfg.get("lowercase", True),
        remove_urls=cleaning_cfg.get("remove_urls", True),
        remove_mentions=cleaning_cfg.get("remove_mentions", False),
        remove_special_chars=cleaning_cfg.get("remove_special_chars", True),
        remove_extra_whitespace=cleaning_cfg.get(
            "remove_extra_whitespace", True
        ),
    )

    # Step 2: Optional lemmatization
    if nlp_cfg.get("lemmatize", False):
        cleaned = lemmatize_text(
            cleaned,
            remove_stopwords=nlp_cfg.get("remove_stopwords", False)
        )

    return cleaned


def create_vectorizer(config: dict) -> Union[TfidfVectorizer, CountVectorizer]:
    """Initialize and return a vectorizer based on configuration."""
    vectorizer_cfg = config.get("preprocessing", {}).get("vectorizer", {})
    nlp_cfg = config.get("preprocessing", {}).get("nlp", {})
    required_keys = ["type", "max_features", "ngram_range"]
    missing = [k for k in required_keys if k not in vectorizer_cfg]
    if missing:
        raise ValueError(f"Missing required vectorizer config keys: {missing}")

    vectorizer_type = vectorizer_cfg.get("type", "tfidf").lower()

    # Common parameters for both vectorizers
    common_params = {
        "max_features": vectorizer_cfg.get("max_features", 2000),
        "ngram_range": tuple(vectorizer_cfg.get("ngram_range", [1, 2])),
        "lowercase": vectorizer_cfg.get("lowercase", True),
        "stop_words": (
            "english"
            if nlp_cfg.get("remove_stopwords", False)
            else None
        ),
    }

    if vectorizer_type == "tfidf":
        logger.info(
            f"Initializing TfidfVectorizer with params: {common_params}"
        )
        return TfidfVectorizer(
            **common_params,
            sublinear_tf=True,  # Apply sublinear tf scaling: 1 + log(tf)
            dtype=np.float64,
        )
    elif vectorizer_type == "count":
        logger.info(
            f"Initializing CountVectorizer with params: {common_params}"
        )
        return CountVectorizer(**common_params, dtype=np.int64)
    else:
        raise ValueError(f"Unknown vectorizer type: {vectorizer_type}")


def preprocess_texts(
    texts: list[str],
    config: dict,
) -> list[str]:
    """Apply text cleaning and optional lemmatization to a list of texts."""
    logger.debug(f"Preprocessing {len(texts)} texts...")

    # Preprocess each text individually
    processed = [preprocess_text(text, config) for text in texts]

    # Warn about empty results
    empty_count = sum(1 for t in processed if not t.strip())
    if empty_count > 0:
        logger.warning(
            f"{empty_count}/{len(processed)} texts resulted in empty strings "
            "after preprocessing. Consider adjusting cleaning parameters."
        )

    # Log sample for debugging
    if logger.isEnabledFor(logging.DEBUG) and processed:
        logger.debug(f"Sample preprocessed text: '{processed[0][:100]}...'")

    return processed
