"""
Text preprocessing module for NLP pipeline.

Provides configurable text cleaning, NLTK-based lemmatization,
and vectorizer factory functions. All functions are idempotent
and safe for batch processing.
"""
import logging
import re

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from .utils.logging_config import setup_logger

logger = setup_logger(__name__)

# Module-level flag for idempotent NLTK resource loading
_NLTK_INITIALIZED = False


def clean_text(
    text: str,
    lowercase: bool = True,
    remove_urls: bool = True,
    remove_mentions: bool = False,
    remove_special_chars: bool = True,
    remove_extra_whitespace: bool = True,
) -> str:
    """
    Clean raw tweet text for NLP processing with configurable steps.

    Args:
        text: Raw input text
        lowercase: Convert to lowercase before processing
        remove_urls: Strip http/https/www links
        remove_mentions: Strip @username mentions
        remove_special_chars: Keep only letters, digits, spaces, and !?.
        remove_extra_whitespace: Collapse multiple spaces/tabs/newlines

    Returns:
        Cleaned text string (empty string if input is non-string)

    Note:
        Order of operations matters: lowercase → URLs → mentions →
        special chars → whitespace
    """
    # Explicit guard for non-string inputs
    if not isinstance(text, str):
        return ""

    if lowercase:
        text = text.lower()

    if remove_urls:
        # [^\s!] stops before punctuation to preserve sentiment markers
        text = re.sub(r"https?://[^\s!]+|www\.[^\s!]+", "", text)

    if remove_mentions:
        text = re.sub(r"@\w+", "", text)

    if remove_special_chars:
        if lowercase:
            # Text already lowercased: use a-z only for efficiency
            text = re.sub(r"[^a-z0-9\s!?.]", "", text)
        else:
            # Preserve uppercase: allow A-Z in character class
            text = re.sub(r"[^a-zA-Z0-9\s!?.]", "", text)

    if remove_extra_whitespace:
        text = re.sub(r"\s+", " ", text).strip()

    return text


def _ensure_nltk_resources() -> None:
    """
    Download required NLTK resources idempotently.

    Uses module-level flag to avoid redundant downloads across multiple calls.
    Lazy-imports nltk to avoid overhead when lemmatization is disabled.

    Raises:
        None: Failures are logged as warnings, not raised
        (graceful degradation)
    """
    global _NLTK_INITIALIZED

    if _NLTK_INITIALIZED:
        return

    # Lazy import avoids NLTK dependency when not needed
    import nltk
    resources = ['punkt', 'wordnet', 'omw-1.4']
    for name in resources:
        try:
            # raise_on_error=True ensures we catch download failures early
            nltk.download(name, quiet=True, raise_on_error=True)
            logger.debug(f"NLTK resource ready: {name}")
        except Exception as e:
            # Graceful degradation — lemmatization will fail later
            # if resources missing
            logger.warning(f"Could not ensure NLTK '{name}': {e}")

    _NLTK_INITIALIZED = True


def lemmatize_text(
        text: str,
        remove_stopwords: bool = False
) -> str:
    """
    Lemmatize tokenized text using NLTK WordNetLemmatizer
    with POS-aware tagging.

    Args:
        text: Preprocessed text string
        remove_stopwords: Filter out English stopwords before lemmatization

    Returns:
        Lemmatized text with tokens joined by spaces

    Note:
        Uses lazy imports to avoid NLTK overhead
        when lemmatization is disabled in config.
        POS tags are mapped to WordNet format (a/v/n/r)
        for accurate lemmatization.
    """
    # Lazy imports reduce startup time when lemmatize=False
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    from nltk import pos_tag

    _ensure_nltk_resources()
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)

    if remove_stopwords:
        stop_words = set(stopwords.words("english"))
        tokens = [t for t in tokens if t not in stop_words]

    # WordNet lemmatizer requires POS tags in specific format
    def get_wordnet_pos(treebank_tag: str) -> str:
        """Map Penn Treebank POS tags to WordNet format."""
        if treebank_tag.startswith('J'):
            return 'a'  # adjective
        elif treebank_tag.startswith('V'):
            return 'v'  # verb
        elif treebank_tag.startswith('N'):
            return 'n'  # noun
        elif treebank_tag.startswith('R'):
            return 'r'  # adverb
        else:
            return 'n'  # default to noun for unknown tags

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
    """
    Apply full preprocessing pipeline to a single text sample.

    Args:
        text: Raw input text
        config: Configuration dict with preprocessing settings

    Returns:
        Fully preprocessed text string

    Note:
        Config structure:
        {
            "preprocessing": {
                "cleaning": { ... },  # Passed to clean_text()
                "nlp": { "lemmatize": bool, "remove_stopwords": bool }
            }
        }
    """
    # Safe nested access with defaults prevents KeyError
    cleaning_cfg = config.get("preprocessing", {}).get("cleaning", {})
    nlp_cfg = config.get("preprocessing", {}).get("nlp", {})

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

    if nlp_cfg.get("lemmatize", False):
        cleaned = lemmatize_text(
            cleaned,
            remove_stopwords=nlp_cfg.get("remove_stopwords", False)
        )

    return cleaned


def create_vectorizer(config: dict) -> TfidfVectorizer | CountVectorizer:
    """
    Initialize and return a vectorizer based on configuration.

    Args:
        config: Configuration dict with vectorizer settings

    Returns:
        Configured TfidfVectorizer or CountVectorizer instance

    Raises:
        ValueError: If required vectorizer config keys are missing
        or type is unknown

    Note:
        TF-IDF uses sublinear_tf=True (1 + log(tf))
        to reduce impact of very frequent terms.
        dtype=np.float64/int64 ensures compatibility with sklearn metrics
        and scipy sparse matrices.
    """
    vectorizer_cfg = config.get("preprocessing", {}).get("vectorizer", {})
    nlp_cfg = config.get("preprocessing", {}).get("nlp", {})

    logger.debug(
        f"Vectorizer config: type={vectorizer_cfg.get('type')}, "
        f"max_features={vectorizer_cfg.get('max_features')}, "
        f"ngram_range={vectorizer_cfg.get('ngram_range')}"
    )

    # Explicit validation with actionable error message
    required_keys = ["type", "max_features", "ngram_range"]
    missing = [k for k in required_keys if k not in vectorizer_cfg]
    if missing:
        raise ValueError(f"Missing required vectorizer config keys: {missing}")

    vectorizer_type = vectorizer_cfg.get("type", "tfidf").lower()

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
            sublinear_tf=True,  # Reduces weight of very frequent terms
            dtype=np.float64,  # Ensures numeric stability in downstream ops
        )
    elif vectorizer_type == "count":
        logger.info(
            f"Initializing CountVectorizer with params: {common_params}"
        )
        return CountVectorizer(**common_params, dtype=np.int64)
    else:
        # Explicit error for unknown vectorizer type
        raise ValueError(f"Unknown vectorizer type: {vectorizer_type}")


def preprocess_texts(
    texts: list[str],
    config: dict,
) -> list[str]:
    """
    Apply text cleaning and optional lemmatization to a list of texts.

    Args:
        texts: List of raw input strings
        config: Configuration dict for preprocessing pipeline

    Returns:
        List of preprocessed text strings (same length as input)

    Note:
        Logs warning if aggressive cleaning produces empty strings —
        useful for tuning remove_special_chars/remove_urls parameters.
    """
    logger.debug(f"Preprocessing {len(texts)} texts...")

    processed = [preprocess_text(text, config) for text in texts]

    # Helps detect over-aggressive cleaning settings
    empty_count = sum(1 for t in processed if not t.strip())
    if empty_count > 0:
        logger.warning(
            f"{empty_count}/{len(processed)} texts resulted in empty strings "
            "after preprocessing. Consider adjusting cleaning parameters."
        )

    # Log sample only if DEBUG enabled to avoid overhead in production
    if logger.isEnabledFor(logging.DEBUG) and processed:
        logger.debug(f"Sample preprocessed text: '{processed[0][:100]}...'")

    return processed
