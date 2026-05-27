"""
Global constants for label mapping and default dataset schema.
Used across data loading, training, evaluation, and API response formatting.
"""
from typing import Final

# Label → index mapping for scikit-learn classifiers
TARGET_MAPPING: Final[dict[str, int]] = {
    "positive": 1,
    "negative": 0,
    "neutral": 2,
}

# Index → label mapping for decoding model predictions
TARGET_MAPPING_INV: Final[
    dict[int, str]
] = {v: k for k, v in TARGET_MAPPING.items()}

# Fallback column names for raw CSV ingestion (overridden by config.yaml)
DEFAULT_COLUMNS: Final[dict[str, str]] = {
    "target": "airline_sentiment",
    "text": "text",
    "confidence_sentiment": "airline_sentiment_confidence",
    "confidence_reason": "negativereason_confidence",
}
