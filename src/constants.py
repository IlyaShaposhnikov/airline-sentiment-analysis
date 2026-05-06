from typing import Dict

# Target mapping for sentiment classification
TARGET_MAPPING: Dict[str, int] = {
    "positive": 1,
    "negative": 0,
    "neutral": 2
}

# Reverse mapping for prediction interpretation
TARGET_MAPPING_INV: Dict[int, str] = {v: k for k, v in TARGET_MAPPING.items()}

# Default column names (fallback)
DEFAULT_COLUMNS = {
    "target": "airline_sentiment",
    "text": "text",
    "confidence_sentiment": "airline_sentiment_confidence",
    "confidence_reason": "negativereason_confidence"
}
