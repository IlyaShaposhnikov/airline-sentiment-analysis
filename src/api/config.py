"""
API configuration loaded from project config.yaml.

Centralizes all configurable settings for the sentiment analysis API,
with optional environment variable overrides for deployment flexibility.
"""

import os
import yaml
from pathlib import Path
from typing import Optional

# ============================================================================
# Path resolution
# ============================================================================

# PROJECT_ROOT = /path/to/airline-sentiment-analysis (repo root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

# ============================================================================
# Load base config from YAML
# ============================================================================


def _load_base_config() -> dict:
    """Load the main config.yaml with error handling."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Config file not found: {CONFIG_PATH}\n"
            "Please ensure configs/config.yaml exists in the project root."
        )

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Load once at module import
_BASE_CONFIG = _load_base_config()

# ============================================================================
# Helper: Get nested config value with optional env override
# ============================================================================


def _get_config_value(
    *keys: str,
    default=None,
    env_var: Optional[str] = None
):
    """
    Get a nested value from config.yaml with optional env var override.

    Usage:
        _get_config_value("serving", "api", "port", default=8000)
        _get_config_value("model", "path", env_var="MODEL_PATH", default="...")
    """
    # Check env var first (if provided) for deployment flexibility
    if env_var and env_var in os.environ:
        return os.environ[env_var]

    # Traverse nested dict
    value = _BASE_CONFIG
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value

# ============================================================================
# Model configuration
# ============================================================================


# Model path: from config.yaml, with optional MODEL_PATH env override
MODEL_PATH = _get_config_value(
    "model", "artifacts", "path",
    default="artifacts/model_bundle.joblib",
    env_var="MODEL_PATH"  # Optional override for deployment
)

# ============================================================================
# API server configuration (from serving.api section)
# ============================================================================

API_HOST = _get_config_value("serving", "api", "host", default="0.0.0.0")
API_PORT = _get_config_value("serving", "api", "port", default=8000)
API_ENABLED = _get_config_value("serving", "api", "enabled", default=True)
CORS_ALLOWED_ORIGINS: list[str] = ["*"]

# ============================================================================
# API metadata
# ============================================================================

API_TITLE = "Airline Sentiment Analysis API"
API_DESCRIPTION = "REST API for sentiment classification of airline tweets"
API_VERSION = "1.0.0"
API_DOCS_URL = "/docs"
API_REDOC_URL = "/redoc"
API_MAX_REQUEST_SIZE = _get_config_value(
    "serving", "api", "limits", "max_request_size_mb", default=10
)

# ============================================================================
# Request/response limits (can be extended in config.yaml if needed)
# ============================================================================

MAX_TEXT_LENGTH = _get_config_value(
    "serving", "api", "limits", "max_text_length", default=1000
)
MAX_BATCH_SIZE = _get_config_value(
    "serving", "api", "limits", "max_batch_size", default=100
)
MIN_EXPLAIN_COUNT = _get_config_value(
    "serving", "api", "limits", "min_explain_count", default=1
)
MAX_EXPLAIN_COUNT = _get_config_value(
    "serving", "api", "limits", "max_explain_count", default=20
)

# ============================================================================
# Logging configuration
# ============================================================================

LOG_LEVEL = _get_config_value(
    "serving", "api", "log_level",
    default="INFO",
    env_var="API_LOG_LEVEL"
)
LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"

# ============================================================================
# Convenience functions
# ============================================================================


def get_model_path() -> Path:
    """Get the configured model path as a Path object."""
    return Path(MODEL_PATH)


def is_model_available() -> bool:
    """Check if the model file exists at the configured path."""
    return get_model_path().exists()


def get_api_config() -> dict:
    """Get all API-related config as a dictionary (for debugging/docs)."""
    return {
        "model_path": MODEL_PATH,
        "host": API_HOST,
        "port": API_PORT,
        "enabled": API_ENABLED,
        "title": API_TITLE,
        "version": API_VERSION,
        "limits": {
            "max_text_length": MAX_TEXT_LENGTH,
            "max_batch_size": MAX_BATCH_SIZE,
        }
    }
