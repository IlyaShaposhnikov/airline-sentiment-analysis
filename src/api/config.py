"""
API configuration loaded from project config.yaml.

Centralizes all configurable settings for the sentiment analysis API,
with optional environment variable overrides for deployment flexibility.
"""

import os
import yaml
from pathlib import Path
from typing import Optional
import json

from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)

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
    env_var: Optional[str] = None,
    cast_type: Optional[type] = None
):
    """
    Get a nested value from config.yaml with optional env var override.

    Args:
        cast_type: Optional type to cast the value to (int, float, bool, list)
    """
    # Check env var first (if provided)
    if env_var and env_var in os.environ:
        value = os.environ[env_var]
        if cast_type and value is not None:
            try:
                if cast_type == list:
                    return json.loads(value)
                elif cast_type == bool:
                    # Safe boolean parsing: "false", "0", "no", "off" → False
                    return value.lower() in ("true", "1", "yes", "on")
                else:
                    return cast_type(value)
            except (ValueError, TypeError, json.JSONDecodeError):
                logger.warning(
                    f"Failed to cast env var '{env_var}' "
                    f"to {cast_type.__name__}: '{value}'. Using default."
                )
                return default
        return value

    # Traverse nested dict from YAML (types preserved)
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
API_PORT = _get_config_value(
    "serving", "api", "port", default=8000, cast_type=int
)
API_ENABLED = _get_config_value(
    "serving", "api", "enabled", default=True, cast_type=bool
)
CORS_ALLOWED_ORIGINS = _get_config_value(
    "serving", "api", "cors_origins",
    default=["*"],
    env_var="CORS_ORIGINS",
    cast_type=list
)
RATE_LIMIT_PER_MINUTE = _get_config_value(
    "serving", "api", "rate_limit_per_minute", default=60, cast_type=int
)

# ============================================================================
# API metadata
# ============================================================================

API_TITLE = "Airline Sentiment Analysis API"
API_DESCRIPTION = "REST API for sentiment classification of airline tweets"
API_VERSION = "1.0.0"
API_DOCS_URL = "/docs"
API_REDOC_URL = "/redoc"
API_MAX_REQUEST_SIZE = _get_config_value(
    "serving", "api", "limits", "max_request_size_mb",
    default=10, cast_type=int
)

# ============================================================================
# Request/response limits (can be extended in config.yaml if needed)
# ============================================================================

MAX_TEXT_LENGTH = _get_config_value(
    "serving", "api", "limits", "max_text_length", default=1000, cast_type=int
)
MAX_BATCH_SIZE = _get_config_value(
    "serving", "api", "limits", "max_batch_size", default=100, cast_type=int
)
MIN_EXPLAIN_COUNT = _get_config_value(
    "serving", "api", "limits", "min_explain_count", default=1, cast_type=int
)
MAX_EXPLAIN_COUNT = _get_config_value(
    "serving", "api", "limits", "max_explain_count", default=20, cast_type=int
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
