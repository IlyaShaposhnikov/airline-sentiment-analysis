"""
Data loading and preparation module.

Handles YAML configuration parsing, CSV ingestion, confidence-based filtering,
and target encoding. All paths are resolved relative to project root.
"""
from pathlib import Path

import pandas as pd
import yaml

from .constants import TARGET_MAPPING
from .utils.logging_config import setup_logger

logger = setup_logger(__name__)
BASE_DIR = Path(__file__).resolve().parents[1]


def load_config(config_path: str | Path = "configs/config.yaml") -> dict:
    """
    Load and parse project configuration from YAML file.

    Resolves relative paths against project root for consistent behavior
    regardless of working directory.

    Args:
        config_path: Path to config.yaml (relative or absolute)

    Returns:
        Parsed configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist at resolved path
        yaml.YAMLError: If YAML syntax is invalid
    """
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = BASE_DIR / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_and_prepare_data(
    config_path: str | Path = "configs/config.yaml",
    base_dir: Path | None = None
) -> pd.DataFrame:
    """
    Load raw CSV data, apply confidence filtering, and encode targets.

    Pipeline:
    1. Validate config structure (required keys, types)
    2. Load CSV with resolved path
    3. Select core columns (target, text, confidence scores)
    4. Drop rows with missing critical values
    5. Filter by sentiment confidence threshold (configurable)
    6. Encode string labels → integers via TARGET_MAPPING
    7. Drop rows with unmapped target values (log warning)

    Args:
        config_path: Path to config.yaml
        base_dir: Optional base directory for path resolution
        (defaults to project root)

    Returns:
        Cleaned DataFrame with columns: text, sentiment_confidence,
        reason_confidence, target

    Raises:
        ValueError: If config is missing required keys or has invalid structure
        TypeError: If confidence_columns is not a dict
        FileNotFoundError: If dataset CSV doesn't exist at resolved path
    """
    if base_dir is None:
        base_dir = BASE_DIR

    cfg = load_config(config_path)
    data_cfg = cfg.get("data", {})

    # Validate top-level required keys
    required_data_keys = [
        "path", "target_column", "text_column", "confidence_columns"
    ]
    missing_keys = [k for k in required_data_keys if k not in data_cfg]
    if missing_keys:
        raise ValueError(
            "Missing required keys in [data] section "
            f"of config: {missing_keys}. "
            f"Expected: {required_data_keys}"
        )

    # Validate nested confidence_columns structure
    conf_cols = data_cfg["confidence_columns"]
    if not isinstance(conf_cols, dict):
        raise TypeError(
            "data.confidence_columns must be a dict, "
            f"got {type(conf_cols).__name__}"
        )

    required_conf_keys = ["sentiment", "reason"]
    missing_conf = [k for k in required_conf_keys if k not in conf_cols]
    if missing_conf:
        raise ValueError(
            "Missing required keys in "
            f"data.confidence_columns: {missing_conf}. "
            f"Expected: {required_conf_keys}"
        )

    # Resolve dataset path relative to base_dir
    data_path = Path(data_cfg["path"])
    if not data_path.is_absolute():
        data_path = base_dir / data_path

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. "
            f"Working directory: {Path.cwd()}. "
            "Please download Tweets.csv to data/"
        )

    df = pd.read_csv(data_path)
    logger.info(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # Select only required columns
    # (.copy() to avoid SettingWithCopyWarning downstream)
    cols_to_keep = [
        data_cfg["target_column"],
        data_cfg["text_column"],
        data_cfg["confidence_columns"]["sentiment"],
        data_cfg["confidence_columns"]["reason"]
    ]
    df = df[cols_to_keep].copy()

    # Drop rows missing critical fields (text or target)
    df = df.dropna(subset=[data_cfg["text_column"], data_cfg["target_column"]])
    logger.info(f"Dropped missing values: {df.shape[0]} rows remaining")

    logger.debug(
        f"Target distribution (before filtering):\n"
        f"{df[data_cfg['target_column']].value_counts(normalize=True).to_dict()}"  # noqa: E501
    )

    # Filter by confidence threshold: keep only high-confidence annotations
    conf_thresh = data_cfg.get("confidence_threshold", 0.7)
    conf_col = data_cfg["confidence_columns"]["sentiment"]
    df_filtered = df[df[conf_col] >= conf_thresh].copy()
    logger.info(
        f"Filtered by confidence (>={conf_thresh}): "
        f"{df_filtered.shape[0]} rows remaining"
    )

    logger.debug(
        f"Target distribution (after filtering):\n"
        f"{df_filtered[data_cfg['target_column']].value_counts(normalize=True).to_dict()}"  # noqa: E501
    )

    # Standardize column names for downstream modules
    df_filtered = df_filtered.rename(columns={
        data_cfg["confidence_columns"]["sentiment"]: "sentiment_confidence",
        data_cfg["confidence_columns"]["reason"]: "reason_confidence"
    })

    # Encode string labels → integers using TARGET_MAPPING
    # Use "Int64" (nullable integer) to preserve NaN for unmapped values
    df_filtered["target"] = (
        df_filtered[data_cfg["target_column"]]
        .map(TARGET_MAPPING)
        .astype("Int64")  # Nullable integer
    )

    # Drop rows with unmapped target values (log which values were dropped)
    invalid_mask = df_filtered["target"].isna()
    if invalid_mask.any():
        logger.warning(
            f"Dropped {invalid_mask.sum()} rows with unknown target values: "
            f"{df_filtered.loc[invalid_mask, data_cfg['target_column']].unique()}"  # noqa: E501
        )
        df_filtered = df_filtered.dropna(subset=["target"])

    # Convert to standard int (no NaNs remain at this point)
    df_filtered["target"] = df_filtered["target"].astype(int)
    logger.info(f"Final dataset shape: {df_filtered.shape[0]} rows")
    return df_filtered


if __name__ == "__main__":
    # Quick validation: run module directly to test data loading
    setup_logger(__name__, level="DEBUG")
    df = load_and_prepare_data()
    logger.info("\nTarget distribution:")
    logger.info(
        df["target"].value_counts(normalize=True).sort_index().to_dict()
    )
    logger.info("\nFirst 3 rows:")
    logger.info("\n" + df.head(3).to_string())
