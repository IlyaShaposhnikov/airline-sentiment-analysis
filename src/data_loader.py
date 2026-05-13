import pandas as pd
from pathlib import Path
import yaml

from .constants import TARGET_MAPPING
from .utils.logging_config import setup_logger

logger = setup_logger(__name__)
BASE_DIR = Path(__file__).resolve().parents[1]


def load_config(config_path: str | Path = "configs/config.yaml") -> dict:
    """Load project configuration from YAML file with path resolution."""
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = BASE_DIR / config_path

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_and_prepare_data(
    config_path: str | Path = "configs/config.yaml",
    base_dir: Path | None = None
) -> pd.DataFrame:
    """
    Load raw CSV, select relevant columns, filter by confidence,
    handle missing values, and return cleaned DataFrame.
    """
    if base_dir is None:
        base_dir = BASE_DIR

    cfg = load_config(config_path)
    data_cfg = cfg.get("data", {})

    # 1. Load raw dataset
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

    # 2. Select core columns
    cols_to_keep = [
        data_cfg["target_column"],
        data_cfg["text_column"],
        data_cfg["confidence_columns"]["sentiment"],
        data_cfg["confidence_columns"]["reason"]
    ]
    df = df[cols_to_keep].copy()

    # 3. Drop rows with missing text or target
    df = df.dropna(subset=[data_cfg["text_column"], data_cfg["target_column"]])
    logger.info(f"Dropped missing values: {df.shape[0]} rows remaining")

    # Log distribution before filtering
    logger.debug(
        f"Target distribution (before filtering):\n"
        f"{df[data_cfg[
            'target_column'
        ]].value_counts(normalize=True).to_dict()}"
    )

    # 4. Filter by sentiment confidence threshold
    conf_thresh = data_cfg.get("confidence_threshold", 0.7)
    conf_col = data_cfg["confidence_columns"]["sentiment"]
    df_filtered = df[df[conf_col] >= conf_thresh].copy()
    logger.info(
        f"Filtered by confidence (>={conf_thresh}): "
        f"{df_filtered.shape[0]} rows remaining"
    )

    # Log distribution after filtering
    logger.debug(
        f"Target distribution (after filtering):\n"
        f"{df_filtered[data_cfg[
            'target_column'
        ]].value_counts(normalize=True).to_dict()}"
    )

    # 5. Rename confidence columns for consistency
    df_filtered = df_filtered.rename(columns={
        data_cfg["confidence_columns"]["sentiment"]: "sentiment_confidence",
        data_cfg["confidence_columns"]["reason"]: "reason_confidence"
    })

    # 6. Encode target variable
    df_filtered["target"] = (
        df_filtered[data_cfg["target_column"]]
        .map(TARGET_MAPPING)
        .astype("Int64")  # Nullable integer
    )

    # Drop unknown target values if any
    invalid_mask = df_filtered["target"].isna()
    if invalid_mask.any():
        logger.warning(
            f"Dropped {invalid_mask.sum()} rows with unknown target values: "
            f"{df_filtered.loc[
                invalid_mask, data_cfg['target_column']
            ].unique()}"
        )
        df_filtered = df_filtered.dropna(subset=["target"])

    df_filtered["target"] = df_filtered["target"].astype(int)
    logger.info(f"Final dataset shape: {df_filtered.shape[0]} rows")
    return df_filtered


if __name__ == "__main__":
    # Quick validation test
    setup_logger(__name__, level="DEBUG")
    df = load_and_prepare_data()
    logger.info("\nTarget distribution:")
    logger.info(
        df["target"].value_counts(normalize=True).sort_index().to_dict()
    )
    logger.info("\nFirst 3 rows:")
    logger.info("\n" + df.head(3).to_string())
