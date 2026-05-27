"""
Centralized logging configuration for the project.
Provides a reusable logger factory with console and optional file output.
"""
import logging
from pathlib import Path
import sys


def setup_logger(
    name: str,
    level: str = "INFO",
    log_file: str | None = None
) -> logging.Logger:
    """
    Configure and return a named logger with console
    and optional file handlers.

    Args:
        name: Logger identifier (typically `__name__`)
        level: Logging threshold (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to append logs to a file

    Returns:
        Configured logging.Logger instance

    Raises:
        ValueError: If an invalid logging level is provided
    """
    level = level.upper()
    valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    if level not in valid_levels:
        raise ValueError(
            f"Invalid log level: {level}. Expected one of {valid_levels}"
        )

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Idempotent: skip reconfiguration if handlers already exist
    # (prevents duplicate logs)
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            log_file, encoding="utf-8", mode="a"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
