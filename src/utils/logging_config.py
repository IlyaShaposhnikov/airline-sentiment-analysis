"""
Centralized logging configuration for the project.
Provides a reusable logger factory with console and optional file output.
"""
import logging
from pathlib import Path
import sys

VALID_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


def _normalize_level(level: str) -> str:
    """Upper-case ``level`` and check it is a standard logging level."""
    level = str(level).upper()
    if level not in VALID_LOG_LEVELS:
        raise ValueError(
            f"Invalid log level: {level}. Expected one of {VALID_LOG_LEVELS}"
        )
    return level


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
    level = _normalize_level(level)

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

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


def set_package_log_level(level: str, prefix: str = "src") -> str:
    """
    Set the level of every existing logger in a package.

    ``setup_logger`` gives each module logger its own level, so changing
    the parent ("src") alone has no effect; this walks all loggers named
    ``prefix`` or ``prefix.*``.

    Args:
        level: Logging level name (case-insensitive)
        prefix: Package name whose loggers are updated

    Returns:
        The normalized level name that was applied

    Raises:
        ValueError: If an invalid logging level is provided
    """
    level = _normalize_level(level)
    numeric = getattr(logging, level)
    for name, logger in list(logging.root.manager.loggerDict.items()):
        if not isinstance(logger, logging.Logger):
            continue  # PlaceHolder for a not-yet-created parent
        if name == prefix or name.startswith(prefix + "."):
            logger.setLevel(numeric)
    logging.getLogger(prefix).setLevel(numeric)
    return level
