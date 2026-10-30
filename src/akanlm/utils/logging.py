"""Logging utilities for reproducible pipeline runs."""

import logging
import os


def get_logger(name: str) -> logging.Logger:
    """Return a configured module logger.

    Args:
        name: Logger namespace.
    Returns:
        A standard library logger instance.
    """
    level = os.environ.get("AKANLM_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=level, format="%(levelname)s | %(name)s | %(message)s")
    return logging.getLogger(name)
