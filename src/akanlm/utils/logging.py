"""Logging utilities for reproducible pipeline runs."""

import logging
import os


def get_logger(name: str) -> logging.Logger:
    """Return the configured logger for ``name``."""
    level = os.environ.get("AKANLM_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=level, format="%(levelname)s | %(name)s | %(message)s")
    return logging.getLogger(name)
