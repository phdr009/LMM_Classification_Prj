"""Logging helpers for the hybrid pipeline."""
from __future__ import annotations

import logging
from logging.config import dictConfig
from typing import Optional


def setup_logging(level: str = "INFO", json: bool = False) -> None:
    """Configure application logging."""

    formatter = {
        "()": "logging.Formatter",
        "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    }
    if json:
        formatter = {
            "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
            "fmt": "%(asctime)s %(levelname)s %(name)s %(message)s",
        }
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"default": formatter},
        "handlers": {
            "default": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "level": level,
            }
        },
        "root": {"handlers": ["default"], "level": level},
    }
    try:
        dictConfig(config)
    except Exception:  # pragma: no cover - fallback to basicConfig
        logging.basicConfig(level=level)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Create or fetch a named logger."""

    return logging.getLogger(name)


__all__ = ["setup_logging", "get_logger"]
