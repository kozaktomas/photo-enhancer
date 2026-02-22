"""Structured JSON logging configuration for the photo enhancer service."""

import logging
import sys

from pythonjsonlogger.json import JsonFormatter


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with JSON structured output.

    Sets up the root logger with a JsonFormatter that outputs one JSON object per line
    to stdout. Fields include: timestamp, level, logger name, message, plus any extras
    passed via the `extra` kwarg on log calls.

    Args:
        level: Logging level (default: logging.INFO).
    """
    handler = logging.StreamHandler(sys.stdout)
    formatter = JsonFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        rename_fields={"asctime": "timestamp", "levelname": "level", "name": "logger"},
    )
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
