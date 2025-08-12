"""Logging setup utilities using Loguru."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

from loguru import logger


def setup_logging(
    *,
    log_dir: str | Path = "logs",
    console_level: Literal[
        "TRACE",
        "DEBUG",
        "INFO",
        "SUCCESS",
        "WARNING",
        "ERROR",
        "CRITICAL",
    ] = "INFO",
    file_level: Literal[
        "TRACE",
        "DEBUG",
        "INFO",
        "SUCCESS",
        "WARNING",
        "ERROR",
        "CRITICAL",
    ] = "DEBUG",
    low_importance_filename: str = "low_importance.log",
    low_importance_max_megabytes: int = 2,
) -> None:
    """Configure Loguru sinks for the app.

    - Console sink shows messages >= console_level (default: INFO)
    - File sink stores lower-importance logs (default: DEBUG) with rotation at
      N megabytes (default: 2MB)
    - Existing handlers are removed to avoid duplicate logs on repeated setup
    """
    logger.remove()

    # Console sink (human facing)
    def _console_sink(message: str) -> None:
        sys.stdout.write(message)
        sys.stdout.flush()

    logger.add(
        sink=_console_sink,
        level=console_level,
        colorize=True,
        backtrace=False,
        diagnose=False,
    )

    # Rotating file sink for low-importance/noisy logs
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    file_path = log_path / low_importance_filename
    rotation_size = f"{max(1, low_importance_max_megabytes)} MB"

    # Delete rotated files immediately to emulate "clean and rewrite"
    logger.add(
        file_path,
        level=file_level,
        rotation=rotation_size,
        retention=0,
        encoding="utf-8",
        enqueue=True,
        backtrace=False,
        diagnose=False,
        filter=lambda record: record["level"].name in ("TRACE", "DEBUG"),
    )


__all__ = ["logger", "setup_logging"]
