"""Logging setup utilities using Loguru.

Enhancements:
- Console sink with colors
- App file (INFO+) rotated daily with retention/compression
- Debug file (TRACE/DEBUG) rotated by size (default 10MB)
- Error file (WARNING+) for quick triage
- Optional JSON serialization for machine parsing
- Intercept stdlib ``logging`` into Loguru
- Component-bound loggers via ``get_logger("component")``
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Literal

from loguru import logger as loguru_logger

# Export a logger pre-bound with a default component placeholder
logger = loguru_logger.bind(component="-")


def setup_logging(  # noqa: PLR0913
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
    # Debug file settings (TRACE/DEBUG only)
    file_level: Literal[
        "TRACE",
        "DEBUG",
        "INFO",
        "SUCCESS",
        "WARNING",
        "ERROR",
        "CRITICAL",
    ] = "DEBUG",
    low_importance_filename: str = "debug.log",
    low_importance_max_megabytes: int = 10,
    # keep previous behavior (delete rotations)
    low_importance_retention: str | int = 0,
    # App and error files
    app_filename: str = "app.log",
    app_rotation: str = "00:00",  # rotate daily at midnight
    app_retention: str | int = "14 days",
    error_filename: str = "errors.log",
    # Output options
    compression: str | None = "zip",
    serialize: bool = False,
    intercept_stdlib: bool = True,
    backtrace: bool | None = None,
    diagnose: bool | None = None,
) -> None:
    """Configure Loguru sinks for the app.

    - Console sink shows messages >= console_level (default: INFO)
    - Debug file stores TRACE/DEBUG logs with size rotation (default: 10MB)
    - App file stores INFO+ with daily rotation and retention
    - Error file stores WARNING+ for quick triage
    - Existing handlers are removed to avoid duplicate logs on repeated setup
    """
    logger.remove()

    # Resolve dynamic flags (env overrides)
    if backtrace is None:
        debug_env = os.getenv("LOGURU_BACKTRACE", os.getenv("DEBUG", "0"))
        backtrace = str(debug_env).lower() in {"1", "true", "yes", "on"}
    if diagnose is None:
        diagnose = backtrace

    # Common format (includes bound component)
    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "{process.name}:{thread.name} | "
        "{extra[component]: <12} | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )

    # Console sink (human facing)
    def _console_sink(message: str) -> None:
        sys.stdout.write(message)
        sys.stdout.flush()

    logger.add(
        sink=_console_sink,
        level=console_level,
        colorize=True,
        backtrace=backtrace,
        diagnose=diagnose,
        format=fmt,
        serialize=False,
    )

    # Files base path
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # App file: INFO+
    app_file = log_path / app_filename
    logger.add(
        app_file,
        level="INFO",
        rotation=app_rotation,
        retention=app_retention,
        encoding="utf-8",
        enqueue=True,
        backtrace=backtrace,
        diagnose=diagnose,
        compression=compression,
        serialize=serialize,
        format=fmt,
    )

    # Debug file: TRACE/DEBUG only with size-based rotation
    debug_file = log_path / low_importance_filename
    rotation_size = f"{max(1, low_importance_max_megabytes)} MB"
    logger.add(
        debug_file,
        level=file_level,
        rotation=rotation_size,
        retention=low_importance_retention,
        encoding="utf-8",
        enqueue=True,
        backtrace=backtrace,
        diagnose=diagnose,
        compression=compression,
        serialize=serialize,
        format=fmt,
        filter=lambda record: record["level"].name in ("TRACE", "DEBUG"),
    )

    # Error file: WARNING+
    err_file = log_path / error_filename
    logger.add(
        err_file,
        level="WARNING",
        rotation=app_rotation,
        retention=app_retention,
        encoding="utf-8",
        enqueue=True,
        backtrace=backtrace,
        diagnose=diagnose,
        compression=compression,
        serialize=serialize,
        format=fmt,
    )

    # Optionally intercept stdlib logging
    if intercept_stdlib:
        intercept_stdlib_logging()


def get_logger(component: str | None = None) -> object:
    """Return a logger bound with a component name for contextual logs.

    Example:
        log = get_logger("chat")
        log.info("Answer generated", extra={"query_id": "..."})

    """
    return logger if not component else logger.bind(component=component)


class _InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        """Redirect stdlib logging record to Loguru preserving level & caller."""
        try:
            # Use name if recognized, else fallback to numeric level
            loguru_logger.level(record.levelname)
            level = record.levelname
        except ValueError:
            level = record.levelno
        loguru_logger.opt(depth=6, exception=record.exc_info).log(
            level,
            record.getMessage(),
        )


def intercept_stdlib_logging(
    names: tuple[str, ...] | None = None,
    level: int = logging.INFO,
) -> None:
    """Intercept stdlib logging and forward to Loguru.

    Args:
        names: Tuple of logger names to intercept. Defaults to root only.
        level: Level to set on intercepted loggers.

    """
    target_names = names or ("",)
    handler = _InterceptHandler()
    for name in target_names:
        std_logger = logging.getLogger(name)
        std_logger.handlers = [handler]
        std_logger.propagate = False
        std_logger.setLevel(level)


__all__ = [
    "get_logger",
    "intercept_stdlib_logging",
    "logger",
    "setup_logging",
]
