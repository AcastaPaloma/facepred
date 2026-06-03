"""Logging setup helpers built on the standard library."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TextIO

DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_LOG_ONCE_KEYS: set[tuple[str, int, str]] = set()


def coerce_log_level(level: int | str) -> int:
    """Convert a logging level name or integer to a standard logging level."""
    if isinstance(level, int):
        return level
    normalized = level.upper()
    if not hasattr(logging, normalized):
        raise ValueError(f"Unknown logging level: {level!r}.")
    resolved = getattr(logging, normalized)
    if not isinstance(resolved, int):
        raise ValueError(f"Unknown logging level: {level!r}.")
    return resolved


def setup_logging(
    *,
    level: int | str = "INFO",
    log_file: str | Path | None = None,
    stream: TextIO | None = None,
    force: bool = False,
    fmt: str = DEFAULT_LOG_FORMAT,
    datefmt: str = DEFAULT_DATE_FORMAT,
) -> logging.Logger:
    """Configure root logging for scripts and notebooks.

    ``force=True`` replaces existing root handlers, which is useful in tests and
    repeated notebook runs. Without it, duplicate stream/file handlers are
    avoided.
    """
    root = logging.getLogger()
    level_no = coerce_log_level(level)
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    if force:
        for handler in list(root.handlers):
            root.removeHandler(handler)
            handler.close()

    has_stream_handler = any(
        isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler)
        for handler in root.handlers
    )
    if not has_stream_handler:
        stream_handler = logging.StreamHandler(stream or sys.stderr)
        stream_handler.setFormatter(formatter)
        root.addHandler(stream_handler)
    else:
        for handler in root.handlers:
            handler.setFormatter(formatter)

    if log_file is not None:
        file_path = Path(log_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_path = str(file_path.resolve())
        has_file_handler = any(
            isinstance(handler, logging.FileHandler)
            and str(Path(handler.baseFilename).resolve()) == resolved_path
            for handler in root.handlers
        )
        if not has_file_handler:
            file_handler = logging.FileHandler(file_path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)

    root.setLevel(level_no)
    for handler in root.handlers:
        handler.setLevel(level_no)
    return root


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a standard-library logger."""
    return logging.getLogger(name)


def log_once(
    logger: logging.Logger,
    level: int | str,
    message: str,
    *,
    key: str | None = None,
) -> bool:
    """Log a message once per process and return whether it was emitted."""
    level_no = coerce_log_level(level)
    dedupe_key = (logger.name, level_no, key or message)
    if dedupe_key in _LOG_ONCE_KEYS:
        return False
    logger.log(level_no, message)
    _LOG_ONCE_KEYS.add(dedupe_key)
    return True
