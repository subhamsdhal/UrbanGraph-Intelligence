"""Opt-in JSON logging; importing this module never configures the root logger."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from threading import RLock
from typing import TextIO

_LOCK = RLock()


class JsonFormatter(logging.Formatter):
    """Emit one JSON object per record with a UTC ISO-8601 timestamp."""

    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(
            {
                "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            },
            ensure_ascii=True,
        )


class _JsonHandler(logging.StreamHandler[TextIO]):
    """Identify handlers owned by this module without changing caller handlers."""


def configure_logging(
    *,
    level: int | str = logging.INFO,
    logger_name: str = "urbangraph",
    stream: TextIO | None = None,
) -> logging.Logger:
    """Configure a named logger; repeated calls replace only our own handler.

    The default stream is stderr. Propagation is disabled to prevent duplicate
    root output. Caller-installed handlers remain caller-owned. Messages must not
    contain secrets: this formatter is not a redaction system. Extras and traceback
    data are deliberately excluded from the minimal schema.
    """
    if not logger_name or logger_name == "root":
        raise ValueError("Use a named logger, not the root logger")
    with _LOCK:
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
        handler = _JsonHandler(stream)
        handler.setFormatter(JsonFormatter())
        for existing in tuple(logger.handlers):
            if isinstance(existing, _JsonHandler):
                logger.removeHandler(existing)
                existing.close()
        logger.addHandler(handler)
        logger.propagate = False
        return logger
