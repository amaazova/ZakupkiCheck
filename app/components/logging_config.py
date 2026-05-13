"""structlog setup: console renderer by default, JSON when LOG_FORMAT=json."""
from __future__ import annotations

import logging
import os
import sys

import structlog

_CONFIGURED = False


def setup_logging(level: str | int = "INFO") -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_level = level if isinstance(level, int) else logging.getLevelName(str(level).upper())
    if not isinstance(log_level, int):
        log_level = logging.INFO

    json_mode = os.environ.get("LOG_FORMAT", "").lower() == "json"
    renderer = (
        structlog.processors.JSONRenderer()
        if json_mode
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )
    _CONFIGURED = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    setup_logging()
    if name:
        return structlog.get_logger(name)
    return structlog.get_logger()
