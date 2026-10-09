"""Structured logging with request correlation.

Two output formats:

* ``json`` — one JSON object per line on stdout. Extras passed via
  ``logger.info("...", extra={...})`` are merged into the JSON payload.
* ``text`` — human-readable single-line format with ``[rid=<request_id>]``.

In both modes, every record carries the current ``request_id``, sourced
from the ``request_id_var`` contextvar set by the request middleware.

The setup function is idempotent — calling it twice does not duplicate
handlers — and accepts optional ``level`` and ``fmt`` overrides so
callers can configure from ``Settings`` or hard-coded test values.
"""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from datetime import UTC, datetime

from app.core.config import settings

# Request correlation

# Set by request-scoped middleware. Logged as ``request_id`` in JSON mode
# and as ``rid=<value>`` in text mode. The default "-" means "no request
# context" (e.g. logs emitted during process startup).
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


class RequestIdFilter(logging.Filter):
    """Inject the current request_id onto every log record.

    A caller that explicitly sets ``request_id`` via ``extra={...}`` wins;
    the filter only fills in the value from the contextvar when the field
    is absent.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = request_id_var.get()
        return True


# JSON formatter

# Attributes that always exist on a LogRecord. Anything else in
# ``record.__dict__`` was added via ``extra={...}`` or a filter and is
# merged into the JSON payload.
_LOG_RECORD_ATTRS = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
    "message",
    "asctime",
    "taskName",
}


class JsonFormatter(logging.Formatter):
    """Emit one JSON object per log record.

    Base fields: ``ts``, ``level``, ``logger``, ``message``. Extras added
    by ``extra={...}`` (including ``request_id`` from ``RequestIdFilter``)
    are merged into the same object.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in _LOG_RECORD_ATTRS or key.startswith("_"):
                continue
            payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


# Text formatter


class TextFormatter(logging.Formatter):
    """Human-readable single-line format that always includes request_id."""

    def __init__(self) -> None:
        super().__init__(
            fmt=("%(asctime)s %(levelname)s [rid=%(request_id)s] %(name)s: %(message)s"),
            datefmt="%Y-%m-%dT%H:%M:%S",
        )

    def format(self, record: logging.LogRecord) -> str:
        # Defensive: if a record bypassed RequestIdFilter, populate the
        # field so the format string does not raise.
        if not hasattr(record, "request_id"):
            record.request_id = request_id_var.get()
        return super().format(record)


# Setup


def setup_logging(level: str | None = None, fmt: str | None = None) -> None:
    """Configure the root logger.

    Args:
        level: log level name (e.g. ``"INFO"``). Defaults to
            ``settings.log_level``.
        fmt: ``"json"`` or ``"text"``. Defaults to ``settings.log_format``.

    Safe to call multiple times: existing handlers on the root logger are
    removed before the new handler is installed.
    """
    resolved_level = (level or settings.log_level).upper()
    resolved_fmt = (fmt or settings.log_format).lower()

    root = logging.getLogger()
    root.setLevel(resolved_level)

    # Replace any existing handlers so a second call does not duplicate
    # output (important because app/main.py calls this at import time and
    # tests may call it again).
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(RequestIdFilter())

    if resolved_fmt == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(TextFormatter())

    root.addHandler(handler)

    # Our request middleware emits the canonical per-request log line,
    # so uvicorn's access log would be redundant.
    logging.getLogger("uvicorn.access").disabled = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger. Thin wrapper for symmetry with other modules."""
    return logging.getLogger(name)
