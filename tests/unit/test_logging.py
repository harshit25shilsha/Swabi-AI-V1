"""Tests for the merged logging setup (Phase 0 + GDv1)."""

from __future__ import annotations

import json
import logging
import sys
from io import StringIO

import pytest

from app.core.logging import (
    JsonFormatter,
    RequestIdFilter,
    TextFormatter,
    request_id_var,
    setup_logging,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _restore_root_logger():
    """Snapshot and restore root logger + uvicorn.access state per test."""
    root = logging.getLogger()
    original_level = root.level
    original_handlers = root.handlers[:]
    access = logging.getLogger("uvicorn.access")
    original_disabled = access.disabled

    yield

    root.setLevel(original_level)
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    for handler in original_handlers:
        root.addHandler(handler)
    access.disabled = original_disabled


def _make_record(msg: str = "hello", level: int = logging.INFO, **extra):
    record = logging.LogRecord(
        name="test.logger",
        level=level,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=(),
        exc_info=None,
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return record


# ---------------------------------------------------------------------------
# JsonFormatter
# ---------------------------------------------------------------------------


class TestJsonFormatter:
    def test_base_fields_present(self):
        out = json.loads(JsonFormatter().format(_make_record("hello")))
        assert out["level"] == "INFO"
        assert out["logger"] == "test.logger"
        assert out["message"] == "hello"
        assert "ts" in out

    def test_extra_fields_merged(self):
        out = json.loads(JsonFormatter().format(_make_record("hello", custom="x", count=3)))
        assert out["custom"] == "x"
        assert out["count"] == 3

    def test_request_id_set_by_filter(self):
        token = request_id_var.set("rid-abc")
        try:
            record = _make_record("hello")
            RequestIdFilter().filter(record)
            out = json.loads(JsonFormatter().format(record))
            assert out["request_id"] == "rid-abc"
        finally:
            request_id_var.reset(token)

    def test_default_request_id_when_no_context(self):
        record = _make_record("hello")
        RequestIdFilter().filter(record)
        out = json.loads(JsonFormatter().format(record))
        assert out["request_id"] == "-"

    def test_explicit_extra_overrides_context(self):
        token = request_id_var.set("from-context")
        try:
            record = _make_record("hello", request_id="from-extra")
            RequestIdFilter().filter(record)
            out = json.loads(JsonFormatter().format(record))
            assert out["request_id"] == "from-extra"
        finally:
            request_id_var.reset(token)

    def test_exception_serialized(self):
        try:
            raise RuntimeError("boom")
        except RuntimeError:
            record = logging.LogRecord(
                name="t",
                level=logging.ERROR,
                pathname=__file__,
                lineno=1,
                msg="oops",
                args=(),
                exc_info=sys.exc_info(),
            )
        out = json.loads(JsonFormatter().format(record))
        assert "exc_info" in out
        assert "boom" in out["exc_info"]


# ---------------------------------------------------------------------------
# TextFormatter
# ---------------------------------------------------------------------------


class TestTextFormatter:
    def test_includes_request_id_from_context(self):
        token = request_id_var.set("rid-xyz")
        try:
            record = _make_record("hello")
            RequestIdFilter().filter(record)
            out = TextFormatter().format(record)
            assert "rid=rid-xyz" in out
            assert "hello" in out
            assert "INFO" in out
        finally:
            request_id_var.reset(token)

    def test_default_request_id(self):
        record = _make_record("hello")
        RequestIdFilter().filter(record)
        out = TextFormatter().format(record)
        assert "rid=-" in out


# ---------------------------------------------------------------------------
# setup_logging
# ---------------------------------------------------------------------------


class TestSetupLogging:
    def test_sets_root_level_from_arg(self):
        setup_logging(level="DEBUG")
        assert logging.getLogger().level == logging.DEBUG

    def test_installs_json_formatter(self):
        setup_logging(level="INFO", fmt="json")
        handler = logging.getLogger().handlers[0]
        assert isinstance(handler.formatter, JsonFormatter)

    def test_installs_text_formatter(self):
        setup_logging(level="INFO", fmt="text")
        handler = logging.getLogger().handlers[0]
        assert isinstance(handler.formatter, TextFormatter)

    def test_idempotent(self):
        setup_logging(level="INFO", fmt="json")
        first = list(logging.getLogger().handlers)
        setup_logging(level="INFO", fmt="json")
        second = list(logging.getLogger().handlers)
        assert len(second) == len(first) == 1

    def test_uvicorn_access_disabled(self):
        logging.getLogger("uvicorn.access").disabled = False
        setup_logging()
        assert logging.getLogger("uvicorn.access").disabled is True

    def test_request_id_filter_installed(self):
        setup_logging()
        handler = logging.getLogger().handlers[0]
        assert any(isinstance(f, RequestIdFilter) for f in handler.filters)

    def test_end_to_end_json_output(self):
        """Install setup_logging, emit through a test logger, verify JSON."""
        setup_logging(level="INFO", fmt="json")

        stream = StringIO()
        # Swap the stream on the installed handler so we can inspect output.
        handler = logging.getLogger().handlers[0]
        original_stream = handler.stream
        handler.stream = stream
        try:
            token = request_id_var.set("rid-e2e")
            try:
                logging.getLogger("e2e").info("hi from test")
            finally:
                request_id_var.reset(token)
        finally:
            handler.stream = original_stream

        line = stream.getvalue().strip().splitlines()[-1]
        parsed = json.loads(line)
        assert parsed["message"] == "hi from test"
        assert parsed["request_id"] == "rid-e2e"
        assert parsed["logger"] == "e2e"

    def test_end_to_end_text_output(self):
        setup_logging(level="INFO", fmt="text")

        stream = StringIO()
        handler = logging.getLogger().handlers[0]
        original_stream = handler.stream
        handler.stream = stream
        try:
            token = request_id_var.set("rid-text")
            try:
                logging.getLogger("e2e").info("hi text")
            finally:
                request_id_var.reset(token)
        finally:
            handler.stream = original_stream

        line = stream.getvalue().strip().splitlines()[-1]
        assert "rid=rid-text" in line
        assert "hi text" in line
        assert "INFO" in line
