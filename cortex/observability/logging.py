"""
cortex.observability.logging
============================
Structured logging for the Cortex stack.

Every log record emitted from Cortex carries these fields:
  - timestamp    (ISO-8601)
  - level
  - logger       (cortex.gate, cortex.memory, etc.)
  - platform_id
  - deployment_id
  - trace_id     (populated per-request via context var)
  - message
  - ...          (any extra fields passed at call site)

Two formats:
  - text:  human-readable for development
  - json:  machine-parseable for Datadog / Loki / Cloud Logging

Usage
-----
    from cortex.observability.logging import get_logger, bind_request

    log = get_logger(__name__)
    log.info("action certified", state="EXECUTE", latency_ms=3.2, confidence=0.87)

    # Bind request-scoped fields for the duration of a certification call
    with bind_request(trace_id="abc-123", action_id="act-456"):
        log.info("starting certification")   # automatically includes trace_id
"""

from __future__ import annotations

import datetime
import json
import logging
import sys
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Generator

# ── Per-request context vars ──────────────────────────────────────────────────
_ctx_trace_id:      ContextVar[str] = ContextVar("trace_id",      default="")
_ctx_action_id:     ContextVar[str] = ContextVar("action_id",     default="")
_ctx_platform_id:   ContextVar[str] = ContextVar("platform_id",   default="")
_ctx_deployment_id: ContextVar[str] = ContextVar("deployment_id", default="")


@contextmanager
def bind_request(
    trace_id:      str = "",
    action_id:     str = "",
    platform_id:   str = "",
    deployment_id: str = "",
) -> Generator[None, None, None]:
    """
    Bind request-scoped fields to all log records emitted in this context.

    Thread-safe via Python ContextVar.

    Example:
        with bind_request(trace_id=cert.trace.trace_id, action_id=action.action_id):
            log.info("certifying action")   # trace_id + action_id auto-included
    """
    t1 = _ctx_trace_id.set(trace_id)
    t2 = _ctx_action_id.set(action_id)
    t3 = _ctx_platform_id.set(platform_id)
    t4 = _ctx_deployment_id.set(deployment_id)
    try:
        yield
    finally:
        _ctx_trace_id.reset(t1)
        _ctx_action_id.reset(t2)
        _ctx_platform_id.reset(t3)
        _ctx_deployment_id.reset(t4)


# ── Formatters ────────────────────────────────────────────────────────────────

class _JsonFormatter(logging.Formatter):
    """
    Emits one JSON object per line.
    Compatible with Datadog, Loki, Cloud Logging, Splunk.
    """

    def format(self, record: logging.LogRecord) -> str:
        obj: dict[str, Any] = {
            "timestamp":     datetime.datetime.fromtimestamp(record.created, tz=datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            "level":         record.levelname,
            "logger":        record.name,
            "message":       record.getMessage(),
            "platform_id":   _ctx_platform_id.get(),
            "deployment_id": _ctx_deployment_id.get(),
        }

        # Request-scoped fields — only include if set
        trace_id = _ctx_trace_id.get()
        if trace_id:
            obj["trace_id"] = trace_id

        action_id = _ctx_action_id.get()
        if action_id:
            obj["action_id"] = action_id

        # Extra keyword fields passed as LogRecord.__dict__ extras
        for key, val in record.__dict__.items():
            if key.startswith("_cx_"):
                obj[key[4:]] = val   # strip the _cx_ prefix

        # Exception info
        if record.exc_info:
            obj["exception"] = self.formatException(record.exc_info)

        return json.dumps(obj, default=str)


class _TextFormatter(logging.Formatter):
    """
    Human-readable text format with context fields.
    """
    GREY    = "\033[90m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    RED     = "\033[91m"
    BOLD    = "\033[1m"
    RESET   = "\033[0m"

    LEVEL_COLORS = {
        "DEBUG":    "\033[90m",
        "INFO":     "\033[92m",
        "WARNING":  "\033[93m",
        "ERROR":    "\033[91m",
        "CRITICAL": "\033[91m\033[1m",
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.LEVEL_COLORS.get(record.levelname, "")
        ts    = datetime.datetime.fromtimestamp(record.created, tz=datetime.timezone.utc).strftime("%H:%M:%S.%f")[:-3]
        level = f"{color}{record.levelname:<8}{self.RESET}"
        name  = f"{self.GREY}{record.name}{self.RESET}"

        trace_id = _ctx_trace_id.get()
        trace    = f" {self.GREY}[{trace_id[:8]}]{self.RESET}" if trace_id else ""

        msg = record.getMessage()

        # Inline extra fields
        extras = []
        for key, val in record.__dict__.items():
            if key.startswith("_cx_"):
                extras.append(f"{key[4:]}={val!r}")
        extra_str = "  " + "  ".join(extras) if extras else ""

        line = f"{ts}  {level}  {name}{trace}  {msg}{extra_str}"

        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


# ── Logger adapter that supports keyword extras ───────────────────────────────

class CortexLogger(logging.LoggerAdapter):
    """
    Drop-in replacement for logging.Logger that supports structured extra fields.

    Usage:
        log = get_logger(__name__)
        log.info("certified", state="EXECUTE", latency_ms=2.3)
        log.warning("low confidence", confidence=0.41, threshold=0.50)
    """

    def process(
        self, msg: str, kwargs: dict
    ) -> tuple[str, dict]:
        extra = kwargs.pop("extra", {}) or {}
        # Pull any keyword args that aren't standard logging kwargs into extras
        standard = {"exc_info", "stack_info", "stacklevel"}
        for key in list(kwargs.keys()):
            if key not in standard:
                extra[f"_cx_{key}"] = kwargs.pop(key)
        kwargs["extra"] = extra
        return msg, kwargs


# ── Setup and get_logger ──────────────────────────────────────────────────────

_configured = False

def configure_logging(
    level:       str = "INFO",
    fmt:         str = "text",
    platform_id: str = "",
    deployment_id: str = "",
) -> None:
    """
    Configure the Cortex logging stack.

    Called once at server startup from the config factory.
    Idempotent — safe to call multiple times.

    Parameters
    ----------
    level:         log level (DEBUG / INFO / WARNING / ERROR / CRITICAL)
    fmt:           "text" or "json"
    platform_id:   stamped on every JSON record
    deployment_id: stamped on every JSON record
    """
    global _configured

    # Set the module-level context vars so all records carry these
    _ctx_platform_id.set(platform_id)
    _ctx_deployment_id.set(deployment_id)

    root = logging.getLogger("cortex")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    if root.handlers:
        root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        _JsonFormatter() if fmt.lower() == "json" else _TextFormatter()
    )
    root.addHandler(handler)
    root.propagate = False
    _configured = True


def get_logger(name: str) -> CortexLogger:
    """
    Get a structured Cortex logger.

    Example:
        log = get_logger(__name__)
        log.info("gate certified", state="EXECUTE", latency_ms=2.1)
    """
    if not _configured:
        configure_logging()
    return CortexLogger(logging.getLogger(name), {})
