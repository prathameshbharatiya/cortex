"""
cortex.observability.tracing
============================
OpenTelemetry distributed tracing for Cortex.

Every certification call becomes a root span with child spans for each
validator and the memory retrieval. Spans carry the full decision context
as attributes — state, confidence, failure modes, latencies.

If opentelemetry-sdk is not installed, all operations are no-ops.
Cortex works without it. Zero coupling.

Span inventory
--------------
  cortex.certify                         (root span per certify() call)
    cortex.validator.sentinel            (child)
    cortex.validator.physicore           (child)
    cortex.validator.memory              (child)
  cortex.context.assemble                (root span per assemble() call)
    cortex.memory.retrieve               (child)
  cortex.experience.record               (root span)

Span attributes (on cortex.certify):
  cortex.state              EXECUTE | EXECUTE_WITH_CONSTRAINTS | ...
  cortex.confidence         0.87
  cortex.latency_ms         2.3
  cortex.platform_id        arm_east_01
  cortex.deployment_id      prod
  cortex.action_id          uuid
  cortex.trace_id           uuid
  cortex.failure_modes      ["speed_limit_exceeded"]

Usage
-----
    from cortex.observability.tracing import get_tracer, span

    tracer = get_tracer()

    with span("cortex.certify", action_id=action.action_id) as s:
        cert = gate.certify(action, ctx)
        s.set_attribute("cortex.state", cert.state.value)
        s.set_attribute("cortex.confidence", cert.confidence.success_probability)

Configure exporter at startup:
    from cortex.observability.tracing import configure_tracing
    configure_tracing(
        exporter="otlp",             # "otlp" | "jaeger" | "console" | "none"
        endpoint="http://otel-collector:4317",
        service_name="cortex",
        platform_id="arm_01",
    )
"""

from __future__ import annotations

import contextlib
from contextlib import contextmanager
from typing import Any, Generator

# ── Optional opentelemetry import ────────────────────────────────────────────

try:
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME
    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False
    otel_trace = None  # type: ignore[assignment]


# ── No-op span for when otel is not available ─────────────────────────────────

class _NoOpSpan:
    """Returned by span() when otel is not installed. All methods are no-ops."""
    def set_attribute(self, key: str, value: Any) -> None: ...
    def set_status(self, *args: Any, **kwargs: Any) -> None: ...
    def record_exception(self, exc: Exception, **kwargs: Any) -> None: ...
    def add_event(self, name: str, attributes: dict | None = None) -> None: ...
    def end(self) -> None: ...
    def __enter__(self) -> "_NoOpSpan": return self
    def __exit__(self, *args: Any) -> None: ...


class _NoOpTracer:
    """Returned by get_tracer() when otel is not installed."""
    def start_span(self, *args: Any, **kwargs: Any) -> _NoOpSpan:
        return _NoOpSpan()

    @contextmanager
    def start_as_current_span(self, name: str, **kwargs: Any) -> Generator[_NoOpSpan, None, None]:
        yield _NoOpSpan()


# ── Global tracer ─────────────────────────────────────────────────────────────

_tracer: Any = None
_configured   = False


def configure_tracing(
    exporter:     str = "none",
    endpoint:     str = "http://localhost:4317",
    service_name: str = "cortex",
    platform_id:  str = "",
    deployment_id: str = "",
    sample_rate:  float = 1.0,
) -> None:
    """
    Initialise the OpenTelemetry tracer.

    Call once at server startup before any certifications run.

    Parameters
    ----------
    exporter:
        "otlp"    — send to an OTLP collector (Jaeger, Tempo, Datadog, etc.)
        "console" — print spans to stdout (development)
        "none"    — disable tracing (default)
    endpoint:
        OTLP collector endpoint. Used when exporter="otlp".
    service_name:
        Shown in trace UIs as the service name.
    platform_id:
        Stamped as a resource attribute on all spans.
    sample_rate:
        Fraction of traces to sample. 1.0 = 100%, 0.1 = 10%.
    """
    global _tracer, _configured

    if not _OTEL_AVAILABLE:
        return

    resource = Resource.create({
        SERVICE_NAME:               service_name,
        "cortex.platform_id":       platform_id,
        "cortex.deployment_id":     deployment_id,
    })

    provider = TracerProvider(resource=resource)

    if exporter == "console":
        provider.add_span_processor(
            BatchSpanProcessor(ConsoleSpanExporter())
        )
    elif exporter == "otlp":
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            otlp_exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
            provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
        except ImportError:
            import logging
            logging.getLogger("cortex.tracing").warning(
                "opentelemetry-exporter-otlp-proto-grpc not installed. "
                "pip install 'cortex-gate[tracing]'"
            )

    otel_trace.set_tracer_provider(provider)
    _tracer    = otel_trace.get_tracer(service_name)
    _configured = True


def get_tracer() -> Any:
    """
    Return the configured tracer.
    Returns a no-op tracer if otel is not configured.
    """
    if not _OTEL_AVAILABLE or not _configured:
        return _NoOpTracer()
    return _tracer


# ── Convenience context manager ───────────────────────────────────────────────

@contextmanager
def span(
    name: str,
    **attributes: Any,
) -> Generator[Any, None, None]:
    """
    Start a span as a context manager, with optional initial attributes.

    Example:
        with span("cortex.certify", action_id=action.action_id) as s:
            cert = gate.certify(action, ctx)
            s.set_attribute("cortex.state", cert.state.value)
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as s:
        for k, v in attributes.items():
            try:
                s.set_attribute(f"cortex.{k}", str(v) if not isinstance(v, (bool, int, float, str)) else v)
            except Exception:
                pass
        try:
            yield s
        except Exception as exc:
            try:
                s.record_exception(exc)
                if _OTEL_AVAILABLE:
                    from opentelemetry.trace import StatusCode
                    s.set_status(StatusCode.ERROR, str(exc))
            except Exception:
                pass
            raise
