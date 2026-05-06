"""
cortex.observability
====================
Structured logging, Prometheus metrics, and OpenTelemetry tracing.

    from cortex.observability import setup_observability
    setup_observability(cfg)       # call once at server startup

Individual components:
    from cortex.observability.logging import get_logger, bind_request
    from cortex.observability.metrics import get_metrics, configure_metrics
    from cortex.observability.tracing import configure_tracing, span
"""

from __future__ import annotations

from cortex.observability.logging import (
    configure_logging,
    get_logger,
    bind_request,
)
from cortex.observability.metrics import (
    configure_metrics,
    get_metrics,
    start_metrics_server,
    CortexMetrics,
)
from cortex.observability.tracing import (
    configure_tracing,
    get_tracer,
    span,
)

__all__ = [
    # Logging
    "configure_logging",
    "get_logger",
    "bind_request",
    # Metrics
    "configure_metrics",
    "get_metrics",
    "start_metrics_server",
    "CortexMetrics",
    # Tracing
    "configure_tracing",
    "get_tracer",
    "span",
    # Convenience
    "setup_observability",
]


def setup_observability(cfg: "CortexSettings") -> None:  # type: ignore[name-defined]
    """
    Initialise logging, metrics, and tracing from a CortexSettings object.

    Call once at server startup, before the first certification.

    Example:
        from cortex.config import get_settings
        from cortex.observability import setup_observability
        setup_observability(get_settings())
    """
    # Structured logging
    configure_logging(
        level=cfg.server.log_level,
        fmt=cfg.server.log_format,
        platform_id=cfg.server.platform_id,
        deployment_id=cfg.server.deployment_id,
    )

    # Prometheus metrics
    configure_metrics(
        platform_id=cfg.server.platform_id,
        deployment_id=cfg.server.deployment_id,
    )
    if cfg.metrics.enabled:
        start_metrics_server(
            port=cfg.metrics.port,
            addr=cfg.metrics.host,
        )

    # OpenTelemetry tracing
    if cfg.tracing.exporter != "none":
        configure_tracing(
            exporter=cfg.tracing.exporter,
            endpoint=cfg.tracing.endpoint,
            service_name=cfg.tracing.service_name or "cortex",
            platform_id=cfg.server.platform_id,
            deployment_id=cfg.server.deployment_id,
            sample_rate=cfg.tracing.sample_rate,
        )
