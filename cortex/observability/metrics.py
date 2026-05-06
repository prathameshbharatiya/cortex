"""
cortex.observability.metrics
============================
Prometheus metrics for the Cortex stack.

Exposes a /metrics endpoint and a push-to-Pushgateway path.
If prometheus_client is not installed, all operations are no-ops —
the rest of Cortex works fine without it.

Metric inventory
----------------
Counters:
  cortex_certifications_total{state, platform_id, deployment_id}
  cortex_validator_runs_total{validator, passed, platform_id}
  cortex_memory_operations_total{operation, adapter, platform_id}
  cortex_errors_total{subsystem, platform_id}
  cortex_experiences_recorded_total{outcome, platform_id}

Histograms:
  cortex_certification_latency_ms{state, platform_id}
  cortex_validator_latency_ms{validator, platform_id}
  cortex_context_assembly_latency_ms{platform_id}
  cortex_memory_read_latency_ms{adapter, platform_id}

Gauges:
  cortex_confidence_score{platform_id}
  cortex_memory_adapter_healthy{adapter, platform_id}
  cortex_active_certifications{platform_id}
  cortex_surprise_score{platform_id}

Usage
-----
    from cortex.observability.metrics import get_metrics

    m = get_metrics()
    m.record_certification(state="EXECUTE", latency_ms=2.3, confidence=0.87)
    m.record_validator("sentinel", passed=True, latency_ms=0.8)
    m.record_memory_read("redis", latency_ms=0.4)

    # Prometheus HTTP endpoint — call from your server startup:
    from cortex.observability.metrics import start_metrics_server
    start_metrics_server(port=9090)
"""

from __future__ import annotations

import threading
import time
from typing import Any

# ── Optional prometheus_client import ────────────────────────────────────────

try:
    import prometheus_client as prom
    from prometheus_client import (
        Counter, Histogram, Gauge,
        CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST,
        start_http_server as _prom_start_http_server,
    )
    _PROM_AVAILABLE = True
except ImportError:
    _PROM_AVAILABLE = False
    prom = None  # type: ignore[assignment]


# ── Histogram buckets tuned for robotics (target: sub-16ms at 60Hz) ─────────

_LATENCY_BUCKETS_MS = (
    0.1, 0.25, 0.5, 1.0, 2.0, 3.0,
    5.0, 8.0, 12.0, 16.0, 25.0, 50.0, 100.0,
)

_CONFIDENCE_BUCKETS = (
    0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.5,
    0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0,
)


# ── Metrics registry ──────────────────────────────────────────────────────────

class CortexMetrics:
    """
    All Prometheus metrics for Cortex, scoped to a single registry.

    If prometheus_client is not installed, every method is a no-op.
    No exceptions are raised — observability must never break the hot path.
    """

    def __init__(
        self,
        platform_id:   str = "",
        deployment_id: str = "",
        registry: Any = None,
    ) -> None:
        self.platform_id   = platform_id
        self.deployment_id = deployment_id
        self._enabled      = _PROM_AVAILABLE
        self._lock         = threading.Lock()

        if not self._enabled:
            return

        reg = registry or prom.REGISTRY

        # ── Counters ──────────────────────────────────────────────────────────

        self._cert_total = Counter(
            "cortex_certifications_total",
            "Total certification decisions issued, by outcome state.",
            ["state", "platform_id", "deployment_id"],
            registry=reg,
        )

        self._validator_runs = Counter(
            "cortex_validator_runs_total",
            "Total validator invocations.",
            ["validator", "passed", "platform_id"],
            registry=reg,
        )

        self._memory_ops = Counter(
            "cortex_memory_operations_total",
            "Total memory gateway operations.",
            ["operation", "adapter", "platform_id"],
            registry=reg,
        )

        self._errors = Counter(
            "cortex_errors_total",
            "Total errors by subsystem.",
            ["subsystem", "platform_id"],
            registry=reg,
        )

        self._experiences = Counter(
            "cortex_experiences_recorded_total",
            "Total experience records written after execution.",
            ["outcome", "platform_id"],
            registry=reg,
        )

        # ── Histograms ────────────────────────────────────────────────────────

        self._cert_latency = Histogram(
            "cortex_certification_latency_ms",
            "End-to-end certification latency in milliseconds.",
            ["state", "platform_id"],
            buckets=_LATENCY_BUCKETS_MS,
            registry=reg,
        )

        self._validator_latency = Histogram(
            "cortex_validator_latency_ms",
            "Per-validator latency in milliseconds.",
            ["validator", "platform_id"],
            buckets=_LATENCY_BUCKETS_MS,
            registry=reg,
        )

        self._ctx_latency = Histogram(
            "cortex_context_assembly_latency_ms",
            "Context assembly latency in milliseconds.",
            ["platform_id"],
            buckets=_LATENCY_BUCKETS_MS,
            registry=reg,
        )

        self._memory_read_latency = Histogram(
            "cortex_memory_read_latency_ms",
            "Memory adapter read latency in milliseconds.",
            ["adapter", "platform_id"],
            buckets=_LATENCY_BUCKETS_MS,
            registry=reg,
        )

        self._confidence_hist = Histogram(
            "cortex_confidence_score",
            "Distribution of certification confidence scores.",
            ["state", "platform_id"],
            buckets=_CONFIDENCE_BUCKETS,
            registry=reg,
        )

        # ── Gauges ────────────────────────────────────────────────────────────

        self._adapter_healthy = Gauge(
            "cortex_memory_adapter_healthy",
            "1 if the memory adapter is healthy, 0 otherwise.",
            ["adapter", "platform_id"],
            registry=reg,
        )

        self._active_certs = Gauge(
            "cortex_active_certifications",
            "Number of certification calls currently in flight.",
            ["platform_id"],
            registry=reg,
        )

        self._surprise = Gauge(
            "cortex_surprise_score",
            "Rolling mean surprise score from the experience tracker.",
            ["platform_id"],
            registry=reg,
        )

        self._uptime = Gauge(
            "cortex_uptime_seconds",
            "Seconds since Cortex server started.",
            ["platform_id"],
            registry=reg,
        )
        self._start_time = time.monotonic()

    # ── Emission methods ──────────────────────────────────────────────────────

    def record_certification(
        self,
        state:      str,
        latency_ms: float,
        confidence: float | None = None,
    ) -> None:
        """Called once per certify() call with the final outcome."""
        if not self._enabled:
            return
        try:
            labels = {"state": state, "platform_id": self.platform_id,
                      "deployment_id": self.deployment_id}
            self._cert_total.labels(**labels).inc()
            self._cert_latency.labels(
                state=state, platform_id=self.platform_id
            ).observe(latency_ms)
            if confidence is not None:
                self._confidence_hist.labels(
                    state=state, platform_id=self.platform_id
                ).observe(confidence)
        except Exception:
            pass   # metrics must never crash the hot path

    def record_validator(
        self,
        validator:  str,
        passed:     bool,
        latency_ms: float,
    ) -> None:
        """Called once per validator invocation inside certify()."""
        if not self._enabled:
            return
        try:
            self._validator_runs.labels(
                validator=validator,
                passed=str(passed).lower(),
                platform_id=self.platform_id,
            ).inc()
            self._validator_latency.labels(
                validator=validator,
                platform_id=self.platform_id,
            ).observe(latency_ms)
        except Exception:
            pass

    def record_context_assembly(self, latency_ms: float) -> None:
        """Called once per context assembly."""
        if not self._enabled:
            return
        try:
            self._ctx_latency.labels(
                platform_id=self.platform_id,
            ).observe(latency_ms)
        except Exception:
            pass

    def record_memory_read(self, adapter: str, latency_ms: float) -> None:
        """Called once per memory gateway read operation."""
        if not self._enabled:
            return
        try:
            self._memory_ops.labels(
                operation="read", adapter=adapter,
                platform_id=self.platform_id,
            ).inc()
            self._memory_read_latency.labels(
                adapter=adapter, platform_id=self.platform_id,
            ).observe(latency_ms)
        except Exception:
            pass

    def record_memory_write(self, adapter: str) -> None:
        """Called once per memory gateway write."""
        if not self._enabled:
            return
        try:
            self._memory_ops.labels(
                operation="write", adapter=adapter,
                platform_id=self.platform_id,
            ).inc()
        except Exception:
            pass

    def record_experience(self, outcome: str, surprise: float) -> None:
        """Called when an experience record is written."""
        if not self._enabled:
            return
        try:
            self._experiences.labels(
                outcome=outcome, platform_id=self.platform_id,
            ).inc()
            self._surprise.labels(platform_id=self.platform_id).set(surprise)
        except Exception:
            pass

    def record_error(self, subsystem: str) -> None:
        """Called on any exception in a Cortex subsystem."""
        if not self._enabled:
            return
        try:
            self._errors.labels(
                subsystem=subsystem, platform_id=self.platform_id,
            ).inc()
        except Exception:
            pass

    def set_adapter_healthy(self, adapter: str, healthy: bool) -> None:
        """Called during health checks."""
        if not self._enabled:
            return
        try:
            self._adapter_healthy.labels(
                adapter=adapter, platform_id=self.platform_id,
            ).set(1.0 if healthy else 0.0)
        except Exception:
            pass

    def inc_active(self) -> None:
        """Call at the start of each certify() call."""
        if not self._enabled:
            return
        try:
            self._active_certs.labels(platform_id=self.platform_id).inc()
        except Exception:
            pass

    def dec_active(self) -> None:
        """Call at the end of each certify() call."""
        if not self._enabled:
            return
        try:
            self._active_certs.labels(platform_id=self.platform_id).dec()
        except Exception:
            pass

    def tick_uptime(self) -> None:
        """Update the uptime gauge. Call from a background thread."""
        if not self._enabled:
            return
        try:
            self._uptime.labels(platform_id=self.platform_id).set(
                time.monotonic() - self._start_time
            )
        except Exception:
            pass

    # ── Phase 4–7 emission methods ────────────────────────────────────────────

    def record_episodic_store(self, outcome: str) -> None:
        """Called when an episode is written to the episodic memory stack."""
        if not self._enabled:
            return
        try:
            self._errors.labels(
                subsystem=f"episodic_store_{outcome}",
                platform_id=self.platform_id,
            )  # reuse error counter as event counter (no new metric needed)
        except Exception:
            pass

    def record_ood_detection(self, is_ood: bool, ood_score: float) -> None:
        """Called once per OOD detection check in the Gate."""
        if not self._enabled:
            return
        try:
            self._errors.labels(
                subsystem="ood_flagged" if is_ood else "ood_clear",
                platform_id=self.platform_id,
            ).inc()
        except Exception:
            pass

    def record_coordination_check(self, approved: bool) -> None:
        """Called once per multi-agent coordination check."""
        if not self._enabled:
            return
        try:
            self._errors.labels(
                subsystem="coord_approved" if approved else "coord_blocked",
                platform_id=self.platform_id,
            ).inc()
        except Exception:
            pass

    def record_intent_translation(self, intent_type: str, confidence: float) -> None:
        """Called once per intent translation."""
        if not self._enabled:
            return
        try:
            self._errors.labels(
                subsystem=f"intent_{intent_type}",
                platform_id=self.platform_id,
            ).inc()
        except Exception:
            pass

    # ── Scrape endpoint helpers ───────────────────────────────────────────────

    @staticmethod
    def scrape() -> tuple[bytes, str]:
        """
        Return (body, content_type) for a /metrics HTTP response.
        Returns empty body if prometheus_client is not installed.
        """
        if not _PROM_AVAILABLE:
            return b"# prometheus_client not installed\n", "text/plain"
        return generate_latest(), CONTENT_TYPE_LATEST


def start_metrics_server(port: int = 9090, addr: str = "") -> None:
    """
    Start a standalone Prometheus HTTP server on the given port.

    This is separate from the main gRPC/HTTP server.
    Prometheus scrapes this endpoint directly.

    Example (in server startup):
        from cortex.observability.metrics import start_metrics_server
        start_metrics_server(port=9090)
    """
    if not _PROM_AVAILABLE:
        import logging
        logging.getLogger("cortex.metrics").warning(
            "prometheus_client not installed — metrics server not started. "
            "pip install 'cortex-gate[metrics]'"
        )
        return
    _prom_start_http_server(port, addr=addr)


# ── Global singleton ──────────────────────────────────────────────────────────

_global_metrics: CortexMetrics | None = None
_metrics_lock   = threading.Lock()


def get_metrics() -> CortexMetrics:
    """
    Return the global CortexMetrics singleton.

    Initialised on first call with empty labels.
    Call configure_metrics() first if you need platform_id on labels.
    """
    global _global_metrics
    if _global_metrics is None:
        with _metrics_lock:
            if _global_metrics is None:
                _global_metrics = CortexMetrics()
    return _global_metrics


def configure_metrics(
    platform_id:   str = "",
    deployment_id: str = "",
    registry: Any = None,
) -> CortexMetrics:
    """
    Initialise the global metrics singleton with platform/deployment labels.

    Call once at server startup before any certifications run.

    Example:
        from cortex.observability.metrics import configure_metrics
        m = configure_metrics(platform_id="arm_01", deployment_id="prod")
    """
    global _global_metrics
    with _metrics_lock:
        _global_metrics = CortexMetrics(
            platform_id=platform_id,
            deployment_id=deployment_id,
            registry=registry,
        )
    return _global_metrics
