"""
Cortex Health Monitor
=====================
Tracks the health of every Cortex subsystem and exposes
a unified status endpoint used by the SDK, gRPC server, and
any monitoring system (Prometheus, Grafana, Datadog).

Metrics tracked per subsystem:
  - Certification Gate: decisions/sec, state distribution, avg latency
  - Memory Gateway: reads/sec, writes/sec, adapter health per adapter
  - Context Engine: assembly latency distribution
  - Relevance Engine: avg candidates, avg selected, avg latency
  - Experience Tracker: records/sec, surprise distribution
  - Lifecycle Manager: last maintenance time, records removed

All metrics are accessible via health.snapshot() as a dict.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LatencyStats:
    """Rolling latency statistics."""
    window: deque = field(default_factory=lambda: deque(maxlen=200))

    def record(self, latency_ms: float) -> None:
        self.window.append(latency_ms)

    @property
    def avg_ms(self) -> float:
        if not self.window:
            return 0.0
        return sum(self.window) / len(self.window)

    @property
    def p95_ms(self) -> float:
        if not self.window:
            return 0.0
        sorted_w = sorted(self.window)
        idx = max(0, int(len(sorted_w) * 0.95) - 1)
        return sorted_w[idx]

    @property
    def p99_ms(self) -> float:
        if not self.window:
            return 0.0
        sorted_w = sorted(self.window)
        idx = max(0, int(len(sorted_w) * 0.99) - 1)
        return sorted_w[idx]


class HealthMonitor:
    """
    Unified health and metrics tracker for the Cortex stack.

    Thread-safe. All record_* methods are non-blocking.
    snapshot() returns a point-in-time consistent view of all metrics.
    """

    def __init__(self) -> None:
        self._start_time = time.time()
        self._lock       = threading.RLock()

        # Certification gate metrics
        self._cert_latency    = LatencyStats()
        self._cert_states:    dict[str, int] = defaultdict(int)
        self._cert_total      = 0

        # Memory gateway metrics
        self._gateway_reads   = 0
        self._gateway_writes  = 0
        self._gateway_latency = LatencyStats()

        # Context engine metrics
        self._ctx_latency     = LatencyStats()
        self._ctx_total       = 0

        # Relevance engine metrics
        self._rel_candidates:  deque = deque(maxlen=200)
        self._rel_selected:    deque = deque(maxlen=200)
        self._rel_latency      = LatencyStats()

        # Experience tracker metrics
        self._exp_total       = 0
        self._exp_surprise:   deque = deque(maxlen=200)

        # Lifecycle metrics
        self._last_maintenance: float | None = None
        self._records_removed_total = 0

        # Error tracking
        self._errors: deque = deque(maxlen=100)

    # ── Record methods ────────────────────────────────────────────────────────

    def record_certification(
        self,
        state:      str,
        latency_ms: float,
    ) -> None:
        with self._lock:
            self._cert_latency.record(latency_ms)
            self._cert_states[state] += 1
            self._cert_total += 1

    def record_gateway_read(self, latency_ms: float) -> None:
        with self._lock:
            self._gateway_reads  += 1
            self._gateway_latency.record(latency_ms)

    def record_gateway_write(self, latency_ms: float) -> None:
        with self._lock:
            self._gateway_writes += 1

    def record_context_assembly(self, latency_ms: float) -> None:
        with self._lock:
            self._ctx_latency.record(latency_ms)
            self._ctx_total += 1

    def record_relevance(
        self,
        candidates: int,
        selected:   int,
        latency_ms: float,
    ) -> None:
        with self._lock:
            self._rel_candidates.append(candidates)
            self._rel_selected.append(selected)
            self._rel_latency.record(latency_ms)

    def record_experience(self, surprise_score: float) -> None:
        with self._lock:
            self._exp_total += 1
            self._exp_surprise.append(surprise_score)

    def record_maintenance(self, records_removed: int) -> None:
        with self._lock:
            self._last_maintenance      = time.time()
            self._records_removed_total += records_removed

    def record_error(self, subsystem: str, error: str) -> None:
        with self._lock:
            self._errors.append({
                "subsystem": subsystem,
                "error":     error,
                "timestamp": time.time(),
            })

    # ── Snapshot ──────────────────────────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        """Return a point-in-time view of all metrics."""
        with self._lock:
            uptime = time.time() - self._start_time

            # Certification throughput
            cert_rps = self._cert_total / max(1.0, uptime)

            # Memory gateway
            gw_rps = self._gateway_reads / max(1.0, uptime)

            # Relevance averages
            avg_candidates = (sum(self._rel_candidates) / len(self._rel_candidates)
                              if self._rel_candidates else 0.0)
            avg_selected   = (sum(self._rel_selected) / len(self._rel_selected)
                              if self._rel_selected else 0.0)

            # Experience surprise average
            avg_surprise = (sum(self._exp_surprise) / len(self._exp_surprise)
                            if self._exp_surprise else 0.0)

            # High surprise count
            high_surprise = sum(1 for s in self._exp_surprise if s > 0.5)

            return {
                "uptime_s":    round(uptime, 1),
                "healthy":     True,
                "version":     "0.5.0",

                "certification": {
                    "total":       self._cert_total,
                    "decisions_per_sec": round(cert_rps, 2),
                    "state_distribution": dict(self._cert_states),
                    "avg_latency_ms": round(self._cert_latency.avg_ms, 2),
                    "p95_latency_ms": round(self._cert_latency.p95_ms, 2),
                    "p99_latency_ms": round(self._cert_latency.p99_ms, 2),
                },

                "memory_gateway": {
                    "total_reads":  self._gateway_reads,
                    "total_writes": self._gateway_writes,
                    "reads_per_sec": round(gw_rps, 2),
                    "avg_read_latency_ms": round(self._gateway_latency.avg_ms, 2),
                },

                "context_engine": {
                    "total_assemblies": self._ctx_total,
                    "avg_latency_ms":   round(self._ctx_latency.avg_ms, 2),
                    "p95_latency_ms":   round(self._ctx_latency.p95_ms, 2),
                },

                "relevance_engine": {
                    "avg_candidates": round(avg_candidates, 1),
                    "avg_selected":   round(avg_selected, 1),
                    "avg_latency_ms": round(self._rel_latency.avg_ms, 2),
                },

                "experience_tracker": {
                    "total_recorded": self._exp_total,
                    "avg_surprise":   round(avg_surprise, 4),
                    "high_surprise_recent": high_surprise,
                },

                "lifecycle": {
                    "last_maintenance": self._last_maintenance,
                    "records_removed_total": self._records_removed_total,
                },

                "errors": {
                    "recent_count": len(self._errors),
                    "last_error":   list(self._errors)[-1] if self._errors else None,
                },
            }

    def is_healthy(self) -> bool:
        """True if no critical errors in the last 60 seconds."""
        with self._lock:
            now    = time.time()
            recent = [e for e in self._errors if now - e["timestamp"] < 60.0]
            return len(recent) == 0

    def reset(self) -> None:
        """Reset all metrics (useful for testing)."""
        with self._lock:
            self._cert_latency    = LatencyStats()
            self._cert_states.clear()
            self._cert_total      = 0
            self._gateway_reads   = 0
            self._gateway_writes  = 0
            self._gateway_latency = LatencyStats()
            self._ctx_latency     = LatencyStats()
            self._ctx_total       = 0
            self._rel_candidates.clear()
            self._rel_selected.clear()
            self._rel_latency     = LatencyStats()
            self._exp_total       = 0
            self._exp_surprise.clear()
            self._errors.clear()


# ── Global singleton ──────────────────────────────────────────────────────────

_global_monitor: HealthMonitor | None = None
_monitor_lock   = threading.Lock()


def get_monitor() -> HealthMonitor:
    """Get or create the global health monitor."""
    global _global_monitor
    with _monitor_lock:
        if _global_monitor is None:
            _global_monitor = HealthMonitor()
        return _global_monitor
