"""
tests/unit/test_observability.py
==================================
Tests for structured logging, Prometheus metrics, and tracing.
All tests are pure — no network, no Prometheus server.
"""

from __future__ import annotations

import json
import io
import logging
import time
import threading
import pytest

from prometheus_client import CollectorRegistry

from cortex.observability.logging import (
    configure_logging, get_logger, bind_request, _JsonFormatter, _TextFormatter,
)
from cortex.observability.metrics import CortexMetrics
from cortex.observability.tracing import span, _NoOpSpan, _NoOpTracer


# ══════════════════════════════════════════════════════════════════════════════
# Structured logging
# ══════════════════════════════════════════════════════════════════════════════

class TestStructuredLogging:

    def _capture_json(self, name: str = "cortex.test") -> tuple:
        """Returns (logger, buffer) with a JSON handler attached."""
        buf = io.StringIO()
        h = logging.StreamHandler(buf)
        h.setFormatter(_JsonFormatter())
        log_obj = logging.getLogger(name)
        log_obj.addHandler(h)
        log_obj.setLevel(logging.DEBUG)
        return get_logger(name), buf, log_obj

    def _last_json(self, buf: io.StringIO) -> dict:
        buf.seek(0)
        lines = [l for l in buf.read().strip().splitlines() if l.strip()]
        return json.loads(lines[-1])

    def test_json_format_produces_valid_json(self):
        log, buf, raw = self._capture_json("cortex.json_test")
        log.info("hello world")
        obj = self._last_json(buf)
        assert "level" in obj
        assert "message" in obj
        assert "timestamp" in obj

    def test_json_level_field(self):
        log, buf, raw = self._capture_json("cortex.level_test")
        log.warning("warn msg")
        obj = self._last_json(buf)
        assert obj["level"] == "WARNING"

    def test_json_extra_fields(self):
        log, buf, raw = self._capture_json("cortex.extra_test")
        log.info("with extras", state="EXECUTE", latency_ms=2.3)
        obj = self._last_json(buf)
        assert obj["state"] == "EXECUTE"
        assert obj["latency_ms"] == 2.3

    def test_json_trace_id_from_context_var(self):
        log, buf, raw = self._capture_json("cortex.ctx_test")
        with bind_request(trace_id="trace-abc-123"):
            log.info("inside request")
        obj = self._last_json(buf)
        assert obj["trace_id"] == "trace-abc-123"

    def test_json_action_id_from_context_var(self):
        log, buf, raw = self._capture_json("cortex.act_test")
        with bind_request(action_id="act-xyz-789"):
            log.info("with action")
        obj = self._last_json(buf)
        assert obj["action_id"] == "act-xyz-789"

    def test_trace_id_not_present_outside_context(self):
        log, buf, raw = self._capture_json("cortex.notrace_test")
        log.info("outside context")
        obj = self._last_json(buf)
        # trace_id should not appear when not set
        assert obj.get("trace_id", "") == ""

    def test_bind_request_is_scoped(self):
        log, buf, raw = self._capture_json("cortex.scope_test")
        with bind_request(trace_id="inside"):
            log.info("inside")
        log.info("outside")
        buf.seek(0)
        lines = [l for l in buf.read().strip().splitlines() if l.strip()]
        obj_inside = json.loads(lines[-2])
        obj_outside = json.loads(lines[-1])
        assert obj_inside.get("trace_id") == "inside"
        assert obj_outside.get("trace_id", "") == ""

    def test_bind_request_thread_safe(self):
        """Different threads must not see each other's trace_id."""
        results = {}

        def worker(tid: str):
            log, buf, raw = self._capture_json(f"cortex.thread_{tid}")
            with bind_request(trace_id=tid):
                time.sleep(0.01)
                log.info("thread log")
            results[tid] = self._last_json(buf).get("trace_id")

        threads = [threading.Thread(target=worker, args=(f"t{i}",)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for tid, seen in results.items():
            assert seen == tid, f"Thread {tid} saw trace_id={seen!r}"

    def test_get_logger_returns_cortex_logger(self):
        log = get_logger("cortex.test")
        assert hasattr(log, "info")
        assert hasattr(log, "warning")
        assert hasattr(log, "error")


# ══════════════════════════════════════════════════════════════════════════════
# Prometheus metrics
# ══════════════════════════════════════════════════════════════════════════════

class TestPrometheusMetrics:

    def _make(self, platform: str = "test_arm") -> CortexMetrics:
        return CortexMetrics(
            platform_id=platform,
            deployment_id="test",
            registry=CollectorRegistry(),
        )

    def test_record_certification_no_exception(self):
        m = self._make()
        m.record_certification("EXECUTE", latency_ms=2.3, confidence=0.87)

    def test_record_all_states(self):
        m = self._make()
        for state in ["EXECUTE", "EXECUTE_WITH_CONSTRAINTS", "REPLAN_REQUIRED",
                      "SAFE_HALT", "HUMAN_OVERRIDE_REQUIRED"]:
            m.record_certification(state, latency_ms=1.0)

    def test_record_validator(self):
        m = self._make()
        m.record_validator("sentinel", passed=True, latency_ms=0.4)
        m.record_validator("physicore", passed=False, latency_ms=1.2)
        m.record_validator("memory", passed=True, latency_ms=0.3)

    def test_record_memory_ops(self):
        m = self._make()
        m.record_memory_read("redis", latency_ms=0.3)
        m.record_memory_write("redis")
        m.record_memory_read("in_memory", latency_ms=0.01)

    def test_record_experience(self):
        m = self._make()
        m.record_experience("success", surprise=0.12)
        m.record_experience("failure", surprise=0.88)

    def test_record_error(self):
        m = self._make()
        m.record_error("gate")
        m.record_error("memory")

    def test_set_adapter_healthy(self):
        m = self._make()
        m.set_adapter_healthy("redis", True)
        m.set_adapter_healthy("weaviate", False)

    def test_inc_dec_active(self):
        m = self._make()
        m.inc_active()
        m.inc_active()
        m.dec_active()

    def test_scrape_returns_bytes(self):
        from prometheus_client import REGISTRY
        body, ct = CortexMetrics.scrape()
        assert isinstance(body, bytes)
        assert "text/plain" in ct or "metrics" in ct

    def test_metrics_on_hot_path_never_raise(self):
        """Metrics emission must never crash certification logic."""
        m = self._make()
        for _ in range(100):
            m.inc_active()
            m.record_certification("EXECUTE", latency_ms=2.0, confidence=0.88)
            m.record_validator("sentinel", passed=True, latency_ms=0.5)
            m.record_validator("physicore", passed=True, latency_ms=0.8)
            m.record_validator("memory", passed=True, latency_ms=0.2)
            m.dec_active()

    def test_metrics_without_prometheus_installed(self, monkeypatch):
        """If prometheus_client is not installed, all methods are no-ops."""
        import cortex.observability.metrics as mod
        monkeypatch.setattr(mod, "_PROM_AVAILABLE", False)
        m = CortexMetrics(registry=CollectorRegistry())
        m._enabled = False
        m.record_certification("EXECUTE", 1.0)  # must not raise
        m.record_validator("sentinel", True, 0.5)
        m.record_error("gate")


# ══════════════════════════════════════════════════════════════════════════════
# OpenTelemetry tracing (no-op path)
# ══════════════════════════════════════════════════════════════════════════════

class TestTracingNoOp:

    def test_span_context_manager_no_exception(self):
        with span("cortex.test", action_id="act-001") as s:
            s.set_attribute("cortex.state", "EXECUTE")

    def test_noop_span_methods_do_nothing(self):
        s = _NoOpSpan()
        s.set_attribute("key", "value")
        s.set_status("ok")
        s.record_exception(ValueError("test"))
        s.add_event("event", {"k": "v"})
        s.end()

    def test_noop_tracer_returns_noop_span(self):
        t = _NoOpTracer()
        with t.start_as_current_span("test") as s:
            assert isinstance(s, _NoOpSpan)

    def test_span_re_raises_exceptions(self):
        with pytest.raises(ValueError):
            with span("cortex.error_test"):
                raise ValueError("test error")

    def test_nested_spans_no_exception(self):
        with span("cortex.outer") as outer:
            outer.set_attribute("cortex.level", "outer")
            with span("cortex.inner") as inner:
                inner.set_attribute("cortex.level", "inner")


# ══════════════════════════════════════════════════════════════════════════════
# End-to-end: certification emits observability
# ══════════════════════════════════════════════════════════════════════════════

class TestObservabilityEndToEnd:

    def test_certification_emits_prometheus_metrics(self):
        """Running certify() must increment prometheus counters."""
        from prometheus_client import CollectorRegistry, generate_latest
        from cortex.observability.metrics import configure_metrics
        import cortex
        from cortex.models.action import Action, ActionSpec, ActionType, ActionConstraints
        from cortex.models.context import RobotState

        reg = CollectorRegistry()
        configure_metrics(platform_id="obs_e2e", registry=reg)

        state = RobotState(
            ee_position=[0.3, 0.0, 0.5],
            ee_orientation=[1.0, 0.0, 0.0, 0.0],
            joint_positions=[0.0] * 7,
            joint_velocities=[0.0] * 7,
            joint_torques=[0.0] * 7,
        )
        ctx = cortex.build_context("test task", state, [])
        action = Action(
            spec=ActionSpec(
                action_type=ActionType.MOVE_EE,
                constraints=ActionConstraints(max_speed_ms=0.5, max_force_n=50.0),
            ),
            source="test", intent="test",
        )
        cortex.certify(action, ctx)

        body = generate_latest(reg).decode()
        assert "obs_e2e" in body
        assert "cortex_certifications_total" in body

    def test_validator_latencies_populated_after_certify(self):
        """The central gap this part fixes: validator latencies must be non-zero."""
        from cortex.gate.certification import CertificationGate
        from tests.conftest import make_action, make_context

        gate = CertificationGate()
        cert = gate.certify(make_action(), make_context())

        # Sentinel always runs
        assert cert.trace.sentinel_result.latency_ms > 0, \
            "Sentinel latency not recorded — Part 3 observability not wired"

        # PhysiCore and Memory run when Sentinel passes
        if cert.trace.physicore_result:
            assert cert.trace.physicore_result.latency_ms > 0, \
                "PhysiCore latency not recorded"
        if cert.trace.memory_result:
            assert cert.trace.memory_result.latency_ms > 0, \
                "Memory latency not recorded"


class TestPhase8NewMetrics:
    """Phase 4-7 metric emission methods must not raise."""

    def _fresh(self):
        from cortex.observability.metrics import CortexMetrics
        try:
            import prometheus_client
            reg = prometheus_client.CollectorRegistry()
        except ImportError:
            reg = None
        return CortexMetrics(platform_id="p8test", registry=reg)

    def test_record_episodic_store_success(self):
        self._fresh().record_episodic_store("success")

    def test_record_episodic_store_failure(self):
        self._fresh().record_episodic_store("failure")

    def test_record_ood_flagged(self):
        self._fresh().record_ood_detection(is_ood=True, ood_score=0.91)

    def test_record_ood_clear(self):
        self._fresh().record_ood_detection(is_ood=False, ood_score=0.05)

    def test_record_coordination_approved(self):
        self._fresh().record_coordination_check(approved=True)

    def test_record_coordination_blocked(self):
        self._fresh().record_coordination_check(approved=False)

    def test_record_intent_nl(self):
        self._fresh().record_intent_translation("natural_language", confidence=0.87)

    def test_record_intent_goal(self):
        self._fresh().record_intent_translation("goal_state", confidence=0.95)

    def test_record_intent_demo(self):
        self._fresh().record_intent_translation("demonstration", confidence=0.72)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
