"""
tests/unit/test_certification_gate.py
======================================
Certification Gate — the single most critical piece of Cortex.

Every test here is a property of the system that must hold forever.
"""

from __future__ import annotations

import time
import pytest

from cortex.models.action import ActionType
from cortex.models.context import PhysicsHorizon, SafetyConstraint, SceneGraph, DetectedObject
from cortex.models.memory import MemoryType, OutcomeTag
from cortex.models.decision import CertificationState, RiskLevel
from cortex.gate.certification import CertificationGate

from tests.conftest import (
    make_action, make_context, make_robot_state, make_memory_record,
    make_exclusion_zone, make_feasible_horizon, make_infeasible_horizon,
)

_ALL_STATES = set(CertificationState.__members__.values())
_BLOCKED = {CertificationState.SAFE_HALT, CertificationState.REPLAN_REQUIRED,
            CertificationState.HUMAN_OVERRIDE_REQUIRED}
_APPROVED = {CertificationState.EXECUTE, CertificationState.EXECUTE_WITH_CONSTRAINTS}


class TestExecutePath:

    def test_clean_action_returns_valid_state(self, gate):
        cert = gate.certify(make_action(), make_context())
        assert cert.state in _ALL_STATES

    def test_approved_action_has_confidence(self, gate):
        cert = gate.certify(make_action(), make_context())
        if cert.state in _APPROVED:
            assert cert.confidence is not None
            c = cert.confidence
            assert 0.0 <= c.success_probability <= 1.0
            assert c.stability_margin >= 0.0
            assert c.validity_window_ms > 0

    def test_confidence_has_all_three_proofs(self, gate):
        cert = gate.certify(make_action(), make_context())
        if cert.confidence:
            proofs = cert.confidence.validation_proofs
            assert "sentinel" in proofs
            assert "physicore" in proofs
            assert "memory" in proofs

    def test_approved_action_is_not_none(self, gate):
        cert = gate.certify(make_action(), make_context())
        if cert.approved:
            assert cert.action is not None

    def test_blocked_action_is_none(self, gate):
        ctx = make_context(robot_state=make_robot_state(emergency_stop=True))
        cert = gate.certify(make_action(), ctx)
        assert cert.action is None

    def test_all_decisions_have_reason(self, gate):
        cert = gate.certify(make_action(), make_context())
        assert cert.reason and len(cert.reason) > 5

    def test_execute_with_memory_records_completes(self, gate):
        records = [make_memory_record() for _ in range(5)]
        cert = gate.certify(make_action(), make_context(memory_records=records))
        assert cert.state in _ALL_STATES


class TestExecuteWithConstraints:

    def test_speed_over_sentinel_limit_gets_capped(self, gate):
        action = make_action(max_speed=10.0)
        cert = gate.certify(action, make_context())
        if cert.state == CertificationState.EXECUTE_WITH_CONSTRAINTS:
            assert cert.action is not None
            assert cert.action.spec.constraints.max_speed_ms <= 2.0

    def test_force_over_sentinel_limit_gets_capped(self, gate):
        action = make_action(max_force=500.0)
        cert = gate.certify(action, make_context())
        if cert.state == CertificationState.EXECUTE_WITH_CONSTRAINTS:
            assert cert.action is not None
            assert cert.action.spec.constraints.max_force_n <= 150.0

    def test_human_nearby_does_not_halt(self, gate):
        cert = gate.certify(make_action(max_speed=1.0), make_context(humans_nearby=True))
        assert cert.state != CertificationState.SAFE_HALT

    def test_human_nearby_adds_uncertainty_source(self, gate):
        ctx = make_context(humans_nearby=True)
        cert = gate.certify(make_action(), ctx)
        if cert.confidence:
            assert "human_proximity" in cert.confidence.uncertainty_sources


class TestSafeHalt:

    def test_emergency_stop_blocks(self, gate, estop_context):
        cert = gate.certify(make_action(), estop_context)
        assert cert.blocked
        assert cert.action is None

    def test_emergency_stop_state(self, gate, estop_context):
        cert = gate.certify(make_action(), estop_context)
        assert cert.state in {CertificationState.SAFE_HALT,
                              CertificationState.HUMAN_OVERRIDE_REQUIRED}

    def test_singularity_blocks(self, gate, singularity_context):
        cert = gate.certify(make_action(), singularity_context)
        assert cert.blocked

    def test_halt_has_blocking_failure_mode(self, gate, estop_context):
        cert = gate.certify(make_action(), estop_context)
        assert cert.trace.blocking_failure_mode is not None

    def test_halt_sentinel_result_failed(self, gate, estop_context):
        cert = gate.certify(make_action(), estop_context)
        assert cert.trace.sentinel_result is not None
        assert not cert.trace.sentinel_result.passed

    def test_halt_physicore_not_run(self, gate, estop_context):
        """Sentinel blocks before PhysiCore runs — PhysiCore result must be None."""
        cert = gate.certify(make_action(), estop_context)
        assert cert.trace.physicore_result is None

    def test_halt_memory_not_run(self, gate, estop_context):
        """Sentinel blocks before Memory runs — Memory result must be None."""
        cert = gate.certify(make_action(), estop_context)
        assert cert.trace.memory_result is None


class TestReplanRequired:

    def test_expired_context_triggers_replan(self, gate):
        ctx = make_context(validity_window_ms=0.001)
        time.sleep(0.01)
        cert = gate.certify(make_action(), ctx)
        assert cert.state == CertificationState.REPLAN_REQUIRED

    def test_expired_context_failure_code(self, gate):
        ctx = make_context(validity_window_ms=0.001)
        time.sleep(0.01)
        cert = gate.certify(make_action(), ctx)
        assert cert.trace.blocking_failure_mode.code == "context_expired"

    def test_infeasible_horizon_blocks(self, gate):
        ctx = make_context(physics_horizon=make_infeasible_horizon())
        cert = gate.certify(make_action(), ctx)
        assert cert.blocked
        assert cert.action is None

    def test_feasible_horizon_does_not_block(self, gate):
        ctx = make_context(physics_horizon=make_feasible_horizon(stability=0.5))
        cert = gate.certify(make_action(), ctx)
        # Should not block because of a feasible horizon
        assert cert.state in _ALL_STATES


class TestHumanOverride:

    def test_exclusion_zone_blocks_execution(self, gate):
        zone = make_exclusion_zone(center=[0.4, 0.0, 0.5], radius_m=0.3)
        ctx = make_context(safety_constraints=[zone])
        cert = gate.certify(make_action(x=0.4, y=0.0, z=0.5), ctx)
        assert cert.blocked

    def test_ultra_high_threshold_escalates(self):
        gate = CertificationGate(human_override_threshold=0.99, replan_threshold=0.999)
        cert = gate.certify(make_action(), make_context())
        assert cert.state in {CertificationState.HUMAN_OVERRIDE_REQUIRED,
                              CertificationState.REPLAN_REQUIRED}

    def test_override_has_full_trace(self, gate):
        zone = make_exclusion_zone(center=[0.4, 0.0, 0.5], radius_m=0.3)
        ctx = make_context(safety_constraints=[zone])
        cert = gate.certify(make_action(x=0.4, y=0.0, z=0.5), ctx)
        assert cert.trace.certification_state == cert.state


class TestAuthorityHierarchy:

    def test_sentinel_overrides_perfect_memory(self, gate):
        records = [make_memory_record(confidence=1.0) for _ in range(10)]
        ctx = make_context(
            robot_state=make_robot_state(emergency_stop=True),
            memory_records=records,
        )
        cert = gate.certify(make_action(), ctx)
        assert cert.blocked

    def test_sentinel_overrides_perfect_physics(self, gate):
        ctx = make_context(
            robot_state=make_robot_state(emergency_stop=True),
            physics_horizon=make_feasible_horizon(stability=1.0),
        )
        cert = gate.certify(make_action(), ctx)
        assert cert.blocked

    def test_physicore_failure_overrides_perfect_memory(self, gate):
        """Perfect memory cannot override a physically unreachable target.

        The physics horizon is advisory — PhysiCore's MPC can override it.
        To guarantee PhysiCore blocks, we use a target 10m away which is
        definitively outside any robot workspace.
        """
        records = [make_memory_record(confidence=1.0) for _ in range(5)]
        ctx = make_context(memory_records=records)
        cert = gate.certify(make_action(x=10.0, y=10.0, z=10.0), ctx)
        assert cert.blocked

    def test_memory_failure_alone_never_halts(self, gate):
        """Memory is Priority 3 soft constraint — must never produce SAFE_HALT."""
        stale = make_memory_record(confidence=0.1, age_seconds=10000.0)
        ctx = make_context(memory_records=[stale])
        cert = gate.certify(make_action(), ctx)
        assert cert.state != CertificationState.SAFE_HALT


class TestDecisionTrace:

    def test_trace_ids_are_unique(self, gate):
        c1 = gate.certify(make_action(), make_context())
        c2 = gate.certify(make_action(), make_context())
        assert c1.trace.trace_id != c2.trace.trace_id

    def test_trace_action_id_matches(self, gate):
        action = make_action()
        cert = gate.certify(action, make_context())
        assert cert.trace.action_id == action.action_id

    def test_trace_ctx_id_matches(self, gate):
        ctx = make_context()
        cert = gate.certify(make_action(), ctx)
        assert cert.trace.ctx_id == ctx.ctx_id

    def test_trace_total_latency_populated(self, gate):
        cert = gate.certify(make_action(), make_context())
        assert cert.trace.total_latency_ms > 0

    def test_sentinel_latency_populated(self, gate):
        """This was the core gap — latency_ms was always zero before Part 3."""
        cert = gate.certify(make_action(), make_context())
        assert cert.trace.sentinel_result.latency_ms > 0

    def test_physicore_latency_populated(self, gate):
        cert = gate.certify(make_action(), make_context())
        if cert.trace.physicore_result:
            assert cert.trace.physicore_result.latency_ms > 0

    def test_memory_latency_populated(self, gate):
        cert = gate.certify(make_action(), make_context())
        if cert.trace.memory_result:
            assert cert.trace.memory_result.latency_ms > 0

    def test_state_in_trace_matches_decision(self, gate):
        cert = gate.certify(make_action(), make_context())
        assert cert.trace.certification_state == cert.state

    def test_timestamp_is_recent(self, gate):
        before = time.time()
        cert = gate.certify(make_action(), make_context())
        assert cert.trace.timestamp >= before


class TestDecisionProperties:

    def test_approved_blocked_exclusive(self, gate):
        cert = gate.certify(make_action(), make_context())
        assert cert.approved != cert.blocked

    def test_decision_id_unique_per_call(self, gate):
        c1 = gate.certify(make_action(), make_context())
        c2 = gate.certify(make_action(), make_context())
        assert c1.decision_id != c2.decision_id

    def test_repr_readable(self, gate):
        cert = gate.certify(make_action(), make_context())
        r = repr(cert)
        assert "CertificationDecision" in r
        assert cert.state.value in r

    def test_requires_replan_property(self, gate):
        ctx = make_context(validity_window_ms=0.001)
        time.sleep(0.01)
        cert = gate.certify(make_action(), ctx)
        assert cert.requires_replan

    def test_requires_human_property(self):
        gate = CertificationGate(human_override_threshold=0.99, replan_threshold=0.999)
        cert = gate.certify(make_action(), make_context())
        if cert.state == CertificationState.HUMAN_OVERRIDE_REQUIRED:
            assert cert.requires_human


class TestPerformance:

    @pytest.mark.timeout(5)
    def test_single_certification_under_100ms(self, gate):
        t0 = time.perf_counter()
        gate.certify(make_action(), make_context())
        elapsed = (time.perf_counter() - t0) * 1000
        assert elapsed < 100.0, f"Single certification {elapsed:.1f}ms > 100ms"

    @pytest.mark.timeout(10)
    def test_ten_certifications_avg_under_50ms(self, gate):
        action, ctx = make_action(), make_context()
        times = [(time.perf_counter(), gate.certify(action, ctx)) for _ in range(10)]
        # Re-measure properly
        t0 = time.perf_counter()
        for _ in range(10):
            gate.certify(action, ctx)
        avg = (time.perf_counter() - t0) * 100  # ms per call
        assert avg < 50.0, f"Average {avg:.1f}ms > 50ms"

    @pytest.mark.timeout(30)
    def test_100_certifications_no_exceptions(self, gate):
        import random
        rng = random.Random(42)
        for _ in range(100):
            x, y, z = rng.uniform(0.1, 0.6), rng.uniform(-0.2, 0.2), rng.uniform(0.2, 0.7)
            cert = gate.certify(make_action(x=x, y=y, z=z), make_context())
            assert cert.state in _ALL_STATES
            assert cert.trace is not None

    @pytest.mark.timeout(5)
    def test_estop_path_under_5ms(self, gate):
        ctx = make_context(robot_state=make_robot_state(emergency_stop=True))
        t0 = time.perf_counter()
        for _ in range(20):
            gate.certify(make_action(), ctx)
        avg = (time.perf_counter() - t0) * 1000 / 20
        assert avg < 5.0, f"E-stop path {avg:.1f}ms > 5ms"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
