"""
tests/unit/test_models.py
==========================
Tests for all Pydantic data models.
These are the contracts that every other component relies on.
"""

from __future__ import annotations

import time
import pytest

from cortex.models.action import Action, ActionSpec, ActionType, ActionConstraints, Pose
from cortex.models.context import (
    Context, RobotState, SceneGraph, DetectedObject,
    SafetyConstraint, PhysicsHorizon,
)
from cortex.models.memory import MemoryRecord, MemoryType, OutcomeTag
from cortex.models.decision import (
    CertificationDecision, CertificationState, ConfidenceContract,
    DecisionTrace, FailureMode, RiskLevel, ValidationResult,
)

from tests.conftest import make_action, make_context, make_robot_state, make_memory_record


# ══════════════════════════════════════════════════════════════════════════════
# Action models
# ══════════════════════════════════════════════════════════════════════════════

class TestPose:

    def test_default_pose_is_origin(self):
        p = Pose()
        assert p.x == 0.0 and p.y == 0.0 and p.z == 0.0
        assert p.qw == 1.0

    def test_position_property(self):
        p = Pose(x=1.0, y=2.0, z=3.0)
        arr = p.position
        assert list(arr) == [1.0, 2.0, 3.0]

    def test_distance_to(self):
        p1 = Pose(x=0.0, y=0.0, z=0.0)
        p2 = Pose(x=3.0, y=4.0, z=0.0)
        assert abs(p1.distance_to(p2) - 5.0) < 1e-9

    def test_pose_is_frozen(self):
        p = Pose(x=1.0)
        with pytest.raises(Exception):  # ValidationError or AttributeError
            p.x = 2.0  # type: ignore[misc]


class TestAction:

    def test_action_has_unique_id(self):
        a1 = make_action()
        a2 = make_action()
        assert a1.action_id != a2.action_id

    def test_with_speed_limit(self):
        a = make_action(max_speed=2.0)
        capped = a.with_speed_limit(1.0)
        assert capped.spec.constraints.max_speed_ms == 1.0
        assert a.spec.constraints.max_speed_ms == 2.0  # original unchanged

    def test_with_force_limit(self):
        a = make_action(max_force=100.0)
        capped = a.with_force_limit(50.0)
        assert capped.spec.constraints.max_force_n == 50.0

    def test_action_type_values(self):
        for t in ActionType:
            assert isinstance(t.value, str)


# ══════════════════════════════════════════════════════════════════════════════
# Context models
# ══════════════════════════════════════════════════════════════════════════════

class TestRobotState:

    def test_safe_to_move_when_no_flags(self):
        rs = make_robot_state()
        assert rs.is_safe_to_move

    def test_not_safe_when_estop(self):
        rs = make_robot_state(emergency_stop=True)
        assert not rs.is_safe_to_move

    def test_not_safe_when_singularity(self):
        rs = make_robot_state(in_singularity=True)
        assert not rs.is_safe_to_move

    def test_ee_pos_array(self):
        rs = make_robot_state(ee_position=[1.0, 2.0, 3.0])
        arr = rs.ee_pos_array
        assert list(arr) == [1.0, 2.0, 3.0]


class TestContext:

    def test_context_has_unique_id(self):
        c1 = make_context()
        c2 = make_context()
        assert c1.ctx_id != c2.ctx_id

    def test_is_expired_when_old(self):
        ctx = make_context(validity_window_ms=0.001)
        time.sleep(0.01)
        assert ctx.is_expired

    def test_is_not_expired_when_fresh(self):
        ctx = make_context(validity_window_ms=60_000)
        assert not ctx.is_expired

    def test_has_physics_horizon(self):
        ctx_with = make_context(physics_horizon=PhysicsHorizon(feasible=True, min_stability_margin=0.4))
        ctx_without = make_context()
        assert ctx_with.has_physics_horizon
        assert not ctx_without.has_physics_horizon

    def test_physics_feasible_without_horizon(self):
        ctx = make_context()
        assert ctx.physics_feasible  # default: feasible if no horizon

    def test_physics_not_feasible_with_infeasible_horizon(self):
        ctx = make_context(
            physics_horizon=PhysicsHorizon(feasible=False, min_stability_margin=0.0)
        )
        assert not ctx.physics_feasible

    def test_absolute_constraints_filters_veto_power(self):
        no_veto = SafetyConstraint(
            constraint_id="c1", description="d", constraint_type="speed_limit",
            veto_power=False,
        )
        veto = SafetyConstraint(
            constraint_id="c2", description="d", constraint_type="object_exclusion",
            veto_power=True,
        )
        ctx = make_context(safety_constraints=[no_veto, veto])
        absolute = ctx.absolute_constraints
        assert veto in absolute
        assert no_veto not in absolute

    def test_has_human_nearby(self):
        assert make_context(humans_nearby=True).has_human_nearby
        assert not make_context(humans_nearby=False).has_human_nearby

    def test_repr_contains_task(self):
        ctx = make_context(task="my test task")
        assert "my test task" in repr(ctx)


# ══════════════════════════════════════════════════════════════════════════════
# Memory models
# ══════════════════════════════════════════════════════════════════════════════

class TestMemoryRecord:

    def test_record_has_unique_id(self):
        r1 = make_memory_record()
        r2 = make_memory_record()
        assert r1.record_id != r2.record_id

    def test_is_currently_valid(self):
        r = make_memory_record()
        assert r.is_currently_valid

    def test_is_invalid_when_expired(self):
        r = make_memory_record(invalid=True)
        assert not r.is_currently_valid

    def test_age_seconds(self):
        r = make_memory_record(age_seconds=120.0)
        assert 119 < r.age_seconds < 121

    def test_success_rate_with_outcomes(self):
        r = make_memory_record(outcome_count=10, success_count=7)
        assert abs(r.success_rate - 0.7) < 1e-9

    def test_success_rate_unknown_is_neutral(self):
        r = make_memory_record(outcome_count=0, success_count=0)
        assert r.success_rate == 0.5  # neutral prior

    def test_sensor_anchored(self):
        r_anchored = make_memory_record(sensor_anchor={"sensor": "lidar"})
        r_plain = make_memory_record()
        assert r_anchored.is_sensor_anchored
        assert not r_plain.is_sensor_anchored

    def test_effective_confidence_decays_with_age(self):
        fresh = make_memory_record(confidence=1.0, age_seconds=0)
        stale = make_memory_record(confidence=1.0, age_seconds=7200)
        assert fresh.effective_confidence >= stale.effective_confidence

    def test_sensor_anchor_boosts_confidence(self):
        anchored = make_memory_record(confidence=0.8, sensor_anchor={"sensor": "camera"})
        plain = make_memory_record(confidence=0.8)
        assert anchored.effective_confidence >= plain.effective_confidence

    def test_repr_contains_type_and_confidence(self):
        r = make_memory_record(memory_type=MemoryType.EPISODIC, confidence=0.75)
        assert "episodic" in repr(r)


# ══════════════════════════════════════════════════════════════════════════════
# Decision models
# ══════════════════════════════════════════════════════════════════════════════

class TestValidationResult:

    def test_score_is_bounded(self):
        vr = ValidationResult(passed=True, score=0.85)
        assert 0.0 <= vr.score <= 1.0

    def test_default_latency_is_zero(self):
        vr = ValidationResult(passed=True, score=0.9)
        assert vr.latency_ms == 0.0

    def test_is_frozen(self):
        vr = ValidationResult(passed=True, score=0.9)
        with pytest.raises(Exception):
            vr.score = 0.5  # type: ignore[misc]


class TestConfidenceContract:

    def test_high_confidence_property(self):
        cc = ConfidenceContract(
            success_probability=0.92,
            stability_margin=0.4,
            worst_case_risk=RiskLevel.LOW,
            validity_window_ms=500.0,
        )
        assert cc.is_high_confidence

    def test_not_high_confidence_when_medium_risk(self):
        cc = ConfidenceContract(
            success_probability=0.95,
            stability_margin=0.4,
            worst_case_risk=RiskLevel.MEDIUM,
            validity_window_ms=500.0,
        )
        assert not cc.is_high_confidence

    def test_acceptable_property(self):
        cc = ConfidenceContract(
            success_probability=0.65,
            stability_margin=0.3,
            worst_case_risk=RiskLevel.MEDIUM,
            validity_window_ms=500.0,
        )
        assert cc.is_acceptable

    def test_not_acceptable_when_critical(self):
        cc = ConfidenceContract(
            success_probability=0.70,
            stability_margin=0.3,
            worst_case_risk=RiskLevel.CRITICAL,
            validity_window_ms=500.0,
        )
        assert not cc.is_acceptable


class TestCertificationState:

    def test_all_states_are_strings(self):
        for s in CertificationState:
            assert isinstance(s.value, str)

    def test_execute_value(self):
        assert CertificationState.EXECUTE.value == "EXECUTE"

    def test_states_count(self):
        assert len(CertificationState) == 5


class TestRiskLevel:

    def test_levels_ordered(self):
        levels = list(RiskLevel)
        assert levels[0] == RiskLevel.NONE
        assert levels[-1] == RiskLevel.CRITICAL


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
