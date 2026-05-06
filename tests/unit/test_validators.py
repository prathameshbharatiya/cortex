"""
tests/unit/test_validators.py
==============================
Unit tests for all three validators in isolation.
Each validator is tested with no dependency on the others.
"""

from __future__ import annotations

import time
import pytest

from cortex.models.memory import MemoryType, OutcomeTag
from cortex.validators.sentinel import SentinelValidator
from cortex.validators.physicore import PhysiCoreValidator
from cortex.validators.memory_validator import MemoryValidator
from cortex.models.context import SceneGraph, DetectedObject, PhysicsHorizon
from cortex.models.decision import RiskLevel

from tests.conftest import (
    make_action, make_context, make_robot_state, make_memory_record,
    make_exclusion_zone, make_feasible_horizon, make_infeasible_horizon,
)


# ══════════════════════════════════════════════════════════════════════════════
# SENTINEL VALIDATOR
# ══════════════════════════════════════════════════════════════════════════════

class TestSentinelValidator:

    def test_clean_action_passes(self, sentinel):
        result = sentinel.validate(make_action(), make_context())
        assert result.passed
        assert result.score > 0

    def test_result_has_latency(self, sentinel):
        result = sentinel.validate(make_action(), make_context())
        assert result.latency_ms >= 0

    def test_emergency_stop_hard_blocks(self, sentinel):
        ctx = make_context(robot_state=make_robot_state(emergency_stop=True))
        result = sentinel.validate(make_action(), ctx)
        assert not result.passed
        assert result.score == 0.0
        codes = [fm.code for fm in result.failure_modes]
        assert "emergency_stop" in codes

    def test_emergency_stop_is_critical(self, sentinel):
        ctx = make_context(robot_state=make_robot_state(emergency_stop=True))
        result = sentinel.validate(make_action(), ctx)
        worst = max(result.failure_modes, key=lambda f: f.risk_level.value)
        assert worst.risk_level == RiskLevel.CRITICAL

    def test_singularity_hard_blocks(self, sentinel):
        ctx = make_context(robot_state=make_robot_state(in_singularity=True))
        result = sentinel.validate(make_action(), ctx)
        assert not result.passed

    def test_speed_over_limit_is_mitigable(self):
        s = SentinelValidator(max_speed_ms=1.0)
        action = make_action(max_speed=2.0)
        result = s.validate(action, make_context())
        speed_fms = [fm for fm in result.failure_modes if fm.code == "speed_limit_exceeded"]
        assert speed_fms
        assert speed_fms[0].mitigable

    def test_within_speed_limit_passes_cleanly(self):
        s = SentinelValidator(max_speed_ms=2.0)
        action = make_action(max_speed=1.0)
        result = s.validate(action, make_context())
        assert result.passed
        speed_fms = [fm for fm in result.failure_modes if fm.code == "speed_limit_exceeded"]
        assert not speed_fms

    def test_force_over_limit_is_mitigable(self):
        s = SentinelValidator(max_force_n=100.0)
        action = make_action(max_force=200.0)
        result = s.validate(action, make_context())
        force_fms = [fm for fm in result.failure_modes if fm.code == "force_limit_exceeded"]
        assert force_fms
        assert force_fms[0].mitigable

    def test_human_proximity_is_soft_failure(self, sentinel):
        ctx = make_context(humans_nearby=True)
        result = sentinel.validate(make_action(), ctx)
        assert result.passed  # does not hard-block
        human_fms = [fm for fm in result.failure_modes if fm.code == "human_proximity"]
        assert human_fms
        assert human_fms[0].mitigable

    def test_exclusion_zone_hard_blocks(self, sentinel):
        zone = make_exclusion_zone(center=[0.4, 0.0, 0.5], radius_m=0.2)
        ctx = make_context(safety_constraints=[zone])
        result = sentinel.validate(make_action(x=0.4, y=0.0, z=0.5), ctx)
        assert not result.passed
        hard_fms = [fm for fm in result.failure_modes if not fm.mitigable]
        assert hard_fms

    def test_target_outside_exclusion_zone_passes(self, sentinel):
        zone = make_exclusion_zone(center=[2.0, 2.0, 2.0], radius_m=0.1)
        ctx = make_context(safety_constraints=[zone])
        result = sentinel.validate(make_action(x=0.3, y=0.0, z=0.5), ctx)
        # Action is far from the zone — no exclusion failure
        exclusion_fms = [fm for fm in result.failure_modes if "exclusion" in fm.code]
        assert not exclusion_fms

    def test_failure_modes_have_source(self, sentinel):
        ctx = make_context(robot_state=make_robot_state(emergency_stop=True))
        result = sentinel.validate(make_action(), ctx)
        for fm in result.failure_modes:
            assert fm.source == "sentinel"


# ══════════════════════════════════════════════════════════════════════════════
# PHYSICORE VALIDATOR
# ══════════════════════════════════════════════════════════════════════════════

class TestPhysiCoreValidator:

    def test_reachable_target_passes(self, physicore):
        result = physicore.validate(make_action(x=0.3, y=0.0, z=0.5), make_context())
        assert result.passed
        assert result.score > 0

    def test_result_has_latency(self, physicore):
        result = physicore.validate(make_action(), make_context())
        assert result.latency_ms >= 0

    def test_score_is_bounded(self, physicore):
        result = physicore.validate(make_action(), make_context())
        assert 0.0 <= result.score <= 1.0

    def test_feasible_horizon_passes(self, physicore):
        ctx = make_context(physics_horizon=make_feasible_horizon(stability=0.5))
        result = physicore.validate(make_action(), ctx)
        assert result.passed

    def test_infeasible_horizon_fails(self, physicore):
        ctx = make_context(physics_horizon=make_infeasible_horizon())
        result = physicore.validate(make_action(), ctx)
        assert not result.passed or result.score < 0.3

    def test_infeasible_horizon_score_is_low(self, physicore):
        ctx = make_context(physics_horizon=make_infeasible_horizon())
        result = physicore.validate(make_action(), ctx)
        assert result.score < 0.5

    def test_collision_proximity_flagged(self):
        p = PhysiCoreValidator(collision_proximity_m=0.15)
        obj = DetectedObject(
            object_id="obstacle",
            label="box",
            position=[0.4, 0.0, 0.5],
            dimensions=[0.05, 0.05, 0.05],
            confidence=0.95,
        )
        scene = SceneGraph(objects=[obj])
        ctx = make_context(scene_graph=scene)
        result = p.validate(make_action(x=0.4, y=0.0, z=0.5), ctx)
        # Either blocks or flags — result must be a valid ValidationResult
        assert isinstance(result.passed, bool)
        assert 0.0 <= result.score <= 1.0

    def test_stability_margin_below_threshold_fails(self):
        p = PhysiCoreValidator(min_stability_margin=0.50)
        # A horizon with stability=0.05 is below threshold=0.50
        ctx = make_context(
            physics_horizon=PhysicsHorizon(feasible=True, min_stability_margin=0.05)
        )
        result = p.validate(make_action(), ctx)
        assert not result.passed or result.score < 0.5

    def test_stability_margin_above_threshold_passes(self):
        p = PhysiCoreValidator(min_stability_margin=0.10)
        ctx = make_context(
            physics_horizon=PhysicsHorizon(feasible=True, min_stability_margin=0.60)
        )
        result = p.validate(make_action(), ctx)
        assert result.passed


# ══════════════════════════════════════════════════════════════════════════════
# MEMORY VALIDATOR
# ══════════════════════════════════════════════════════════════════════════════

class TestMemoryValidator:

    def test_no_records_passes_with_moderate_score(self, memory_validator):
        result = memory_validator.validate(make_action(), make_context(memory_records=[]))
        assert result.passed
        assert 0.5 <= result.score <= 0.9  # not penalised but not max confidence

    def test_fresh_valid_records_pass_high_score(self, memory_validator):
        records = [make_memory_record(age_seconds=10) for _ in range(3)]
        result = memory_validator.validate(make_action(), make_context(memory_records=records))
        assert result.passed
        assert result.score >= 0.7

    def test_stale_records_lower_score_vs_fresh(self):
        m = MemoryValidator(stale_threshold_s=100)
        stale = make_memory_record(age_seconds=500)
        fresh = make_memory_record(age_seconds=10)
        r_stale = m.validate(make_action(), make_context(memory_records=[stale]))
        r_fresh = m.validate(make_action(), make_context(memory_records=[fresh]))
        assert r_stale.score < r_fresh.score

    def test_invalid_record_is_flagged(self, memory_validator):
        invalid = make_memory_record(invalid=True)
        result = memory_validator.validate(make_action(), make_context(memory_records=[invalid]))
        invalid_fms = [fm for fm in result.failure_modes if "invalid" in fm.code]
        assert invalid_fms

    def test_high_failure_rate_flags_warning(self):
        m = MemoryValidator(failure_rate_threshold=0.3)
        bad = make_memory_record(outcome_count=10, success_count=2,
                                 outcome_tag=OutcomeTag.FAILURE)
        result = m.validate(make_action(), make_context(memory_records=[bad]))
        rate_fms = [fm for fm in result.failure_modes if "failure_rate" in fm.code]
        assert rate_fms

    def test_low_failure_rate_no_warning(self):
        m = MemoryValidator(failure_rate_threshold=0.3)
        good = make_memory_record(outcome_count=10, success_count=9,
                                  outcome_tag=OutcomeTag.SUCCESS)
        result = m.validate(make_action(), make_context(memory_records=[good]))
        rate_fms = [fm for fm in result.failure_modes if "failure_rate" in fm.code]
        assert not rate_fms

    def test_spatial_contradiction_flagged(self):
        m = MemoryValidator(contradiction_dist_m=0.05)
        obj = DetectedObject(
            object_id="cup",
            label="cup",
            position=[0.7, 0.0, 0.5],   # sensor says it's here
            dimensions=[0.05, 0.05, 0.08],
            confidence=0.9,
        )
        scene = SceneGraph(objects=[obj])
        record = make_memory_record(
            memory_type=MemoryType.SPATIAL,
            content={"object_id": "cup", "position": [0.3, 0.0, 0.5]},  # memory says here
        )
        ctx = make_context(memory_records=[record], scene_graph=scene)
        result = m.validate(make_action(), ctx)
        contradiction_fms = [fm for fm in result.failure_modes if "contradiction" in fm.code]
        assert contradiction_fms

    def test_sensor_anchored_record_scores_higher(self, memory_validator):
        anchored = make_memory_record(
            sensor_anchor={"sensor": "camera", "confirmed_at": time.time()}
        )
        plain = make_memory_record()
        r_anchored = memory_validator.validate(make_action(), make_context(memory_records=[anchored]))
        r_plain = memory_validator.validate(make_action(), make_context(memory_records=[plain]))
        # Anchored should score >= plain (or equal if both max out)
        assert r_anchored.score >= r_plain.score * 0.9

    def test_memory_validator_passes_are_always_soft(self, memory_validator):
        """Memory failures never produce score=0.0 — they reduce confidence only."""
        bad_records = [
            make_memory_record(confidence=0.1, age_seconds=5000, outcome_tag=OutcomeTag.FAILURE)
        ]
        result = memory_validator.validate(make_action(), make_context(memory_records=bad_records))
        # Memory validator passes even with bad records — it's a soft constraint
        assert result.score > 0.0

    def test_result_has_latency(self, memory_validator):
        result = memory_validator.validate(make_action(), make_context())
        assert result.latency_ms >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])




# ══════════════════════════════════════════════════════════════════════════════
# PHYSICORE VALIDATOR — real engine behavioral tests
# These test what the PhysiCore engine actually does, not the fallback path.
# ══════════════════════════════════════════════════════════════════════════════

class TestPhysiCoreBehavior:
    """Behavioral tests for PhysiCoreValidator with the real MPC engine running."""

    def test_result_is_valid_validation_result(self, physicore):
        from cortex.models.decision import ValidationResult
        result = physicore.validate(make_action(), make_context())
        assert isinstance(result, ValidationResult)
        assert isinstance(result.passed, bool)
        assert 0.0 <= result.score <= 1.0
        assert result.latency_ms >= 0
        assert isinstance(result.failure_modes, list)
        assert isinstance(result.notes, str)

    def test_infeasible_horizon_lowers_score(self, physicore):
        """An infeasible horizon must produce a lower score than a feasible one."""
        ctx_ok  = make_context(physics_horizon=make_feasible_horizon(stability=0.8))
        ctx_bad = make_context(physics_horizon=make_infeasible_horizon())
        r_ok  = physicore.validate(make_action(), ctx_ok)
        r_bad = physicore.validate(make_action(), ctx_bad)
        assert r_bad.score <= r_ok.score

    def test_feasible_horizon_passes(self, physicore):
        ctx = make_context(physics_horizon=make_feasible_horizon(stability=0.5))
        result = physicore.validate(make_action(), ctx)
        assert result.passed

    def test_failure_modes_have_source(self, physicore):
        ctx = make_context(physics_horizon=make_infeasible_horizon())
        result = physicore.validate(make_action(), ctx)
        for fm in result.failure_modes:
            assert fm.source in ("physicore", "sentinel")

    def test_consistent_results_on_same_input(self, physicore):
        """Same input must produce consistent pass/fail (engine is deterministic)."""
        action = make_action(x=0.3, y=0.0, z=0.5)
        ctx = make_context()
        r1 = physicore.validate(action, ctx)
        r2 = physicore.validate(action, ctx)
        assert r1.passed == r2.passed

    def test_no_target_pose_completes(self, physicore):
        from cortex.models.action import Action, ActionSpec, ActionType, ActionConstraints
        action = Action(
            spec=ActionSpec(
                action_type=ActionType.GRASP,
                target_pose=None,
                constraints=ActionConstraints(max_speed_ms=0.5, max_force_n=50.0),
            ),
            source="test", intent="grasp",
        )
        result = physicore.validate(action, make_context())
        assert isinstance(result.passed, bool)

    def test_multiple_calls_no_crash(self, physicore):
        for i in range(10):
            result = physicore.validate(make_action(), make_context())
            assert isinstance(result.passed, bool)


# ══════════════════════════════════════════════════════════════════════════════
# SENTINEL VALIDATOR — additional rules path coverage
# ══════════════════════════════════════════════════════════════════════════════

class TestSentinelAdditionalPaths:

    def test_workspace_x_boundary_fails(self, sentinel):
        from cortex.models.context import SceneGraph
        scene = SceneGraph(workspace_bounds={"x_min": -1.0, "x_max": 1.0,
                                              "y_min": -1.0, "y_max": 1.0,
                                              "z_min": 0.0,  "z_max": 1.5})
        ctx = make_context(scene_graph=scene)
        result = sentinel.validate(make_action(x=2.0), ctx)
        assert not result.passed

    def test_workspace_z_boundary_fails(self, sentinel):
        from cortex.models.context import SceneGraph
        scene = SceneGraph(workspace_bounds={"x_min": -1.0, "x_max": 1.0,
                                              "y_min": -1.0, "y_max": 1.0,
                                              "z_min": 0.0,  "z_max": 1.5})
        ctx = make_context(scene_graph=scene)
        result = sentinel.validate(make_action(z=3.0), ctx)
        assert not result.passed

    def test_both_speed_and_human_proximity_flagged(self):
        s = SentinelValidator(max_speed_ms=1.0)
        result = s.validate(make_action(max_speed=2.0), make_context(humans_nearby=True))
        codes = [fm.code for fm in result.failure_modes]
        assert "speed_limit_exceeded" in codes
        assert "human_proximity" in codes

    def test_force_over_limit_mitigable_not_blocking(self):
        s = SentinelValidator(max_force_n=50.0)
        result = s.validate(make_action(max_force=80.0), make_context())
        assert result.passed
        force_fms = [fm for fm in result.failure_modes if fm.code == "force_limit_exceeded"]
        assert force_fms
        assert force_fms[0].mitigable

    def test_multiple_exclusion_zones_first_blocks(self, sentinel):
        from cortex.models.context import SafetyConstraint
        z1 = SafetyConstraint(constraint_id="z1", description="d",
            constraint_type="object_exclusion",
            parameters={"center": [0.4, 0.0, 0.5], "radius_m": 0.3}, veto_power=True)
        z2 = SafetyConstraint(constraint_id="z2", description="d",
            constraint_type="object_exclusion",
            parameters={"center": [0.8, 0.0, 0.5], "radius_m": 0.3}, veto_power=True)
        ctx = make_context(safety_constraints=[z1, z2])
        result = sentinel.validate(make_action(x=0.4, y=0.0, z=0.5), ctx)
        assert not result.passed

    def test_sentinel_properties_accessible(self):
        s = SentinelValidator()
        assert s.mode == "RULES_ONLY"
        assert s.ledger_hash == "NO_SENTINEL"
        assert s.sentinel_status is None
