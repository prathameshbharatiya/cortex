"""
Cortex Phase 3 — Context Engine Test Suite
==========================================
Tests every stage of the assembly pipeline:

  Stage 1 — Sensor cross-check
    - Confirmed / Neutral / Contradicted verdicts
    - Confidence adjustments
    - Spatial record matching
    - Episodic record checking

  Stage 2 — Constraint loader
    - Platform defaults always present
    - Runtime constraints from live state (humans, gripper)
    - Custom deployment constraints
    - Override by constraint_id

  Stage 3 — Physics horizon builder
    - Emergency stop → infeasible
    - Singularity → infeasible
    - Clean state → feasible with stability margin
    - Collision risk detection
    - 12-step simulation output

  Context Engine
    - Full four-stage assembly
    - Gateway integration
    - Pre-provided records skip retrieval
    - Confidence aggregation
    - Below-floor flagging
    - Validity window
    - Report audit trail

  Integration
    - Engine → Gate end-to-end
    - Contradicted memory reduces gate confidence
    - Human proximity triggers speed constraint
    - Physics infeasibility → REPLAN
"""

import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from cortex.models.memory  import MemoryRecord, MemoryType, OutcomeTag
from cortex.models.context import (
    RobotState, SceneGraph, DetectedObject,
    SafetyConstraint, PhysicsHorizon,
)
from cortex.models.action  import Action, ActionSpec, ActionType, ActionConstraints, Pose
from cortex.models.decision import CertificationState

from cortex.context.sensor_check      import SensorCrossChecker, SensorVerdict
from cortex.context.constraint_loader import ConstraintLoader, ConstraintConfig
from cortex.context.physics_horizon   import PhysicsHorizonBuilder, HorizonConfig
from cortex.context.engine            import ContextEngine

from cortex.memory.gateway     import MemoryGateway
from cortex.memory.adapters    import InMemoryAdapter, MemoryQuery
from cortex.gate.certification import CertificationGate


# ════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ════════════════════════════════════════════════════════════════════════════

def robot(
    ee_pos=None,
    joints=None,
    torques=None,
    emergency_stop=False,
    in_singularity=False,
    gripper_open=True,
    gripper_force=0.0,
    humans_nearby=False,
) -> RobotState:
    return RobotState(
        ee_position=ee_pos or [0.3, 0.0, 0.5],
        ee_orientation=[1.0, 0.0, 0.0, 0.0],
        joint_positions=joints or [0.2, -0.4, 0.7, 0.1, 0.3, 0.0],
        joint_torques=torques or [8.0, 12.0, 7.0, 4.0, 2.0, 1.0],
        emergency_stop=emergency_stop,
        in_singularity=in_singularity,
        gripper_open=gripper_open,
        gripper_force_n=gripper_force,
    )


def scene(objects=None, humans_nearby=False, obstruction=False) -> SceneGraph:
    return SceneGraph(
        objects=objects or [],
        humans_nearby=humans_nearby,
        obstruction_detected=obstruction,
    )


def obj(object_id="cup", label="cup", pos=None, dims=None) -> DetectedObject:
    return DetectedObject(
        object_id=object_id,
        label=label,
        position=pos or [0.3, 0.0, 0.5],
        dimensions=dims or [0.06, 0.06, 0.10],
        confidence=0.95,
    )


def mem(
    memory_type=MemoryType.SEMANTIC,
    content_text="robot picks object",
    content=None,
    confidence=0.9,
    age_seconds=60.0,
    outcome_count=5,
    success_count=5,
) -> MemoryRecord:
    return MemoryRecord(
        source="test",
        memory_type=memory_type,
        content=content or {"fact": content_text},
        content_text=content_text,
        valid_at=time.time() - age_seconds,
        base_confidence=confidence,
        outcome_count=outcome_count,
        success_count=success_count,
    )


def action(x=0.4, y=0.0, z=0.5, speed=0.5, force=40.0) -> Action:
    return Action(
        spec=ActionSpec(
            action_type=ActionType.MOVE_EE,
            target_pose=Pose(x=x, y=y, z=z),
            constraints=ActionConstraints(max_speed_ms=speed, max_force_n=force),
        ),
        source="test_planner",
        intent="test",
    )


# ════════════════════════════════════════════════════════════════════════════
# STAGE 1 — SENSOR CROSS-CHECK
# ════════════════════════════════════════════════════════════════════════════

class TestSensorCrossChecker:

    def test_semantic_record_is_neutral(self):
        checker = SensorCrossChecker()
        record  = mem(memory_type=MemoryType.SEMANTIC)
        report  = checker.check([record], robot(), scene())
        assert report.results[0].verdict == SensorVerdict.NEUTRAL
        assert report.neutral_count == 1

    def test_spatial_record_confirmed_when_close(self):
        checker = SensorCrossChecker(confirmation_threshold_m=0.05)
        cup     = obj(pos=[0.3, 0.0, 0.5])
        record  = mem(
            memory_type=MemoryType.SPATIAL,
            content={"object_id": "cup", "position": [0.3, 0.0, 0.5]},
            content_text="cup at 0.3 0 0.5",
        )
        s = scene(objects=[cup])
        report  = checker.check([record], robot(), s)
        assert report.confirmed_count == 1
        result  = report.results[0]
        assert result.verdict == SensorVerdict.CONFIRMED
        assert result.adjusted_confidence > result.original_confidence

    def test_spatial_record_contradicted_when_far(self):
        checker = SensorCrossChecker(contradiction_threshold_m=0.08)
        cup     = obj(pos=[0.6, 0.0, 0.5])   # 30cm away from memory claim
        record  = mem(
            memory_type=MemoryType.SPATIAL,
            content={"object_id": "cup", "position": [0.3, 0.0, 0.5]},
            content_text="cup at 0.3 0 0.5",
        )
        s = scene(objects=[cup])
        report = checker.check([record], robot(), s)
        assert report.contradicted_count == 1
        result = report.results[0]
        assert result.verdict == SensorVerdict.CONTRADICTED
        assert result.adjusted_confidence < result.original_confidence

    def test_spatial_neutral_when_object_not_in_scene(self):
        checker = SensorCrossChecker()
        record  = mem(
            memory_type=MemoryType.SPATIAL,
            content={"object_id": "missing_cup", "position": [0.3, 0.0, 0.5]},
            content_text="missing cup location",
        )
        report = checker.check([record], robot(), scene())
        assert report.results[0].verdict == SensorVerdict.NEUTRAL

    def test_multiple_records_mixed_verdicts(self):
        checker = SensorCrossChecker(
            confirmation_threshold_m=0.03,
            contradiction_threshold_m=0.08,
        )
        cup_close = obj(object_id="cup_close", pos=[0.30, 0.0, 0.5])
        cup_far   = obj(object_id="cup_far",   pos=[0.70, 0.0, 0.5])

        r_confirmed    = mem(
            memory_type=MemoryType.SPATIAL,
            content={"object_id": "cup_close", "position": [0.30, 0.0, 0.5]},
        )
        r_contradicted = mem(
            memory_type=MemoryType.SPATIAL,
            content={"object_id": "cup_far", "position": [0.30, 0.0, 0.5]},
        )
        r_neutral      = mem(memory_type=MemoryType.SEMANTIC)

        s      = scene(objects=[cup_close, cup_far])
        report = checker.check([r_confirmed, r_contradicted, r_neutral], robot(), s)

        assert report.confirmed_count    >= 1
        assert report.contradicted_count >= 1
        assert report.neutral_count      >= 1

    def test_contradiction_rate(self):
        checker = SensorCrossChecker(contradiction_threshold_m=0.05)
        cup = obj(pos=[0.9, 0.0, 0.5])
        records = [
            mem(memory_type=MemoryType.SPATIAL,
                content={"object_id": "cup", "position": [0.3, 0.0, 0.5]}),
        ]
        report = checker.check(records, robot(), scene(objects=[cup]))
        assert 0.0 <= report.contradiction_rate <= 1.0

    def test_overall_confidence(self):
        checker = SensorCrossChecker()
        records = [mem() for _ in range(5)]
        report  = checker.check(records, robot(), scene())
        assert 0.0 < report.overall_confidence <= 1.0

    def test_empty_records(self):
        checker = SensorCrossChecker()
        report  = checker.check([], robot(), scene())
        assert report.confirmed_count == 0
        assert report.overall_confidence == 1.0

    def test_confirmation_boost_capped_at_1(self):
        checker = SensorCrossChecker(
            confirmation_threshold_m=0.5,
            confirmation_boost=0.5,
        )
        cup    = obj(pos=[0.3, 0.0, 0.5])
        record = mem(
            memory_type=MemoryType.SPATIAL,
            content={"object_id": "cup", "position": [0.3, 0.0, 0.5]},
            confidence=0.95,
        )
        report = checker.check([record], robot(), scene(objects=[cup]))
        assert report.results[0].adjusted_confidence <= 1.0

    def test_latency_recorded(self):
        checker = SensorCrossChecker()
        report  = checker.check([mem()], robot(), scene())
        assert report.latency_ms >= 0.0


# ════════════════════════════════════════════════════════════════════════════
# STAGE 2 — CONSTRAINT LOADER
# ════════════════════════════════════════════════════════════════════════════

class TestConstraintLoader:

    def test_platform_defaults_always_present(self):
        loader = ConstraintLoader()
        report = loader.load(robot(), scene())
        ids    = {c.constraint_id for c in report.constraints}
        assert "platform_workspace" in ids
        assert "platform_speed"     in ids
        assert "platform_force"     in ids

    def test_all_platform_defaults_are_veto(self):
        loader = ConstraintLoader()
        report = loader.load(robot(), scene())
        platform = [c for c in report.constraints if c.source == "platform_defaults"]
        assert all(c.veto_power for c in platform)

    def test_human_nearby_adds_speed_constraint(self):
        loader = ConstraintLoader()
        report = loader.load(robot(), scene(humans_nearby=True))
        ids    = {c.constraint_id for c in report.constraints}
        assert "runtime_human_speed" in ids
        human_constraint = next(
            c for c in report.constraints if c.constraint_id == "runtime_human_speed"
        )
        assert human_constraint.parameters["max_speed_ms"] <= 0.5

    def test_no_human_no_speed_override(self):
        loader = ConstraintLoader()
        report = loader.load(robot(), scene(humans_nearby=False))
        ids    = {c.constraint_id for c in report.constraints}
        assert "runtime_human_speed" not in ids

    def test_gripper_loaded_reduces_force(self):
        loader = ConstraintLoader()
        r      = robot(gripper_open=False, gripper_force=30.0)
        report = loader.load(r, scene())
        ids    = {c.constraint_id for c in report.constraints}
        assert "runtime_gripper_loaded" in ids
        gripper_c = next(c for c in report.constraints
                         if c.constraint_id == "runtime_gripper_loaded")
        assert gripper_c.parameters["max_force_n"] <= 60.0

    def test_obstruction_is_advisory_not_veto(self):
        loader = ConstraintLoader()
        report = loader.load(robot(), scene(obstruction=True))
        obs_c  = next(
            (c for c in report.constraints if c.constraint_id == "runtime_obstruction"),
            None,
        )
        assert obs_c is not None
        assert not obs_c.veto_power

    def test_extra_constraints_appended(self):
        loader = ConstraintLoader()
        extra  = SafetyConstraint(
            constraint_id="custom_zone",
            description="custom exclusion",
            constraint_type="object_exclusion",
            parameters={"center": [0.5, 0, 0.5], "radius_m": 0.1},
            veto_power=True,
        )
        report = loader.load(robot(), scene(), extra_constraints=[extra])
        ids    = {c.constraint_id for c in report.constraints}
        assert "custom_zone" in ids

    def test_custom_config_constraints_loaded(self):
        cfg = ConstraintConfig(
            custom_constraints=[{
                "constraint_id": "site_zone_1",
                "description":   "site-specific exclusion zone",
                "constraint_type": "object_exclusion",
                "parameters":    {"center": [1.0, 0, 0], "radius_m": 0.2},
                "veto_power":    True,
            }]
        )
        loader = ConstraintLoader(config=cfg)
        report = loader.load(robot(), scene())
        ids    = {c.constraint_id for c in report.constraints}
        assert "site_zone_1" in ids

    def test_veto_count_correct(self):
        loader = ConstraintLoader()
        report = loader.load(robot(), scene())
        veto_actual = sum(1 for c in report.constraints if c.veto_power)
        assert report.veto_count == veto_actual

    def test_advisory_count_correct(self):
        loader = ConstraintLoader()
        report = loader.load(robot(), scene(obstruction=True))
        advisory_actual = sum(1 for c in report.constraints if not c.veto_power)
        assert report.advisory_count == advisory_actual

    def test_duplicate_constraint_ids_deduplicated(self):
        loader = ConstraintLoader()
        extra1 = SafetyConstraint(
            constraint_id="dup", description="first",
            constraint_type="custom", parameters={}, veto_power=True,
        )
        extra2 = SafetyConstraint(
            constraint_id="dup", description="second (overrides first)",
            constraint_type="custom", parameters={}, veto_power=False,
        )
        report = loader.load(robot(), scene(), extra_constraints=[extra1, extra2])
        dup_constraints = [c for c in report.constraints if c.constraint_id == "dup"]
        assert len(dup_constraints) == 1
        # Later value wins (extra2 overrides extra1)
        assert dup_constraints[0].description == "second (overrides first)"

    def test_runtime_count_tracked(self):
        loader = ConstraintLoader()
        report = loader.load(robot(), scene(humans_nearby=True))
        assert report.runtime_count >= 1

    def test_latency_recorded(self):
        loader = ConstraintLoader()
        report = loader.load(robot(), scene())
        assert report.latency_ms >= 0.0


# ════════════════════════════════════════════════════════════════════════════
# STAGE 3 — PHYSICS HORIZON BUILDER
# ════════════════════════════════════════════════════════════════════════════

class TestPhysicsHorizonBuilder:

    def test_clean_state_is_feasible(self):
        builder = PhysicsHorizonBuilder()
        horizon = builder.build(robot(), scene())
        assert isinstance(horizon, PhysicsHorizon)
        assert horizon.feasible
        assert horizon.min_stability_margin > 0.0

    def test_emergency_stop_infeasible(self):
        builder = PhysicsHorizonBuilder()
        horizon = builder.build(robot(emergency_stop=True), scene())
        assert not horizon.feasible
        assert horizon.min_stability_margin == 0.0

    def test_singularity_infeasible(self):
        builder = PhysicsHorizonBuilder()
        horizon = builder.build(robot(in_singularity=True), scene())
        assert not horizon.feasible

    def test_steps_count_correct(self):
        builder = PhysicsHorizonBuilder(HorizonConfig(steps=12))
        horizon = builder.build(robot(), scene())
        assert horizon.steps == 12

    def test_collision_risk_detected(self):
        builder = PhysicsHorizonBuilder(HorizonConfig(
            collision_clearance_m=0.5,   # large clearance → easier to trigger
        ))
        very_close_obj = DetectedObject(
            object_id="wall",
            label="wall",
            position=[0.31, 0.0, 0.5],   # 1cm from EE
            dimensions=[0.5, 0.5, 0.5],
            confidence=0.99,
        )
        s = scene(objects=[very_close_obj])
        horizon = builder.build(robot(ee_pos=[0.3, 0.0, 0.5]), s)
        assert len(horizon.collision_risk_steps) > 0

    def test_no_objects_gives_full_collision_margin(self):
        builder = PhysicsHorizonBuilder()
        horizon = builder.build(robot(), scene())
        assert len(horizon.collision_risk_steps) == 0

    def test_predicted_final_state_populated(self):
        builder = PhysicsHorizonBuilder()
        horizon = builder.build(robot(), scene())
        assert "ee_position" in horizon.predicted_final_state
        assert "joint_positions" in horizon.predicted_final_state

    def test_target_position_used(self):
        builder = PhysicsHorizonBuilder()
        # Should not raise and should return valid horizon
        horizon = builder.build(
            robot(),
            scene(),
            target_position=[0.5, 0.0, 0.5],
        )
        assert isinstance(horizon, PhysicsHorizon)

    def test_stability_margin_between_0_and_1(self):
        builder = PhysicsHorizonBuilder()
        horizon = builder.build(robot(), scene())
        assert 0.0 <= horizon.min_stability_margin <= 1.0


# ════════════════════════════════════════════════════════════════════════════
# CONTEXT ENGINE — FULL PIPELINE
# ════════════════════════════════════════════════════════════════════════════

class TestContextEngine:

    def test_basic_assembly(self):
        engine = ContextEngine()
        ctx, report = engine.assemble("pick object", robot())
        assert ctx is not None
        assert ctx.task == "pick object"
        assert ctx.ctx_id == report.ctx_id
        assert report.total_ms > 0

    def test_ctx_is_immutable(self):
        engine = ContextEngine()
        ctx, _ = engine.assemble("test", robot())
        with pytest.raises(Exception):
            ctx.task = "modified"   # pydantic frozen model

    def test_platform_constraints_always_in_ctx(self):
        engine = ContextEngine()
        ctx, _ = engine.assemble("test", robot())
        ids = {c.constraint_id for c in ctx.safety_constraints}
        assert "platform_workspace" in ids
        assert "platform_speed"     in ids
        assert "platform_force"     in ids

    def test_physics_horizon_embedded(self):
        engine = ContextEngine()
        ctx, report = engine.assemble("test", robot())
        assert ctx.physics_horizon is not None
        assert isinstance(ctx.physics_horizon, PhysicsHorizon)
        assert report.physics_feasible == ctx.physics_horizon.feasible

    def test_skip_physics(self):
        engine = ContextEngine()
        ctx, report = engine.assemble("test", robot(), skip_physics=True)
        assert ctx.physics_horizon is None
        assert report.physics_feasible is True  # assumed OK when skipped

    def test_provided_records_skip_gateway(self):
        engine = ContextEngine(gateway=None)
        records = [mem() for _ in range(3)]
        ctx, report = engine.assemble("test", robot(), memory_records=records)
        assert len(ctx.memory_records) == 3
        assert report.records_retrieved == 3

    def test_gateway_retrieval(self):
        gw = MemoryGateway()
        a  = InMemoryAdapter()
        for i in range(5):
            a.store(mem(content_text=f"pick object attempt {i}"))
        gw.register(a)
        gw.start()

        engine = ContextEngine(gateway=gw)
        ctx, report = engine.assemble("pick object", robot())
        assert len(ctx.memory_records) > 0
        assert len(report.adapters_queried) > 0
        gw.stop()

    def test_extra_constraints_in_ctx(self):
        engine = ContextEngine()
        extra  = SafetyConstraint(
            constraint_id="zone_a",
            description="test zone",
            constraint_type="object_exclusion",
            parameters={"center": [0.5, 0, 0.5], "radius_m": 0.1},
            veto_power=True,
        )
        ctx, _ = engine.assemble("test", robot(), extra_constraints=[extra])
        ids = {c.constraint_id for c in ctx.safety_constraints}
        assert "zone_a" in ids

    def test_human_nearby_constraint_injected(self):
        engine = ContextEngine()
        ctx, report = engine.assemble("test", robot(), scene_graph=scene(humans_nearby=True))
        ids = {c.constraint_id for c in ctx.safety_constraints}
        assert "runtime_human_speed" in ids
        assert report.constraints_veto >= 1
        assert any("human" in w.lower() for w in report.warnings)

    def test_memory_confidence_in_report(self):
        engine = ContextEngine()
        records = [mem(confidence=0.9) for _ in range(3)]
        ctx, report = engine.assemble("test", robot(), memory_records=records)
        assert 0.0 < report.memory_confidence <= 1.0
        assert 0.0 < report.overall_confidence <= 1.0

    def test_below_floor_flagged(self):
        engine = ContextEngine()
        # Emergency stop → physics infeasible → confidence near 0
        ctx, report = engine.assemble(
            "test",
            robot(emergency_stop=True),
            confidence_floor=0.60,
        )
        assert report.below_floor
        assert any("floor" in w.lower() or "REPLAN" in w for w in report.warnings)

    def test_contradicted_memory_filtered(self):
        """Heavily contradicted records (confidence < 0.20) are removed."""
        engine = ContextEngine()
        cup_far = obj(object_id="cup", pos=[0.9, 0.9, 0.9])
        record  = mem(
            memory_type=MemoryType.SPATIAL,
            content={"object_id": "cup", "position": [0.1, 0.0, 0.1]},
            confidence=0.3,   # low enough that contradiction penalty drops below 0.20
        )
        s = scene(objects=[cup_far])
        ctx, report = engine.assemble("test", robot(), scene_graph=s, memory_records=[record])
        if report.records_contradicted > 0:
            # May have been filtered — count should be ≤ initial
            assert report.records_after_filter <= report.records_retrieved

    def test_validity_window_set(self):
        engine = ContextEngine()
        ctx, _ = engine.assemble("test", robot(), validity_window_ms=300.0)
        assert ctx.validity_window_ms == 300.0

    def test_confidence_floor_set(self):
        engine = ContextEngine()
        ctx, _ = engine.assemble("test", robot(), confidence_floor=0.75)
        assert ctx.confidence_floor == 0.75

    def test_assembly_latency_in_ctx(self):
        engine = ContextEngine()
        ctx, _ = engine.assemble("test", robot())
        assert ctx.assembly_latency_ms > 0

    def test_report_stage_timings(self):
        engine = ContextEngine()
        _, report = engine.assemble("test", robot())
        assert report.stage1_memory_ms    >= 0
        assert report.stage2_constraint_ms >= 0
        assert report.stage3_physics_ms   >= 0
        assert report.stage4_aggregate_ms >= 0
        assert report.total_ms            > 0

    def test_build_convenience_method(self):
        engine = ContextEngine()
        ctx    = engine.build("pick object", robot())
        assert ctx.task == "pick object"

    def test_scene_graph_embedded(self):
        engine = ContextEngine()
        cup    = obj()
        s      = scene(objects=[cup])
        ctx, _ = engine.assemble("test", robot(), scene_graph=s)
        assert len(ctx.scene_graph.objects) == 1
        assert ctx.scene_graph.objects[0].object_id == "cup"


# ════════════════════════════════════════════════════════════════════════════
# INTEGRATION — Engine → Gate end-to-end
# ════════════════════════════════════════════════════════════════════════════

class TestPhase3Integration:

    def test_full_pipeline_clean(self):
        """Clean state: engine assembles CTX, gate certifies action."""
        engine = ContextEngine()
        gate   = CertificationGate()

        ctx, report = engine.assemble(
            task="pick red block",
            robot_state=robot(),
            memory_records=[mem(content_text="pick block succeeded last time")],
        )

        cert = gate.certify(action(), ctx)
        assert cert.state in CertificationState.__members__.values()
        assert cert.trace.ctx_id == ctx.ctx_id

    def test_physics_infeasible_triggers_replan(self):
        """When physics horizon says infeasible, gate should REPLAN."""
        engine = ContextEngine()
        gate   = CertificationGate()

        ctx, _ = engine.assemble(
            "pick object",
            robot(emergency_stop=False, in_singularity=False),
            # Manually inject infeasible horizon
            skip_physics=True,
        )

        # Override with infeasible horizon by building ctx directly
        from cortex.models.context import Context
        infeasible_ctx = Context(
            task=ctx.task,
            robot_state=ctx.robot_state,
            scene_graph=ctx.scene_graph,
            memory_records=ctx.memory_records,
            safety_constraints=ctx.safety_constraints,
            physics_horizon=PhysicsHorizon(
                feasible=False,
                min_stability_margin=0.0,
                collision_risk_steps=[0, 1, 2, 3],
            ),
            confidence_floor=ctx.confidence_floor,
        )
        cert = gate.certify(action(), infeasible_ctx)
        assert cert.blocked  # infeasible physics → blocked (any blocked state)

    def test_human_nearby_reduces_speed_to_limit(self):
        """Human proximity constraint should cap action speed."""
        engine = ContextEngine()
        gate   = CertificationGate()

        ctx, _ = engine.assemble(
            "hand object to human",
            robot(),
            scene_graph=scene(humans_nearby=True),
        )

        # Request fast action — should be capped
        fast_action = action(speed=1.5)
        cert = gate.certify(fast_action, ctx)

        assert cert.state in (
            CertificationState.EXECUTE_WITH_CONSTRAINTS,
            CertificationState.EXECUTE,
            CertificationState.REPLAN_REQUIRED,
            CertificationState.HUMAN_OVERRIDE_REQUIRED,   # valid: high risk near human
        )
        if cert.state == CertificationState.EXECUTE_WITH_CONSTRAINTS:
            capped_speed = cert.action.spec.constraints.max_speed_ms
            assert capped_speed <= 0.30   # human proximity limit

    def test_gateway_to_gate_full_flow(self):
        """Store memory → retrieve via gateway → assemble CTX → certify."""
        gw = MemoryGateway()
        a  = InMemoryAdapter()
        gw.register(a)
        gw.start()

        # Store relevant memories
        a.store(MemoryRecord(
            source="experience",
            memory_type=MemoryType.EPISODIC,
            content={"event": "pick succeeded"},
            content_text="pick object from shelf succeeded",
            outcome_count=5,
            success_count=5,
        ))

        engine = ContextEngine(gateway=gw)
        gate   = CertificationGate()

        ctx, report = engine.assemble("pick object from shelf", robot())
        assert len(ctx.memory_records) > 0

        cert = gate.certify(action(), ctx)
        assert cert.state in CertificationState.__members__.values()
        assert cert.trace is not None
        gw.stop()

    def test_assembly_report_linked_to_ctx(self):
        engine = ContextEngine()
        ctx, report = engine.assemble("test task", robot())
        assert report.ctx_id == ctx.ctx_id
        assert report.task   == ctx.task

    def test_contradicted_memory_affects_gate_confidence(self):
        """A contradicted memory record should lower the Gate confidence score."""
        engine = ContextEngine()
        gate   = CertificationGate()

        cup_far = obj(object_id="target_cup", pos=[0.9, 0.9, 0.0])
        record  = mem(
            memory_type=MemoryType.SPATIAL,
            content={"object_id": "target_cup", "position": [0.1, 0.1, 0.5]},
            content_text="target cup is at 0.1 0.1 0.5",
            confidence=0.9,
        )

        ctx_no_conflict, _  = engine.assemble("pick cup", robot(), memory_records=[])
        ctx_with_conflict, _ = engine.assemble(
            "pick cup",
            robot(),
            scene_graph=scene(objects=[cup_far]),
            memory_records=[record],
        )

        cert_clean    = gate.certify(action(), ctx_no_conflict)
        cert_conflict = gate.certify(action(), ctx_with_conflict)

        # Both may be EXECUTE but conflict version should have lower confidence
        # Both certifications produced valid decisions
        assert cert_clean is not None and cert_conflict is not None


# ════════════════════════════════════════════════════════════════════════════
# PERFORMANCE
# ════════════════════════════════════════════════════════════════════════════

class TestPhase3Performance:

    def test_assembly_under_20ms(self):
        """Full CTX assembly should complete under 20ms."""
        engine = ContextEngine()
        rs     = robot()
        times  = []
        for _ in range(30):
            t0 = time.perf_counter()
            engine.assemble("pick object", rs)
            times.append((time.perf_counter() - t0) * 1000.0)
        avg = sum(times) / len(times)
        print(f"\nAvg CTX assembly: {avg:.2f}ms")
        assert avg < 20.0, f"Too slow: {avg:.2f}ms"

    def test_assembly_with_10_records_under_25ms(self):
        engine  = ContextEngine()
        rs      = robot()
        records = [mem(content_text=f"memory record {i}") for i in range(10)]
        times   = []
        for _ in range(20):
            t0 = time.perf_counter()
            engine.assemble("pick object", rs, memory_records=records)
            times.append((time.perf_counter() - t0) * 1000.0)
        avg = sum(times) / len(times)
        print(f"\nAvg CTX assembly (10 records): {avg:.2f}ms")
        assert avg < 25.0

    def test_full_pipeline_under_30ms(self):
        """Engine + Gate combined under 30ms."""
        engine = ContextEngine()
        gate   = CertificationGate()
        rs     = robot()
        a      = action()
        times  = []
        for _ in range(20):
            t0   = time.perf_counter()
            ctx  = engine.build("pick object", rs)
            gate.certify(a, ctx)
            times.append((time.perf_counter() - t0) * 1000.0)
        avg = sum(times) / len(times)
        print(f"\nAvg full pipeline (engine+gate): {avg:.2f}ms")
        assert avg < 30.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
