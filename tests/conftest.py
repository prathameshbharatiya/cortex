"""
tests/conftest.py
=================
Shared pytest fixtures. Imported automatically by every test module.

All fixtures are deterministic — no randomness, no I/O, no network.
Fixtures that create mutable state are function-scoped so tests
cannot pollute each other.
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
from cortex.models.decision import CertificationState
from cortex.gate.certification import CertificationGate
from cortex.validators.sentinel import SentinelValidator
from cortex.validators.physicore import PhysiCoreValidator
from cortex.validators.memory_validator import MemoryValidator
from cortex.memory.gateway import MemoryGateway
from cortex.memory.adapters.in_memory import InMemoryAdapter


# ── Robot state ───────────────────────────────────────────────────────────────

def make_robot_state(**overrides) -> RobotState:
    """Standard, safe, non-singular robot state at a reachable pose."""
    defaults = dict(
        ee_position=[0.3, 0.0, 0.5],
        ee_orientation=[1.0, 0.0, 0.0, 0.0],
        ee_velocity=[0.0, 0.0, 0.0],
        joint_positions=[0.1, -0.5, 0.8, 0.0, 0.4, 0.0],
        joint_velocities=[0.0] * 6,
        joint_torques=[5.0, 10.0, 8.0, 3.0, 2.0, 1.0],
        gripper_open=True,
        gripper_force_n=0.0,
        is_moving=False,
        in_singularity=False,
        emergency_stop=False,
    )
    defaults.update(overrides)
    return RobotState(**defaults)


# ── Actions ───────────────────────────────────────────────────────────────────

def make_action(
    x: float = 0.4,
    y: float = 0.0,
    z: float = 0.5,
    action_type: ActionType = ActionType.MOVE_EE,
    max_speed: float = 0.5,
    max_force: float = 50.0,
    source: str = "test_planner",
    intent: str = "pick up object",
    ai_confidence: float = 0.85,
) -> Action:
    """Standard action within workspace bounds, under speed/force limits."""
    return Action(
        spec=ActionSpec(
            action_type=action_type,
            target_pose=Pose(x=x, y=y, z=z),
            constraints=ActionConstraints(
                max_speed_ms=max_speed,
                max_force_n=max_force,
            ),
        ),
        source=source,
        intent=intent,
        ai_confidence=ai_confidence,
    )


# ── Contexts ──────────────────────────────────────────────────────────────────

def make_context(
    task: str = "pick up object",
    robot_state: RobotState | None = None,
    scene_graph: SceneGraph | None = None,
    memory_records: list[MemoryRecord] | None = None,
    safety_constraints: list[SafetyConstraint] | None = None,
    physics_horizon: PhysicsHorizon | None = None,
    confidence_floor: float = 0.60,
    humans_nearby: bool = False,
    validity_window_ms: float = 5000.0,
    **kwargs,
) -> Context:
    """Standard context: safe state, empty scene, no memory, no constraints."""
    return Context(
        task=task,
        robot_state=robot_state or make_robot_state(),
        scene_graph=scene_graph or SceneGraph(humans_nearby=humans_nearby),
        memory_records=memory_records or [],
        safety_constraints=safety_constraints or [],
        physics_horizon=physics_horizon,
        confidence_floor=confidence_floor,
        validity_window_ms=validity_window_ms,
        **kwargs,
    )


# ── Memory records ────────────────────────────────────────────────────────────

def make_memory_record(
    memory_type: MemoryType = MemoryType.SEMANTIC,
    confidence: float = 0.9,
    age_seconds: float = 60.0,
    outcome_tag: OutcomeTag = OutcomeTag.SUCCESS,
    outcome_count: int = 5,
    success_count: int = 5,
    content: dict | None = None,
    sensor_anchor: dict | None = None,
    source: str = "test_memory",
    invalid: bool = False,
) -> MemoryRecord:
    """Standard fresh, high-confidence, successful memory record."""
    valid_at = time.time() - age_seconds
    invalid_at = time.time() - 1.0 if invalid else None
    return MemoryRecord(
        source=source,
        memory_type=memory_type,
        content=content or {"fact": "test_fact", "object": "block"},
        content_text="test memory record",
        valid_at=valid_at,
        invalid_at=invalid_at,
        base_confidence=confidence,
        outcome_tag=outcome_tag,
        outcome_count=outcome_count,
        success_count=success_count,
        sensor_anchor=sensor_anchor,
    )


# ── Standard safety constraint builders ──────────────────────────────────────

def make_exclusion_zone(
    center: list[float] | None = None,
    radius_m: float = 0.2,
    constraint_id: str = "test_zone",
) -> SafetyConstraint:
    return SafetyConstraint(
        constraint_id=constraint_id,
        description="Test exclusion zone",
        constraint_type="object_exclusion",
        parameters={"center": center or [0.4, 0.0, 0.5], "radius_m": radius_m},
        veto_power=True,
        source="test",
    )


def make_feasible_horizon(stability: float = 0.4) -> PhysicsHorizon:
    return PhysicsHorizon(feasible=True, min_stability_margin=stability)


def make_infeasible_horizon() -> PhysicsHorizon:
    return PhysicsHorizon(
        feasible=False,
        min_stability_margin=0.0,
        collision_risk_steps=[2, 4, 6],
    )


# ── pytest fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def gate() -> CertificationGate:
    return CertificationGate()


@pytest.fixture
def sentinel() -> SentinelValidator:
    return SentinelValidator()


@pytest.fixture
def physicore() -> PhysiCoreValidator:
    return PhysiCoreValidator()


@pytest.fixture
def memory_validator() -> MemoryValidator:
    return MemoryValidator()


@pytest.fixture
def gateway() -> MemoryGateway:
    gw = MemoryGateway()
    gw.register(InMemoryAdapter(name="test_l1"), priority=1)
    gw.start()
    yield gw
    gw.stop()


@pytest.fixture
def clean_context() -> Context:
    return make_context()


@pytest.fixture
def estop_context() -> Context:
    return make_context(robot_state=make_robot_state(emergency_stop=True))


@pytest.fixture
def singularity_context() -> Context:
    return make_context(robot_state=make_robot_state(in_singularity=True))


@pytest.fixture
def human_context() -> Context:
    return make_context(humans_nearby=True)


@pytest.fixture
def standard_action() -> Action:
    return make_action()


# Export builders so test modules can import from conftest
__all__ = [
    "make_robot_state",
    "make_action",
    "make_context",
    "make_memory_record",
    "make_exclusion_zone",
    "make_feasible_horizon",
    "make_infeasible_horizon",
]
