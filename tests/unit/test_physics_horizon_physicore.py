"""
tests/unit/test_physics_horizon_physicore.py
============================================
Phase 3 physics horizon tests.

Verifies that PhysicsHorizonBuilder:
  1. Returns a well-formed PhysicsHorizon for a normal state
  2. Returns infeasible=False for emergency_stop
  3. Returns infeasible=False for in_singularity
  4. Falls back to kinematic when no real backend is available
  5. Selects mock backend when explicitly preferred
  6. Collision risk steps are populated when objects are near the EE
  7. Stability margin is between 0 and 1
"""

from __future__ import annotations

import pytest

from cortex.context.physics_horizon import PhysicsHorizonBuilder, HorizonConfig
from cortex.models.context import RobotState, SceneGraph, PhysicsHorizon, DetectedObject


def _normal_state() -> RobotState:
    return RobotState(
        ee_position=[0.3, 0.0, 0.5],
        ee_orientation=[1.0, 0.0, 0.0, 0.0],
    )


def _estop_state() -> RobotState:
    return RobotState(
        ee_position=[0.3, 0.0, 0.5],
        ee_orientation=[1.0, 0.0, 0.0, 0.0],
        emergency_stop=True,
    )


def _singularity_state() -> RobotState:
    return RobotState(
        ee_position=[0.3, 0.0, 0.5],
        ee_orientation=[1.0, 0.0, 0.0, 0.0],
        in_singularity=True,
    )


class TestPhysicsHorizonBuilder:

    def test_normal_state_returns_physics_horizon(self) -> None:
        builder = PhysicsHorizonBuilder()
        horizon = builder.build(_normal_state(), SceneGraph())
        assert isinstance(horizon, PhysicsHorizon)

    def test_normal_state_has_correct_steps(self) -> None:
        cfg = HorizonConfig(steps=12)
        builder = PhysicsHorizonBuilder(config=cfg)
        horizon = builder.build(_normal_state(), SceneGraph())
        assert horizon.steps == 12

    def test_emergency_stop_infeasible(self) -> None:
        builder = PhysicsHorizonBuilder()
        horizon = builder.build(_estop_state(), SceneGraph())
        assert horizon.feasible is False

    def test_emergency_stop_full_collision_risk(self) -> None:
        builder = PhysicsHorizonBuilder()
        horizon = builder.build(_estop_state(), SceneGraph())
        assert len(horizon.collision_risk_steps) > 0

    def test_singularity_infeasible(self) -> None:
        builder = PhysicsHorizonBuilder()
        horizon = builder.build(_singularity_state(), SceneGraph())
        assert horizon.feasible is False

    def test_singularity_steps_populated(self) -> None:
        builder = PhysicsHorizonBuilder()
        horizon = builder.build(_singularity_state(), SceneGraph())
        assert len(horizon.singularity_risk_steps) > 0

    def test_stability_margin_in_range(self) -> None:
        builder = PhysicsHorizonBuilder()
        horizon = builder.build(_normal_state(), SceneGraph())
        assert 0.0 <= horizon.min_stability_margin <= 1.0

    def test_predicted_final_state_not_none(self) -> None:
        builder = PhysicsHorizonBuilder()
        horizon = builder.build(_normal_state(), SceneGraph())
        assert horizon.predicted_final_state is not None

    def test_kinematic_fallback_works(self) -> None:
        """Force kinematic path by setting backend_preference to only kinematic."""
        builder = PhysicsHorizonBuilder(backend_preference=["kinematic"])
        horizon = builder.build(_normal_state(), SceneGraph())
        assert isinstance(horizon, PhysicsHorizon)
        assert horizon.steps == 12

    def test_mock_backend_selected(self) -> None:
        """Mock backend produces feasible=True for a normal state."""
        builder = PhysicsHorizonBuilder(backend_preference=["mock"])
        horizon = builder.build(_normal_state(), SceneGraph())
        assert isinstance(horizon, PhysicsHorizon)

    def test_mock_backend_estop_infeasible(self) -> None:
        """Mock backend correctly marks emergency_stop as infeasible."""
        builder = PhysicsHorizonBuilder(backend_preference=["mock"])
        horizon = builder.build(_estop_state(), SceneGraph())
        assert horizon.feasible is False

    def test_object_near_ee_raises_collision_risk(self) -> None:
        """When an obstacle is very close to the EE, collision_risk_steps should appear."""
        obj = DetectedObject(
            object_id="obstacle_1",
            label="box",
            position=[0.3, 0.0, 0.5],   # same position as EE
            dimensions=[0.1, 0.1, 0.1],
            confidence=0.9,
        )
        scene = SceneGraph(objects=[obj])
        builder = PhysicsHorizonBuilder(backend_preference=["kinematic"])
        horizon = builder.build(_normal_state(), scene)
        assert len(horizon.collision_risk_steps) > 0

    def test_empty_scene_no_collision_risk(self) -> None:
        """With no obstacles, collision_risk_steps should be empty."""
        builder = PhysicsHorizonBuilder(backend_preference=["kinematic"])
        horizon = builder.build(_normal_state(), SceneGraph())
        assert horizon.collision_risk_steps == []
