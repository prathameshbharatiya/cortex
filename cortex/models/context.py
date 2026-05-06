"""
Context models
==============
The Context (CTX) is the single validated representation of everything
Cortex knows about the current situation. It is the mandatory input to
the Certification Gate — no action can be certified without one.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from cortex.models.memory import MemoryRecord


class RobotState(BaseModel):
    """
    Live proprioceptive state of the robot.
    This is ground truth from sensors — not from memory.
    """

    # End-effector
    ee_position:    list[float] = Field(description="EE position [x, y, z] metres")
    ee_orientation: list[float] = Field(description="EE orientation quaternion [w, x, y, z]")
    ee_velocity:    list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])

    # Joints
    joint_positions:     list[float] = Field(default_factory=list, description="Joint angles rad")
    joint_velocities:    list[float] = Field(default_factory=list)
    joint_torques:       list[float] = Field(default_factory=list)

    # Gripper
    gripper_open:        bool  = Field(default=True)
    gripper_force_n:     float = Field(default=0.0, description="Current contact force N")

    # Robot-level flags
    is_moving:           bool  = Field(default=False)
    in_singularity:      bool  = Field(default=False)
    emergency_stop:      bool  = Field(default=False)

    timestamp: float = Field(default_factory=time.time)

    model_config = {"frozen": True}

    @property
    def ee_pos_array(self) -> np.ndarray:
        return np.array(self.ee_position)

    @property
    def is_safe_to_move(self) -> bool:
        return not self.emergency_stop and not self.in_singularity


class DetectedObject(BaseModel):
    """An object detected in the current scene."""

    object_id:   str
    label:       str
    position:    list[float] = Field(description="[x, y, z] metres in world frame")
    dimensions:  list[float] = Field(description="[width, height, depth] metres")
    confidence:  float       = Field(ge=0.0, le=1.0)
    is_grasped:  bool        = Field(default=False)
    is_occluded: bool        = Field(default=False)

    model_config = {"frozen": True}

    @property
    def pos_array(self) -> np.ndarray:
        return np.array(self.position)


class SceneGraph(BaseModel):
    """Structured representation of what is in the environment right now."""

    objects:      list[DetectedObject] = Field(default_factory=list)
    workspace_bounds: dict[str, float] = Field(
        default_factory=lambda: {
            "x_min": -1.0, "x_max": 1.0,
            "y_min": -1.0, "y_max": 1.0,
            "z_min":  0.0, "z_max": 2.0,
        }
    )
    humans_nearby: bool = Field(default=False)
    obstruction_detected: bool = Field(default=False)
    timestamp: float = Field(default_factory=time.time)

    model_config = {"frozen": True}

    def get_object(self, object_id: str) -> DetectedObject | None:
        return next((o for o in self.objects if o.object_id == object_id), None)

    def objects_within_radius(self, position: list[float], radius_m: float) -> list[DetectedObject]:
        pos = np.array(position)
        return [
            o for o in self.objects
            if float(np.linalg.norm(o.pos_array - pos)) <= radius_m
        ]


class SafetyConstraint(BaseModel):
    """A hard safety constraint that must not be violated."""

    constraint_id: str
    description:   str
    constraint_type: str = Field(
        description=(
            "Type: 'workspace_boundary' | 'speed_limit' | 'force_limit' "
            "| 'human_proximity' | 'object_exclusion' | 'custom'"
        )
    )
    parameters: dict[str, Any] = Field(default_factory=dict)
    veto_power: bool = Field(
        default=True,
        description="If True, violation blocks execution absolutely (Sentinel-grade)"
    )
    source: str = Field(default="sentinel", description="Which system issued this constraint")

    model_config = {"frozen": True}


class PhysicsHorizon(BaseModel):
    """
    12-step predictive lookahead from PhysiCore.
    Encodes whether the action is physically feasible across future control steps.
    """

    steps: int = Field(default=12)
    feasible: bool = Field(description="True if all steps are within physical limits")
    min_stability_margin: float = Field(
        ge=0.0,
        description="Closest approach to a physical limit across all steps (fraction)"
    )
    predicted_final_state: dict[str, Any] = Field(default_factory=dict)
    collision_risk_steps: list[int] = Field(
        default_factory=list,
        description="Step indices where collision risk was detected"
    )
    singularity_risk_steps: list[int] = Field(
        default_factory=list,
        description="Step indices where kinematic singularity risk was detected"
    )

    model_config = {"frozen": True}


class Context(BaseModel):
    """
    The Context Packet (CTX) — the validated, unified world-state representation.

    This is the mandatory input to cortex.certify().
    It is assembled by ContextBuilder and is immutable once created.
    The AI planner receives this and uses it to generate action proposals.
    The Certification Gate uses it to validate those proposals.
    """

    ctx_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    # ── What is the goal? ────────────────────────────────────────────────────
    task: str = Field(description="Current task description")

    # ── What is the current physical state? ─────────────────────────────────
    robot_state: RobotState
    scene_graph: SceneGraph = Field(default_factory=SceneGraph)

    # ── What does memory say? ────────────────────────────────────────────────
    memory_records: list[MemoryRecord] = Field(default_factory=list)
    memory_confidence: float = Field(
        ge=0.0, le=1.0,
        default=1.0,
        description="Overall confidence in the assembled memory basis"
    )

    # ── What are the hard rules? ─────────────────────────────────────────────
    safety_constraints: list[SafetyConstraint] = Field(default_factory=list)

    # ── What does physics say about the near future? ─────────────────────────
    physics_horizon: PhysicsHorizon | None = Field(default=None)

    # ── Meta ─────────────────────────────────────────────────────────────────
    confidence_floor: float = Field(
        ge=0.0, le=1.0,
        default=0.60,
        description="Minimum confidence for an EXECUTE decision. Below this → REPLAN."
    )
    validity_window_ms: float = Field(
        default=500.0,
        description="How long this context is valid before requiring reassembly"
    )
    assembly_latency_ms: float = Field(default=0.0)
    timestamp: float = Field(default_factory=time.time)

    model_config = {"frozen": True}

    @property
    def has_physics_horizon(self) -> bool:
        return self.physics_horizon is not None

    @property
    def physics_feasible(self) -> bool:
        return self.physics_horizon is None or self.physics_horizon.feasible

    @property
    def has_human_nearby(self) -> bool:
        return self.scene_graph.humans_nearby

    @property
    def absolute_constraints(self) -> list[SafetyConstraint]:
        """Safety constraints with veto power — Sentinel-grade."""
        return [c for c in self.safety_constraints if c.veto_power]

    @property
    def is_expired(self) -> bool:
        age_ms = (time.time() - self.timestamp) * 1000
        return age_ms > self.validity_window_ms

    def __repr__(self) -> str:
        return (
            f"Context("
            f"task={self.task!r:.30}, "
            f"memory_records={len(self.memory_records)}, "
            f"constraints={len(self.safety_constraints)}, "
            f"physics={'ok' if self.physics_feasible else 'INFEASIBLE'}"
            f")"
        )
