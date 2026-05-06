"""
Action models
=============
Represents any action proposed by an AI planning system.
Cortex is model-agnostic — it accepts actions from any source.
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel, Field, field_validator


class ActionType(str, Enum):
    MOVE_EE         = "move_ee"          # Move end-effector to pose
    MOVE_JOINT      = "move_joint"       # Move to joint configuration
    GRASP           = "grasp"            # Close gripper on object
    RELEASE         = "release"          # Open gripper
    PUSH            = "push"             # Push object along surface
    PLACE           = "place"            # Place held object at pose
    NAVIGATE        = "navigate"         # Mobile base navigation
    COMPOSITE       = "composite"        # Multi-step sequence
    CUSTOM          = "custom"           # Platform-specific action


class Pose(BaseModel):
    """6-DOF pose: position (x,y,z metres) + orientation (quaternion w,x,y,z)."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    qw: float = 1.0
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0

    model_config = {"frozen": True}

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    @property
    def quaternion(self) -> np.ndarray:
        return np.array([self.qw, self.qx, self.qy, self.qz])

    def distance_to(self, other: "Pose") -> float:
        return float(np.linalg.norm(self.position - other.position))


class JointConfiguration(BaseModel):
    """Robot joint angles in radians."""
    angles: list[float] = Field(description="Joint angles (radians), ordered from base to EE")
    velocities: list[float] = Field(default_factory=list)
    accelerations: list[float] = Field(default_factory=list)

    model_config = {"frozen": True}

    @field_validator("angles")
    @classmethod
    def angles_not_empty(cls, v: list[float]) -> list[float]:
        if not v:
            raise ValueError("Joint angles cannot be empty")
        return v


class ActionConstraints(BaseModel):
    """
    Execution parameters and hard limits for this action.
    These are the limits the AI proposes — Cortex may tighten them further.
    """
    max_speed_ms:       float | None = Field(default=None, description="Max EE speed m/s")
    max_force_n:        float | None = Field(default=None, description="Max contact force N")
    max_torque_nm:      float | None = Field(default=None, description="Max joint torque Nm")
    timeout_ms:         float | None = Field(default=5000.0, description="Action timeout ms")
    abort_on_contact:   bool         = Field(default=False)
    require_confirmation: bool       = Field(default=False)

    model_config = {"frozen": True}


class ActionSpec(BaseModel):
    """Full specification of what an action intends to do."""

    action_type: ActionType
    target_pose: Pose | None = Field(default=None)
    target_joints: JointConfiguration | None = Field(default=None)
    target_object_id: str | None = Field(default=None)
    trajectory: list[Pose] | None = Field(default=None)
    constraints: ActionConstraints = Field(default_factory=ActionConstraints)
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Platform-specific extra parameters"
    )

    model_config = {"frozen": True}


class Action(BaseModel):
    """
    An action proposed by an AI system, ready for Cortex certification.
    """

    action_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    spec: ActionSpec
    source: str = Field(
        default="unknown",
        description="Which AI system proposed this action (for audit)"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        default=1.0,
        description="AI's own confidence in this proposal (not Cortex's assessment)"
    )
    intent: str = Field(
        default="",
        description="Human-readable description of what this action tries to achieve"
    )

    model_config = {"frozen": True}

    def with_speed_limit(self, max_speed_ms: float) -> "Action":
        """Return a copy of this action with a reduced speed limit."""
        new_constraints = self.spec.constraints.model_copy(
            update={"max_speed_ms": min(
                max_speed_ms,
                self.spec.constraints.max_speed_ms or max_speed_ms
            )}
        )
        new_spec = self.spec.model_copy(update={"constraints": new_constraints})
        return self.model_copy(update={"spec": new_spec})

    def with_force_limit(self, max_force_n: float) -> "Action":
        """Return a copy of this action with a reduced force limit."""
        new_constraints = self.spec.constraints.model_copy(
            update={"max_force_n": min(
                max_force_n,
                self.spec.constraints.max_force_n or max_force_n
            )}
        )
        new_spec = self.spec.model_copy(update={"constraints": new_constraints})
        return self.model_copy(update={"spec": new_spec})

    def __repr__(self) -> str:
        return (
            f"Action(type={self.spec.action_type.value}, "
            f"source={self.source!r}, "
            f"id={self.action_id[:8]})"
        )
