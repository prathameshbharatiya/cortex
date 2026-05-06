"""
State Relevance Scorer
======================
Axis 2 of the Relevance Engine.

Answers: is this memory applicable to the robot's current physical state?

A memory of successfully picking an object from the left side of the
workspace is not directly applicable when the robot is currently
positioned on the right side — the joint configuration is different,
the approach angle differs, and the relevant workspace region differs.

This scorer evaluates:
  1. EE proximity     — how close is the memory's EE context to current EE?
  2. Joint similarity — how similar is the current joint config to the
                        configuration when this memory was formed?
  3. Workspace region — are we in the same region of the workspace?
  4. Gripper state    — does the memory assume the same gripper state?

Records with no spatial content score 0.6 (neutral — can't confirm or deny).
Records with spatial content that matches score close to 1.0.
Records with spatial content that doesn't match score closer to 0.0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from cortex.models.memory  import MemoryRecord, MemoryType
from cortex.models.context import RobotState


# ── Workspace regions (simplified) ───────────────────────────────────────────
# A 6-region partition of the workspace by quadrant and height.
# In production this uses a finer spatial index.

def _workspace_region(pos: list[float]) -> str:
    x, y, z = pos[0], pos[1], pos[2] if len(pos) > 2 else 0.5
    h = "high" if z > 0.6 else ("mid" if z > 0.3 else "low")
    q = ("right" if x > 0 else "left") + "_" + ("front" if y > 0 else "back")
    return f"{q}_{h}"


@dataclass(frozen=True)
class StateScore:
    record_id:        str
    ee_proximity:     float   # EE position match [0, 1]
    joint_similarity: float   # Joint config match [0, 1]
    region_match:     float   # Workspace region match [0, 1]
    gripper_match:    float   # Gripper state match [0, 1]
    composite:        float   # weighted composite [0, 1]


class StateRelevanceScorer:
    """
    Scores how applicable a memory record is to the current robot state.

    Weights:
      ee_proximity     — 0.45  (strongest spatial signal)
      joint_similarity — 0.25  (kinematic configuration)
      region_match     — 0.20  (coarse workspace region)
      gripper_match    — 0.10  (gripper open/closed)
    """

    def __init__(
        self,
        w_ee:      float = 0.45,
        w_joint:   float = 0.25,
        w_region:  float = 0.20,
        w_gripper: float = 0.10,
        ee_match_radius_m: float  = 0.15,   # within 15cm = strong match
        joint_match_rad:   float  = 0.30,   # within 0.3 rad per joint = match
    ) -> None:
        total = w_ee + w_joint + w_region + w_gripper
        self.w_ee      = w_ee      / total
        self.w_joint   = w_joint   / total
        self.w_region  = w_region  / total
        self.w_gripper = w_gripper / total

        self.ee_match_radius  = ee_match_radius_m
        self.joint_match_rad  = joint_match_rad

    def score(self, record: MemoryRecord, robot_state: RobotState) -> StateScore:
        content = record.content if isinstance(record.content, dict) else {}

        # ── EE proximity ──────────────────────────────────────────────────────
        ee_proximity = self._score_ee_proximity(content, robot_state)

        # ── Joint similarity ──────────────────────────────────────────────────
        joint_similarity = self._score_joints(content, robot_state)

        # ── Workspace region ──────────────────────────────────────────────────
        region_match = self._score_region(content, robot_state)

        # ── Gripper state ─────────────────────────────────────────────────────
        gripper_match = self._score_gripper(content, robot_state)

        composite = (
            self.w_ee      * ee_proximity
            + self.w_joint   * joint_similarity
            + self.w_region  * region_match
            + self.w_gripper * gripper_match
        )

        return StateScore(
            record_id=record.record_id,
            ee_proximity=round(ee_proximity, 4),
            joint_similarity=round(joint_similarity, 4),
            region_match=round(region_match, 4),
            gripper_match=round(gripper_match, 4),
            composite=round(composite, 4),
        )

    # ── Individual scorers ────────────────────────────────────────────────────

    def _score_ee_proximity(
        self, content: dict, robot_state: RobotState
    ) -> float:
        # Try known keys for EE position in memory content
        ee_keys = ["ee_position", "position", "ee_pos", "robot_ee"]
        claimed_ee = None
        for key in ee_keys:
            if key in content and isinstance(content[key], (list, tuple)):
                claimed_ee = content[key]
                break

        if claimed_ee is None:
            return 0.65   # no EE context — neutral

        try:
            claimed = np.array(claimed_ee[:3], dtype=float)
            current = np.array(robot_state.ee_position[:3], dtype=float)
            dist    = float(np.linalg.norm(claimed - current))
            # Score decays from 1.0 at dist=0 to 0.1 at dist=ee_match_radius*3
            score = max(0.1, 1.0 - dist / (self.ee_match_radius * 3))
            if dist <= self.ee_match_radius:
                score = max(score, 0.75)   # within radius: guaranteed at least 0.75
            return score
        except (ValueError, TypeError):
            return 0.65

    def _score_joints(
        self, content: dict, robot_state: RobotState
    ) -> float:
        joint_keys = ["joint_positions", "joints", "joint_angles"]
        claimed_joints = None
        for key in joint_keys:
            if key in content and isinstance(content[key], (list, tuple)):
                claimed_joints = content[key]
                break

        if claimed_joints is None or not robot_state.joint_positions:
            return 0.65   # no joint context — neutral

        try:
            claimed = np.array(claimed_joints, dtype=float)
            current = np.array(robot_state.joint_positions, dtype=float)
            n       = min(len(claimed), len(current))
            if n == 0:
                return 0.65
            diffs  = np.abs(claimed[:n] - current[:n])
            within = float(np.mean(diffs < self.joint_match_rad))
            return max(0.1, within)
        except (ValueError, TypeError):
            return 0.65

    def _score_region(
        self, content: dict, robot_state: RobotState
    ) -> float:
        # Check if memory records a workspace region
        if "workspace_region" in content:
            mem_region = content["workspace_region"]
            cur_region = _workspace_region(robot_state.ee_position)
            return 1.0 if mem_region == cur_region else 0.3

        # Derive region from position if available
        pos_keys = ["ee_position", "position", "ee_pos"]
        for key in pos_keys:
            if key in content and isinstance(content[key], (list, tuple)):
                try:
                    mem_region = _workspace_region(list(content[key]))
                    cur_region = _workspace_region(robot_state.ee_position)
                    return 1.0 if mem_region == cur_region else 0.4
                except (ValueError, TypeError):
                    pass

        return 0.65   # no region info — neutral

    @staticmethod
    def _score_gripper(content: dict, robot_state: RobotState) -> float:
        gripper_keys = ["gripper_open", "gripper_state", "is_grasping"]
        for key in gripper_keys:
            if key in content:
                mem_val = content[key]
                # Normalise to bool
                if isinstance(mem_val, bool):
                    mem_open = mem_val
                elif isinstance(mem_val, str):
                    mem_open = mem_val.lower() in ("open", "true", "yes")
                elif isinstance(mem_val, (int, float)):
                    mem_open = mem_val > 0.5
                else:
                    continue
                cur_open = robot_state.gripper_open
                return 1.0 if mem_open == cur_open else 0.3

        return 0.7   # no gripper info — slightly positive (most memories assume open)
