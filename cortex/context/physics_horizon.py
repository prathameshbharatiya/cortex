"""
Physics Horizon Builder
=======================
Stage 3 of Context assembly.

Generates the 12-step predictive forward simulation embedded in the CTX.
This tells the Certification Gate — before any action is proposed —
what the physics looks like in the near future given the current state.

In production this wraps a real physics engine (MuJoCo, PyBullet,
a proprietary RK4 solver, or PhysiCore's native simulation).

In Phase 3, it uses a kinematics-based approximation that gives
deterministic, testable results without external dependencies.
The interface is identical — swap the backend without touching the Gate.

What the horizon provides
-------------------------
  feasible              — is any motion physically possible right now?
  min_stability_margin  — closest approach to any limit across all steps
  collision_risk_steps  — which steps have collision risk
  singularity_risk_steps — which steps approach singularity
  predicted_final_state — estimated robot state after the horizon
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any

_log = logging.getLogger("cortex.context.physics_horizon")

import numpy as np

from cortex.models.context import RobotState, SceneGraph, PhysicsHorizon


# ── Horizon config ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class HorizonConfig:
    steps:              int   = 12
    dt_s:               float = 0.02        # 20ms per step = 50Hz control
    workspace_radius_m: float = 0.85
    max_joint_vel_rads: float = 1.5         # rad/s — typical arm limit
    max_torque_nm:      float = 180.0
    singularity_threshold: float = 0.08    # manipulability measure
    collision_clearance_m: float = 0.05    # minimum obstacle clearance


class PhysicsHorizonBuilder:
    """
    Builds the predictive physics horizon for the CTX.

    Called during Context assembly — before the AI proposes any action.
    The horizon tells the Gate what the physics envelope looks like
    so it can reject or constrain actions that approach limits.

    Backend selection (in order of preference):
      1. MuJoCo — if mujoco>=3.0 is installed
      2. PyBullet — if pybullet is installed
      3. Kinematic approximation — always available (CPU-only fallback)

    The kinematic fallback is deterministic, zero-dependency, and sufficient
    for development.  Production deployments should use a real physics backend.
    """

    def __init__(
        self,
        config: HorizonConfig | None = None,
        backend_preference: list[str] | None = None,
    ) -> None:
        self.cfg = config or HorizonConfig()
        self._backend = self._load_best_backend(
            backend_preference or ["mujoco", "pybullet", "kinematic"]
        )

    def _load_best_backend(self, preference: list[str]) -> "Any | None":
        """Try backends in preference order, return first available non-kinematic one."""
        for name in preference:
            if name == "kinematic":
                return None  # signals use of built-in kinematic path
            try:
                from cortex.physicore_engine.backends import load_backend, BackendError
                backend = load_backend(name)
                if backend.is_available():
                    return backend
            except Exception:
                pass
        return None  # fall back to kinematic

    def build(
        self,
        robot_state: RobotState,
        scene_graph: SceneGraph,
        target_position: list[float] | None = None,
    ) -> PhysicsHorizon:
        """
        Generate the physics horizon from current robot state.

        target_position: optional hint about intended direction of motion.
        If None, horizon is built around current state stability only.
        """
        t0 = time.perf_counter()

        # Immediate feasibility checks
        if robot_state.emergency_stop:
            return PhysicsHorizon(
                steps=self.cfg.steps,
                feasible=False,
                min_stability_margin=0.0,
                predicted_final_state={"reason": "emergency_stop"},
                collision_risk_steps=list(range(self.cfg.steps)),
                singularity_risk_steps=[],
            )

        if robot_state.in_singularity:
            return PhysicsHorizon(
                steps=self.cfg.steps,
                feasible=False,
                min_stability_margin=0.0,
                predicted_final_state={"reason": "in_singularity"},
                collision_risk_steps=[],
                singularity_risk_steps=list(range(self.cfg.steps)),
            )

        # Try real physics backend first, fall back to kinematic approximation
        if self._backend is not None and self._backend.is_available():
            return self._build_with_backend(robot_state, target_position)
        return self._simulate(robot_state, scene_graph, target_position)

    def _build_with_backend(
        self,
        robot_state:     RobotState,
        target_position: list[float] | None,
    ) -> PhysicsHorizon:
        """Delegate to a real physics backend (MuJoCo or PyBullet)."""
        try:
            result = self._backend.simulate_horizon(
                robot_state,
                n_steps=self.cfg.steps,
                dt=self.cfg.dt_s,
                target_position=target_position,
            )
            return PhysicsHorizon(
                steps=self.cfg.steps,
                feasible=result.feasible,
                min_stability_margin=result.min_stability_margin,
                predicted_final_state=result.predicted_final_state,
                collision_risk_steps=result.collision_risk_steps,
                singularity_risk_steps=result.singularity_risk_steps,
            )
        except Exception as exc:
            _log.warning(
                "Physics backend failed, falling back to kinematic approximation",
                extra={"error": str(exc)},
            )
            from cortex.models.context import SceneGraph
            return self._simulate(robot_state, SceneGraph(), target_position)

    # ── Forward simulation ────────────────────────────────────────────────────

    def _simulate(
        self,
        robot_state:     RobotState,
        scene_graph:     SceneGraph,
        target_position: list[float] | None,
    ) -> PhysicsHorizon:
        """
        12-step kinematic forward simulation.

        Each step represents dt_s seconds of motion at current velocity.
        Checks workspace, torque, collision, and singularity at each step.
        """
        joints      = list(robot_state.joint_positions) or [0.0] * 6
        velocities  = list(robot_state.joint_velocities) or [0.0] * len(joints)
        torques     = list(robot_state.joint_torques) or [0.0] * len(joints)
        ee_pos      = list(robot_state.ee_position)

        stability_margins: list[float] = []
        collision_risk_steps:    list[int] = []
        singularity_risk_steps:  list[int] = []

        for step in range(self.cfg.steps):
            # ── Workspace boundary margin ─────────────────────────────────────
            dist_from_origin = float(np.linalg.norm(ee_pos))
            workspace_margin = max(0.0,
                (self.cfg.workspace_radius_m - dist_from_origin) / self.cfg.workspace_radius_m
            )
            stability_margins.append(workspace_margin)

            # ── Torque margin ─────────────────────────────────────────────────
            if torques:
                max_torque = max(abs(t) for t in torques)
                torque_margin = max(0.0, 1.0 - max_torque / self.cfg.max_torque_nm)
                stability_margins.append(torque_margin)

            # ── Velocity margin ───────────────────────────────────────────────
            if velocities:
                max_vel = max(abs(v) for v in velocities)
                vel_margin = max(0.0, 1.0 - max_vel / self.cfg.max_joint_vel_rads)
                stability_margins.append(vel_margin)

            # ── Singularity proximity ─────────────────────────────────────────
            manipulability = self._manipulability(joints)
            if manipulability < self.cfg.singularity_threshold:
                singularity_risk_steps.append(step)
            sing_margin = min(1.0, manipulability / (self.cfg.singularity_threshold * 3))
            stability_margins.append(sing_margin)

            # ── Collision proximity ───────────────────────────────────────────
            coll_margin = self._collision_margin(ee_pos, scene_graph)
            stability_margins.append(coll_margin)
            if coll_margin < 0.2:
                collision_risk_steps.append(step)

            # ── Propagate state one step ──────────────────────────────────────
            joints, velocities, ee_pos = self._step_kinematics(
                joints, velocities, ee_pos, target_position
            )

        min_margin = min(stability_margins) if stability_margins else 0.5
        feasible   = (
            min_margin > 0.0
            and len(collision_risk_steps) == 0
            and len(singularity_risk_steps) < self.cfg.steps // 2
        )

        return PhysicsHorizon(
            steps=self.cfg.steps,
            feasible=feasible,
            min_stability_margin=round(min_margin, 4),
            collision_risk_steps=collision_risk_steps,
            singularity_risk_steps=singularity_risk_steps,
            predicted_final_state={
                "ee_position":        ee_pos,
                "joint_positions":    joints,
                "simulation_steps":   self.cfg.steps,
                "dt_s":               self.cfg.dt_s,
            },
        )

    # ── Kinematic helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _manipulability(joints: list[float]) -> float:
        """
        Approximate manipulability measure.
        Near-zero = near singularity.
        Production uses full Jacobian determinant.
        """
        if not joints:
            return 1.0
        sines = [abs(math.sin(j)) for j in joints]
        product = math.prod([max(s, 0.01) for s in sines])
        return product ** (1.0 / len(joints))

    def _collision_margin(
        self, ee_pos: list[float], scene_graph: SceneGraph
    ) -> float:
        """Fraction of clearance budget remaining to nearest obstacle."""
        if not scene_graph.objects:
            return 1.0

        ee = np.array(ee_pos)
        min_clearance = float("inf")
        for obj in scene_graph.objects:
            if getattr(obj, "is_grasped", False):
                continue
            obj_pos = np.array(obj.position)
            obj_radius = max(obj.dimensions) / 2.0 if obj.dimensions else 0.05
            clearance = float(np.linalg.norm(ee - obj_pos)) - obj_radius
            min_clearance = min(min_clearance, clearance)

        if min_clearance == float("inf"):
            return 1.0

        return max(0.0, min(1.0, min_clearance / (self.cfg.collision_clearance_m * 3)))

    @staticmethod
    def _step_kinematics(
        joints:          list[float],
        velocities:      list[float],
        ee_pos:          list[float],
        target_position: list[float] | None,
        dt:              float = 0.02,
        damping:         float = 0.95,
    ) -> tuple[list[float], list[float], list[float]]:
        """
        Single kinematic step: integrate velocities, damp, propagate EE.
        This is a placeholder for a real FK solver in production.
        """
        new_joints = [j + v * dt for j, v in zip(joints, velocities)]
        new_vels   = [v * damping for v in velocities]

        # Approximate EE displacement from first two joints (shoulder + elbow)
        if len(new_joints) >= 2:
            dx = math.sin(new_joints[0]) * 0.4
            dy = math.cos(new_joints[0]) * 0.4
            dz = math.sin(new_joints[1]) * 0.3
            # Blend toward target if provided
            if target_position:
                tgt = np.array(target_position)
                cur = np.array(ee_pos)
                direction = tgt - cur
                norm = float(np.linalg.norm(direction))
                if norm > 0.001:
                    direction = direction / norm * 0.01   # 1cm step toward target
                    new_ee = (cur + direction).tolist()
                else:
                    new_ee = ee_pos
            else:
                new_ee = [
                    ee_pos[0] + dx * dt * 0.1,
                    ee_pos[1] + dy * dt * 0.1,
                    ee_pos[2] + dz * dt * 0.1,
                ]
        else:
            new_ee = ee_pos

        return new_joints, new_vels, new_ee
