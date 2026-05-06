"""
PyBullet Physics Backend
========================
Rigid-body dynamics simulation using PyBullet.
Good dynamics model, widely available, GPU-optional.

Requirements:
  pip install pybullet

Usage:
    from cortex.physicore_engine.backends import load_backend
    backend = load_backend("pybullet", urdf_path="ur5e.urdf")
    result  = backend.simulate_horizon(state, n_steps=12, dt=0.02)
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from cortex.models.context import RobotState
from cortex.physicore_engine.backends.base import PhysicsBackend, HorizonResult, BackendError


class PyBulletBackend(PhysicsBackend):
    """
    PyBullet rigid-body dynamics backend.

    Loads a robot URDF (or uses a built-in KUKA iiwa model as fallback)
    and runs PyBullet forward dynamics to compute the physics horizon.

    Requires: pip install pybullet
    """

    def __init__(
        self,
        urdf_path:          str | None = None,
        workspace_radius_m: float      = 0.85,
        max_torque_nm:      float      = 180.0,
        use_gui:            bool       = False,
    ) -> None:
        self._urdf_path        = urdf_path
        self._workspace_radius = workspace_radius_m
        self._max_torque       = max_torque_nm
        self._use_gui          = use_gui
        self._physics_client: Any | None = None
        self._robot_id: int | None       = None
        self._available: bool | None     = None

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            import pybullet  # noqa: F401
            self._available = True
        except ImportError:
            self._available = False
        return self._available

    def _ensure_loaded(self) -> None:
        if self._physics_client is not None:
            return
        if not self.is_available():
            raise BackendError(
                "PyBullet is not installed. Install with: pip install pybullet"
            )
        import pybullet as pb
        import pybullet_data

        mode = pb.GUI if self._use_gui else pb.DIRECT
        self._physics_client = pb.connect(mode)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        pb.setGravity(0, 0, -9.81)

        urdf = self._urdf_path or "kuka_iiwa/model.urdf"
        self._robot_id = pb.loadURDF(
            urdf,
            basePosition=[0, 0, 0],
            useFixedBase=True,
        )

    def simulate_horizon(
        self,
        robot_state:     RobotState,
        n_steps:         int   = 12,
        dt:              float = 0.02,
        target_position: list[float] | None = None,
    ) -> HorizonResult:
        if not self.is_available():
            raise BackendError(
                "PyBullet is not installed. Install with: pip install pybullet"
            )
        t0 = time.perf_counter()
        try:
            return self._pybullet_simulate(robot_state, n_steps, dt, target_position, t0)
        except BackendError:
            raise
        except Exception as exc:
            raise BackendError(f"PyBullet simulation failed: {exc}") from exc

    def _pybullet_simulate(
        self,
        robot_state:     RobotState,
        n_steps:         int,
        dt:              float,
        target_position: list[float] | None,
        t0:              float,
    ) -> HorizonResult:
        import pybullet as pb

        self._ensure_loaded()
        robot = self._robot_id

        # Set joint states
        n_joints = pb.getNumJoints(robot)
        n_state  = min(len(robot_state.joint_positions), n_joints)
        for ji in range(n_state):
            pb.resetJointState(
                robot, ji,
                targetValue=robot_state.joint_positions[ji],
                targetVelocity=(
                    robot_state.joint_velocities[ji]
                    if ji < len(robot_state.joint_velocities)
                    else 0.0
                ),
            )

        pb.setTimeStep(dt)

        stability_margins: list[float] = []
        collision_risk_steps:   list[int] = []
        singularity_risk_steps: list[int] = []

        for step in range(n_steps):
            pb.stepSimulation()

            # EE position (last link)
            link_state = pb.getLinkState(robot, n_joints - 1)
            ee_pos = np.array(link_state[0])

            # Workspace margin
            dist      = float(np.linalg.norm(ee_pos))
            ws_margin = max(0.0, (self._workspace_radius - dist) / self._workspace_radius)
            stability_margins.append(ws_margin)

            # Joint torques
            joint_states = pb.getJointStates(robot, range(n_joints))
            for js in joint_states:
                torque = abs(js[3]) if len(js) > 3 else 0.0
                t_margin = max(0.0, 1.0 - torque / self._max_torque)
                stability_margins.append(t_margin)

            # Collision contacts
            contacts = pb.getContactPoints(robot)
            if contacts:
                collision_risk_steps.append(step)
                stability_margins.append(0.1)
            else:
                stability_margins.append(1.0)

        min_margin = min(stability_margins) if stability_margins else 0.5
        feasible   = min_margin > 0.0 and len(collision_risk_steps) < n_steps // 3

        # Final EE position
        link_state = pb.getLinkState(robot, n_joints - 1)
        final_state = {
            "ee_position": list(link_state[0]),
            "simulation_steps": n_steps,
            "backend": "pybullet",
        }

        return HorizonResult(
            feasible=feasible,
            min_stability_margin=round(min_margin, 4),
            collision_risk_steps=collision_risk_steps,
            singularity_risk_steps=singularity_risk_steps,
            predicted_final_state=final_state,
            backend_name="pybullet",
            simulation_time_ms=(time.perf_counter() - t0) * 1000,
        )

    def __del__(self) -> None:
        if self._physics_client is not None:
            try:
                import pybullet as pb
                pb.disconnect(self._physics_client)
            except Exception:
                pass
