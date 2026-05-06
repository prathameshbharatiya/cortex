"""
MuJoCo Physics Backend
======================
Full rigid-body dynamics simulation using MuJoCo >= 3.0.
Highest fidelity simulation backend available.

Requirements:
  pip install mujoco>=3.0

If a robot URDF/MJCF is provided, the simulation uses the actual robot model.
Without a model, falls back to a generic 6-DOF arm model for feasibility checks.

Usage:
    from cortex.physicore_engine.backends import load_backend
    backend = load_backend("mujoco", urdf_path="ur5e.urdf")
    result  = backend.simulate_horizon(state, n_steps=12, dt=0.02)
"""

from __future__ import annotations

import time
import math
from typing import Any

import numpy as np

from cortex.models.context import RobotState
from cortex.physicore_engine.backends.base import PhysicsBackend, HorizonResult, BackendError


# Generic 6-DOF arm MJCF model used when no robot model is provided
_GENERIC_ARM_MJCF = """
<mujoco model="cortex_generic_arm">
  <compiler angle="radian"/>
  <option timestep="0.02" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="link0" pos="0 0 0">
      <joint name="joint0" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
      <geom type="cylinder" size="0.05 0.2" mass="1.0"/>
      <body name="link1" pos="0 0 0.4">
        <joint name="joint1" type="hinge" axis="0 1 0" range="-2.5 2.5"/>
        <geom type="cylinder" size="0.04 0.2" mass="0.8"/>
        <body name="link2" pos="0 0 0.4">
          <joint name="joint2" type="hinge" axis="0 1 0" range="-2.5 2.5"/>
          <geom type="cylinder" size="0.035 0.15" mass="0.6"/>
          <body name="link3" pos="0 0 0.3">
            <joint name="joint3" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
            <geom type="cylinder" size="0.03 0.12" mass="0.4"/>
            <body name="link4" pos="0 0 0.25">
              <joint name="joint4" type="hinge" axis="0 1 0" range="-2.5 2.5"/>
              <geom type="cylinder" size="0.025 0.1" mass="0.3"/>
              <body name="end_effector" pos="0 0 0.2">
                <joint name="joint5" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
                <geom type="sphere" size="0.02" mass="0.1"/>
                <site name="ee_site" pos="0 0 0.05"/>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


class MuJoCoBackend(PhysicsBackend):
    """
    MuJoCo rigid-body dynamics backend.

    Loads a robot model (MJCF or URDF) and runs forward dynamics simulation
    to compute the physics horizon. Provides collision detection, joint limit
    checking, and stability analysis.

    Requires: pip install mujoco>=3.0
    """

    def __init__(
        self,
        urdf_path: str | None = None,
        mjcf_str:  str | None = None,
        workspace_radius_m: float = 0.85,
        max_torque_nm:      float = 180.0,
    ) -> None:
        self._urdf_path         = urdf_path
        self._mjcf_str          = mjcf_str or _GENERIC_ARM_MJCF
        self._workspace_radius  = workspace_radius_m
        self._max_torque        = max_torque_nm
        self._model: Any        = None
        self._data:  Any        = None
        self._available: bool | None = None

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            import mujoco  # noqa: F401
            self._available = True
        except ImportError:
            self._available = False
        return self._available

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if not self.is_available():
            raise BackendError(
                "MuJoCo is not installed. Install with: pip install mujoco>=3.0"
            )
        import mujoco
        if self._urdf_path:
            self._model = mujoco.MjModel.from_xml_path(self._urdf_path)
        else:
            self._model = mujoco.MjModel.from_xml_string(self._mjcf_str)
        self._data = mujoco.MjData(self._model)

    def simulate_horizon(
        self,
        robot_state:     RobotState,
        n_steps:         int   = 12,
        dt:              float = 0.02,
        target_position: list[float] | None = None,
    ) -> HorizonResult:
        if not self.is_available():
            raise BackendError(
                "MuJoCo is not installed. Install with: pip install mujoco>=3.0"
            )

        t0 = time.perf_counter()

        try:
            return self._mujoco_simulate(robot_state, n_steps, dt, target_position)
        except BackendError:
            raise
        except Exception as exc:
            raise BackendError(f"MuJoCo simulation failed: {exc}") from exc
        finally:
            pass

    def _mujoco_simulate(
        self,
        robot_state:     RobotState,
        n_steps:         int,
        dt:              float,
        target_position: list[float] | None,
    ) -> HorizonResult:
        import mujoco

        t0 = time.perf_counter()
        self._ensure_loaded()

        model = self._model
        data  = self._data

        # Reset data
        mujoco.mj_resetData(model, data)

        # Set timestep
        model.opt.timestep = dt

        # Initialise joint positions from robot state
        n_joints = min(len(robot_state.joint_positions), model.nq)
        for i in range(n_joints):
            data.qpos[i] = robot_state.joint_positions[i]

        n_vels = min(len(robot_state.joint_velocities), model.nv)
        for i in range(n_vels):
            data.qvel[i] = robot_state.joint_velocities[i]

        # Forward kinematics to sync
        mujoco.mj_forward(model, data)

        stability_margins: list[float] = []
        collision_risk_steps:   list[int] = []
        singularity_risk_steps: list[int] = []

        for step in range(n_steps):
            # Step dynamics
            mujoco.mj_step(model, data)

            # ── Workspace margin ──────────────────────────────────────────
            # Find end-effector site if it exists
            ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
            if ee_site_id >= 0:
                ee_pos = data.site_xpos[ee_site_id].copy()
            else:
                ee_pos = data.qpos[:3] if model.nq >= 3 else np.zeros(3)

            dist = float(np.linalg.norm(ee_pos))
            ws_margin = max(0.0, (self._workspace_radius - dist) / self._workspace_radius)
            stability_margins.append(ws_margin)

            # ── Joint limits ──────────────────────────────────────────────
            if model.nq > 0:
                for ji in range(model.nq):
                    lo = model.jnt_range[ji, 0]
                    hi = model.jnt_range[ji, 1]
                    if hi > lo:
                        pos = data.qpos[ji]
                        range_size = hi - lo
                        from_lo = (pos - lo) / range_size
                        from_hi = (hi - pos) / range_size
                        margin = min(from_lo, from_hi) * 2.0  # 0=at limit, 1=at center
                        stability_margins.append(max(0.0, margin))

            # ── Torque margin ─────────────────────────────────────────────
            if model.nv > 0 and len(data.qfrc_constraint) > 0:
                max_t = float(np.max(np.abs(data.qfrc_constraint[:model.nv])))
                t_margin = max(0.0, 1.0 - max_t / self._max_torque)
                stability_margins.append(t_margin)

            # ── Contact / collision ───────────────────────────────────────
            if data.ncon > 0:
                collision_risk_steps.append(step)
                stability_margins.append(0.1)
            else:
                stability_margins.append(1.0)

        min_margin = min(stability_margins) if stability_margins else 0.5
        feasible   = (
            min_margin > 0.0
            and len(collision_risk_steps) < n_steps // 3
        )

        # Get final state
        n_q = min(model.nq, 7)
        final_state = {
            "joint_positions": data.qpos[:n_q].tolist(),
            "simulation_steps": n_steps,
            "backend": "mujoco",
        }
        ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        if ee_site_id >= 0:
            final_state["ee_position"] = data.site_xpos[ee_site_id].tolist()

        return HorizonResult(
            feasible=feasible,
            min_stability_margin=round(min_margin, 4),
            collision_risk_steps=collision_risk_steps,
            singularity_risk_steps=singularity_risk_steps,
            predicted_final_state=final_state,
            backend_name="mujoco",
            simulation_time_ms=(time.perf_counter() - t0) * 1000,
        )
