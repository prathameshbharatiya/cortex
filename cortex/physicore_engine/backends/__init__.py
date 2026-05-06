"""
Physics simulation backends for Cortex PhysicsHorizonBuilder.

Available backends (in order of preference):
  1. mujoco  — Full dynamics, highest fidelity (pip install mujoco>=3.0)
  2. pybullet — Good dynamics, widely available (pip install pybullet)
  3. mock    — Deterministic stub for testing

Usage:
    from cortex.physicore_engine.backends import load_backend, BackendError

    backend = load_backend("mujoco")
    result  = backend.simulate_horizon(robot_state, n_steps=12, dt=0.02)
"""

from cortex.physicore_engine.backends.base import (
    PhysicsBackend,
    HorizonResult,
    BackendError,
)
from cortex.physicore_engine.backends.mock_backend import MockPhysicsBackend


def load_backend(name: str, **kwargs) -> "PhysicsBackend":
    """
    Load a physics backend by name.

    Parameters
    ----------
    name : "mujoco" | "pybullet" | "mock"
    **kwargs : passed to backend constructor (e.g. urdf_path)

    Raises BackendError if the backend is unavailable.
    """
    if name == "mujoco":
        from cortex.physicore_engine.backends.mujoco_backend import MuJoCoBackend
        return MuJoCoBackend(**kwargs)
    if name == "pybullet":
        from cortex.physicore_engine.backends.pybullet_backend import PyBulletBackend
        return PyBulletBackend(**kwargs)
    if name == "mock":
        return MockPhysicsBackend(**kwargs)
    raise BackendError(f"Unknown backend: {name!r}. Choose from: mujoco, pybullet, mock")


__all__ = [
    "PhysicsBackend",
    "HorizonResult",
    "BackendError",
    "MockPhysicsBackend",
    "load_backend",
]
