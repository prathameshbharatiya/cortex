"""
Physics backend abstract interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from cortex.models.context import RobotState


class BackendError(Exception):
    """Raised when a backend is unavailable or encounters a simulation failure."""


@dataclass
class HorizonResult:
    """Result of a physics horizon simulation."""
    feasible:              bool
    min_stability_margin:  float          # 0–1, 0 = at limit, 1 = full margin
    collision_risk_steps:  list[int]      = field(default_factory=list)
    singularity_risk_steps: list[int]     = field(default_factory=list)
    predicted_final_state: dict[str, Any] = field(default_factory=dict)
    backend_name:          str            = ""
    simulation_time_ms:    float          = 0.0


class PhysicsBackend(ABC):
    """
    Abstract base for Cortex physics simulation backends.

    Implementors provide simulate_horizon() which takes the current robot state
    and returns a HorizonResult.  The PhysicsHorizonBuilder calls this and
    converts the result into a PhysicsHorizon CTX field.
    """

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this backend is installed and functional."""
        ...

    @abstractmethod
    def simulate_horizon(
        self,
        robot_state:     RobotState,
        n_steps:         int   = 12,
        dt:              float = 0.02,
        target_position: list[float] | None = None,
    ) -> HorizonResult:
        """
        Run forward physics simulation for n_steps timesteps of dt seconds each.

        Parameters
        ----------
        robot_state     : current robot state from sensors
        n_steps         : number of simulation steps (default 12 = 240ms at 50Hz)
        dt              : timestep in seconds
        target_position : optional hint for intended direction of motion

        Returns
        -------
        HorizonResult with feasibility and risk information
        """
        ...
