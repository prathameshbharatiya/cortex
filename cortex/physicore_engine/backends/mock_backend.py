"""
Mock Physics Backend
====================
Deterministic stub for testing and environments without physics engines.
Passes all feasibility checks unless emergency_stop or singularity is set.
"""

from __future__ import annotations

import time

from cortex.models.context import RobotState
from cortex.physicore_engine.backends.base import PhysicsBackend, HorizonResult


class MockPhysicsBackend(PhysicsBackend):
    """Deterministic physics stub for testing."""

    def is_available(self) -> bool:
        return True

    def simulate_horizon(
        self,
        robot_state:     RobotState,
        n_steps:         int   = 12,
        dt:              float = 0.02,
        target_position: list[float] | None = None,
    ) -> HorizonResult:
        t0 = time.perf_counter()

        if robot_state.emergency_stop or robot_state.in_singularity:
            return HorizonResult(
                feasible=False,
                min_stability_margin=0.0,
                collision_risk_steps=list(range(n_steps)),
                singularity_risk_steps=[],
                predicted_final_state={"reason": "mock_blocked"},
                backend_name="mock",
                simulation_time_ms=(time.perf_counter() - t0) * 1000,
            )

        return HorizonResult(
            feasible=True,
            min_stability_margin=0.75,
            collision_risk_steps=[],
            singularity_risk_steps=[],
            predicted_final_state={"ee_position": list(robot_state.ee_position)},
            backend_name="mock",
            simulation_time_ms=(time.perf_counter() - t0) * 1000,
        )
