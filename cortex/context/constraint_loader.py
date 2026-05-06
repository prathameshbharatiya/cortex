"""
Constraint Loader
=================
Stage 2 of Context assembly.

Collects, validates, and injects safety constraints into the CTX.
Constraints come from three sources:

  1. Platform defaults — hard limits that are always active.
     (workspace bounds, absolute speed ceiling, force ceiling)

  2. Deployment configuration — site-specific rules loaded from config.
     (exclusion zones, human proximity rules, task-specific limits)

  3. Runtime signals — dynamic constraints from live system state.
     (reduced limits when battery low, restricted zones when humans present)

Every constraint in the CTX has veto_power=True (Sentinel-grade) or
veto_power=False (advisory). The Certification Gate uses this distinction
to decide whether to block or just reduce confidence.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from cortex.models.context import SafetyConstraint, RobotState, SceneGraph


# ── Platform defaults ─────────────────────────────────────────────────────────

DEFAULT_WORKSPACE_BOUNDS = {
    "x_min": -1.0, "x_max": 1.0,
    "y_min": -1.0, "y_max": 1.0,
    "z_min":  0.0, "z_max": 2.0,
}

DEFAULT_SPEED_LIMIT_MS   = 2.0    # m/s  — absolute ceiling
DEFAULT_FORCE_LIMIT_N    = 150.0  # N    — absolute ceiling
HUMAN_SPEED_LIMIT_MS     = 0.25   # m/s  — when humans nearby
HUMAN_EXCLUSION_M        = 0.5    # m    — hard exclusion around humans


@dataclass(frozen=True)
class ConstraintConfig:
    """Deployment-level constraint configuration."""
    workspace_bounds:   dict[str, float] = field(default_factory=lambda: DEFAULT_WORKSPACE_BOUNDS.copy())
    max_speed_ms:       float = DEFAULT_SPEED_LIMIT_MS
    max_force_n:        float = DEFAULT_FORCE_LIMIT_N
    custom_constraints: list[dict[str, Any]] = field(default_factory=list)
    human_speed_limit_ms: float = HUMAN_SPEED_LIMIT_MS
    human_exclusion_m:  float = HUMAN_EXCLUSION_M


@dataclass
class ConstraintLoadReport:
    constraints:     list[SafetyConstraint] = field(default_factory=list)
    veto_count:      int = 0
    advisory_count:  int = 0
    runtime_count:   int = 0   # dynamic constraints from live state
    latency_ms:      float = 0.0


class ConstraintLoader:
    """
    Assembles the full constraint set for the CTX.

    Called during Context assembly — after sensor cross-check,
    before physics horizon. Constraints are injected into the CTX
    so both the AI planner and the Certification Gate see them.
    """

    def __init__(self, config: ConstraintConfig | None = None) -> None:
        self.cfg = config or ConstraintConfig()

    def load(
        self,
        robot_state:         RobotState,
        scene_graph:         SceneGraph,
        extra_constraints:   list[SafetyConstraint] | None = None,
    ) -> ConstraintLoadReport:
        """
        Build the full active constraint set for the current situation.
        """
        t0 = time.perf_counter()
        constraints: list[SafetyConstraint] = []

        # ── 1. Platform defaults (always active) ──────────────────────────────
        constraints.extend(self._platform_defaults())

        # ── 2. Runtime signals (live state) ───────────────────────────────────
        runtime = self._runtime_constraints(robot_state, scene_graph)
        constraints.extend(runtime)

        # ── 3. Deployment configuration ───────────────────────────────────────
        constraints.extend(self._config_constraints())

        # ── 4. Caller-provided extras ─────────────────────────────────────────
        if extra_constraints:
            constraints.extend(extra_constraints)

        # Deduplicate by constraint_id (caller extras override defaults)
        seen: dict[str, SafetyConstraint] = {}
        for c in constraints:
            seen[c.constraint_id] = c
        final = list(seen.values())

        veto_count     = sum(1 for c in final if c.veto_power)
        advisory_count = sum(1 for c in final if not c.veto_power)
        runtime_count  = len(runtime)

        return ConstraintLoadReport(
            constraints=final,
            veto_count=veto_count,
            advisory_count=advisory_count,
            runtime_count=runtime_count,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )

    # ── Constraint sources ────────────────────────────────────────────────────

    def _platform_defaults(self) -> list[SafetyConstraint]:
        bounds = self.cfg.workspace_bounds
        return [
            SafetyConstraint(
                constraint_id="platform_workspace",
                description="Platform workspace boundary limits",
                constraint_type="workspace_boundary",
                parameters=bounds,
                veto_power=True,
                source="platform_defaults",
            ),
            SafetyConstraint(
                constraint_id="platform_speed",
                description=f"Absolute speed ceiling {self.cfg.max_speed_ms} m/s",
                constraint_type="speed_limit",
                parameters={"max_speed_ms": self.cfg.max_speed_ms},
                veto_power=True,
                source="platform_defaults",
            ),
            SafetyConstraint(
                constraint_id="platform_force",
                description=f"Absolute force ceiling {self.cfg.max_force_n} N",
                constraint_type="force_limit",
                parameters={"max_force_n": self.cfg.max_force_n},
                veto_power=True,
                source="platform_defaults",
            ),
        ]

    def _runtime_constraints(
        self,
        robot_state: RobotState,
        scene_graph: SceneGraph,
    ) -> list[SafetyConstraint]:
        """Dynamic constraints derived from live robot and scene state."""
        constraints: list[SafetyConstraint] = []

        # Human proximity — tighten speed limit
        if scene_graph.humans_nearby:
            constraints.append(SafetyConstraint(
                constraint_id="runtime_human_speed",
                description=(
                    f"Human detected: speed limited to {self.cfg.human_speed_limit_ms} m/s"
                ),
                constraint_type="speed_limit",
                parameters={"max_speed_ms": self.cfg.human_speed_limit_ms},
                veto_power=True,
                source="runtime_human_detection",
            ))

        # Scene obstruction — advisory
        if scene_graph.obstruction_detected:
            constraints.append(SafetyConstraint(
                constraint_id="runtime_obstruction",
                description="Obstruction detected in workspace — proceed with caution",
                constraint_type="custom",
                parameters={"severity": "advisory"},
                veto_power=False,
                source="runtime_scene_analysis",
            ))

        # Gripper already loaded — force limit lower
        if not robot_state.gripper_open and robot_state.gripper_force_n > 20.0:
            constraints.append(SafetyConstraint(
                constraint_id="runtime_gripper_loaded",
                description="Gripper loaded — reduced force ceiling to protect payload",
                constraint_type="force_limit",
                parameters={"max_force_n": min(self.cfg.max_force_n, 50.0)},
                veto_power=True,
                source="runtime_gripper_state",
            ))

        return constraints

    def _config_constraints(self) -> list[SafetyConstraint]:
        """Convert deployment config entries to SafetyConstraints."""
        constraints: list[SafetyConstraint] = []
        for i, cfg in enumerate(self.cfg.custom_constraints):
            constraints.append(SafetyConstraint(
                constraint_id=cfg.get("constraint_id", f"config_{i}"),
                description=cfg.get("description", f"Config constraint {i}"),
                constraint_type=cfg.get("constraint_type", "custom"),
                parameters=cfg.get("parameters", {}),
                veto_power=cfg.get("veto_power", True),
                source="deployment_config",
            ))
        return constraints
