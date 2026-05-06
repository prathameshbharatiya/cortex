"""
PhysiCore Validator — Production Grade
=======================================
Replaces Cortex's original kinematic approximation with the real
PhysiCore Hybrid Uncertainty-Aware Sim-to-Real Engine v2.1.

Old validator:
  - Cosine-based manipulability heuristic
  - Simple Euler forward simulation
  - Basic collision proximity check
  No MPC. No residual learning. No system identification.

This validator:
  - CEM (Cross-Entropy Method) Model Predictive Control
  - Residual ensemble of 3 neural networks (sim-to-real gap learning)
  - Online System ID with innovation-driven adaptive learning rate
  - RK4 integration with quaternion normalisation
  - 12 platform dynamics (quadrotor, manipulator, rocket, surgical robot, AUV...)
  - ISA atmosphere + Dryden turbulence + J2 gravitational perturbation
  - SHA-256 forensic hash chain per control step
  - FailureLog: residual, uncertainty, loop time, state explosion, SysID divergence

Drop-in compatible with the Cortex Certification Gate.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# ── PhysiCore engine from uploaded system ─────────────────────────────────────
_ENGINE_DIR = Path(__file__).parent.parent / "physicore_engine"
if str(_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(_ENGINE_DIR))

try:
    from core.engine import (
        PhysiCore, PhysiCoreConfig, ControlStep,
        PLATFORM_DYNAMICS,
    )
    _PHYSICORE_AVAILABLE = True
except ImportError as _e:
    _PHYSICORE_AVAILABLE = False
    _PHYSICORE_IMPORT_ERR = str(_e)

from cortex.models.action  import Action, ActionType
from cortex.models.context import Context
from cortex.models.decision import FailureMode, RiskLevel, ValidationResult

# Platform map: Cortex action types → PhysiCore platform names
_DEFAULT_PLATFORM = "manipulator_arm"
_PLATFORM_ALIASES = {
    "manipulator_arm": "manipulator_arm",
    "quadrotor":       "quadrotor",
    "ground_rover":    "ground_rover",
    "balancing_bot":   "balancing_bot",
    "surgical_robot":  "surgical_robot",
    "legged_robot":    "legged_robot",
    "rocket":          "rocket",
    "auv":             "auv",
    "satellite":       "satellite",
    "fixed_wing":      "fixed_wing",
    "evtol":           "evtol",
}

# Failure severity thresholds from PhysiCore FailureLog
_RESIDUAL_WARN   = 0.30
_RESIDUAL_ERROR  = 0.80
_UNCERT_WARN     = 0.05
_UNCERT_ERROR    = 0.15
_STATE_CEILING   = 1e4


class PhysiCoreValidator:
    """
    Production-grade physics validator — real PhysiCore MPC engine.

    When a platform engine is created (via get_engine() or attach_engine()):
      Full pipeline: CEM-MPC → residual ensemble → system ID → FailureLog
      → SHA-256 forensic chain per control step.

    Without an engine:
      Analytic kinematic fallback (still better than original Cortex).

    Usage:
        validator = PhysiCoreValidator(platform="manipulator_arm")
        result = validator.validate(action, ctx)

        # After execution, feed back the real transition:
        validator.observe(prev_state, action_array, real_next_state)
    """

    def __init__(
        self,
        platform:              str   = _DEFAULT_PLATFORM,
        initial_params:        dict | None = None,
        control_hz:            float = 60.0,
        wind_intensity:        float = 0.0,
        # Legacy Cortex parameters (used in fallback path)
        workspace_radius_m:    float = 0.85,
        min_stability_margin:  float = 0.10,
        collision_proximity_m: float = 0.05,
    ) -> None:
        self.platform              = platform
        self.initial_params        = initial_params or {"mass": 2.0, "friction": 0.3, "inertia": 0.1}
        self.control_hz            = control_hz
        self.wind_intensity        = wind_intensity
        self.workspace_radius_m    = workspace_radius_m
        self.min_stability_margin  = min_stability_margin
        self.collision_proximity_m = collision_proximity_m

        self._engine: Any = None
        self._step = 0

        # Build the engine immediately if PhysiCore is available
        if _PHYSICORE_AVAILABLE:
            self._build_engine()

    def attach_engine(self, engine: Any) -> None:
        """Attach a pre-built PhysiCore engine."""
        self._engine = engine

    def get_engine(self) -> Any | None:
        """Return the internal PhysiCore engine (for SentinelOS attachment)."""
        return self._engine

    def observe(
        self,
        state:      np.ndarray,
        action:     np.ndarray,
        next_state: np.ndarray,
    ) -> None:
        """
        Feed back a real transition.
        Drives residual ensemble learning and system ID.
        Call this after every execution for continuous adaptation.
        """
        if self._engine is not None:
            self._engine.observe(
                np.asarray(state,      dtype=float),
                np.asarray(action,     dtype=float),
                np.asarray(next_state, dtype=float),
            )

    # ── Main interface ────────────────────────────────────────────────────────

    def validate(self, action: Action, ctx: Context) -> ValidationResult:
        t0 = time.perf_counter()

        if self._engine is not None:
            return self._physicore_path(action, ctx, t0)
        return self._analytic_path(action, ctx, t0)

    # ── PhysiCore MPC path ────────────────────────────────────────────────────

    def _physicore_path(
        self, action: Action, ctx: Context, t0: float
    ) -> ValidationResult:
        # ── Pre-checks shared with analytic path ──────────────────────────────
        # Check these before running the expensive MPC engine so failures
        # that are deterministically blocked don't pay the CEM cost.

        # Physics horizon pre-check — respect planning module's feasibility
        if ctx.physics_horizon is not None and not ctx.physics_horizon.feasible:
            h = ctx.physics_horizon
            return ValidationResult(
                passed=False, score=0.0,
                failure_modes=[FailureMode(
                    code="physics_horizon_infeasible",
                    description=f"Physics horizon infeasible | collision_steps={h.collision_risk_steps}",
                    risk_level=RiskLevel.HIGH, source="physicore",
                    mitigable=len(h.collision_risk_steps) == 0,
                )],
                latency_ms=(time.perf_counter() - t0) * 1000,
                notes="Physics horizon infeasible (pre-check)",
            )

        # Workspace reachability pre-check
        target = action.spec.target_pose
        if target is not None:
            dist = float(np.linalg.norm(np.array([target.x, target.y, target.z])))
            if dist > self.workspace_radius_m:
                return ValidationResult(
                    passed=False, score=0.0,
                    failure_modes=[FailureMode(
                        code="target_unreachable",
                        description=f"Target dist {dist:.3f}m > workspace {self.workspace_radius_m:.3f}m",
                        risk_level=RiskLevel.HIGH, source="physicore", mitigable=False,
                    )],
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    notes="Unreachable target (pre-check)",
                )

        rs    = ctx.robot_state
        state = self._build_state(rs, self._engine.cfg.state_dim)
        x_ref = state.copy()

        # Target pose as reference state
        if target is not None:
            x_ref[:3] = [target.x, target.y, target.z]

        try:
            ctrl: ControlStep = self._engine.step(state, x_ref)
        except Exception as exc:
            return ValidationResult(
                passed=False, score=0.0,
                failure_modes=[FailureMode(
                    code="physicore_error",
                    description=f"PhysiCore engine: {exc}",
                    risk_level=RiskLevel.HIGH, source="physicore", mitigable=False,
                )],
                latency_ms=(time.perf_counter()-t0)*1000,
                notes=f"PhysiCore error: {exc}",
            )

        self._step += 1
        failure_modes: list[FailureMode] = []

        # ── Residual analysis ─────────────────────────────────────────────
        res = ctrl.residual_norm
        unc = ctrl.uncertainty

        if res > _RESIDUAL_ERROR:
            failure_modes.append(FailureMode(
                code="residual_critical",
                description=f"Sim-to-real residual critically high: {res:.4f} > {_RESIDUAL_ERROR}",
                risk_level=RiskLevel.HIGH, source="physicore", mitigable=True,
            ))
        elif res > _RESIDUAL_WARN:
            failure_modes.append(FailureMode(
                code="residual_elevated",
                description=f"Sim-to-real residual elevated: {res:.4f} > {_RESIDUAL_WARN}",
                risk_level=RiskLevel.MEDIUM, source="physicore", mitigable=True,
            ))

        if unc > _UNCERT_ERROR:
            failure_modes.append(FailureMode(
                code="uncertainty_critical",
                description=f"Ensemble uncertainty critically high: {unc:.4f}",
                risk_level=RiskLevel.HIGH, source="physicore", mitigable=True,
            ))
        elif unc > _UNCERT_WARN:
            failure_modes.append(FailureMode(
                code="uncertainty_elevated",
                description=f"Ensemble uncertainty elevated: {unc:.4f}",
                risk_level=RiskLevel.MEDIUM, source="physicore", mitigable=True,
            ))

        # ── FailureLog events from PhysiCore ──────────────────────────────
        for ev in ctrl.failure_events:
            sev_str = ev.severity
            risk = {
                "WARNING": RiskLevel.LOW,
                "ERROR":   RiskLevel.MEDIUM,
                "CRITICAL":RiskLevel.HIGH,
            }.get(sev_str, RiskLevel.MEDIUM)
            failure_modes.append(FailureMode(
                code=ev.failure_type,
                description=ev.description,
                risk_level=risk, source="physicore", mitigable=True,
            ))

        # ── State explosion check ─────────────────────────────────────────
        state_norm = float(np.linalg.norm(state))
        if state_norm > _STATE_CEILING:
            return ValidationResult(
                passed=False, score=0.0,
                failure_modes=[FailureMode(
                    code="state_exploded",
                    description=f"State norm {state_norm:.1f} > {_STATE_CEILING}",
                    risk_level=RiskLevel.CRITICAL, source="physicore", mitigable=False,
                )],
                latency_ms=(time.perf_counter()-t0)*1000,
                notes="State explosion — catastrophic failure",
            )

        # ── Action clipping ───────────────────────────────────────────────
        if ctrl.action_clipped:
            failure_modes.append(FailureMode(
                code="action_clipped",
                description="Optimal action clipped to actuator bounds",
                risk_level=RiskLevel.LOW, source="physicore", mitigable=True,
            ))

        # ── Loop timing ───────────────────────────────────────────────────
        if ctrl.loop_time_ms > 50.0:
            failure_modes.append(FailureMode(
                code="loop_slow",
                description=f"Control loop {ctrl.loop_time_ms:.1f}ms > 50ms budget",
                risk_level=RiskLevel.MEDIUM, source="physicore", mitigable=True,
            ))

        hard = [f for f in failure_modes if not f.mitigable]
        if hard:
            return ValidationResult(
                passed=False, score=0.0, failure_modes=failure_modes,
                latency_ms=(time.perf_counter()-t0)*1000,
                notes=f"PhysiCore hard block: {hard[0].code}",
            )

        # Score: 1.0 at zero residual, degrades with res/unc
        score = max(0.1, 1.0 - 0.6*min(res/_RESIDUAL_ERROR, 1.0) - 0.4*min(unc/_UNCERT_ERROR, 1.0))

        params = ctrl.params
        notes = (
            f"PhysiCore step={ctrl.step_count} | "
            f"res={res:.4f} unc={unc:.4f} | "
            f"mass={params.get('mass',0):.3f} "
            f"friction={params.get('friction',0):.4f} | "
            f"loop={ctrl.loop_time_ms:.1f}ms | "
            f"cert={ctrl.certificate[:8]}"
        )

        return ValidationResult(
            passed=True, score=score, failure_modes=failure_modes,
            latency_ms=(time.perf_counter()-t0)*1000,
            notes=notes,
        )

    # ── Analytic fallback (no PhysiCore available) ────────────────────────────

    def _analytic_path(
        self, action: Action, ctx: Context, t0: float
    ) -> ValidationResult:
        """
        Kinematic approximation fallback.
        Used when PhysiCore is not installed.
        Still stronger than original Cortex: checks horizon from CTX,
        proper collision proximity, and stability margin from physics_horizon.
        """
        import math
        failure_modes: list[FailureMode] = []
        stability_scores: list[float]    = []
        rs = ctx.robot_state

        # Reachability
        target = action.spec.target_pose
        if target is not None:
            dist = float(np.linalg.norm(np.array([target.x, target.y, target.z])))
            if dist > self.workspace_radius_m:
                return ValidationResult(
                    passed=False, score=0.0,
                    failure_modes=[FailureMode(
                        code="target_unreachable",
                        description=f"Target dist {dist:.3f}m > workspace {self.workspace_radius_m:.3f}m",
                        risk_level=RiskLevel.HIGH, source="physicore", mitigable=False,
                    )],
                    latency_ms=(time.perf_counter()-t0)*1000,
                    notes="Unreachable target",
                )
            margin = (self.workspace_radius_m - dist) / self.workspace_radius_m
            stability_scores.append(min(1.0, margin * 2.0))

        # Physics horizon from CTX
        if ctx.physics_horizon is not None:
            h = ctx.physics_horizon
            stability_scores.append(h.min_stability_margin)
            if not h.feasible:
                return ValidationResult(
                    passed=False, score=0.0,
                    failure_modes=[FailureMode(
                        code="physics_horizon_infeasible",
                        description=f"Physics horizon infeasible | collision_steps={h.collision_risk_steps}",
                        risk_level=RiskLevel.HIGH, source="physicore",
                        mitigable=len(h.collision_risk_steps) == 0,
                    )],
                    latency_ms=(time.perf_counter()-t0)*1000,
                    notes="Physics horizon infeasible",
                )
            if h.min_stability_margin < self.min_stability_margin:
                failure_modes.append(FailureMode(
                    code="low_stability_margin",
                    description=f"Stability margin {h.min_stability_margin:.2f} < {self.min_stability_margin:.2f}",
                    risk_level=RiskLevel.MEDIUM, source="physicore", mitigable=True,
                ))
        else:
            stability_scores.append(0.75)

        # Collision proximity
        if target is not None:
            tgt = np.array([target.x, target.y, target.z])
            for obj in ctx.scene_graph.objects:
                if getattr(obj, "is_grasped", False):
                    continue
                obj_pos    = np.array(obj.position)
                obj_radius = max(obj.dimensions) / 2.0 if obj.dimensions else 0.05
                clearance  = float(np.linalg.norm(tgt - obj_pos)) - obj_radius
                if clearance < self.collision_proximity_m:
                    risk = RiskLevel.CRITICAL if clearance < 0 else RiskLevel.HIGH
                    failure_modes.append(FailureMode(
                        code="collision_proximity",
                        description=f"Clearance {clearance:.3f}m < {self.collision_proximity_m:.3f}m to '{obj.label}'",
                        risk_level=risk, source="physicore", mitigable=clearance >= 0,
                    ))

        # Singularity (manipulability heuristic)
        joints = rs.joint_positions
        if joints:
            sines = [abs(math.sin(j)) for j in joints]
            manip = float(np.prod([max(s, 0.01) for s in sines])) ** (1.0 / len(joints))
            stability_scores.append(min(1.0, manip / 0.08))

        hard = [f for f in failure_modes if not f.mitigable]
        if hard:
            return ValidationResult(
                passed=False, score=0.0, failure_modes=failure_modes,
                latency_ms=(time.perf_counter()-t0)*1000,
                notes=f"PhysiCore analytic block: {hard[0].code}",
            )

        score = float(np.mean(stability_scores)) if stability_scores else 0.8
        return ValidationResult(
            passed=True, score=score, failure_modes=failure_modes,
            latency_ms=(time.perf_counter()-t0)*1000,
            notes=f"PhysiCore analytic path (score={score:.2f})",
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_engine(self) -> None:
        if not _PHYSICORE_AVAILABLE:
            print(f"[PhysiCoreValidator] PhysiCore unavailable: {_PHYSICORE_IMPORT_ERR}")
            return
        platform = _PLATFORM_ALIASES.get(self.platform, _DEFAULT_PLATFORM)
        if platform not in PLATFORM_DYNAMICS:
            platform = _DEFAULT_PLATFORM
        try:
            self._engine = PhysiCore.for_platform(
                platform       = platform,
                initial_params = self.initial_params,
                control_hz     = self.control_hz,
                wind_intensity = self.wind_intensity,
            )
        except Exception as exc:
            print(f"[PhysiCoreValidator] Engine build failed: {exc}")
            self._engine = None

    @staticmethod
    def _build_state(rs, state_dim: int) -> np.ndarray:
        pos    = list(rs.ee_position[:3]) if rs.ee_position else [0.0]*3
        vel    = list(rs.joint_velocities[:3]) if rs.joint_velocities else [0.0]*3
        quat   = list(rs.ee_orientation[:4]) if rs.ee_orientation else [1.0, 0.0, 0.0, 0.0]
        joints = list(rs.joint_positions[:6]) if rs.joint_positions else [0.0]*6
        raw    = pos + vel + quat + joints
        arr    = np.array(raw[:state_dim], dtype=float)
        if len(arr) < state_dim:
            arr = np.pad(arr, (0, state_dim - len(arr)))
        return arr

    @property
    def engine_params(self) -> dict:
        if self._engine is None:
            return {}
        return self._engine.physics.params.copy()

    @property
    def engine_diagnostics(self) -> dict:
        if self._engine is None:
            return {"available": False}
        return self._engine.diagnostics_full
