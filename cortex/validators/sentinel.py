"""
Sentinel Validator — Production Grade
======================================
Replaces the original rule-based SentinelValidator with the full
8-layer Sentinel OS from physicore/sentinel/core.py.

Old validator: workspace bounds + speed/force if-statements. That's all.
This validator: 8 mathematical layers (L0-L7), Lyapunov stability,
RLS estimator, fault signatures, shadow path, forensic hash chain.

Drop-in compatible with the Certification Gate.
"""
from __future__ import annotations
import sys
import time
from pathlib import Path
from typing import Any
import numpy as np

# ── Sentinel OS from uploaded system ─────────────────────────────────────────
_ENGINE_DIR = Path(__file__).parent.parent / "physicore_engine"
if str(_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(_ENGINE_DIR))

try:
    from sentinel.core import (
        SentinelOS, SentinelConfig, SentinelMode,
        get_sentinel_config, MissionEvent,
    )
    _SENTINEL_AVAILABLE = True
except ImportError as _e:
    _SENTINEL_AVAILABLE = False
    _SENTINEL_IMPORT_ERR = str(_e)

from cortex.models.action  import Action
from cortex.models.context import Context
from cortex.models.decision import FailureMode, RiskLevel, ValidationResult

_SEV_RISK = {
    "WARNING": RiskLevel.MEDIUM, "CRITICAL": RiskLevel.HIGH,
    "CATASTROPHIC": RiskLevel.CRITICAL, "ERROR": RiskLevel.HIGH,
}


class SentinelValidator:
    """
    Production-grade safety validator — full 8-layer Sentinel OS.

    If a PhysiCore engine is attached (via attach_engine()):
      Runs L0-L7: Lyapunov stability, RLS, fault signatures, shadow path,
      SHA-256 forensic chain, FTS, intent coherence, jerk limiting.

    If no engine is attached:
      Runs enhanced rule-based fallback (still better than original Cortex).

    Usage:
        validator = SentinelValidator(platform="manipulator_arm")
        validator.attach_engine(physicore_engine)   # optional but strongly recommended
        result = validator.validate(action, ctx)
        # After execution:
        validator.observe(real_next_state)
    """

    def __init__(
        self,
        platform:          str   = "manipulator_arm",
        engine:            Any   = None,
        verbose:           bool  = False,
        max_speed_ms:      float = 2.0,
        max_force_n:       float = 150.0,
        human_exclusion_m: float = 0.5,
        human_slow_zone_m: float = 1.5,
    ) -> None:
        self.platform          = platform
        self.max_speed_ms      = max_speed_ms
        self.max_force_n       = max_force_n
        self.human_exclusion_m = human_exclusion_m
        self._verbose          = verbose
        self._sentinel: Any    = None
        self._last_state:  np.ndarray | None = None
        self._last_action: np.ndarray | None = None

        if engine is not None:
            self.attach_engine(engine)

    def attach_engine(self, engine: Any) -> None:
        """Attach a PhysiCore engine to unlock the full Sentinel OS stack."""
        if not _SENTINEL_AVAILABLE:
            print(f"[SentinelValidator] SentinelOS unavailable: {_SENTINEL_IMPORT_ERR}")
            return
        try:
            cfg = get_sentinel_config(self.platform)
            self._sentinel = SentinelOS(
                engine=engine, platform=self.platform,
                config=cfg, verbose=self._verbose,
            )
        except Exception as exc:
            print(f"[SentinelValidator] SentinelOS init failed: {exc}")
            self._sentinel = None

    def observe(self, next_state: np.ndarray) -> None:
        """Feed back the real next state. Drives RLS and residual learning."""
        if (self._sentinel is not None
                and self._last_state is not None
                and self._last_action is not None):
            self._sentinel.observe(
                self._last_state, self._last_action,
                np.asarray(next_state, dtype=float),
            )

    # ── Main interface ────────────────────────────────────────────────────────

    def validate(self, action: Action, ctx: Context) -> ValidationResult:
        t0 = time.perf_counter()

        # Hard blocks regardless of mode
        rs = ctx.robot_state
        if rs.emergency_stop:
            return self._hard_block("emergency_stop", "Emergency stop active.", t0)
        if rs.in_singularity:
            return self._hard_block("kinematic_singularity", "Kinematic singularity.", t0)

        if self._sentinel is not None:
            return self._sentinel_path(action, ctx, t0)
        return self._rules_path(action, ctx, t0)

    # ── Sentinel OS path (full 8 layers) ──────────────────────────────────────

    def _sentinel_path(self, action: Action, ctx: Context, t0: float) -> ValidationResult:
        rs    = ctx.robot_state
        state = self._build_state(rs)
        x_ref = state.copy()
        alt   = float(rs.ee_position[2]) if len(rs.ee_position) > 2 else 0.0
        vel   = np.array(rs.joint_velocities[:3]) if rs.joint_velocities else np.zeros(3)

        try:
            safe_action = self._sentinel.step(state=state, x_ref=x_ref,
                                              altitude=alt, velocity=vel)
        except Exception as exc:
            return self._hard_block("sentinel_error", f"SentinelOS: {exc}", t0)

        self._last_state  = state.copy()
        self._last_action = safe_action.copy()

        status = self._sentinel.status
        mode   = SentinelMode(status["mode"])
        passed = mode not in (SentinelMode.FALLBACK, SentinelMode.INTERNAL_FAULT)
        score  = self._mode_score(mode, status)

        failure_modes: list[FailureMode] = []
        if status.get("fault"):
            f   = status["fault"]
            sev = _SEV_RISK.get(f.get("severity", "WARNING"), RiskLevel.MEDIUM)
            failure_modes.append(FailureMode(
                code=f.get("fault_type", "SENTINEL_FAULT"),
                description=f.get("description", "Sentinel fault"),
                risk_level=sev, source="sentinel",
                mitigable=mode == SentinelMode.CAUTIOUS,
            ))
        if ctx.scene_graph.humans_nearby:
            failure_modes.append(FailureMode(
                code="human_proximity",
                description="Human in workspace — Sentinel L6 jerk limiting active",
                risk_level=RiskLevel.MEDIUM, source="sentinel", mitigable=True,
            ))

        lya  = status.get("lyapunov", {})
        rls  = status.get("rls", {})
        notes = (
            f"Sentinel {mode.value} | "
            f"V={lya.get('V',0):.3f} Vs={lya.get('V_shadow',0):.3f} "
            f"Δ={lya.get('shadow_delta',0):.4f} | "
            f"mass={rls.get('total_mass',0):.3f} "
            f"cov={rls.get('avg_covariance',0):.0f} | "
            f"faults={status.get('fault_count',0)} | "
            f"hash={status.get('ledger_hash','?')[:8]}"
        )

        return ValidationResult(
            passed=passed, score=score,
            failure_modes=failure_modes,
            latency_ms=(time.perf_counter() - t0) * 1000,
            notes=notes,
        )

    # ── Rule-based fallback (no engine) ──────────────────────────────────────

    def _rules_path(self, action: Action, ctx: Context, t0: float) -> ValidationResult:
        failure_modes: list[FailureMode] = []
        rs     = ctx.robot_state
        target = action.spec.target_pose

        # Workspace bounds
        if target is not None:
            bounds = ctx.scene_graph.workspace_bounds
            for axis, val, lo, hi in [
                ("x", target.x, "x_min", "x_max"),
                ("y", target.y, "y_min", "y_max"),
                ("z", target.z, "z_min", "z_max"),
            ]:
                if val < bounds.get(lo, -1e9) or val > bounds.get(hi, 1e9):
                    return ValidationResult(
                        passed=False, score=0.0,
                        failure_modes=[FailureMode(
                            code="workspace_boundary_violation",
                            description=(f"Target {axis}={val:.3f} outside "
                                         f"[{bounds.get(lo):.3f}, {bounds.get(hi):.3f}]"),
                            risk_level=RiskLevel.HIGH, source="sentinel", mitigable=False,
                        )],
                        latency_ms=(time.perf_counter()-t0)*1000,
                        notes="Workspace boundary violated",
                    )

        # Speed / force
        spd = action.spec.constraints.max_speed_ms
        if spd is not None and spd > self.max_speed_ms:
            failure_modes.append(FailureMode(
                code="speed_limit_exceeded",
                description=f"Speed {spd:.2f} > {self.max_speed_ms:.2f} m/s",
                risk_level=RiskLevel.HIGH, source="sentinel", mitigable=True,
            ))
        frc = action.spec.constraints.max_force_n
        if frc is not None and frc > self.max_force_n:
            failure_modes.append(FailureMode(
                code="force_limit_exceeded",
                description=f"Force {frc:.1f} > {self.max_force_n:.1f} N",
                risk_level=RiskLevel.HIGH, source="sentinel", mitigable=True,
            ))

        # Human proximity
        if ctx.scene_graph.humans_nearby:
            failure_modes.append(FailureMode(
                code="human_proximity",
                description="Human in workspace",
                risk_level=RiskLevel.HIGH, source="sentinel", mitigable=True,
            ))

        # Exclusion zones from CTX constraints
        for c in ctx.absolute_constraints:
            if c.constraint_type == "object_exclusion" and target is not None:
                center = np.array(c.parameters.get("center", [0,0,0]))
                radius = c.parameters.get("radius_m", 0.1)
                dist   = float(np.linalg.norm(
                    np.array([target.x, target.y, target.z]) - center
                ))
                if dist < radius:
                    return ValidationResult(
                        passed=False, score=0.0,
                        failure_modes=[FailureMode(
                            code=f"constraint_{c.constraint_id}_exclusion",
                            description=f"Target inside '{c.description}' (dist={dist:.3f}m < {radius:.3f}m)",
                            risk_level=RiskLevel.CRITICAL, source="sentinel", mitigable=False,
                        )],
                        latency_ms=(time.perf_counter()-t0)*1000,
                        notes="Exclusion zone violated",
                    )

        hard = [f for f in failure_modes if not f.mitigable]
        if hard:
            return ValidationResult(
                passed=False, score=0.0, failure_modes=failure_modes,
                latency_ms=(time.perf_counter()-t0)*1000,
                notes=f"Blocked: {hard[0].code}",
            )

        score = max(0.5, 1.0 - len(failure_modes) * 0.1)
        return ValidationResult(
            passed=True, score=score, failure_modes=failure_modes,
            latency_ms=(time.perf_counter()-t0)*1000,
            notes="Sentinel rules passed (no engine)",
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _build_state(rs) -> np.ndarray:
        pos    = list(rs.ee_position[:3]) if rs.ee_position else [0.0]*3
        vel    = list(rs.joint_velocities[:3]) if rs.joint_velocities else [0.0]*3
        joints = list(rs.joint_positions[:6]) if rs.joint_positions else [0.0]*6
        return np.array(pos + vel + joints[:3], dtype=float)

    @staticmethod
    def _mode_score(mode, status: dict) -> float:
        base = {
            "NOMINAL": 0.90, "CAUTIOUS": 0.65,
            "FALLBACK": 0.10, "INTERNAL_FAULT": 0.00,
        }.get(mode.value if hasattr(mode, "value") else mode, 0.50)
        cov = status.get("rls", {}).get("avg_covariance", 1000.0)
        return min(1.0, base + 0.1 * max(0.0, 1.0 - cov/5000.0))

    @staticmethod
    def _hard_block(code: str, description: str, t0: float) -> ValidationResult:
        return ValidationResult(
            passed=False, score=0.0,
            failure_modes=[FailureMode(
                code=code, description=description,
                risk_level=RiskLevel.CRITICAL, source="sentinel", mitigable=False,
            )],
            latency_ms=(time.perf_counter()-t0)*1000,
            notes=f"HARD BLOCK: {code}",
        )

    @property
    def sentinel_status(self) -> dict | None:
        return self._sentinel.status if self._sentinel else None

    @property
    def ledger_hash(self) -> str:
        return self._sentinel.chain_hash if self._sentinel else "NO_SENTINEL"

    @property
    def mode(self) -> str:
        return self._sentinel.mode.value if self._sentinel else "RULES_ONLY"
