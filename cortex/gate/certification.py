"""
Certification Gate
==================
The core of Cortex. The mandatory boundary between AI intent and physical action.

phi(action, context) → CertificationDecision

The Gate runs three validators in strict priority order:

  1. Sentinel  (Priority 1 — Absolute veto)
     Hard safety constraints. A Sentinel rejection is final.
     → SAFE_HALT or HUMAN_OVERRIDE_REQUIRED

  2. PhysiCore (Priority 2 — Hard physical constraint)
     Physics feasibility. Can issue EXECUTE_WITH_CONSTRAINTS if fixable.
     → REPLAN_REQUIRED or modifies action for EXECUTE_WITH_CONSTRAINTS

  3. Memory    (Priority 3 — Soft constraint)
     Memory consistency. Reduces confidence, can trigger HUMAN_OVERRIDE.
     → Never blocks alone, but can push below confidence floor.

The Gate then assembles the ConfidenceContract and issues the final decision.

No action reaches robot.execute() without passing through here.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

import numpy as np

from cortex.models.action  import Action
from cortex.models.context import Context
from cortex.models.decision import (
    CertificationDecision,
    CertificationState,
    ConfidenceContract,
    DecisionTrace,
    FailureMode,
    RiskLevel,
    ValidationResult,
)
from cortex.validators.sentinel         import SentinelValidator
from cortex.validators.physicore        import PhysiCoreValidator
from cortex.validators.memory_validator import MemoryValidator
from cortex.observability.logging  import get_logger, bind_request
from cortex.observability.metrics  import get_metrics
from cortex.observability.tracing  import span as otel_span
from cortex.ood.detector           import OODDetector, _default_state_to_vector

_log = get_logger(__name__)

# ── Confidence thresholds ─────────────────────────────────────────────────────
# Below HUMAN_THRESHOLD → HUMAN_OVERRIDE_REQUIRED (even if all validators pass)
HUMAN_OVERRIDE_THRESHOLD = 0.35
# Below REPLAN_THRESHOLD → REPLAN_REQUIRED
REPLAN_THRESHOLD         = 0.50


class CertificationGate:
    """
    The Cortex Certification Gate.

    Usage
    -----
        gate = CertificationGate()
        cert = gate.certify(action, ctx)

        if cert.approved:
            robot.execute(cert.action)
        elif cert.requires_replan:
            new_action = planner.replan(ctx)
        elif cert.requires_human:
            operator.escalate(cert)
        else:  # SAFE_HALT
            robot.halt()
    """

    def __init__(
        self,
        sentinel:  SentinelValidator  | None = None,
        physicore: PhysiCoreValidator | None = None,
        memory:    MemoryValidator    | None = None,
        ood_detector: OODDetector | None = None,
        # Confidence thresholds
        human_override_threshold: float = HUMAN_OVERRIDE_THRESHOLD,
        replan_threshold:         float = REPLAN_THRESHOLD,
        # Validator weights for composite confidence
        sentinel_weight:  float = 0.50,
        physicore_weight: float = 0.35,
        memory_weight:    float = 0.15,
    ) -> None:
        self.sentinel     = sentinel  or SentinelValidator()
        self.physicore    = physicore or PhysiCoreValidator()
        self.memory       = memory    or MemoryValidator()
        self.ood_detector = ood_detector or OODDetector()

        self.human_override_threshold = human_override_threshold
        self.replan_threshold         = replan_threshold

        # Weights must sum to 1.0
        total = sentinel_weight + physicore_weight + memory_weight
        self.w_sentinel  = sentinel_weight  / total
        self.w_physicore = physicore_weight / total
        self.w_memory    = memory_weight    / total

    # ── Main entry point ──────────────────────────────────────────────────────

    def certify(self, action: Action, ctx: Context) -> CertificationDecision:
        """
        The core certification function: phi(action, ctx) → CertificationDecision.

        This is the only function that produces a CertificationDecision.
        It is the mandatory checkpoint between any AI system and physical execution.
        """
        metrics = get_metrics()
        metrics.inc_active()
        t0 = time.perf_counter()

        with otel_span(
            "cortex.certify",
            action_id=action.action_id,
            ctx_id=ctx.ctx_id,
            task=getattr(ctx, "task", ""),
        ) as root_span:
            with bind_request(
                action_id=str(action.action_id),
                trace_id="",          # filled in _make_decision once trace_id known
            ):
                try:
                    result = self._certify_inner(action, ctx, t0, root_span)
                except Exception as exc:
                    _log.error("certification error", exc_info=True,
                               action_id=str(action.action_id))
                    metrics.record_error("gate")
                    raise
                finally:
                    metrics.dec_active()
                return result

    def _certify_inner(
        self,
        action: Action,
        ctx: Context,
        t0: float,
        root_span: Any,
    ) -> CertificationDecision:
        """Inner certify — all the real logic, called from certify()."""
        import time as _time

        # ── Guard: expired context ────────────────────────────────────────────
        if ctx.is_expired:
            return self._make_decision(
                state=CertificationState.REPLAN_REQUIRED,
                action=None,
                reason="Context has expired. Reassemble CTX before certifying.",
                sentinel_result=None,
                physicore_result=None,
                memory_result=None,
                action_original=action,
                ctx=ctx,
                t0=t0,
                blocking_fm=FailureMode(
                    code="context_expired",
                    description="Context validity window has elapsed.",
                    risk_level=RiskLevel.MEDIUM,
                    source="gate",
                    mitigable=True,
                ),
            )

        # ════════════════════════════════════════════════════════════════════
        # STAGE 1 — SENTINEL (Priority 1, Absolute Veto)
        # ════════════════════════════════════════════════════════════════════
        _s1 = time.perf_counter()
        sentinel_result = self.sentinel.validate(action, ctx)
        _sentinel_ms = (time.perf_counter() - _s1) * 1000.0
        sentinel_result = ValidationResult(
            passed=sentinel_result.passed,
            score=sentinel_result.score,
            failure_modes=sentinel_result.failure_modes,
            latency_ms=_sentinel_ms,
            notes=sentinel_result.notes,
        )
        get_metrics().record_validator("sentinel", sentinel_result.passed, _sentinel_ms)
        root_span.set_attribute("cortex.sentinel.passed", sentinel_result.passed)
        root_span.set_attribute("cortex.sentinel.score",  sentinel_result.score)
        root_span.set_attribute("cortex.sentinel.latency_ms", round(_sentinel_ms, 2))

        if not sentinel_result.passed:
            # Determine whether this needs a human or a full halt
            has_critical = any(
                fm.risk_level == RiskLevel.CRITICAL
                for fm in sentinel_result.failure_modes
            )
            state = (
                CertificationState.HUMAN_OVERRIDE_REQUIRED
                if has_critical
                else CertificationState.SAFE_HALT
            )
            blocking = self._worst_failure(sentinel_result.failure_modes)
            return self._make_decision(
                state=state,
                action=None,
                reason=f"Sentinel veto: {blocking.code if blocking else 'safety constraint violated'}",
                sentinel_result=sentinel_result,
                physicore_result=None,
                memory_result=None,
                action_original=action,
                ctx=ctx,
                t0=t0,
                blocking_fm=blocking,
            )

        # Check if Sentinel passed but there are human-proximity soft constraints
        # that require speed reduction
        action_modified = action
        constraints_applied: list[str] = []

        human_proximity_flags = [
            fm for fm in sentinel_result.failure_modes
            if fm.code == "human_proximity" and fm.mitigable
        ]
        if human_proximity_flags:
            action_modified = action_modified.with_speed_limit(0.25)  # 25cm/s near humans
            constraints_applied.append("speed_reduced_for_human_proximity")

        speed_exceeded_flags = [
            fm for fm in sentinel_result.failure_modes
            if fm.code == "speed_limit_exceeded" and fm.mitigable
        ]
        if speed_exceeded_flags:
            action_modified = action_modified.with_speed_limit(self.sentinel.max_speed_ms)
            constraints_applied.append("speed_capped_to_sentinel_limit")

        force_exceeded_flags = [
            fm for fm in sentinel_result.failure_modes
            if fm.code == "force_limit_exceeded" and fm.mitigable
        ]
        if force_exceeded_flags:
            action_modified = action_modified.with_force_limit(self.sentinel.max_force_n)
            constraints_applied.append("force_capped_to_sentinel_limit")

        # ════════════════════════════════════════════════════════════════════
        # STAGE 1.5 — OOD DETECTION (non-blocking, confidence reduction)
        # ════════════════════════════════════════════════════════════════════
        if self.ood_detector.is_fitted:
            _ood_vec    = _default_state_to_vector(ctx.robot_state)
            _ood_result = self.ood_detector.detect(_ood_vec)
            root_span.set_attribute("cortex.ood.is_ood",    _ood_result.is_ood)
            root_span.set_attribute("cortex.ood.ood_score", _ood_result.ood_score)
            if _ood_result.is_ood:
                _log.warning(
                    "ood_detected",
                    ood_score=_ood_result.ood_score,
                    method=_ood_result.method,
                )
                return self._make_decision(
                    state=CertificationState.REPLAN_REQUIRED,
                    action=None,
                    reason=(
                        f"OOD anomaly detected (score={_ood_result.ood_score:.3f}). "
                        "Robot state is outside the calibrated distribution."
                    ),
                    sentinel_result=sentinel_result,
                    physicore_result=None,
                    memory_result=None,
                    action_original=action,
                    ctx=ctx,
                    t0=t0,
                    blocking_fm=FailureMode(
                        code="OOD_ANOMALY",
                        description=(
                            f"State embedding OOD score {_ood_result.ood_score:.3f} "
                            f"exceeds threshold (method={_ood_result.method})"
                        ),
                        risk_level=RiskLevel.HIGH,
                        source="ood_detector",
                        mitigable=True,
                    ),
                )

        # ════════════════════════════════════════════════════════════════════
        # STAGE 2 — PHYSICORE (Priority 2, Hard Physical Constraint)
        # ════════════════════════════════════════════════════════════════════
        _s2 = time.perf_counter()
        physicore_result = self.physicore.validate(action_modified, ctx)
        _physicore_ms = (time.perf_counter() - _s2) * 1000.0
        physicore_result = ValidationResult(
            passed=physicore_result.passed,
            score=physicore_result.score,
            failure_modes=physicore_result.failure_modes,
            latency_ms=_physicore_ms,
            notes=physicore_result.notes,
        )
        get_metrics().record_validator("physicore", physicore_result.passed, _physicore_ms)
        root_span.set_attribute("cortex.physicore.passed", physicore_result.passed)
        root_span.set_attribute("cortex.physicore.score",  physicore_result.score)
        root_span.set_attribute("cortex.physicore.latency_ms", round(_physicore_ms, 2))

        if not physicore_result.passed:
            blocking = self._worst_failure(physicore_result.failure_modes)
            return self._make_decision(
                state=CertificationState.REPLAN_REQUIRED,
                action=None,
                reason=f"PhysiCore: no feasible trajectory — {blocking.code if blocking else 'physics infeasible'}",
                sentinel_result=sentinel_result,
                physicore_result=physicore_result,
                memory_result=None,
                action_original=action,
                ctx=ctx,
                t0=t0,
                blocking_fm=blocking,
            )

        # Apply mitigable PhysiCore constraints
        for fm in physicore_result.failure_modes:
            if fm.mitigable and fm.code in ("low_stability_margin", "torque_near_limit"):
                action_modified = action_modified.with_speed_limit(
                    (action_modified.spec.constraints.max_speed_ms or 1.0) * 0.7
                )
                constraints_applied.append(f"speed_reduced_for_{fm.code}")

        # ════════════════════════════════════════════════════════════════════
        # STAGE 3 — MEMORY (Priority 3, Soft Constraint)
        # ════════════════════════════════════════════════════════════════════
        _s3 = time.perf_counter()
        memory_result = self.memory.validate(action_modified, ctx)
        _memory_ms = (time.perf_counter() - _s3) * 1000.0
        memory_result = ValidationResult(
            passed=memory_result.passed,
            score=memory_result.score,
            failure_modes=memory_result.failure_modes,
            latency_ms=_memory_ms,
            notes=memory_result.notes,
        )
        get_metrics().record_validator("memory", memory_result.passed, _memory_ms)
        root_span.set_attribute("cortex.memory.score", memory_result.score)
        root_span.set_attribute("cortex.memory.latency_ms", round(_memory_ms, 2))

        # Memory never hard-blocks alone — but it lowers confidence

        # ════════════════════════════════════════════════════════════════════
        # CONFIDENCE CONTRACT ASSEMBLY
        # ════════════════════════════════════════════════════════════════════
        confidence = self._assemble_confidence(
            sentinel_result, physicore_result, memory_result, ctx
        )

        # ════════════════════════════════════════════════════════════════════
        # FINAL STATE DETERMINATION
        # ════════════════════════════════════════════════════════════════════
        state, reason = self._determine_final_state(
            confidence, constraints_applied, action_modified, action, ctx
        )

        all_failure_modes = (
            sentinel_result.failure_modes
            + physicore_result.failure_modes
            + memory_result.failure_modes
        )

        return self._make_decision(
            state=state,
            action=action_modified if state in (
                CertificationState.EXECUTE,
                CertificationState.EXECUTE_WITH_CONSTRAINTS,
            ) else None,
            reason=reason,
            sentinel_result=sentinel_result,
            physicore_result=physicore_result,
            memory_result=memory_result,
            action_original=action,
            ctx=ctx,
            t0=t0,
            confidence=confidence,
            all_failure_modes=all_failure_modes,
        )

    # ── Confidence assembly ───────────────────────────────────────────────────

    def _assemble_confidence(
        self,
        sentinel_result:  ValidationResult,
        physicore_result: ValidationResult,
        memory_result:    ValidationResult,
        ctx: Context,
    ) -> ConfidenceContract:
        """
        Build the ConfidenceContract from all three validator scores.
        This is a weighted composite, not a single scalar.
        """
        composite_score = (
            self.w_sentinel  * sentinel_result.score
            + self.w_physicore * physicore_result.score
            + self.w_memory    * memory_result.score
        )

        all_failures = (
            sentinel_result.failure_modes
            + physicore_result.failure_modes
            + memory_result.failure_modes
        )
        worst_risk = self._worst_risk(all_failures)

        # Stability margin from PhysiCore (or estimate from score if no horizon)
        stability_margin = (
            ctx.physics_horizon.min_stability_margin * 100
            if ctx.physics_horizon
            else physicore_result.score * 50.0  # estimate as % of limit
        )

        uncertainty_sources = []
        if memory_result.score < 0.8:
            uncertainty_sources.append("memory_uncertainty")
        if not ctx.has_physics_horizon:
            uncertainty_sources.append("no_physics_horizon")
        if ctx.has_human_nearby:
            uncertainty_sources.append("human_proximity")
        if ctx.scene_graph.obstruction_detected:
            uncertainty_sources.append("scene_obstruction")

        # Validity window: minimum of CTX window and validator-specific windows
        validity_window = min(
            ctx.validity_window_ms,
            500.0 if ctx.has_human_nearby else 2000.0,
        )

        return ConfidenceContract(
            success_probability=round(composite_score, 4),
            stability_margin=round(stability_margin, 2),
            worst_case_risk=worst_risk,
            uncertainty_sources=uncertainty_sources,
            validity_window_ms=validity_window,
            validation_proofs={
                "sentinel":  sentinel_result,
                "physicore": physicore_result,
                "memory":    memory_result,
            },
        )

    # ── State determination ───────────────────────────────────────────────────

    def _determine_final_state(
        self,
        confidence: ConfidenceContract,
        constraints_applied: list[str],
        action_modified: Action,
        action_original: Action,
        ctx: Context,
    ) -> tuple[CertificationState, str]:
        """
        Given the assembled confidence contract, determine the final certification state.
        """
        p = confidence.success_probability

        # Critical risk always requires human
        if confidence.worst_case_risk == RiskLevel.CRITICAL:
            return (
                CertificationState.HUMAN_OVERRIDE_REQUIRED,
                f"Critical risk identified. Human decision required. (confidence={p:.0%})",
            )

        # Too uncertain for algorithm
        if p < self.human_override_threshold:
            return (
                CertificationState.HUMAN_OVERRIDE_REQUIRED,
                f"Confidence {p:.0%} below human-override threshold "
                f"{self.human_override_threshold:.0%}. Escalating.",
            )

        # Below replan threshold — goal reachable but not this action
        if p < self.replan_threshold:
            return (
                CertificationState.REPLAN_REQUIRED,
                f"Confidence {p:.0%} below replan threshold "
                f"{self.replan_threshold:.0%}. Request alternative action.",
            )

        # Below context floor — defined per context
        if p < ctx.confidence_floor:
            return (
                CertificationState.REPLAN_REQUIRED,
                f"Confidence {p:.0%} below context floor {ctx.confidence_floor:.0%}.",
            )

        # High risk with moderate confidence — request human if severe
        if (
            confidence.worst_case_risk == RiskLevel.HIGH
            and p < 0.70
        ):
            return (
                CertificationState.HUMAN_OVERRIDE_REQUIRED,
                f"High risk at confidence {p:.0%}. Human verification needed.",
            )

        # Constraints were applied — action has been modified
        if constraints_applied:
            return (
                CertificationState.EXECUTE_WITH_CONSTRAINTS,
                f"Certified with {len(constraints_applied)} constraint(s): "
                + ", ".join(constraints_applied),
            )

        # Clean pass
        return (
            CertificationState.EXECUTE,
            f"All validators passed. Confidence={p:.0%}. Execute as proposed.",
        )

    # ── Decision factory ──────────────────────────────────────────────────────

    def _make_decision(
        self,
        state: CertificationState,
        action: Action | None,
        reason: str,
        sentinel_result:  ValidationResult | None,
        physicore_result: ValidationResult | None,
        memory_result:    ValidationResult | None,
        action_original: Action,
        ctx: Context,
        t0: float,
        confidence: ConfidenceContract | None = None,
        all_failure_modes: list[FailureMode] | None = None,
        blocking_fm: FailureMode | None = None,
    ) -> CertificationDecision:

        latency_ms = (time.perf_counter() - t0) * 1000.0

        trace = DecisionTrace(
            ctx_id=ctx.ctx_id,
            action_id=action_original.action_id,
            sentinel_result=sentinel_result,
            physicore_result=physicore_result,
            memory_result=memory_result,
            confidence_contract=confidence,
            certification_state=state,
            blocking_failure_mode=blocking_fm,
            total_latency_ms=latency_ms,
        )

        decision = CertificationDecision(
            state=state,
            action=action,
            confidence=confidence,
            trace=trace,
            failure_modes=all_failure_modes or [],
            reason=reason,
        )

        # ── Emit observability ────────────────────────────────────────────────
        conf_score = confidence.success_probability if confidence else None
        fail_codes = [fm.code for fm in (all_failure_modes or [])]

        # Structured log — one line per certification, every field machine-parseable
        log_fn = _log.warning if state.value not in ("EXECUTE", "EXECUTE_WITH_CONSTRAINTS") else _log.info
        log_fn(
            "certified",
            state=state.value,
            latency_ms=round(latency_ms, 2),
            confidence=round(conf_score, 4) if conf_score is not None else None,
            trace_id=trace.trace_id,
            action_id=str(action_original.action_id),
            ctx_id=str(ctx.ctx_id),
            failure_codes=fail_codes or None,
            reason=reason,
        )

        # Prometheus
        get_metrics().record_certification(
            state=state.value,
            latency_ms=latency_ms,
            confidence=conf_score,
        )

        return decision

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _worst_failure(failures: list[FailureMode]) -> FailureMode | None:
        if not failures:
            return None
        order = list(RiskLevel)
        return max(failures, key=lambda f: order.index(f.risk_level))

    @staticmethod
    def _worst_risk(failures: list[FailureMode]) -> RiskLevel:
        if not failures:
            return RiskLevel.NONE
        order = list(RiskLevel)
        return max((f.risk_level for f in failures), key=lambda r: order.index(r))
