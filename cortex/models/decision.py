"""
Decision models
===============
CertificationDecision is the fundamental output of the Cortex Certification Gate.
Every action that touches the physical world must produce one of these before executing.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ── Certification states ─────────────────────────────────────────────────────

class CertificationState(str, Enum):
    """
    The five possible outcomes of a certification decision.

    Priority order (highest veto power first):
        SAFE_HALT > HUMAN_OVERRIDE_REQUIRED > REPLAN_REQUIRED
            > EXECUTE_WITH_CONSTRAINTS > EXECUTE
    """

    EXECUTE = "EXECUTE"
    """Action is safe, feasible, and consistent with memory. Execute as proposed."""

    EXECUTE_WITH_CONSTRAINTS = "EXECUTE_WITH_CONSTRAINTS"
    """
    Action is feasible but requires modification.
    The certified action is in CertificationDecision.action — use that, not the original.
    Modifications may include: speed reduction, force capping, trajectory adjustment.
    """

    REPLAN_REQUIRED = "REPLAN_REQUIRED"
    """
    Goal is achievable but the proposed action cannot be certified.
    Return the context to the AI planner and request a different action.
    """

    SAFE_HALT = "SAFE_HALT"
    """
    No safe action is available within the current planning horizon.
    Stop all motion. Hold current pose. Log and await further instruction.
    """

    HUMAN_OVERRIDE_REQUIRED = "HUMAN_OVERRIDE_REQUIRED"
    """
    Uncertainty or conflicting constraints cannot be resolved algorithmically.
    Escalate to a human operator with the full decision trace.
    """


# ── Risk levels ──────────────────────────────────────────────────────────────

class RiskLevel(str, Enum):
    NONE     = "none"
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"


# ── Failure modes ────────────────────────────────────────────────────────────

class FailureMode(BaseModel):
    """A structured risk identified during certification."""

    code: str = Field(description="Machine-readable risk code, e.g. 'collision_risk'")
    description: str = Field(description="Human-readable explanation")
    risk_level: RiskLevel = Field(default=RiskLevel.MEDIUM)
    source: str = Field(
        description="Which validator raised this: 'sentinel' | 'physicore' | 'memory'",
        default="unknown",
    )
    mitigable: bool = Field(
        description="True if EXECUTE_WITH_CONSTRAINTS can address this risk",
        default=False,
    )

    model_config = {"frozen": True}


# ── Confidence contract ───────────────────────────────────────────────────────

class ValidationResult(BaseModel):
    """Result from a single validator subsystem."""

    passed: bool
    score: float = Field(ge=0.0, le=1.0, description="Confidence [0,1]")
    failure_modes: list[FailureMode] = Field(default_factory=list)
    latency_ms: float = Field(default=0.0)
    notes: str = Field(default="")

    model_config = {"frozen": True}


class ConfidenceContract(BaseModel):
    """
    Structured execution guarantee replacing a single scalar probability.

    This is what Cortex guarantees, not just what it predicts.
    Every field is derived from an independent validation source.
    """

    success_probability: float = Field(
        ge=0.0, le=1.0,
        description="P(action achieves goal | current validated context)"
    )
    stability_margin: float = Field(
        ge=0.0,
        description="Distance to nearest constraint violation as percentage of limit"
    )
    worst_case_risk: RiskLevel = Field(
        description="Worst risk level identified across all validators"
    )
    uncertainty_sources: list[str] = Field(
        default_factory=list,
        description="Named sources of remaining uncertainty after validation"
    )
    validity_window_ms: float = Field(
        description="How long this certification remains valid in milliseconds"
    )
    validation_proofs: dict[str, ValidationResult] = Field(
        default_factory=dict,
        description="Per-validator results: {'sentinel': ..., 'physicore': ..., 'memory': ...}"
    )

    model_config = {"frozen": True}

    @property
    def is_high_confidence(self) -> bool:
        return (
            self.success_probability >= 0.80
            and self.worst_case_risk in (RiskLevel.NONE, RiskLevel.LOW)
        )

    @property
    def is_acceptable(self) -> bool:
        return (
            self.success_probability >= 0.60
            and self.worst_case_risk not in (RiskLevel.HIGH, RiskLevel.CRITICAL)
        )


# ── Decision trace ────────────────────────────────────────────────────────────

class DecisionTrace(BaseModel):
    """
    Full audit record of a certification decision.

    Stored for every certification, regardless of outcome.
    Answers: why did Cortex make this decision?
    """

    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    ctx_id: str = Field(description="ID of the Context that was active")
    action_id: str = Field(description="ID of the action that was proposed")

    sentinel_result:  ValidationResult | None = Field(default=None)
    physicore_result: ValidationResult | None = Field(default=None)
    memory_result:    ValidationResult | None = Field(default=None)

    confidence_contract: ConfidenceContract | None = Field(default=None)
    certification_state: CertificationState = Field(description="Final decision")

    rejected_alternatives: list[str] = Field(
        default_factory=list,
        description="IDs of alternative actions that were considered and rejected"
    )
    blocking_failure_mode: FailureMode | None = Field(
        default=None,
        description="The failure mode that caused a non-EXECUTE decision, if any"
    )

    total_latency_ms: float = Field(default=0.0)
    timestamp: float = Field(default_factory=time.time)

    model_config = {"frozen": True}


# ── The certification decision ────────────────────────────────────────────────

class CertificationDecision(BaseModel):
    """
    The output of cortex.certify(action, ctx).

    This is the only thing that should ever reach robot.execute().
    """

    decision_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    state: CertificationState
    action: Any = Field(
        description=(
            "The action to execute. For EXECUTE: same as proposed. "
            "For EXECUTE_WITH_CONSTRAINTS: a modified version. "
            "For all other states: None."
        )
    )

    confidence: ConfidenceContract | None = Field(
        default=None,
        description="Only populated for EXECUTE and EXECUTE_WITH_CONSTRAINTS"
    )
    trace: DecisionTrace = Field(description="Full audit trail of this decision")
    failure_modes: list[FailureMode] = Field(default_factory=list)
    reason: str = Field(
        default="",
        description="Human-readable explanation of the decision"
    )

    timestamp: float = Field(default_factory=time.time)

    # Cryptographic signature over (ctx_id, state, confidence.success_probability, timestamp_ns)
    # Populated by CertificationGate when a CTXSigner is configured.
    cortex_signature: str | None = Field(
        default=None,
        description="Hex-encoded Ed25519 signature covering key decision fields",
    )

    model_config = {"frozen": True}

    # ── convenience properties ────────────────────────────────────────────────

    @property
    def approved(self) -> bool:
        """True if execution is permitted (with or without constraints)."""
        return self.state in (
            CertificationState.EXECUTE,
            CertificationState.EXECUTE_WITH_CONSTRAINTS,
        )

    @property
    def blocked(self) -> bool:
        return not self.approved

    @property
    def requires_replan(self) -> bool:
        return self.state == CertificationState.REPLAN_REQUIRED

    @property
    def requires_human(self) -> bool:
        return self.state == CertificationState.HUMAN_OVERRIDE_REQUIRED

    def __repr__(self) -> str:
        prob = (
            f"{self.confidence.success_probability:.0%}"
            if self.confidence
            else "n/a"
        )
        return (
            f"CertificationDecision("
            f"state={self.state.value}, "
            f"confidence={prob}, "
            f"latency={self.trace.total_latency_ms:.1f}ms)"
        )
