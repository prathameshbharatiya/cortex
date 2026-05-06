"""
Experience Record
=================
The fundamental unit of learning in Cortex.

Every action that executes produces an ExperienceRecord.
The record links three things together with a causal index:

  1. The decision — what Cortex certified (from DecisionTrace)
  2. The context  — what the world looked like (CTX snapshot)
  3. The outcome  — what actually happened in the physical world

This linkage is what makes Phase 5 possible:
  - The Relevance Engine can look up outcome history per memory record
  - The Lifecycle Manager can weight records by outcome quality
  - Failures can be surfaced as warnings in future similar situations
  - Surprise scores (predicted vs actual) calibrate confidence contracts

The ExperienceRecord is write-once and immutable.
It is the ground truth of what happened — it cannot be edited.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ── Outcome classification ────────────────────────────────────────────────────

class OutcomeClass(str, Enum):
    SUCCESS          = "success"           # goal achieved, no issues
    PARTIAL          = "partial"           # goal partially achieved
    FAILURE          = "failure"           # goal not achieved
    UNSAFE           = "unsafe"            # action caused or approached harm
    TIMEOUT          = "timeout"           # action did not complete in time
    INTERRUPTED      = "interrupted"       # action was stopped externally
    SKIPPED          = "skipped"           # action was not executed (SAFE_HALT)


# ── Causal factor ─────────────────────────────────────────────────────────────

class CausalFactor(BaseModel):
    """
    An identified cause contributing to the outcome.
    Cortex attributes outcomes to specific causal factors
    so future decisions can learn from them.
    """
    factor_type: str = Field(
        description=(
            "Type: 'memory_error' | 'physics_violation' | "
            "'sensor_failure' | 'constraint_violated' | "
            "'model_error' | 'environment_change' | 'unknown'"
        )
    )
    description:      str
    memory_record_id: str | None = Field(
        default=None,
        description="ID of the memory record implicated in this cause"
    )
    confidence:       float = Field(ge=0.0, le=1.0, default=0.5)

    model_config = {"frozen": True}


# ── Outcome specification ─────────────────────────────────────────────────────

class OutcomeSpec(BaseModel):
    """
    The actual physical outcome of executing a certified action.
    Provided by the caller (robot control layer) after execution.
    """
    outcome_class:    OutcomeClass
    description:      str = ""

    # Physical measurements
    actual_ee_position:   list[float] | None = Field(default=None)
    position_error_m:     float | None = Field(
        default=None,
        description="Distance between intended and actual EE position"
    )
    force_peak_n:         float | None = Field(default=None)
    duration_ms:          float | None = Field(default=None)

    # Contact and grasp
    contact_detected:     bool  = False
    grasp_successful:     bool | None = None

    # Causal attribution
    causal_factors:       list[CausalFactor] = Field(default_factory=list)
    operator_notes:       str = ""

    model_config = {"frozen": True}

    @property
    def succeeded(self) -> bool:
        return self.outcome_class == OutcomeClass.SUCCESS

    @property
    def failed(self) -> bool:
        return self.outcome_class in (
            OutcomeClass.FAILURE,
            OutcomeClass.UNSAFE,
        )


# ── Experience record ─────────────────────────────────────────────────────────

class ExperienceRecord(BaseModel):
    """
    The immutable record of what happened when Cortex certified an action.

    Written once after execution. Never modified.
    Indexed by: task, memory_record_ids, outcome_class, surprise_score.
    """

    experience_id:    str = Field(default_factory=lambda: str(uuid.uuid4()))

    # ── What was decided ──────────────────────────────────────────────────────
    trace_id:         str = Field(description="DecisionTrace that produced this action")
    ctx_id:           str = Field(description="Context that was active")
    action_id:        str = Field(description="Action that was certified and executed")
    certification_state: str = Field(description="What Cortex decided")

    # ── What the world looked like ────────────────────────────────────────────
    task:             str
    robot_ee_position: list[float] = Field(default_factory=list)
    memory_record_ids: list[str]   = Field(
        default_factory=list,
        description="IDs of memory records that informed this decision"
    )
    predicted_confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence from ConfidenceContract at certification time"
    )

    # ── What actually happened ────────────────────────────────────────────────
    outcome:          OutcomeSpec

    # ── Surprise score ────────────────────────────────────────────────────────
    surprise_score:   float = Field(
        ge=0.0, le=1.0,
        description=(
            "Deviation from predicted outcome. High surprise = "
            "confidence contract was poorly calibrated for this situation."
        )
    )

    # ── Meta ──────────────────────────────────────────────────────────────────
    timestamp:        float = Field(default_factory=time.time)
    platform_id:      str   = Field(default="", description="Robot/platform identifier")
    deployment_id:    str   = Field(default="", description="Deployment environment identifier")

    model_config = {"frozen": True}

    @property
    def succeeded(self) -> bool:
        return self.outcome.succeeded

    @property
    def failed(self) -> bool:
        return self.outcome.failed

    @property
    def is_high_surprise(self) -> bool:
        return self.surprise_score > 0.5

    def __repr__(self) -> str:
        return (
            f"ExperienceRecord("
            f"outcome={self.outcome.outcome_class.value}, "
            f"surprise={self.surprise_score:.2f}, "
            f"memory_records={len(self.memory_record_ids)})"
        )
