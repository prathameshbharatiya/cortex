"""
Memory models
=============
MemoryRecord is the canonical internal format for any memory retrieved
from any external store. The Memory Gateway normalises all sources into this.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    EPISODIC   = "episodic"    # "Last time I did X, Y happened"
    SEMANTIC   = "semantic"    # "Object A is a type of B"
    SPATIAL    = "spatial"     # "Object A is at position P"
    PROCEDURAL = "procedural"  # "To do X, follow steps 1,2,3"


class OutcomeTag(str, Enum):
    SUCCESS  = "success"
    FAILURE  = "failure"
    PARTIAL  = "partial"
    UNKNOWN  = "unknown"


class MemoryRecord(BaseModel):
    """
    A single memory record in Cortex's canonical internal format.

    Key fields:
    - valid_at / invalid_at: temporal validity. If a record has no invalid_at,
      it is considered still true. Records without valid_at cannot be used for
      physical action certification.
    - sensor_anchor: if the content of this record has been confirmed by a
      live sensor reading, that confirmation is stored here. Sensor-anchored
      records receive elevated weight in the Relevance Engine.
    - outcome_tag: was acting on this record previously successful?
    """

    record_id:   str = Field(default_factory=lambda: str(uuid.uuid4()))
    source:      str = Field(description="Which system provided this record")
    memory_type: MemoryType

    content: Any = Field(description="The memory payload (normalised to canonical format)")
    content_text: str = Field(
        default="",
        description="Human-readable summary of content (for logging and audit)"
    )

    # ── Temporal validity ────────────────────────────────────────────────────
    valid_at:   float = Field(
        default_factory=time.time,
        description="Unix timestamp when this fact was known to be true"
    )
    invalid_at: float | None = Field(
        default=None,
        description="Unix timestamp when this fact became false. None = still valid."
    )

    # ── Confidence ──────────────────────────────────────────────────────────
    base_confidence: float = Field(
        ge=0.0, le=1.0,
        default=1.0,
        description="Source-reported confidence at time of retrieval"
    )

    # ── Sensor grounding ─────────────────────────────────────────────────────
    sensor_anchor: dict[str, Any] | None = Field(
        default=None,
        description=(
            "If this memory has been confirmed by live sensor data, "
            "the confirming sensor reading is stored here."
        )
    )
    sensor_confirmed_at: float | None = Field(default=None)

    # ── Outcome history ──────────────────────────────────────────────────────
    outcome_tag: OutcomeTag = Field(default=OutcomeTag.UNKNOWN)
    outcome_count: int = Field(
        default=0,
        description="How many times this record was used and produced an outcome"
    )
    success_count: int = Field(default=0)

    # ── Meta ─────────────────────────────────────────────────────────────────
    retrieval_latency_ms: float = Field(default=0.0)

    model_config = {"frozen": True}

    @property
    def is_currently_valid(self) -> bool:
        """True if this record has not been marked as invalid."""
        if self.invalid_at is None:
            return True
        return time.time() < self.invalid_at

    @property
    def age_seconds(self) -> float:
        return time.time() - self.valid_at

    @property
    def is_sensor_anchored(self) -> bool:
        return self.sensor_anchor is not None

    @property
    def success_rate(self) -> float:
        if self.outcome_count == 0:
            return 0.5  # unknown → neutral prior
        return self.success_count / self.outcome_count

    @property
    def effective_confidence(self) -> float:
        """
        Confidence adjusted for age and sensor grounding.
        Records decay over time if not sensor-confirmed.
        """
        age_hours = self.age_seconds / 3600.0
        decay = max(0.5, 1.0 - (age_hours * 0.05))   # 5% decay per hour, floor 0.5

        if self.is_sensor_anchored:
            decay = min(1.0, decay + 0.2)             # sensor-confirmed: +20% boost

        return self.base_confidence * decay

    def __repr__(self) -> str:
        return (
            f"MemoryRecord("
            f"type={self.memory_type.value}, "
            f"confidence={self.effective_confidence:.2f}, "
            f"valid={self.is_currently_valid}, "
            f"source={self.source!r})"
        )
