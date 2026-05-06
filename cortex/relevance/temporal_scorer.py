"""
Temporal Validity Scorer
========================
Axis 3 of the Relevance Engine.

Answers: is this memory still true right now?

The physical world changes. An object that was at position X five
minutes ago may not be there now. A constraint that applied during
the last task may not apply to this one. Memory has a shelf life.

This scorer evaluates:
  1. Age decay       — confidence decays as record ages (type-specific rates)
  2. Sensor anchor   — sensor-confirmed records get a significant boost
  3. Explicit window — records with invalid_at set are treated as expired
  4. Outcome recency — records used successfully recently score higher

Decay rates by memory type
--------------------------
  SPATIAL    — decays fast (5 min half-life). Objects move.
  EPISODIC   — decays medium (2 hour half-life). Events matter less over time.
  SEMANTIC   — decays slow (24 hour half-life). Facts remain stable.
  PROCEDURAL — decays very slow (7 day half-life). Procedures rarely change.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from cortex.models.memory import MemoryRecord, MemoryType


# ── Decay half-lives by memory type (seconds) ────────────────────────────────
_HALF_LIVES: dict[MemoryType, float] = {
    MemoryType.SPATIAL:    5   * 60,        #  5 minutes
    MemoryType.EPISODIC:   2   * 3600,      #  2 hours
    MemoryType.SEMANTIC:   24  * 3600,      # 24 hours
    MemoryType.PROCEDURAL: 7   * 24 * 3600, #  7 days
}

_SENSOR_ANCHOR_BOOST = 0.20    # sensor-confirmed records get +20%
_RECENT_USE_BOOST    = 0.10    # used successfully in last hour: +10%
_RECENT_USE_WINDOW   = 3600.0  # "recent" = within 1 hour


@dataclass(frozen=True)
class TemporalScore:
    record_id:       str
    age_seconds:     float
    half_life:       float
    age_decay:       float   # raw exponential decay [0, 1]
    sensor_boost:    float   # boost from sensor confirmation
    recency_boost:   float   # boost from recent successful use
    expired:         bool    # True if invalid_at has passed
    composite:       float   # final temporal score [0, 1]


class TemporalValidityScorer:
    """
    Scores how temporally valid a memory record is right now.

    The temporal score is NOT a replacement for effective_confidence —
    it is a separate axis that specifically measures how recently
    the fact encoded in this record was confirmed to be true.
    """

    def __init__(
        self,
        half_lives:          dict[MemoryType, float] | None = None,
        sensor_anchor_boost: float = _SENSOR_ANCHOR_BOOST,
        recent_use_boost:    float = _RECENT_USE_BOOST,
        recent_use_window:   float = _RECENT_USE_WINDOW,
        min_score:           float = 0.05,   # floor — never zero unless expired
    ) -> None:
        self.half_lives       = half_lives or _HALF_LIVES
        self.sensor_boost     = sensor_anchor_boost
        self.recent_use_boost = recent_use_boost
        self.recent_use_window = recent_use_window
        self.min_score        = min_score

    def score(self, record: MemoryRecord) -> TemporalScore:
        now = time.time()

        # ── Hard expiry ───────────────────────────────────────────────────────
        if record.invalid_at is not None and now > record.invalid_at:
            return TemporalScore(
                record_id=record.record_id,
                age_seconds=now - record.valid_at,
                half_life=self._half_life(record.memory_type),
                age_decay=0.0,
                sensor_boost=0.0,
                recency_boost=0.0,
                expired=True,
                composite=0.0,
            )

        # ── Age decay ─────────────────────────────────────────────────────────
        age_seconds = now - record.valid_at
        half_life   = self._half_life(record.memory_type)
        age_decay   = math.exp(-math.log(2) * age_seconds / half_life)
        age_decay   = max(self.min_score, age_decay)

        # ── Sensor anchor boost ───────────────────────────────────────────────
        sensor_boost = 0.0
        if record.is_sensor_anchored and record.sensor_confirmed_at:
            confirmation_age = now - record.sensor_confirmed_at
            # Boost decays if confirmation itself is old (use spatial half-life)
            confirmation_freshness = math.exp(
                -math.log(2) * confirmation_age / _HALF_LIVES[MemoryType.SPATIAL]
            )
            sensor_boost = self.sensor_boost * confirmation_freshness

        # ── Recent successful use boost ───────────────────────────────────────
        recency_boost = 0.0
        if (
            record.outcome_count > 0
            and record.success_count > 0
            and age_seconds <= self.recent_use_window
        ):
            recency_boost = self.recent_use_boost * (record.success_count / record.outcome_count)

        # ── Composite ─────────────────────────────────────────────────────────
        composite = min(1.0, age_decay + sensor_boost + recency_boost)

        return TemporalScore(
            record_id=record.record_id,
            age_seconds=age_seconds,
            half_life=half_life,
            age_decay=round(age_decay, 4),
            sensor_boost=round(sensor_boost, 4),
            recency_boost=round(recency_boost, 4),
            expired=False,
            composite=round(composite, 4),
        )

    def _half_life(self, memory_type: MemoryType) -> float:
        return self.half_lives.get(memory_type, _HALF_LIVES[MemoryType.EPISODIC])

    def batch_score(self, records: list[MemoryRecord]) -> list[TemporalScore]:
        return [self.score(r) for r in records]
