"""
Memory Consistency Validator
=============================
Priority 3 — Soft constraint. Does not block execution alone,
but reduces confidence and can escalate to REPLAN or HUMAN_OVERRIDE.

Memory validation checks:
  1. Temporal validity — are the memory records in the CTX still valid?
  2. Sensor contradiction — does any memory record contradict live sensor data?
  3. Object state consistency — does memory agree with what the scene graph sees?
  4. Outcome-weighted reliability — are we about to use memories that
     previously led to failures in similar situations?
  5. Memory staleness — are the records fresh enough to trust?

A memory inconsistency alone does not block execution.
It raises the uncertainty, reduces the confidence contract,
and — if severe enough — triggers REPLAN or HUMAN_OVERRIDE.
"""

from __future__ import annotations

import time

import numpy as np

from cortex.models.action  import Action
from cortex.models.context import Context
from cortex.models.memory  import MemoryRecord, MemoryType, OutcomeTag
from cortex.models.decision import FailureMode, RiskLevel, ValidationResult

# ── Thresholds ────────────────────────────────────────────────────────────────
STALE_THRESHOLD_SECONDS   = 3600.0   # 1 hour — records older than this are stale
FAILURE_RATE_THRESHOLD    = 0.3      # >30% failure rate → flag the record
CONTRADICTION_DIST_M      = 0.10     # >10cm mismatch between memory and sensor = contradiction
MIN_MEMORY_CONFIDENCE     = 0.50     # below this, memory basis is considered unreliable


class MemoryValidator:
    """
    Validates that the memory basis of the context is consistent with
    current physical reality and historically reliable.
    """

    def __init__(
        self,
        stale_threshold_s:     float = STALE_THRESHOLD_SECONDS,
        failure_rate_threshold: float = FAILURE_RATE_THRESHOLD,
        contradiction_dist_m:  float = CONTRADICTION_DIST_M,
        min_confidence:        float = MIN_MEMORY_CONFIDENCE,
    ) -> None:
        self.stale_threshold_s      = stale_threshold_s
        self.failure_rate_threshold = failure_rate_threshold
        self.contradiction_dist_m   = contradiction_dist_m
        self.min_confidence         = min_confidence

    # ── Main entry point ──────────────────────────────────────────────────────

    def validate(self, action: Action, ctx: Context) -> ValidationResult:
        t0 = time.perf_counter()
        failure_modes: list[FailureMode] = []
        confidence_scores: list[float] = []

        if not ctx.memory_records:
            # No memory — not a failure, just lower confidence
            return ValidationResult(
                passed=True,
                score=0.7,
                failure_modes=[],
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                notes="Memory: no records in context — operating without memory basis",
            )

        for record in ctx.memory_records:
            # Check 1 — Is the record currently valid?
            fm = self._check_validity(record)
            if fm:
                failure_modes.append(fm)

            # Check 2 — Is the record stale?
            fm, staleness_score = self._check_staleness(record)
            if fm:
                failure_modes.append(fm)
            confidence_scores.append(staleness_score * record.effective_confidence)

            # Check 3 — Does the record contradict the live scene?
            fm, consistency_score = self._check_scene_consistency(record, ctx)
            if fm:
                failure_modes.append(fm)
            confidence_scores.append(consistency_score)

            # Check 4 — Outcome reliability
            fm, outcome_score = self._check_outcome_reliability(record)
            if fm:
                failure_modes.append(fm)
            confidence_scores.append(outcome_score)

        # Overall memory confidence
        overall_confidence = float(np.mean(confidence_scores)) if confidence_scores else 0.5
        overall_confidence = min(overall_confidence, ctx.memory_confidence)

        latency_ms = (time.perf_counter() - t0) * 1000.0

        if overall_confidence < self.min_confidence:
            failure_modes.append(FailureMode(
                code="low_memory_confidence",
                description=(
                    f"Overall memory confidence {overall_confidence:.2f} is below "
                    f"minimum threshold {self.min_confidence:.2f}. "
                    "Context may not reliably represent current state."
                ),
                risk_level=RiskLevel.MEDIUM,
                source="memory",
                mitigable=True,
            ))

        hard_failures = [f for f in failure_modes if not f.mitigable]
        notes = (
            f"Memory: confidence={overall_confidence:.2f}, "
            f"records={len(ctx.memory_records)}, "
            f"issues={len(failure_modes)}"
        )

        return ValidationResult(
            passed=len(hard_failures) == 0,
            score=overall_confidence,
            failure_modes=failure_modes,
            latency_ms=latency_ms,
            notes=notes,
        )

    # ── Individual checks ─────────────────────────────────────────────────────

    def _check_validity(self, record: MemoryRecord) -> FailureMode | None:
        """Has this record been explicitly invalidated?"""
        if not record.is_currently_valid:
            return FailureMode(
                code="invalid_memory_record",
                description=(
                    f"Memory record {record.record_id[:8]} "
                    f"(type={record.memory_type.value}) is marked invalid "
                    f"(expired at {record.invalid_at:.0f})"
                ),
                risk_level=RiskLevel.MEDIUM,
                source="memory",
                mitigable=True,
            )
        return None

    def _check_staleness(self, record: MemoryRecord) -> tuple[FailureMode | None, float]:
        """Is the record older than the staleness threshold?"""
        age = record.age_seconds
        if age > self.stale_threshold_s:
            return (
                FailureMode(
                    code="stale_memory_record",
                    description=(
                        f"Memory record {record.record_id[:8]} "
                        f"is {age/3600:.1f} hours old "
                        f"(threshold: {self.stale_threshold_s/3600:.1f}h)"
                    ),
                    risk_level=RiskLevel.LOW,
                    source="memory",
                    mitigable=True,
                ),
                max(0.3, 1.0 - (age / self.stale_threshold_s) * 0.5),
            )

        # Freshness score: 1.0 at t=0, 0.5 at threshold
        freshness = max(0.5, 1.0 - (age / self.stale_threshold_s) * 0.5)
        return None, freshness

    def _check_scene_consistency(
        self, record: MemoryRecord, ctx: Context
    ) -> tuple[FailureMode | None, float]:
        """
        Does the memory record's content agree with what the scene graph reports?
        Focus on spatial records that make claims about object positions.
        """
        if record.memory_type != MemoryType.SPATIAL:
            return None, 1.0

        content = record.content
        if not isinstance(content, dict):
            return None, 0.9

        obj_id = content.get("object_id")
        claimed_pos = content.get("position")
        if obj_id is None or claimed_pos is None:
            return None, 0.9

        scene_obj = ctx.scene_graph.get_object(obj_id)
        if scene_obj is None:
            # Object not in scene — might be occluded, not necessarily a contradiction
            return None, 0.75

        claimed = np.array(claimed_pos)
        actual  = np.array(scene_obj.position)
        dist = float(np.linalg.norm(claimed - actual))

        if dist > self.contradiction_dist_m:
            return (
                FailureMode(
                    code="memory_sensor_contradiction",
                    description=(
                        f"Memory claims object '{obj_id}' at {claimed_pos}, "
                        f"but sensor reports it at {scene_obj.position} "
                        f"(discrepancy: {dist:.3f}m > {self.contradiction_dist_m:.3f}m)"
                    ),
                    risk_level=RiskLevel.MEDIUM,
                    source="memory",
                    mitigable=True,
                ),
                max(0.0, 1.0 - dist / (self.contradiction_dist_m * 3)),
            )

        # Good consistency — score based on proximity
        consistency = max(0.5, 1.0 - dist / self.contradiction_dist_m)
        return None, consistency

    def _check_outcome_reliability(
        self, record: MemoryRecord
    ) -> tuple[FailureMode | None, float]:
        """
        Is this memory record historically associated with failures?
        Records with high failure rates are flagged as warnings.
        """
        if record.outcome_count < 3:
            return None, 0.8  # too few uses to judge

        failure_rate = 1.0 - record.success_rate
        if failure_rate > self.failure_rate_threshold:
            return (
                FailureMode(
                    code="high_failure_rate_memory",
                    description=(
                        f"Memory record {record.record_id[:8]} has "
                        f"{failure_rate:.0%} failure rate "
                        f"({record.outcome_count - record.success_count} failures "
                        f"out of {record.outcome_count} uses)"
                    ),
                    risk_level=RiskLevel.MEDIUM,
                    source="memory",
                    mitigable=True,
                ),
                max(0.2, record.success_rate),
            )

        return None, record.success_rate
