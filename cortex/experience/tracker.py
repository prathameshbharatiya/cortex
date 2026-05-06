"""
Experience Tracker
==================
Closes the feedback loop.

When an action executes and produces an outcome, the Experience Tracker:

  1. Writes the ExperienceRecord — immutable, timestamped, causally indexed
  2. Computes the surprise score — how far was actual from predicted?
  3. Updates memory record outcome counts — so the Relevance Engine
     knows which records historically led to success or failure
  4. Indexes by causal factors — so similar situations in the future
     surface relevant failure history
  5. Feeds back to the Gateway — updates the underlying memory stores
     so outcome data is available at next retrieval

The surprise score is the most important output.
High surprise on a SUCCESS means Cortex was more pessimistic than
necessary — confidence thresholds can be relaxed for this situation.
High surprise on a FAILURE means Cortex was over-confident —
thresholds should tighten.

This is how Cortex gets smarter without retraining.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import defaultdict
from typing import Any

from cortex.experience.record import ExperienceRecord, OutcomeSpec, OutcomeClass, CausalFactor
from cortex.models.decision   import CertificationDecision, CertificationState
from cortex.models.context    import Context
from cortex.models.memory     import MemoryRecord, MemoryType, OutcomeTag
from cortex.memory.gateway    import MemoryGateway


# ── Surprise computation ──────────────────────────────────────────────────────

def _compute_surprise(
    predicted_confidence: float,
    outcome:              OutcomeSpec,
) -> float:
    """
    Surprise = |predicted_probability - actual_outcome|

    SUCCESS → actual = 1.0
    PARTIAL → actual = 0.5
    FAILURE / UNSAFE / TIMEOUT → actual = 0.0
    INTERRUPTED / SKIPPED → actual = 0.5 (ambiguous)
    """
    actual_map = {
        OutcomeClass.SUCCESS:     1.0,
        OutcomeClass.PARTIAL:     0.5,
        OutcomeClass.FAILURE:     0.0,
        OutcomeClass.UNSAFE:      0.0,
        OutcomeClass.TIMEOUT:     0.1,
        OutcomeClass.INTERRUPTED: 0.5,
        OutcomeClass.SKIPPED:     0.5,
    }
    actual_outcome = actual_map.get(outcome.outcome_class, 0.5)
    return round(abs(predicted_confidence - actual_outcome), 4)


# ── Tracker ───────────────────────────────────────────────────────────────────

class ExperienceTracker:
    """
    Records execution outcomes and feeds them back into memory.

    Thread-safe. All writes are non-blocking — the tracker queues
    experience records and processes them asynchronously so robot
    control loops are never blocked by tracker I/O.
    """

    def __init__(
        self,
        gateway:           MemoryGateway | None = None,
        platform_id:       str = "",
        deployment_id:     str = "",
        async_writes:      bool = True,
        high_surprise_threshold: float = 0.5,
        max_records:       int = 100_000,
    ) -> None:
        self.gateway               = gateway
        self.platform_id           = platform_id
        self.deployment_id         = deployment_id
        self.async_writes          = async_writes
        self.high_surprise_threshold = high_surprise_threshold

        # In-process store (always available, regardless of gateway)
        self._records:        dict[str, ExperienceRecord] = {}
        self._by_task:        dict[str, list[str]] = defaultdict(list)
        self._by_memory_id:   dict[str, list[str]] = defaultdict(list)
        self._by_outcome:     dict[str, list[str]] = defaultdict(list)
        self._high_surprise:  list[str] = []
        self._memory_updates: dict[str, dict[str, int]] = defaultdict(
            lambda: {"count": 0, "successes": 0}
        )
        self._lock  = threading.RLock()
        self._stats = defaultdict(int)
        self._max   = max_records

    # ── Main entry points ─────────────────────────────────────────────────────

    def record(
        self,
        decision: CertificationDecision,
        ctx:      Context,
        outcome:  OutcomeSpec,
    ) -> ExperienceRecord:
        """
        Record the outcome of a certified action execution.

        Call this immediately after robot.execute() returns.

        Parameters
        ----------
        decision : the CertificationDecision that authorised the action
        ctx      : the Context that was active when the action was certified
        outcome  : what actually happened (provided by the robot control layer)

        Returns
        -------
        The written ExperienceRecord.
        """
        # Extract predicted confidence
        predicted_confidence = 0.5
        if decision.confidence:
            predicted_confidence = decision.confidence.success_probability

        # Compute surprise
        surprise = _compute_surprise(predicted_confidence, outcome)

        # Extract memory record IDs that informed this decision
        memory_ids = [r.record_id for r in ctx.memory_records]

        exp = ExperienceRecord(
            trace_id=decision.trace.trace_id,
            ctx_id=ctx.ctx_id,
            action_id=decision.trace.action_id,
            certification_state=decision.state.value,
            task=ctx.task,
            robot_ee_position=list(ctx.robot_state.ee_position),
            memory_record_ids=memory_ids,
            predicted_confidence=predicted_confidence,
            outcome=outcome,
            surprise_score=surprise,
            platform_id=self.platform_id,
            deployment_id=self.deployment_id,
        )

        if self.async_writes:
            t = threading.Thread(
                target=self._write,
                args=(exp, ctx.memory_records),
                daemon=True,
            )
            t.start()
        else:
            self._write(exp, ctx.memory_records)

        return exp

    def record_simple(
        self,
        task:                 str,
        outcome_class:        OutcomeClass,
        trace_id:             str = "",
        ctx_id:               str = "",
        action_id:            str = "",
        certification_state:  str = "EXECUTE",
        memory_record_ids:    list[str] | None = None,
        predicted_confidence: float = 0.5,
        ee_position:          list[float] | None = None,
        description:          str = "",
        causal_factors:       list[CausalFactor] | None = None,
    ) -> ExperienceRecord:
        """
        Simplified entry point for recording without a full Decision/Context.
        Useful for testing and for recording outcomes from external systems.
        """
        outcome = OutcomeSpec(
            outcome_class=outcome_class,
            description=description,
            causal_factors=causal_factors or [],
        )
        surprise = _compute_surprise(predicted_confidence, outcome)

        exp = ExperienceRecord(
            trace_id=trace_id or str(uuid.uuid4()),
            ctx_id=ctx_id or str(uuid.uuid4()),
            action_id=action_id or str(uuid.uuid4()),
            certification_state=certification_state,
            task=task,
            robot_ee_position=ee_position or [],
            memory_record_ids=memory_record_ids or [],
            predicted_confidence=predicted_confidence,
            outcome=outcome,
            surprise_score=surprise,
            platform_id=self.platform_id,
            deployment_id=self.deployment_id,
        )

        self._write(exp, [])
        return exp

    # ── Query interface ───────────────────────────────────────────────────────

    def get(self, experience_id: str) -> ExperienceRecord | None:
        with self._lock:
            return self._records.get(experience_id)

    def by_task(self, task: str, limit: int = 50) -> list[ExperienceRecord]:
        with self._lock:
            ids = self._by_task.get(task, [])[-limit:]
            return [self._records[i] for i in ids if i in self._records]

    def by_memory_record(self, record_id: str) -> list[ExperienceRecord]:
        with self._lock:
            ids = self._by_memory_id.get(record_id, [])
            return [self._records[i] for i in ids if i in self._records]

    def high_surprise_experiences(self, limit: int = 20) -> list[ExperienceRecord]:
        with self._lock:
            ids = self._high_surprise[-limit:]
            return [self._records[i] for i in ids if i in self._records]

    def outcome_stats(self, task: str | None = None) -> dict[str, Any]:
        """Aggregate outcome statistics, optionally filtered by task."""
        with self._lock:
            if task:
                ids = self._by_task.get(task, [])
                records = [self._records[i] for i in ids if i in self._records]
            else:
                records = list(self._records.values())

        if not records:
            return {"total": 0}

        counts: dict[str, int] = defaultdict(int)
        total_surprise = 0.0
        for r in records:
            counts[r.outcome.outcome_class.value] += 1
            total_surprise += r.surprise_score

        total = len(records)
        successes = counts.get(OutcomeClass.SUCCESS.value, 0)

        return {
            "total":           total,
            "success_count":   successes,
            "failure_count":   counts.get(OutcomeClass.FAILURE.value, 0),
            "partial_count":   counts.get(OutcomeClass.PARTIAL.value, 0),
            "unsafe_count":    counts.get(OutcomeClass.UNSAFE.value, 0),
            "success_rate":    round(successes / total, 4) if total > 0 else 0.0,
            "avg_surprise":    round(total_surprise / total, 4) if total > 0 else 0.0,
            "high_surprise":   len(self._high_surprise),
            "outcome_counts":  dict(counts),
        }

    def memory_outcome_stats(self, record_id: str) -> dict[str, Any]:
        """How did actions informed by this memory record turn out?"""
        experiences = self.by_memory_record(record_id)
        if not experiences:
            return {"total": 0, "success_rate": 0.5, "avg_surprise": 0.0}

        successes = sum(1 for e in experiences if e.succeeded)
        total     = len(experiences)
        avg_surp  = sum(e.surprise_score for e in experiences) / total

        return {
            "total":        total,
            "successes":    successes,
            "failures":     total - successes,
            "success_rate": round(successes / total, 4),
            "avg_surprise": round(avg_surp, 4),
        }

    def count(self) -> int:
        with self._lock:
            return len(self._records)

    def stats(self) -> dict:
        with self._lock:
            return dict(self._stats)

    # ── Internal write ────────────────────────────────────────────────────────

    def _write(
        self,
        exp:     ExperienceRecord,
        records: list[MemoryRecord],
    ) -> None:
        with self._lock:
            # Evict oldest if at capacity
            if len(self._records) >= self._max:
                oldest = min(self._records.values(), key=lambda r: r.timestamp)
                self._evict(oldest)

            # Store
            self._records[exp.experience_id] = exp
            self._by_task[exp.task].append(exp.experience_id)
            self._by_outcome[exp.outcome.outcome_class.value].append(exp.experience_id)

            for mid in exp.memory_record_ids:
                self._by_memory_id[mid].append(exp.experience_id)
                # Update in-process outcome counters
                self._memory_updates[mid]["count"] += 1
                if exp.succeeded:
                    self._memory_updates[mid]["successes"] += 1

            if exp.is_high_surprise:
                self._high_surprise.append(exp.experience_id)

            self._stats["total_recorded"] += 1
            if exp.succeeded:
                self._stats["successes"] += 1
            else:
                self._stats["failures"] += 1
            if exp.is_high_surprise:
                self._stats["high_surprise"] += 1

        # Write back to Gateway (outside lock to avoid blocking)
        self._write_back_to_gateway(exp, records)

    def _write_back_to_gateway(
        self,
        exp:     ExperienceRecord,
        records: list[MemoryRecord],
    ) -> None:
        """
        Write updated outcome counts back to the Gateway/memory stores
        so the Relevance Engine picks up the latest outcome history.
        """
        if not self.gateway or not records:
            return

        for record in records:
            stats = self._get_memory_stats(record.record_id)
            if stats["count"] == 0:
                continue

            # Build updated record with new outcome counts
            new_tag = (
                OutcomeTag.SUCCESS if stats["success_rate"] >= 0.7
                else OutcomeTag.FAILURE if stats["success_rate"] < 0.3
                else OutcomeTag.PARTIAL
            )
            updated = MemoryRecord(
                record_id=record.record_id,
                source=record.source,
                memory_type=record.memory_type,
                content=record.content,
                content_text=record.content_text,
                valid_at=record.valid_at,
                invalid_at=record.invalid_at,
                base_confidence=record.base_confidence,
                sensor_anchor=record.sensor_anchor,
                sensor_confirmed_at=record.sensor_confirmed_at,
                outcome_tag=new_tag,
                outcome_count=stats["count"],
                success_count=stats["successes"],
                retrieval_latency_ms=record.retrieval_latency_ms,
            )
            try:
                self.gateway.store(updated)
            except Exception:
                pass   # never block on Gateway failure

    def _get_memory_stats(self, record_id: str) -> dict[str, Any]:
        with self._lock:
            s = self._memory_updates.get(record_id, {"count": 0, "successes": 0})
            count     = s["count"]
            successes = s["successes"]
        return {
            "count":        count,
            "successes":    successes,
            "success_rate": (successes / count) if count > 0 else 0.5,
        }

    def _evict(self, exp: ExperienceRecord) -> None:
        """Remove an experience record from all indices."""
        eid = exp.experience_id
        self._records.pop(eid, None)
        task_list = self._by_task.get(exp.task, [])
        if eid in task_list:
            task_list.remove(eid)
        for mid in exp.memory_record_ids:
            mid_list = self._by_memory_id.get(mid, [])
            if eid in mid_list:
                mid_list.remove(eid)
        out_list = self._by_outcome.get(exp.outcome.outcome_class.value, [])
        if eid in out_list:
            out_list.remove(eid)
        if eid in self._high_surprise:
            self._high_surprise.remove(eid)
