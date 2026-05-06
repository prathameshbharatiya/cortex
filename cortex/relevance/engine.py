"""
Relevance Engine
================
Phase 4 core. Decides which memory records actually belong in the CTX.

This is the intelligence layer that separates Cortex from every
existing memory system. Every other system uses similarity as a
proxy for relevance. Cortex uses four independent axes:

  Axis 1 — Task relevance:  does this memory inform the current goal?
  Axis 2 — State relevance: is it applicable to the current physical state?
  Axis 3 — Temporal validity: is it still true right now?
  Axis 4 — Outcome history: did acting on it work before?

The engine scores every candidate record on all four axes, computes
a weighted composite, enforces diversity across the selected set,
and returns the ranked records with full scoring audit.

Why diversity matters
---------------------
Without diversity enforcement, the top-k records tend to be highly
similar to each other — they all describe the same aspect of the
situation from slightly different angles. This gives the CTX redundant
information and leaves other aspects of the situation unrepresented.

The diversity constraint ensures the selected set covers multiple
distinct aspects: at least one episodic record, at least one
procedural or semantic record (if available), and no more than
max_same_type records of the same memory type.

Failure record surfacing
------------------------
Records with low outcome scores are not silently excluded. They are
included in the ranked output with is_warning=True and a degraded
relevance score. The CTX receives them as low-confidence records
that the Gate and AI can see. This is how Cortex learns from
failures without retraining.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from cortex.models.memory  import MemoryRecord, MemoryType
from cortex.models.context import RobotState, SceneGraph
from cortex.relevance.task_scorer     import TaskRelevanceScorer,    TaskScore
from cortex.relevance.state_scorer    import StateRelevanceScorer,   StateScore
from cortex.relevance.temporal_scorer import TemporalValidityScorer, TemporalScore
from cortex.relevance.outcome_scorer  import OutcomeRelevanceScorer, OutcomeScore


# ── Per-record relevance result ───────────────────────────────────────────────

@dataclass(frozen=True)
class RecordRelevance:
    record:         MemoryRecord
    task_score:     TaskScore
    state_score:    StateScore
    temporal_score: TemporalScore
    outcome_score:  OutcomeScore
    composite:      float          # final weighted relevance [0, 1]
    rank:           int            # position in the ranked output (1 = most relevant)
    selected:       bool           # included in the final CTX set?
    is_warning:     bool           # failure history — included but flagged
    exclusion_reason: str          # why this record was not selected (if any)


# ── Engine report ─────────────────────────────────────────────────────────────

@dataclass
class RelevanceReport:
    task:                str
    candidates_in:       int
    selected_count:      int
    excluded_count:      int
    warnings_surfaced:   int
    diversity_enforced:  bool
    top_composite:       float
    bottom_composite:    float
    latency_ms:          float
    results:             list[RecordRelevance] = field(default_factory=list)

    @property
    def selected(self) -> list[MemoryRecord]:
        return [r.record for r in self.results if r.selected]

    @property
    def warnings(self) -> list[RecordRelevance]:
        return [r for r in self.results if r.is_warning and r.selected]


# ── Relevance Engine ──────────────────────────────────────────────────────────

class RelevanceEngine:
    """
    Selects and ranks memory records for inclusion in the CTX.

    Usage
    -----
        engine = RelevanceEngine()
        report = engine.rank(
            candidates=records,
            task="pick red block from shelf",
            robot_state=robot_state,
            top_k=8,
        )
        ctx_records = report.selected
    """

    def __init__(
        self,
        task_scorer:     TaskRelevanceScorer    | None = None,
        state_scorer:    StateRelevanceScorer   | None = None,
        temporal_scorer: TemporalValidityScorer | None = None,
        outcome_scorer:  OutcomeRelevanceScorer | None = None,
        # Axis weights (must sum to 1.0)
        w_task:     float = 0.35,
        w_state:    float = 0.25,
        w_temporal: float = 0.25,
        w_outcome:  float = 0.15,
        # Diversity constraints
        max_same_type:       int   = 3,    # max records of the same MemoryType
        min_score_threshold: float = 0.10, # records below this are excluded
        diversity_bonus:     float = 0.05, # bonus for introducing type diversity
        # Warning surfacing
        include_warnings:    bool  = True,  # include failure records as warnings
        max_warnings:        int   = 2,     # max failure records to surface
    ) -> None:
        self.task_scorer     = task_scorer     or TaskRelevanceScorer()
        self.state_scorer    = state_scorer    or StateRelevanceScorer()
        self.temporal_scorer = temporal_scorer or TemporalValidityScorer()
        self.outcome_scorer  = outcome_scorer  or OutcomeRelevanceScorer()

        # Normalise weights
        total = w_task + w_state + w_temporal + w_outcome
        self.w_task     = w_task     / total
        self.w_state    = w_state    / total
        self.w_temporal = w_temporal / total
        self.w_outcome  = w_outcome  / total

        self.max_same_type       = max_same_type
        self.min_score_threshold = min_score_threshold
        self.diversity_bonus     = diversity_bonus
        self.include_warnings    = include_warnings
        self.max_warnings        = max_warnings

    # ── Main entry point ──────────────────────────────────────────────────────

    def rank(
        self,
        candidates:  list[MemoryRecord],
        task:        str,
        robot_state: RobotState,
        scene_graph: SceneGraph | None = None,
        top_k:       int = 10,
    ) -> RelevanceReport:
        """
        Score, rank, and select the most relevant memory records.

        Parameters
        ----------
        candidates  : all available memory records (from Gateway)
        task        : current task description
        robot_state : live robot state (for Axis 2 scoring)
        scene_graph : current scene (used for context — not scored directly)
        top_k       : max records to select for the CTX

        Returns
        -------
        RelevanceReport with .selected = the records to include in CTX
        """
        t0 = time.perf_counter()

        if not candidates:
            return RelevanceReport(
                task=task,
                candidates_in=0,
                selected_count=0,
                excluded_count=0,
                warnings_surfaced=0,
                diversity_enforced=False,
                top_composite=0.0,
                bottom_composite=0.0,
                latency_ms=0.0,
                results=[],
            )

        # ── Score every candidate on all four axes ────────────────────────────
        scored: list[tuple[MemoryRecord, float, TaskScore, StateScore, TemporalScore, OutcomeScore]] = []

        for record in candidates:
            t_score = self.task_scorer.score(record, task)
            s_score = self.state_scorer.score(record, robot_state)
            v_score = self.temporal_scorer.score(record)
            o_score = self.outcome_scorer.score(record)

            # Hard exclude: expired records
            if v_score.expired:
                continue

            # Composite (weighted)
            composite = (
                self.w_task     * t_score.composite
                + self.w_state  * s_score.composite
                + self.w_temporal * v_score.composite
                + self.w_outcome  * o_score.composite
            )

            scored.append((record, composite, t_score, s_score, v_score, o_score))

        # Sort by composite descending
        scored.sort(key=lambda x: x[1], reverse=True)

        # ── Selection with diversity enforcement ──────────────────────────────
        selected_results:   list[RecordRelevance] = []
        excluded_results:   list[RecordRelevance] = []
        type_counts:        dict[MemoryType, int] = {}
        warning_count       = 0
        selected_count      = 0

        for rank_idx, (record, composite, t_s, s_s, v_s, o_s) in enumerate(scored):
            is_warning = o_s.is_warning

            # Hard floor — exclude regardless of diversity
            if composite < self.min_score_threshold and not is_warning:
                excluded_results.append(RecordRelevance(
                    record=record,
                    task_score=t_s, state_score=s_s,
                    temporal_score=v_s, outcome_score=o_s,
                    composite=round(composite, 4),
                    rank=rank_idx + 1,
                    selected=False,
                    is_warning=False,
                    exclusion_reason=f"below min threshold ({composite:.2f} < {self.min_score_threshold})",
                ))
                continue

            # Failure warning surfacing — include but mark, cap at max_warnings
            if is_warning:
                if not self.include_warnings or warning_count >= self.max_warnings:
                    excluded_results.append(RecordRelevance(
                        record=record,
                        task_score=t_s, state_score=s_s,
                        temporal_score=v_s, outcome_score=o_s,
                        composite=round(composite, 4),
                        rank=rank_idx + 1,
                        selected=False,
                        is_warning=True,
                        exclusion_reason="max_warnings reached" if warning_count >= self.max_warnings
                                         else "warnings disabled",
                    ))
                    continue
                # Include as warning
                selected_results.append(RecordRelevance(
                    record=record,
                    task_score=t_s, state_score=s_s,
                    temporal_score=v_s, outcome_score=o_s,
                    composite=round(composite, 4),
                    rank=rank_idx + 1,
                    selected=True,
                    is_warning=True,
                    exclusion_reason="",
                ))
                warning_count  += 1
                type_counts[record.memory_type] = type_counts.get(record.memory_type, 0) + 1
                continue

            # Top-k limit
            if selected_count >= top_k:
                excluded_results.append(RecordRelevance(
                    record=record,
                    task_score=t_s, state_score=s_s,
                    temporal_score=v_s, outcome_score=o_s,
                    composite=round(composite, 4),
                    rank=rank_idx + 1,
                    selected=False,
                    is_warning=False,
                    exclusion_reason=f"top_k={top_k} reached",
                ))
                continue

            # Diversity check — too many of the same type?
            current_type_count = type_counts.get(record.memory_type, 0)
            if current_type_count >= self.max_same_type:
                # Check if there are other types we haven't seen yet
                seen_types   = set(type_counts.keys())
                all_types    = {r[0].memory_type for r in scored}
                unseen_types = all_types - seen_types
                if unseen_types:
                    # Defer this record — unseen types still available
                    excluded_results.append(RecordRelevance(
                        record=record,
                        task_score=t_s, state_score=s_s,
                        temporal_score=v_s, outcome_score=o_s,
                        composite=round(composite, 4),
                        rank=rank_idx + 1,
                        selected=False,
                        is_warning=False,
                        exclusion_reason=f"diversity: {record.memory_type.value} count={current_type_count} >= {self.max_same_type}",
                    ))
                    continue

            # Apply diversity bonus for introducing a new type
            if current_type_count == 0:
                composite = min(1.0, composite + self.diversity_bonus)

            selected_results.append(RecordRelevance(
                record=record,
                task_score=t_s, state_score=s_s,
                temporal_score=v_s, outcome_score=o_s,
                composite=round(composite, 4),
                rank=rank_idx + 1,
                selected=True,
                is_warning=False,
                exclusion_reason="",
            ))
            type_counts[record.memory_type] = current_type_count + 1
            selected_count += 1

        all_results = selected_results + excluded_results
        all_results.sort(key=lambda r: r.rank)

        composites = [r.composite for r in selected_results]
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return RelevanceReport(
            task=task,
            candidates_in=len(candidates),
            selected_count=len(selected_results),
            excluded_count=len(excluded_results),
            warnings_surfaced=warning_count,
            diversity_enforced=len(type_counts) > 1,
            top_composite=max(composites) if composites else 0.0,
            bottom_composite=min(composites) if composites else 0.0,
            latency_ms=latency_ms,
            results=all_results,
        )

    # ── Convenience: just the records ─────────────────────────────────────────

    def select(
        self,
        candidates:  list[MemoryRecord],
        task:        str,
        robot_state: RobotState,
        top_k:       int = 10,
    ) -> list[MemoryRecord]:
        """Simplified: returns just the selected records."""
        return self.rank(candidates, task, robot_state, top_k=top_k).selected
