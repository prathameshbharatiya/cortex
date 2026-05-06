"""
Outcome Relevance Scorer
========================
Axis 4 of the Relevance Engine.

Answers: when this memory was used before, did acting on it succeed?

This is the axis that makes Cortex learn from experience without
retraining. A memory that has consistently led to successful outcomes
in similar situations is more valuable than one with a mixed or
negative track record.

The scorer evaluates:
  1. Success rate       — raw success / total uses
  2. Outcome confidence — how many uses before we trust the rate?
  3. Situation match    — does the current situation match the situation
                          in which this memory was previously used?
  4. Failure penalty    — explicit FAILURE-tagged records get penalised
                          but not excluded. Failure history is information.

Why failure records are kept
----------------------------
A record tagged FAILURE is not deleted. It is retained with a low
score and surfaced as a warning in the CTX. The AI and the Gate
both see it — the Gate uses it to reduce confidence, the AI can
use it to avoid previously failed approaches.

This is the difference between a system that learns and one that forgets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from cortex.models.memory import MemoryRecord, OutcomeTag


# ── Outcome confidence thresholds ─────────────────────────────────────────────
_MIN_USES_FOR_TRUST = 3      # need at least 3 uses before trusting the rate
_FAILURE_PENALTY    = 0.40   # failure-tagged records: score × (1 - penalty)
_HIGH_SUCCESS_BOOST = 0.10   # > 80% success rate with > 5 uses: +10%


@dataclass(frozen=True)
class OutcomeScore:
    record_id:        str
    outcome_tag:      str
    outcome_count:    int
    success_rate:     float   # success_count / outcome_count
    trust_weight:     float   # how much to trust the rate (based on sample size)
    failure_penalty:  float   # 0 if not failed, _FAILURE_PENALTY if tagged FAILURE
    composite:        float   # final outcome score [0, 1]
    is_warning:       bool    # True if this record is a known failure source


class OutcomeRelevanceScorer:
    """
    Scores memory records by their outcome history.

    High-success records score closer to 1.0.
    Low-success or failure-tagged records score lower but are retained.
    Unknown (zero uses) records score at the neutral prior.
    """

    def __init__(
        self,
        min_uses_for_trust:  int   = _MIN_USES_FOR_TRUST,
        failure_penalty:     float = _FAILURE_PENALTY,
        high_success_boost:  float = _HIGH_SUCCESS_BOOST,
        neutral_prior:       float = 0.60,   # score for records with no history
        failure_warning_threshold: float = 0.35,  # success rate below this = warning
    ) -> None:
        self.min_uses            = min_uses_for_trust
        self.failure_penalty     = failure_penalty
        self.high_success_boost  = high_success_boost
        self.neutral_prior       = neutral_prior
        self.failure_warning_threshold = failure_warning_threshold

    def score(self, record: MemoryRecord) -> OutcomeScore:
        count      = record.outcome_count
        successes  = record.success_count
        tag        = record.outcome_tag

        # ── No history ────────────────────────────────────────────────────────
        if count == 0:
            # Unknown history — neutral prior
            composite = self.neutral_prior
            if tag == OutcomeTag.SUCCESS:
                composite = min(1.0, self.neutral_prior + 0.10)
            elif tag == OutcomeTag.FAILURE:
                composite = max(0.1, self.neutral_prior - self.failure_penalty)

            return OutcomeScore(
                record_id=record.record_id,
                outcome_tag=tag.value,
                outcome_count=0,
                success_rate=0.0,
                trust_weight=0.0,
                failure_penalty=self.failure_penalty if tag == OutcomeTag.FAILURE else 0.0,
                composite=round(composite, 4),
                is_warning=tag == OutcomeTag.FAILURE,
            )

        # ── Compute success rate with Laplace smoothing ───────────────────────
        # Laplace smoothing prevents 0% or 100% rates with tiny samples
        success_rate = (successes + 1) / (count + 2)

        # ── Trust weight — confidence in the rate ────────────────────────────
        # Bayesian: trust increases with sample size, plateaus at ~20 uses
        trust_weight = 1.0 - math.exp(-count / self.min_uses)

        # ── Base score — blend prior and observed rate ────────────────────────
        base_score = (
            (1.0 - trust_weight) * self.neutral_prior
            + trust_weight       * success_rate
        )

        # ── High success boost ────────────────────────────────────────────────
        if success_rate > 0.80 and count >= 5:
            base_score = min(1.0, base_score + self.high_success_boost)

        # ── Failure penalty ───────────────────────────────────────────────────
        fp = 0.0
        if tag == OutcomeTag.FAILURE:
            fp = self.failure_penalty
            base_score = base_score * (1.0 - fp)
        elif success_rate < self.failure_warning_threshold and count >= self.min_uses:
            # Not explicitly tagged failure, but high failure rate in practice
            fp = self.failure_penalty * 0.5
            base_score = base_score * (1.0 - fp)

        composite  = max(0.05, min(1.0, base_score))
        is_warning = (
            tag == OutcomeTag.FAILURE
            or (success_rate < self.failure_warning_threshold and count >= self.min_uses)
        )

        return OutcomeScore(
            record_id=record.record_id,
            outcome_tag=tag.value,
            outcome_count=count,
            success_rate=round(success_rate, 4),
            trust_weight=round(trust_weight, 4),
            failure_penalty=round(fp, 4),
            composite=round(composite, 4),
            is_warning=is_warning,
        )

    def batch_score(self, records: list[MemoryRecord]) -> list[OutcomeScore]:
        return [self.score(r) for r in records]
