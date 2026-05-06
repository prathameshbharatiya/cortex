"""
Task Relevance Scorer
=====================
Axis 1 of the Relevance Engine.

Answers: does this memory record actually help accomplish the current task?

Similarity alone is not enough. A memory of "picking a red cup" is
semantically similar to "picking a blue cup" — but if the task is
specifically about the red cup, the blue cup memory is noise.

This scorer combines:
  1. Token overlap  — keyword match between task and memory content
  2. Action match   — does the memory involve the same type of action?
  3. Object match   — does the memory reference the same object?
  4. Goal alignment — does the memory's recorded outcome serve the goal?

In production the token overlap is replaced by a proper embedding
model (sentence-transformers, OpenAI ada, etc). The interface is
identical — swap the embedding function without changing anything else.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from cortex.models.memory import MemoryRecord, MemoryType


# ── Action verb vocabulary for action-type matching ──────────────────────────
_PICK_VERBS  = {"pick", "grasp", "grab", "lift", "retrieve", "take"}
_PLACE_VERBS = {"place", "put", "set", "deposit", "release", "drop"}
_PUSH_VERBS  = {"push", "slide", "move", "shift", "nudge"}
_NAV_VERBS   = {"navigate", "move", "go", "travel", "approach", "reach"}
_INSPECT_VERBS = {"inspect", "check", "verify", "observe", "look"}

_VERB_GROUPS = [_PICK_VERBS, _PLACE_VERBS, _PUSH_VERBS, _NAV_VERBS, _INSPECT_VERBS]


@dataclass(frozen=True)
class TaskScore:
    record_id:      str
    token_score:    float   # keyword overlap [0, 1]
    action_score:   float   # action-type alignment [0, 1]
    object_score:   float   # object reference match [0, 1]
    composite:      float   # weighted composite [0, 1]


class TaskRelevanceScorer:
    """
    Scores how relevant a memory record is to the current task.

    Weights:
      token   — 0.50  (primary signal)
      action  — 0.30  (important: wrong action type = wrong memory)
      object  — 0.20  (useful but not always present)
    """

    def __init__(
        self,
        w_token:  float = 0.50,
        w_action: float = 0.30,
        w_object: float = 0.20,
    ) -> None:
        total = w_token + w_action + w_object
        self.w_token  = w_token  / total
        self.w_action = w_action / total
        self.w_object = w_object / total

    def score(self, record: MemoryRecord, task: str) -> TaskScore:
        task_tokens  = self._tokenise(task)
        task_verbs   = self._extract_verbs(task_tokens)
        task_nouns   = self._extract_nouns(task_tokens, task_verbs)

        # ── Token overlap ─────────────────────────────────────────────────────
        content_tokens  = self._tokenise(record.content_text)
        token_score     = self._cosine_overlap(task_tokens, content_tokens)

        # Also check structured content
        if isinstance(record.content, dict):
            for v in record.content.values():
                if isinstance(v, str):
                    content_tokens |= self._tokenise(v)
            token_score = max(token_score, self._cosine_overlap(task_tokens, content_tokens))

        # ── Action type alignment ─────────────────────────────────────────────
        content_verbs = self._extract_verbs(content_tokens)
        action_score  = self._verb_group_match(task_verbs, content_verbs)

        # ── Object match ──────────────────────────────────────────────────────
        content_nouns = self._extract_nouns(content_tokens, content_verbs)
        object_score  = self._noun_overlap(task_nouns, content_nouns)

        composite = (
            self.w_token  * token_score
            + self.w_action * action_score
            + self.w_object * object_score
        )

        return TaskScore(
            record_id=record.record_id,
            token_score=round(token_score, 4),
            action_score=round(action_score, 4),
            object_score=round(object_score, 4),
            composite=round(composite, 4),
        )

    # ── Text processing ───────────────────────────────────────────────────────

    @staticmethod
    def _tokenise(text: str) -> set[str]:
        """Lowercase, strip punctuation, split into tokens."""
        text   = text.lower()
        text   = re.sub(r"[^\w\s]", " ", text)
        tokens = set(text.split())
        # Remove very common stop words
        stops  = {"the", "a", "an", "is", "are", "was", "were", "in", "on",
                   "at", "to", "for", "of", "and", "or", "it", "its",
                   "this", "that", "with", "from", "by", "be", "has", "had"}
        return tokens - stops

    @staticmethod
    def _cosine_overlap(a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        intersection = len(a & b)
        return intersection / (len(a) ** 0.5 * len(b) ** 0.5)

    @staticmethod
    def _extract_verbs(tokens: set[str]) -> set[str]:
        all_verbs: set[str] = set()
        for group in _VERB_GROUPS:
            all_verbs |= group
        return tokens & all_verbs

    @staticmethod
    def _extract_nouns(tokens: set[str], verbs: set[str]) -> set[str]:
        """Approximate nouns as non-verb content tokens."""
        function_words = {"robot", "arm", "gripper", "end", "effector",
                          "position", "pose", "target", "object"}
        return (tokens - verbs) | (tokens & function_words)

    @staticmethod
    def _verb_group_match(task_verbs: set[str], content_verbs: set[str]) -> float:
        """1.0 if both belong to same verb group, 0.5 if one is empty, 0.0 if different."""
        if not task_verbs or not content_verbs:
            return 0.5   # unknown → neutral

        for group in _VERB_GROUPS:
            task_in    = bool(task_verbs    & group)
            content_in = bool(content_verbs & group)
            if task_in and content_in:
                return 1.0
            if task_in != content_in:
                return 0.0

        return 0.5   # different groups, but no strong mismatch

    @staticmethod
    def _noun_overlap(task_nouns: set[str], content_nouns: set[str]) -> float:
        if not task_nouns or not content_nouns:
            return 0.5
        overlap = len(task_nouns & content_nouns)
        if overlap == 0:
            return 0.1
        return min(1.0, overlap / max(1, min(len(task_nouns), len(content_nouns))))
