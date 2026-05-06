"""
Episodic Memory Stack
=====================
In-process store for geometry-anchored episodes.

Each Episode records what happened at a specific pose + time.
Retrieval is by geometry similarity (nearest-neighbour in embedding
space) optionally filtered by task type and outcome.

Design follows Chameleon's principle that spatial context is the
primary retrieval key for manipulation tasks — "where was I and what
did I do there?" is more reliable than pure semantic search.

Thread-safe: uses a threading.Lock around all mutations.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from cortex.memory.episodic.geometry_encoder import GeometryEncoder
from cortex.models.memory import MemoryRecord, MemoryType, OutcomeTag


@dataclass
class Episode:
    """
    A single episodic memory: what happened at this pose.

    Fields
    ------
    episode_id     : unique identifier
    task           : task description that was being performed
    position       : EE position when episode occurred
    orientation    : EE orientation (quaternion) when episode occurred
    embedding      : geometry embedding (set automatically on store)
    outcome        : was the action successful?
    action_taken   : description of the action that was taken
    failure_modes  : any failure modes encountered (codes)
    confidence     : how confident was the certifier at the time
    timestamp      : Unix time when the episode was recorded
    metadata       : arbitrary extra fields
    """

    episode_id:   str   = field(default_factory=lambda: str(uuid.uuid4()))
    task:         str   = ""
    position:     list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    orientation:  list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])
    embedding:    np.ndarray | None = field(default=None, repr=False)

    outcome:      OutcomeTag = OutcomeTag.UNKNOWN
    action_taken: str        = ""
    failure_modes: list[str] = field(default_factory=list)
    confidence:   float      = 1.0
    timestamp:    float      = field(default_factory=time.time)
    metadata:     dict       = field(default_factory=dict)

    def to_memory_record(self) -> MemoryRecord:
        """Convert to canonical MemoryRecord for the Gateway."""
        return MemoryRecord(
            record_id=self.episode_id,
            source="episodic_memory_stack",
            memory_type=MemoryType.EPISODIC,
            content={
                "task":         self.task,
                "position":     self.position,
                "orientation":  self.orientation,
                "action_taken": self.action_taken,
                "failure_modes": self.failure_modes,
                "confidence":   self.confidence,
                "metadata":     self.metadata,
            },
            content_text=f"{self.task} @ {self.position} → {self.outcome.value}",
            valid_at=self.timestamp,
            base_confidence=self.confidence,
            outcome_tag=self.outcome,
        )


class EpisodicMemoryStack:
    """
    Geometry-anchored episodic memory store.

    Store episodes with store(), retrieve similar ones with retrieve().
    Similarity is computed in geometry embedding space (cosine distance).

    Parameters
    ----------
    encoder      : GeometryEncoder instance (shared or private)
    max_episodes : max number of episodes to retain (FIFO eviction)
    """

    def __init__(
        self,
        encoder:      GeometryEncoder | None = None,
        max_episodes: int = 10_000,
    ) -> None:
        self._encoder      = encoder or GeometryEncoder()
        self._max_episodes = max_episodes
        self._episodes:    list[Episode]    = []
        self._embeddings:  list[np.ndarray] = []
        self._lock         = threading.Lock()

    # ── Storage ───────────────────────────────────────────────────────────────

    def store(self, episode: Episode) -> str:
        """
        Store an episode. Computes geometry embedding if not already set.

        Returns the episode_id.
        """
        if episode.embedding is None:
            emb = self._encoder.encode(episode.position, episode.orientation)
            object.__setattr__(episode, "embedding", emb) if hasattr(episode, "__dataclass_fields__") else setattr(episode, "embedding", emb)
            episode = Episode(
                episode_id=episode.episode_id,
                task=episode.task,
                position=episode.position,
                orientation=episode.orientation,
                embedding=emb,
                outcome=episode.outcome,
                action_taken=episode.action_taken,
                failure_modes=episode.failure_modes,
                confidence=episode.confidence,
                timestamp=episode.timestamp,
                metadata=episode.metadata,
            )

        with self._lock:
            # FIFO eviction
            if len(self._episodes) >= self._max_episodes:
                self._episodes.pop(0)
                self._embeddings.pop(0)
            self._episodes.append(episode)
            self._embeddings.append(episode.embedding)

        return episode.episode_id

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        position:    Sequence[float],
        orientation: Sequence[float],
        top_k:       int   = 5,
        min_sim:     float = 0.5,
        task_filter: str | None = None,
        outcome_filter: OutcomeTag | None = None,
    ) -> list[tuple[Episode, float]]:
        """
        Retrieve the most geometrically similar episodes.

        Returns a list of (episode, similarity_score) sorted descending by score.

        Parameters
        ----------
        position, orientation : query pose
        top_k                 : max results
        min_sim               : minimum cosine similarity to include
        task_filter           : if set, only return episodes for this task
        outcome_filter        : if set, only return episodes with this outcome
        """
        query_emb = self._encoder.encode(position, orientation)

        with self._lock:
            episodes   = list(self._episodes)
            embeddings = list(self._embeddings)

        if not episodes:
            return []

        # Vectorised cosine similarity
        emb_matrix = np.stack(embeddings)                 # (N, dim)
        scores     = emb_matrix @ query_emb               # (N,)  — unit vectors

        results: list[tuple[Episode, float]] = []
        for ep, score in zip(episodes, scores):
            if float(score) < min_sim:
                continue
            if task_filter and ep.task != task_filter:
                continue
            if outcome_filter and ep.outcome != outcome_filter:
                continue
            results.append((ep, float(score)))

        results.sort(key=lambda t: t[1], reverse=True)
        return results[:top_k]

    # ── Queries ───────────────────────────────────────────────────────────────

    def retrieve_failures(
        self,
        position:    Sequence[float],
        orientation: Sequence[float],
        top_k:       int   = 5,
        min_sim:     float = 0.4,
    ) -> list[tuple[Episode, float]]:
        """Retrieve nearby failure episodes — useful for pre-flight checks."""
        return self.retrieve(
            position, orientation,
            top_k=top_k, min_sim=min_sim,
            outcome_filter=OutcomeTag.FAILURE,
        )

    def to_memory_records(
        self,
        position:    Sequence[float],
        orientation: Sequence[float],
        top_k:       int   = 5,
        min_sim:     float = 0.5,
    ) -> list[MemoryRecord]:
        """Retrieve as canonical MemoryRecords for Gateway integration."""
        hits = self.retrieve(position, orientation, top_k=top_k, min_sim=min_sim)
        return [ep.to_memory_record() for ep, _ in hits]

    # ── Stats ─────────────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._episodes)

    def outcome_stats(self) -> dict[str, int]:
        with self._lock:
            stats: dict[str, int] = {}
            for ep in self._episodes:
                key = ep.outcome.value
                stats[key] = stats.get(key, 0) + 1
        return stats


# Public alias — canonical name used by external callers
HierarchicalEpisodicStack = EpisodicMemoryStack
