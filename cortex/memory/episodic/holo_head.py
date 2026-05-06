"""
HoloHead — Spatial Attention Head for Episode Retrieval
========================================================
A lightweight attention mechanism that re-ranks retrieved episodes by
combining geometry similarity with task relevance.

Inspired by holographic associative memory: each query is treated as an
interference pattern that is matched against stored episode embeddings.

Architecture
------------
  1. Receive a set of candidate (episode, geometry_score) pairs
  2. Compute task-text embedding via a bag-of-words TF-IDF projection
  3. Combine geometry score + task score with a learned (or fixed) weight
  4. Re-rank and return top-k

In Phase 4 the task embedding uses a fixed random hash projection so
there are no ML training requirements.  The weighting (alpha=0.7 for
geometry, beta=0.3 for task) is configurable.
"""

from __future__ import annotations

import hashlib
import math
from typing import Sequence

import numpy as np

from cortex.memory.episodic.memory_stack import Episode


class HoloHead:
    """
    Spatial-linguistic attention head for episodic episode re-ranking.

    Parameters
    ----------
    embedding_dim : int
        Dimensionality of the internal task embeddings (default 64).
    alpha         : float
        Weight for geometry similarity score (default 0.7).
    beta          : float
        Weight for task text similarity score (default 0.3).
    seed          : int
        RNG seed for task projection matrix.
    """

    def __init__(
        self,
        embedding_dim: int   = 64,
        alpha:         float = 0.7,
        beta:          float = 0.3,
        seed:          int   = 99,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.alpha         = alpha
        self.beta          = beta

        # Fixed projection from token hash space → embedding_dim
        rng = np.random.default_rng(seed)
        self._task_proj: np.ndarray = rng.standard_normal(
            (512, embedding_dim)
        ).astype(np.float32)

    # ── Task embedding ────────────────────────────────────────────────────────

    def _embed_task(self, task: str) -> np.ndarray:
        """
        Embed a task string into the task embedding space.

        Uses character-level bigram hashing into a 512-dimensional bag,
        then projects to embedding_dim.
        """
        tokens = self._tokenise(task)
        bow = np.zeros(512, dtype=np.float32)
        for tok in tokens:
            h = int(hashlib.sha256(tok.encode()).hexdigest(), 16) % 512
            bow[h] += 1.0

        # TF normalise
        norm = np.linalg.norm(bow)
        if norm > 1e-8:
            bow /= norm

        emb = bow @ self._task_proj
        enorm = np.linalg.norm(emb)
        if enorm > 1e-8:
            emb /= enorm
        return emb

    @staticmethod
    def _tokenise(text: str) -> list[str]:
        """Simple word + bigram tokeniser."""
        words  = text.lower().split()
        bigrams = [f"{a}_{b}" for a, b in zip(words, words[1:])]
        return words + bigrams

    # ── Re-ranking ────────────────────────────────────────────────────────────

    def rerank(
        self,
        query_task:  str,
        candidates:  list[tuple[Episode, float]],
        top_k:       int = 5,
    ) -> list[tuple[Episode, float]]:
        """
        Re-rank candidate episodes by combining geometry + task similarity.

        Parameters
        ----------
        query_task  : the current task description
        candidates  : list of (episode, geometry_score) from EpisodicMemoryStack
        top_k       : max results to return

        Returns
        -------
        list of (episode, combined_score), sorted descending
        """
        if not candidates:
            return []

        query_emb = self._embed_task(query_task)
        results: list[tuple[Episode, float]] = []

        for episode, geo_score in candidates:
            ep_emb  = self._embed_task(episode.task)
            task_sim = float(np.dot(query_emb, ep_emb))
            combined = self.alpha * geo_score + self.beta * task_sim
            results.append((episode, combined))

        results.sort(key=lambda t: t[1], reverse=True)
        return results[:top_k]

    def attend(
        self,
        query_task:  str,
        candidates:  list[tuple[Episode, float]],
    ) -> np.ndarray:
        """
        Return a weighted aggregate embedding of the top candidates.

        Useful for injecting a soft episodic memory vector into
        the Context confidence calculation.
        """
        ranked = self.rerank(query_task, candidates, top_k=len(candidates))
        if not ranked:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        total_weight = sum(score for _, score in ranked)
        if total_weight < 1e-8:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        agg = np.zeros(self.embedding_dim, dtype=np.float32)
        for episode, score in ranked:
            emb = self._embed_task(episode.task)
            agg += (score / total_weight) * emb
        return agg
