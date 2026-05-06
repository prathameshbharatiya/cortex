"""
In-Memory Adapter
=================
Stores records in a Python dict. No external dependencies.
Fastest possible reads. Data lives for the process lifetime only.

Use cases:
  - Development and testing (default adapter)
  - Single-robot deployments where persistence is not required
  - L1 cache layer within a multi-adapter Gateway setup
  - Unit tests — deterministic, no I/O, instant

This is the adapter Cortex uses when no other adapters are registered.
It is not a toy. A well-tuned in-memory store handles thousands of
records at sub-millisecond latency, which is exactly what Phase 1 needs.
"""

from __future__ import annotations

import time
import threading
from typing import Any

import numpy as np

from cortex.models.memory import MemoryRecord, MemoryType
from cortex.memory.adapters.base import (
    MemoryAdapter, MemoryQuery, AdapterSchema, StoreAck, QueryType
)


class InMemoryAdapter(MemoryAdapter):
    """
    Thread-safe in-memory memory store.
    Supports semantic similarity (cosine on content_text), temporal
    filtering, spatial proximity, and exact lookup.
    """

    def __init__(self, name: str = "in_memory", max_records: int = 50_000) -> None:
        self.name        = name
        self.max_records = max_records
        self._store: dict[str, MemoryRecord] = {}
        self._lock  = threading.RLock()

    # ── Core interface ────────────────────────────────────────────────────────

    def retrieve(self, query: MemoryQuery) -> list[MemoryRecord]:
        with self._lock:
            records = list(self._store.values())

        # ── Filter pass ───────────────────────────────────────────────────────

        if query.record_ids:
            records = [r for r in records if r.record_id in query.record_ids]
            return records[:query.top_k]

        if query.memory_types:
            records = [r for r in records if r.memory_type in query.memory_types]

        if query.source_filter:
            records = [r for r in records if r.source in query.source_filter]

        if not query.include_failed:
            from cortex.models.memory import OutcomeTag
            records = [
                r for r in records
                if r.outcome_tag != OutcomeTag.FAILURE or r.outcome_count < 3
            ]

        # Temporal filter
        if query.max_age_seconds is not None:
            cutoff = time.time() - query.max_age_seconds
            records = [r for r in records if r.valid_at >= cutoff]

        # Only currently valid records
        records = [r for r in records if r.is_currently_valid]

        # Confidence floor
        if query.min_confidence > 0:
            records = [r for r in records if r.effective_confidence >= query.min_confidence]

        # Spatial filter
        if query.near_position and query.query_type == QueryType.SPATIAL:
            records = self._filter_spatial(records, query.near_position, query.radius_m)

        # ── Ranking pass ──────────────────────────────────────────────────────

        if query.query_type == QueryType.SEMANTIC and query.query_text:
            records = self._rank_semantic(records, query.query_text)
        elif query.query_type == QueryType.TEMPORAL:
            records = sorted(records, key=lambda r: r.valid_at, reverse=True)
        elif query.query_type == QueryType.TASK and query.task:
            records = self._rank_task(records, query.task)
        else:
            # Default: sort by effective confidence × recency
            records = sorted(
                records,
                key=lambda r: r.effective_confidence * (1.0 / max(1.0, r.age_seconds / 3600)),
                reverse=True,
            )

        return records[:query.top_k]

    def store(self, record: MemoryRecord) -> StoreAck:
        t0 = time.perf_counter()
        with self._lock:
            if len(self._store) >= self.max_records:
                self._evict_oldest(count=max(1, self.max_records // 10))
            self._store[record.record_id] = record
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return StoreAck(record_id=record.record_id, success=True, latency_ms=latency_ms)

    def schema(self) -> AdapterSchema:
        return AdapterSchema(
            name=self.name,
            backend_type="in_memory",
            supports_semantic_search=True,
            supports_temporal_filter=True,
            supports_spatial_search=True,
            supports_exact_lookup=True,
            supports_write=True,
            memory_types=list(MemoryType),
            avg_read_latency_ms=0.1,
            avg_write_latency_ms=0.1,
            max_records=self.max_records,
            description="Thread-safe in-memory store. No persistence. Fastest reads.",
        )

    def delete(self, record_id: str) -> bool:
        with self._lock:
            if record_id in self._store:
                del self._store[record_id]
                return True
            return False

    def count(self) -> int:
        with self._lock:
            return len(self._store)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def all_records(self) -> list[MemoryRecord]:
        with self._lock:
            return list(self._store.values())

    # ── Semantic ranking (simple bag-of-words cosine) ─────────────────────────

    @staticmethod
    def _rank_semantic(
        records: list[MemoryRecord], query_text: str
    ) -> list[MemoryRecord]:
        """
        Rank records by cosine similarity between query and content_text.
        Production replaces this with a real embedding model.
        This gives deterministic, fast, testable results without dependencies.
        """
        if not records:
            return records

        def tokenise(text: str) -> dict[str, int]:
            tokens = text.lower().split()
            freq: dict[str, int] = {}
            for t in tokens:
                freq[t] = freq.get(t, 0) + 1
            return freq

        query_tokens = tokenise(query_text)
        if not query_tokens:
            return records

        def cosine(a: dict[str, int], b: dict[str, int]) -> float:
            keys = set(a) | set(b)
            dot  = sum(a.get(k, 0) * b.get(k, 0) for k in keys)
            norm_a = sum(v * v for v in a.values()) ** 0.5
            norm_b = sum(v * v for v in b.values()) ** 0.5
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot / (norm_a * norm_b)

        scored = [
            (r, cosine(query_tokens, tokenise(r.content_text)) * r.effective_confidence)
            for r in records
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [r for r, _ in scored]

    @staticmethod
    def _rank_task(records: list[MemoryRecord], task: str) -> list[MemoryRecord]:
        """Rank by task relevance using keyword overlap."""
        task_words = set(task.lower().split())

        def task_score(r: MemoryRecord) -> float:
            content_words = set(r.content_text.lower().split())
            overlap = len(task_words & content_words)
            return overlap * r.effective_confidence

        return sorted(records, key=task_score, reverse=True)

    @staticmethod
    def _filter_spatial(
        records: list[MemoryRecord],
        position: list[float],
        radius_m: float,
    ) -> list[MemoryRecord]:
        """Keep only spatial records whose content position is within radius."""
        pos = np.array(position)
        result = []
        for r in records:
            if r.memory_type != MemoryType.SPATIAL:
                result.append(r)  # non-spatial records pass through
                continue
            content = r.content
            if not isinstance(content, dict):
                result.append(r)
                continue
            obj_pos = content.get("position")
            if obj_pos is None:
                result.append(r)
                continue
            dist = float(np.linalg.norm(np.array(obj_pos) - pos))
            if dist <= radius_m:
                result.append(r)
        return result

    def _evict_oldest(self, count: int) -> None:
        """Remove the oldest records when store is full."""
        sorted_ids = sorted(
            self._store.keys(),
            key=lambda rid: self._store[rid].valid_at,
        )
        for rid in sorted_ids[:count]:
            del self._store[rid]
