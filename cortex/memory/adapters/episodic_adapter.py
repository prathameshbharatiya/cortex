"""
Episodic Memory Adapter
=======================
MemoryGateway-compatible adapter that wraps both:
  - EpisodicMemoryStack (Chameleon geometry-grounded)
  - EpisodicGraph       (EMCA causal graph)

The adapter exposes a unified retrieve/store interface and selects the
appropriate backend based on the query type:
  SPATIAL  → EpisodicMemoryStack (geometry nearest-neighbour)
  default  → EpisodicGraph (causal / recent / task-matched retrieval)

A single EpisodicAdapter instance keeps both stores in sync — every
stored record is written to both.
"""

from __future__ import annotations

import time
from typing import Any

from cortex.memory.adapters.base import (
    MemoryAdapter, MemoryQuery, AdapterSchema, StoreAck, QueryType
)
from cortex.models.memory import MemoryRecord, MemoryType, OutcomeTag
from cortex.memory.episodic.memory_stack import EpisodicMemoryStack, Episode
from cortex.memory.episodic.geometry_encoder import GeometryEncoder
from cortex.memory.episodic.holo_head import HoloHead
from cortex.memory.graph.episodic_graph import EpisodicGraph, EventNode


class EpisodicAdapter(MemoryAdapter):
    """
    Unified episodic memory adapter (geometry stack + causal graph).

    Register with the MemoryGateway like any other adapter:
        gw = MemoryGateway()
        gw.register(EpisodicAdapter(), priority=3)
    """

    def __init__(
        self,
        encoder:          GeometryEncoder | None = None,
        max_episodes:     int = 10_000,
        max_graph_nodes:  int = 50_000,
    ) -> None:
        self._stack  = EpisodicMemoryStack(
            encoder=encoder or GeometryEncoder(),
            max_episodes=max_episodes,
        )
        self._graph  = EpisodicGraph(max_nodes=max_graph_nodes)
        self._holo   = HoloHead()
        self._prev_node_id: str | None = None

    # ── Schema ────────────────────────────────────────────────────────────────

    def schema(self) -> AdapterSchema:
        return AdapterSchema(
            name="episodic_adapter",
            backend_type="episodic",
            version="1.0",
            supports_semantic_search=True,
            supports_temporal_filter=True,
            supports_spatial_search=True,
            supports_exact_lookup=False,
            supports_write=True,
            memory_types=[MemoryType.EPISODIC, MemoryType.SPATIAL],
            avg_read_latency_ms=2.0,
            avg_write_latency_ms=1.0,
            max_records=max(10_000, 50_000),
            description=(
                "Geometry-anchored Chameleon stack + EMCA causal graph. "
                "Spatial queries use geometry NN; other queries use causal graph."
            ),
        )

    # ── Store ─────────────────────────────────────────────────────────────────

    def store(self, record: MemoryRecord) -> StoreAck:
        t0 = time.perf_counter()
        try:
            content  = record.content if isinstance(record.content, dict) else {}
            position = content.get("position", [0.0, 0.0, 0.0])
            orient   = content.get("orientation", [1.0, 0.0, 0.0, 0.0])

            # Write to geometry stack
            episode = Episode(
                episode_id=record.record_id,
                task=content.get("task", record.content_text),
                position=position,
                orientation=orient,
                outcome=record.outcome_tag,
                action_taken=content.get("action_taken", ""),
                failure_modes=content.get("failure_modes", []),
                confidence=record.base_confidence,
                timestamp=record.valid_at,
                metadata=content.get("metadata", {}),
            )
            self._stack.store(episode)

            # Write to causal graph
            node = EventNode(
                node_id=record.record_id,
                task=content.get("task", record.content_text),
                action=content.get("action_taken", ""),
                state_before=content.get("state_before", {}),
                state_after=content.get("state_after", {}),
                outcome=record.outcome_tag,
                failure_codes=content.get("failure_modes", []),
                confidence=record.base_confidence,
                position=position,
                timestamp=record.valid_at,
                metadata=content.get("metadata", {}),
            )
            delay_ms = 0.0
            if self._prev_node_id:
                prev = self._graph.get_node(self._prev_node_id)
                if prev:
                    delay_ms = max(0.0, (record.valid_at - prev.timestamp) * 1000)
            self._graph.record_event(
                node,
                caused_by=self._prev_node_id,
                delay_ms=delay_ms,
            )
            self._prev_node_id = record.record_id

            return StoreAck(
                record_id=record.record_id,
                success=True,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        except Exception as exc:
            return StoreAck(
                record_id=record.record_id,
                success=False,
                latency_ms=(time.perf_counter() - t0) * 1000,
                error=str(exc),
            )

    # ── Retrieve ──────────────────────────────────────────────────────────────

    def retrieve(self, query: MemoryQuery) -> list[MemoryRecord]:
        t0 = time.perf_counter()

        if query.query_type == QueryType.SPATIAL and query.near_position:
            return self._retrieve_spatial(query)

        if query.query_type == QueryType.TEMPORAL:
            return self._retrieve_temporal(query)

        return self._retrieve_task(query)

    def _retrieve_spatial(self, query: MemoryQuery) -> list[MemoryRecord]:
        pos    = query.near_position or [0.0, 0.0, 0.0]
        orient = [1.0, 0.0, 0.0, 0.0]   # default orientation for spatial-only queries

        candidates = self._stack.retrieve(
            position=pos,
            orientation=orient,
            top_k=query.top_k,
            min_sim=0.3,
            task_filter=query.task or None,
        )

        if query.query_text:
            candidates = self._holo.rerank(
                query_task=query.query_text,
                candidates=candidates,
                top_k=query.top_k,
            )

        records = [ep.to_memory_record() for ep, _ in candidates]
        return self._apply_filters(records, query)

    def _retrieve_task(self, query: MemoryQuery) -> list[MemoryRecord]:
        text = query.query_text or query.task
        if text:
            nodes = self._graph.similar_plan(text, top_k=query.top_k)
        else:
            nodes = self._graph.recent(query.top_k)
        records = [n.to_memory_record() for n in nodes]
        return self._apply_filters(records, query)

    def _retrieve_temporal(self, query: MemoryQuery) -> list[MemoryRecord]:
        nodes = self._graph.recent(query.top_k * 3)
        records = [n.to_memory_record() for n in nodes]
        return self._apply_filters(records, query)[:query.top_k]

    @staticmethod
    def _apply_filters(records: list[MemoryRecord], query: MemoryQuery) -> list[MemoryRecord]:
        out = records
        if not query.include_failed:
            out = [r for r in out if r.outcome_tag != OutcomeTag.FAILURE]
        if query.min_confidence > 0.0:
            out = [r for r in out if r.base_confidence >= query.min_confidence]
        if query.max_age_seconds is not None:
            cutoff = time.time() - query.max_age_seconds
            out = [r for r in out if r.valid_at >= cutoff]
        if query.memory_types:
            out = [r for r in out if r.memory_type in query.memory_types]
        return out

    # ── Passthrough accessors for tests ──────────────────────────────────────

    @property
    def stack(self) -> EpisodicMemoryStack:
        return self._stack

    @property
    def graph(self) -> EpisodicGraph:
        return self._graph

    @property
    def holo_head(self) -> HoloHead:
        return self._holo

    def connect(self) -> None:
        pass   # in-process; no connection needed

    def disconnect(self) -> None:
        pass

    def health_check(self) -> bool:
        return True
