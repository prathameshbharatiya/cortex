"""
tests/unit/test_episodic_memory.py
===================================
Phase 4 tests: Chameleon geometry-grounded episodic memory,
EMCA causal graph, and the unified EpisodicAdapter.
"""

from __future__ import annotations

import time
import uuid

import numpy as np
import pytest

from cortex.memory.episodic.geometry_encoder import GeometryEncoder
from cortex.memory.episodic.memory_stack import EpisodicMemoryStack, Episode
from cortex.memory.episodic.holo_head import HoloHead
from cortex.memory.graph.episodic_graph import EpisodicGraph, EventNode, CausalEdge
from cortex.memory.adapters.episodic_adapter import EpisodicAdapter
from cortex.memory.adapters.base import MemoryQuery, QueryType
from cortex.models.memory import MemoryType, OutcomeTag


# ── GeometryEncoder ──────────────────────────────────────────────────────────

class TestGeometryEncoder:

    def test_encode_returns_correct_shape(self) -> None:
        enc = GeometryEncoder(embedding_dim=64)
        emb = enc.encode([0.3, 0.0, 0.5], [1.0, 0.0, 0.0, 0.0])
        assert emb.shape == (64,)

    def test_encode_is_unit_vector(self) -> None:
        enc = GeometryEncoder(embedding_dim=64)
        emb = enc.encode([0.3, 0.0, 0.5], [1.0, 0.0, 0.0, 0.0])
        norm = float(np.linalg.norm(emb))
        assert abs(norm - 1.0) < 1e-5, f"Expected unit vector, got norm={norm}"

    def test_deterministic(self) -> None:
        enc = GeometryEncoder(embedding_dim=64)
        pos, ori = [0.3, 0.0, 0.5], [1.0, 0.0, 0.0, 0.0]
        assert np.allclose(enc.encode(pos, ori), enc.encode(pos, ori))

    def test_nearby_poses_high_similarity(self) -> None:
        enc = GeometryEncoder(embedding_dim=64)
        e1  = enc.encode([0.30, 0.0, 0.5], [1.0, 0.0, 0.0, 0.0])
        e2  = enc.encode([0.31, 0.0, 0.5], [1.0, 0.0, 0.0, 0.0])
        sim = enc.similarity(e1, e2)
        assert sim > 0.9, f"Nearby poses should be similar, got sim={sim:.4f}"

    def test_far_poses_lower_similarity(self) -> None:
        enc = GeometryEncoder(embedding_dim=64)
        e1  = enc.encode([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
        e2  = enc.encode([0.8, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])
        sim = enc.similarity(e1, e2)
        assert sim < 0.99, f"Distant poses should not be nearly identical"

    def test_batch_encode_shape(self) -> None:
        enc  = GeometryEncoder(embedding_dim=32)
        batch = enc.encode_batch(
            [[0.1, 0.0, 0.5], [0.2, 0.0, 0.5]],
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        )
        assert batch.shape == (2, 32)


# ── EpisodicMemoryStack ──────────────────────────────────────────────────────

class TestEpisodicMemoryStack:

    def _episode(self, pos=(0.3, 0.0, 0.5), outcome=OutcomeTag.SUCCESS) -> Episode:
        return Episode(
            task="pick object",
            position=list(pos),
            orientation=[1.0, 0.0, 0.0, 0.0],
            outcome=outcome,
            action_taken="MOVE_EE",
            confidence=0.9,
        )

    def test_store_increases_size(self) -> None:
        stack = EpisodicMemoryStack()
        assert stack.size == 0
        stack.store(self._episode())
        assert stack.size == 1

    def test_retrieve_returns_similar_episodes(self) -> None:
        stack = EpisodicMemoryStack()
        stack.store(self._episode(pos=(0.3, 0.0, 0.5)))
        stack.store(self._episode(pos=(0.7, 0.4, 0.2)))

        results = stack.retrieve([0.3, 0.0, 0.5], [1.0, 0.0, 0.0, 0.0], top_k=5)
        assert len(results) >= 1
        top_ep, top_score = results[0]
        assert top_score > 0.5

    def test_retrieve_failure_filter(self) -> None:
        stack = EpisodicMemoryStack()
        stack.store(self._episode(outcome=OutcomeTag.SUCCESS))
        stack.store(self._episode(outcome=OutcomeTag.FAILURE))

        failures = stack.retrieve_failures([0.3, 0.0, 0.5], [1.0, 0.0, 0.0, 0.0])
        for ep, _ in failures:
            assert ep.outcome == OutcomeTag.FAILURE

    def test_to_memory_records(self) -> None:
        stack = EpisodicMemoryStack()
        stack.store(self._episode())
        records = stack.to_memory_records([0.3, 0.0, 0.5], [1.0, 0.0, 0.0, 0.0])
        assert len(records) >= 1
        assert records[0].memory_type == MemoryType.EPISODIC

    def test_fifo_eviction_respects_max(self) -> None:
        stack = EpisodicMemoryStack(max_episodes=3)
        for i in range(5):
            stack.store(self._episode())
        assert stack.size == 3

    def test_outcome_stats(self) -> None:
        stack = EpisodicMemoryStack()
        stack.store(self._episode(outcome=OutcomeTag.SUCCESS))
        stack.store(self._episode(outcome=OutcomeTag.FAILURE))
        stats = stack.outcome_stats()
        assert stats.get("success", 0) >= 1
        assert stats.get("failure", 0) >= 1


# ── HoloHead ─────────────────────────────────────────────────────────────────

class TestHoloHead:

    def _make_episodes(self) -> list[tuple[Episode, float]]:
        eps = [
            Episode(task="pick red cube", outcome=OutcomeTag.SUCCESS),
            Episode(task="place on shelf", outcome=OutcomeTag.FAILURE),
            Episode(task="pick blue ball", outcome=OutcomeTag.SUCCESS),
        ]
        return [(e, 0.8) for e in eps]

    def test_rerank_returns_top_k(self) -> None:
        holo = HoloHead()
        results = holo.rerank("pick object", self._make_episodes(), top_k=2)
        assert len(results) == 2

    def test_rerank_sorted_descending(self) -> None:
        holo    = HoloHead()
        results = holo.rerank("pick cube", self._make_episodes(), top_k=3)
        scores  = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_attend_returns_embedding_shape(self) -> None:
        holo = HoloHead(embedding_dim=64)
        agg  = holo.attend("pick cube", self._make_episodes())
        assert agg.shape == (64,)

    def test_attend_empty_returns_zeros(self) -> None:
        holo = HoloHead(embedding_dim=64)
        agg  = holo.attend("pick cube", [])
        assert np.allclose(agg, 0.0)


# ── EpisodicGraph ─────────────────────────────────────────────────────────────

class TestEpisodicGraph:

    def _node(self, action="MOVE_EE", outcome=OutcomeTag.SUCCESS) -> EventNode:
        return EventNode(
            task="pick object",
            action=action,
            outcome=outcome,
            confidence=0.9,
        )

    def test_add_node_increments_count(self) -> None:
        graph = EpisodicGraph()
        graph.add_node(self._node())
        assert graph.node_count == 1

    def test_add_edge_increments_count(self) -> None:
        graph = EpisodicGraph()
        n1 = self._node("MOVE_EE")
        n2 = self._node("GRASP")
        graph.add_node(n1)
        graph.add_node(n2)
        edge = CausalEdge(source_id=n1.node_id, target_id=n2.node_id)
        graph.add_edge(edge)
        assert graph.edge_count == 1

    def test_record_event_creates_causal_chain(self) -> None:
        graph = EpisodicGraph()
        n1_id = graph.record_event(self._node("APPROACH"))
        n2_id = graph.record_event(self._node("GRASP"), caused_by=n1_id)
        n3_id = graph.record_event(self._node("LIFT"), caused_by=n2_id)

        chain = graph.causal_chain(n3_id)
        assert len(chain) == 3
        assert chain[0].node_id == n1_id
        assert chain[-1].node_id == n3_id

    def test_by_outcome_filters_correctly(self) -> None:
        graph = EpisodicGraph()
        graph.record_event(self._node(outcome=OutcomeTag.SUCCESS))
        graph.record_event(self._node(outcome=OutcomeTag.FAILURE))
        failures = graph.by_outcome(OutcomeTag.FAILURE)
        assert all(n.outcome == OutcomeTag.FAILURE for n in failures)

    def test_recent_returns_most_recent(self) -> None:
        graph = EpisodicGraph()
        for i in range(5):
            graph.record_event(EventNode(task=f"task_{i}", action=f"action_{i}"))
        recent = graph.recent(3)
        assert len(recent) == 3

    def test_similar_plan_finds_matching_nodes(self) -> None:
        graph = EpisodicGraph()
        graph.record_event(EventNode(task="pick red cube", action="MOVE_EE"))
        graph.record_event(EventNode(task="place on shelf", action="PLACE"))
        matches = graph.similar_plan("pick")
        assert any("pick" in n.task for n in matches)

    def test_fifo_eviction(self) -> None:
        graph = EpisodicGraph(max_nodes=3)
        for i in range(5):
            graph.record_event(EventNode(task=f"task_{i}"))
        assert graph.node_count == 3

    def test_stats_returns_dict(self) -> None:
        graph = EpisodicGraph()
        graph.record_event(self._node(outcome=OutcomeTag.SUCCESS))
        stats = graph.stats()
        assert "total_nodes" in stats
        assert stats["total_nodes"] >= 1

    def test_failure_patterns_length(self) -> None:
        graph = EpisodicGraph()
        n1_id = graph.record_event(self._node("APPROACH"))
        n2_id = graph.record_event(self._node("GRASP", OutcomeTag.FAILURE), caused_by=n1_id)
        patterns = graph.failure_patterns(min_chain_depth=2)
        assert len(patterns) >= 1

    def test_add_edge_invalid_source_raises(self) -> None:
        graph = EpisodicGraph()
        n = self._node()
        graph.add_node(n)
        edge = CausalEdge(source_id="nonexistent", target_id=n.node_id)
        with pytest.raises(KeyError):
            graph.add_edge(edge)


# ── EpisodicAdapter ───────────────────────────────────────────────────────────

class TestEpisodicAdapter:

    def _make_record(
        self,
        position=(0.3, 0.0, 0.5),
        outcome=OutcomeTag.SUCCESS,
        task="pick object",
    ):
        from cortex.models.memory import MemoryRecord
        return MemoryRecord(
            source="test",
            memory_type=MemoryType.EPISODIC,
            content={
                "task": task,
                "position": list(position),
                "orientation": [1.0, 0.0, 0.0, 0.0],
                "action_taken": "MOVE_EE",
                "failure_modes": [],
            },
            content_text=f"{task} @ {list(position)} → {outcome.value}",
            base_confidence=0.9,
            outcome_tag=outcome,
        )

    def test_store_returns_success(self) -> None:
        adapter = EpisodicAdapter()
        record  = self._make_record()
        ack     = adapter.store(record)
        assert ack.success

    def test_store_populates_both_backends(self) -> None:
        adapter = EpisodicAdapter()
        adapter.store(self._make_record())
        assert adapter.stack.size == 1
        assert adapter.graph.node_count == 1

    def test_retrieve_spatial_returns_records(self) -> None:
        adapter = EpisodicAdapter()
        adapter.store(self._make_record())
        query = MemoryQuery(
            query_type=QueryType.SPATIAL,
            near_position=[0.3, 0.0, 0.5],
            top_k=5,
            include_failed=True,
        )
        records = adapter.retrieve(query)
        assert len(records) >= 1

    def test_retrieve_task_returns_records(self) -> None:
        adapter = EpisodicAdapter()
        adapter.store(self._make_record(task="pick blue cube"))
        query = MemoryQuery(
            query_text="pick blue",
            top_k=5,
            include_failed=True,
        )
        records = adapter.retrieve(query)
        assert len(records) >= 1

    def test_retrieve_excludes_failures_by_default(self) -> None:
        adapter = EpisodicAdapter()
        adapter.store(self._make_record(outcome=OutcomeTag.FAILURE))
        query = MemoryQuery(query_type=QueryType.TEMPORAL, top_k=10, include_failed=False)
        records = adapter.retrieve(query)
        for r in records:
            assert r.outcome_tag != OutcomeTag.FAILURE

    def test_schema_describes_capabilities(self) -> None:
        adapter = EpisodicAdapter()
        schema  = adapter.schema()
        assert schema.supports_spatial_search
        assert schema.supports_write
        assert MemoryType.EPISODIC in schema.memory_types

    def test_health_check_true(self) -> None:
        assert EpisodicAdapter().health_check()

    def test_causal_chain_across_multiple_stores(self) -> None:
        adapter = EpisodicAdapter()
        for task in ["approach", "grasp", "lift"]:
            adapter.store(self._make_record(task=task))
        # Most recent node should have a causal chain of 3
        recent = adapter.graph.recent(1)
        chain  = adapter.graph.causal_chain(recent[0].node_id)
        assert len(chain) == 3
