"""
tests/unit/test_federated_memory.py
=====================================
Phase 9 tests: federated memory network with differential privacy
and consent management.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from cortex.federation.protocol import (
    FederatedMemoryProtocol,
    FederatedNode,
    FederatedRecord,
    PrivacyBudget,
    ConsentPolicy,
)
from cortex.models.memory import MemoryRecord, MemoryType, OutcomeTag


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _node(
    node_id: str,
    epsilon_total: float = 10.0,
    per_record_cost: float = 0.1,
    allowed_peers: list[str] | None = None,
    share_positions: bool = True,
    share_task_text: bool = True,
    min_confidence: float = 0.0,
) -> FederatedNode:
    return FederatedNode(
        node_id=node_id,
        name=node_id,
        privacy_budget=PrivacyBudget(
            epsilon_total=epsilon_total,
            per_record_cost=per_record_cost,
        ),
        consent_policy=ConsentPolicy(
            allowed_peer_ids=allowed_peers or [],
            share_positions=share_positions,
            share_task_text=share_task_text,
            min_confidence=min_confidence,
        ),
    )


def _record(
    task: str = "pick object",
    position: list[float] | None = None,
    confidence: float = 0.9,
    memory_type: MemoryType = MemoryType.EPISODIC,
) -> MemoryRecord:
    return MemoryRecord(
        source="test",
        memory_type=memory_type,
        content={
            "task": task,
            "position": position or [0.3, 0.0, 0.5],
        },
        content_text=task,
        base_confidence=confidence,
        outcome_tag=OutcomeTag.SUCCESS,
    )


# ── PrivacyBudget ─────────────────────────────────────────────────────────────

class TestPrivacyBudget:

    def test_remaining_starts_at_total(self):
        b = PrivacyBudget(epsilon_total=5.0)
        assert b.remaining == 5.0

    def test_consume_deducts_budget(self):
        b = PrivacyBudget(epsilon_total=1.0, per_record_cost=0.1)
        b.consume()
        assert abs(b.epsilon_used - 0.1) < 1e-9

    def test_consume_returns_false_when_exhausted(self):
        b = PrivacyBudget(epsilon_total=0.1, per_record_cost=0.1)
        b.consume()
        assert not b.consume()

    def test_exhausted_flag(self):
        b = PrivacyBudget(epsilon_total=0.1, per_record_cost=0.1)
        b.consume()
        assert b.exhausted

    def test_reset_clears_used(self):
        b = PrivacyBudget(epsilon_total=1.0, per_record_cost=0.1)
        b.consume()
        b.reset()
        assert b.epsilon_used == 0.0

    def test_custom_cost_consume(self):
        b = PrivacyBudget(epsilon_total=1.0)
        b.consume(cost=0.5)
        assert abs(b.epsilon_used - 0.5) < 1e-9


# ── ConsentPolicy ─────────────────────────────────────────────────────────────

class TestConsentPolicy:

    def test_empty_peer_list_allows_all(self):
        p = ConsentPolicy(allowed_peer_ids=[])
        assert p.allows_peer("anyone")

    def test_specific_peer_list(self):
        p = ConsentPolicy(allowed_peer_ids=["arm_02"])
        assert p.allows_peer("arm_02")
        assert not p.allows_peer("arm_03")

    def test_allows_type(self):
        p = ConsentPolicy(allowed_memory_types=[MemoryType.EPISODIC])
        assert p.allows_type(MemoryType.EPISODIC)
        assert not p.allows_type(MemoryType.SEMANTIC)


# ── FederatedMemoryProtocol ───────────────────────────────────────────────────

class TestFederatedProtocol:

    def _proto(self, rng_seed: int = 42) -> FederatedMemoryProtocol:
        return FederatedMemoryProtocol(rng=np.random.default_rng(rng_seed))

    def test_register_increments_count(self):
        proto = self._proto()
        proto.register_node(_node("n1"))
        assert proto.node_count == 1

    def test_deregister_decrements_count(self):
        proto = self._proto()
        proto.register_node(_node("n1"))
        proto.deregister_node("n1")
        assert proto.node_count == 0

    def test_share_delivers_to_peer(self):
        proto = self._proto()
        proto.register_node(_node("n1"))
        proto.register_node(_node("n2"))
        sent = proto.share("n1", [_record()])
        assert sent >= 1
        received = proto.get_received("n2")
        assert len(received) >= 1

    def test_share_not_received_by_sender(self):
        proto = self._proto()
        proto.register_node(_node("n1"))
        proto.register_node(_node("n2"))
        proto.share("n1", [_record()])
        received_by_n1 = proto.get_received("n1")
        assert len(received_by_n1) == 0

    def test_shared_record_has_federated_source(self):
        proto = self._proto()
        proto.register_node(_node("n1"))
        proto.register_node(_node("n2"))
        proto.share("n1", [_record(task="pick cube")])
        received = proto.get_received("n2")
        assert len(received) == 1
        assert "federated" in received[0].record.source

    def test_shared_record_gets_new_id(self):
        proto = self._proto()
        proto.register_node(_node("n1"))
        proto.register_node(_node("n2"))
        original = _record()
        proto.share("n1", [original])
        received = proto.get_received("n2")
        assert received[0].record.record_id != original.record_id

    def test_position_noise_applied(self):
        proto = self._proto(rng_seed=0)
        proto.register_node(_node("n1"))
        proto.register_node(_node("n2"))
        original = _record(position=[0.3, 0.0, 0.5])
        proto.share("n1", [original])
        received = proto.get_received("n2")
        recv_pos = received[0].record.content.get("position", [0.0, 0.0, 0.0])
        # Position should be noisy (not exactly equal to original)
        # With ε=0.1 and sensitivity=1, scale=10 → significant noise
        orig_pos = [0.3, 0.0, 0.5]
        assert recv_pos != orig_pos, "Position should have DP noise applied"

    def test_position_redacted_when_consent_false(self):
        proto = self._proto()
        proto.register_node(_node("n1", share_positions=False))
        proto.register_node(_node("n2"))
        proto.share("n1", [_record(position=[0.3, 0.0, 0.5])])
        received = proto.get_received("n2")
        recv_pos = received[0].record.content.get("position")
        assert recv_pos == [0.0, 0.0, 0.0]

    def test_task_text_redacted_when_consent_false(self):
        proto = self._proto()
        proto.register_node(_node("n1", share_task_text=False))
        proto.register_node(_node("n2"))
        proto.share("n1", [_record(task="pick red cube")])
        received = proto.get_received("n2")
        assert received[0].record.content_text == MemoryType.EPISODIC.value

    def test_budget_exhaustion_stops_sharing(self):
        proto = self._proto()
        # Budget for exactly 2 records
        proto.register_node(_node("n1", epsilon_total=0.2, per_record_cost=0.1))
        proto.register_node(_node("n2"))
        records = [_record() for _ in range(10)]
        sent = proto.share("n1", records)
        assert sent == 2

    def test_low_confidence_record_not_shared(self):
        proto = self._proto()
        proto.register_node(_node("n1", min_confidence=0.8))
        proto.register_node(_node("n2"))
        proto.share("n1", [_record(confidence=0.3)])
        assert len(proto.get_received("n2")) == 0

    def test_consent_peer_restriction(self):
        proto = self._proto()
        proto.register_node(_node("n1", allowed_peers=["n3"]))  # only n3 allowed
        proto.register_node(_node("n2"))
        proto.register_node(_node("n3"))
        proto.share("n1", [_record()])
        assert len(proto.get_received("n2")) == 0
        assert len(proto.get_received("n3")) >= 1

    def test_non_episodic_type_blocked_by_consent(self):
        proto = self._proto()
        # Consent only allows EPISODIC; try sharing SEMANTIC
        proto.register_node(_node("n1"))
        proto.register_node(_node("n2"))
        semantic = _record(memory_type=MemoryType.SEMANTIC)
        sent = proto.share("n1", [semantic])
        assert sent == 0

    def test_clear_received(self):
        proto = self._proto()
        proto.register_node(_node("n1"))
        proto.register_node(_node("n2"))
        proto.share("n1", [_record()])
        count = proto.clear_received("n2")
        assert count >= 1
        assert len(proto.get_received("n2")) == 0

    def test_unregistered_sender_returns_zero(self):
        proto = self._proto()
        proto.register_node(_node("n2"))
        sent = proto.share("ghost", [_record()])
        assert sent == 0

    def test_network_summary_structure(self):
        proto = self._proto()
        proto.register_node(_node("n1"))
        proto.register_node(_node("n2"))
        summary = proto.network_summary()
        assert summary["node_count"] == 2
        assert len(summary["nodes"]) == 2
        assert "budget_used" in summary["nodes"][0]

    def test_budget_usage_tracked_in_summary(self):
        proto = self._proto()
        proto.register_node(_node("n1"))
        proto.register_node(_node("n2"))
        proto.share("n1", [_record()])
        summary = proto.network_summary()
        n1_entry = next(n for n in summary["nodes"] if n["node_id"] == "n1")
        assert n1_entry["budget_used"] > 0.0

    def test_multi_node_broadcast(self):
        proto = self._proto()
        for i in range(5):
            proto.register_node(_node(f"n{i}"))
        proto.share("n0", [_record()])
        total_received = sum(
            len(proto.get_received(f"n{i}")) for i in range(1, 5)
        )
        assert total_received == 4   # 1 record × 4 peers
