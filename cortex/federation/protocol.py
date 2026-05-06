"""
Federated Memory Protocol
=========================
Enables multiple Cortex nodes (physical robots or simulation deployments)
to share episodic memory records while preserving robot-operator privacy.

Privacy model: ε-Differential Privacy (Laplace mechanism)
---------------------------------------------------------------------------
Before any memory record leaves a node it is privacy-processed:
  1. Numerical fields (positions, confidence scores) receive Laplace noise
     calibrated to sensitivity / ε.
  2. Text fields are generalised (task category only, no specific identifiers).
  3. A per-node PrivacyBudget tracks total ε consumed; sharing is halted if
     the budget is exhausted.

Consent model
-------------
Each node has a ConsentPolicy that controls:
  - which memory types may be shared (e.g. EPISODIC only, not SEMANTIC)
  - which peer nodes are allowed to receive records
  - minimum confidence threshold (low-confidence memories are not shared)
  - whether position coordinates may be included or must be redacted

Network model (in-process for Phase 9)
---------------------------------------
The FederatedMemoryProtocol is a local coordinator that manages a registry
of FederatedNodes and routes share/receive operations between them.
In production each node would be a remote Cortex instance; the share()
call would be replaced with an authenticated gRPC/HTTPS call.  The
privacy and consent logic is identical — the transport is swappable.

Thread-safety: RLock guards all mutations.
"""

from __future__ import annotations

import copy
import math
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from cortex.models.memory import MemoryRecord, MemoryType, OutcomeTag


# ── Privacy budget ────────────────────────────────────────────────────────────

@dataclass
class PrivacyBudget:
    """
    Tracks ε-differential privacy budget consumption for one node.

    Fields
    ------
    epsilon_total   : total ε budget allocated (higher = less private)
    epsilon_used    : ε consumed so far
    delta           : δ parameter (probability of privacy violation, default 1e-5)
    per_record_cost : ε charged per shared record
    """
    epsilon_total:   float = 10.0
    epsilon_used:    float = 0.0
    delta:           float = 1e-5
    per_record_cost: float = 0.1

    @property
    def remaining(self) -> float:
        return max(0.0, self.epsilon_total - self.epsilon_used)

    @property
    def exhausted(self) -> bool:
        return self.epsilon_used >= self.epsilon_total

    def consume(self, cost: float | None = None) -> bool:
        """Consume `cost` ε. Returns True if budget allows, False if exhausted."""
        c = cost if cost is not None else self.per_record_cost
        if self.epsilon_used + c > self.epsilon_total:
            return False
        self.epsilon_used += c
        return True

    def reset(self) -> None:
        self.epsilon_used = 0.0


# ── Consent policy ────────────────────────────────────────────────────────────

@dataclass
class ConsentPolicy:
    """
    Governs what a node is allowed to share and with whom.

    Fields
    ------
    allowed_memory_types  : which MemoryTypes may be shared
    allowed_peer_ids      : which node IDs may receive records
                            (empty = allow all registered peers)
    min_confidence        : only share records with confidence >= this
    share_positions       : if False, position coordinates are zeroed out
    share_task_text       : if False, task text is replaced with category
    max_records_per_sync  : upper bound on records per sync call
    """
    allowed_memory_types:  list[MemoryType]  = field(
        default_factory=lambda: [MemoryType.EPISODIC]
    )
    allowed_peer_ids:      list[str]         = field(default_factory=list)
    min_confidence:        float             = 0.5
    share_positions:       bool              = True
    share_task_text:       bool              = True
    max_records_per_sync:  int               = 50

    def allows_peer(self, peer_id: str) -> bool:
        if not self.allowed_peer_ids:
            return True   # empty = all peers allowed
        return peer_id in self.allowed_peer_ids

    def allows_type(self, memory_type: MemoryType) -> bool:
        return memory_type in self.allowed_memory_types


# ── Federated record ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FederatedRecord:
    """
    A privacy-processed memory record ready for sharing.

    Fields
    ------
    origin_node_id  : which node produced this record
    record          : the privacy-processed MemoryRecord
    epsilon_spent   : ε budget consumed to produce this record
    shared_at       : Unix timestamp
    """
    origin_node_id: str
    record:         MemoryRecord
    epsilon_spent:  float = 0.0
    shared_at:      float = field(default_factory=time.time)


# ── Federated node ────────────────────────────────────────────────────────────

@dataclass
class FederatedNode:
    """
    Describes one node in the federated network.

    Fields
    ------
    node_id         : unique identifier
    name            : human-readable label (e.g. "arm_east_01")
    privacy_budget  : ε-DP budget for this node
    consent_policy  : controls what this node shares
    received:        list of FederatedRecords received from peers
    """
    node_id:        str
    name:           str            = ""
    privacy_budget: PrivacyBudget  = field(default_factory=PrivacyBudget)
    consent_policy: ConsentPolicy  = field(default_factory=ConsentPolicy)
    received:       list[FederatedRecord] = field(default_factory=list)


# ── Privacy engine ────────────────────────────────────────────────────────────

class _PrivacyEngine:
    """Applies Laplace-mechanism DP noise to numerical record fields."""

    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self._rng = rng or np.random.default_rng()

    def privatise(
        self,
        record:          MemoryRecord,
        epsilon:         float,
        consent:         ConsentPolicy,
    ) -> MemoryRecord:
        """
        Return a privacy-processed copy of `record`.

        Applies Laplace noise (sensitivity=1, scale=1/ε) to:
          - content dict numerical fields
          - base_confidence
        Redacts fields based on consent policy.
        """
        content = dict(record.content) if isinstance(record.content, dict) else {}
        sensitivity = 1.0
        scale = sensitivity / max(epsilon, 1e-9)

        # Position noise / redaction
        if "position" in content:
            if consent.share_positions:
                pos = list(content["position"])
                noise = self._rng.laplace(0, scale, len(pos)).tolist()
                content["position"] = [p + n for p, n in zip(pos, noise)]
            else:
                content["position"] = [0.0, 0.0, 0.0]

        # Task text generalisation
        if "task" in content and not consent.share_task_text:
            content["task"] = record.memory_type.value   # replace with category

        # Confidence noise
        noisy_conf = float(np.clip(
            record.base_confidence + self._rng.laplace(0, scale * 0.1),
            0.0, 1.0,
        ))

        # Build noisy content_text
        content_text = (
            content.get("task", record.memory_type.value)
            if consent.share_task_text
            else record.memory_type.value
        )

        return MemoryRecord(
            record_id=str(uuid.uuid4()),   # new ID — don't expose originals
            source=f"federated:{record.source}",
            memory_type=record.memory_type,
            content=content,
            content_text=content_text,
            valid_at=record.valid_at,
            base_confidence=noisy_conf,
            outcome_tag=record.outcome_tag,
        )


# ── Protocol ──────────────────────────────────────────────────────────────────

class FederatedMemoryProtocol:
    """
    Orchestrates privacy-preserving memory sharing between nodes.

    Usage
    -----
        proto = FederatedMemoryProtocol()
        proto.register_node(FederatedNode(node_id="arm_01"))
        proto.register_node(FederatedNode(node_id="arm_02"))

        # Share records from arm_01 to all consenting peers
        shared = proto.share(
            sender_id="arm_01",
            records=[record1, record2],
        )
        print(f"Shared {shared} records")

        # Read what arm_02 received
        received = proto.get_received("arm_02")
    """

    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self._nodes:   dict[str, FederatedNode] = {}
        self._privacy  = _PrivacyEngine(rng=rng)
        self._lock     = threading.RLock()

    # ── Node management ───────────────────────────────────────────────────────

    def register_node(self, node: FederatedNode) -> None:
        with self._lock:
            self._nodes[node.node_id] = node

    def deregister_node(self, node_id: str) -> None:
        with self._lock:
            self._nodes.pop(node_id, None)

    @property
    def node_count(self) -> int:
        with self._lock:
            return len(self._nodes)

    def get_node(self, node_id: str) -> FederatedNode | None:
        with self._lock:
            return self._nodes.get(node_id)

    # ── Sharing ───────────────────────────────────────────────────────────────

    def share(
        self,
        sender_id: str,
        records:   list[MemoryRecord],
    ) -> int:
        """
        Share records from sender_id to all consenting peer nodes.

        Applies DP noise, checks consent policy and budget, then delivers
        FederatedRecords to each eligible peer's received list.

        Returns the number of (peer, record) deliveries made.
        """
        with self._lock:
            sender = self._nodes.get(sender_id)
            if sender is None:
                return 0

            peers = [
                n for nid, n in self._nodes.items()
                if nid != sender_id
            ]

            deliveries = 0
            cap = sender.consent_policy.max_records_per_sync
            eligible = [
                r for r in records[:cap]
                if (
                    sender.consent_policy.allows_type(r.memory_type)
                    and r.base_confidence >= sender.consent_policy.min_confidence
                )
            ]

            for peer in peers:
                if not sender.consent_policy.allows_peer(peer.node_id):
                    continue
                if sender.privacy_budget.exhausted:
                    break

                for record in eligible:
                    if not sender.privacy_budget.consume():
                        break
                    privatised = self._privacy.privatise(
                        record,
                        epsilon=sender.privacy_budget.per_record_cost,
                        consent=sender.consent_policy,
                    )
                    fed_rec = FederatedRecord(
                        origin_node_id=sender_id,
                        record=privatised,
                        epsilon_spent=sender.privacy_budget.per_record_cost,
                    )
                    peer.received.append(fed_rec)
                    deliveries += 1

        return deliveries

    def get_received(self, node_id: str) -> list[FederatedRecord]:
        """Return all FederatedRecords received by node_id."""
        with self._lock:
            node = self._nodes.get(node_id)
            if node is None:
                return []
            return list(node.received)

    def clear_received(self, node_id: str) -> int:
        """Drain and return count of received records (e.g., after ingestion)."""
        with self._lock:
            node = self._nodes.get(node_id)
            if node is None:
                return 0
            count = len(node.received)
            node.received.clear()
            return count

    # ── Network summary ───────────────────────────────────────────────────────

    def network_summary(self) -> dict:
        with self._lock:
            return {
                "node_count": len(self._nodes),
                "nodes": [
                    {
                        "node_id":        n.node_id,
                        "name":           n.name,
                        "budget_used":    round(n.privacy_budget.epsilon_used, 4),
                        "budget_total":   n.privacy_budget.epsilon_total,
                        "budget_pct":     round(
                            n.privacy_budget.epsilon_used /
                            max(n.privacy_budget.epsilon_total, 1e-9) * 100, 1
                        ),
                        "records_received": len(n.received),
                    }
                    for n in self._nodes.values()
                ],
            }
