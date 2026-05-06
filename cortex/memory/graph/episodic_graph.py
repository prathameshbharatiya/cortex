"""
EMCA — Episodic Memory with Causal Architecture
================================================
Stores episodic memories as nodes in a directed causal graph.

Each EventNode captures one discrete event (action taken, outcome observed).
CausalEdges connect events that have a causal relationship: "action A at
state S caused transition to state S', producing outcome O."

This lets Cortex answer queries like:
  "What sequence of actions led to the last CRITICAL failure near this pose?"
  "Which prior success episode is most causally similar to the current plan?"

Graph properties
----------------
  - Directed: edges run from cause → effect
  - Temporal: edges carry a delay_ms field (time between events)
  - Weighted: edges carry a strength field (0.0–1.0, inferred from outcome)
  - Thread-safe: internal lock guards all mutations

Retrieval modes
---------------
  by_outcome   — all nodes with a given outcome (e.g., all FAILURE nodes)
  causal_chain — the ancestor chain leading to a given node
  similar_plan — nodes whose action matches a query string (substring)
  recent       — most recently added nodes
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Iterator

from cortex.models.memory import MemoryRecord, MemoryType, OutcomeTag


@dataclass
class EventNode:
    """
    A single event in the episodic causal graph.

    Fields
    ------
    node_id       : unique identifier
    task          : the task being performed when this event occurred
    action        : the action that was taken
    state_before  : description of system state before the action
    state_after   : description of system state after the action
    outcome       : result of the action
    failure_codes : any failure mode codes raised
    confidence    : certification confidence at the time
    position      : EE position when event occurred
    timestamp     : Unix time
    metadata      : arbitrary extra fields
    """

    node_id:      str        = field(default_factory=lambda: str(uuid.uuid4()))
    task:         str        = ""
    action:       str        = ""
    state_before: dict       = field(default_factory=dict)
    state_after:  dict       = field(default_factory=dict)
    outcome:      OutcomeTag = OutcomeTag.UNKNOWN
    failure_codes: list[str] = field(default_factory=list)
    confidence:   float      = 1.0
    position:     list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    timestamp:    float      = field(default_factory=time.time)
    metadata:     dict       = field(default_factory=dict)

    def to_memory_record(self) -> MemoryRecord:
        return MemoryRecord(
            record_id=self.node_id,
            source="episodic_graph",
            memory_type=MemoryType.EPISODIC,
            content={
                "task":          self.task,
                "action":        self.action,
                "state_before":  self.state_before,
                "state_after":   self.state_after,
                "failure_codes": self.failure_codes,
                "confidence":    self.confidence,
                "position":      self.position,
            },
            content_text=f"{self.task}: {self.action} → {self.outcome.value}",
            valid_at=self.timestamp,
            base_confidence=self.confidence,
            outcome_tag=self.outcome,
        )


@dataclass
class CausalEdge:
    """
    A directed causal link: source_id → target_id.

    Fields
    ------
    edge_id    : unique identifier
    source_id  : the causing event node
    target_id  : the resulting event node
    strength   : 0.0 (weak) – 1.0 (strong) causal relationship
    delay_ms   : time between events in milliseconds
    label      : optional human-readable description of the causal relationship
    """

    edge_id:   str   = field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str   = ""
    target_id: str   = ""
    strength:  float = 1.0
    delay_ms:  float = 0.0
    label:     str   = ""


class EpisodicGraph:
    """
    EMCA episodic memory graph.

    Thread-safe directed graph of EventNodes connected by CausalEdges.
    Provides causal chain retrieval, failure pattern detection, and
    export to MemoryRecord lists for the Gateway.
    """

    def __init__(self, max_nodes: int = 50_000) -> None:
        self._max_nodes  = max_nodes
        self._nodes:     dict[str, EventNode]  = {}
        self._edges:     dict[str, CausalEdge] = {}
        self._out_edges: dict[str, list[str]]  = {}  # node_id → [edge_id]
        self._in_edges:  dict[str, list[str]]  = {}  # node_id → [edge_id]
        self._insertion_order: list[str]       = []   # node_ids in order
        self._lock = threading.Lock()

    # ── Graph mutations ───────────────────────────────────────────────────────

    def add_node(self, node: EventNode) -> str:
        """Add an event node. Returns node_id."""
        with self._lock:
            if len(self._nodes) >= self._max_nodes:
                self._evict_oldest()
            self._nodes[node.node_id] = node
            self._out_edges[node.node_id] = []
            self._in_edges[node.node_id]  = []
            self._insertion_order.append(node.node_id)
        return node.node_id

    def add_edge(self, edge: CausalEdge) -> str:
        """Add a causal edge between two existing nodes. Returns edge_id."""
        with self._lock:
            if edge.source_id not in self._nodes:
                raise KeyError(f"Source node {edge.source_id!r} not in graph")
            if edge.target_id not in self._nodes:
                raise KeyError(f"Target node {edge.target_id!r} not in graph")
            self._edges[edge.edge_id] = edge
            self._out_edges[edge.source_id].append(edge.edge_id)
            self._in_edges[edge.target_id].append(edge.edge_id)
        return edge.edge_id

    def record_event(
        self,
        node:        EventNode,
        caused_by:   str | None = None,
        delay_ms:    float      = 0.0,
        edge_strength: float    = 1.0,
    ) -> str:
        """
        Add an event node and optionally wire it as the effect of a prior node.

        Parameters
        ----------
        node       : the new event
        caused_by  : node_id of the preceding causal event (if any)
        delay_ms   : time since the causing event
        edge_strength : confidence in the causal link

        Returns the new node_id.
        """
        nid = self.add_node(node)
        if caused_by is not None:
            edge = CausalEdge(
                source_id=caused_by,
                target_id=nid,
                strength=edge_strength,
                delay_ms=delay_ms,
                label=f"{caused_by[:8]}→{nid[:8]}",
            )
            try:
                self.add_edge(edge)
            except KeyError:
                pass  # source may have been evicted
        return nid

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def get_node(self, node_id: str) -> EventNode | None:
        with self._lock:
            return self._nodes.get(node_id)

    def by_outcome(self, outcome: OutcomeTag) -> list[EventNode]:
        """Return all nodes with the given outcome."""
        with self._lock:
            return [n for n in self._nodes.values() if n.outcome == outcome]

    def recent(self, n: int = 20) -> list[EventNode]:
        """Return the n most recently added nodes."""
        with self._lock:
            ids = self._insertion_order[-n:]
            return [self._nodes[nid] for nid in reversed(ids) if nid in self._nodes]

    def causal_chain(
        self, node_id: str, max_depth: int = 10
    ) -> list[EventNode]:
        """
        Walk backwards through causal edges from node_id.

        Returns the ancestor chain [root, ..., parent, node], with root first.
        Stops at max_depth to prevent runaway traversal.
        """
        with self._lock:
            chain: list[str] = []
            current = node_id
            visited: set[str] = set()

            for _ in range(max_depth):
                if current in visited or current not in self._nodes:
                    break
                visited.add(current)
                chain.append(current)

                in_eids = self._in_edges.get(current, [])
                if not in_eids:
                    break
                # Follow highest-strength incoming edge
                best_edge = max(
                    (self._edges[eid] for eid in in_eids if eid in self._edges),
                    key=lambda e: e.strength,
                    default=None,
                )
                if best_edge is None:
                    break
                current = best_edge.source_id

            chain.reverse()
            return [self._nodes[nid] for nid in chain if nid in self._nodes]

    def similar_plan(
        self, action_query: str, top_k: int = 10
    ) -> list[EventNode]:
        """Return nodes whose action contains the query string (case-insensitive)."""
        q = action_query.lower()
        with self._lock:
            matches = [
                n for n in self._nodes.values()
                if q in n.action.lower() or q in n.task.lower()
            ]
        matches.sort(key=lambda n: n.timestamp, reverse=True)
        return matches[:top_k]

    def failure_patterns(self, min_chain_depth: int = 2) -> list[list[EventNode]]:
        """
        Find all failure nodes and return their causal chains.

        Useful for detecting repeated failure patterns.
        """
        failures = self.by_outcome(OutcomeTag.FAILURE)
        patterns: list[list[EventNode]] = []
        for f in failures:
            chain = self.causal_chain(f.node_id)
            if len(chain) >= min_chain_depth:
                patterns.append(chain)
        return patterns

    # ── Gateway integration ───────────────────────────────────────────────────

    def recent_as_memory_records(self, n: int = 10) -> list[MemoryRecord]:
        return [node.to_memory_record() for node in self.recent(n)]

    def failures_as_memory_records(self) -> list[MemoryRecord]:
        return [node.to_memory_record() for node in self.by_outcome(OutcomeTag.FAILURE)]

    # ── Stats ─────────────────────────────────────────────────────────────────

    @property
    def node_count(self) -> int:
        with self._lock:
            return len(self._nodes)

    @property
    def edge_count(self) -> int:
        with self._lock:
            return len(self._edges)

    def stats(self) -> dict[str, int]:
        with self._lock:
            outcome_counts: dict[str, int] = {}
            for n in self._nodes.values():
                k = n.outcome.value
                outcome_counts[k] = outcome_counts.get(k, 0) + 1
        return {
            "total_nodes": self.node_count,
            "total_edges": self.edge_count,
            **outcome_counts,
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _evict_oldest(self) -> None:
        """Remove the oldest node (FIFO). Must hold _lock."""
        if not self._insertion_order:
            return
        oldest_id = self._insertion_order.pop(0)
        if oldest_id not in self._nodes:
            return
        # Remove edges referencing this node
        for eid in list(self._out_edges.get(oldest_id, [])):
            edge = self._edges.pop(eid, None)
            if edge:
                in_list = self._in_edges.get(edge.target_id, [])
                if eid in in_list:
                    in_list.remove(eid)
        for eid in list(self._in_edges.get(oldest_id, [])):
            edge = self._edges.pop(eid, None)
            if edge:
                out_list = self._out_edges.get(edge.source_id, [])
                if eid in out_list:
                    out_list.remove(eid)
        self._out_edges.pop(oldest_id, None)
        self._in_edges.pop(oldest_id, None)
        self._nodes.pop(oldest_id, None)
