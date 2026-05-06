"""
Memory Adapter Base
===================
The contract every memory source must implement to connect to Cortex.

Three methods. That's it. Any system that implements these three methods
becomes a first-class memory source for the Cortex Memory Gateway:

    retrieve(query)  → list[MemoryRecord]
    store(record)    → str (record_id)
    schema()         → AdapterSchema

The Gateway calls these. It doesn't care whether the adapter talks to
Redis, Weaviate, Mem0, Zep, a flat JSON file, or a custom database.
That is the entire point of this layer.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from cortex.models.memory import MemoryRecord, MemoryType


# ── Query model ───────────────────────────────────────────────────────────────

class QueryType(str, Enum):
    SEMANTIC   = "semantic"    # similarity search over embeddings
    TEMPORAL   = "temporal"    # time-bounded retrieval
    SPATIAL    = "spatial"     # position-aware retrieval
    EXACT      = "exact"       # lookup by record_id or exact key
    TASK       = "task"        # task-conditioned retrieval


@dataclass(frozen=True)
class MemoryQuery:
    """
    A structured retrieval request sent to any adapter.

    The adapter uses whichever fields are relevant to its backend.
    Fields it cannot use are safely ignored.
    """

    # What are we looking for?
    query_text:    str            = ""
    query_type:    QueryType      = QueryType.SEMANTIC
    memory_types:  list[MemoryType] = field(default_factory=list)

    # Filters
    task:          str            = ""
    min_confidence: float         = 0.0
    max_age_seconds: float | None = None   # None = no age limit
    object_ids:    list[str]      = field(default_factory=list)
    source_filter: list[str]      = field(default_factory=list)

    # Spatial
    near_position: list[float] | None = None   # [x, y, z]
    radius_m:      float              = 1.0

    # Results
    top_k:         int            = 10
    include_failed: bool          = False   # include outcome=FAILURE records?

    # Exact lookup
    record_ids:    list[str]      = field(default_factory=list)


# ── Adapter schema ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AdapterSchema:
    """
    Self-description of an adapter's capabilities.
    The Gateway uses this to route queries optimally.
    """

    name:            str
    backend_type:    str          # "redis" | "weaviate" | "mem0" | "json" | ...
    version:         str          = "1.0"

    # What can this adapter do?
    supports_semantic_search: bool = False
    supports_temporal_filter: bool = True
    supports_spatial_search:  bool = False
    supports_exact_lookup:    bool = True
    supports_write:           bool = True

    # What memory types does it hold?
    memory_types: list[MemoryType] = field(default_factory=list)

    # Performance characteristics (used by Gateway for routing)
    avg_read_latency_ms:  float = 10.0
    avg_write_latency_ms: float = 20.0
    max_records:          int   = 100_000

    description: str = ""


# ── Store acknowledgement ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class StoreAck:
    """Returned by adapter.store() to confirm a write."""
    record_id:  str
    success:    bool
    latency_ms: float = 0.0
    error:      str   = ""


# ── Exceptions ────────────────────────────────────────────────────────────────

class AdapterConnectionError(Exception):
    """
    Raised when an adapter cannot connect to its backend.

    Unlike silent fallback, this surfaces real misconfiguration immediately.
    The MemoryGateway catches this, removes the failing adapter from the pool,
    and logs ERROR.  Callers that need a specific adapter should catch this.
    """


# ── Base adapter ──────────────────────────────────────────────────────────────

class MemoryAdapter(ABC):
    """
    Abstract base for all Cortex memory adapters.

    To add a new memory source to Cortex:
      1. Subclass MemoryAdapter
      2. Implement retrieve(), store(), schema()
      3. Register with the Gateway

    That's it. Cortex handles everything else.
    """

    @abstractmethod
    def retrieve(self, query: MemoryQuery) -> list[MemoryRecord]:
        """
        Retrieve memory records matching the query.
        Must return within the latency budget of the caller.
        Must never raise — return [] on error, log the failure.
        """
        ...

    @abstractmethod
    def store(self, record: MemoryRecord) -> StoreAck:
        """
        Persist a memory record.
        Must return a StoreAck regardless of success or failure.
        """
        ...

    @abstractmethod
    def schema(self) -> AdapterSchema:
        """
        Describe this adapter's capabilities and performance profile.
        Called once at registration time by the Gateway.
        """
        ...

    # ── Optional lifecycle hooks ───────────────────────────────────────────────

    def connect(self) -> None:
        """Called by Gateway when adapter is registered. Override to init connections."""
        pass

    def disconnect(self) -> None:
        """Called by Gateway on shutdown. Override to clean up connections."""
        pass

    def health_check(self) -> bool:
        """Returns True if the backend is reachable. Override for real checks."""
        return True

    def delete(self, record_id: str) -> bool:
        """Delete a record by ID. Returns True if deleted. Optional."""
        return False

    def count(self) -> int:
        """Return total number of records in this store. Optional."""
        return -1

    # ── Timing helper ─────────────────────────────────────────────────────────

    @staticmethod
    def _timed_retrieve(
        fn, query: MemoryQuery
    ) -> tuple[list[MemoryRecord], float]:
        t0 = time.perf_counter()
        results = fn(query)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return results, latency_ms
