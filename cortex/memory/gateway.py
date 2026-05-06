"""
Memory Gateway
==============
The single point of contact between Cortex and any memory system.

The Gateway:
  1. Accepts adapter registrations from any memory system
  2. Routes queries to the right adapter(s) based on query type and schema
  3. Merges and deduplicates results from multiple adapters
  4. Normalises all records into the canonical MemoryRecord format
  5. Enforces temporal validity — invalid records never leave the Gateway
  6. Tracks adapter health and routes around failures automatically

Usage
-----
    from cortex.memory.gateway import MemoryGateway
    from cortex.memory.adapters import InMemoryAdapter, RedisAdapter

    gw = MemoryGateway()
    gw.register(InMemoryAdapter(), priority=1)   # L1 cache
    gw.register(RedisAdapter(),    priority=2)   # L2 store

    records = gw.retrieve(MemoryQuery(query_text="pick object"))
    gw.store(record)

The Certification Gate calls gw.retrieve() to populate the CTX.
The Experience Tracker calls gw.store() to write outcomes back.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from cortex.models.memory import MemoryRecord, MemoryType
from cortex.memory.adapters.base import (
    MemoryAdapter, MemoryQuery, AdapterSchema, StoreAck, QueryType
)
from cortex.memory.adapters.in_memory import InMemoryAdapter


# ── Registered adapter entry ──────────────────────────────────────────────────

@dataclass
class AdapterEntry:
    adapter:    MemoryAdapter
    schema:     AdapterSchema
    priority:   int              # lower = higher priority (1 = first)
    enabled:    bool = True
    last_health_check: float = field(default_factory=time.time)
    health_ok:  bool = True
    read_count: int  = 0
    write_count: int = 0
    error_count: int = 0


# ── Retrieval result ──────────────────────────────────────────────────────────

@dataclass
class GatewayResult:
    records:          list[MemoryRecord]
    total_latency_ms: float
    adapters_queried: list[str]
    adapters_failed:  list[str]
    deduplicated:     int             # how many duplicates were removed
    filtered_invalid: int             # how many expired records were removed


# ── Gateway ───────────────────────────────────────────────────────────────────

class MemoryGateway:
    """
    The unified memory interface for Cortex.

    Model-agnostic: doesn't care which AI generated the memory.
    Memory-agnostic: doesn't care which system stores it.
    Stack-agnostic: doesn't care which robot is running.
    """

    def __init__(
        self,
        health_check_interval_s: float = 30.0,
        default_top_k:           int   = 10,
        fan_out:                 bool  = False,   # query all adapters vs priority order
    ) -> None:
        self._adapters:   list[AdapterEntry] = []
        self._lock        = threading.RLock()
        self._hc_interval = health_check_interval_s
        self.default_top_k = default_top_k
        self.fan_out       = fan_out
        self._stats: dict[str, int] = defaultdict(int)
        self._running      = False
        self._hc_thread:   threading.Thread | None = None

        # Always have an in-memory fallback — the Gateway is never empty
        self._fallback = InMemoryAdapter(name="gateway_fallback")

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> "MemoryGateway":
        """Connect all adapters and start health-check thread. Returns self for chaining."""
        with self._lock:
            for entry in self._adapters:
                self._connect_adapter(entry)
        self._running = True
        self._hc_thread = threading.Thread(
            target=self._health_check_loop,
            daemon=True,
            name="cortex-gateway-hc",
        )
        self._hc_thread.start()
        return self

    def stop(self) -> None:
        self._running = False
        with self._lock:
            for entry in self._adapters:
                try:
                    entry.adapter.disconnect()
                except Exception:
                    pass

    def __enter__(self) -> "MemoryGateway":
        return self.start()

    def __exit__(self, *_: Any) -> None:
        self.stop()

    # ── Registration ──────────────────────────────────────────────────────────

    def register(
        self,
        adapter:  MemoryAdapter,
        priority: int = 10,
        connect:  bool = True,
    ) -> "MemoryGateway":
        """
        Register a memory adapter with the Gateway.

        priority: lower number = higher priority (1 = first queried).
        connect:  if True, calls adapter.connect() immediately.
        Returns self for chaining.

        Example:
            gw.register(RedisAdapter(),    priority=1)   # fastest, check first
            gw.register(WeaviateAdapter(), priority=2)   # semantic search
            gw.register(JSONFileAdapter(), priority=3)   # persistent fallback
        """
        schema = adapter.schema()
        entry  = AdapterEntry(
            adapter=adapter,
            schema=schema,
            priority=priority,
        )
        if connect:
            self._connect_adapter(entry)

        with self._lock:
            self._adapters.append(entry)
            self._adapters.sort(key=lambda e: e.priority)

        return self

    def unregister(self, adapter_name: str) -> bool:
        with self._lock:
            for i, entry in enumerate(self._adapters):
                if entry.schema.name == adapter_name:
                    entry.adapter.disconnect()
                    self._adapters.pop(i)
                    return True
        return False

    # ── Core interface ────────────────────────────────────────────────────────

    def retrieve(self, query: MemoryQuery) -> GatewayResult:
        """
        Retrieve memory records from registered adapters.

        Routing strategy:
          fan_out=False (default): query adapters in priority order,
            stop when top_k results are found.
          fan_out=True: query all adapters, merge and deduplicate results.
            Slower but more complete. Use for context assembly.
        """
        t0 = time.perf_counter()
        top_k = query.top_k or self.default_top_k

        if not self._adapters:
            records = self._fallback.retrieve(query)
            return GatewayResult(
                records=records,
                total_latency_ms=(time.perf_counter() - t0) * 1000,
                adapters_queried=["gateway_fallback"],
                adapters_failed=[],
                deduplicated=0,
                filtered_invalid=0,
            )

        healthy = self._healthy_adapters()
        if not healthy:
            records = self._fallback.retrieve(query)
            return GatewayResult(
                records=records,
                total_latency_ms=(time.perf_counter() - t0) * 1000,
                adapters_queried=["gateway_fallback"],
                adapters_failed=[e.schema.name for e in self._adapters],
                deduplicated=0,
                filtered_invalid=0,
            )

        # Route to capable adapters
        capable = self._capable_adapters(query, healthy)
        if not capable:
            capable = healthy   # fall back to any healthy adapter

        queried: list[str] = []
        failed:  list[str] = []
        all_records: list[MemoryRecord] = []

        if self.fan_out:
            # Query all capable adapters and merge
            for entry in capable:
                result, ok = self._query_adapter(entry, query)
                queried.append(entry.schema.name)
                if ok:
                    all_records.extend(result)
                else:
                    failed.append(entry.schema.name)
        else:
            # Priority order — stop when satisfied
            for entry in capable:
                result, ok = self._query_adapter(entry, query)
                queried.append(entry.schema.name)
                if ok:
                    all_records.extend(result)
                    if len(all_records) >= top_k:
                        break
                else:
                    failed.append(entry.schema.name)

        # Post-process
        before_filter = len(all_records)
        all_records   = [r for r in all_records if r.is_currently_valid]
        filtered      = before_filter - len(all_records)

        before_dedup  = len(all_records)
        all_records   = self._deduplicate(all_records)
        deduped       = before_dedup - len(all_records)

        all_records   = self._rank(all_records, query)[:top_k]

        self._stats["total_retrieves"] += 1
        self._stats["total_records_returned"] += len(all_records)

        return GatewayResult(
            records=all_records,
            total_latency_ms=(time.perf_counter() - t0) * 1000,
            adapters_queried=queried,
            adapters_failed=failed,
            deduplicated=deduped,
            filtered_invalid=filtered,
        )

    def store(
        self,
        record:    MemoryRecord,
        adapters:  list[str] | None = None,   # None = write to all write-capable
    ) -> list[StoreAck]:
        """
        Store a memory record in one or more adapters.

        adapters: list of adapter names to write to.
                  None = write to all registered write-capable adapters.
        Returns list of StoreAck, one per adapter written.
        """
        targets = self._write_targets(adapters)
        acks: list[StoreAck] = []

        for entry in targets:
            try:
                ack = entry.adapter.store(record)
                entry.write_count += 1
                if not ack.success:
                    entry.error_count += 1
            except Exception as e:
                ack = StoreAck(record_id=record.record_id, success=False, error=str(e))
                entry.error_count += 1
            acks.append(ack)

        # Always write to fallback too
        self._fallback.store(record)
        self._stats["total_stores"] += 1
        return acks

    def retrieve_records(self, query: MemoryQuery) -> list[MemoryRecord]:
        """Convenience wrapper — returns just the records list."""
        return self.retrieve(query).records

    # ── Adapter introspection ─────────────────────────────────────────────────

    @property
    def adapter_count(self) -> int:
        with self._lock:
            return len(self._adapters)

    def adapters(self) -> list[AdapterSchema]:
        with self._lock:
            return [e.schema for e in self._adapters]

    def adapter_status(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "name":        e.schema.name,
                    "backend":     e.schema.backend_type,
                    "priority":    e.priority,
                    "healthy":     e.health_ok,
                    "enabled":     e.enabled,
                    "reads":       e.read_count,
                    "writes":      e.write_count,
                    "errors":      e.error_count,
                }
                for e in self._adapters
            ]

    def stats(self) -> dict:
        return dict(self._stats)

    def enable(self, name: str) -> bool:
        return self._set_enabled(name, True)

    def disable(self, name: str) -> bool:
        return self._set_enabled(name, False)

    def count_all(self) -> dict[str, int]:
        with self._lock:
            return {e.schema.name: e.adapter.count() for e in self._adapters}

    # ── Internal routing helpers ──────────────────────────────────────────────

    def _healthy_adapters(self) -> list[AdapterEntry]:
        with self._lock:
            return [e for e in self._adapters if e.enabled and e.health_ok]

    def _capable_adapters(
        self, query: MemoryQuery, entries: list[AdapterEntry]
    ) -> list[AdapterEntry]:
        """Filter to adapters that support the query type."""
        qt = query.query_type
        capable = []
        for e in entries:
            s = e.schema
            if qt == QueryType.SEMANTIC and not s.supports_semantic_search:
                continue
            if qt == QueryType.SPATIAL and not s.supports_spatial_search:
                continue
            if qt == QueryType.EXACT and not s.supports_exact_lookup:
                continue
            if query.memory_types:
                overlap = set(query.memory_types) & set(s.memory_types)
                if not overlap and s.memory_types:
                    continue
            capable.append(e)
        return capable or entries   # if nothing is capable, try all

    def _write_targets(self, names: list[str] | None) -> list[AdapterEntry]:
        with self._lock:
            entries = [
                e for e in self._adapters
                if e.enabled and e.health_ok and e.schema.supports_write
            ]
        if names:
            entries = [e for e in entries if e.schema.name in names]
        return entries

    def _query_adapter(
        self, entry: AdapterEntry, query: MemoryQuery
    ) -> tuple[list[MemoryRecord], bool]:
        try:
            records = entry.adapter.retrieve(query)
            entry.read_count += 1
            return records, True
        except Exception:
            entry.error_count += 1
            return [], False

    @staticmethod
    def _deduplicate(records: list[MemoryRecord]) -> list[MemoryRecord]:
        """Remove duplicate records by record_id, keeping highest confidence."""
        seen: dict[str, MemoryRecord] = {}
        for r in records:
            if r.record_id not in seen:
                seen[r.record_id] = r
            elif r.effective_confidence > seen[r.record_id].effective_confidence:
                seen[r.record_id] = r
        return list(seen.values())

    @staticmethod
    def _rank(records: list[MemoryRecord], query: MemoryQuery) -> list[MemoryRecord]:
        """Final ranking of merged results."""
        if query.query_type == QueryType.TEMPORAL:
            return sorted(records, key=lambda r: r.valid_at, reverse=True)
        # Default: confidence × freshness
        now = time.time()
        def score(r: MemoryRecord) -> float:
            age_h = (now - r.valid_at) / 3600.0
            freshness = max(0.2, 1.0 - age_h * 0.05)
            return r.effective_confidence * freshness
        return sorted(records, key=score, reverse=True)

    # ── Health checking ───────────────────────────────────────────────────────

    def _connect_adapter(self, entry: AdapterEntry) -> None:
        try:
            entry.adapter.connect()
            entry.health_ok = entry.adapter.health_check()
        except Exception:
            entry.health_ok = False

    def _health_check_loop(self) -> None:
        while self._running:
            time.sleep(self._hc_interval)
            with self._lock:
                entries = list(self._adapters)
            for entry in entries:
                try:
                    entry.health_ok = entry.adapter.health_check()
                    entry.last_health_check = time.time()
                except Exception:
                    entry.health_ok = False

    def _set_enabled(self, name: str, value: bool) -> bool:
        with self._lock:
            for entry in self._adapters:
                if entry.schema.name == name:
                    entry.enabled = value
                    return True
        return False
