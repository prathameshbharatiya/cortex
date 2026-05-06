"""
Mem0 Adapter
============
Connects Cortex to Mem0 (mem0.ai) — the leading managed memory API
for AI agents. $24M funded, 48K+ GitHub stars, 186M API calls/month.

Mem0 handles storage, retrieval, and memory management.
Cortex wraps it to:
  1. Normalise Mem0's output into MemoryRecord format
  2. Add temporal validity fields Mem0 doesn't track
  3. Plug Mem0 memories into the Certification Gate

Connection:
  pip install mem0ai
  export MEM0_API_KEY=your_key_here

Or self-hosted:
  Mem0Adapter(api_key="key", base_url="http://localhost:8000")

When Mem0 is unavailable this adapter falls back to InMemoryAdapter.
"""

from __future__ import annotations

import time
from typing import Any

from cortex.models.memory import MemoryRecord, MemoryType, OutcomeTag
from cortex.memory.adapters.base import (
    MemoryAdapter, MemoryQuery, AdapterSchema, StoreAck, QueryType
)
from cortex.memory.adapters.in_memory import InMemoryAdapter


class Mem0Adapter(MemoryAdapter):
    """
    Mem0-backed memory adapter.

    Uses Mem0's API for storage/retrieval, normalises into MemoryRecord.
    Falls back to InMemoryAdapter if Mem0 is unreachable.
    """

    def __init__(
        self,
        api_key:   str        = "",
        user_id:   str        = "cortex_robot",
        base_url:  str | None = None,
        name:      str        = "mem0",
        agent_id:  str | None = None,
    ) -> None:
        self.api_key  = api_key
        self.user_id  = user_id
        self.base_url = base_url
        self.name     = name
        self.agent_id = agent_id
        self._client: Any = None
        self._fallback    = InMemoryAdapter(name=f"{name}_fallback")
        self._connected   = False

    def connect(self) -> None:
        try:
            from mem0 import MemoryClient
            kwargs: dict[str, Any] = {"api_key": self.api_key}
            if self.base_url:
                kwargs["host"] = self.base_url
            self._client    = MemoryClient(**kwargs)
            self._connected = True
        except ImportError:
            self._connected = False   # mem0ai not installed
        except Exception:
            self._connected = False

    def health_check(self) -> bool:
        if not self._connected:
            return False
        try:
            self._client.get_all(user_id=self.user_id, limit=1)
            return True
        except Exception:
            return False

    # ── Core interface ────────────────────────────────────────────────────────

    def retrieve(self, query: MemoryQuery) -> list[MemoryRecord]:
        if not self._connected:
            return self._fallback.retrieve(query)
        try:
            return self._mem0_retrieve(query)
        except Exception:
            return self._fallback.retrieve(query)

    def store(self, record: MemoryRecord) -> StoreAck:
        if not self._connected:
            return self._fallback.store(record)

        t0 = time.perf_counter()
        try:
            messages = [{"role": "assistant", "content": record.content_text or str(record.content)}]
            kwargs: dict[str, Any] = {"user_id": self.user_id}
            if self.agent_id:
                kwargs["agent_id"] = self.agent_id
            self._client.add(messages, **kwargs)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            self._fallback.store(record)
            return StoreAck(record_id=record.record_id, success=True, latency_ms=latency_ms)
        except Exception as e:
            return StoreAck(record_id=record.record_id, success=False, error=str(e))

    def schema(self) -> AdapterSchema:
        return AdapterSchema(
            name=self.name,
            backend_type="mem0",
            supports_semantic_search=True,
            supports_temporal_filter=False,   # Mem0 doesn't expose temporal filters
            supports_spatial_search=False,
            supports_exact_lookup=False,
            supports_write=True,
            memory_types=[MemoryType.EPISODIC, MemoryType.SEMANTIC],
            avg_read_latency_ms=80.0,   # API round-trip
            avg_write_latency_ms=120.0,
            max_records=1_000_000,
            description=(
                f"Mem0 managed API (user_id={self.user_id}). "
                "Best for episodic and semantic memories. "
                "186M+ API calls/month in production."
            ),
        )

    # ── Mem0 retrieval ────────────────────────────────────────────────────────

    def _mem0_retrieve(self, query: MemoryQuery) -> list[MemoryRecord]:
        kwargs: dict[str, Any] = {
            "user_id": self.user_id,
            "limit":   query.top_k,
        }
        if self.agent_id:
            kwargs["agent_id"] = self.agent_id

        if query.query_text and query.query_type == QueryType.SEMANTIC:
            raw = self._client.search(query.query_text, **kwargs)
        else:
            raw = self._client.get_all(**kwargs)

        results = raw if isinstance(raw, list) else raw.get("results", [])

        records: list[MemoryRecord] = []
        for item in results:
            try:
                records.append(self._mem0_item_to_record(item))
            except Exception:
                pass
        return records[:query.top_k]

    def _mem0_item_to_record(self, item: dict) -> MemoryRecord:
        score = item.get("score", 0.8)
        return MemoryRecord(
            record_id=item.get("id", f"mem0_{time.time()}"),
            source=self.name,
            memory_type=MemoryType.EPISODIC,
            content=item,
            content_text=item.get("memory", ""),
            valid_at=time.time(),
            base_confidence=float(score) if score else 0.8,
            outcome_tag=OutcomeTag.UNKNOWN,
        )
