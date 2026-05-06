"""
JSON File Adapter
=================
Persists memory records to a JSON file on disk.
No database. No server. No dependencies beyond Python stdlib.

Use cases:
  - Single-robot deployments where simplicity matters more than speed
  - Development environments where you want persistence across restarts
  - Air-gapped systems where no network connectivity is available
  - Bootstrapping: start here, migrate to Weaviate when scale demands it

Performance:
  Reads: loads full file into memory, filters in-process (~5-20ms for <10k records)
  Writes: appends to in-memory index, flushes to disk asynchronously
  Not suitable for: > 50k records, high write frequency (> 100 writes/sec)

Migration path:
  When this adapter becomes too slow, swap it for WeaviateAdapter.
  The Gateway interface is identical — zero code changes in Cortex.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from cortex.models.memory import MemoryRecord, MemoryType, OutcomeTag
from cortex.memory.adapters.base import (
    MemoryAdapter, MemoryQuery, AdapterSchema, StoreAck, QueryType
)
from cortex.memory.adapters.in_memory import InMemoryAdapter


class JSONFileAdapter(MemoryAdapter):
    """
    File-backed memory adapter using JSON for persistence.

    Internally keeps an InMemoryAdapter as a hot cache.
    On init, loads all records from disk into the cache.
    On store, writes to cache immediately and flushes to disk.
    On retrieve, serves from cache (fast path, no disk I/O).
    """

    def __init__(
        self,
        file_path: str | Path,
        name:      str = "json_file",
        auto_flush: bool = True,
        flush_interval_s: float = 5.0,
    ) -> None:
        self.file_path       = Path(file_path)
        self.name            = name
        self.auto_flush      = auto_flush
        self.flush_interval  = flush_interval_s
        self._cache          = InMemoryAdapter(name=f"{name}_cache")
        self._lock           = threading.RLock()
        self._dirty          = False
        self._flush_thread:  threading.Thread | None = None
        self._running        = False

    def connect(self) -> None:
        """Load existing records from disk and start flush thread."""
        self._load_from_disk()
        if self.auto_flush:
            self._running = True
            self._flush_thread = threading.Thread(
                target=self._flush_loop,
                daemon=True,
                name=f"cortex-json-flush-{self.name}",
            )
            self._flush_thread.start()

    def disconnect(self) -> None:
        self._running = False
        self._flush_to_disk()   # final flush on shutdown

    # ── Core interface ────────────────────────────────────────────────────────

    def retrieve(self, query: MemoryQuery) -> list[MemoryRecord]:
        return self._cache.retrieve(query)

    def store(self, record: MemoryRecord) -> StoreAck:
        t0 = time.perf_counter()
        ack = self._cache.store(record)
        with self._lock:
            self._dirty = True
        if not self.auto_flush:
            self._flush_to_disk()
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return StoreAck(
            record_id=record.record_id,
            success=ack.success,
            latency_ms=latency_ms,
        )

    def schema(self) -> AdapterSchema:
        return AdapterSchema(
            name=self.name,
            backend_type="json_file",
            supports_semantic_search=True,
            supports_temporal_filter=True,
            supports_spatial_search=True,
            supports_exact_lookup=True,
            supports_write=True,
            memory_types=list(MemoryType),
            avg_read_latency_ms=1.0,
            avg_write_latency_ms=5.0,
            max_records=50_000,
            description=(
                f"JSON file persistence at {self.file_path}. "
                "In-memory cache with async disk flush."
            ),
        )

    def health_check(self) -> bool:
        try:
            parent = self.file_path.parent
            return parent.exists() and os.access(parent, os.W_OK)
        except Exception:
            return False

    def delete(self, record_id: str) -> bool:
        deleted = self._cache.delete(record_id)
        if deleted:
            with self._lock:
                self._dirty = True
        return deleted

    def count(self) -> int:
        return self._cache.count()

    def flush(self) -> None:
        """Manually flush cache to disk."""
        self._flush_to_disk()

    # ── Disk I/O ──────────────────────────────────────────────────────────────

    def _load_from_disk(self) -> None:
        if not self.file_path.exists():
            return
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            records_data = data.get("records", [])
            loaded = 0
            from cortex.migrations.schema import detect_version, migrate_record, CURRENT_VERSION as _SV
            for rd in records_data:
                try:
                    stored_ver = rd.get("_schema_version") or detect_version(rd)
                    if stored_ver != _SV:
                        rd = migrate_record(rd, stored_ver)
                        rd["_schema_version"] = _SV
                    record_data = {k: v for k, v in rd.items() if k != "_schema_version"}
                    record = MemoryRecord(**record_data)
                    self._cache.store(record)
                    loaded += 1
                except Exception:
                    pass
        except (json.JSONDecodeError, OSError):
            pass

    def _flush_to_disk(self) -> None:
        with self._lock:
            if not self._dirty:
                return
            records = self._cache.all_records()
            self._dirty = False

        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.file_path.with_suffix(".tmp")
        try:
            from cortex.migrations.schema import CURRENT_VERSION as _SV
            data = {
                "version": _SV,
                "adapter": self.name,
                "flushed_at": time.time(),
                "count": len(records),
                "records": [dict(**r.model_dump(), _schema_version=_SV) for r in records],
            }
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
            tmp_path.replace(self.file_path)   # atomic rename
        except OSError:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    def _flush_loop(self) -> None:
        while self._running:
            time.sleep(self.flush_interval)
            try:
                self._flush_to_disk()
            except Exception:
                pass
