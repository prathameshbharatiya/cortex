"""
Redis Adapter
=============
Sub-millisecond reads. Production L1 cache layer for Cortex.

Connection: requires a running Redis instance.
  Default:      redis://localhost:6379
  Environment:  REDIS_URL=redis://host:port/db
  Docker:       docker run -d -p 6379:6379 redis:7-alpine

Dependency: redis-py
  pip install redis

Connection pooling:
  Uses redis ConnectionPool with max_connections=50.
  Shared across all store/retrieve calls.

Retry policy:
  3 attempts, exponential backoff (0.1s, 0.2s, 0.4s) on transient errors.

Circuit breaker:
  After _CB_THRESHOLD consecutive failures the breaker opens.
  It re-probes after _CB_RECOVERY_S seconds.
  While open, all calls raise AdapterConnectionError immediately.

If Redis is not installed or never reachable, raises AdapterConnectionError
on connect().  The MemoryGateway handles this by removing the adapter and
logging ERROR — do not silence it.

Key schema
  cortex:record:{record_id}  → JSON-serialised MemoryRecord
  cortex:idx:type:{type}     → SET of record_ids with that MemoryType
  cortex:idx:source:{source} → SET of record_ids from that source
  cortex:meta                → HASH of adapter metadata
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from cortex.models.memory import MemoryRecord, MemoryType
from cortex.memory.adapters.base import (
    AdapterConnectionError,
    AdapterSchema,
    MemoryAdapter,
    MemoryQuery,
    QueryType,
    StoreAck,
)
from cortex.memory.adapters.in_memory import InMemoryAdapter

_log = logging.getLogger("cortex.memory.redis")

_PREFIX   = "cortex"
_RECORD   = f"{_PREFIX}:record"
_IDX_TYPE = f"{_PREFIX}:idx:type"
_IDX_SRC  = f"{_PREFIX}:idx:source"
_META     = f"{_PREFIX}:meta"
_DEFAULT_TTL_S = 3600

_CB_THRESHOLD   = 5     # consecutive failures before circuit opens
_CB_RECOVERY_S  = 30.0  # seconds before re-probe
_RETRY_ATTEMPTS = 3
_RETRY_BASE_S   = 0.1


class RedisAdapter(MemoryAdapter):
    """
    Redis-backed memory adapter with connection pooling, retry, and circuit breaker.

    Raises AdapterConnectionError on connect() if Redis is unreachable.
    During operation uses a warm InMemoryAdapter fallback for reads so the
    Gateway always gets results even during transient Redis failures.
    """

    def __init__(
        self,
        url:           str  = "",
        name:          str  = "redis",
        ttl_seconds:   int  = _DEFAULT_TTL_S,
        key_prefix:    str  = _PREFIX,
        db:            int  = 0,
        max_connections: int = 50,
    ) -> None:
        self.url        = url or os.environ.get("REDIS_URL", "redis://localhost:6379")
        self.name       = name
        self.ttl        = ttl_seconds
        self.prefix     = key_prefix
        self.db         = db
        self._max_conn  = max_connections
        self._pool: Any = None
        self._client: Any = None
        self._fallback  = InMemoryAdapter(name=f"{name}_fallback")
        self._connected = False

        # Circuit breaker state
        self._cb_failures  = 0
        self._cb_open      = False
        self._cb_open_at   = 0.0

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def connect(self) -> None:
        """
        Establish connection pool.  Raises AdapterConnectionError if Redis
        is not reachable so the Gateway can remove this adapter cleanly.
        """
        try:
            import redis
        except ImportError as exc:
            raise AdapterConnectionError(
                "redis-py is not installed. Install with: pip install redis"
            ) from exc

        try:
            import redis as _redis
            self._pool = _redis.ConnectionPool.from_url(
                self.url,
                db=self.db,
                decode_responses=True,
                max_connections=self._max_conn,
                socket_connect_timeout=2,
                socket_timeout=1,
            )
            self._client = _redis.Redis(connection_pool=self._pool)
            self._client.ping()
            self._connected = True
            self._cb_failures = 0
            self._cb_open = False
            self._client.hset(_META, mapping={
                "adapter": self.name,
                "connected_at": str(time.time()),
                "version": "1.0",
            })
            _log.info("Redis adapter connected", extra={"url": self.url})
        except Exception as exc:
            self._connected = False
            raise AdapterConnectionError(
                f"Redis unavailable at {self.url}: {exc}. "
                "Start Redis with: docker run -d -p 6379:6379 redis:7-alpine"
            ) from exc

    def disconnect(self) -> None:
        if self._pool:
            try:
                self._pool.disconnect()
            except Exception:
                pass
        self._connected = False

    def health_check(self) -> bool:
        if not self._connected or self._client is None:
            return False
        if self._cb_open:
            if time.time() - self._cb_open_at > _CB_RECOVERY_S:
                self._cb_open = False  # allow re-probe
            else:
                return False
        try:
            ok: bool = bool(self._client.ping())
            if ok:
                self._cb_failures = 0
            return ok
        except Exception:
            self._cb_failures += 1
            self._connected = False
            return False

    # ── Core interface ────────────────────────────────────────────────────────

    def retrieve(self, query: MemoryQuery) -> list[MemoryRecord]:
        if not self._connected or self._cb_open:
            return self._fallback.retrieve(query)
        try:
            return self._with_retry(lambda: self._redis_retrieve(query))
        except Exception as exc:
            _log.warning("Redis retrieve failed, using fallback", extra={"error": str(exc)})
            return self._fallback.retrieve(query)

    def store(self, record: MemoryRecord) -> StoreAck:
        if not self._connected or self._cb_open:
            return self._fallback.store(record)

        t0 = time.perf_counter()
        try:
            def _do_store() -> None:
                from cortex.migrations.schema import CURRENT_VERSION as _SV
                raw_dict = record.model_dump()
                raw_dict["_schema_version"] = _SV
                payload = json.dumps(raw_dict, default=str)
                pipe = self._client.pipeline()
                pipe.setex(f"{_RECORD}:{record.record_id}", self.ttl, payload)
                pipe.sadd(f"{_IDX_TYPE}:{record.memory_type.value}", record.record_id)
                pipe.sadd(f"{_IDX_SRC}:{record.source}", record.record_id)
                pipe.execute()

            self._with_retry(_do_store)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            self._fallback.store(record)
            return StoreAck(record_id=record.record_id, success=True, latency_ms=latency_ms)
        except Exception as exc:
            _log.error("Redis store failed", extra={"record_id": record.record_id, "error": str(exc)})
            return self._fallback.store(record)

    def schema(self) -> AdapterSchema:
        return AdapterSchema(
            name=self.name,
            backend_type="redis",
            supports_semantic_search=False,
            supports_temporal_filter=True,
            supports_spatial_search=False,
            supports_exact_lookup=True,
            supports_write=True,
            memory_types=list(MemoryType),
            avg_read_latency_ms=0.3,
            avg_write_latency_ms=0.5,
            max_records=500_000,
            description=(
                f"Redis at {self.url}. Sub-millisecond reads. "
                f"TTL={self.ttl}s. Connection pool={self._max_conn}."
            ),
        )

    def delete(self, record_id: str) -> bool:
        if not self._connected or self._cb_open:
            return self._fallback.delete(record_id)
        try:
            deleted = self._client.delete(f"{_RECORD}:{record_id}")
            self._fallback.delete(record_id)
            return bool(deleted)
        except Exception:
            return False

    def count(self) -> int:
        if not self._connected or self._cb_open:
            return self._fallback.count()
        try:
            keys = self._client.keys(f"{_RECORD}:*")
            return len(keys)
        except Exception:
            return -1

    # ── Retry + circuit breaker ───────────────────────────────────────────────

    def _with_retry(self, fn: Any) -> Any:
        """
        Call fn with up to _RETRY_ATTEMPTS attempts, exponential backoff.
        Updates circuit breaker on repeated failures.
        """
        last_exc: Exception | None = None
        for attempt in range(_RETRY_ATTEMPTS):
            try:
                result = fn()
                self._cb_failures = 0
                return result
            except Exception as exc:
                last_exc = exc
                self._cb_failures += 1
                if self._cb_failures >= _CB_THRESHOLD:
                    self._cb_open    = True
                    self._cb_open_at = time.time()
                    _log.error(
                        "Redis circuit breaker opened",
                        extra={"failures": self._cb_failures, "url": self.url},
                    )
                    raise AdapterConnectionError(
                        f"Redis circuit breaker open after {self._cb_failures} failures"
                    ) from exc
                if attempt < _RETRY_ATTEMPTS - 1:
                    delay = _RETRY_BASE_S * (2 ** attempt)
                    time.sleep(delay)
        raise last_exc  # type: ignore[misc]

    # ── Redis retrieval internals ─────────────────────────────────────────────

    def _redis_retrieve(self, query: MemoryQuery) -> list[MemoryRecord]:
        record_ids: set[str] | None = None

        if query.record_ids:
            record_ids = set(query.record_ids)
        elif query.memory_types:
            sets = [f"{_IDX_TYPE}:{mt.value}" for mt in query.memory_types]
            record_ids = (
                set(self._client.smembers(sets[0]))
                if len(sets) == 1
                else set(self._client.sunion(*sets))
            )

        if query.source_filter and record_ids is None:
            src_sets = [f"{_IDX_SRC}:{s}" for s in query.source_filter]
            src_ids  = set(self._client.sunion(*src_sets))
            record_ids = src_ids if record_ids is None else record_ids & src_ids

        if record_ids is not None:
            keys = [f"{_RECORD}:{rid}" for rid in record_ids]
        else:
            keys = self._client.keys(f"{_RECORD}:*")

        if not keys:
            return []

        pipe = self._client.pipeline()
        for k in keys:
            pipe.get(k)
        raw_list = pipe.execute()

        records: list[MemoryRecord] = []
        for raw in raw_list:
            if raw is None:
                continue
            try:
                raw_dict = json.loads(raw)
                stored_ver = raw_dict.get("_schema_version")
                from cortex.migrations.schema import CURRENT_VERSION as _SV, migrate_record
                if stored_ver is None:
                    from cortex.migrations.schema import detect_version
                    stored_ver = detect_version(raw_dict)
                if stored_ver != _SV:
                    raw_dict = migrate_record(raw_dict, stored_ver)
                    raw_dict["_schema_version"] = _SV
                records.append(MemoryRecord.model_validate(raw_dict))
            except Exception:
                pass

        now = time.time()
        if query.max_age_seconds is not None:
            cutoff = now - query.max_age_seconds
            records = [r for r in records if r.valid_at >= cutoff]

        records = [r for r in records if r.is_currently_valid]

        if query.min_confidence > 0:
            records = [r for r in records if r.effective_confidence >= query.min_confidence]

        if not query.include_failed:
            from cortex.models.memory import OutcomeTag
            records = [
                r for r in records
                if r.outcome_tag != OutcomeTag.FAILURE or r.outcome_count < 3
            ]

        records.sort(key=lambda r: r.effective_confidence, reverse=True)
        return records[:query.top_k]
