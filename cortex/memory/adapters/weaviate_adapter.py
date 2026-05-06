"""
Weaviate Adapter
================
Production semantic memory store for Cortex.  Vector + hybrid search,
millions of records, millisecond retrieval.

Requires weaviate-client >= 4.0:
  pip install "weaviate-client>=4.0"

Connection:
  Default URL: http://localhost:8080
  Docker:      docker run -d -p 8080:8080 semitechnologies/weaviate:latest
  Env:         WEAVIATE_URL=http://host:port

Raises AdapterConnectionError on connect() if Weaviate is unreachable.
Falls back to InMemoryAdapter during operation on transient errors.

Schema:
  Collection "CortexMemoryRecord" is created automatically on first connect.
  Uses vectorizer "text2vec-transformers" for semantic search on content_text.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any

from cortex.models.memory import MemoryRecord, MemoryType, OutcomeTag
from cortex.memory.adapters.base import (
    AdapterConnectionError,
    AdapterSchema,
    MemoryAdapter,
    MemoryQuery,
    QueryType,
    StoreAck,
)
from cortex.memory.adapters.in_memory import InMemoryAdapter

_log = logging.getLogger("cortex.memory.weaviate")

_CLASS_NAME = "CortexMemoryRecord"


def _wv_uuid(record_id: str) -> str:
    """Deterministic Weaviate UUID from Cortex record_id."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, record_id))


class WeaviateAdapter(MemoryAdapter):
    """
    Weaviate-backed semantic memory adapter using the v4 client API.

    Raises AdapterConnectionError on connect() if Weaviate is unreachable.
    Falls back to InMemoryAdapter on transient retrieval/store errors.
    """

    def __init__(
        self,
        url:     str        = "",
        name:    str        = "weaviate",
        api_key: str | None = None,
    ) -> None:
        self.url        = url or os.environ.get("WEAVIATE_URL", "http://localhost:8080")
        self.name       = name
        self.api_key    = api_key
        self._client: Any = None
        self._fallback  = InMemoryAdapter(name=f"{name}_fallback")
        self._connected = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def connect(self) -> None:
        """
        Connect to Weaviate using the v4 client.
        Raises AdapterConnectionError if unreachable.
        Auto-creates the CortexMemoryRecord collection on first connect.
        """
        try:
            import weaviate
        except ImportError as exc:
            raise AdapterConnectionError(
                "weaviate-client is not installed. "
                "Install with: pip install 'weaviate-client>=4.0'"
            ) from exc

        try:
            import weaviate
            import weaviate.classes as wvc

            auth = (
                weaviate.auth.AuthApiKey(api_key=self.api_key)
                if self.api_key
                else None
            )

            # Parse host/port from URL for v4 connect_to_custom
            from urllib.parse import urlparse
            parsed = urlparse(self.url)
            host   = parsed.hostname or "localhost"
            port   = parsed.port    or 8080
            secure = parsed.scheme  == "https"

            self._client = weaviate.connect_to_custom(
                http_host=host,
                http_port=port,
                http_secure=secure,
                grpc_host=host,
                grpc_port=50051,
                grpc_secure=secure,
                auth_credentials=auth,
            )

            if not self._client.is_ready():
                raise ConnectionError("Weaviate not ready")

            self._ensure_collection()
            self._connected = True
            _log.info("Weaviate adapter connected", extra={"url": self.url})

        except AdapterConnectionError:
            raise
        except Exception as exc:
            self._connected = False
            raise AdapterConnectionError(
                f"Weaviate unavailable at {self.url}: {exc}. "
                "Start Weaviate with: "
                "docker run -d -p 8080:8080 semitechnologies/weaviate:latest"
            ) from exc

    def disconnect(self) -> None:
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass
        self._connected = False

    def health_check(self) -> bool:
        if not self._connected or self._client is None:
            return False
        try:
            return bool(self._client.is_ready())
        except Exception:
            self._connected = False
            return False

    # ── Core interface ────────────────────────────────────────────────────────

    def retrieve(self, query: MemoryQuery) -> list[MemoryRecord]:
        if not self._connected:
            return self._fallback.retrieve(query)
        try:
            return self._weaviate_retrieve(query)
        except Exception as exc:
            _log.warning("Weaviate retrieve failed, using fallback", extra={"error": str(exc)})
            return self._fallback.retrieve(query)

    def store(self, record: MemoryRecord) -> StoreAck:
        if not self._connected:
            return self._fallback.store(record)

        t0 = time.perf_counter()
        try:
            import weaviate.classes as wvc
            collection = self._client.collections.get(_CLASS_NAME)
            props = {
                "recordId":       record.record_id,
                "source":         record.source,
                "memoryType":     record.memory_type.value,
                "contentText":    record.content_text,
                "validAt":        record.valid_at,
                "invalidAt":      record.invalid_at or 0.0,
                "baseConfidence": record.base_confidence,
                "outcomeTag":     record.outcome_tag.value,
                "outcomeCount":   record.outcome_count,
                "successCount":   record.success_count,
                "contentJson":    json.dumps(record.content, default=str),
            }
            collection.data.insert(
                properties=props,
                uuid=_wv_uuid(record.record_id),
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0
            self._fallback.store(record)
            return StoreAck(record_id=record.record_id, success=True, latency_ms=latency_ms)
        except Exception as exc:
            _log.error("Weaviate store failed", extra={"record_id": record.record_id, "error": str(exc)})
            return self._fallback.store(record)

    def schema(self) -> AdapterSchema:
        return AdapterSchema(
            name=self.name,
            backend_type="weaviate",
            supports_semantic_search=True,
            supports_temporal_filter=True,
            supports_spatial_search=False,
            supports_exact_lookup=True,
            supports_write=True,
            memory_types=list(MemoryType),
            avg_read_latency_ms=5.0,
            avg_write_latency_ms=10.0,
            max_records=10_000_000,
            description=(
                f"Weaviate v4 at {self.url}. "
                "Vector + hybrid search. Auto-schema on first connect."
            ),
        )

    def delete(self, record_id: str) -> bool:
        if not self._connected:
            return self._fallback.delete(record_id)
        try:
            collection = self._client.collections.get(_CLASS_NAME)
            collection.data.delete_by_id(_wv_uuid(record_id))
            self._fallback.delete(record_id)
            return True
        except Exception:
            return False

    def count(self) -> int:
        if not self._connected:
            return self._fallback.count()
        try:
            collection = self._client.collections.get(_CLASS_NAME)
            agg = collection.aggregate.over_all(total_count=True)
            return agg.total_count or 0
        except Exception:
            return -1

    # ── Weaviate retrieval ────────────────────────────────────────────────────

    def _weaviate_retrieve(self, query: MemoryQuery) -> list[MemoryRecord]:
        import weaviate.classes as wvc

        collection = self._client.collections.get(_CLASS_NAME)

        filters = self._build_filters(query)

        if query.query_type == QueryType.SEMANTIC and query.query_text:
            response = collection.query.near_text(
                query=query.query_text,
                limit=query.top_k,
                filters=filters,
                return_properties=[
                    "recordId", "source", "memoryType", "contentText",
                    "validAt", "invalidAt", "baseConfidence",
                    "outcomeTag", "outcomeCount", "successCount", "contentJson",
                ],
            )
        else:
            response = collection.query.fetch_objects(
                limit=query.top_k,
                filters=filters,
                return_properties=[
                    "recordId", "source", "memoryType", "contentText",
                    "validAt", "invalidAt", "baseConfidence",
                    "outcomeTag", "outcomeCount", "successCount", "contentJson",
                ],
            )

        records: list[MemoryRecord] = []
        for obj in response.objects:
            try:
                records.append(self._obj_to_record(obj.properties))
            except Exception:
                pass
        return records

    @staticmethod
    def _build_filters(query: MemoryQuery) -> Any:
        """Build Weaviate v4 filter from MemoryQuery."""
        try:
            import weaviate.classes as wvc
        except ImportError:
            return None

        conditions = []

        if query.memory_types:
            type_values = [mt.value for mt in query.memory_types]
            conditions.append(
                wvc.query.Filter.by_property("memoryType").contains_any(type_values)
            )

        if query.max_age_seconds is not None:
            cutoff = time.time() - query.max_age_seconds
            conditions.append(
                wvc.query.Filter.by_property("validAt").greater_than(cutoff)
            )

        if query.min_confidence > 0:
            conditions.append(
                wvc.query.Filter.by_property("baseConfidence").greater_or_equal(query.min_confidence)
            )

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        result = conditions[0]
        for c in conditions[1:]:
            result = result & c
        return result

    @staticmethod
    def _obj_to_record(props: dict[str, Any]) -> MemoryRecord:
        content    = json.loads(props.get("contentJson") or "{}")
        invalid_at = props.get("invalidAt") or None
        if invalid_at == 0.0:
            invalid_at = None
        return MemoryRecord(
            record_id=props["recordId"],
            source=props["source"],
            memory_type=MemoryType(props["memoryType"]),
            content=content,
            content_text=props.get("contentText", ""),
            valid_at=props["validAt"],
            invalid_at=invalid_at,
            base_confidence=props["baseConfidence"],
            outcome_tag=OutcomeTag(props.get("outcomeTag", "unknown")),
            outcome_count=props.get("outcomeCount", 0),
            success_count=props.get("successCount", 0),
        )

    # ── Schema management ─────────────────────────────────────────────────────

    def _ensure_collection(self) -> None:
        """Create the CortexMemoryRecord collection if it does not exist."""
        import weaviate.classes as wvc
        import weaviate.classes.config as wvcc

        existing = [c.name for c in self._client.collections.list_all(simple=True)]
        if _CLASS_NAME in existing:
            return

        self._client.collections.create(
            name=_CLASS_NAME,
            description="Cortex certified memory records",
            vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_transformers(),
            properties=[
                wvcc.Property(name="recordId",       data_type=wvcc.DataType.TEXT),
                wvcc.Property(name="source",         data_type=wvcc.DataType.TEXT),
                wvcc.Property(name="memoryType",     data_type=wvcc.DataType.TEXT),
                wvcc.Property(name="contentText",    data_type=wvcc.DataType.TEXT),
                wvcc.Property(name="validAt",        data_type=wvcc.DataType.NUMBER),
                wvcc.Property(name="invalidAt",      data_type=wvcc.DataType.NUMBER),
                wvcc.Property(name="baseConfidence", data_type=wvcc.DataType.NUMBER),
                wvcc.Property(name="outcomeTag",     data_type=wvcc.DataType.TEXT),
                wvcc.Property(name="outcomeCount",   data_type=wvcc.DataType.INT),
                wvcc.Property(name="successCount",   data_type=wvcc.DataType.INT),
                wvcc.Property(name="contentJson",    data_type=wvcc.DataType.TEXT),
            ],
        )
        _log.info("Weaviate collection created", extra={"collection": _CLASS_NAME})
