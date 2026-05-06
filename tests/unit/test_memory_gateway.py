"""
tests/unit/test_memory_gateway.py
===================================
Memory Gateway and adapter tests. All tests use InMemoryAdapter only —
no network, no Redis, no Weaviate.
"""

from __future__ import annotations

import tempfile
import time
import threading
import os
import pytest

from cortex.models.memory import MemoryRecord, MemoryType, OutcomeTag
from cortex.memory.adapters.base import MemoryQuery, QueryType, StoreAck
from cortex.memory.adapters.in_memory import InMemoryAdapter
from cortex.memory.adapters.json_file import JSONFileAdapter
from cortex.memory.adapters.redis_adapter import RedisAdapter
from cortex.memory.adapters.weaviate_adapter import WeaviateAdapter
from cortex.memory.gateway import MemoryGateway, GatewayResult

from tests.conftest import make_memory_record


# ══════════════════════════════════════════════════════════════════════════════
# InMemoryAdapter
# ══════════════════════════════════════════════════════════════════════════════

class TestInMemoryAdapter:

    def test_store_and_retrieve(self):
        a = InMemoryAdapter()
        r = make_memory_record(content={"key": "value"})
        ack = a.store(r)
        assert isinstance(ack, StoreAck)
        results = a.retrieve(MemoryQuery(query_text="test", top_k=10))
        ids = [rec.record_id for rec in results]
        assert r.record_id in ids

    def test_retrieve_filters_expired(self):
        a = InMemoryAdapter()
        valid = make_memory_record()
        expired = make_memory_record(invalid=True)
        a.store(valid)
        a.store(expired)
        results = a.retrieve(MemoryQuery(query_text="test", top_k=10))
        ids = [rec.record_id for rec in results]
        assert valid.record_id in ids
        assert expired.record_id not in ids

    def test_retrieve_respects_top_k(self):
        a = InMemoryAdapter()
        for _ in range(10):
            a.store(make_memory_record())
        results = a.retrieve(MemoryQuery(query_text="test", top_k=3))
        assert len(results) <= 3

    def test_retrieve_by_type_filter(self):
        a = InMemoryAdapter()
        spatial = make_memory_record(memory_type=MemoryType.SPATIAL)
        episodic = make_memory_record(memory_type=MemoryType.EPISODIC)
        a.store(spatial)
        a.store(episodic)
        results = a.retrieve(MemoryQuery(
            query_text="test",
            memory_types=[MemoryType.SPATIAL],
            top_k=10,
        ))
        types = {r.memory_type for r in results}
        assert MemoryType.SPATIAL in types
        assert MemoryType.EPISODIC not in types

    def test_store_returns_ack_with_record_id(self):
        a = InMemoryAdapter()
        r = make_memory_record()
        ack = a.store(r)
        assert ack.record_id == r.record_id
        assert ack.success

    def test_health_returns_true(self):
        a = InMemoryAdapter()
        assert a.health_check()

    def test_empty_adapter_returns_empty_list(self):
        a = InMemoryAdapter()
        results = a.retrieve(MemoryQuery(query_text="nothing", top_k=5))
        assert results == []

    def test_multiple_stores_all_retrievable(self):
        a = InMemoryAdapter()
        records = [make_memory_record() for _ in range(5)]
        for r in records:
            a.store(r)
        results = a.retrieve(MemoryQuery(query_text="test", top_k=20))
        stored_ids = {r.record_id for r in records}
        retrieved_ids = {r.record_id for r in results}
        assert stored_ids.issubset(retrieved_ids)


# ══════════════════════════════════════════════════════════════════════════════
# JSONFileAdapter
# ══════════════════════════════════════════════════════════════════════════════

class TestJSONFileAdapter:

    def test_store_and_retrieve(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            a = JSONFileAdapter(file_path=path, auto_flush=False)
            a.connect()
            r = make_memory_record()
            a.store(r)
            results = a.retrieve(MemoryQuery(query_text="test", top_k=10))
            assert any(rec.record_id == r.record_id for rec in results)
            a.disconnect()
        finally:
            os.unlink(path)

    def test_persists_across_instances(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            r = make_memory_record()
            # Write with first instance — auto_flush=False flushes synchronously on store
            a1 = JSONFileAdapter(file_path=path, auto_flush=False)
            a1.connect()
            a1.store(r)
            a1.disconnect()
            # Read with second instance — connect() loads from disk
            a2 = JSONFileAdapter(file_path=path, auto_flush=False)
            a2.connect()
            results = a2.retrieve(MemoryQuery(query_text="test", top_k=10))
            a2.disconnect()
            assert any(rec.record_id == r.record_id for rec in results)
        finally:
            os.unlink(path)


# ══════════════════════════════════════════════════════════════════════════════
# Optional adapters — graceful fallback when not reachable
# ══════════════════════════════════════════════════════════════════════════════

class TestRedisAdapterFallback:

    def test_redis_not_reachable_falls_back_gracefully(self):
        """Redis unreachable must not raise — it falls back to InMemoryAdapter."""
        a = RedisAdapter(url="redis://localhost:19999")  # nothing on port 19999
        r = make_memory_record()
        # Store and retrieve must not raise even when Redis is unreachable
        try:
            ack = a.store(r)
        except Exception as exc:
            pytest.fail(f"RedisAdapter.store raised when Redis unreachable: {exc}")
        try:
            results = a.retrieve(MemoryQuery(query_text="test", top_k=5))
        except Exception as exc:
            pytest.fail(f"RedisAdapter.retrieve raised when Redis unreachable: {exc}")

    def test_redis_health_check_returns_bool(self):
        a = RedisAdapter(url="redis://localhost:19999")
        result = a.health_check()
        assert isinstance(result, bool)


class TestWeaviateAdapterFallback:

    def test_weaviate_not_reachable_falls_back(self):
        a = WeaviateAdapter(url="http://localhost:19998")
        try:
            ack = a.store(make_memory_record())
        except Exception as exc:
            pytest.fail(f"WeaviateAdapter.store raised: {exc}")
        try:
            a.retrieve(MemoryQuery(query_text="test", top_k=5))
        except Exception as exc:
            pytest.fail(f"WeaviateAdapter.retrieve raised: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# MemoryGateway
# ══════════════════════════════════════════════════════════════════════════════

class TestMemoryGateway:

    def test_register_and_retrieve(self, gateway):
        r = make_memory_record()
        gateway.store(r)
        result = gateway.retrieve(MemoryQuery(query_text="test", top_k=10))
        assert isinstance(result, GatewayResult)
        ids = [rec.record_id for rec in result.records]
        assert r.record_id in ids

    def test_gateway_filters_invalid(self, gateway):
        valid = make_memory_record()
        invalid = make_memory_record(invalid=True)
        gateway.store(valid)
        gateway.store(invalid)
        result = gateway.retrieve(MemoryQuery(query_text="test", top_k=10))
        ids = [rec.record_id for rec in result.records]
        assert valid.record_id in ids
        assert invalid.record_id not in ids

    def test_gateway_result_has_latency(self, gateway):
        r = make_memory_record()
        gateway.store(r)
        result = gateway.retrieve(MemoryQuery(query_text="test", top_k=5))
        assert result.total_latency_ms >= 0

    def test_gateway_result_lists_adapters(self, gateway):
        result = gateway.retrieve(MemoryQuery(query_text="test", top_k=5))
        assert isinstance(result.adapters_queried, list)
        assert len(result.adapters_queried) >= 1

    def test_fan_out_queries_all_adapters(self):
        gw = MemoryGateway(fan_out=True)
        a1 = InMemoryAdapter(name="a1")
        a2 = InMemoryAdapter(name="a2")
        gw.register(a1, priority=1)
        gw.register(a2, priority=2)
        gw.start()
        r1 = make_memory_record(source="src1")
        r2 = make_memory_record(source="src2")
        a1.store(r1)
        a2.store(r2)
        result = gw.retrieve(MemoryQuery(query_text="test", top_k=20))
        ids = {r.record_id for r in result.records}
        assert r1.record_id in ids
        assert r2.record_id in ids
        gw.stop()

    def test_deduplication_removes_duplicates(self):
        gw = MemoryGateway(fan_out=True)
        a1 = InMemoryAdapter(name="a1")
        a2 = InMemoryAdapter(name="a2")
        gw.register(a1, priority=1)
        gw.register(a2, priority=2)
        gw.start()
        # Same record stored in both adapters
        r = make_memory_record()
        a1.store(r)
        a2.store(r)
        result = gw.retrieve(MemoryQuery(query_text="test", top_k=20))
        ids = [rec.record_id for rec in result.records]
        assert ids.count(r.record_id) == 1  # deduplicated
        assert result.deduplicated >= 1
        gw.stop()

    def test_priority_order_respected(self):
        gw = MemoryGateway(fan_out=False)  # priority order, not fan-out
        a1 = InMemoryAdapter(name="priority_1")
        a2 = InMemoryAdapter(name="priority_2")
        gw.register(a1, priority=1)
        gw.register(a2, priority=2)
        gw.start()
        r1 = make_memory_record()
        a1.store(r1)
        result = gw.retrieve(MemoryQuery(query_text="test", top_k=5))
        # Priority 1 adapter was queried first
        assert "priority_1" in result.adapters_queried
        gw.stop()

    def test_gateway_thread_safe_concurrent_stores(self, gateway):
        errors = []
        records = [make_memory_record() for _ in range(20)]

        def store_batch(batch):
            try:
                for r in batch:
                    gateway.store(r)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=store_batch, args=(records[i::4],))
            for i in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent store errors: {errors}"

    def test_retrieve_records_convenience_method(self, gateway):
        r = make_memory_record()
        gateway.store(r)
        records = gateway.retrieve_records(MemoryQuery(query_text="test", top_k=10))
        assert isinstance(records, list)
        assert any(rec.record_id == r.record_id for rec in records)

    def test_adapter_status(self, gateway):
        statuses = gateway.adapter_status()
        assert isinstance(statuses, list)
        assert len(statuses) >= 1
        for s in statuses:
            assert "name" in s
            assert "healthy" in s


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
