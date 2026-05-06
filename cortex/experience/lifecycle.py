"""
Lifecycle Manager
=================
Keeps memory stores healthy over time.

Without active lifecycle management, memory stores degrade:
  - Retrieval slows as record counts grow
  - Relevance Engine scores become noisy (too many irrelevant candidates)
  - Contradicted and stale records pollute the CTX
  - Duplicate experiences fragment outcome statistics

The Lifecycle Manager runs four operations on a configurable schedule:

  1. Compression
     Cluster semantically similar records. Replace the cluster with a
     single representative record preserving the aggregate outcome
     statistics and the highest-confidence content. Reduces store size
     without losing signal.

  2. Temporal decay
     Records that have not been sensor-confirmed and approach their
     type-specific half-life get their confidence weight reduced.
     Records at zero effective confidence are marked for archival.

  3. Outcome-weighted prioritisation
     Elevate records associated with consistent successes.
     Retain failure records with negative outcome tags — failure history
     is information. Discard records with no outcome association after
     the forgetting window.

  4. Forgetting
     Records that consistently score below the relevance threshold
     across multiple retrieval cycles are soft-deleted (archived with
     metadata) and eventually hard-deleted after the retention period.

The Manager operates on the in-memory adapter directly and pushes
changes through the Gateway to persistent stores.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from cortex.models.memory import MemoryRecord, MemoryType, OutcomeTag
from cortex.memory.gateway import MemoryGateway
from cortex.memory.adapters.base import MemoryQuery, QueryType
from cortex.memory.adapters.in_memory import InMemoryAdapter


# ── Half-lives for decay (match temporal scorer) ──────────────────────────────
_HALF_LIVES = {
    MemoryType.SPATIAL:    5    * 60,
    MemoryType.EPISODIC:   2    * 3600,
    MemoryType.SEMANTIC:   24   * 3600,
    MemoryType.PROCEDURAL: 7    * 24 * 3600,
}


# ── Lifecycle configuration ───────────────────────────────────────────────────

@dataclass
class LifecycleConfig:
    # Compression
    compression_enabled:     bool  = True
    compression_threshold:   int   = 1000     # compress when store > N records
    min_cluster_size:        int   = 3        # min records to form a cluster

    # Decay
    decay_enabled:           bool  = True
    decay_confidence_floor:  float = 0.10     # mark for archival below this
    decay_check_interval_s:  float = 300.0    # every 5 minutes

    # Forgetting
    forgetting_enabled:      bool  = True
    forgetting_window_s:     float = 7 * 24 * 3600   # 7 days
    min_retrievals_to_keep:  int   = 1                # keep if retrieved ≥ N times
    hard_delete_after_s:     float = 30 * 24 * 3600  # hard delete after 30 days

    # Maintenance schedule
    maintenance_interval_s:  float = 600.0    # full maintenance every 10 minutes
    max_records_per_adapter: int   = 50_000   # trigger compression above this


# ── Operation reports ─────────────────────────────────────────────────────────

@dataclass
class MaintenanceReport:
    timestamp:           float = field(default_factory=time.time)
    records_before:      int   = 0
    records_after:       int   = 0
    compressed:          int   = 0
    decayed:             int   = 0
    forgotten:           int   = 0
    archived:            int   = 0
    latency_ms:          float = 0.0
    adapter_name:        str   = ""
    warnings:            list[str] = field(default_factory=list)

    @property
    def records_removed(self) -> int:
        return self.records_before - self.records_after


# ── Lifecycle Manager ─────────────────────────────────────────────────────────

class LifecycleManager:
    """
    Maintains health of all memory stores connected via the Gateway.

    Runs maintenance operations on a background thread.
    Can also be triggered manually via run_maintenance().
    """

    def __init__(
        self,
        gateway: MemoryGateway | None = None,
        config:  LifecycleConfig | None = None,
    ) -> None:
        self.gateway  = gateway
        self.cfg      = config or LifecycleConfig()
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock    = threading.RLock()
        self._reports: list[MaintenanceReport] = []
        self._archive: list[MemoryRecord] = []   # soft-deleted records

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> "LifecycleManager":
        """Start the background maintenance thread."""
        self._running = True
        self._thread  = threading.Thread(
            target=self._maintenance_loop,
            daemon=True,
            name="cortex-lifecycle",
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        self._running = False

    def __enter__(self) -> "LifecycleManager":
        return self.start()

    def __exit__(self, *_: Any) -> None:
        self.stop()

    # ── Manual trigger ────────────────────────────────────────────────────────

    def run_maintenance(
        self,
        adapter: InMemoryAdapter | None = None,
    ) -> list[MaintenanceReport]:
        """
        Run a full maintenance cycle immediately.
        If adapter is provided, runs on that adapter only.
        Otherwise runs on all InMemoryAdapters in the Gateway.
        """
        adapters = self._get_in_memory_adapters(adapter)
        reports  = []
        for a, name in adapters:
            report = self._maintain_adapter(a, name)
            with self._lock:
                self._reports.append(report)
            reports.append(report)
        return reports

    # ── Individual operations (public for direct use) ─────────────────────────

    def compress(
        self,
        adapter:  InMemoryAdapter,
        report:   MaintenanceReport,
    ) -> int:
        """
        Compress clusters of similar records into single representative records.
        Returns number of records removed.
        """
        if not self.cfg.compression_enabled:
            return 0

        records = adapter.all_records()
        if len(records) < self.cfg.min_cluster_size:
            return 0

        removed = 0
        clusters = self._cluster_by_type_and_source(records)

        for cluster in clusters:
            if len(cluster) < self.cfg.min_cluster_size:
                continue
            representative = self._build_representative(cluster)
            if representative is None:
                continue
            # Remove all cluster members
            for r in cluster:
                adapter.delete(r.record_id)
                removed += 1
            # Add representative
            adapter.store(representative)
            removed -= 1   # we added one back
            report.compressed += len(cluster) - 1

        return removed

    def apply_decay(
        self,
        adapter: InMemoryAdapter,
        report:  MaintenanceReport,
    ) -> int:
        """
        Apply temporal decay to records.
        Records at or below confidence floor are archived.
        Returns number of records decayed.
        """
        if not self.cfg.decay_enabled:
            return 0

        records = adapter.all_records()
        decayed = 0
        now     = time.time()

        for record in records:
            if record.is_sensor_anchored:
                continue   # sensor-confirmed records are immune to decay

            half_life = _HALF_LIVES.get(record.memory_type, 7200.0)
            age       = now - record.valid_at

            # Calculate decayed confidence
            import math
            decay_factor   = math.exp(-math.log(2) * age / half_life)
            new_confidence = record.base_confidence * decay_factor

            if new_confidence <= self.cfg.decay_confidence_floor:
                # Archive and remove
                self._archive.append(record)
                adapter.delete(record.record_id)
                report.archived += 1
                decayed         += 1

        report.decayed = decayed
        return decayed

    def apply_forgetting(
        self,
        adapter: InMemoryAdapter,
        report:  MaintenanceReport,
    ) -> int:
        """
        Forget records that have not been used and are past the forgetting window.
        Records with outcome history are retained regardless.
        Returns number of records forgotten.
        """
        if not self.cfg.forgetting_enabled:
            return 0

        records  = adapter.all_records()
        now      = time.time()
        forgotten = 0

        for record in records:
            age = now - record.valid_at

            # Never forget records with outcome history — they are information
            if record.outcome_count > 0:
                continue

            # Never forget sensor-confirmed records
            if record.is_sensor_anchored:
                continue

            # Forget if past the window and never retrieved for outcome
            if age > self.cfg.forgetting_window_s:
                self._archive.append(record)
                adapter.delete(record.record_id)
                report.forgotten += 1
                forgotten        += 1

        return forgotten

    def prioritise(
        self,
        adapter: InMemoryAdapter,
        report:  MaintenanceReport,
    ) -> None:
        """
        Outcome-weighted prioritisation.
        Records with consistently high success rates are tagged SUCCESS.
        Records with consistently low success rates are tagged FAILURE.
        This feeds directly into the Relevance Engine's Axis 4 scoring.
        """
        records = adapter.all_records()
        for record in records:
            if record.outcome_count < 3:
                continue   # not enough data to re-tag

            success_rate = record.success_count / record.outcome_count
            if success_rate >= 0.75 and record.outcome_tag != OutcomeTag.SUCCESS:
                self._retag(adapter, record, OutcomeTag.SUCCESS)
            elif success_rate < 0.30 and record.outcome_tag != OutcomeTag.FAILURE:
                self._retag(adapter, record, OutcomeTag.FAILURE)
            elif 0.30 <= success_rate < 0.75 and record.outcome_tag not in (
                OutcomeTag.PARTIAL, OutcomeTag.UNKNOWN
            ):
                self._retag(adapter, record, OutcomeTag.PARTIAL)

    # ── Archive access ────────────────────────────────────────────────────────

    def archive(self) -> list[MemoryRecord]:
        """Returns all soft-deleted records."""
        return list(self._archive)

    def reports(self) -> list[MaintenanceReport]:
        with self._lock:
            return list(self._reports)

    def last_report(self) -> MaintenanceReport | None:
        with self._lock:
            return self._reports[-1] if self._reports else None

    # ── Internals ─────────────────────────────────────────────────────────────

    def _maintenance_loop(self) -> None:
        while self._running:
            time.sleep(self.cfg.maintenance_interval_s)
            try:
                self.run_maintenance()
            except Exception:
                pass

    def _maintain_adapter(
        self,
        adapter: InMemoryAdapter,
        name:    str,
    ) -> MaintenanceReport:
        t0     = time.perf_counter()
        before = adapter.count()
        report = MaintenanceReport(
            records_before=before,
            adapter_name=name,
        )

        # Run in order: prioritise → decay → compress → forget
        self.prioritise(adapter, report)
        self.apply_decay(adapter, report)

        if adapter.count() > self.cfg.compression_threshold:
            self.compress(adapter, report)

        self.apply_forgetting(adapter, report)

        report.records_after = adapter.count()
        report.latency_ms    = (time.perf_counter() - t0) * 1000.0
        return report

    def _get_in_memory_adapters(
        self,
        single: InMemoryAdapter | None,
    ) -> list[tuple[InMemoryAdapter, str]]:
        if single is not None:
            return [(single, getattr(single, "name", "adapter"))]

        if self.gateway is None:
            return []

        result = []
        for schema in self.gateway.adapters():
            if schema.backend_type == "in_memory":
                # Access the adapter from the gateway's internal entries
                with self.gateway._lock:
                    for entry in self.gateway._adapters:
                        if entry.schema.name == schema.name:
                            if isinstance(entry.adapter, InMemoryAdapter):
                                result.append((entry.adapter, schema.name))
        return result

    @staticmethod
    def _cluster_by_type_and_source(
        records: list[MemoryRecord],
    ) -> list[list[MemoryRecord]]:
        """
        Group records into clusters by (memory_type, source, content_text_prefix).
        Simple bucketing — production uses embedding-based clustering.
        """
        buckets: dict[tuple, list[MemoryRecord]] = {}
        for r in records:
            prefix = r.content_text[:30].lower().strip() if r.content_text else ""
            key    = (r.memory_type, r.source, prefix)
            buckets.setdefault(key, []).append(r)
        return [v for v in buckets.values() if len(v) > 1]

    @staticmethod
    def _build_representative(cluster: list[MemoryRecord]) -> MemoryRecord | None:
        """Build a single representative record from a cluster."""
        if not cluster:
            return None

        # Best confidence as base
        best = max(cluster, key=lambda r: r.effective_confidence)

        # Aggregate outcomes
        total_count    = sum(r.outcome_count for r in cluster)
        total_successes = sum(r.success_count for r in cluster)
        new_tag = (
            OutcomeTag.SUCCESS if total_count > 0 and total_successes / total_count >= 0.70
            else OutcomeTag.FAILURE if total_count > 0 and total_successes / total_count < 0.30
            else OutcomeTag.PARTIAL if total_count > 0
            else best.outcome_tag
        )

        # Earliest valid_at as provenance
        earliest_valid = min(r.valid_at for r in cluster)

        return MemoryRecord(
            source=best.source,
            memory_type=best.memory_type,
            content=best.content,
            content_text=best.content_text,
            valid_at=earliest_valid,
            invalid_at=best.invalid_at,
            base_confidence=best.base_confidence,
            sensor_anchor=best.sensor_anchor,
            sensor_confirmed_at=best.sensor_confirmed_at,
            outcome_tag=new_tag,
            outcome_count=total_count,
            success_count=total_successes,
        )

    @staticmethod
    def _retag(
        adapter: InMemoryAdapter,
        record:  MemoryRecord,
        new_tag: OutcomeTag,
    ) -> None:
        """Replace a record's outcome tag in the adapter."""
        updated = MemoryRecord(
            record_id=record.record_id,
            source=record.source,
            memory_type=record.memory_type,
            content=record.content,
            content_text=record.content_text,
            valid_at=record.valid_at,
            invalid_at=record.invalid_at,
            base_confidence=record.base_confidence,
            sensor_anchor=record.sensor_anchor,
            sensor_confirmed_at=record.sensor_confirmed_at,
            outcome_tag=new_tag,
            outcome_count=record.outcome_count,
            success_count=record.success_count,
        )
        adapter.delete(record.record_id)
        adapter.store(updated)
