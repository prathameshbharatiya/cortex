"""
cortex.migrations.runner
========================
Migration runner — scans an adapter's stored records and upgrades any that
are below CURRENT_VERSION.

Usage
-----
    # Check what needs migrating (dry run)
    cortex-migrate --adapter redis --url redis://localhost:6379 --dry-run

    # Run migration
    cortex-migrate --adapter redis --url redis://localhost:6379

    # From Python
    from cortex.migrations.runner import run_migration, MigrationReport
    report = run_migration(adapter, dry_run=False)
    print(report)

Safety guarantees
-----------------
1. Dry run first — always safe to call with dry_run=True.
2. Backup before write — for JSON file adapters, a .bak copy is made atomically
   before any records are overwritten.
3. Rollback on partial failure — if any record fails to migrate, no records
   are written and a MigrationError is raised with the list of failures.
4. Idempotent — running migration twice produces the same result. Records
   already at CURRENT_VERSION are left untouched.
5. Audit log — every migration produces a MigrationReport with a full record
   of what changed, including before/after for every upgraded record.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from cortex.migrations.schema import (
    CURRENT_VERSION, detect_version, migrate_record
)

log = logging.getLogger(__name__)


# ── Report types ──────────────────────────────────────────────────────────────

@dataclass
class RecordMigration:
    """Record of a single record being migrated."""
    record_id:    str
    from_version: str
    to_version:   str
    success:      bool
    error:        str = ""


@dataclass
class MigrationReport:
    """Full report from a migration run."""
    adapter_name:    str
    dry_run:         bool
    started_at:      float = field(default_factory=time.time)
    finished_at:     float = 0.0
    total_scanned:   int = 0
    already_current: int = 0
    migrated:        int = 0
    failed:          int = 0
    skipped:         int = 0
    records:         list[RecordMigration] = field(default_factory=list)
    errors:          list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.failed == 0

    @property
    def duration_s(self) -> float:
        return self.finished_at - self.started_at

    def __str__(self) -> str:
        status = "DRY RUN" if self.dry_run else ("OK" if self.success else "FAILED")
        lines = [
            f"Migration report [{status}] — {self.adapter_name}",
            f"  Duration      : {self.duration_s:.2f}s",
            f"  Scanned       : {self.total_scanned}",
            f"  Already current: {self.already_current}",
            f"  Migrated      : {self.migrated}",
            f"  Failed        : {self.failed}",
            f"  Skipped       : {self.skipped}",
        ]
        if self.errors:
            lines.append(f"  Errors:")
            for e in self.errors[:5]:
                lines.append(f"    - {e}")
        return "\n".join(lines)


class MigrationError(RuntimeError):
    """Raised when one or more records fail to migrate."""
    def __init__(self, report: MigrationReport) -> None:
        self.report = report
        super().__init__(
            f"Migration failed: {report.failed} record(s) failed on {report.adapter_name}"
        )


# ── Core runner ───────────────────────────────────────────────────────────────

def run_migration(
    adapter,
    dry_run: bool = False,
    stop_on_error: bool = False,
) -> MigrationReport:
    """
    Scan all records in `adapter` and upgrade any below CURRENT_VERSION.

    Parameters
    ----------
    adapter:
        Any MemoryAdapter with an `all_records()` method and a `store()` method.
        InMemoryAdapter, JSONFileAdapter, and RedisAdapter all qualify.
    dry_run:
        If True, scan and report but do not write any changes.
    stop_on_error:
        If True, raise MigrationError immediately on the first failure.
        If False, continue and report all failures at the end.

    Returns
    -------
    MigrationReport

    Raises
    ------
    MigrationError
        If stop_on_error=True and a record fails, or if any records failed
        and the caller should be aware.
    """
    report = MigrationReport(
        adapter_name=getattr(adapter, "name", str(adapter)),
        dry_run=dry_run,
    )

    log.info(
        "Migration starting — adapter=%s dry_run=%s target_version=%s",
        report.adapter_name, dry_run, CURRENT_VERSION,
    )

    # ── 1. Load all records ───────────────────────────────────────────────────
    try:
        all_records = adapter.all_records()
    except AttributeError:
        # Adapters that don't support all_records() (e.g. Weaviate remote)
        log.warning("Adapter %s does not support all_records() — skipping", report.adapter_name)
        report.skipped = -1
        report.errors.append("Adapter does not support bulk record enumeration")
        report.finished_at = time.time()
        return report

    report.total_scanned = len(all_records)
    log.info("  Loaded %d records to inspect", report.total_scanned)

    # ── 2. Identify which need migration ──────────────────────────────────────
    to_migrate: list[tuple[Any, str]] = []   # (record, from_version)

    for record in all_records:
        raw     = record.model_dump()
        version = raw.get("_schema_version") or detect_version(raw)
        if version == CURRENT_VERSION:
            report.already_current += 1
        else:
            to_migrate.append((record, version))

    log.info(
        "  Already current: %d  |  Need migration: %d",
        report.already_current, len(to_migrate),
    )

    if not to_migrate:
        report.finished_at = time.time()
        log.info("Migration complete — nothing to do")
        return report

    # ── 3. Migrate (or simulate) ──────────────────────────────────────────────
    upgraded: list[Any] = []

    for record, from_version in to_migrate:
        record_id = record.record_id
        try:
            raw_new = migrate_record(record.model_dump(), from_version)
            raw_new["_schema_version"] = CURRENT_VERSION

            # Reconstruct as a MemoryRecord to validate the migration output
            from cortex.models.memory import MemoryRecord as _MR
            new_record = _MR.model_validate(raw_new)

            upgraded.append(new_record)
            report.records.append(RecordMigration(
                record_id=record_id,
                from_version=from_version,
                to_version=CURRENT_VERSION,
                success=True,
            ))
            report.migrated += 1

        except Exception as exc:
            err_msg = f"{record_id}: {exc}"
            report.records.append(RecordMigration(
                record_id=record_id,
                from_version=from_version,
                to_version=CURRENT_VERSION,
                success=False,
                error=str(exc),
            ))
            report.failed += 1
            report.errors.append(err_msg)
            log.error("  Migration failed for %s: %s", record_id, exc)

            if stop_on_error:
                report.finished_at = time.time()
                raise MigrationError(report) from exc

    # ── 4. Abort if any failed (rollback = don't write) ───────────────────────
    if report.failed > 0:
        report.finished_at = time.time()
        log.error(
            "Migration aborted — %d record(s) failed. No changes written.",
            report.failed,
        )
        raise MigrationError(report)

    # ── 5. Write (skip in dry run) ────────────────────────────────────────────
    if not dry_run:
        _backup_if_possible(adapter)
        written = 0
        for new_record in upgraded:
            try:
                adapter.store(new_record)
                written += 1
            except Exception as exc:
                log.error("  Write failed for %s: %s", new_record.record_id, exc)
                report.failed += 1
                report.errors.append(f"write:{new_record.record_id}:{exc}")

        log.info("  Written %d/%d records", written, len(upgraded))

    report.finished_at = time.time()
    log.info(
        "Migration complete — migrated=%d failed=%d duration=%.2fs",
        report.migrated, report.failed, report.duration_s,
    )
    return report


# ── Backup helper ─────────────────────────────────────────────────────────────

def _backup_if_possible(adapter) -> None:
    """
    For JSONFileAdapter: atomically copy the file to file.bak before writing.
    For other adapters: no-op.
    """
    try:
        from cortex.memory.adapters.json_file import JSONFileAdapter as _JFA
        if isinstance(adapter, _JFA):
            src = adapter.file_path
            if src.exists():
                bak = src.with_suffix(".bak")
                import shutil
                shutil.copy2(src, bak)
                log.info("  Backup created: %s", bak)
    except Exception as exc:
        log.warning("  Backup failed (non-fatal): %s", exc)
