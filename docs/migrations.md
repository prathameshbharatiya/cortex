# Schema migrations

Cortex stores `MemoryRecord` objects in Redis, Weaviate, and JSON files. When a new version of Cortex adds or renames fields, records written by the old version need to be upgraded before the new code can read them reliably.

Cortex handles this automatically for reads (records are upgraded on the fly when loaded) and provides a CLI tool for bulk migration of existing stores.

---

## Schema version history

```
1.0   Initial release — 12 fields
1.1   Added content_text (str, default "")
1.2   Added retrieval_latency_ms (float, default 0.0)   ← current
```

```bash
cortex-migrate --status
```

```
Cortex schema versions:
  1.0  (12 fields)
  1.1  (13 fields)
  1.2  (14 fields) ← current
```

---

## Automatic migration on read

Records written by older Cortex versions are upgraded automatically when loaded. You do not need to run the migration CLI for normal operation — it is for explicit bulk upgrades of existing stores before deploying a new version.

When a 1.0 record is read from Redis or a JSON file, Cortex:
1. Detects its version from the fields present.
2. Runs it through the migration chain (1.0 → 1.1 → 1.2).
3. Returns a fully valid `MemoryRecord` for the current version.
4. Does **not** write the upgraded version back to the store automatically.

This means a mixed cluster (some nodes on 1.0, some on 1.2) works safely — but the store fills up with old-format records over time. Run the migration CLI to upgrade the store and remove the per-read overhead.

---

## Running migrations

### Step 1: dry run (always do this first)

```bash
# JSON file
cortex-migrate --adapter json --path ./cortex_memory.json --dry-run

# Redis
cortex-migrate --adapter redis --url redis://localhost:6379 --dry-run
```

The dry run scans every record, shows how many need upgrading, and exits without writing anything. It is safe to run in production.

```
cortex-migrate  adapter=redis  dry_run=True
  Target schema version: 1.2
  DRY RUN — no changes will be written

Migration report [DRY RUN] — redis
  Duration      : 0.12s
  Scanned       : 14820
  Already current: 14820
  Migrated      : 0
  Failed        : 0
  Skipped       : 0
```

### Step 2: run the migration

```bash
cortex-migrate --adapter json --path ./cortex_memory.json
cortex-migrate --adapter redis --url redis://localhost:6379
```

Safety guarantees:
- **JSON file**: a `.bak` copy is made atomically before any records are written.
- **Rollback**: if any record fails to migrate, no records are written at all.
- **Idempotent**: running migration twice produces the same result.

### Step 3: verify

```bash
cortex-migrate --adapter redis --url redis://localhost:6379 --dry-run
# Should show: Already current: <total>  Migrated: 0
```

---

## From Python

```python
from cortex.migrations import run_migration, MigrationError, CURRENT_VERSION
from cortex.memory.adapters.redis_adapter import RedisAdapter

adapter = RedisAdapter(url="redis://localhost:6379")
adapter.connect()

# Dry run
report = run_migration(adapter, dry_run=True)
print(report)

# Actual migration
try:
    report = run_migration(adapter, dry_run=False)
    print(f"Migrated {report.migrated} records in {report.duration_s:.2f}s")
except MigrationError as exc:
    print(exc.report)
    # No records were written — the store is untouched
```

---

## Deploying a new Cortex version

Standard procedure for every version bump:

```bash
# 1. Back up your stores (belt-and-suspenders)
redis-cli BGSAVE

# 2. Dry run against production
REDIS_URL=redis://prod-redis:6379 \
cortex-migrate --adapter redis --url $REDIS_URL --dry-run

# 3. Run migration (with old Cortex still running — safe)
cortex-migrate --adapter redis --url $REDIS_URL

# 4. Deploy the new Cortex version
kubectl set image deployment/cortex-grpc cortex-grpc=cortex:0.1.1 -n cortex
kubectl rollout status deployment/cortex-grpc -n cortex

# 5. Verify
kubectl exec -n cortex deploy/cortex-grpc -- cortex-migrate --status
```

---

## Writing a migration for a new field

When you add a field to `MemoryRecord`:

1. Bump the version in `cortex/migrations/schema.py`:

```python
CURRENT_VERSION: str = "1.3"
VERSION_HISTORY: list[str] = ["1.0", "1.1", "1.2", "1.3"]
```

2. Add the new field to the manifest:

```python
SCHEMA_MANIFEST["1.3"] = SCHEMA_MANIFEST["1.2"] | {"my_new_field"}
```

3. Add the fingerprint:

```python
_VERSION_FINGERPRINTS["1.3"] = {"content_text", "retrieval_latency_ms", "my_new_field"}
```

4. Write the migration function:

```python
def _migrate_1_2_to_1_3(record: dict) -> dict:
    record = dict(record)
    record.setdefault("my_new_field", "default_value")
    return record
```

5. Register it:

```python
MIGRATION_STEPS[("1.2", "1.3")] = _migrate_1_2_to_1_3
```

6. Update `CHANGELOG.md`.

That's it. The runner handles the rest.
