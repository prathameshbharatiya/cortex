"""
cortex.migrations
=================
Schema versioning and migration tooling for Cortex persistence.

    from cortex.migrations import CURRENT_VERSION, run_migration, MigrationReport

    # Dry run — safe, no writes
    from cortex.memory.adapters.json_file import JSONFileAdapter
    adapter = JSONFileAdapter(file_path="./cortex_memory.json")
    adapter.connect()
    report = run_migration(adapter, dry_run=True)
    print(report)

    # Actual migration
    report = run_migration(adapter, dry_run=False)
    if not report.success:
        raise RuntimeError(str(report))

CLI:
    cortex-migrate --adapter json --path ./cortex_memory.json --dry-run
    cortex-migrate --adapter redis --url redis://localhost:6379
"""

from cortex.migrations.schema import (
    CURRENT_VERSION,
    VERSION_HISTORY,
    SCHEMA_MANIFEST,
    migrate_record,
    detect_version,
)
from cortex.migrations.runner import (
    run_migration,
    MigrationReport,
    RecordMigration,
    MigrationError,
)

__all__ = [
    "CURRENT_VERSION",
    "VERSION_HISTORY",
    "SCHEMA_MANIFEST",
    "migrate_record",
    "detect_version",
    "run_migration",
    "MigrationReport",
    "RecordMigration",
    "MigrationError",
    "main",
]


def main() -> None:
    """
    cortex-migrate — run schema migrations on a Cortex memory adapter.

    Examples:
        cortex-migrate --adapter json --path ./cortex_memory.json --dry-run
        cortex-migrate --adapter json --path ./cortex_memory.json
        cortex-migrate --adapter redis --url redis://localhost:6379
        cortex-migrate --status
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="cortex-migrate",
        description="Run Cortex memory schema migrations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  cortex-migrate --status
  cortex-migrate --adapter json --path ./cortex_memory.json --dry-run
  cortex-migrate --adapter json --path ./cortex_memory.json
  cortex-migrate --adapter redis --url redis://localhost:6379 --dry-run
  cortex-migrate --adapter redis --url redis://localhost:6379
        """,
    )
    parser.add_argument(
        "--adapter", choices=["json", "redis", "memory"],
        default="json",
        help="Adapter to migrate (default: json)",
    )
    parser.add_argument(
        "--path", default="./cortex_memory.json",
        help="JSON file path (for --adapter json)",
    )
    parser.add_argument(
        "--url", default="redis://localhost:6379",
        help="Redis URL (for --adapter redis)",
    )
    parser.add_argument(
        "--dry-run", "-n", action="store_true",
        help="Scan and report — do not write any changes",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Print current schema version and exit",
    )
    args = parser.parse_args()

    if args.status:
        print(f"\nCortex schema versions:")
        for i, v in enumerate(VERSION_HISTORY):
            current = " ← current" if v == CURRENT_VERSION else ""
            fields  = len(SCHEMA_MANIFEST.get(v, set()))
            print(f"  {v}  ({fields} fields){current}")
        print()
        return

    # Build adapter
    adapter = None
    if args.adapter == "json":
        from cortex.memory.adapters.json_file import JSONFileAdapter
        adapter = JSONFileAdapter(file_path=args.path, auto_flush=False)
        adapter.connect()
    elif args.adapter == "redis":
        from cortex.memory.adapters.redis_adapter import RedisAdapter
        adapter = RedisAdapter(url=args.url)
        adapter.connect()
    elif args.adapter == "memory":
        from cortex.memory.adapters.in_memory import InMemoryAdapter
        adapter = InMemoryAdapter()

    if adapter is None:
        print("Error: could not build adapter", file=sys.stderr)
        sys.exit(1)

    print(f"\ncortex-migrate  adapter={args.adapter}  dry_run={args.dry_run}")
    print(f"  Target schema version: {CURRENT_VERSION}")
    if args.dry_run:
        print("  DRY RUN — no changes will be written\n")
    else:
        print()

    try:
        report = run_migration(adapter, dry_run=args.dry_run)
        print(report)
        if hasattr(adapter, "disconnect"):
            adapter.disconnect()
        sys.exit(0)
    except MigrationError as exc:
        print(exc.report)
        print(f"\nMigration FAILED — no changes were written.", file=sys.stderr)
        sys.exit(1)
