"""
cortex.migrations.schema
========================
Schema version registry for Cortex persistence.

Every time the MemoryRecord model gains, removes, or renames a field, a new
schema version is registered here. Migrations are functions that transform a
raw record dict from version N to version N+1.

Current schema versions
------------------------
1.0   Initial release — 12 fields
1.1   Added `content_text` (str, default "")
1.2   Added `retrieval_latency_ms` (float, default 0.0)
2.0   Renamed `sensor_confirmed_at` → kept, added `platform_id` (str, default "")

The CURRENT_VERSION constant is the version that running code writes.
Any record with a lower version is automatically upgraded on read.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

# ── Version constants ─────────────────────────────────────────────────────────

CURRENT_VERSION: str = "1.2"

# Ordered list of all versions ever released
VERSION_HISTORY: list[str] = ["1.0", "1.1", "1.2"]


# ── Schema manifest (fields in each version) ──────────────────────────────────
# Used to detect records written by older code and to validate migration output.

SCHEMA_MANIFEST: dict[str, set[str]] = {
    "1.0": {
        "record_id", "source", "memory_type", "content",
        "valid_at", "invalid_at", "base_confidence",
        "sensor_anchor", "sensor_confirmed_at",
        "outcome_tag", "outcome_count", "success_count",
    },
    "1.1": {
        "record_id", "source", "memory_type", "content", "content_text",
        "valid_at", "invalid_at", "base_confidence",
        "sensor_anchor", "sensor_confirmed_at",
        "outcome_tag", "outcome_count", "success_count",
    },
    "1.2": {
        "record_id", "source", "memory_type", "content", "content_text",
        "valid_at", "invalid_at", "base_confidence",
        "sensor_anchor", "sensor_confirmed_at",
        "outcome_tag", "outcome_count", "success_count",
        "retrieval_latency_ms",
    },
}


# ── Migration functions ────────────────────────────────────────────────────────
# Each function takes a raw dict and returns an upgraded raw dict.
# Functions must be pure and must not raise on any valid input for their source version.

def _migrate_1_0_to_1_1(record: dict[str, Any]) -> dict[str, Any]:
    """Add content_text field (default empty string)."""
    record = dict(record)
    record.setdefault("content_text", "")
    return record


def _migrate_1_1_to_1_2(record: dict[str, Any]) -> dict[str, Any]:
    """Add retrieval_latency_ms field (default 0.0)."""
    record = dict(record)
    record.setdefault("retrieval_latency_ms", 0.0)
    return record


# ── Migration chain ────────────────────────────────────────────────────────────
# Maps (from_version, to_version) → migration function.
# Only adjacent versions are listed — the runner chains them automatically.

MIGRATION_STEPS: dict[tuple[str, str], Callable[[dict], dict]] = {
    ("1.0", "1.1"): _migrate_1_0_to_1_1,
    ("1.1", "1.2"): _migrate_1_1_to_1_2,
}


# ── Migration runner ───────────────────────────────────────────────────────────

def migrate_record(record: dict[str, Any], from_version: str) -> dict[str, Any]:
    """
    Upgrade a raw record dict from `from_version` to CURRENT_VERSION.

    Applies migrations step-by-step through the version chain.
    Returns the upgraded dict. Does not modify the input.

    Raises ValueError if from_version is unknown.
    """
    if from_version not in VERSION_HISTORY:
        raise ValueError(
            f"Unknown schema version {from_version!r}. "
            f"Known versions: {VERSION_HISTORY}"
        )

    if from_version == CURRENT_VERSION:
        return record

    versions = VERSION_HISTORY
    start    = versions.index(from_version)
    end      = versions.index(CURRENT_VERSION)

    result = dict(record)
    for i in range(start, end):
        step = (versions[i], versions[i + 1])
        fn   = MIGRATION_STEPS.get(step)
        if fn is None:
            raise RuntimeError(
                f"No migration registered for {step[0]} → {step[1]}"
            )
        result = fn(result)

    return result


# ── Version detection fingerprints ────────────────────────────────────────────
# For each version: the set of fields that ARE present but NOT in earlier versions.
# A record is identified as version V if it has all the distinguishing fields for V.

_VERSION_FINGERPRINTS: dict[str, set[str]] = {
    "1.0": set(),                               # baseline — no distinguishing fields
    "1.1": {"content_text"},                    # added in 1.1
    "1.2": {"content_text", "retrieval_latency_ms"},  # added in 1.2
}


def detect_version(record: dict[str, Any]) -> str:
    """
    Infer the schema version of a raw record dict from its fields.

    Used when loading records that have no explicit `_schema_version` key
    (all records written before schema versioning was introduced).

    Returns the highest version whose distinguishing fields are all present.
    """
    keys = set(record.keys()) - {"_schema_version"}
    # Walk versions from newest to oldest — return the first that fits
    for version in reversed(VERSION_HISTORY):
        fingerprint = _VERSION_FINGERPRINTS.get(version, set())
        if fingerprint.issubset(keys):
            return version
    return VERSION_HISTORY[0]
