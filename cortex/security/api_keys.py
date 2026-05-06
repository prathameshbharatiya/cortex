"""
cortex.security.api_keys
========================
API key authentication for Cortex transports.

Keys are stored as SHA-256 hashes server-side — the raw key is never saved.
Every key carries a robot_id (which platform it belongs to) and a set of
permissions (which operations it may call).

Key format:  cx_live_<43-char-urlsafe-base64>
             cx_dev_<43-char-urlsafe-base64>   (local development only)

Environment variables
---------------------
CORTEX_API_KEYS
    Comma-separated list of raw API keys that are valid.
    e.g. "cx_live_abc123,cx_live_xyz789"

CORTEX_SKIP_AUTH
    Set to "1" to disable authentication entirely (local dev only).
    Never set this in production.

CORTEX_AUTH_REQUIRED
    Set to "1" to force authentication even when no keys are configured.
    Use this to ensure a misconfigured server fails closed instead of open.

Permissions
-----------
Each key is assigned a set of permissions:
    certify     — call /certify and the gRPC Certify RPC
    record      — call /record (write experience outcomes)
    memory      — call /memory (read/write memory records)
    context     — call /context (read context packets)
    health      — call /health (unauthenticated by default)
    admin       — all of the above

Usage — HTTP (FastAPI dependency)
----------------------------------
    from cortex.security.api_keys import require_permission
    from fastapi import Depends

    @app.post("/certify")
    async def certify(body: ..., _key=Depends(require_permission("certify"))):
        ...

Usage — gRPC interceptor
------------------------
    from cortex.security.api_keys import ApiKeyInterceptor
    server = grpc.server(..., interceptors=[ApiKeyInterceptor()])

Usage — generate a new key
---------------------------
    cortex-keygen
    # or:
    python -m cortex.security.api_keys
"""

from __future__ import annotations

import hashlib
import logging
import os
import secrets
import time
from dataclasses import dataclass, field
from typing import Optional, Set

log = logging.getLogger(__name__)

# ── Key prefix constants ──────────────────────────────────────────────────────
PREFIX_LIVE = "cx_live"
PREFIX_DEV  = "cx_dev"

# ── All valid permissions ─────────────────────────────────────────────────────
ALL_PERMISSIONS: Set[str] = {"certify", "record", "memory", "context", "health", "admin"}


# ── Key record ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ApiKeyRecord:
    """
    A validated API key and the claims it carries.
    Returned by authenticate() on success — passed to authorise().
    """
    key_hash:    str
    robot_id:    str          = ""       # which platform this key belongs to
    permissions: Set[str]     = field(default_factory=lambda: {"certify", "health"})
    created_at:  float        = field(default_factory=time.time)

    def has(self, permission: str) -> bool:
        return "admin" in self.permissions or permission in self.permissions


# ── Key store ─────────────────────────────────────────────────────────────────

class ApiKeyStore:
    """
    In-memory API key store loaded from environment variables.

    Production deployments inject keys via Kubernetes Secrets → env vars.
    The store is reload-safe: call reload() to pick up rotated keys without
    restarting the server.
    """

    def __init__(self) -> None:
        self._hashes: dict[str, ApiKeyRecord] = {}
        self._skip_auth:     bool = False
        self._auth_required: bool = False
        self.reload()

    def reload(self) -> None:
        """Reload key configuration from environment variables."""
        self._skip_auth     = os.environ.get("CORTEX_SKIP_AUTH",     "0") == "1"
        self._auth_required = os.environ.get("CORTEX_AUTH_REQUIRED", "0") == "1"
        self._hashes        = {}

        raw_keys = os.environ.get("CORTEX_API_KEYS", "")
        if raw_keys:
            for entry in raw_keys.split(","):
                entry = entry.strip()
                if not entry:
                    continue
                # Format: raw_key[:robot_id[:perm1+perm2]]
                # Split from the RIGHT so the key (which may contain '-' but not ':')
                # is preserved. Metadata is always the last 0, 1, or 2 colon-segments.
                parts = entry.rsplit(":", 2)
                raw_key   = parts[0].strip()
                robot_id  = parts[1].strip() if len(parts) > 1 else ""
                perms_str = parts[2].strip() if len(parts) > 2 else "certify+health"
                if not raw_key:
                    continue
                key_hash    = _sha256(raw_key)
                permissions = set(perms_str.split("+")) & ALL_PERMISSIONS
                if not permissions:
                    permissions = {"certify", "health"}
                self._hashes[key_hash] = ApiKeyRecord(
                    key_hash=key_hash,
                    robot_id=robot_id,
                    permissions=permissions,
                )
            log.info("Loaded %d API key(s)", len(self._hashes))
        else:
            if not self._skip_auth:
                if self._auth_required:
                    log.error(
                        "CORTEX_AUTH_REQUIRED=1 but CORTEX_API_KEYS is empty — "
                        "server will reject all requests"
                    )
                else:
                    log.warning(
                        "CORTEX_API_KEYS not configured and CORTEX_SKIP_AUTH != 1. "
                        "The server is UNAUTHENTICATED. Set CORTEX_API_KEYS or "
                        "CORTEX_SKIP_AUTH=1 (local dev only)."
                    )

    def authenticate(self, raw_key: str) -> ApiKeyRecord | None:
        """
        Validate a raw API key. Returns an ApiKeyRecord on success, None on failure.
        Constant-time comparison to prevent timing attacks.
        """
        if self._skip_auth:
            return ApiKeyRecord(key_hash="dev", robot_id="dev", permissions=ALL_PERMISSIONS)

        if not raw_key:
            return None

        key_hash    = _sha256(raw_key)
        stored_hash = next(iter(self._hashes.keys()), None)  # any key for timing

        # Constant-time comparison across all stored hashes
        match: ApiKeyRecord | None = None
        for stored, record in self._hashes.items():
            if secrets.compare_digest(stored, key_hash):
                match = record

        if match is None:
            log.warning("Authentication failed: invalid API key (hash prefix: %s...)", key_hash[:8])
        return match

    def is_skip_auth(self) -> bool:
        return self._skip_auth

    def is_auth_required(self) -> bool:
        return self._auth_required

    def has_keys(self) -> bool:
        return bool(self._hashes)


# ── Module-level singleton ────────────────────────────────────────────────────

_store: ApiKeyStore | None = None


def get_store() -> ApiKeyStore:
    global _store
    if _store is None:
        _store = ApiKeyStore()
    return _store


def reset_store() -> None:
    """Force a fresh reload — used in tests and after key rotation."""
    global _store
    _store = None


# ── FastAPI dependency ────────────────────────────────────────────────────────

def require_permission(permission: str):
    """
    FastAPI dependency factory. Returns a dependency that validates the API key
    and checks it has the required permission.

    Usage:
        @app.post("/certify")
        async def certify(body: ..., _=Depends(require_permission("certify"))):
            ...
    """
    from fastapi import Header, HTTPException, status as http_status

    async def _dep(
        x_api_key:     Optional[str] = Header(None, alias="X-API-Key"),
        authorization: Optional[str] = Header(None),
    ) -> ApiKeyRecord:
        store = get_store()

        if store.is_skip_auth():
            return ApiKeyRecord(key_hash="dev", robot_id="dev", permissions=ALL_PERMISSIONS)

        # No keys configured and not required → unauthenticated pass-through with warning
        if not store.has_keys() and not store.is_auth_required():
            return ApiKeyRecord(key_hash="unconfigured", permissions=ALL_PERMISSIONS)

        # Extract key from headers
        raw_key: Optional[str] = None
        if x_api_key:
            raw_key = x_api_key.strip()
        elif authorization and authorization.lower().startswith("bearer "):
            raw_key = authorization[7:].strip()

        if not raw_key:
            raise HTTPException(
                status_code=http_status.HTTP_401_UNAUTHORIZED,
                detail=(
                    "Authentication required. "
                    "Provide your API key via X-API-Key header or "
                    "Authorization: Bearer <key>."
                ),
                headers={"WWW-Authenticate": "Bearer"},
            )

        record = store.authenticate(raw_key)
        if record is None:
            raise HTTPException(
                status_code=http_status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key.",
                headers={"WWW-Authenticate": "Bearer"},
            )

        if not record.has(permission):
            raise HTTPException(
                status_code=http_status.HTTP_403_FORBIDDEN,
                detail=f"API key does not have '{permission}' permission.",
            )

        return record

    return _dep


# ── gRPC interceptor ──────────────────────────────────────────────────────────

class ApiKeyInterceptor:
    """
    gRPC server interceptor that validates API keys on every RPC call.

    The key is read from the gRPC metadata key "x-api-key".
    Health RPCs are always allowed through without a key.

    Usage:
        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=10),
            interceptors=[ApiKeyInterceptor()],
        )
    """

    # RPCs that don't require authentication
    _PUBLIC_RPCS = frozenset({"/cortex.CortexHealth/Health"})

    # RPC → permission mapping
    _RPC_PERMISSIONS: dict[str, str] = {
        "/cortex.CortexCertification/Certify":  "certify",
        "/cortex.CortexMemory/Retrieve":        "memory",
        "/cortex.CortexMemory/Store":           "memory",
        "/cortex.CortexExperience/Record":      "record",
        "/cortex.CortexHealth/Health":          "health",
    }

    def intercept_service(self, continuation, handler_call_details):
        """Called for each incoming RPC."""
        method = handler_call_details.method

        # Public RPCs pass through unconditionally
        if method in self._PUBLIC_RPCS:
            return continuation(handler_call_details)

        store = get_store()
        if store.is_skip_auth():
            return continuation(handler_call_details)

        if not store.has_keys() and not store.is_auth_required():
            return continuation(handler_call_details)

        # Extract key from metadata
        metadata = dict(handler_call_details.invocation_metadata or [])
        raw_key  = metadata.get("x-api-key", "").strip()

        record = store.authenticate(raw_key)
        if record is None:
            import grpc as _grpc
            def _abort(request, context):
                context.abort(
                    _grpc.StatusCode.UNAUTHENTICATED,
                    "API key required. Set x-api-key metadata.",
                )
            return _abort

        required_perm = self._RPC_PERMISSIONS.get(method, "certify")
        if not record.has(required_perm):
            import grpc as _grpc
            def _forbid(request, context):
                context.abort(
                    _grpc.StatusCode.PERMISSION_DENIED,
                    f"API key does not have '{required_perm}' permission.",
                )
            return _forbid

        return continuation(handler_call_details)


# ── Key generation ────────────────────────────────────────────────────────────

def generate_api_key(prefix: str = PREFIX_LIVE) -> str:
    """
    Generate a cryptographically secure Cortex API key.

    Example:
        from cortex.security.api_keys import generate_api_key
        key = generate_api_key()
        # → cx_live_Xk3mN9...
        # Add to CORTEX_API_KEYS on your server
    """
    return f"{prefix}_{secrets.token_urlsafe(32)}"


def hash_key(raw_key: str) -> str:
    """Return the SHA-256 hex digest of a key — store this, not the raw key."""
    return _sha256(raw_key)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


# ── CLI entrypoint ────────────────────────────────────────────────────────────

def main() -> None:
    """
    cortex-keygen — generate a new API key and print setup instructions.
    """
    import argparse
    parser = argparse.ArgumentParser(
        prog="cortex-keygen",
        description="Generate a Cortex API key.",
    )
    parser.add_argument("--dev",      action="store_true", help="Generate a dev key (cx_dev_ prefix)")
    parser.add_argument("--robot-id", default="",          help="Robot/platform ID to embed in key metadata")
    parser.add_argument("--perms",    default="certify+health", help="Permissions: certify+record+memory+context+admin")
    args = parser.parse_args()

    prefix = PREFIX_DEV if args.dev else PREFIX_LIVE
    key    = generate_api_key(prefix=prefix)

    robot_suffix = f":{args.robot_id}:{args.perms}" if args.robot_id else ""
    env_value    = f"{key}{robot_suffix}"
    key_hash     = hash_key(key)

    print(f"\n{'='*60}")
    print(f"  New Cortex API key")
    print(f"{'='*60}")
    print(f"\n  Key:       {key}")
    print(f"  Hash:      {key_hash[:16]}... (SHA-256)")
    if args.robot_id:
        print(f"  Robot ID:  {args.robot_id}")
    print(f"  Perms:     {args.perms}")
    print(f"\n  Add to your server environment:")
    print(f"\n    export CORTEX_API_KEYS=\"{env_value}\"")
    print(f"\n  For multiple keys (comma-separated):")
    print(f"\n    export CORTEX_API_KEYS=\"{env_value},cx_live_<other_key>\"")
    print(f"\n  The raw key is shown ONCE. Store it securely.")
    print(f"  The server stores only the SHA-256 hash.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
