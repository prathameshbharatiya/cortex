"""
cortex.config
=============
Configuration loading for the Cortex stack.

    from cortex.config import get_settings, CortexSettings

    cfg = get_settings()                    # load once, cached
    cfg = CortexSettings.load()             # explicit load (respects file + env)
    cfg = CortexSettings.from_yaml("...")   # from a YAML string
    cfg = CortexSettings.from_file("/etc/cortex/cortex.yaml")

    print(cfg.display())                    # redacted startup summary
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from cortex.config.settings import (
    CortexSettings,
    ServerSettings,
    GrpcSettings,
    HttpSettings,
    GateSettings,
    SentinelSettings,
    PhysicoreSettings,
    ContextSettings,
    RedisSettings,
    WeaviateSettings,
    Mem0Settings,
    PersistenceSettings,
    LifecycleSettings,
    MetricsSettings,
    TracingSettings,
)

__all__ = [
    "CortexSettings",
    "ServerSettings",
    "GrpcSettings",
    "HttpSettings",
    "GateSettings",
    "SentinelSettings",
    "PhysicoreSettings",
    "ContextSettings",
    "RedisSettings",
    "WeaviateSettings",
    "Mem0Settings",
    "PersistenceSettings",
    "LifecycleSettings",
    "MetricsSettings",
    "TracingSettings",
    "get_settings",
    "reset_settings",
]

# ── global singleton ──────────────────────────────────────────────────────────
_lock:     threading.Lock           = threading.Lock()
_settings: CortexSettings | None   = None


def get_settings() -> CortexSettings:
    """
    Return the global CortexSettings singleton.

    Loaded once on first call using the full priority chain:
      CORTEX_CONFIG_FILE → auto-discovered cortex.yaml → env vars → defaults

    Thread-safe.

    Example:
        from cortex.config import get_settings
        cfg = get_settings()
        print(cfg.server.platform_id)
        print(cfg.redis.url)
    """
    global _settings
    if _settings is None:
        with _lock:
            if _settings is None:
                _settings = CortexSettings.load()
    return _settings


def reset_settings(new: CortexSettings | None = None) -> None:
    """
    Reset the singleton — primarily for testing.

    Example:
        from cortex.config import reset_settings, CortexSettings
        reset_settings(CortexSettings(redis={"enabled": True, "url": "redis://test:6379"}))
    """
    global _settings
    with _lock:
        _settings = new
