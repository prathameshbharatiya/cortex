"""
cortex.config.factory
=====================
Builds a fully wired CortexServer from a CortexSettings object.

This is the ONLY place where settings values are translated into
component constructor arguments. No other file should read settings
and pass them manually into components.

Usage
-----
    from cortex.config import get_settings
    from cortex.config.factory import build_server

    cfg    = get_settings()
    server = build_server(cfg)
    server.start()

The gRPC and HTTP server entry points call this automatically.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from cortex.config.settings import CortexSettings
from cortex.gate.certification  import CertificationGate
from cortex.context.engine      import ContextEngine
from cortex.memory.gateway      import MemoryGateway
from cortex.memory.adapters     import InMemoryAdapter, JSONFileAdapter, RedisAdapter
from cortex.experience.tracker  import ExperienceTracker
from cortex.experience.lifecycle import LifecycleManager, LifecycleConfig
from cortex.validators.sentinel  import SentinelValidator
from cortex.validators.physicore import PhysiCoreValidator
from cortex.validators.memory_validator import MemoryValidator
from cortex.integration.server  import CortexServer
from cortex._version import __version__

log = logging.getLogger(__name__)


def build_server(cfg: CortexSettings | None = None) -> CortexServer:
    """
    Build and return a fully wired CortexServer from settings.

    Parameters
    ----------
    cfg:
        CortexSettings to build from. If None, calls get_settings() to
        load from the environment / config file.

    Returns
    -------
    CortexServer
        Ready to call .start() on, or use as a context manager.

    Example
    -------
        from cortex.config.factory import build_server

        with build_server() as server:
            response = server.certify(request)
    """
    if cfg is None:
        from cortex.config import get_settings
        cfg = get_settings()

    # ── Observability — must be first so all subsequent logs are structured ──
    from cortex.observability import setup_observability
    setup_observability(cfg)

    version = cfg.server.version or __version__

    log.info("Building Cortex server v%s", version)
    log.info("  platform_id   : %s", cfg.server.platform_id or "(not set)")
    log.info("  deployment_id : %s", cfg.server.deployment_id or "(not set)")

    # ── Memory Gateway ────────────────────────────────────────────────────────
    gateway = _build_gateway(cfg)

    # ── Validators ────────────────────────────────────────────────────────────
    sentinel  = _build_sentinel(cfg)
    physicore = _build_physicore(cfg)
    memory_v  = MemoryValidator()

    # ── Certification Gate ────────────────────────────────────────────────────
    gate = CertificationGate(
        sentinel=sentinel,
        physicore=physicore,
        memory=memory_v,
        human_override_threshold=cfg.gate.human_override_threshold,
        replan_threshold=cfg.gate.replan_threshold,
        sentinel_weight=cfg.gate.sentinel_weight,
        physicore_weight=cfg.gate.physicore_weight,
        memory_weight=cfg.gate.memory_weight,
    )
    log.info(
        "  gate thresholds: human=%.2f replan=%.2f",
        cfg.gate.human_override_threshold,
        cfg.gate.replan_threshold,
    )

    # ── Context Engine ────────────────────────────────────────────────────────
    engine = ContextEngine(
        gateway=gateway,
        default_top_k=cfg.context.default_top_k,
        default_confidence_floor=cfg.context.confidence_floor,
    )

    # ── Experience Tracker ────────────────────────────────────────────────────
    tracker = ExperienceTracker(
        gateway=gateway,
        platform_id=cfg.server.platform_id,
        deployment_id=cfg.server.deployment_id,
    )

    # ── Lifecycle Manager ─────────────────────────────────────────────────────
    lc_config = LifecycleConfig(
        decay_enabled=cfg.lifecycle.decay_enabled,
        decay_check_interval_s=cfg.lifecycle.decay_check_interval_s,
        decay_confidence_floor=cfg.lifecycle.decay_confidence_floor,
        forgetting_enabled=cfg.lifecycle.forgetting_enabled,
        forgetting_window_s=cfg.lifecycle.forgetting_window_s,
        hard_delete_after_s=cfg.lifecycle.hard_delete_after_s,
        maintenance_interval_s=cfg.lifecycle.maintenance_interval_s,
        max_records_per_adapter=cfg.lifecycle.max_records_per_adapter,
    )
    lifecycle = LifecycleManager(gateway=gateway, config=lc_config)

    if cfg.lifecycle.enabled:
        lifecycle.start()
        log.info("  lifecycle      : enabled (maintenance every %.0fs)",
                 cfg.lifecycle.maintenance_interval_s)
    else:
        log.info("  lifecycle      : disabled")

    return CortexServer(
        gateway=gateway,
        engine=engine,
        gate=gate,
        tracker=tracker,
        lifecycle=lifecycle,
        version=version,
    )


# ── private builders ──────────────────────────────────────────────────────────

def _build_gateway(cfg: CortexSettings) -> MemoryGateway:
    """Wire memory adapters based on which are enabled in settings."""
    gateway = MemoryGateway(fan_out=True)

    # L1 — always present, in-memory
    gateway.register(InMemoryAdapter(name="l1_cache"), priority=1)
    log.info("  memory adapter : InMemory (L1, always on)")

    # L2 — JSON file persistence (optional)
    if cfg.persist.enabled:
        gateway.register(
            JSONFileAdapter(file_path=cfg.persist.path, name="json_persist"),
            priority=2,
        )
        log.info("  memory adapter : JSONFile at %s", cfg.persist.path)

    # L3 — Redis (optional)
    if cfg.redis.enabled:
        try:
            adapter = RedisAdapter(
                url=cfg.redis.url,
                db=cfg.redis.db,
                ttl_seconds=cfg.redis.ttl_seconds,
                name="redis",
            )
            gateway.register(adapter, priority=10)
            log.info("  memory adapter : Redis at %s (db=%d)", cfg.redis.url, cfg.redis.db)
        except Exception as exc:
            log.error("  Redis adapter failed to initialise: %s — falling back", exc)

    # L4 — Weaviate (optional)
    if cfg.weaviate.enabled:
        try:
            from cortex.memory.adapters import WeaviateAdapter
            adapter = WeaviateAdapter(
                url=cfg.weaviate.url,
                api_key=cfg.weaviate.api_key,
                name="weaviate",
            )
            gateway.register(adapter, priority=20)
            log.info("  memory adapter : Weaviate at %s", cfg.weaviate.url)
        except Exception as exc:
            log.error("  Weaviate adapter failed to initialise: %s — falling back", exc)

    # L5 — Mem0 (optional)
    if cfg.mem0.enabled:
        try:
            from cortex.memory.adapters import Mem0Adapter
            adapter = Mem0Adapter(
                api_key=cfg.mem0.api_key or "",
                base_url=cfg.mem0.base_url,
                name="mem0",
            )
            gateway.register(adapter, priority=30)
            log.info("  memory adapter : Mem0 (managed)")
        except Exception as exc:
            log.error("  Mem0 adapter failed to initialise: %s — falling back", exc)

    gateway.start()
    return gateway


def _build_sentinel(cfg: CortexSettings) -> SentinelValidator:
    return SentinelValidator(
        max_speed_ms=cfg.sentinel.max_speed_ms,
        max_force_n=cfg.sentinel.max_force_n,
    )


def _build_physicore(cfg: CortexSettings) -> PhysiCoreValidator:
    return PhysiCoreValidator(
        min_stability_margin=cfg.physicore.min_stability_margin,
    )
