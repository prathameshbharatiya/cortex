"""
cortex.config.settings
======================
The single, typed configuration model for the entire Cortex stack.

Loading priority (highest wins):
  1. Explicit kwargs passed to CortexSettings(...)
  2. Environment variables  CORTEX_<SECTION>_<FIELD>=value
  3. cortex.yaml / cortex.yml in the working directory
  4. /etc/cortex/cortex.yaml  (system-wide default)
  5. Built-in defaults defined here

Quick start
-----------
    # Zero config — works out of the box, in-memory only
    from cortex.config import get_settings
    cfg = get_settings()

    # From env vars
    CORTEX_REDIS_URL=redis://localhost:6379 python run.py

    # From file
    CORTEX_CONFIG_FILE=/etc/cortex/cortex.yaml python run.py

    # From a YAML string in code
    from cortex.config import CortexSettings
    cfg = CortexSettings.from_yaml(\"\"\"
    server:
      platform_id: arm_east_01
    redis:
      url: redis://prod-redis:6379
    gate:
      human_override_threshold: 0.40
    \"\"\")

All fields are documented inline. See docs/configuration.md for the full guide.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ── helpers ───────────────────────────────────────────────────────────────────

def _find_config_file() -> Path | None:
    """Search for cortex.yaml in standard locations."""
    env_val = os.environ.get("CORTEX_CONFIG_FILE", "")
    candidates = []
    if env_val:
        candidates.append(Path(env_val))
    candidates += [
        Path.cwd() / "cortex.yaml",
        Path.cwd() / "cortex.yml",
        Path("/etc/cortex/cortex.yaml"),
        Path("/etc/cortex/cortex.yml"),
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            return p
    return None


# ── sub-models ────────────────────────────────────────────────────────────────

class ServerSettings(BaseSettings):
    """
    Core server identity and behaviour.

    Env vars: CORTEX_SERVER_*
    """
    model_config = SettingsConfigDict(env_prefix="CORTEX_SERVER_", extra="ignore")

    platform_id: str = Field(
        default="",
        description="Physical platform identifier (e.g. 'arm_east_01', 'mobile_base_02'). "
                    "Stamped on every DecisionTrace. Required in production.",
    )
    deployment_id: str = Field(
        default="",
        description="Deployment environment identifier (e.g. 'prod', 'staging', 'lab'). "
                    "Stamped on every DecisionTrace.",
    )
    version: str = Field(
        default="",
        description="Override the reported Cortex version. Defaults to cortex.__version__.",
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Root log level. Individual components respect this unless overridden.",
    )
    log_format: Literal["text", "json"] = Field(
        default="text",
        description="'text' for human-readable output, 'json' for structured log shipping.",
    )


class GrpcSettings(BaseSettings):
    """
    gRPC transport settings.

    Env vars: CORTEX_GRPC_*
    """
    model_config = SettingsConfigDict(env_prefix="CORTEX_GRPC_", extra="ignore")

    enabled: bool = Field(
        default=True,
        description="Enable the gRPC server transport.",
    )
    host: str = Field(
        default="0.0.0.0",
        description="Interface to bind the gRPC server.",
    )
    port: int = Field(
        default=50051,
        ge=1, le=65535,
        description="TCP port for the gRPC server.",
    )
    max_workers: int = Field(
        default=10,
        ge=1, le=256,
        description="gRPC thread pool size.",
    )
    max_message_size_mb: int = Field(
        default=16,
        ge=1,
        description="Maximum gRPC message size in megabytes.",
    )
    keepalive_time_s: int = Field(
        default=30,
        description="Keepalive ping interval in seconds.",
    )
    reflection_enabled: bool = Field(
        default=False,
        description="Enable gRPC server reflection (useful for debugging with grpcurl).",
    )


class HttpSettings(BaseSettings):
    """
    HTTP REST transport settings.

    Env vars: CORTEX_HTTP_*
    """
    model_config = SettingsConfigDict(env_prefix="CORTEX_HTTP_", extra="ignore")

    enabled: bool = Field(
        default=False,
        description="Enable the HTTP REST server transport.",
    )
    host: str = Field(
        default="0.0.0.0",
        description="Interface to bind the HTTP server.",
    )
    port: int = Field(
        default=8765,
        ge=1, le=65535,
        description="TCP port for the HTTP server.",
    )
    workers: int = Field(
        default=4,
        ge=1, le=64,
        description="Uvicorn worker count.",
    )
    cors_origins: list[str] = Field(
        default_factory=list,
        description="Allowed CORS origins. Empty list disables CORS.",
    )
    request_timeout_s: float = Field(
        default=5.0,
        gt=0,
        description="Per-request timeout in seconds.",
    )


class GateSettings(BaseSettings):
    """
    Certification Gate thresholds and validator weights.

    Env vars: CORTEX_GATE_*
    """
    model_config = SettingsConfigDict(env_prefix="CORTEX_GATE_", extra="ignore")

    human_override_threshold: float = Field(
        default=0.35,
        ge=0.0, le=1.0,
        description="Composite confidence below this → HUMAN_OVERRIDE_REQUIRED. "
                    "Raise to be more conservative (escalate more often). "
                    "Lower to trust the system more.",
    )
    replan_threshold: float = Field(
        default=0.50,
        ge=0.0, le=1.0,
        description="Composite confidence below this → REPLAN_REQUIRED. "
                    "Must be > human_override_threshold.",
    )
    sentinel_weight: float = Field(
        default=0.50,
        ge=0.0, le=1.0,
        description="Sentinel's weight in the composite confidence score. "
                    "sentinel + physicore + memory weights must sum to 1.0.",
    )
    physicore_weight: float = Field(
        default=0.35,
        ge=0.0, le=1.0,
        description="PhysiCore's weight in the composite confidence score.",
    )
    memory_weight: float = Field(
        default=0.15,
        ge=0.0, le=1.0,
        description="Memory validator's weight in the composite confidence score.",
    )

    @field_validator("replan_threshold")
    @classmethod
    def replan_above_human(cls, v: float, info: Any) -> float:
        hot = info.data.get("human_override_threshold", 0.35)
        if v <= hot:
            raise ValueError(
                f"replan_threshold ({v}) must be greater than "
                f"human_override_threshold ({hot})"
            )
        return v

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> "GateSettings":
        total = self.sentinel_weight + self.physicore_weight + self.memory_weight
        if abs(total - 1.0) > 0.001:
            raise ValueError(
                f"Gate weights must sum to 1.0, got {total:.3f} "
                f"(sentinel={self.sentinel_weight}, physicore={self.physicore_weight}, "
                f"memory={self.memory_weight})"
            )
        return self


class SentinelSettings(BaseSettings):
    """
    Sentinel validator safety limits.

    Env vars: CORTEX_SENTINEL_*
    """
    model_config = SettingsConfigDict(env_prefix="CORTEX_SENTINEL_", extra="ignore")

    max_speed_ms: float = Field(
        default=2.0,
        gt=0,
        description="Maximum allowed end-effector speed in m/s. "
                    "Actions requesting higher speed are rejected.",
    )
    max_force_n: float = Field(
        default=150.0,
        gt=0,
        description="Maximum allowed end-effector force in Newtons.",
    )
    human_proximity_m: float = Field(
        default=0.5,
        ge=0,
        description="Minimum safe distance to detected humans in metres. "
                    "Detections closer than this trigger L6 jerk-limiting.",
    )


class PhysicoreSettings(BaseSettings):
    """
    PhysiCore validator settings.

    Env vars: CORTEX_PHYSICORE_*
    """
    model_config = SettingsConfigDict(env_prefix="CORTEX_PHYSICORE_", extra="ignore")

    min_stability_margin: float = Field(
        default=0.10,
        ge=0.0, le=1.0,
        description="Minimum Lyapunov stability margin. Actions producing margins below "
                    "this are rejected with REPLAN_REQUIRED.",
    )
    lookahead_steps: int = Field(
        default=12,
        ge=1, le=100,
        description="Number of RK4 integration steps in the physics horizon.",
    )
    remote_url: str | None = Field(
        default=None,
        description="If set, delegate physics validation to a remote PhysiCore server "
                    "(e.g. 'http://physicore:8000'). If None, runs local simulation.",
    )
    remote_timeout_s: float = Field(
        default=0.05,
        gt=0,
        description="Timeout for remote PhysiCore calls in seconds. "
                    "Must be well under the certification budget (typically 16ms at 60Hz).",
    )


class ContextSettings(BaseSettings):
    """
    Context engine settings.

    Env vars: CORTEX_CONTEXT_*
    """
    model_config = SettingsConfigDict(env_prefix="CORTEX_CONTEXT_", extra="ignore")

    default_top_k: int = Field(
        default=10,
        ge=1, le=1000,
        description="Maximum number of memory records retrieved per certification call.",
    )
    confidence_floor: float = Field(
        default=0.60,
        ge=0.0, le=1.0,
        description="Minimum overall CTX confidence. Below this the context is "
                    "marked degraded and REPLAN_REQUIRED is more likely.",
    )


class RedisSettings(BaseSettings):
    """
    Redis memory adapter settings.

    Env vars: CORTEX_REDIS_*
    """
    model_config = SettingsConfigDict(env_prefix="CORTEX_REDIS_", extra="ignore")

    enabled: bool = Field(
        default=False,
        description="Enable the Redis memory adapter. "
                    "Set to true and provide url when Redis is available.",
    )
    url: str = Field(
        default="redis://localhost:6379",
        description="Redis connection URL. Supports redis://, rediss://, unix://.",
    )
    db: int = Field(
        default=0,
        ge=0, le=15,
        description="Redis database index.",
    )
    ttl_seconds: int = Field(
        default=3600,
        ge=1,
        description="Default TTL for memory records in Redis. "
                    "Records not retrieved within this window expire automatically.",
    )
    password: str | None = Field(
        default=None,
        description="Redis AUTH password. Leave None for passwordless Redis.",
    )
    pool_size: int = Field(
        default=10,
        ge=1, le=100,
        description="Redis connection pool size.",
    )

    @field_validator("url")
    @classmethod
    def url_has_scheme(cls, v: str) -> str:
        if not (v.startswith("redis://") or v.startswith("rediss://") or v.startswith("unix://")):
            raise ValueError(
                f"Redis URL must start with redis://, rediss://, or unix://. Got: {v!r}"
            )
        return v


class WeaviateSettings(BaseSettings):
    """
    Weaviate memory adapter settings.

    Env vars: CORTEX_WEAVIATE_*
    """
    model_config = SettingsConfigDict(env_prefix="CORTEX_WEAVIATE_", extra="ignore")

    enabled: bool = Field(
        default=False,
        description="Enable the Weaviate memory adapter (L2 semantic vector store).",
    )
    url: str = Field(
        default="http://localhost:8080",
        description="Weaviate instance URL.",
    )
    api_key: str | None = Field(
        default=None,
        description="Weaviate API key. Required for Weaviate Cloud Services (WCS). "
                    "Leave None for self-hosted instances without auth.",
    )


class Mem0Settings(BaseSettings):
    """
    Mem0 managed memory adapter settings.

    Env vars: CORTEX_MEM0_*
    """
    model_config = SettingsConfigDict(env_prefix="CORTEX_MEM0_", extra="ignore")

    enabled: bool = Field(
        default=False,
        description="Enable the Mem0 managed memory adapter.",
    )
    api_key: str | None = Field(
        default=None,
        description="Mem0 API key. Required when enabled=true.",
    )
    base_url: str | None = Field(
        default=None,
        description="Override Mem0 API base URL. Used for self-hosted Mem0 instances.",
    )

    @model_validator(mode="after")
    def key_required_if_enabled(self) -> "Mem0Settings":
        if self.enabled and not self.api_key:
            raise ValueError(
                "CORTEX_MEM0_API_KEY must be set when CORTEX_MEM0_ENABLED=true"
            )
        return self


class PersistenceSettings(BaseSettings):
    """
    JSON file persistence adapter settings.

    Env vars: CORTEX_PERSIST_*
    """
    model_config = SettingsConfigDict(env_prefix="CORTEX_PERSIST_", extra="ignore")

    enabled: bool = Field(
        default=False,
        description="Enable the JSON file persistence adapter. "
                    "Useful for single-robot setups without Redis.",
    )
    path: str = Field(
        default="./cortex_memory.json",
        description="Path to the JSON file used for persistent memory storage. "
                    "Directory must exist and be writable.",
    )


class SecuritySettings(BaseSettings):
    """
    Authentication and transport security settings.

    Env vars: CORTEX_SECURITY_* (or bare CORTEX_API_KEYS, CORTEX_SKIP_AUTH, etc.)
    """
    model_config = SettingsConfigDict(env_prefix="CORTEX_SECURITY_", extra="ignore")

    # API key auth
    api_keys: str = Field(
        default="",
        description="Comma-separated raw API keys. Each key may carry optional "
                    "metadata: 'key:robot_id:perm1+perm2'. "
                    "Env: CORTEX_API_KEYS (preferred) or CORTEX_SECURITY_API_KEYS.",
    )
    skip_auth: bool = Field(
        default=False,
        description="Disable all authentication. Local development only. "
                    "Never set in production. Env: CORTEX_SKIP_AUTH.",
    )
    auth_required: bool = Field(
        default=False,
        description="Reject all requests when no API keys are configured (fail closed). "
                    "Env: CORTEX_AUTH_REQUIRED.",
    )

    # TLS / mTLS
    tls_enabled: bool = Field(
        default=False,
        description="Enable TLS on the gRPC server. "
                    "Requires tls_cert_file and tls_key_file. "
                    "Env: CORTEX_TLS_ENABLED.",
    )
    mtls_enabled: bool = Field(
        default=False,
        description="Require client certificates (mTLS). "
                    "Requires tls_ca_file. "
                    "Env: CORTEX_TLS_MTLS.",
    )
    tls_cert_file: str = Field(
        default="",
        description="Path to server TLS certificate (PEM). Env: CORTEX_TLS_CERT_FILE.",
    )
    tls_key_file: str = Field(
        default="",
        description="Path to server TLS private key (PEM). Env: CORTEX_TLS_KEY_FILE.",
    )
    tls_ca_file: str = Field(
        default="",
        description="Path to CA certificate for client verification (PEM). "
                    "Required for mTLS. Env: CORTEX_TLS_CA_FILE.",
    )


class MetricsSettings(BaseSettings):
    """
    Prometheus metrics settings.

    Env vars: CORTEX_METRICS_*
    """
    model_config = SettingsConfigDict(env_prefix="CORTEX_METRICS_", extra="ignore")

    enabled: bool = Field(
        default=False,
        description="Start a Prometheus /metrics HTTP server.",
    )
    host: str = Field(
        default="0.0.0.0",
        description="Interface for the metrics server.",
    )
    port: int = Field(
        default=9090,
        ge=1, le=65535,
        description="Port for the Prometheus metrics server.",
    )


class TracingSettings(BaseSettings):
    """
    OpenTelemetry tracing settings.

    Env vars: CORTEX_TRACING_*
    """
    model_config = SettingsConfigDict(env_prefix="CORTEX_TRACING_", extra="ignore")

    exporter: Literal["none", "console", "otlp"] = Field(
        default="none",
        description="Trace exporter backend. "
                    "'none' disables tracing. "
                    "'console' prints spans to stdout. "
                    "'otlp' sends to an OTLP collector (Jaeger, Tempo, Datadog).",
    )
    endpoint: str = Field(
        default="http://localhost:4317",
        description="OTLP collector endpoint. Used when exporter='otlp'.",
    )
    service_name: str = Field(
        default="cortex",
        description="Service name shown in trace UIs.",
    )
    sample_rate: float = Field(
        default=1.0,
        ge=0.0, le=1.0,
        description="Fraction of traces to sample. 1.0=100%, 0.1=10%.",
    )


class LifecycleSettings(BaseSettings):
    """
    Memory lifecycle manager settings — aging, compression, forgetting.

    Env vars: CORTEX_LIFECYCLE_*
    """
    model_config = SettingsConfigDict(env_prefix="CORTEX_LIFECYCLE_", extra="ignore")

    enabled: bool = Field(
        default=True,
        description="Enable automatic memory lifecycle management.",
    )
    decay_enabled: bool = Field(
        default=True,
        description="Enable confidence decay for old records.",
    )
    decay_check_interval_s: float = Field(
        default=300.0,
        gt=0,
        description="How often to run decay checks, in seconds.",
    )
    decay_confidence_floor: float = Field(
        default=0.10,
        ge=0.0, le=1.0,
        description="Records with confidence below this are marked for archival.",
    )
    forgetting_enabled: bool = Field(
        default=True,
        description="Enable hard deletion of old, unretrieved records.",
    )
    forgetting_window_s: float = Field(
        default=7 * 24 * 3600,
        gt=0,
        description="Records not retrieved within this window are candidates for forgetting. "
                    "Default: 7 days.",
    )
    hard_delete_after_s: float = Field(
        default=30 * 24 * 3600,
        gt=0,
        description="Records older than this are hard-deleted regardless of retrieval count. "
                    "Default: 30 days.",
    )
    maintenance_interval_s: float = Field(
        default=600.0,
        gt=0,
        description="Full maintenance cycle interval in seconds. Default: 10 minutes.",
    )
    max_records_per_adapter: int = Field(
        default=50_000,
        ge=100,
        description="Compression is triggered when an adapter exceeds this record count.",
    )


# ── root settings model ───────────────────────────────────────────────────────

class CortexSettings(BaseSettings):
    """
    Root Cortex configuration model.

    Assembles all sub-models. Can be loaded from:
      - Environment variables (CORTEX_*)
      - cortex.yaml file
      - Explicit constructor kwargs

    Example (environment):
        CORTEX_SERVER_PLATFORM_ID=arm_01
        CORTEX_REDIS_ENABLED=true
        CORTEX_REDIS_URL=redis://prod-cache:6379
        CORTEX_GATE_HUMAN_OVERRIDE_THRESHOLD=0.40

    Example (YAML file at ./cortex.yaml):
        server:
          platform_id: arm_east_01
          log_level: INFO
        redis:
          enabled: true
          url: redis://localhost:6379
        gate:
          human_override_threshold: 0.40
    """
    model_config = SettingsConfigDict(
        env_prefix="CORTEX_",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    # ── config file location ──────────────────────────────────────────────
    config_file: str | None = Field(
        default=None,
        description="Path to a YAML config file. "
                    "Env: CORTEX_CONFIG_FILE. Auto-discovered if not set.",
    )

    # ── sub-models ─────────────────────────────────────────────────────────
    server:     ServerSettings     = Field(default_factory=ServerSettings)
    grpc:       GrpcSettings       = Field(default_factory=GrpcSettings)
    http:       HttpSettings       = Field(default_factory=HttpSettings)
    gate:       GateSettings       = Field(default_factory=GateSettings)
    sentinel:   SentinelSettings   = Field(default_factory=SentinelSettings)
    physicore:  PhysicoreSettings  = Field(default_factory=PhysicoreSettings)
    context:    ContextSettings    = Field(default_factory=ContextSettings)
    redis:      RedisSettings      = Field(default_factory=RedisSettings)
    weaviate:   WeaviateSettings   = Field(default_factory=WeaviateSettings)
    mem0:       Mem0Settings       = Field(default_factory=Mem0Settings)
    persist:    PersistenceSettings = Field(default_factory=PersistenceSettings)
    lifecycle:  LifecycleSettings  = Field(default_factory=LifecycleSettings)
    metrics:    MetricsSettings    = Field(default_factory=MetricsSettings)
    tracing:    TracingSettings    = Field(default_factory=TracingSettings)
    security:   SecuritySettings   = Field(default_factory=SecuritySettings)

    # ── constructors ──────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, yaml_text: str) -> "CortexSettings":
        """
        Build settings from a YAML string.

        Example:
            cfg = CortexSettings.from_yaml(\"\"\"
            server:
              platform_id: arm_01
            redis:
              enabled: true
              url: redis://localhost:6379
            \"\"\")
        """
        try:
            import yaml  # type: ignore[import]
        except ImportError as e:
            raise ImportError("pip install pyyaml to use CortexSettings.from_yaml()") from e
        data = yaml.safe_load(yaml_text) or {}
        return cls(**_flatten(data))

    @classmethod
    def from_file(cls, path: str | Path) -> "CortexSettings":
        """
        Build settings from a YAML file.

        Example:
            cfg = CortexSettings.from_file("/etc/cortex/cortex.yaml")
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Cortex config file not found: {p}")
        return cls.from_yaml(p.read_text(encoding="utf-8"))

    @classmethod
    def load(cls) -> "CortexSettings":
        """
        Load settings using the full priority chain:
          env CORTEX_CONFIG_FILE  > auto-discovered cortex.yaml  > env vars > defaults

        This is the function called by get_settings() and by all server entry points.
        """
        # Check for a file to load
        config_file_env = os.environ.get("CORTEX_CONFIG_FILE")
        if config_file_env:
            file_path = Path(config_file_env)
            if not file_path.exists():
                raise FileNotFoundError(
                    f"CORTEX_CONFIG_FILE={config_file_env!r} does not exist."
                )
            base = cls.from_file(file_path)
        else:
            discovered = _find_config_file()
            if discovered:
                base = cls.from_file(discovered)
            else:
                base = cls()

        # Layer env vars on top (pydantic-settings handles this automatically
        # when we reconstruct — but since from_yaml() bypasses env reading,
        # we do a second pass here to let env vars override file values)
        env_override = cls()
        return _merge(base, env_override)

    # ── utilities ─────────────────────────────────────────────────────────

    def display(self) -> str:
        """
        Return a redacted human-readable summary for logging at startup.
        Secrets (passwords, API keys) are masked.
        """
        lines = ["Cortex configuration:"]

        def _fmt(label: str, value: Any, secret: bool = False) -> None:
            v = "***" if secret and value else (value if value else "(not set)")
            lines.append(f"  {label:<38} {v}")

        _fmt("server.platform_id",           self.server.platform_id)
        _fmt("server.deployment_id",          self.server.deployment_id)
        _fmt("server.log_level",              self.server.log_level)
        _fmt("server.log_format",             self.server.log_format)
        lines.append("")
        _fmt("grpc.enabled",                  self.grpc.enabled)
        _fmt("grpc.host:port",                f"{self.grpc.host}:{self.grpc.port}")
        _fmt("grpc.max_workers",              self.grpc.max_workers)
        _fmt("http.enabled",                  self.http.enabled)
        _fmt("http.host:port",                f"{self.http.host}:{self.http.port}")
        lines.append("")
        _fmt("gate.human_override_threshold", self.gate.human_override_threshold)
        _fmt("gate.replan_threshold",         self.gate.replan_threshold)
        _fmt("gate.weights (s/p/m)",
             f"{self.gate.sentinel_weight}/{self.gate.physicore_weight}/{self.gate.memory_weight}")
        lines.append("")
        _fmt("sentinel.max_speed_ms",         self.sentinel.max_speed_ms)
        _fmt("sentinel.max_force_n",          self.sentinel.max_force_n)
        _fmt("sentinel.human_proximity_m",    self.sentinel.human_proximity_m)
        lines.append("")
        _fmt("context.default_top_k",         self.context.default_top_k)
        _fmt("context.confidence_floor",      self.context.confidence_floor)
        lines.append("")
        _fmt("redis.enabled",                 self.redis.enabled)
        _fmt("redis.url",                     self.redis.url if self.redis.enabled else "-")
        _fmt("redis.password",                self.redis.password, secret=True)
        _fmt("weaviate.enabled",              self.weaviate.enabled)
        _fmt("weaviate.url",                  self.weaviate.url if self.weaviate.enabled else "-")
        _fmt("weaviate.api_key",              self.weaviate.api_key, secret=True)
        _fmt("mem0.enabled",                  self.mem0.enabled)
        _fmt("mem0.api_key",                  self.mem0.api_key, secret=True)
        _fmt("persist.enabled",               self.persist.enabled)
        _fmt("persist.path",                  self.persist.path if self.persist.enabled else "-")
        lines.append("")
        _fmt("lifecycle.enabled",             self.lifecycle.enabled)
        _fmt("lifecycle.maintenance_interval_s", self.lifecycle.maintenance_interval_s)
        _fmt("lifecycle.hard_delete_after_s", self.lifecycle.hard_delete_after_s)

        return "\n".join(lines)


# ── helpers ───────────────────────────────────────────────────────────────────

def _flatten(data: dict, prefix: str = "") -> dict:
    """
    Flatten a nested YAML dict into CortexSettings constructor kwargs.

    {"server": {"platform_id": "arm"}} → {"server": ServerSettings(platform_id="arm")}
    """
    # We pass sub-dicts directly; pydantic will coerce them into sub-models.
    return data


def _merge(base: CortexSettings, override: CortexSettings) -> CortexSettings:
    """
    Merge two settings objects: env vars in `override` win over file values in `base`
    only when they differ from the default.
    """
    # Since pydantic-settings already handles env priority within a single
    # instantiation, and from_yaml() ignores env, we simply return the override
    # (which was built from env) if any env vars were set; otherwise return base.
    # A simple heuristic: if the override differs from a plain default, env was set.
    default = CortexSettings()
    # For secrets / critical fields, prefer override if it deviates from default
    if override.redis.enabled != default.redis.enabled:
        return override
    if override.server.platform_id != default.server.platform_id:
        return override
    if override.gate.human_override_threshold != default.gate.human_override_threshold:
        return override
    return base
