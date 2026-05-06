# Configuration reference

Cortex is configured through a layered priority system. You never have to set anything — every value has a safe default. In production you configure exactly what differs from those defaults and nothing else.

## Priority order

From highest to lowest:

1. Explicit Python kwargs (`CortexSettings(redis={"enabled": True})`)
2. Environment variables (`CORTEX_REDIS_ENABLED=true`)
3. Config file specified by `CORTEX_CONFIG_FILE`
4. Auto-discovered `cortex.yaml` in the working directory
5. `/etc/cortex/cortex.yaml`
6. Built-in defaults

## Quick start

**Zero config — development:**
```bash
python run.py   # in-memory only, all defaults
```

**Environment variables — CI / containers:**
```bash
CORTEX_SERVER_PLATFORM_ID=arm_01 \
CORTEX_REDIS_ENABLED=true \
CORTEX_REDIS_URL=redis://localhost:6379 \
python run.py
```

**Config file — production:**
```bash
CORTEX_CONFIG_FILE=/etc/cortex/cortex.yaml cortex-serve
```

**In Python:**
```python
from cortex.config import get_settings, CortexSettings

# Auto-load (respects env + file)
cfg = get_settings()

# Explicit file
cfg = CortexSettings.from_file("/etc/cortex/cortex.yaml")

# Inline YAML
cfg = CortexSettings.from_yaml("""
server:
  platform_id: arm_east_01
redis:
  enabled: true
  url: redis://prod-redis:6379
""")

# Print redacted startup summary
print(cfg.display())
```

## Secrets management

**Never put secrets in `cortex.yaml`.** API keys and passwords belong in environment variables only:

```bash
# Good — env var, not in file
export CORTEX_REDIS_PASSWORD=s3cr3t
export CORTEX_WEAVIATE_API_KEY=wcs-abc123
export CORTEX_MEM0_API_KEY=mem0-xyz

# Bad — do not commit this
# redis:
#   password: s3cr3t
```

---

## Full reference

### `server` — core identity

| Key | Env var | Default | Description |
|-----|---------|---------|-------------|
| `platform_id` | `CORTEX_SERVER_PLATFORM_ID` | `""` | Physical platform identifier. Stamped on every `DecisionTrace`. Required in production. |
| `deployment_id` | `CORTEX_SERVER_DEPLOYMENT_ID` | `""` | Deployment environment label (`prod`, `staging`, `lab`). |
| `log_level` | `CORTEX_SERVER_LOG_LEVEL` | `INFO` | One of `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. |
| `log_format` | `CORTEX_SERVER_LOG_FORMAT` | `text` | `text` for human-readable, `json` for log-shipping. |

### `grpc` — gRPC transport

| Key | Env var | Default | Description |
|-----|---------|---------|-------------|
| `enabled` | `CORTEX_GRPC_ENABLED` | `true` | Enable gRPC server. |
| `host` | `CORTEX_GRPC_HOST` | `0.0.0.0` | Bind interface. |
| `port` | `CORTEX_GRPC_PORT` | `50051` | TCP port. |
| `max_workers` | `CORTEX_GRPC_MAX_WORKERS` | `10` | Thread pool size. |
| `max_message_size_mb` | `CORTEX_GRPC_MAX_MESSAGE_SIZE_MB` | `16` | Max gRPC message size. |
| `reflection_enabled` | `CORTEX_GRPC_REFLECTION_ENABLED` | `false` | Enable gRPC reflection (for `grpcurl`). |

### `http` — HTTP REST transport

| Key | Env var | Default | Description |
|-----|---------|---------|-------------|
| `enabled` | `CORTEX_HTTP_ENABLED` | `false` | Enable HTTP server. |
| `host` | `CORTEX_HTTP_HOST` | `0.0.0.0` | Bind interface. |
| `port` | `CORTEX_HTTP_PORT` | `8765` | TCP port. |
| `workers` | `CORTEX_HTTP_WORKERS` | `4` | Uvicorn workers. |

### `gate` — Certification Gate thresholds

These are the most important tuning knobs in the system. Understand them before changing them.

| Key | Env var | Default | Description |
|-----|---------|---------|-------------|
| `human_override_threshold` | `CORTEX_GATE_HUMAN_OVERRIDE_THRESHOLD` | `0.35` | Confidence below this → `HUMAN_OVERRIDE_REQUIRED`. Raise to escalate more. |
| `replan_threshold` | `CORTEX_GATE_REPLAN_THRESHOLD` | `0.50` | Confidence below this → `REPLAN_REQUIRED`. Must be > `human_override_threshold`. |
| `sentinel_weight` | `CORTEX_GATE_SENTINEL_WEIGHT` | `0.50` | Sentinel's share of composite confidence. |
| `physicore_weight` | `CORTEX_GATE_PHYSICORE_WEIGHT` | `0.35` | PhysiCore's share. |
| `memory_weight` | `CORTEX_GATE_MEMORY_WEIGHT` | `0.15` | Memory validator's share. Weights must sum to 1.0. |

**Tuning guide:**
- A robot in a busy lab with humans → raise `human_override_threshold` to 0.45
- A fully automated cell with no humans → lower `human_override_threshold` to 0.25
- Stale memory is causing too many replans → lower `memory_weight` to 0.10

### `sentinel` — safety limits

Sentinel has **absolute veto power**. These limits are enforced regardless of confidence.

| Key | Env var | Default | Description |
|-----|---------|---------|-------------|
| `max_speed_ms` | `CORTEX_SENTINEL_MAX_SPEED_MS` | `2.0` | Maximum end-effector speed (m/s). |
| `max_force_n` | `CORTEX_SENTINEL_MAX_FORCE_N` | `150.0` | Maximum end-effector force (N). |
| `human_proximity_m` | `CORTEX_SENTINEL_HUMAN_PROXIMITY_M` | `0.5` | Minimum safe human distance (m). |

### `physicore` — physics validation

| Key | Env var | Default | Description |
|-----|---------|---------|-------------|
| `min_stability_margin` | `CORTEX_PHYSICORE_MIN_STABILITY_MARGIN` | `0.10` | Minimum Lyapunov margin. Below this → `REPLAN_REQUIRED`. |
| `lookahead_steps` | `CORTEX_PHYSICORE_LOOKAHEAD_STEPS` | `12` | RK4 horizon integration steps. |
| `remote_url` | `CORTEX_PHYSICORE_REMOTE_URL` | `null` | Remote PhysiCore API URL (e.g. `http://physicore:8000`). |
| `remote_timeout_s` | `CORTEX_PHYSICORE_REMOTE_TIMEOUT_S` | `0.05` | Remote call timeout. Must be << 16ms for 60Hz. |

### `context` — context engine

| Key | Env var | Default | Description |
|-----|---------|---------|-------------|
| `default_top_k` | `CORTEX_CONTEXT_DEFAULT_TOP_K` | `10` | Max memory records per certification call. |
| `confidence_floor` | `CORTEX_CONTEXT_CONFIDENCE_FLOOR` | `0.60` | Min CTX confidence before context is marked degraded. |

### `redis` — Redis memory adapter

| Key | Env var | Default | Description |
|-----|---------|---------|-------------|
| `enabled` | `CORTEX_REDIS_ENABLED` | `false` | Enable Redis adapter. |
| `url` | `CORTEX_REDIS_URL` | `redis://localhost:6379` | Redis connection URL. |
| `db` | `CORTEX_REDIS_DB` | `0` | Database index. |
| `ttl_seconds` | `CORTEX_REDIS_TTL_SECONDS` | `3600` | Record TTL. |
| `password` | `CORTEX_REDIS_PASSWORD` | `null` | AUTH password. Set via env, never in file. |
| `pool_size` | `CORTEX_REDIS_POOL_SIZE` | `10` | Connection pool size. |

### `weaviate` — Weaviate memory adapter

| Key | Env var | Default | Description |
|-----|---------|---------|-------------|
| `enabled` | `CORTEX_WEAVIATE_ENABLED` | `false` | Enable Weaviate adapter. |
| `url` | `CORTEX_WEAVIATE_URL` | `http://localhost:8080` | Instance URL. |
| `api_key` | `CORTEX_WEAVIATE_API_KEY` | `null` | API key for Weaviate Cloud. Set via env only. |

### `mem0` — Mem0 managed memory

| Key | Env var | Default | Description |
|-----|---------|---------|-------------|
| `enabled` | `CORTEX_MEM0_ENABLED` | `false` | Enable Mem0 adapter. |
| `api_key` | `CORTEX_MEM0_API_KEY` | `null` | Required when enabled. Set via env only. |
| `base_url` | `CORTEX_MEM0_BASE_URL` | `null` | Override for self-hosted Mem0. |

### `persist` — JSON file adapter

| Key | Env var | Default | Description |
|-----|---------|---------|-------------|
| `enabled` | `CORTEX_PERSIST_ENABLED` | `false` | Enable JSON file persistence. |
| `path` | `CORTEX_PERSIST_PATH` | `./cortex_memory.json` | File path. Must be writable. |

### `lifecycle` — memory aging

| Key | Env var | Default | Description |
|-----|---------|---------|-------------|
| `enabled` | `CORTEX_LIFECYCLE_ENABLED` | `true` | Enable lifecycle management. |
| `decay_check_interval_s` | `CORTEX_LIFECYCLE_DECAY_CHECK_INTERVAL_S` | `300` | Decay check frequency (5 min). |
| `forgetting_window_s` | `CORTEX_LIFECYCLE_FORGETTING_WINDOW_S` | `604800` | Records unretrieved > this are forgotten (7 days). |
| `hard_delete_after_s` | `CORTEX_LIFECYCLE_HARD_DELETE_AFTER_S` | `2592000` | Hard delete threshold (30 days). |
| `maintenance_interval_s` | `CORTEX_LIFECYCLE_MAINTENANCE_INTERVAL_S` | `600` | Full maintenance cycle (10 min). |

---

## Production deployment examples

### Minimal production (gRPC + Redis)

```yaml
# /etc/cortex/cortex.yaml
server:
  platform_id: arm_east_01
  deployment_id: prod
  log_format: json

gate:
  human_override_threshold: 0.40

sentinel:
  max_speed_ms: 1.0

redis:
  enabled: true
  url: redis://prod-redis:6379
```

```bash
CORTEX_REDIS_PASSWORD=s3cr3t \
CORTEX_CONFIG_FILE=/etc/cortex/cortex.yaml \
cortex-serve
```

### Multi-adapter (Redis + Weaviate)

```yaml
redis:
  enabled: true
  url: redis://cache:6379
weaviate:
  enabled: true
  url: http://weaviate:8080
```

```bash
CORTEX_WEAVIATE_API_KEY=wcs-abc123 cortex-serve
```

### Docker / Kubernetes

```dockerfile
ENV CORTEX_SERVER_PLATFORM_ID=arm_01
ENV CORTEX_SERVER_LOG_FORMAT=json
ENV CORTEX_REDIS_ENABLED=true
ENV CORTEX_REDIS_URL=redis://redis:6379
```

All secrets injected via Kubernetes Secrets into environment — never baked into the image.
