# Cortex

**Context Certification Infrastructure for Physical AI**

> No action may execute in the physical world unless Cortex has certified it.

[![CI](https://github.com/cortex-ai/cortex/actions/workflows/ci.yml/badge.svg)](https://github.com/cortex-ai/cortex/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/cortex-gate)](https://pypi.org/project/cortex-gate/)
[![Python](https://img.shields.io/pypi/pyversions/cortex-gate)](https://pypi.org/project/cortex-gate/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

---

## What Cortex is

Cortex is the mandatory boundary between an AI model's decision and a robot's actuator.

A model can *suggest* an action. Cortex decides whether that action is allowed to run.

```
AI model  ──plan──►  Cortex.certify()  ──EXECUTE──►  robot.move()
                           │
                      SAFE_HALT ──►  robot.stop()
```

Cortex runs three validators in strict priority order before any action touches hardware:

1. **Sentinel** — hard safety constraints. Absolute veto. Workspace limits, human proximity, e-stop state, force/speed limits.
2. **PhysiCore** — physics feasibility. Can the robot physically do this? Reachability, collision prediction, stability margins, singularity detection.
3. **Memory** — context consistency. Have we done this before? What happened? Is the context stale?

Every decision produces a `CertificationDecision` with a bounded `ConfidenceContract` and a full `DecisionTrace` — every factor, every rejection reason, every alternative considered.

---

## Install

```bash
# Core only (zero external dependencies)
pip install cortex-gate

# With gRPC transport (robot fleet deployments)
pip install "cortex-gate[grpc]"

# With Redis memory cache (recommended for production)
pip install "cortex-gate[grpc,redis]"

# Full production stack
pip install "cortex-gate[grpc,redis,weaviate,http]"

# Everything
pip install "cortex-gate[all]"
```

Requires Python 3.10+.

---

## Four lines to integrate

```python
import cortex

ctx  = cortex.build_context("pick red block", robot_state)
cert = cortex.certify(action, ctx)

if cert.state == cortex.CertificationState.EXECUTE:
    robot.execute(cert.action)
```

That's it. Cortex handles the rest.

---

## The five certification states

| State | Meaning | What to do |
|---|---|---|
| `EXECUTE` | Safe, feasible, consistent. | Run the action as-is. |
| `EXECUTE_WITH_CONSTRAINTS` | Safe, but modified. | Run `cert.action` — not the original. |
| `REPLAN_REQUIRED` | Goal reachable, action not certifiable. | Ask the planner for a different action. |
| `SAFE_HALT` | No safe action available. | Stop all motion. Hold current pose. |
| `HUMAN_OVERRIDE_REQUIRED` | Uncertainty cannot be resolved algorithmically. | Escalate to an operator with `cert.trace`. |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                        Cortex                           │
│                                                         │
│   ┌───────────────┐     ┌───────────────────────────┐  │
│   │ Memory        │     │ Certification Gate         │  │
│   │ Gateway       │────►│                           │  │
│   │               │     │  1. Sentinel   (veto)     │  │
│   │ adapters:     │     │  2. PhysiCore  (physics)  │  │
│   │  - InMemory   │     │  3. Memory     (context)  │  │
│   │  - Redis      │     │                           │  │
│   │  - Weaviate   │     │  → CertificationDecision  │  │
│   │  - Mem0       │     └───────────────────────────┘  │
│   └───────────────┘                                     │
│                                                         │
│   ┌────────────────────────────────────────────────┐   │
│   │ Transports                                     │   │
│   │  Python SDK  │  gRPC server  │  HTTP REST      │   │
│   │  ROS 2 node  │               │                 │   │
│   └────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### Memory adapters

| Adapter | Use case | Latency |
|---|---|---|
| `InMemoryAdapter` | Development, testing | < 0.01 ms |
| `JSONFileAdapter` | Single-robot persistence | < 1 ms |
| `RedisAdapter` | Production L1 cache | < 1 ms |
| `WeaviateAdapter` | Semantic vector search | 5–20 ms |
| `Mem0Adapter` | Managed memory API | network |

---

## Transports

### Python SDK (same process)

```python
from cortex.integration.sdk.client import CortexSDK

sdk = CortexSDK()
ctx  = sdk.build_context("grasp object", robot_state)
cert = sdk.certify(action, ctx)
sdk.record_outcome(cert, ctx, "success")
```

### gRPC server (fleet deployment)

```bash
cortex-serve --host 0.0.0.0 --port 50051
```

```python
sdk = CortexSDK(server_address="grpc://cortex-server:50051")
```

### HTTP REST server

```bash
cortex-serve-http --host 0.0.0.0 --port 8765
```

```bash
curl -X POST http://localhost:8765/certify \
  -H "Content-Type: application/json" \
  -d '{"action": {...}, "task": "pick red block", "robot_state": {...}}'
```

### ROS 2

```python
from cortex.integration.ros2.node import CortexNode
# Mount as a standard ROS 2 lifecycle node
```

---

## Running the demo

```bash
git clone https://github.com/cortex-ai/cortex
cd cortex
pip install -e ".[dev]"
python examples/demo.py
```

The demo exercises all five certification states with a simulated robot arm scenario.

---

## Development

```bash
# Clone and install in editable mode with all dev extras
git clone https://github.com/cortex-ai/cortex
cd cortex
pip install -e ".[dev,grpc,redis,http]"

# Run unit tests
pytest tests/unit -v

# Run with coverage
pytest tests/unit --cov=cortex --cov-report=term-missing

# Lint
ruff check .

# Format
ruff format .

# Type check
mypy cortex
```

---

## Documentation

| Guide | What it covers |
|---|---|
| [Quickstart](docs/quickstart.md) | Install, four-line integration, first certification in 10 minutes |
| [API Reference](docs/api/reference.md) | Every class, method, and property |
| [Configuration](docs/configuration.md) | cortex.yaml, environment variables, all settings |
| [Error codes](docs/error_codes.md) | Every failure code — cause, risk level, resolution |
| [Deployment](docs/deployment.md) | Docker, Kubernetes, Helm, production security |
| [Migrations](docs/migrations.md) | Schema versioning, upgrading between versions |

---

## Security

Cortex is infrastructure for physical robots. Every transport is authenticated.

```bash
# Generate an API key
cortex-keygen

# Set it on the server
export CORTEX_API_KEYS=cx_live_your_key_here
cortex-serve

# Use it from the client
sdk = CortexSDK(server_address="grpc://cortex:50051", api_key="cx_live_your_key_here")
```

For mTLS (robot fleet with per-robot certificates):

```bash
cortex-gen-certs --robots arm_east_01,arm_west_02
export CORTEX_TLS_ENABLED=1 CORTEX_TLS_MTLS=1
cortex-serve
```

See [SECURITY.md](SECURITY.md) for the vulnerability disclosure policy.

---

## Schema migrations

When upgrading Cortex, run migrations before deploying the new version:

```bash
cortex-migrate --adapter redis --url redis://localhost:6379 --dry-run
cortex-migrate --adapter redis --url redis://localhost:6379
```

Records written by older versions are also upgraded automatically on read — the CLI is for bulk pre-deployment cleanup.

---



1. Update `cortex/_version.py` — set `__version__`
2. Update `pyproject.toml` — set `project.version` to match
3. Update `CHANGELOG.md` — add an entry under `[Unreleased]` → `[x.y.z]`
4. Commit: `git commit -m "chore: release vX.Y.Z"`
5. Tag: `git tag vX.Y.Z && git push --tags`

The release workflow triggers automatically, runs the full test suite, builds the wheel, and publishes to PyPI.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Security

See [SECURITY.md](SECURITY.md) for the vulnerability disclosure policy.

> ⚠️ v0.1.x: The gRPC and HTTP servers have no authentication. Do not expose them to untrusted networks without adding auth. This is addressed in the security hardening milestone.
