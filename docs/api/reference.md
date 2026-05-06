# API Reference

## Top-level functions

These are available directly from `import cortex`.

---

### `cortex.build_context(task, robot_state, memory_records=None, ...)` → `Context`

Assemble a validated Context Packet (CTX) from the robot's current state.

The context is the mandatory input to `certify()`. It encodes: what the robot is trying to do, its current physical state, relevant memories, safety constraints, and a physics horizon.

```python
ctx = cortex.build_context(
    task="pick red block",
    robot_state=robot_state,
    memory_records=None,       # auto-retrieved from gateway if None
    scene_graph=None,          # current scene detections
    extra_constraints=None,    # additional SafetyConstraints
    confidence_floor=0.60,     # minimum confidence for EXECUTE
    skip_physics=False,        # skip PhysiCore horizon (faster, less safe)
    top_k=10,                  # max memory records to retrieve
)
```

**Returns:** `Context` — immutable, valid for `validity_window_ms` milliseconds (default 500ms).

---

### `cortex.certify(action, ctx)` → `CertificationDecision`

The core function. Runs three validators in priority order and issues a decision.

```python
cert = cortex.certify(action, ctx)
```

**Returns:** `CertificationDecision`

**Never raises.** All errors produce a `HUMAN_OVERRIDE_REQUIRED` decision rather than an exception.

---

## CortexSDK

The recommended integration layer. Manages memory, lifecycle, and the full pipeline.

```python
from cortex import CortexSDK

sdk = CortexSDK(
    server_address=None,    # None = local mode. "grpc://host:port" = remote mode.
    platform_id="",         # Robot identifier. Stamped on every ExperienceRecord.
    deployment_id="",       # Deployment environment. Stamped on every ExperienceRecord.
    auto_lifecycle=True,    # Start memory lifecycle manager automatically.
)
```

Use as a context manager (recommended):

```python
with CortexSDK(platform_id="arm_east_01") as sdk:
    ctx  = sdk.build_context("pick block", robot_state)
    cert = sdk.certify(action, ctx)
```

### `sdk.build_context(task, robot_state, ...)` → `Context`

Same signature as `cortex.build_context()`. Memory is retrieved from the gateway automatically.

### `sdk.certify(action, ctx)` → `CertificationDecision`

Same as `cortex.certify()`. Routes to gRPC if in remote mode.

### `sdk.record_outcome(decision, ctx, outcome_class="success", description="")` → `ExperienceRecord`

Record what actually happened after execution. This closes the feedback loop and makes future certifications more accurate.

```python
exp = sdk.record_outcome(cert, ctx, "success")
exp = sdk.record_outcome(cert, ctx, "failure", description="Gripper slipped")
```

`outcome_class` accepts a string or `OutcomeClass` enum:
`"success"` | `"failure"` | `"partial"` | `"unsafe"` | `"timeout"` | `"interrupted"` | `"skipped"`

**Returns:** `ExperienceRecord` with `surprise_score` (how unexpected was this outcome?).

### `sdk.remember(record)` → `bool`

Store a `MemoryRecord` in the gateway.

```python
sdk.remember(MemoryRecord(
    source="operator",
    memory_type=MemoryType.SPATIAL,
    content={"object_id": "cup", "position": [0.4, 0.1, 0.3]},
    content_text="Cup is on the right shelf",
    base_confidence=0.95,
))
```

### `sdk.recall(query_text, task="", top_k=10, ...)` → `list[MemoryRecord]`

Retrieve memory records by semantic query.

```python
records = sdk.recall("pick red block", top_k=5)
```

### `sdk.status()` → `dict`

Return a status snapshot including uptime, adapter health, and experience statistics.

### `sdk.stop()` → `None`

Shut down the SDK cleanly. Called automatically by the `with` block.

---

## CertificationDecision

The output of `certify()`. Always check `cert.state` or `cert.approved` before executing.

```python
cert.state           # CertificationState enum
cert.approved        # True if EXECUTE or EXECUTE_WITH_CONSTRAINTS
cert.blocked         # not cert.approved
cert.requires_replan # True if REPLAN_REQUIRED
cert.requires_human  # True if HUMAN_OVERRIDE_REQUIRED
cert.action          # Action to execute (may be modified). None if blocked.
cert.confidence      # ConfidenceContract (None if blocked)
cert.trace           # DecisionTrace — full audit trail
cert.failure_modes   # list[FailureMode]
cert.reason          # str — human-readable explanation
```

---

## CertificationState

```python
from cortex import CertificationState

CertificationState.EXECUTE                   # safe, run as proposed
CertificationState.EXECUTE_WITH_CONSTRAINTS  # safe, run cert.action (modified)
CertificationState.REPLAN_REQUIRED           # goal reachable, action not certifiable
CertificationState.SAFE_HALT                 # no safe action available
CertificationState.HUMAN_OVERRIDE_REQUIRED   # escalate to operator
```

---

## ConfidenceContract

The structured guarantee returned for approved decisions.

```python
conf = cert.confidence

conf.success_probability   # float [0, 1] — P(action achieves goal)
conf.stability_margin      # float — distance to nearest constraint violation (%)
conf.worst_case_risk       # RiskLevel — NONE | LOW | MEDIUM | HIGH | CRITICAL
conf.uncertainty_sources   # list[str] — named sources of remaining uncertainty
conf.validity_window_ms    # float — how long this certification is valid
conf.validation_proofs     # dict — per-validator ValidationResult
conf.is_high_confidence    # bool — probability >= 0.80 and risk <= LOW
conf.is_acceptable         # bool — probability >= 0.60 and risk < HIGH
```

---

## DecisionTrace

Full audit record of every certification decision.

```python
trace = cert.trace

trace.trace_id             # str — unique ID for this decision
trace.ctx_id               # str — ID of the context used
trace.action_id            # str — ID of the proposed action
trace.sentinel_result      # ValidationResult | None
trace.physicore_result     # ValidationResult | None
trace.memory_result        # ValidationResult | None
trace.confidence_contract  # ConfidenceContract | None
trace.certification_state  # CertificationState
trace.blocking_failure_mode # FailureMode | None — what caused a non-EXECUTE decision
trace.total_latency_ms     # float — end-to-end certification time
trace.timestamp            # float — Unix timestamp
```

---

## ValidationResult

Returned by each validator. Available in `cert.trace.sentinel_result`, etc.

```python
result.passed         # bool
result.score          # float [0, 1]
result.failure_modes  # list[FailureMode]
result.latency_ms     # float — how long this validator took
result.notes          # str — human-readable detail
```

---

## MemoryRecord

The canonical memory format. All adapters normalise to this.

```python
from cortex import MemoryRecord, MemoryType, OutcomeTag

record = MemoryRecord(
    source="my_system",
    memory_type=MemoryType.SPATIAL,   # EPISODIC | SEMANTIC | SPATIAL | PROCEDURAL
    content={"position": [0.4, 0.1, 0.3]},
    content_text="Object is on the shelf",
    base_confidence=0.9,
    outcome_tag=OutcomeTag.SUCCESS,   # SUCCESS | FAILURE | PARTIAL | UNKNOWN
)

# Properties
record.record_id            # str — unique UUID
record.is_currently_valid   # bool — not marked invalid
record.age_seconds          # float
record.effective_confidence # float — decays with age, boosted by sensor anchoring
record.success_rate         # float — success_count / outcome_count
record.is_sensor_anchored   # bool
```

---

## RobotState

The live proprioceptive state of the robot.

```python
from cortex import RobotState

state = RobotState(
    ee_position=[x, y, z],           # metres, world frame
    ee_orientation=[w, x, y, z],      # unit quaternion
    ee_velocity=[vx, vy, vz],         # m/s (optional)
    joint_positions=[...],            # radians
    joint_velocities=[...],           # rad/s
    joint_torques=[...],              # Nm
    gripper_open=True,
    emergency_stop=False,
    in_singularity=False,
)
```

---

## FailureMode

A structured risk identified during certification.

```python
fm.code         # str — machine-readable (see Error Codes)
fm.description  # str — human-readable explanation
fm.risk_level   # RiskLevel — NONE | LOW | MEDIUM | HIGH | CRITICAL
fm.source       # str — "sentinel" | "physicore" | "memory" | "gate"
fm.mitigable    # bool — True if EXECUTE_WITH_CONSTRAINTS can address this
```

---

## Memory adapters

```python
from cortex.memory.adapters import (
    InMemoryAdapter,    # development and testing, no persistence
    JSONFileAdapter,    # single-robot, file-backed persistence
    RedisAdapter,       # production L1 cache, sub-millisecond reads
    WeaviateAdapter,    # semantic vector search, large memory stores
    Mem0Adapter,        # managed memory API
)
```

All adapters share the same interface: `store(record)`, `retrieve(query)`, `health_check()`.

---

## Configuration

```python
from cortex.config import get_settings, CortexSettings

cfg = get_settings()                          # load from env / cortex.yaml
cfg = CortexSettings.from_yaml("...")         # from YAML string
cfg = CortexSettings.from_file("path.yaml")  # from file
print(cfg.display())                          # redacted startup summary
```

See [configuration.md](../configuration.md) for the full reference.

---

## Security

```python
from cortex.security import generate_api_key, load_server_credentials

# Generate a key
key = generate_api_key()   # cx_live_...

# FastAPI dependency
from cortex.security import require_permission
@app.post("/certify")
async def certify(..., _=Depends(require_permission("certify"))):
    ...

# gRPC interceptor
from cortex.security import ApiKeyInterceptor
server = grpc.server(..., interceptors=[ApiKeyInterceptor()])
```

---

## Schema migrations

```python
from cortex.migrations import run_migration, CURRENT_VERSION, MigrationError

report = run_migration(adapter, dry_run=True)
print(report)  # scan without writing

report = run_migration(adapter, dry_run=False)
if not report.success:
    raise MigrationError(report)
```

```bash
cortex-migrate --status
cortex-migrate --adapter redis --url redis://localhost:6379 --dry-run
cortex-migrate --adapter json  --path ./cortex_memory.json
```
