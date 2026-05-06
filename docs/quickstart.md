# Quickstart

Get Cortex running and certifying actions in under 10 minutes.

## Install

```bash
pip install cortex-gate
```

For a robot deployment with Redis and gRPC:

```bash
pip install "cortex-gate[grpc,redis]"
```

## The four-line integration

```python
import cortex

ctx  = cortex.build_context("pick red block", robot_state)
cert = cortex.certify(action, ctx)

if cert.approved:
    robot.execute(cert.action)
```

That's the complete API. Cortex handles everything else — memory retrieval, physics validation, safety constraints, confidence scoring.

## Complete minimal example

```python
import cortex
from cortex import CortexSDK
from cortex.models.action  import Action, ActionSpec, ActionType, ActionConstraints
from cortex.models.context import RobotState

# 1. Describe the robot's current state
robot_state = RobotState(
    ee_position=[0.3, 0.0, 0.5],       # end-effector [x, y, z] metres
    ee_orientation=[1.0, 0.0, 0.0, 0.0],  # quaternion [w, x, y, z]
    joint_positions=[0.0, -0.5, 0.0, -1.5, 0.0, 1.0, 0.0],
    joint_velocities=[0.0] * 7,
    joint_torques=[0.0] * 7,
)

# 2. Describe the proposed action
action = Action(
    spec=ActionSpec(
        action_type=ActionType.MOVE_EE,
        constraints=ActionConstraints(max_speed_ms=0.5, max_force_n=50.0),
    ),
    source="my_planner",
    intent="pick red block",
    ai_confidence=0.85,
)

# 3. Build context (assembles memory, constraints, physics horizon)
ctx = cortex.build_context("pick red block", robot_state)

# 4. Certify
cert = cortex.certify(action, ctx)

# 5. Act on the decision
if cert.approved:
    # cert.action may be modified (speed capped, etc.)
    robot.execute(cert.action)

elif cert.requires_replan:
    new_action = planner.replan(ctx)

elif cert.requires_human:
    operator.escalate(cert.trace)

else:  # SAFE_HALT
    robot.halt()
```

## Understanding the five outcomes

Every `certify()` call returns exactly one of these:

| State | `cert.approved` | What to do |
|---|---|---|
| `EXECUTE` | ✅ | Run `cert.action` as-is |
| `EXECUTE_WITH_CONSTRAINTS` | ✅ | Run `cert.action` — it has been modified |
| `REPLAN_REQUIRED` | ❌ | Ask the planner for a different action |
| `SAFE_HALT` | ❌ | Stop all motion |
| `HUMAN_OVERRIDE_REQUIRED` | ❌ | Escalate with `cert.trace` |

`EXECUTE_WITH_CONSTRAINTS` means Cortex modified the action — for example, capped the speed because a human is nearby. Always execute `cert.action`, not your original action.

## Using the SDK (recommended)

The `CortexSDK` is the easiest way to integrate Cortex. It manages memory, lifecycle, and the full pipeline:

```python
from cortex import CortexSDK
from cortex.models.action import Action, ActionSpec, ActionType, ActionConstraints
from cortex.models.context import RobotState

with CortexSDK(platform_id="arm_east_01") as sdk:

    # Build context (retrieves relevant memory automatically)
    ctx = sdk.build_context("pick red block", robot_state)

    # Certify
    cert = sdk.certify(action, ctx)

    if cert.approved:
        robot.execute(cert.action)

    # Close the feedback loop — makes Cortex smarter over time
    sdk.record_outcome(cert, ctx, "success")
```

The `with` block handles startup and shutdown automatically.

## Adding memory

Cortex becomes more useful when it remembers past executions:

```python
from cortex.models.memory import MemoryRecord, MemoryType

with CortexSDK() as sdk:
    # Store a memory (e.g. from a previous session)
    sdk.remember(MemoryRecord(
        source="operator",
        memory_type=MemoryType.SPATIAL,
        content={"object_id": "red_block", "position": [0.4, 0.1, 0.3]},
        content_text="Red block is on the left shelf",
        base_confidence=0.95,
    ))

    # Recall relevant memories
    records = sdk.recall("pick red block", top_k=5)
    for r in records:
        print(r.content_text)
```

## Connecting to a remote Cortex server

For robot fleet deployments, run Cortex as a server and connect from each robot:

```bash
# Start the gRPC server
CORTEX_API_KEYS=cx_live_your_key cortex-serve --port 50051
```

```python
with CortexSDK(server_address="grpc://cortex-server:50051") as sdk:
    # Same API — everything routes over gRPC
    ctx  = sdk.build_context("pick block", robot_state)
    cert = sdk.certify(action, ctx)
```

Set `X-API-Key: cx_live_your_key` in gRPC metadata, or generate a key:

```bash
cortex-keygen
```

## Health check

```bash
cortex-health              # local stack
cortex-health --server grpc://localhost:50051
cortex-health --server http://localhost:8765
```

## What's next

- [API Reference](api/reference.md) — every class and method
- [Configuration](configuration.md) — cortex.yaml, env vars
- [Deployment](deployment.md) — Docker, Kubernetes, Helm
- [Error codes](error_codes.md) — every failure code explained
- [Schemas & migrations](migrations.md) — upgrading between versions
