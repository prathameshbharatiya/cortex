# Cortex

**Context Certification Infrastructure for Physical AI**

> No action may execute in the physical world unless Cortex has certified it.

---

Cortex is the mandatory boundary between an AI model's decision and a robot's actuator.

A model can *suggest* an action. Cortex decides whether that action is allowed to run.

```
AI model  ──plan──►  Cortex.certify()  ──EXECUTE──►  robot.move()
                           │
                      SAFE_HALT ──►  robot.stop()
```

## Get started in 10 minutes

```bash
pip install cortex-gate
```

```python
import cortex

ctx  = cortex.build_context("pick red block", robot_state)
cert = cortex.certify(action, ctx)

if cert.approved:
    robot.execute(cert.action)
```

→ **[Quickstart guide](quickstart.md)**

## How it works

Cortex runs three validators in strict priority order:

1. **Sentinel** — hard safety constraints. Absolute veto power. Workspace limits, human proximity, e-stop state, force and speed limits. Sentinel rejection cannot be overridden by any other component.

2. **PhysiCore** — physics feasibility. Can the robot physically execute this? Reachability, collision prediction, stability margins, singularity detection.

3. **Memory** — context consistency. Have we done this before? What happened? Is any memory stale or contradicting the current sensor data?

Every decision is a `CertificationDecision` with one of five outcomes:

| Outcome | Meaning |
|---|---|
| `EXECUTE` | Safe. Run as proposed. |
| `EXECUTE_WITH_CONSTRAINTS` | Safe, but Cortex modified the action. Run `cert.action`. |
| `REPLAN_REQUIRED` | Goal reachable, this action not certifiable. Ask the planner for another. |
| `SAFE_HALT` | No safe action available. Stop all motion. |
| `HUMAN_OVERRIDE_REQUIRED` | Uncertainty cannot be resolved algorithmically. Escalate. |

## Documentation

| | |
|---|---|
| **[Quickstart](quickstart.md)** | Install and certify your first action |
| **[API Reference](api/reference.md)** | Every class and method |
| **[Configuration](configuration.md)** | cortex.yaml and environment variables |
| **[Error Codes](error_codes.md)** | Every failure code explained |
| **[Deployment](deployment.md)** | Docker, Kubernetes, Helm |
| **[Migrations](migrations.md)** | Schema versioning and upgrades |
