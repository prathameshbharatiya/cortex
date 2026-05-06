# Error codes

Every `FailureMode` in a `CertificationDecision` carries a `code` — a
machine-readable string you can branch on in your robot controller.

```python
for fm in cert.failure_modes:
    if fm.code == "emergency_stop":
        robot.acknowledge_estop()
    elif fm.code == "human_proximity":
        robot.slow_down()
```

The `blocking_failure_mode` on the `DecisionTrace` is the single code that
caused a non-EXECUTE decision. All others are warnings.

---

## Gate-level codes

These are issued by the Certification Gate itself, before any validator runs.

| Code | State produced | Cause | Resolution |
|---|---|---|---|
| `context_expired` | `REPLAN_REQUIRED` | The CTX packet has exceeded its `validity_window_ms`. | Call `build_context()` again before certifying. |

---

## Sentinel codes

Sentinel has **absolute veto power**. A Sentinel failure can never be overridden by PhysiCore or Memory scores.

| Code | Mitigable | Risk | Cause | Resolution |
|---|---|---|---|---|
| `emergency_stop` | ❌ | CRITICAL | `robot_state.emergency_stop == True` | Clear the e-stop before certifying any action. |
| `kinematic_singularity` | ❌ | CRITICAL | `robot_state.in_singularity == True` | Move the arm out of the singular configuration first. |
| `workspace_boundary_violation` | ❌ | HIGH | Target pose is outside the workspace boundary configured in `SceneGraph.workspace_bounds`. | Retarget to a pose inside the workspace. |
| `constraint_<id>_exclusion` | ❌ | CRITICAL | Target is inside an object exclusion zone defined in `safety_constraints`. | Retarget around the exclusion zone. |
| `speed_limit_exceeded` | ✅ | HIGH | Requested speed exceeds `sentinel.max_speed_ms`. Cortex will cap the speed and produce `EXECUTE_WITH_CONSTRAINTS`. | Use `cert.action` (with capped speed) instead of the original. |
| `force_limit_exceeded` | ✅ | HIGH | Requested force exceeds `sentinel.max_force_n`. | Use `cert.action` (with capped force). |
| `human_proximity` | ✅ | MEDIUM | A human was detected within `sentinel.human_proximity_m`. Speed will be reduced to 25 cm/s. | Use `cert.action`. Monitor `cert.confidence.uncertainty_sources` for `"human_proximity"`. |

---

## PhysiCore codes

| Code | Mitigable | Risk | Cause | Resolution |
|---|---|---|---|---|
| `target_unreachable` | ❌ | HIGH | Target distance from origin exceeds `physicore.workspace_radius_m`. | Retarget to a reachable pose. |
| `physics_horizon_infeasible` | ❌ | HIGH | The PhysiCore 12-step lookahead found no feasible trajectory. Collision detected on steps listed in `collision_risk_steps`. | Replanning required. Change the target or approach angle. |
| `low_stability_margin` | ✅ | MEDIUM | The trajectory passes close to a stability limit. `min_stability_margin` < threshold. Cortex reduces speed by 30%. | Use `cert.action`. Consider changing approach angle. |
| `collision_proximity` | ✅/❌ | HIGH/CRITICAL | Target is within `collision_proximity_m` of a detected object. Mitigable if clearance >= 0 (near miss). Non-mitigable if clearance < 0 (intersection). | Increase clearance to the object. |
| `residual_elevated` | ✅ | MEDIUM | PhysiCore residual model detects a sim-to-real gap above the warning threshold. Action is cautious but allowed. | Use `cert.action`. Consider re-running system identification. |
| `residual_critical` | ❌ | HIGH | Sim-to-real gap is critically large. The model cannot reliably predict this trajectory. | Halt and re-run system identification. |
| `uncertainty_elevated` | ✅ | MEDIUM | MPC uncertainty is above the warning threshold. | Use `cert.action` with reduced speed. |
| `uncertainty_critical` | ❌ | HIGH | MPC uncertainty is critically large. | Replanning required. Simplify the trajectory. |
| `loop_slow` | ✅ | LOW | PhysiCore control loop took longer than expected. Performance warning only. | Check CPU load. The action can still execute. |
| `state_exploded` | ❌ | HIGH | A state variable diverged during simulation. Numerical instability. | Replanning required. Change initial conditions. |
| `physicore_error` | ❌ | HIGH | Unexpected error in the PhysiCore engine. | Check logs. May indicate a configuration issue. |
| `action_clipped` | ✅ | LOW | One or more action parameters were clipped to safe bounds by the MPC. | Use `cert.action`. Review action parameter bounds. |

---

## Memory Validator codes

Memory failures **never block alone**. They reduce confidence and may push the composite score below the replan or human-override threshold.

| Code | Mitigable | Risk | Cause | Resolution |
|---|---|---|---|---|
| `invalid_memory_record` | ✅ | LOW | A memory record in the context has `is_currently_valid == False`. | The record has been superseded. Run a fresh `build_context()` call. |
| `stale_memory_record` | ✅ | LOW | A memory record's age exceeds `stale_threshold_s` (default 1 hour). | Refresh with a new sensor reading or mark the record invalid. |
| `high_failure_rate_memory` | ✅ | MEDIUM | A memory record has a failure rate > `failure_rate_threshold` (default 30%). Previous use of this record led to failures. | Review what caused previous failures in similar situations. |
| `memory_sensor_contradiction` | ✅ | MEDIUM | A spatial memory record contradicts current sensor data by more than `contradiction_dist_m`. Object has moved or was misidentified. | Update the spatial memory with the current sensor reading. |
| `low_memory_confidence` | ✅ | LOW | The effective confidence of memory records is below `min_confidence`. | Confirm with a fresh sensor reading and update memory. |

---

## RiskLevel

| Level | Meaning |
|---|---|
| `NONE` | No risk identified. |
| `LOW` | Minor issue, does not affect execution. |
| `MEDIUM` | Meaningful uncertainty. Execution allowed with caution. |
| `HIGH` | Significant risk. Execution blocked unless mitigable. |
| `CRITICAL` | Absolute block. No override possible via `EXECUTE_WITH_CONSTRAINTS`. |

---

## Handling failure codes in your controller

```python
from cortex.models.decision import RiskLevel

cert = cortex.certify(action, ctx)

if cert.approved:
    # Check for warnings even on approved decisions
    for fm in cert.failure_modes:
        if fm.risk_level == RiskLevel.MEDIUM:
            log.warning("Certified with warning: %s — %s", fm.code, fm.description)
    robot.execute(cert.action)

elif cert.state == CertificationState.REPLAN_REQUIRED:
    # Log the blocking reason
    if cert.trace.blocking_failure_mode:
        log.info("Replan needed: %s", cert.trace.blocking_failure_mode.code)
    action = planner.replan(ctx)

elif cert.state == CertificationState.HUMAN_OVERRIDE_REQUIRED:
    # Pass the full trace to the operator — it has everything they need
    operator.escalate(
        trace=cert.trace,
        reason=cert.reason,
        failure_modes=cert.failure_modes,
    )

elif cert.state == CertificationState.SAFE_HALT:
    robot.halt()
    # SAFE_HALT means Sentinel found something it could not mitigate at all.
    # Do not retry without changing the physical situation.
```
