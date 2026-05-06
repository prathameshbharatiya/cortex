"""
Cortex Server
=============
The central request handler.

All transports (HTTP, gRPC, Python SDK, ROS 2) funnel through here.
The server owns the pipeline: Gateway → Engine → Gate → Tracker.
It converts protocol requests into pipeline calls and protocol responses
back out.

This is the single place where the full Cortex stack is wired together.
Every integration is just a transport wrapper around this server.

Usage
-----
    server = CortexServer.create()         # default: in-memory everything
    server = CortexServer.create(
        memory_sources=["redis://localhost:6379"],
        platform_id="arm_01",
    )
    server.start()

    # Any transport can now call:
    response = server.certify(request)
    response = server.record_outcome(request)
    response = server.get_context(request)
    response = server.health()
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from cortex.models.action import Action, ActionSpec, ActionType, ActionConstraints, Pose
from cortex.models.context import RobotState, SceneGraph
from cortex.models.memory import MemoryRecord, MemoryType
from cortex.models.decision import CertificationState

from cortex.gate.certification import CertificationGate
from cortex.context.engine     import ContextEngine
from cortex.memory.gateway     import MemoryGateway
from cortex.memory.adapters    import (
    InMemoryAdapter, JSONFileAdapter, RedisAdapter
)
from cortex.experience.tracker   import ExperienceTracker
from cortex.experience.lifecycle import LifecycleManager, LifecycleConfig
from cortex.experience.record    import OutcomeSpec, OutcomeClass

from cortex.integration.protocol import (
    CertifyRequest, CertifyResponse,
    RecordOutcomeRequest, RecordOutcomeResponse,
    GetContextRequest, GetContextResponse,
    HealthRequest, HealthResponse,
    StoreMemoryRequest, StoreMemoryResponse,
    CortexError,
)


class CortexServer:
    """
    The wired-together Cortex pipeline.
    One instance per deployment. Thread-safe.
    """

    def __init__(
        self,
        gateway:   MemoryGateway,
        engine:    ContextEngine,
        gate:      CertificationGate,
        tracker:   ExperienceTracker,
        lifecycle: LifecycleManager,
        version:   str = "0.1.0",
    ) -> None:
        self.gateway   = gateway
        self.engine    = engine
        self.gate      = gate
        self.tracker   = tracker
        self.lifecycle = lifecycle
        self.version   = version
        self._started_at = time.time()

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        memory_sources: list[str] | None = None,
        persist_path:   str | None        = None,
        platform_id:    str               = "",
        deployment_id:  str               = "",
        version:        str               = "0.1.0",
    ) -> "CortexServer":
        """
        Build a fully wired CortexServer.

        memory_sources: list of connection strings
          "memory://"              → in-memory only (default)
          "redis://host:port"      → Redis adapter
          "file:///path/to/store"  → JSON file adapter

        persist_path: if set, adds a JSON file adapter at this path
        """
        gw = MemoryGateway(fan_out=True)

        # Always have in-memory L1
        gw.register(InMemoryAdapter(name="l1_cache"), priority=1)

        if persist_path:
            gw.register(
                JSONFileAdapter(file_path=persist_path, name="persistent"),
                priority=2,
            )

        for i, source in enumerate(memory_sources or []):
            if source.startswith("redis://"):
                gw.register(RedisAdapter(url=source, name=f"redis_{i}"), priority=10 + i)
            elif source.startswith("file://"):
                path = source[7:]
                gw.register(JSONFileAdapter(file_path=path, name=f"file_{i}"), priority=5 + i)

        gw.start()

        engine   = ContextEngine(gateway=gw)
        gate     = CertificationGate()
        tracker  = ExperienceTracker(gateway=gw, platform_id=platform_id,
                                     deployment_id=deployment_id)
        lifecycle = LifecycleManager(gateway=gw)
        lifecycle.start()

        return cls(
            gateway=gw,
            engine=engine,
            gate=gate,
            tracker=tracker,
            lifecycle=lifecycle,
            version=version,
        )

    def start(self) -> "CortexServer":
        return self

    def stop(self) -> None:
        self.lifecycle.stop()
        self.gateway.stop()

    def __enter__(self) -> "CortexServer":
        return self.start()

    def __exit__(self, *_: Any) -> None:
        self.stop()

    # ── Core operations ───────────────────────────────────────────────────────

    def certify(self, req: CertifyRequest) -> CertifyResponse:
        """
        The main operation. Assemble CTX, certify action, return decision.
        """
        t0 = time.perf_counter()
        try:
            # Build robot state from request
            robot_state = RobotState(
                ee_position=req.ee_position or [0.0, 0.0, 0.0],
                ee_orientation=req.ee_quaternion or [1.0, 0.0, 0.0, 0.0],
                joint_positions=req.joint_positions,
                joint_torques=req.joint_torques,
                gripper_open=req.gripper_open,
                emergency_stop=req.emergency_stop,
                in_singularity=req.in_singularity,
            )

            scene = SceneGraph(humans_nearby=req.humans_nearby)

            # Assemble CTX
            ctx, report = self.engine.assemble(
                task=req.task or req.intent,
                robot_state=robot_state,
                scene_graph=scene,
                confidence_floor=req.confidence_floor,
                validity_window_ms=req.validity_window_ms,
                skip_physics=req.skip_physics,
            )

            # Build action
            action = Action(
                spec=ActionSpec(
                    action_type=self._parse_action_type(req.action_type),
                    target_pose=Pose(
                        x=req.target_position[0] if len(req.target_position) > 0 else 0.0,
                        y=req.target_position[1] if len(req.target_position) > 1 else 0.0,
                        z=req.target_position[2] if len(req.target_position) > 2 else 0.0,
                        qw=req.target_quaternion[0] if len(req.target_quaternion) > 0 else 1.0,
                        qx=req.target_quaternion[1] if len(req.target_quaternion) > 1 else 0.0,
                        qy=req.target_quaternion[2] if len(req.target_quaternion) > 2 else 0.0,
                        qz=req.target_quaternion[3] if len(req.target_quaternion) > 3 else 0.0,
                    ),
                    constraints=ActionConstraints(
                        max_speed_ms=req.max_speed_ms,
                        max_force_n=req.max_force_n,
                        timeout_ms=req.timeout_ms,
                    ),
                ),
                source=req.source,
                intent=req.intent,
            )

            # Certify
            cert = self.gate.certify(action, ctx)
            total_ms = (time.perf_counter() - t0) * 1000.0

            # Build response
            modified_speed = None
            modified_force = None
            if (cert.state == CertificationState.EXECUTE_WITH_CONSTRAINTS
                    and cert.action):
                modified_speed = cert.action.spec.constraints.max_speed_ms
                modified_force = cert.action.spec.constraints.max_force_n

            return CertifyResponse(
                request_id=req.request_id,
                decision_id=cert.decision_id,
                trace_id=cert.trace.trace_id,
                ctx_id=ctx.ctx_id,
                state=cert.state.value,
                approved=cert.approved,
                reason=cert.reason,
                success_probability=(
                    cert.confidence.success_probability if cert.confidence else None
                ),
                stability_margin=(
                    cert.confidence.stability_margin if cert.confidence else None
                ),
                worst_case_risk=(
                    cert.confidence.worst_case_risk.value if cert.confidence else None
                ),
                validity_window_ms=(
                    cert.confidence.validity_window_ms if cert.confidence else None
                ),
                modified_max_speed=modified_speed,
                modified_max_force=modified_force,
                failure_modes=[
                    {"code": fm.code, "description": fm.description,
                     "risk_level": fm.risk_level.value, "mitigable": fm.mitigable}
                    for fm in cert.failure_modes
                ],
                certification_latency_ms=cert.trace.total_latency_ms,
                assembly_latency_ms=report.total_ms,
            )

        except Exception as e:
            return CertifyResponse(
                request_id=req.request_id,
                decision_id="",
                trace_id="",
                ctx_id="",
                state=CertificationState.SAFE_HALT.value,
                approved=False,
                reason=f"Internal error: {type(e).__name__}",
                error=str(e),
            )

    def record_outcome(self, req: RecordOutcomeRequest) -> RecordOutcomeResponse:
        try:
            exp = self.tracker.record_simple(
                task=req.task,
                outcome_class=OutcomeClass(req.outcome_class),
                trace_id=req.trace_id,
                ctx_id=req.ctx_id,
                action_id=req.action_id,
                certification_state="EXECUTE",
                memory_record_ids=req.memory_record_ids,
                predicted_confidence=req.predicted_confidence,
                ee_position=req.ee_position,
                description=req.description,
            )
            return RecordOutcomeResponse(
                request_id=req.request_id,
                experience_id=exp.experience_id,
                surprise_score=exp.surprise_score,
            )
        except Exception as e:
            return RecordOutcomeResponse(
                request_id=req.request_id,
                experience_id="",
                surprise_score=0.0,
                error=str(e),
            )

    def get_context(self, req: GetContextRequest) -> GetContextResponse:
        try:
            robot_state = RobotState(
                ee_position=req.ee_position,
                ee_orientation=req.ee_quaternion,
                joint_positions=req.joint_positions,
                joint_torques=req.joint_torques,
                gripper_open=req.gripper_open,
                emergency_stop=req.emergency_stop,
            )
            scene = SceneGraph(humans_nearby=req.humans_nearby)
            ctx, report = self.engine.assemble(
                task=req.task,
                robot_state=robot_state,
                scene_graph=scene,
                top_k=req.top_k,
                skip_physics=req.skip_physics,
            )
            return GetContextResponse(
                request_id=req.request_id,
                ctx_id=ctx.ctx_id,
                task=ctx.task,
                memory_record_count=len(ctx.memory_records),
                constraint_count=len(ctx.safety_constraints),
                physics_feasible=report.physics_feasible,
                memory_confidence=report.memory_confidence,
                overall_confidence=report.overall_confidence,
                assembly_latency_ms=report.total_ms,
            )
        except Exception as e:
            return GetContextResponse(
                request_id=req.request_id,
                ctx_id="",
                task=req.task,
                memory_record_count=0,
                constraint_count=0,
                physics_feasible=False,
                memory_confidence=0.0,
                overall_confidence=0.0,
                assembly_latency_ms=0.0,
                error=str(e),
            )

    def store_memory(self, req: StoreMemoryRequest) -> StoreMemoryResponse:
        try:
            record = MemoryRecord(
                source=req.source,
                memory_type=MemoryType(req.memory_type),
                content=req.content or {"text": req.content_text},
                content_text=req.content_text,
                base_confidence=req.confidence,
            )
            acks = self.gateway.store(record)
            success = any(a.success for a in acks) if acks else True
            return StoreMemoryResponse(
                request_id=req.request_id,
                record_id=record.record_id,
                success=success,
            )
        except Exception as e:
            return StoreMemoryResponse(
                request_id=req.request_id,
                record_id="",
                success=False,
                error=str(e),
            )

    def health(self, req: HealthRequest | None = None) -> HealthResponse:
        rid = req.request_id if req else str(uuid.uuid4())
        try:
            adapters = self.gateway.adapter_status()
            all_healthy = all(a["healthy"] for a in adapters)
            status = "ok" if all_healthy else "degraded"
            return HealthResponse(
                request_id=rid,
                status=status,
                version=self.version,
                uptime_s=round(time.time() - self._started_at, 1),
                adapter_count=len(adapters),
                adapters=adapters,
            )
        except Exception as e:
            return HealthResponse(
                request_id=rid,
                status="error",
                error=str(e),
            )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_action_type(s: str) -> ActionType:
        mapping = {
            "move_ee":  ActionType.MOVE_EE,
            "grasp":    ActionType.GRASP,
            "release":  ActionType.RELEASE,
            "place":    ActionType.PLACE,
            "push":     ActionType.PUSH,
            "navigate": ActionType.NAVIGATE,
        }
        return mapping.get(s.lower(), ActionType.MOVE_EE)
