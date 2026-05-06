"""
Cortex Python SDK — the four-line integration API.
"""
from __future__ import annotations
import time, uuid
from typing import Any

from cortex.models.action  import Action, ActionSpec, ActionType, ActionConstraints, Pose
from cortex.models.context import Context, RobotState, SceneGraph, SafetyConstraint
from cortex.models.memory  import MemoryRecord, MemoryType
from cortex.models.decision import CertificationDecision, CertificationState

from cortex.gate.certification   import CertificationGate
from cortex.context.engine       import ContextEngine
from cortex.memory.gateway       import MemoryGateway
from cortex.memory.adapters      import InMemoryAdapter, JSONFileAdapter, RedisAdapter, MemoryQuery, QueryType
from cortex.experience.tracker   import ExperienceTracker
from cortex.experience.record    import OutcomeClass, OutcomeSpec, ExperienceRecord
from cortex.experience.lifecycle import LifecycleManager


class CortexSDK:
    """
    The Cortex Python SDK.

    Quick start (local mode — zero config):
        sdk = CortexSDK()
        ctx  = sdk.build_context("pick red block", robot_state)
        cert = sdk.certify(action, ctx)
        if cert.approved:
            robot.execute(cert.action)
        sdk.record_outcome(cert, ctx, "success")

    Remote mode (gRPC fleet deployment):
        sdk = CortexSDK(server_address="grpc://cortex-server:50051")
        # Same API — everything routes over gRPC
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        server_address:  str | None = None,
        gateway:         MemoryGateway     | None = None,
        gate:            CertificationGate | None = None,
        engine:          ContextEngine     | None = None,
        tracker:         ExperienceTracker | None = None,
        deployment_id:   str  = "",
        platform_id:     str  = "",
        auto_lifecycle:  bool = True,
    ) -> None:
        self.server_address = server_address
        self.deployment_id  = deployment_id
        self.platform_id    = platform_id
        self._mode          = "remote" if server_address else "local"
        self._start_time    = time.time()
        self._channel       = None

        if self._mode == "local":
            self._gateway  = gateway or self._default_gateway()
            self._gate     = gate    or CertificationGate()
            self._engine   = engine  or ContextEngine(gateway=self._gateway)
            self._tracker  = tracker or ExperienceTracker(
                gateway=self._gateway, platform_id=platform_id, deployment_id=deployment_id,
            )
            self._lifecycle = LifecycleManager(gateway=self._gateway) if auto_lifecycle else None
            self._gateway.start()
            if self._lifecycle:
                self._lifecycle.start()
        else:
            self._gateway = gateway; self._gate = gate
            self._engine = engine;   self._tracker = tracker
            self._lifecycle = None
            self._cert_stub = None; self._mem_stub = None; self._exp_stub = None

    # ── Core API ──────────────────────────────────────────────────────────────

    def build_context(
        self,
        task: str,
        robot_state: RobotState,
        memory_records: list[MemoryRecord] | None = None,
        scene_graph: SceneGraph | None = None,
        extra_constraints: list[SafetyConstraint] | None = None,
        confidence_floor: float = 0.60,
        skip_physics: bool = False,
        top_k: int = 10,
    ) -> Context:
        """Assemble a validated Context. Feed to certify()."""
        if self._mode == "local":
            ctx, _ = self._engine.assemble(
                task=task, robot_state=robot_state,
                memory_records=memory_records, scene_graph=scene_graph,
                extra_constraints=extra_constraints,
                confidence_floor=confidence_floor,
                skip_physics=skip_physics, top_k=top_k,
            )
            return ctx
        return self._remote_get_context(task, robot_state, top_k, skip_physics)

    def certify(self, action: Action, ctx: Context) -> CertificationDecision:
        """
        Certify an action. The mandatory call before any physical execution.
        Returns: EXECUTE | EXECUTE_WITH_CONSTRAINTS | REPLAN_REQUIRED | SAFE_HALT | HUMAN_OVERRIDE_REQUIRED
        """
        if self._mode == "local":
            return self._gate.certify(action, ctx)
        return self._remote_certify(action, ctx)

    def record_outcome(
        self,
        decision: CertificationDecision,
        ctx: Context,
        outcome_class: str | OutcomeClass = "success",
        description: str = "",
    ) -> ExperienceRecord:
        """Record what happened after execution. Closes the feedback loop."""
        oc = OutcomeClass(outcome_class.lower()) if isinstance(outcome_class, str) else outcome_class
        if self._mode == "local":
            return self._tracker.record(
                decision=decision, ctx=ctx,
                outcome=OutcomeSpec(outcome_class=oc, description=description),
            )
        return self._remote_record_outcome(decision, ctx, oc, description)

    # ── Memory API ────────────────────────────────────────────────────────────

    def remember(self, record: MemoryRecord) -> bool:
        """Store a memory record in the Gateway."""
        if self._mode == "local":
            acks = self._gateway.store(record)
            return any(a.success for a in acks)
        return self._remote_store(record)

    def recall(
        self,
        query_text: str,
        task: str = "",
        top_k: int = 10,
        query_type: str = "semantic",
        max_age_s: float | None = None,
    ) -> list[MemoryRecord]:
        """Retrieve memory records by query."""
        if self._mode == "local":
            qt = QueryType(query_type) if query_type in [q.value for q in QueryType] else QueryType.SEMANTIC
            return self._gateway.retrieve_records(MemoryQuery(
                query_text=query_text, query_type=qt, task=task,
                top_k=top_k, max_age_seconds=max_age_s,
            ))
        return self._remote_retrieve(query_text, task, top_k)

    # ── Status ────────────────────────────────────────────────────────────────

    def status(self) -> dict[str, Any]:
        base = {"mode": self._mode, "version": self.VERSION,
                "uptime_seconds": round(time.time() - self._start_time, 1),
                "deployment_id": self.deployment_id}
        if self._mode == "local":
            base.update({
                "adapters": self._gateway.adapter_status(),
                "experience_count": self._tracker.count(),
                "outcome_stats": self._tracker.outcome_stats(),
                "memory_counts": self._gateway.count_all(),
            })
        return base

    def add_memory_adapter(self, adapter_type: str, priority: int = 10, **kwargs) -> "CortexSDK":
        """Add a memory adapter: 'memory' | 'json' | 'redis' | 'weaviate' | 'mem0'"""
        if self._mode != "local":
            raise RuntimeError("add_memory_adapter only available in local mode")
        adapter_map: dict[str, Any] = {
            "memory": InMemoryAdapter, "json": JSONFileAdapter, "redis": RedisAdapter,
        }
        try:
            from cortex.memory.adapters import WeaviateAdapter, Mem0Adapter
            adapter_map["weaviate"] = WeaviateAdapter
            adapter_map["mem0"]     = Mem0Adapter
        except ImportError:
            pass
        cls = adapter_map.get(adapter_type.lower())
        if not cls:
            raise ValueError(f"Unknown adapter: {adapter_type}. Options: {list(adapter_map)}")
        self._gateway.register(cls(**kwargs), priority=priority, connect=True)
        return self

    def stop(self) -> None:
        if self._mode == "local":
            if self._lifecycle: self._lifecycle.stop()
            self._gateway.stop()
        elif self._channel:
            self._channel.close()

    def __enter__(self) -> "CortexSDK":
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop()

    # ── Remote stubs ──────────────────────────────────────────────────────────

    def _ensure_channel(self) -> None:
        if self._channel is None:
            import grpc
            from cortex.integration.grpc import cortex_pb2_grpc as pb_grpc
            addr = (self.server_address or "localhost:50051").replace("grpc://", "")
            self._channel    = grpc.insecure_channel(addr)
            self._cert_stub  = pb_grpc.CortexCertificationStub(self._channel)
            self._mem_stub   = pb_grpc.CortexMemoryStub(self._channel)
            self._exp_stub   = pb_grpc.CortexExperienceStub(self._channel)

    def _remote_certify(self, action: Action, ctx: Context) -> CertificationDecision:
        self._ensure_channel()
        from cortex.integration.grpc import cortex_pb2 as pb
        import json
        tp = action.spec.target_pose
        ap = pb.ActionProto(
            action_id=action.action_id, action_type=action.spec.action_type.value,
            source=action.source, intent=action.intent, ai_confidence=action.confidence,
            max_speed_ms=action.spec.constraints.max_speed_ms or 0,
            max_force_n=action.spec.constraints.max_force_n or 0,
        )
        if tp:
            ap.target_pose.CopyFrom(pb.Pose(
                position=pb.Vec3(x=tp.x, y=tp.y, z=tp.z),
                orientation=pb.Quaternion(w=tp.qw, x=tp.qx, y=tp.qy, z=tp.qz),
            ))
        ee, ori = ctx.robot_state.ee_position, ctx.robot_state.ee_orientation
        rs = pb.RobotStateProto(
            ee_position=pb.Vec3(x=ee[0], y=ee[1], z=ee[2]),
            ee_orientation=pb.Quaternion(w=ori[0], x=ori[1], y=ori[2], z=ori[3]),
            joint_positions=ctx.robot_state.joint_positions,
            emergency_stop=ctx.robot_state.emergency_stop,
        )
        resp = self._cert_stub.Certify(pb.CertifyRequest(
            action=ap, task=ctx.task, robot_state=rs, confidence_floor=ctx.confidence_floor,
        ))
        return self._response_to_decision(resp, action, ctx)

    def _remote_get_context(self, task, robot_state, top_k, skip_physics) -> Context:
        self._ensure_channel()
        from cortex.integration.grpc import cortex_pb2 as pb
        ee, ori = robot_state.ee_position, robot_state.ee_orientation
        rs = pb.RobotStateProto(
            ee_position=pb.Vec3(x=ee[0], y=ee[1], z=ee[2]),
            ee_orientation=pb.Quaternion(w=ori[0], x=ori[1], y=ori[2], z=ori[3]),
        )
        resp = self._cert_stub.GetContext(pb.GetContextRequest(
            task=task, robot_state=rs, top_k=top_k, skip_physics=skip_physics,
        ))
        return Context(ctx_id=resp.ctx_id, task=resp.task,
                       robot_state=robot_state, scene_graph=SceneGraph(),
                       memory_confidence=resp.memory_confidence,
                       assembly_latency_ms=resp.assembly_latency_ms)

    def _remote_store(self, record: MemoryRecord) -> bool:
        self._ensure_channel()
        from cortex.integration.grpc import cortex_pb2 as pb
        import json
        proto = pb.MemoryRecordProto(
            record_id=record.record_id, source=record.source,
            memory_type=record.memory_type.value,
            content_json=json.dumps(record.content, default=str),
            content_text=record.content_text, valid_at=record.valid_at,
            base_confidence=record.base_confidence, outcome_tag=record.outcome_tag.value,
        )
        return self._mem_stub.Store(pb.StoreRequest(record=proto)).success

    def _remote_retrieve(self, query_text, task, top_k) -> list[MemoryRecord]:
        self._ensure_channel()
        from cortex.integration.grpc import cortex_pb2 as pb
        import json
        resp = self._mem_stub.Retrieve(pb.RetrieveRequest(
            query_text=query_text, task=task, top_k=top_k, query_type="semantic",
        ))
        records = []
        for r in resp.records:
            try: content = json.loads(r.content_json) if r.content_json else {}
            except: content = {}
            records.append(MemoryRecord(
                record_id=r.record_id, source=r.source or "grpc",
                memory_type=MemoryType(r.memory_type) if r.memory_type else MemoryType.SEMANTIC,
                content=content, content_text=r.content_text,
                valid_at=r.valid_at or time.time(), base_confidence=r.base_confidence or 0.9,
            ))
        return records

    def _remote_record_outcome(self, decision, ctx, oc, description) -> ExperienceRecord:
        self._ensure_channel()
        from cortex.integration.grpc import cortex_pb2 as pb
        self._exp_stub.RecordOutcome(pb.RecordOutcomeRequest(
            decision_id=decision.decision_id, trace_id=decision.trace.trace_id,
            ctx_id=ctx.ctx_id, task=ctx.task, outcome_class=oc.value, description=description,
            predicted_confidence=decision.confidence.success_probability if decision.confidence else 0.5,
            memory_record_ids=[r.record_id for r in ctx.memory_records],
            ee_position=ctx.robot_state.ee_position,
        ))
        t = ExperienceTracker(async_writes=False)
        return t.record_simple(task=ctx.task, outcome_class=oc,
                                trace_id=decision.trace.trace_id,
                                predicted_confidence=decision.confidence.success_probability if decision.confidence else 0.5)

    @staticmethod
    def _response_to_decision(resp, action, ctx) -> CertificationDecision:
        from cortex.models.decision import (
            CertificationDecision, CertificationState,
            ConfidenceContract, DecisionTrace, ValidationResult, RiskLevel,
        )
        try: state = CertificationState(resp.state)
        except ValueError: state = CertificationState.REPLAN_REQUIRED

        def vr(p):
            if p is None: return None
            return ValidationResult(passed=p.passed, score=p.score, notes=p.notes, latency_ms=p.latency_ms)

        trace = DecisionTrace(
            trace_id=resp.trace_id or str(uuid.uuid4()), ctx_id=resp.ctx_id or ctx.ctx_id,
            action_id=action.action_id, sentinel_result=vr(resp.sentinel_result),
            physicore_result=vr(resp.physicore_result), memory_result=vr(resp.memory_result),
            certification_state=state, total_latency_ms=resp.total_latency_ms,
        )
        confidence = None
        if resp.HasField("confidence"):
            c = resp.confidence
            try: risk = RiskLevel(c.worst_case_risk)
            except: risk = RiskLevel.MEDIUM
            confidence = ConfidenceContract(
                success_probability=c.success_probability, stability_margin=c.stability_margin,
                worst_case_risk=risk, uncertainty_sources=list(c.uncertainty_sources),
                validity_window_ms=c.validity_window_ms, validation_proofs={},
            )
        return CertificationDecision(
            decision_id=resp.decision_id or str(uuid.uuid4()), state=state,
            action=action if state in (CertificationState.EXECUTE, CertificationState.EXECUTE_WITH_CONSTRAINTS) else None,
            confidence=confidence, trace=trace, reason=resp.reason,
        )

    @staticmethod
    def _default_gateway() -> MemoryGateway:
        gw = MemoryGateway()
        gw.register(InMemoryAdapter(name="default"), priority=1)
        return gw
