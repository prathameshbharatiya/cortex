
"""Cortex gRPC Server — implements all four Cortex services."""
from __future__ import annotations
import json, time, os
from concurrent import futures
from typing import Any
import grpc

from cortex.integration.grpc import cortex_pb2 as pb
from cortex.integration.grpc import cortex_pb2_grpc as pb_grpc
from cortex.models.action  import Action, ActionSpec, ActionType, ActionConstraints, Pose
from cortex.models.context import RobotState, SceneGraph, SafetyConstraint, Context
from cortex.models.memory  import MemoryRecord, MemoryType, OutcomeTag
from cortex.models.decision import CertificationState
from cortex.gate.certification  import CertificationGate
from cortex.context.engine      import ContextEngine
from cortex.memory.gateway      import MemoryGateway
from cortex.memory.adapters     import InMemoryAdapter, MemoryQuery, QueryType
from cortex.experience.tracker  import ExperienceTracker
from cortex.experience.record   import OutcomeClass, OutcomeSpec
from cortex.experience.lifecycle import LifecycleManager

_VERSION = "1.0.0"

def _proto_rs(p):
    return RobotState(
        ee_position=[p.ee_position.x, p.ee_position.y, p.ee_position.z],
        ee_orientation=[p.ee_orientation.w, p.ee_orientation.x, p.ee_orientation.y, p.ee_orientation.z],
        joint_positions=list(p.joint_positions), joint_velocities=list(p.joint_velocities),
        joint_torques=list(p.joint_torques), gripper_open=p.gripper_open,
        gripper_force_n=p.gripper_force_n, emergency_stop=p.emergency_stop, in_singularity=p.in_singularity,
    )

def _proto_mem(p):
    try: content = json.loads(p.content_json) if p.content_json else {}
    except: content = {}
    return MemoryRecord(
        record_id=p.record_id if p.record_id else str(__import__("uuid").uuid4()), source=p.source or "grpc",
        memory_type=MemoryType(p.memory_type) if p.memory_type else MemoryType.SEMANTIC,
        content=content, content_text=p.content_text,
        valid_at=p.valid_at or time.time(), invalid_at=p.invalid_at if p.invalid_at > 0 else None,
        base_confidence=p.base_confidence or 0.9,
        outcome_tag=OutcomeTag(p.outcome_tag) if p.outcome_tag else OutcomeTag.UNKNOWN,
        outcome_count=p.outcome_count, success_count=p.success_count,
    )

def _mem_proto(r):
    return pb.MemoryRecordProto(
        record_id=r.record_id, source=r.source, memory_type=r.memory_type.value,
        content_json=json.dumps(r.content, default=str), content_text=r.content_text,
        valid_at=r.valid_at, invalid_at=r.invalid_at or 0.0, base_confidence=r.base_confidence,
        outcome_tag=r.outcome_tag.value, outcome_count=r.outcome_count, success_count=r.success_count,
        is_sensor_anchored=r.is_sensor_anchored,
    )

def _proto_action(p):
    tp = None
    if p.HasField("target_pose"):
        tp = Pose(x=p.target_pose.position.x, y=p.target_pose.position.y, z=p.target_pose.position.z,
                  qw=p.target_pose.orientation.w, qx=p.target_pose.orientation.x,
                  qy=p.target_pose.orientation.y, qz=p.target_pose.orientation.z)
    try: at = ActionType(p.action_type)
    except: at = ActionType.MOVE_EE
    return Action(**({"action_id": p.action_id} if p.action_id else {}),
        spec=ActionSpec(action_type=at, target_pose=tp,
            constraints=ActionConstraints(max_speed_ms=p.max_speed_ms or None, max_force_n=p.max_force_n or None)),
        source=p.source or "grpc", intent=p.intent, confidence=p.ai_confidence or 1.0)

def _action_proto(a):
    pr = pb.ActionProto(action_id=a.action_id, action_type=a.spec.action_type.value,
        source=a.source, intent=a.intent, ai_confidence=a.confidence,
        max_speed_ms=a.spec.constraints.max_speed_ms or 0, max_force_n=a.spec.constraints.max_force_n or 0)
    if a.spec.target_pose:
        p = a.spec.target_pose
        pr.target_pose.CopyFrom(pb.Pose(position=pb.Vec3(x=p.x,y=p.y,z=p.z),
            orientation=pb.Quaternion(w=p.qw,x=p.qx,y=p.qy,z=p.qz)))
    return pr

def _vr_proto(r):
    if r is None: return pb.ValidationResultProto(passed=True, score=0)
    return pb.ValidationResultProto(passed=r.passed, score=r.score, notes=r.notes,
        latency_ms=r.latency_ms, failure_codes=[fm.code for fm in r.failure_modes])

def _constraint_proto(c):
    from cortex.models.context import SafetyConstraint
    try: params = json.loads(c.parameters_json) if c.parameters_json else {}
    except: params = {}
    return SafetyConstraint(constraint_id=c.constraint_id, description=c.description,
        constraint_type=c.constraint_type, parameters=params, veto_power=c.veto_power, source=c.source or "grpc")

class CertServicer(pb_grpc.CortexCertificationServicer):
    def __init__(self, gate, engine):
        self.gate = gate; self.engine = engine

    def Certify(self, req, ctx):
        rs = _proto_rs(req.robot_state)
        mems = [_proto_mem(r) for r in req.memory_records]
        cons = [_constraint_proto(c) for c in req.extra_constraints]
        action = _proto_action(req.action)
        context, report = self.engine.assemble(
            task=req.task or "unknown", robot_state=rs,
            memory_records=mems if mems else None, extra_constraints=cons if cons else None,
            confidence_floor=req.confidence_floor or 0.60, skip_physics=req.skip_physics)
        cert = self.gate.certify(action, context)
        resp = pb.CertifyResponse(
            decision_id=cert.decision_id, state=cert.state.value, reason=cert.reason,
            total_latency_ms=cert.trace.total_latency_ms, trace_id=cert.trace.trace_id,
            ctx_id=context.ctx_id, timestamp=time.time(),
            failure_mode_codes=[fm.code for fm in cert.failure_modes],
            sentinel_result=_vr_proto(cert.trace.sentinel_result),
            physicore_result=_vr_proto(cert.trace.physicore_result),
            memory_result=_vr_proto(cert.trace.memory_result),
        )
        if cert.action: resp.certified_action.CopyFrom(_action_proto(cert.action))
        if cert.confidence:
            c = cert.confidence
            resp.confidence.CopyFrom(pb.ConfidenceContractProto(
                success_probability=c.success_probability, stability_margin=c.stability_margin,
                worst_case_risk=c.worst_case_risk.value, uncertainty_sources=c.uncertainty_sources,
                validity_window_ms=c.validity_window_ms))
        return resp

    def GetContext(self, req, ctx):
        rs = _proto_rs(req.robot_state)
        context, report = self.engine.assemble(task=req.task or "unknown", robot_state=rs,
            top_k=req.top_k or 10, skip_physics=req.skip_physics)
        return pb.GetContextResponse(
            ctx_id=context.ctx_id, task=context.task,
            memory_records=[_mem_proto(r) for r in context.memory_records],
            memory_confidence=context.memory_confidence,
            assembly_latency_ms=context.assembly_latency_ms,
            physics_feasible=context.physics_feasible, warnings=report.warnings)

class MemServicer(pb_grpc.CortexMemoryServicer):
    def __init__(self, gateway): self.gateway = gateway

    def Retrieve(self, req, ctx):
        try: qt = QueryType(req.query_type)
        except: qt = QueryType.SEMANTIC
        mts = []
        for mt in req.memory_types:
            try: mts.append(MemoryType(mt))
            except: pass
        result = self.gateway.retrieve(MemoryQuery(
            query_text=req.query_text, query_type=qt, memory_types=mts,
            top_k=req.top_k or 10, max_age_seconds=req.max_age_seconds or None,
            min_confidence=req.min_confidence, task=req.task))
        return pb.RetrieveResponse(records=[_mem_proto(r) for r in result.records],
            total_latency_ms=result.total_latency_ms, adapters_queried=result.adapters_queried,
            deduplicated=result.deduplicated)

    def Store(self, req, ctx):
        record = _proto_mem(req.record)
        acks = self.gateway.store(record)
        ok = any(a.success for a in acks) if acks else False
        return pb.StoreResponse(record_id=record.record_id, success=ok,
            latency_ms=acks[0].latency_ms if acks else 0.0, error="" if ok else "write failed")

    def Delete(self, req, ctx):
        deleted = False
        try:
            for schema in self.gateway.adapters():
                with self.gateway._lock:
                    for entry in self.gateway._adapters:
                        if entry.schema.name == schema.name:
                            if entry.adapter.delete(req.record_id): deleted = True
        except Exception: pass
        return pb.DeleteResponse(deleted=deleted)

class ExpServicer(pb_grpc.CortexExperienceServicer):
    def __init__(self, tracker): self.tracker = tracker

    def RecordOutcome(self, req, ctx):
        try: oc = OutcomeClass(req.outcome_class)
        except: oc = OutcomeClass.PARTIAL
        exp = self.tracker.record_simple(task=req.task, outcome_class=oc,
            trace_id=req.trace_id, ctx_id=req.ctx_id, action_id=req.decision_id,
            predicted_confidence=req.predicted_confidence or 0.5,
            memory_record_ids=list(req.memory_record_ids), description=req.description)
        return pb.RecordOutcomeResponse(experience_id=exp.experience_id,
            surprise_score=exp.surprise_score, is_high_surprise=exp.is_high_surprise)

    def GetOutcomeStats(self, req, ctx):
        s = self.tracker.outcome_stats(task=req.task or None)
        return pb.OutcomeStatsResponse(total=s.get("total",0), success_count=s.get("success_count",0),
            failure_count=s.get("failure_count",0), success_rate=s.get("success_rate",0.0),
            avg_surprise=s.get("avg_surprise",0.0), high_surprise=s.get("high_surprise",0))

class HealthServicer(pb_grpc.CortexHealthServicer):
    def __init__(self, gateway, tracker, t0):
        self.gateway = gateway; self.tracker = tracker; self.t0 = t0

    def Check(self, req, ctx):
        statuses = [pb.AdapterStatus(name=s["name"], backend=s["backend"], healthy=s["healthy"],
            enabled=s["enabled"], reads=s["reads"], writes=s["writes"], errors=s["errors"])
            for s in self.gateway.adapter_status()]
        return pb.HealthResponse(alive=True, ready=True, version=_VERSION,
            adapters=statuses, experience_count=self.tracker.count(),
            uptime_seconds=time.time()-self.t0)

class CortexServer:
    def __init__(self, port=50051, gateway=None, gate=None, engine=None,
                 tracker=None, lifecycle=None, max_workers=10, deployment_id=""):
        self.port = port
        self._t0  = time.time()
        self._max_workers = max_workers
        self._gateway  = gateway  or self._default_gw()
        self._gate     = gate     or CertificationGate()
        self._engine   = engine   or ContextEngine(gateway=self._gateway)
        self._tracker  = tracker  or ExperienceTracker(gateway=self._gateway)
        self._lifecycle = lifecycle or LifecycleManager(gateway=self._gateway)
        self._running  = False

    def start(self):
        from cortex.security.api_keys import ApiKeyInterceptor
        import grpc as _grpc

        interceptors = [ApiKeyInterceptor()]
        self._server = _grpc.server(
            futures.ThreadPoolExecutor(max_workers=self._max_workers),
            interceptors=interceptors,
        )

        pb_grpc.add_CortexCertificationServicer_to_server(CertServicer(self._gate, self._engine), self._server)
        pb_grpc.add_CortexMemoryServicer_to_server(MemServicer(self._gateway), self._server)
        pb_grpc.add_CortexExperienceServicer_to_server(ExpServicer(self._tracker), self._server)
        pb_grpc.add_CortexHealthServicer_to_server(HealthServicer(self._gateway, self._tracker, self._t0), self._server)

        # TLS / mTLS
        tls_enabled = os.environ.get("CORTEX_TLS_ENABLED", "0") == "1"
        if tls_enabled:
            from cortex.security.mtls import load_server_credentials
            creds = load_server_credentials()
            self._server.add_secure_port(f"[::]:{self.port}", creds)
        else:
            self._server.add_insecure_port(f"[::]:{self.port}")

        self._server.start()
        self._gateway.start()
        self._lifecycle.start()
        self._running = True
        return self

    def stop(self, grace=1.0):
        if self._running:
            self._server.stop(grace)
            self._gateway.stop()
            self._lifecycle.stop()
            self._running = False

    def wait_for_termination(self): self._server.wait_for_termination()
    def __enter__(self): return self.start()
    def __exit__(self, *_): self.stop()

    @property
    def address(self): return f"localhost:{self.port}"

    @staticmethod
    def _default_gw():
        gw = MemoryGateway()
        gw.register(InMemoryAdapter(name="default"), priority=1)
        return gw


def main() -> None:
    """
    Entry point for the `cortex-serve` CLI command.

    Loads configuration from the full priority chain
    (CORTEX_CONFIG_FILE → cortex.yaml → env vars → defaults),
    then starts the gRPC server and blocks until interrupted.

    Quick start:
        cortex-serve
        cortex-serve --host 0.0.0.0 --port 50051
        CORTEX_GRPC_PORT=50052 cortex-serve
        CORTEX_CONFIG_FILE=/etc/cortex/cortex.yaml cortex-serve
    """
    import argparse
    import logging
    import signal
    import sys

    parser = argparse.ArgumentParser(
        prog="cortex-serve",
        description="Start the Cortex gRPC certification server.",
    )
    parser.add_argument("--host",     default=None, help="Bind host (overrides config)")
    parser.add_argument("--port",     default=None, type=int, help="Bind port (overrides config)")
    parser.add_argument("--workers",  default=None, type=int, help="Thread pool size (overrides config)")
    parser.add_argument("--config",   default=None, help="Path to cortex.yaml config file")
    parser.add_argument("--platform", default=None, help="Platform ID (overrides config)")
    args = parser.parse_args()

    # Set config file path before loading settings
    if args.config:
        import os
        os.environ["CORTEX_CONFIG_FILE"] = args.config

    from cortex.config import get_settings
    from cortex.config.factory import build_server

    cfg = get_settings()

    # CLI args override config
    if args.host:     cfg.grpc.host = args.host
    if args.port:     cfg.grpc.port = args.port
    if args.workers:  cfg.grpc.max_workers = args.workers
    if args.platform: cfg.server.platform_id = args.platform

    logging.basicConfig(
        level=getattr(logging, cfg.server.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    log = logging.getLogger("cortex.grpc")
    log.info(cfg.display())

    core_server = build_server(cfg)

    grpc_server = CortexServer(
        port=cfg.grpc.port,
        gateway=core_server.gateway,
        gate=core_server.gate,
        engine=core_server.engine,
        tracker=core_server.tracker,
        lifecycle=core_server.lifecycle,
        max_workers=cfg.grpc.max_workers,
        deployment_id=cfg.server.deployment_id,
    )

    def _shutdown(sig, frame):
        log.info("Shutting down Cortex gRPC server...")
        grpc_server.stop()
        core_server.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    grpc_server.start()
    log.info("Cortex gRPC server listening on %s:%d", cfg.grpc.host, cfg.grpc.port)
    grpc_server.wait_for_termination()


if __name__ == "__main__":
    main()
