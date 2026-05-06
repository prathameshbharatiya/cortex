"""
Cortex Python SDK
=================
The integration layer for Python-based robot stacks.

This is the four-line API:

    from cortex.sdk import CortexClient

    client = CortexClient()                    # local in-process
    # or
    client = CortexClient(host="localhost", port=8765)  # remote HTTP

    # Step 1: get a certified context
    ctx = client.get_context(task, robot_state)

    # Step 2: your AI proposes an action
    action = your_planner.plan(ctx)

    # Step 3: certify it
    cert = client.certify(action, robot_state, task)

    # Step 4: execute only if certified
    if cert.approved:
        robot.execute(cert.action_params)
    else:
        handle(cert)

Two modes
---------
  In-process (default):
    Runs the full Cortex pipeline inside the same Python process.
    Zero network overhead. Best for single-robot, low-latency deployments.

  Remote (HTTP):
    Connects to a running Cortex HTTP server.
    Best for multi-robot deployments and when Cortex runs on edge hardware.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any

from cortex.integration.protocol import (
    CertifyRequest, CertifyResponse,
    RecordOutcomeRequest, RecordOutcomeResponse,
    GetContextRequest, GetContextResponse,
    StoreMemoryRequest, StoreMemoryResponse,
    HealthRequest, HealthResponse,
)
from cortex.integration.server import CortexServer


# ── High-level result types ───────────────────────────────────────────────────

@dataclass
class CertResult:
    """
    The result of client.certify().
    Simple, flat structure — no nested Cortex models needed.
    """
    approved:       bool
    state:          str
    reason:         str
    trace_id:       str
    ctx_id:         str

    # Confidence (only if approved)
    confidence:     float | None    = None
    stability_margin: float | None  = None
    risk_level:     str | None      = None
    valid_for_ms:   float | None    = None

    # Modified action params (only if EXECUTE_WITH_CONSTRAINTS)
    max_speed_ms:   float | None    = None
    max_force_n:    float | None    = None

    # Warnings from failure history
    warnings:       list[str]       = None

    # Timing
    latency_ms:     float           = 0.0

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []

    @property
    def execute(self) -> bool:
        return self.state == "EXECUTE"

    @property
    def execute_with_constraints(self) -> bool:
        return self.state == "EXECUTE_WITH_CONSTRAINTS"

    @property
    def replan(self) -> bool:
        return self.state == "REPLAN_REQUIRED"

    @property
    def halt(self) -> bool:
        return self.state == "SAFE_HALT"

    @property
    def needs_human(self) -> bool:
        return self.state == "HUMAN_OVERRIDE_REQUIRED"

    def __repr__(self) -> str:
        conf = f"{self.confidence:.0%}" if self.confidence else "n/a"
        return (
            f"CertResult(state={self.state}, "
            f"confidence={conf}, "
            f"latency={self.latency_ms:.1f}ms)"
        )


@dataclass
class ContextResult:
    """The result of client.get_context()."""
    ctx_id:              str
    task:                str
    memory_records:      int
    constraints:         int
    physics_feasible:    bool
    memory_confidence:   float
    overall_confidence:  float
    assembly_latency_ms: float


# ── In-process transport ──────────────────────────────────────────────────────

class _InProcessTransport:
    """Calls CortexServer directly in the same process."""

    def __init__(self, server: CortexServer) -> None:
        self._server = server

    def certify(self, req: CertifyRequest) -> CertifyResponse:
        return self._server.certify(req)

    def record_outcome(self, req: RecordOutcomeRequest) -> RecordOutcomeResponse:
        return self._server.record_outcome(req)

    def get_context(self, req: GetContextRequest) -> GetContextResponse:
        return self._server.get_context(req)

    def store_memory(self, req: StoreMemoryRequest) -> StoreMemoryResponse:
        return self._server.store_memory(req)

    def health(self, req: HealthRequest) -> HealthResponse:
        return self._server.health(req)


# ── HTTP transport ────────────────────────────────────────────────────────────

class _HttpTransport:
    """Calls a remote CortexServer over HTTP."""

    def __init__(self, base_url: str, timeout_s: float = 5.0) -> None:
        self.base_url   = base_url.rstrip("/")
        self.timeout    = timeout_s
        self._session: Any = None

    def _get_session(self) -> Any:
        if self._session is None:
            try:
                import requests
                self._session = requests.Session()
            except ImportError:
                raise ImportError("requests is required for remote mode: pip install requests")
        return self._session

    def _post(self, path: str, data: dict) -> dict:
        session = self._get_session()
        resp    = session.post(
            f"{self.base_url}{path}",
            json=data,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def _get(self, path: str) -> dict:
        session = self._get_session()
        resp    = session.get(f"{self.base_url}{path}", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def certify(self, req: CertifyRequest) -> CertifyResponse:
        d = req.__dict__.copy()
        d.pop("request_id", None)
        r = self._post("/certify", d)
        return CertifyResponse(**{k: v for k, v in r.items()
                                  if k in CertifyResponse.__dataclass_fields__})

    def record_outcome(self, req: RecordOutcomeRequest) -> RecordOutcomeResponse:
        d = req.__dict__.copy()
        d.pop("request_id", None)
        r = self._post("/record", d)
        return RecordOutcomeResponse(**{k: v for k, v in r.items()
                                        if k in RecordOutcomeResponse.__dataclass_fields__})

    def get_context(self, req: GetContextRequest) -> GetContextResponse:
        d = req.__dict__.copy()
        d.pop("request_id", None)
        r = self._post("/context", d)
        return GetContextResponse(**{k: v for k, v in r.items()
                                     if k in GetContextResponse.__dataclass_fields__})

    def store_memory(self, req: StoreMemoryRequest) -> StoreMemoryResponse:
        d = req.__dict__.copy()
        d.pop("request_id", None)
        r = self._post("/memory", d)
        return StoreMemoryResponse(**{k: v for k, v in r.items()
                                      if k in StoreMemoryResponse.__dataclass_fields__})

    def health(self, req: HealthRequest) -> HealthResponse:
        r = self._get("/health")
        return HealthResponse(**{k: v for k, v in r.items()
                                 if k in HealthResponse.__dataclass_fields__})


# ── Client ────────────────────────────────────────────────────────────────────

class CortexClient:
    """
    The Cortex Python SDK client.

    In-process (default):
        client = CortexClient()

    Remote HTTP:
        client = CortexClient(host="localhost", port=8765)

    With explicit server:
        server = CortexServer.create()
        client = CortexClient(server=server)
    """

    def __init__(
        self,
        server:  CortexServer | None = None,
        host:    str | None          = None,
        port:    int                 = 8765,
        timeout: float               = 5.0,
    ) -> None:
        if host is not None:
            self._transport = _HttpTransport(f"http://{host}:{port}", timeout)
            self._server    = None
        else:
            if server is None:
                server = CortexServer.create()
            self._server    = server
            self._transport = _InProcessTransport(server)

    # ── Core API — four lines ─────────────────────────────────────────────────

    def certify(
        self,
        action_type:      str,
        target_position:  list[float],
        task:             str           = "",
        ee_position:      list[float]   = None,
        ee_quaternion:    list[float]   = None,
        joint_positions:  list[float]   = None,
        joint_torques:    list[float]   = None,
        target_quaternion: list[float]  = None,
        max_speed_ms:     float | None  = None,
        max_force_n:      float | None  = None,
        source:           str           = "unknown",
        intent:           str           = "",
        gripper_open:     bool          = True,
        humans_nearby:    bool          = False,
        emergency_stop:   bool          = False,
        in_singularity:   bool          = False,
        confidence_floor: float         = 0.60,
        skip_physics:     bool          = False,
    ) -> CertResult:
        """
        Certify an action.

        Returns a CertResult with .approved, .state, .confidence, etc.

        The one method every integration calls.
        """
        t0  = time.perf_counter()
        req = CertifyRequest(
            request_id=self._rid(),
            action_type=action_type,
            source=source,
            intent=intent or task,
            target_position=target_position,
            target_quaternion=target_quaternion or [1.0, 0.0, 0.0, 0.0],
            max_speed_ms=max_speed_ms,
            max_force_n=max_force_n,
            task=task,
            ee_position=ee_position or [0.0, 0.0, 0.0],
            ee_quaternion=ee_quaternion or [1.0, 0.0, 0.0, 0.0],
            joint_positions=joint_positions or [],
            joint_torques=joint_torques or [],
            gripper_open=gripper_open,
            humans_nearby=humans_nearby,
            emergency_stop=emergency_stop,
            in_singularity=in_singularity,
            confidence_floor=confidence_floor,
            skip_physics=skip_physics,
        )
        resp     = self._transport.certify(req)
        total_ms = (time.perf_counter() - t0) * 1000.0

        warnings = [
            fm["description"]
            for fm in (resp.failure_modes or [])
            if fm.get("risk_level") in ("medium", "high", "critical")
        ]

        return CertResult(
            approved=resp.approved,
            state=resp.state,
            reason=resp.reason,
            trace_id=resp.trace_id,
            ctx_id=resp.ctx_id,
            confidence=resp.success_probability,
            stability_margin=resp.stability_margin,
            risk_level=resp.worst_case_risk,
            valid_for_ms=resp.validity_window_ms,
            max_speed_ms=resp.modified_max_speed,
            max_force_n=resp.modified_max_force,
            warnings=warnings,
            latency_ms=total_ms,
        )

    def get_context(
        self,
        task:           str,
        ee_position:    list[float]  = None,
        ee_quaternion:  list[float]  = None,
        joint_positions: list[float] = None,
        joint_torques:  list[float]  = None,
        humans_nearby:  bool         = False,
        emergency_stop: bool         = False,
        top_k:          int          = 10,
    ) -> ContextResult:
        req  = GetContextRequest(
            request_id=self._rid(),
            task=task,
            ee_position=ee_position or [0.0, 0.0, 0.0],
            ee_quaternion=ee_quaternion or [1.0, 0.0, 0.0, 0.0],
            joint_positions=joint_positions or [],
            joint_torques=joint_torques or [],
            humans_nearby=humans_nearby,
            emergency_stop=emergency_stop,
            top_k=top_k,
        )
        resp = self._transport.get_context(req)
        return ContextResult(
            ctx_id=resp.ctx_id,
            task=resp.task,
            memory_records=resp.memory_record_count,
            constraints=resp.constraint_count,
            physics_feasible=resp.physics_feasible,
            memory_confidence=resp.memory_confidence,
            overall_confidence=resp.overall_confidence,
            assembly_latency_ms=resp.assembly_latency_ms,
        )

    def record_outcome(
        self,
        cert:          CertResult,
        outcome:       str,              # "success" | "failure" | "partial" | ...
        task:          str       = "",
        description:   str       = "",
    ) -> float:
        """
        Record the outcome of an executed action.
        Returns the surprise score.
        """
        req  = RecordOutcomeRequest(
            request_id=self._rid(),
            trace_id=cert.trace_id,
            ctx_id=cert.ctx_id,
            action_id="",
            task=task,
            outcome_class=outcome,
            description=description,
            predicted_confidence=cert.confidence or 0.5,
        )
        resp = self._transport.record_outcome(req)
        return resp.surprise_score

    def store_memory(
        self,
        content_text: str,
        source:       str   = "sdk",
        memory_type:  str   = "episodic",
        content:      dict  = None,
        confidence:   float = 0.9,
    ) -> str:
        """Store a memory record. Returns record_id."""
        req  = StoreMemoryRequest(
            request_id=self._rid(),
            source=source,
            memory_type=memory_type,
            content_text=content_text,
            content=content or {},
            confidence=confidence,
        )
        resp = self._transport.store_memory(req)
        return resp.record_id

    def health(self) -> dict:
        resp = self._transport.health(HealthRequest(request_id=self._rid()))
        return resp.__dict__

    def stop(self) -> None:
        if self._server:
            self._server.stop()

    def __enter__(self) -> "CortexClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop()

    @staticmethod
    def _rid() -> str:
        return str(uuid.uuid4())[:8]
