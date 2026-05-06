"""
Cortex Protocol
===============
The canonical wire format for all Cortex integrations.

Every integration — gRPC, HTTP, Python SDK, ROS 2 — speaks this protocol.
The protocol is defined here in Python dataclasses so it is language-
portable and does not require protobuf compilation to use.

When gRPC is available, these map directly to .proto message types.
When it is not, the HTTP and SDK layers use these directly as JSON.

Design principles:
  - Every request carries a request_id for tracing
  - Every response carries the same request_id back
  - Errors are structured, never bare strings
  - All timestamps are Unix epoch floats (seconds)
  - All positions are [x, y, z] float lists in metres
  - All quaternions are [w, x, y, z] float lists
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ── Request types ─────────────────────────────────────────────────────────────

@dataclass
class CertifyRequest:
    """
    Request to certify an action against a context.
    The core operation — this is what every integration calls.
    """
    request_id:      str

    # Action to certify
    action_type:     str             # "move_ee" | "grasp" | "place" | etc.
    source:          str             # which AI system proposed this
    intent:          str             # human-readable description

    # Target (for move_ee, grasp, place)
    target_position: list[float]     # [x, y, z] metres
    target_quaternion: list[float]   # [w, x, y, z]

    # Action constraints
    max_speed_ms:    float | None = None
    max_force_n:     float | None = None
    timeout_ms:      float        = 5000.0

    # Context
    task:            str           = ""
    ee_position:     list[float]   = field(default_factory=lambda: [0.0, 0.0, 0.0])
    ee_quaternion:   list[float]   = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])
    joint_positions: list[float]   = field(default_factory=list)
    joint_torques:   list[float]   = field(default_factory=list)
    gripper_open:    bool          = True
    humans_nearby:   bool          = False
    emergency_stop:  bool          = False
    in_singularity:  bool          = False

    # Optional overrides
    confidence_floor:   float      = 0.60
    validity_window_ms: float      = 500.0
    skip_physics:       bool       = False


@dataclass
class RecordOutcomeRequest:
    """Record the outcome of an executed certified action."""
    request_id:      str
    trace_id:        str
    ctx_id:          str
    action_id:       str
    task:            str
    outcome_class:   str    # "success" | "partial" | "failure" | "unsafe" | ...
    description:     str    = ""
    predicted_confidence: float = 0.5
    memory_record_ids: list[str] = field(default_factory=list)
    ee_position:     list[float] = field(default_factory=list)


@dataclass
class GetContextRequest:
    """Request a validated CTX without certifying an action."""
    request_id:  str
    task:        str
    ee_position: list[float]
    ee_quaternion: list[float]
    joint_positions: list[float] = field(default_factory=list)
    joint_torques:   list[float] = field(default_factory=list)
    gripper_open:    bool        = True
    humans_nearby:   bool        = False
    emergency_stop:  bool        = False
    top_k:           int         = 10
    skip_physics:    bool        = False


@dataclass
class HealthRequest:
    request_id: str


@dataclass
class StoreMemoryRequest:
    """Store a memory record via the Gateway."""
    request_id:   str
    source:       str
    memory_type:  str     # "episodic" | "semantic" | "spatial" | "procedural"
    content_text: str
    content:      dict    = field(default_factory=dict)
    confidence:   float   = 0.9


# ── Response types ────────────────────────────────────────────────────────────

@dataclass
class CertifyResponse:
    """The response to a CertifyRequest."""
    request_id:          str
    decision_id:         str
    trace_id:            str
    ctx_id:              str

    # The decision
    state:               str     # "EXECUTE" | "EXECUTE_WITH_CONSTRAINTS" | etc.
    approved:            bool
    reason:              str

    # Confidence (only if approved)
    success_probability: float | None = None
    stability_margin:    float | None = None
    worst_case_risk:     str | None   = None
    validity_window_ms:  float | None = None

    # Modified action (only if EXECUTE_WITH_CONSTRAINTS)
    modified_max_speed:  float | None = None
    modified_max_force:  float | None = None

    # Failure modes
    failure_modes:       list[dict]   = field(default_factory=list)

    # Timing
    certification_latency_ms: float   = 0.0
    assembly_latency_ms:      float   = 0.0

    # Error (only if something went wrong at the protocol level)
    error:               str | None   = None


@dataclass
class RecordOutcomeResponse:
    request_id:    str
    experience_id: str
    surprise_score: float
    error:         str | None = None


@dataclass
class GetContextResponse:
    request_id:         str
    ctx_id:             str
    task:               str
    memory_record_count: int
    constraint_count:   int
    physics_feasible:   bool
    memory_confidence:  float
    overall_confidence: float
    assembly_latency_ms: float
    error:              str | None = None


@dataclass
class HealthResponse:
    request_id:    str
    status:        str    # "ok" | "degraded" | "error"
    version:       str    = "0.1.0"
    uptime_s:      float  = 0.0
    adapter_count: int    = 0
    adapters:      list[dict] = field(default_factory=list)
    error:         str | None = None


@dataclass
class StoreMemoryResponse:
    request_id: str
    record_id:  str
    success:    bool
    error:      str | None = None


# ── Error codes ───────────────────────────────────────────────────────────────

class CortexError:
    INVALID_REQUEST  = "INVALID_REQUEST"
    CONTEXT_EXPIRED  = "CONTEXT_EXPIRED"
    GATEWAY_ERROR    = "GATEWAY_ERROR"
    INTERNAL_ERROR   = "INTERNAL_ERROR"
    NOT_INITIALISED  = "NOT_INITIALISED"
