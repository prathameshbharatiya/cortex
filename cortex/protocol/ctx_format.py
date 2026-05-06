"""
CTX Wire Format — v1.0
======================
The .ctx format is the universal interface between AI intention and physical
execution.  It is versioned, language-agnostic, and cryptographically signed.

Format:  MessagePack binary (compact, schema-preserving, fast)
Fallback: JSON (human-readable, for debugging and tooling)

Signing: Ed25519 (32-byte key, 64-byte signature, ~0.3ms verify)
         Handled by CTXSigner in signing.py.

Versioning
----------
Every packet carries a "v" field at the top level.
  Current: CTX_VERSION = "1.0"
  Older packets are migrated by _migrate_dict() before deserialising.
  Future packets with higher minor versions are accepted (forward compat).
  Different major versions raise CTXVersionError.

Wire layout (MessagePack dict)
------------------------------
  {
    "v":   "1.0",                  # format version
    "ctx": { ... Context fields }, # serialised Context
    "ts":  1234567890123456789,    # timestamp_ns when packed
  }
"""

from __future__ import annotations

import json
import time
from typing import Any

import msgpack  # pip install msgpack

from cortex.models.context import (
    Context, RobotState, SceneGraph, SafetyConstraint, PhysicsHorizon,
    DetectedObject,
)
from cortex.models.memory import MemoryRecord, MemoryType, OutcomeTag

CTX_VERSION = "1.0"
_MAJOR = int(CTX_VERSION.split(".")[0])


class CTXVersionError(Exception):
    """Raised when a .ctx packet has an incompatible major version."""


class CTXFormatError(Exception):
    """Raised when a .ctx packet has an invalid structure."""


class CTXSerializer:
    """
    Serialise / deserialise Context to/from the .ctx binary wire format.

    Thread-safe — all methods are stateless.
    """

    # ── Binary (MessagePack) ─────────────────────────────────────────────────

    def to_bytes(self, ctx: Context) -> bytes:
        """Serialise Context to .ctx binary (MessagePack)."""
        envelope = self._make_envelope(ctx)
        return msgpack.packb(envelope, use_bin_type=True)

    def from_bytes(self, data: bytes) -> Context:
        """Deserialise .ctx binary to Context."""
        try:
            envelope: dict[str, Any] = msgpack.unpackb(data, raw=False)
        except Exception as exc:
            raise CTXFormatError(f"Cannot parse .ctx binary: {exc}") from exc
        return self._from_envelope(envelope)

    # ── JSON (debugging / tooling) ───────────────────────────────────────────

    def to_dict(self, ctx: Context) -> dict[str, Any]:
        """Serialise to JSON-compatible dict (for debugging and ISO schema)."""
        return self._make_envelope(ctx)

    def from_dict(self, data: dict[str, Any]) -> Context:
        """Deserialise from JSON-compatible dict."""
        return self._from_envelope(data)

    def to_json(self, ctx: Context) -> str:
        return json.dumps(self.to_dict(ctx), default=str)

    def from_json(self, text: str) -> Context:
        return self.from_dict(json.loads(text))

    # ── File I/O ──────────────────────────────────────────────────────────────

    def save(self, ctx: Context, path: str) -> None:
        """Save Context to a .ctx file (binary MessagePack)."""
        with open(path, "wb") as fh:
            fh.write(self.to_bytes(ctx))

    def load(self, path: str) -> Context:
        """Load and validate a .ctx file."""
        with open(path, "rb") as fh:
            return self.from_bytes(fh.read())

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _make_envelope(self, ctx: Context) -> dict[str, Any]:
        return {
            "v":   CTX_VERSION,
            "ctx": self._ctx_to_dict(ctx),
            "ts":  time.time_ns(),
        }

    def _from_envelope(self, envelope: dict[str, Any]) -> Context:
        if "v" not in envelope or "ctx" not in envelope:
            raise CTXFormatError("Missing required fields 'v' or 'ctx'")

        version: str = envelope["v"]
        major = int(version.split(".")[0])
        if major != _MAJOR:
            raise CTXVersionError(
                f"Incompatible .ctx version {version!r} "
                f"(this runtime supports major={_MAJOR})"
            )

        raw = envelope["ctx"]
        if major == 1:
            raw = _migrate_v1(raw)  # idempotent

        return self._ctx_from_dict(raw)

    # ── Context serialisation ─────────────────────────────────────────────────

    @staticmethod
    def _ctx_to_dict(ctx: Context) -> dict[str, Any]:
        d: dict[str, Any] = {
            "ctx_id":             ctx.ctx_id,
            "task":               ctx.task,
            "assembly_latency_ms": ctx.assembly_latency_ms,
            "memory_confidence":  ctx.memory_confidence,
            "confidence_floor":   ctx.confidence_floor,
            "timestamp":          ctx.timestamp,
            "validity_window_ms": ctx.validity_window_ms,
            "robot_state":        _robot_state_to_dict(ctx.robot_state),
            "scene_graph":        _scene_graph_to_dict(ctx.scene_graph),
            "memory_records":     [_memory_record_to_dict(r) for r in ctx.memory_records],
            "safety_constraints": [_constraint_to_dict(c) for c in ctx.safety_constraints],
        }
        if ctx.physics_horizon is not None:
            d["physics_horizon"] = _physics_horizon_to_dict(ctx.physics_horizon)
        return d

    @staticmethod
    def _ctx_from_dict(d: dict[str, Any]) -> Context:
        robot_state = _robot_state_from_dict(d["robot_state"])
        scene_graph = _scene_graph_from_dict(d.get("scene_graph", {}))
        memory_records = [_memory_record_from_dict(r) for r in d.get("memory_records", [])]
        safety_constraints = [_constraint_from_dict(c) for c in d.get("safety_constraints", [])]
        physics_horizon: PhysicsHorizon | None = None
        if "physics_horizon" in d and d["physics_horizon"]:
            physics_horizon = _physics_horizon_from_dict(d["physics_horizon"])

        return Context(
            ctx_id=d["ctx_id"],
            task=d["task"],
            robot_state=robot_state,
            scene_graph=scene_graph,
            memory_records=memory_records,
            safety_constraints=safety_constraints,
            physics_horizon=physics_horizon,
            assembly_latency_ms=float(d.get("assembly_latency_ms", 0.0)),
            memory_confidence=float(d.get("memory_confidence", 1.0)),
            confidence_floor=float(d.get("confidence_floor", 0.6)),
            timestamp=float(d.get("timestamp", time.time())),
            validity_window_ms=float(d.get("validity_window_ms", 500.0)),
        )


# ── Per-model serialisation helpers ──────────────────────────────────────────

def _robot_state_to_dict(rs: RobotState) -> dict[str, Any]:
    return {
        "ee_position":       rs.ee_position,
        "ee_orientation":    rs.ee_orientation,
        "ee_velocity":       rs.ee_velocity,
        "joint_positions":   rs.joint_positions,
        "joint_velocities":  rs.joint_velocities,
        "joint_torques":     rs.joint_torques,
        "gripper_open":      rs.gripper_open,
        "gripper_force_n":   rs.gripper_force_n,
        "is_moving":         rs.is_moving,
        "in_singularity":    rs.in_singularity,
        "emergency_stop":    rs.emergency_stop,
        "timestamp":         rs.timestamp,
    }


def _robot_state_from_dict(d: dict[str, Any]) -> RobotState:
    return RobotState(
        ee_position=d.get("ee_position", [0.0, 0.0, 0.0]),
        ee_orientation=d.get("ee_orientation", [1.0, 0.0, 0.0, 0.0]),
        ee_velocity=d.get("ee_velocity", [0.0, 0.0, 0.0]),
        joint_positions=d.get("joint_positions", []),
        joint_velocities=d.get("joint_velocities", []),
        joint_torques=d.get("joint_torques", []),
        gripper_open=d.get("gripper_open", True),
        gripper_force_n=d.get("gripper_force_n", 0.0),
        is_moving=d.get("is_moving", False),
        in_singularity=d.get("in_singularity", False),
        emergency_stop=d.get("emergency_stop", False),
        timestamp=d.get("timestamp", time.time()),
    )


def _scene_graph_to_dict(sg: SceneGraph) -> dict[str, Any]:
    return {
        "objects": [
            {
                "object_id":   o.object_id,
                "label":       o.label,
                "position":    o.position,
                "dimensions":  o.dimensions,
                "confidence":  o.confidence,
                "is_grasped":  o.is_grasped,
                "is_occluded": o.is_occluded,
            }
            for o in sg.objects
        ],
        "workspace_bounds":     sg.workspace_bounds,
        "humans_nearby":        sg.humans_nearby,
        "obstruction_detected": sg.obstruction_detected,
        "timestamp":            sg.timestamp,
    }


def _scene_graph_from_dict(d: dict[str, Any]) -> SceneGraph:
    objects = [
        DetectedObject(
            object_id=o.get("object_id", ""),
            label=o.get("label", ""),
            position=o.get("position", [0.0, 0.0, 0.0]),
            dimensions=o.get("dimensions", [0.0, 0.0, 0.0]),
            confidence=o.get("confidence", 1.0),
            is_grasped=o.get("is_grasped", False),
            is_occluded=o.get("is_occluded", False),
        )
        for o in d.get("objects", [])
    ]
    return SceneGraph(
        objects=objects,
        workspace_bounds=d.get("workspace_bounds", {}),
        humans_nearby=d.get("humans_nearby", False),
        obstruction_detected=d.get("obstruction_detected", False),
        timestamp=d.get("timestamp", time.time()),
    )


def _memory_record_to_dict(r: MemoryRecord) -> dict[str, Any]:
    return {
        "record_id":       r.record_id,
        "source":          r.source,
        "memory_type":     r.memory_type.value,
        "content":         r.content,
        "content_text":    r.content_text,
        "valid_at":        r.valid_at,
        "invalid_at":      r.invalid_at,
        "base_confidence": r.base_confidence,
        "outcome_tag":     r.outcome_tag.value,
        "outcome_count":   r.outcome_count,
        "success_count":   r.success_count,
    }


def _memory_record_from_dict(d: dict[str, Any]) -> MemoryRecord:
    return MemoryRecord(
        record_id=d.get("record_id", ""),
        source=d.get("source", ""),
        memory_type=MemoryType(d.get("memory_type", "episodic")),
        content=d.get("content", {}),
        content_text=d.get("content_text", ""),
        valid_at=d.get("valid_at", time.time()),
        invalid_at=d.get("invalid_at"),
        base_confidence=d.get("base_confidence", 1.0),
        outcome_tag=OutcomeTag(d.get("outcome_tag", "unknown")),
        outcome_count=d.get("outcome_count", 0),
        success_count=d.get("success_count", 0),
    )


def _constraint_to_dict(c: SafetyConstraint) -> dict[str, Any]:
    return {
        "constraint_id":   c.constraint_id,
        "description":     c.description,
        "constraint_type": c.constraint_type,
        "parameters":      c.parameters,
        "veto_power":      c.veto_power,
        "source":          c.source,
    }


def _constraint_from_dict(d: dict[str, Any]) -> SafetyConstraint:
    return SafetyConstraint(
        constraint_id=d.get("constraint_id", ""),
        description=d.get("description", ""),
        constraint_type=d.get("constraint_type", "custom"),
        parameters=d.get("parameters", {}),
        veto_power=d.get("veto_power", True),
        source=d.get("source", ""),
    )


def _physics_horizon_to_dict(ph: PhysicsHorizon) -> dict[str, Any]:
    return {
        "steps":                   ph.steps,
        "feasible":                ph.feasible,
        "min_stability_margin":    ph.min_stability_margin,
        "predicted_final_state":   ph.predicted_final_state,
        "collision_risk_steps":    ph.collision_risk_steps,
        "singularity_risk_steps":  ph.singularity_risk_steps,
    }


def _physics_horizon_from_dict(d: dict[str, Any]) -> PhysicsHorizon:
    return PhysicsHorizon(
        steps=d.get("steps", 12),
        feasible=d.get("feasible", True),
        min_stability_margin=d.get("min_stability_margin", 1.0),
        predicted_final_state=d.get("predicted_final_state", {}),
        collision_risk_steps=d.get("collision_risk_steps", []),
        singularity_risk_steps=d.get("singularity_risk_steps", []),
    )


def _migrate_v1(raw: dict[str, Any]) -> dict[str, Any]:
    """Idempotent migration for v1.x packets. Currently a no-op."""
    return raw
