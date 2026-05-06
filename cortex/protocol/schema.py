"""
CTX JSON Schema Generator
==========================
Generates a formal JSON Schema (draft-07) for the .ctx wire format.
This is the artifact submitted to ISO for standardisation.

Usage
-----
    from cortex.protocol.schema import generate_json_schema
    schema = generate_json_schema()

    # Or from CLI:
    python -m cortex.protocol.schema > docs/ctx-schema-v1.json
"""

from __future__ import annotations

import json


def generate_json_schema() -> dict:
    """
    Generate the JSON Schema for the CTX v1.0 wire format.
    This schema is suitable for ISO submission and third-party tooling.
    """
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "$id": "https://cortex.ai/schemas/ctx-v1.0.json",
        "title": "CTX — Cortex Context Packet",
        "description": (
            "The universal context packet format for Physical AI execution "
            "certification. Versioned, language-agnostic, signed."
        ),
        "type": "object",
        "required": ["v", "ctx", "ts"],
        "additionalProperties": False,
        "properties": {
            "v": {
                "type": "string",
                "description": "CTX format version (semver major.minor)",
                "pattern": r"^\d+\.\d+$",
                "examples": ["1.0"],
            },
            "ts": {
                "type": "integer",
                "description": "Nanosecond timestamp when this packet was packed (Unix epoch)",
                "minimum": 0,
            },
            "ctx": _ctx_schema(),
        },
    }


def _ctx_schema() -> dict:
    return {
        "type": "object",
        "description": "The certified context (CTX) packet",
        "required": ["ctx_id", "task", "robot_state"],
        "properties": {
            "ctx_id": {
                "type": "string",
                "format": "uuid",
                "description": "Unique ID for this context packet",
            },
            "task": {
                "type": "string",
                "description": "Human-readable description of the task to be certified",
            },
            "assembly_latency_ms": {
                "type": "number",
                "description": "Time taken to assemble this CTX (milliseconds)",
                "minimum": 0,
            },
            "memory_confidence": {
                "type": "number",
                "description": "Aggregate confidence from memory subsystem",
                "minimum": 0.0,
                "maximum": 1.0,
            },
            "confidence_floor": {
                "type": "number",
                "description": "Minimum confidence required for EXECUTE certification",
                "minimum": 0.0,
                "maximum": 1.0,
            },
            "timestamp": {
                "type": "number",
                "description": "Unix timestamp when this CTX was created",
            },
            "validity_window_ms": {
                "type": "number",
                "description": "How long this CTX is valid (milliseconds)",
                "minimum": 0,
            },
            "is_degraded": {
                "type": "boolean",
                "description": "True if CTX was assembled with reduced capability",
            },
            "robot_state": _robot_state_schema(),
            "scene_graph":       _scene_graph_schema(),
            "memory_records":    {
                "type": "array",
                "items": _memory_record_schema(),
                "description": "Certified memory records used in this context",
            },
            "safety_constraints": {
                "type": "array",
                "items": _safety_constraint_schema(),
                "description": "Active safety constraints for this certification",
            },
            "physics_horizon": _physics_horizon_schema(),
        },
    }


def _robot_state_schema() -> dict:
    return {
        "type": "object",
        "description": "Live proprioceptive state of the robot at CTX assembly time",
        "required": ["ee_position", "ee_orientation"],
        "properties": {
            "ee_position": {
                "type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3,
                "description": "End-effector position [x, y, z] in metres",
            },
            "ee_orientation": {
                "type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4,
                "description": "End-effector orientation quaternion [w, x, y, z]",
            },
            "ee_velocity": {
                "type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3,
            },
            "joint_positions":  {"type": "array", "items": {"type": "number"}},
            "joint_velocities": {"type": "array", "items": {"type": "number"}},
            "joint_torques":    {"type": "array", "items": {"type": "number"}},
            "gripper_open":     {"type": "boolean"},
            "gripper_force_n":  {"type": "number", "minimum": 0},
            "is_moving":        {"type": "boolean"},
            "in_singularity":   {"type": "boolean"},
            "emergency_stop":   {"type": "boolean"},
            "timestamp":        {"type": "number", "description": "Unix timestamp of sensor reading"},
        },
    }


def _scene_graph_schema() -> dict:
    return {
        "type": "object",
        "description": "Detected objects and scene structure",
        "properties": {
            "objects": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["object_id", "label"],
                    "properties": {
                        "object_id":  {"type": "string"},
                        "label":      {"type": "string"},
                        "position":   {"type": "array", "items": {"type": "number"}, "minItems": 3},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "attributes": {"type": "object"},
                    },
                },
            },
            "scene_confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "frame_id":         {"type": "string"},
        },
    }


def _memory_record_schema() -> dict:
    return {
        "type": "object",
        "required": ["record_id", "source", "memory_type"],
        "properties": {
            "record_id":       {"type": "string", "format": "uuid"},
            "source":          {"type": "string"},
            "memory_type":     {
                "type": "string",
                "enum": ["episodic", "semantic", "procedural", "spatial", "social"],
            },
            "content":         {"type": "object"},
            "content_text":    {"type": "string"},
            "valid_at":        {"type": "number"},
            "invalid_at":      {"type": ["number", "null"]},
            "base_confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "outcome_tag":     {
                "type": "string",
                "enum": ["success", "failure", "partial", "unknown"],
            },
            "outcome_count":   {"type": "integer", "minimum": 0},
            "success_count":   {"type": "integer", "minimum": 0},
        },
    }


def _safety_constraint_schema() -> dict:
    return {
        "type": "object",
        "required": ["constraint_id", "description"],
        "properties": {
            "constraint_id": {"type": "string"},
            "description":   {"type": "string"},
            "is_hard":       {"type": "boolean"},
            "value":         {"type": ["number", "null"]},
            "unit":          {"type": "string"},
            "source":        {"type": "string"},
        },
    }


def _physics_horizon_schema() -> dict:
    return {
        "type": ["object", "null"],
        "description": "Forward physics simulation result",
        "properties": {
            "steps":           {"type": "integer", "minimum": 0},
            "dt_seconds":      {"type": "number", "minimum": 0},
            "feasible":        {"type": "boolean"},
            "min_clearance_m": {"type": "number"},
            "max_force_n":     {"type": "number"},
            "margin":          {"type": "number"},
            "positions":       {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
            "velocities":      {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
            "warnings":        {"type": "array", "items": {"type": "string"}},
        },
    }


if __name__ == "__main__":
    schema = generate_json_schema()
    print(json.dumps(schema, indent=2))
