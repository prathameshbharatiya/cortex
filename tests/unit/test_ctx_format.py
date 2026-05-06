"""
tests/unit/test_ctx_format.py
==============================
Tests for the .ctx wire format: serialisation, versioning, round-trips.
No external services required.
"""

from __future__ import annotations

import json
import time

import pytest
import msgpack

import cortex
from cortex.models.context import RobotState, SceneGraph, DetectedObject
from cortex.models.memory import MemoryRecord, MemoryType, OutcomeTag
from cortex.protocol.ctx_format import (
    CTXSerializer, CTX_VERSION, CTXVersionError, CTXFormatError,
)


def _make_state() -> RobotState:
    return RobotState(
        ee_position=[0.3, 0.0, 0.5],
        ee_orientation=[1.0, 0.0, 0.0, 0.0],
        joint_positions=[0.0, -0.5, 0.0, -1.5, 0.0, 1.0, 0.8],
    )


def _make_ctx(with_memory: bool = True):
    state  = _make_state()
    mems   = [] if not with_memory else [
        MemoryRecord(
            source="test_robot",
            memory_type=MemoryType.EPISODIC,
            content={"task": "pick block", "result": "success"},
            content_text="pick block succeeded",
            valid_at=time.time(),
            base_confidence=0.9,
            outcome_tag=OutcomeTag.SUCCESS,
            outcome_count=2,
            success_count=2,
        )
    ]
    return cortex.build_context("pick red block", state, mems)


class TestCTXSerialization:

    def test_to_bytes_returns_bytes(self) -> None:
        ctx = _make_ctx()
        raw = CTXSerializer().to_bytes(ctx)
        assert isinstance(raw, bytes)
        assert len(raw) > 0

    def test_round_trip_binary(self) -> None:
        """Serialise to bytes and back — all key fields preserved."""
        ctx = _make_ctx()
        s   = CTXSerializer()
        raw = s.to_bytes(ctx)
        ctx2 = s.from_bytes(raw)

        assert ctx2.ctx_id  == ctx.ctx_id
        assert ctx2.task    == ctx.task
        assert len(ctx2.robot_state.ee_position) == 3
        assert ctx2.robot_state.ee_position[0] == pytest.approx(0.3, abs=1e-6)

    def test_round_trip_with_memory(self) -> None:
        """Memory records survive round-trip."""
        ctx = _make_ctx(with_memory=True)
        s   = CTXSerializer()
        ctx2 = s.from_bytes(s.to_bytes(ctx))

        assert len(ctx2.memory_records) == len(ctx.memory_records)
        if ctx.memory_records:
            assert ctx2.memory_records[0].record_id == ctx.memory_records[0].record_id

    def test_to_dict_is_json_serialisable(self) -> None:
        ctx = _make_ctx()
        d   = CTXSerializer().to_dict(ctx)
        # Must not raise
        json.dumps(d, default=str)

    def test_version_field_in_envelope(self) -> None:
        ctx = _make_ctx()
        raw = CTXSerializer().to_bytes(ctx)
        envelope = msgpack.unpackb(raw, raw=False)
        assert envelope["v"] == CTX_VERSION
        assert "ts" in envelope
        assert "ctx" in envelope

    def test_from_bytes_wrong_major_version_raises(self) -> None:
        ctx = _make_ctx()
        raw = CTXSerializer().to_bytes(ctx)
        envelope = msgpack.unpackb(raw, raw=False)
        envelope["v"] = "99.0"
        bad_raw = msgpack.packb(envelope, use_bin_type=True)

        with pytest.raises(CTXVersionError):
            CTXSerializer().from_bytes(bad_raw)

    def test_from_bytes_garbage_raises(self) -> None:
        with pytest.raises(CTXFormatError):
            CTXSerializer().from_bytes(b"not msgpack at all!!")

    def test_json_round_trip(self) -> None:
        ctx  = _make_ctx()
        s    = CTXSerializer()
        text = s.to_json(ctx)
        ctx2 = s.from_json(text)
        assert ctx2.ctx_id == ctx.ctx_id

    def test_save_load_file(self, tmp_path) -> None:
        ctx  = _make_ctx()
        path = str(tmp_path / "test.ctx")
        s    = CTXSerializer()
        s.save(ctx, path)
        ctx2 = s.load(path)
        assert ctx2.ctx_id == ctx.ctx_id

    def test_cortex_api_save_load(self, tmp_path) -> None:
        ctx  = _make_ctx()
        path = str(tmp_path / "api.ctx")
        cortex.save_ctx(ctx, path)
        ctx2 = cortex.load_ctx(path)
        assert ctx2.ctx_id == ctx.ctx_id

    def test_confidence_fields_preserved(self) -> None:
        ctx = _make_ctx()
        s   = CTXSerializer()
        ctx2 = s.from_bytes(s.to_bytes(ctx))
        assert ctx2.confidence_floor   == pytest.approx(ctx.confidence_floor,   abs=1e-6)
        assert ctx2.validity_window_ms == pytest.approx(ctx.validity_window_ms, abs=1e-3)
        assert ctx2.memory_confidence  == pytest.approx(ctx.memory_confidence,  abs=1e-6)
        assert ctx2.timestamp          == pytest.approx(ctx.timestamp,          abs=1e-3)

    def test_robot_state_preserved(self) -> None:
        ctx = _make_ctx()
        s   = CTXSerializer()
        ctx2 = s.from_bytes(s.to_bytes(ctx))
        assert ctx2.robot_state.gripper_open == ctx.robot_state.gripper_open
        assert ctx2.robot_state.emergency_stop == ctx.robot_state.emergency_stop
