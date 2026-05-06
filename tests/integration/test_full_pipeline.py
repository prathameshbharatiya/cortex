"""
tests/integration/test_full_pipeline.py
========================================
End-to-end integration test for the full Cortex certification pipeline.
Verifies the complete path:
  build_context() → certify() → CertificationDecision

No external services required — uses only in-process memory.
"""

from __future__ import annotations

import time
import pytest

import cortex
from cortex.models.action import ActionType, ActionSpec, Action, Pose
from cortex.models.context import RobotState
from cortex.models.decision import CertificationState
from cortex.models.memory import MemoryRecord, MemoryType, OutcomeTag


def _make_robot_state() -> RobotState:
    return RobotState(
        ee_position=[0.3, 0.0, 0.5],
        ee_orientation=[1.0, 0.0, 0.0, 0.0],
        ee_velocity=[0.0, 0.0, 0.0],
        joint_positions=[0.0, -0.5, 0.0, -1.5, 0.0, 1.0, 0.8],
        joint_velocities=[0.0] * 7,
        joint_torques=[0.0] * 7,
        gripper_open=True,
    )


def _make_memory_record(task: str = "pick red block") -> MemoryRecord:
    return MemoryRecord(
        source="test_robot",
        memory_type=MemoryType.EPISODIC,
        content={"task": task, "result": "success", "position": [0.3, 0.0, 0.5]},
        content_text=f"Successfully completed: {task}",
        valid_at=time.time(),
        base_confidence=0.85,
        outcome_tag=OutcomeTag.SUCCESS,
        outcome_count=3,
        success_count=3,
    )


def _make_action(target: list[float] | None = None) -> Action:
    pose = Pose(x=(target or [0.4, 0.0, 0.3])[0],
                y=(target or [0.4, 0.0, 0.3])[1],
                z=(target or [0.4, 0.0, 0.3])[2])
    spec = ActionSpec(
        action_type=ActionType.MOVE_EE,
        target_pose=pose,
    )
    return Action(spec=spec, source="test_planner", intent="pick red block")


class TestFullPipeline:

    def test_basic_certify_returns_decision(self) -> None:
        """Happy path: build context, certify action, get valid decision."""
        state  = _make_robot_state()
        mem    = _make_memory_record()
        ctx    = cortex.build_context("pick red block", state, [mem])
        action = _make_action()
        decision = cortex.certify(action, ctx)

        assert decision.state in list(CertificationState)
        assert decision.trace is not None
        assert decision.confidence is not None

    def test_context_contains_memory_records(self) -> None:
        state = _make_robot_state()
        mem   = _make_memory_record()
        ctx   = cortex.build_context("pick red block", state, [mem])

        assert len(ctx.memory_records) > 0

    def test_emergency_stop_blocks_execution(self) -> None:
        """E-stop robot state must block execution via Sentinel P1 veto."""
        state = RobotState(
            ee_position=[0.3, 0.0, 0.5],
            ee_orientation=[1.0, 0.0, 0.0, 0.0],
            emergency_stop=True,
        )
        ctx    = cortex.build_context("pick red block", state, [])
        action = _make_action()
        decision = cortex.certify(action, ctx)

        # E-stop is CRITICAL — gate returns HUMAN_OVERRIDE_REQUIRED (operator must clear)
        assert decision.state in (
            CertificationState.SAFE_HALT,
            CertificationState.HUMAN_OVERRIDE_REQUIRED,
        )
        assert decision.state != CertificationState.EXECUTE

    def test_certification_decision_has_trace(self) -> None:
        state    = _make_robot_state()
        ctx      = cortex.build_context("move arm", state, [])
        action   = _make_action()
        decision = cortex.certify(action, ctx)

        assert decision.trace is not None
        assert decision.trace.trace_id != ""

    def test_no_memory_still_certifies(self) -> None:
        """Certify without any memory records (degraded confidence path)."""
        state    = _make_robot_state()
        ctx      = cortex.build_context("pick blue block", state, [])
        action   = _make_action()
        decision = cortex.certify(action, ctx)

        assert decision.state in list(CertificationState)

    def test_context_builder_wired_to_engine(self) -> None:
        """ContextBuilder must use ContextEngine (not manual assembly)."""
        from cortex.gate.builder import ContextBuilder
        from cortex.context.engine import ContextEngine

        builder = ContextBuilder()
        assert isinstance(builder._engine, ContextEngine), (
            "ContextBuilder must delegate to ContextEngine"
        )

    def test_multiple_certify_calls_independent(self) -> None:
        """Each certify() call is independent — no state leak."""
        state = _make_robot_state()
        ctx1  = cortex.build_context("task A", state, [_make_memory_record("task A")])
        ctx2  = cortex.build_context("task B", state, [_make_memory_record("task B")])

        d1 = cortex.certify(_make_action([0.3, 0.0, 0.4]), ctx1)
        d2 = cortex.certify(_make_action([0.5, 0.1, 0.3]), ctx2)

        assert d1.trace.trace_id != d2.trace.trace_id

    def test_certify_latency_under_100ms(self) -> None:
        """The full pipeline must complete in under 100ms for the test harness."""
        state  = _make_robot_state()
        mem    = _make_memory_record()
        ctx    = cortex.build_context("pick red block", state, [mem])
        action = _make_action()

        t0 = time.perf_counter()
        cortex.certify(action, ctx)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        assert elapsed_ms < 100.0, f"certify() took {elapsed_ms:.1f}ms — exceeds 100ms budget"
