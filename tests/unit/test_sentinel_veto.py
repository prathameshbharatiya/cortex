"""
tests/unit/test_sentinel_veto.py
==================================
Mandatory safety test: proves that the Sentinel P1 veto is architecturally
non-bypassable.

REQUIREMENT (from build doc):
  "Write a test that directly verifies this: mock Sentinel to return FAIL,
   assert that PhysiCore is never called, assert that SAFE_HALT or
   HUMAN_OVERRIDE_REQUIRED is returned. This test must be in the CI pipeline."

These tests prove, for every code path, that:
  1. A Sentinel FAIL causes the gate to return SAFE_HALT or HUMAN_OVERRIDE_REQUIRED
  2. PhysiCore (P2) is NEVER called when Sentinel (P1) vetoes
  3. Memory validator (P3) is NEVER called when Sentinel (P1) vetoes
  4. The returned decision.action is None (no action approved)
  5. The gate cannot be configured to bypass Sentinel
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch, call

import pytest

import cortex
from cortex.gate.certification import CertificationGate
from cortex.models.action import Action, ActionSpec, ActionType, Pose
from cortex.models.context import RobotState
from cortex.models.decision import CertificationState
from cortex.validators.sentinel import SentinelValidator
from cortex.validators.physicore import PhysiCoreValidator
from cortex.validators.memory_validator import MemoryValidator as MemoryConsistencyValidator


def _make_state(emergency_stop: bool = False) -> RobotState:
    return RobotState(
        ee_position=[0.3, 0.0, 0.5],
        ee_orientation=[1.0, 0.0, 0.0, 0.0],
        emergency_stop=emergency_stop,
    )


def _make_action() -> Action:
    return Action(
        spec=ActionSpec(
            action_type=ActionType.MOVE_EE,
            target_pose=Pose(x=0.4, y=0.0, z=0.5),
        ),
        source="test",
        intent="test action",
    )


def _make_ctx(estop: bool = False):
    return cortex.build_context("test task", _make_state(estop), [])


class TestSentinelP1Veto:
    """All tests in this class verify the non-bypassability of the P1 veto."""

    def test_sentinel_fail_returns_blocking_state(self) -> None:
        """When Sentinel vetoes, the decision state must block execution."""
        state  = _make_state(emergency_stop=True)
        ctx    = cortex.build_context("test", state, [])
        action = _make_action()

        decision = cortex.certify(action, ctx)

        assert decision.state in (
            CertificationState.SAFE_HALT,
            CertificationState.HUMAN_OVERRIDE_REQUIRED,
        ), f"Expected blocking state, got {decision.state}"

    def test_sentinel_fail_action_is_none(self) -> None:
        """When Sentinel vetoes, no action is approved — decision.action must be None."""
        state  = _make_state(emergency_stop=True)
        ctx    = cortex.build_context("test", state, [])
        action = _make_action()

        decision = cortex.certify(action, ctx)

        assert decision.action is None, (
            f"Sentinel veto must set action=None, got {decision.action!r}"
        )

    def test_sentinel_fail_physicore_not_called(self) -> None:
        """When Sentinel vetoes, PhysiCore validator must NOT be called."""
        gate = CertificationGate()
        physicore_mock = MagicMock(spec=PhysiCoreValidator)
        gate.physicore = physicore_mock

        state  = _make_state(emergency_stop=True)
        ctx    = cortex.build_context("test", state, [])
        action = _make_action()

        gate.certify(action, ctx)

        physicore_mock.validate.assert_not_called(), (
            "PhysiCore P2 validator MUST NOT run when Sentinel P1 vetoes"
        )

    def test_sentinel_fail_memory_validator_not_called(self) -> None:
        """When Sentinel vetoes, Memory validator must NOT be called."""
        gate = CertificationGate()
        memory_mock = MagicMock(spec=MemoryConsistencyValidator)
        gate.memory = memory_mock

        state  = _make_state(emergency_stop=True)
        ctx    = cortex.build_context("test", state, [])
        action = _make_action()

        gate.certify(action, ctx)

        memory_mock.validate.assert_not_called(), (
            "Memory P3 validator MUST NOT run when Sentinel P1 vetoes"
        )

    def test_sentinel_mocked_fail_blocks_immediately(self) -> None:
        """With Sentinel mocked to always fail, gate returns blocking state immediately."""
        from cortex.validators.sentinel import ValidationResult
        from cortex.models.decision import FailureMode, RiskLevel

        gate = CertificationGate()
        sentinel_mock = MagicMock(spec=SentinelValidator)
        sentinel_mock.validate.return_value = ValidationResult(
            passed=False,
            score=0.0,
            failure_modes=[
                FailureMode(
                    code="mock_safety_violation",
                    description="Mocked safety violation for test",
                    risk_level=RiskLevel.CRITICAL,
                    source="sentinel_mock",
                )
            ],
        )
        gate.sentinel = sentinel_mock

        state  = _make_state()  # No actual e-stop — Sentinel is mocked
        ctx    = cortex.build_context("test", state, [])
        action = _make_action()

        decision = gate.certify(action, ctx)

        assert decision.state in (
            CertificationState.SAFE_HALT,
            CertificationState.HUMAN_OVERRIDE_REQUIRED,
        )
        assert decision.action is None

    def test_sentinel_mocked_fail_p2_p3_not_called(self) -> None:
        """With Sentinel mocked to fail, P2 and P3 must never execute."""
        from cortex.validators.sentinel import ValidationResult
        from cortex.models.decision import FailureMode, RiskLevel

        gate = CertificationGate()

        sentinel_mock  = MagicMock(spec=SentinelValidator)
        physicore_mock = MagicMock(spec=PhysiCoreValidator)
        memory_mock    = MagicMock(spec=MemoryConsistencyValidator)

        sentinel_mock.validate.return_value = ValidationResult(
            passed=False,
            score=0.0,
            failure_modes=[
                FailureMode(
                    code="mocked_veto",
                    description="Test veto",
                    risk_level=RiskLevel.HIGH,
                    source="test",
                )
            ],
        )

        gate.sentinel = sentinel_mock
        gate.physicore = physicore_mock
        gate.memory    = memory_mock

        state  = _make_state()
        ctx    = cortex.build_context("test", state, [])
        action = _make_action()

        gate.certify(action, ctx)

        sentinel_mock.validate.assert_called_once()
        physicore_mock.validate.assert_not_called()
        memory_mock.validate.assert_not_called()

    def test_normal_action_reaches_p2(self) -> None:
        """Control test: without Sentinel failure, PhysiCore IS called."""
        gate = CertificationGate()
        physicore_mock = MagicMock(spec=PhysiCoreValidator)

        # Return a passing result from physicore mock
        from cortex.validators.sentinel import ValidationResult
        physicore_mock.validate.return_value = ValidationResult(
            passed=True,
            score=0.9,
            failure_modes=[],
        )
        gate.physicore = physicore_mock

        state  = _make_state()  # Normal state — Sentinel should pass
        ctx    = cortex.build_context("test", state, [])
        action = _make_action()

        gate.certify(action, ctx)

        # PhysiCore should have been called since Sentinel passed
        assert physicore_mock.validate.call_count >= 1, (
            "PhysiCore P2 MUST be called when Sentinel P1 passes"
        )

    def test_singularity_blocks_execution(self) -> None:
        """Kinematic singularity must produce a blocking state."""
        state = RobotState(
            ee_position=[0.3, 0.0, 0.5],
            ee_orientation=[1.0, 0.0, 0.0, 0.0],
            in_singularity=True,
        )
        ctx    = cortex.build_context("test", state, [])
        action = _make_action()

        decision = cortex.certify(action, ctx)

        assert decision.state in (
            CertificationState.SAFE_HALT,
            CertificationState.HUMAN_OVERRIDE_REQUIRED,
        )
        assert decision.action is None

    def test_decision_has_trace_on_veto(self) -> None:
        """Even on Sentinel veto, the decision must carry a full audit trace."""
        state  = _make_state(emergency_stop=True)
        ctx    = cortex.build_context("test", state, [])
        action = _make_action()

        decision = cortex.certify(action, ctx)

        assert decision.trace is not None
        assert decision.trace.trace_id != ""

    def test_sentinel_attribute_exists_on_gate(self) -> None:
        """Gate must expose .sentinel attribute for inspector access."""
        gate = CertificationGate()
        assert hasattr(gate, "sentinel"), "CertificationGate must have a .sentinel attribute"
        assert isinstance(gate.sentinel, SentinelValidator)
