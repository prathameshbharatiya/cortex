"""
Tests for Phase 10: Simulation Harness
"""

from __future__ import annotations

import pytest

from cortex.models.action import Action, ActionSpec, ActionType, Pose, ActionConstraints
from cortex.models.context import RobotState
from cortex.models.decision import CertificationDecision, CertificationState
from cortex.sim.harness import (
    SimHarness,
    SimScenario,
    SimResult,
    SimReporter,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _simple_action() -> Action:
    return Action(
        spec=ActionSpec(
            action_type=ActionType.MOVE_EE,
            target_pose=Pose(x=0.4, y=0.0, z=0.5),
            constraints=ActionConstraints(max_speed_ms=0.3),
        ),
        source="test",
        intent="test move",
    )


def _normal_state(**kwargs) -> RobotState:
    return RobotState(
        ee_position=[0.3, 0.0, 0.5],
        ee_orientation=[1.0, 0.0, 0.0, 0.0],
        **kwargs,
    )


def _make_scenario(
    name: str = "test_scenario",
    expected: list[CertificationState] | None = None,
    **state_kwargs,
) -> SimScenario:
    return SimScenario(
        name=name,
        robot_state=_normal_state(**state_kwargs),
        action=_simple_action(),
        expected_states=expected or [CertificationState.EXECUTE],
        description="unit test scenario",
        tags=["test"],
    )


# ── SimScenario ───────────────────────────────────────────────────────────────

class TestSimScenario:
    def test_construction(self):
        s = _make_scenario()
        assert s.name == "test_scenario"
        assert len(s.expected_states) == 1
        assert s.tags == ["test"]

    def test_default_tags_empty(self):
        s = SimScenario(
            name="x",
            robot_state=_normal_state(),
            action=_simple_action(),
            expected_states=[CertificationState.EXECUTE],
        )
        assert s.tags == []
        assert s.description == ""


# ── SimResult ─────────────────────────────────────────────────────────────────

class TestSimResult:
    def _dummy_decision(self, state: CertificationState) -> CertificationDecision:
        from cortex.models.decision import CertificationDecision, DecisionTrace
        trace = DecisionTrace(ctx_id="test-ctx", action_id="test-action", certification_state=state)
        return CertificationDecision(state=state, action=None, trace=trace, reason="test")

    def test_passed_when_state_in_expected(self):
        scenario = _make_scenario(expected=[CertificationState.EXECUTE])
        decision = self._dummy_decision(CertificationState.EXECUTE)
        r = SimResult(scenario=scenario, decision=decision, passed=True, latency_ms=1.0)
        assert r.passed is True

    def test_failed_when_state_not_in_expected(self):
        scenario = _make_scenario(expected=[CertificationState.SAFE_HALT])
        decision = self._dummy_decision(CertificationState.EXECUTE)
        r = SimResult(scenario=scenario, decision=decision, passed=False, latency_ms=1.0)
        assert r.passed is False

    def test_actual_state_returns_decision_value(self):
        scenario = _make_scenario()
        decision = self._dummy_decision(CertificationState.EXECUTE)
        r = SimResult(scenario=scenario, decision=decision, passed=True, latency_ms=1.0)
        assert r.actual_state == CertificationState.EXECUTE.value

    def test_actual_state_error_when_no_decision(self):
        scenario = _make_scenario()
        r = SimResult(scenario=scenario, decision=None, passed=False, latency_ms=1.0, error="boom")
        assert r.actual_state == "ERROR"

    def test_name_property(self):
        r = SimResult(
            scenario=_make_scenario(name="my_scenario"),
            decision=None,
            passed=False,
            latency_ms=0.5,
        )
        assert r.name == "my_scenario"

    def test_expected_states_str(self):
        scenario = _make_scenario(
            expected=[CertificationState.SAFE_HALT, CertificationState.EXECUTE]
        )
        r = SimResult(scenario=scenario, decision=None, passed=False, latency_ms=0.0)
        s = r.expected_states_str
        assert "SAFE_HALT" in s
        assert "EXECUTE" in s


# ── SimHarness ────────────────────────────────────────────────────────────────

class TestSimHarness:
    def test_empty_harness_runs_zero_results(self):
        harness = SimHarness()
        results = harness.run_all()
        assert results == []

    def test_add_scenario_increments_count(self):
        harness = SimHarness()
        harness.add_scenario(_make_scenario("a"))
        harness.add_scenario(_make_scenario("b"))
        assert harness.scenario_count == 2

    def test_add_scenario_returns_self(self):
        harness = SimHarness()
        result = harness.add_scenario(_make_scenario())
        assert result is harness

    def test_run_all_returns_one_result_per_scenario(self):
        harness = SimHarness()
        harness.add_scenario(_make_scenario("s1"))
        harness.add_scenario(_make_scenario("s2"))
        results = harness.run_all()
        assert len(results) == 2

    def test_run_tagged_filters_correctly(self):
        harness = SimHarness()
        s1 = SimScenario(
            name="tagged", robot_state=_normal_state(), action=_simple_action(),
            expected_states=[CertificationState.EXECUTE], tags=["smoke"],
        )
        s2 = SimScenario(
            name="untagged", robot_state=_normal_state(), action=_simple_action(),
            expected_states=[CertificationState.EXECUTE], tags=["integration"],
        )
        harness.add_scenario(s1).add_scenario(s2)
        results = harness.run_tagged("smoke")
        assert len(results) == 1
        assert results[0].name == "tagged"

    def test_run_tagged_no_matches_returns_empty(self):
        harness = SimHarness()
        harness.add_scenario(_make_scenario())
        assert harness.run_tagged("nonexistent") == []

    def test_add_standard_scenarios_loads_five(self):
        harness = SimHarness()
        harness.add_standard_scenarios()
        assert harness.scenario_count == 5

    def test_result_has_latency(self):
        harness = SimHarness()
        harness.add_scenario(_make_scenario())
        results = harness.run_all()
        assert results[0].latency_ms >= 0.0

    def test_result_decision_is_not_none_on_success(self):
        harness = SimHarness()
        harness.add_scenario(_make_scenario())
        results = harness.run_all()
        assert results[0].decision is not None
        assert results[0].error == ""


# ── Standard scenarios ────────────────────────────────────────────────────────

class TestStandardScenarios:
    """Run the canonical scenario set and verify outcomes."""

    def setup_method(self):
        self.harness = SimHarness()
        self.harness.add_standard_scenarios()
        self.results = self.harness.run_all()
        self.by_name = {r.name: r for r in self.results}

    def test_all_five_scenarios_run(self):
        assert len(self.results) == 5

    def test_normal_move_certifies_passes(self):
        r = self.by_name["normal_move_certifies"]
        assert r.passed, f"expected pass, got {r.actual_state}"

    def test_emergency_stop_blocks(self):
        r = self.by_name["emergency_stop_blocks"]
        assert r.passed, f"expected blocking state, got {r.actual_state}"

    def test_emergency_stop_not_execute(self):
        r = self.by_name["emergency_stop_blocks"]
        if r.decision:
            assert r.decision.state != CertificationState.EXECUTE

    def test_singularity_blocks(self):
        r = self.by_name["singularity_blocks"]
        assert r.passed, f"expected blocking state, got {r.actual_state}"

    def test_out_of_workspace_blocks(self):
        r = self.by_name["out_of_workspace_blocks"]
        assert r.passed, f"expected blocking state, got {r.actual_state}"

    def test_slow_safe_move_allowed(self):
        r = self.by_name["slow_safe_move_allowed"]
        assert r.passed, f"expected allowed state, got {r.actual_state}"

    def test_smoke_tag_has_five_scenarios(self):
        harness = SimHarness()
        harness.add_standard_scenarios()
        results = harness.run_tagged("smoke")
        assert len(results) == 5

    def test_safety_tag_has_two_scenarios(self):
        harness = SimHarness()
        harness.add_standard_scenarios()
        results = harness.run_tagged("safety")
        assert len(results) == 2

    def test_sentinel_tag_has_one_scenario(self):
        harness = SimHarness()
        harness.add_standard_scenarios()
        results = harness.run_tagged("sentinel")
        assert len(results) == 1
        assert results[0].name == "emergency_stop_blocks"

    def test_no_scenario_raises_exception(self):
        for r in self.results:
            assert r.error == "", f"scenario {r.name} raised: {r.error}"


# ── SimReporter ───────────────────────────────────────────────────────────────

class TestSimReporter:
    def _make_result(self, passed: bool, name: str = "s") -> SimResult:
        from cortex.models.decision import CertificationDecision, DecisionTrace
        scenario = _make_scenario(name=name)
        state = CertificationState.EXECUTE if passed else CertificationState.SAFE_HALT
        trace = DecisionTrace(ctx_id="test-ctx", action_id="test-action", certification_state=state)
        decision = CertificationDecision(state=state, action=None, trace=trace, reason="test")
        return SimResult(scenario=scenario, decision=decision, passed=passed, latency_ms=1.0)

    def test_all_passed_when_all_pass(self):
        results = [self._make_result(True, f"s{i}") for i in range(3)]
        reporter = SimReporter(results)
        assert reporter.all_passed is True
        assert reporter.passed == 3
        assert reporter.failed == 0

    def test_all_passed_false_when_any_fail(self):
        results = [self._make_result(True), self._make_result(False, "f")]
        reporter = SimReporter(results)
        assert reporter.all_passed is False
        assert reporter.failed == 1

    def test_total_count(self):
        results = [self._make_result(True) for _ in range(4)]
        reporter = SimReporter(results)
        assert reporter.total == 4

    def test_summary_lines_non_empty(self):
        results = [self._make_result(True, "pass_scenario")]
        reporter = SimReporter(results)
        lines = reporter.summary_lines()
        assert len(lines) > 0

    def test_summary_lines_contain_scenario_name(self):
        results = [self._make_result(True, "my_unique_scenario")]
        reporter = SimReporter(results)
        text = "\n".join(reporter.summary_lines())
        assert "my_unique_scenario" in text

    def test_summary_lines_show_pass_marker(self):
        results = [self._make_result(True, "good")]
        reporter = SimReporter(results)
        text = "\n".join(reporter.summary_lines())
        assert "PASS" in text

    def test_summary_lines_show_fail_marker(self):
        results = [self._make_result(False, "bad")]
        reporter = SimReporter(results)
        text = "\n".join(reporter.summary_lines())
        assert "FAIL" in text

    def test_summary_shows_error_when_present(self):
        scenario = _make_scenario(name="err_scenario")
        r = SimResult(scenario=scenario, decision=None, passed=False, latency_ms=1.0, error="kaboom")
        reporter = SimReporter([r])
        text = "\n".join(reporter.summary_lines())
        assert "kaboom" in text

    def test_print_summary_no_exception(self, capsys):
        harness = SimHarness()
        harness.add_standard_scenarios()
        results = harness.run_all()
        reporter = SimReporter(results)
        reporter.print_summary()
        captured = capsys.readouterr()
        assert "Cortex Simulation Harness" in captured.out
