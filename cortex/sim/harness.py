"""
Simulation Harness
==================
Runs the full Cortex certification pipeline against synthetic scenarios —
no robot hardware, no physics simulator, no network required.

Use this for:
  - Local development and smoke testing
  - CI regression checks
  - Demonstrating expected gate behaviour to new contributors
  - Benchmarking certification latency under load

Design
------
A SimScenario describes one test case: robot state + action + expected
outcome (EXECUTE, SAFE_HALT, REPLAN_REQUIRED, etc.).

SimHarness runs all registered scenarios, collects SimResults, and
returns a pass/fail summary.  Each run is fully deterministic.

Integration
-----------
  harness = SimHarness()
  harness.add_scenario(SimScenario(...))
  results = harness.run_all()
  SimReporter(results).print_summary()
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

import cortex
from cortex.gate.certification import CertificationGate
from cortex.models.action import Action, ActionSpec, ActionType, Pose, ActionConstraints
from cortex.models.context import RobotState, SceneGraph
from cortex.models.decision import CertificationDecision, CertificationState


# ── Scenario descriptor ───────────────────────────────────────────────────────

@dataclass
class SimScenario:
    """
    Describes one simulation test case.

    Fields
    ------
    name                : human-readable scenario name
    robot_state         : the robot state to certify against
    action              : the action being certified
    expected_states     : acceptable CertificationState outcomes (pass if any match)
    description         : optional notes about what this scenario tests
    tags                : optional categorisation tags
    """
    name:            str
    robot_state:     RobotState
    action:          Action
    expected_states: list[CertificationState]
    description:     str       = ""
    tags:            list[str] = field(default_factory=list)


# ── Result ────────────────────────────────────────────────────────────────────

@dataclass
class SimResult:
    """
    Outcome of running one SimScenario.

    Fields
    ------
    scenario      : which scenario was run
    decision      : the CertificationDecision produced
    passed        : True if decision.state is in scenario.expected_states
    latency_ms    : wall-clock time for certify()
    error         : set if an exception was raised during certification
    """
    scenario:    SimScenario
    decision:    CertificationDecision | None
    passed:      bool
    latency_ms:  float
    error:       str = ""

    @property
    def name(self) -> str:
        return self.scenario.name

    @property
    def actual_state(self) -> str:
        if self.decision:
            return self.decision.state.value
        return "ERROR"

    @property
    def expected_states_str(self) -> str:
        return " | ".join(s.value for s in self.scenario.expected_states)


# ── Harness ───────────────────────────────────────────────────────────────────

class SimHarness:
    """
    Runs SimScenarios through CertificationGate and collects results.

    Usage
    -----
        harness = SimHarness()
        harness.add_standard_scenarios()
        results = harness.run_all()
        SimReporter(results).print_summary()
    """

    def __init__(self, gate: CertificationGate | None = None) -> None:
        self._gate      = gate or CertificationGate()
        self._scenarios: list[SimScenario] = []

    def add_scenario(self, scenario: SimScenario) -> "SimHarness":
        self._scenarios.append(scenario)
        return self

    def add_standard_scenarios(self) -> "SimHarness":
        """Register the canonical set of smoke-test scenarios."""
        for s in _STANDARD_SCENARIOS:
            self.add_scenario(s)
        return self

    def run_all(self) -> list[SimResult]:
        """Run every registered scenario and return results."""
        results: list[SimResult] = []
        for scenario in self._scenarios:
            results.append(self._run_one(scenario))
        return results

    def run_tagged(self, tag: str) -> list[SimResult]:
        """Run only scenarios that carry a specific tag."""
        return [
            self._run_one(s)
            for s in self._scenarios
            if tag in s.tags
        ]

    def _run_one(self, scenario: SimScenario) -> SimResult:
        t0 = time.perf_counter()
        try:
            ctx = cortex.build_context(
                scenario.name, scenario.robot_state, []
            )
            decision = self._gate.certify(scenario.action, ctx)
            latency_ms = (time.perf_counter() - t0) * 1000
            passed = decision.state in scenario.expected_states
            return SimResult(
                scenario=scenario,
                decision=decision,
                passed=passed,
                latency_ms=round(latency_ms, 3),
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000
            return SimResult(
                scenario=scenario,
                decision=None,
                passed=False,
                latency_ms=round(latency_ms, 3),
                error=str(exc),
            )

    @property
    def scenario_count(self) -> int:
        return len(self._scenarios)


# ── Reporter ──────────────────────────────────────────────────────────────────

class SimReporter:
    """Formats SimResult lists for terminal output."""

    def __init__(self, results: list[SimResult]) -> None:
        self._results = results

    @property
    def passed(self) -> int:
        return sum(1 for r in self._results if r.passed)

    @property
    def failed(self) -> int:
        return sum(1 for r in self._results if not r.passed)

    @property
    def total(self) -> int:
        return len(self._results)

    @property
    def all_passed(self) -> bool:
        return self.failed == 0

    def summary_lines(self) -> list[str]:
        lines: list[str] = []
        lines.append(f"\n{'─'*62}")
        lines.append(f" Cortex Simulation Harness — {self.total} scenarios")
        lines.append(f"{'─'*62}")
        for r in self._results:
            status = "PASS" if r.passed else "FAIL"
            marker = "✓" if r.passed else "✗"
            lines.append(
                f"  {marker} [{status}] {r.name:<38} "
                f"{r.actual_state:<28} {r.latency_ms:6.1f}ms"
            )
            if r.error:
                lines.append(f"           ERROR: {r.error}")
            elif not r.passed:
                lines.append(
                    f"           expected: {r.expected_states_str}"
                )
        lines.append(f"{'─'*62}")
        lines.append(
            f"  Result: {self.passed}/{self.total} passed"
            f"{'  ✓ ALL PASS' if self.all_passed else f'  ✗ {self.failed} FAILED'}"
        )
        lines.append(f"{'─'*62}\n")
        return lines

    def print_summary(self) -> None:
        for line in self.summary_lines():
            print(line)


# ── Standard scenarios ────────────────────────────────────────────────────────

def _move_action(x: float = 0.4, y: float = 0.0, z: float = 0.5) -> Action:
    return Action(
        spec=ActionSpec(
            action_type=ActionType.MOVE_EE,
            target_pose=Pose(x=x, y=y, z=z),
            constraints=ActionConstraints(max_speed_ms=0.3),
        ),
        source="sim_harness",
        intent="simulation scenario",
    )


def _normal_state(**kwargs) -> RobotState:
    return RobotState(
        ee_position=[0.3, 0.0, 0.5],
        ee_orientation=[1.0, 0.0, 0.0, 0.0],
        **kwargs,
    )


_STANDARD_SCENARIOS: list[SimScenario] = [
    SimScenario(
        name="normal_move_certifies",
        robot_state=_normal_state(),
        action=_move_action(),
        expected_states=[
            CertificationState.EXECUTE,
            CertificationState.EXECUTE_WITH_CONSTRAINTS,
            CertificationState.REPLAN_REQUIRED,
            CertificationState.HUMAN_OVERRIDE_REQUIRED,
        ],
        description="Baseline: a normal move action should not be SAFE_HALT.",
        tags=["smoke", "baseline"],
    ),
    SimScenario(
        name="emergency_stop_blocks",
        robot_state=_normal_state(emergency_stop=True),
        action=_move_action(),
        expected_states=[
            CertificationState.SAFE_HALT,
            CertificationState.HUMAN_OVERRIDE_REQUIRED,
        ],
        description="E-stop active → Sentinel P1 must veto immediately.",
        tags=["smoke", "safety", "sentinel"],
    ),
    SimScenario(
        name="singularity_blocks",
        robot_state=_normal_state(in_singularity=True),
        action=_move_action(),
        expected_states=[
            CertificationState.SAFE_HALT,
            CertificationState.HUMAN_OVERRIDE_REQUIRED,
            CertificationState.REPLAN_REQUIRED,
        ],
        description="Kinematic singularity → must produce a blocking state.",
        tags=["smoke", "safety", "physics"],
    ),
    SimScenario(
        name="out_of_workspace_blocks",
        robot_state=_normal_state(),
        action=_move_action(x=2.0, y=2.0, z=2.0),   # well outside workspace
        expected_states=[
            CertificationState.REPLAN_REQUIRED,
            CertificationState.SAFE_HALT,
            CertificationState.HUMAN_OVERRIDE_REQUIRED,
        ],
        description="Target outside workspace → PhysiCore should block.",
        tags=["smoke", "physics"],
    ),
    SimScenario(
        name="slow_safe_move_allowed",
        robot_state=_normal_state(),
        action=Action(
            spec=ActionSpec(
                action_type=ActionType.MOVE_EE,
                target_pose=Pose(x=0.35, y=0.0, z=0.5),
                constraints=ActionConstraints(max_speed_ms=0.05),
            ),
            source="sim_harness",
            intent="slow careful move",
        ),
        expected_states=[
            CertificationState.EXECUTE,
            CertificationState.EXECUTE_WITH_CONSTRAINTS,
            CertificationState.REPLAN_REQUIRED,
            CertificationState.HUMAN_OVERRIDE_REQUIRED,
        ],
        description="Very slow motion should not trigger Sentinel speed limits.",
        tags=["smoke", "baseline"],
    ),
]
