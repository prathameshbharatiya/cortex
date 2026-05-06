"""
cortex-demo — run the Cortex certification demo.

Shows all five certification states with a simulated robot arm scenario.

Usage:
    cortex-demo
    cortex-demo --quiet
"""

from __future__ import annotations

import argparse
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="cortex-demo",
        description="Run the Cortex certification demo (all five states).",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress colour output.",
    )
    args = parser.parse_args()

    # Locate the examples/demo.py relative to this package
    pkg_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    demo_path = os.path.join(pkg_root, "examples", "demo.py")

    if not os.path.exists(demo_path):
        # Fallback: run the inline demo directly
        _run_inline_demo(args.quiet)
        return

    # Run examples/demo.py
    sys.path.insert(0, pkg_root)
    spec_globals: dict = {"__file__": demo_path, "__name__": "__main__"}
    with open(demo_path) as f:
        exec(compile(f.read(), demo_path, "exec"), spec_globals)  # noqa: S102


def _run_inline_demo(quiet: bool) -> None:
    """Minimal inline demo if examples/demo.py is not present (installed wheel)."""
    import time
    import cortex
    from cortex.models.action  import Action, ActionSpec, ActionType, ActionConstraints, Pose
    from cortex.models.context import RobotState, SceneGraph
    from cortex.models.memory  import MemoryRecord, MemoryType

    RESET = "" if quiet else "\033[0m"
    BOLD  = "" if quiet else "\033[1m"
    GREEN = "" if quiet else "\033[92m"
    CYAN  = "" if quiet else "\033[96m"

    print(f"\n{BOLD}{CYAN}Cortex v{cortex.__version__} — certification demo{RESET}\n")

    state = RobotState(
        ee_position=Pose(position=(0.3, 0.0, 0.5), orientation=(0, 0, 0, 1)),
        joint_positions=[0.0, -0.5, 0.0, -1.5, 0.0, 1.0, 0.0],
        joint_velocities=[0.0] * 7,
        joint_torques=[0.0] * 7,
    )
    scene = SceneGraph(objects=[], workspace_bounds=(-1, -1, 0, 1, 1, 1.5))
    ctx = cortex.build_context("pick red block", state, [])

    spec   = ActionSpec(action_type=ActionType.MOVE_TO)
    constraints = ActionConstraints(max_speed_ms=0.3, max_force_n=50.0)
    action = Action(
        spec=spec,
        target_pose=Pose(position=(0.4, 0.1, 0.3), orientation=(0, 0, 0, 1)),
        constraints=constraints,
        source="demo_planner",
        intent="pick red block",
        ai_confidence=0.82,
    )

    t0   = time.monotonic()
    cert = cortex.certify(action, ctx)
    ms   = (time.monotonic() - t0) * 1000

    state_color = GREEN
    print(f"  State     : {state_color}{cert.state.value}{RESET}")
    print(f"  Confidence: {cert.confidence.success_probability:.2f}")
    print(f"  Latency   : {ms:.1f} ms")
    print(f"  Approved  : {cert.approved}\n")


if __name__ == "__main__":
    main()
