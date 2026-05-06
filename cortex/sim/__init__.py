"""
Simulation Harness
==================
Deterministic, zero-hardware test environment for Cortex development.

Run a full certification pipeline cycle in-process without any robot
or physics backend dependency.

Exports:
  SimHarness      — orchestrates end-to-end simulated runs
  SimScenario     — describes one test scenario
  SimResult       — outcome of a simulated run
  SimReporter     — formats results for terminal output
"""

from cortex.sim.harness import SimHarness, SimScenario, SimResult, SimReporter

__all__ = ["SimHarness", "SimScenario", "SimResult", "SimReporter"]
