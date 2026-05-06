"""
Multi-Agent Coordination Protocol
===================================
Detects and resolves spatial/temporal conflicts between multiple robots
or AI agents sharing the same workspace.

Exports:
  AgentCoordinator   — the main coordinator managing a fleet of agents
  AgentRegistration  — describes one agent in the fleet
  ConflictReport     — describes a detected conflict
  CoordinationResult — outcome of a coordination check
"""

from cortex.multiagent.coordinator import (
    AgentCoordinator,
    AgentRegistration,
    ConflictReport,
    CoordinationResult,
    ConflictType,
)

__all__ = [
    "AgentCoordinator",
    "AgentRegistration",
    "ConflictReport",
    "CoordinationResult",
    "ConflictType",
]
