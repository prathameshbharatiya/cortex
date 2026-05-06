"""
tests/unit/test_multiagent_coordinator.py
==========================================
Phase 6 tests: multi-agent conflict detection and coordination.
"""

from __future__ import annotations

import time

import pytest

from cortex.multiagent.coordinator import (
    AgentCoordinator,
    AgentRegistration,
    ConflictType,
    ResolutionStrategy,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _reg(
    agent_id: str,
    position: list[float] = None,
    priority: int = 10,
    safety_radius_m: float = 0.3,
    resources: frozenset[str] = frozenset(),
) -> AgentRegistration:
    return AgentRegistration(
        agent_id=agent_id,
        name=agent_id,
        position=position or [0.0, 0.0, 0.0],
        safety_radius_m=safety_radius_m,
        priority=priority,
        resources_held=resources,
    )


# ── Registration ──────────────────────────────────────────────────────────────

class TestAgentRegistration:

    def test_register_increments_count(self) -> None:
        coord = AgentCoordinator()
        coord.register(_reg("r1"))
        assert coord.agent_count == 1

    def test_register_multiple(self) -> None:
        coord = AgentCoordinator()
        coord.register(_reg("r1"))
        coord.register(_reg("r2"))
        assert coord.agent_count == 2

    def test_deregister(self) -> None:
        coord = AgentCoordinator()
        coord.register(_reg("r1"))
        coord.deregister("r1")
        assert coord.agent_count == 0

    def test_get_agent(self) -> None:
        coord = AgentCoordinator()
        coord.register(_reg("r1", position=[1.0, 0.0, 0.0]))
        agent = coord.get_agent("r1")
        assert agent is not None
        assert agent.position == [1.0, 0.0, 0.0]

    def test_update_position(self) -> None:
        coord = AgentCoordinator()
        coord.register(_reg("r1", position=[0.0, 0.0, 0.0]))
        coord.update_position("r1", [1.0, 0.0, 0.0])
        agent = coord.get_agent("r1")
        assert agent.position == [1.0, 0.0, 0.0]

    def test_unregistered_agent_check_rejected(self) -> None:
        coord  = AgentCoordinator()
        result = coord.check_action("ghost")
        assert not result.approved
        assert result.strategy == ResolutionStrategy.REJECT


# ── No conflict ───────────────────────────────────────────────────────────────

class TestNoConflict:

    def test_single_agent_always_approved(self) -> None:
        coord = AgentCoordinator()
        coord.register(_reg("r1", position=[0.0, 0.0, 0.0]))
        result = coord.check_action("r1", target_position=[0.5, 0.0, 0.0])
        assert result.approved

    def test_far_apart_agents_no_conflict(self) -> None:
        coord = AgentCoordinator()
        coord.register(_reg("r1", position=[0.0, 0.0, 0.0]))
        coord.register(_reg("r2", position=[5.0, 0.0, 0.0]))
        result = coord.check_action("r1", target_position=[0.3, 0.0, 0.0])
        assert result.approved

    def test_approved_result_has_empty_conflicts(self) -> None:
        coord = AgentCoordinator()
        coord.register(_reg("r1", position=[0.0, 0.0, 0.0]))
        result = coord.check_action("r1", target_position=[0.3, 0.0, 0.0])
        assert result.conflicts == []


# ── Spatial overlap ───────────────────────────────────────────────────────────

class TestSpatialConflicts:

    def test_spatial_overlap_detected(self) -> None:
        coord = AgentCoordinator()
        coord.register(_reg("r1", position=[0.0, 0.0, 0.0]))
        coord.register(_reg("r2", position=[0.2, 0.0, 0.0]))   # very close
        # r1 moving toward r2
        result = coord.check_action("r1", target_position=[0.15, 0.0, 0.0])
        assert not result.approved
        conflict_types = [c.conflict_type for c in result.conflicts]
        assert ConflictType.SPATIAL_OVERLAP in conflict_types

    def test_spatial_overlap_suggests_hold_or_replan(self) -> None:
        coord = AgentCoordinator()
        coord.register(_reg("r1", position=[0.0, 0.0, 0.0]))
        coord.register(_reg("r2", position=[0.2, 0.0, 0.0]))
        result = coord.check_action("r1", target_position=[0.15, 0.0, 0.0])
        assert result.strategy in (ResolutionStrategy.HOLD_AND_WAIT, ResolutionStrategy.REPLAN)

    def test_trajectory_conflict_detected(self) -> None:
        coord = AgentCoordinator()
        coord.register(_reg("r1", position=[0.0, 0.0, 0.0]))
        coord.register(_reg("r2", position=[0.5, 0.0, 0.0]))
        # Path goes right through r2
        path = [[0.1, 0.0, 0.0], [0.5, 0.0, 0.0], [0.9, 0.0, 0.0]]
        result = coord.check_action("r1", planned_path=path)
        assert not result.approved
        conflict_types = [c.conflict_type for c in result.conflicts]
        assert ConflictType.SPATIAL_TRAJECTORY in conflict_types

    def test_trajectory_clear_path_approved(self) -> None:
        coord = AgentCoordinator()
        coord.register(_reg("r1", position=[0.0, 0.0, 0.0]))
        coord.register(_reg("r2", position=[0.0, 5.0, 0.0]))   # far away in Y
        path = [[0.1, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]]
        result = coord.check_action("r1", planned_path=path)
        assert result.approved


# ── Resource contention ───────────────────────────────────────────────────────

class TestResourceContention:

    def test_resource_held_by_other_detected(self) -> None:
        coord = AgentCoordinator()
        coord.register(_reg("r1", position=[0.0, 0.0, 0.0]))
        coord.register(_reg("r2", position=[2.0, 0.0, 0.0],
                            resources=frozenset(["object_A"])))
        result = coord.check_action(
            "r1",
            resources_requested=frozenset(["object_A"]),
        )
        assert not result.approved
        conflict_types = [c.conflict_type for c in result.conflicts]
        assert ConflictType.RESOURCE_CONTENTION in conflict_types

    def test_resource_not_held_approved(self) -> None:
        coord = AgentCoordinator()
        coord.register(_reg("r1", position=[0.0, 0.0, 0.0]))
        coord.register(_reg("r2", position=[2.0, 0.0, 0.0],
                            resources=frozenset(["object_B"])))
        result = coord.check_action(
            "r1",
            resources_requested=frozenset(["object_A"]),
        )
        assert result.approved

    def test_resource_conflict_strategy_hold(self) -> None:
        coord = AgentCoordinator()
        coord.register(_reg("r1", position=[0.0, 0.0, 0.0]))
        coord.register(_reg("r2", position=[2.0, 0.0, 0.0],
                            resources=frozenset(["object_A"])))
        result = coord.check_action(
            "r1", resources_requested=frozenset(["object_A"])
        )
        assert result.strategy == ResolutionStrategy.HOLD_AND_WAIT


# ── Workspace summary ─────────────────────────────────────────────────────────

class TestWorkspaceSummary:

    def test_summary_contains_agents(self) -> None:
        coord = AgentCoordinator()
        coord.register(_reg("r1"))
        coord.register(_reg("r2"))
        summary = coord.workspace_summary()
        assert summary["agent_count"] == 2
        assert len(summary["agents"]) == 2

    def test_summary_agent_has_position(self) -> None:
        coord = AgentCoordinator()
        coord.register(_reg("r1", position=[1.0, 2.0, 3.0]))
        agent_entry = coord.workspace_summary()["agents"][0]
        assert agent_entry["position"] == [1.0, 2.0, 3.0]


# ── Stale eviction ────────────────────────────────────────────────────────────

class TestStaleEviction:

    def test_stale_agent_evicted(self) -> None:
        coord = AgentCoordinator(stale_registration_s=0.01)  # 10ms stale
        coord.register(_reg("r1", position=[0.0, 0.0, 0.0]))
        coord.register(_reg("r2", position=[2.0, 0.0, 0.0]))
        time.sleep(0.05)
        # Trigger eviction by checking an action (evict_stale called inside)
        coord.check_action("r1")
        # After eviction, only r1 might remain (r2 was stale)
        # Both could be evicted if check happens after r1 also expires
        assert coord.agent_count <= 1
