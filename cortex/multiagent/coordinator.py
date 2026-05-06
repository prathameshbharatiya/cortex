"""
Multi-Agent Coordination Protocol
===================================
Detects spatial and temporal conflicts between multiple robots/agents
that share a physical workspace.

Conflict types
--------------
SPATIAL_OVERLAP      — two agents' safety bubbles overlap right now
SPATIAL_TRAJECTORY   — predicted trajectories cross within the horizon
RESOURCE_CONTENTION  — two agents want the same object/resource
TEMPORAL_RACE        — two agents scheduled to act at the same time
PRIORITY_CONFLICT    — lower-priority agent action blocks higher-priority one

Resolution strategies (returned in CoordinationResult)
-------------------------------------------------------
HOLD_AND_WAIT   — pause the lower-priority agent until the conflict resolves
YIELD           — one agent defers to the other
REPLAN          — conflicting agent must replan around the other
REJECT          — action is flatly incompatible with fleet state
APPROVE         — no conflict; proceed

Architecture
------------
The coordinator is a shared, thread-safe service.  Each robot calls
check_action() before executing.  The coordinator keeps a registry of
all agents' current states and planned trajectories.

Thread safety: a single RLock guards all mutations.
"""

from __future__ import annotations

import math
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence


# ── Domain types ──────────────────────────────────────────────────────────────

class ConflictType(str, Enum):
    SPATIAL_OVERLAP    = "spatial_overlap"
    SPATIAL_TRAJECTORY = "spatial_trajectory"
    RESOURCE_CONTENTION = "resource_contention"
    TEMPORAL_RACE      = "temporal_race"
    PRIORITY_CONFLICT  = "priority_conflict"


class ResolutionStrategy(str, Enum):
    APPROVE        = "approve"
    HOLD_AND_WAIT  = "hold_and_wait"
    YIELD          = "yield"
    REPLAN         = "replan"
    REJECT         = "reject"


@dataclass(frozen=True)
class AgentRegistration:
    """
    Describes one agent in the multi-agent fleet.

    Fields
    ------
    agent_id        : unique identifier
    name            : human-readable name
    position        : current end-effector / base position [x, y, z]
    safety_radius_m : personal safety bubble radius in metres (default 0.3m)
    priority        : lower number = higher priority (1 = highest)
    planned_path    : optional list of waypoints (list of [x,y,z])
    resources_held  : set of resource IDs the agent currently holds
    last_updated    : Unix timestamp of last state update
    """
    agent_id:       str
    name:           str               = ""
    position:       list[float]       = field(default_factory=lambda: [0.0, 0.0, 0.0])
    safety_radius_m: float            = 0.3
    priority:       int               = 10
    planned_path:   list[list[float]] = field(default_factory=list)
    resources_held: frozenset[str]    = field(default_factory=frozenset)
    last_updated:   float             = field(default_factory=time.time)

    def with_position(self, position: list[float]) -> "AgentRegistration":
        return AgentRegistration(
            agent_id=self.agent_id,
            name=self.name,
            position=position,
            safety_radius_m=self.safety_radius_m,
            priority=self.priority,
            planned_path=self.planned_path,
            resources_held=self.resources_held,
            last_updated=time.time(),
        )


@dataclass(frozen=True)
class ConflictReport:
    """Describes a single detected conflict between two agents."""
    conflict_id:     str       = field(default_factory=lambda: str(uuid.uuid4()))
    conflict_type:   ConflictType = ConflictType.SPATIAL_OVERLAP
    agent_a:         str       = ""
    agent_b:         str       = ""
    description:     str       = ""
    severity:        float     = 0.5   # 0.0 (trivial) – 1.0 (critical)
    overlap_m:       float     = 0.0   # metres of spatial overlap (if spatial)
    resource_id:     str       = ""    # contested resource (if resource)
    suggested_delay_s: float   = 0.0   # how long to hold if HOLD_AND_WAIT


@dataclass(frozen=True)
class CoordinationResult:
    """
    Outcome of a coordination check for one agent's proposed action.

    Fields
    ------
    approved        : True if the action may proceed
    strategy        : what the agent should do
    conflicts       : list of detected conflicts (empty if approved)
    hold_for_s      : if strategy=HOLD_AND_WAIT, how long to wait
    reason          : human-readable explanation
    latency_ms      : how long the check took
    """
    approved:    bool                 = True
    strategy:    ResolutionStrategy   = ResolutionStrategy.APPROVE
    conflicts:   list[ConflictReport] = field(default_factory=list)
    hold_for_s:  float                = 0.0
    reason:      str                  = "No conflicts detected."
    latency_ms:  float                = 0.0


# ── Coordinator ───────────────────────────────────────────────────────────────

class AgentCoordinator:
    """
    Thread-safe multi-agent coordination service.

    Usage
    -----
        coord = AgentCoordinator()

        # Register all robots at startup
        coord.register(AgentRegistration(agent_id="robot_1", position=[0,0,0]))
        coord.register(AgentRegistration(agent_id="robot_2", position=[1,0,0]))

        # Before each action, check for conflicts
        result = coord.check_action(
            agent_id="robot_1",
            target_position=[0.5, 0, 0],
            resources_requested=frozenset(["object_A"]),
        )
        if not result.approved:
            handle_conflict(result)
    """

    def __init__(
        self,
        default_safety_clearance_m: float = 0.05,
        stale_registration_s:       float = 30.0,
    ) -> None:
        self._clearance       = default_safety_clearance_m
        self._stale_threshold = stale_registration_s
        self._agents: dict[str, AgentRegistration] = {}
        self._lock   = threading.RLock()

    # ── Registration ──────────────────────────────────────────────────────────

    def register(self, agent: AgentRegistration) -> None:
        """Register or update an agent in the fleet."""
        with self._lock:
            self._agents[agent.agent_id] = agent

    def update_position(
        self,
        agent_id: str,
        position: list[float],
        planned_path: list[list[float]] | None = None,
    ) -> None:
        """Update an agent's current position (called continuously by the robot)."""
        with self._lock:
            if agent_id not in self._agents:
                return
            current = self._agents[agent_id]
            self._agents[agent_id] = AgentRegistration(
                agent_id=current.agent_id,
                name=current.name,
                position=position,
                safety_radius_m=current.safety_radius_m,
                priority=current.priority,
                planned_path=planned_path if planned_path is not None else current.planned_path,
                resources_held=current.resources_held,
                last_updated=time.time(),
            )

    def deregister(self, agent_id: str) -> None:
        """Remove an agent from the fleet (e.g., after shutdown)."""
        with self._lock:
            self._agents.pop(agent_id, None)

    @property
    def agent_count(self) -> int:
        with self._lock:
            return len(self._agents)

    def get_agent(self, agent_id: str) -> AgentRegistration | None:
        with self._lock:
            return self._agents.get(agent_id)

    def all_agents(self) -> list[AgentRegistration]:
        with self._lock:
            return list(self._agents.values())

    # ── Conflict detection ────────────────────────────────────────────────────

    def check_action(
        self,
        agent_id:            str,
        target_position:     list[float] | None = None,
        resources_requested: frozenset[str]     = frozenset(),
        planned_path:        list[list[float]]  | None = None,
        action_duration_s:   float              = 1.0,
    ) -> CoordinationResult:
        """
        Check whether an agent's proposed action conflicts with other agents.

        Parameters
        ----------
        agent_id             : the requesting agent
        target_position      : where the agent wants to move
        resources_requested  : resource IDs the agent wants to acquire
        planned_path         : list of waypoints for the planned trajectory
        action_duration_s    : expected duration of the action in seconds

        Returns
        -------
        CoordinationResult with approved=True if no blocking conflicts found.
        """
        t0 = time.perf_counter()
        with self._lock:
            if agent_id not in self._agents:
                return CoordinationResult(
                    approved=False,
                    strategy=ResolutionStrategy.REJECT,
                    reason=f"Agent {agent_id!r} is not registered.",
                    latency_ms=(time.perf_counter() - t0) * 1000,
                )

            # Snapshot requester before eviction (eviction must not remove caller)
            requester = self._agents[agent_id]
            self._evict_stale()
            # Re-register requester in case it was just evicted
            if agent_id not in self._agents:
                self._agents[agent_id] = requester
            others = [a for aid, a in self._agents.items() if aid != agent_id]

        conflicts: list[ConflictReport] = []

        for other in others:
            # Spatial overlap at current positions
            spatial = self._check_spatial_overlap(
                requester, other, target_position
            )
            if spatial:
                conflicts.append(spatial)

            # Trajectory intersection
            if planned_path:
                traj = self._check_trajectory_conflict(
                    requester, other, planned_path
                )
                if traj:
                    conflicts.append(traj)

            # Resource contention
            if resources_requested:
                res = self._check_resource_contention(
                    requester, other, resources_requested
                )
                if res:
                    conflicts.append(res)

        if not conflicts:
            return CoordinationResult(
                approved=True,
                strategy=ResolutionStrategy.APPROVE,
                reason="No conflicts detected.",
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        # Resolve: pick the worst conflict and determine strategy
        worst     = max(conflicts, key=lambda c: c.severity)
        strategy, hold_for_s = self._resolve(requester, worst)

        approved = strategy == ResolutionStrategy.APPROVE
        return CoordinationResult(
            approved=approved,
            strategy=strategy,
            conflicts=conflicts,
            hold_for_s=hold_for_s,
            reason=(
                f"{worst.conflict_type.value}: {worst.description} "
                f"(strategy={strategy.value})"
            ),
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    # ── Spatial detection ─────────────────────────────────────────────────────

    def _check_spatial_overlap(
        self,
        requester:       AgentRegistration,
        other:           AgentRegistration,
        target_position: list[float] | None,
    ) -> ConflictReport | None:
        """Check if moving to target_position would overlap with `other`."""
        check_pos = target_position if target_position else requester.position
        dist      = _euclidean(check_pos, other.position)
        combined  = requester.safety_radius_m + other.safety_radius_m + self._clearance
        overlap   = combined - dist

        if overlap > 0:
            severity = min(1.0, overlap / combined)
            return ConflictReport(
                conflict_type=ConflictType.SPATIAL_OVERLAP,
                agent_a=requester.agent_id,
                agent_b=other.agent_id,
                description=(
                    f"Target position {check_pos} overlaps with {other.name or other.agent_id} "
                    f"at {other.position} (overlap={overlap:.3f}m)"
                ),
                severity=severity,
                overlap_m=round(overlap, 4),
                suggested_delay_s=max(0.5, overlap * 5),  # rough: 1cm/s retreat
            )
        return None

    def _check_trajectory_conflict(
        self,
        requester:    AgentRegistration,
        other:        AgentRegistration,
        planned_path: list[list[float]],
    ) -> ConflictReport | None:
        """Check if the planned trajectory passes within safety radius of `other`."""
        combined = requester.safety_radius_m + other.safety_radius_m + self._clearance
        for waypoint in planned_path:
            dist = _euclidean(waypoint, other.position)
            if dist < combined:
                return ConflictReport(
                    conflict_type=ConflictType.SPATIAL_TRAJECTORY,
                    agent_a=requester.agent_id,
                    agent_b=other.agent_id,
                    description=(
                        f"Planned waypoint {waypoint} passes within "
                        f"{dist:.3f}m of {other.name or other.agent_id} "
                        f"(minimum safe={combined:.3f}m)"
                    ),
                    severity=min(1.0, (combined - dist) / combined),
                    overlap_m=round(combined - dist, 4),
                    suggested_delay_s=1.0,
                )
        return None

    def _check_resource_contention(
        self,
        requester:    AgentRegistration,
        other:        AgentRegistration,
        requested:    frozenset[str],
    ) -> ConflictReport | None:
        """Check if a resource is already held by another agent."""
        contested = requested & other.resources_held
        if contested:
            return ConflictReport(
                conflict_type=ConflictType.RESOURCE_CONTENTION,
                agent_a=requester.agent_id,
                agent_b=other.agent_id,
                description=(
                    f"Resources {contested} are held by "
                    f"{other.name or other.agent_id}"
                ),
                severity=0.8,
                resource_id=next(iter(contested)),
                suggested_delay_s=2.0,
            )
        return None

    # ── Resolution ────────────────────────────────────────────────────────────

    def _resolve(
        self,
        requester: AgentRegistration,
        conflict:  ConflictReport,
    ) -> tuple[ResolutionStrategy, float]:
        """Map a conflict to a resolution strategy."""
        if conflict.conflict_type == ConflictType.RESOURCE_CONTENTION:
            return ResolutionStrategy.HOLD_AND_WAIT, conflict.suggested_delay_s

        if conflict.conflict_type == ConflictType.SPATIAL_OVERLAP:
            if conflict.severity > 0.7:
                return ResolutionStrategy.REPLAN, 0.0
            return ResolutionStrategy.HOLD_AND_WAIT, conflict.suggested_delay_s

        if conflict.conflict_type == ConflictType.SPATIAL_TRAJECTORY:
            return ResolutionStrategy.YIELD, conflict.suggested_delay_s

        if conflict.conflict_type == ConflictType.PRIORITY_CONFLICT:
            return ResolutionStrategy.YIELD, 0.5

        return ResolutionStrategy.REPLAN, 0.0

    # ── Workspace summary ─────────────────────────────────────────────────────

    def workspace_summary(self) -> dict:
        """Return a summary of all registered agents for observability."""
        with self._lock:
            return {
                "agent_count": len(self._agents),
                "agents": [
                    {
                        "agent_id":  a.agent_id,
                        "name":      a.name,
                        "position":  a.position,
                        "priority":  a.priority,
                        "resources": list(a.resources_held),
                    }
                    for a in self._agents.values()
                ],
            }

    # ── Stale eviction ────────────────────────────────────────────────────────

    def _evict_stale(self) -> None:
        """Remove agents that haven't updated recently. Must hold _lock."""
        cutoff  = time.time() - self._stale_threshold
        stale   = [aid for aid, a in self._agents.items() if a.last_updated < cutoff]
        for aid in stale:
            del self._agents[aid]


# ── Geometry helper ───────────────────────────────────────────────────────────

def _euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


# Public alias — canonical name used by external callers
MultiAgentCoordinator = AgentCoordinator
