"""
Context Engine
==============
The heart of Phase 3. Assembles the full validated Context (CTX) from
raw inputs through a four-stage pipeline.

The CTX is the single validated packet that both the AI planner and
the Certification Gate operate on. Nothing enters the Gate without
passing through the Context Engine first.

Assembly pipeline
-----------------
  Stage 1 — Memory retrieval and sensor cross-check
    Retrieve relevant records from the Gateway.
    Validate each record against live sensor data.
    Adjust per-record confidence based on sensor agreement.

  Stage 2 — Constraint injection
    Load platform defaults, runtime constraints, and deployment config.
    Attach the full constraint set to the CTX so the Gate sees it.

  Stage 3 — Physics horizon computation
    Run 12-step forward simulation from current robot state.
    Embed the horizon in the CTX so the Gate and AI both see
    what the physics envelope looks like before any action is proposed.

  Stage 4 — Confidence aggregation and CTX finalisation
    Compute overall CTX confidence from all sources.
    Apply confidence floor — if below floor, return degraded CTX
    flagged for REPLAN rather than a CTX that will fail at the Gate.
    Sign and freeze the CTX (immutable once returned).

Usage
-----
    engine = ContextEngine(gateway=gw)

    ctx = engine.assemble(
        task="pick red block",
        robot_state=robot_state,
        scene_graph=scene_graph,
    )

    cert = gate.certify(action, ctx)
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from cortex.models.context import (
    Context, RobotState, SceneGraph, SafetyConstraint, PhysicsHorizon
)
from cortex.models.memory import MemoryRecord, MemoryType
from cortex.memory.gateway import MemoryGateway
from cortex.memory.adapters.base import MemoryQuery, QueryType
from cortex.context.sensor_check import SensorCrossChecker, CrossCheckReport, SensorVerdict
from cortex.relevance.engine import RelevanceEngine
from cortex.context.constraint_loader import ConstraintLoader, ConstraintConfig, ConstraintLoadReport
from cortex.context.physics_horizon import PhysicsHorizonBuilder, HorizonConfig


# ── Assembly report ───────────────────────────────────────────────────────────

@dataclass
class AssemblyReport:
    """
    Full audit record of how the CTX was assembled.
    Stored alongside the CTX for traceability.
    """
    ctx_id:              str
    task:                str

    # Stage timings
    stage1_memory_ms:    float = 0.0
    stage2_constraint_ms: float = 0.0
    stage3_physics_ms:   float = 0.0
    stage4_aggregate_ms: float = 0.0
    total_ms:            float = 0.0

    # Stage results
    records_retrieved:   int = 0
    records_confirmed:   int = 0
    records_contradicted: int = 0
    records_after_filter: int = 0
    constraints_loaded:  int = 0
    constraints_veto:    int = 0
    physics_feasible:    bool = True

    # Final confidence
    memory_confidence:   float = 1.0
    overall_confidence:  float = 1.0
    below_floor:         bool = False

    adapters_queried:    list[str] = field(default_factory=list)
    warnings:            list[str] = field(default_factory=list)


# ── Context Engine ────────────────────────────────────────────────────────────

class ContextEngine:
    """
    Assembles validated CTX packets from raw inputs.

    The engine is stateless — each call to assemble() is independent.
    It orchestrates the four pipeline stages and returns a frozen CTX.
    """

    def __init__(
        self,
        gateway:           MemoryGateway | None = None,
        sensor_checker:    SensorCrossChecker | None = None,
        constraint_loader: ConstraintLoader | None = None,
        horizon_builder:   PhysicsHorizonBuilder | None = None,
        # Defaults
        default_top_k:     int   = 10,
        default_confidence_floor: float = 0.60,
        default_validity_ms: float = 500.0,
    ) -> None:
        self.gateway           = gateway
        self.sensor_checker    = sensor_checker    or SensorCrossChecker()
        self.constraint_loader = constraint_loader or ConstraintLoader()
        self.horizon_builder   = horizon_builder   or PhysicsHorizonBuilder()
        self.relevance_engine  = RelevanceEngine()
        self.default_top_k     = default_top_k
        self.default_confidence_floor = default_confidence_floor
        self.default_validity_ms      = default_validity_ms

    # ── Main entry point ──────────────────────────────────────────────────────

    def assemble(
        self,
        task:                str,
        robot_state:         RobotState,
        scene_graph:         SceneGraph | None = None,
        memory_records:      list[MemoryRecord] | None = None,
        extra_constraints:   list[SafetyConstraint] | None = None,
        target_position:     list[float] | None = None,
        confidence_floor:    float | None = None,
        validity_window_ms:  float | None = None,
        top_k:               int | None = None,
        skip_physics:        bool = False,
    ) -> tuple[Context, AssemblyReport]:
        """
        Assemble a validated CTX through the four-stage pipeline.

        Returns (ctx, report) — the frozen CTX and its assembly audit record.

        Parameters
        ----------
        task              : human-readable task description
        robot_state       : live proprioceptive state from sensors
        scene_graph       : detected objects and scene structure (optional)
        memory_records    : pre-retrieved records (skips Stage 1 retrieval
                            if provided — useful when caller controls retrieval)
        extra_constraints : additional safety constraints to inject
        target_position   : intended motion direction hint for physics sim
        confidence_floor  : minimum confidence for EXECUTE (default 0.60)
        validity_window_ms: how long CTX is valid (default 500ms)
        top_k             : max memory records to retrieve (default 10)
        skip_physics      : skip Stage 3 physics simulation (faster, less safe)
        """
        t_total = time.perf_counter()
        ctx_id  = str(uuid.uuid4())
        scene   = scene_graph or SceneGraph()
        floor   = confidence_floor  or self.default_confidence_floor
        window  = validity_window_ms or self.default_validity_ms
        k       = top_k or self.default_top_k

        report  = AssemblyReport(ctx_id=ctx_id, task=task)

        # ════════════════════════════════════════════════════════════════════
        # STAGE 1 — Memory retrieval + sensor cross-check
        # ════════════════════════════════════════════════════════════════════
        t1 = time.perf_counter()

        records, cross_check = self._stage1_memory(
            task, robot_state, scene, memory_records, k, report
        )

        report.stage1_memory_ms    = (time.perf_counter() - t1) * 1000.0
        report.records_retrieved   = len(records)
        report.records_confirmed   = cross_check.confirmed_count if cross_check else 0
        report.records_contradicted = cross_check.contradicted_count if cross_check else 0
        report.records_after_filter = len(records)

        if cross_check and cross_check.contradiction_rate > 0.5:
            report.warnings.append(
                f"High contradiction rate: {cross_check.contradiction_rate:.0%} of "
                "memory records contradict sensor data"
            )

        # ════════════════════════════════════════════════════════════════════
        # STAGE 2 — Constraint injection
        # ════════════════════════════════════════════════════════════════════
        t2 = time.perf_counter()

        constraint_report = self.constraint_loader.load(
            robot_state=robot_state,
            scene_graph=scene,
            extra_constraints=extra_constraints,
        )
        constraints = constraint_report.constraints

        report.stage2_constraint_ms = (time.perf_counter() - t2) * 1000.0
        report.constraints_loaded   = len(constraints)
        report.constraints_veto     = constraint_report.veto_count

        if constraint_report.runtime_count > 0:
            report.warnings.append(
                f"{constraint_report.runtime_count} runtime constraint(s) active "
                "(e.g. human nearby, gripper loaded)"
            )

        # ════════════════════════════════════════════════════════════════════
        # STAGE 3 — Physics horizon
        # ════════════════════════════════════════════════════════════════════
        t3 = time.perf_counter()
        physics_horizon: PhysicsHorizon | None = None

        if not skip_physics:
            physics_horizon = self.horizon_builder.build(
                robot_state=robot_state,
                scene_graph=scene,
                target_position=target_position,
            )
            report.physics_feasible = physics_horizon.feasible

            if not physics_horizon.feasible:
                report.warnings.append(
                    "Physics horizon indicates INFEASIBLE state — "
                    f"collision_steps={physics_horizon.collision_risk_steps}, "
                    f"singularity_steps={physics_horizon.singularity_risk_steps}"
                )
        else:
            report.physics_feasible = True

        report.stage3_physics_ms = (time.perf_counter() - t3) * 1000.0

        # ════════════════════════════════════════════════════════════════════
        # STAGE 4 — Confidence aggregation + CTX finalisation
        # ════════════════════════════════════════════════════════════════════
        t4 = time.perf_counter()

        memory_confidence = self._compute_memory_confidence(records, cross_check)
        overall_confidence = self._compute_overall_confidence(
            memory_confidence, physics_horizon, constraints, scene
        )

        report.memory_confidence  = round(memory_confidence, 4)
        report.overall_confidence = round(overall_confidence, 4)
        report.below_floor        = overall_confidence < floor
        report.stage4_aggregate_ms = (time.perf_counter() - t4) * 1000.0

        if report.below_floor:
            report.warnings.append(
                f"CTX confidence {overall_confidence:.2%} below floor {floor:.2%} — "
                "Gate will issue REPLAN_REQUIRED"
            )

        # Build the frozen CTX
        ctx = Context(
            ctx_id=ctx_id,
            task=task,
            robot_state=robot_state,
            scene_graph=scene,
            memory_records=records,
            memory_confidence=memory_confidence,
            safety_constraints=constraints,
            physics_horizon=physics_horizon,
            confidence_floor=floor,
            validity_window_ms=window,
            assembly_latency_ms=(time.perf_counter() - t_total) * 1000.0,
        )

        report.total_ms = (time.perf_counter() - t_total) * 1000.0
        return ctx, report

    # ── Stage implementations ─────────────────────────────────────────────────

    def _stage1_memory(
        self,
        task:            str,
        robot_state:     RobotState,
        scene_graph:     SceneGraph,
        provided_records: list[MemoryRecord] | None,
        top_k:           int,
        report:          AssemblyReport,
    ) -> tuple[list[MemoryRecord], CrossCheckReport | None]:
        """Retrieve memory records and run sensor cross-check."""

        if provided_records is not None:
            # Caller pre-retrieved records — skip Gateway retrieval
            records = [r for r in provided_records if r.is_currently_valid]
        elif self.gateway is not None:
            # Retrieve from Gateway
            query = MemoryQuery(
                query_text=task,
                query_type=QueryType.SEMANTIC,
                top_k=top_k,
                min_confidence=0.30,
            )
            gw_result = self.gateway.retrieve(query)
            records   = gw_result.records
            report.adapters_queried = gw_result.adapters_queried

            # Also fetch recent spatial records for cross-check
            if scene_graph.objects:
                spatial_query = MemoryQuery(
                    query_type=QueryType.SPATIAL,
                    memory_types=[MemoryType.SPATIAL],
                    top_k=top_k // 2,
                    max_age_seconds=3600.0,
                )
                spatial_result = self.gateway.retrieve(spatial_query)
                # Merge without duplicates
                existing_ids = {r.record_id for r in records}
                for r in spatial_result.records:
                    if r.record_id not in existing_ids:
                        records.append(r)
                        existing_ids.add(r.record_id)
        else:
            records = []

        # Sensor cross-check
        if records:
            cross_check = self.sensor_checker.check(records, robot_state, scene_graph)
            adjusted_map = {r.record_id: r.adjusted_confidence for r in cross_check.results}
            records = [r for r in records
                       if adjusted_map.get(r.record_id, r.effective_confidence) > 0.20]
        else:
            cross_check = None

        # Relevance Engine — four-axis scoring + diversity selection
        if records:
            rel_report = self.relevance_engine.rank(
                candidates=records, task=task,
                robot_state=robot_state, scene_graph=scene_graph, top_k=top_k,
            )
            records = rel_report.selected

        return records, cross_check

    @staticmethod
    def _compute_memory_confidence(
        records:     list[MemoryRecord],
        cross_check: CrossCheckReport | None,
    ) -> float:
        if not records:
            return 0.80   # no memory = conservative confidence, not zero

        if cross_check:
            return round(cross_check.overall_confidence, 4)

        return round(
            sum(r.effective_confidence for r in records) / len(records), 4
        )

    @staticmethod
    def _compute_overall_confidence(
        memory_confidence: float,
        physics_horizon:   PhysicsHorizon | None,
        constraints:       list[SafetyConstraint],
        scene_graph:       SceneGraph,
    ) -> float:
        """
        Composite confidence from all assembly stages.
        Weighted: physics > memory > scene factors.
        """
        scores: list[tuple[float, float]] = []   # (score, weight)

        # Memory confidence (weight 0.40)
        scores.append((memory_confidence, 0.40))

        # Physics (weight 0.40)
        if physics_horizon is not None:
            physics_score = (
                physics_horizon.min_stability_margin
                if physics_horizon.feasible
                else 0.0
            )
            scores.append((physics_score, 0.40))
        else:
            scores.append((0.75, 0.40))  # no horizon = conservative estimate

        # Scene factors (weight 0.20)
        scene_score = 1.0
        if scene_graph.humans_nearby:
            scene_score *= 0.80
        if scene_graph.obstruction_detected:
            scene_score *= 0.85
        scores.append((scene_score, 0.20))

        total_weight = sum(w for _, w in scores)
        if total_weight == 0:
            return 0.5
        return sum(s * w for s, w in scores) / total_weight

    # ── Convenience method: fast build without report ─────────────────────────

    def build(
        self,
        task:        str,
        robot_state: RobotState,
        **kwargs,
    ) -> Context:
        """
        Simplified entry point — returns just the CTX.
        Use assemble() when you need the audit report.
        """
        ctx, _ = self.assemble(task, robot_state, **kwargs)
        return ctx
