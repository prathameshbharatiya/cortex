"""
Sensor Cross-Check
==================
Stage 1 of Context assembly.

Takes retrieved memory records and the live robot/scene state,
and checks every memory claim against what sensors currently report.

Three outcomes per record:
  CONFIRMED   — sensor agrees with memory. Confidence elevated.
  NEUTRAL     — no sensor data to confirm or deny. Confidence unchanged.
  CONTRADICTED — sensor disagrees with memory. Confidence reduced.
                 Record is flagged but not removed — the Gate decides
                 whether to block on it.

Why this matters
----------------
A robot may retrieve a memory that says "cup is at position [0.3, 0, 0.5]"
but the camera currently sees the cup at [0.6, 0, 0.5]. Acting on the old
memory without catching this contradiction causes incorrect grasps.

The cross-check catches this before the CTX is finalised.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from cortex.models.memory  import MemoryRecord, MemoryType
from cortex.models.context import RobotState, SceneGraph


class SensorVerdict(str, Enum):
    CONFIRMED    = "confirmed"
    NEUTRAL      = "neutral"
    CONTRADICTED = "contradicted"


@dataclass(frozen=True)
class CrossCheckResult:
    record_id:        str
    verdict:          SensorVerdict
    original_confidence: float
    adjusted_confidence: float
    contradiction_dist_m: float = 0.0
    notes:            str = ""


@dataclass
class CrossCheckReport:
    results:          list[CrossCheckResult] = field(default_factory=list)
    confirmed_count:  int = 0
    neutral_count:    int = 0
    contradicted_count: int = 0
    latency_ms:       float = 0.0

    @property
    def contradiction_rate(self) -> float:
        total = len(self.results)
        return self.contradicted_count / total if total > 0 else 0.0

    @property
    def overall_confidence(self) -> float:
        if not self.results:
            return 1.0
        return sum(r.adjusted_confidence for r in self.results) / len(self.results)


class SensorCrossChecker:
    """
    Validates retrieved memory records against live sensor data.

    Adjusts per-record confidence based on sensor agreement.
    Flags contradictions for the Context Engine to handle.
    """

    def __init__(
        self,
        contradiction_threshold_m: float = 0.08,   # 8cm mismatch = contradiction
        confirmation_threshold_m:  float = 0.03,   # 3cm mismatch = confirmed
        contradiction_penalty:     float = 0.40,   # confidence × (1 - penalty)
        confirmation_boost:        float = 0.15,   # confidence + boost, capped at 1.0
    ) -> None:
        self.contradiction_threshold_m = contradiction_threshold_m
        self.confirmation_threshold_m  = confirmation_threshold_m
        self.contradiction_penalty     = contradiction_penalty
        self.confirmation_boost        = confirmation_boost

    def check(
        self,
        records:     list[MemoryRecord],
        robot_state: RobotState,
        scene_graph: SceneGraph,
    ) -> CrossCheckReport:
        t0 = time.perf_counter()
        results: list[CrossCheckResult] = []

        for record in records:
            result = self._check_record(record, robot_state, scene_graph)
            results.append(result)

        report = CrossCheckReport(
            results=results,
            confirmed_count=sum(1 for r in results if r.verdict == SensorVerdict.CONFIRMED),
            neutral_count=sum(1 for r in results if r.verdict == SensorVerdict.NEUTRAL),
            contradicted_count=sum(1 for r in results if r.verdict == SensorVerdict.CONTRADICTED),
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )
        return report

    # ── Per-record check ──────────────────────────────────────────────────────

    def _check_record(
        self,
        record:      MemoryRecord,
        robot_state: RobotState,
        scene_graph: SceneGraph,
    ) -> CrossCheckResult:
        orig = record.effective_confidence

        # ── Spatial records — check object positions ─────────────────────────
        if record.memory_type == MemoryType.SPATIAL:
            return self._check_spatial(record, scene_graph, orig)

        # ── Episodic records — check robot state assumptions ─────────────────
        if record.memory_type == MemoryType.EPISODIC:
            return self._check_episodic(record, robot_state, orig)

        # ── All other types — neutral (cannot directly verify) ───────────────
        return CrossCheckResult(
            record_id=record.record_id,
            verdict=SensorVerdict.NEUTRAL,
            original_confidence=orig,
            adjusted_confidence=orig,
            notes=f"type={record.memory_type.value}: no sensor check applicable",
        )

    def _check_spatial(
        self,
        record:      MemoryRecord,
        scene_graph: SceneGraph,
        orig:        float,
    ) -> CrossCheckResult:
        content = record.content
        if not isinstance(content, dict):
            return self._neutral(record.record_id, orig, "no structured content")

        obj_id       = content.get("object_id")
        claimed_pos  = content.get("position")
        if not obj_id or not claimed_pos:
            return self._neutral(record.record_id, orig, "missing object_id or position")

        scene_obj = scene_graph.get_object(obj_id)
        if scene_obj is None:
            # Object not currently visible — could be occluded, not a contradiction
            return self._neutral(
                record.record_id, orig,
                f"object '{obj_id}' not in current scene (may be occluded)",
            )

        claimed = np.array(claimed_pos)
        actual  = np.array(scene_obj.position)
        dist    = float(np.linalg.norm(claimed - actual))

        if dist <= self.confirmation_threshold_m:
            adjusted = min(1.0, orig + self.confirmation_boost)
            return CrossCheckResult(
                record_id=record.record_id,
                verdict=SensorVerdict.CONFIRMED,
                original_confidence=orig,
                adjusted_confidence=adjusted,
                contradiction_dist_m=dist,
                notes=f"sensor confirms: '{obj_id}' within {dist:.3f}m of memory claim",
            )

        if dist > self.contradiction_threshold_m:
            adjusted = orig * (1.0 - self.contradiction_penalty)
            return CrossCheckResult(
                record_id=record.record_id,
                verdict=SensorVerdict.CONTRADICTED,
                original_confidence=orig,
                adjusted_confidence=adjusted,
                contradiction_dist_m=dist,
                notes=(
                    f"CONTRADICTION: memory says '{obj_id}' at {claimed_pos}, "
                    f"sensor reports {scene_obj.position} (dist={dist:.3f}m)"
                ),
            )

        # Between thresholds — partially confirmed
        ratio    = 1.0 - (dist / self.contradiction_threshold_m)
        adjusted = orig + (self.confirmation_boost * ratio)
        return CrossCheckResult(
            record_id=record.record_id,
            verdict=SensorVerdict.NEUTRAL,
            original_confidence=orig,
            adjusted_confidence=min(1.0, adjusted),
            contradiction_dist_m=dist,
            notes=f"partial match: '{obj_id}' within {dist:.3f}m (threshold {self.contradiction_threshold_m}m)",
        )

    def _check_episodic(
        self,
        record:      MemoryRecord,
        robot_state: RobotState,
        orig:        float,
    ) -> CrossCheckResult:
        content = record.content
        if not isinstance(content, dict):
            return self._neutral(record.record_id, orig, "no structured content")

        # Check if the memory references an assumed EE position
        assumed_ee = content.get("ee_position")
        if assumed_ee and robot_state.ee_position:
            claimed = np.array(assumed_ee)
            actual  = np.array(robot_state.ee_position)
            dist    = float(np.linalg.norm(claimed - actual))

            if dist > 0.5:   # 50cm — robot has moved significantly since this episode
                adjusted = orig * (1.0 - self.contradiction_penalty * 0.5)
                return CrossCheckResult(
                    record_id=record.record_id,
                    verdict=SensorVerdict.NEUTRAL,
                    original_confidence=orig,
                    adjusted_confidence=adjusted,
                    notes=(
                        f"robot moved {dist:.2f}m since episode was recorded — "
                        "spatial context may differ"
                    ),
                )

        return self._neutral(record.record_id, orig, "episodic record: no direct sensor check")

    @staticmethod
    def _neutral(record_id: str, confidence: float, notes: str) -> CrossCheckResult:
        return CrossCheckResult(
            record_id=record_id,
            verdict=SensorVerdict.NEUTRAL,
            original_confidence=confidence,
            adjusted_confidence=confidence,
            notes=notes,
        )
