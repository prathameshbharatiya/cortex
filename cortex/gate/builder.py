"""
Context Builder
===============
Assembles a validated Context (CTX) from raw task description,
robot state, and memory records. The CTX is the mandatory input
to the Certification Gate.

Internals delegate to ContextEngine for the full four-stage assembly
pipeline (memory retrieval, constraint injection, physics horizon,
confidence aggregation).  The public API is unchanged.
"""

from __future__ import annotations

from cortex.models.context import Context, RobotState, SceneGraph
from cortex.models.memory  import MemoryRecord


class ContextBuilder:
    """Assembles a validated Context from its components via ContextEngine."""

    def __init__(self) -> None:
        # Import here to avoid circular imports at module load
        from cortex.context.engine import ContextEngine
        self._engine = ContextEngine()

    def build(
        self,
        task: str,
        robot_state: RobotState,
        memory_records: list[MemoryRecord] | None = None,
        scene_graph: SceneGraph | None = None,
        confidence_floor: float = 0.60,
        validity_window_ms: float = 500.0,
    ) -> Context:
        ctx, _ = self._engine.assemble(
            task=task,
            robot_state=robot_state,
            scene_graph=scene_graph,
            memory_records=memory_records,
            confidence_floor=confidence_floor,
            validity_window_ms=validity_window_ms,
        )
        return ctx
