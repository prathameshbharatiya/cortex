"""
Cortex — Context Certification Infrastructure for Physical AI
=============================================================

No action may execute in the physical world unless Cortex has certified it.

Quick start
-----------
    import cortex

    ctx   = cortex.build_context(task, robot_state, memory_records)
    cert  = cortex.certify(action, ctx)

    if cert.state == CertificationState.EXECUTE:
        robot.execute(cert.action)
    else:
        handle(cert)
"""

from cortex.models.action      import Action, ActionSpec
from cortex.models.context     import Context, RobotState, SceneGraph
from cortex.models.memory      import MemoryRecord, MemoryType
from cortex.models.decision    import (
    CertificationDecision,
    CertificationState,
    ConfidenceContract,
    DecisionTrace,
    FailureMode,
    RiskLevel,
)
from cortex.gate.certification import CertificationGate
from cortex.gate.builder       import ContextBuilder

# ── convenience singleton ────────────────────────────────────────────────────
_default_gate: CertificationGate | None = None


def _gate() -> CertificationGate:
    global _default_gate
    if _default_gate is None:
        _default_gate = CertificationGate()
    return _default_gate


def build_context(
    task: str,
    robot_state: RobotState,
    memory_records: list[MemoryRecord] | None = None,
) -> Context:
    """Assemble a validated Context from task description, robot state, and memory."""
    return ContextBuilder().build(task, robot_state, memory_records or [])


def certify(action: Action, ctx: Context) -> CertificationDecision:
    """
    Certify an action against a context.

    Returns a CertificationDecision with state:
        EXECUTE                  — safe, run it
        EXECUTE_WITH_CONSTRAINTS — run the modified action in cert.action
        REPLAN_REQUIRED          — goal reachable, try a different action
        SAFE_HALT                — stop all motion
        HUMAN_OVERRIDE_REQUIRED  — escalate to operator
    """
    return _gate().certify(action, ctx)


__all__ = [
    "Action",
    "ActionSpec",
    "Context",
    "RobotState",
    "SceneGraph",
    "MemoryRecord",
    "MemoryType",
    "CertificationDecision",
    "CertificationState",
    "ConfidenceContract",
    "DecisionTrace",
    "FailureMode",
    "RiskLevel",
    "CertificationGate",
    "ContextBuilder",
    "build_context",
    "certify",
]

# Phase 2 — Memory Gateway
from cortex.memory.gateway  import MemoryGateway, GatewayResult
from cortex.memory.adapters import (
    MemoryQuery, QueryType,
    InMemoryAdapter, JSONFileAdapter, RedisAdapter,
    WeaviateAdapter, Mem0Adapter,
)

# Phase 3 — Context Engine
from cortex.context.engine import ContextEngine, AssemblyReport
from cortex.context.sensor_check import SensorCrossChecker, SensorVerdict
from cortex.context.constraint_loader import ConstraintLoader, ConstraintConfig
from cortex.context.physics_horizon import PhysicsHorizonBuilder, HorizonConfig

# Phase 4 — Relevance Engine
from cortex.relevance.engine          import RelevanceEngine, RelevanceReport, RecordRelevance
from cortex.relevance.task_scorer     import TaskRelevanceScorer
from cortex.relevance.state_scorer    import StateRelevanceScorer
from cortex.relevance.temporal_scorer import TemporalValidityScorer
from cortex.relevance.outcome_scorer  import OutcomeRelevanceScorer

# Phase 5 — Experience Tracker + Lifecycle Manager
from cortex.experience.record    import ExperienceRecord, OutcomeSpec, OutcomeClass, CausalFactor
from cortex.experience.tracker   import ExperienceTracker
from cortex.experience.lifecycle import LifecycleManager, LifecycleConfig, MaintenanceReport


# Protocol — .ctx wire format
from cortex.protocol           import CTXSerializer, CTXSigner, CTXSignatureError, generate_json_schema
from cortex.protocol.validator import validate_ctx_file

_ctx_serializer = CTXSerializer()


def save_ctx(ctx: "Context", path: str) -> None:
    """Save a Context to a .ctx binary file (MessagePack + versioned)."""
    _ctx_serializer.save(ctx, path)


def load_ctx(path: str) -> "Context":
    """Load and validate a .ctx binary file."""
    return _ctx_serializer.load(path)


# Phase 6 — Integration Layer
from cortex.integration.sdk.client import CortexSDK
from cortex.integration.ros2.node  import CortexNode

# Package metadata
from cortex._version import __version__, __version_info__, __build__, __commit__

__all__ += [
    "__version__",
    "__version_info__",
    "__build__",
    "__commit__",
]
