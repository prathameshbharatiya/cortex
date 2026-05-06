"""
Intent-to-Action Translation Layer
====================================
Translates three categories of high-level intent into ActionProposals:

  1. NLIntent            — natural language ("pick up the red cube")
  2. GoalStateIntent     — desired end state ({object: "cube", at: pose})
  3. DemonstrationIntent — a recorded trajectory of waypoints

Each intent type has its own strategy class.  The top-level
IntentTranslator selects the correct strategy and returns an
ActionProposal that CertificationGate can directly certify.

Design principles
-----------------
- No LLM dependency: NL parsing uses a rule-based keyword extraction
  approach in Phase 7.  A real deployment swaps in an LLM backend by
  replacing the NLStrategy._parse() method — the rest of the pipeline
  is unchanged.
- Deterministic: given the same intent + state, produces the same
  ActionProposal (important for testing and reproducibility).
- Confidence scoring: every proposal carries a translation_confidence
  (0.0–1.0) that feeds into the Certification Gate confidence.
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

from cortex.models.action import Action, ActionSpec, ActionType, ActionConstraints, Pose


# ── Intent types ──────────────────────────────────────────────────────────────

class IntentType(str, Enum):
    NATURAL_LANGUAGE = "natural_language"
    GOAL_STATE       = "goal_state"
    DEMONSTRATION    = "demonstration"


@dataclass(frozen=True)
class NLIntent:
    """
    A natural-language intent string.

    Fields
    ------
    text        : the raw NL command ("pick up the red cube")
    source      : who issued this (e.g., "operator", "llm_planner")
    language    : ISO 639-1 language code (default "en")
    """
    text:     str
    source:   str  = "operator"
    language: str  = "en"


@dataclass(frozen=True)
class GoalStateIntent:
    """
    A desired goal state: what the world should look like after the action.

    Fields
    ------
    target_pose     : desired EE pose after the action
    object_id       : ID of the object to manipulate (optional)
    action_hint     : coarse action type hint from planner (optional)
    grasp_required  : whether the agent must grasp something
    constraints     : additional action constraints
    """
    target_pose:    Pose                   = field(default_factory=Pose)
    object_id:      str | None            = None
    action_hint:    ActionType | None     = None
    grasp_required: bool                  = False
    constraints:    ActionConstraints     = field(default_factory=ActionConstraints)


@dataclass(frozen=True)
class DemonstrationIntent:
    """
    A kinesthetic or teleoperated demonstration.

    Fields
    ------
    waypoints       : list of [x, y, z] positions from the demonstration
    timestamps_ms   : optional timestamps (ms) for each waypoint
    gripper_states  : optional gripper state (True=closed) per waypoint
    source          : e.g. "kinesthetic", "teleoperation", "replay"
    """
    waypoints:      list[list[float]]       = field(default_factory=list)
    timestamps_ms:  list[float]             = field(default_factory=list)
    gripper_states: list[bool]              = field(default_factory=list)
    source:         str                     = "kinesthetic"


# ── Output ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ActionProposal:
    """
    A structured, certifiable action produced by the translator.

    Fields
    ------
    action                : the Action object ready for CertificationGate
    intent_type           : which translator produced this
    translation_confidence: 0.0–1.0 confidence in the translation
    raw_intent            : the original intent that was translated
    notes                 : translator-specific diagnostic info
    latency_ms            : how long translation took
    """
    action:                Action
    intent_type:           IntentType
    translation_confidence: float         = 1.0
    raw_intent:            Any            = None
    notes:                 list[str]      = field(default_factory=list)
    latency_ms:            float          = 0.0


# ── NL keyword rules ──────────────────────────────────────────────────────────

_NL_RULES: list[tuple[re.Pattern, ActionType, float]] = [
    (re.compile(r"\b(pick\s*up|grab|grasp|take)\b", re.I), ActionType.GRASP,   0.85),
    (re.compile(r"\b(place|put\s*down|set\s*down|deposit)\b", re.I), ActionType.PLACE, 0.85),
    (re.compile(r"\b(release|let\s*go|drop|open\s*gripper)\b", re.I), ActionType.RELEASE, 0.9),
    (re.compile(r"\b(push|slide|shove)\b", re.I),          ActionType.PUSH,    0.8),
    (re.compile(r"\b(move|go\s*to|reach|approach|navigate)\b", re.I), ActionType.MOVE_EE, 0.7),
    (re.compile(r"\b(home|reset|zero)\b", re.I),           ActionType.MOVE_JOINT, 0.75),
]

_COORD_RE = re.compile(
    r"(?:at|to|toward|position)?\s*"
    r"\(?(-?\d+\.?\d*)[,\s]+(-?\d+\.?\d*)[,\s]+(-?\d+\.?\d*)\)?",
    re.I,
)


class _NLStrategy:
    """Rule-based NL intent → Action."""

    def translate(self, intent: NLIntent) -> ActionProposal:
        t0 = time.perf_counter()
        text = intent.text.strip()

        action_type, confidence = self._classify(text)
        target_pose, pose_notes = self._extract_pose(text)
        notes = pose_notes

        action = Action(
            spec=ActionSpec(
                action_type=action_type,
                target_pose=target_pose,
            ),
            source=f"nl_translator:{intent.source}",
            intent=text,
        )
        return ActionProposal(
            action=action,
            intent_type=IntentType.NATURAL_LANGUAGE,
            translation_confidence=confidence,
            raw_intent=intent,
            notes=notes,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    def _classify(self, text: str) -> tuple[ActionType, float]:
        for pattern, action_type, conf in _NL_RULES:
            if pattern.search(text):
                return action_type, conf
        return ActionType.MOVE_EE, 0.5   # fallback with low confidence

    def _extract_pose(self, text: str) -> tuple[Pose, list[str]]:
        notes: list[str] = []
        m = _COORD_RE.search(text)
        if m:
            x, y, z = float(m.group(1)), float(m.group(2)), float(m.group(3))
            return Pose(x=x, y=y, z=z), notes
        notes.append("no_coordinates_in_text; defaulting to (0,0,0)")
        return Pose(), notes


# ── Goal-state strategy ───────────────────────────────────────────────────────

class _GoalStateStrategy:
    """GoalStateIntent → Action (direct mapping, high confidence)."""

    def translate(self, intent: GoalStateIntent) -> ActionProposal:
        t0 = time.perf_counter()

        action_type = intent.action_hint or (
            ActionType.GRASP if intent.grasp_required else ActionType.MOVE_EE
        )
        notes: list[str] = []

        action = Action(
            spec=ActionSpec(
                action_type=action_type,
                target_pose=intent.target_pose,
                constraints=intent.constraints,
                target_object_id=intent.object_id,
            ),
            source="goal_state_translator",
            intent=(
                f"goal: {action_type.value} "
                f"object={intent.object_id or 'none'} "
                f"pose=({intent.target_pose.x:.3f},"
                f"{intent.target_pose.y:.3f},"
                f"{intent.target_pose.z:.3f})"
            ),
        )
        return ActionProposal(
            action=action,
            intent_type=IntentType.GOAL_STATE,
            translation_confidence=0.95,
            raw_intent=intent,
            notes=notes,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )


# ── Demonstration strategy ────────────────────────────────────────────────────

class _DemonstrationStrategy:
    """DemonstrationIntent → Action (extracts final waypoint as target pose)."""

    def translate(self, intent: DemonstrationIntent) -> ActionProposal:
        t0 = time.perf_counter()
        notes: list[str] = []
        confidence = 0.9

        if not intent.waypoints:
            # No waypoints: return a no-op MOVE_EE to current position
            notes.append("empty_demonstration; no waypoints provided")
            confidence = 0.3
            action = Action(
                spec=ActionSpec(action_type=ActionType.MOVE_EE, target_pose=Pose()),
                source=f"demonstration_translator:{intent.source}",
                intent="empty demonstration",
            )
            return ActionProposal(
                action=action,
                intent_type=IntentType.DEMONSTRATION,
                translation_confidence=confidence,
                raw_intent=intent,
                notes=notes,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        # Use final waypoint as target pose
        final_wp = intent.waypoints[-1]
        x = float(final_wp[0]) if len(final_wp) > 0 else 0.0
        y = float(final_wp[1]) if len(final_wp) > 1 else 0.0
        z = float(final_wp[2]) if len(final_wp) > 2 else 0.0
        target_pose = Pose(x=x, y=y, z=z)

        # Determine if demonstration ends with a grasp (gripper closed at end)
        ends_with_grasp = (
            intent.gripper_states
            and intent.gripper_states[-1] is True
        )
        action_type = ActionType.GRASP if ends_with_grasp else ActionType.MOVE_EE

        if len(intent.waypoints) < 3:
            notes.append("short_demonstration; fewer than 3 waypoints")
            confidence = 0.7

        action = Action(
            spec=ActionSpec(
                action_type=action_type,
                target_pose=target_pose,
            ),
            source=f"demonstration_translator:{intent.source}",
            intent=(
                f"demonstration: {len(intent.waypoints)} waypoints "
                f"from {intent.source}"
            ),
        )
        return ActionProposal(
            action=action,
            intent_type=IntentType.DEMONSTRATION,
            translation_confidence=confidence,
            raw_intent=intent,
            notes=notes,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )


# ── Top-level translator ──────────────────────────────────────────────────────

class IntentTranslator:
    """
    Top-level intent-to-action translator.

    Selects the correct strategy based on intent type and returns an
    ActionProposal ready for CertificationGate.certify().

    Usage
    -----
        translator = IntentTranslator()

        # Natural language
        proposal = translator.translate(NLIntent("pick up the red cube"))

        # Goal state
        proposal = translator.translate(GoalStateIntent(
            target_pose=Pose(x=0.4, y=0.0, z=0.5),
            object_id="red_cube",
            grasp_required=True,
        ))

        # Pass to gate
        decision = gate.certify(proposal.action, ctx)
    """

    def __init__(self) -> None:
        self._nl_strategy    = _NLStrategy()
        self._goal_strategy  = _GoalStateStrategy()
        self._demo_strategy  = _DemonstrationStrategy()

    def translate(
        self,
        intent: "NLIntent | GoalStateIntent | DemonstrationIntent",
    ) -> ActionProposal:
        """
        Translate an intent into an ActionProposal.

        Raises
        ------
        TypeError if the intent type is not recognised.
        """
        if isinstance(intent, NLIntent):
            return self._nl_strategy.translate(intent)
        if isinstance(intent, GoalStateIntent):
            return self._goal_strategy.translate(intent)
        if isinstance(intent, DemonstrationIntent):
            return self._demo_strategy.translate(intent)
        raise TypeError(
            f"Unsupported intent type: {type(intent).__name__}. "
            "Expected NLIntent, GoalStateIntent, or DemonstrationIntent."
        )

    def translate_nl(self, text: str, source: str = "operator") -> ActionProposal:
        """Convenience: translate a raw NL string."""
        return self.translate(NLIntent(text=text, source=source))

    def translate_goal(
        self,
        target_pose:    Pose,
        object_id:      str | None = None,
        grasp_required: bool       = False,
    ) -> ActionProposal:
        """Convenience: translate a goal pose."""
        return self.translate(GoalStateIntent(
            target_pose=target_pose,
            object_id=object_id,
            grasp_required=grasp_required,
        ))

    def translate_demo(
        self,
        waypoints: list[list[float]],
        gripper_states: list[bool] | None = None,
        source: str = "kinesthetic",
    ) -> ActionProposal:
        """Convenience: translate a waypoint demonstration."""
        return self.translate(DemonstrationIntent(
            waypoints=waypoints,
            gripper_states=gripper_states or [],
            source=source,
        ))
