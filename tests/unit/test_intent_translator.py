"""
tests/unit/test_intent_translator.py
======================================
Phase 7 tests: intent-to-action translation.
"""

from __future__ import annotations

import pytest

from cortex.intent.translator import (
    IntentTranslator,
    NLIntent,
    GoalStateIntent,
    DemonstrationIntent,
    IntentType,
    ActionProposal,
)
from cortex.models.action import Action, ActionType, Pose


# ── NLIntent ──────────────────────────────────────────────────────────────────

class TestNLTranslation:

    def _t(self) -> IntentTranslator:
        return IntentTranslator()

    def test_pick_maps_to_grasp(self) -> None:
        proposal = self._t().translate_nl("pick up the red cube")
        assert proposal.action.spec.action_type == ActionType.GRASP

    def test_place_maps_to_place(self) -> None:
        proposal = self._t().translate_nl("place the object on the shelf")
        assert proposal.action.spec.action_type == ActionType.PLACE

    def test_release_maps_to_release(self) -> None:
        proposal = self._t().translate_nl("release the gripper")
        assert proposal.action.spec.action_type == ActionType.RELEASE

    def test_push_maps_to_push(self) -> None:
        proposal = self._t().translate_nl("push the box to the left")
        assert proposal.action.spec.action_type == ActionType.PUSH

    def test_move_maps_to_move_ee(self) -> None:
        proposal = self._t().translate_nl("move to the target position")
        assert proposal.action.spec.action_type == ActionType.MOVE_EE

    def test_unknown_defaults_to_move_ee(self) -> None:
        proposal = self._t().translate_nl("xyzzy frobnicator")
        assert proposal.action.spec.action_type == ActionType.MOVE_EE
        assert proposal.translation_confidence <= 0.6

    def test_coordinate_extraction(self) -> None:
        proposal = self._t().translate_nl("move to 0.4 0.0 0.5")
        pose = proposal.action.spec.target_pose
        assert pose is not None
        assert abs(pose.x - 0.4) < 1e-6
        assert abs(pose.z - 0.5) < 1e-6

    def test_no_coordinates_has_note(self) -> None:
        proposal = self._t().translate_nl("grab the cube")
        assert any("no_coordinates" in n for n in proposal.notes)

    def test_intent_type_is_nl(self) -> None:
        proposal = self._t().translate_nl("pick the cube")
        assert proposal.intent_type == IntentType.NATURAL_LANGUAGE

    def test_action_is_action_instance(self) -> None:
        proposal = self._t().translate_nl("pick up the cube")
        assert isinstance(proposal.action, Action)

    def test_latency_recorded(self) -> None:
        proposal = self._t().translate_nl("pick up the cube")
        assert proposal.latency_ms >= 0.0

    def test_high_confidence_for_known_verb(self) -> None:
        proposal = self._t().translate_nl("grasp the object")
        assert proposal.translation_confidence >= 0.8


# ── GoalStateIntent ───────────────────────────────────────────────────────────

class TestGoalStateTranslation:

    def _t(self) -> IntentTranslator:
        return IntentTranslator()

    def test_goal_maps_to_move_ee_by_default(self) -> None:
        proposal = self._t().translate_goal(Pose(x=0.4, y=0.0, z=0.5))
        assert proposal.action.spec.action_type == ActionType.MOVE_EE

    def test_grasp_required_maps_to_grasp(self) -> None:
        proposal = self._t().translate_goal(
            Pose(x=0.4, y=0.0, z=0.5), grasp_required=True
        )
        assert proposal.action.spec.action_type == ActionType.GRASP

    def test_pose_preserved(self) -> None:
        pose     = Pose(x=0.3, y=0.1, z=0.6)
        proposal = self._t().translate_goal(pose)
        assert proposal.action.spec.target_pose == pose

    def test_object_id_preserved(self) -> None:
        proposal = self._t().translate_goal(
            Pose(x=0.4, y=0.0, z=0.5), object_id="cube_1"
        )
        assert proposal.action.spec.target_object_id == "cube_1"

    def test_high_confidence(self) -> None:
        proposal = self._t().translate_goal(Pose(x=0.4, y=0.0, z=0.5))
        assert proposal.translation_confidence >= 0.9

    def test_intent_type_is_goal_state(self) -> None:
        proposal = self._t().translate_goal(Pose())
        assert proposal.intent_type == IntentType.GOAL_STATE

    def test_action_hint_overrides_type(self) -> None:
        intent   = GoalStateIntent(
            target_pose=Pose(x=0.4, y=0.0, z=0.5),
            action_hint=ActionType.PLACE,
        )
        proposal = self._t().translate(intent)
        assert proposal.action.spec.action_type == ActionType.PLACE


# ── DemonstrationIntent ───────────────────────────────────────────────────────

class TestDemonstrationTranslation:

    def _t(self) -> IntentTranslator:
        return IntentTranslator()

    def test_final_waypoint_becomes_target(self) -> None:
        waypoints = [[0.1, 0.0, 0.5], [0.2, 0.0, 0.5], [0.4, 0.0, 0.5]]
        proposal  = self._t().translate_demo(waypoints)
        pose = proposal.action.spec.target_pose
        assert abs(pose.x - 0.4) < 1e-6

    def test_empty_demo_has_low_confidence(self) -> None:
        proposal = self._t().translate_demo([])
        assert proposal.translation_confidence < 0.5

    def test_gripper_closed_maps_to_grasp(self) -> None:
        waypoints = [[0.1, 0.0, 0.5], [0.4, 0.0, 0.5]]
        proposal  = self._t().translate_demo(
            waypoints, gripper_states=[False, True]
        )
        assert proposal.action.spec.action_type == ActionType.GRASP

    def test_gripper_open_maps_to_move_ee(self) -> None:
        waypoints = [[0.1, 0.0, 0.5], [0.4, 0.0, 0.5]]
        proposal  = self._t().translate_demo(
            waypoints, gripper_states=[False, False]
        )
        assert proposal.action.spec.action_type == ActionType.MOVE_EE

    def test_short_demo_has_note(self) -> None:
        proposal = self._t().translate_demo([[0.1, 0.0, 0.5]])
        assert any("short" in n for n in proposal.notes)

    def test_intent_type_is_demonstration(self) -> None:
        proposal = self._t().translate_demo([[0.4, 0.0, 0.5]])
        assert proposal.intent_type == IntentType.DEMONSTRATION


# ── IntentTranslator dispatch ─────────────────────────────────────────────────

class TestIntentTranslatorDispatch:

    def test_unsupported_type_raises(self) -> None:
        t = IntentTranslator()
        with pytest.raises(TypeError):
            t.translate("not an intent")   # type: ignore

    def test_all_three_strategies_return_proposal(self) -> None:
        t = IntentTranslator()
        assert isinstance(t.translate(NLIntent("pick up")),             ActionProposal)
        assert isinstance(t.translate(GoalStateIntent()),               ActionProposal)
        assert isinstance(t.translate(DemonstrationIntent(waypoints=[[0.4, 0.0, 0.5]])), ActionProposal)

    def test_proposal_action_has_source(self) -> None:
        t = IntentTranslator()
        p = t.translate_nl("move to position")
        assert "nl_translator" in p.action.source
