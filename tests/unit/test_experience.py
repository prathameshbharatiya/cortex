"""
tests/unit/test_experience.py
==============================
Experience Tracker and memory record lifecycle.
"""

from __future__ import annotations

import time
import pytest

from cortex.experience.record import ExperienceRecord, OutcomeClass, OutcomeSpec
from cortex.experience.tracker import ExperienceTracker
from cortex.experience.lifecycle import LifecycleManager, LifecycleConfig
from cortex.memory.gateway import MemoryGateway
from cortex.memory.adapters.in_memory import InMemoryAdapter
from cortex.models.decision import CertificationState

from tests.conftest import make_context, make_action, make_memory_record


@pytest.fixture
def tracker_gateway():
    gw = MemoryGateway()
    gw.register(InMemoryAdapter(name="test"), priority=1)
    gw.start()
    tracker = ExperienceTracker(gateway=gw, platform_id="test", deployment_id="unit")
    yield tracker, gw
    gw.stop()


class TestExperienceTracker:

    def test_record_simple_success(self, tracker_gateway):
        tracker, gw = tracker_gateway
        exp = tracker.record_simple(
            task="pick block",
            outcome_class=OutcomeClass.SUCCESS,
            trace_id="trace-001",
            ctx_id="ctx-001",
            action_id="act-001",
            certification_state="EXECUTE",
            predicted_confidence=0.85,
        )
        assert exp is not None
        assert exp.experience_id
        assert exp.outcome.outcome_class == OutcomeClass.SUCCESS

    def test_record_simple_failure(self, tracker_gateway):
        tracker, gw = tracker_gateway
        exp = tracker.record_simple(
            task="grasp object",
            outcome_class=OutcomeClass.FAILURE,
            trace_id="trace-002",
            ctx_id="ctx-002",
            action_id="act-002",
            certification_state="EXECUTE",
            predicted_confidence=0.72,
        )
        assert exp.outcome.outcome_class == OutcomeClass.FAILURE

    def test_surprise_score_low_when_expected(self, tracker_gateway):
        tracker, gw = tracker_gateway
        # Predicted 0.95 → got success → low surprise
        exp = tracker.record_simple(
            task="test", outcome_class=OutcomeClass.SUCCESS,
            trace_id="t", ctx_id="c", action_id="a",
            certification_state="EXECUTE", predicted_confidence=0.95,
        )
        assert exp.surprise_score is not None
        assert exp.surprise_score < 0.5

    def test_surprise_score_high_when_unexpected(self, tracker_gateway):
        tracker, gw = tracker_gateway
        # Predicted 0.95 → got failure → high surprise
        exp = tracker.record_simple(
            task="test", outcome_class=OutcomeClass.FAILURE,
            trace_id="t", ctx_id="c", action_id="a",
            certification_state="EXECUTE", predicted_confidence=0.95,
        )
        assert exp.surprise_score > 0.4

    def test_experience_has_timestamp(self, tracker_gateway):
        tracker, gw = tracker_gateway
        before = time.time()
        exp = tracker.record_simple(
            task="test", outcome_class=OutcomeClass.SUCCESS,
            trace_id="t", ctx_id="c", action_id="a",
            certification_state="EXECUTE",
        )
        assert exp.timestamp >= before

    def test_experience_ids_are_unique(self, tracker_gateway):
        tracker, gw = tracker_gateway
        ids = set()
        for i in range(10):
            exp = tracker.record_simple(
                task=f"task_{i}", outcome_class=OutcomeClass.SUCCESS,
                trace_id=f"t{i}", ctx_id=f"c{i}", action_id=f"a{i}",
                certification_state="EXECUTE",
            )
            ids.add(exp.experience_id)
        assert len(ids) == 10

    def test_platform_id_stamped(self, tracker_gateway):
        tracker, gw = tracker_gateway
        exp = tracker.record_simple(
            task="test", outcome_class=OutcomeClass.SUCCESS,
            trace_id="t", ctx_id="c", action_id="a",
            certification_state="EXECUTE",
        )
        assert exp.platform_id == "test"

    def test_outcome_classes_all_recordable(self, tracker_gateway):
        tracker, gw = tracker_gateway
        for i, oc in enumerate(OutcomeClass):
            exp = tracker.record_simple(
                task="test", outcome_class=oc,
                trace_id=f"t{i}", ctx_id=f"c{i}", action_id=f"a{i}",
                certification_state="EXECUTE",
            )
            assert exp is not None


class TestLifecycleManager:

    def test_lifecycle_starts_and_stops(self):
        gw = MemoryGateway()
        gw.register(InMemoryAdapter(name="lc_test"), priority=1)
        gw.start()
        cfg = LifecycleConfig(
            maintenance_interval_s=9999,
            decay_check_interval_s=9999,
        )
        lm = LifecycleManager(gateway=gw, config=cfg)
        lm.start()
        time.sleep(0.05)
        lm.stop()
        gw.stop()

    def test_manual_maintenance_runs(self):
        gw = MemoryGateway()
        a = InMemoryAdapter(name="maint_test")
        gw.register(a, priority=1)
        gw.start()
        # Store some records
        for _ in range(5):
            a.store(make_memory_record())
        lm = LifecycleManager(gateway=gw, config=LifecycleConfig())
        report = lm.run_maintenance()
        assert isinstance(report, list)
        assert report[0].records_before >= 0
        gw.stop()

    def test_maintenance_removes_invalid_records(self):
        gw = MemoryGateway()
        a = InMemoryAdapter(name="del_test")
        gw.register(a, priority=1)
        gw.start()
        # Store valid and invalid
        a.store(make_memory_record())
        a.store(make_memory_record(invalid=True))
        lm = LifecycleManager(gateway=gw, config=LifecycleConfig())
        lm.run_maintenance()
        results = a.retrieve(__import__("cortex.memory.adapters.base", fromlist=["MemoryQuery"]).MemoryQuery(query_text="test", top_k=20))
        for r in results:
            assert r.is_currently_valid
        gw.stop()


class TestOutcomeClass:

    def test_all_classes_are_strings(self):
        for oc in OutcomeClass:
            assert isinstance(oc.value, str)

    def test_success_value(self):
        assert OutcomeClass.SUCCESS.value == "success"

    def test_failure_value(self):
        assert OutcomeClass.FAILURE.value == "failure"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
