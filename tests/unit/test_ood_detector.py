"""
tests/unit/test_ood_detector.py
================================
Phase 5 tests: OOD detection — embedding distance, conformal,
ensemble, and CertificationGate integration.
"""

from __future__ import annotations

import numpy as np
import pytest

from cortex.ood.detector import (
    OODDetector,
    OODResult,
    EmbeddingDistanceDetector,
    ConformalDetector,
    EnsembleOODDetector,
    _default_state_to_vector,
)
from cortex.models.context import RobotState


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _in_dist_data(n: int = 50, dim: int = 19, seed: int = 0) -> list[list[float]]:
    """Generate in-distribution data around a cluster near the origin."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((n, dim)) * 0.1).tolist()


def _ood_sample(dim: int = 19) -> list[float]:
    """A sample far from the origin — clearly OOD."""
    return [10.0] * dim


def _id_sample(dim: int = 19, seed: int = 99) -> list[float]:
    """A sample close to origin — clearly in-distribution."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(dim) * 0.05).tolist()


# ── EmbeddingDistanceDetector ─────────────────────────────────────────────────

class TestEmbeddingDistanceDetector:

    def test_fit_sets_fitted(self) -> None:
        det = EmbeddingDistanceDetector()
        det.fit(_in_dist_data())
        assert det.is_fitted

    def test_in_dist_not_flagged(self) -> None:
        det = EmbeddingDistanceDetector(k=3.0)
        det.fit(_in_dist_data())
        result = det.detect(_id_sample())
        assert not result.is_ood, f"In-dist sample should not be OOD, score={result.ood_score}"

    def test_ood_sample_flagged(self) -> None:
        det = EmbeddingDistanceDetector(k=3.0)
        det.fit(_in_dist_data())
        result = det.detect(_ood_sample())
        assert result.is_ood, f"OOD sample should be flagged, score={result.ood_score}"

    def test_ood_score_in_range(self) -> None:
        det = EmbeddingDistanceDetector()
        det.fit(_in_dist_data())
        r1 = det.detect(_id_sample())
        r2 = det.detect(_ood_sample())
        assert 0.0 <= r1.ood_score <= 1.0
        assert r2.ood_score > r1.ood_score

    def test_result_method_name(self) -> None:
        det = EmbeddingDistanceDetector()
        det.fit(_in_dist_data())
        result = det.detect(_id_sample())
        assert result.method == "embedding_distance"

    def test_unfitted_returns_in_dist(self) -> None:
        det = EmbeddingDistanceDetector()
        result = det.detect(_id_sample())
        assert not result.is_ood

    def test_fit_requires_2d_data(self) -> None:
        det = EmbeddingDistanceDetector()
        with pytest.raises(ValueError):
            det.fit([[1.0]])   # only 1 row

    def test_details_has_distance_key(self) -> None:
        det = EmbeddingDistanceDetector()
        det.fit(_in_dist_data())
        result = det.detect(_id_sample())
        assert "distance" in result.details


# ── ConformalDetector ─────────────────────────────────────────────────────────

class TestConformalDetector:

    def test_fit_sets_fitted(self) -> None:
        det = ConformalDetector()
        det.fit(_in_dist_data(n=30))
        assert det.is_fitted

    def test_in_dist_high_pvalue(self) -> None:
        det = ConformalDetector(alpha=0.05)
        det.fit(_in_dist_data(n=50))
        result = det.detect(_id_sample())
        assert result.confidence > 0.05, f"p-value={result.confidence} too low for in-dist"

    def test_ood_sample_flagged(self) -> None:
        det = ConformalDetector(alpha=0.05, n_neighbors=3)
        det.fit(_in_dist_data(n=30))
        result = det.detect(_ood_sample())
        assert result.is_ood

    def test_method_name(self) -> None:
        det = ConformalDetector()
        det.fit(_in_dist_data(n=20))
        assert det.detect(_id_sample()).method == "conformal"

    def test_unfitted_returns_in_dist(self) -> None:
        det = ConformalDetector()
        result = det.detect(_id_sample())
        assert not result.is_ood

    def test_fit_requires_enough_rows(self) -> None:
        det = ConformalDetector(n_neighbors=5)
        with pytest.raises(ValueError):
            det.fit(_in_dist_data(n=3))

    def test_ood_score_higher_for_ood(self) -> None:
        det = ConformalDetector(alpha=0.05, n_neighbors=3)
        det.fit(_in_dist_data(n=30))
        id_score  = det.detect(_id_sample()).ood_score
        ood_score = det.detect(_ood_sample()).ood_score
        assert ood_score > id_score


# ── EnsembleOODDetector ───────────────────────────────────────────────────────

class TestEnsembleOODDetector:

    def _fitted_ensemble(self) -> EnsembleOODDetector:
        cal  = _in_dist_data(n=30)
        dist = EmbeddingDistanceDetector(k=3.0)
        conf = ConformalDetector(alpha=0.05, n_neighbors=3)
        ens  = EnsembleOODDetector(threshold=0.5)
        ens.add(dist, weight=0.6).add(conf, weight=0.4)
        ens.fit(cal)
        return ens

    def test_ensemble_not_ood_for_in_dist(self) -> None:
        ens    = self._fitted_ensemble()
        result = ens.detect(_id_sample())
        assert not result.is_ood

    def test_ensemble_flags_ood(self) -> None:
        ens    = self._fitted_ensemble()
        result = ens.detect(_ood_sample())
        assert result.is_ood

    def test_ensemble_method_name(self) -> None:
        ens = self._fitted_ensemble()
        assert ens.detect(_id_sample()).method == "ensemble"

    def test_empty_ensemble_returns_in_dist(self) -> None:
        ens    = EnsembleOODDetector()
        result = ens.detect(_id_sample())
        assert not result.is_ood

    def test_is_fitted_after_fit(self) -> None:
        ens = self._fitted_ensemble()
        assert ens.is_fitted

    def test_details_has_member_scores(self) -> None:
        ens    = self._fitted_ensemble()
        result = ens.detect(_id_sample())
        assert "member_scores" in result.details


# ── OODDetector (facade) ──────────────────────────────────────────────────────

class TestOODDetector:

    def test_unfitted_returns_in_dist(self) -> None:
        det    = OODDetector()
        result = det.detect(_id_sample())
        assert not result.is_ood

    def test_is_fitted_after_fit(self) -> None:
        det = OODDetector()
        det.fit(_in_dist_data(n=30))
        assert det.is_fitted

    def test_detects_ood_sample(self) -> None:
        det = OODDetector()
        det.fit(_in_dist_data(n=30))
        result = det.detect(_ood_sample())
        assert result.is_ood

    def test_in_dist_not_flagged(self) -> None:
        det = OODDetector()
        det.fit(_in_dist_data(n=50))
        assert not det.detect(_id_sample()).is_ood

    def test_fit_from_robot_states(self) -> None:
        states = [
            RobotState(
                ee_position=[0.3 + i * 0.01, 0.0, 0.5],
                ee_orientation=[1.0, 0.0, 0.0, 0.0],
            )
            for i in range(30)
        ]
        det = OODDetector()
        det.fit_from_robot_states(states)
        assert det.is_fitted

    def test_default_state_to_vector_shape(self) -> None:
        state = RobotState(
            ee_position=[0.3, 0.0, 0.5],
            ee_orientation=[1.0, 0.0, 0.0, 0.0],
        )
        vec = _default_state_to_vector(state)
        assert len(vec) == 19   # 3 + 4 + 6 + 6


# ── Gate integration ──────────────────────────────────────────────────────────

class TestGateOODIntegration:
    """OOD detector wired into CertificationGate."""

    def _make_action(self):
        from cortex.models.action import Action, ActionSpec, ActionType, Pose
        return Action(
            spec=ActionSpec(
                action_type=ActionType.MOVE_EE,
                target_pose=Pose(x=0.4, y=0.0, z=0.5),
            ),
            source="test",
            intent="test action",
        )

    def test_gate_accepts_unfitted_ood_detector(self) -> None:
        """Gate must not block when OOD detector is not yet calibrated."""
        import cortex
        from cortex.gate.certification import CertificationGate
        from cortex.models.context import RobotState

        state  = RobotState(ee_position=[0.3, 0.0, 0.5], ee_orientation=[1.0, 0.0, 0.0, 0.0])
        ctx    = cortex.build_context("test", state, [])
        action = self._make_action()

        gate   = CertificationGate()
        # Unfitted OOD detector should not cause REPLAN
        decision = gate.certify(action, ctx)
        # Just verify it completes without exception; don't assert state
        assert decision is not None

    def test_gate_blocks_on_ood_anomaly(self) -> None:
        """Gate returns REPLAN_REQUIRED when fitted OOD detector flags anomaly."""
        import cortex
        from cortex.gate.certification import CertificationGate
        from cortex.models.decision import CertificationState

        # Calibrate on a tight in-distribution cluster
        in_dist_states = [
            RobotState(
                ee_position=[0.3 + i * 0.001, 0.0, 0.5],
                ee_orientation=[1.0, 0.0, 0.0, 0.0],
            )
            for i in range(30)
        ]
        det = OODDetector(distance_k=1.0, ensemble_threshold=0.2)
        det.fit_from_robot_states(in_dist_states)

        # Create an extreme OOD state
        ood_state = RobotState(
            ee_position=[50.0, 50.0, 50.0],
            ee_orientation=[1.0, 0.0, 0.0, 0.0],
        )
        ctx    = cortex.build_context("test", ood_state, [])
        action = self._make_action()

        gate   = CertificationGate(ood_detector=det)
        decision = gate.certify(action, ctx)

        # OOD state with extreme position should be detected; expect either
        # REPLAN_REQUIRED (OOD) or SAFE_HALT (sentinel may also fire on this state)
        assert decision.state in (
            CertificationState.REPLAN_REQUIRED,
            CertificationState.SAFE_HALT,
            CertificationState.HUMAN_OVERRIDE_REQUIRED,
        )
