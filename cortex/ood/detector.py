"""
OOD Detection Module
====================
Detects when the current robot state / action falls outside the
distribution seen during training or nominal operation.

Three detection methods:

1. EmbeddingDistanceDetector
   Compares the L2/cosine distance of the input embedding to the
   centroid (mean) of a calibration set.  Simple and fast.
   Threshold: distance > mean + k*std of calibration distances.

2. ConformalDetector
   Non-parametric conformal prediction: computes a p-value for
   each new sample against the calibration set using a distance-
   based non-conformity score.  Guaranteed false-positive rate at
   significance level α.

3. EnsembleOODDetector
   Combines multiple detectors with configurable weights.  Flags
   OOD if the weighted OOD score exceeds a threshold.

All detectors are:
  - Stateless after calibration (fit() once, detect() any time)
  - Thread-safe (read-only after fit)
  - Zero external ML dependencies (numpy only)

Integration with CertificationGate
-----------------------------------
The gate calls detector.detect(state_vector) during P2 validation.
If result.is_ood is True:
  - result.ood_score  → 0.0–1.0  (1.0 = maximally OOD)
  - result.method     → which detector triggered
  - result.confidence → p-value or distance ratio (method-dependent)

Wire-in example (in CertificationGate):
    ood_result = self._ood_detector.detect(state.to_vector())
    if ood_result.is_ood:
        return REPLAN_REQUIRED with OOD_ANOMALY failure mode
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


# ── Result ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class OODResult:
    """
    Result of an OOD detection call.

    Fields
    ------
    is_ood      : True if the sample is detected as out-of-distribution
    ood_score   : 0.0 (in-distribution) to 1.0 (maximally OOD)
    method      : name of the detector that produced this result
    confidence  : method-specific confidence (p-value, distance ratio, etc.)
    details     : optional extra diagnostic info
    latency_ms  : detection latency
    """
    is_ood:     bool
    ood_score:  float
    method:     str
    confidence: float = 1.0
    details:    dict  = field(default_factory=dict)
    latency_ms: float = 0.0


# ── Embedding distance detector ───────────────────────────────────────────────

class EmbeddingDistanceDetector:
    """
    Detects OOD via distance from the calibration distribution centroid.

    After fit(), stores:
      - centroid: mean of calibration embeddings
      - threshold: mean_dist + k * std_dist (default k=3.0)

    detect() returns is_ood=True if the distance from the query to the
    centroid exceeds the threshold.
    """

    def __init__(self, k: float = 3.0, metric: str = "l2") -> None:
        """
        Parameters
        ----------
        k      : threshold multiplier (mean_dist + k*std_dist)
        metric : "cosine" or "l2"
        """
        self.k       = k
        self.metric  = metric
        self._fitted = False
        self._centroid:   np.ndarray | None = None
        self._threshold:  float             = float("inf")
        self._mean_dist:  float             = 0.0
        self._std_dist:   float             = 0.0

    def fit(self, calibration_data: Sequence[Sequence[float]]) -> "EmbeddingDistanceDetector":
        """
        Fit on calibration (in-distribution) embeddings.

        Parameters
        ----------
        calibration_data : list of embedding vectors (must all be same length)
        """
        X = np.array(calibration_data, dtype=np.float32)
        if X.ndim != 2 or len(X) < 2:
            raise ValueError("calibration_data must be a 2-D array with >= 2 rows")

        if self.metric == "cosine":
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms = np.where(norms < 1e-8, 1.0, norms)
            X = X / norms

        self._centroid = X.mean(axis=0)

        dists = self._dist_to_centroid(X)
        self._mean_dist = float(dists.mean())
        self._std_dist  = float(dists.std())
        self._threshold = self._mean_dist + self.k * max(self._std_dist, 1e-6)
        self._fitted    = True
        return self

    def detect(self, embedding: Sequence[float]) -> OODResult:
        t0  = time.perf_counter()
        emb = np.array(embedding, dtype=np.float32)

        if not self._fitted or self._centroid is None:
            # Return in-distribution if not fitted (graceful no-op)
            return OODResult(
                is_ood=False, ood_score=0.0, method="embedding_distance",
                confidence=1.0,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        if self.metric == "cosine":
            norm = np.linalg.norm(emb)
            if norm > 1e-8:
                emb = emb / norm

        dist = float(np.linalg.norm(emb - self._centroid))

        ood_score = min(1.0, dist / max(self._threshold, 1e-8))
        is_ood    = dist > self._threshold

        return OODResult(
            is_ood=is_ood,
            ood_score=ood_score,
            method="embedding_distance",
            confidence=dist / max(self._threshold, 1e-8),
            details={
                "distance":   round(dist, 6),
                "threshold":  round(self._threshold, 6),
                "mean_dist":  round(self._mean_dist, 6),
                "std_dist":   round(self._std_dist, 6),
            },
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    def _dist_to_centroid(self, X: np.ndarray) -> np.ndarray:
        assert self._centroid is not None
        return np.linalg.norm(X - self._centroid, axis=1)

    @property
    def is_fitted(self) -> bool:
        return self._fitted


# ── Conformal detector ────────────────────────────────────────────────────────

class ConformalDetector:
    """
    Non-parametric conformal OOD detector.

    Computes a p-value for each new sample: the fraction of calibration
    samples with a non-conformity score >= that of the query.

    Non-conformity score: distance to the k-th nearest neighbour (kNN).

    is_ood when p_value < alpha (significance level, default 0.05).

    Properties
    ----------
    - Distribution-free valid coverage: FPR <= alpha when calibration is i.i.d.
    - Computationally O(N*d) per query where N = calibration size, d = dim.
    """

    def __init__(
        self,
        alpha:    float = 0.05,
        n_neighbors: int = 5,
    ) -> None:
        self.alpha       = alpha
        self.n_neighbors = n_neighbors
        self._fitted     = False
        self._cal_data:  np.ndarray | None = None
        self._cal_scores: np.ndarray | None = None

    def fit(self, calibration_data: Sequence[Sequence[float]]) -> "ConformalDetector":
        X = np.array(calibration_data, dtype=np.float32)
        if X.ndim != 2 or len(X) < self.n_neighbors + 1:
            raise ValueError(
                f"calibration_data must have >= {self.n_neighbors + 1} rows"
            )
        self._cal_data   = X
        self._cal_scores = self._nonconformity_scores(X)
        self._fitted     = True
        return self

    def detect(self, embedding: Sequence[float]) -> OODResult:
        t0  = time.perf_counter()
        emb = np.array(embedding, dtype=np.float32).reshape(1, -1)

        if not self._fitted or self._cal_data is None or self._cal_scores is None:
            return OODResult(
                is_ood=False, ood_score=0.0, method="conformal",
                confidence=1.0,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        # Non-conformity score for the query
        query_score = self._knn_score(emb, self._cal_data)

        # p-value: fraction of calibration scores >= query_score
        p_value = float((self._cal_scores >= query_score).mean())

        # OOD score: 1 - p_value (high p means in-distribution)
        ood_score = 1.0 - p_value
        is_ood    = p_value < self.alpha

        return OODResult(
            is_ood=is_ood,
            ood_score=ood_score,
            method="conformal",
            confidence=p_value,
            details={
                "p_value":     round(p_value, 6),
                "alpha":       self.alpha,
                "query_score": round(float(query_score), 6),
            },
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    def _nonconformity_scores(self, X: np.ndarray) -> np.ndarray:
        """kNN distance for each point in X (leaving self out)."""
        n = len(X)
        scores = np.zeros(n, dtype=np.float32)
        k = min(self.n_neighbors, n - 1)
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            dists = np.linalg.norm(X[mask] - X[i], axis=1)
            knn   = np.sort(dists)[:k]
            scores[i] = float(knn.mean())
        return scores

    def _knn_score(self, query: np.ndarray, X: np.ndarray) -> float:
        """kNN distance from query to calibration set."""
        dists = np.linalg.norm(X - query, axis=1)
        k     = min(self.n_neighbors, len(dists))
        return float(np.sort(dists)[:k].mean())

    @property
    def is_fitted(self) -> bool:
        return self._fitted


# ── Ensemble ──────────────────────────────────────────────────────────────────

@dataclass
class _DetectorEntry:
    detector: "EmbeddingDistanceDetector | ConformalDetector"
    weight:   float


class EnsembleOODDetector:
    """
    Weighted ensemble of OOD detectors.

    Aggregates ood_score from each member detector using weights.
    Flags is_ood if the weighted average exceeds the ensemble threshold.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self._threshold = threshold
        self._members:  list[_DetectorEntry] = []

    def add(
        self,
        detector: "EmbeddingDistanceDetector | ConformalDetector",
        weight: float = 1.0,
    ) -> "EnsembleOODDetector":
        self._members.append(_DetectorEntry(detector=detector, weight=weight))
        return self

    def fit(self, calibration_data: Sequence[Sequence[float]]) -> "EnsembleOODDetector":
        """Fit all member detectors on the same calibration data."""
        for entry in self._members:
            entry.detector.fit(calibration_data)
        return self

    def detect(self, embedding: Sequence[float]) -> OODResult:
        t0 = time.perf_counter()
        if not self._members:
            return OODResult(
                is_ood=False, ood_score=0.0, method="ensemble",
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        total_weight = sum(e.weight for e in self._members)
        weighted_score = 0.0
        member_results: dict[str, float] = {}

        for entry in self._members:
            r = entry.detector.detect(embedding)
            weighted_score += r.ood_score * (entry.weight / total_weight)
            member_results[r.method] = round(r.ood_score, 4)

        is_ood = weighted_score > self._threshold

        return OODResult(
            is_ood=is_ood,
            ood_score=round(weighted_score, 4),
            method="ensemble",
            confidence=1.0 - weighted_score,
            details={"member_scores": member_results, "threshold": self._threshold},
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    @property
    def is_fitted(self) -> bool:
        return all(e.detector.is_fitted for e in self._members)


# ── Convenience factory ───────────────────────────────────────────────────────

class OODDetector:
    """
    Pre-configured ensemble OOD detector for use in CertificationGate.

    Combines EmbeddingDistanceDetector (weight=0.6) and
    ConformalDetector (weight=0.4) into a single ensemble.

    Usage
    -----
        detector = OODDetector()
        detector.fit(calibration_vectors)
        result = detector.detect(state_vector)
        if result.is_ood:
            ...
    """

    def __init__(
        self,
        distance_k:       float = 3.0,
        conformal_alpha:  float = 0.05,
        conformal_knn:    int   = 5,
        ensemble_threshold: float = 0.5,
    ) -> None:
        self._distance   = EmbeddingDistanceDetector(k=distance_k)
        self._conformal  = ConformalDetector(
            alpha=conformal_alpha, n_neighbors=conformal_knn
        )
        self._ensemble   = (
            EnsembleOODDetector(threshold=ensemble_threshold)
            .add(self._distance,  weight=0.6)
            .add(self._conformal, weight=0.4)
        )
        self._fitted = False

    def fit(self, calibration_data: Sequence[Sequence[float]]) -> "OODDetector":
        """
        Calibrate the detector on in-distribution samples.

        Parameters
        ----------
        calibration_data : list of state/embedding vectors
                           Must have >= 7 samples (kNN requirement).
        """
        self._ensemble.fit(calibration_data)
        self._fitted = True
        return self

    def detect(self, embedding: Sequence[float]) -> OODResult:
        """
        Detect if the given embedding is out-of-distribution.

        Returns OODResult with is_ood=False if not yet calibrated
        (graceful degradation — don't block if no calibration data).
        """
        return self._ensemble.detect(embedding)

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit_from_robot_states(
        self,
        states: "list",
        to_vector_fn: "callable | None" = None,
    ) -> "OODDetector":
        """
        Convenience: fit from a list of RobotState objects.

        to_vector_fn(state) → list[float]
        If None, uses a built-in default that concatenates
        ee_position + ee_orientation + joint_positions.
        """
        if to_vector_fn is None:
            to_vector_fn = _default_state_to_vector
        vectors = [to_vector_fn(s) for s in states]
        return self.fit(vectors)


def _default_state_to_vector(state: "object") -> list[float]:
    """Default RobotState → float vector for OOD calibration."""
    pos   = list(getattr(state, "ee_position",   [0.0, 0.0, 0.0]))[:3]
    ori   = list(getattr(state, "ee_orientation", [1.0, 0.0, 0.0, 0.0]))[:4]
    jpos  = list(getattr(state, "joint_positions", []) or [])[:6]
    jvel  = list(getattr(state, "joint_velocities", []) or [])[:6]
    # Pad to fixed length 19 = 3 + 4 + 6 + 6
    jpos  += [0.0] * max(0, 6 - len(jpos))
    jvel  += [0.0] * max(0, 6 - len(jvel))
    return pos + ori + jpos + jvel
