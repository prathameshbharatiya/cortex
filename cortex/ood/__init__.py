"""
Out-of-Distribution (OOD) Detection Module
==========================================

Three complementary OOD detection methods, composable into an ensemble:

  EmbeddingDistanceDetector  — cosine distance from training distribution centroid
  ConformalDetector          — conformal prediction p-values from calibration set
  EnsembleOODDetector        — majority vote / weighted average of the above

The main entry point for the CertificationGate is OODDetector (a pre-configured
ensemble).  Lower-level detectors are exposed for testing and specialised use.

Usage
-----
    from cortex.ood import OODDetector
    detector = OODDetector()
    result   = detector.detect(state_vector)
    if result.is_ood:
        # raise SAFE_HALT or REPLAN_REQUIRED
"""

from cortex.ood.detector import (
    OODDetector,
    OODResult,
    EmbeddingDistanceDetector,
    ConformalDetector,
    EnsembleOODDetector,
)

__all__ = [
    "OODDetector",
    "OODResult",
    "EmbeddingDistanceDetector",
    "ConformalDetector",
    "EnsembleOODDetector",
]
