"""
Geometry Encoder
================
Encodes a robot's end-effector pose (position + orientation) into a
compact geometry embedding used to anchor episodic memories spatially.

Design (Chameleon-inspired):
  - Position is encoded with Fourier positional features (sin/cos at
    multiple frequencies) — gives the network a smooth spatial basis.
  - Orientation quaternion is normalised and appended directly.
  - The combined vector is projected to a fixed embedding_dim via a
    learned or fixed random projection matrix.

In Phase 4 the projection uses a fixed (seeded) random matrix so there
are no training dependencies.  In production this is replaced with a
small MLP trained on robot trajectory data.

Embedding properties:
  - Deterministic for same (position, orientation) input.
  - Nearby poses produce similar embeddings (smooth interpolation).
  - Dimension: embedding_dim (default 64).
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


class GeometryEncoder:
    """
    Encodes 3-D pose (position + quaternion) into a geometry embedding.

    Parameters
    ----------
    embedding_dim : int
        Output dimensionality (default 64).
    n_freqs : int
        Number of Fourier frequency bands for position (default 8).
    seed : int
        RNG seed for the fixed random projection matrix.
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        n_freqs:       int = 8,
        seed:          int = 42,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.n_freqs       = n_freqs

        # Fourier features: sin + cos for each freq, for 3 position dims
        # → raw_dim = 3 * n_freqs * 2  +  4 (quaternion)
        self._raw_dim = 3 * n_freqs * 2 + 4

        # Fixed random projection: raw_dim → embedding_dim
        rng = np.random.default_rng(seed)
        self._projection: np.ndarray = rng.standard_normal(
            (self._raw_dim, embedding_dim)
        ).astype(np.float32)
        # Orthonormalise columns for stable norms
        if embedding_dim <= self._raw_dim:
            Q, _ = np.linalg.qr(self._projection)
            self._projection = Q[:, :embedding_dim]

    # ── Public API ────────────────────────────────────────────────────────────

    def encode(
        self,
        position:    Sequence[float],
        orientation: Sequence[float],
    ) -> np.ndarray:
        """
        Encode a pose into a geometry embedding.

        Parameters
        ----------
        position    : (x, y, z) in metres
        orientation : (qw, qx, qy, qz) normalised quaternion

        Returns
        -------
        embedding : float32 array of shape (embedding_dim,)
        """
        pos  = np.array(position,    dtype=np.float32)[:3]
        quat = np.array(orientation, dtype=np.float32)[:4]

        # Normalise quaternion
        qnorm = np.linalg.norm(quat)
        if qnorm > 1e-8:
            quat = quat / qnorm

        # Fourier positional features
        freqs    = 2.0 ** np.arange(self.n_freqs, dtype=np.float32)  # 1,2,4,...
        pos_feat = np.concatenate([
            np.sin(np.outer(pos, freqs).ravel()),
            np.cos(np.outer(pos, freqs).ravel()),
        ])  # shape: 3*n_freqs*2

        raw = np.concatenate([pos_feat, quat])  # shape: raw_dim

        # Project to embedding_dim
        emb = raw @ self._projection  # shape: embedding_dim
        norm = np.linalg.norm(emb)
        if norm > 1e-8:
            emb = emb / norm
        return emb.astype(np.float32)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two geometry embeddings."""
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-8 or nb < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def encode_batch(
        self,
        positions:    list[Sequence[float]],
        orientations: list[Sequence[float]],
    ) -> np.ndarray:
        """Encode a batch of poses. Returns shape (N, embedding_dim)."""
        return np.stack([
            self.encode(p, o)
            for p, o in zip(positions, orientations)
        ])
