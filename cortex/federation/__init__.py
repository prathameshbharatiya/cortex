"""
Federated Memory Network
========================
Enables multiple Cortex nodes (robots, deployments) to share episodic
memories while preserving privacy via differential privacy noise injection
and explicit consent management.

Exports:
  FederatedMemoryProtocol  — orchestrates share/receive across nodes
  FederatedNode            — describes a peer node
  PrivacyBudget            — tracks ε-DP budget consumption
  ConsentPolicy            — governs what each node is allowed to share
  FederatedRecord          — a privacy-processed memory ready for sharing
"""

from cortex.federation.protocol import (
    FederatedMemoryProtocol,
    FederatedNode,
    PrivacyBudget,
    ConsentPolicy,
    FederatedRecord,
)

__all__ = [
    "FederatedMemoryProtocol",
    "FederatedNode",
    "PrivacyBudget",
    "ConsentPolicy",
    "FederatedRecord",
]
