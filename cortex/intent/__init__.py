"""
Intent-to-Action Translation Layer
====================================
Translates high-level intents (natural language, goal states,
demonstrations) into concrete ActionProposals ready for CertificationGate.

Exports:
  IntentTranslator   — top-level translator (selects the right strategy)
  ActionProposal     — structured translation output
  NLIntent           — natural-language intent container
  GoalStateIntent    — target state descriptor
  DemonstrationIntent — kinesthetic/teleoperation demonstration
  IntentType         — enum of supported intent types
"""

from cortex.intent.translator import (
    IntentTranslator,
    ActionProposal,
    NLIntent,
    GoalStateIntent,
    DemonstrationIntent,
    IntentType,
)

__all__ = [
    "IntentTranslator",
    "ActionProposal",
    "NLIntent",
    "GoalStateIntent",
    "DemonstrationIntent",
    "IntentType",
]
