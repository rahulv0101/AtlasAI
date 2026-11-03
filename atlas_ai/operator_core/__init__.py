"""Legacy operator-core compatibility exports.

Brain v2 and AskAtlasService are the canonical answer route for UI, typed,
conversation, and voice surfaces. This package remains available for older
tests and compatibility adapters only.
"""

from atlas_ai.operator_core.intent import classify_intent
from atlas_ai.operator_core.synthesis import synthesize_answer
from atlas_ai.operator_core.memory import ConversationMemory
from atlas_ai.operator_core.knowledge import KnowledgeBase
from atlas_ai.operator_core.answer_engine import AnswerEngine

LEGACY_COMPATIBILITY_ONLY = True

__all__ = [
    "LEGACY_COMPATIBILITY_ONLY",
    "classify_intent",
    "synthesize_answer",
    "ConversationMemory",
    "KnowledgeBase",
    "AnswerEngine",
]
