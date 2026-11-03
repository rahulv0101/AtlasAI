from atlas_ai.brain.answer_engine import AnswerEngine
from atlas_ai.brain.ai_decision_core import AIDecisionCore
from atlas_ai.brain.intent import classify_intent_candidates
from atlas_ai.brain.domain import detect_domain_candidates
from atlas_ai.brain.knowledge import BrainKnowledge
from atlas_ai.brain.memory import BrainMemory
import importlib.util
import sys
from pathlib import Path


def _load_legacy_brain_symbols() -> dict:
    symbols: dict = {}
    try:
        legacy_path = Path(__file__).resolve().parents[1] / "brain.py"
        if not legacy_path.exists():
            return symbols
        spec = importlib.util.spec_from_file_location("atlas_ai._legacy_brain_module", legacy_path)
        if spec is None or spec.loader is None:
            return symbols
        module = importlib.util.module_from_spec(spec)
        # Dataclasses and forward refs expect the module to be present in sys.modules
        # while executing, otherwise class creation can fail.
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        for name in ("AtlasBrain", "BrainSnapshot", "BrainTurn", "BrainMemory"):
            if hasattr(module, name):
                symbols[name] = getattr(module, name)
    except Exception:
        return {}
    return symbols


_legacy = _load_legacy_brain_symbols()
AtlasBrain = _legacy.get("AtlasBrain")
BrainSnapshot = _legacy.get("BrainSnapshot")
BrainTurn = _legacy.get("BrainTurn")

__all__ = [
    "AnswerEngine",
    "AIDecisionCore",
    "classify_intent_candidates",
    "detect_domain_candidates",
    "BrainKnowledge",
    "BrainMemory",
    "AtlasBrain",
    "BrainSnapshot",
    "BrainTurn",
]
