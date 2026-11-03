from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from atlas_ai.voice.wake_loop import run_voice_loop


def _normalize_phrase(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").strip().lower()).strip()


@dataclass
class ClapState:
    clap_times: list[float]
    armed_until: float


def _contains_phrase(text: str, phrase: str) -> bool:
    normalized_text = _normalize_phrase(text)
    normalized_phrase = _normalize_phrase(phrase)
    if not normalized_text or not normalized_phrase:
        return False
    if normalized_text == normalized_phrase:
        return True
    pattern = rf"(?:^|\s){re.escape(normalized_phrase)}(?:\s|$)"
    if re.search(pattern, normalized_text):
        return True
    return SequenceMatcher(None, normalized_text, normalized_phrase).ratio() >= 0.88


def _should_trigger_phrase(text: str, config, clap_state: ClapState) -> str | None:
    if _contains_phrase(text, "atlas sleep"):
        return "Atlas sleep"
    wake_variants = [
        str(getattr(config, "trigger_phrase", "Atlas wake") or "Atlas wake"),
        "Atlas wake",
        "Atlas wake up",
    ]
    for wake_phrase in wake_variants:
        if _contains_phrase(text, wake_phrase):
            return wake_phrase
    return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Atlas voice listener (wake/sleep + Ask Atlas speech loop).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    parse_args(argv)
    return run_voice_loop()


if __name__ == "__main__":
    raise SystemExit(main())
