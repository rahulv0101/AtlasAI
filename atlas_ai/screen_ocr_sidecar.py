from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass
class OCRSnapshot:
    timestamp: str
    symbol: str
    bid_text: str
    ask_text: str
    spread_text: str
    confidence: float
    advisory_only: bool = True


class ScreenOCRSidecar:
    # purpose: provide read-only screen OCR observability for HFM terminal UI.
    # what it does: captures OCR context as advisory data only (never execution authority).
    def __init__(self, output_path: str = "storage/screen_context.json") -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def run_once(self) -> dict[str, Any]:
        # FIX: safe placeholder snapshot to keep stack stable when OCR deps are not installed.
        snap = OCRSnapshot(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            symbol="XAUUSD",
            bid_text="",
            ask_text="",
            spread_text="",
            confidence=0.0,
            advisory_only=True,
        )
        payload = {
            "enabled": True,
            "advisory_only": True,
            "source": "screen_ocr_sidecar",
            "status": "unavailable_until_ocr_backend_configured",
            "snapshot": asdict(snap),
            "note": "OCR sidecar is read-only and cannot bypass ExecutionGate.",
        }
        self.output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload


def main() -> int:
    ScreenOCRSidecar().run_once()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

