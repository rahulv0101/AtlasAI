from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from urllib.request import urlopen

from atlas_ai.atlas_wake_config import load_atlas_wake_config


DEFAULT_VOSK_MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"


def ensure_vosk_model(model_url: str = DEFAULT_VOSK_MODEL_URL) -> Path:
    config = load_atlas_wake_config()
    target = Path(config.audio_model_path)
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    archive_path = target.parent / "vosk-model-small-en-us-0.15.zip"
    with urlopen(model_url, timeout=120) as resp:
        archive_path.write_bytes(resp.read())
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(target.parent)
    archive_path.unlink(missing_ok=True)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download the local offline Vosk model for Atlas audio triggers.")
    parser.add_argument("--json", action="store_true", help="Print the installed model path as JSON.")
    args = parser.parse_args(argv)
    path = ensure_vosk_model()
    if args.json:
        print(json.dumps({"audio_model_path": str(path)}, indent=2))
    else:
        print(str(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
