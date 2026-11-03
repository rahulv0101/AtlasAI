from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atlas_ai.config.settings import load_settings
from atlas_ai.memory.knowledge_store import KnowledgeStore
from atlas_ai.ui_server import create_ui_app, serve_ui_app


def main() -> None:
    settings = load_settings()
    app = create_ui_app(KnowledgeStore(settings.storage_dir))
    serve_ui_app(app, "127.0.0.1", 8765)


if __name__ == "__main__":
    main()
