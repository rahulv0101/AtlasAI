from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from atlas_ai.config.settings import load_settings


DEFAULT_UI_HOST = "127.0.0.1"
DEFAULT_UI_PORT = 8765


@dataclass(frozen=True)
class AtlasWakeConfig:
    project_root: str
    python_executable: str
    mt5_executable_path: str
    launch_profile: str
    backend_launch_command: list[str]
    frontend_launch_command: list[str]
    comparison_root: str
    comparison_run_id: str
    paper_preset: str
    poll_seconds: int
    ui_host: str
    ui_port: int
    backend_ready_timeout_seconds: int
    frontend_ready_timeout_seconds: int
    mt5_ready_timeout_seconds: int
    health_stale_after_seconds: int
    audio_model_path: str
    audio_sample_rate: int
    audio_block_size: int
    audio_device_name: str
    audio_clap_enabled: bool
    audio_double_clap_window_seconds: float
    audio_clap_arm_seconds: float
    audio_clap_peak_threshold: float
    trigger_phrase: str = "Atlas wake"
    sleep_phrase: str = "Atlas sleep"

    @property
    def api_base_url(self) -> str:
        return f"http://{self.ui_host}:{self.ui_port}"

    @property
    def status_url(self) -> str:
        return f"{self.api_base_url}/api/status"

    def to_safe_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if str(self.launch_profile or "").strip().lower() == "guarded_demo":
            payload["comparison_root"] = ""
            payload["comparison_run_id"] = ""
        return payload


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_python(project_root: Path) -> Path:
    candidates = [
        project_root / ".venv" / "Scripts" / "python.exe",
        Path(sys.executable),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path(sys.executable)


def _config_file_path(project_root: Path) -> Path:
    return project_root / "atlas_wake_config.json"


def _load_json_config(project_root: Path) -> dict[str, Any]:
    path = _config_file_path(project_root)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to parse {path.name}: {exc}") from exc


def _as_command(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [part for part in value.strip().split(" ") if part]
    return []


def _contains_forbidden_guarded_demo_routing(command: list[str]) -> str:
    joined = " ".join(str(x) for x in list(command or [])).strip().lower()
    if not joined:
        return ""
    if "paper-compare" in joined:
        return "paper-compare"
    if "--comparison-root" in joined:
        return "--comparison-root"
    if "storage_compare_runs" in joined:
        return "storage_compare_runs"
    return ""


def load_atlas_wake_config() -> AtlasWakeConfig:
    project_root = _project_root()
    json_config = _load_json_config(project_root)
    settings = load_settings()
    python_executable = Path(
        str(
            json_config.get("python_executable")
            or os.getenv("ATLAS_WAKE_PYTHON", "")
            or _default_python(project_root)
        )
    )
    comparison_root = Path(
        str(
            json_config.get("comparison_root")
            or os.getenv("ATLAS_WAKE_COMPARISON_ROOT", "")
            or (project_root / "storage_compare_runs")
        )
    )
    comparison_run_id = str(
        json_config.get("comparison_run_id")
        or os.getenv("ATLAS_WAKE_COMPARISON_RUN_ID", "")
        or "atlas_wake_latest"
    ).strip()
    paper_preset = str(
        json_config.get("paper_preset")
        or os.getenv("ATLAS_WAKE_PAPER_PRESET", "")
        or "conservative_filter_rollout_v1"
    ).strip()
    poll_seconds = int(
        json_config.get("poll_seconds")
        or os.getenv("ATLAS_WAKE_POLL_SECONDS", "")
        or 60
    )
    ui_host = str(
        json_config.get("ui_host")
        or os.getenv("ATLAS_WAKE_UI_HOST", "")
        or DEFAULT_UI_HOST
    ).strip()
    ui_port = int(
        json_config.get("ui_port")
        or os.getenv("ATLAS_WAKE_UI_PORT", "")
        or DEFAULT_UI_PORT
    )
    launch_profile = str(
        json_config.get("launch_profile")
        or os.getenv("ATLAS_WAKE_LAUNCH_PROFILE", "")
        or "guarded_demo"
    ).strip().lower()
    if launch_profile not in {"guarded_demo", "paper_compare"}:
        launch_profile = "guarded_demo"

    backend_launch_command = _as_command(json_config.get("backend_launch_command"))
    if not backend_launch_command:
        if launch_profile == "paper_compare":
            backend_launch_command = [
                str(python_executable),
                "-m",
                "atlas_ai.main",
                "paper-compare",
                "--poll-seconds",
                str(poll_seconds),
                "--comparison-root",
                str(comparison_root),
                "--comparison-run-id",
                comparison_run_id,
                "--paper-preset",
                paper_preset,
                "--resume",
                "--allow-market-closed",
            ]
        else:
            backend_launch_command = [
                str(python_executable),
                "-m",
                "atlas_ai.main",
                "live",
                "--poll-seconds",
                str(poll_seconds),
                "--mode-override",
                "live",
            ]

    frontend_launch_command = _as_command(json_config.get("frontend_launch_command"))
    if not frontend_launch_command:
        if launch_profile == "paper_compare":
            frontend_launch_command = [
                str(python_executable),
                "-m",
                "atlas_ai.main",
                "ui",
                "--comparison-root",
                str(comparison_root),
                "--comparison-run-id",
                comparison_run_id,
            ]
        else:
            frontend_launch_command = [
                str(python_executable),
                "-m",
                "atlas_ai.main",
                "ui",
                "--mode-override",
                "live",
            ]

    if launch_profile == "guarded_demo":
        backend_forbidden = _contains_forbidden_guarded_demo_routing(backend_launch_command)
        if backend_forbidden:
            raise RuntimeError(
                "Invalid guarded_demo backend command contains forbidden paper routing token: "
                f"{backend_forbidden}"
            )
        frontend_forbidden = _contains_forbidden_guarded_demo_routing(frontend_launch_command)
        if frontend_forbidden:
            raise RuntimeError(
                "Invalid guarded_demo frontend command contains forbidden paper routing token: "
                f"{frontend_forbidden}"
            )

    return AtlasWakeConfig(
        project_root=str(project_root),
        python_executable=str(python_executable),
        mt5_executable_path=str(
            json_config.get("mt5_executable_path")
            or os.getenv("ATLAS_WAKE_MT5_PATH", "")
            or settings.hfm_terminal_path
        ).strip(),
        launch_profile=launch_profile,
        backend_launch_command=backend_launch_command,
        frontend_launch_command=frontend_launch_command,
        comparison_root=str(comparison_root),
        comparison_run_id=comparison_run_id,
        paper_preset=paper_preset,
        poll_seconds=poll_seconds,
        ui_host=ui_host,
        ui_port=ui_port,
        backend_ready_timeout_seconds=int(
            json_config.get("backend_ready_timeout_seconds")
            or os.getenv("ATLAS_WAKE_BACKEND_READY_TIMEOUT", "")
            or 180
        ),
        frontend_ready_timeout_seconds=int(
            json_config.get("frontend_ready_timeout_seconds")
            or os.getenv("ATLAS_WAKE_FRONTEND_READY_TIMEOUT", "")
            or 90
        ),
        mt5_ready_timeout_seconds=int(
            json_config.get("mt5_ready_timeout_seconds")
            or os.getenv("ATLAS_WAKE_MT5_READY_TIMEOUT", "")
            or 90
        ),
        health_stale_after_seconds=int(
            json_config.get("health_stale_after_seconds")
            or os.getenv("ATLAS_WAKE_HEALTH_STALE_AFTER_SECONDS", "")
            or max(180, poll_seconds * 3)
        ),
        audio_model_path=str(
            json_config.get("audio_model_path")
            or os.getenv("ATLAS_AUDIO_MODEL_PATH", "")
            or (project_root / "models" / "vosk-model-small-en-us-0.15")
        ).strip(),
        audio_sample_rate=int(
            json_config.get("audio_sample_rate")
            or os.getenv("ATLAS_AUDIO_SAMPLE_RATE", "")
            or 16000
        ),
        audio_block_size=int(
            json_config.get("audio_block_size")
            or os.getenv("ATLAS_AUDIO_BLOCK_SIZE", "")
            or 8000
        ),
        audio_device_name=str(
            json_config.get("audio_device_name")
            or os.getenv("ATLAS_AUDIO_DEVICE_NAME", "")
            or ""
        ).strip(),
        audio_clap_enabled=str(
            json_config.get("audio_clap_enabled")
            or os.getenv("ATLAS_AUDIO_CLAP_ENABLED", "false")
        ).strip().lower() in {"1", "true", "yes", "on"},
        audio_double_clap_window_seconds=float(
            json_config.get("audio_double_clap_window_seconds")
            or os.getenv("ATLAS_AUDIO_DOUBLE_CLAP_WINDOW_SECONDS", "")
            or 1.5
        ),
        audio_clap_arm_seconds=float(
            json_config.get("audio_clap_arm_seconds")
            or os.getenv("ATLAS_AUDIO_CLAP_ARM_SECONDS", "")
            or 8.0
        ),
        audio_clap_peak_threshold=float(
            json_config.get("audio_clap_peak_threshold")
            or os.getenv("ATLAS_AUDIO_CLAP_PEAK_THRESHOLD", "")
            or 12000.0
        ),
        trigger_phrase=str(
            json_config.get("trigger_phrase")
            or os.getenv("ATLAS_WAKE_TRIGGER_PHRASE", "")
            or "Atlas wake"
        ).strip(),
        sleep_phrase=str(
            json_config.get("sleep_phrase")
            or os.getenv("ATLAS_SLEEP_TRIGGER_PHRASE", "")
            or "Atlas sleep"
        ).strip(),
    )
