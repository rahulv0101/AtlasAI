from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from atlas_ai.atlas_process_guard import (
    backend_process_state,
    frontend_process_state,
    json_file_status,
    mt5_process_state,
    process_exists,
    stop_process,
    wait_for,
)
from atlas_ai.atlas_runtime_control import write_stop_request
from atlas_ai.atlas_wake_config import AtlasWakeConfig, load_atlas_wake_config


def _log_path(config: AtlasWakeConfig) -> Path:
    return Path(config.project_root) / "storage" / "atlas_wake.log"


def setup_logging(config: AtlasWakeConfig) -> None:
    log_path = _log_path(config)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )


def _stop_frontend(config: AtlasWakeConfig) -> dict[str, Any]:
    state = frontend_process_state()
    if not state.running or not state.pid:
        return {"stopped": True, "was_running": False, "detail": "frontend not running"}
    logging.info("Stopping frontend pid=%s", state.pid)
    stop_process(state.pid, force=False)
    stopped, _ = wait_for(lambda: not process_exists(state.pid), timeout_seconds=20, poll_seconds=1.0)
    if not stopped:
        stop_process(state.pid, force=True)
        stopped, _ = wait_for(lambda: not process_exists(state.pid), timeout_seconds=10, poll_seconds=0.5)
    return {
        "stopped": bool(stopped),
        "was_running": True,
        "pid": state.pid,
        "detail": "frontend stopped" if stopped else "frontend stop failed",
    }


def _stop_backend(config: AtlasWakeConfig) -> dict[str, Any]:
    state = backend_process_state(config)
    if not state.running or not state.pid:
        return {"stopped": True, "was_running": False, "detail": "backend not running"}
    if str(getattr(config, "launch_profile", "paper_compare")).strip().lower() == "guarded_demo":
        logging.info("Stopping guarded demo backend pid=%s", state.pid)
        stop_process(state.pid, force=False)
        stopped, _ = wait_for(lambda: not process_exists(state.pid), timeout_seconds=max(20, config.poll_seconds + 20), poll_seconds=2.0)
        forced = False
        if not stopped:
            logging.warning("Guarded demo backend did not exit after graceful wait; forcing pid=%s", state.pid)
            stop_process(state.pid, force=True)
            forced = True
            stopped, _ = wait_for(lambda: not process_exists(state.pid), timeout_seconds=10, poll_seconds=0.5)
        return {
            "stopped": bool(stopped),
            "was_running": True,
            "pid": state.pid,
            "forced": forced,
            "detail": "guarded demo backend stopped cleanly" if stopped and not forced else ("guarded demo backend forced to stop" if stopped else "guarded demo backend stop failed"),
        }
    stop_file = write_stop_request(config.comparison_root, config.comparison_run_id, "Atlas sleep requested")
    logging.info("Requested graceful backend stop via %s", stop_file)
    stopped, _ = wait_for(lambda: not process_exists(state.pid), timeout_seconds=max(20, config.poll_seconds + 20), poll_seconds=2.0)
    forced = False
    if not stopped:
        logging.warning("Backend did not exit after graceful wait; forcing pid=%s", state.pid)
        stop_process(state.pid, force=True)
        forced = True
        stopped, _ = wait_for(lambda: not process_exists(state.pid), timeout_seconds=10, poll_seconds=0.5)
    return {
        "stopped": bool(stopped),
        "was_running": True,
        "pid": state.pid,
        "forced": forced,
        "stop_request_path": str(stop_file),
        "detail": "backend stopped cleanly" if stopped and not forced else ("backend forced to stop" if stopped else "backend stop failed"),
    }


def _stop_mt5(config: AtlasWakeConfig) -> dict[str, Any]:
    state = mt5_process_state(config)
    if not state.running or not state.pid:
        return {"stopped": True, "was_running": False, "detail": "mt5 not running"}
    logging.info("Stopping MT5 pid=%s", state.pid)
    stop_process(state.pid, force=False)
    stopped, _ = wait_for(lambda: not process_exists(state.pid), timeout_seconds=15, poll_seconds=1.0)
    if not stopped:
        stop_process(state.pid, force=True)
        stopped, _ = wait_for(lambda: not process_exists(state.pid), timeout_seconds=10, poll_seconds=0.5)
    return {
        "stopped": bool(stopped),
        "was_running": True,
        "pid": state.pid,
        "detail": "mt5 stopped" if stopped else "mt5 stop failed",
    }


def verify_data_integrity(config: AtlasWakeConfig) -> dict[str, Any]:
    if str(getattr(config, "launch_profile", "paper_compare")).strip().lower() == "guarded_demo":
        return {
            "intact": True,
            "files": {},
            "last_saved_timestamps": {},
            "detail": "comparison data integrity not applicable for guarded demo launch profile",
        }
    run_root = Path(config.comparison_root) / config.comparison_run_id
    files = {
        "comparison_summary": json_file_status(run_root / "comparison_summary.json"),
        "comparison_report": json_file_status(run_root / "comparison_report.json"),
        "comparison_tracker": json_file_status(run_root / "comparison_tracker.json"),
    }
    intact = all(bool(item["readable"]) for item in files.values())
    last_saved_timestamps = {name: data["timestamp"] for name, data in files.items()}
    return {
        "intact": intact,
        "files": files,
        "last_saved_timestamps": last_saved_timestamps,
    }


def run_full_shutdown(
    config: AtlasWakeConfig | None = None,
    *,
    keep_listener: bool = False,
    keep_mt5: bool = False,
) -> dict[str, Any]:
    config = config or load_atlas_wake_config()
    setup_logging(config)
    logging.info(
        "Atlas sleep shutdown requested for run_id=%s keep_listener=%s keep_mt5=%s",
        config.comparison_run_id,
        keep_listener,
        keep_mt5,
    )
    frontend = _stop_frontend(config)
    backend = _stop_backend(config)
    mt5 = (
        {
            "stopped": True,
            "was_running": bool(mt5_process_state(config).running),
            "detail": "mt5 preserved",
        }
        if keep_mt5
        else _stop_mt5(config)
    )
    data = verify_data_integrity(config)
    stopped_all = frontend["stopped"] and backend["stopped"] and mt5["stopped"]
    overall = "FULL SYSTEM STOPPED CLEANLY"
    if not stopped_all or not data["intact"]:
        overall = "PARTIAL STOP" if any([frontend["stopped"], backend["stopped"], mt5["stopped"], data["intact"]]) else "STOP FAILED"
    return {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "frontend_stopped": frontend,
        "backend_stopped": backend,
        "mt5_hfm_stopped": mt5,
        "listener_preserved": bool(keep_listener),
        "experiment_data_intact": data,
        "overall_result": overall,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stop the full local Atlas paper stack.")
    parser.add_argument("--json", action="store_true", help="Print the final structured summary as JSON.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_full_shutdown()
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(json.dumps({"overall_result": summary["overall_result"]}, indent=2))
    return 0 if summary["overall_result"] == "FULL SYSTEM STOPPED CLEANLY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
