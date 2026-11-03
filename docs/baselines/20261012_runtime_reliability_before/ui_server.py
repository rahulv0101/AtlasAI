from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import subprocess
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.serving import WSGIRequestHandler, make_server

from atlas_ai.config.settings import load_settings
from atlas_ai.memory.knowledge_store import KnowledgeStore
from atlas_ai.memory.jsonl_tail import tail_text_lines
from atlas_ai.execution.execution_gate import ExecutionGate
from atlas_ai.execution.mt5_broker_adapter import MT5BrokerAdapter
from atlas_ai.execution.trade_executor import TradeExecutor
from atlas_ai.analysis.decision_analysis import build_structured_decision_analysis
from atlas_ai.analysis.signal_correlation import build_signal_correlation_report_from_store
from atlas_ai.analysis.setup_hunter import build_setup_hunter
from atlas_ai.learning.experiment_runner import build_self_improvement_validation_report
from atlas_ai.learning.fast_path_audit import build_fast_path_promotion_audit_report
from atlas_ai.learning.missed_entry_audit import build_consensus_selectivity_operator_review_packet
from atlas_ai.learning.operator_teaching import (
    build_current_decision_teaching_prompt,
    build_operator_teaching_digest,
    build_structured_lesson_from_status,
)
from atlas_ai.operator_assistant import load_comparison_meta_from_run_root
from atlas_ai.runtime_heartbeat import get_backend_health
from atlas_ai.system_health import build_freshness_runtime, build_system_health
from atlas_ai.strategies.dna import StrategyChromosome
from atlas_ai.strategies.evolution_reporter import EvolutionReporter
from atlas_ai.strategies.strategy_explainer import StrategyExplainer
from atlas_ai.brain.ai_decision_journal import AIDecisionJournal
from atlas_ai.brain.ai_decision_reviewer import AIDecisionReviewer
from atlas_ai.brain.ai_promotion_intelligence import AIPromotionIntelligence
from atlas_ai.brain.ai_research_queue import AIResearchQueue
from atlas_ai.brain.ai_demo_review_packet import AIDemoReviewPacket
from atlas_ai.brain.ai_demo_activation_proposal import AIDemoActivationProposal
from atlas_ai.brain.ai_demo_activation_preflight import AIDemoActivationPreflight
from atlas_ai.brain.ai_demo_activation import AIDemoActivator
from atlas_ai.brain.adaptation import build_adaptation_summary
from atlas_ai.brain.cognitive_core import build_cognitive_core
from atlas_ai.brain.why_engine import build_why_trace
from atlas_ai.self_improvement import (
    approve_proposal_for_rollout,
    build_self_improvement_runtime_guard,
    build_self_improvement_summary,
    close_proposal_without_rollout,
    ensure_self_improvement_knowledge,
    load_self_improvement_bundle,
    promote_proposal_to_active_rules,
    record_proposal_validation,
    reconcile_proposal_queue_with_rollouts,
    rollback_rollout_promotion,
    transition_rollout_status,
)
from atlas_ai.ui.ask_atlas import AskAtlasService
from atlas_ai.voice.startup import announce_ui_ready_once
from atlas_ai.vision.chart_vision import (
    analyze_chart_visual_context,
    capture_desktop_screenshot,
    load_latest_visual_read,
    save_visual_read,
)


_LOGGER = logging.getLogger(__name__)
_MACRO_REFRESH_LOCK = threading.Lock()
_MACRO_REFRESH_IN_FLIGHT = False
_MACRO_REFRESH_LAST_STARTED = 0.0


class AtlasUIRequestHandler(WSGIRequestHandler):
    """Short-lived local HTTP connections for the desktop/webview wrapper."""

    protocol_version = "HTTP/1.0"

    def setup(self) -> None:
        super().setup()
        with contextlib.suppress(Exception):
            self.connection.settimeout(8.0)


def _hidden_creationflags() -> int:
    if os.name != "nt":
        return 0
    return int(getattr(subprocess, "CREATE_NO_WINDOW", 0))


def _load_json_file(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_automation_payload(store: KnowledgeStore) -> dict[str, Any]:
    raw = _load_json_file(Path(store.base) / "automation_state.json")
    payload = raw.get("payload", {}) if isinstance(raw, dict) else {}
    return dict(payload or {}) if isinstance(payload, dict) else {}


def _automation_truth_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    policy_id = str(payload.get("active_automation_policy_id", "") or "")
    policy = dict(dict(payload.get("automation_profiles", {}) or {}).get(policy_id, {}) or {})
    automation_paused = bool(payload.get("automation_paused", False))
    last_event = dict(payload.get("last_automation_event", {}) or {})
    last_block_reason = str(payload.get("last_automation_block_reason", "") or "")
    try:
        last_event_age_seconds = max(0.0, time.time() - float(last_event.get("ts", 0.0) or 0.0))
    except Exception:
        last_event_age_seconds = 0.0
    if last_block_reason and not automation_paused and last_event_age_seconds > 3600.0:
        last_block_reason = ""
    return {
        "automation_enabled": bool(policy.get("enabled", False)) and not automation_paused,
        "automation_paused": automation_paused,
        "automation_pause_reason": str(payload.get("automation_pause_reason", "") or ""),
        "automation_explicit_resume_active": bool(payload.get("automation_explicit_resume_active", False)),
        "last_auto_execution_result": last_event,
        "last_auto_execution_block_reason": last_block_reason,
        "last_auto_execution_block_stale": bool(
            str(payload.get("last_automation_block_reason", "") or "")
            and not automation_paused
            and last_event_age_seconds > 3600.0
        ),
    }


def _compact_event_for_ui(event: dict[str, Any]) -> dict[str, Any]:
    row = dict(event or {})
    meta = dict(row.get("meta", {}) or {})
    compact_meta: dict[str, Any] = {}
    for key in ("decision_explanation", "system_health", "context_snapshot", "shadow_advice"):
        value = meta.get(key)
        if isinstance(value, dict):
            compact_meta[key] = value
    message = str(row.get("message", "") or "")
    row["message"] = message[:600]
    row["meta"] = compact_meta
    return row


def _load_evolution_scheduler_state(store: KnowledgeStore) -> dict[str, Any]:
    """Load evolution scheduler state from persisted file."""
    state_file = Path(store.base) / "evolution_scheduler_state.json"
    if not state_file.exists():
        return {
            "evolution_enabled": False,
            "scheduler_status": "not_configured",
            "current_generation": 0,
            "last_evolution_run_at": "",
            "next_evolution_run_at": "",
            "best_fitness": 0.0,
            "best_uid": "",
            "active_demo_strategy_count": 0,
            "active_shadow_count": 0,
            "last_error": "",
            "total_runs": 0,
        }
    try:
        raw = json.loads(state_file.read_text(encoding="utf-8"))
        return {
            "evolution_enabled": True,
            "scheduler_status": raw.get("scheduler_status", "unknown"),
            "current_generation": raw.get("current_generation", 0),
            "last_evolution_run_at": raw.get("last_evolution_run_at", ""),
            "next_evolution_run_at": raw.get("next_evolution_run_at", ""),
            "best_fitness": raw.get("best_fitness", 0.0),
            "best_uid": raw.get("best_uid", ""),
            "active_demo_strategy_count": raw.get("active_demo_count", 0),
            "active_shadow_count": raw.get("active_shadow_count", 0),
            "last_error": raw.get("last_error", ""),
            "total_runs": raw.get("total_runs", 0),
            "dry_run": raw.get("dry_run", False),
        }
    except Exception:
        return {
            "evolution_enabled": True,
            "scheduler_status": "error_loading_state",
            "current_generation": 0,
            "last_evolution_run_at": "",
            "next_evolution_run_at": "",
            "best_fitness": 0.0,
            "best_uid": "",
            "active_demo_strategy_count": 0,
            "active_shadow_count": 0,
            "last_error": "Failed to load scheduler state",
            "total_runs": 0,
        }


def _canonical_freshness_runtime(status_payload: dict, settings) -> dict:
    freshness = build_freshness_runtime(status_payload, settings)
    return dict(freshness or {})


def _process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        if os.name == "nt":
            output = subprocess.check_output(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                text=True,
                stderr=subprocess.DEVNULL,
                creationflags=_hidden_creationflags(),
                timeout=2.0,
            ).strip()
            lowered = output.lower()
            return str(pid) in output and ("python.exe" in lowered or "pythonw.exe" in lowered)
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _discover_ui_wrapper_pid() -> int:
    if os.name != "nt":
        return 0
    ps_cmd = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.Name -in @('python.exe','pythonw.exe') -and $_.CommandLine -match 'atlas_ai\\.main\\s+ui(?:\\s|$)' } | "
        "Select-Object ProcessId,CommandLine | ConvertTo-Json -Depth 3"
    )
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", ps_cmd],
            text=True,
            stderr=subprocess.DEVNULL,
            creationflags=_hidden_creationflags(),
            timeout=2.0,
        ).strip()
    except Exception:
        return 0
    if not out:
        return 0
    try:
        data = json.loads(out)
    except Exception:
        return 0
    items = data if isinstance(data, list) else [data]
    best_pid = 0
    best_rank: tuple[int, int] | None = None
    for item in items:
        if not isinstance(item, dict):
            continue
        cmdline = str(item.get("CommandLine", "") or "").strip().lower()
        if "atlas_ai.main ui-server" in cmdline or "atlas_ai.main ui-browser" in cmdline:
            continue
        try:
            pid = int(item.get("ProcessId", 0) or 0)
        except Exception:
            pid = 0
        if pid <= 0:
            continue
        is_base_python = 1 if ".venv\\scripts\\python.exe" not in cmdline else 0
        rank = (is_base_python, pid)
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_pid = pid
    return best_pid


def _parse_iso_timestamp(value: object) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        return None


def _latest_successful_desktop_start(store: KnowledgeStore) -> datetime | None:
    path = Path(store.base) / "logs" / "ui_startup_diagnostics.log"
    if not path.exists():
        return None
    latest: datetime | None = None
    try:
        lines = tail_text_lines(path, 80)
    except Exception:
        return None
    for line in lines:
        if "ok=True" not in line or "stage=started" not in line:
            continue
        timestamp = _parse_iso_timestamp(line.split("|", 1)[0].strip())
        if timestamp is not None and (latest is None or timestamp > latest):
            latest = timestamp
    return latest


def _reconcile_desktop_window_state(payload: dict, store: KnowledgeStore | None = None) -> dict:
    desktop_window = dict(payload or {})
    if not desktop_window:
        return desktop_window
    try:
        current_wrapper_pid = int(desktop_window.get("wrapper_pid", 0) or 0)
    except Exception:
        current_wrapper_pid = 0
    wrapper_running = _process_exists(current_wrapper_pid)
    if wrapper_running and store is not None:
        payload_ts = _parse_iso_timestamp(desktop_window.get("updated_at") or desktop_window.get("ts"))
        latest_started = _latest_successful_desktop_start(store)
        if latest_started is not None and (payload_ts is None or latest_started >= payload_ts):
            desktop_window.update(
                {
                    "ok": True,
                    "ready": True,
                    "desktop_window_ready": True,
                    "desktop_shell_healthy": True,
                    "desktop_ui_available": True,
                    "frontend_mode": "desktop",
                    "failure_reason": "",
                    "error": "",
                    "exception_type": "",
                    "stage": "started",
                    "updated_at": latest_started.isoformat(),
                    "ts": latest_started.isoformat(),
                }
            )
            desktop_window["classification"] = {"category": "ok", "code": "", "operator_action": "", "operator_cause": ""}
            desktop_window["degraded_since"] = ""
            evidence = dict(desktop_window.get("failure_evidence", {}) or {})
            evidence["desktop_started_reconciled_from_diagnostics"] = latest_started.isoformat()
            desktop_window["failure_evidence"] = evidence
            return desktop_window
    if wrapper_running:
        return desktop_window
    discovered_pid = _discover_ui_wrapper_pid()
    if discovered_pid > 0:
        desktop_window["wrapper_pid"] = discovered_pid
        evidence = dict(desktop_window.get("failure_evidence", {}) or {})
        evidence["wrapper_pid_reconciled_from_process_scan"] = discovered_pid
        desktop_window["failure_evidence"] = evidence
        return desktop_window
    if current_wrapper_pid > 0:
        desktop_window["wrapper_pid"] = 0
        if bool(desktop_window.get("frontend_mode", "") == "desktop"):
            desktop_window["ready"] = False
            desktop_window["desktop_window_ready"] = False
        evidence = dict(desktop_window.get("failure_evidence", {}) or {})
        evidence["stale_wrapper_pid_cleared"] = current_wrapper_pid
        desktop_window["failure_evidence"] = evidence
    return desktop_window


def _resolve_comparison_run_root(store: KnowledgeStore, comparison_root: str | None = None, comparison_run_id: str | None = None) -> Path | None:
    if not comparison_root and not comparison_run_id:
        return None
    base_root = Path(comparison_root) if comparison_root else (Path(store.base).parent / "storage_compare_runs")
    if comparison_run_id:
        root = base_root / str(comparison_run_id)
        if (root / "comparison_summary.json").exists():
            return root
        return None
    summaries = list(base_root.glob("*/comparison_summary.json"))
    if not summaries:
        return None
    latest = max(summaries, key=lambda p: p.stat().st_mtime)
    return latest.parent


def _safe_parse_event_time(value: object) -> float:
    text = str(value or "").strip()
    if not text:
        return 0.0
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0.0


def _latest_training_summary(source: "UIDashboardSource") -> dict:
    events = list(source.load_events(limit=240))
    training_events = [
        event
        for event in events
        if str(event.get("type", "")).lower() == "training"
        and "historical training completed" in str(event.get("message", "")).lower()
    ]
    if not training_events:
        return {}
    latest = max(training_events, key=lambda event: _safe_parse_event_time(event.get("time")))
    meta = dict(latest.get("meta", {}) or {})
    return {
        "completed_at": str(latest.get("time", "") or ""),
        "bars": int(meta.get("bars", 0) or 0),
        "return_pct": round(float(meta.get("total_return", 0.0) or 0.0) * 100.0, 2),
        "trades": int(meta.get("trades", 0) or 0),
        "wins": int(meta.get("wins", 0) or 0),
        "losses": int(meta.get("losses", 0) or 0),
        "best_strategy": dict(meta.get("best_strategy", {}) or {}),
        "weak_point_summary": dict(meta.get("weak_point_summary", {}) or {}),
        "setup_family_summary": dict(meta.get("setup_family_summary", {}) or {}),
        "labels_summary": dict(meta.get("labels_summary", {}) or {}),
    }


def _loss_reason_summary(outcomes: list[dict]) -> dict:
    rows = [dict(item or {}) for item in outcomes if str(dict(item or {}).get("outcome", "")).lower() == "loss"]
    total = len(rows)
    if total <= 0:
        return {"total_losses": 0, "top_reason": "none", "reasons": []}

    counts: dict[str, int] = defaultdict(int)
    family_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        reason = str(row.get("loss_classification", "") or dict(row.get("post_trade_review", {}) or {}).get("classification", "") or "unknown").strip().lower()
        if not reason:
            reason = "unknown"
        family = str(row.get("setup_family", "unknown") or "unknown").strip().lower()
        counts[reason] += 1
        family_counts[family] += 1

    reasons = [
        {
            "reason": reason,
            "count": count,
            "share": round(count / total, 4),
        }
        for reason, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    family_breakdown = [
        {
            "family": family,
            "count": count,
            "share": round(count / total, 4),
        }
        for family, count in sorted(family_counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    return {
        "total_losses": total,
        "top_reason": str(reasons[0]["reason"]) if reasons else "none",
        "reasons": reasons,
        "family_breakdown": family_breakdown,
    }


def _last_confirmed_trade_outcome(outcomes: list[dict]) -> dict:
    rows = [
        dict(item or {})
        for item in list(outcomes or [])
        if str(dict(item or {}).get("outcome", "")).strip().lower() in {"win", "loss", "draw"}
    ]
    if not rows:
        return {
            "available": False,
            "outcome": "unknown",
            "lesson": "No broker-confirmed closed trade outcome is available yet.",
        }

    last = rows[-1]
    outcome = str(last.get("outcome", "unknown") or "unknown").lower()
    pnl = float(last.get("net_pnl_zar", 0.0) or 0.0)
    strategy = str(last.get("strategy", "unknown") or "unknown")
    side = str(last.get("side", "unknown") or "unknown")
    confidence = float(last.get("confidence", 0.0) or 0.0)
    trend_strength = float(last.get("trend_strength", 0.0) or 0.0)
    htf_aligned = bool(last.get("htf_aligned", False))
    lesson = (
        "Last confirmed trade was a loss; similar setups should require stronger trend, confidence, and entry quality before approval."
        if outcome == "loss"
        else (
            "Last confirmed trade was a win; keep it as supportive evidence only until the sample size is larger."
            if outcome == "win"
            else "Last confirmed trade was flat; treat it as neutral evidence."
        )
    )
    return {
        "available": True,
        "ticket": str(last.get("ticket", "") or ""),
        "decision_id": str(last.get("decision_id", "") or ""),
        "time": str(last.get("time", "") or ""),
        "strategy": strategy,
        "side": side,
        "session": str(last.get("session", "unknown") or "unknown"),
        "regime": str(last.get("regime", "unknown") or "unknown"),
        "higher_tf_bias": str(last.get("higher_tf_bias", "unknown") or "unknown"),
        "htf_aligned": htf_aligned,
        "confidence": round(confidence, 4),
        "trend_strength": round(trend_strength, 4),
        "net_pnl_zar": round(pnl, 2),
        "outcome": outcome,
        "lesson": lesson,
    }


def _build_learning_summary(source: "UIDashboardSource") -> dict:
    state = dict(source.load_learning_state().get("state", {}) or {})
    best_strategy = dict(state.get("best_evolved_strategy", {}) or {})
    promoted_pool = dict(state.get("evolved_strategy_pool", {}) or {})
    loss_memory = dict(state.get("loss_memory", {}) or {})
    recent_outcomes = list(loss_memory.get("recent_outcomes", []) or [])
    recent_slice = recent_outcomes[-80:]
    sample_size = len(recent_slice)
    wins = sum(1 for item in recent_slice if str(dict(item).get("outcome", "")).lower() == "win")
    losses = sum(1 for item in recent_slice if str(dict(item).get("outcome", "")).lower() == "loss")
    draws = sum(1 for item in recent_slice if str(dict(item).get("outcome", "")).lower() == "draw")
    top_family = "none"
    families = list(dict(promoted_pool.get("setup_family_summary", {}) or {}).get("families", []) or [])
    if not families:
        latest_training = _latest_training_summary(source)
        families = list(dict(latest_training.get("setup_family_summary", {}) or {}).get("families", []) or [])
        top_family = str(dict(latest_training.get("setup_family_summary", {}) or {}).get("best_family", "none") or "none")
    else:
        top_family = str(dict(promoted_pool.get("setup_family_summary", {}) or {}).get("best_family", "none") or "none")

    latest_training = _latest_training_summary(source)
    weak_point_summary = dict(latest_training.get("weak_point_summary", {}) or {})
    setup_family_summary = dict(latest_training.get("setup_family_summary", {}) or {})
    if not top_family:
        top_family = str(setup_family_summary.get("best_family", "none") or "none")
    win_rate = round(wins / max(1, wins + losses), 4) if sample_size else 0.0

    return {
        "champion_signature": str(best_strategy.get("signature", "") or promoted_pool.get("champion_signature", "") or ""),
        "champion_source": str(best_strategy.get("source", "") or ""),
        "sample_size": sample_size,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "recent_win_rate": win_rate,
        "top_setup_family": top_family,
        "setup_family_summary": setup_family_summary,
        "weak_point_summary": weak_point_summary,
        "loss_reason_summary": _loss_reason_summary(recent_slice),
        "last_confirmed_trade": _last_confirmed_trade_outcome(recent_outcomes),
        "latest_training": latest_training,
    }


def _load_near_setup_learning_summary(store: KnowledgeStore, *, limit: int = 10) -> dict[str, Any]:
    payload = _load_json_file(Path(store.base) / "monitoring" / "atlas_near_setup_learning.json")
    top_patterns = []
    for item in list(payload.get("top_patterns", []) or [])[: max(1, limit)]:
        row = dict(item or {})
        top_patterns.append(
            {
                "fingerprint": str(row.get("fingerprint", "") or ""),
                "count": int(row.get("count", 0) or 0),
                "last_seen": str(row.get("last_seen", "") or ""),
                "teaching_focus": [str(x) for x in list(row.get("teaching_focus", []) or [])[:4]],
            }
        )
    return {
        "mode": str(payload.get("mode", "observability_only") or "observability_only"),
        "updated_at": str(payload.get("updated_at", "") or ""),
        "total_samples": int(payload.get("total_samples", 0) or 0),
        "top_patterns": top_patterns,
        "policy_note": str(
            payload.get("policy_note")
            or "Near-setup history is observability only; it does not change strategy, risk, guard, or execution policy."
        ),
    }


def _build_blocker_summary(status_payload: dict[str, Any], near_setup_history: dict[str, Any] | None = None) -> dict[str, Any]:
    blocker = dict(status_payload.get("blocker_classification", {}) or {})
    analysis = dict(status_payload.get("decision_analysis", {}) or {})
    setup_radar = dict(analysis.get("setup_radar", {}) or {})
    top_watch = dict(setup_radar.get("top_watch", {}) or setup_radar.get("top_candidate", {}) or {})
    chosen = dict(status_payload.get("chosen", {}) or {})
    thresholds = dict(status_payload.get("thresholds", {}) or {})
    confidence = float(chosen.get("confidence", status_payload.get("confidence", 0.0)) or 0.0)
    threshold = float(thresholds.get("confidence", status_payload.get("min_confidence_threshold", 0.0)) or 0.0)
    hard_blockers = list(analysis.get("hard_blockers", []) or analysis.get("blockers", []) or [])
    top_blockers = list(analysis.get("top_blockers", []) or [])
    near_patterns = list(dict(near_setup_history or {}).get("top_patterns", []) or [])
    return {
        "action": str(status_payload.get("action", "") or ""),
        "blocker_class": str(blocker.get("class", status_payload.get("blocker_class", "unknown")) or "unknown"),
        "primary": str(
            blocker.get("primary")
            or status_payload.get("primary_blocker")
            or status_payload.get("message")
            or "No primary blocker reported."
        ),
        "recoverability": str(blocker.get("recoverability", status_payload.get("blocker_recoverability", "")) or ""),
        "confidence": round(confidence, 4),
        "threshold": round(threshold, 4),
        "confidence_gap": round(confidence - threshold, 4),
        "hard_blockers": [str(x) for x in hard_blockers[:8]],
        "top_blockers": top_blockers[:5],
        "setup_radar": {
            "state": str(setup_radar.get("timing_state", setup_radar.get("state", "wait")) or "wait"),
            "top_watch": top_watch,
            "waiting_for": list(top_watch.get("missing_to_entry", setup_radar.get("waiting_for", [])) or [])[:6],
            "conditions_slowing_setups": list(setup_radar.get("conditions_slowing_setups", []) or [])[:8],
            "summary": str(setup_radar.get("operator_summary", setup_radar.get("summary", "")) or ""),
        },
        "near_setup_learning": {
            "total_samples": int(dict(near_setup_history or {}).get("total_samples", 0) or 0),
            "top_patterns": near_patterns[:5],
        },
        "policy_note": "Blocker summary is observability only; it does not change strategy, risk, guard, or execution policy.",
    }


def _build_compact_status_payload(
    status_payload: dict[str, Any],
    runtime_flags: dict[str, Any],
    store: KnowledgeStore,
) -> dict[str, Any]:
    status_payload = dict(status_payload or {})
    runtime_flags = dict(runtime_flags or {})
    settings = load_settings()
    desktop_window = _reconcile_desktop_window_state(dict(runtime_flags.get("ui_desktop_window", {}) or {}), store)
    automation_payload = _load_automation_payload(store)
    structured_blocker = _build_structured_blocker(status_payload, automation_payload)
    reconciled_runtime_flags = dict(runtime_flags)
    reconciled_runtime_flags["ui_desktop_window"] = desktop_window
    runtime_execution = dict(runtime_flags.get("execution", {}) or {})
    session_control = dict(automation_payload.get("session_control", {}) or runtime_execution.get("session_control", {}) or {})
    automation_truth = _automation_truth_from_payload(automation_payload)
    stored_system_health = dict(status_payload.get("system_health", {}) or {})
    computed_system_health = build_system_health(status_payload, settings, runtime_flags=runtime_flags)
    system_health = dict(computed_system_health)
    for key, value in stored_system_health.items():
        if value not in ("", None, {}, []):
            system_health[key] = value
    mode_truth = dict(system_health.get("mode_truth", {}) or {})
    startup_verification = dict(runtime_flags.get("startup_verification", {}) or {})
    effective_mode = str(system_health.get("effective_mode", status_payload.get("effective_mode", "")) or "")
    effective_mode_label = str(system_health.get("effective_mode_label", status_payload.get("effective_mode_label", "")) or "")
    gate_execution_environment = str(
        status_payload.get("gate_execution_environment", "")
        or mode_truth.get("gate_execution_environment", "")
        or startup_verification.get("gate_execution_environment", "")
        or ""
    )
    if gate_execution_environment:
        mode_truth["gate_execution_environment"] = gate_execution_environment
    execute_trades_raw = status_payload.get("execute_trades", runtime_flags.get("execute_trades", startup_verification.get("execute_trades", None)))
    execute_trades = bool(execute_trades_raw) if execute_trades_raw is not None else bool(getattr(settings, "execute_trades", False))
    blocker_summary = dict(status_payload.get("blocker_classification", {}) or {})
    decision_analysis = dict(status_payload.get("decision_analysis", {}) or {})
    decision_trace = dict(status_payload.get("decision_trace", {}) or {})
    trace_signal = dict(decision_trace.get("signal", {}) or {})
    setup_quality = dict(decision_analysis.get("setup_quality_observation", {}) or {})
    chosen = dict(status_payload.get("chosen", {}) or {})
    risk = dict(status_payload.get("risk", {}) or {})
    market = dict(status_payload.get("market", {}) or {})
    status_visual = dict(status_payload.get("visual_chart_read", {}) or {})
    latest_visual = load_latest_visual_read(Path(store.base) / "chart_vision", max_age_seconds=900)
    visual_chart_read = latest_visual if bool(latest_visual.get("available", False)) and not bool(latest_visual.get("stale", False)) else status_visual
    status_for_readiness = dict(status_payload)
    status_for_readiness["visual_chart_read"] = dict(visual_chart_read or {})
    operator_readiness = _build_operator_readiness(
        status_for_readiness,
        reconciled_runtime_flags,
        automation_payload,
        desktop_window,
    )
    execution_readiness = _build_execution_readiness(
        status_for_readiness,
        reconciled_runtime_flags,
        automation_payload,
        operator_readiness,
    )
    canonical_session_state = str(session_control.get("current_session", "") or "")
    raw_session_state = str(system_health.get("session_state", "") or "")
    warnings = [str(item) for item in list(system_health.get("warnings", []) or [])]
    if bool(automation_payload.get("mismatch_detected", False)) and "broker_reconciliation_mismatch" not in warnings:
        warnings.append("broker_reconciliation_mismatch")
    compact_market = {
        "recent_close": list(market.get("recent_close", []) or [])[-80:],
    }
    loss_pattern_block = _build_loss_pattern_block_status(status_payload)
    recovery_status = _build_recovery_status(status_payload, structured_blocker, loss_pattern_block)
    setup_hunter = dict(status_payload.get("setup_hunter", {}) or {})
    if not setup_hunter:
        setup_hunter = build_setup_hunter(
            status=status_payload,
            decision_analysis=decision_analysis,
            runtime=runtime_flags,
        )
    learning_critic = _build_learning_critic(status_payload, store)
    loss_learning_review = _build_loss_learning_review(status_payload, store)
    learning_lanes = _build_learning_lanes_status(status_payload, runtime_flags, store)
    cognitive_state = _build_cognitive_state_payload(status_payload, runtime_flags, store)
    trader_cognition = dict(status_payload.get("trader_cognition", {}) or {})
    safety_lock_review = _build_safety_lock_review(status_payload)
    volatility_protection_review = _build_volatility_protection_review(status_payload)
    visual_evidence_review = _build_visual_evidence_review(status_for_readiness, runtime_flags, store)
    setup_execution_handoff = _build_setup_execution_handoff_review(
        status_for_readiness,
        setup_hunter,
        execution_readiness,
    )
    signal_selection_review = _build_signal_selection_review(
        status_for_readiness,
        setup_hunter,
        execution_readiness,
    )
    try:
        backend_health = get_backend_health(store)
    except Exception:
        backend_health = {}
    backend_health_status = str(dict(backend_health).get("status", "") or "").strip().lower()
    backend_warning = ""
    if backend_health_status == "unknown":
        backend_warning = "Backend heartbeat is unknown; dashboard status is last-known state."
    elif backend_health_status == "stale":
        backend_warning = (
            f"Backend heartbeat is stale; last report was "
            f"{int(dict(backend_health).get('seconds_since_heartbeat', 0) or 0)} seconds ago."
        )
    return {
        "compact": True,
        "updated_at": str(status_payload.get("updated_at", "") or ""),
        "action": str(status_payload.get("action", "") or ""),
        "message": str(status_payload.get("message", "") or ""),
        "execute_trades": execute_trades,
        "effective_mode": effective_mode,
        "effective_mode_label": effective_mode_label,
        "gate_execution_environment": gate_execution_environment,
        "symbol": str(status_payload.get("symbol", "XAUUSD") or "XAUUSD"),
        "timeframe": str(status_payload.get("timeframe", "") or ""),
        "gemini_enabled": bool(status_payload.get("gemini_enabled", False)),
        "gemini_note": str(status_payload.get("gemini_note", "") or ""),
        "blocker_class": str(status_payload.get("blocker_class", blocker_summary.get("class", "")) or ""),
        "blocker_classification": blocker_summary,
        "structured_blocker": structured_blocker,
        "primary_blocker": str(status_payload.get("primary_blocker", blocker_summary.get("primary", "")) or ""),
        "loss_learning": dict(status_payload.get("loss_learning", {}) or {}),
        "loss_pattern_block": loss_pattern_block,
        "recovery_status": recovery_status,
        "safety_lock_review": safety_lock_review,
        "volatility_protection_review": volatility_protection_review,
        "visual_evidence_review": visual_evidence_review,
        "setup_hunter": setup_hunter,
        "setup_execution_handoff": setup_execution_handoff,
        "signal_selection_review": signal_selection_review,
        "learning_critic": learning_critic,
        "loss_learning_review": loss_learning_review,
        "learning_lanes": learning_lanes,
        "cognitive_state": {
            "schema": cognitive_state.get("schema"),
            "state": cognitive_state.get("state"),
            "headline": cognitive_state.get("headline"),
            "mind_model": cognitive_state.get("mind_model"),
            "coordinator": cognitive_state.get("coordinator"),
            "policy_note": cognitive_state.get("policy_note"),
        },
        "trader_cognition": {
            "schema": trader_cognition.get("schema"),
            "mode": trader_cognition.get("mode"),
            "observation": trader_cognition.get("observation"),
            "thesis": trader_cognition.get("thesis"),
            "counter_thesis": trader_cognition.get("counter_thesis"),
            "discipline_check": dict(trader_cognition.get("discipline_check", {}) or {}),
            "service_intent": dict(trader_cognition.get("service_intent", {}) or {}),
            "decision": trader_cognition.get("decision"),
            "confidence": trader_cognition.get("confidence"),
            "threshold": trader_cognition.get("threshold"),
            "required_next_evidence": list(trader_cognition.get("required_next_evidence", []) or [])[:6],
            "autonomy_boundary": dict(trader_cognition.get("autonomy_boundary", {}) or {}),
        },
        "loss_to_win_playbook": dict(status_payload.get("loss_to_win_playbook", {}) or {}),
        "winner_playbook": dict(status_payload.get("winner_playbook", {}) or {}),
        "learning_loop": _build_learning_loop_status(store),
        "visual_chart_read": dict(visual_chart_read or {}),
        "chosen": chosen,
        "best_actionable_candidate": dict(status_payload.get("best_actionable_candidate", {}) or {}),
        "min_confidence_threshold": float(status_payload.get("min_confidence_threshold", dict(status_payload.get("thresholds", {}) or {}).get("confidence", 0.0)) or 0.0),
        "thresholds": dict(status_payload.get("thresholds", {}) or {}),
        "risk": {
            "equity": risk.get("equity"),
            "balance": risk.get("balance"),
            "daily_loss_pct": risk.get("daily_loss_pct"),
            "weekly_loss_pct": risk.get("weekly_loss_pct"),
            "open_positions": risk.get("open_positions"),
        },
        "market": compact_market,
        "signals": list(status_payload.get("signals", []) or [])[:12],
        "signal_summary": dict(status_payload.get("signal_summary", {}) or {}),
        "same_side_support": dict(status_payload.get("same_side_support", {}) or {}),
        "decision_analysis": {
            "account_profile": dict(decision_analysis.get("account_profile", {}) or {}),
            "confidence_score": decision_analysis.get("confidence_score"),
            "confidence_level": decision_analysis.get("confidence_level"),
            "confluence": dict(decision_analysis.get("confluence", {}) or {}),
            "blockers": list(decision_analysis.get("blockers", []) or [])[:8],
            "hard_blockers": list(decision_analysis.get("hard_blockers", []) or decision_analysis.get("blockers", []) or [])[:8],
            "top_blockers": list(decision_analysis.get("top_blockers", []) or [])[:5],
            "setup_quality_observation": setup_quality,
            "setup_radar": dict(decision_analysis.get("setup_radar", {}) or {}),
            "timing_engine": dict(decision_analysis.get("timing_engine", {}) or {}),
            "bias_breakdown": dict(decision_analysis.get("bias_breakdown", {}) or {}),
            "signal_bias": decision_analysis.get("signal_bias"),
            "chart_structure_bias": decision_analysis.get("chart_structure_bias"),
            "higher_timeframe_bias": decision_analysis.get("higher_timeframe_bias"),
            "final_trade_bias": decision_analysis.get("final_trade_bias"),
            "confluence_score": decision_analysis.get("confluence_score"),
            "confluence_threshold": decision_analysis.get("confluence_threshold"),
            "recommended_action": decision_analysis.get("recommended_action"),
            "entry_quality": decision_analysis.get("entry_quality"),
            "trend": dict(decision_analysis.get("trend", {}) or {}),
            "momentum": dict(decision_analysis.get("momentum", {}) or {}),
            "location": dict(decision_analysis.get("location", {}) or {}),
            "setup_type": decision_analysis.get("setup_type"),
            "market_regime": dict(decision_analysis.get("market_regime", {}) or {}),
            "reversal_detection": dict(decision_analysis.get("reversal_detection", {}) or {}),
            "context_alignment": dict(decision_analysis.get("context_alignment", {}) or {}),
            "macro_context": dict(decision_analysis.get("macro_context", {}) or {}),
            "trading_response": {
                "size_band_rationale": dict(
                    dict(decision_analysis.get("trading_response", {}) or {}).get("size_band_rationale", {}) or {}
                ),
            },
            "decision_protocol": dict(decision_analysis.get("decision_protocol", {}) or {}),
            "reasoning": list(decision_analysis.get("reasoning", []) or [])[:6],
            "warnings": list(decision_analysis.get("warnings", []) or [])[:6],
            "explanation": decision_analysis.get("explanation"),
        },
        "decision_explanation": dict(status_payload.get("decision_explanation", {}) or {}),
        "decision_trace": {
            "schema": str(decision_trace.get("schema", "") or ""),
            "symbol": str(decision_trace.get("symbol", status_payload.get("symbol", "")) or ""),
            "timeframe": str(decision_trace.get("timeframe", status_payload.get("timeframe", "")) or ""),
            "final_action": str(decision_trace.get("final_action", status_payload.get("action", "")) or ""),
            "final_decision": str(decision_trace.get("final_decision", "") or ""),
            "message": str(decision_trace.get("message", status_payload.get("message", "")) or ""),
            "selected_candidate": dict(decision_trace.get("selected_candidate", {}) or {}),
            "context": dict(decision_trace.get("context", {}) or {}),
            "filters": dict(decision_trace.get("filters", {}) or {}),
            "bias_breakdown": dict(decision_trace.get("bias_breakdown", {}) or {}),
            "risk": dict(decision_trace.get("risk", {}) or {}),
            "execution_gate": dict(decision_trace.get("execution_gate", {}) or {}),
            "blocker_class": str(decision_trace.get("blocker_class", "") or ""),
            "primary_blocker": str(decision_trace.get("primary_blocker", "") or ""),
            "blocker_recoverability": str(decision_trace.get("blocker_recoverability", "") or ""),
            "blocker_summary": dict(decision_trace.get("blocker_summary", {}) or {}),
            "ranking": dict(decision_trace.get("ranking", {}) or {}),
            "signal": {
                "active": list(trace_signal.get("active", []) or [])[:8],
                "raw_count": trace_signal.get("raw_count"),
                "active_count": trace_signal.get("active_count"),
                "filtered_out_count": trace_signal.get("filtered_out_count"),
            },
            "trace_notes": list(decision_trace.get("trace_notes", []) or [])[:8],
        },
        "freshness_runtime": _canonical_freshness_runtime(status_payload, settings),
        "system_health": {
            "effective_mode": effective_mode,
            "effective_mode_label": effective_mode_label,
            "mode_truth": mode_truth,
            "backend_health": dict(backend_health),
            "backend_staleness_warning": backend_warning,
            "session_state": canonical_session_state or raw_session_state,
            "raw_session_state": raw_session_state,
            "session_state_source": "session_control" if canonical_session_state else "system_health",
            "broker_connectivity_status": dict(system_health.get("broker_connectivity_status", {}) or {}),
            "warnings": warnings,
            "routing_risk_detected": bool(system_health.get("routing_risk_detected", False)),
            "config_mismatch_detected": bool(system_health.get("config_mismatch_detected", False)),
            "broker_reconciliation_mismatch": bool(automation_payload.get("mismatch_detected", False)),
        },
        "ui_desktop_window": desktop_window,
        "ui_desktop_window_ready": bool(desktop_window.get("ready", False)),
        "automation_truth": automation_truth,
        "operator_readiness": operator_readiness,
        "execution_readiness": execution_readiness,
        "execution": {
            "session_control": session_control,
            "automation_enabled": bool(automation_truth.get("automation_enabled", False)),
            "automation_paused": bool(automation_truth.get("automation_paused", False)),
            "automation_pause_reason": str(automation_truth.get("automation_pause_reason", "") or ""),
            "automation_explicit_resume_active": bool(automation_truth.get("automation_explicit_resume_active", False)),
            "automation_live_allowed": bool(
                automation_payload.get("automation_live_allowed", runtime_execution.get("automation_live_allowed", False))
            ),
            "auto_trade_allowed": (
                bool(automation_payload["auto_trade_allowed"])
                if "auto_trade_allowed" in automation_payload
                else (
                    bool(runtime_execution["auto_trade_allowed"])
                    if "auto_trade_allowed" in runtime_execution
                    else None
                )
            ),
            "auto_trade_status_known": (
                "auto_trade_allowed" in automation_payload
                or "auto_trade_allowed" in runtime_execution
                or "auto_trade_reason" in automation_payload
                or "auto_trade_reason" in runtime_execution
            ),
            "auto_trade_reason": str(
                automation_payload.get(
                    "auto_trade_reason",
                    runtime_execution.get("auto_trade_reason", session_control.get("session_block_reason", "")),
                )
                or session_control.get("session_block_reason", "")
                or ""
            ),
            "session_override_active": bool(
                automation_payload.get("session_override_active", runtime_execution.get("session_override_active", False))
            ),
            "session_override_real_session_active": bool(
                automation_payload.get(
                    "session_override_real_session_active",
                    runtime_execution.get("session_override_real_session_active", False),
                )
            ),
            "session_override_simulated_session_active": bool(
                automation_payload.get(
                    "session_override_simulated_session_active",
                    runtime_execution.get("session_override_simulated_session_active", False),
                )
            ),
            "reconciliation_status": str(automation_payload.get("reconciliation_status", "") or ""),
            "mismatch_detected": bool(automation_payload.get("mismatch_detected", False)),
            "orphan_order_count": len(dict(automation_payload.get("orphan_orders", {}) or {})),
        },
    }


def _build_recovery_status(
    status_payload: dict[str, Any],
    structured_blocker: dict[str, Any],
    loss_pattern_block: dict[str, Any],
) -> dict[str, Any]:
    if bool(loss_pattern_block.get("blocked", False)) or bool(loss_pattern_block.get("soft_penalty_active", False)):
        return {
            "category": "loss_learning",
            "blocked": bool(loss_pattern_block.get("blocked", False)),
            "soft_penalty_active": bool(loss_pattern_block.get("soft_penalty_active", False)),
            "title": "Loss Learning",
            "reason": str(loss_pattern_block.get("reason", "") or status_payload.get("message", "")),
            "primary_metric": {
                "label": "WR / Streak",
                "value": f"{float(loss_pattern_block.get('win_rate', 0.0) or 0.0):.2f} / {int(loss_pattern_block.get('loss_streak', 0) or 0)}",
            },
            "scope": str(dict(loss_pattern_block.get("hour_context", {}) or {}).get("key") or loss_pattern_block.get("scope", "")),
            "checklist": list(loss_pattern_block.get("recovery_checklist", []) or [])[:8],
            "policy_note": str(loss_pattern_block.get("policy_note", "") or ""),
        }

    message = str(status_payload.get("message", "") or "")
    same_side = dict(status_payload.get("same_side_support", {}) or {})
    if "Consensus guard blocked trade" in message or same_side:
        count = int(same_side.get("count", same_side.get("active_count", 0)) or 0)
        required = 0
        match = re.search(r"required=(\d+)", message)
        if match:
            required = int(match.group(1))
        return {
            "category": "consensus",
            "blocked": "Consensus guard blocked trade" in message,
            "soft_penalty_active": False,
            "title": "Consensus",
            "reason": message if "Consensus guard" in message else "Waiting for same-side support.",
            "primary_metric": {"label": "Support", "value": f"{count}/{required or '?'}"},
            "scope": str(same_side.get("source", "same_side_support") or "same_side_support"),
            "checklist": [
                {
                    "code": "same_side_support",
                    "label": "At least two active/raw signals must support the chosen side.",
                    "status": "blocked" if required and count < required else "watching",
                    "detail": f"support={count}, required={required or 'unknown'}",
                },
                {
                    "code": "signal_quality",
                    "label": "The next candidate should keep confidence above threshold after support checks.",
                    "status": "watching",
                    "detail": "",
                },
            ],
            "policy_note": "Consensus recovery status is read-only observability; it does not change strategy, risk, guards, or execution policy.",
        }

    top_blockers = list(structured_blocker.get("top_blockers", []) or [])
    checklist = []
    for blocker in top_blockers[:3]:
        row = dict(blocker or {})
        checklist.append(
            {
                "code": str(row.get("name", "") or "blocker"),
                "label": str(row.get("detail", row.get("name", "Blocker must clear")) or ""),
                "status": "waiting",
                "detail": str(row.get("delta_text", "") or ""),
            }
        )
    return {
        "category": str(structured_blocker.get("class", "general") or "general"),
        "blocked": str(status_payload.get("action", "") or "").lower() == "blocked",
        "soft_penalty_active": False,
        "title": str(structured_blocker.get("code", "") or structured_blocker.get("class", "Recovery") or "Recovery"),
        "reason": str(structured_blocker.get("detail", status_payload.get("message", "")) or ""),
        "primary_metric": {"label": "Action", "value": str(status_payload.get("action", "") or "")},
        "scope": str(structured_blocker.get("recoverability", "") or ""),
        "checklist": checklist,
        "policy_note": "Recovery status is read-only observability; it does not change strategy, risk, guards, or execution policy.",
    }


def _build_safety_lock_review(status_payload: dict[str, Any]) -> dict[str, Any]:
    status_payload = dict(status_payload or {})
    system_health = dict(status_payload.get("system_health", {}) or {})
    raw_safety_layer = dict(status_payload.get("safety_layer", {}) or {})
    health_safety_layer = dict(system_health.get("safety_layer", {}) or {})
    safety_layer = {**raw_safety_layer, **health_safety_layer}
    for key in (
        "loss_streak",
        "supporting_win_rate",
        "rolling_drawdown_pct",
        "rolling_expectancy_zar",
        "dominant_loss_class",
        "dominant_loss_class_count",
        "recovery_requirements",
        "volatility_reason",
    ):
        if key not in health_safety_layer and key in raw_safety_layer:
            safety_layer[key] = raw_safety_layer[key]
    reason_text = str(safety_layer.get("safety_lock_reason", "") or "")
    reasons = [part.strip() for part in reason_text.split(",") if part.strip()]
    if not reasons and bool(safety_layer.get("safety_lock_active", False)):
        reasons = ["safety_lock_active"]
    metric_present = {
        "loss_streak": "loss_streak" in safety_layer,
        "supporting_win_rate": "supporting_win_rate" in safety_layer,
        "rolling_drawdown_pct": "rolling_drawdown_pct" in safety_layer,
        "rolling_expectancy_zar": "rolling_expectancy_zar" in safety_layer,
    }
    metric_map = {
        "consecutive_loss_threshold": {
            "label": "Loss streak",
            "value": int(float(safety_layer.get("loss_streak", 0) or 0)) if metric_present["loss_streak"] else "not_reported",
            "detail": "Recent or session loss streak reached the protective threshold.",
        },
        "supporting_win_rate_weak": {
            "label": "Supporting win rate",
            "value": round(float(safety_layer.get("supporting_win_rate", 0.0) or 0.0), 4) if metric_present["supporting_win_rate"] else "not_reported",
            "detail": "Recent supporting evidence is too weak for normal live entry flow.",
        },
        "rolling_drawdown_limit": {
            "label": "Rolling drawdown",
            "value": round(float(safety_layer.get("rolling_drawdown_pct", 0.0) or 0.0), 4) if metric_present["rolling_drawdown_pct"] else "not_reported",
            "detail": "Drawdown is near or above the configured protection level.",
        },
        "rolling_expectancy_degradation": {
            "label": "Rolling expectancy",
            "value": round(float(safety_layer.get("rolling_expectancy_zar", 0.0) or 0.0), 2) if metric_present["rolling_expectancy_zar"] else "not_reported",
            "detail": "Recent expectancy is degraded enough to require protection.",
        },
    }
    evidence = []
    for reason in reasons:
        metric = dict(metric_map.get(reason, {}) or {})
        evidence.append(
            {
                "code": reason,
                "label": str(metric.get("label", reason.replace("_", " ").title()) or ""),
                "value": metric.get("value", ""),
                "detail": str(metric.get("detail", "Safety layer reported this active lock reason.") or ""),
            }
        )
    if not evidence:
        evidence = [
            {
                "code": "clear",
                "label": "Safety lock",
                "value": "clear",
                "detail": "No active safety lock reason is reported.",
            }
        ]
    recovery_requirements = [str(item) for item in list(safety_layer.get("recovery_requirements", []) or [])[:8]]
    if bool(safety_layer.get("safety_lock_active", False)) and not recovery_requirements:
        recovery_requirements = [
            "manual_operator_unlock_required",
            "cooldown_expired",
            "no_active_hard_blocker",
            "volatility_back_to_acceptable",
            "next_setup_meets_high_confluence",
        ]
    recovery_blockers = [str(item) for item in list(safety_layer.get("recovery_blockers", []) or [])[:8]]
    checklist = []
    requirement_labels = {
        "manual_operator_unlock_required": "Manual operator review is required before unlock.",
        "cooldown_expired": "Cooldown must be expired or inactive.",
        "no_active_hard_blocker": "No hard market, timing, broker, or risk blocker may remain.",
        "volatility_back_to_acceptable": "Volatility must be acceptable for new entries.",
        "next_setup_meets_high_confluence": "The next setup must meet high-confluence normal gates.",
    }
    for requirement in recovery_requirements:
        blocked = requirement in recovery_blockers or (
            requirement == "manual_operator_unlock_required" and bool(safety_layer.get("safety_lock_active", False))
        )
        checklist.append(
            {
                "code": requirement,
                "label": requirement_labels.get(requirement, requirement.replace("_", " ")),
                "status": "blocked" if blocked else "review",
                "detail": "",
            }
        )
    if bool(safety_layer.get("cooldown_active", False)) and not any(item["code"] == "cooldown_expired" for item in checklist):
        checklist.append(
            {
                "code": "cooldown_expired",
                "label": requirement_labels["cooldown_expired"],
                "status": "blocked",
                "detail": str(safety_layer.get("cooldown_until", "") or ""),
            }
        )
    if str(safety_layer.get("volatility_action", "") or "") == "block_new_entries":
        checklist.append(
            {
                "code": "volatility_back_to_acceptable",
                "label": requirement_labels["volatility_back_to_acceptable"],
                "status": "blocked",
                "detail": str(safety_layer.get("volatility_reason", "") or ""),
            }
        )
    active = bool(safety_layer.get("safety_lock_active", False))
    return {
        "enabled": True,
        "active": active,
        "status": "locked" if active else ("cooldown" if bool(safety_layer.get("cooldown_active", False)) else "clear"),
        "reason": reason_text,
        "scope": str(safety_layer.get("safety_lock_scope", "new_entries") or "new_entries"),
        "started_at": str(safety_layer.get("safety_lock_started_at", "") or ""),
        "recovery_mode": str(safety_layer.get("recovery_mode", "normal") or "normal"),
        "recovery_ready": bool(safety_layer.get("recovery_ready", False)),
        "recovery_blockers": recovery_blockers,
        "evidence": evidence[:8],
        "metrics": {
            "loss_streak": int(float(safety_layer.get("loss_streak", 0) or 0)),
            "supporting_win_rate": round(float(safety_layer.get("supporting_win_rate", 0.0) or 0.0), 4),
            "rolling_drawdown_pct": round(float(safety_layer.get("rolling_drawdown_pct", 0.0) or 0.0), 4),
            "rolling_expectancy_zar": round(float(safety_layer.get("rolling_expectancy_zar", 0.0) or 0.0), 2),
            "metric_availability": metric_present,
            "dominant_loss_class": str(safety_layer.get("dominant_loss_class", "none") or "none"),
            "volatility_state": str(safety_layer.get("volatility_state", "normal") or "normal"),
            "cooldown_active": bool(safety_layer.get("cooldown_active", False)),
            "cooldown_until": str(safety_layer.get("cooldown_until", "") or ""),
        },
        "recovery_checklist": checklist[:10],
        "separate_from_learning": {
            "pattern_ban": False,
            "note": "Safety lock protects the account. It is separate from loss-pattern learning and does not permanently ban a setup family.",
        },
        "next_action": (
            "Keep trading blocked until operator review plus recovery requirements clear."
            if active
            else "No safety lock is active; normal market, risk, timing, and execution gates still apply."
        ),
        "policy_note": "Safety-lock review is read-only observability; it does not unlock safety, change risk, force entries, or bypass execution gates.",
    }


def _build_volatility_protection_review(status_payload: dict[str, Any]) -> dict[str, Any]:
    status_payload = dict(status_payload or {})
    system_health = dict(status_payload.get("system_health", {}) or {})
    safety_layer = {
        **dict(status_payload.get("safety_layer", {}) or {}),
        **dict(system_health.get("safety_layer", {}) or {}),
    }
    decision_analysis = dict(status_payload.get("decision_analysis", {}) or {})
    regime = dict(decision_analysis.get("market_regime", {}) or dict(decision_analysis.get("context_alignment", {}) or {}).get("regime", {}) or {})
    spread_liquidity = dict(system_health.get("spread_liquidity", {}) or dict(decision_analysis.get("macro_context", {}) or {}).get("execution_friction", {}) or {})
    vol_intel = dict(status_payload.get("volatility_intelligence", {}) or {})
    vol_perf = dict(status_payload.get("volatility_learning_performance", {}) or {})
    action = str(safety_layer.get("volatility_action", status_payload.get("volatility_action", "")) or "")
    state = str(safety_layer.get("volatility_state", status_payload.get("volatility_state", "")) or "")
    reason = str(safety_layer.get("volatility_reason", status_payload.get("volatility_reason", "")) or "")
    vol_class = str(vol_intel.get("class", "") or ("extreme_volatility" if state == "extreme" else state or "unknown"))
    response_rule = str(vol_intel.get("response_rule", "") or ("volatility_hard_block" if action == "block_new_entries" else "normal_analysis"))
    hard_block = action == "block_new_entries" or response_rule in {"volatility_hard_block", "news_hard_block", "spread_slippage_block"}
    regime_reasons = [str(item) for item in list(regime.get("regime_reasons", regime.get("reasons", [])) or [])[:6]]
    recovery_blockers = [str(item) for item in list(safety_layer.get("recovery_blockers", []) or [])[:8]]
    requirements = [
        {
            "code": "volatility_back_to_acceptable",
            "status": "blocked" if hard_block else "clear",
            "detail": reason or ("volatility hard block active" if hard_block else "volatility is not blocking new entries"),
        },
        {
            "code": "spread_liquidity_acceptable",
            "status": "clear" if str(spread_liquidity.get("spread_status", spread_liquidity.get("spread_state", "normal")) or "normal") == "normal" else "review",
            "detail": str(spread_liquidity.get("spread_explanation", "") or ""),
        },
        {
            "code": "setup_meets_high_confluence",
            "status": "blocked" if hard_block else "review",
            "detail": "Wait for normal gates after volatility protection clears.",
        },
    ]
    return {
        "schema": "atlas_volatility_protection_review_v1",
        "status": "hard_block" if hard_block else ("reduced_aggression" if action == "reduce_aggressiveness" else "clear"),
        "hard_block": hard_block,
        "state": state or "unknown",
        "class": vol_class,
        "response_rule": response_rule,
        "action": action or "unknown",
        "reason": reason,
        "evidence": {
            "regime": str(regime.get("regime", regime.get("state", "")) or ""),
            "regime_confidence": regime.get("regime_confidence", regime.get("score")),
            "regime_reasons": regime_reasons,
            "spread_status": str(spread_liquidity.get("spread_status", spread_liquidity.get("spread_state", "")) or ""),
            "spread_points": spread_liquidity.get("spread_points"),
            "liquidity_state": str(spread_liquidity.get("liquidity_state", "") or ""),
            "tags": [str(item) for item in list(vol_intel.get("tags", []) or [])[:8]],
            "blockers": [str(item) for item in list(vol_intel.get("blockers", []) or recovery_blockers)[:8]],
        },
        "learning_performance": {
            "sample_size": int(vol_perf.get("sample_size", 0) or 0),
            "summary": dict(vol_perf.get("summary", {}) or {}),
            "recommendation_checkpoint": dict(vol_perf.get("recommendation_checkpoint", {}) or {}),
        },
        "recovery_requirements": requirements,
        "next_action": (
            "Do not open new trades until volatility returns to acceptable conditions and normal timing/confluence gates pass."
            if hard_block
            else "Volatility is not hard-blocking new entries; normal gates still apply."
        ),
        "policy_note": "Volatility protection review is read-only observability; it does not lower thresholds, change risk, unlock safety, or bypass execution gates.",
    }


def _build_loss_pattern_block_status(status_payload: dict[str, Any]) -> dict[str, Any]:
    direct = dict(status_payload.get("loss_pattern_block_detail", {}) or {})
    loss_learning = dict(status_payload.get("loss_learning", {}) or {})
    hard_block = dict(loss_learning.get("hard_block", {}) or {})
    detail = direct or hard_block
    if not detail:
        if bool(loss_learning.get("applied", False)) or float(loss_learning.get("penalty", 0.0) or 0.0) > 0:
            details = dict(loss_learning.get("details", {}) or {})
            recent_side = dict(details.get("recent_side", {}) or {})
            hour_pattern = dict(details.get("hour_pattern", {}) or {})
            side_detail = recent_side if bool(recent_side.get("applied", False)) else hour_pattern
            return {
                "blocked": False,
                "soft_penalty_active": True,
                "code": "loss_learning_penalty",
                "trigger": str(side_detail.get("source", "") or "recent_loss_memory"),
                "reason": str(loss_learning.get("reason", status_payload.get("message", "")) or ""),
                "side": str(side_detail.get("side", "") or ""),
                "scope": str(side_detail.get("scope", "") or ""),
                "samples": int(side_detail.get("samples", 0) or 0),
                "wins": int(side_detail.get("wins", 0) or 0),
                "losses": int(side_detail.get("losses", 0) or 0),
                "win_rate": round(float(side_detail.get("win_rate", 0.0) or 0.0), 4),
                "loss_streak": int(side_detail.get("loss_streak", 0) or 0),
                "hour_utc": int(side_detail.get("hour_utc", 0) or 0),
                "hour_context": {},
                "thresholds": {},
                "penalty": round(float(loss_learning.get("penalty", 0.0) or 0.0), 4),
                "clears_when": str(loss_learning.get("clears_when", side_detail.get("clears_when", "")) or ""),
                "evidence_required": list(loss_learning.get("evidence_required", side_detail.get("evidence_required", [])) or [])[:8],
                "recovery_checklist": [
                    {
                        "code": "loss_streak_breaks",
                        "label": "A fresh win breaks or reduces the current loss streak.",
                        "status": "waiting",
                        "detail": f"current_streak={int(side_detail.get('loss_streak', 0) or 0)}",
                    },
                    {
                        "code": "win_rate_improves",
                        "label": "Recent win rate improves enough for the penalty to fade.",
                        "status": "waiting",
                        "detail": f"wr={float(side_detail.get('win_rate', 0.0) or 0.0):.2f}",
                    },
                    {
                        "code": "cleaner_setup_evidence",
                        "label": "Next setup shows cleaner confirmation than recent losing examples.",
                        "status": "review",
                        "detail": "",
                    },
                ],
                "policy_note": "Loss-learning penalty status is read-only observability; it does not change strategy, risk, guards, or execution policy.",
            }
        return {
            "blocked": False,
            "soft_penalty_active": False,
            "code": "",
            "recovery_checklist": [],
            "policy_note": "Loss-pattern block status is read-only observability.",
        }
    recovery = []
    for item in list(detail.get("recovery_checklist", []) or [])[:8]:
        row = dict(item or {})
        recovery.append(
            {
                "code": str(row.get("code", "") or ""),
                "label": str(row.get("label", "") or ""),
                "status": str(row.get("status", "") or ""),
                "detail": str(row.get("detail", "") or ""),
            }
        )
    return {
        "blocked": bool(detail.get("blocked", False) or loss_learning.get("hard_blocked", False)),
        "soft_penalty_active": bool(loss_learning.get("applied", False)) and not bool(detail.get("blocked", False)),
        "code": str(detail.get("code", "") or ""),
        "trigger": str(detail.get("trigger", "") or ""),
        "reason": str(detail.get("reason", status_payload.get("message", "")) or ""),
        "side": str(detail.get("side", "") or ""),
        "scope": str(detail.get("scope", "") or ""),
        "samples": int(detail.get("samples", 0) or 0),
        "wins": int(detail.get("wins", 0) or 0),
        "losses": int(detail.get("losses", 0) or 0),
        "win_rate": round(float(detail.get("win_rate", 0.0) or 0.0), 4),
        "loss_streak": int(detail.get("loss_streak", 0) or 0),
        "hour_utc": int(detail.get("hour_utc", 0) or 0),
        "hour_context": dict(detail.get("hour_context", {}) or {}),
        "thresholds": dict(detail.get("thresholds", {}) or {}),
        "clears_when": str(detail.get("clears_when", "") or ""),
        "evidence_required": list(detail.get("evidence_required", []) or [])[:8],
        "recovery_checklist": recovery,
        "policy_note": str(
            detail.get(
                "policy_note",
                "Loss-pattern block status is read-only observability; it does not change strategy, risk, guards, or execution policy.",
            )
            or ""
        ),
    }


def _build_learning_critic(status_payload: dict[str, Any], store: KnowledgeStore) -> dict[str, Any]:
    current = dict(status_payload.get("learning_critic", {}) or {})
    current_status = str(current.get("status", current.get("reason", "")) or "")
    should_use_current = bool(current) and (
        bool(current.get("applied", False))
        or (int(current.get("samples", 0) or 0) > 0)
        or current_status not in {"", "flat_side", "no_matching_pattern_lesson", "lesson_not_loss_dominant"}
    )
    if should_use_current:
        current.setdefault("enabled", True)
        current.setdefault(
            "policy_note",
            "Learning critic is observability only unless an existing bounded lesson adjustment applies.",
        )
        return current

    try:
        state = dict(store.load_learning_state().get("state", {}) or {})
    except Exception:
        state = {}
    lessons = dict(dict(state.get("loss_memory", {}) or {}).get("pattern_lessons", {}) or {})
    strongest: dict[str, Any] = {}
    strongest_score = -1.0
    for lesson in lessons.values():
        row = dict(lesson or {})
        samples = int(row.get("samples", 0) or 0)
        losses = int(row.get("losses", 0) or 0)
        loss_rate = float(row.get("loss_rate", 0.0) or 0.0)
        if samples < 1 or losses < 1:
            continue
        score = loss_rate * min(5, samples)
        if score > strongest_score:
            strongest = row
            strongest_score = score
    if not strongest:
        return {
            "enabled": True,
            "applied": False,
            "status": "no_loss_pattern_lessons_yet",
            "match_strength": "not_enough_evidence",
            "match_score": 0.0,
            "policy_note": "Learning critic is observability only unless an existing bounded lesson adjustment applies.",
        }

    post_trade_lesson = dict(strongest.get("post_trade_lesson", {}) or {})
    return {
        "enabled": True,
        "applied": False,
        "status": "watching_for_repeated_patterns",
        "fingerprint": str(strongest.get("fingerprint", "") or ""),
        "matched_fingerprint": str(strongest.get("fingerprint", "") or ""),
        "match_strength": "stored_loss_memory_watch",
        "match_score": round(float(strongest.get("loss_rate", 0.0) or 0.0), 4),
        "samples": int(strongest.get("samples", 0) or 0),
        "loss_rate": round(float(strongest.get("loss_rate", 0.0) or 0.0), 4),
        "previous_failure_axis": str(post_trade_lesson.get("failure_axis", strongest.get("failure_axis", "")) or ""),
        "idea_quality": str(post_trade_lesson.get("idea_quality", strongest.get("idea_quality", "")) or ""),
        "timing_quality": str(post_trade_lesson.get("timing_quality", strongest.get("timing_quality", "")) or ""),
        "next_time_action": str(post_trade_lesson.get("next_time_action", "") or ""),
        "future_recognition": list(post_trade_lesson.get("future_recognition", []) or [])[:8],
        "policy_note": "Learning critic is observability only unless an existing bounded lesson adjustment applies.",
    }


def _build_loss_learning_review(status_payload: dict[str, Any], store: KnowledgeStore) -> dict[str, Any]:
    status_payload = dict(status_payload or {})
    critic = _build_learning_critic(status_payload, store)
    loss_to_win = dict(status_payload.get("loss_to_win_playbook", {}) or {})
    winner_playbook = dict(status_payload.get("winner_playbook", {}) or {})
    top_loss_lesson = dict(loss_to_win.get("top_lesson", {}) or {})
    top_winner_lesson = dict(winner_playbook.get("top_lesson", {}) or {})
    loss_missing = [str(item) for item in list(top_loss_lesson.get("missing", []) or [])[:8]]
    winner_missing = [str(item) for item in list(top_winner_lesson.get("missing", []) or [])[:8]]
    corrected = bool(top_loss_lesson.get("confirmed", False)) or str(critic.get("status", "")) == "cleared_by_current_confirmation"
    bounded_caution = bool(critic.get("applied", False)) or str(loss_to_win.get("status", "")) == "loss_pattern_needs_correction"
    safety_layer = dict(dict(status_payload.get("system_health", {}) or {}).get("safety_layer", status_payload.get("safety_layer", {})) or {})
    previous_mistake = (
        str(top_loss_lesson.get("classification", "") or "")
        or str(critic.get("previous_failure_axis", "") or "")
        or str(critic.get("status", "") or "no_specific_loss_pattern_matched")
    )
    corrected_entry = str(top_loss_lesson.get("corrected_entry", "") or critic.get("next_time_action", "") or "")
    required = [str(item) for item in list(top_loss_lesson.get("confirmation_required", []) or [])[:8]]
    current_missing = loss_missing or [str(item) for item in list(critic.get("future_recognition", []) or [])[:8]]
    teachings_payload = _load_json_file(Path(store.base).parent / "knowledge" / "loss_pattern_teachings.json")
    if not teachings_payload:
        teachings_payload = _load_json_file(Path.cwd() / "knowledge" / "loss_pattern_teachings.json")
    history_teachings_payload = _load_json_file(Path(store.base).parent / "knowledge" / "loss_history_teachings.json")
    if not history_teachings_payload:
        history_teachings_payload = _load_json_file(Path.cwd() / "knowledge" / "loss_history_teachings.json")
    teachings = [dict(item or {}) for item in list(teachings_payload.get("teachings", []) or [])]
    history_clusters = [dict(item or {}) for item in list(history_teachings_payload.get("loss_clusters", []) or [])]
    relevance_terms = {
        str(previous_mistake).lower(),
        *[str(item).lower() for item in current_missing],
        *[str(item).lower() for item in required],
    }
    relevant_teachings: list[dict[str, Any]] = []
    for teaching in teachings:
        text = " ".join(
            [
                str(teaching.get("id", "")),
                str(teaching.get("title", "")),
                str(teaching.get("lesson", "")),
                " ".join(str(x) for x in list(teaching.get("confirmation_required", []) or [])),
            ]
        ).lower()
        if any(term and term in text for term in relevance_terms):
            relevant_teachings.append(teaching)
    if not relevant_teachings:
        relevant_teachings = teachings[:3]
    status = "corrected_pattern_ready_for_normal_gates" if corrected else (
        "same_pattern_allowed_after_corrections" if bounded_caution else "watching_for_loss_pattern_match"
    )
    return {
        "enabled": True,
        "status": status,
        "is_pattern_ban": False,
        "same_pattern_can_trade_again": True,
        "current_setup_corrected_previous_loss": corrected,
        "bounded_caution_active": bounded_caution and not corrected,
        "previous_mistake": previous_mistake,
        "corrected_entry": corrected_entry,
        "confirmation_required": required,
        "current_missing_confirmations": current_missing,
        "winner_confirmation_missing": winner_missing,
        "learning_critic": critic,
        "loss_to_win_top_lesson": top_loss_lesson,
        "winner_top_lesson": top_winner_lesson,
        "durable_teachings": {
            "schema_version": teachings_payload.get("schema_version", 1),
            "relevant": relevant_teachings[:6],
            "all": teachings[:12],
            "policy_note": str(teachings_payload.get("policy_note", "") or "Teachings are bounded knowledge only."),
        },
        "historical_loss_teachings": {
            "schema_version": history_teachings_payload.get("schema_version", 1),
            "sample_summary": dict(history_teachings_payload.get("sample_summary", {}) or {}),
            "grounding_note": str(history_teachings_payload.get("grounding_note", "") or ""),
            "clusters": history_clusters[:8],
            "global_teachings": [str(item) for item in list(history_teachings_payload.get("global_teachings", []) or [])[:8]],
            "policy_note": str(history_teachings_payload.get("policy_note", "") or "Historical teachings are bounded knowledge only."),
        },
        "safety_lock_separate": {
            "active": bool(safety_layer.get("safety_lock_active", False)),
            "reason": str(safety_layer.get("safety_lock_reason", "") or ""),
            "note": "Safety lock is account protection, not a permanent pattern ban.",
        },
        "operator_summary": (
            "Atlas can trade this pattern again only after the failed ingredient is corrected."
            if not corrected
            else "Current evidence corrects the prior loss pattern; normal gates still decide."
        ),
        "policy_note": "Loss learning is bounded teaching evidence; it does not ban patterns, force entries, lower thresholds, or bypass risk/execution gates.",
    }


def _build_learning_loop_status(store: KnowledgeStore) -> dict[str, Any]:
    from atlas_ai.learning.evidence_provenance import source_counts

    try:
        state = dict(store.load_learning_state().get("state", {}) or {})
    except Exception:
        state = {}
    loss_memory = dict(state.get("loss_memory", {}) or {})
    backfill = dict(loss_memory.get("lesson_backfill", {}) or {})
    decision_context = dict(backfill.get("decision_context", {}) or {})
    context_quality = dict(backfill.get("context_quality", {}) or {})
    events = _tail_learning_events_fast(store)
    last_closed_learning: dict[str, Any] = {}
    last_backfill_event: dict[str, Any] = {}
    last_error: dict[str, Any] = {}
    for event in events:
        message = str(event.get("message", "") or "")
        if message in {
            "Closed-trade learning summary updated.",
            "Processed closed deal(s) for win/loss learning.",
        } or "closed deal" in message.lower():
            last_closed_learning = dict(event)
        if message in {
            "Loss-pattern lesson memory refreshed after closed trade.",
            "Loss-pattern lesson memory backfilled.",
        }:
            last_backfill_event = dict(event)
        if "failed" in message.lower() or "error" in message.lower():
            last_error = dict(event)

    lessons = dict(loss_memory.get("pattern_lessons", {}) or {})
    recent_outcomes = list(loss_memory.get("recent_outcomes", []) or [])
    derived_quality = {
        "location_known": 0,
        "momentum_known": 0,
        "volatility_known": 0,
        "decision_context_linked": 0,
    }
    for lesson in lessons.values():
        quality = dict(dict(lesson or {}).get("context_quality", {}) or {})
        derived_quality["location_known"] += int(bool(quality.get("location_known", False)))
        derived_quality["momentum_known"] += int(bool(quality.get("momentum_known", False)))
        derived_quality["volatility_known"] += int(bool(quality.get("volatility_known", False)))
        derived_quality["decision_context_linked"] += int(bool(quality.get("decision_context_linked", False)))
    status = "waiting"
    if last_error and str(last_error.get("time", "")) >= str(last_backfill_event.get("time", "")):
        status = "error"
    elif backfill or last_backfill_event:
        status = "active"
    return {
        "status": status,
        "last_closed_trade_learning_at": str(last_closed_learning.get("time", "") or ""),
        "last_lesson_refresh_at": str(backfill.get("rebuilt_at") or last_backfill_event.get("time", "") or ""),
        "recent_outcomes": int(backfill.get("recent_outcomes", len(recent_outcomes)) or 0),
        "pattern_lessons": int(backfill.get("pattern_lessons", len(lessons)) or 0),
        "loss_dominant_lessons": int(backfill.get("loss_dominant_lessons", 0) or 0),
        "outcome_sources": source_counts(loss_memory),
        "decision_context": {
            "decision_samples_scanned": int(decision_context.get("decision_samples_scanned", 0) or 0),
            "outcomes_scanned": int(decision_context.get("outcomes_scanned", len(recent_outcomes)) or 0),
            "outcomes_linked": int(decision_context.get("outcomes_linked", derived_quality["decision_context_linked"]) or 0),
            "link_rate": round(float(decision_context.get("link_rate", 0.0) or 0.0), 4),
        },
        "context_quality": {
            "location_known": int(context_quality.get("location_known", derived_quality["location_known"]) or 0),
            "momentum_known": int(context_quality.get("momentum_known", derived_quality["momentum_known"]) or 0),
            "volatility_known": int(context_quality.get("volatility_known", derived_quality["volatility_known"]) or 0),
        },
        "last_error": {
            "time": str(last_error.get("time", "") or ""),
            "message": str(last_error.get("message", "") or ""),
            "meta": dict(last_error.get("meta", {}) or {}),
        },
        "policy_note": "Learning loop status is read-only observability; it does not change strategy, risk, guards, or execution policy.",
    }


def _count_proposals_for_learning_lanes(store: KnowledgeStore, status_payload: dict[str, Any]) -> dict[str, Any]:
    proposals_payload = _load_json_file(Path(store.base).parent / "knowledge" / "proposal_queue.json")
    if not proposals_payload:
        proposals_payload = _load_json_file(Path.cwd() / "knowledge" / "proposal_queue.json")
    proposals = [dict(item or {}) for item in list(proposals_payload.get("proposals", []) or [])]
    self_improvement = dict(status_payload.get("self_improvement_state", {}) or {})
    pending_statuses = {"proposed", "pending", "pending_review", "operator_review", "review_required"}
    active_statuses = {"approved_for_rollout", "shadow_active", "active", "monitoring"}
    pending = [item for item in proposals if str(item.get("status", "") or "") in pending_statuses]
    active = [item for item in proposals if str(item.get("status", "") or "") in active_statuses]
    return {
        "total": int(self_improvement.get("proposal_count", len(proposals)) or len(proposals)),
        "pending_review": len(pending),
        "active_or_rollout": len(active),
        "latest_ids": [str(item.get("proposal_id", "") or "") for item in proposals[-5:] if item.get("proposal_id")],
        "queue_updated_at": str(proposals_payload.get("updated_at", "") or ""),
    }


def _build_learning_lanes_status(
    status_payload: dict[str, Any],
    runtime_flags: dict[str, Any],
    store: KnowledgeStore,
) -> dict[str, Any]:
    status_payload = dict(status_payload or {})
    runtime_flags = dict(runtime_flags or {})
    decision_analysis = dict(status_payload.get("decision_analysis", {}) or {})
    decision_trace = dict(status_payload.get("decision_trace", {}) or {})
    visual_chart_read = dict(status_payload.get("visual_chart_read", {}) or {})
    latest_visual = load_latest_visual_read(Path(store.base) / "chart_vision", max_age_seconds=900)
    if bool(latest_visual.get("available", False)) and not bool(latest_visual.get("stale", False)):
        visual_chart_read = latest_visual
    market = dict(status_payload.get("market", {}) or {})
    structured_blocker = dict(status_payload.get("structured_blocker", {}) or {})
    if not structured_blocker:
        structured_blocker = _build_structured_blocker(status_payload, _load_automation_payload(store))
    learning_loop = _build_learning_loop_status(store)
    learning_critic = _build_learning_critic(status_payload, store)
    loss_learning_review = _build_loss_learning_review(status_payload, store)
    loss_pattern_block = _build_loss_pattern_block_status(status_payload)
    teachings_payload = _load_json_file(Path(store.base).parent / "knowledge" / "loss_pattern_teachings.json")
    if not teachings_payload:
        teachings_payload = _load_json_file(Path.cwd() / "knowledge" / "loss_pattern_teachings.json")
    proposal_counts = _count_proposals_for_learning_lanes(store, status_payload)
    timing_engine = dict(decision_analysis.get("timing_engine", {}) or {})
    timing_readiness = dict(status_payload.get("timing_readiness", {}) or {})
    timing_discipline = dict(status_payload.get("timing_discipline", {}) or {})
    explanation = dict(status_payload.get("decision_explanation", {}) or {})
    explanation_text = str(
        explanation.get("summary")
        or explanation.get("operator_summary")
        or decision_analysis.get("explanation")
        or status_payload.get("message", "")
        or ""
    )
    recent_close = list(market.get("recent_close", []) or [])
    observe_active = bool(visual_chart_read.get("available", False)) or bool(recent_close) or bool(status_payload.get("updated_at"))
    classify_active = bool(structured_blocker.get("code") or structured_blocker.get("class") or decision_trace or decision_analysis)
    remember_active = learning_loop.get("status") == "active" or int(learning_loop.get("pattern_lessons", 0) or 0) > 0
    explain_active = bool(explanation_text or loss_learning_review.get("operator_summary") or structured_blocker.get("summary"))
    bounded_caution_active = bool(
        learning_critic.get("applied", False)
        or loss_pattern_block.get("blocked", False)
        or loss_pattern_block.get("soft_penalty_active", False)
        or timing_readiness.get("hard_block", False)
        or timing_discipline.get("hard_block", False)
        or str(timing_engine.get("status", "") or "").lower() in {"blocked", "caution", "waiting"}
    )
    lanes = {
        "observe": {
            "status": "active" if observe_active else "waiting",
            "active": observe_active,
            "source": "visual_chart_read|market_recent_close|status_freshness",
            "evidence": {
                "visual_available": bool(visual_chart_read.get("available", False)),
                "visual_stale": bool(visual_chart_read.get("stale", False)),
                "visual_pattern_seen": str(visual_chart_read.get("pattern_seen", "") or ""),
                "recent_close_points": len(recent_close),
                "status_updated_at": str(status_payload.get("updated_at", "") or ""),
            },
            "policy_note": "Observation collects evidence only; it does not execute trades.",
        },
        "classify": {
            "status": "active" if classify_active else "waiting",
            "active": classify_active,
            "source": "structured_blocker|decision_trace|playbooks",
            "evidence": {
                "blocker_code": str(structured_blocker.get("code", "") or ""),
                "blocker_class": str(structured_blocker.get("class", status_payload.get("blocker_class", "")) or ""),
                "final_action": str(decision_trace.get("final_action", status_payload.get("action", "")) or ""),
                "loss_pattern_previous_mistake": str(loss_learning_review.get("previous_mistake", "") or ""),
            },
            "policy_note": "Classification labels evidence and blockers; it does not rewrite strategy logic.",
        },
        "remember": {
            "status": str(learning_loop.get("status", "waiting") or "waiting"),
            "active": remember_active,
            "source": "learning_state|loss_memory|durable_teachings",
            "evidence": {
                "recent_outcomes": int(learning_loop.get("recent_outcomes", 0) or 0),
                "pattern_lessons": int(learning_loop.get("pattern_lessons", 0) or 0),
                "loss_dominant_lessons": int(learning_loop.get("loss_dominant_lessons", 0) or 0),
                "durable_teachings": len(list(teachings_payload.get("teachings", []) or [])),
                "last_lesson_refresh_at": str(learning_loop.get("last_lesson_refresh_at", "") or ""),
            },
            "policy_note": "Memory supports continuity and bounded lesson evidence; it does not silently change trading controls.",
        },
        "explain": {
            "status": "active" if explain_active else "waiting",
            "active": explain_active,
            "source": "decision_explanation|blocker_summary|loss_learning_review",
            "evidence": {
                "summary": explanation_text,
                "loss_learning_summary": str(loss_learning_review.get("operator_summary", "") or ""),
                "primary_blocker": str(status_payload.get("primary_blocker", structured_blocker.get("detail", "")) or ""),
            },
            "policy_note": "Explanation is operator-facing reasoning; it does not create permission to trade.",
        },
        "bounded_caution": {
            "status": "active" if bounded_caution_active else "watching",
            "active": bounded_caution_active,
            "source": "learning_critic|timing_discipline|loss_pattern_block",
            "evidence": {
                "learning_critic_status": str(learning_critic.get("status", "") or ""),
                "learning_critic_applied": bool(learning_critic.get("applied", False)),
                "loss_pattern_blocked": bool(loss_pattern_block.get("blocked", False)),
                "loss_pattern_soft_penalty": bool(loss_pattern_block.get("soft_penalty_active", False)),
                "timing_hard_block": bool(timing_readiness.get("hard_block", False) or timing_discipline.get("hard_block", False)),
                "missing_confirmations": list(loss_learning_review.get("current_missing_confirmations", []) or [])[:8],
            },
            "policy_note": "Bounded caution may only use existing allowed caution/penalty paths; it cannot lower thresholds, force entries, or bypass risk/execution gates.",
        },
        "propose_improvements": {
            "status": "active" if proposal_counts["total"] else "waiting",
            "active": bool(proposal_counts["total"]),
            "source": "proposal_queue|self_improvement_state|control_queue",
            "evidence": proposal_counts,
            "policy_note": "Improvement proposals are review items only until explicitly approved and validated under governance.",
        },
    }
    active_count = sum(1 for lane in lanes.values() if bool(lane.get("active", False)))
    return {
        "enabled": True,
        "status": "active" if active_count else "waiting",
        "active_lanes": active_count,
        "lanes": lanes,
        "policy_note": "Learning lanes are read-only observability and bounded decision support; this does not change strategy, risk, or execution policy by itself.",
    }


def _build_cognitive_state_payload(
    status_payload: dict[str, Any],
    runtime_flags: dict[str, Any],
    store: KnowledgeStore,
) -> dict[str, Any]:
    status_payload = dict(status_payload or {})
    runtime_flags = dict(runtime_flags or {})
    automation_payload = _load_automation_payload(store)
    desktop_window = _reconcile_desktop_window_state(dict(runtime_flags.get("ui_desktop_window", {}) or {}), store)
    reconciled_runtime = dict(runtime_flags)
    reconciled_runtime["ui_desktop_window"] = desktop_window
    operator_readiness = _build_operator_readiness(status_payload, reconciled_runtime, automation_payload, desktop_window)
    execution_readiness = _build_execution_readiness(
        status_payload,
        reconciled_runtime,
        automation_payload,
        operator_readiness,
    )
    learning_lanes = _build_learning_lanes_status(status_payload, reconciled_runtime, store)
    try:
        learning_state = dict(store.load_learning_state() or {})
    except Exception:
        learning_state = {}
    events = _tail_learning_events_fast(store, limit=120)
    adaptation = build_adaptation_summary(status_payload, events)
    cognitive_core = build_cognitive_core(
        status=status_payload,
        runtime={"execution": automation_payload, **reconciled_runtime},
        events=events,
        learning_state=learning_state,
        knowledge_payload={
            "lessons_learned": _load_json_file(Path(store.base).parent / "knowledge" / "lessons_learned.json")
            or _load_json_file(Path.cwd() / "knowledge" / "lessons_learned.json"),
            "operator_teaching_reviews": _load_json_file(Path(store.base).parent / "knowledge" / "operator_teaching_reviews.json")
            or _load_json_file(Path.cwd() / "knowledge" / "operator_teaching_reviews.json"),
        },
        adaptation_summary=adaptation,
        evidence_separation={
            "market_bias": str(dict(status_payload.get("decision_analysis", {}) or {}).get("final_trade_bias", "")),
            "signal_direction": str(dict(status_payload.get("chosen", {}) or {}).get("side", "flat")),
            "higher_timeframe_bias": str(dict(status_payload.get("decision_analysis", {}) or {}).get("higher_timeframe_bias", "")),
            "final_trade_bias": str(dict(status_payload.get("decision_analysis", {}) or {}).get("final_trade_bias", "neutral")),
            "setup_readiness": str(dict(status_payload.get("decision_analysis", {}) or {}).get("setup_readiness", "")),
            "execution_permission": "allowed" if bool(execution_readiness.get("ready", False)) else "not_allowed",
            "primary_blocker": str(dict(execution_readiness.get("primary_blocker", {}) or {}).get("detail", status_payload.get("message", "")) or ""),
        },
    )
    lanes = dict(learning_lanes.get("lanes", {}) or {})
    active_lane_names = [name for name, lane in lanes.items() if bool(dict(lane or {}).get("active", False))]
    primary_blocker = dict(execution_readiness.get("primary_blocker", {}) or {})
    mind_state = "blocked_but_learning" if primary_blocker else "coordinated"
    if bool(execution_readiness.get("ready", False)):
        mind_state = "ready_to_evaluate_execution_gate"
    return {
        "schema": "atlas_cognitive_state_v1",
        "enabled": True,
        "mind_model": "one_coordinated_mind_with_modular_specialists",
        "state": mind_state,
        "headline": (
            f"Atlas is coordinating {len(active_lane_names)}/6 learning lanes; "
            f"execution is {'ready' if bool(execution_readiness.get('ready', False)) else 'blocked by gates'}."
        ),
        "coordinator": {
            "sees": dict(cognitive_core.get("world_model", {}) or {}),
            "thinks": dict(cognitive_core.get("decision_understanding", {}) or {}),
            "remembers": dict(cognitive_core.get("trade_memory", {}) or {}),
            "cautions": dict(lanes.get("bounded_caution", {}) or {}),
            "proposes": dict(lanes.get("propose_improvements", {}) or {}),
            "primary_blocker": primary_blocker,
        },
        "learning_lanes": learning_lanes,
        "cognitive_core": {
            "schema": cognitive_core.get("schema"),
            "mode": cognitive_core.get("mode"),
            "learning_state": dict(cognitive_core.get("learning_state", {}) or {}),
            "decision_critic": dict(cognitive_core.get("decision_critic", {}) or {}),
            "safety_contract": dict(cognitive_core.get("safety_contract", {}) or {}),
        },
        "execution_readiness": execution_readiness,
        "policy_note": "Cognitive state presents Atlas as one coordinated mind, but does not merge or bypass strategy, risk, or execution authority.",
    }


def _tail_learning_events_fast(store: KnowledgeStore, *, max_bytes: int = 768_000, limit: int = 250) -> list[dict[str, Any]]:
    path = Path(store.events_file)
    if not path.exists():
        return []
    try:
        size = path.stat().st_size
        with path.open("rb") as fh:
            fh.seek(max(0, size - max_bytes))
            raw = fh.read().decode("utf-8", errors="ignore")
    except Exception:
        return []
    rows: list[dict[str, Any]] = []
    for line in raw.splitlines():
        if '"learning"' not in line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if str(row.get("type", "")).strip().lower() == "learning":
            rows.append(row)
    return rows[-max(1, int(limit)) :]


def _external_context_age_seconds(path: Path) -> float | None:
    payload = _load_json_file(path)
    generated_at_raw = str(payload.get("generated_at", "") or "").strip()
    if not generated_at_raw:
        return None
    try:
        generated_at = datetime.fromisoformat(generated_at_raw.replace("Z", "+00:00"))
        if generated_at.tzinfo is None:
            generated_at = generated_at.replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(timezone.utc) - generated_at.astimezone(timezone.utc)).total_seconds())
    except Exception:
        return None


def _refresh_macro_context_if_needed(store: KnowledgeStore, *, max_age_seconds: float = 45 * 60) -> None:
    global _MACRO_REFRESH_IN_FLIGHT, _MACRO_REFRESH_LAST_STARTED
    if os.getenv("PYTEST_CURRENT_TEST"):
        return
    path = Path(__file__).resolve().parent / "knowledge" / "external_context.json"
    age = _external_context_age_seconds(path)
    if age is not None and age <= max_age_seconds:
        return
    with _MACRO_REFRESH_LOCK:
        if _MACRO_REFRESH_IN_FLIGHT:
            return
        if time.time() - _MACRO_REFRESH_LAST_STARTED < 10 * 60:
            return
        _MACRO_REFRESH_IN_FLIGHT = True
        _MACRO_REFRESH_LAST_STARTED = time.time()

    def _worker() -> None:
        global _MACRO_REFRESH_IN_FLIGHT
        try:
            from atlas_ai.macro_context_updater import write_external_context

            payload = write_external_context(path)
            store.log_event(
                "system",
                "External macro context refreshed.",
                {
                    "generated_at": payload.get("generated_at", ""),
                    "source": payload.get("source", ""),
                    "errors": dict(payload.get("errors", {}) or {}),
                },
            )
        except Exception as exc:
            _LOGGER.debug("Macro context refresh failed", exc_info=True)
            try:
                store.log_event("warning", "External macro context refresh failed.", {"error": str(exc)})
            except Exception:
                pass
        finally:
            with _MACRO_REFRESH_LOCK:
                _MACRO_REFRESH_IN_FLIGHT = False

    threading.Thread(target=_worker, name="atlas-macro-context-refresh", daemon=True).start()


class UIDashboardSource:
    def __init__(self, store: KnowledgeStore):
        self.store = store

    @property
    def read_only(self) -> bool:
        return False

    def load_status(self) -> dict:
        return dict(self.store.load_status() or {})

    def load_events(self, limit: int = 120) -> list[dict]:
        return list(self.store.tail_events(limit=limit))

    def load_learning_state(self) -> dict:
        return dict(self.store.load_learning_state() or {})

    def load_runtime_flags(self) -> dict:
        return dict(self.store.load_runtime_flags() or {})

    def save_runtime_flags(self, flags: dict) -> None:
        self.store.save_runtime_flags(flags)

    def log_event(self, event_type: str, message: str, meta: dict | None = None) -> None:
        self.store.log_event(event_type, message, meta)

    def load_comparison(self) -> dict:
        return {}


class ComparisonDashboardSource(UIDashboardSource):
    def __init__(self, store: KnowledgeStore, run_root: Path):
        super().__init__(store)
        self.run_root = Path(run_root)
        self.control_store = KnowledgeStore(str(self.run_root / "control"))
        self.experiment_store = KnowledgeStore(str(self.run_root / "experiment"))

    @property
    def read_only(self) -> bool:
        return True

    def _preflight(self) -> dict:
        return _load_json_file(self.run_root / "preflight.json")

    def _summary(self) -> dict:
        return _load_json_file(self.run_root / "comparison_summary.json")

    def _report(self) -> dict:
        return _load_json_file(self.run_root / "comparison_report.json")

    def _comparison_meta(self) -> dict:
        return load_comparison_meta_from_run_root(self.run_root)

    def load_status(self) -> dict:
        status = dict(self.experiment_store.load_status() or {})
        if not status:
            status = dict((self._report().get("lane_diagnostics", {}) or {}).get("latest_experiment", {}) or {})
        status["comparison"] = self._comparison_meta()
        status["comparison_active"] = True
        status["comparison_run_id"] = str(status["comparison"].get("run_id", ""))
        status["comparison_preset"] = str(status["comparison"].get("preset", ""))
        status.setdefault("lane", "experiment")
        return status

    def load_events(self, limit: int = 120) -> list[dict]:
        return list(self.experiment_store.tail_events(limit=limit))

    def load_learning_state(self) -> dict:
        return dict(self.experiment_store.load_learning_state() or {})

    def load_runtime_flags(self) -> dict:
        flags = dict(self.experiment_store.load_runtime_flags() or {})
        flags["comparison_active"] = True
        flags["read_only_ui"] = True
        flags["comparison_run_id"] = str(self._comparison_meta().get("run_id", ""))
        return flags

    def save_runtime_flags(self, flags: dict) -> None:
        raise RuntimeError("Runtime flag changes are disabled while viewing an active comparison run.")

    def log_event(self, event_type: str, message: str, meta: dict | None = None) -> None:
        self.experiment_store.log_event(event_type, message, meta)

    def load_comparison(self) -> dict:
        return self._comparison_meta()


def _build_operator_health_summary(
    status_payload: dict,
    runtime_flags: dict,
    desktop_window_override: dict | None = None,
) -> dict:
    status_payload = dict(status_payload or {})
    runtime_flags = dict(runtime_flags or {})
    startup_verification = dict(
        runtime_flags.get("startup_verification", status_payload.get("guarded_demo_startup_sanity", {})) or {}
    )
    desktop_window = dict(desktop_window_override or runtime_flags.get("ui_desktop_window", {}) or {})
    system_health = dict(status_payload.get("system_health", {}) or {})
    runtime_execute = bool(runtime_flags.get("execute_trades", False))
    status_execute = bool(status_payload.get("execute_trades", False))
    startup_execute_raw = startup_verification.get("execute_trades", None)
    startup_execute = bool(startup_execute_raw) if startup_execute_raw is not None else None
    if startup_execute is not None:
        effective_mode = "live" if startup_execute else "paper"
        effective_source = "startup_verification"
    elif "execute_trades" in runtime_flags:
        effective_mode = "live" if runtime_execute else "paper"
        effective_source = "runtime_flags"
    else:
        effective_mode = "live" if status_execute else "paper"
        effective_source = "status"
    mismatch_reasons: list[str] = []
    if startup_execute is not None and startup_execute != runtime_execute:
        mismatch_reasons.append("startup_vs_runtime")
    if runtime_execute != status_execute:
        mismatch_reasons.append("runtime_vs_status")
    frontend_mode = str(desktop_window.get("frontend_mode", "") or ("desktop" if desktop_window.get("ok") else "unknown"))
    degraded_reason = str(desktop_window.get("failure_reason", "") or "")
    mode_explanation = "Mode is coherent across startup, runtime, and status."
    if mismatch_reasons:
        mode_explanation = (
            "Mode mismatch detected between "
            + ", ".join(mismatch_reasons)
            + f". Operator mode is currently anchored to {effective_source}."
        )
    if degraded_reason:
        mode_explanation += f" Frontend is degraded because: {degraded_reason}."
    effective_mode_label = str(system_health.get("effective_mode_label", "") or "").strip()
    if not effective_mode_label:
        profile = str(startup_verification.get("profile", "") or "").strip().lower()
        if profile == "guarded_demo":
            effective_mode_label = "GUARDED_DEMO"
        else:
            effective_mode_label = effective_mode.upper()
    return {
        "effective_mode": effective_mode,
        "effective_mode_label": effective_mode_label,
        "effective_mode_source": effective_source,
        "startup_verification": startup_verification,
        "startup_ok": bool(startup_verification.get("startup_ok", False)),
        "runtime_execute_trades": runtime_execute,
        "status_execute_trades": status_execute,
        "mode_mismatch_detected": bool(mismatch_reasons) or bool(system_health.get("config_mismatch_detected", False)),
        "mode_mismatch_reasons": mismatch_reasons,
        "mode_explanation": mode_explanation,
        "core_health": {
            "status_action": str(status_payload.get("action", "") or ""),
            "status_message": str(status_payload.get("message", "") or ""),
            "warnings": list(system_health.get("warnings", []) or []),
        },
        "frontend_mode": frontend_mode,
        "desktop_window_ready": bool(desktop_window.get("desktop_window_ready", desktop_window.get("ready", False))),
        "fallback_active": bool(desktop_window.get("fallback_active", False)),
        "degraded_reason": degraded_reason,
        "degraded_since": str(desktop_window.get("degraded_since", "") or ""),
    }


def _blocker_code(reason: str) -> str:
    text = str(reason or "").strip().lower()
    if not text:
        return "none"
    if ":" in text and re.match(r"^[a-z0-9_:-]+$", text):
        return text.split(":", 1)[0]
    if "higher-timeframe alignment" in text or "htf_bias" in text:
        return "htf_alignment_conflict"
    if "loss-pattern" in text or "loss-learning guard" in text or "loss memory penalty" in text:
        return "loss_pattern_memory"
    if "no directional trade signal" in text:
        return "no_directional_trade_signal"
    if "no high-confidence" in text:
        return "no_high_confidence_setup"
    if "online learning reduced confidence" in text:
        return "online_learning_confidence_penalty"
    if "adaptive guard reduced confidence" in text:
        return "adaptive_confidence_penalty"
    if "outside active session" in text:
        return "outside_active_session"
    if "broker connected" in text and "live loop started" in text:
        return "startup_connected"
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")[:80] or "unknown"


def _recoverability_for_code(code: str) -> str:
    mapping = {
        "htf_alignment_conflict": "wait_for_m1_and_higher_timeframe_alignment",
        "loss_pattern_memory": "wait_for_cleaner_pattern_or_review_loss_lesson",
        "no_directional_trade_signal": "wait_for_buy_or_sell_signal_confirmation",
        "no_high_confidence_setup": "wait_for_setup_or_review_signal_fit",
        "online_learning_confidence_penalty": "wait_for_stronger_confirmation_or_more_samples",
        "adaptive_confidence_penalty": "wait_for_stability_to_improve",
        "outside_active_session": "wait_for_allowed_session",
        "execution_gate_disabled": "operator_enable_execution_gate",
        "kill_switch_active": "operator_clear_kill_switch",
        "approval_missing": "operator_approval_required",
    }
    return mapping.get(str(code or ""), "review_blocker_detail")


def _build_structured_blocker(status_payload: dict[str, Any], automation_payload: dict[str, Any]) -> dict[str, Any]:
    status_payload = dict(status_payload or {})
    automation_payload = dict(automation_payload or {})
    blocker = dict(status_payload.get("blocker_classification", {}) or {})
    analysis = dict(status_payload.get("decision_analysis", {}) or {})
    message = str(
        blocker.get("primary")
        or status_payload.get("primary_blocker")
        or status_payload.get("message")
        or ""
    )
    code = _blocker_code(message)
    hard_blockers = [str(x) for x in list(analysis.get("hard_blockers", []) or analysis.get("blockers", []) or [])]
    session_control = dict(automation_payload.get("session_control", {}) or {})
    gate_decision = dict(
        automation_payload.get("gate_decisions_last_cycle", {})
        or automation_payload.get("last_gate_decision", {})
        or {}
    )
    orphan_orders = dict(automation_payload.get("orphan_orders", {}) or {})
    orphan_rows = [dict(row or {}) for row in orphan_orders.values()]
    orphan_rows.sort(key=lambda row: str(row.get("detected_at", "") or ""), reverse=True)
    return {
        "code": code,
        "class": str(blocker.get("class", status_payload.get("blocker_class", "unknown")) or "unknown"),
        "detail": message,
        "recoverability": str(blocker.get("recoverability") or _recoverability_for_code(code)),
        "hard_blockers": hard_blockers[:8],
        "top_blockers": list(analysis.get("top_blockers", []) or [])[:5],
        "session": {
            "active": bool(session_control.get("session_active", False)),
            "current": str(session_control.get("current_session", "") or ""),
            "reason": str(session_control.get("session_block_reason", "") or ""),
        },
        "gate_decision": {
            "allowed": bool(gate_decision.get("allowed", False)),
            "blocked_by": str(gate_decision.get("blocked_by", "") or ""),
            "code": str(gate_decision.get("code", "") or ""),
            "reason": str(gate_decision.get("reason", "") or ""),
            "detail": dict(gate_decision.get("detail", {}) or {}),
        },
        "reconciliation": {
            "status": str(automation_payload.get("reconciliation_status", "") or ""),
            "mismatch_detected": bool(automation_payload.get("mismatch_detected", False)),
            "orphan_order_count": len(orphan_rows),
            "top_orphan_orders": [
                {
                    "broker_order_id": str(row.get("broker_order_id", "") or ""),
                    "instrument": str(row.get("instrument", "") or ""),
                    "side": str(row.get("side", "") or ""),
                    "size": row.get("size"),
                    "reason": str(row.get("reason", "") or ""),
                    "reviewed": bool(row.get("reviewed", False)),
                }
                for row in orphan_rows[:5]
            ],
        },
        "policy_note": "Structured blocker is observability only; it does not change strategy, risk, guard, or execution policy.",
    }


def _build_operator_readiness(
    status_payload: dict[str, Any],
    runtime_flags: dict[str, Any],
    automation_payload: dict[str, Any],
    desktop_window_override: dict | None = None,
) -> dict[str, Any]:
    status_payload = dict(status_payload or {})
    runtime_flags = dict(runtime_flags or {})
    automation_payload = dict(automation_payload or {})
    automation_truth = _automation_truth_from_payload(automation_payload)
    system_health = dict(status_payload.get("system_health", {}) or {})
    broker = dict(system_health.get("broker_connectivity_status", {}) or {})
    desktop_window = dict(desktop_window_override or {})
    if not desktop_window:
        desktop_window = _reconcile_desktop_window_state(dict(runtime_flags.get("ui_desktop_window", {}) or {}), None)
    structured_blocker = _build_structured_blocker(status_payload, automation_payload)
    market_ready = str(status_payload.get("action", "") or "").lower() in {"buy", "sell", "execute"}
    automation_ready = bool(automation_truth.get("automation_enabled", False)) and not bool(automation_truth.get("automation_paused", False))
    session_control = dict(automation_payload.get("session_control", {}) or {})
    session_ready = bool(session_control.get("session_active", False)) and not bool(session_control.get("session_pause", False))
    action = str(status_payload.get("action", "") or "").strip().lower()
    message = str(status_payload.get("message", "") or "")
    startup_connected = action == "connected" or "broker connected" in message.lower()
    broker_ready = (bool(broker.get("connected", False)) and bool(broker.get("symbol_selected", True))) or startup_connected
    ui_ready = bool(desktop_window.get("ok", False)) or bool(desktop_window.get("ready", False))
    blockers: list[dict[str, str]] = []
    if not market_ready and action in {"hold", "blocked", "failed"}:
        blockers.append({"lane": "market_quality", "code": structured_blocker["code"], "detail": structured_blocker["detail"]})
    if not automation_ready:
        blockers.append({"lane": "automation", "code": "automation_not_enabled", "detail": str(automation_truth.get("automation_pause_reason", "") or "automation disabled")})
    if not session_ready:
        blockers.append({"lane": "session", "code": "session_not_ready", "detail": str(session_control.get("session_block_reason", "") or "session inactive")})
    if not broker_ready:
        blockers.append({"lane": "broker", "code": "broker_not_ready", "detail": "MT5/HFM bridge is not confirmed connected"})
    if not ui_ready:
        blockers.append({"lane": "operator_ui", "code": "ui_degraded", "detail": str(desktop_window.get("failure_reason", "") or "desktop/API surface degraded")})
    return {
        "ready_for_execution_attempt": bool(market_ready and automation_ready and session_ready and broker_ready),
        "lanes": {
            "market_quality": {"ready": market_ready, "blocker": structured_blocker},
            "automation": {"ready": automation_ready, "truth": automation_truth},
            "session": {"ready": session_ready, "control": session_control},
            "broker": {"ready": broker_ready, "status": broker},
            "operator_ui": {"ready": ui_ready, "desktop": desktop_window},
        },
        "primary_lane": blockers[0]["lane"] if blockers else "ready",
        "blockers": blockers,
        "next_action": (
            "Wait for the market-quality blocker to clear."
            if blockers and blockers[0]["lane"] == "market_quality"
            else (
                "Fix the first system/operator blocker before expecting execution."
                if blockers
                else "Existing gates can evaluate the next execution request."
            )
        ),
        "policy_note": "Readiness lanes are explanatory only; they do not lower thresholds or bypass gates.",
    }


def _build_execution_readiness(
    status_payload: dict[str, Any],
    runtime_flags: dict[str, Any],
    automation_payload: dict[str, Any],
    operator_readiness: dict[str, Any] | None = None,
) -> dict[str, Any]:
    status_payload = dict(status_payload or {})
    runtime_flags = dict(runtime_flags or {})
    automation_payload = dict(automation_payload or {})
    operator_readiness = dict(operator_readiness or {})
    action = str(status_payload.get("action", "") or "").strip().lower()
    message = str(status_payload.get("message", "") or "")
    system_health = dict(status_payload.get("system_health", {}) or {})
    safety_layer = dict(system_health.get("safety_layer", status_payload.get("safety_layer", {})) or {})
    structured_blocker = _build_structured_blocker(status_payload, automation_payload)
    execution_truth = dict(
        automation_payload.get("execution_truth", {})
        or automation_payload.get("last_execution_truth", {})
        or automation_payload.get("execution_state", {})
        or automation_payload.get("last_execution_state", {})
        or dict(runtime_flags.get("execution", {}) or {}).get("execution_truth", {})
        or dict(runtime_flags.get("execution", {}) or {}).get("last_execution_truth", {})
        or {}
    )
    execution_stale = bool(execution_truth.get("stale", False)) or bool(execution_truth.get("current") is False)
    execution_reason = str(execution_truth.get("display_reason", execution_truth.get("reason", "")) or "")
    visual = dict(status_payload.get("visual_chart_read", {}) or {})
    timing = dict(status_payload.get("timing_discipline", {}) or {})
    automation_truth = _automation_truth_from_payload(automation_payload)
    execute_trades = bool(runtime_flags.get("execute_trades", status_payload.get("execute_trades", False)))
    safety_locked = bool(safety_layer.get("safety_lock_active", False))
    safety_reason = str(safety_layer.get("safety_lock_reason", "") or "")
    runtime_execution = dict(runtime_flags.get("execution", {}) or {})
    auto_trade_allowed_raw = (
        automation_payload["auto_trade_allowed"]
        if "auto_trade_allowed" in automation_payload
        else runtime_execution.get("auto_trade_allowed", None)
    )
    auto_trade_reason = str(
        (
            automation_payload["auto_trade_reason"]
            if "auto_trade_reason" in automation_payload
            else runtime_execution.get("auto_trade_reason", "")
        )
        or ""
    )
    auto_trade_status_known = auto_trade_allowed_raw is not None or bool(auto_trade_reason)
    auto_trade_allowed = bool(auto_trade_allowed_raw)
    broker_status = dict(system_health.get("broker_connectivity_status", {}) or {})
    broker_ready = bool(broker_status.get("connected", False)) and bool(broker_status.get("symbol_selected", True))
    market_ready = action in {"buy", "sell", "execute"}
    visual_capture = dict(visual.get("capture", {}) or {})
    runtime_visual_watch = dict(runtime_flags.get("visual_watch", {}) or {})
    visual_source = str(visual.get("source", "") or "")
    visual_capture_ok = bool(visual_capture.get("ok", False)) or bool(runtime_visual_watch.get("last_capture_ok", False))
    visual_capture_degraded = (
        bool(visual.get("available", False))
        and not visual_capture_ok
        and visual_source == "chart_features_visual_proxy"
    )
    visual_ready = bool(visual.get("visual_watch_used_for_decision", False)) or bool(
        runtime_visual_watch.get("last_capture_ok", False)
    )
    timing_ready = not bool(timing.get("hard_block", False))

    blockers: list[dict[str, Any]] = []
    if not execute_trades:
        blockers.append({"lane": "runtime", "code": "execute_trades_disabled", "detail": "Runtime execution flag is off."})
    if safety_locked:
        blockers.append({"lane": "safety", "code": "safety_lock_active", "detail": safety_reason or message})
    if not broker_ready:
        blockers.append({"lane": "broker", "code": "broker_not_ready", "detail": "MT5/HFM bridge is not confirmed connected."})
    auto_trade_waiting_for_context = (
        not market_ready
        and (
            auto_trade_reason == "runtime_context_required"
            or not auto_trade_status_known
        )
    )
    if auto_trade_status_known and not auto_trade_allowed and not auto_trade_waiting_for_context:
        blockers.append({"lane": "automation", "code": "auto_trade_not_allowed", "detail": auto_trade_reason or "automation context is not allowing auto-trade."})
    if not market_ready:
        blockers.append({"lane": "market", "code": structured_blocker.get("code", "market_not_ready"), "detail": structured_blocker.get("detail", message)})
    if not timing_ready:
        blockers.append({"lane": "timing", "code": "timing_discipline_block", "detail": ", ".join(list(timing.get("missing", []) or [])[:6])})

    return {
        "ready": not blockers,
        "ready_for_execution_attempt": not blockers,
        "primary_blocker": blockers[0] if blockers else {},
        "blocker_chain": blockers,
        "state": {
            "execute_trades": execute_trades,
            "safety_lock_active": safety_locked,
            "automation_enabled": bool(automation_truth.get("automation_enabled", False)),
            "auto_trade_allowed": auto_trade_allowed,
            "auto_trade_status_known": auto_trade_status_known,
            "auto_trade_waiting_for_context": auto_trade_waiting_for_context,
            "broker_ready": broker_ready,
            "market_action": action or "unknown",
            "visual_watch_ready": visual_ready,
            "visual_watch_used_for_decision": bool(visual.get("visual_watch_used_for_decision", False)),
            "visual_capture_degraded": visual_capture_degraded,
            "visual_capture_source": str(visual_capture.get("capture_source", runtime_visual_watch.get("last_capture_source", "")) or ""),
            "visual_capture_error": str(visual_capture.get("error", runtime_visual_watch.get("last_error", "")) or ""),
            "timing_ready": timing_ready,
            "stale_execution_truth_present": execution_stale,
        },
        "stale_execution_truth": {
            "present": execution_stale,
            "reason": execution_reason,
            "request_id": str(execution_truth.get("request_id", "") or ""),
            "policy_note": "Stale execution truth is audit history, not the current blocker.",
        },
        "next_action": (
            "Do not unlock or trade until the safety lock clears or is explicitly reviewed."
            if safety_locked
            else (
                "Wait for market and timing gates to produce a valid directional setup."
                if blockers
                else "Atlas can submit the next gated execution request."
            )
        ),
        "operator_readiness_primary_lane": str(operator_readiness.get("primary_lane", "") or ""),
        "policy_note": "Execution readiness is read-only observability; it does not change strategy, risk, guard, or execution policy.",
    }


def _build_visual_evidence_review(
    status_payload: dict[str, Any],
    runtime_flags: dict[str, Any],
    store: KnowledgeStore | None = None,
) -> dict[str, Any]:
    status_payload = dict(status_payload or {})
    runtime_flags = dict(runtime_flags or {})
    visual = dict(status_payload.get("visual_chart_read", {}) or {})
    if store is not None:
        latest = load_latest_visual_read(Path(store.base) / "chart_vision", max_age_seconds=900)
        if bool(latest.get("available", False)) and not bool(latest.get("stale", False)):
            visual = latest
    capture = dict(visual.get("capture", {}) or {})
    image = dict(visual.get("image", visual.get("visual_features", {})) or {})
    visual_watch = dict(runtime_flags.get("visual_watch", {}) or {})
    source = str(visual.get("source", "") or "")
    capture_ok = bool(capture.get("ok", False)) or bool(visual_watch.get("last_capture_ok", False))
    screenshot_source = str(capture.get("capture_source", visual_watch.get("last_capture_source", "")) or "")
    error = str(capture.get("error", visual_watch.get("last_error", "")) or "")
    window = dict(capture.get("window", {}) or {})
    actual_screenshot = capture_ok and screenshot_source == "mt5_window" and bool(image.get("available", True))
    proxy_only = source == "chart_features_visual_proxy" or (bool(visual.get("available", False)) and not capture_ok)
    degraded = bool(proxy_only and not actual_screenshot)
    status = "actual_chart_capture" if actual_screenshot else ("proxy_only" if bool(visual.get("available", False)) else "waiting_for_live_terminal_capture")
    headline = (
        "Atlas has an actual MT5/HFM chart screenshot."
        if actual_screenshot
        else (
            "Atlas is using candle-derived chart features as a visual proxy; the live MT5/HFM chart window was not captured."
            if proxy_only
            else "Atlas is waiting for live terminal visual evidence."
        )
    )
    next_action = (
        "Open or foreground the MT5/HFM chart window so the visual watcher can capture the actual chart."
        if degraded and error == "mt5_or_hfm_window_not_found"
        else (
            "Keep the chart window visible; visual evidence is advisory and existing gates still decide."
            if actual_screenshot
            else "Wait for the visual watcher to capture the live terminal chart."
        )
    )
    return {
        "schema": "atlas_visual_evidence_review_v1",
        "status": status,
        "headline": headline,
        "available": bool(visual.get("available", False)),
        "degraded": degraded,
        "actual_screenshot": actual_screenshot,
        "proxy_only": proxy_only,
        "source": source or "unknown",
        "capture": {
            "ok": capture_ok,
            "source": screenshot_source,
            "method": str(dict(capture.get("capture_method", {}) or {}).get("method", visual_watch.get("last_capture_method", "")) or ""),
            "target_window": str(capture.get("target_window", "mt5") or "mt5"),
            "window_found": bool(window.get("found", capture_ok)),
            "error": error,
            "captured_at": str(capture.get("captured_at", visual.get("captured_at", visual_watch.get("last_captured_at", ""))) or ""),
        },
        "image": {
            "available": bool(image.get("available", False)),
            "status": str(image.get("status", "") or ""),
        },
        "read": {
            "pattern_seen": str(visual.get("pattern_seen", "") or ""),
            "setup_cleanliness": str(visual.get("setup_cleanliness", "") or ""),
            "setup_quality": visual.get("setup_quality"),
            "visual_bias": str(visual.get("visual_bias", "") or ""),
            "visual_confirmation": str(visual.get("visual_confirmation", "") or ""),
            "used_for_decision": bool(visual.get("visual_watch_used_for_decision", False)),
            "stale": bool(visual.get("stale", False)),
        },
        "next_action": next_action,
        "policy_note": "Visual evidence review is read-only observability; it does not execute trades, lower thresholds, or bypass strategy, risk, timing, or execution gates.",
    }


def _build_setup_execution_handoff_review(
    status_payload: dict[str, Any],
    setup_hunter: dict[str, Any],
    execution_readiness: dict[str, Any],
) -> dict[str, Any]:
    status_payload = dict(status_payload or {})
    setup_hunter = dict(setup_hunter or {})
    execution_readiness = dict(execution_readiness or {})
    closest = dict(setup_hunter.get("closest_setup", {}) or {})
    primary = dict(execution_readiness.get("primary_blocker", {}) or {})
    chosen = dict(status_payload.get("chosen", {}) or {})
    chosen_side = str(chosen.get("side", "flat") or "flat").lower()
    candidate_side = str(closest.get("side", "neutral") or "neutral").lower()
    candidate_name = str(closest.get("name", "") or "")
    candidate_exists = bool(candidate_name)
    execution_ready = bool(execution_readiness.get("ready", False))
    selected_directional_signal = chosen_side in {"buy", "sell"}
    candidate_is_execution_signal = (
        candidate_exists
        and selected_directional_signal
        and candidate_side == chosen_side
        and execution_ready
    )
    status = "no_candidate"
    headline = "No setup candidate is ready for handoff."
    next_action = "Keep scanning known setup playbooks and raw strategy signals."
    if execution_ready:
        status = "execution_ready"
        headline = "Execution gates can evaluate the selected directional signal."
        next_action = "Let existing execution gates handle the next approved request."
    elif candidate_exists and not selected_directional_signal:
        status = "watched_candidate_not_selected_signal"
        headline = f"Setup Hunter is watching {candidate_name} {candidate_side}, but execution has no selected directional trade signal."
        next_action = "Wait for strategy selection and final gates to convert the watched setup into an executable signal."
    elif candidate_exists:
        status = "candidate_blocked_by_execution_gates"
        headline = f"Setup Hunter is watching {candidate_name} {candidate_side}, but execution gates are still blocking it."
        next_action = str(primary.get("detail", "") or "Wait for the first execution blocker to clear.")
    return {
        "schema": "atlas_setup_execution_handoff_v1",
        "status": status,
        "headline": headline,
        "candidate_exists": candidate_exists,
        "candidate_is_execution_signal": candidate_is_execution_signal,
        "selected_directional_signal": selected_directional_signal,
        "closest_setup": closest,
        "selected_signal": {
            "side": chosen_side,
            "confidence": chosen.get("confidence"),
            "strategy": str(chosen.get("strategy", "") or ""),
        },
        "execution_ready": execution_ready,
        "primary_blocker": primary,
        "next_action": next_action,
        "policy_note": "Setup-to-execution handoff is read-only observability; it does not create signals, lower thresholds, alter strategy, change risk, or execute trades.",
    }


def _build_signal_selection_review(
    status_payload: dict[str, Any],
    setup_hunter: dict[str, Any],
    execution_readiness: dict[str, Any],
) -> dict[str, Any]:
    status_payload = dict(status_payload or {})
    setup_hunter = dict(setup_hunter or {})
    execution_readiness = dict(execution_readiness or {})
    decision_analysis = dict(status_payload.get("decision_analysis", {}) or {})
    chosen = dict(status_payload.get("chosen", {}) or {})
    closest = dict(setup_hunter.get("closest_setup", {}) or {})
    primary = dict(execution_readiness.get("primary_blocker", {}) or {})
    top_watch = dict(dict(decision_analysis.get("setup_radar", {}) or {}).get("top_watch", {}) or {})
    confluence = dict(decision_analysis.get("confluence", {}) or {})
    top_blockers = [dict(item or {}) for item in list(decision_analysis.get("top_blockers", []) or [])[:5]]
    hard_blockers = [str(item) for item in list(decision_analysis.get("hard_blockers", decision_analysis.get("blockers", [])) or [])[:8]]
    chosen_side = str(chosen.get("side", "flat") or "flat").lower()
    candidate_side = str(closest.get("side", "neutral") or "neutral").lower()
    selected_directional = chosen_side in {"buy", "sell"}
    candidate_exists = bool(closest.get("name"))
    missing = [str(item) for item in list(top_watch.get("missing_to_entry", closest.get("missing_to_entry", [])) or [])[:8]]
    candidate_conf = float(closest.get("confidence", 0.0) or 0.0)
    chosen_conf = float(chosen.get("confidence", 0.0) or 0.0)
    threshold = float(dict(status_payload.get("thresholds", {}) or {}).get("confidence", status_payload.get("min_confidence_threshold", 0.0)) or 0.0)
    score = confluence.get("score", decision_analysis.get("confluence_score"))
    score_threshold = confluence.get("threshold", decision_analysis.get("confluence_threshold"))
    status = "selected_signal_ready" if selected_directional and bool(execution_readiness.get("ready", False)) else "not_selected"
    if candidate_exists and not selected_directional:
        status = "raw_candidate_not_selected"
    elif selected_directional and hard_blockers:
        status = "selected_signal_blocked"
    elif not candidate_exists:
        status = "no_candidate"
    reasons: list[str] = []
    if not selected_directional:
        reasons.append("chosen signal is flat")
    if hard_blockers:
        reasons.extend(hard_blockers[:4])
    if missing:
        reasons.extend(missing[:4])
    if score is not None and score_threshold is not None:
        try:
            if float(score) < float(score_threshold):
                reasons.append("confluence_below_threshold")
        except Exception:
            pass
    if candidate_side in {"buy", "sell"} and selected_directional and candidate_side != chosen_side:
        reasons.append("candidate_side_differs_from_selected_signal")
    headline = (
        f"Atlas is watching {closest.get('name')} {candidate_side}, but it has not selected an executable directional signal."
        if status == "raw_candidate_not_selected"
        else (
            "Atlas selected a directional signal, but execution gates are still blocking it."
            if status == "selected_signal_blocked"
            else (
                "No setup candidate is strong enough for selection yet."
                if status == "no_candidate"
                else "The selected signal is ready for normal execution gates."
            )
        )
    )
    return {
        "schema": "atlas_signal_selection_review_v1",
        "status": status,
        "headline": headline,
        "candidate": closest,
        "selected_signal": {
            "side": chosen_side,
            "confidence": round(chosen_conf, 4),
            "strategy": str(chosen.get("strategy", "") or ""),
        },
        "candidate_confidence": round(candidate_conf, 4),
        "confidence_threshold": round(threshold, 4),
        "selected_directional_signal": selected_directional,
        "hard_blockers": hard_blockers,
        "missing_confirmations": missing,
        "top_blockers": top_blockers,
        "confluence": {
            "score": score,
            "threshold": score_threshold,
            "qualified": bool(confluence.get("qualified", False)),
        },
        "primary_execution_blocker": primary,
        "selection_reasons": list(dict.fromkeys(reasons))[:10],
        "next_action": (
            "Wait for the raw candidate to gain setup confirmation before treating it as selected."
            if status == "raw_candidate_not_selected"
            else (
                str(primary.get("detail", "") or "Wait for the first execution blocker to clear.")
                if status == "selected_signal_blocked"
                else "Keep scanning until a setup candidate meets selection and normal gates."
            )
        ),
        "policy_note": "Signal selection review is read-only observability; it does not create signals, lower thresholds, change strategy, alter risk, or execute trades.",
    }


def _build_system_audit_payload(
    status_payload: dict[str, Any],
    runtime_flags: dict[str, Any],
    automation_payload: dict[str, Any],
    *,
    store: KnowledgeStore | None = None,
) -> dict[str, Any]:
    status_payload = dict(status_payload or {})
    runtime_flags = dict(runtime_flags or {})
    automation_payload = dict(automation_payload or {})
    desktop_window = _reconcile_desktop_window_state(dict(runtime_flags.get("ui_desktop_window", {}) or {}), store)
    reconciled_runtime = dict(runtime_flags)
    reconciled_runtime["ui_desktop_window"] = desktop_window
    operator_readiness = _build_operator_readiness(status_payload, reconciled_runtime, automation_payload, desktop_window)
    execution_readiness = _build_execution_readiness(status_payload, reconciled_runtime, automation_payload, operator_readiness)
    setup_hunter = dict(status_payload.get("setup_hunter", {}) or {})
    if not setup_hunter:
        setup_hunter = build_setup_hunter(status=status_payload, decision_analysis=dict(status_payload.get("decision_analysis", {}) or {}), runtime=reconciled_runtime)
    setup_execution_handoff = _build_setup_execution_handoff_review(status_payload, setup_hunter, execution_readiness)
    system_health = dict(status_payload.get("system_health", {}) or {})
    safety_layer = dict(system_health.get("safety_layer", status_payload.get("safety_layer", {})) or {})
    visual_watch = dict(runtime_flags.get("visual_watch", {}) or {})
    visual_evidence_review = _build_visual_evidence_review(status_payload, reconciled_runtime, store)
    volatility_protection_review = _build_volatility_protection_review(status_payload)
    warnings = [str(item) for item in list(system_health.get("warnings", []) or [])]
    issues: list[dict[str, str]] = []
    if not bool(desktop_window.get("ready", desktop_window.get("ok", False))):
        issues.append({"severity": "warning", "area": "ui", "code": "desktop_not_ready", "detail": str(desktop_window.get("failure_reason", "") or "desktop wrapper not ready")})
    if bool(safety_layer.get("safety_lock_active", False)):
        issues.append({"severity": "protective_lock", "area": "risk", "code": "safety_lock_active", "detail": str(safety_layer.get("safety_lock_reason", "") or "safety lock active")})
    if not bool(execution_readiness.get("ready", False)):
        primary = dict(execution_readiness.get("primary_blocker", {}) or {})
        issues.append({"severity": "blocked", "area": str(primary.get("lane", "execution") or "execution"), "code": str(primary.get("code", "execution_not_ready") or "execution_not_ready"), "detail": str(primary.get("detail", "") or "execution not ready")})
    if warnings:
        for warning in warnings[:8]:
            issues.append({"severity": "warning", "area": "system", "code": warning, "detail": warning})
    if bool(visual_evidence_review.get("degraded", False)):
        issues.append({"severity": "warning", "area": "vision", "code": "visual_capture_degraded", "detail": str(visual_evidence_review.get("headline", "") or "visual capture degraded")})
    elif not bool(visual_watch.get("last_capture_ok", False)):
        issues.append({"severity": "warning", "area": "vision", "code": "visual_watch_not_confirmed", "detail": "continuous visual watch has no confirmed recent capture"})
    if bool(volatility_protection_review.get("hard_block", False)) and not any(item.get("code") == "volatility_hard_block" for item in issues):
        issues.append({"severity": "protective_lock", "area": "risk", "code": "volatility_hard_block", "detail": str(volatility_protection_review.get("reason", "") or "volatility protection is blocking new entries")})
    learning_lanes = _build_learning_lanes_status(status_payload, reconciled_runtime, store) if store is not None else {}
    cognitive_state = _build_cognitive_state_payload(status_payload, reconciled_runtime, store) if store is not None else {}
    safety_lock_review = _build_safety_lock_review(status_payload)

    return {
        "ok": not any(str(item.get("severity", "")) in {"blocked"} for item in issues),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "desktop_ready": bool(desktop_window.get("ready", desktop_window.get("ok", False))),
            "execute_trades": bool(runtime_flags.get("execute_trades", status_payload.get("execute_trades", False))),
            "execution_ready": bool(execution_readiness.get("ready", False)),
            "primary_blocker": dict(execution_readiness.get("primary_blocker", {}) or {}),
            "safety_lock_active": bool(safety_layer.get("safety_lock_active", False)),
            "volatility_hard_block": bool(volatility_protection_review.get("hard_block", False)),
            "visual_watch_ready": bool(dict(execution_readiness.get("state", {}) or {}).get("visual_watch_ready", False)),
            "visual_capture_degraded": bool(visual_evidence_review.get("degraded", False)),
            "broker_ready": bool(dict(execution_readiness.get("state", {}) or {}).get("broker_ready", False)),
        },
        "desktop": desktop_window,
        "operator_readiness": operator_readiness,
        "execution_readiness": execution_readiness,
        "learning_lanes": learning_lanes,
        "cognitive_state": cognitive_state,
        "safety_lock_review": safety_lock_review,
        "volatility_protection_review": volatility_protection_review,
        "visual_evidence_review": visual_evidence_review,
        "setup_execution_handoff": setup_execution_handoff,
        "visual_watch": visual_watch,
        "warnings": warnings,
        "issues": issues,
        "upgrade_notes": [
            "Pillow pixel access is compatible with Pillow 14 get_flattened_data.",
            "Dependency upgrades should be staged separately while Atlas is not trading.",
        ],
        "policy_note": "System audit is read-only observability; it does not change strategy, risk, guard, or execution policy.",
    }


def create_ui_app(
    store: KnowledgeStore,
    comparison_root: str | None = None,
    comparison_run_id: str | None = None,
) -> Flask:
    ui_dir = Path(__file__).resolve().parent / "ui"
    app = Flask(__name__, static_folder=str(ui_dir), static_url_path="")
    @app.after_request
    def _force_local_connection_close(response):
        response.headers["Connection"] = "close"
        response.headers["Cache-Control"] = "no-store"
        return response

    run_root = _resolve_comparison_run_root(store, comparison_root=comparison_root, comparison_run_id=comparison_run_id)
    source: UIDashboardSource = ComparisonDashboardSource(store, run_root) if run_root else UIDashboardSource(store)
    execution_state_store_path = str(Path(store.base) / "automation_state.json")
    shared_execution_gate = ExecutionGate(state_store_path=execution_state_store_path)
    ui_settings = load_settings()
    shared_execution_gate.set_broker_adapter(MT5BrokerAdapter(TradeExecutor(ui_settings)))
    assistant = AskAtlasService(source, execution_gate=shared_execution_gate)
    ensure_self_improvement_knowledge(store)
    learning_summary_cache: dict[str, Any] = {"expires_at": 0.0, "payload": {}}
    learning_summary_lock = threading.Lock()

    def _cached_learning_summary(*, max_age_seconds: float = 30.0) -> dict[str, Any]:
        now = time.time()
        with learning_summary_lock:
            if now < float(learning_summary_cache.get("expires_at", 0.0) or 0.0):
                cached = dict(learning_summary_cache.get("payload", {}) or {})
                if cached:
                    return cached
            payload = _build_learning_summary(source)
            payload["cache"] = {
                "cached": False,
                "ttl_seconds": float(max_age_seconds),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            learning_summary_cache["payload"] = payload
            learning_summary_cache["expires_at"] = now + max(1.0, float(max_age_seconds))
            return dict(payload)

    def _warm_digest_cache() -> None:
        try:
            source.load_status()
            source.load_runtime_flags()
            source.load_events(limit=240)
            _cached_learning_summary()
        except Exception:
            _LOGGER.debug("UI digest cache warm-up failed", exc_info=True)

    threading.Thread(target=_warm_digest_cache, name="atlas-ui-digest-warmup", daemon=True).start()

    # Primary backend startup trigger for greeting; no frontend callback required.
    # One-shot semantics are enforced in announce_ui_ready_once().
    try:
        announce_ui_ready_once(store)
    except Exception:
        pass

    @app.get("/")
    def index():
        return send_from_directory(ui_dir, "index.html")

    @app.get("/api/health")
    def health():
        return jsonify({"ok": True, "service": "atlas-ui", "ts": datetime.now(timezone.utc).isoformat()})

    @app.get("/api/status")
    def status():
        payload = dict(source.load_status() or {})
        flags = dict(source.load_runtime_flags() or {})
        settings = load_settings()
        desktop_window = _reconcile_desktop_window_state(dict(flags.get("ui_desktop_window", {}) or {}), store)
        payload["ui_desktop_window"] = desktop_window
        payload["ui_desktop_window_ready"] = bool(desktop_window.get("ready", False))
        payload["freshness_runtime"] = _canonical_freshness_runtime(payload, settings)
        payload["operator_health"] = _build_operator_health_summary(payload, flags, desktop_window)
        payload["volatility_intelligence"] = dict(payload.get("volatility_intelligence", {}) or {})
        payload["volatility_learning_performance"] = dict(payload.get("volatility_learning_performance", {}) or {})
        payload["learning_summary"] = _cached_learning_summary()
        payload["near_setup_learning"] = _load_near_setup_learning_summary(store)
        automation_payload = _load_automation_payload(store)
        execution_snapshot = dict(automation_payload)
        payload["execution"] = execution_snapshot
        payload["execution_truth"] = dict(execution_snapshot.get("execution_truth", {}) or execution_snapshot.get("last_execution_truth", {}) or {})
        payload["automation_truth"] = _automation_truth_from_payload(automation_payload)
        payload["structured_blocker"] = _build_structured_blocker(payload, automation_payload)
        payload["loss_learning_review"] = _build_loss_learning_review(payload, store)
        payload["learning_lanes"] = _build_learning_lanes_status(payload, flags, store)
        payload["cognitive_state"] = _build_cognitive_state_payload(payload, flags, store)
        payload["safety_lock_review"] = _build_safety_lock_review(payload)
        payload["volatility_protection_review"] = _build_volatility_protection_review(payload)
        payload["visual_evidence_review"] = _build_visual_evidence_review(payload, flags, store)
        reconciled_flags = dict(flags)
        reconciled_flags["ui_desktop_window"] = desktop_window
        payload["operator_readiness"] = _build_operator_readiness(payload, reconciled_flags, automation_payload, desktop_window)
        payload["execution_readiness"] = _build_execution_readiness(
            payload,
            reconciled_flags,
            automation_payload,
            payload["operator_readiness"],
        )
        setup_hunter = dict(payload.get("setup_hunter", {}) or {})
        if not setup_hunter:
            setup_hunter = build_setup_hunter(status=payload, decision_analysis=dict(payload.get("decision_analysis", {}) or {}), runtime=reconciled_flags)
            payload["setup_hunter"] = setup_hunter
        payload["setup_execution_handoff"] = _build_setup_execution_handoff_review(
            payload,
            setup_hunter,
            payload["execution_readiness"],
        )
        payload["signal_selection_review"] = _build_signal_selection_review(
            payload,
            setup_hunter,
            payload["execution_readiness"],
        )
        return jsonify(payload)

    @app.get("/api/status/compact")
    def status_compact():
        return jsonify(_build_compact_status_payload(source.load_status(), source.load_runtime_flags(), store))

    @app.get("/api/status/full")
    def status_full():
        payload = dict(source.load_status() or {})
        flags = dict(source.load_runtime_flags() or {})
        settings = load_settings()
        _refresh_macro_context_if_needed(store)
        self_improvement_bundle = load_self_improvement_bundle(store)
        payload["learning_summary"] = _cached_learning_summary()
        payload["near_setup_learning"] = _load_near_setup_learning_summary(store)
        payload["self_improvement_state"] = dict(payload.get("self_improvement_state", {}) or build_self_improvement_summary(self_improvement_bundle))
        desktop_window = _reconcile_desktop_window_state(dict(flags.get("ui_desktop_window", {}) or {}), store)
        payload["ui_desktop_window"] = desktop_window
        payload["ui_desktop_window_ready"] = bool(desktop_window.get("ready", False))
        payload["freshness_runtime"] = _canonical_freshness_runtime(payload, settings)
        payload["operator_health"] = _build_operator_health_summary(payload, flags, desktop_window)
        payload["volatility_intelligence"] = dict(payload.get("volatility_intelligence", {}) or {})
        payload["volatility_learning_performance"] = dict(payload.get("volatility_learning_performance", {}) or {})
        try:
            assistant.execution.refresh_critical_controls_from_persisted_state()
        except Exception:
            pass
        execution_snapshot = assistant.execution_snapshot()
        payload["execution"] = execution_snapshot
        payload["execution_truth"] = dict(execution_snapshot.get("execution_truth", {}) or execution_snapshot.get("last_execution_truth", {}) or {})
        payload["automation_truth"] = {
            "automation_enabled": bool(execution_snapshot.get("automation_enabled", False)),
            "automation_paused": bool(execution_snapshot.get("automation_paused", False)),
            "automation_pause_reason": str(execution_snapshot.get("automation_pause_reason", "") or ""),
            "automation_explicit_resume_active": bool(execution_snapshot.get("automation_explicit_resume_active", False)),
            "last_auto_execution_result": dict(execution_snapshot.get("last_auto_execution_result", {}) or {}),
            "last_auto_execution_block_reason": str(execution_snapshot.get("last_auto_execution_block_reason", "") or ""),
        }
        payload["structured_blocker"] = _build_structured_blocker(payload, execution_snapshot)
        payload["loss_learning_review"] = _build_loss_learning_review(payload, store)
        payload["learning_lanes"] = _build_learning_lanes_status(payload, flags, store)
        payload["cognitive_state"] = _build_cognitive_state_payload(payload, flags, store)
        payload["safety_lock_review"] = _build_safety_lock_review(payload)
        payload["volatility_protection_review"] = _build_volatility_protection_review(payload)
        payload["visual_evidence_review"] = _build_visual_evidence_review(payload, flags, store)
        reconciled_flags = dict(flags)
        reconciled_flags["ui_desktop_window"] = desktop_window
        payload["operator_readiness"] = _build_operator_readiness(payload, reconciled_flags, execution_snapshot, desktop_window)
        payload["execution_readiness"] = _build_execution_readiness(
            payload,
            reconciled_flags,
            execution_snapshot,
            payload["operator_readiness"],
        )
        setup_hunter = dict(payload.get("setup_hunter", {}) or {})
        if not setup_hunter:
            setup_hunter = build_setup_hunter(status=payload, decision_analysis=dict(payload.get("decision_analysis", {}) or {}), runtime=reconciled_flags)
            payload["setup_hunter"] = setup_hunter
        payload["setup_execution_handoff"] = _build_setup_execution_handoff_review(
            payload,
            setup_hunter,
            payload["execution_readiness"],
        )
        payload["signal_selection_review"] = _build_signal_selection_review(
            payload,
            setup_hunter,
            payload["execution_readiness"],
        )
        return jsonify(payload)

    @app.get("/api/why")
    def why():
        payload = dict(source.load_status() or {})
        flags = dict(source.load_runtime_flags() or {})
        execution_snapshot = assistant.execution_snapshot()
        return jsonify({"ok": True, "why": build_why_trace(payload, flags, execution_snapshot)})

    @app.get("/api/events")
    def events():
        return jsonify([_compact_event_for_ui(row) for row in source.load_events(limit=40)])

    @app.get("/api/blocker-summary")
    def blocker_summary():
        status_payload = dict(source.load_status() or {})
        near_setup_history = _load_near_setup_learning_summary(store)
        payload = _build_blocker_summary(status_payload, near_setup_history)
        payload["structured_blocker"] = _build_structured_blocker(status_payload, _load_automation_payload(store))
        return jsonify(payload)

    @app.get("/api/operator-readiness")
    def operator_readiness():
        status_payload = dict(source.load_status() or {})
        runtime_payload = dict(source.load_runtime_flags() or {})
        automation_payload = _load_automation_payload(store)
        desktop_window = _reconcile_desktop_window_state(dict(runtime_payload.get("ui_desktop_window", {}) or {}), store)
        runtime_payload["ui_desktop_window"] = desktop_window
        readiness = _build_operator_readiness(status_payload, runtime_payload, automation_payload, desktop_window)
        readiness["execution_readiness"] = _build_execution_readiness(
            status_payload,
            runtime_payload,
            automation_payload,
            readiness,
        )
        return jsonify(readiness)

    @app.get("/api/system-audit")
    def system_audit():
        status_payload = dict(source.load_status() or {})
        runtime_payload = dict(source.load_runtime_flags() or {})
        automation_payload = _load_automation_payload(store)
        return jsonify(
            _build_system_audit_payload(
                status_payload,
                runtime_payload,
                automation_payload,
                store=store,
            )
        )

    @app.get("/api/loss-learning-review")
    def loss_learning_review():
        return jsonify(_build_loss_learning_review(dict(source.load_status() or {}), store))

    @app.get("/api/learning-lanes")
    def learning_lanes():
        return jsonify(
            _build_learning_lanes_status(
                dict(source.load_status() or {}),
                dict(source.load_runtime_flags() or {}),
                store,
            )
        )

    @app.get("/api/cognitive-state")
    def cognitive_state():
        return jsonify(
            _build_cognitive_state_payload(
                dict(source.load_status() or {}),
                dict(source.load_runtime_flags() or {}),
                store,
            )
        )

    @app.get("/api/safety-lock-review")
    def safety_lock_review():
        return jsonify(_build_safety_lock_review(dict(source.load_status() or {})))

    @app.get("/api/volatility-protection-review")
    def volatility_protection_review():
        return jsonify(_build_volatility_protection_review(dict(source.load_status() or {})))

    @app.get("/api/signal-selection-review")
    def signal_selection_review():
        status_payload = dict(source.load_status() or {})
        runtime_payload = dict(source.load_runtime_flags() or {})
        automation_payload = _load_automation_payload(store)
        desktop_window = _reconcile_desktop_window_state(dict(runtime_payload.get("ui_desktop_window", {}) or {}), store)
        runtime_payload["ui_desktop_window"] = desktop_window
        setup_hunter = dict(status_payload.get("setup_hunter", {}) or {})
        if not setup_hunter:
            setup_hunter = build_setup_hunter(
                status=status_payload,
                decision_analysis=dict(status_payload.get("decision_analysis", {}) or {}),
                runtime=runtime_payload,
            )
        operator_readiness = _build_operator_readiness(status_payload, runtime_payload, automation_payload, desktop_window)
        execution_readiness = _build_execution_readiness(status_payload, runtime_payload, automation_payload, operator_readiness)
        return jsonify(_build_signal_selection_review(status_payload, setup_hunter, execution_readiness))

    @app.get("/api/reports/signal-correlation")
    def signal_correlation_report():
        try:
            window = max(2, int(request.args.get("window", "500")))
        except Exception:
            window = 500
        try:
            threshold = float(request.args.get("threshold", "0.8"))
        except Exception:
            threshold = 0.8
        threshold = max(0.0, min(1.0, threshold))
        report = build_signal_correlation_report_from_store(
            store,
            rolling_window=window,
            threshold=threshold,
            write_files=True,
        )
        return jsonify({"ok": True, "report": report})

    @app.get("/api/reports/fast-path-audit")
    def fast_path_audit_report():
        try:
            window_hours = max(0.0, float(request.args.get("window_hours", "168")))
        except Exception:
            window_hours = 168.0
        report = build_fast_path_promotion_audit_report(
            store,
            window_hours=window_hours,
            write_file=True,
        )
        return jsonify({"ok": True, "report": report})

    @app.get("/api/visual-evidence-review")
    def visual_evidence_review():
        return jsonify(
            _build_visual_evidence_review(
                dict(source.load_status() or {}),
                dict(source.load_runtime_flags() or {}),
                store,
            )
        )

    @app.get("/api/vision/latest")
    def vision_latest():
        latest = load_latest_visual_read(Path(store.base) / "chart_vision")
        if not bool(latest.get("available", False)):
            latest = {
                "enabled": True,
                "advisory_only": True,
                "available": False,
                "source": "live_terminal_required",
                "status": "waiting_for_live_terminal_capture",
                "visual_confirmation": "unavailable",
                "confidence_effect": "no_visual_evidence",
                "reasons": ["live MT5/HFM terminal screenshot is required for visual chart evidence"],
                "policy_note": "Visual chart reading must use the live terminal screenshot; candle-derived chart features remain separate.",
            }
        return jsonify(latest)

    @app.post("/api/vision/capture")
    def vision_capture():
        status_payload = dict(source.load_status() or {})
        capture = capture_desktop_screenshot(Path(store.base) / "chart_vision")
        image_path = str(capture.get("path", "") or "") if bool(capture.get("ok", False)) else ""
        visual = analyze_chart_visual_context(
            status=status_payload,
            chart_features=dict(status_payload.get("chart_features", {}) or {}),
            image_path=image_path or None,
        )
        visual["capture"] = capture
        saved = save_visual_read(visual, Path(store.base) / "chart_vision")
        return jsonify({"ok": True, "visual_chart_read": saved})

    @app.get("/api/setup-radar/history")
    def setup_radar_history():
        return jsonify(_load_near_setup_learning_summary(store))

    @app.get("/api/allocator")
    def allocator():
        state = source.load_learning_state().get("state", {}).get("allocator", {})
        return jsonify(state)

    @app.get("/api/learning-insights")
    def learning_insights():
        state = source.load_learning_state().get("state", {})
        learning_summary = _cached_learning_summary()
        loss_memory = dict(state.get("loss_memory", {}))
        outcomes = list(loss_memory.get("recent_outcomes", []))
        recent = outcomes[-20:]
        live_events = list(source.load_events(limit=220))
        live_recent = live_events[-60:]

        live_exec = [e for e in live_recent if str(e.get("type", "")).lower() in {"execution", "failed"}]
        live_blocks = [e for e in live_recent if str(e.get("type", "")).lower() == "risk"]
        live_learning = [e for e in live_recent if str(e.get("type", "")).lower() == "learning"]
        drift_hits = [
            e
            for e in live_recent
            if "performance drift detected" in str(e.get("message", "")).lower()
        ]

        live_wins = sum(
            1
            for e in live_exec
            if bool((e.get("meta", {}) or {}).get("execution", {}).get("success", False))
        )
        live_fails = sum(
            1
            for e in live_exec
            if not bool((e.get("meta", {}) or {}).get("execution", {}).get("success", False))
        )
        live_attempts = live_wins + live_fails
        live_success_rate = (live_wins / live_attempts) if live_attempts else 0.0

        if not recent:
            summary = "Live learning active from current session events; closed-trade memory is still building."
            live_insights = [
                f"Live attempts: {live_attempts}, success: {live_wins}, failed: {live_fails}, success rate: {live_success_rate * 100:.1f}%.",
                f"Recent risk blocks: {len(live_blocks)}. Drift alerts: {len(drift_hits)}.",
                f"Learning updates this session: {len(live_learning)} event(s).",
            ]
            return jsonify(
                {
                    "sample_size": 0,
                    "live_sample_size": live_attempts,
                    "summary": summary,
                    "insights": live_insights,
                    "last_confirmed_trade": learning_summary.get("last_confirmed_trade", {}),
                    "learning_summary": learning_summary,
                }
            )

        wins = sum(1 for x in recent if str(x.get("outcome", "")) == "win")
        losses = sum(1 for x in recent if str(x.get("outcome", "")) == "loss")
        draws = sum(1 for x in recent if str(x.get("outcome", "")) == "draw")
        pnl = round(sum(float(x.get("net_pnl_zar", 0.0) or 0.0) for x in recent), 2)
        sample = len(recent)
        win_rate = wins / max(1, (wins + losses))

        by_side: dict[str, dict[str, float]] = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0.0, "n": 0})
        by_session: dict[str, dict[str, float]] = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0.0, "n": 0})
        for x in recent:
            side = str(x.get("side", "flat"))
            ses = str(x.get("session", "unknown"))
            out = str(x.get("outcome", "draw"))
            p = float(x.get("net_pnl_zar", 0.0) or 0.0)

            by_side[side]["n"] += 1
            by_side[side]["pnl"] += p
            by_session[ses]["n"] += 1
            by_session[ses]["pnl"] += p

            if out == "win":
                by_side[side]["wins"] += 1
                by_session[ses]["wins"] += 1
            elif out == "loss":
                by_side[side]["losses"] += 1
                by_session[ses]["losses"] += 1

        side_scores = []
        for side, st in by_side.items():
            wl = st["wins"] + st["losses"]
            wr = (st["wins"] / wl) if wl else 0.0
            side_scores.append((wr, st["pnl"], int(st["n"]), side))
        side_scores.sort(reverse=True)

        session_scores = []
        for ses, st in by_session.items():
            wl = st["wins"] + st["losses"]
            wr = (st["wins"] / wl) if wl else 0.0
            session_scores.append((wr, st["pnl"], int(st["n"]), ses))
        session_scores.sort(reverse=True)

        insights: list[str] = []
        insights.append(
            f"Last {sample} closed trades: {wins} wins, {losses} losses, {draws} draws. Net PnL {pnl:+.2f}. Win rate {win_rate * 100:.1f}%."
        )
        if side_scores:
            best = side_scores[0]
            insights.append(
                f"Best side recently: {best[3]} with {best[2]} samples and net PnL {best[1]:+.2f}."
            )
            if len(side_scores) > 1:
                worst = side_scores[-1]
                insights.append(
                    f"Weak side recently: {worst[3]} with {worst[2]} samples and net PnL {worst[1]:+.2f}. System should be stricter here."
                )
        if session_scores:
            best_s = session_scores[0]
            insights.append(
                f"Best session recently: {best_s[3]} (samples {best_s[2]}, net PnL {best_s[1]:+.2f})."
            )
            if len(session_scores) > 1:
                weak_s = session_scores[-1]
                insights.append(
                    f"Weak session recently: {weak_s[3]} (samples {weak_s[2]}, net PnL {weak_s[1]:+.2f})."
                )
        if losses > wins:
            insights.append("Learning mode: losses are dominating; system should prioritize quality over trade frequency.")
        else:
            insights.append("Learning mode: quality improving; keep risk caps unchanged while scaling only on consistent edge.")

        insights.insert(
            1,
            (
                f"Live session: attempts {live_attempts}, success {live_wins}, failed {live_fails}, "
                f"success rate {live_success_rate * 100:.1f}%."
            ),
        )
        if drift_hits:
            insights.append(f"Drift monitor triggered {len(drift_hits)} time(s) recently.")
        if live_learning:
            insights.append(f"Learning events recorded recently: {len(live_learning)}.")

        return jsonify(
            {
                "sample_size": sample,
                "live_sample_size": live_attempts,
                "summary": "Live + closed-trade learning summary.",
                "insights": insights[:6],
                "last_confirmed_trade": learning_summary.get("last_confirmed_trade", {}),
                "learning_summary": learning_summary,
            }
        )

    @app.get("/api/teaching/reviews")
    def teaching_reviews():
        payload = store.load_structured_knowledge_file(
            "operator_teaching_reviews.json",
            {"schema_version": 1, "reviews": []},
        )
        reviews = list(dict(payload or {}).get("reviews", []) or [])
        return jsonify(
            {
                "schema_version": 1,
                "count": len(reviews),
                "reviews": reviews[-120:],
                "digest": build_operator_teaching_digest(store),
            }
        )

    @app.post("/api/teaching/reviews")
    def record_teaching_review():
        if source.read_only:
            return jsonify({"ok": False, "error": "UI is read-only while viewing an active comparison run."}), 409
        payload = request.get_json(silent=True) or {}
        label = str(payload.get("label", "") or "").strip().lower()
        allowed_labels = {
            "correct_hold",
            "too_cautious",
            "bad_setup",
            "missed_trade",
            "good_trade",
            "bad_trade",
            "execution_issue",
            "confusing_answer",
        }
        if label not in allowed_labels:
            return jsonify({"ok": False, "error": "invalid_label", "allowed_labels": sorted(allowed_labels)}), 400

        status_payload = dict(source.load_status() or {})
        runtime_payload = dict(source.load_runtime_flags() or {})
        analysis = dict(status_payload.get("decision_analysis", {}) or {})
        playbook = dict(status_payload.get("setup_playbook", analysis.get("setup_playbook", {})) or {})
        top_playbook = dict(playbook.get("top_candidate", {}) or {})
        setup_radar = dict(analysis.get("setup_radar", {}) or {})
        top_watch = dict(setup_radar.get("top_watch", {}) or setup_radar.get("top_candidate", {}) or {})
        chosen = dict(status_payload.get("chosen", {}) or {})
        visual = dict(status_payload.get("visual_chart_read", {}) or {})
        visual_capture = dict(visual.get("capture", {}) or {})
        market_regime = dict(analysis.get("market_regime", {}) or {})
        trend = dict(analysis.get("trend", {}) or {})
        momentum = dict(analysis.get("momentum", {}) or {})
        location = dict(analysis.get("location", {}) or {})
        context_alignment = dict(analysis.get("context_alignment", {}) or {})
        execution_friction = dict(dict(analysis.get("macro_context", {}) or {}).get("execution_friction", {}) or {})
        if not top_playbook and top_watch:
            top_playbook = top_watch
        context_snapshot = {
            "symbol": str(status_payload.get("symbol", "") or ""),
            "timeframe": str(status_payload.get("timeframe", "") or ""),
            "session": str(dict(context_alignment.get("session", {}) or {}).get("state", dict(status_payload.get("system_health", {}) or {}).get("session_state", "")) or ""),
            "market_regime": str(market_regime.get("regime", market_regime.get("state", "")) or ""),
            "regime_confidence": market_regime.get("regime_confidence", market_regime.get("score")),
            "trend_state": str(trend.get("state", "") or ""),
            "trend_structure": str(trend.get("structure", "") or ""),
            "location_state": str(location.get("state", "") or ""),
            "momentum_state": str(momentum.get("state", "") or ""),
            "higher_timeframe_bias": str(analysis.get("higher_timeframe_bias", dict(analysis.get("bias_breakdown", {}) or {}).get("higher_timeframe_bias", "")) or ""),
            "final_trade_bias": str(analysis.get("final_trade_bias", dict(analysis.get("bias_breakdown", {}) or {}).get("final_trade_bias", "")) or ""),
            "selected_signal_side": str(chosen.get("side", "") or ""),
            "selected_signal_strategy": str(chosen.get("strategy", "") or ""),
            "selected_signal_confidence": chosen.get("confidence"),
            "top_playbook": str(top_playbook.get("name", "") or ""),
            "top_playbook_side": str(top_playbook.get("side", "") or ""),
            "top_playbook_readiness": str(top_playbook.get("readiness", "") or ""),
            "top_playbook_trigger_state": str(top_playbook.get("trigger_state", top_playbook.get("phase", "")) or ""),
            "missing_to_entry": [str(item) for item in list(top_playbook.get("missing_to_entry", top_playbook.get("missing", [])) or [])[:8]],
            "setup_risks": [str(item) for item in list(top_playbook.get("risks", top_playbook.get("risk_notes", [])) or [])[:8]],
            "hard_blockers": [str(item) for item in list(analysis.get("hard_blockers", analysis.get("blockers", [])) or [])[:8]],
            "top_blockers": [dict(item or {}) for item in list(analysis.get("top_blockers", []) or [])[:5]],
            "visual_source": str(visual.get("source", "") or ""),
            "visual_confirmation": str(visual.get("visual_confirmation", "") or ""),
            "visual_pattern_seen": str(visual.get("pattern_seen", "") or ""),
            "visual_setup_quality": visual.get("setup_quality"),
            "visual_capture_ok": bool(visual_capture.get("ok", False)),
            "visual_capture_source": str(visual_capture.get("capture_source", "") or ""),
            "spread_state": str(execution_friction.get("spread_state", execution_friction.get("spread_status", "")) or ""),
            "spread_points": execution_friction.get("spread_points"),
            "liquidity_state": str(execution_friction.get("liquidity_state", "") or ""),
            "slippage_state": str(execution_friction.get("slippage_state", "") or ""),
            "volatility_state": str(dict(status_payload.get("safety_layer", {}) or {}).get("volatility_state", dict(dict(status_payload.get("system_health", {}) or {}).get("safety_layer", {}) or {}).get("volatility_state", "")) or ""),
        }
        context_quality = {
            "decision_context_linked": True,
            "location_known": bool(context_snapshot["location_state"] and context_snapshot["location_state"] != "unknown"),
            "momentum_known": bool(context_snapshot["momentum_state"] and context_snapshot["momentum_state"] != "unknown"),
            "volatility_known": bool(context_snapshot["volatility_state"] and context_snapshot["volatility_state"] != "unknown"),
            "visual_context_known": bool(context_snapshot["visual_source"] or context_snapshot["visual_confirmation"]),
            "selected_signal_known": bool(context_snapshot["selected_signal_side"]),
            "setup_context_known": bool(context_snapshot["top_playbook"]),
        }
        review = {
            "review_id": f"teach_{int(time.time() * 1000)}",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "label": label,
            "note": str(payload.get("note", "") or "").strip()[:1000],
            "operator": str(payload.get("operator", "") or "operator").strip()[:80],
            "decision_id": str(
                payload.get("decision_id")
                or dict(status_payload.get("decision_explanation", {}) or {}).get("candidate_id")
                or dict(status_payload.get("decision_trace", {}) or {}).get("decision_id")
                or ""
            ),
            "status_action": str(status_payload.get("action", "") or ""),
            "primary_blocker": str(status_payload.get("primary_blocker", status_payload.get("message", "")) or ""),
            "setup_family": str(status_payload.get("setup_family", "") or ""),
            "playbook_name": str(top_playbook.get("name", "") or ""),
            "playbook_score": float(top_playbook.get("score", 0.0) or 0.0),
            "playbook_readiness": str(top_playbook.get("readiness", "") or ""),
            "trade_decision": str(analysis.get("trade_decision", "") or ""),
            "recommended_action": str(analysis.get("recommended_action", "") or ""),
            "confluence_score": int(analysis.get("confluence_score", 0) or 0),
            "confluence_threshold": int(analysis.get("confluence_threshold", 0) or 0),
            "decision_context_snapshot": context_snapshot,
            "context_quality": context_quality,
            "risk_snapshot": {
                "spread_points": float(dict(status_payload.get("risk", {}) or {}).get("spread_points", 0.0) or 0.0),
                "open_positions": int(dict(status_payload.get("risk", {}) or {}).get("open_positions", 0) or 0),
                "daily_loss_pct": float(dict(status_payload.get("risk", {}) or {}).get("daily_loss_pct", 0.0) or 0.0),
                "weekly_loss_pct": float(dict(status_payload.get("risk", {}) or {}).get("weekly_loss_pct", 0.0) or 0.0),
            },
            "mode_snapshot": {
                "effective_mode": str(dict(status_payload.get("system_health", {}) or {}).get("effective_mode", "") or ""),
                "execute_trades": bool(runtime_payload.get("execute_trades", status_payload.get("execute_trades", False))),
            },
            "policy_note": "Operator teaching label only. No automatic strategy, risk, or execution policy change.",
        }
        teaching = dict(
            store.load_structured_knowledge_file(
                "operator_teaching_reviews.json",
                {"schema_version": 1, "reviews": []},
            )
            or {}
        )
        reviews = list(teaching.get("reviews", []) or [])
        reviews.append(review)
        teaching = {
            "schema_version": 1,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "reviews": reviews[-2000:],
        }
        store.save_structured_knowledge_file("operator_teaching_reviews.json", teaching)
        source.log_event(
            "learning",
            f"Operator teaching review recorded: {label}",
            {
                "review_id": review["review_id"],
                "label": label,
                "decision_id": review["decision_id"],
                "playbook_name": review["playbook_name"],
                "policy_note": review["policy_note"],
            },
        )
        return jsonify({"ok": True, "review": review, "count": len(teaching["reviews"])})

    @app.get("/api/teaching/digest")
    def teaching_digest():
        status_payload = dict(source.load_status() or {})
        digest = build_operator_teaching_digest(store)
        digest["current_decision_prompt"] = build_current_decision_teaching_prompt(status_payload)
        return jsonify(digest)

    @app.get("/api/teaching/current-lesson")
    def teaching_current_lesson():
        status_payload = dict(source.load_status() or {})
        return jsonify(build_structured_lesson_from_status(status_payload))

    @app.get("/api/runtime")
    def runtime_flags():
        flags = dict(source.load_runtime_flags() or {})
        status_payload = dict(source.load_status() or {})
        settings = load_settings()
        self_improvement_bundle = load_self_improvement_bundle(store)
        desktop_window = _reconcile_desktop_window_state(dict(flags.get("ui_desktop_window", {}) or {}), store)
        classification = dict(desktop_window.get("classification", {}) or {})
        desktop_disabled = bool(desktop_window.get("disabled_due_to_repeated_failures", False))
        fallback_url = str(desktop_window.get("fallback_url", "") or "")
        operator_action = str(classification.get("operator_action", "") or "")
        if desktop_disabled and fallback_url:
            operator_action = (
                "Desktop host disabled due to repeated failures. "
                f"Use browser fallback: {fallback_url}"
            )
        flags["ui_desktop_window"] = desktop_window
        flags["ui_desktop_operator_status"] = {
            "desktop_available": bool(desktop_window.get("ok", False)),
            "stage": str(desktop_window.get("stage", "") or ""),
            "failure_code": str(classification.get("code", "") or ""),
            "cause": str(classification.get("operator_cause", "") or ""),
            "action": operator_action,
            "disabled_due_to_repeated_failures": desktop_disabled,
            "failure_count": int(desktop_window.get("failure_count", 0) or 0),
            "fallback_active": bool(desktop_window.get("fallback_active", False)),
            "fallback_url": fallback_url,
            "updated_at": str(desktop_window.get("ts", "") or desktop_window.get("updated_at", "") or ""),
        }
        freshness = _canonical_freshness_runtime(status_payload, settings)
        # Preserve explicit runtime freshness overrides (e.g., tests injecting stale)
        runtime_freshness = dict(flags.get("freshness_runtime", {}) or {})
        merged_freshness = dict(freshness)
        if "is_stale" in runtime_freshness:
            merged_freshness["is_stale"] = bool(runtime_freshness["is_stale"])
        if "source" in runtime_freshness:
            merged_freshness["source"] = str(runtime_freshness["source"])
        elif "is_stale" in runtime_freshness:
            merged_freshness["source"] = "runtime_flags"
        flags["freshness_runtime"] = merged_freshness
        flags["max_freshness_age_ms"] = float(merged_freshness.get("max_freshness_age_ms", 0.0) or 0.0)
        flags["self_improvement"] = self_improvement_bundle
        flags["self_improvement_state"] = build_self_improvement_summary(self_improvement_bundle)
        flags["self_improvement_runtime"] = build_self_improvement_runtime_guard(store)
        # Evolution scheduler state
        flags["evolution"] = _load_evolution_scheduler_state(store)
        flags["learning_summary"] = _cached_learning_summary()
        flags["near_setup_learning"] = _load_near_setup_learning_summary(store)
        if not isinstance(flags.get("startup_verification"), dict):
            status_startup_verification = dict(status_payload.get("guarded_demo_startup_sanity", {}) or {})
            if status_startup_verification:
                flags["startup_verification"] = status_startup_verification
        flags.update({"execution": assistant.execution_snapshot()})
        flags["decision_analysis"] = build_structured_decision_analysis(status=status_payload, runtime=flags)
        flags["guarded_demo_workflow"] = assistant.guarded_demo_workflow_snapshot()
        flags["operator_health"] = _build_operator_health_summary(status_payload, flags, desktop_window)
        flags["operator_readiness"] = _build_operator_readiness(
            status_payload,
            flags,
            dict(flags.get("execution", {}) or {}),
            desktop_window,
        )
        flags["execution_readiness"] = _build_execution_readiness(
            status_payload,
            flags,
            dict(flags.get("execution", {}) or {}),
            flags["operator_readiness"],
        )
        flags["volatility_intelligence"] = dict(status_payload.get("volatility_intelligence", {}) or {})
        flags["volatility_learning_performance"] = dict(status_payload.get("volatility_learning_performance", {}) or {})
        return jsonify(flags)

    @app.get("/api/diagnostics")
    def diagnostics():
        status = source.load_status()
        return jsonify(
            {
                "context_snapshot": dict(status.get("context_snapshot", {}) or {}),
                "decision_explanation": dict(status.get("decision_explanation", {}) or {}),
                "decision_trace": dict(status.get("decision_trace", {}) or {}),
                "system_health": dict(status.get("system_health", {}) or {}),
                "shadow_advice": dict(status.get("shadow_advice", {}) or {}),
                "comparison": dict(status.get("comparison", {}) or {}),
                "learning_summary": _cached_learning_summary(),
                "near_setup_learning": _load_near_setup_learning_summary(store),
            }
        )

    @app.get("/api/self-improvement/policy")
    def self_improvement_policy():
        return jsonify(dict(load_self_improvement_bundle(store).get("self_improvement_policy.json", {}) or {}))

    @app.get("/api/self-improvement/proposals")
    def self_improvement_proposals():
        # Reconcile stale pending approvals before returning queue state.
        reconcile_proposal_queue_with_rollouts(store)
        bundle = load_self_improvement_bundle(store)
        payload = dict(bundle.get("proposal_queue.json", {}) or {})
        payload["summary"] = build_self_improvement_summary(bundle)
        return jsonify(payload)

    @app.get("/api/self-improvement/insights")
    def self_improvement_insights():
        payload = dict(load_self_improvement_bundle(store).get("learned_insights.json", {}) or {})
        payload["summary"] = build_self_improvement_summary(load_self_improvement_bundle(store))
        return jsonify(payload)

    @app.get("/api/self-improvement/rollouts")
    def self_improvement_rollouts():
        payload = dict(load_self_improvement_bundle(store).get("rollout_registry.json", {}) or {})
        payload["summary"] = build_self_improvement_summary(load_self_improvement_bundle(store))
        return jsonify(payload)

    @app.get("/api/self-improvement/audit")
    def self_improvement_audit():
        return jsonify(store.tail_self_improvement_audit(limit=120))

    @app.get("/api/self-improvement/validation/<proposal_id>")
    def self_improvement_validation(proposal_id: str):
        return jsonify(build_self_improvement_validation_report(store, proposal_id))

    @app.get("/api/self-improvement/proposals/<proposal_id>/operator-review")
    def self_improvement_operator_review_packet(proposal_id: str):
        return jsonify(build_consensus_selectivity_operator_review_packet(store, proposal_id=proposal_id))

    @app.post("/api/self-improvement/proposals/<proposal_id>/validate")
    def self_improvement_validate_proposal(proposal_id: str):
        if source.read_only:
            return jsonify({"ok": False, "error": "UI is read-only while viewing an active comparison run."}), 409
        report = build_self_improvement_validation_report(store, proposal_id)
        if not bool(report.get("ok", True)):
            return jsonify({"ok": False, "error": str(report.get("reason", "validation_failed") or "validation_failed")}), 404
        result = record_proposal_validation(store, proposal_id, report, actor="operator")
        if not bool(result.get("ok", False)):
            return jsonify(result), 404
        return jsonify(
            {
                "ok": True,
                "proposal": dict(result.get("proposal", {}) or {}),
                "report": report,
            }
        )

    @app.post("/api/self-improvement/proposals/<proposal_id>/approve")
    def self_improvement_approve_proposal(proposal_id: str):
        if source.read_only:
            return jsonify({"ok": False, "error": "UI is read-only while viewing an active comparison run."}), 409
        payload = request.get_json(silent=True) or {}
        approved_by = str(payload.get("approved_by", "") or "operator")
        note = str(payload.get("note", "") or "")
        confirmation = str(payload.get("confirmation", "") or "").strip()
        expected = f"approve {proposal_id}"
        if confirmation != expected:
            return jsonify(
                {
                    "ok": False,
                    "error": "confirmation_required",
                    "confirmation_required": expected,
                    "message": f"Approval requires confirmation: {expected}",
                }
            ), 400
        result = approve_proposal_for_rollout(store, proposal_id, approved_by=approved_by, note=note, actor="operator")
        if not bool(result.get("ok", False)):
            return jsonify(result), 409
        return jsonify(result)

    @app.post("/api/self-improvement/proposals/<proposal_id>/close")
    def self_improvement_close_proposal(proposal_id: str):
        if source.read_only:
            return jsonify({"ok": False, "error": "UI is read-only while viewing an active comparison run."}), 409
        payload = request.get_json(silent=True) or {}
        target_status = str(payload.get("target_status", "") or "").strip().lower()
        if target_status not in {"deferred", "rejected"}:
            return jsonify({"ok": False, "error": "unsupported_close_status", "target_status": target_status}), 400
        command = "defer" if target_status == "deferred" else "reject"
        confirmation = str(payload.get("confirmation", "") or "").strip()
        expected = f"{command} {proposal_id}"
        if confirmation != expected:
            return jsonify(
                {
                    "ok": False,
                    "error": "confirmation_required",
                    "confirmation_required": expected,
                    "message": f"Proposal close requires confirmation: {expected}",
                }
            ), 400
        result = close_proposal_without_rollout(
            store,
            proposal_id,
            target_status=target_status,
            actor=str(payload.get("actor", "") or "operator"),
            reason=str(payload.get("reason", "") or f"operator_{target_status}"),
            note=str(payload.get("note", "") or ""),
        )
        if not bool(result.get("ok", False)):
            return jsonify(result), 409
        return jsonify(result)

    @app.post("/api/self-improvement/rollouts/<rollout_id>/transition")
    def self_improvement_transition_rollout(rollout_id: str):
        if source.read_only:
            return jsonify({"ok": False, "error": "UI is read-only while viewing an active comparison run."}), 409
        payload = request.get_json(silent=True) or {}
        target_status = str(payload.get("target_status", "") or "").strip()
        if not target_status:
            return jsonify({"ok": False, "error": "target_status_required"}), 400
        confirmation = str(payload.get("confirmation", "") or "").strip()
        expected = f"transition {rollout_id} {target_status}"
        if confirmation != expected:
            return jsonify(
                {
                    "ok": False,
                    "error": "confirmation_required",
                    "confirmation_required": expected,
                    "message": f"Rollout transition requires confirmation: {expected}",
                }
            ), 400
        result = transition_rollout_status(
            store,
            rollout_id,
            target_status=target_status,
            actor="operator",
            reason=str(payload.get("reason", "") or ""),
            note=str(payload.get("note", "") or ""),
        )
        if not bool(result.get("ok", False)):
            return jsonify(result), 409
        return jsonify(result)

    @app.post("/api/self-improvement/proposals/<proposal_id>/promote")
    def self_improvement_promote_proposal(proposal_id: str):
        if source.read_only:
            return jsonify({"ok": False, "error": "UI is read-only while viewing an active comparison run."}), 409
        payload = request.get_json(silent=True) or {}
        confirmation = str(payload.get("confirmation", "") or "").strip()
        expected = f"promote {proposal_id}"
        if confirmation != expected:
            return jsonify(
                {
                    "ok": False,
                    "error": "confirmation_required",
                    "confirmation_required": expected,
                    "message": f"Promotion requires confirmation: {expected}",
                }
            ), 400
        result = promote_proposal_to_active_rules(
            store,
            proposal_id,
            approved_by=str(payload.get("approved_by", "") or "operator"),
            actor="operator",
            note=str(payload.get("note", "") or ""),
        )
        if not bool(result.get("ok", False)):
            return jsonify(result), 409
        return jsonify(result)

    @app.post("/api/self-improvement/rollouts/<rollout_id>/rollback")
    def self_improvement_rollback_rollout(rollout_id: str):
        if source.read_only:
            return jsonify({"ok": False, "error": "UI is read-only while viewing an active comparison run."}), 409
        payload = request.get_json(silent=True) or {}
        confirmation = str(payload.get("confirmation", "") or "").strip()
        expected = f"rollback {rollout_id}"
        if confirmation != expected:
            return jsonify(
                {
                    "ok": False,
                    "error": "confirmation_required",
                    "confirmation_required": expected,
                    "message": f"Rollback requires confirmation: {expected}",
                }
            ), 400
        result = rollback_rollout_promotion(
            store,
            rollout_id,
            actor="operator",
            reason=str(payload.get("reason", "") or "operator_requested_rollback"),
            note=str(payload.get("note", "") or ""),
        )
        if not bool(result.get("ok", False)):
            return jsonify(result), 409
        return jsonify(result)

    @app.post("/api/runtime/execute-trades")
    def set_execute_trades():
        if source.read_only:
            return jsonify({"ok": False, "error": "UI is read-only while viewing an active comparison run."}), 409
        payload = request.get_json(silent=True) or {}
        enabled = bool(payload.get("enabled", False))
        confirmation = str(payload.get("confirmation", "") or "").strip()
        expected = f"set execute_trades {str(enabled).lower()}"
        if confirmation != expected:
            return jsonify(
                {
                    "ok": False,
                    "error": "confirmation_required",
                    "confirmation_required": expected,
                    "message": f"Execution toggle requires confirmation: {expected}",
                }
            ), 400
        flags = source.load_runtime_flags()
        flags["execute_trades"] = enabled
        source.save_runtime_flags(flags)
        source.log_event("system", f"UI toggled execute_trades to {enabled}")
        return jsonify({"ok": True, "execute_trades": enabled})

    @app.post("/api/runtime/gemini-influence")
    def set_gemini_influence():
        if source.read_only:
            return jsonify({"ok": False, "error": "UI is read-only while viewing an active comparison run."}), 409
        payload = request.get_json(silent=True) or {}
        enabled = bool(payload.get("enabled", False))
        confirmation = str(payload.get("confirmation", "") or "").strip()
        expected = f"set gemini_influence {str(enabled).lower()}"
        if confirmation != expected:
            return jsonify(
                {
                    "ok": False,
                    "error": "confirmation_required",
                    "confirmation_required": expected,
                    "message": f"AI influence toggle requires confirmation: {expected}",
                }
            ), 400
        flags = source.load_runtime_flags()
        flags["gemini_influence_enabled"] = enabled
        source.save_runtime_flags(flags)
        source.log_event("system", f"UI toggled gemini_influence_enabled to {enabled}")
        return jsonify({"ok": True, "gemini_influence_enabled": enabled})

    @app.post("/api/safety/unlock")
    def safety_unlock_request():
        payload = request.get_json(silent=True) or {}
        confirmation = str(payload.get("confirmation", "") or "").strip()
        reason = str(payload.get("reason", "") or "operator_requested_unlock")
        expected = "unlock safety"
        if confirmation != expected:
            return jsonify(
                {
                    "ok": False,
                    "error": "confirmation_required",
                    "confirmation_required": expected,
                    "unlock_requires_confirmation": True,
                    "message": f"Safety unlock requires confirmation: {expected}",
                }
            ), 400
        flags = source.load_runtime_flags()
        flags["safety_unlock_requested"] = True
        flags["safety_unlock_requested_at"] = datetime.now(timezone.utc).isoformat()
        flags["safety_unlock_reason"] = reason
        source.save_runtime_flags(flags)
        source.log_event("risk", f"Safety unlock requested by operator: {reason}")
        return jsonify(
            {
                "ok": True,
                "requested": True,
                "executed": False,
                "message": "Unlock request recorded. Safety controls still enforce final conditions.",
            }
        )

    @app.post("/api/ask-atlas")
    def ask_atlas():
        payload = request.get_json(silent=True) or {}
        query = str(payload.get("query", "") or "")
        mode = str(payload.get("mode", "") or "")
        return jsonify(assistant.answer(query, mode=mode))

    @app.post("/api/ui-ready")
    def ui_ready():
        spoken = False
        try:
            spoken = bool(announce_ui_ready_once(store))
        except Exception:
            spoken = False
        return jsonify({"ok": True, "spoken": spoken})

    # ------------------------------------------------------------------
    # AI Observability API endpoints
    # ------------------------------------------------------------------

    def _ai_journal() -> AIDecisionJournal:
        return AIDecisionJournal(store.base)

    def _ai_reviewer() -> AIDecisionReviewer:
        return AIDecisionReviewer(store.base)

    def _ai_promotion() -> AIPromotionIntelligence:
        return AIPromotionIntelligence(store.base)

    def _ai_research_queue() -> AIResearchQueue:
        return AIResearchQueue(store.base)

    def _ai_demo_review_packet() -> AIDemoReviewPacket:
        return AIDemoReviewPacket(store.base)

    def _ai_demo_activation_proposal() -> AIDemoActivationProposal:
        return AIDemoActivationProposal(store.base)

    def _ai_demo_activation_preflight() -> AIDemoActivationPreflight:
        return AIDemoActivationPreflight(store.base)

    def _ai_demo_activator() -> AIDemoActivator:
        return AIDemoActivator(store.base)

    @app.get("/api/ai/decision/latest")
    def ai_decision_latest():
        record = _ai_journal().latest()
        return jsonify({"ok": True, "latest": record, "available": bool(record)})

    @app.get("/api/ai/decision/journal")
    def ai_decision_journal():
        limit = max(1, min(200, int(request.args.get("limit", 20) or 20)))
        return jsonify({"ok": True, "records": _ai_journal().recent(limit=limit), "limit": limit})

    @app.get("/api/ai/decision/reviews")
    def ai_decision_reviews():
        limit = max(1, min(200, int(request.args.get("limit", 20) or 20)))
        return jsonify({"ok": True, "reviews": _ai_reviewer().recent_reviews(limit=limit), "limit": limit})

    @app.post("/api/ai/decision/review")
    def ai_decision_review_refresh():
        reviewer = _ai_reviewer()
        events = store.tail_events(limit=160) if hasattr(store, "tail_events") else []
        result = reviewer.review(
            learning_state=store.load_learning_state(),
            events=events,
            limit=max(1, min(200, int((request.get_json(silent=True) or {}).get("limit", 20) or 20))),
        )
        return jsonify({"ok": True, "executed": False, "review": result})

    @app.get("/api/ai/strategy/preferences")
    def ai_strategy_preferences():
        prefs = _ai_reviewer().preferences.load()
        return jsonify({"ok": True, "preferences": prefs})

    @app.get("/api/ai/promotion/recommendations")
    def ai_promotion_recommendations():
        payload = _ai_promotion().load_latest()
        return jsonify({"ok": True, "recommendations": payload, "available": bool(payload)})

    @app.post("/api/ai/promotion/recompute")
    def ai_promotion_recompute():
        population = _load_evolution_population()
        reviewer = _ai_reviewer()
        payload = _ai_promotion().evaluate(
            population,
            strategy_preferences=reviewer.preferences.load(),
        )
        queue = _ai_research_queue().sync_from_promotion_report(payload)
        payload["research_queue"] = {"summary": AIResearchQueue.summary(queue)}
        return jsonify({"ok": True, "executed": False, "recommendations": payload})

    @app.get("/api/ai/research/queue")
    def ai_research_queue():
        payload = _ai_research_queue().load()
        policy = dict(payload.get("policy", {}) or {})
        policy["shadow_drawdown_policy"] = str(getattr(ui_settings, "evolution_shadow_drawdown_policy", "raw_peak"))
        policy["shadow_drawdown_limit_pct"] = float(getattr(ui_settings, "evolution_max_shadow_drawdown_pct", 0.05))
        payload["policy"] = policy
        return jsonify({"ok": True, "queue": payload, "summary": AIResearchQueue.summary(payload)})

    def _research_queue_action(item_id: str, status: str, required_confirmation: str = ""):
        payload = request.get_json(silent=True) or {}
        expected = str(required_confirmation or "").strip()
        if expected:
            confirmation = str(payload.get("confirmation", "") or "").strip()
            if confirmation != expected:
                return jsonify(
                    {
                        "ok": False,
                        "error": "confirmation_required",
                        "confirmation_required": expected,
                        "message": f"Research queue action requires confirmation: {expected}",
                    }
                ), 400
        result = _ai_research_queue().update_item(
            item_id,
            status,
            note=str(payload.get("note", "") or ""),
            operator=str(payload.get("operator", "operator") or "operator"),
        )
        if not result.get("ok", False):
            return jsonify(result), 404
        return jsonify(result)

    @app.post("/api/ai/research/items/<path:item_id>/review")
    def ai_research_item_review(item_id: str):
        return _research_queue_action(item_id, "reviewed", required_confirmation=f"review {item_id}")

    @app.post("/api/ai/research/items/<path:item_id>/defer")
    def ai_research_item_defer(item_id: str):
        return _research_queue_action(item_id, "deferred", required_confirmation=f"defer {item_id}")

    @app.post("/api/ai/research/items/<path:item_id>/reject")
    def ai_research_item_reject(item_id: str):
        return _research_queue_action(item_id, "rejected", required_confirmation=f"reject {item_id}")

    @app.post("/api/ai/research/items/<path:item_id>/paper-test")
    def ai_research_item_paper_test(item_id: str):
        return _research_queue_action(item_id, "paper_test_requested", required_confirmation=f"paper test {item_id}")

    @app.post("/api/ai/research/items/<path:item_id>/demo-review")
    def ai_research_item_demo_review(item_id: str):
        return _research_queue_action(item_id, "demo_review_requested", required_confirmation=f"demo review {item_id}")

    @app.post("/api/ai/research/items/<path:item_id>/demo-review-approve")
    def ai_research_item_demo_review_approve(item_id: str):
        return _research_queue_action(item_id, "demo_review_approved", required_confirmation=f"approve demo review {item_id}")

    @app.post("/api/ai/research/items/<path:item_id>/demo-review-packet")
    def ai_research_item_demo_review_packet(item_id: str):
        queue = _ai_research_queue().load()
        item = next((dict(row) for row in list(queue.get("items", []) or []) if str(dict(row).get("id", "")) == str(item_id)), None)
        if item is None:
            return jsonify({"ok": False, "reason": "item_not_found", "item_id": item_id, "executed": False}), 404
        uid = str(item.get("uid", "") or "")
        packet = _ai_demo_review_packet().build(uid)
        if not packet.get("ok", False):
            return jsonify(packet), 404
        attached = _ai_research_queue().attach_demo_review_packet(item_id, packet)
        packet["queue_update"] = attached
        return jsonify(packet)

    @app.post("/api/ai/research/items/<path:item_id>/demo-activation-proposal")
    def ai_research_item_demo_activation_proposal(item_id: str):
        queue = _ai_research_queue().load()
        item = next((dict(row) for row in list(queue.get("items", []) or []) if str(dict(row).get("id", "")) == str(item_id)), None)
        if item is None:
            return jsonify({"ok": False, "reason": "item_not_found", "item_id": item_id, "executed": False}), 404
        uid = str(item.get("uid", "") or "")
        proposal = _ai_demo_activation_proposal().build(uid)
        if not proposal.get("ok", False):
            return jsonify(proposal), 404
        queue_update = _ai_research_queue().update_item(
            item_id,
            "demo_activation_proposed",
            note="demo_activation_proposal_generated",
            operator="system",
        )
        proposal["queue_update"] = queue_update
        return jsonify(proposal)

    @app.post("/api/ai/research/items/<path:item_id>/demo-activation-preflight")
    def ai_research_item_demo_activation_preflight(item_id: str):
        queue = _ai_research_queue().load()
        item = next((dict(row) for row in list(queue.get("items", []) or []) if str(dict(row).get("id", "")) == str(item_id)), None)
        if item is None:
            return jsonify({"ok": False, "reason": "item_not_found", "item_id": item_id, "executed": False}), 404
        uid = str(item.get("uid", "") or "")
        preflight = _ai_demo_activation_preflight().run(uid)
        if not preflight.get("ok", False):
            return jsonify(preflight), 409
        return jsonify(preflight)

    @app.post("/api/ai/research/items/<path:item_id>/activate-demo")
    def ai_research_item_activate_demo(item_id: str):
        queue = _ai_research_queue().load()
        item = next((dict(row) for row in list(queue.get("items", []) or []) if str(dict(row).get("id", "")) == str(item_id)), None)
        if item is None:
            return jsonify({"ok": False, "reason": "item_not_found", "item_id": item_id, "trade_executed": False}), 404
        payload = request.get_json(silent=True) or {}
        uid = str(item.get("uid", "") or "")
        command = str(payload.get("explicit_command", "") or payload.get("confirm", "") or "")
        result = _ai_demo_activator().activate(
            uid,
            command,
            operator=str(payload.get("operator", "operator") or "operator"),
        )
        if not result.get("ok", False):
            return jsonify(result), 409
        return jsonify(result)

    # ------------------------------------------------------------------
    # Strategy Evolution API endpoints
    # ------------------------------------------------------------------

    def _load_evolution_population() -> list[StrategyChromosome]:
        pop_file = Path(store.base) / "evolution_population.json"
        if not pop_file.exists():
            return []
        try:
            payload = json.loads(pop_file.read_text(encoding="utf-8"))
            return [StrategyChromosome.from_dict(c) for c in payload.get("chromosomes", [])]
        except Exception:
            return []

    @app.get("/api/strategies/leaderboard")
    def evolution_leaderboard():
        population = _load_evolution_population()
        reporter = EvolutionReporter(output_dir=store.base)
        reporter.write_leaderboard(population)
        rows = reporter._build_leaderboard_rows(population)
        return jsonify({
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "total_population": len(population),
            "stages": reporter._stage_counts(population),
            "leaderboard": rows[:20],
            "top_candidate": reporter._top_candidate(population),
        })

    @app.get("/api/strategies/evolution-log")
    def evolution_log():
        log_file = Path(store.base) / "evolution_log.json"
        if not log_file.exists():
            return jsonify({"updated_at": datetime.now(timezone.utc).isoformat(), "events": []})
        try:
            payload = _load_json_file(log_file)
            return jsonify(payload)
        except Exception:
            return jsonify({"updated_at": datetime.now(timezone.utc).isoformat(), "events": []})

    @app.get("/api/strategies/<strategy_id>")
    def strategy_detail(strategy_id: str):
        population = _load_evolution_population()
        chromosome = next((c for c in population if c.uid == strategy_id or c.uid.startswith(strategy_id)), None)
        if chromosome is None:
            return jsonify({"ok": False, "error": "strategy_not_found"}), 404
        reporter = EvolutionReporter(output_dir=store.base)
        detail = reporter.strategy_detail(chromosome)
        detail["markdown_summary"] = StrategyExplainer.explain_markdown(chromosome)
        return jsonify({"ok": True, "strategy": detail})

    return app


def serve_ui_app(app: Flask, host: str = "127.0.0.1", port: int = 8765) -> None:
    """Run the Atlas UI server as a durable control-surface host.

    This intentionally separates server lifetime from desktop/browser wrapper
    lifetime so native window failures do not implicitly tear down the API.
    """
    _LOGGER.info("Starting durable UI server on http://%s:%s", host, port)
    server = make_server(host, port, app, threaded=True, request_handler=AtlasUIRequestHandler)
    # If the serving socket stops, do not let abandoned request threads keep a
    # zombie pythonw.exe alive while the API port is already offline.
    with contextlib.suppress(Exception):
        server.daemon_threads = True
    with contextlib.suppress(Exception):
        server.block_on_close = False
    try:
        server.serve_forever()
    finally:
        with contextlib.suppress(Exception):
            server.server_close()
        _LOGGER.info("Durable UI server stopped on http://%s:%s", host, port)


def start_visual_watch_loop(
    store: KnowledgeStore,
    *,
    interval_seconds: float | None = None,
    enabled_default: bool = True,
) -> threading.Event:
    stop_event = threading.Event()
    default_raw = "1" if enabled_default else "0"
    enabled = str(os.getenv("ATLAS_VISUAL_WATCH_ENABLED", default_raw)).strip().lower() not in {"0", "false", "no", "off"}
    if not enabled:
        return stop_event
    interval = float(interval_seconds if interval_seconds is not None else os.getenv("ATLAS_VISUAL_WATCH_SECONDS", "3") or 3)
    interval = max(1.0, min(30.0, interval))
    visual_dir = Path(store.base) / "chart_vision"
    with contextlib.suppress(Exception):
        flags = dict(store.load_runtime_flags() or {})
        visual_watch = dict(flags.get("visual_watch", {}) or {})
        visual_watch.update(
            {
                "enabled": True,
                "interval_seconds": interval,
                "policy_note": "Continuous visual watch is advisory-only and cannot execute trades or bypass gates.",
            }
        )
        flags["visual_watch"] = visual_watch
        store.save_runtime_flags(flags)

    def _run() -> None:
        while not stop_event.wait(interval):
            started = time.time()
            try:
                status_payload = dict(store.load_status() or {})
                capture = capture_desktop_screenshot(visual_dir)
                image_path = str(capture.get("path", "") or "") if bool(capture.get("ok", False)) else ""
                visual = analyze_chart_visual_context(
                    status=status_payload,
                    chart_features=dict(status_payload.get("chart_features", {}) or {}),
                    image_path=image_path or None,
                )
                visual["capture"] = capture
                visual["visual_watch"] = {
                    "enabled": True,
                    "interval_seconds": interval,
                    "captured_at_epoch": round(started, 3),
                    "advisory_only": True,
                    "does_not_execute": True,
                }
                saved = save_visual_read(visual, visual_dir)
                flags = dict(store.load_runtime_flags() or {})
                visual_watch = dict(flags.get("visual_watch", {}) or {})
                visual_watch.update(
                    {
                        "enabled": True,
                        "interval_seconds": interval,
                        "last_capture_ok": bool(capture.get("ok", False)),
                        "last_capture_source": str(capture.get("capture_source", "") or ""),
                        "last_capture_method": str(dict(capture.get("capture_method", {}) or {}).get("method", "") or ""),
                        "last_visual_status": str(dict(saved.get("visual_features", {}) or {}).get("status", "") or ""),
                        "last_captured_at": str(saved.get("captured_at", "") or ""),
                        "policy_note": "Continuous visual watch is advisory-only and cannot execute trades or bypass gates.",
                    }
                )
                flags["visual_watch"] = visual_watch
                store.save_runtime_flags(flags)
            except Exception as exc:
                with contextlib.suppress(Exception):
                    flags = dict(store.load_runtime_flags() or {})
                    visual_watch = dict(flags.get("visual_watch", {}) or {})
                    visual_watch.update(
                        {
                            "enabled": True,
                            "interval_seconds": interval,
                            "last_capture_ok": False,
                            "last_error": str(exc),
                            "last_error_at": datetime.now(timezone.utc).isoformat(),
                            "policy_note": "Continuous visual watch is advisory-only and cannot execute trades or bypass gates.",
                        }
                    )
                    flags["visual_watch"] = visual_watch
                    store.save_runtime_flags(flags)

    threading.Thread(target=_run, name="atlas-visual-watch", daemon=True).start()
    return stop_event


def start_background_live_loop(trader, poll_seconds: int) -> threading.Thread:
    def _run() -> None:
        consecutive_failures = 0
        while True:
            try:
                trader.step()
                consecutive_failures = 0
                time.sleep(poll_seconds)
            except Exception as exc:
                consecutive_failures += 1
                sleep_seconds = min(120, max(poll_seconds, 2 ** min(consecutive_failures, 6)))
                trader.store.log_event("error", f"Live loop error: {exc}")
                trader.store.save_status(
                    {
                        "action": "error",
                        "message": (
                            f"Live loop error: {exc}. "
                            f"Reconnecting with backoff ({sleep_seconds}s, attempt {consecutive_failures})."
                        ),
                        "execute_trades": bool(trader._execute_trades_enabled()),
                        "gemini_enabled": bool(getattr(trader.gemini, "enabled", False)),
                        "symbol": trader.settings.symbol,
                        "timeframe": trader.settings.timeframe,
                    }
                )
                time.sleep(sleep_seconds)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread
