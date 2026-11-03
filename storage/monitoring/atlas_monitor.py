import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from atlas_ai.learning.online_learning import build_online_learning_observation, update_online_learning_state
from atlas_ai.execution.comment_codec import parse_broker_comment
from atlas_ai.memory.knowledge_store import KnowledgeStore

LOG = ROOT / "monitoring" / "atlas_watchdog.log"
INSIGHTS = ROOT / "monitoring" / "atlas_monitor_insights.jsonl"
STATE = ROOT / "monitoring" / "atlas_monitor_state.json"
NEAR_SETUP_REPORT = ROOT / "monitoring" / "atlas_near_setup_learning.json"
ONLINE_LEARNING_REPORT = ROOT / "monitoring" / "atlas_online_learning.json"
POST_TRADE_REPORT = ROOT / "monitoring" / "atlas_post_trade_learning.json"


def now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat()


def log(message):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"{now_iso()} {message}\n")


def insight(payload):
    payload = dict(payload)
    payload["ts"] = now_iso()
    INSIGHTS.parent.mkdir(parents=True, exist_ok=True)
    with INSIGHTS.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, separators=(",", ":"), default=str) + "\n")


def load_state():
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except Exception:
        return {"seen_suspicious_deals": []}


def save_state(state):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")


def save_near_setup_report(report):
    NEAR_SETUP_REPORT.parent.mkdir(parents=True, exist_ok=True)
    NEAR_SETUP_REPORT.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")


def save_online_learning_report(report):
    ONLINE_LEARNING_REPORT.parent.mkdir(parents=True, exist_ok=True)
    ONLINE_LEARNING_REPORT.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")


def save_post_trade_report(report):
    POST_TRADE_REPORT.parent.mkdir(parents=True, exist_ok=True)
    POST_TRADE_REPORT.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")


def persist_online_learning(status, near_setup_sample, near_setup_learning):
    observation = build_online_learning_observation(
        status=dict(status or {}),
        near_setup_sample=dict(near_setup_sample or {}),
        near_setup_learning=dict(near_setup_learning or {}),
    )
    store = KnowledgeStore(str(ROOT))
    payload = store.update_learning_state(lambda current: update_online_learning_state(current, observation))
    online = dict(dict(payload.get("state", {}) or {}).get("online_learning", {}) or {})
    save_online_learning_report(online)
    return online


def classify_execution_failure(reason):
    text = str(reason or "").strip().lower()
    if not text:
        return ""
    if "broker_rejected" in text or "insufficient_margin" in text:
        return "broker_reject"
    if "broker_response_invalid" in text or "broker_confirmation_missing" in text or "missing_broker_order_id" in text:
        return "broker_confirmation"
    if "order_reconciliation_required" in text or "timeout" in text or "unknown" in text:
        return "reconciliation_required"
    if "approval" in text or "explicit_confirmation" in text:
        return "approval_or_authority"
    if "freshness" in text or "stale" in text:
        return "stale_runtime"
    if "cluster" in text:
        return "execution_cluster"
    if "consensus guard" in text or "higher-timeframe alignment" in text or "directional-structure guard" in text:
        return "quality_gate"
    if "no high-confidence" in text or "no_clear_setup" in text or "m1_quality" in text or "structure" in text or "confidence" in text or "evidence" in text:
        return "setup_quality"
    if "daily_loss" in text or "recent_losses" in text or "drawdown" in text:
        return "risk_limit"
    if "session" in text:
        return "session_gate"
    if "executor" in text:
        return "executor_error"
    return "other"


def build_execution_diagnostics(status):
    execution = dict(status.get("execution", {}) or {})
    truth = dict(status.get("execution_truth", {}) or {})
    truth_ts = 0.0
    try:
        truth_ts = float(truth.get("ts", 0.0) or 0.0)
    except Exception:
        truth_ts = 0.0
    truth_age_seconds = max(0.0, time.time() - truth_ts) if truth_ts > 0 else 0.0
    stale_truth = bool(
        truth
        and truth_age_seconds > 900
        and str(truth.get("reason", "") or "") in {"approval_missing", "live_requires_explicit_confirmation"}
        and str(status.get("action", "") or "").lower() in {"hold", "blocked"}
    )
    reason = str(
        ("" if stale_truth else truth.get("reason"))
        or execution.get("comment")
        or status.get("primary_blocker")
        or status.get("message")
        or ""
    )
    failure_class = str(("" if stale_truth else truth.get("failure_class")) or classify_execution_failure(reason))
    active_truth = {} if stale_truth else truth
    return {
        "status_action": str(status.get("action", "") or ""),
        "status_message": str(status.get("message", "") or ""),
        "execution_success": bool(execution.get("success", active_truth.get("ok", False))),
        "retcode": execution.get("retcode", ""),
        "broker_order_id": str(execution.get("broker_order_id") or active_truth.get("broker_order_id") or ""),
        "decision_id": str(execution.get("decision_id") or ""),
        "reason": reason,
        "failure_class": failure_class,
        "truth": active_truth,
        "truth_stale": stale_truth,
        "truth_age_seconds": round(truth_age_seconds, 2),
        "ignored_stale_truth": {
            "reason": truth.get("reason", ""),
            "failure_class": truth.get("failure_class", ""),
            "request_id": truth.get("request_id", ""),
            "age_seconds": round(truth_age_seconds, 2),
        }
        if stale_truth
        else {},
        "policy_note": "Execution diagnostics are observability only; they do not change strategy, risk, or execution policy.",
    }


def is_outcome_deal(deal):
    row = dict(deal or {})
    profit = float(row.get("profit", 0.0) or 0.0)
    comment = str(row.get("comment", "") or "").lower()
    # MT5 can expose zero-profit entry legs with the same position id as the
    # closing deal. Only outcome-bearing legs should teach post-trade lessons.
    return abs(profit) > 1e-9 or "[tp" in comment or "[sl" in comment


def ps_json(command):
    completed = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", command],
        cwd=str(PROJECT),
        capture_output=True,
        text=True,
        timeout=20,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        return []
    try:
        data = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else [data]


def live_processes():
    return ps_json(
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.CommandLine -match 'atlas_ai[.]main live' } | "
        "Select-Object ProcessId,ParentProcessId,Name,CommandLine | ConvertTo-Json -Depth 4"
    )


def weekend_learning_only_enabled():
    try:
        flags = json.loads((PROJECT / "storage" / "runtime_flags.json").read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(flags.get("weekend_learning_only", False))


def stop_live_processes():
    subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_Process | "
            "Where-Object { $_.CommandLine -match 'atlas_ai[.]main live' } | "
            "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }",
        ],
        cwd=str(PROJECT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=20,
    )


def start_backend():
    py = PROJECT / ".venv" / "Scripts" / "python.exe"
    exe = str(py) if py.exists() else sys.executable
    subprocess.Popen(
        [exe, "-m", "atlas_ai.main", "live", "--poll-seconds", "60", "--mode-override", "live"],
        cwd=str(PROJECT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    log("backend_restart_requested reason=not_running_or_stale")


def refresh_full_stack(reason):
    py = PROJECT / ".venv" / "Scripts" / "python.exe"
    if not py.exists():
        log(f"full_stack_refresh_skipped reason={reason} error=venv_python_missing")
        return {"ok": False, "error": "venv_python_missing"}
    try:
        completed = subprocess.run(
            [str(py), "-m", "atlas_ai.atlas_full_launcher", "--json"],
            cwd=str(PROJECT),
            capture_output=True,
            text=True,
            timeout=180,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        ok = completed.returncode == 0
        log(f"full_stack_refresh_completed reason={reason} ok={ok} returncode={completed.returncode}")
        return {
            "ok": ok,
            "returncode": completed.returncode,
            "stdout_tail": (completed.stdout or "")[-1200:],
            "stderr_tail": (completed.stderr or "")[-1200:],
        }
    except Exception as exc:
        log(f"full_stack_refresh_error reason={reason} error={exc}")
        return {"ok": False, "error": str(exc)}


def get_status():
    try:
        with urlopen("http://127.0.0.1:8765/api/status/compact", timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        with urlopen("http://127.0.0.1:8765/api/status", timeout=8) as response:
            return json.loads(response.read().decode("utf-8"))


def get_health():
    with urlopen("http://127.0.0.1:8765/api/health", timeout=2) as response:
        return json.loads(response.read().decode("utf-8"))


def mt5_snapshot():
    py = PROJECT / ".venv" / "Scripts" / "python.exe"
    if not py.exists():
        return {"error": "venv_python_missing"}
    code = r"""
import json
from datetime import datetime, timedelta
import MetaTrader5 as mt5
from atlas_ai.config.settings import load_settings
from atlas_ai.execution.comment_codec import parse_broker_comment
s = load_settings()
ok = mt5.initialize()
acc = mt5.account_info() if ok else None
pos = mt5.positions_get(symbol=s.symbol) if ok else None
pos = pos or []
start = datetime.now() - timedelta(hours=24)
end = datetime.now() + timedelta(minutes=2)
deals = mt5.history_deals_get(start, end) if ok else None
deals = deals or []
suspicious = []
atlas_deals = []
for d in deals:
    if str(getattr(d, "symbol", "")) != str(s.symbol):
        continue
    magic = int(getattr(d, "magic", 0) or 0)
    comment = str(getattr(d, "comment", "") or "")
    compact = "".join(ch for ch in comment.lower() if ch.isalnum())
    parsed_comment = parse_broker_comment(comment)
    ours = (
        magic == int(getattr(s, "magic_number", 0) or 0)
        or bool(parsed_comment.get("is_atlas"))
        or compact.startswith("aai")
        or compact.startswith("atlasai")
    )
    row = {
        "ticket": int(getattr(d, "ticket", 0)),
        "order": int(getattr(d, "order", 0)),
        "position_id": int(getattr(d, "position_id", 0)),
        "time": datetime.fromtimestamp(int(getattr(d, "time", 0))).isoformat(),
        "symbol": str(getattr(d, "symbol", "")),
        "type": int(getattr(d, "type", -1)),
        "entry": int(getattr(d, "entry", -1)),
        "volume": float(getattr(d, "volume", 0.0) or 0.0),
        "price": float(getattr(d, "price", 0.0) or 0.0),
        "profit": float(getattr(d, "profit", 0.0) or 0.0),
        "magic": magic,
        "comment": comment,
    }
    if ours:
        atlas_deals.append(row)
        continue
    if not ours:
        suspicious.append(row)
closed_atlas = [d for d in atlas_deals if int(d.get("entry", -1)) != 0]
print(json.dumps({
    "connected": bool(ok and acc),
    "balance": float(getattr(acc, "balance", 0.0)) if acc else None,
    "equity": float(getattr(acc, "equity", 0.0)) if acc else None,
    "profit": float(getattr(acc, "profit", 0.0)) if acc else None,
    "positions": len(pos),
    "floating_profit": round(sum(float(getattr(p, "profit", 0.0)) for p in pos), 2),
    "atlas_deals": atlas_deals[-20:],
    "atlas_deal_count": len(atlas_deals),
    "atlas_closed_count": len(closed_atlas),
    "atlas_realized_profit": round(sum(float(d.get("profit", 0.0) or 0.0) for d in closed_atlas), 2),
    "latest_atlas_deal": atlas_deals[-1] if atlas_deals else {},
    "suspicious_deals": suspicious[-20:],
}))
"""
    try:
        completed = subprocess.run(
            [str(py), "-c", code],
            cwd=str(PROJECT),
            capture_output=True,
            text=True,
            timeout=20,
        )
        if completed.stdout.strip():
            return json.loads(completed.stdout)
        return {
            "error": "empty_mt5_snapshot",
            "returncode": completed.returncode,
            "stderr_tail": (completed.stderr or "")[-1000:],
        }
    except Exception as exc:
        return {"error": str(exc)}


def dig(data, *keys):
    cur = data
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def confusion_signals(status):
    signals = []
    checks = [
        ("market_regime_unknown", dig(status, "market_regime", "regime") == "unknown"),
        ("event_risk_unknown", dig(status, "context_snapshot", "event_risk_state") == "unknown"),
        ("news_risk_unknown", dig(status, "context_snapshot", "news_risk_classification") == "unknown"),
        ("dxy_alignment_unknown", dig(status, "context_snapshot", "dxy_alignment") == "unknown"),
        ("us10y_alignment_unknown", dig(status, "context_snapshot", "us10y_alignment") == "unknown"),
        ("vix_alignment_unknown", dig(status, "context_snapshot", "vix_alignment") == "unknown"),
        ("oil_alignment_unknown", dig(status, "context_snapshot", "oil_alignment") == "unknown"),
        ("dxy_data_unavailable", dig(status, "decision_analysis", "macro_context", "dxy", "availability") == "unavailable"),
        ("us10y_data_unavailable", dig(status, "decision_analysis", "macro_context", "us10y", "availability") == "unavailable"),
        ("vix_data_unavailable", dig(status, "decision_analysis", "macro_context", "vix", "availability") == "unavailable"),
        ("oil_data_unavailable", dig(status, "decision_analysis", "macro_context", "oil", "availability") == "unavailable"),
        ("setup_family_no_trade", status.get("setup_family") == "no_trade"),
        ("momentum_mixed", dig(status, "chart_features", "momentum_state") == "mixed"),
        ("momentum_weak", dig(status, "chart_features", "momentum_state") == "weak"),
        ("entry_location_mid_range", dig(status, "chart_features", "location_state") == "mid_range"),
    ]
    for name, active in checks:
        if active:
            signals.append(name)
    return signals


def _top_blocker_name(raw):
    text = str(raw or "").strip()
    if ":" in text:
        text = text.split(":", 1)[0].strip()
    return text or "unknown"


def classify_near_setup(status, market, confusion, top_blockers):
    decision = dict(status.get("decision_analysis", {}) or {})
    action = str(status.get("action") or "").lower()
    blocker_class = str(status.get("blocker_class") or "").lower()
    confluence = dig(status, "decision_analysis", "confluence_score")
    threshold = dig(status, "decision_analysis", "confluence_threshold")
    try:
        confluence_gap = round(float(threshold) - float(confluence), 2)
    except Exception:
        confluence_gap = None

    hard_blockers = [str(x) for x in list(decision.get("hard_blockers", []) or decision.get("blockers", []) or [])]
    near_enough = confluence_gap is not None and 0 <= confluence_gap <= 12
    signal_quality_hold = action in {"hold", "blocked"} and blocker_class in {"signal_quality", "quality_gate", "safety_gate", "unknown"}
    relevant_blocker = bool({"no_clear_setup", "structure_too_weak"} & set(hard_blockers))
    relevant_confusion = bool({"setup_family_no_trade", "momentum_mixed", "momentum_weak", "entry_location_mid_range"} & set(confusion))
    if not (signal_quality_hold and (near_enough or relevant_blocker or relevant_confusion)):
        return {}

    blockers = [_top_blocker_name(x) for x in list(top_blockers or [])][:4]
    if not blockers and hard_blockers:
        blockers = hard_blockers[:4]

    momentum = str((market or {}).get("momentum") or "unknown")
    location = str((market or {}).get("location") or "unknown")
    regime = str((market or {}).get("regime") or "unknown")
    trend_structure = str((market or {}).get("trend_structure") or "unknown")
    fingerprint_parts = [
        f"regime={regime}",
        f"structure={trend_structure}",
        f"momentum={momentum}",
        f"location={location}",
    ]
    if blockers:
        fingerprint_parts.append("blockers=" + "+".join(blockers[:3]))
    fingerprint = "|".join(fingerprint_parts)

    teaching_focus = []
    if location in {"mid_range", "consolidation"} or "entry_location_mid_range" in confusion:
        teaching_focus.append("wait for a cleaner edge or pullback location instead of treating mid-range movement as a setup")
    if momentum in {"mixed", "weak"} or {"momentum_mixed", "momentum_weak"} & set(confusion):
        teaching_focus.append("separate direction bias from entry momentum; require confirmation before acting")
    if "setup_family_no_trade" in confusion or "no_clear_setup" in hard_blockers:
        teaching_focus.append("label this as no-trade until a specific setup family is visible")
    if location == "breakout" and momentum == "strong" and ("setup_family_no_trade" in confusion or "no_clear_setup" in hard_blockers):
        teaching_focus.append("distinguish raw breakout movement from a confirmed breakout/retest playbook before treating it as actionable")
    if "structure_too_weak" in hard_blockers:
        teaching_focus.append("prefer clearer HH/HL or LH/LL structure before continuation entries")
    if not teaching_focus:
        teaching_focus.append("collect more examples before changing any trading behavior")

    return {
        "event": "near_setup_learning",
        "mode": "observability_only",
        "fingerprint": fingerprint,
        "action": status.get("action"),
        "blocker_class": status.get("blocker_class"),
        "primary_blocker": status.get("primary_blocker") or status.get("message"),
        "confluence_score": confluence,
        "confluence_threshold": threshold,
        "confluence_gap": confluence_gap,
        "hard_blockers": hard_blockers,
        "top_blockers": blockers,
        "confusion_signals": list(confusion or []),
        "market": dict(market or {}),
        "teaching_focus": teaching_focus,
        "policy_note": "Near-setup learning is observability only; it does not change strategy, risk, guard, or execution policy.",
    }


def update_near_setup_learning(state, sample):
    if not sample:
        return {}
    learning = dict(state.get("near_setup_learning", {}) or {})
    patterns = dict(learning.get("patterns", {}) or {})
    fingerprint = str(sample.get("fingerprint") or "unknown")
    row = dict(patterns.get(fingerprint, {}) or {})
    row["count"] = int(row.get("count", 0) or 0) + 1
    row["first_seen"] = row.get("first_seen") or now_iso()
    row["last_seen"] = now_iso()
    row["latest"] = sample
    patterns[fingerprint] = row
    ranked = sorted(patterns.items(), key=lambda item: int(dict(item[1]).get("count", 0) or 0), reverse=True)
    learning = {
        "mode": "observability_only",
        "updated_at": now_iso(),
        "total_samples": int(learning.get("total_samples", 0) or 0) + 1,
        "top_patterns": [
            {
                "fingerprint": key,
                "count": int(dict(value).get("count", 0) or 0),
                "last_seen": dict(value).get("last_seen"),
                "teaching_focus": list(dict(dict(value).get("latest", {}) or {}).get("teaching_focus", []) or []),
            }
            for key, value in ranked[:10]
        ],
        "patterns": dict(ranked[:50]),
        "policy_note": "This report teaches Atlas what to inspect next; it does not alter live trading decisions.",
    }
    state["near_setup_learning"] = learning
    return learning


def update_post_trade_learning(state, closed_deals, status, market, near_setup_sample):
    deals = [dict(d or {}) for d in list(closed_deals or [])]
    if not deals:
        return {}
    learning = dict(state.get("post_trade_learning", {}) or {})
    seen = {str(x) for x in list(learning.get("seen_tickets", []) or [])}
    outcomes = list(learning.get("recent_outcomes", []) or [])
    patterns = dict(learning.get("patterns", {}) or {})
    new_rows = []
    execution_diag = build_execution_diagnostics(status)
    setup_fingerprint = str((near_setup_sample or {}).get("fingerprint") or "")
    if not setup_fingerprint:
        setup_fingerprint = "|".join(
            [
                f"regime={str((market or {}).get('regime') or 'unknown')}",
                f"structure={str((market or {}).get('trend_structure') or 'unknown')}",
                f"momentum={str((market or {}).get('momentum') or 'unknown')}",
                f"location={str((market or {}).get('location') or 'unknown')}",
            ]
        )
    for deal in deals:
        ticket = str(deal.get("ticket", "") or "")
        if not ticket or ticket in seen or not is_outcome_deal(deal):
            continue
        seen.add(ticket)
        profit = float(deal.get("profit", 0.0) or 0.0)
        outcome = "win" if profit > 0 else ("loss" if profit < 0 else "flat")
        row = {
            "ticket": ticket,
            "order": int(deal.get("order", 0) or 0),
            "position_id": int(deal.get("position_id", 0) or 0),
            "time": str(deal.get("time", "") or ""),
            "symbol": str(deal.get("symbol", "") or ""),
            "volume": float(deal.get("volume", 0.0) or 0.0),
            "price": float(deal.get("price", 0.0) or 0.0),
            "profit": round(profit, 2),
            "outcome": outcome,
            "setup_fingerprint": setup_fingerprint,
            "market_context": dict(market or {}),
            "execution_diagnostics": execution_diag,
            "correlation_quality": "monitor_tick_context",
            "lesson": (
                "reinforce similar setup context after confirmed profit"
                if outcome == "win"
                else "review entry timing, HTF alignment, momentum, and location before repeating this context"
                if outcome == "loss"
                else "flat outcome; collect more examples before drawing a lesson"
            ),
            "policy_note": "Post-trade learning is observability only; it does not alter live trading decisions.",
        }
        outcomes.append(row)
        pattern = dict(patterns.get(setup_fingerprint, {}) or {})
        pattern["count"] = int(pattern.get("count", 0) or 0) + 1
        pattern["wins"] = int(pattern.get("wins", 0) or 0) + (1 if outcome == "win" else 0)
        pattern["losses"] = int(pattern.get("losses", 0) or 0) + (1 if outcome == "loss" else 0)
        pattern["profit"] = round(float(pattern.get("profit", 0.0) or 0.0) + profit, 2)
        pattern["last_seen"] = now_iso()
        pattern["latest"] = row
        patterns[setup_fingerprint] = pattern
        new_rows.append(row)

    if not new_rows:
        return dict(learning or {})
    ranked = sorted(patterns.items(), key=lambda item: (float(dict(item[1]).get("profit", 0.0) or 0.0), int(dict(item[1]).get("count", 0) or 0)), reverse=True)
    learning = {
        "mode": "observability_only",
        "updated_at": now_iso(),
        "seen_tickets": sorted(seen)[-1000:],
        "recent_outcomes": outcomes[-300:],
        "latest_outcomes": new_rows[-20:],
        "total_closed_trades_seen": len(seen),
        "top_patterns": [
            {
                "fingerprint": key,
                "count": int(dict(value).get("count", 0) or 0),
                "wins": int(dict(value).get("wins", 0) or 0),
                "losses": int(dict(value).get("losses", 0) or 0),
                "profit": round(float(dict(value).get("profit", 0.0) or 0.0), 2),
                "last_seen": dict(value).get("last_seen", ""),
            }
            for key, value in ranked[:10]
        ],
        "patterns": dict(ranked[:80]),
        "policy_note": "This connects MT5 outcomes to observed setup context; it does not change strategy, risk, guard, or execution policy.",
    }
    state["post_trade_learning"] = learning
    return learning


def main():
    interval = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    stale_minutes = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    log(f"python_monitor_started interval={interval} stale_minutes={stale_minutes}")
    insight({"event": "python_monitor_started", "interval_seconds": interval, "stale_minutes": stale_minutes})

    while True:
        try:
            state = load_state()
            live = live_processes()
            if weekend_learning_only_enabled():
                if live:
                    insight({"event": "weekend_learning_only_live_worker_found", "action_taken": "stop_live_backend"})
                    stop_live_processes()
                time.sleep(interval)
                continue
            if not live:
                insight({"event": "backend_down", "action_taken": "restart_backend"})
                start_backend()
                time.sleep(interval)
                continue

            try:
                status = get_status()
            except Exception as exc:
                failures = int(state.get("status_api_failure_count", 0) or 0) + 1
                state["status_api_failure_count"] = failures
                log(f"status_api_unavailable backend_process_present=true error={exc}")
                health_ok = False
                try:
                    health_ok = bool((get_health() or {}).get("ok"))
                except Exception:
                    health_ok = False
                payload = {
                    "event": "status_api_unavailable",
                    "backend_process_count": len(live),
                    "error": str(exc),
                    "consecutive_failures": failures,
                    "health_ok": health_ok,
                }
                now_ts = time.time()
                last_refresh = float(state.get("last_status_api_recovery_at", 0.0) or 0.0)
                refresh_threshold = 4 if health_ok else 2
                if failures >= refresh_threshold and now_ts - last_refresh >= 600:
                    state["last_status_api_recovery_at"] = now_ts
                    save_state(state)
                    result = refresh_full_stack("status_api_unavailable")
                    payload["action_taken"] = "full_stack_refresh"
                    payload["refresh_ok"] = bool(result.get("ok"))
                    payload["refresh_returncode"] = result.get("returncode")
                else:
                    save_state(state)
                insight(payload)
                time.sleep(interval)
                continue
            if int(state.get("status_api_failure_count", 0) or 0) > 0:
                state["status_api_failure_count"] = 0
                save_state(state)

            age_minutes = 0.0
            updated_at = status.get("updated_at")
            if updated_at:
                parsed = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    parsed = parsed.astimezone()
                age_minutes = (datetime.now(parsed.tzinfo) - parsed).total_seconds() / 60.0

            if age_minutes > stale_minutes:
                log(f"status_stale age_minutes={age_minutes:.2f} action=restart_backend")
                insight({"event": "status_stale", "age_minutes": round(age_minutes, 2), "action_taken": "restart_backend"})
                stop_live_processes()
                time.sleep(2)
                start_backend()
                time.sleep(interval)
                continue

            risk = status.get("risk") or {}
            market = {
                "session": status.get("session"),
                "regime": dig(status, "market_regime", "regime"),
                "trend_structure": dig(status, "chart_features", "trend_structure"),
                "momentum": dig(status, "chart_features", "momentum_state"),
                "location": dig(status, "chart_features", "location_state"),
                "spread_points": risk.get("spread_points"),
            }
            confusion = confusion_signals(status)
            top_blockers = []
            for blocker in dig(status, "pre_trade_memo", "top_blockers") or []:
                top_blockers.append(f"{blocker.get('name')}:{blocker.get('actual')}/{blocker.get('target')}")

            mt5 = mt5_snapshot()
            atlas_deals = list((mt5 or {}).get("atlas_deals", []) or [])
            atlas_deal_count = len(atlas_deals)
            atlas_closed_count = int((mt5 or {}).get("atlas_closed_count", 0) or 0)
            atlas_realized_profit = round(float((mt5 or {}).get("atlas_realized_profit", 0.0) or 0.0), 2)
            latest_atlas_deal = dict((mt5 or {}).get("latest_atlas_deal", {}) or {})
            suspicious_deals = list((mt5 or {}).get("suspicious_deals", []) or [])
            suspicious_deal_count = len(suspicious_deals)
            suspicious_volume = round(sum(float(d.get("volume", 0.0) or 0.0) for d in suspicious_deals), 2)
            suspicious_profit = round(sum(float(d.get("profit", 0.0) or 0.0) for d in suspicious_deals), 2)
            mt5_public = dict(mt5 or {})
            mt5_public.pop("atlas_deals", None)
            mt5_public.pop("suspicious_deals", None)
            mt5_public.update(
                {
                    "atlas_deal_count": atlas_deal_count,
                    "atlas_closed_count": atlas_closed_count,
                    "atlas_realized_profit": atlas_realized_profit,
                    "latest_atlas_deal": latest_atlas_deal,
                    "suspicious_deal_count": suspicious_deal_count,
                    "suspicious_volume": suspicious_volume,
                    "suspicious_profit": suspicious_profit,
                    "suspicious_examples": suspicious_deals[:3],
                    "suspicious_interpretation": "Broker history contains XAUUSD deals not tagged with Atlas magic/comment.",
                }
            )
            seen_atlas = {str(x) for x in list(state.get("seen_atlas_deals", []) or [])}
            new_atlas_deals = []
            new_closed = []
            for deal in atlas_deals:
                ticket = str(deal.get("ticket", ""))
                if ticket and ticket not in seen_atlas:
                    new_atlas_deals.append(deal)
                    seen_atlas.add(ticket)
            if new_atlas_deals:
                state["seen_atlas_deals"] = sorted(seen_atlas)[-500:]
                save_state(state)
                new_closed = [d for d in new_atlas_deals if int(d.get("entry", -1) or -1) != 0 and is_outcome_deal(d)]
                new_profit = round(sum(float(d.get("profit", 0.0) or 0.0) for d in new_closed), 2)
                log(f"atlas_trade_activity deals={len(new_atlas_deals)} closed={len(new_closed)} profit={new_profit}")
                insight(
                    {
                        "event": "atlas_trade_activity",
                        "deal_count": len(new_atlas_deals),
                        "closed_count": len(new_closed),
                        "realized_profit": new_profit,
                        "deals": new_atlas_deals[:20],
                        "interpretation": "MT5 history contains Atlas-tagged trade activity.",
                    }
                )

            seen = {str(x) for x in list(state.get("seen_suspicious_deals", []) or [])}
            new_suspicious = []
            for deal in suspicious_deals:
                ticket = str(deal.get("ticket", ""))
                if ticket and ticket not in seen:
                    new_suspicious.append(deal)
                    seen.add(ticket)
            if new_suspicious:
                state["seen_suspicious_deals"] = sorted(seen)[-500:]
                save_state(state)
                total_volume = round(sum(float(d.get("volume", 0.0) or 0.0) for d in new_suspicious), 2)
                total_profit = round(sum(float(d.get("profit", 0.0) or 0.0) for d in new_suspicious), 2)
                log(f"untagged_mt5_activity deals={len(new_suspicious)} volume={total_volume} profit={total_profit}")
                insight(
                    {
                        "event": "untagged_mt5_activity",
                        "deal_count": len(new_suspicious),
                        "total_volume": total_volume,
                        "total_profit": total_profit,
                        "deals": new_suspicious[:20],
                        "interpretation": "XAUUSD broker history contains deals not tagged with Atlas magic/comment",
                    }
                )

            near_setup_sample = classify_near_setup(status, market, confusion, top_blockers)
            near_setup_learning = update_near_setup_learning(state, near_setup_sample)
            if near_setup_learning:
                save_state(state)
                save_near_setup_report(near_setup_learning)
                insight(near_setup_sample)
            execution_diagnostics = build_execution_diagnostics(status)
            post_trade_learning = update_post_trade_learning(state, new_closed, status, market, near_setup_sample)
            latest_post_trade = list((post_trade_learning or {}).get("latest_outcomes", []) or [])
            if latest_post_trade:
                save_state(state)
                save_post_trade_report(post_trade_learning)
                insight(
                    {
                        "event": "post_trade_learning",
                        "mode": "observability_only",
                        "latest_outcomes": latest_post_trade,
                        "top_patterns": list((post_trade_learning or {}).get("top_patterns", []) or [])[:5],
                        "policy_note": str((post_trade_learning or {}).get("policy_note", "")),
                    }
                )
            online_learning = persist_online_learning(status, near_setup_sample, near_setup_learning)
            latest_online = dict(online_learning.get("latest", {}) or {})
            if latest_online:
                insight({"event": "online_learning_observation", **latest_online})

            log(
                "ok "
                f"action={status.get('action')} blocker_class={status.get('blocker_class')} "
                f"primary='{status.get('primary_blocker')}' weekly={risk.get('weekly_loss_pct')} "
                f"daily={risk.get('daily_loss_pct')} positions={risk.get('open_positions')} "
                f"confusion={','.join(confusion)}"
            )
            insight(
                {
                    "event": "monitor_tick",
                    "action": status.get("action"),
                    "message": status.get("message"),
                    "blocker_class": status.get("blocker_class"),
                    "primary_blocker": status.get("primary_blocker"),
                    "hard_blockers": dig(status, "decision_analysis", "hard_blockers") or [],
                    "top_blockers": top_blockers,
                    "confusion_signals": confusion,
                    "risk": {
                        "balance": risk.get("balance"),
                        "equity": risk.get("equity"),
                        "weekly_loss_pct": risk.get("weekly_loss_pct"),
                        "daily_loss_pct": risk.get("daily_loss_pct"),
                        "trades_today": risk.get("trades_today"),
                        "open_positions": risk.get("open_positions"),
                    },
                    "market": market,
                    "near_setup_learning": {
                        "active": bool(near_setup_sample),
                        "fingerprint": near_setup_sample.get("fingerprint") if near_setup_sample else "",
                        "top_pattern_count": (
                            int(dict((near_setup_learning.get("top_patterns") or [{}])[0]).get("count", 0) or 0)
                            if near_setup_learning
                            else 0
                        ),
                        "policy_note": "observability_only",
                    },
                    "online_learning": {
                        "active": bool(latest_online),
                        "mode": latest_online.get("mode", ""),
                        "allowed_influence": latest_online.get("allowed_influence", ""),
                        "evidence_strength": latest_online.get("evidence_strength", 0.0),
                        "overreaction_risk": latest_online.get("overreaction_risk", 0.0),
                    },
                    "execution_diagnostics": execution_diagnostics,
                    "post_trade_learning": {
                        "active": bool(post_trade_learning),
                        "latest_count": len(latest_post_trade),
                        "total_closed_trades_seen": int((post_trade_learning or {}).get("total_closed_trades_seen", 0) or 0),
                        "top_patterns": list((post_trade_learning or {}).get("top_patterns", []) or [])[:3],
                        "policy_note": str((post_trade_learning or {}).get("policy_note", "")),
                    },
                    "mt5": mt5_public,
                    "atlas_activity": {
                        "recent_deal_count": atlas_deal_count,
                        "recent_closed_count": atlas_closed_count,
                        "recent_realized_profit": atlas_realized_profit,
                        "latest_deal": latest_atlas_deal,
                        "new_deal_count": len(new_atlas_deals),
                    },
                    "untagged_activity": {
                        "new_deal_count": len(new_suspicious),
                        "recent_suspicious_count": suspicious_deal_count,
                        "recent_suspicious_volume": suspicious_volume,
                        "recent_suspicious_profit": suspicious_profit,
                    },
                }
            )
        except Exception as exc:
            log(f"python_monitor_error {exc}")
            insight({"event": "python_monitor_error", "error": str(exc)})
        time.sleep(interval)


if __name__ == "__main__":
    main()
