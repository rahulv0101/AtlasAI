from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UI_DIR = ROOT / "atlas_ai" / "ui"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_operator_dashboard_panels_exist() -> None:
    index_html = _read(UI_DIR / "index.html")
    assert "Operator Dashboard" in index_html
    assert "System Snapshot" in index_html
    assert "Safety-First Recommendation" in index_html
    assert "Performance Feedback" in index_html
    assert "Paper-Mode Tuning" in index_html
    assert "Live Pilot Review" in index_html
    assert "Control Queue" in index_html
    assert "Operator Actions" in index_html
    assert "Pilot Runbook" in index_html


def test_operator_dashboard_fact_inference_recommendation_structure() -> None:
    index_html = _read(UI_DIR / "index.html")
    assert 'id="op-snapshot-facts"' in index_html
    assert 'id="op-snapshot-inference"' in index_html
    assert 'id="op-snapshot-recommendation"' in index_html
    assert 'id="op-reco-facts"' in index_html
    assert 'id="op-feedback-facts"' in index_html
    assert 'id="op-tuning-facts"' in index_html
    assert "Facts first. Inference next. Recommendation last." in index_html


def test_operator_actions_are_explicit_and_safety_labeled() -> None:
    index_html = _read(UI_DIR / "index.html")
    app_js = _read(UI_DIR / "app.js")
    assert "No automatic execution" in index_html


def test_teaching_console_assets_are_present() -> None:
    index_html = Path("atlas_ai/ui/index.html").read_text(encoding="utf-8")
    app_js = Path("atlas_ai/ui/app.js").read_text(encoding="utf-8")
    assert "Teach Atlas" in index_html
    assert 'data-teach-label="correct_hold"' in index_html
    assert 'data-teach-label="missed_trade"' in index_html
    assert "/api/teaching/reviews" in app_js
    assert 'id="teach-current-lesson"' in index_html
    assert "structured_lesson" in app_js
    assert "Confidence effect:" in app_js
    assert "recordTeachingReview" in app_js
    assert "Approval required" in index_html
    assert 'id="op-approve-item"' in index_html
    assert 'id="op-approve-all"' in index_html
    assert 'id="op-reject-item"' in index_html
    assert 'id="op-defer-item"' in index_html
    assert 'id="op-show-pending"' in index_html
    assert 'id="op-explain-item"' in index_html
    assert 'id="op-why-item"' in index_html
    assert "approve item" in app_js
    assert "approve all pending items" in app_js
    assert "reject item" in app_js
    assert "defer item" in app_js
    assert "show pending items" in app_js
    assert "explain item" in app_js
    assert "why did you suggest item" in app_js


def test_visual_chart_read_assets_are_present() -> None:
    index_html = _read(UI_DIR / "index.html")
    app_js = _read(UI_DIR / "app.js")
    assert "Visual Chart Read" in index_html
    assert 'id="visual-pattern"' in index_html
    assert 'id="visual-quality"' in index_html
    assert 'id="visual-confirmation"' in index_html
    assert "renderVisualChartRead" in app_js
    assert "visual_chart_read" in app_js


def test_operator_dashboard_uses_existing_endpoints_and_polling() -> None:
    app_js = _read(UI_DIR / "app.js")
    assert 'fetchJson("/api/status/compact")' in app_js
    assert 'fetchJson("/api/runtime")' in app_js
    assert 'fetch("/api/ask-atlas"' in app_js
    assert "const ASK_ATLAS_TIMEOUT_MS = 10000;" in app_js
    assert "const OPERATOR_REFRESH_MS = 60000;" in app_js
    assert "const REFRESH_INTERVAL_MS = 10000;" in app_js
    assert "lastAuxRefreshAt" in app_js
    assert "if (document.hidden) return;" in app_js
    assert "function mergedExecutionState(status, runtime)" in app_js
    assert "const statusExec" in app_js
    assert "automation_truth" in app_js
    assert "health.effective_mode_label" in app_js
    assert "startOperatorPolling();" in app_js
    assert "refreshOperatorPanels();" in app_js
    assert "bindOperatorActions();" in app_js


def test_command_deck_view_and_actions_are_present() -> None:
    index_html = _read(UI_DIR / "index.html")
    app_js = _read(UI_DIR / "app.js")
    style_css = _read(UI_DIR / "style.css")
    assert 'id="view-command"' in index_html
    assert 'id="view-diagnostics"' in index_html
    assert 'id="cmd-resume-automation"' in index_html
    assert 'id="cmd-pause-automation"' in index_html
    assert 'id="cmd-why-blocked"' in index_html
    assert 'id="cmd-what-next"' in index_html
    assert 'id="lane-market"' in index_html
    assert 'id="lane-automation"' in index_html
    assert 'id="lane-session"' in index_html
    assert 'id="lane-broker"' in index_html
    assert 'id="lane-ui"' in index_html
    assert "function setDashboardView(view)" in app_js
    assert "function bindCommandDeckActions()" in app_js
    assert "function renderReadinessLanes(status)" in app_js
    assert "operator_readiness" in app_js
    assert "structured_blocker" in app_js
    assert 'askAtlas(command, 20000)' in app_js
    assert "Command still processing; refreshing status..." in app_js
    assert 'runCommandDeckAction("resume automation")' in app_js
    assert 'runCommandDeckAction("pause automation")' in app_js
    assert 'runCommandDeckAction("what is holding you back right now?")' in app_js
    assert 'runCommandDeckAction("what is the next safe step?")' in app_js
    assert "bindCommandDeckActions();" in app_js
    assert "body.command-view .diagnostics-panel" in style_css
    assert ".command-actions-bar" in style_css
    assert ".readiness-lanes" in style_css


def test_learning_card_shows_last_broker_confirmed_trade() -> None:
    index_html = _read(UI_DIR / "index.html")
    app_js = _read(UI_DIR / "app.js")
    assert 'id="last-trade-outcome"' in index_html
    assert 'id="last-trade-lesson"' in index_html
    assert "last_confirmed_trade" in app_js
    assert "last-trade-pnl" in app_js


def test_command_deck_prefers_blocker_classification_summary() -> None:
    app_js = _read(UI_DIR / "app.js")
    assert "blocker_classification" in app_js
    assert "trace.blocker_summary" in app_js
    assert "blocker_recoverability" in app_js
    assert "Fix Runtime" in app_js
    assert "Safety Review" in app_js
    assert "Review Gate" in app_js
    assert "Wait For Setup" in app_js
    assert "const hadRefreshFailure = refreshFailureCount > 0;" in app_js
    assert 'lastStatusSignature = "";' in app_js
    assert "refreshFailureCount >= 3 ? \"OFFLINE\" : \"RECONNECTING\"" in app_js


def test_ui_mode_label_uses_backend_execution_environment_label_or_display_mode() -> None:
    app_js = _read(UI_DIR / "app.js")
    assert "function resolveDisplayMode(" in app_js
    assert "exec.execution_environment_label || dbg.display_mode" in app_js
    assert '"Unknown (safe fallback)"' in app_js
    assert 'const statModeEl = document.getElementById("stat-mode");' in app_js
    assert "const modeMeta = resolveDisplayModeMeta(execState, null);" in app_js
    assert "MODE: ${modeLabel}" in app_js


def test_ui_mode_label_no_longer_uses_legacy_execute_trades_for_stat_mode() -> None:
    app_js = _read(UI_DIR / "app.js")
    assert '(view.execute_trades ? "LIVE" : "PAPER")' not in app_js


def test_decision_analysis_panel_is_present_and_wired() -> None:
    index_html = _read(UI_DIR / "index.html")
    app_js = _read(UI_DIR / "app.js")
    assert "Decision Analysis" in index_html
    assert 'id="decision-analysis-panel"' in index_html
    assert 'id="da-bias"' in index_html
    assert "Final Bias" in index_html
    assert 'id="da-signal-bias"' in index_html
    assert 'id="da-chart-bias"' in index_html
    assert 'id="da-htf-bias"' in index_html
    assert 'id="da-confidence-score"' in index_html
    assert 'id="da-confidence-level"' in index_html
    assert 'id="da-trend"' in index_html
    assert 'id="da-momentum"' in index_html
    assert 'id="da-location"' in index_html
    assert 'id="da-setup-type"' in index_html
    assert 'id="da-entry-quality"' in index_html
    assert 'id="da-reasoning"' in index_html
    assert 'id="da-warnings"' in index_html
    assert 'id="dp-observe"' in index_html
    assert 'id="dp-orient"' in index_html
    assert 'id="dp-critique"' in index_html
    assert 'id="dp-decide"' in index_html
    assert 'id="setup-radar-watchlist"' in index_html
    assert 'id="setup-radar-timing"' in index_html
    assert 'id="setup-radar-conditions"' in index_html
    assert 'id="da-explanation"' in index_html
    assert "function renderDecisionAnalysis(view)" in app_js
    assert "bias_breakdown" in app_js
    assert 'setText("da-signal-bias"' in app_js
    assert 'setText("da-chart-bias"' in app_js
    assert 'setText("da-htf-bias"' in app_js
    assert "decision_analysis" in app_js
    assert 'renderDetailList("dp-observe"' in app_js
    assert 'renderChipList("dp-critique"' in app_js
    assert "setup_radar" in app_js
    assert "timing_engine" in app_js
    assert 'renderDetailList(\n    "setup-radar-watchlist"' in app_js
    assert 'renderDetailList(\n    "setup-radar-timing"' in app_js
    assert 'renderChipList("setup-radar-conditions"' in app_js
    assert 'renderDecisionAnalysis(view);' in app_js


def test_decision_trace_panel_is_present_and_wired() -> None:
    index_html = _read(UI_DIR / "index.html")
    app_js = _read(UI_DIR / "app.js")
    assert "Decision Trace" in index_html
    assert 'id="decision-trace-panel"' in index_html
    assert 'id="trace-final-action"' in index_html
    assert 'id="trace-final-decision"' in index_html
    assert 'id="trace-selected-candidate"' in index_html
    assert 'id="trace-active-signals"' in index_html
    assert 'id="trace-hard-blockers"' in index_html
    assert 'id="trace-risk-state"' in index_html
    assert 'id="trace-bias-breakdown"' in index_html
    assert 'id="trace-execution-result"' in index_html
    assert "function renderDecisionTrace(view)" in app_js
    assert "decision_trace" in app_js
    assert 'renderDetailList(\n    "trace-active-signals"' in app_js
    assert 'renderChipList(\n    "trace-hard-blockers"' in app_js
    assert 'renderDetailList(\n    "trace-bias-breakdown"' in app_js
    assert "renderDecisionTrace(view);" in app_js
