const dashboard = document.getElementById("dashboard");
const startup = document.getElementById("startup");
const loaderFill = document.getElementById("loader-fill");
const loaderText = document.getElementById("loader-text");
const chart = document.getElementById("gold-chart");
const chartCtx = chart ? chart.getContext("2d") : null;
let priceHistory = [];
let startupDismissed = false;
let firstRefreshDone = false;
const startupMinMs = 3400;
const startupStartedAt = Date.now();
let lastRichStatus = {};
let lastStatusSignature = "";
let lastSignalsSignature = "";
let lastEventsSignature = "";
let lastLearningSignature = "";
let refreshInFlight = false;
let refreshTimer = null;
let refreshFailureCount = 0;
let lastAuxRefreshAt = 0;
let askAtlasHistory = [];
const STATUS_STALE_MS = 5 * 60 * 1000;
const EVENTS_WINDOW_MS = 30 * 60 * 1000;
const REFRESH_INTERVAL_MS = 10000;
const FETCH_TIMEOUT_MS = 12000;
const ASK_ATLAS_TIMEOUT_MS = 10000;
const OPERATOR_REFRESH_MS = 60000;
const OPERATOR_QUERIES = {
  recommendation: "What is the next safe step?",
  feedback: "How well are you adapting?",
  tuning: "What small tuning do you recommend?",
  live_review: "Show me the live pilot review",
  queue: "show pending items",
};
let operatorPanelCache = {
  recommendation: null,
  feedback: null,
  tuning: null,
  queue: null,
  last_update_iso: null,
};
let selfImprovementCache = {
  proposals: null,
  rollouts: null,
  audit: null,
  reviewPacket: null,
  last_update_iso: null,
};
let aiCoreCache = {
  latest: null,
  reviews: null,
  preferences: null,
  promotion: null,
  researchQueue: null,
  last_update_iso: null,
};
let operatorRefreshInFlight = false;
let operatorRefreshTimer = null;
let dashboardView = "command";

let pct = 0;
const loaderTimer = setInterval(() => {
  pct = Math.min(100, pct + Math.ceil(Math.random() * 8));
  loaderFill.style.width = `${pct}%`;
  loaderText.textContent = `${pct}%`;
  if (pct >= 100) clearInterval(loaderTimer);
}, 160);

function dismissStartup() {
  if (startupDismissed) return;
  if (!firstRefreshDone) return;
  if (Date.now() - startupStartedAt < startupMinMs) return;
  startupDismissed = true;
  dashboard.classList.remove("hidden");
  dashboard.style.opacity = "1";
  dashboard.style.transform = "none";
  if (startup) {
    startup.style.opacity = "0";
    startup.style.visibility = "hidden";
    startup.style.pointerEvents = "none";
    startup.style.display = "none";
  }
}

setInterval(dismissStartup, 120);

async function fetchJson(url) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);
  const requestUrl = `${url}${url.includes("?") ? "&" : "?"}_ts=${Date.now()}`;
  try {
    const res = await fetch(requestUrl, {
      cache: "no-store",
      headers: { "Cache-Control": "no-store" },
      signal: controller.signal,
    });
    if (!res.ok) {
      throw new Error(`${url} returned ${res.status}`);
    }
    const payload = await res.json();
    console.debug(`[Atlas UI] fetch ok: ${url}`);
    return payload;
  } catch (err) {
    console.warn(`[Atlas UI] fetch failed: ${url}`, err);
    throw err;
  } finally {
    clearTimeout(timeoutId);
  }
}

async function loadStatus() { return fetchJson("/api/status/compact"); }
async function loadEvents() { return fetchJson("/api/events"); }
async function loadRuntime() { return fetchJson("/api/runtime"); }
async function loadLearningInsights() { return fetchJson("/api/learning-insights"); }
async function loadSelfImprovementProposals() { return fetchJson("/api/self-improvement/proposals"); }
async function loadSelfImprovementRollouts() { return fetchJson("/api/self-improvement/rollouts"); }
async function loadSelfImprovementAudit() { return fetchJson("/api/self-improvement/audit"); }
async function loadSelfImprovementOperatorReview(proposalId) {
  return fetchJson(`/api/self-improvement/proposals/${encodeURIComponent(proposalId)}/operator-review`);
}
async function loadAIDecisionLatest() { return fetchJson("/api/ai/decision/latest"); }
async function loadAIDecisionReviews() { return fetchJson("/api/ai/decision/reviews?limit=8"); }
async function loadAIStrategyPreferences() { return fetchJson("/api/ai/strategy/preferences"); }
async function loadAIPromotionRecommendations() { return fetchJson("/api/ai/promotion/recommendations"); }
async function loadAIResearchQueue() { return fetchJson("/api/ai/research/queue"); }
async function loadTeachingDigest() { return fetchJson("/api/teaching/digest"); }
async function askAtlas(query, timeoutMs = ASK_ATLAS_TIMEOUT_MS) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch("/api/ask-atlas", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
      signal: controller.signal,
    });
    if (!res.ok) {
      throw new Error(`/api/ask-atlas returned ${res.status}`);
    }
    return res.json();
  } finally {
    clearTimeout(timeoutId);
  }
}

async function postJson(url, payload = {}) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok || data.ok === false) {
    const message = compactText(data.error || data.reason || `${url} failed`, `${url} failed`);
    throw new Error(message);
  }
  return data;
}

async function recordTeachingReview(label) {
  const noteEl = document.getElementById("teach-note");
  const statusEl = document.getElementById("teach-status");
  if (statusEl) statusEl.textContent = "Recording teaching label...";
  const data = await postJson("/api/teaching/reviews", {
    label,
    note: noteEl ? noteEl.value : "",
    operator: "operator",
  });
  const review = (data || {}).review || {};
  if (statusEl) {
    statusEl.textContent = `Recorded ${compactText(review.label, label)} for ${compactText(review.playbook_name, "current decision")}.`;
  }
  if (noteEl) noteEl.value = "";
  loadTeachingDigest().then(renderTeachingDigest).catch(() => {});
  return data;
}

function renderTeachingDigest(payload) {
  const prompt = ((payload || {}).current_decision_prompt || {});
  const lesson = ((prompt || {}).structured_lesson || {});
  const statusEl = document.getElementById("teach-status");
  if (!statusEl) return;
  const sampleCount = Number((payload || {}).sample_count || 0);
  const suggested = compactText(prompt.suggested_label, "no suggestion");
  const reason = compactText(prompt.reason, "Teaching prompt unavailable.");
  statusEl.textContent = `Suggested label: ${suggested}. ${reason} Samples: ${sampleCount}.`;
  renderDetailList(
    "teach-current-lesson",
    [
      `Condition: ${compactText(lesson.condition, "waiting for decision")}`,
      `Blocker: ${compactText(lesson.blocker, "none")}`,
      `Correction: ${compactText(lesson.correction, "wait for a cleaner setup")}`,
      `Confidence effect: ${compactText(lesson.confidence_effect, "no hidden change")}`,
      `Evidence required: ${Array.isArray(lesson.evidence_required) ? lesson.evidence_required.join(", ") : compactText(lesson.evidence_required, "-")}`,
    ],
    "No structured lesson yet"
  );
}

function getSettledValue(result, fallback) {
  return result && result.status === "fulfilled" ? result.value : fallback;
}
function fmt(v, digits = 2) { if (v === undefined || v === null || Number.isNaN(v)) return "-"; return Number(v).toFixed(digits); }
function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;")
    .replace(/'/g, "&#39;");
}
function fmtText(value, fallback = "-") {
  const text = String(value ?? "").trim();
  return text || fallback;
}
function renderChipList(elementId, items, emptyLabel = "None") {
  const el = document.getElementById(elementId);
  if (!el) return;
  const values = Array.isArray(items)
    ? items.map((item) => String(item ?? "").trim()).filter(Boolean)
    : [];
  if (!values.length) {
    el.innerHTML = `<span class="chip muted-chip">${escapeHtml(emptyLabel)}</span>`;
    return;
  }
  el.innerHTML = values.map((item) => `<span class="chip">${escapeHtml(item)}</span>`).join("");
}
function renderWarningChipList(elementId, items, emptyLabel = "No active warnings") {
  const el = document.getElementById(elementId);
  if (!el) return;
  const values = Array.isArray(items)
    ? items.map((item) => String(item ?? "").trim()).filter(Boolean)
    : [];
  if (!values.length) {
    el.innerHTML = `<span class="chip muted-chip">${escapeHtml(emptyLabel)}</span>`;
    return;
  }
  el.innerHTML = values.map((item) => `<span class="chip warning-chip">${escapeHtml(item)}</span>`).join("");
}
function parseTimeMs(value) {
  const t = Date.parse(String(value || ""));
  return Number.isFinite(t) ? t : 0;
}
function isFreshTime(value, maxAgeMs = STATUS_STALE_MS) {
  const t = parseTimeMs(value);
  if (!t) return false;
  return (Date.now() - t) <= maxAgeMs;
}
function getFreshnessRuntime(view) {
  const freshness = (((view || {}).freshness_runtime) || {});
  const state = String(freshness.state || "").toLowerCase();
  const primaryAgeMs = Number(freshness.primary_age_ms);
  const statusAgeMs = Number(freshness.status_age_ms);
  return {
    state: (state === "fresh" || state === "aging" || state === "stale") ? state : "unknown",
    primaryAgeMs,
    statusAgeMs,
    staleThresholdMs: Number(freshness.stale_threshold_ms),
    agingThresholdMs: Number(freshness.aging_threshold_ms),
    source: String(freshness.source || ""),
  };
}
function compactGeminiNote(note) {
  const text = String(note || "").trim();
  if (!text) return "Gemini note not available.";
  const lower = text.toLowerCase();
  if (lower.includes("temporarily paused")) return "Gemini temporarily paused due to API quota/rate limits.";
  if (lower.includes("gemini error")) return "Gemini unavailable due to API quota/rate limits.";
  return text;
}

function isActionableTrade(view) {
  const side = String(((view || {}).side) || "").toLowerCase();
  const confidence = Number(((view || {}).confidence) || 0);
  return (side === "buy" || side === "sell") && confidence > 0;
}

function stableStringify(value) {
  try {
    return JSON.stringify(value ?? null);
  } catch {
    return String(value ?? "");
  }
}

function setGeminiState(enabled, note) {
  const badge = document.getElementById("gemini-status");
  const unavailable = !enabled || (note || "").toLowerCase().includes("gemini error") || (note || "").toLowerCase().includes("disabled");
  badge.textContent = unavailable ? "GEMINI: UNAVAILABLE" : "GEMINI: ONLINE";
  badge.classList.toggle("muted-pill", unavailable);
}

function renderDecisionExplanation(view) {
  const explanation = ((view || {}).decision_explanation || {});
  const fallbackTimestamp = (view || {}).updated_at || "-";
  const fields = [
    ["decision-candidate-id", fmtText(explanation.candidate_id)],
    ["decision-trade-id", fmtText(explanation.trade_id)],
    ["decision-final-decision", fmtText(explanation.final_decision)],
    ["decision-lane", fmtText(explanation.lane || (view || {}).lane)],
    ["decision-mode", fmtText(explanation.mode || ((view || {}).execute_trades ? "live" : "paper"))],
    ["decision-symbol", fmtText(explanation.symbol || (view || {}).symbol)],
    ["decision-signal-source", fmtText(explanation.signal_source)],
    ["decision-regime", fmtText(explanation.regime)],
    ["decision-session", fmtText(explanation.session)],
    ["decision-trend-strength", fmtText(explanation.trend_strength)],
    ["decision-timestamp", fmtText(explanation.timestamp || fallbackTimestamp)],
  ];
  fields.forEach(([id, value]) => {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
  });
  const explanationText = document.getElementById("decision-explanation-text");
  if (explanationText) explanationText.textContent = fmtText(explanation.explanation_text, "No decision explanation available yet.");
  renderChipList("decision-reason-codes", explanation.reason_codes, "No reason codes");
  renderChipList("decision-setup-quality", explanation.setup_quality_reasons, "No setup quality reasons");
  renderChipList("decision-active-guards", explanation.active_guards, "No active guards");
  renderChipList("decision-penalties", explanation.penalties, "No penalties");
}

function renderDecisionTrace(view) {
  const trace = ((view || {}).decision_trace || {});
  const selected = ((trace || {}).selected_candidate || {});
  const context = ((trace || {}).context || {});
  const filters = ((trace || {}).filters || {});
  const biasBreakdown = ((trace || {}).bias_breakdown || {});
  const risk = ((trace || {}).risk || {});
  const gate = ((trace || {}).execution_gate || {});
  const gateResult = ((gate || {}).result || {});
  const signal = ((trace || {}).signal || {});
  const activeSignals = Array.isArray(signal.active) ? signal.active : [];
  const hardBlockers = Array.isArray(filters.hard_blockers) ? filters.hard_blockers : [];
  const candidateLabel = `${compactText(selected.side, "flat")} / ${compactText(selected.strategy, "none")}`;
  const gateLabel = Boolean(gate.execute_trades)
    ? compactText(gate.environment, "execution enabled")
    : "paper/read-only";
  const fields = [
    ["trace-final-action", compactText(trace.final_action, "-").toUpperCase()],
    ["trace-final-decision", compactText(trace.final_decision, "-")],
    ["trace-selected-candidate", candidateLabel],
    ["trace-confidence", fmt(selected.confidence, 4)],
    ["trace-session", compactText(context.session, "-")],
    ["trace-regime", compactText(context.regime, "-")],
    ["trace-spread", `${compactText(context.spread_state, "unknown")} / ${fmt(risk.spread_points, 1)} pts`],
    ["trace-execution-gate", gateLabel],
  ];
  fields.forEach(([id, value]) => setText(id, value));
  setText("trace-message", compactText(trace.message, "No decision trace available yet."));
  renderDetailList(
    "trace-active-signals",
    activeSignals.slice(0, 6).map((row) => {
      const side = compactText(row.side, "flat");
      const strategy = compactText(row.strategy, "unknown");
      const confidence = Number.isFinite(Number(row.confidence)) ? Number(row.confidence).toFixed(3) : "-";
      const score = Number.isFinite(Number(row.score)) ? Number(row.score).toFixed(3) : "-";
      return `${strategy} | ${side} | conf ${confidence} | score ${score}`;
    }),
    "No active signals"
  );
  renderChipList(
    "trace-hard-blockers",
    hardBlockers.map((row) => {
      const name = compactText((row || {}).name, "blocker");
      const reason = compactText((row || {}).reason, "");
      return reason && reason !== "-" ? `${name}: ${reason}` : name;
    }),
    "No hard blockers"
  );
  renderDetailList(
    "trace-risk-state",
    [
      `Daily loss: ${fmt(risk.daily_loss_pct, 4)}`,
      `Weekly loss: ${fmt(risk.weekly_loss_pct, 4)}`,
      `Open positions: ${compactText(risk.open_positions, "-")}`,
      `Trades today: ${compactText(risk.trades_today, "-")}`,
      `Consecutive losses: ${compactText(risk.consecutive_losses, "-")}`,
    ],
    "No risk trace"
  );
  renderDetailList(
    "trace-bias-breakdown",
    [
      `Signal: ${compactText(biasBreakdown.signal_bias, "neutral")}`,
      `Chart structure: ${compactText(biasBreakdown.chart_structure_bias, "neutral")}`,
      `Higher timeframe: ${compactText(biasBreakdown.higher_timeframe_bias, "unknown")}`,
      `Final trade: ${compactText(biasBreakdown.final_trade_bias, "neutral")}`,
      `Conflict: ${biasBreakdown.conflict === undefined ? "-" : (biasBreakdown.conflict ? "yes" : "no")}`,
    ],
    "No bias breakdown"
  );
  renderDetailList(
    "trace-execution-result",
    [
      `Success: ${gateResult.success === undefined ? "-" : String(Boolean(gateResult.success))}`,
      `Retcode: ${compactText(gateResult.retcode, "-")}`,
      `Comment: ${compactText(gateResult.comment, "-")}`,
      `Decision ID: ${compactText(gate.decision_id || gateResult.decision_id, "-")}`,
    ],
    "No execution result"
  );
}

function summarizeObjectChips(obj, fallback = "Not available") {
  const source = obj && typeof obj === "object" ? obj : {};
  const items = Object.entries(source)
    .filter(([, value]) => value !== undefined && value !== null && String(value).trim() !== "")
    .map(([key, value]) => `${key}: ${String(value)}`);
  return items.length ? items : [fallback];
}

function renderSystemHealth(view) {
  const health = ((view || {}).system_health || {});
  const safetyLayer = ((view || {}).safety_layer || {});
  const broker = ((health || {}).broker_connectivity_status || {});
  const filterState = ((health || {}).filter_state || {});
  const mlState = ((health || {}).ml_state || {});
  const geminiState = ((health || {}).gemini_state || {});
  const fields = [
    ["health-effective-mode", fmtText(health.effective_mode_label || health.effective_mode)],
    ["health-paper-confirmed", health.paper_mode_confirmed === undefined ? "-" : (health.paper_mode_confirmed ? "YES" : "NO")],
    ["health-broker-connectivity", broker.connected ? "CONNECTED" : "DISCONNECTED"],
    ["health-routing-risk", health.routing_risk_detected ? "DETECTED" : "CLEAR"],
    ["health-stale-data", health.stale_data_detected ? "STALE" : "FRESH"],
    ["health-config-mismatch", health.config_mismatch_detected ? "DETECTED" : "CLEAR"],
    ["health-safety-lock", safetyLayer.safety_lock_active ? "ACTIVE" : "CLEAR"],
    ["health-cooldown", safetyLayer.cooldown_active ? compactText(safetyLayer.cooldown_level, "active") : "CLEAR"],
    ["health-volatility", compactText(safetyLayer.volatility_state, "normal").toUpperCase()],
    ["health-recovery-mode", compactText(safetyLayer.recovery_mode, "normal")],
  ];
  fields.forEach(([id, value]) => {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
  });
  renderWarningChipList("health-warnings", health.warnings, "No active warnings");
  renderChipList("health-filter-state", summarizeObjectChips(filterState, "No filter state"), "No filter state");
  renderChipList("health-ml-state", summarizeObjectChips(mlState, "No ML state"), "No ML state");
  renderChipList("health-gemini-state", summarizeObjectChips({
    enabled: geminiState.enabled,
    influence_enabled: geminiState.influence_enabled,
  }, "No Gemini state"), "No Gemini state");
}

function renderSelfImprovementPanel(runtime, proposalsPayload, rolloutsPayload, auditPayload) {
  const runtimeGuard = (((runtime || {}).self_improvement_runtime) || {});
  const summary = (((runtime || {}).self_improvement_state) || {});
  const proposals = Array.isArray((proposalsPayload || {}).proposals) ? proposalsPayload.proposals : [];
  const rollouts = Array.isArray((rolloutsPayload || {}).rollouts) ? rolloutsPayload.rollouts : [];
  const audit = Array.isArray(auditPayload) ? auditPayload : [];

  setText("si-active-registry-version", compactText(runtimeGuard.active_rule_registry_version, compactText(summary.active_rule_version, "-")));
  setText("si-active-rule-version-id", compactText(runtimeGuard.active_rule_version_id, "-"));
  setText("si-latest-proposal-id", compactText(summary.latest_proposal_id, "-"));
  setText("si-latest-rollout-id", compactText(summary.latest_rollout_id, "-"));
  setText("si-approval-pending", String(Number(summary.approval_pending_count || 0)));
  setText("si-rollback-available", Boolean(summary.rollback_available) ? "YES" : "NO");
  const pendingReview = proposals.find((proposal) => {
    const status = String((proposal || {}).status || "").toLowerCase();
    const missing = (((proposal || {}).validation || {}).missing) || [];
    return status === "awaiting_approval" || missing.includes("operator_review");
  }) || {};
  const pendingEvidence = ((pendingReview || {}).evidence) || {};
  const pendingMetrics = ((pendingEvidence || {}).metrics) || {};
  const pendingScope = ((pendingReview || {}).scope) || {};
  const pendingProposalId = compactText(summary.pending_proposal_id || pendingReview.proposal_id, "-");
  const pendingTargetBlocker = compactText(summary.pending_proposal_target_blocker || pendingScope.target_blocker, "-");
  const pendingReviewLine = pendingReview.proposal_id
    ? `${compactText(pendingReview.proposal_id, "unknown")} | ${compactText(pendingReview.status, "awaiting_approval")} | samples ${Number(pendingEvidence.sample_count || 0)} | net ${fmt(pendingMetrics.net_review_edge, 3)} | explicit approval required`
    : "No proposal waiting for operator review";
  setText("si-pending-proposal-id", pendingProposalId);
  setText("si-pending-target-blocker", pendingTargetBlocker);
  setText("si-pending-operator-review", pendingReviewLine);

  renderDetailList(
    "si-runtime-guard",
    [
      `Execution source: ${compactText(runtimeGuard.execution_rule_source, "approved_registry_only")}`,
      `Approved registry loaded: ${Boolean(runtimeGuard.approved_registry_loaded) ? "yes" : "no"}`,
      `Draft rules loaded into execution: ${Boolean(runtimeGuard.draft_rules_loaded_into_execution) ? "yes" : "no"}`,
      `Promoted rule count: ${Number(runtimeGuard.promoted_rule_count || 0)}`,
      `Rollout active: ${Boolean(runtimeGuard.rollout_active) ? "yes" : "no"}`,
      `Pending proposal: ${compactText(runtimeGuard.pending_proposal_id || summary.pending_proposal_id, "none")}`,
      `Pending target: ${compactText(runtimeGuard.pending_proposal_target_blocker || summary.pending_proposal_target_blocker, "none")}`,
      `Latest promoted proposal: ${compactText(runtimeGuard.latest_promoted_proposal_id, "none")}`,
    ],
    "No self-improvement runtime guard data"
  );

  renderDetailList(
    "si-proposals",
    proposals.slice(-6).reverse().map((proposal) => {
      const id = compactText(proposal.proposal_id, "unknown");
      const status = compactText(proposal.status, "draft");
      const category = compactText(proposal.category, "unknown");
      const strength = Number.isFinite(Number((((proposal || {}).evidence || {}).strength))) ? Number(proposal.evidence.strength).toFixed(2) : "-";
      return `${id} | ${status} | ${category} | strength ${strength}`;
    }),
    "No proposals yet"
  );

  renderDetailList(
    "si-rollouts",
    rollouts.slice(-6).reverse().map((rollout) => {
      const id = compactText(rollout.rollout_id, "unknown");
      const status = compactText(rollout.status, "unknown");
      const proposalId = compactText(rollout.proposal_id, "none");
      const snapshotId = compactText(rollout.snapshot_id, "none");
      return `${id} | ${status} | proposal ${proposalId} | snapshot ${snapshotId}`;
    }),
    "No rollouts yet"
  );

  renderDetailList(
    "si-audit",
    audit.slice(-8).reverse().map((row) => {
      const event = compactText(row.event, "event");
      const proposalId = compactText(row.proposal_id, "none");
      const rolloutId = compactText(row.rollout_id, "none");
      return `${event} | proposal ${proposalId} | rollout ${rolloutId}`;
    }),
    "No audit history yet"
  );

  const proposalInput = document.getElementById("si-proposal-id");
  const defaultProposalId = summary.pending_proposal_id || summary.latest_proposal_id;
  if (proposalInput && !proposalInput.value && defaultProposalId) {
    proposalInput.value = defaultProposalId;
  }
  const rolloutInput = document.getElementById("si-rollout-id");
  if (rolloutInput && !rolloutInput.value && summary.latest_rollout_id) {
    rolloutInput.value = summary.latest_rollout_id;
  }
}

function renderSelfImprovementReviewPacket(packet) {
  const el = document.getElementById("si-review-packet");
  if (!el) return;
  if (!packet || !packet.ok) {
    renderDetailList("si-review-packet", [], "No review packet loaded");
    return;
  }
  const collector = (((packet || {}).evidence || {}).collector) || {};
  const shadow = (((packet || {}).evidence || {}).shadow_compare) || {};
  const validation = ((packet || {}).validation) || {};
  const approval = ((packet || {}).approval) || {};
  const safety = ((packet || {}).safety) || {};
  const posture = ((packet || {}).operator_posture) || {};
  const postureInputs = ((posture || {}).inputs) || {};
  renderDetailList(
    "si-review-packet",
    [
      `Status: ${compactText(packet.status, "unknown")} | recommendation: ${compactText(packet.recommendation, "review")}`,
      `Operator posture: ${compactText(posture.posture, "review")} | ${compactText(posture.reason, "review evidence before action")}`,
      `Suggested operator action: ${compactText(posture.suggested_operator_action, "review_packet")}`,
      `Posture inputs: collector ${fmt(postureInputs.collector_net_review_edge, 3)} | replay ${fmt(postureInputs.replay_net_review_edge, 3)} | shadow ${fmt(postureInputs.shadow_net_review_edge, 3)} | regression ${Boolean(postureInputs.regression_ok) ? "ok" : "not-ok"}`,
      `Evidence: ${Number(collector.enriched_samples || 0)}/${Number(collector.required_samples || 0)} enriched | outcomes ${Number(collector.reviewed_outcome_samples || 0)} | net ${fmt(((collector.metrics || {}).net_review_edge), 3)}`,
      `Shadow compare: ${compactText(shadow.decision, "unknown")} | capture ${fmt(shadow.capture_rate, 3)} | false unlock ${fmt(shadow.false_unlock_risk_rate, 3)}`,
      `Validation: completed ${(validation.completed || []).join(", ") || "none"} | missing ${(validation.missing || []).join(", ") || "none"}`,
      `Approval: required=${Boolean(approval.required) ? "yes" : "no"} approved=${Boolean(approval.approved) ? "yes" : "no"}`,
      `Safety: approve=${Boolean(safety.does_not_approve) ? "separate explicit action" : "unknown"} | execute=${Boolean(safety.does_not_execute) ? "no" : "unknown"}`,
    ],
    "No review packet loaded"
  );
}

function renderAICorePanel(latestPayload, reviewsPayload, preferencesPayload, promotionPayload, researchQueuePayload) {
  const latest = ((latestPayload || {}).latest || {});
  const trace = ((latest || {}).trace || {});
  const recommendation = ((trace || {}).recommendation || {});
  const blockers = ((trace || {}).blocker_analysis || {});
  const strategyInvention = ((trace || {}).strategy_invention || {});
  const reviews = Array.isArray((reviewsPayload || {}).reviews) ? reviewsPayload.reviews : [];
  const matchingReviews = reviews.filter((row) =>
    row.schema === "atlas_ai_decision_review_v2" &&
    !row.legacy_unverified &&
    (latest.decision_id
      ? row.decision_id === latest.decision_id
      : row.decision_recorded_at === latest.recorded_at &&
        row.source === latest.source && row.query === latest.query)
  );
  const latestReview = matchingReviews.length ? matchingReviews[matchingReviews.length - 1] : {};
  const preferences = ((preferencesPayload || {}).preferences || {});
  const ranking = preferences.schema === "atlas_ai_strategy_preferences_v2" &&
    Array.isArray(preferences.ranking) ? preferences.ranking : [];
  const topPreference = ranking.length ? ranking[0] : {};
  const promotion = ((promotionPayload || {}).recommendations || {});
  const topPromotion = ((promotion || {}).top_recommendation || {});
  const queuePayload = ((researchQueuePayload || {}).queue || {});
  const queuePolicy = ((queuePayload || {}).policy || {});
  const queueItems = Array.isArray(queuePayload.items) ? queuePayload.items : [];

  setText("ai-latest-action", compactText(recommendation.action, "-").replace(/_/g, " "));
  const confidence = Number(recommendation.confidence);
  setText("ai-decision-confidence", Number.isFinite(confidence) ? confidence.toFixed(3) : "-");
  setText("ai-primary-blocker", compactText(blockers.primary_blocker, "none").replace(/_/g, " "));
  setText("ai-review-outcome", compactText(latestReview.outcome, "unverified / not reviewed").replace(/_/g, " "));
  setText("ai-top-archetype", Number(topPreference.score) > 0
    ? compactText(topPreference.archetype, "-").replace(/_/g, " ") : "No positive verified preference");
  setText("ai-promotion-action", compactText(topPromotion.recommendation, "-").replace(/_/g, " "));

  renderDetailList(
    "ai-decision-reason",
    [
      `Recorded: ${compactText(latest.recorded_at, "unknown")} | source: ${compactText(latest.source, "unknown")}`,
      `Query: ${compactText(latest.query, "none")}`,
      compactText(recommendation.reason, "No decision trace recorded yet."),
      `Trade bias: ${compactText(recommendation.trade_bias, "neutral").replace(/_/g, " ")}`,
      `Invented candidates: ${Number(strategyInvention.candidate_count || 0)}`,
    ],
    "No decision trace recorded yet"
  );

  renderDetailList(
    "ai-review-detail",
    matchingReviews.slice(-5).reverse().map((row) => {
      const outcome = compactText(row.outcome, "pending").replace(/_/g, " ");
      const quality = compactText(row.blocker_quality, "pending").replace(/_/g, " ");
      const lesson = compactText(row.lesson, "No lesson recorded.");
      return `Reviewed: ${compactText(row.reviewed_at, "unknown")} | ${outcome} | ${quality} | ${lesson}`;
    }),
    "No verified-format review matched to this decision. Legacy or unrelated reviews are not outcome evidence."
  );

  renderDetailList(
    "ai-preference-ranking",
    ranking.slice(0, 6).map((row) => {
      const archetype = compactText(row.archetype, "unknown").replace(/_/g, " ");
      const score = Number(row.score);
      return `${archetype} | score ${Number.isFinite(score) ? score.toFixed(2) : "-"}`;
    }),
    "No verified preference evidence. Legacy scores are excluded."
  );

  renderDetailList(
    "ai-promotion-detail",
    [
      `Report timestamp: ${compactText(promotion.updated_at, "unknown")} | separate research assessment`,
      `Top UID: ${compactText(topPromotion.uid, "none")}`,
      `Stage: ${compactText(topPromotion.stage, "none").replace(/_/g, " ")}`,
      `Confidence: ${Number.isFinite(Number(topPromotion.confidence)) ? Number(topPromotion.confidence).toFixed(3) : "-"}`,
      `Safety: ${(((promotion || {}).safety || {}).does_not_execute === false) ? "review policy issue" : "recommendation only"}`,
    ],
    "No promotion recommendation yet"
  );

  renderDetailList(
    "ai-research-queue",
    [
      `Shadow drawdown policy: ${compactText(queuePolicy.shadow_drawdown_policy, "raw_peak").replace(/_/g, " ")}`,
      `Shadow drawdown limit: ${Number.isFinite(Number(queuePolicy.shadow_drawdown_limit_pct)) ? `${(Number(queuePolicy.shadow_drawdown_limit_pct) * 100).toFixed(2)}%` : "-"}`,
      ...queueItems.slice(0, 6).map((item) => {
      const id = compactText(item.id, "unknown");
      const status = compactText(item.status, "pending_review").replace(/_/g, " ");
      const rec = compactText(item.recommendation, "research").replace(/_/g, " ");
      const packet = ((item || {}).demo_review_packet || {});
      const packetFlag = packet.path ? " | packet ready" : "";
      return `${id} | ${status} | ${rec}${packetFlag}`;
      }),
    ],
    "No AI research queue items yet"
  );
  const researchInput = document.getElementById("ai-research-item-id");
  if (researchInput && !researchInput.value && queueItems.length) {
    researchInput.value = compactText(queueItems[0].id, "");
  }
}

function setAnalysisPillClass(elementId, kind, value) {
  const el = document.getElementById(elementId);
  if (!el) return;
  el.classList.remove("analysis-buy", "analysis-sell", "analysis-neutral", "analysis-good", "analysis-risky", "analysis-avoid");
  const normalized = String(value || "").toLowerCase();
  if (kind === "bias") {
    if (normalized === "buy") el.classList.add("analysis-buy");
    else if (normalized === "sell") el.classList.add("analysis-sell");
    else el.classList.add("analysis-neutral");
    return;
  }
  if (kind === "entry") {
    if (normalized === "good") el.classList.add("analysis-good");
    else if (normalized === "risky") el.classList.add("analysis-risky");
    else el.classList.add("analysis-avoid");
    return;
  }
  if (kind === "confidence") {
    if (normalized === "high") el.classList.add("analysis-good");
    else if (normalized === "medium") el.classList.add("analysis-risky");
    else el.classList.add("analysis-neutral");
  }
}

function renderDecisionAnalysis(view) {
  const analysis = ((((view || {}).decision_analysis) || {}));
  const hasAnalysis = analysis && typeof analysis === "object" && Object.keys(analysis).length > 0;
  if (!hasAnalysis) {
    setText("da-bias", "NEUTRAL");
    setText("da-confidence-score", "- / 100");
    setText("da-confluence-score", "- / 100");
    setText("da-confidence-level", "LOW");
    setText("da-trend", "-");
    setText("da-momentum", "-");
    setText("da-location", "-");
    setText("da-setup-type", "no_trade");
    setText("da-regime", "-");
    setText("da-reversal", "-");
    setText("da-session-quality", "-");
    setText("da-account-tier", "-");
    setText("da-risk-profile", "-");
    setText("da-size-tier", "-");
    setText("da-recommended-action", "hold");
    setText("da-entry-quality", "AVOID");
    setAnalysisPillClass("da-bias", "bias", "neutral");
    setAnalysisPillClass("da-confidence-level", "confidence", "low");
    setAnalysisPillClass("da-entry-quality", "entry", "avoid");
    renderChipList("da-blockers", [], "No active blockers");
    renderDetailList("da-reasoning", ["Decision analysis unavailable"], "Decision analysis unavailable");
    renderWarningChipList("da-warnings", [], "No active warnings");
    renderDetailList("dp-observe", ["Observe stage unavailable"], "Observe stage unavailable");
    renderDetailList("dp-orient", ["Orient stage unavailable"], "Orient stage unavailable");
    renderChipList("dp-critique", [], "No critique reasons");
    renderDetailList("dp-decide", ["Decision stage unavailable"], "Decision stage unavailable");
    renderDetailList("setup-radar-watchlist", ["Setup radar unavailable"], "Setup radar unavailable");
    renderDetailList("setup-radar-timing", ["Timing engine unavailable"], "Timing engine unavailable");
    renderChipList("setup-radar-conditions", [], "No slowing conditions");
    setText("da-explanation", "Decision analysis unavailable.");
    renderMacroTerminal({});
    return;
  }

  const biasBreakdown = ((analysis || {}).bias_breakdown || (((analysis || {}).trading_response || {}).bias_breakdown || {}));
  const signalBias = compactText((analysis || {}).signal_bias || biasBreakdown.signal_bias, "neutral").toLowerCase();
  const chartBias = compactText((analysis || {}).chart_structure_bias || biasBreakdown.chart_structure_bias, "neutral").toLowerCase();
  const htfBias = compactText((analysis || {}).higher_timeframe_bias || biasBreakdown.higher_timeframe_bias, "unknown").toLowerCase();
  const finalBias = compactText((analysis || {}).final_trade_bias || biasBreakdown.final_trade_bias || (((analysis || {}).trading_response || {}).bias), "neutral").toLowerCase();
  const bias = finalBias;
  const confidenceScore = Number(analysis.confidence_score);
  const confidenceLevel = compactText(analysis.confidence_level, "low").toLowerCase();
  const trend = (analysis.trend && typeof analysis.trend === "object")
    ? compactText(analysis.trend.state, compactText(analysis.trend.structure, "unclear"))
    : "unclear";
  const momentum = (analysis.momentum && typeof analysis.momentum === "object")
    ? compactText(analysis.momentum.state, "mixed")
    : "mixed";
  const location = (analysis.location && typeof analysis.location === "object")
    ? compactText(analysis.location.state, "unclear")
    : "unclear";
  const setupType = compactText(analysis.setup_type, "no_trade");
  const regime = compactText((((analysis || {}).market_regime || {}).regime), "unknown");
  const accountTier = compactText((((analysis || {}).account_profile || {}).account_tier), "standard_account");
  const riskProfile = compactText((((analysis || {}).account_profile || {}).risk_profile), "capital_preservation_standard");
  const sizeTier = compactText((((analysis || {}).account_profile || {}).current_size_tier), "-");
  const reversalState = Boolean((((analysis || {}).reversal_detection || {}).confirmed_reversal))
    ? "confirmed"
    : (Boolean((((analysis || {}).reversal_detection || {}).reversal_candidate)) ? "candidate" : "none");
  const confluenceScore = Number(((analysis || {}).confluence || {}).score);
  const recommendedAction = compactText(analysis.recommended_action, "hold");
  const sessionQuality = compactText(((((analysis || {}).context_alignment || {}).session || {}).state_class), "unknown");
  const blockers = Array.isArray(analysis.hard_blockers)
    ? analysis.hard_blockers
    : (Array.isArray(analysis.blockers) ? analysis.blockers : []);
  const topBlockers = Array.isArray(analysis.top_blockers) ? analysis.top_blockers.slice(0, 3) : [];
  const entryQuality = compactText(analysis.entry_quality, "avoid").toLowerCase();
  const reasoning = Array.isArray(analysis.reasoning) ? analysis.reasoning.slice(0, 4).map((x) => compactText(x)).filter((x) => x !== "-") : [];
  const warnings = Array.isArray(analysis.warnings) ? analysis.warnings.slice(0, 3).map((x) => compactText(x)).filter((x) => x !== "-") : [];
  const explanation = compactText(analysis.explanation, "Decision analysis unavailable.");
  const protocol = ((analysis || {}).decision_protocol || {});
  const observeSummary = Array.isArray((((protocol || {}).observe || {}).summary))
    ? (((protocol || {}).observe || {}).summary)
    : [];
  const orientSummary = Array.isArray((((protocol || {}).orient || {}).summary))
    ? (((protocol || {}).orient || {}).summary)
    : [];
  const critiqueReasons = Array.isArray((((protocol || {}).critique || {}).top_failure_reasons))
    ? (((protocol || {}).critique || {}).top_failure_reasons)
    : [];
  const decide = (((protocol || {}).decide) || {});
  const act = (((protocol || {}).act) || {});
  const setupRadar = ((analysis || {}).setup_radar || {});
  const timingEngine = ((analysis || {}).timing_engine || {});
  const radarWatchlist = Array.isArray(setupRadar.watchlist) ? setupRadar.watchlist.slice(0, 4) : [];
  const protocolOutcome = [
    `Confluence: ${Number.isFinite(Number(decide.confluence_score)) ? Math.round(Number(decide.confluence_score)) : "-"} / ${Number.isFinite(Number(decide.confluence_threshold)) ? Math.round(Number(decide.confluence_threshold)) : "-"}`,
    `Blockers override: ${Boolean(decide.blockers_override_score) ? "yes" : "no"}`,
    `Action: ${fmtText(decide.recommended_action, "hold")}`,
    `Execution request eligible: ${Boolean(act.execution_request_eligible) ? "yes" : "no"}`,
    `Top blockers: ${topBlockers.length ? topBlockers.map((b) => `${compactText(b.name)} (${compactText(b.delta_text)})`).join("; ") : "none"}`,
  ];

  setText("da-bias", bias.toUpperCase());
  setText("da-signal-bias", signalBias.toUpperCase());
  setText("da-chart-bias", chartBias.toUpperCase());
  setText("da-htf-bias", htfBias.toUpperCase());
  setText("da-confidence-score", `${Number.isFinite(confidenceScore) ? Math.max(0, Math.min(100, Math.round(confidenceScore))) : "-"} / 100`);
  setText("da-confluence-score", `${Number.isFinite(confluenceScore) ? Math.max(0, Math.min(100, Math.round(confluenceScore))) : "-"} / 100`);
  setText("da-confidence-level", confidenceLevel.toUpperCase());
  setText("da-trend", trend);
  setText("da-momentum", momentum);
  setText("da-location", location);
  setText("da-setup-type", setupType);
  setText("da-regime", regime);
  setText("da-reversal", reversalState);
  setText("da-session-quality", sessionQuality);
  setText("da-account-tier", accountTier);
  setText("da-risk-profile", riskProfile);
  setText("da-size-tier", sizeTier);
  setText("da-recommended-action", recommendedAction);
  setText("da-entry-quality", entryQuality.toUpperCase());
  setAnalysisPillClass("da-bias", "bias", bias);
  setAnalysisPillClass("da-confidence-level", "confidence", confidenceLevel);
  setAnalysisPillClass("da-entry-quality", "entry", entryQuality);
  renderChipList("da-blockers", blockers, "No active blockers");
  renderDetailList("da-reasoning", reasoning, "No strong reasoning points.");
  renderWarningChipList("da-warnings", warnings, "No active warnings");
  renderDetailList("dp-observe", observeSummary, "No observe summary available");
  renderDetailList("dp-orient", orientSummary, "No orient summary available");
  renderChipList("dp-critique", critiqueReasons, "No critique reasons");
  renderDetailList("dp-decide", protocolOutcome, "No decision protocol output");
  renderDetailList(
    "setup-radar-watchlist",
    radarWatchlist.map((row) => {
      const missing = Array.isArray(row.missing_to_entry) && row.missing_to_entry.length
        ? ` wait=${row.missing_to_entry.slice(0, 2).join(", ")}`
        : "";
      return `${compactText(row.name, "setup")} | ${compactText(row.side, "neutral")} | ${compactText(row.phase, "not_forming")} | ${compactText(row.score, "0")}/100${missing}`;
    }),
    "No setup forming"
  );
  renderDetailList(
    "setup-radar-timing",
    [
      `State: ${compactText(timingEngine.state, "wait")}`,
      `Waiting for: ${Array.isArray(timingEngine.waiting_for) && timingEngine.waiting_for.length ? timingEngine.waiting_for.join(", ") : "existing gates"}`,
      compactText(timingEngine.summary, ""),
    ].filter(Boolean),
    "No timing state"
  );
  renderChipList("setup-radar-conditions", timingEngine.conditions_slowing_setups || setupRadar.conditions_slowing_setups, "No slowing conditions");
  setText("da-explanation", explanation);
  renderMacroTerminal(analysis);
}

function renderMacroTerminal(analysis) {
  const macro = (((analysis || {}).macro_context) || {});
  const ctx = (((analysis || {}).context_alignment) || {});
  const sizeBand = ((((analysis || {}).trading_response) || {}).size_band_rationale || {});
  setText("macro-dxy", `${fmtText((((ctx || {}).dxy || {}).state), "-")} / ${fmtText((((ctx || {}).dxy || {}).trend), "-")}`);
  setText("macro-us10y", `${fmtText((((ctx || {}).us10y || {}).state), "-")} / ${fmtText((((ctx || {}).us10y || {}).trend), "-")}`);
  setText("macro-vix", `${fmtText((((ctx || {}).vix || {}).regime), "-")} / ${fmtText((((ctx || {}).vix || {}).state), "-")}`);
  setText("macro-oil", `${fmtText((((ctx || {}).oil || {}).backdrop), "-")} / ${fmtText((((ctx || {}).oil || {}).state), "-")}`);
  setText("macro-event-risk", fmtText((((ctx || {}).event_news || {}).event_risk_state), "-"));
  setText("macro-news-context", fmtText((((ctx || {}).event_news || {}).macro_classification), "-"));
  setText(
    "macro-friction",
    `${fmtText((((ctx || {}).spread_liquidity || {}).spread_state), "-")} / ${fmtText((((ctx || {}).spread_liquidity || {}).slippage_state), "-")}`
  );
  setText(
    "macro-size-band",
    `${fmtText(sizeBand.band, "-")} @ ${Number.isFinite(Number(sizeBand.cap_multiplier)) ? Number(sizeBand.cap_multiplier).toFixed(2) : "-"}x`
  );
  renderDetailList("macro-reads", (((macro || {}).operator_terminal || {}).reads), "No macro reads available");
  renderChipList("macro-allow-reasons", (((macro || {}).operator_terminal || {}).allow_reason), "No active support");
  renderChipList("macro-block-reasons", (((macro || {}).operator_terminal || {}).block_reason), "No active block reason");
}

function renderDetailList(elementId, rows, emptyLabel = "No details available") {
  const el = document.getElementById(elementId);
  if (!el) return;
  const values = Array.isArray(rows)
    ? rows.map((row) => String(row ?? "").trim()).filter(Boolean)
    : [];
  if (!values.length) {
    el.innerHTML = `<div class="detail-row muted">${escapeHtml(emptyLabel)}</div>`;
    return;
  }
  el.innerHTML = values.map((row) => `<div class="detail-row">${escapeHtml(row)}</div>`).join("");
}

function renderRunbookRows(elementId, rows, emptyLabel = "No checklist items") {
  const el = document.getElementById(elementId);
  if (!el) return;
  const values = Array.isArray(rows) ? rows : [];
  if (!values.length) {
    el.innerHTML = `<div class="detail-row muted">${escapeHtml(emptyLabel)}</div>`;
    return;
  }
  el.innerHTML = values.map((row) => {
    const status = String((row || {}).status || "REVIEW").toUpperCase();
    const text = String((row || {}).text || "").trim();
    const cls = status === "OK" ? "runbook-ok" : (status === "BLOCKED" ? "runbook-blocked" : "runbook-review");
    return `<div class="detail-row ${cls}"><strong>${escapeHtml(status)}</strong> - ${escapeHtml(text || "-")}</div>`;
  }).join("");
}

function renderRunbookStopRows(elementId, rows, emptyLabel = "No stop conditions active") {
  const el = document.getElementById(elementId);
  if (!el) return;
  const values = Array.isArray(rows) ? rows.map((x) => String(x || "").trim()).filter(Boolean) : [];
  if (!values.length) {
    el.innerHTML = `<div class="detail-row runbook-ok">${escapeHtml(emptyLabel)}</div>`;
    return;
  }
  el.innerHTML = values.map((row) => `<div class="detail-row runbook-stop">${escapeHtml(row)}</div>`).join("");
}

function renderAskAtlas(response, query = "") {
  const ack = document.getElementById("ask-atlas-ack");
  const understanding = document.getElementById("ask-atlas-understanding");
  const result = document.getElementById("ask-atlas-result");
  const approval = document.getElementById("ask-atlas-approval");
  const status = document.getElementById("ask-atlas-status");
  const primaryLabel = document.querySelector(".ask-primary-label");
  if (ack) ack.textContent = fmtText((response || {}).acknowledge, "No acknowledgement available.");
  if (understanding) understanding.textContent = fmtText((response || {}).understanding, "No understanding available.");
  if (result) result.textContent = fmtText((response || {}).result, "No result available.");
  if (approval) approval.textContent = fmtText((response || {}).approval, "No approval guidance available.");
  renderChipList("ask-atlas-sources", (response || {}).data_sources || [], "No sources listed");
  if (primaryLabel) {
    const kind = fmtText((response || {}).response_kind, "question_answer");
    primaryLabel.textContent = kind === "task_result" ? "Task Result" : "Atlas Answer";
  }
  if (status) {
    const handledAs = fmtText((response || {}).handled_as, "");
    const intent = fmtText(handledAs || (response || {}).intent, "unknown");
    const modeMeta = resolveDisplayModeMeta((response || {}).execution_state, (response || {}).debug);
    const modeLabel = modeMeta.label;
    status.textContent = query
      ? `Handled: ${intent} | MODE: ${modeLabel}`
      : `Awaiting operator prompt. MODE: ${modeLabel}`;
    status.setAttribute("data-mode-source", modeMeta.source);
    status.setAttribute("data-mode-fallback", String(modeMeta.fallback));
  }
}

function renderAskAtlasHistory() {
  renderDetailList(
    "ask-atlas-history",
    askAtlasHistory.map((row) => `${row.query} -> ${row.result}`),
    "No recent requests"
  );
}

function setText(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

function toPct(value) {
  const num = Number(value);
  if (!Number.isFinite(num)) return "-";
  return `${(num * 100).toFixed(1)}%`;
}

function compactText(value, fallback = "-") {
  const txt = String(value || "").replace(/\s+/g, " ").trim();
  return txt || fallback;
}

function titleCase(value) {
  return String(value || "")
    .split(/\s+/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function resolveDisplayMode(executionState, askAtlasDebug) {
  const exec = executionState && typeof executionState === "object" ? executionState : {};
  const dbg = askAtlasDebug && typeof askAtlasDebug === "object" ? askAtlasDebug : {};
  const mode = String(exec.execution_environment_label || dbg.display_mode || "").trim();
  return mode || "Unknown (safe fallback)";
}

function resolveDisplayModeMeta(executionState, askAtlasDebug) {
  const exec = executionState && typeof executionState === "object" ? executionState : {};
  const dbg = askAtlasDebug && typeof askAtlasDebug === "object" ? askAtlasDebug : {};
  const fromExec = String(exec.execution_environment_label || "").trim();
  if (fromExec) return { label: fromExec, source: "execution_environment_label", fallback: false };
  const fromDebug = String(dbg.display_mode || "").trim();
  if (fromDebug) return { label: fromDebug, source: "display_mode", fallback: false };
  return { label: "Unknown (safe fallback)", source: "safe_fallback_unknown", fallback: true };
}

function mergedExecutionState(status, runtime) {
  const statusExec = (((status || {}).execution) || {});
  const exec = {
    ...statusExec,
    ...(((runtime || {}).execution) || {}),
  };
  const truth = (((status || {}).automation_truth) || {});
  const health = (((status || {}).system_health) || {});
  return {
    ...exec,
    automation_enabled: truth.automation_enabled ?? exec.automation_enabled,
    automation_paused: truth.automation_paused ?? exec.automation_paused,
    automation_pause_reason: truth.automation_pause_reason ?? exec.automation_pause_reason,
    automation_explicit_resume_active: truth.automation_explicit_resume_active ?? exec.automation_explicit_resume_active,
    last_auto_execution_result: truth.last_auto_execution_result ?? exec.last_auto_execution_result,
    last_auto_execution_block_reason: truth.last_auto_execution_block_reason ?? exec.last_auto_execution_block_reason,
    execution_environment_label: exec.execution_environment_label || health.effective_mode_label || health.effective_mode || "",
  };
}

function parseResultLine(resultText, prefix) {
  const lines = String(resultText || "")
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
  const match = lines.find((line) => line.toLowerCase().startsWith(prefix.toLowerCase()));
  return match || "";
}

function setCommandTileState(id, state) {
  const el = document.getElementById(id);
  if (!el) return;
  el.classList.remove("state-ready", "state-waiting", "state-blocked", "state-review");
  el.classList.add(`state-${state}`);
}

function readinessLabel(ready) {
  return ready ? "Ready" : "Blocked";
}

function laneDetail(laneName, lane) {
  const data = lane && typeof lane === "object" ? lane : {};
  if (laneName === "market") {
    const blocker = ((data.blocker) || {});
    return compactText(blocker.code || blocker.detail, "waiting for decision");
  }
  if (laneName === "automation") {
    const truth = ((data.truth) || {});
    return truth.automation_paused
      ? compactText(truth.automation_pause_reason, "paused")
      : (truth.automation_enabled ? "resume active" : "disabled");
  }
  if (laneName === "session") {
    const control = ((data.control) || {});
    return data.ready
      ? compactText(control.current_session, "active")
      : compactText(control.session_block_reason || control.next_session_local, "inactive");
  }
  if (laneName === "broker") {
    const broker = ((data.status) || {});
    if (data.ready) return broker.connected === false ? "startup connected" : "MT5 connected";
    return "MT5 not confirmed";
  }
  if (laneName === "ui") {
    const desktop = ((data.desktop) || {});
    return data.ready ? compactText(desktop.frontend_mode, "connected") : compactText(desktop.failure_reason, "degraded");
  }
  return "-";
}

function renderReadinessLanes(status) {
  const readiness = ((status || {}).operator_readiness || {});
  const lanes = ((readiness || {}).lanes || {});
  const mapping = [
    ["lane-market", "market", lanes.market_quality],
    ["lane-automation", "automation", lanes.automation],
    ["lane-session", "session", lanes.session],
    ["lane-broker", "broker", lanes.broker],
    ["lane-ui", "ui", lanes.operator_ui],
  ];
  mapping.forEach(([id, name, lane]) => {
    const el = document.getElementById(id);
    if (!el) return;
    const ready = Boolean((lane || {}).ready);
    el.classList.remove("lane-ready", "lane-wait", "lane-blocked");
    el.classList.add(ready ? "lane-ready" : (name === "market" || name === "session" ? "lane-wait" : "lane-blocked"));
    const strong = el.querySelector("strong");
    const small = el.querySelector("small");
    if (strong) strong.textContent = readinessLabel(ready);
    if (small) small.textContent = laneDetail(name, lane);
  });
}

function renderCognitiveState(status) {
  const cognitive = ((status || {}).cognitive_state || {});
  const coordinator = ((cognitive || {}).coordinator || {});
  const sees = ((coordinator || {}).sees || {});
  const thinks = ((coordinator || {}).thinks || {});
  const remembers = ((coordinator || {}).remembers || {});
  const cautions = ((coordinator || {}).cautions || {});
  const blocker = ((coordinator || {}).primary_blocker || {});
  const state = compactText(cognitive.state, "checking").replace(/_/g, " ");
  const badge = document.getElementById("mind-state-badge");
  if (badge) {
    badge.textContent = state.toUpperCase();
    badge.classList.toggle("state-connected", state.includes("ready") || state === "coordinated");
    badge.classList.toggle("state-training", state.includes("learning") || state.includes("blocked"));
    badge.classList.toggle("muted-pill", state === "checking");
  }
  setText("mind-headline", compactText(cognitive.headline, "Waiting for cognitive state."));
  const symbol = compactText(sees.symbol, "XAUUSD");
  const regime = compactText(sees.regime, "unknown");
  const session = compactText(sees.session, "unknown");
  const location = compactText(sees.location, "unknown");
  setText("mind-sees", `${symbol} / ${regime}`);
  setText("mind-sees-detail", `${session} | ${location}`);
  const bias = compactText(thinks.final_trade_bias || thinks.market_bias, "neutral");
  const readiness = compactText(thinks.setup_readiness, "unknown").replace(/_/g, " ");
  setText("mind-thinks", bias);
  setText("mind-thinks-detail", readiness);
  const sample = Number(remembers.closed_sample_size || remembers.recent_sample_size);
  const winRate = Number(remembers.win_rate);
  setText("mind-remembers", Number.isFinite(sample) ? `${sample} samples` : "memory waiting");
  setText("mind-remembers-detail", Number.isFinite(winRate) ? `WR ${Math.round(winRate * 100)}%` : "no closed sample rate");
  const cautionStatus = compactText(cautions.status, blocker.code ? "blocked" : "watching").replace(/_/g, " ");
  setText("mind-cautions", cautionStatus);
  setText("mind-cautions-detail", compactText(blocker.detail || blocker.code || cautions.source, "no primary caution"));
  setText("mind-policy", compactText(cognitive.policy_note, "Coordinated intelligence. Risk and execution gates remain separate."));
}

function renderCommandStrip(status, runtime) {
  const exec = mergedExecutionState(status, runtime);
  const session = ((exec.session_control) || {});
  const analysis = ((status || {}).decision_analysis || {});
  const trace = ((status || {}).decision_trace || {});
  const operatorReadiness = ((status || {}).operator_readiness || {});
  const structuredBlocker = ((status || {}).structured_blocker || {});
  const blockerSummary = ((status || {}).blocker_classification || trace.blocker_summary || {});
  const topBlockers = Array.isArray(analysis.top_blockers) ? analysis.top_blockers : [];
  const hardBlockers = Array.isArray(analysis.hard_blockers) ? analysis.hard_blockers : [];
  const decision = ((status || {}).decision_explanation || {});
  const message = compactText((status || {}).message, "Waiting for decision cycle");
  const automationEnabled = Boolean(exec.automation_enabled || exec.automation_explicit_resume_active);
  const automationPaused = Boolean(exec.automation_paused);
  const liveAllowed = Boolean(exec.automation_live_allowed);
  const sessionActive = Boolean(session.session_active);
  const killActive = Boolean(exec.kill_switch_active);
  const recon = Boolean(exec.reconciliation_required);
  const forcedFlat = Boolean(exec.forced_flat_pending || session.forced_flat_required);
  const action = compactText((status || {}).action, "hold").toLowerCase();
  const finalDecision = compactText(decision.final_decision, action);
  const blocker = topBlockers[0] || {};
  const blockerClass = compactText(blockerSummary.class || (status || {}).blocker_class, "unknown").replace(/_/g, " ");
  const primaryBlocker = compactText(blockerSummary.primary || (status || {}).primary_blocker || blocker.name || hardBlockers[0], message);
  const blockerRecoverability = compactText(blockerSummary.recoverability || (status || {}).blocker_recoverability, "");
  const structuredCode = compactText(structuredBlocker.code, "");
  const blockerDetail = compactText(structuredBlocker.detail || blockerSummary.detail || blocker.delta_text || blocker.detail, message);
  const displayBlockerClass = compactText(structuredBlocker.class || blockerClass, "unknown").replace(/_/g, " ");
  const blockerName = displayBlockerClass !== "unknown" ? `${titleCase(displayBlockerClass)}: ${primaryBlocker}` : primaryBlocker;
  const blockerDelta = structuredCode
    ? `${structuredCode} | ${compactText(structuredBlocker.recoverability || blockerRecoverability, blockerDetail)}`
    : (blockerRecoverability ? `${titleCase(blockerRecoverability)} | ${blockerDetail}` : blockerDetail);

  let readiness = "WAITING";
  let readinessDetail = message;
  let readinessState = "waiting";
  if (killActive || recon || forcedFlat) {
    readiness = "BLOCKED";
    readinessDetail = killActive ? "Kill switch active" : (recon ? "Reconciliation required" : "Forced-flat required");
    readinessState = "blocked";
  } else if (!automationEnabled || automationPaused) {
    readiness = "REVIEW";
    readinessDetail = automationPaused ? compactText(exec.automation_pause_reason, "Automation paused") : "Automation disabled";
    readinessState = "review";
  } else if (!sessionActive) {
    readiness = "WAITING";
    readinessDetail = compactText(session.session_block_reason, "Outside active session");
    readinessState = "waiting";
  } else if (action === "buy" || action === "sell") {
    readiness = "READY";
    readinessDetail = `${action.toUpperCase()} candidate inside active session`;
    readinessState = "ready";
  } else {
    readiness = finalDecision === "block" ? "BLOCKED" : "WAITING";
    readinessDetail = message;
    readinessState = finalDecision === "block" ? "blocked" : "waiting";
  }

  setText("cmd-readiness", readiness);
  setText("cmd-readiness-detail", readinessDetail);
  setCommandTileState("cmd-readiness-card", readinessState);
  setText("cmd-automation", automationEnabled && !automationPaused ? "Enabled" : "Paused");
  setText("cmd-automation-detail", liveAllowed ? "Live demo gate allowed" : compactText(exec.automation_pause_reason || exec.auto_trade_reason, "Gate not allowed"));
  setText("cmd-session", sessionActive ? compactText(session.current_session, "Active") : "Closed");
  setText("cmd-session-detail", sessionActive
    ? `${compactText(session.current_session_window_local, "session open")} | ${compactText(session.minutes_to_session_close, "-")}m left`
    : `Next: ${compactText(session.next_session_local, "unknown")}`);
  setText("cmd-blocker", blockerName);
  setText("cmd-blocker-detail", blockerDelta);
  let nextAction = sessionActive ? "Observe" : "Wait";
  const primaryLane = compactText(operatorReadiness.primary_lane, "");
  if (readiness === "READY") nextAction = "Review Signal";
  else if (primaryLane === "automation") nextAction = "Resume/Review";
  else if (primaryLane === "session") nextAction = "Wait Session";
  else if (primaryLane === "broker") nextAction = "Fix Broker";
  else if (primaryLane === "operator_ui") nextAction = "Fix UI";
  else if (blockerClass === "runtime or data") nextAction = "Fix Runtime";
  else if (blockerClass === "safety gate") nextAction = "Safety Review";
  else if (blockerClass === "quality gate") nextAction = "Review Gate";
  else if (blockerClass === "signal quality") nextAction = "Wait For Setup";
  setText("cmd-next-action", nextAction);
  setText("cmd-next-detail", compactText(operatorReadiness.next_action, readiness === "READY" ? "Approval still required by policy" : message));
  renderReadinessLanes(status);
}

function renderVisualChartRead(status) {
  const visual = ((status || {}).visual_chart_read || {});
  const review = ((status || {}).visual_evidence_review || {});
  const pattern = compactText(visual.pattern_seen, "unavailable");
  const quality = Number(visual.setup_quality);
  const qualityText = Number.isFinite(quality) ? `${Math.round(quality * 100)}%` : "-";
  const confirmation = compactText(visual.visual_confirmation, "waiting").replace(/_/g, " ");
  const reasons = Array.isArray(visual.reasons) ? visual.reasons : [];
  const source = compactText(review.source || visual.source, "no visual source").replace(/_/g, " ");
  const statusText = compactText(review.status, visual.available ? "available" : "waiting").replace(/_/g, " ");
  setText("visual-pattern", pattern.replace(/_/g, " "));
  setText("visual-quality", qualityText);
  setText("visual-confirmation", confirmation);
  setText("visual-status", statusText);
  setText("visual-source", source);
  setText("visual-reason", compactText(review.headline, "") || (reasons.length ? `${reasons.slice(0, 2).join(" | ")} (${source})` : `Waiting for visual chart evidence (${source}).`));
}

function renderSetupHunter(status) {
  const hunter = ((status || {}).setup_hunter || {});
  const handoff = ((status || {}).setup_execution_handoff || {});
  const closest = ((hunter || {}).closest_setup || {});
  const state = compactText(hunter.state, "scanning").replace(/_/g, " ");
  const handoffState = compactText(handoff.status, "checking").replace(/_/g, " ");
  const count = Number(hunter.candidate_count);
  const name = compactText(closest.name, "none").replace(/_/g, " ");
  const side = compactText(closest.side, "neutral").toUpperCase();
  const score = Number(closest.score);
  const next = compactText(hunter.next_action || closest.next_confirmation, "scan all known setup playbooks and raw strategy signals");
  setText("hunter-state", state);
  setText("hunter-handoff", handoffState);
  setText(
    "hunter-closest",
    name === "none"
      ? "none"
      : `${name} ${side}${Number.isFinite(score) ? ` ${Math.round(score)}/100` : ""}`
  );
  setText("hunter-count", Number.isFinite(count) ? String(count) : "-");
  setText("hunter-next", compactText(handoff.headline, "") || next);
}

function renderLearningCritic(status) {
  const critic = ((status || {}).learning_critic || {});
  const statusText = compactText(critic.status, "watching").replace(/_/g, " ");
  const matchStrength = compactText(critic.match_strength, "").replace(/_/g, " ");
  const matchScore = Number(critic.match_score);
  const lossRate = Number(critic.loss_rate);
  const haircut = Number(critic.haircut);
  const lossRateText = Number.isFinite(lossRate)
    ? `${Math.round(lossRate * 100)}%${Number.isFinite(haircut) && haircut > 0 ? ` | haircut ${(haircut * 100).toFixed(1)}%` : ""}`
    : "-";
  const failureAxis = compactText(critic.previous_failure_axis || critic.failure_axis, "none").replace(/_/g, " ");
  const next = compactText(critic.next_time_action, "");
  const recognition = Array.isArray(critic.future_recognition) ? critic.future_recognition.filter(Boolean).slice(0, 2) : [];
  const sampleText = Number.isFinite(Number(critic.samples)) && Number(critic.samples) > 0
    ? `${Number(critic.samples)} samples`
    : "no confirmed samples";
  setText("critic-status", statusText);
  setText("critic-loss-rate", lossRateText);
  setText("critic-failure-axis", failureAxis);
  setText(
    "critic-next-action",
    next
      || (matchStrength ? `Match: ${matchStrength}${Number.isFinite(matchScore) ? ` (${matchScore.toFixed(2)})` : ""}` : "")
      || (recognition.length ? recognition.join(" | ") : `Watching lesson memory (${sampleText}).`)
  );
}

function renderLearningLoop(status) {
  const loop = ((status || {}).learning_loop || {});
  const decisionContext = ((loop || {}).decision_context || {});
  const quality = ((loop || {}).context_quality || {});
  const state = compactText(loop.status, "waiting").replace(/_/g, " ");
  const lessons = Number(loop.pattern_lessons);
  const lossLessons = Number(loop.loss_dominant_lessons);
  const linked = Number(decisionContext.outcomes_linked);
  const scanned = Number(decisionContext.outcomes_scanned || loop.recent_outcomes);
  const momentumKnown = Number(quality.momentum_known);
  const volatilityKnown = Number(quality.volatility_known);
  const lastRefresh = compactText(loop.last_lesson_refresh_at, "");
  const lastError = ((loop || {}).last_error || {});
  setText("loop-status", state);
  setText("loop-lessons", Number.isFinite(lessons) ? `${lessons} patterns${Number.isFinite(lossLessons) ? ` (${lossLessons} with at least 50% losses)` : ""}` : "-");
  const sources = loop.outcome_sources;
  setText("loop-sources", sources
    ? `${sources.broker_recorded || 0} broker-recorded, ${sources.historical_backtest || 0} backtest, ${sources.forced_probe || 0} forced probes, ${sources.unknown || 0} unknown; ${sources.excluded_manual_tests || 0} manual tests excluded`
    : "Source breakdown unavailable");
  setText("loop-context", Number.isFinite(linked) && Number.isFinite(scanned) ? `${linked}/${scanned}` : "-");
  const qualityText = [
    Number.isFinite(momentumKnown) ? `momentum ${momentumKnown}` : "",
    Number.isFinite(volatilityKnown) ? `volatility ${volatilityKnown}` : "",
  ].filter(Boolean).join(" | ");
  const errorText = compactText(lastError.message, "");
  setText(
    "loop-refresh",
    errorText
      ? `Last issue: ${errorText}`
      : (lastRefresh ? `Last refresh: ${lastRefresh}${qualityText ? ` | ${qualityText}` : ""}` : "Waiting for closed-trade learning.")
  );
}

function learningLaneEvidenceText(name, lane) {
  const evidence = ((lane || {}).evidence || {});
  if (name === "observe") {
    const pattern = compactText(evidence.visual_pattern_seen, "");
    const points = Number(evidence.recent_close_points);
    return pattern || (Number.isFinite(points) ? `${points} close points` : "watching feed");
  }
  if (name === "classify") {
    return compactText(evidence.blocker_class || evidence.blocker_code || evidence.final_action, "waiting");
  }
  if (name === "remember") {
    const lessons = Number(evidence.pattern_lessons);
    const outcomes = Number(evidence.recent_outcomes);
    return Number.isFinite(lessons) ? `${lessons} lessons${Number.isFinite(outcomes) ? ` / ${outcomes} outcomes` : ""}` : "lesson memory";
  }
  if (name === "explain") {
    return compactText(evidence.primary_blocker || evidence.loss_learning_summary || evidence.summary, "operator reasoning");
  }
  if (name === "bounded_caution") {
    const missing = Array.isArray(evidence.missing_confirmations) ? evidence.missing_confirmations.filter(Boolean) : [];
    return missing.length ? missing.slice(0, 2).join(" | ") : compactText(evidence.learning_critic_status, "watching");
  }
  if (name === "propose_improvements") {
    const total = Number(evidence.total);
    const pending = Number(evidence.pending_review);
    return Number.isFinite(total) ? `${total} proposals${Number.isFinite(pending) ? ` / ${pending} pending` : ""}` : "review queue";
  }
  return compactText(lane.source, "waiting");
}

function renderLearningLanes(status) {
  const payload = ((status || {}).learning_lanes || {});
  const lanes = ((payload || {}).lanes || {});
  const grid = document.getElementById("learning-lanes-grid");
  if (!grid) return;
  const order = [
    ["observe", "Observe"],
    ["classify", "Classify"],
    ["remember", "Remember"],
    ["explain", "Explain"],
    ["bounded_caution", "Caution"],
    ["propose_improvements", "Propose"],
  ];
  const rows = order.map(([name, label]) => {
    const lane = lanes[name] || {};
    const active = Boolean(lane.active);
    const statusText = compactText(lane.status, active ? "active" : "waiting").replace(/_/g, " ");
    const cssState = active ? "lane-active" : (statusText.includes("watch") ? "lane-watch" : "lane-wait");
    const evidence = learningLaneEvidenceText(name, lane);
    return `<div class="learning-lane ${cssState}" title="${escapeHtml(compactText(lane.policy_note, ""))}"><span>${label}</span><strong>${escapeHtml(statusText)}</strong><small>${escapeHtml(evidence)}</small></div>`;
  });
  grid.innerHTML = rows.join("");
  const activeCount = Number(payload.active_lanes);
  setText(
    "learning-lanes-note",
    Number.isFinite(activeCount)
      ? `${activeCount}/6 lanes active. Bounded status only; no policy bypass.`
      : "Learning lanes are waiting for status."
  );
}

function renderRecoveryStatus(status) {
  const recovery = ((status || {}).recovery_status || (status || {}).loss_pattern_block || {});
  const blocked = Boolean(recovery.blocked);
  const softPenalty = Boolean(recovery.soft_penalty_active);
  const reason = compactText(recovery.reason, blocked ? "block active" : "clear");
  const metric = ((recovery || {}).primary_metric || {});
  const checklist = Array.isArray(recovery.checklist)
    ? recovery.checklist
    : (Array.isArray(recovery.recovery_checklist) ? recovery.recovery_checklist : []);
  const activeItems = checklist
    .filter((row) => ["blocked", "waiting", "review"].includes(String((row || {}).status || "").toLowerCase()))
    .slice(0, 2)
    .map((row) => compactText(row.label || row.code, "review recovery"));
  setText("recovery-block", blocked ? compactText(recovery.title || recovery.category, "blocked").replace(/_/g, " ") : (softPenalty ? "soft penalty" : "clear"));
  setText("recovery-stats", compactText(metric.value, "-"));
  setText("recovery-scope", compactText(recovery.scope, "-"));
  setText("recovery-next", (blocked || softPenalty || activeItems.length) ? (activeItems.join(" | ") || reason) : "No active recovery blocker.");
}

function renderSafetyLockReview(status) {
  const review = ((status || {}).safety_lock_review || {});
  const metrics = ((review || {}).metrics || {});
  const availability = ((metrics || {}).metric_availability || {});
  const active = Boolean(review.active);
  const lockStatus = compactText(review.status, active ? "locked" : "clear").replace(/_/g, " ");
  const reason = compactText(review.reason, active ? "safety lock active" : "none").replace(/_/g, " ");
  const wr = Number(metrics.supporting_win_rate);
  const streak = Number(metrics.loss_streak);
  const wrText = availability.supporting_win_rate === false ? "n/a" : (Number.isFinite(wr) ? `${Math.round(wr * 100)}%` : "-");
  const streakText = availability.loss_streak === false ? "n/a" : (Number.isFinite(streak) ? String(streak) : "-");
  const checklist = Array.isArray(review.recovery_checklist) ? review.recovery_checklist : [];
  const blockedItems = checklist
    .filter((row) => String((row || {}).status || "").toLowerCase() === "blocked")
    .slice(0, 2)
    .map((row) => compactText(row.label || row.code, "review required"));
  setText("safety-lock-status", active ? lockStatus.toUpperCase() : lockStatus);
  setText("safety-lock-reason", reason);
  setText("safety-lock-metrics", `WR ${wrText} / streak ${streakText}`);
  setText(
    "safety-lock-next",
    active
      ? (blockedItems.join(" | ") || compactText(review.next_action, "Operator review required."))
      : compactText(review.next_action, "Safety lock clear; normal gates still apply.")
  );
}

function renderOperatorSnapshot(status, runtime) {
  const regime = compactText((((status || {}).regime || {}).name), "unknown");
  const regimeConfidence = toPct((((status || {}).regime || {}).confidence));
  const adaptation = (((status || {}).adaptation || {}));
  const learningMode = compactText(adaptation.active_learning_mode, "unknown");
  const evidenceStrength = Number.isFinite(Number(adaptation.evidence_strength_score))
    ? Number(adaptation.evidence_strength_score).toFixed(2)
    : "-";
  const stabilityScore = Number.isFinite(Number(adaptation.stability_score))
    ? Number(adaptation.stability_score).toFixed(2)
    : "-";
  const overreactionScore = Number.isFinite(Number(adaptation.overreaction_score))
    ? Number(adaptation.overreaction_score).toFixed(2)
    : "-";
  const execState = mergedExecutionState(status, runtime);
  const freshnessRuntime = (((status || {}).freshness_runtime) || {});
  const freshnessTiming = (((status || {}).freshness_timing) || {});
  const marketFetch = (((freshnessTiming || {}).market_fetch) || {});
  const modeMeta = resolveDisplayModeMeta(execState, null);
  const modeLabel = modeMeta.label;
  const readiness = compactText(modeLabel, "Unknown (safe fallback)");
  const posture = compactText((status || {}).action, "hold").toUpperCase();
  const execEnabled = Boolean(execState.execution_gate_enabled);
  const killActive = Boolean(execState.kill_switch_active);
  const pendingExec = Number(execState.pending_count || 0);
  const dryRunDefault = Boolean(execState.dry_run_default);
  const rolloutStage = compactText(execState.rollout_stage, "paper_only");
  const executionEnvironment = compactText(modeLabel, "Unknown (safe fallback)");
  const realLiveState = Boolean(execState.real_live_enabled) ? "enabled" : "disabled / not in use";
  const sleepMode = Boolean(execState.sleep_mode_active);
  const sleepAlerts = Number(execState.sleep_alert_count || 0);
  const operatorPresent = Boolean(execState.operator_present);
  const autoTradeAllowed = Boolean(execState.auto_trade_allowed);
  const autoTradeReason = compactText(execState.auto_trade_reason, "n/a");
  const sessionOverride = Boolean(execState.session_override_active);
  const liveEligibility = Boolean(execState.live_execution_allowed) ? "yes" : "no";
  const approvalRequired = Boolean(execState.approval_required) ? "yes" : "no";
  const regression = Boolean(execState.regression_triggered) ? `triggered (${compactText(execState.regression_reason, "safety")})` : "clear";
  const lastExec = execState.last_execution_result && Object.keys(execState.last_execution_result).length
    ? compactText(JSON.stringify(execState.last_execution_result))
    : "none";
  const automationEnabled = Boolean(execState.automation_enabled) ? "yes" : "no";
  const automationPaused = Boolean(execState.automation_paused) ? `yes (${compactText(execState.automation_pause_reason, "paused")})` : "no";
  const automationPolicyId = compactText(execState.automation_active_policy_id, "none");
  const automationLive = Boolean(execState.automation_live_allowed) ? "yes" : "no";
  const automationBlock = compactText(execState.automation_last_block_reason, "none");
  const sessionControl = ((execState.session_control) || {});
  const sessionName = compactText(sessionControl.current_session, "none");
  const sessionActive = Boolean(sessionControl.session_active) ? "yes" : "no";
  const sessionMinsLeft = Number.isFinite(Number(sessionControl.minutes_to_session_close))
    ? Number(sessionControl.minutes_to_session_close)
    : "-";
  const sessionTimezone = compactText(sessionControl.session_timezone, "UTC");
  const localSessionTime = compactText(sessionControl.current_local_time, "n/a");
  const calendarStatus = compactText(sessionControl.session_calendar_status, "unknown");
  const nextSession = compactText(sessionControl.next_session_local, "unknown");
  const dayTradesUsed = `${Number(sessionControl.daily_trade_count || 0)} / ${Number(sessionControl.daily_trade_limit || 0)}`;
  const consecutiveLosses = `${Number(sessionControl.consecutive_losses || 0)} / ${Number(sessionControl.consecutive_loss_limit || 0)}`;
  const dailyStop = Boolean(sessionControl.daily_stop_triggered) ? "triggered" : "clear";
  const sessionBlockReason = compactText(sessionControl.session_block_reason, "none");
  const forcedFlatRequired = Boolean(sessionControl.forced_flat_required);
  const forcedFlatUrgency = forcedFlatRequired
    ? ((Number(sessionMinsLeft) <= 5 && Number(sessionMinsLeft) >= 0) ? "immediate_review" : ((Number(sessionMinsLeft) <= 15 && Number(sessionMinsLeft) >= 0) ? "urgent" : "high"))
    : "none";
  const recoveryMode = compactText(execState.recovery_mode, "fresh_start");
  const tinyPilot = ((execState.tiny_live_pilot) || {});
  const tinyPilotEnabled = Boolean(tinyPilot.enabled) ? "yes" : "no";
  const tinyPilotApproved = Boolean(tinyPilot.approved) ? "yes" : "no";
  const tinyPilotPaused = Boolean(tinyPilot.paused) ? `yes (${compactText(tinyPilot.pause_reason, "paused")})` : "no";
  const tinyPilotScope = `${compactText(tinyPilot.allowed_instrument, "n/a")} / ${compactText(tinyPilot.allowed_setup_family, "n/a")}`;
  const tinyPilotSizeCap = Number(tinyPilot.size_cap || 0).toFixed(3);
  const tinyPilotTrades = `${Number(execState.tiny_live_pilot_trade_count || 0)} / ${Number(tinyPilot.max_trades_per_day || 0)}`;
  const tinyPilotLossUsed = Number(execState.tiny_live_pilot_daily_loss_used || 0).toFixed(3);
  const tinyPilotLossCap = Number(tinyPilot.daily_loss_cap || 0).toFixed(3);
  const tinyPilotBlock = compactText(execState.tiny_live_pilot_last_block_reason, "none");
  const lastLiveOrderStatus = compactText(execState.last_live_order_status, "none");
  const pendingLiveOrders = Number(execState.pending_live_orders_count || 0);
  const partialLiveOrders = Number(execState.partially_filled_count || 0);
  const reconciliationRequired = Boolean(execState.reconciliation_required) ? "yes" : "no";
  const lastBrokerReject = compactText(execState.last_broker_reject_reason, "none");
  const uncertainLiveState = compactText(execState.last_uncertain_execution_state, "none");
  const recoveryReason = compactText(execState.recovery_reason, "n/a");
  const recoveryResume = Boolean(execState.recovery_requires_explicit_resume) ? "yes" : "no";
  const restoredPolicy = compactText(execState.restored_policy_id, "none");
  const cleanShutdown = Boolean(execState.last_shutdown_clean) ? "yes" : "no";
  const confDiag = (((status || {}).confidence_diagnostics) || {});
  const confGap = Number(confDiag.confidence_gap);
  const confGapText = Number.isFinite(confGap) ? confGap.toFixed(3) : "n/a";
  const confDrag = Array.isArray(confDiag.dominant_drag_reasons) ? confDiag.dominant_drag_reasons.slice(0, 3).join(", ") : "none";
  const statusAgeMs = Number(freshnessRuntime.status_age_ms);
  const statusFreshness = Number.isFinite(statusAgeMs)
    ? `${statusAgeMs.toFixed(0)}ms${Boolean(freshnessRuntime.is_status_stale) ? " (STALE)" : ""}`
    : "unknown";
  const fetchRetries = Number(marketFetch.retries || 0);
  const fetchRetryDelayMs = Number(marketFetch.retry_sleep_ms || 0);
  const facts = [
    `Regime: ${regime}`,
    `Regime confidence: ${regimeConfidence}`,
    `Learning mode: ${learningMode}`,
    `Evidence strength: ${evidenceStrength}`,
    `Stability: ${stabilityScore}`,
    `Overreaction: ${overreactionScore}`,
    `Readiness: ${readiness}`,
    `Current posture: ${posture}`,
    `Execution gate: ${execEnabled ? "enabled" : "disabled"}`,
    `Execution environment: ${executionEnvironment}`,
    `Real Live: ${realLiveState}`,
    `Sleep safety mode: ${sleepMode ? "active" : "inactive"}`,
    `Execution disabled due to sleep: ${Boolean(execState.sleep_execution_blocked) ? "yes" : "no"}`,
    `Monitoring while sleeping: ${Boolean(execState.monitoring_active_while_sleeping) ? "active" : "active"}`,
    `Alerts generated while sleeping: ${sleepAlerts}`,
    `Operator present mode: ${operatorPresent ? "active" : "inactive"}`,
    `Status freshness: ${statusFreshness}`,
    `Market fetch retries: ${fetchRetries} (delay ${Number.isFinite(fetchRetryDelayMs) ? fetchRetryDelayMs.toFixed(0) : "0"}ms)`,
    `Auto-trading allowed: ${autoTradeAllowed ? "yes" : "no"} (demo-only)`,
    `Auto-trading block reason: ${autoTradeReason}`,
    `Session override (testing only): ${sessionOverride ? "active" : "inactive"} | real session active: ${Boolean(execState.session_override_real_session_active) ? "yes" : "no"} | simulated session active: ${Boolean(execState.session_override_simulated_session_active) ? "yes" : "no"}`,
    `Kill switch: ${killActive ? "active" : "inactive"}`,
    `Pending execution requests: ${pendingExec}`,
    `Dry-run default: ${dryRunDefault ? "yes" : "no"}`,
    `Rollout stage: ${rolloutStage}`,
    `Tiny live pilot enabled: ${tinyPilotEnabled}`,
    `Tiny live pilot approved: ${tinyPilotApproved}`,
    `Tiny live pilot paused: ${tinyPilotPaused}`,
    `Tiny pilot scope: ${tinyPilotScope}`,
    `Tiny pilot size cap: ${tinyPilotSizeCap}`,
    `Tiny pilot trades used: ${tinyPilotTrades}`,
    `Tiny pilot risk used/cap: ${tinyPilotLossUsed}/${tinyPilotLossCap}`,
    `Tiny pilot block reason: ${tinyPilotBlock}`,
    `Last live order status: ${lastLiveOrderStatus}`,
    `Pending live orders: ${pendingLiveOrders}`,
    `Partially filled live orders: ${partialLiveOrders}`,
    `Reconciliation required: ${reconciliationRequired}`,
    `Last broker reject reason: ${lastBrokerReject}`,
    `Last uncertain live state: ${uncertainLiveState}`,
    `Live eligibility: ${liveEligibility}`,
    `Approval required: ${approvalRequired}`,
    `Last execution result: ${lastExec}`,
    `Regression status: ${regression}`,
    `Automation enabled: ${automationEnabled}`,
    `Automation paused: ${automationPaused}`,
    `Automation policy: ${automationPolicyId}`,
    `Automation live allowed: ${automationLive}`,
    `Last automation block: ${automationBlock}`,
    `Session: ${sessionName}`,
    `Session active: ${sessionActive}`,
    `Session timezone: ${sessionTimezone}`,
    `Local session time: ${localSessionTime}`,
    `Calendar status: ${calendarStatus}`,
    `Time to session close (min): ${sessionMinsLeft}`,
    `Next session: ${nextSession}`,
    `Day trades used: ${dayTradesUsed}`,
    `Consecutive losses: ${consecutiveLosses}`,
    `Daily stop: ${dailyStop}`,
    `Session block reason: ${sessionBlockReason}`,
    `Forced flat required: ${forcedFlatRequired ? "yes" : "no"}`,
    `Forced-flat urgency: ${forcedFlatUrgency}`,
    `Recovery mode: ${recoveryMode}`,
    `Recovery reason: ${recoveryReason}`,
    `Recovery explicit resume required: ${recoveryResume}`,
    `Restored policy: ${restoredPolicy}`,
    `Last shutdown clean: ${cleanShutdown}`,
    `Confidence gap (chosen-threshold): ${confGapText}`,
    `Dominant confidence drag: ${compactText(confDrag, "none")}`,
  ].join(" | ");
  const inference = overreactionScore !== "-" && Number(overreactionScore) > 0.65
    ? "Stability risk is elevated; keep interpretation conservative."
    : (learningMode === "fast_path"
      ? "Evidence appears persistent enough for faster adaptation, still bounded."
      : "System is in cautious adaptation posture.");
  const recommendation = "Keep actions approval-controlled. No automatic execution.";
  setText("op-snapshot-facts", facts);
  setText("op-snapshot-inference", inference);
  setText("op-snapshot-recommendation", recommendation);
}

function renderOperatorRecommendation(payload) {
  const result = compactText((payload || {}).result, "No recommendation available.");
  const firstLine = result.split("\n")[0] || result;
  const constraints = [
    "No automatic execution",
    "No automatic live parameter change",
    "Operator approval required for actionable items",
  ].join(" | ");
  setText("op-reco-facts", firstLine);
  setText("op-reco-inference", compactText((payload || {}).understanding, "Recommendation intent unresolved."));
  setText("op-reco-recommendation", `${result}\nConstraints: ${constraints}`);
}

function renderOperatorFeedback(payload) {
  const result = compactText((payload || {}).result, "No performance feedback available.");
  const adaptation = parseResultLine(result, "Adaptation review:");
  const recommendationReview = parseResultLine(result, "Recommendation review:");
  const stability = parseResultLine(result, "Stability review:");
  const direction = parseResultLine(result, "Recent direction:");
  const weakness = parseResultLine(result, "Main weakness:");
  const nextFocus = parseResultLine(result, "Next safe focus:");
  const facts = [adaptation, recommendationReview, stability].filter(Boolean).join(" | ") || result;
  const inference = [direction, weakness].filter(Boolean).join(" | ") || compactText((payload || {}).understanding, "Feedback inference unavailable.");
  const recommendation = nextFocus || "Review weakest area first with paper-only changes and explicit approval.";
  setText("op-feedback-facts", facts);
  setText("op-feedback-inference", inference);
  setText("op-feedback-recommendation", recommendation);
}

function renderOperatorTuning(payload) {
  const result = compactText((payload || {}).result, "No tuning recommendation available.");
  const candidate = parseResultLine(result, "Recommended paper-mode tuning:");
  const rationale = parseResultLine(result, "Why:");
  const expectedEffect = parseResultLine(result, "Expected effect:");
  const outcome = parseResultLine(result, "Paper tuning result:");
  const facts = [candidate, outcome].filter(Boolean).join(" | ") || result;
  const inference = rationale || compactText((payload || {}).understanding, "No tuning inference available.");
  const recommendation = `${expectedEffect || "Run bounded baseline-vs-tuned paper comparison first."} | Promotion status: operator approval required.`;
  setText("op-tuning-facts", facts);
  setText("op-tuning-inference", inference);
  setText("op-tuning-recommendation", recommendation);
}

function renderOperatorQueue(payload) {
  const controlState = (((payload || {}).control_state) || {});
  const pending = Array.isArray(controlState.pending) ? controlState.pending : [];
  const approved = Array.isArray(controlState.approved) ? controlState.approved : [];
  const rejected = Array.isArray(controlState.rejected) ? controlState.rejected : [];
  const deferred = Array.isArray(controlState.deferred) ? controlState.deferred : [];
  pending.sort((a, b) => Number(a.priority || 9) - Number(b.priority || 9));
  setText(
    "op-queue-facts",
    `Pending: ${pending.length} | Approved: ${approved.length} | Rejected: ${rejected.length} | Deferred: ${deferred.length}`
  );
  setText(
    "op-queue-inference",
    pending.length ? `Top priority item is #${pending[0].id} (${pending[0].type}) at p${pending[0].priority}.` : "No pending control items."
  );
  setText(
    "op-queue-recommendation",
    pending.length ? `Review item #${pending[0].id} first. Approval required before any further action.` : "No action needed right now."
  );
  renderDetailList(
    "op-pending-items",
    pending.slice(0, 8).map((item) => `#${item.id} [${item.type}] p${item.priority} conf ${fmt(item.confidence, 2)} - ${item.summary}`),
    "No pending items"
  );
  renderDetailList(
    "op-queue-history",
    [
      `Approved: ${approved.slice(0, 3).map((x) => `#${x.id}`).join(", ") || "none"}`,
      `Rejected: ${rejected.slice(0, 3).map((x) => `#${x.id}`).join(", ") || "none"}`,
      `Deferred: ${deferred.slice(0, 3).map((x) => `#${x.id}`).join(", ") || "none"}`,
    ],
    "No history yet"
  );
}

function renderOperatorLivePilotReview(status, runtime, queuePayload, reviewPayload) {
  const exec = (((runtime || {}).execution) || {});
  const pilot = ((exec.tiny_live_pilot) || {});
  const session = ((exec.session_control) || {});
  const controlState = (((queuePayload || {}).control_state) || {});
  const pending = Array.isArray(controlState.pending) ? controlState.pending : [];
  const liveReviewItems = pending.filter((item) => {
    const t = String(item.type || "").toLowerCase();
    const d = String(item.related_domain || "").toLowerCase();
    return t.includes("execution") || t.includes("forced_flat") || d.includes("session") || d.includes("execution");
  });
  const blocked = compactText(exec.tiny_live_pilot_last_block_reason, "none");
  const pauseReason = compactText(pilot.pause_reason, "none");
  const recon = Boolean(exec.reconciliation_required);
  const orphanCount = Number(exec.orphan_order_count || 0);
  const forcedFlat = Boolean(exec.forced_flat_pending);
  const facts = [
    `Environment: ${compactText(exec.execution_environment_label || exec.execution_environment, "Paper")} | real live: ${Boolean(exec.real_live_enabled) ? "enabled" : "disabled / not in use"}`,
    `Sleep safety mode: ${Boolean(exec.sleep_mode_active) ? "active" : "inactive"} | sleep alerts: ${Number(exec.sleep_alert_count || 0)}`,
    `Operator present: ${Boolean(exec.operator_present) ? "yes" : "no"} | auto-trading allowed: ${Boolean(exec.auto_trade_allowed) ? "yes" : "no"} (demo-only)`,
    `Session override (testing only): ${Boolean(exec.session_override_active) ? "active" : "inactive"} | real session active: ${Boolean(exec.session_override_real_session_active) ? "yes" : "no"} | simulated session active: ${Boolean(exec.session_override_simulated_session_active) ? "yes" : "no"}`,
    `Pilot enabled: ${Boolean(pilot.enabled) ? "yes" : "no"} | approved: ${Boolean(pilot.approved) ? "yes" : "no"} | paused: ${Boolean(pilot.paused) ? "yes" : "no"}`,
    `Scope: ${compactText(pilot.allowed_instrument, "n/a")} / ${compactText(pilot.allowed_setup_family, "n/a")} | size cap: ${Number(pilot.size_cap || 0).toFixed(3)}`,
    `Activity: attempts today=${Number(exec.tiny_live_pilot_attempts_today || 0)} approvals used=${Number(exec.live_approvals_used || 0)} trades used=${Number(exec.tiny_live_pilot_trade_count || 0)}/${Number(pilot.max_trades_per_day || 0)}`,
    `Risk: daily loss=${Number(exec.tiny_live_pilot_daily_loss_used || 0).toFixed(3)}/${Number(pilot.daily_loss_cap || 0).toFixed(3)} consecutive losses=${Number(exec.tiny_live_pilot_consecutive_losses || 0)}/${Number(pilot.max_consecutive_losses || 0)}`,
    `Order health: last=${compactText(exec.last_live_order_status, "none")} pending=${Number(exec.pending_live_orders_count || 0)} partial=${Number(exec.partially_filled_count || 0)} reconciliation=${recon ? "required" : "clear"}`,
    `Orphan orders: ${orphanCount} | top orphan: ${compactText(exec.top_orphan_summary, "none")}`,
  ].join(" | ");
  const inference = [
    blocked !== "none" ? `Blocked by: ${blocked}` : "",
    Boolean(pilot.paused) ? `Pause reason: ${pauseReason}` : "",
    forcedFlat ? `Forced-flat pending (${compactText(exec.forced_flat_urgency, "high")}).` : "",
    recon ? "Uncertain order state detected; reconciliation required." : "",
    (!Boolean(pilot.paused) && !forcedFlat && !recon && blocked === "none") ? "Pilot state appears bounded and stable." : "",
  ].filter(Boolean).join(" | ");
  const recommendation = [
    recon ? "Resolve reconciliation-required order state before next live attempt." : "",
    orphanCount > 0 ? "Review orphan broker orders before any further live routing." : "",
    forcedFlat ? "Prioritize forced-flat review now." : "",
    Boolean(pilot.paused) ? "Keep pilot paused until blocker is reviewed and cleared." : "",
    "Action hints: show execution requests | show execution audit | show pending items | explain item <id> | reconcile now | reconcile order <id>",
  ].filter(Boolean).join(" ");
  setText("op-live-facts", facts);
  setText("op-live-inference", inference || compactText((reviewPayload || {}).understanding, "Live pilot interpretation unavailable."));
  setText("op-live-recommendation", recommendation || "No urgent live pilot review action required.");
  renderDetailList(
    "op-live-details",
    [
      `Top reconciliation request: ${compactText(exec.top_reconciliation_request_id, "none")}`,
      `Top reconciliation reason: ${compactText(exec.top_reconciliation_reason, "none")}`,
      `Orphan order count: ${orphanCount}`,
      `Top orphan order: ${compactText(exec.top_orphan_order_id, "none")}`,
      `Top orphan summary: ${compactText(exec.top_orphan_summary, "none")}`,
      `Session block reason: ${compactText(session.session_block_reason, "none")}`,
      `Pilot block reason: ${blocked}`,
      `Auto-trading block reason: ${compactText(exec.auto_trade_reason, "none")}`,
      `Last broker reject: ${compactText(exec.last_broker_reject_reason, "none")}`,
      `Last uncertain state: ${compactText(exec.last_uncertain_execution_state, "none")}`,
      `Forced flat pending: ${forcedFlat ? "yes" : "no"} (${compactText(exec.forced_flat_urgency, "none")})`,
      `Live review items pending: ${liveReviewItems.length}`,
    ],
    "No live pilot review details"
  );
}

function renderPilotRunbook(status, runtime) {
  const exec = (((runtime || {}).execution) || {});
  const pilot = ((exec.tiny_live_pilot) || {});
  const session = ((exec.session_control) || {});
  const realSessionActive = Boolean(session.session_active);
  const simulatedSessionActive = Boolean(exec.session_override_simulated_session_active);
  const effectiveSessionActive = realSessionActive || simulatedSessionActive;
  const openPositions = Number((runtime || {}).open_positions ?? (status || {}).open_positions ?? 0);
  const pre = [
    {
      status: (Boolean(exec.execution_gate_enabled) && !Boolean(exec.kill_switch_active)) ? "OK" : "BLOCKED",
      text: `System status: gate ${Boolean(exec.execution_gate_enabled) ? "enabled" : "disabled"}, kill switch ${Boolean(exec.kill_switch_active) ? "active" : "inactive"}`,
    },
    {
      status: (Boolean(pilot.enabled) && Boolean(pilot.approved)) ? "OK" : "BLOCKED",
      text: `Pilot readiness: enabled=${Boolean(pilot.enabled)} approved=${Boolean(pilot.approved)}`,
    },
    {
      status: effectiveSessionActive ? "OK" : "BLOCKED",
      text: `Session active (effective): ${effectiveSessionActive} | real=${realSessionActive} | simulated=${simulatedSessionActive} (${compactText(session.current_session, "none")})`,
    },
    {
      status: Boolean(exec.reconciliation_required) ? "BLOCKED" : "OK",
      text: `Reconciliation clear: ${!Boolean(exec.reconciliation_required)}`,
    },
    {
      status: Number(exec.orphan_order_count || 0) > 0 ? "REVIEW" : "OK",
      text: `Orphan orders: ${Number(exec.orphan_order_count || 0)}`,
    },
    {
      status: (Number(exec.tiny_live_pilot_attempts_today || 0) === 0 && Number(exec.tiny_live_pilot_trade_count || 0) === 0) ? "OK" : "REVIEW",
      text: `Risk counters reset check: attempts=${Number(exec.tiny_live_pilot_attempts_today || 0)} trades=${Number(exec.tiny_live_pilot_trade_count || 0)}`,
    },
    {
      status: Boolean(exec.forced_flat_pending) ? "REVIEW" : "OK",
      text: `Forced-flat pending from prior state: ${Boolean(exec.forced_flat_pending)}`,
    },
  ];

  const during = [
    { status: "REVIEW", text: `Trades used: ${Number(exec.tiny_live_pilot_trade_count || 0)} / ${Number(pilot.max_trades_per_day || 0)}` },
    { status: "REVIEW", text: `Daily loss used: ${Number(exec.tiny_live_pilot_daily_loss_used || 0).toFixed(3)} / ${Number(pilot.daily_loss_cap || 0).toFixed(3)}` },
    { status: "REVIEW", text: `Consecutive losses: ${Number(exec.tiny_live_pilot_consecutive_losses || 0)} / ${Number(pilot.max_consecutive_losses || 0)}` },
    { status: "REVIEW", text: `Session/time: ${compactText(session.current_session, "none")} | mins to close=${Number(session.minutes_to_session_close || -1)}` },
    { status: Boolean(pilot.paused) ? "BLOCKED" : "OK", text: `Pilot paused: ${Boolean(pilot.paused)} (${compactText(pilot.pause_reason, "none")})` },
    { status: (Number(session.minutes_to_session_close || -1) >= 0 && Number(session.minutes_to_session_close || -1) <= 15) ? "REVIEW" : "OK", text: `Near session close warning: ${(Number(session.minutes_to_session_close || -1) >= 0 && Number(session.minutes_to_session_close || -1) <= 15)}` },
    { status: Boolean(exec.forced_flat_pending) ? "BLOCKED" : "OK", text: `Forced-flat urgency: ${compactText(exec.forced_flat_urgency, "none")}` },
  ];

  const stop = [];
  if (Boolean(session.daily_stop_triggered)) stop.push("STOP: daily loss cap hit.");
  if (Number(exec.tiny_live_pilot_consecutive_losses || 0) >= Number(pilot.max_consecutive_losses || 9999)) stop.push("STOP: consecutive loss limit hit.");
  if (Boolean(exec.forced_flat_pending) && String(exec.forced_flat_urgency || "") !== "high") stop.push(`STOP: forced-flat ${compactText(exec.forced_flat_urgency, "urgent")}.`);
  if (Boolean(pilot.paused)) stop.push(`STOP: pilot paused (${compactText(pilot.pause_reason, "unknown")}).`);

  const post = [
    { status: openPositions > 0 ? "REVIEW" : "OK", text: `All positions closed: ${openPositions <= 0}` },
    { status: Number(exec.pending_live_orders_count || 0) > 0 ? "REVIEW" : "OK", text: `Pending live orders: ${Number(exec.pending_live_orders_count || 0)}` },
    { status: Boolean(exec.reconciliation_required) ? "REVIEW" : "OK", text: `Reconciliation required: ${Boolean(exec.reconciliation_required)}` },
    { status: Number(exec.orphan_order_count || 0) > 0 ? "REVIEW" : "OK", text: `Orphan orders: ${Number(exec.orphan_order_count || 0)}` },
    { status: "REVIEW", text: `Today summary: attempts=${Number(exec.tiny_live_pilot_attempts_today || 0)} | last=${compactText(exec.last_live_order_status, "none")} | paused=${Boolean(pilot.paused)} | anomalies=${(Boolean(exec.reconciliation_required) || Number(exec.orphan_order_count || 0) > 0) ? "yes" : "no"}` },
  ];

  renderRunbookRows("op-runbook-pre", pre, "No pre-session checklist.");
  renderRunbookRows("op-runbook-during", during, "No during-session checklist.");
  renderRunbookStopRows("op-runbook-stop", stop, "No stop conditions active.");
  renderRunbookRows("op-runbook-post", post, "No post-session checklist.");
}

async function refreshOperatorPanels() {
  if (operatorRefreshInFlight) return;
  if (document.hidden) return;
  operatorRefreshInFlight = true;
  try {
    const [
      recommendation,
      feedback,
      tuning,
      liveReview,
      queue,
      proposals,
      rollouts,
      audit,
      aiLatest,
      aiReviews,
      aiPreferences,
      aiPromotion,
      aiResearchQueue,
    ] = await Promise.all([
      askAtlas(OPERATOR_QUERIES.recommendation),
      askAtlas(OPERATOR_QUERIES.feedback),
      askAtlas(OPERATOR_QUERIES.tuning),
      askAtlas(OPERATOR_QUERIES.live_review),
      askAtlas(OPERATOR_QUERIES.queue),
      loadSelfImprovementProposals(),
      loadSelfImprovementRollouts(),
      loadSelfImprovementAudit(),
      loadAIDecisionLatest(),
      loadAIDecisionReviews(),
      loadAIStrategyPreferences(),
      loadAIPromotionRecommendations(),
      loadAIResearchQueue(),
    ]);
    operatorPanelCache = {
      recommendation,
      feedback,
      tuning,
      liveReview,
      queue,
      last_update_iso: new Date().toISOString(),
    };
    selfImprovementCache = {
      proposals,
      rollouts,
      audit,
      last_update_iso: new Date().toISOString(),
    };
    aiCoreCache = {
      latest: aiLatest,
      reviews: aiReviews,
      preferences: aiPreferences,
      promotion: aiPromotion,
      researchQueue: aiResearchQueue,
      last_update_iso: new Date().toISOString(),
    };
    renderOperatorRecommendation(recommendation);
    renderOperatorFeedback(feedback);
    renderOperatorTuning(tuning);
    renderOperatorQueue(queue);
    const liveReviewExec = ((liveReview || {}).execution_state || {});
    const queueExec = ((queue || {}).execution_state || {});
    const mergedRuntime = {
      execution: {
        ...queueExec,
        ...liveReviewExec,
      },
    };
    renderOperatorLivePilotReview(lastRichStatus, mergedRuntime, queue, liveReview);
    renderSelfImprovementPanel(lastRichStatus.runtime || {}, proposals, rollouts, audit);
    renderSelfImprovementReviewPacket(selfImprovementCache.reviewPacket);
    renderAICorePanel(aiLatest, aiReviews, aiPreferences, aiPromotion, aiResearchQueue);
    loadTeachingDigest().then(renderTeachingDigest).catch(() => {});
  } catch (err) {
    console.warn("[Atlas UI] operator panel refresh failed", err);
    setText("op-reco-recommendation", "Operator panel refresh failed. Last known values remain visible.");
  } finally {
    operatorRefreshInFlight = false;
  }
}

function getSelfImprovementProposalId() {
  const el = document.getElementById("si-proposal-id");
  return compactText(el ? el.value : "", "");
}

function getSelfImprovementRolloutId() {
  const el = document.getElementById("si-rollout-id");
  return compactText(el ? el.value : "", "");
}

async function refreshSelfImprovementPanelData(runtimeOverride = null) {
  const runtime = runtimeOverride || (await loadRuntime());
  const [proposals, rollouts, audit] = await Promise.all([
    loadSelfImprovementProposals(),
    loadSelfImprovementRollouts(),
    loadSelfImprovementAudit(),
  ]);
  selfImprovementCache = {
    proposals,
    rollouts,
    audit,
    reviewPacket: selfImprovementCache.reviewPacket,
    last_update_iso: new Date().toISOString(),
  };
  renderSelfImprovementPanel(runtime, proposals, rollouts, audit);
  renderSelfImprovementReviewPacket(selfImprovementCache.reviewPacket);
  return { runtime, proposals, rollouts, audit };
}

async function refreshAICorePanelData() {
  const [latest, reviews, preferences, promotion, researchQueue] = await Promise.all([
    loadAIDecisionLatest(),
    loadAIDecisionReviews(),
    loadAIStrategyPreferences(),
    loadAIPromotionRecommendations(),
    loadAIResearchQueue(),
  ]);
  aiCoreCache = {
    latest,
    reviews,
    preferences,
    promotion,
    researchQueue,
    last_update_iso: new Date().toISOString(),
  };
  renderAICorePanel(latest, reviews, preferences, promotion, researchQueue);
  return aiCoreCache;
}

async function runAICoreAction(label, runner) {
  setText("ai-action-status", `Running: ${label}`);
  try {
    const result = await runner();
    await refreshAICorePanelData();
    setText("ai-action-status", compactText(result.reason || "AI core panel refreshed.", "AI core panel refreshed."));
    return result;
  } catch (err) {
    setText("ai-action-status", `Action failed: ${err}`);
    throw err;
  }
}

async function runSelfImprovementAction(label, runner) {
  setText("si-action-status", `Running: ${label}`);
  try {
    const result = await runner();
    await refresh();
    await refreshSelfImprovementPanelData();
    setText("si-action-status", compactText(result.reason || result.rule_version_id || "Action completed.", "Action completed."));
    return result;
  } catch (err) {
    setText("si-action-status", `Action failed: ${err}`);
    throw err;
  }
}

function getOperatorItemId() {
  const el = document.getElementById("op-item-id");
  const id = Number(el ? el.value : 0);
  return Number.isFinite(id) && id > 0 ? Math.floor(id) : 0;
}

async function runOperatorAction(command) {
  const statusEl = document.getElementById("op-action-status");
  if (statusEl) statusEl.textContent = `Running: ${command}`;
  try {
    const result = await askAtlas(command);
    if (statusEl) statusEl.textContent = compactText(result.result, "Action completed.");
    renderOperatorQueue(result);
    return result;
  } catch (err) {
    if (statusEl) statusEl.textContent = `Action failed: ${err}`;
    throw err;
  }
}

function setDashboardView(view) {
  dashboardView = view === "diagnostics" ? "diagnostics" : "command";
  document.body.classList.toggle("diagnostics-view", dashboardView === "diagnostics");
  document.body.classList.toggle("command-view", dashboardView !== "diagnostics");
  const commandBtn = document.getElementById("view-command");
  const diagnosticsBtn = document.getElementById("view-diagnostics");
  if (commandBtn) commandBtn.classList.toggle("active", dashboardView === "command");
  if (diagnosticsBtn) diagnosticsBtn.classList.toggle("active", dashboardView === "diagnostics");
}

async function runCommandDeckAction(command) {
  const statusEl = document.getElementById("cmd-action-status");
  if (statusEl) statusEl.textContent = `Running: ${command}`;
  try {
    const result = await askAtlas(command, 20000);
    if (statusEl) statusEl.textContent = compactText(result.result, "Command completed.");
    renderAskAtlas(result, command);
    askAtlasHistory.unshift({
      query: command,
      result: compactText(result.result, "Command completed.").slice(0, 120),
    });
    askAtlasHistory = askAtlasHistory.slice(0, 8);
    renderAskAtlasHistory();
    await refresh();
    return result;
  } catch (err) {
    if (statusEl) {
      statusEl.textContent = err && err.name === "AbortError"
        ? "Command still processing; refreshing status..."
        : `Command failed: ${err}`;
    }
    await refresh().catch(() => {});
    throw err;
  }
}

function bindCommandDeckActions() {
  const on = (id, fn) => {
    const el = document.getElementById(id);
    if (el) el.addEventListener("click", fn);
  };
  on("view-command", () => {
    setDashboardView("command");
    if (window.location.hash) history.replaceState(null, "", window.location.pathname);
  });
  on("view-diagnostics", () => {
    setDashboardView("diagnostics");
    history.replaceState(null, "", "#diagnostics");
  });
  on("cmd-resume-automation", async () => {
    await runCommandDeckAction("resume automation");
  });
  on("cmd-pause-automation", async () => {
    await runCommandDeckAction("pause automation");
  });
  on("cmd-why-blocked", async () => {
    await runCommandDeckAction("what is holding you back right now?");
  });
  on("cmd-what-next", async () => {
    await runCommandDeckAction("what is the next safe step?");
  });
  setDashboardView(window.location.hash === "#diagnostics" ? "diagnostics" : "command");
}

function bindOperatorActions() {
  const requireId = () => {
    const id = getOperatorItemId();
    if (!id) {
      setText("op-action-status", "Enter a valid item id.");
      return 0;
    }
    return id;
  };
  const on = (id, fn) => {
    const el = document.getElementById(id);
    if (el) el.addEventListener("click", fn);
  };
  on("op-approve-item", async () => {
    const itemId = requireId();
    if (!itemId) return;
    await runOperatorAction(`approve item ${itemId}`);
  });
  on("op-approve-all", async () => {
    await runOperatorAction("approve all pending items");
  });
  on("op-reject-item", async () => {
    const itemId = requireId();
    if (!itemId) return;
    await runOperatorAction(`reject item ${itemId}`);
  });
  on("op-defer-item", async () => {
    const itemId = requireId();
    if (!itemId) return;
    await runOperatorAction(`defer item ${itemId}`);
  });
  on("op-show-pending", async () => {
    await runOperatorAction("show pending items");
  });
  on("op-explain-item", async () => {
    const itemId = requireId();
    if (!itemId) return;
    await runOperatorAction(`explain item ${itemId}`);
  });
  on("op-why-item", async () => {
    const itemId = requireId();
    if (!itemId) return;
    await runOperatorAction(`why did you suggest item ${itemId}`);
  });
}

function bindSelfImprovementActions() {
  const requireProposalId = () => {
    const proposalId = getSelfImprovementProposalId();
    if (!proposalId) {
      setText("si-action-status", "Enter a valid proposal id.");
      return "";
    }
    return proposalId;
  };
  const requireRolloutId = () => {
    const rolloutId = getSelfImprovementRolloutId();
    if (!rolloutId) {
      setText("si-action-status", "Enter a valid rollout id.");
      return "";
    }
    return rolloutId;
  };
  const on = (id, fn) => {
    const el = document.getElementById(id);
    if (el) el.addEventListener("click", fn);
  };
  on("si-validate", async () => {
    const proposalId = requireProposalId();
    if (!proposalId) return;
    await runSelfImprovementAction(`validate ${proposalId}`, () => postJson(`/api/self-improvement/proposals/${proposalId}/validate`));
  });
  on("si-review", async () => {
    const proposalId = requireProposalId();
    if (!proposalId) return;
    await runSelfImprovementAction(`review packet ${proposalId}`, async () => {
      const packet = await loadSelfImprovementOperatorReview(proposalId);
      selfImprovementCache.reviewPacket = packet;
      renderSelfImprovementReviewPacket(packet);
      return { reason: packet.ok ? "Review packet loaded. Approval remains a separate explicit action." : (packet.reason || "Review packet unavailable.") };
    });
  });
  on("si-approve", async () => {
    const proposalId = requireProposalId();
    if (!proposalId) return;
    const confirmed = window.confirm(
      `Approve ${proposalId} for guarded rollout?\n\nThis records operator approval and creates a guarded demo rollout entry. It does not promote live rules directly.`
    );
    if (!confirmed) {
      setText("si-action-status", "Approval cancelled. No rollout was created.");
      return;
    }
    await runSelfImprovementAction(
      `approve ${proposalId}`,
      () => postJson(`/api/self-improvement/proposals/${proposalId}/approve`, {
        approved_by: "operator",
        confirmation: `approve ${proposalId}`,
      })
    );
  });
  on("si-defer", async () => {
    const proposalId = requireProposalId();
    if (!proposalId) return;
    const confirmed = window.confirm(`Defer ${proposalId}?\n\nThis keeps the proposal non-executing and creates no rollout.`);
    if (!confirmed) {
      setText("si-action-status", "Defer cancelled. No proposal status changed.");
      return;
    }
    await runSelfImprovementAction(
      `defer ${proposalId}`,
      () => postJson(`/api/self-improvement/proposals/${proposalId}/close`, {
        target_status: "deferred",
        reason: "operator_deferred",
        note: "dashboard_deferred_no_rollout",
        confirmation: `defer ${proposalId}`,
      })
    );
  });
  on("si-reject", async () => {
    const proposalId = requireProposalId();
    if (!proposalId) return;
    const confirmed = window.confirm(`Reject ${proposalId}?\n\nThis keeps the proposal non-executing and creates no rollout.`);
    if (!confirmed) {
      setText("si-action-status", "Reject cancelled. No proposal status changed.");
      return;
    }
    await runSelfImprovementAction(
      `reject ${proposalId}`,
      () => postJson(`/api/self-improvement/proposals/${proposalId}/close`, {
        target_status: "rejected",
        reason: "operator_rejected",
        note: "dashboard_rejected_no_rollout",
        confirmation: `reject ${proposalId}`,
      })
    );
  });
  on("si-transition", async () => {
    const rolloutId = requireRolloutId();
    if (!rolloutId) return;
    const target = compactText(document.getElementById("si-rollout-target")?.value, "shadow_active");
    const confirmed = window.confirm(
      `Move ${rolloutId} to ${target}?\n\nThis changes the guarded rollout state. It does not promote active rules by itself.`
    );
    if (!confirmed) {
      setText("si-action-status", "Transition cancelled. Rollout state was not changed.");
      return;
    }
    await runSelfImprovementAction(
      `transition ${rolloutId} -> ${target}`,
      () => postJson(`/api/self-improvement/rollouts/${rolloutId}/transition`, {
        target_status: target,
        reason: "operator_dashboard_transition",
        confirmation: `transition ${rolloutId} ${target}`,
      })
    );
  });
  on("si-promote", async () => {
    const proposalId = requireProposalId();
    if (!proposalId) return;
    const confirmed = window.confirm(
      `Promote ${proposalId} to active rules?\n\nThis is a final promotion step and requires a rollback snapshot.`
    );
    if (!confirmed) {
      setText("si-action-status", "Promotion cancelled. Active rules were not changed.");
      return;
    }
    await runSelfImprovementAction(
      `promote ${proposalId}`,
      () => postJson(`/api/self-improvement/proposals/${proposalId}/promote`, {
        approved_by: "operator",
        confirmation: `promote ${proposalId}`,
      })
    );
  });
  on("si-rollback", async () => {
    const rolloutId = requireRolloutId();
    if (!rolloutId) return;
    const confirmed = window.confirm(
      `Rollback ${rolloutId}?\n\nThis restores the prior active rule snapshot and marks the rollout rolled back.`
    );
    if (!confirmed) {
      setText("si-action-status", "Rollback cancelled. Active rules were not changed.");
      return;
    }
    await runSelfImprovementAction(
      `rollback ${rolloutId}`,
      () => postJson(`/api/self-improvement/rollouts/${rolloutId}/rollback`, {
        reason: "operator_dashboard_rollback",
        confirmation: `rollback ${rolloutId}`,
      })
    );
  });
  on("si-refresh", async () => {
    await runSelfImprovementAction("refresh self-improvement panel", () => refreshSelfImprovementPanelData().then(() => ({ reason: "Panel refreshed." })));
  });
}

function bindAICoreActions() {
  const on = (id, fn) => {
    const el = document.getElementById(id);
    if (el) el.addEventListener("click", fn);
  };
  const requireResearchItemId = () => {
    const itemId = compactText(document.getElementById("ai-research-item-id")?.value, "");
    if (!itemId) {
      setText("ai-action-status", "Enter a valid research item id.");
      return "";
    }
    return itemId;
  };
  on("ai-refresh", async () => {
    await runAICoreAction("refresh AI core panel", () => refreshAICorePanelData().then(() => ({ reason: "AI core panel refreshed." })));
  });
  on("ai-review-refresh", async () => {
    await runAICoreAction("review AI decisions", () => postJson("/api/ai/decision/review", { limit: 20 }));
  });
  on("ai-promotion-refresh", async () => {
    await runAICoreAction("recompute promotion intelligence", () => postJson("/api/ai/promotion/recompute", {}));
  });
  on("ai-research-review", async () => {
    const itemId = requireResearchItemId();
    if (!itemId) return;
    const confirmed = window.confirm(`Mark ${itemId} reviewed?\n\nThis changes the AI research queue state only.`);
    if (!confirmed) return;
    await runAICoreAction(`mark reviewed ${itemId}`, () => postJson(`/api/ai/research/items/${encodeURIComponent(itemId)}/review`, {
      note: "dashboard_reviewed",
      confirmation: `review ${itemId}`,
    }));
  });
  on("ai-research-defer", async () => {
    const itemId = requireResearchItemId();
    if (!itemId) return;
    const confirmed = window.confirm(`Defer ${itemId}?\n\nThis keeps the research item non-executing.`);
    if (!confirmed) return;
    await runAICoreAction(`defer ${itemId}`, () => postJson(`/api/ai/research/items/${encodeURIComponent(itemId)}/defer`, {
      note: "dashboard_deferred",
      confirmation: `defer ${itemId}`,
    }));
  });
  on("ai-research-reject", async () => {
    const itemId = requireResearchItemId();
    if (!itemId) return;
    const confirmed = window.confirm(`Reject ${itemId}?\n\nThis keeps the research item non-executing.`);
    if (!confirmed) return;
    await runAICoreAction(`reject ${itemId}`, () => postJson(`/api/ai/research/items/${encodeURIComponent(itemId)}/reject`, {
      note: "dashboard_rejected",
      confirmation: `reject ${itemId}`,
    }));
  });
  on("ai-research-paper", async () => {
    const itemId = requireResearchItemId();
    if (!itemId) return;
    const confirmed = window.confirm(`Request paper test for ${itemId}?\n\nThis does not activate demo or live trading.`);
    if (!confirmed) return;
    await runAICoreAction(`paper test ${itemId}`, () => postJson(`/api/ai/research/items/${encodeURIComponent(itemId)}/paper-test`, {
      note: "dashboard_paper_test_requested",
      confirmation: `paper test ${itemId}`,
    }));
  });
  on("ai-research-demo-review", async () => {
    const itemId = requireResearchItemId();
    if (!itemId) return;
    const confirmed = window.confirm(`Request demo review for ${itemId}?\n\nThis does not activate demo by itself.`);
    if (!confirmed) return;
    await runAICoreAction(`demo review ${itemId}`, () => postJson(`/api/ai/research/items/${encodeURIComponent(itemId)}/demo-review`, {
      note: "dashboard_demo_review_requested",
      confirmation: `demo review ${itemId}`,
    }));
  });
  on("ai-research-demo-packet", async () => {
    const itemId = requireResearchItemId();
    if (!itemId) return;
    await runAICoreAction(`demo review packet ${itemId}`, () => postJson(`/api/ai/research/items/${encodeURIComponent(itemId)}/demo-review-packet`, {}));
  });
  on("ai-research-demo-approve", async () => {
    const itemId = requireResearchItemId();
    if (!itemId) return;
    const confirmed = window.confirm(
      `Approve demo review for ${itemId}?\n\nThis marks the research item as demo-review approved. It does not activate demo by itself.`
    );
    if (!confirmed) {
      setText("ai-action-status", "Demo review approval cancelled.");
      return;
    }
    await runAICoreAction(
      `approve demo review ${itemId}`,
      () => postJson(`/api/ai/research/items/${encodeURIComponent(itemId)}/demo-review-approve`, {
        note: "dashboard_demo_review_approved_no_promotion",
        confirmation: `approve demo review ${itemId}`,
      })
    );
  });
  on("ai-research-demo-proposal", async () => {
    const itemId = requireResearchItemId();
    if (!itemId) return;
    await runAICoreAction(`demo activation proposal ${itemId}`, () => postJson(`/api/ai/research/items/${encodeURIComponent(itemId)}/demo-activation-proposal`, {}));
  });
  on("ai-research-demo-preflight", async () => {
    const itemId = requireResearchItemId();
    if (!itemId) return;
    await runAICoreAction(`demo activation preflight ${itemId}`, () => postJson(`/api/ai/research/items/${encodeURIComponent(itemId)}/demo-activation-preflight`, {}));
  });
}

async function submitAskAtlas(query) {
  const trimmed = String(query || "").trim();
  const status = document.getElementById("ask-atlas-status");
  if (!trimmed) {
    if (status) status.textContent = "Enter a question or command for Atlas.";
    return;
  }
  if (status) status.textContent = "Atlas is reviewing current state...";
  const response = await askAtlas(trimmed);
  renderAskAtlas(response, trimmed);
  askAtlasHistory = [
    {
      query: trimmed,
      result: fmtText((response || {}).result, "No result"),
    },
    ...askAtlasHistory,
  ].slice(0, 6);
  renderAskAtlasHistory();
}

function renderContextSnapshot(view) {
  const context = ((view || {}).context_snapshot || {});
  const fields = [
    ["context-symbol", fmtText(context.symbol || (view || {}).symbol)],
    ["context-timestamp", fmtText(context.timestamp || (view || {}).updated_at)],
    ["context-regime", fmtText(context.regime)],
    ["context-session", fmtText(context.session)],
    ["context-trend-strength", fmtText(context.trend_strength)],
    ["context-volatility-state", fmtText(context.volatility_state)],
    ["context-spread-state", fmtText(context.spread_state)],
    ["context-liquidity-state", fmtText(context.liquidity_state)],
    ["context-vix-regime", fmtText(context.vix_regime)],
    ["context-oil-backdrop", fmtText(context.oil_backdrop)],
    ["context-event-risk-state", fmtText(context.event_risk_state)],
    ["context-news-risk-classification", fmtText(context.news_risk_classification)],
    ["context-slippage-state", fmtText(context.slippage_state)],
    ["context-repeated-entry-risk", context.repeated_entry_risk === undefined ? "-" : (context.repeated_entry_risk ? "HIGH" : "LOW")],
    ["context-overstretched-entry-risk", context.overstretched_entry_risk === undefined ? "-" : (context.overstretched_entry_risk ? "HIGH" : "LOW")],
  ];
  fields.forEach(([id, value]) => {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
  });
}

function renderShadowAdvice(view) {
  const advice = ((view || {}).shadow_advice || {});
  const altRanking = Array.isArray(advice.alternative_ranking) ? advice.alternative_ranking : [];
  const fields = [
    ["shadow-trade-id", fmtText(advice.trade_id)],
    ["shadow-setup-score", fmtText(advice.setup_score)],
    ["shadow-confidence", fmtText(advice.confidence)],
    ["shadow-recommendation", fmtText(advice.recommendation)],
  ];
  fields.forEach(([id, value]) => {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
  });
  renderChipList("shadow-anomaly-flags", advice.anomaly_flags, "No anomaly flags");
  renderDetailList(
    "shadow-alternative-ranking",
    altRanking.map((row, index) => {
      const strategy = String((row || {}).strategy || "unknown");
      const score = fmt((row || {}).score, 4);
      const confidence = fmt((row || {}).confidence, 4);
      return `${index + 1}. ${strategy} | score ${score} | conf ${confidence}`;
    }),
    "No alternative ranking"
  );
}

function drawGoldChart() {
  if (!chartCtx || !chart) return;

  const w = chart.width;
  const h = chart.height;
  chartCtx.clearRect(0, 0, w, h);

  chartCtx.strokeStyle = "rgba(141,168,191,0.35)";
  chartCtx.lineWidth = 1;
  for (let i = 1; i <= 4; i++) {
    const y = (h / 5) * i;
    chartCtx.beginPath();
    chartCtx.moveTo(0, y);
    chartCtx.lineTo(w, y);
    chartCtx.stroke();
  }

  if (priceHistory.length < 2) return;

  const minPrice = Math.min(...priceHistory);
  const maxPrice = Math.max(...priceHistory);
  const range = Math.max(maxPrice - minPrice, 0.0001);

  const pad = 12;
  const innerW = w - pad * 2;
  const innerH = h - pad * 2;

  chartCtx.beginPath();
  priceHistory.forEach((v, i) => {
    const x = pad + (i / (priceHistory.length - 1)) * innerW;
    const normalized = (v - minPrice) / range;
    const y = pad + (1 - normalized) * innerH;
    if (i === 0) chartCtx.moveTo(x, y);
    else chartCtx.lineTo(x, y);
  });

  chartCtx.strokeStyle = "#24d2ff";
  chartCtx.lineWidth = 2;
  chartCtx.shadowColor = "rgba(36,210,255,0.6)";
  chartCtx.shadowBlur = 8;
  chartCtx.stroke();
  chartCtx.shadowBlur = 0;

  const last = priceHistory[priceHistory.length - 1];
  chartCtx.fillStyle = "#10ef9d";
  chartCtx.font = "12px Segoe UI";
  chartCtx.fillText(`Last: ${last.toFixed(2)}`, pad, 14);
}

function renderStatus(s, runtime) {
  const hasRichPayload = Boolean(
    s &&
    ((Array.isArray((s.market || {}).recent_close) && (s.market || {}).recent_close.length > 1) ||
      Array.isArray(s.signals) ||
      (s.risk && typeof s.risk === "object") ||
      s.chosen)
  );
  if (hasRichPayload) lastRichStatus = s;
  const view = hasRichPayload ? s : { ...lastRichStatus, ...s };

  const status = document.getElementById("system-status");
  const decision = document.getElementById("decision");
  const gemini = document.getElementById("gemini-note");
  const signals = document.getElementById("signals");
  const updated = document.getElementById("last-updated");
  const health = document.getElementById("health-summary");

  if (!view || Object.keys(view).length === 0) {
    status.textContent = "IDLE";
    decision.textContent = "Waiting for first cycle...";
    setGeminiState(false, "");
    return;
  }

  const statusSignature = stableStringify({
    updated_at: view.updated_at,
    action: view.action,
    message: view.message,
    chosen: view.chosen,
    decision_explanation: view.decision_explanation,
    decision_trace: view.decision_trace,
    system_health: view.system_health,
    risk: view.risk,
    market: view.market,
    health: view.health,
    signal_summary: view.signal_summary,
    contextual_ranking: view.contextual_ranking,
    visual_chart_read: view.visual_chart_read,
    setup_hunter: view.setup_hunter,
  });
  const signalsSignature = stableStringify(view.signals || []);
  const shouldRenderStatus = statusSignature !== lastStatusSignature;
  const shouldRenderSignals = signalsSignature !== lastSignalsSignature;
  lastStatusSignature = statusSignature;
  lastSignalsSignature = signalsSignature;

  if (shouldRenderStatus) {
    status.textContent = `${(view.action || "idle").toUpperCase()} | ${view.symbol || "XAUUSD"}`;
    status.classList.remove("state-connected", "state-training", "state-connecting", "state-error");
    const action = String(view.action || "").toLowerCase();
    if (action === "connected") status.classList.add("state-connected");
    else if (action === "training") status.classList.add("state-training");
    else if (action === "connecting") status.classList.add("state-connecting");
    else if (action === "error") status.classList.add("state-error");

    document.getElementById("stat-symbol").textContent = view.symbol || "XAUUSD";
    document.getElementById("stat-timeframe").textContent = view.timeframe || "-";
    const execState = mergedExecutionState(view, runtime);
    const modeMeta = resolveDisplayModeMeta(execState, null);
    const statModeEl = document.getElementById("stat-mode");
    if (statModeEl) {
      statModeEl.textContent = modeMeta.label;
      statModeEl.setAttribute("data-mode-source", modeMeta.source);
      statModeEl.setAttribute("data-mode-fallback", String(modeMeta.fallback));
    }
    document.getElementById("stat-action").textContent = (view.action || "-").toUpperCase();
    const chosen = view.chosen || {};
    const bestActionable = view.best_actionable_candidate || ((view.contextual_ranking || {}).best_actionable) || {};
    const confidenceSource = isActionableTrade(chosen) ? chosen : (isActionableTrade(bestActionable) ? bestActionable : chosen);
    const chosenConfidence = confidenceSource.confidence;
    const threshold = Number(view.min_confidence_threshold);
    const confidenceText = Number.isFinite(threshold) && threshold > 0
      ? `${fmt(chosenConfidence, 3)} / ${fmt(threshold, 3)}`
      : fmt(chosenConfidence, 3);
    const confidenceEl = document.getElementById("stat-confidence");
    confidenceEl.textContent = confidenceText;
    confidenceEl.title = isActionableTrade(chosen)
      ? "Chosen trade confidence / minimum threshold"
      : (isActionableTrade(bestActionable)
        ? "Best actionable candidate confidence / minimum threshold"
        : "Current chosen confidence / minimum threshold");
    const freshnessEl = document.getElementById("stat-freshness");
    if (freshnessEl) {
      const freshness = getFreshnessRuntime(view);
      const ageMs = Number.isFinite(freshness.primaryAgeMs) ? freshness.primaryAgeMs : freshness.statusAgeMs;
      freshnessEl.classList.remove("freshness-green", "freshness-yellow", "freshness-red", "freshness-unknown");
      if (!Number.isFinite(ageMs) || freshness.state === "unknown") {
        freshnessEl.textContent = "Unknown";
        freshnessEl.classList.add("freshness-unknown");
        freshnessEl.title = "Status freshness unavailable.";
      } else if (freshness.state === "fresh") {
        freshnessEl.textContent = `${Math.round(ageMs)}ms (Fresh)`;
        freshnessEl.classList.add("freshness-green");
        freshnessEl.title = `Freshness: green (<= ${Math.round(freshness.agingThresholdMs)}ms)`;
      } else if (freshness.state === "aging") {
        freshnessEl.textContent = `${Math.round(ageMs)}ms (Aging)`;
        freshnessEl.classList.add("freshness-yellow");
        freshnessEl.title = `Freshness: yellow (${Math.round(freshness.agingThresholdMs + 1)}ms to ${Math.round(freshness.staleThresholdMs)}ms)`;
      } else {
        freshnessEl.textContent = `${Math.round(ageMs)}ms (Stale)`;
        freshnessEl.classList.add("freshness-red");
        freshnessEl.title = `Freshness: red (> ${Math.round(freshness.staleThresholdMs)}ms)`;
      }
    }
    const attentionEl = document.getElementById("stat-attention");
    if (attentionEl) {
      const execStateNow = mergedExecutionState(view, runtime);
      const scNow = ((execStateNow.session_control) || {});
      const flags = [];
      if (Boolean(execStateNow.kill_switch_active)) flags.push("KILL SWITCH");
      if (Boolean(execStateNow.forced_flat_pending) || Boolean(scNow.forced_flat_required)) flags.push("FORCED-FLAT");
      if (Boolean(execStateNow.reconciliation_required)) flags.push("RECON");
      if (Number(execStateNow.orphan_order_count || 0) > 0) flags.push("ORPHAN");
      if (Boolean((((execStateNow.tiny_live_pilot) || {}).paused))) flags.push("PILOT PAUSED");
      attentionEl.textContent = flags.length ? flags.join(" | ") : "CLEAR";
      attentionEl.title = flags.length ? "Safety attention flags are active." : "No active safety attention flags.";
    }
    const market = view.market || {};
    if (Array.isArray(market.recent_close) && market.recent_close.length > 1) {
      priceHistory = market.recent_close.map(Number).filter((x) => Number.isFinite(x));
      drawGoldChart();
    }

    decision.textContent = view.message || "No message";
    gemini.textContent = compactGeminiNote(view.gemini_note);
    const updatedAt = view.updated_at ? new Date(view.updated_at).toLocaleString() : "-";
    const freshness = getFreshnessRuntime(view);
    const stale = freshness.state === "stale";
    if (updated) updated.textContent = stale ? `Last update: ${updatedAt} (stale)` : `Last update: ${updatedAt}`;
    if (health) {
      const systemHealth = ((view || {}).system_health || {});
      const routing = systemHealth.routing_risk_detected ? "routing:risk" : "routing:clear";
      const config = systemHealth.config_mismatch_detected ? "config:mismatch" : "config:ok";
      const warnings = Array.isArray(systemHealth.warnings) ? systemHealth.warnings.length : 0;
      const broker = ((systemHealth.broker_connectivity_status || {}).connected) ? "broker:ok" : "broker:down";
      const freshnessLabel = freshness.state === "aging"
        ? "data:aging"
        : (freshness.state === "stale" ? "data:stale" : (freshness.state === "fresh" ? "data:fresh" : "data:unknown"));
      health.textContent = `Health: ${broker} | ${routing} | ${config} | warnings:${warnings} | ${freshnessLabel}`;
    }
    setGeminiState(view.gemini_enabled, view.gemini_note || "");

    const risk = view.risk || {};
    document.getElementById("equity").textContent = fmt(risk.equity);
    document.getElementById("balance").textContent = fmt(risk.balance);
    document.getElementById("loss").textContent = `${fmt((risk.daily_loss_pct || 0) * 100)}%`;
    document.getElementById("positions").textContent = risk.open_positions ?? "-";
    renderDecisionExplanation(view);
    renderDecisionTrace(view);
    renderSystemHealth(view);
    renderContextSnapshot(view);
    renderShadowAdvice(view);
    renderCognitiveState(view);
    renderVisualChartRead(view);
    renderSetupHunter(view);
    renderLearningCritic(view);
    renderLearningLoop(view);
    renderLearningLanes(view);
    renderRecoveryStatus(view);
    renderSafetyLockReview(view);
  }

  if (shouldRenderSignals) {
    const sortedSignals = [...(view.signals || [])].sort((a, b) => {
      const aSide = String((a || {}).side || "flat").toLowerCase();
      const bSide = String((b || {}).side || "flat").toLowerCase();
      const aActive = aSide === "flat" ? 0 : 1;
      const bActive = bSide === "flat" ? 0 : 1;
      if (aActive !== bActive) return bActive - aActive;
      const aConf = Number((a || {}).confidence || 0);
      const bConf = Number((b || {}).confidence || 0);
      return bConf - aConf;
    });
    const rows = sortedSignals.map((x) => `<div class="signal-item"><strong title="${x.strategy}">${x.strategy}</strong>Side: ${x.side} | Confidence: ${fmt(x.confidence, 3)}</div>`);
    signals.innerHTML = rows.join("") || "No signals yet.";
  }
  renderDecisionAnalysis(view);
}

function renderEvents(items) {
  const events = document.getElementById("events");
  const liveTypes = new Set(["system", "decision", "paper", "execution", "failed", "blocked", "risk", "day_trader", "error", "training"]);
  const filtered = (items || []).filter((x) => liveTypes.has(String(x.type || "").toLowerCase()) && isFreshTime(x.time, EVENTS_WINDOW_MS));
  const newestFirst = filtered.slice().reverse();
  const deduped = [];
  let driftThrottleSeen = false;
  for (const x of newestFirst) {
    const msg = String(x.message || "");
    const isDriftThrottle = msg.toLowerCase().startsWith("drift throttle active");
    if (isDriftThrottle) {
      if (driftThrottleSeen) continue;
      driftThrottleSeen = true;
    }
    deduped.push(x);
    if (deduped.length >= 8) break;
  }
  const recent = deduped;
  const signature = stableStringify(recent);
  if (signature === lastEventsSignature) return;
  lastEventsSignature = signature;
  const rows = recent.map((x) => {
    const explanation = (((x || {}).meta || {}).decision_explanation || {});
    const systemHealth = (((x || {}).meta || {}).system_health || {});
    const context = (((x || {}).meta || {}).context_snapshot || {});
    const shadow = (((x || {}).meta || {}).shadow_advice || {});
    const detail = explanation.explanation_text
      ? `<div class="event-detail">${escapeHtml(String(explanation.final_decision || "").toUpperCase() || "DECISION")}: ${escapeHtml(explanation.explanation_text)}</div>`
      : "";
    const warningDetail = Array.isArray(systemHealth.warnings) && systemHealth.warnings.length
      ? `<div class="event-detail">WARNINGS: ${escapeHtml(systemHealth.warnings.join(", "))}</div>`
      : "";
    const contextDetail = (context.regime || context.session)
      ? `<div class="event-detail">CONTEXT: ${escapeHtml(String(context.regime || "unknown"))} | ${escapeHtml(String(context.session || "unknown"))} | spread=${escapeHtml(String(context.spread_state || "unknown"))}</div>`
      : "";
    const shadowDetail = shadow.recommendation
      ? `<div class="event-detail">SHADOW: ${escapeHtml(String(shadow.recommendation))} | anomalies=${escapeHtml(String((shadow.anomaly_flags || []).join(", ") || "none"))}</div>`
      : "";
    return `<div class="event"><strong>${escapeHtml(x.type)}</strong> ${escapeHtml(x.message)}${detail}${warningDetail}${contextDetail}${shadowDetail}<br><span class="muted">${escapeHtml(x.time)}</span></div>`;
  });
  events.innerHTML = rows.join("") || "No events yet.";
}

function renderLearningInsights(payload) {
  const summaryEl = document.getElementById("learning-summary");
  const listEl = document.getElementById("learning-insights");
  if (!summaryEl || !listEl) return;
  const signature = stableStringify(payload);
  if (signature === lastLearningSignature) return;
  lastLearningSignature = signature;
  const summary = String((payload || {}).summary || "Learning summary unavailable.");
  const insights = Array.isArray((payload || {}).insights) ? payload.insights : [];
  const lastTrade = ((payload || {}).last_confirmed_trade || ((payload || {}).learning_summary || {}).last_confirmed_trade || {});
  summaryEl.textContent = summary;
  const lastTradeFields = [
    ["last-trade-outcome", compactText(lastTrade.outcome, "unknown").toUpperCase()],
    ["last-trade-pnl", Number.isFinite(Number(lastTrade.net_pnl_zar)) ? `${Number(lastTrade.net_pnl_zar).toFixed(2)} ZAR` : "-"],
    ["last-trade-strategy", compactText(lastTrade.strategy, "-")],
    ["last-trade-side", compactText(lastTrade.side, "-")],
    ["last-trade-confidence", Number.isFinite(Number(lastTrade.confidence)) ? Number(lastTrade.confidence).toFixed(4) : "-"],
    ["last-trade-htf", lastTrade.htf_aligned === undefined ? "-" : (lastTrade.htf_aligned ? "yes" : "no")],
  ];
  lastTradeFields.forEach(([id, value]) => {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
  });
  const lessonEl = document.getElementById("last-trade-lesson");
  if (lessonEl) lessonEl.textContent = compactText(lastTrade.lesson, "No broker-confirmed trade outcome is available yet.");
  listEl.innerHTML = insights.map((x) => `<div class="event">${escapeHtml(String(x))}</div>`).join("") || "No learning insights yet.";
}

function renderModeToggle(runtime, status) {
  const btn = document.getElementById("mode-toggle");
  const readOnly = Boolean(runtime.read_only_ui);
  btn.disabled = readOnly;
  const enabled = Boolean(runtime.execute_trades ?? status.execute_trades ?? false);
  btn.textContent = readOnly
    ? "Read-only (comparison)"
    : (enabled ? "Switch to PAPER" : "Switch to LIVE");
  btn.classList.toggle("live", enabled);
}

function renderGeminiInfluenceToggle(runtime, status) {
  const btn = document.getElementById("gemini-influence-toggle");
  const readOnly = Boolean(runtime.read_only_ui);
  btn.disabled = readOnly;
  const enabled = Boolean(runtime.gemini_influence_enabled ?? status.gemini_influence_enabled ?? false);
  btn.textContent = readOnly
    ? "Gemini Influence: Read-only"
    : (enabled ? "Gemini Influence: ON" : "Gemini Influence: OFF");
  btn.classList.toggle("live", enabled);
}

async function toggleMode() {
  const runtime = await loadRuntime();
  const current = Boolean(runtime.execute_trades);
  const target = !current;
  const confirmed = window.confirm(
    target
      ? "Switch Atlas to LIVE trade execution?\n\nThis enables broker execution when all strategy, risk, and safety gates pass."
      : "Switch Atlas to PAPER mode?\n\nThis disables broker execution and keeps analysis running."
  );
  if (!confirmed) return;
  await fetch("/api/runtime/execute-trades", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      enabled: target,
      confirmation: `set execute_trades ${String(target)}`,
    }),
  });
}

async function toggleGeminiInfluence() {
  const runtime = await loadRuntime();
  const current = Boolean(runtime.gemini_influence_enabled);
  const target = !current;
  const confirmed = window.confirm(
    target
      ? "Enable AI influence?\n\nAtlas may use AI advice as bounded proposal evidence, but execution gates still control trades."
      : "Disable AI influence?\n\nAtlas will keep AI observation available without using it as influence evidence."
  );
  if (!confirmed) return;
  await fetch("/api/runtime/gemini-influence", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      enabled: target,
      confirmation: `set gemini_influence ${String(target)}`,
    }),
  });
}

async function refresh() {
  if (refreshInFlight) return;
  refreshInFlight = true;
  try {
    const now = Date.now();
    const loadAuxiliary = !firstRefreshDone || (now - lastAuxRefreshAt) >= 30000;
    const requests = loadAuxiliary
      ? [loadStatus(), loadRuntime(), loadEvents(), loadLearningInsights()]
      : [loadStatus(), loadRuntime()];
    const results = await Promise.allSettled(requests);
    const s = getSettledValue(results[0], null);
    const r = getSettledValue(results[1], {});
    const e = loadAuxiliary ? getSettledValue(results[2], []) : null;
    const l = loadAuxiliary ? getSettledValue(results[3], null) : null;

    if (!s) {
      throw new Error("Core status endpoint unavailable");
    }

    const hadRefreshFailure = refreshFailureCount > 0;
    refreshFailureCount = 0;
    if (hadRefreshFailure) {
      lastStatusSignature = "";
      lastSignalsSignature = "";
      lastLearningSignature = "";
    }
    firstRefreshDone = true;
    if (loadAuxiliary) lastAuxRefreshAt = now;
    dismissStartup();
    renderStatus(s, r);
    if (e) renderEvents(e);
    if (l) {
      renderLearningInsights(l);
    }
    renderModeToggle(r, s);
    renderGeminiInfluenceToggle(r, s);
    renderCommandStrip(s, r);
    renderOperatorSnapshot(s, r);
    renderOperatorLivePilotReview(s, r, operatorPanelCache.queue, operatorPanelCache.liveReview);
    renderPilotRunbook(s, r);
    renderSelfImprovementPanel(r, selfImprovementCache.proposals, selfImprovementCache.rollouts, selfImprovementCache.audit);
    renderSelfImprovementReviewPacket(selfImprovementCache.reviewPacket);
    renderAICorePanel(aiCoreCache.latest, aiCoreCache.reviews, aiCoreCache.preferences, aiCoreCache.promotion, aiCoreCache.researchQueue);
    lastRichStatus = { status: s, runtime: r, events: e, learning: l };
    console.debug("[Atlas UI] refresh complete", s.updated_at || "no-status-timestamp");
  } catch (err) {
    refreshFailureCount += 1;
    lastStatusSignature = "";
    lastSignalsSignature = "";
    lastLearningSignature = "";
    const statusEl = document.getElementById("system-status");
    if (statusEl) {
      statusEl.textContent = refreshFailureCount >= 3 ? "OFFLINE" : "RECONNECTING";
      statusEl.classList.remove("state-connected", "state-training", "state-error");
      statusEl.classList.add(refreshFailureCount >= 3 ? "state-error" : "state-connecting");
    }
    if (refreshFailureCount >= 3) setGeminiState(false, "");
    if (refreshFailureCount === 1) {
      const decision = document.getElementById("decision");
      if (decision) decision.textContent = "Waiting for live status...";
    }
    console.warn(`[Atlas UI] refresh failed (${refreshFailureCount})`, err);
  } finally {
    refreshInFlight = false;
  }
}

function startPolling() {
  if (refreshTimer !== null) {
    clearInterval(refreshTimer);
  }
  refreshTimer = setInterval(() => {
    if (document.hidden) return;
    refresh().catch((err) => console.warn("[Atlas UI] polling recovery failed", err));
  }, REFRESH_INTERVAL_MS);
}

function startOperatorPolling() {
  if (operatorRefreshTimer !== null) {
    clearInterval(operatorRefreshTimer);
  }
  operatorRefreshTimer = setInterval(() => {
    refreshOperatorPanels().catch((err) => console.warn("[Atlas UI] operator polling failed", err));
  }, OPERATOR_REFRESH_MS);
}

document.getElementById("mode-toggle").addEventListener("click", async () => {
  try {
    await toggleMode();
    await refresh();
  } catch (err) {
    console.error(err);
  }
});

document.getElementById("gemini-influence-toggle").addEventListener("click", async () => {
  try {
    await toggleGeminiInfluence();
    await refresh();
  } catch (err) {
    console.error(err);
  }
});

startPolling();
refresh();
startOperatorPolling();
refreshOperatorPanels();
bindCommandDeckActions();
bindOperatorActions();
bindSelfImprovementActions();
bindAICoreActions();

window.addEventListener("online", () => {
  console.info("[Atlas UI] network restored, forcing refresh");
  refresh().catch((err) => console.warn("[Atlas UI] refresh after online event failed", err));
});

async function bindWindowControls() {
  const bar = document.getElementById("window-bar");
  let tries = 0;
  const maxTries = 30;

  const timer = setInterval(() => {
    const isDesktop = typeof window.pywebview !== "undefined" && window.pywebview.api;
    if (isDesktop) {
      clearInterval(timer);
      bar.style.display = "flex";

      document.getElementById("win-min").addEventListener("click", async () => {
        try {
          const ok = await window.pywebview.api.minimize();
          if (!ok) console.warn("Minimize not available");
        } catch (e) { console.error(e); }
      });

      document.getElementById("win-restore").addEventListener("click", async () => {
        try {
          const ok = await window.pywebview.api.toggle_restore();
          if (!ok) console.warn("Restore/Maximize not available");
        } catch (e) { console.error(e); }
      });

      document.getElementById("win-close").addEventListener("click", async () => {
        try {
          const ok = await window.pywebview.api.close();
          if (!ok) console.warn("Close not available");
        } catch (e) { console.error(e); }
      });
      return;
    }

    tries += 1;
    if (tries >= maxTries) {
      clearInterval(timer);
      bar.style.display = "none";
    }
  }, 150);
}

bindWindowControls();

const askAtlasForm = document.getElementById("ask-atlas-form");
if (askAtlasForm) {
  askAtlasForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const input = document.getElementById("ask-atlas-input");
    try {
      await submitAskAtlas(input ? input.value : "");
    } catch (err) {
      const status = document.getElementById("ask-atlas-status");
      if (status) status.textContent = "Atlas assistant request failed.";
      console.error(err);
    }
  });
}

document.querySelectorAll(".teach-btn").forEach((button) => {
  button.addEventListener("click", async () => {
    const label = button.getAttribute("data-teach-label") || "";
    try {
      await recordTeachingReview(label);
    } catch (err) {
      const status = document.getElementById("teach-status");
      if (status) status.textContent = "Teaching label failed.";
      console.error(err);
    }
  });
});

renderAskAtlas({}, "");
renderAskAtlasHistory();
