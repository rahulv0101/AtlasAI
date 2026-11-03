from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class IntentCandidate:
    name: str
    confidence: float
    matched_patterns: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass(frozen=True)
class DomainCandidate:
    name: str
    confidence: float
    reason: str = ""


@dataclass(frozen=True)
class EvidenceItem:
    source: str
    domain: str
    content: str
    recency: float
    confidence: float
    fact_vs_inference: str
    relevance_score: float


@dataclass(frozen=True)
class AnswerPlan:
    name: str
    sections: list[str]
    preferred_mode: str = "standard"
    constraints: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RelevanceCheckResult:
    passed: bool
    reason: str


@dataclass(frozen=True)
class AnswerResult:
    text: str
    selected_intent: str
    candidate_intents: list[IntentCandidate]
    selected_domain: str
    selected_plan: str
    confidence: float
    evidence_used: list[EvidenceItem]
    relevance_passed: bool
    debug_summary: dict


@dataclass(frozen=True)
class RecommendationReason:
    text: str
    kind: str = "fact"


@dataclass(frozen=True)
class RecommendationConstraint:
    text: str


@dataclass(frozen=True)
class RecommendationItem:
    category: str
    recommendation: str
    reason: str
    supporting_evidence: list[str]
    confidence: float
    constraint: str
    priority: int
    safety_level: str


@dataclass(frozen=True)
class RecommendationResult:
    items: list[RecommendationItem]
    rejected: list[str]
    policy_notes: list[str]


@dataclass(frozen=True)
class PerformanceFeedbackItem:
    category: str
    label: str
    summary: str
    confidence: float
    priority: int


@dataclass(frozen=True)
class PerformanceFeedbackResult:
    scores: dict[str, float]
    items: list[PerformanceFeedbackItem]
    biggest_weakness: str
    next_safe_focus: str
    direction: str
    notes: list[str]
    inputs_snapshot: dict[str, float]
    scoring_breakdown: dict[str, str] = field(default_factory=dict)
    weakness_ranking: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TuningConstraint:
    text: str


@dataclass(frozen=True)
class TuningCandidate:
    target: str
    category: str
    current_value: float
    proposed_delta: float
    bounded_range: tuple[float, float]
    rationale: str
    expected_effect: str
    safety_constraints: list[str]
    confidence: float
    safe_to_paper_test: bool
    priority: int
    promotion_allowed: bool = False


@dataclass(frozen=True)
class TuningComparison:
    candidate_target: str
    baseline_summary: str
    tuned_summary: str
    delta_acceptance_rate: float
    delta_stability: float
    delta_overreaction: float
    outcome: str
    recommendation: str


@dataclass(frozen=True)
class TuningResult:
    candidates: list[TuningCandidate]
    rejected: list[str]
    comparisons: list[TuningComparison]
    top_recommendation: str
    notes: list[str]
    safety_gates_triggered: list[str] = field(default_factory=list)
    opportunities_considered: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ComparisonIntelligenceResult:
    direction: str
    compared_to: str
    deltas: dict[str, float]
    improved: list[str]
    worsened: list[str]
    safe_interpretation: str
    sparse_data: bool = False


@dataclass(frozen=True)
class ControlItem:
    id: int
    type: Literal[
        "recommendation",
        "tuning_candidate",
        "paper_experiment",
        "performance_warning",
        "improvement_suggestion",
        "paper_tuning_promotion_candidate",
        "extended_paper_validation",
        "recommendation_confidence_escalation",
        "strategy_review_flag",
        "persistent_weakness_investigation",
        "execution_request",
        "forced_flat_review",
        "order_reconciliation_review",
        "order_orphan_review",
    ]
    summary: str
    detailed_reason: str
    supporting_evidence: list[str]
    safety_constraints: list[str]
    confidence: float
    priority: int
    timestamp: float
    status: Literal["pending", "approved", "rejected", "deferred"]
    workflow_state: Literal[
        "proposed",
        "approved",
        "rejected",
        "deferred",
        "under_observation",
        "validated_in_paper",
        "expired",
    ] = "proposed"
    approval_level: Literal[
        "informational",
        "review_required",
        "approval_required",
        "high_risk_requires_explicit_confirmation",
    ] = "review_required"
    approval_reason: str = ""
    approval_timestamp: float | None = None
    approval_context: str = ""
    related_domain: str = ""
    source_module: str = ""
    priority_reason: str = ""
    trigger_conditions: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ExecutionRequest:
    id: str
    environment: Literal["paper", "live"]
    action_type: str
    payload_summary: str
    created_at: float
    requested_by: str
    approval_required: bool
    approval_status: Literal["pending", "approved", "rejected"]
    validation_status: Literal["not_validated", "passed", "failed"]
    execution_status: Literal["pending", "dry_run_completed", "executed", "blocked", "failed"]
    dry_run: bool
    audit_ref: str
