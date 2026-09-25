"""Typed decisions: only a ready result contains an executable action."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

from computer.actions import Action


class DecisionEffect(StrEnum):
    OBSERVE = "observe"
    ACT = "act"
    TERMINAL = "terminal"


@dataclass(frozen=True, slots=True)
class VisualGroundingNeed:
    """Observation request selected by Jev; never an executable action."""

    objective: str
    reason: str
    max_candidates: int = 5

    def __post_init__(self) -> None:
        if not self.objective.strip() or len(self.objective) > 240:
            raise ValueError("grounding objective must contain 1 to 240 characters")
        if not self.reason.strip() or len(self.reason) > 240:
            raise ValueError("grounding reason must contain 1 to 240 characters")
        if not 1 <= self.max_candidates <= 100:
            raise ValueError("grounding max_candidates must be between 1 and 100")


@dataclass(frozen=True, slots=True)
class OfferedApplication:
    app_id: str
    name: str


@dataclass(frozen=True, slots=True)
class TaskProgress:
    query_required: bool
    query_entered_or_submitted: bool
    result_grounding_eligible: bool


@dataclass(frozen=True, slots=True)
class ChoiceProbability:
    option_id: str
    probability: float


@dataclass(frozen=True, slots=True)
class DecisionAttempt:
    attempt: int
    result: str
    error_category: str | None = None
    http_status: int | None = None
    provider_error_code: str | None = None
    response_shape_summary: dict[str, object] = field(default_factory=dict)
    returned_option_id: str | None = None
    returned_confidence: float | None = None
    returned_confidence_raw_type: str | None = None
    selected_option_probability: float | None = None
    probability_count: int | None = None
    probability_sum: float | None = None


@dataclass(frozen=True, slots=True)
class OptionFilterSummary:
    uia_click_candidates_seen: int = 0
    uia_click_options_offered: int = 0
    type_candidates_seen: int = 0
    type_options_offered: int = 0
    press_key_capabilities_seen: int = 0
    press_key_options_offered: int = 0
    filtered_reasons: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ObservationPolicyDiagnostic:
    eligible: bool | None = None
    reason: str = "not_applicable"
    confidence_required: bool = False


@dataclass(frozen=True, slots=True)
class HybridDecisionResult:
    status: Literal["ready", "grounding", "needs_human", "error"]
    action: Action | None
    grounding_need: VisualGroundingNeed | None
    confidence: float | None
    message: str
    observation_id: str = ""
    selected_option: str | None = None
    probabilities: dict[str, float] = field(default_factory=dict)
    model: str | None = None
    error: str | None = None
    diagnostic: str | None = None
    offered_option_types: tuple[str, ...] = ()
    offered_grounding_needs: tuple[VisualGroundingNeed, ...] = ()
    offered_applications: tuple[OfferedApplication, ...] = ()
    decision_error_category: str | None = None
    http_status: int | None = None
    provider_error_code: str | None = None
    response_shape_summary: dict[str, object] = field(default_factory=dict)
    expected_primitive: str = "choice"
    offered_option_count: int = 0
    returned_option_id: str | None = None
    returned_confidence_raw_type: str | None = None
    retry_attempted: bool = False
    retry_result_category: str | None = None
    task_progress: TaskProgress = field(
        default_factory=lambda: TaskProgress(False, False, True),
    )
    choice_probabilities: tuple[ChoiceProbability, ...] = ()
    selected_option_probability: float | None = None
    decision_attempts: tuple[DecisionAttempt, ...] = ()
    option_filter_summary: OptionFilterSummary = field(default_factory=OptionFilterSummary)
    effect: DecisionEffect = DecisionEffect.ACT
    offered_visual_option_count: int = 0


@dataclass(frozen=True, slots=True)
class DecisionResult:
    status: Literal["ready", "needs_human", "error"]
    action: Action | None
    confidence: float | None
    message: str
    observation_id: str = ""
    selected_option: str | None = None
    probabilities: dict[str, float] = field(default_factory=dict)
    model: str | None = None
    error: str | None = None
    diagnostic: str | None = None


@dataclass(frozen=True, slots=True)
class TargetChoiceResult:
    """Jev's bounded choice among locally admissible target candidates."""

    status: Literal["ready", "stop", "error"]
    candidate_id: str | None
    confidence: float | None
    message: str
    error: str | None = None
    diagnostic_reason: Literal[
        "resolution_ineligible", "candidate_limit", "candidate_invalid",
        "evidence_indistinguishable", "model_stop",
        "model_confidence_below_threshold", "candidate_selected",
        "provider_error", "invalid_response", "internal_error",
    ] | None = None
    provider_called: bool | None = None
    proposed_candidate_id: str | None = None
