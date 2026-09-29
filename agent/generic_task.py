"""Small experimental controller for bounded generic target activation."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from enum import StrEnum
import math
import re
import time
import unicodedata
from typing import Any, Protocol, TypeVar

from agent.activation_postcondition import (
    ActivationEvidence, ActivationEvidenceSource, ActivationPostconditionResult,
    TargetActivationPostcondition, evaluate_target_activation_postcondition,
)
from agent.application_activation import (
    ActivationLifecycleDiagnostics, ApplicationActivationDiagnostics,
    ApplicationActivationWait,
    wait_for_trusted_application_activation,
)
from agent.visual_field_verification import (
    has_credential_sensitive_evidence, role_is_text_entry, safe_candidate_ids,
    same_trusted_foreground_identity, verify_visual_field_correspondence,
)
from computer.actions import Action, ClickAction, OpenAppAction, QuerySubmitAction, TypeAction, VisualClickAction
from computer.applications import ApplicationCandidate, ApplicationCatalog
from computer.models import (
    Observation, ProviderErrorDiagnostic, Rect, ScreenshotMetadata, UIElement, VisualGroundingStatus,
    VisualElement, VisualPreclickCandidateDiagnostic,
    VisualPreclickRevalidationDiagnostic, VisualPreclickRevalidationStatus,
    VisualPreclickTargetSpecDiagnostic,
    VisualProviderAttempt, VisualReadinessReason,
)
from computer.results import ActionResult, VisualActivationDiagnostic
from computer.visual import (
    VisualFieldValueRead, VisualGroundingRequest, bounded_grounding_request,
)
from decision.context import Redactor, requested_target_spec
from decision.models import TargetChoiceResult, VisualGroundingNeed
from decision.target_resolution import (
    CandidateResolution, IdentityEvidence, PresentationCompatibility, RoleCompatibility,
    TargetResolution, TargetResolutionStatus, TargetSpec,
    normalize_presentation_role, resolve_target,
)
from agent.target_evidence import (
    DirectedGroundingBinding, TargetEvidenceSet, adapt_observation_candidates,
    bind_directed_grounding, is_query_field,
    is_visual_query_field, query_literal_from_target,
)
from agent.target_sufficiency import (
    TargetEvidenceSufficiency, assess_target_evidence_sufficiency,
)
from safety.policy import (
    GenericTargetActivationPolicy, generic_uia_semantic_role, visual_rect_to_screen,
)


MAX_DIAGNOSTIC_CANDIDATES = 12
MAX_DIAGNOSTIC_FRONTIER = 5
MAX_DIAGNOSTIC_NEAR_MATCHES = 5
MAX_DIAGNOSTIC_LABEL_CHARS = 100
_SAFE_DIAGNOSTIC_CANDIDATE_ID = re.compile(r"(?:c|v)\d{1,5}\Z")
_DiscoveryValue = TypeVar("_DiscoveryValue")


class GenericTaskComputer(Protocol):
    visual_provider: object | None

    def observe_local(self) -> Observation: ...
    def observe_directed(self, grounding: VisualGroundingRequest) -> Observation: ...
    def observe_preclick_local(self, expected_context: Observation) -> Observation: ...
    def observe_preclick_directed(
        self, grounding: VisualGroundingRequest, expected_context: Observation,
    ) -> Observation: ...
    def read_visual_field_value(
        self, observation: Observation, field_id: str,
    ) -> VisualFieldValueRead: ...
    def execute(self, action: Action, observation: Observation | None = None) -> ActionResult: ...
    def execute_type_phase2(
        self, action: TypeAction, observation: Observation, *, visual_verified: bool,
    ) -> ActionResult: ...
    def execute_generic_target_activation(
        self, action: VisualClickAction, observation: Observation,
    ) -> ActionResult: ...
    def execute_generic_query_submit(
        self, action: QuerySubmitAction, observation: Observation,
    ) -> ActionResult: ...


class GenericTargetDecisionMaker(Protocol):
    min_confidence: float

    def decide_target_activation(
        self, target: TargetSpec, resolution: TargetResolution,
    ) -> TargetChoiceResult: ...


class GenericCapability(StrEnum):
    ACTIVATE_TARGET = "ACTIVATE_TARGET"
    FIND_QUERY_FIELD = "FIND_QUERY_FIELD"
    ENTER_LITERAL_QUERY = "ENTER_LITERAL_QUERY"
    SUBMIT_QUERY = "SUBMIT_QUERY"
    WAIT_FOR_TRANSITION = "WAIT_FOR_TRANSITION"
    RESOLVE_TARGET = "RESOLVE_TARGET"
    OPEN_APPLICATION = "OPEN_APPLICATION"
    OBSERVE = "OBSERVE"
    FINISH = "FINISH"
    STOP = "STOP"


class GenericTaskOperation(StrEnum):
    OPEN_ONLY = "open_only"
    ACTIVATE_TARGET = "activate_target"
    SEARCH = "search"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True, slots=True)
class GenericTaskIntent:
    """Small typed split between an optional app clause and its operation payload."""

    application_request: str | None
    operation: GenericTaskOperation
    operation_target: str | None
    decomposition_method: str
    source_language: str = "en"


@dataclass(frozen=True, slots=True)
class GenericTaskDecompositionDiagnostics:
    source_language: str
    application_text: str | None
    operation: str
    operation_target_text: str | None
    decomposition_method: str


@dataclass(frozen=True, slots=True)
class GenericPerceptionAttemptDiagnostics:
    capability: str
    observation_id: str | None
    foreground_identity_stable: bool
    uia_candidate_count: int
    visual_candidate_count: int
    visual_observation_id: str | None = None
    structural_observation_complete: bool = False
    visual_observation_complete: bool = False
    visual_provider_result: str = "unavailable"
    same_trusted_app: bool = False
    same_window: bool = False
    fresh_structural: bool = False
    fresh_visual: bool = False
    discovery_classification: str = "incomplete"
    incomplete_reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GenericPerceptionRetryDiagnostics:
    capability: str
    applicable: bool
    reason: str
    attempts: int
    max_attempts: int
    elapsed_ms: int
    success: bool
    stop_reason: str
    attempt_diagnostics: tuple[GenericPerceptionAttemptDiagnostics, ...]
    initial_discovery_elapsed_ms: int = 0
    readiness_wait_elapsed_ms: int = 0
    retry_discovery_elapsed_ms: int = 0
    total_elapsed_ms: int = 0
    initial_failure_class: str | None = None
    retry_performed: bool = False
    retry_result: str = "not_attempted"


# Compatibility aliases for diagnostics consumers that used the earlier names.
GenericReadinessAttemptDiagnostics = GenericPerceptionAttemptDiagnostics
GenericReadinessRetryDiagnostics = GenericPerceptionRetryDiagnostics


@dataclass(frozen=True, slots=True)
class _ReadinessAssessment:
    observation_id: str | None
    uia_candidate_count: int
    visual_candidate_count: int
    empty: bool
    complete: bool
    usable_candidate: bool
    stop_reason: str
    visual_observation: Observation | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class GenericTaskBudgets:
    max_steps: int = 16
    app_launches: int = 1
    query_field_activations: int = 1
    visual_target_activations: int = 2
    literal_types: int = 1
    query_submits: int = 1
    final_target_activations: int = 1
    visual_grounding_calls: int = 4
    max_frontier_candidates: int = 5
    transition_timeout_seconds: float = 2.0
    transition_poll_seconds: float = 0.2
    app_activation_timeout_seconds: float = 3.0
    readiness_retries: int = 2
    readiness_wait_budget_seconds: float = 0.8
    readiness_poll_seconds: float = 0.2

    def __post_init__(self) -> None:
        if not 1 <= self.max_steps <= 16 or not 1 <= self.max_frontier_candidates <= 5:
            raise ValueError("Step and candidate budgets exceed the generic debug limits.")
        if self.app_launches != 1 or self.query_field_activations != 1:
            raise ValueError("Generic task activation budgets are fixed at one.")
        if self.visual_target_activations != 2 or self.literal_types != 1:
            raise ValueError("Generic task input budgets are fixed and bounded.")
        if self.query_submits != 1 or self.final_target_activations != 1:
            raise ValueError("Generic task activation budgets are fixed at one.")
        if not 1 <= self.visual_grounding_calls <= 4:
            raise ValueError("Visual grounding budget must be between one and four.")
        if not 0 <= self.transition_timeout_seconds <= 5:
            raise ValueError("Transition timeout must be between zero and five seconds.")
        if not 0.01 <= self.transition_poll_seconds <= 1:
            raise ValueError("Transition poll interval must be between 10 ms and one second.")
        if not 0 <= self.app_activation_timeout_seconds <= 15:
            raise ValueError("Application activation timeout must be between zero and 15 seconds.")
        if not 0 <= self.readiness_retries <= 2:
            raise ValueError("At most two fresh readiness observations are allowed per run.")
        if not 0 <= self.readiness_wait_budget_seconds <= 1.5:
            raise ValueError("Readiness wait budget must be between zero and 1.5 seconds.")
        if not 0.01 <= self.readiness_poll_seconds <= 0.5:
            raise ValueError("Readiness poll interval must be between 10 ms and 500 ms.")


@dataclass(frozen=True, slots=True)
class GenericTaskStep:
    step: int
    capability: GenericCapability
    observation_id: str | None = None
    candidate_count: int = 0
    resolution_status: str | None = None
    action_kind: str | None = None
    action_succeeded: bool | None = None


@dataclass(frozen=True, slots=True)
class GenericTargetSpecDiagnostic:
    primary_identity: str
    qualifiers: tuple[str, ...]
    desired_role: str | None
    generic_intent: str


@dataclass(frozen=True, slots=True)
class CandidateSemanticDiagnostic:
    candidate_id: str
    source: str
    snapshot_id: str
    primary_text: str | None
    secondary_text: tuple[str, ...]
    semantic_role: str | None
    provider_role: str | None
    primary_identity: str
    qualifier_evidence: tuple[str, ...]
    role_compatibility: str
    actionable: bool
    geometry_valid: bool
    safety_eligible: bool
    admissible: bool
    rejection_reasons: tuple[str, ...]
    target_semantic_evidence: str | None
    domain_semantic_compatibility: str
    presentation_role: str
    presentation_compatibility: str


@dataclass(frozen=True, slots=True)
class CandidateSourceDiagnostics:
    snapshot_id: str | None
    raw_candidate_count: int
    adapted_candidate_count: int
    admissible_candidate_count: int
    rejected_candidate_count: int
    rejection_reason_counts: tuple[tuple[str, int], ...]
    frontier_candidate_count: int
    grounding_called: bool | None = None
    grounding_objective: str | None = None
    provider_raw_element_count: int | None = None
    parsed_element_count: int | None = None
    validated_element_count: int | None = None
    visual_provider_attempts: tuple[VisualProviderAttempt, ...] = ()
    selected_visual_provider: str | None = None
    provider_failover_used: bool = False
    provider_failover_reason: str | None = None
    visual_provider_error: ProviderErrorDiagnostic | None = None


@dataclass(frozen=True, slots=True)
class GenericTargetResolutionDiagnostics:
    attempt_stage: str
    target_spec: GenericTargetSpecDiagnostic
    uia: CandidateSourceDiagnostics
    visual: CandidateSourceDiagnostics
    status: str
    resolution_reason: str
    candidate_ids: tuple[str, ...]
    frontier_candidate_count: int
    frontier_candidate_ids: tuple[str, ...]
    frontier_evidence: tuple[CandidateSemanticDiagnostic, ...]
    near_matches: tuple[CandidateSemanticDiagnostic, ...]


@dataclass(frozen=True, slots=True)
class GenericTargetDecisionDiagnostics:
    resolver_status: str
    selection_mode: str
    evidence_sufficiency: str
    evidence_sufficiency_reasons: tuple[str, ...]
    admissible_candidate_count_total: int
    frontier_candidate_count: int
    jev_required: bool
    jev_called: bool
    jev_provider_called: bool | None
    jev_result_kind: str | None
    jev_selected_candidate_id: str | None
    jev_confidence: float | None
    jev_release_result: str
    stop_reason: str | None
    chosen_candidate_id: str | None = None


@dataclass(frozen=True, slots=True)
class QueryFieldCandidateDiagnostic:
    candidate_id: str
    source: str
    snapshot_id: str
    label: str
    role: str
    clickable: bool | None
    considered_query_field: bool
    consideration_reason: str
    semantic_safety_eligible: bool | None
    semantic_safety_reason: str | None


@dataclass(frozen=True, slots=True)
class QueryFieldVerificationDiagnostics:
    foreground_stable: bool | None
    same_hwnd: bool | None
    same_pid: bool | None
    same_trusted_app_id: bool
    fresh_observation_id: str | None
    fresh_observation_distinct_from_click: bool | None
    focused_control_present: bool
    focused_control_count: int
    focused_control_id: str | None
    focused_control_role: str | None
    focused_control_editable: bool | None
    focused_control_password: bool | None
    focused_control_enabled: bool | None
    focused_control_visible: bool | None
    uia_value_pattern_available: bool | None
    uia_text_pattern_available: bool | None
    clicked_visual_target_id: str | None
    clicked_visual_target_role: str | None
    visual_focus_verification_attempted: bool
    visual_focus_verification_result: str
    visual_focus_candidate_count: int | None
    query_field_verification_method: str
    query_field_verification_failure_reason: str | None
    strong_visual_verification_available: bool
    strong_visual_verification_used: bool
    strong_visual_verification_attempted: bool = False
    strong_visual_grounding_objective: str | None = None
    strong_visual_candidate_count: int = 0
    strong_visual_candidate_ids: tuple[str, ...] = ()
    spatial_correspondence_result: str | None = None
    semantic_correspondence_result: str | None = None
    credential_safety_result: str = "unknown"
    window_geometry_stable: bool | None = None
    strong_visual_verification_result: str = "not_attempted"


@dataclass(frozen=True, slots=True)
class GenericQueryFieldDiagnostics:
    uia_candidate_count: int
    uia_candidates: tuple[QueryFieldCandidateDiagnostic, ...]
    grounding_called: bool
    grounding_objective: str | None
    visual_candidate_count: int
    visual_candidates: tuple[QueryFieldCandidateDiagnostic, ...]
    selected_candidate_id: str | None
    selected_source: str | None
    selected_snapshot_id: str | None
    selection_reason: str
    semantic_safety_eligible: bool | None
    semantic_safety_reason: str | None
    visual_click_snapshot_id: str | None = None
    visual_click_succeeded: bool | None = None
    verification: QueryFieldVerificationDiagnostics | None = None
    visual_observation_id: str | None = None


@dataclass(frozen=True, slots=True)
class _VerifiedVisualQueryFieldBinding:
    """Immutable local provenance for the uniquely selected and verified query field."""

    selected_candidate_id: str
    original_snapshot_id: str
    selected_candidate: VisualElement = field(repr=False)
    original_metadata: ScreenshotMetadata = field(repr=False)
    verified_candidate_id: str
    verified_snapshot_id: str
    verified_candidate: VisualElement = field(repr=False)
    verified_metadata: ScreenshotMetadata = field(repr=False)
    application_id: str
    process_id: int
    window_handle: int
    semantic_role: str
    verification_provenance: str
    unique: bool


def _visual_query_field_geometry_available(
    candidate: VisualElement | None, metadata: ScreenshotMetadata | None,
    snapshot_id: str | None,
) -> bool:
    if (candidate is None or metadata is None or not snapshot_id
            or candidate.source != "visual" or not candidate.clickable
            or not role_is_text_entry(candidate.role)
            or metadata.snapshot_id != snapshot_id
            or metadata.pixel_width <= 0 or metadata.pixel_height <= 0
            or metadata.window_handle <= 0
            or not all(math.isfinite(value) and value > 0
                       for value in (metadata.scale_x, metadata.scale_y))):
        return False
    rect = candidate.rectangle
    return bool(
        all(type(value) is int for value in (
            rect.left, rect.top, rect.right, rect.bottom,
            metadata.window_bounds.left, metadata.window_bounds.top,
            metadata.window_bounds.right, metadata.window_bounds.bottom,
            metadata.capture_bounds.left, metadata.capture_bounds.top,
            metadata.capture_bounds.right, metadata.capture_bounds.bottom,
        ))
        and 0 <= rect.left < rect.right <= metadata.pixel_width
        and 0 <= rect.top < rect.bottom <= metadata.pixel_height
        and metadata.window_bounds.left < metadata.window_bounds.right
        and metadata.window_bounds.top < metadata.window_bounds.bottom
        and metadata.capture_bounds.left < metadata.capture_bounds.right
        and metadata.capture_bounds.top < metadata.capture_bounds.bottom
    )


@dataclass(frozen=True, slots=True)
class QuerySubmitVisualCandidateDiagnostic:
    candidate_id: str
    field_label: str | None
    field_value: str | None
    role: str
    activity: str
    is_query_field: bool
    field_identity_relation: str
    literal_relation: str
    credential_safe: bool | None
    candidate_acceptance: str


@dataclass(frozen=True, slots=True)
class QuerySubmitContinuityDiagnostics:
    result: str
    reason: str
    source: str
    fresh_observation_id: str | None
    same_trusted_app: bool | None
    same_window: bool | None
    field_identity_match: bool | None
    literal_confirmed: bool
    field_focused_or_active: bool | None
    visual_verification_attempted: bool = False
    visual_candidate_count: int = 0
    visual_provider_attempts: tuple[VisualProviderAttempt, ...] = ()
    structural_observation_complete: bool = False
    incompleteness_reasons: tuple[str, ...] = ()
    visual_fallback_eligible: bool = False
    literal_relation: str = "not_evaluated"
    field_continuity_relation: str = "not_evaluated"
    active_focused_relation: str = "not_evaluated"
    credential_safe: bool | None = None
    final_continuity_result: str | None = None
    query_field_candidate_count: int = 0
    visual_candidates: tuple[QuerySubmitVisualCandidateDiagnostic, ...] = ()
    value_read_fallback_eligible: bool = False
    value_read_fallback_attempted: bool = False
    crop_valid: bool | None = None
    value_read_provider_attempts: tuple[VisualProviderAttempt, ...] = ()
    extracted_field_value: str | None = None
    final_submit_release: bool = False
    original_binding_present: bool = False
    original_candidate_id: str | None = None
    original_snapshot_id: str | None = None
    original_binding_unique: bool = False
    original_binding_source: str | None = None
    original_geometry_available: bool = False
    fresh_candidate_count: int = 0
    continuity_comparison_started: bool = False
    failure_stage: str | None = None


@dataclass(frozen=True, slots=True)
class GenericVisualDebugScreenshotDiagnostic:
    stage: str
    saved: bool
    path: str | None = None
    sha256: str | None = None
    byte_length: int | None = None
    matches_request_fingerprint: bool = False
    error: str | None = None


@dataclass(frozen=True, slots=True)
class GenericTaskDebugResult:
    success: bool
    stop_reason: str
    message: str
    steps: int
    target_resolution_status: str | None = None
    frontier_candidate_ids: tuple[str, ...] = ()
    chosen_candidate_id: str | None = None
    jev_confidence: float | None = None
    app_launches: int = 0
    query_field_activations: int = 0
    visual_target_activations: int = 0
    literal_types: int = 0
    query_submits: int = 0
    final_target_activations: int = 0
    visual_grounding_calls: int = 0
    post_action_observation_obtained: bool = False
    post_action_foreground_stable: bool | None = None
    trace: tuple[GenericTaskStep, ...] = ()
    application_activation: ApplicationActivationDiagnostics | None = None
    target_resolution_diagnostics: tuple[GenericTargetResolutionDiagnostics, ...] = ()
    query_field_diagnostics: GenericQueryFieldDiagnostics | None = None
    visual_debug_screenshots: tuple[GenericVisualDebugScreenshotDiagnostic, ...] = ()
    activation_lifecycle: ActivationLifecycleDiagnostics | None = None
    target_decision: GenericTargetDecisionDiagnostics | None = None
    task_decomposition: GenericTaskDecompositionDiagnostics | None = None
    perception_retry: tuple[GenericPerceptionRetryDiagnostics, ...] = ()
    visual_activation: VisualActivationDiagnostic | None = None
    target_activation_postcondition: TargetActivationPostcondition | None = None
    visual_preclick_revalidation: VisualPreclickRevalidationDiagnostic | None = None
    query_submit_continuity: QuerySubmitContinuityDiagnostics | None = None

    @property
    def readiness_retry(self) -> tuple[GenericPerceptionRetryDiagnostics, ...]:
        """Compatibility alias; serialized diagnostics use perception_retry."""
        return self.perception_retry


@dataclass(frozen=True, slots=True)
class TransitionWaitResult:
    changed: bool
    observation: Observation | None
    polls: int
    reason: str


@dataclass(slots=True)
class _RunBudget:
    app_launches: int = 0
    query_field_activations: int = 0
    visual_target_activations: int = 0
    literal_types: int = 0
    query_submits: int = 0
    final_target_activations: int = 0
    visual_grounding_calls: int = 0
    query_field_visual_verification_calls: int = 0
    readiness_retries: int = 0
    readiness_wait_elapsed_seconds: float = 0.0
    steps: int = 0


def _task_piece(value: str) -> str | None:
    """Validate a bounded source slice without rewriting its interior text."""
    piece = value.strip()
    if (not piece or len(piece) > 200 or "\x00" in piece
            or any(ord(character) < 32 and character not in "\t" for character in piece)):
        return None
    if Redactor().clean(piece) != piece:
        return None
    return piece


_COMPOUND_ACTION = re.compile(
    r"\s+(?:and\s+then|and|then)\s+(?:search|find|open|select|choose|play|send|"
    r"delete|buy|purchase|submit|type|click|press)\b", re.I,
)


def _decompose_spanish_task(text: str) -> GenericTaskIntent | None:
    """Parse a small set of explicit Spanish forms into the shared intent model."""
    invalid = GenericTaskIntent(
        None, GenericTaskOperation.UNSUPPORTED, None,
        "unsupported_or_ambiguous", "es",
    )
    if not re.match(r"^\s*(?:abre|busca)\b", text, re.I):
        return None

    app_search = re.fullmatch(
        r"\s*abre\s+(.+?)\s+y\s+busca(?:\s+por)?\s+(.+?)\s*", text, re.I,
    )
    if app_search is not None:
        application = _task_piece(app_search.group(1))
        query = _task_piece(app_search.group(2))
        if (application is None or query is None
                or re.search(r"\s+y\s+(?:abre|busca)\b", query, re.I)):
            return invalid
        return GenericTaskIntent(
            application, GenericTaskOperation.SEARCH, query,
            "app_then_search", "es",
        )

    app_target = re.fullmatch(
        r"\s*abre\s+(.+?)\s+y\s+abre\s+(.+?)\s*", text, re.I,
    )
    if app_target is not None:
        application = _task_piece(app_target.group(1))
        target_source = _task_piece(app_target.group(2))
        if (application is None or target_source is None
                or re.search(r"\s+y\s+(?:abre|busca)\b", target_source, re.I)):
            return invalid
        conversation = re.fullmatch(
            r"(?:el|la)\s+(?:chat|conversaci[oó]n)\s+con\s+(.+?)\s*",
            target_source, re.I,
        )
        target_phrase = (
            f"open the conversation with {conversation.group(1).strip()}"
            if conversation is not None else f"open {target_source}"
        )
        target_phrase = _task_piece(target_phrase)
        if target_phrase is None or requested_target_spec(
            target_phrase, experimental_generic=True,
        ) is None:
            return invalid
        return GenericTaskIntent(
            application, GenericTaskOperation.ACTIVATE_TARGET, target_phrase,
            "app_then_target_action", "es",
        )

    search_first = re.fullmatch(
        r"\s*busca(?:\s+por)?\s+(.+?)\s+en\s+(.+?)\s*", text, re.I,
    )
    if search_first is not None:
        if len(tuple(re.finditer(r"\s+en\s+", text, re.I))) != 1:
            return invalid
        query = _task_piece(search_first.group(1))
        application = _task_piece(search_first.group(2))
        if query is None or application is None:
            return invalid
        return GenericTaskIntent(
            application, GenericTaskOperation.SEARCH, query,
            "search_in_app", "es",
        )

    app_only = re.fullmatch(r"\s*abre\s+(.+?)\s*", text, re.I)
    if app_only is not None:
        application = _task_piece(app_only.group(1))
        if application is not None and re.search(r"\s+y\s+(?:abre|busca)\b", application, re.I):
            return invalid
        return (GenericTaskIntent(
            application, GenericTaskOperation.OPEN_ONLY, None, "app_only", "es",
        ) if application is not None else invalid)
    return invalid


def decompose_generic_task(request: str) -> GenericTaskIntent:
    """Parse only explicit, generic clauses; ambiguous compound phrasing fails closed."""
    text = request.strip()
    source_language = "es" if re.match(r"^\s*(?:abre|busca)\b", text, re.I) else "en"
    invalid = GenericTaskIntent(
        None, GenericTaskOperation.UNSUPPORTED, None,
        "unsupported_or_ambiguous", source_language,
    )
    spanish_intent = _decompose_spanish_task(text)
    if spanish_intent is not None:
        return spanish_intent

    # An explicit app-first search clause makes the boundary unambiguous. The
    # query capture intentionally consumes the rest, preserving conjunctions.
    app_search_separators = tuple(re.finditer(
        r"\s+(?:and\s+then|and|then)\s+(?:search|find)\b", text, re.I,
    ))
    if app_search_separators:
        if len(app_search_separators) != 1:
            return invalid
        match = re.fullmatch(
            r"\s*(?:open|launch|start|run)\s+(.+?)\s+(?:and\s+then|and|then)\s+"
            r"(search|find)(?:\s+for)?\s+(.+?)\s*", text, re.I,
        )
        if match is None:
            return invalid
        application, query = _task_piece(match.group(1)), _task_piece(match.group(3))
        if (application is None or query is None or query.casefold() == "for"
                or _COMPOUND_ACTION.search(query)):
            return invalid
        return GenericTaskIntent(application, GenericTaskOperation.SEARCH, query, "app_then_search")

    app_target_separators = tuple(re.finditer(
        r"\s+(?:and\s+then|and|then)\s+(?:open|select|choose|play|navigate\s+to|go\s+to)\b",
        text, re.I,
    ))
    if app_target_separators:
        if len(app_target_separators) != 1:
            return invalid
        match = re.fullmatch(
            r"\s*(?:open|launch|start|run)\s+(.+?)\s+(?:and\s+then|and|then)\s+"
            r"(open|select|choose|play|navigate\s+to|go\s+to)\s+(.+?)\s*", text, re.I,
        )
        if match is None:
            return invalid
        application = _task_piece(match.group(1))
        verb = match.group(2).casefold()
        target_phrase = _task_piece(f"{verb} {match.group(3)}")
        if application is None or target_phrase is None:
            return invalid
        if requested_target_spec(target_phrase, experimental_generic=True) is None:
            return invalid
        return GenericTaskIntent(
            application, GenericTaskOperation.ACTIVATE_TARGET, target_phrase,
            "app_then_target_action",
        )

    if _COMPOUND_ACTION.search(text):
        return invalid

    # Search-first syntax uses one explicit final app boundary. More than one
    # "in" delimiter is ambiguous and is not resolved by guessing.
    if re.match(r"^\s*search\b", text, re.I):
        in_separators = tuple(re.finditer(r"\s+in\s+", text, re.I))
        if len(in_separators) > 1:
            return invalid
        reverse = re.fullmatch(
            r"\s*search\s+(?:for\s+)?(.+?)\s+in\s+(.+?)\s*", text, re.I,
        )
        if reverse is not None:
            query, application = _task_piece(reverse.group(1)), _task_piece(reverse.group(2))
            if (query is None or application is None or query.casefold() == "for"
                    or _COMPOUND_ACTION.search(query)):
                return invalid
            return GenericTaskIntent(
                application, GenericTaskOperation.SEARCH, query, "search_in_app",
            )
        bare = re.fullmatch(r"\s*search\s+(?:for\s+)?(.+?)\s*", text, re.I)
        if bare is None:
            return invalid
        query = _task_piece(bare.group(1))
        if query is not None and (query.casefold() == "for" or _COMPOUND_ACTION.search(query)):
            return invalid
        return (GenericTaskIntent(None, GenericTaskOperation.SEARCH, query, "search_current_app")
                if query is not None else invalid)

    # "Open X" without a post-launch operation denotes an application unless
    # the phrase carries an established typed UI target role (for example a
    # file or a conversation). This keeps those legacy target forms intact.
    open_only = re.fullmatch(r"\s*(?:open|launch|start|run)\s+(.+?)\s*", text, re.I)
    if open_only is not None:
        target = requested_target_spec(text, experimental_generic=True)
        if target is not None and target.desired_role is not None:
            return GenericTaskIntent(
                None, GenericTaskOperation.ACTIVATE_TARGET, text, "typed_target_request",
            )
        application = _task_piece(open_only.group(1))
        return (GenericTaskIntent(application, GenericTaskOperation.OPEN_ONLY, None, "app_only")
                if application is not None else invalid)

    # Requests without an app clause retain the existing generic target parser.
    target = requested_target_spec(text, experimental_generic=True)
    if target is not None:
        return GenericTaskIntent(
            None, GenericTaskOperation.ACTIVATE_TARGET, text, "target_request",
        )
    # Do not reinterpret unsupported compound instructions as an application
    # name or as one long target phrase.
    if _COMPOUND_ACTION.search(text):
        return invalid
    return invalid


def _target_spec_for_intent(intent: GenericTaskIntent) -> TargetSpec | None:
    if intent.operation is GenericTaskOperation.OPEN_ONLY:
        return None
    if intent.operation is GenericTaskOperation.SEARCH:
        if intent.operation_target is None:
            return None
        try:
            return TargetSpec(intent.operation_target, action_intent="search")
        except ValueError:
            return None
    if intent.operation is GenericTaskOperation.ACTIVATE_TARGET and intent.operation_target:
        return requested_target_spec(intent.operation_target, experimental_generic=True)
    return None


def _observation_signature(observation: Observation) -> tuple[object, ...]:
    return (
        observation.application_id, observation.process_id,
        tuple((
            item.control_type, item.name[:120], item.automation_id[:80],
            (item.observed_text or "")[:120], item.visible, item.enabled,
        ) for item in observation.elements[:80]),
    )


def wait_for_local_transition(
    computer: GenericTaskComputer,
    baseline: Observation,
    *,
    timeout_seconds: float = 2.0,
    poll_seconds: float = 0.2,
    clock: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
    max_polls: int = 30,
) -> TransitionWaitResult:
    """Poll local UIA only until a bounded foreground-stable change is visible."""
    start = clock()
    deadline = start + timeout_seconds
    signature = _observation_signature(baseline)
    latest: Observation | None = None
    for poll in range(1, max_polls + 1):
        if clock() >= deadline:
            break
        try:
            latest = computer.observe_local()
        except KeyboardInterrupt:
            raise
        except Exception:
            return TransitionWaitResult(False, None, poll, "local_observation_failed")
        if latest.error:
            return TransitionWaitResult(False, latest, poll, "local_observation_error")
        if ((baseline.application_id and latest.application_id != baseline.application_id)
                or (baseline.process_id is not None and latest.process_id != baseline.process_id)):
            return TransitionWaitResult(False, latest, poll, "foreground_changed")
        base_meta, latest_meta = baseline.screenshot, latest.screenshot
        if (base_meta is not None and latest_meta is not None
                and base_meta.window_handle != latest_meta.window_handle):
            return TransitionWaitResult(False, latest, poll, "foreground_changed")
        if _observation_signature(latest) != signature:
            return TransitionWaitResult(True, latest, poll, "local_transition_observed")
        remaining = deadline - clock()
        if remaining <= 0:
            break
        sleep_fn(min(poll_seconds, remaining))
    return TransitionWaitResult(False, latest, min(max_polls, poll if 'poll' in locals() else 0),
                                "transition_timeout")


@dataclass
class GenericTaskDebugAgent:
    computer: GenericTaskComputer
    decision_maker: GenericTargetDecisionMaker
    app_catalog: ApplicationCatalog | None = None
    policy: GenericTargetActivationPolicy = field(default_factory=GenericTargetActivationPolicy)
    budgets: GenericTaskBudgets = field(default_factory=GenericTaskBudgets)
    redactor: Redactor = field(default_factory=Redactor)
    sleep_fn: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic
    visual_debug_capture_callback: Callable[
        [str, Observation], GenericVisualDebugScreenshotDiagnostic,
    ] | None = None
    _active_budget: _RunBudget = field(init=False, repr=False)
    _active_trace: list[GenericTaskStep] = field(init=False, repr=False)
    _active_resolution_diagnostics: list[GenericTargetResolutionDiagnostics] = field(
        init=False, repr=False,
    )
    _active_visual_debug_screenshots: list[GenericVisualDebugScreenshotDiagnostic] = field(
        init=False, repr=False,
    )
    _active_query_field_diagnostics: GenericQueryFieldDiagnostics | None = field(
        init=False, default=None, repr=False,
    )
    _active_visual_query_field_binding: _VerifiedVisualQueryFieldBinding | None = field(
        init=False, default=None, repr=False,
    )
    _active_query_submit_continuity: QuerySubmitContinuityDiagnostics | None = field(
        init=False, default=None, repr=False,
    )
    _active_target_decision_diagnostics: GenericTargetDecisionDiagnostics | None = field(
        init=False, default=None, repr=False,
    )
    _active_visual_activation: VisualActivationDiagnostic | None = field(
        init=False, default=None, repr=False,
    )
    _active_target_activation_postcondition: TargetActivationPostcondition | None = field(
        init=False, default=None, repr=False,
    )
    _active_visual_preclick_revalidation: VisualPreclickRevalidationDiagnostic | None = field(
        init=False, default=None, repr=False,
    )
    _active_chosen_id: str | None = field(init=False, default=None, repr=False)
    _active_confidence: float | None = field(init=False, default=None, repr=False)
    _active_post_observation: bool = field(init=False, default=False, repr=False)
    _active_post_foreground_stable: bool | None = field(init=False, default=None, repr=False)
    _active_last_visual_observation: Observation | None = field(
        init=False, default=None, repr=False,
    )
    _active_last_visual_grounding_outcome: str = field(
        init=False, default="not_attempted", repr=False,
    )

    def run(self, request: str) -> GenericTaskDebugResult:
        budget = _RunBudget()
        trace: list[GenericTaskStep] = []
        self._active_budget, self._active_trace = budget, trace
        self._active_resolution_diagnostics = []
        self._active_visual_debug_screenshots = []
        self._active_query_field_diagnostics = None
        self._active_visual_query_field_binding = None
        self._active_query_submit_continuity = None
        self._active_target_decision_diagnostics = None
        self._active_visual_activation = None
        self._active_target_activation_postcondition = None
        self._active_visual_preclick_revalidation = None
        self._active_chosen_id = None
        self._active_confidence = None
        self._active_post_observation = False
        self._active_post_foreground_stable = None
        self._active_last_visual_observation = None
        self._active_last_visual_grounding_outcome = "not_attempted"
        last_resolution: TargetResolution | None = None
        last_observation: Observation | None = None
        activation_diagnostics: ApplicationActivationDiagnostics | None = None
        activation_lifecycle: ActivationLifecycleDiagnostics | None = None
        readiness_activation_observation: Observation | None = None
        readiness_activation_app_id: str | None = None
        readiness_activation_verified = False
        readiness_diagnostics: list[GenericReadinessRetryDiagnostics] = []
        task_decomposition: GenericTaskDecompositionDiagnostics | None = None

        def finish(success: bool, reason: str, message: str) -> GenericTaskDebugResult:
            return GenericTaskDebugResult(
                success, reason, message, budget.steps,
                last_resolution.status.value if last_resolution else None,
                last_resolution.frontier_candidate_ids if last_resolution else (),
                self._active_chosen_id, self._active_confidence, budget.app_launches,
                budget.query_field_activations, budget.visual_target_activations,
                budget.literal_types, budget.query_submits, budget.final_target_activations,
                budget.visual_grounding_calls, self._active_post_observation,
                self._active_post_foreground_stable,
                tuple(trace), activation_diagnostics,
                tuple(self._active_resolution_diagnostics),
                self._active_query_field_diagnostics,
                tuple(self._active_visual_debug_screenshots),
                activation_lifecycle,
                self._active_target_decision_diagnostics,
                task_decomposition,
                tuple(readiness_diagnostics),
                self._active_visual_activation,
                self._active_target_activation_postcondition,
                self._active_visual_preclick_revalidation,
                self._active_query_submit_continuity,
            )

        if not isinstance(request, str) or not request.strip() or self.redactor.clean(request) != request:
            return finish(False, "invalid_request", "A non-empty request without credentials is required.")
        intent = decompose_generic_task(request)
        target = _target_spec_for_intent(intent)
        diagnostic_target = target.primary_identity if target is not None else intent.operation_target
        task_decomposition = GenericTaskDecompositionDiagnostics(
            intent.source_language,
            self._diagnostic_text(intent.application_request, 120) or None,
            intent.operation.value,
            self._diagnostic_text(diagnostic_target, 120) or None,
            intent.decomposition_method,
        )
        if intent.operation is GenericTaskOperation.UNSUPPORTED:
            return finish(False, "unsupported_instruction", "The request could not be decomposed safely.")
        if intent.operation is not GenericTaskOperation.OPEN_ONLY and target is None:
            return finish(False, "target_unavailable", "A bounded operation target could not be derived.")
        if intent.operation is GenericTaskOperation.OPEN_ONLY and self.app_catalog is None:
            return finish(False, "application_unavailable", "A trusted application catalog is required to open an application.")

        try:
            observation = self.computer.observe_local()
        except KeyboardInterrupt:
            raise
        except Exception:
            return finish(False, "observation_failed", "Initial local observation failed.")
        budget.steps += 1
        last_observation = observation
        trace.append(GenericTaskStep(budget.steps, GenericCapability.OBSERVE, observation.observation_id))
        if observation.error or not observation.observation_id:
            return finish(False, "observation_failed", "A fresh local observation is required.")

        app_query = intent.application_request
        if (app_query is not None and self.app_catalog is None
                and intent.operation is GenericTaskOperation.SEARCH):
            return finish(False, "application_unavailable", "The requested application cannot be verified against the trusted catalog.")
        if app_query is not None and self.app_catalog is not None:
            matches = self.app_catalog.find(app_query, 5)
            if not matches:
                return finish(False, "application_unavailable", "The requested application is not in the trusted catalog.")
            best_score = matches[0].score
            best = tuple(item for item in matches if item.score == best_score)
            if len(best) != 1:
                return finish(False, "application_ambiguous", "The requested application did not resolve uniquely.")
            candidate = best[0].candidate
            if observation.application_id != candidate.id:
                if candidate.launch_policy != "allow":
                    return finish(False, "application_not_approved", "The catalog does not approve unattended launch.")
                if budget.app_launches >= self.budgets.app_launches:
                    return finish(False, "budget_exhausted", "The application launch budget is exhausted.")
                if budget.steps + 2 > self.budgets.max_steps:
                    return finish(False, "budget_exhausted", "The overall step budget is exhausted.")
                action = OpenAppAction(candidate.id)
                verdict = self.policy.validate(action, observation)
                if verdict.disposition != "allow":
                    return finish(False, "application_safety_denied", verdict.reason)
                budget.app_launches += 1
                budget.steps += 1
                open_app_action_started_at = self.clock()
                result = self.computer.execute(action, observation)
                open_app_action_elapsed_ms = max(
                    0, round((self.clock() - open_app_action_started_at) * 1000),
                )
                trace.append(GenericTaskStep(
                    budget.steps, GenericCapability.OPEN_APPLICATION, observation.observation_id,
                    action_kind=action.kind, action_succeeded=result.success,
                ))
                if not result.success:
                    return finish(False, "application_launch_failed", "The trusted application did not launch.")
                activation = self._wait_for_application(
                    candidate, observation,
                    open_app_action_started_at=open_app_action_started_at,
                    open_app_action_elapsed_ms=open_app_action_elapsed_ms,
                )
                observation = activation.observation
                activation_diagnostics = activation.diagnostics
                activation_lifecycle = activation.lifecycle
                budget.steps += 1
                last_observation = observation
                trace.append(GenericTaskStep(
                    budget.steps, GenericCapability.WAIT_FOR_TRANSITION,
                    observation.observation_id if observation else None,
                ))
                if observation is None or observation.error or observation.application_id != candidate.id:
                    return finish(False, "application_activation_timeout", "The requested application did not become foreground.")
                readiness_activation_observation = observation
                readiness_activation_app_id = candidate.id
                lifecycle_ready = (
                    activation_lifecycle is None
                    or (
                        activation_lifecycle.target_foreground_observed is True
                        and activation_lifecycle.target_probe_complete is True
                        and activation_lifecycle.enumeration_complete is True
                        and activation_lifecycle.window_resolution_status == "unique"
                        and activation_lifecycle.primary_surface_resolution == "unique"
                        and (
                            activation_lifecycle.target_window_state is None
                            or (
                                activation_lifecycle.target_window_state.trusted_identity_match
                                and activation_lifecycle.target_window_state.visible is True
                                and activation_lifecycle.target_window_state.minimized is False
                                and activation_lifecycle.target_window_state.foreground is True
                            )
                        )
                    )
                )
                readiness_activation_verified = bool(
                    activation.reason == "activated"
                    and activation.diagnostics.identity_match is True
                    and lifecycle_ready
                )

        if intent.operation is GenericTaskOperation.OPEN_ONLY:
            budget.steps += 1
            trace.append(GenericTaskStep(
                budget.steps, GenericCapability.FINISH,
                observation.observation_id,
            ))
            return finish(True, "application_opened", "The requested trusted application is foreground.")

        # Resolve current UIA first. A remote visual call happens only when the
        # current local evidence cannot safely resolve the target.
        if intent.operation is not GenericTaskOperation.SEARCH:
            def target_discovery(source: Observation):
                resolved_observation, resolution, evidence_set = self._resolve_with_optional_visual(
                    target, source, allow_grounding=True, reason="target-resolution",
                )
                assessment = self._target_readiness_assessment(
                    source, resolved_observation, resolution,
                )
                return resolved_observation, (resolution, evidence_set), assessment

            observation, target_result, readiness_stop = self._run_readiness_discovery(
                GenericCapability.RESOLVE_TARGET, observation,
                activation_observation=readiness_activation_observation,
                activated_app_id=readiness_activation_app_id,
                activation_verified=readiness_activation_verified,
                budget=budget, trace=trace, diagnostics=readiness_diagnostics,
                discover=target_discovery,
            )
            last_resolution, evidence_set = target_result
            last_observation = observation
            if readiness_stop in {"foreground_changed", "trusted_identity_changed",
                                  "window_disappeared", "stale_observation",
                                  "observation_incomplete"}:
                return finish(False, "readiness_context_changed", "The foreground context changed during readiness observation.")
            if self._is_resolvable(last_resolution):
                return self._select_activate_and_stop(
                    request, target, last_resolution, evidence_set, observation,
                    budget, trace, finish,
                )
            if last_resolution.status is TargetResolutionStatus.AMBIGUOUS:
                return finish(False, "target_ambiguous", "Local and visual evidence does not distinguish the target.")

        # Search requests already have a deterministic operation payload, so
        # they go directly to query-field discovery without resolving the full
        # original command as a UI target.
        capabilities = (
            self.derive_capabilities(target, observation, last_resolution, budget)
            if last_resolution is not None else ()
        )
        if (intent.operation is not GenericTaskOperation.SEARCH
                and not ({GenericCapability.FIND_QUERY_FIELD, GenericCapability.ENTER_LITERAL_QUERY}
                         & set(capabilities))):
            return finish(False, "search_not_available", "No bounded search capability is available from the current state.")
        def query_field_discovery(source: Observation):
            self._active_last_visual_observation = None
            field_observation, action = self._find_query_field(source, budget, trace)
            assessment = self._query_field_readiness_assessment(source, field_observation)
            return field_observation, (field_observation, action), assessment

        observation, query_result, readiness_stop = self._run_readiness_discovery(
            GenericCapability.FIND_QUERY_FIELD, observation,
            activation_observation=readiness_activation_observation,
            activated_app_id=readiness_activation_app_id,
            activation_verified=readiness_activation_verified,
            budget=budget, trace=trace, diagnostics=readiness_diagnostics,
            discover=query_field_discovery,
        )
        field_observation, query_field_action = query_result
        if readiness_stop in {"foreground_changed", "trusted_identity_changed",
                              "window_disappeared", "stale_observation",
                              "observation_incomplete"}:
            return finish(False, "readiness_context_changed", "The foreground context changed during readiness observation.")
        if field_observation is None:
            return finish(False, "query_field_unavailable", "No safe generic search field was available.")
        observation = field_observation
        query_field_click_observation = observation
        last_observation = observation
        click_result: ActionResult | None = None
        if query_field_action is not None:
            if budget.steps + 2 > self.budgets.max_steps:
                return finish(False, "budget_exhausted", "The overall step budget is exhausted.")
            if budget.query_field_activations >= self.budgets.query_field_activations:
                return finish(False, "budget_exhausted", "The query-field activation budget is exhausted.")
            if isinstance(query_field_action, VisualClickAction):
                if budget.visual_target_activations >= self.budgets.visual_target_activations:
                    return finish(False, "budget_exhausted", "The visual activation budget is exhausted.")
                budget.visual_target_activations += 1
            budget.query_field_activations += 1
            click_result = self._execute_target_action(query_field_action, observation, None)
            budget.steps += 1
            trace.append(GenericTaskStep(
                budget.steps, GenericCapability.FIND_QUERY_FIELD, observation.observation_id,
                action_kind=query_field_action.kind, action_succeeded=click_result.success,
            ))
            if isinstance(query_field_action, VisualClickAction) and self._active_query_field_diagnostics is not None:
                self._active_query_field_diagnostics = replace(
                    self._active_query_field_diagnostics,
                    visual_click_snapshot_id=query_field_action.snapshot_id,
                    visual_click_succeeded=click_result.success and click_result.input_issued,
                )
            if not click_result.success:
                return finish(False, "query_field_activation_failed", "The query field could not be activated safely.")
            try:
                observation = self.computer.observe_local()
            except KeyboardInterrupt:
                raise
            except Exception:
                return finish(False, "observation_failed", "Observation after query-field activation failed.")
            budget.steps += 1
            last_observation = observation
            trace.append(GenericTaskStep(budget.steps, GenericCapability.OBSERVE, observation.observation_id))
        type_observation, verification_method, verification_diagnostics = (
            self._verify_query_field_after_click(
                query_field_click_observation, observation, query_field_action,
                click_result, request, budget, trace,
            )
        )
        if self._active_query_field_diagnostics is not None:
            self._active_query_field_diagnostics = replace(
                self._active_query_field_diagnostics,
                verification=verification_diagnostics,
            )
        if type_observation is None or verification_method == "none":
            return finish(False, "query_field_not_verified", "A focused, non-password query field was not verified.")

        if budget.literal_types >= self.budgets.literal_types:
            return finish(False, "budget_exhausted", "The literal Type budget is exhausted.")
        try:
            literal = query_literal_from_target(target)
        except ValueError:
            return finish(False, "query_unavailable", "A deterministic literal query was unavailable.")
        type_action = TypeAction(literal)
        if budget.steps + 2 > self.budgets.max_steps:
            return finish(False, "budget_exhausted", "The overall step budget is exhausted.")
        type_verdict = self.policy.validate(type_action, type_observation)
        if type_verdict.disposition != "allow":
            return finish(False, "type_safety_denied", type_verdict.reason)
        budget.literal_types += 1
        budget.steps += 1
        if verification_method == "strong_visual":
            verified_type_executor = getattr(self.computer, "execute_type_phase2", None)
            if not callable(verified_type_executor):
                return finish(False, "query_field_not_verified", "Verified literal typing is unavailable.")
            try:
                type_result = verified_type_executor(
                    type_action, type_observation, visual_verified=True,
                )
            except KeyboardInterrupt:
                raise
            except Exception:
                type_result = ActionResult(
                    False, type_action, "Verified literal input failed.",
                    error="windows_operation_failed", input_issued=False,
                )
        else:
            type_result = self.computer.execute(type_action, type_observation)
        trace.append(GenericTaskStep(
            budget.steps, GenericCapability.ENTER_LITERAL_QUERY, type_observation.observation_id,
            action_kind=type_action.kind, action_succeeded=type_result.success,
        ))
        if not type_result.success:
            return finish(False, "literal_type_failed", "The bounded literal query was not safely entered.")
        typed_from = type_observation

        try:
            observation = self.computer.observe_local()
        except KeyboardInterrupt:
            raise
        except Exception:
            return finish(False, "observation_failed", "Observation after literal typing failed.")
        budget.steps += 1
        last_observation = observation
        trace.append(GenericTaskStep(budget.steps, GenericCapability.OBSERVE, observation.observation_id))
        submit_visual_observation: Observation | None = None
        if intent.operation is GenericTaskOperation.SEARCH:
            continuity, submit_observation = self._verify_query_field_continuity_after_type(
                query_field_click_observation, typed_from, observation,
                query_field_action, click_result, request, literal, budget, trace,
            )
            self._active_query_submit_continuity = continuity
            if continuity.result != "verified" or submit_observation is None:
                return finish(
                    False, "query_submit_not_eligible",
                    "The previously verified query field and literal could not be revalidated.",
                )
            submit_visual_observation = (
                submit_observation
                if continuity.source == "visual" else None
            )
            observation = submit_observation
            last_observation = observation
            # A SEARCH query is text to preserve in its original field, not a
            # new actionable target to resolve before submitting.
            last_resolution = None
        else:
            observation, last_resolution, evidence_set = self._resolve_with_optional_visual(
                target, observation, allow_grounding=True,
                reason="target-resolution-after-type",
            )
            last_observation = observation
            if self._is_resolvable(last_resolution):
                return self._select_activate_and_stop(
                    request, target, last_resolution, evidence_set, observation,
                    budget, trace, finish,
                )
            if last_resolution.status is TargetResolutionStatus.AMBIGUOUS:
                return finish(False, "target_ambiguous", "The visible target remains ambiguous after typing.")

        capabilities = self.derive_capabilities(
            target, observation, last_resolution, budget,
            query_field_continuity_verified=intent.operation is GenericTaskOperation.SEARCH,
        )
        if GenericCapability.SUBMIT_QUERY not in capabilities:
            return finish(False, "query_submit_not_eligible", "Local capability policy did not authorize query submission.")
        if budget.query_submits >= self.budgets.query_submits:
            return finish(False, "budget_exhausted", "The query-submit budget is exhausted.")
        if budget.steps + 2 > self.budgets.max_steps:
            return finish(False, "budget_exhausted", "The overall step budget is exhausted.")
        submit_action = QuerySubmitAction()
        budget.query_submits += 1
        budget.steps += 1
        submit_result = self._execute_query_submit(
            submit_action, observation,
            verified_search_observation=submit_visual_observation,
        )
        trace.append(GenericTaskStep(
            budget.steps, GenericCapability.SUBMIT_QUERY, observation.observation_id,
            action_kind=submit_action.kind, action_succeeded=submit_result.success,
        ))
        if not submit_result.success or not submit_result.input_issued:
            return finish(False, "query_submit_failed", "The verified query was not submitted.")

        capabilities = self.derive_capabilities(target, observation, last_resolution, budget)
        if GenericCapability.WAIT_FOR_TRANSITION not in capabilities:
            return finish(False, "transition_not_available", "Local state does not permit a transition wait.")

        transition = wait_for_local_transition(
            self.computer, observation,
            timeout_seconds=self.budgets.transition_timeout_seconds,
            poll_seconds=self.budgets.transition_poll_seconds,
            clock=self.clock, sleep_fn=self.sleep_fn,
        )
        budget.steps += 1
        observation = transition.observation
        last_observation = observation
        trace.append(GenericTaskStep(
            budget.steps, GenericCapability.WAIT_FOR_TRANSITION,
            observation.observation_id if observation else None,
        ))
        if not transition.changed or observation is None:
            return finish(False, transition.reason, "A bounded local transition was not observed.")
        if intent.operation is GenericTaskOperation.SEARCH:
            observation, last_resolution, evidence_set = self._resolve_with_optional_visual(
                target, observation, allow_grounding=True,
                reason="search-results-after-submit",
            )
            last_observation = observation
            if not self._is_resolvable(last_resolution):
                reason = (
                    "target_ambiguous"
                    if last_resolution.status is TargetResolutionStatus.AMBIGUOUS
                    else "search_results_not_verified"
                )
                return finish(False, reason, "Submitted search results could not be verified safely.")
            budget.steps += 1
            trace.append(GenericTaskStep(
                budget.steps, GenericCapability.FINISH, observation.observation_id,
            ))
            return finish(
                True, "search_completed",
                "The verified query was submitted once and a matching result was observed; no result was activated.",
            )
        observation, last_resolution, evidence_set = self._resolve_with_optional_visual(
            target, observation, allow_grounding=True, reason="target-resolution-after-submit",
        )
        last_observation = observation
        if not self._is_resolvable(last_resolution):
            reason = "target_ambiguous" if last_resolution.status is TargetResolutionStatus.AMBIGUOUS else "target_not_found"
            return finish(False, reason, "No safe target was resolved after the single query submission.")
        return self._select_activate_and_stop(
            request, target, last_resolution, evidence_set, observation,
            budget, trace, finish,
        )

    @staticmethod
    def _is_resolvable(resolution: TargetResolution) -> bool:
        return resolution.status in {TargetResolutionStatus.UNIQUE, TargetResolutionStatus.CHOICE}

    def derive_capabilities(
        self, target: TargetSpec, observation: Observation,
        resolution: TargetResolution | None, budget: _RunBudget, *,
        query_field_continuity_verified: bool = False,
    ) -> tuple[GenericCapability, ...]:
        """Expose only next steps justified by the current snapshot and budgets."""
        if target.action_intent == "search":
            available: list[GenericCapability] = []
            if (budget.literal_types < self.budgets.literal_types
                    and budget.query_field_activations < self.budgets.query_field_activations):
                available.append(
                    GenericCapability.ENTER_LITERAL_QUERY
                    if self._focused_query_field(observation) is not None
                    else GenericCapability.FIND_QUERY_FIELD
                )
            if (budget.literal_types > 0 and budget.query_submits < self.budgets.query_submits
                    and (query_field_continuity_verified
                         or (focused := self._focused_query_field(observation)) is not None)):
                try:
                    literal = query_literal_from_target(target)
                except ValueError:
                    literal = ""
                if literal and (query_field_continuity_verified
                                or self._query_value_matches(focused, literal)):
                    available.append(GenericCapability.SUBMIT_QUERY)
            if budget.query_submits > 0:
                available.append(GenericCapability.WAIT_FOR_TRANSITION)
                if (getattr(self.computer, "visual_provider", True) is not None
                        and not observation.visual_elements
                        and budget.visual_grounding_calls < self.budgets.visual_grounding_calls):
                    available.append(GenericCapability.RESOLVE_TARGET)
            available.append(GenericCapability.STOP)
            return tuple(available)
        if resolution is None:
            return (GenericCapability.STOP,)
        if self._is_resolvable(resolution):
            return (GenericCapability.ACTIVATE_TARGET, GenericCapability.FINISH,
                    GenericCapability.STOP)
        available: list[GenericCapability] = []
        if (budget.literal_types < self.budgets.literal_types
                and budget.query_field_activations < self.budgets.query_field_activations):
            available.append(
                GenericCapability.ENTER_LITERAL_QUERY
                if self._focused_query_field(observation) is not None
                else GenericCapability.FIND_QUERY_FIELD
            )
        if (budget.literal_types > 0 and budget.query_submits < self.budgets.query_submits
                and (focused := self._focused_query_field(observation)) is not None):
            try:
                literal = query_literal_from_target(target)
            except ValueError:
                literal = ""
            if literal and self._query_value_matches(focused, literal):
                available.append(GenericCapability.SUBMIT_QUERY)
        if budget.query_submits > 0:
            available.append(GenericCapability.WAIT_FOR_TRANSITION)
        if (getattr(self.computer, "visual_provider", True) is not None
                and not observation.visual_elements
                and budget.visual_grounding_calls < self.budgets.visual_grounding_calls):
            available.append(GenericCapability.RESOLVE_TARGET)
        available.append(GenericCapability.STOP)
        return tuple(available)

    def _resolve(
        self, target: TargetSpec, observation: Observation, *,
        grounding_binding: DirectedGroundingBinding | None = None,
    ) -> tuple[TargetResolution, TargetEvidenceSet]:
        evidence_set = adapt_observation_candidates(
            observation, self.policy, self.redactor, grounding_binding=grounding_binding,
        )
        resolution = resolve_target(
            target, evidence_set.candidates,
            expected_snapshot_id=observation.observation_id,
            max_frontier_candidates=self.budgets.max_frontier_candidates,
            frontier_mode=True,
        )
        return resolution, evidence_set

    def _diagnostic_text(self, value: str | None, limit: int = MAX_DIAGNOSTIC_LABEL_CHARS) -> str:
        if not value:
            return ""
        cleaned = " ".join(self.redactor.clean(value).split())
        return cleaned[:limit]

    def _target_spec_diagnostic(self, target: TargetSpec) -> GenericTargetSpecDiagnostic:
        return GenericTargetSpecDiagnostic(
            self._diagnostic_text(target.primary_identity, 120),
            tuple(self._diagnostic_text(value, 80) for value in target.qualifiers[:8]),
            self._diagnostic_text(target.desired_role, 80) or None,
            self._diagnostic_text(target.action_intent, 40),
        )

    def _source_diagnostics(
        self, source: str, observation: Observation, evidence: TargetEvidenceSet,
        resolution: TargetResolution, *, grounding_called: bool | None = None,
        grounding_objective: str | None = None,
    ) -> CandidateSourceDiagnostics:
        source_name = source.upper()
        adapted = tuple(item for item in evidence.candidates if item.source.upper() == source_name)
        rows = tuple(item for item in resolution.candidates if item.source.upper() == source_name)
        reasons = Counter(reason for row in rows for reason in row.rejection_reasons)
        frontier = set(resolution.frontier_candidate_ids)
        pipeline = observation.visual_pipeline if source_name == "VISUAL" else None
        raw_count = len(observation.visual_elements) if source_name == "VISUAL" else len(observation.elements)
        return CandidateSourceDiagnostics(
            observation.observation_id or None,
            raw_count,
            len(adapted),
            sum(row.admissible for row in rows),
            sum(not row.admissible for row in rows),
            tuple(sorted(reasons.items())),
            sum(row.candidate_id in frontier for row in rows),
            grounding_called,
            self._diagnostic_text(grounding_objective, 180) or None,
            pipeline.provider_raw_element_count if pipeline else None,
            pipeline.parsed_element_count if pipeline else None,
            pipeline.validated_element_count if pipeline else None,
            observation.visual_provider_attempts if source_name == "VISUAL" else (),
            observation.selected_visual_provider if source_name == "VISUAL" else None,
            observation.provider_failover_used if source_name == "VISUAL" else False,
            observation.provider_failover_reason if source_name == "VISUAL" else None,
            observation.visual_provider_error if source_name == "VISUAL" else None,
        )

    def _candidate_semantic_diagnostic(
        self, row, observation: Observation,
    ) -> CandidateSemanticDiagnostic:
        primary_text: str | None = None
        secondary = tuple(self._diagnostic_text(value, 80) for value in row.secondary_text[:3])
        if row.source.upper() == "UIA":
            control = next((item for item in observation.elements
                            if item.id == row.candidate_id), None)
            # A nameless editable control can carry user-entered text in its
            # resolver evidence. Keep that value out of debug output.
            if control is not None and control.name.strip():
                primary_text = self._diagnostic_text(control.name, 100)
        else:
            element = next((item for item in observation.visual_elements
                            if item.id == row.candidate_id), None)
            if element is not None:
                primary_text = self._diagnostic_text(element.label, 100)
        return CandidateSemanticDiagnostic(
            row.candidate_id[:80], row.source[:40], row.snapshot_id[:128], primary_text,
            tuple(value for value in secondary if value),
            self._diagnostic_text(row.semantic_role, 60) or None,
            self._diagnostic_text(row.provider_role, 60) or None,
            row.primary_identity.value,
            tuple(item.value for item in row.qualifier_evidence[:8]),
            row.role_compatibility.value,
            row.actionable, row.geometry_valid, row.safety_eligible, row.admissible,
            tuple(reason[:80] for reason in row.rejection_reasons[:8]),
            self._diagnostic_text(row.target_semantic_evidence, 60) or None,
            row.domain_semantic_compatibility.value,
            row.presentation_role.value,
            row.presentation_compatibility.value,
        )

    @staticmethod
    def _text_tokens(value: str) -> set[str]:
        return set(re.findall(r"[^\W_]+", value.casefold(), flags=re.UNICODE))

    def _append_resolution_diagnostics(
        self, target: TargetSpec, attempt_stage: str,
        uia_observation: Observation, uia_evidence: TargetEvidenceSet,
        uia_resolution: TargetResolution,
        visual_observation: Observation | None, visual_evidence: TargetEvidenceSet,
        visual_resolution: TargetResolution,
        final_observation: Observation, final_resolution: TargetResolution,
        *, grounding_called: bool, grounding_objective: str | None,
    ) -> None:
        visual_source_observation = visual_observation or final_observation
        uia_debug = self._source_diagnostics(
            "UIA", uia_observation, uia_evidence, uia_resolution,
        )
        visual_debug = self._source_diagnostics(
            "VISUAL", visual_source_observation, visual_evidence, visual_resolution,
            grounding_called=grounding_called, grounding_objective=grounding_objective,
        )
        by_id = {item.candidate_id: item for item in final_resolution.candidates}
        frontier_ids = final_resolution.frontier_candidate_ids
        frontier_evidence = tuple(
            self._candidate_semantic_diagnostic(by_id[candidate_id], final_observation)
            for candidate_id in frontier_ids[:MAX_DIAGNOSTIC_FRONTIER]
            if candidate_id in by_id
        )
        query_tokens = self._text_tokens(" ".join((target.primary_identity, *target.qualifiers)))
        near_ranked: list[tuple[int, int, object]] = []
        for index, row in enumerate(final_resolution.candidates):
            display: str | None = None
            if row.source.upper() == "UIA":
                control = next((item for item in final_observation.elements
                                if item.id == row.candidate_id), None)
                if control is not None and control.name.strip():
                    display = control.name
            else:
                element = next((item for item in final_observation.visual_elements
                                if item.id == row.candidate_id), None)
                if element is not None:
                    display = element.label
            overlap = len(query_tokens & self._text_tokens(display or ""))
            identity_match = row.primary_identity.value == "match"
            if overlap or identity_match:
                near_ranked.append((overlap, -index, row))
        near_ranked.sort(key=lambda value: (-value[0], -value[1]))
        near_matches = tuple(
            self._candidate_semantic_diagnostic(item, final_observation)
            for _overlap, _order, item in near_ranked[:MAX_DIAGNOSTIC_NEAR_MATCHES]
        )
        self._active_resolution_diagnostics.append(GenericTargetResolutionDiagnostics(
            attempt_stage[:80], self._target_spec_diagnostic(target),
            uia_debug, visual_debug, final_resolution.status.value,
            final_resolution.resolution_reason[:120],
            tuple(row.candidate_id[:80] for row in final_resolution.candidates[:MAX_DIAGNOSTIC_CANDIDATES]),
            len(frontier_ids),
            tuple(candidate_id[:80] for candidate_id in frontier_ids[:MAX_DIAGNOSTIC_FRONTIER]),
            frontier_evidence, near_matches,
        ))

    def _record_visual_debug_capture(self, stage: str, observation: Observation) -> None:
        callback = self.visual_debug_capture_callback
        if callback is None:
            return
        try:
            result = callback(stage, observation)
        except KeyboardInterrupt:
            raise
        except Exception:
            result = GenericVisualDebugScreenshotDiagnostic(
                stage, False, error="capture_callback_failed",
            )
        if not isinstance(result, GenericVisualDebugScreenshotDiagnostic):
            result = GenericVisualDebugScreenshotDiagnostic(
                stage, False, error="invalid_capture_callback_result",
            )
        self._active_visual_debug_screenshots.append(result)

    def _resolve_with_optional_visual(
        self, target: TargetSpec, observation: Observation, *,
        allow_grounding: bool, reason: str,
    ) -> tuple[Observation, TargetResolution, TargetEvidenceSet]:
        self._active_last_visual_observation = None
        self._active_last_visual_grounding_outcome = "not_attempted"
        local_observation = observation
        local_resolution, local_evidence = self._resolve(target, local_observation)
        capabilities = self.derive_capabilities(
            target, local_observation, local_resolution, self._active_budget,
        )
        can_ground = (
            not self._is_resolvable(local_resolution) and allow_grounding
            and GenericCapability.RESOLVE_TARGET in capabilities
        )
        if not can_ground:
            self._append_resolution_diagnostics(
                target, reason, local_observation, local_evidence, local_resolution,
                local_observation, local_evidence, local_resolution,
                local_observation, local_resolution,
                grounding_called=False, grounding_objective=None,
            )
            return local_observation, local_resolution, local_evidence
        need = VisualGroundingNeed(
            self._target_objective(target),
            "Local UIA evidence could not resolve the requested target safely.", 5,
        )
        grounding = bounded_grounding_request(need.objective, need.max_candidates)
        if self._active_budget.steps + 1 > self.budgets.max_steps:
            self._active_last_visual_grounding_outcome = "incomplete"
            self._append_resolution_diagnostics(
                target, reason, local_observation, local_evidence, local_resolution,
                None, TargetEvidenceSet((), {}), local_resolution,
                local_observation, local_resolution,
                grounding_called=False, grounding_objective=grounding.objective,
            )
            return local_observation, local_resolution, local_evidence
        self._active_budget.visual_grounding_calls += 1
        try:
            grounded = self.computer.observe_directed(grounding)
        except KeyboardInterrupt:
            raise
        except Exception:
            self._active_last_visual_grounding_outcome = "failed"
            self._append_resolution_diagnostics(
                target, reason, local_observation, local_evidence, local_resolution,
                None, TargetEvidenceSet((), {}), local_resolution,
                local_observation, local_resolution,
                grounding_called=True, grounding_objective=grounding.objective,
            )
            return local_observation, local_resolution, local_evidence
        self._active_last_visual_observation = grounded
        self._record_visual_debug_capture(reason, grounded)
        self._active_budget.steps += 1
        self._active_trace.append(GenericTaskStep(
            self._active_budget.steps, GenericCapability.RESOLVE_TARGET, grounded.observation_id,
            resolution_status="visual_grounding", candidate_count=len(grounded.visual_elements),
        ))
        if grounded.error:
            self._active_last_visual_grounding_outcome = "failed"
            diagnostic_resolution, diagnostic_evidence = self._resolve(target, grounded)
            self._append_resolution_diagnostics(
                target, reason, local_observation, local_evidence, local_resolution,
                grounded, diagnostic_evidence, diagnostic_resolution,
                local_observation, local_resolution,
                grounding_called=True, grounding_objective=grounding.objective,
            )
            return grounded, local_resolution, local_evidence
        self._active_last_visual_grounding_outcome = "complete"
        grounding_binding = bind_directed_grounding(
            target, grounding, local_observation, grounded,
            expected_objective=self._target_objective(target),
        )
        resolution, evidence_set = self._resolve(
            target, grounded, grounding_binding=grounding_binding,
        )
        self._append_resolution_diagnostics(
            target, reason, local_observation, local_evidence, local_resolution,
            grounded, evidence_set, resolution, grounded, resolution,
            grounding_called=True, grounding_objective=grounding.objective,
        )
        return grounded, resolution, evidence_set

    @staticmethod
    def _readiness_visual_result(
        observation: Observation | None,
    ) -> tuple[bool, str, str | None]:
        """Classify the provider pipeline without treating a valid empty result as failure."""
        if observation is None:
            return False, "unavailable", "visual_observation_unavailable"
        status = observation.visual_grounding_status
        if status is VisualGroundingStatus.PARSE_ERROR:
            return False, "invalid", "visual_provider_invalid_response"
        if status is VisualGroundingStatus.PROVIDER_ERROR or observation.visual_provider_error is not None:
            category = (
                observation.visual_provider_error.provider_error_category
                or observation.visual_provider_error.category
                if observation.visual_provider_error is not None else "unknown"
            )
            if category == "timeout":
                return False, "error", "visual_provider_timeout"
            if category == "invalid_response":
                return False, "invalid", "visual_provider_invalid_response"
            return False, "error", "visual_provider_error"
        if observation.error:
            return False, "error", "visual_observation_error"
        if observation.truncated or observation.inspection_errors != 0:
            return False, "invalid", "visual_observation_incomplete"
        readiness = observation.visual_readiness
        if readiness is not None and not readiness.ready:
            if readiness.reason is VisualReadinessReason.CREDENTIAL_SENSITIVE:
                return False, "blocked", "credential_sensitive_context"
            if readiness.reason in {
                VisualReadinessReason.TIMEOUT, VisualReadinessReason.CAPTURE_ERROR,
                VisualReadinessReason.INSUFFICIENT_VISUAL_INFORMATION,
            }:
                return False, "unavailable", "capture_not_ready"
        if observation.screenshot is None:
            return False, "unavailable", "capture_not_ready"
        if not observation.observation_id:
            return False, "invalid", "visual_observation_id_missing"
        if observation.screenshot.snapshot_id != observation.observation_id:
            return False, "invalid", "visual_snapshot_id_mismatch"
        if (observation.foreground_hwnd is not None
                and observation.foreground_hwnd != observation.screenshot.window_handle):
            return False, "invalid", "visual_capture_window_mismatch"
        if observation.visual_directed_grounding is not True:
            return False, "unavailable", "directed_visual_grounding_unavailable"
        if (type(observation.visual_requested_max_elements) is not int
                or observation.visual_requested_max_elements <= 0):
            return False, "invalid", "visual_request_metadata_invalid"
        if (type(observation.visual_provider_call_count) is not int
                or observation.visual_provider_call_count < 1):
            return False, "unavailable", "visual_provider_not_called"
        pipeline = observation.visual_pipeline
        if (pipeline is None
                or pipeline.provider_requested_max_elements != observation.visual_requested_max_elements):
            return False, "invalid", "visual_pipeline_missing_or_mismatched"
        counts = (
            pipeline.provider_raw_element_count, pipeline.parsed_element_count,
            pipeline.validated_element_count, pipeline.deduplicated_element_count,
            pipeline.observation_visual_control_count,
            observation.visual_returned_elements,
        )
        if any(type(count) is not int or count < 0 for count in counts):
            return False, "invalid", "visual_pipeline_counts_missing_or_invalid"
        raw, parsed, validated, deduplicated, observed, returned = counts
        if (observed != len(observation.visual_elements)
                or returned != len(observation.visual_elements)):
            return False, "invalid", "visual_pipeline_candidate_count_mismatch"
        if status is VisualGroundingStatus.SUCCESS_EMPTY:
            if any((raw, parsed, validated, deduplicated, observed, returned)):
                return False, "invalid", "visual_empty_status_count_mismatch"
            return True, "success_empty", None
        if status is VisualGroundingStatus.SUCCESS_WITH_CANDIDATES:
            if not (raw > 0 and parsed > 0 and validated > 0 and deduplicated > 0
                    and observed > 0 and returned > 0):
                return False, "invalid", "visual_candidate_status_count_mismatch"
            return True, "success_candidates", None
        if status is VisualGroundingStatus.VALIDATION_EMPTY:
            if not (raw > 0 and parsed > 0 and validated == 0
                    and deduplicated == 0 and observed == 0 and returned == 0):
                return False, "invalid", "visual_validation_status_count_mismatch"
            return True, "success_candidates", None
        if status is VisualGroundingStatus.DEDUP_EMPTY:
            if not (raw > 0 and parsed > 0 and validated > 0
                    and deduplicated == 0 and observed == 0 and returned == 0):
                return False, "invalid", "visual_dedup_status_count_mismatch"
            return True, "success_candidates", None
        return (
            False, "unavailable" if status is None else "invalid",
            "visual_grounding_status_unavailable" if status is None
            else "visual_grounding_status_invalid",
        )

    @staticmethod
    def _readiness_same_trusted_app(left: Observation, right: Observation) -> bool:
        if not left.application_id or left.application_id != right.application_id:
            return False
        if (left.process_id is not None and right.process_id is not None
                and left.process_id != right.process_id):
            return False
        return True

    @staticmethod
    def _readiness_same_window(left: Observation, right: Observation) -> bool:
        def window_handle(observation: Observation) -> int | None:
            screenshot_handle = (
                observation.screenshot.window_handle
                if observation.screenshot is not None else None
            )
            if (screenshot_handle is not None and observation.foreground_hwnd is not None
                    and screenshot_handle != observation.foreground_hwnd):
                return None
            return screenshot_handle or observation.foreground_hwnd

        left_hwnd = window_handle(left)
        right_hwnd = window_handle(right)
        return left_hwnd is not None and right_hwnd is not None and left_hwnd == right_hwnd

    def _readiness_attempt_diagnostic(
        self,
        capability: GenericCapability,
        structural: Observation,
        visual: Observation | None,
        activation_observation: Observation | None,
        assessment: _ReadinessAssessment,
        *,
        fresh_structural: bool,
        fresh_visual: bool,
    ) -> GenericReadinessAttemptDiagnostics:
        structural_complete = self._readiness_observation_complete(structural)
        visual_complete, provider_result, visual_incomplete_reason = (
            self._readiness_visual_result(visual)
        )
        same_app = bool(
            activation_observation is not None
            and self._readiness_same_trusted_app(activation_observation, structural)
            and (visual is None or (
                self._readiness_same_trusted_app(activation_observation, visual)
                and self._readiness_same_trusted_app(structural, visual)
            ))
        )
        same_window = bool(
            activation_observation is not None
            and self._readiness_same_window(activation_observation, structural)
            and (visual is None or (
                self._readiness_same_window(activation_observation, visual)
                and self._readiness_same_window(structural, visual)
            ))
        )
        foreground_stable = same_app and same_window
        incomplete_reasons: list[str] = []
        if not self._readiness_observation_complete(structural):
            incomplete_reasons.extend(
                self._readiness_structural_incomplete_reasons(structural)
            )
        if visual_incomplete_reason is not None:
            incomplete_reasons.append(visual_incomplete_reason)
        if not same_app:
            incomplete_reasons.append("trusted_app_identity_mismatch")
        if not same_window:
            incomplete_reasons.append("window_identity_mismatch")
        if not fresh_structural:
            incomplete_reasons.append("stale_structural_observation")
        if visual is not None and not fresh_visual:
            incomplete_reasons.append("stale_or_unbound_visual_snapshot")
        has_uia_candidate = assessment.uia_candidate_count > 0
        has_visual_candidate = assessment.visual_candidate_count > 0
        if (structural_complete and foreground_stable and fresh_structural
                and has_uia_candidate and visual is None):
            classification = "complete_with_candidates"
        elif (structural_complete and visual_complete and foreground_stable
              and fresh_structural and fresh_visual):
            if (assessment.uia_candidate_count == 0 and not has_visual_candidate
                    and provider_result == "success_empty"):
                classification = "complete_empty"
            elif (has_uia_candidate or has_visual_candidate
                  or provider_result == "success_candidates"):
                classification = "complete_with_candidates"
            else:
                classification = "incomplete"
        else:
            classification = "incomplete"
        return GenericReadinessAttemptDiagnostics(
            capability.value, structural.observation_id or None, foreground_stable,
            assessment.uia_candidate_count, assessment.visual_candidate_count,
            visual.observation_id if visual is not None else None,
            structural_complete, visual_complete, provider_result,
            same_app, same_window, fresh_structural, fresh_visual, classification,
            tuple(dict.fromkeys(incomplete_reasons)),
        )

    @staticmethod
    def _readiness_structural_incomplete_reasons(observation: Observation) -> tuple[str, ...]:
        reasons: list[str] = []
        if observation.error:
            reasons.append("structural_observation_error")
        if not observation.observation_id:
            reasons.append("structural_observation_id_missing")
        if observation.truncated:
            reasons.append("structural_observation_truncated")
        if observation.inspection_errors != 0:
            reasons.append("structural_observation_inspection_errors")
        return tuple(reasons)

    def _query_field_readiness_assessment(
        self, source: Observation, selected_observation: Observation | None,
    ) -> _ReadinessAssessment:
        diagnostics = self._active_query_field_diagnostics
        visual = self._active_last_visual_observation
        if diagnostics is None:
            return _ReadinessAssessment(
                source.observation_id or None, 0, 0, False, False, False,
                "discovery_incomplete",
            )
        visual_count = diagnostics.visual_candidate_count
        visual_complete, provider_result, _ = self._readiness_visual_result(visual)
        structural_complete = self._readiness_observation_complete(source)
        same_app = visual is not None and self._readiness_same_trusted_app(source, visual)
        same_window = visual is not None and self._readiness_same_window(source, visual)
        complete = bool(structural_complete and visual_complete and same_app and same_window)
        empty = bool(
            complete and diagnostics.grounding_called
            and diagnostics.selection_reason == "no_visual_query_field"
            and diagnostics.uia_candidate_count == 0 and visual_count == 0
            and provider_result == "success_empty"
        )
        usable = bool(
            selected_observation is not None
            and diagnostics.semantic_safety_eligible is True
        )
        if empty:
            stop_reason = "empty_discovery"
        elif diagnostics.selection_reason.startswith("multiple_"):
            stop_reason = "ambiguous_candidates"
        elif diagnostics.uia_candidate_count or visual_count or provider_result == "success_candidates":
            stop_reason = "candidates_present"
        elif usable:
            stop_reason = "candidate_found"
        else:
            stop_reason = "discovery_incomplete"
        return _ReadinessAssessment(
            (visual.observation_id if visual is not None else source.observation_id) or None,
            diagnostics.uia_candidate_count, visual_count, empty, complete, usable,
            stop_reason, visual,
        )

    def _target_readiness_assessment(
        self, structural: Observation, observation: Observation,
        resolution: TargetResolution,
    ) -> _ReadinessAssessment:
        diagnostics = (
            self._active_resolution_diagnostics[-1]
            if self._active_resolution_diagnostics else None
        )
        if diagnostics is None:
            return _ReadinessAssessment(
                observation.observation_id or None, 0, 0, False, False,
                self._is_resolvable(resolution), "discovery_incomplete",
            )
        uia_count = diagnostics.uia.raw_candidate_count
        visual_count = diagnostics.visual.raw_candidate_count
        visual = self._active_last_visual_observation
        visual_complete, provider_result, _ = self._readiness_visual_result(visual)
        structural_complete = self._readiness_observation_complete(structural)
        same_app = visual is not None and self._readiness_same_trusted_app(structural, visual)
        same_window = visual is not None and self._readiness_same_window(structural, visual)
        complete = bool(
            structural_complete and visual_complete and same_app and same_window
            and diagnostics.visual.grounding_called is True
            and self._active_last_visual_grounding_outcome == "complete"
        )
        empty = bool(
            complete and resolution.status is TargetResolutionStatus.NO_MATCH
            and uia_count == 0 and visual_count == 0
            and provider_result == "success_empty"
        )
        if empty:
            stop_reason = "empty_discovery"
        elif resolution.status is TargetResolutionStatus.AMBIGUOUS:
            stop_reason = "ambiguous_candidates"
        elif uia_count or visual_count or provider_result == "success_candidates":
            stop_reason = "visible_candidates_no_match"
        elif self._is_resolvable(resolution):
            stop_reason = "candidate_found"
        else:
            stop_reason = "discovery_incomplete"
        return _ReadinessAssessment(
            diagnostics.visual.snapshot_id or diagnostics.uia.snapshot_id,
            uia_count, visual_count, empty, complete,
            self._is_resolvable(resolution), stop_reason, visual,
        )

    def _run_readiness_discovery(
        self,
        capability: GenericCapability,
        initial_observation: Observation,
        *,
        activation_observation: Observation | None,
        activated_app_id: str | None,
        activation_verified: bool,
        budget: _RunBudget,
        trace: list[GenericTaskStep],
        diagnostics: list[GenericReadinessRetryDiagnostics],
        discover: Callable[[Observation], tuple[
            Observation | None, _DiscoveryValue, _ReadinessAssessment,
        ]],
    ) -> tuple[Observation, _DiscoveryValue, str | None]:
        started = self.clock()
        max_attempts = min(
            2, 1 + max(0, self.budgets.readiness_retries - budget.readiness_retries),
        )
        attempts: list[GenericPerceptionAttemptDiagnostics] = []
        seen_local_ids: set[str] = set()
        seen_visual_observation_ids: set[str] = set()
        seen_visual_snapshot_ids: set[str] = set()
        anchor = activation_observation or initial_observation
        expected_app_id = activated_app_id or initial_observation.application_id
        current = initial_observation
        initial_discovery_started = self.clock()
        action_state_before = (
            budget.query_field_activations, budget.visual_target_activations,
            budget.literal_types, budget.query_submits, budget.final_target_activations,
        )
        selected_observation, value, assessment = discover(current)
        initial_discovery_elapsed_ms = max(
            0, round((self.clock() - initial_discovery_started) * 1000),
        )
        visual = assessment.visual_observation

        def visual_ids_are_fresh(observation: Observation | None) -> bool:
            if observation is None or observation.screenshot is None:
                return False
            snapshot_id = observation.screenshot.snapshot_id
            return bool(
                observation.observation_id and snapshot_id
                and snapshot_id == observation.observation_id
                and observation.observation_id not in seen_visual_observation_ids
                and snapshot_id not in seen_visual_snapshot_ids
            )

        def stable(observation: Observation | None) -> bool:
            return bool(
                expected_app_id
                and self._readiness_foreground_stable(anchor, observation, expected_app_id)
            )

        initial_stable = stable(current) and (visual is None or stable(visual))
        initial_detail = self._readiness_attempt_diagnostic(
            capability, current, visual, anchor, assessment,
            fresh_structural=bool(current.observation_id),
            fresh_visual=visual_ids_are_fresh(visual),
        )
        attempts.append(initial_detail)
        if current.observation_id:
            seen_local_ids.add(current.observation_id)
        if visual is not None and visual.screenshot is not None:
            if visual.observation_id:
                seen_visual_observation_ids.add(visual.observation_id)
            if visual.screenshot.snapshot_id:
                seen_visual_snapshot_ids.add(visual.screenshot.snapshot_id)
        applicable = False
        reason = "perception_complete"
        stop_reason = assessment.stop_reason
        success = False
        retry_performed = False
        readiness_wait_elapsed_ms = 0
        retry_discovery_elapsed_ms = 0
        action_state_after = (
            budget.query_field_activations, budget.visual_target_activations,
            budget.literal_types, budget.query_submits, budget.final_target_activations,
        )
        recoverable_reasons = {
            "visual_provider_error", "visual_provider_timeout",
            "visual_provider_invalid_response", "structural_observation_truncated",
            "capture_not_ready",
        }
        failure_class = "+".join(
            reason for reason in initial_detail.incomplete_reasons
            if reason in recoverable_reasons
        ) or None

        if assessment.stop_reason == "ambiguous_candidates":
            reason, stop_reason = "ambiguous_candidates", "ambiguous_candidates"
        elif initial_detail.discovery_classification != "incomplete":
            reason = (
                "complete_semantic_no_match"
                if initial_detail.discovery_classification == "complete_empty"
                else "perception_complete"
            )
        elif not expected_app_id or not initial_stable:
            reason, stop_reason = "foreground_identity_not_stable", "foreground_changed"
        elif not current.observation_id or not initial_detail.fresh_structural:
            reason, stop_reason = "initial_observation_not_fresh", "stale_observation"
        elif visual is None or not initial_detail.fresh_visual:
            reason, stop_reason = "visual_observation_not_fresh", "observation_incomplete"
        elif not initial_detail.same_trusted_app or not initial_detail.same_window:
            reason, stop_reason = "foreground_identity_not_stable", "foreground_changed"
        elif assessment.usable_candidate or assessment.stop_reason in {
            "ambiguous_candidates", "unsafe_candidate", "safety_rejected",
        }:
            reason, stop_reason = assessment.stop_reason, assessment.stop_reason
        elif not recoverable_reasons.intersection(initial_detail.incomplete_reasons):
            reason, stop_reason = "incomplete_reason_not_retryable", "observation_incomplete"
        elif action_state_before != action_state_after:
            reason, stop_reason = "action_occurred_since_failed_observation", "observation_incomplete"
        elif self._active_post_observation:
            reason, stop_reason = "post_action_cycle_not_started", "observation_incomplete"
        elif max_attempts < 2:
            reason, stop_reason = "perception_retry_budget_exhausted", "observation_incomplete"
        elif budget.steps + 1 > self.budgets.max_steps:
            reason, stop_reason = "step_budget_exhausted", "observation_incomplete"
        else:
            applicable = True
            reason = "recoverable_incomplete_perception"
            remaining_wait = max(
                0.0,
                self.budgets.readiness_wait_budget_seconds
                - budget.readiness_wait_elapsed_seconds,
            )
            requested_wait = min(
                0.3, self.budgets.readiness_poll_seconds, remaining_wait,
            )
            if requested_wait > 0:
                wait_started = self.clock()
                self.sleep_fn(requested_wait)
                observed_wait = max(0.0, self.clock() - wait_started)
                charged_wait = min(remaining_wait, max(requested_wait, observed_wait))
                budget.readiness_wait_elapsed_seconds += charged_wait
                readiness_wait_elapsed_ms = max(0, round(observed_wait * 1000))

            budget.readiness_retries += 1
            budget.steps += 1
            retry_started = self.clock()
            try:
                fresh = self.computer.observe_local()
            except KeyboardInterrupt:
                raise
            except Exception:
                trace.append(GenericTaskStep(
                    budget.steps, GenericCapability.OBSERVE,
                    resolution_status="perception_retry_observation_failed",
                ))
                stop_reason = "observation_incomplete"
                fresh = None
            if fresh is not None:
                trace.append(GenericTaskStep(
                    budget.steps, GenericCapability.OBSERVE, fresh.observation_id or None,
                    resolution_status="perception_retry",
                ))
                fresh_structural = bool(
                    fresh.observation_id and fresh.observation_id not in seen_local_ids
                )
                if not fresh_structural:
                    empty_assessment = _ReadinessAssessment(
                        fresh.observation_id or None, 0, 0, False, False, False,
                        "discovery_incomplete",
                    )
                    attempts.append(self._readiness_attempt_diagnostic(
                        capability, fresh, None, anchor, empty_assessment,
                        fresh_structural=False, fresh_visual=False,
                    ))
                    stop_reason = "stale_observation"
                elif not stable(fresh):
                    context_assessment = _ReadinessAssessment(
                        fresh.observation_id or None, 0, 0, False, False, False,
                        "discovery_incomplete",
                    )
                    attempts.append(self._readiness_attempt_diagnostic(
                        capability, fresh, None, anchor, context_assessment,
                        fresh_structural=True, fresh_visual=False,
                    ))
                    if not fresh.application_id or fresh.application_id != expected_app_id:
                        stop_reason = "trusted_identity_changed"
                    else:
                        stop_reason = "foreground_changed"
                else:
                    current_action_state = (
                        budget.query_field_activations, budget.visual_target_activations,
                        budget.literal_types, budget.query_submits, budget.final_target_activations,
                    )
                    if current_action_state != action_state_before:
                        stop_reason = "observation_incomplete"
                        reason = "action_occurred_since_failed_observation"
                    else:
                        seen_local_ids.add(fresh.observation_id)
                        retry_performed = True
                        selected_observation, value, assessment = discover(fresh)
                        retry_discovery_elapsed_ms = max(
                            0, round((self.clock() - retry_started) * 1000),
                        )
                        visual = assessment.visual_observation
                        fresh_visual = visual_ids_are_fresh(visual)
                        detail = self._readiness_attempt_diagnostic(
                            capability, fresh, visual, anchor, assessment,
                            fresh_structural=True, fresh_visual=fresh_visual,
                        )
                        attempts.append(detail)
                        if visual is not None and fresh_visual:
                            seen_visual_observation_ids.add(visual.observation_id)
                            seen_visual_snapshot_ids.add(visual.screenshot.snapshot_id)
                        if not fresh_visual:
                            stale_visual_id = bool(
                                visual is not None and (
                                    visual.observation_id in seen_visual_observation_ids
                                    or (visual.screenshot is not None
                                        and visual.screenshot.snapshot_id in seen_visual_snapshot_ids)
                                )
                            )
                            stop_reason = "stale_observation" if stale_visual_id else "observation_incomplete"
                        elif not detail.same_trusted_app or not detail.same_window:
                            stop_reason = "foreground_changed"
                        elif assessment.usable_candidate:
                            success = True
                            stop_reason = "candidate_found"
                        elif detail.discovery_classification == "complete_empty":
                            stop_reason = "complete_semantic_no_match"
                        else:
                            stop_reason = "observation_incomplete"
                        current = selected_observation or fresh
                if not retry_discovery_elapsed_ms:
                    retry_discovery_elapsed_ms = max(
                        0, round((self.clock() - retry_started) * 1000),
                    )
        total_elapsed_ms = max(0, round((self.clock() - started) * 1000))
        retry_result = (
            "not_attempted" if not retry_performed
            else "candidate_found" if success
            else "complete_semantic_no_match"
            if attempts[-1].discovery_classification == "complete_empty"
            else stop_reason
        )
        diagnostics.append(GenericPerceptionRetryDiagnostics(
            capability.value, applicable, reason, len(attempts), max_attempts,
            total_elapsed_ms, success, stop_reason,
            tuple(attempts[:4]),
            initial_discovery_elapsed_ms, readiness_wait_elapsed_ms,
            retry_discovery_elapsed_ms, total_elapsed_ms,
            failure_class, retry_performed,
            retry_result,
        ))
        return selected_observation or current, value, stop_reason if stop_reason in {
            "foreground_changed", "trusted_identity_changed", "window_disappeared",
            "stale_observation", "observation_incomplete",
        } else None

    @staticmethod
    def _readiness_observation_complete(observation: Observation) -> bool:
        return bool(
            not observation.error and observation.observation_id
            and observation.inspection_errors == 0 and not observation.truncated
        )

    @staticmethod
    def _readiness_foreground_stable(
        baseline: Observation, current: Observation | None, expected_app_id: str,
    ) -> bool:
        if (current is None or baseline.error or current.error
                or not baseline.observation_id or not current.observation_id
                or baseline.application_id != expected_app_id
                or current.application_id != expected_app_id):
            return False
        if (baseline.process_id is not None
                and current.process_id != baseline.process_id):
            return False
        baseline_hwnd = GenericTaskDebugAgent._observation_hwnd(baseline)
        current_hwnd = GenericTaskDebugAgent._observation_hwnd(current)
        if baseline_hwnd is None or current_hwnd is None or current_hwnd != baseline_hwnd:
            return False
        return True

    def _target_objective(self, target: TargetSpec) -> str:
        identity = self.redactor.clean(target.primary_identity)[:120]
        role = self.redactor.clean(target.desired_role or "actionable target")[:50]
        return f'Find visible actionable elements corresponding to "{identity}" with semantic role "{role}".'

    def _find_query_field(
        self, observation: Observation, budget: _RunBudget, trace: list[GenericTaskStep],
    ) -> tuple[Observation | None, Action | None]:
        fields = [item for item in observation.elements if is_query_field(item, self.policy)]
        editable_controls = [item for item in observation.elements
                             if item.control_type in {"Edit", "Document"}]
        field_ids = {item.id for item in fields}
        diagnostic_controls = (
            [*fields, *(item for item in editable_controls if item.id not in field_ids)]
        )[:MAX_DIAGNOSTIC_CANDIDATES]
        uia_candidates: list[QueryFieldCandidateDiagnostic] = []
        for control in diagnostic_controls:
            considered = is_query_field(control, self.policy)
            verdict = self.policy.validate_candidate(
                ClickAction(control.id), observation,
                semantic_role="search_field" if considered else generic_uia_semantic_role(control),
            )
            if control.visible is not True:
                consideration = "not_visible_or_visibility_unknown"
            elif control.enabled is not True:
                consideration = "not_enabled_or_enabled_state_unknown"
            elif control.is_password is not False:
                consideration = "password_or_password_state_unknown"
            elif not considered:
                consideration = "no_generic_query_semantics"
            else:
                consideration = "eligible_generic_query_field"
            uia_candidates.append(QueryFieldCandidateDiagnostic(
                control.id[:80], "UIA", observation.observation_id[:128],
                self._diagnostic_text(control.name, 80), control.control_type[:60],
                control.visible is True and control.enabled is True, considered,
                consideration, verdict.disposition == "allow", verdict.reason[:120],
            ))

        def record_selection(
            *, selected=None, action: Action | None = None, source: str | None = None,
            snapshot_id: str | None = None, selection_reason: str,
            safety_eligible: bool | None = None, safety_reason: str | None = None,
            grounding_called: bool = False, objective: str | None = None,
            visual_count: int = 0,
            visual_candidates: tuple[QueryFieldCandidateDiagnostic, ...] = (),
            visual_observation_id: str | None = None,
        ) -> None:
            self._active_query_field_diagnostics = GenericQueryFieldDiagnostics(
                len(editable_controls), tuple(uia_candidates), grounding_called,
                self._diagnostic_text(objective, 180) or None,
                visual_count, visual_candidates,
                getattr(selected, "id", None), source, snapshot_id,
                selection_reason, safety_eligible, safety_reason,
                visual_observation_id=visual_observation_id,
            )

        focused = [item for item in fields if item.focused is True]
        if len(focused) == 1:
            selected = focused[0]
            verdict = self.policy.validate_candidate(
                ClickAction(selected.id), observation, semantic_role="search_field",
            )
            record_selection(
                selected=selected, source="UIA", snapshot_id=observation.observation_id,
                selection_reason="already_focused_uia_query_field",
                safety_eligible=verdict.disposition == "allow", safety_reason=verdict.reason,
            )
            return observation, None
        if len(fields) == 1:
            selected = fields[0]
            verdict = self.policy.validate_candidate(
                ClickAction(selected.id), observation, semantic_role="search_field",
            )
            record_selection(
                selected=selected, source="UIA", snapshot_id=observation.observation_id,
                selection_reason="unique_uia_query_field",
                safety_eligible=verdict.disposition == "allow", safety_reason=verdict.reason,
            )
            return observation, ClickAction(fields[0].id)
        if len(fields) > 1:
            record_selection(selection_reason="multiple_uia_query_fields")
            return None, None
        objective = "Find one visible generic search or query text field for entering the requested literal."
        if (getattr(self.computer, "visual_provider", True) is None
                or budget.visual_grounding_calls >= self.budgets.visual_grounding_calls):
            record_selection(
                selection_reason=("no_uia_query_field_visual_unavailable"
                                  if getattr(self.computer, "visual_provider", True) is None
                                  else "visual_grounding_budget_exhausted"),
                objective=objective,
            )
            return None, None
        request = VisualGroundingNeed(
            objective,
            "No uniquely eligible search field was exposed by UIA.", 5,
        )
        if budget.steps + 1 > self.budgets.max_steps:
            record_selection(
                selection_reason="step_budget_exhausted", objective=objective,
            )
            return None, None
        budget.visual_grounding_calls += 1
        grounding = bounded_grounding_request(request.objective, request.max_candidates)
        try:
            grounded = self.computer.observe_directed(grounding)
        except KeyboardInterrupt:
            raise
        except Exception:
            record_selection(
                selection_reason="visual_grounding_failed", grounding_called=True,
                objective=objective,
            )
            return None, None
        self._active_last_visual_observation = grounded
        self._record_visual_debug_capture("query-field", grounded)
        budget.steps += 1
        trace.append(GenericTaskStep(
            budget.steps, GenericCapability.FIND_QUERY_FIELD, grounded.observation_id,
            candidate_count=len(grounded.visual_elements),
        ))
        if grounded.error:
            record_selection(
                selection_reason="visual_grounding_observation_error", grounding_called=True,
                objective=objective, visual_observation_id=grounded.observation_id,
            )
            return None, None
        visual_candidates_list: list[QueryFieldCandidateDiagnostic] = []
        visual_fields = [item for item in grounded.visual_elements if is_visual_query_field(item)]
        visual_field_ids = {item.id for item in visual_fields}
        diagnostic_visual = (
            [*visual_fields, *(item for item in grounded.visual_elements
                               if item.id not in visual_field_ids)]
        )[:MAX_DIAGNOSTIC_CANDIDATES]
        for item in diagnostic_visual:
            considered = is_visual_query_field(item)
            action = VisualClickAction(grounded.observation_id, item.id)
            verdict = self.policy.validate_candidate(action, grounded)
            normalized_role = re.sub(r"[_-]+", " ", item.role.casefold()).strip()
            if not item.clickable:
                consideration = "not_clickable"
            elif not item.label.strip():
                consideration = "missing_label"
            elif normalized_role not in {"search field", "search box", "text field", "text box", "edit"}:
                consideration = "unsupported_visual_query_role"
            else:
                consideration = "eligible_visual_query_field"
            visual_candidates_list.append(QueryFieldCandidateDiagnostic(
                item.id[:80], "VISUAL", grounded.observation_id[:128],
                self._diagnostic_text(item.label, 80), self._diagnostic_text(item.role, 60),
                item.clickable, considered, consideration,
                verdict.disposition == "allow", verdict.reason[:120],
            ))
        fields_visual = visual_fields
        if len(fields_visual) != 1:
            record_selection(
                selection_reason="no_visual_query_field" if not fields_visual
                else "multiple_visual_query_fields",
                grounding_called=True, objective=objective,
                visual_count=len(grounded.visual_elements),
                visual_candidates=tuple(visual_candidates_list),
                visual_observation_id=grounded.observation_id,
            )
            return None, None
        selected = fields_visual[0]
        selected_action = VisualClickAction(grounded.observation_id, selected.id)
        selected_verdict = self.policy.validate_candidate(selected_action, grounded)
        record_selection(
            selected=selected, source="VISUAL", snapshot_id=grounded.observation_id,
            selection_reason="unique_visual_query_field", grounding_called=True,
            objective=objective, visual_count=len(grounded.visual_elements),
            visual_candidates=tuple(visual_candidates_list),
            safety_eligible=selected_verdict.disposition == "allow",
            safety_reason=selected_verdict.reason,
            visual_observation_id=grounded.observation_id,
        )
        return grounded, selected_action

    @staticmethod
    def _observation_hwnd(observation: Observation) -> int | None:
        if observation.screenshot is not None:
            return observation.screenshot.window_handle
        return observation.foreground_hwnd

    def _query_field_verification_diagnostics(
        self, before: Observation, after: Observation, click_action: Action | None,
        focused_field: UIElement | None,
    ) -> QueryFieldVerificationDiagnostics:
        focused_controls = tuple(item for item in after.elements if item.focused is True)
        focused_control = focused_controls[0] if len(focused_controls) == 1 else None
        focused_query_fields = tuple(item for item in focused_controls
                                     if is_query_field(item, self.policy))
        same_hwnd = None
        before_hwnd, after_hwnd = self._observation_hwnd(before), self._observation_hwnd(after)
        if before_hwnd is not None and after_hwnd is not None:
            same_hwnd = before_hwnd == after_hwnd
        same_pid = (
            before.process_id == after.process_id
            if isinstance(before.process_id, int) and isinstance(after.process_id, int)
            else None
        )
        same_app = bool(
            before.application_id and after.application_id
            and before.application_id == after.application_id
        )
        verification_method = (
            "strong_local" if focused_field is not None and not after.error else "none"
        )
        failure_reason: str | None = None
        if verification_method == "none":
            if after.error:
                failure_reason = "observation_error"
            elif not after.observation_id:
                failure_reason = "missing_observation_id"
            elif not focused_controls:
                failure_reason = "no_focused_control"
            elif len(focused_query_fields) > 1:
                failure_reason = "multiple_focused_query_fields"
            elif focused_control is None:
                failure_reason = "multiple_focused_controls"
            elif focused_control.control_type not in {"Edit", "Document"}:
                failure_reason = "focused_control_not_editable"
            elif focused_control.visible is not True:
                failure_reason = "focused_control_not_visible_or_unknown"
            elif focused_control.enabled is not True:
                failure_reason = "focused_control_not_enabled_or_unknown"
            elif focused_control.is_password is not False:
                failure_reason = "password_state_not_confirmed"
            elif not is_query_field(focused_control, self.policy):
                failure_reason = "missing_generic_query_semantics"
            else:
                failure_reason = "focused_query_field_not_unique"

        clicked_visual_target_id: str | None = None
        clicked_visual_target_role: str | None = None
        distinct_from_click: bool | None = None
        if click_action is not None:
            distinct_from_click = bool(
                before.observation_id and after.observation_id
                and before.observation_id != after.observation_id
            )
        if isinstance(click_action, VisualClickAction):
            clicked_visual_target_id = click_action.target_id[:80]
            clicked = next((item for item in before.visual_elements
                            if item.id == click_action.target_id), None)
            if clicked is not None:
                clicked_visual_target_role = self._diagnostic_text(clicked.role, 60) or None

        return QueryFieldVerificationDiagnostics(
            self._foreground_matches(before, after), same_hwnd, same_pid, same_app,
            after.observation_id or None,
            distinct_from_click,
            bool(focused_controls), len(focused_controls),
            focused_control.id[:80] if focused_control else None,
            (self._diagnostic_text(
                generic_uia_semantic_role(focused_control) or focused_control.control_type, 60,
            ) or None) if focused_control else None,
            (focused_control.control_type in {"Edit", "Document"}) if focused_control else None,
            focused_control.is_password if focused_control else None,
            focused_control.enabled if focused_control else None,
            focused_control.visible if focused_control else None,
            None, None,
            clicked_visual_target_id, clicked_visual_target_role,
            False, "not_attempted", None,
            verification_method, failure_reason,
            False, False,
        )

    def _computer_visual_context_matches(self, observation: Observation) -> bool:
        checker = getattr(self.computer, "visual_context_matches", None)
        if not callable(checker):
            return False
        try:
            return checker(observation) is True
        except KeyboardInterrupt:
            raise
        except Exception:
            return False

    @staticmethod
    def _local_geometry_matches_visual_snapshot(
        visual_observation: Observation, local_observation: Observation,
    ) -> bool:
        local_meta = local_observation.screenshot
        if local_meta is None:
            return True
        visual_meta = visual_observation.screenshot
        if visual_meta is None:
            return False
        return bool(
            visual_meta.window_handle == local_meta.window_handle
            and visual_meta.window_bounds == local_meta.window_bounds
            and visual_meta.capture_bounds == local_meta.capture_bounds
            and visual_meta.pixel_width == local_meta.pixel_width
            and visual_meta.pixel_height == local_meta.pixel_height
            and visual_meta.dpi_x == local_meta.dpi_x
            and visual_meta.dpi_y == local_meta.dpi_y
            and visual_meta.scale_x == local_meta.scale_x
            and visual_meta.scale_y == local_meta.scale_y
        )

    def _verify_query_field_after_click(
        self,
        clicked_observation: Observation,
        post_click_observation: Observation,
        click_action: Action | None,
        click_result: ActionResult | None,
        request: str,
        budget: _RunBudget,
        trace: list[GenericTaskStep],
    ) -> tuple[Observation | None, str, QueryFieldVerificationDiagnostics]:
        focused_field = self._focused_query_field(post_click_observation)
        diagnostics = self._query_field_verification_diagnostics(
            clicked_observation, post_click_observation, click_action, focused_field,
        )
        same_foreground = same_trusted_foreground_identity(
            clicked_observation, post_click_observation,
        )
        fresh_local_observation = bool(
            post_click_observation.observation_id
            and (click_action is None
                 or post_click_observation.observation_id != clicked_observation.observation_id)
        )
        clicked_target = None
        if isinstance(click_action, VisualClickAction):
            clicked_target = next((
                item for item in clicked_observation.visual_elements
                if item.id == click_action.target_id
            ), None)
        credential_safe = not has_credential_sensitive_evidence(
            request, clicked_target, clicked_observation, post_click_observation,
        )
        diagnostics = replace(
            diagnostics,
            foreground_stable=same_foreground,
            credential_safety_result="clear" if credential_safe else "blocked",
            strong_visual_verification_result="not_needed" if (
                focused_field is not None and same_foreground and credential_safe
                and fresh_local_observation and not post_click_observation.error
            ) else "not_attempted",
        )

        if (focused_field is not None and same_foreground and fresh_local_observation
                and credential_safe and not post_click_observation.error):
            return post_click_observation, "strong_local", replace(
                diagnostics, query_field_verification_method="strong_local",
                query_field_verification_failure_reason=None,
                strong_visual_verification_available=False,
                strong_visual_verification_used=False,
            )

        if post_click_observation.error:
            reason = "observation_error"
        elif not credential_safe:
            reason = "credential_sensitive_context"
        elif not same_foreground:
            reason = "foreground_context_changed"
        elif not fresh_local_observation:
            reason = "fresh_post_click_observation_required"
        else:
            reason = diagnostics.query_field_verification_failure_reason or "local_query_field_unverified"

        def stop(
            result: str, *, available: bool = False, credential_blocked: bool = False,
        ) -> tuple[None, str, QueryFieldVerificationDiagnostics]:
            return None, "none", replace(
                diagnostics,
                query_field_verification_method="none",
                query_field_verification_failure_reason=result,
                strong_visual_verification_available=available,
                strong_visual_verification_result=result,
                visual_focus_verification_result=result,
                credential_safety_result=(
                    "blocked" if credential_blocked
                    else diagnostics.credential_safety_result
                ),
            )

        if not isinstance(click_action, VisualClickAction):
            return stop("visual_click_required")
        if (click_result is None or not click_result.success
                or click_result.input_issued is not True):
            return stop("visual_click_not_confirmed")
        if (not clicked_observation.visual_directed_grounding
                or click_action.snapshot_id != clicked_observation.observation_id
                or clicked_observation.screenshot is None
                or clicked_target is None or not is_visual_query_field(clicked_target)
                or not clicked_target.clickable):
            return stop("visual_click_context_ineligible")
        selected_query_fields = tuple(
            item for item in clicked_observation.visual_elements
            if is_visual_query_field(item)
        )
        if (len(selected_query_fields) != 1
                or selected_query_fields[0].id != clicked_target.id):
            return stop("original_visual_query_field_not_unique")
        if not same_foreground:
            return stop("foreground_context_changed")
        if not self._local_geometry_matches_visual_snapshot(
            clicked_observation, post_click_observation,
        ):
            return stop("window_geometry_changed")
        query_diagnostics = self._active_query_field_diagnostics
        if (query_diagnostics is None or not query_diagnostics.grounding_called
                or query_diagnostics.selected_source != "VISUAL"
                or query_diagnostics.selected_candidate_id != click_action.target_id
                or query_diagnostics.selected_snapshot_id != click_action.snapshot_id
                or query_diagnostics.semantic_safety_eligible is not True
                or self.policy.validate_candidate(click_action, clicked_observation).disposition != "allow"):
            return stop("visual_target_not_semantically_safe")
        if (budget.query_field_activations != 1
                or budget.visual_target_activations != 1
                or budget.literal_types != 0):
            return stop("query_field_action_budget_ineligible")
        if budget.query_field_visual_verification_calls != 0:
            return stop("verification_budget_exhausted")
        if getattr(self.computer, "visual_provider", True) is None:
            return stop("visual_provider_unavailable")
        if budget.visual_grounding_calls >= self.budgets.visual_grounding_calls:
            return stop("visual_grounding_budget_exhausted")
        if budget.steps + 1 > self.budgets.max_steps:
            return stop("step_budget_exhausted")
        if not credential_safe:
            return stop("credential_sensitive_context", credential_blocked=True)
        if not self._computer_visual_context_matches(clicked_observation):
            return stop("foreground_context_changed")

        objective = (
            "Verify the visible text/search/query field that was just activated and is "
            "appropriate for entering the requested literal."
        )
        diagnostics = replace(
            diagnostics,
            strong_visual_verification_available=True,
            strong_visual_verification_attempted=True,
            strong_visual_grounding_objective=objective,
            visual_focus_verification_attempted=True,
            visual_focus_verification_result="verification_requested",
            query_field_verification_failure_reason=None,
        )
        # Consume both budgets before the one remote visual request; never retry it.
        budget.query_field_visual_verification_calls += 1
        budget.visual_grounding_calls += 1
        grounding = bounded_grounding_request(objective, 5)
        verification: Observation | None
        try:
            verification = self.computer.observe_directed(grounding)
        except KeyboardInterrupt:
            raise
        except Exception:
            verification = None
        budget.steps += 1
        trace.append(GenericTaskStep(
            budget.steps, GenericCapability.FIND_QUERY_FIELD,
            verification.observation_id if verification is not None else None,
            candidate_count=(
                min(5, len(verification.visual_elements)) if verification is not None else 0
            ),
            resolution_status="strong_visual_query_field_verification",
        ))
        if verification is None:
            return stop("provider_error", available=True)

        candidate_ids = safe_candidate_ids(verification, 5)
        candidate_count = min(5, len(verification.visual_elements))
        diagnostics = replace(
            diagnostics,
            strong_visual_candidate_count=candidate_count,
            strong_visual_candidate_ids=candidate_ids,
            visual_focus_candidate_count=candidate_count,
            visual_focus_verification_result=(
                "provider_error" if (
                    verification.error or verification.visual_provider_error is not None
                ) else "verification_received"
            ),
        )
        if (verification.error or verification.visual_provider_error is not None):
            return stop("provider_error", available=True)
        if not verification.visual_elements:
            return stop("provider_empty", available=True)
        if has_credential_sensitive_evidence(
            request, clicked_target, clicked_observation,
            post_click_observation, verification,
        ):
            return stop(
                "credential_sensitive_context", available=True, credential_blocked=True,
            )
        if (not same_trusted_foreground_identity(clicked_observation, verification)
                or not same_trusted_foreground_identity(post_click_observation, verification)
                or not self._computer_visual_context_matches(clicked_observation)):
            return stop("foreground_context_changed", available=True)

        correspondence = verify_visual_field_correspondence(
            clicked_target, clicked_observation, verification,
        )
        diagnostics = replace(
            diagnostics,
            window_geometry_stable=correspondence.window_geometry_stable,
            spatial_correspondence_result=correspondence.spatial_correspondence_result,
            semantic_correspondence_result=correspondence.semantic_correspondence_result,
            visual_focus_verification_result=(
                "verified" if correspondence.verified else "verification_rejected"
            ),
        )
        if not correspondence.verified:
            reason = (
                "window_geometry_changed" if not correspondence.window_geometry_stable
                else "spatial_correspondence_failed"
                if correspondence.spatial_correspondence_result != "unique_match"
                else "semantic_correspondence_failed"
            )
            return None, "none", replace(
                diagnostics,
                query_field_verification_method="none",
                query_field_verification_failure_reason=reason,
                strong_visual_verification_result=reason,
                strong_visual_verification_used=False,
            )
        verified_candidate = next((
            item for item in verification.visual_elements
            if item.id == correspondence.selected_candidate_id
        ), None)
        if (verified_candidate is None
                or not _visual_query_field_geometry_available(
                    clicked_target, clicked_observation.screenshot,
                    clicked_observation.observation_id,
                )
                or not _visual_query_field_geometry_available(
                    verified_candidate, verification.screenshot,
                    verification.observation_id,
                )):
            return None, "none", replace(
                diagnostics,
                query_field_verification_method="none",
                query_field_verification_failure_reason="window_geometry_changed",
                strong_visual_verification_result="window_geometry_changed",
                strong_visual_verification_used=False,
            )
        self._active_visual_query_field_binding = _VerifiedVisualQueryFieldBinding(
            clicked_target.id,
            clicked_observation.observation_id,
            clicked_target,
            clicked_observation.screenshot,
            verified_candidate.id,
            verification.observation_id,
            verified_candidate,
            verification.screenshot,
            clicked_observation.application_id,
            clicked_observation.process_id or 0,
            clicked_observation.screenshot.window_handle,
            " ".join(re.sub(
                r"[_-]+", " ", unicodedata.normalize("NFKC", clicked_target.role).casefold(),
            ).split()),
            "strong_visual_unique_spatial_correspondence",
            True,
        )
        return verification, "strong_visual", replace(
            diagnostics,
            query_field_verification_method="strong_visual",
            query_field_verification_failure_reason=None,
            strong_visual_verification_used=True,
            strong_visual_verification_result="verified",
        )

    def _focused_query_field(self, observation: Observation) -> UIElement | None:
        fields = [item for item in observation.elements
                  if item.focused is True and is_query_field(item, self.policy)]
        return fields[0] if len(fields) == 1 else None

    @staticmethod
    def _is_visual_query_continuity_field(element: VisualElement) -> bool:
        return bool(
            element.is_query_field is True and element.clickable is True
            and role_is_text_entry(element.role)
        )

    @staticmethod
    def _query_value_matches(control: UIElement, literal: str) -> bool:
        if control.observed_text is None or control.observed_text_truncated:
            return False
        return control.observed_text in {literal, literal + "\r", literal + "\r\n"}

    @staticmethod
    def _foreground_matches(before: Observation, after: Observation) -> bool:
        if (before.application_id != after.application_id
                or before.process_id != after.process_id):
            return False
        if before.screenshot and after.screenshot:
            return before.screenshot.window_handle == after.screenshot.window_handle
        if (before.foreground_hwnd is not None and after.foreground_hwnd is not None):
            return before.foreground_hwnd == after.foreground_hwnd
        return True

    def _activation_postcondition(
        self, target: TargetSpec, before: Observation, after: Observation,
        budget: _RunBudget,
    ) -> TargetActivationPostcondition:
        stable = self._foreground_matches(before, after)
        context_check = getattr(
            self.computer, "activation_postcondition_context_matches", None,
        )
        if stable and callable(context_check):
            try:
                stable = bool(context_check(before, after))
            except KeyboardInterrupt:
                raise
            except Exception:
                stable = False

        diagnostic = evaluate_target_activation_postcondition(
            target, after, trusted_context_stable=stable,
            clean_text=self.redactor.clean,
        )
        if (not stable or diagnostic.result in {
                ActivationPostconditionResult.VERIFIED,
                ActivationPostconditionResult.CONTRADICTED,
        }):
            return diagnostic

        provider = getattr(self.computer, "visual_provider", None)
        verify_visual = getattr(
            self.computer, "verify_activation_postcondition_visual", None,
        )
        if provider is None or not callable(verify_visual):
            return diagnostic
        if budget.visual_grounding_calls >= self.budgets.visual_grounding_calls:
            return replace(diagnostic, failure_reason="visual_verification_budget_exhausted")

        identity = self.redactor.clean(target.primary_identity)[:80]
        role = self.redactor.clean(target.desired_role or "target")[:40]
        objective = (
            f'Determine whether "{identity}" (role "{role}") is the active, open, or '
            "selected entity, not merely visible. Return visible region and state evidence."
        )
        grounding = VisualGroundingRequest(objective, max_elements=5, verification_only=True)
        budget.visual_grounding_calls += 1
        visual_called = True
        try:
            visually_observed = verify_visual(after, grounding)
            if not isinstance(visually_observed, Observation):
                visually_observed = replace(
                    after, visual_provider_error=ProviderErrorDiagnostic(
                        "malformed_response",
                        message="Post-activation visual verification returned an invalid result.",
                    ),
                )
        except KeyboardInterrupt:
            raise
        except Exception:
            visually_observed = replace(
                after, visual_provider_error=ProviderErrorDiagnostic(
                    "unknown_api_error",
                    message="Post-activation visual verification could not be completed.",
                ),
            )
        if stable and callable(context_check):
            try:
                stable = bool(context_check(before, after))
            except KeyboardInterrupt:
                raise
            except Exception:
                stable = False
        attempts = visually_observed.visual_provider_attempts[:4]
        return evaluate_target_activation_postcondition(
            target, visually_observed, trusted_context_stable=stable,
            visual_verification_called=visual_called,
            visual_provider_attempts=attempts,
            visual_grounding_objective=objective,
            provider_incomplete=(
                visually_observed.visual_provider_error is not None
                or visually_observed.visual_grounding_status is VisualGroundingStatus.PROVIDER_ERROR
            ),
            clean_text=self.redactor.clean,
        )

    def _execute_target_action(
        self, action: Action, observation: Observation, semantic_role: str | None,
    ) -> ActionResult:
        verdict = self.policy.validate_candidate(action, observation, semantic_role=semantic_role)
        if verdict.disposition != "allow":
            diagnostic = None
            if isinstance(action, VisualClickAction):
                candidate = next((
                    item for item in observation.visual_elements if item.id == action.target_id
                ), None)
                provider = observation.visual_provider
                diagnostic = VisualActivationDiagnostic(
                    candidate_id=(action.target_id if _SAFE_DIAGNOSTIC_CANDIDATE_ID.fullmatch(
                        action.target_id,
                    ) and action.target_id.startswith("v") else None),
                    snapshot_binding_valid=(
                        observation.screenshot is not None
                        and action.snapshot_id == observation.observation_id
                        and observation.screenshot.snapshot_id == action.snapshot_id
                    ),
                    candidate_lookup_succeeded=candidate is not None,
                    provenance_valid=(candidate.source == "visual" if candidate else False),
                    provider_name=(
                        provider if provider in {"openai", "gemini"}
                        else "other" if provider else "unknown"
                    ),
                    provider_execution_agnostic=True,
                    preflight_started=True,
                    preflight_failure_reason="local_safety_policy_rejected",
                    snapshot_consumed=False,
                    failure_stage="agent_safety_policy",
                    failure_reason="local_safety_policy_rejected",
                )
                self._active_visual_activation = diagnostic
            return ActionResult(
                False, action, verdict.reason, error="policy_blocked",
                visual_activation_diagnostic=diagnostic,
            )
        if isinstance(action, VisualClickAction):
            return self.computer.execute_generic_target_activation(action, observation)
        return self.computer.execute(action, observation)

    def _preclick_context_matches(
        self, expected: Observation, fresh: Observation,
    ) -> bool:
        checker = getattr(self.computer, "activation_postcondition_context_matches", None)
        if callable(checker):
            try:
                return bool(checker(expected, fresh))
            except KeyboardInterrupt:
                raise
            except Exception:
                return False
        expected_hwnd = self._observation_hwnd(expected)
        fresh_hwnd = self._observation_hwnd(fresh)
        return bool(
            expected.application_id and expected.application_id == fresh.application_id
            and isinstance(expected.process_id, int) and expected.process_id > 0
            and expected.process_id == fresh.process_id
            and expected_hwnd is not None and expected_hwnd == fresh_hwnd
        )

    @staticmethod
    def _preclick_text(value: str) -> str:
        normalized = unicodedata.normalize("NFKC", value).casefold()
        return " ".join(normalized.split())

    def _preclick_candidate_matches(
        self, target: TargetSpec, original: CandidateResolution,
        fresh: CandidateResolution,
    ) -> bool:
        return self._preclick_candidate_rejection_reason(target, original, fresh) is None

    def _preclick_candidate_rejection_reason(
        self, target: TargetSpec, original: CandidateResolution,
        fresh: CandidateResolution,
    ) -> str | None:
        """Compare continuity with an already selected target, not initial matching."""
        if fresh.primary_identity is not IdentityEvidence.MATCH:
            return f"primary_identity_{fresh.primary_identity.value}"
        if original.primary_identity is not IdentityEvidence.MATCH:
            return "original_primary_identity_not_match"
        if self._preclick_text(fresh.primary_text) != self._preclick_text(original.primary_text):
            return "primary_text_differs_from_original_candidate"
        expected_qualifiers = len(target.qualifiers)
        if (len(original.qualifier_evidence) != expected_qualifiers
                or len(fresh.qualifier_evidence) != expected_qualifiers
                or any(evidence is not IdentityEvidence.MATCH
                       for evidence in original.qualifier_evidence)
                or any(evidence is not IdentityEvidence.MATCH
                       for evidence in fresh.qualifier_evidence)):
            return "qualifier_mismatch"
        if (not original.actionable or not original.geometry_valid
                or not original.safety_eligible or not original.snapshot_valid
                or not original.admissible):
            return "original_candidate_not_safely_selected"

        if (original.presentation_compatibility is PresentationCompatibility.INCOMPATIBLE
                or fresh.presentation_compatibility is PresentationCompatibility.INCOMPATIBLE):
            return "presentation_role_incompatible"

        if original.semantic_role is None and fresh.semantic_role is None:
            if (original.presentation_compatibility is not PresentationCompatibility.COMPATIBLE
                    or fresh.presentation_compatibility is not PresentationCompatibility.COMPATIBLE):
                return "unknown_semantic_role_requires_compatible_presentation"
        elif original.semantic_role is None or fresh.semantic_role is None:
            return "semantic_role_continuity_unproven"
        elif original.semantic_role != fresh.semantic_role:
            return "semantic_role_mismatch"
        return None

    def _preclick_target_diagnostic(
        self, target: TargetSpec,
    ) -> VisualPreclickTargetSpecDiagnostic:
        return VisualPreclickTargetSpecDiagnostic(
            primary_identity=self._diagnostic_text(target.primary_identity, 120),
            qualifiers=tuple(self._diagnostic_text(value, 80) for value in target.qualifiers[:8]),
            desired_role=self._diagnostic_text(target.desired_role, 80) or None,
            action_intent=self._diagnostic_text(target.action_intent, 40),
        )

    def _preclick_candidate_diagnostic(
        self, target: TargetSpec, original: CandidateResolution,
        row: CandidateResolution | None, *,
        raw: VisualElement | None = None,
        reached_frontier: bool = False,
        same_target_rejection_reason: str | None = None,
        considered_same_target: bool = False,
    ) -> VisualPreclickCandidateDiagnostic:
        if row is not None:
            candidate_id = row.candidate_id
            primary_text = row.primary_text
            secondary_text = row.secondary_text
            provider_role = row.provider_role
            semantic_role = row.semantic_role
            presentation_role = row.presentation_role.value
            identity_relation = row.primary_identity.value
            qualifier_evidence = tuple(value.value for value in row.qualifier_evidence)
            role_compatibility = row.role_compatibility.value
            presentation_compatibility = row.presentation_compatibility.value
            actionable = row.actionable
            geometry_valid = row.geometry_valid
            safety_eligible = row.safety_eligible
            admissible = row.admissible
            rejection_reasons = row.rejection_reasons
        else:
            candidate_id = raw.id if raw is not None else ""
            primary_text = raw.label if raw is not None else ""
            secondary_text = (raw.parent,) if raw is not None and raw.parent else ()
            provider_role = raw.role if raw is not None else None
            semantic_role = None
            presentation_role = normalize_presentation_role(provider_role).value
            identity_relation = "not_resolved"
            qualifier_evidence = ()
            role_compatibility = RoleCompatibility.UNKNOWN.value
            presentation_compatibility = "unknown"
            actionable = raw.clickable is True if raw is not None else False
            geometry_valid = False
            safety_eligible = False
            admissible = False
            rejection_reasons = ("not_in_local_candidate_resolution",)
            same_target_rejection_reason = same_target_rejection_reason or "candidate_not_locally_resolved"

        return VisualPreclickCandidateDiagnostic(
            candidate_id=(candidate_id[:80] if _SAFE_DIAGNOSTIC_CANDIDATE_ID.fullmatch(candidate_id) else ""),
            primary_text=self._diagnostic_text(primary_text, 120),
            secondary_text=tuple(self._diagnostic_text(value, 120) for value in secondary_text[:4]),
            provider_role=self._diagnostic_text(provider_role, 80) or None,
            semantic_role=self._diagnostic_text(semantic_role, 80) or None,
            presentation_role=presentation_role,
            primary_identity_relation=identity_relation,
            qualifier_evidence=qualifier_evidence,
            role_compatibility=role_compatibility,
            presentation_compatibility=presentation_compatibility,
            actionable=actionable,
            geometry_valid=geometry_valid,
            safety_eligible=safety_eligible,
            admissible=admissible,
            rejection_reasons=tuple(rejection_reasons[:8]),
            reached_frontier=reached_frontier,
            considered_same_target=considered_same_target,
            same_target_rejection_reason=same_target_rejection_reason,
        )

    @staticmethod
    def _preclick_action_rect(action: Action, observation: Observation) -> Rect | None:
        if isinstance(action, ClickAction):
            control = next((item for item in observation.elements if item.id == action.target_id), None)
            return control.rectangle if control is not None else None
        if isinstance(action, VisualClickAction):
            candidate = next((item for item in observation.visual_elements
                              if item.id == action.target_id), None)
            metadata = observation.screenshot
            if candidate is None or metadata is None or metadata.snapshot_id != action.snapshot_id:
                return None
            try:
                return visual_rect_to_screen(candidate.rectangle, metadata)
            except Exception:
                return None
        return None

    @staticmethod
    def _preclick_displacement_bucket(old: Rect, fresh: Rect) -> str:
        old_x, old_y = (old.left + old.right) / 2, (old.top + old.bottom) / 2
        new_x, new_y = (fresh.left + fresh.right) / 2, (fresh.top + fresh.bottom) / 2
        distance = math.hypot(new_x - old_x, new_y - old_y)
        if distance == 0:
            return "0_px"
        if distance <= 4:
            return "1_4_px"
        if distance <= 15:
            return "5_15_px"
        if distance <= 40:
            return "16_40_px"
        if distance <= 100:
            return "41_100_px"
        return "over_100_px"

    def _preclick_geometry_diagnostic(
        self, original_action: VisualClickAction, original_observation: Observation,
        fresh_action: Action, fresh_observation: Observation,
        *, source: str, provider_attempts: tuple[VisualProviderAttempt, ...] = (),
    ) -> VisualPreclickRevalidationDiagnostic | None:
        old_rect = self._preclick_action_rect(original_action, original_observation)
        new_rect = self._preclick_action_rect(fresh_action, fresh_observation)
        if old_rect is None or new_rect is None:
            return None
        changed = old_rect != new_rect
        return VisualPreclickRevalidationDiagnostic(
            attempted=True,
            original_candidate_id=(
                original_action.target_id[:80]
                if _SAFE_DIAGNOSTIC_CANDIDATE_ID.fullmatch(original_action.target_id) else None
            ),
            original_snapshot_id_present=bool(
                original_observation.observation_id
                and original_action.snapshot_id == original_observation.observation_id
            ),
            fresh_snapshot_obtained=bool(
                fresh_observation.observation_id
                and fresh_observation.observation_id != original_observation.observation_id
            ),
            trusted_context_stable=True,
            same_target_found=True,
            result=(VisualPreclickRevalidationStatus.MOVED if changed
                    else VisualPreclickRevalidationStatus.STABLE),
            geometry_changed=changed,
            displacement_bucket=self._preclick_displacement_bucket(old_rect, new_rect),
            size_changed=(old_rect.right - old_rect.left != new_rect.right - new_rect.left
                          or old_rect.bottom - old_rect.top != new_rect.bottom - new_rect.top),
            provider_attempts=provider_attempts[:4],
            rebound_to_fresh_snapshot=True,
            source=source,
        )

    def _revalidate_visual_target_before_click(
        self, target: TargetSpec, original_row: CandidateResolution,
        original_action: VisualClickAction, original_observation: Observation,
        budget: _RunBudget, trace: list[GenericTaskStep],
    ) -> tuple[Action | None, Observation | None, CandidateResolution | None,
               VisualPreclickRevalidationDiagnostic]:
        """Bind a visual activation to fresh, unique UIA or directed visual evidence."""
        diagnostic = VisualPreclickRevalidationDiagnostic(
            attempted=True,
            original_candidate_id=(
                original_action.target_id[:80]
                if _SAFE_DIAGNOSTIC_CANDIDATE_ID.fullmatch(original_action.target_id) else None
            ),
            original_snapshot_id_present=bool(
                original_observation.observation_id
                and original_action.snapshot_id == original_observation.observation_id
            ),
            fresh_snapshot_obtained=False,
            trusted_context_stable=None,
            same_target_found=False,
            result=VisualPreclickRevalidationStatus.INCOMPLETE,
            failure_reason="local_observation_unavailable",
            original_target=self._preclick_target_diagnostic(target),
            original_candidate=self._preclick_candidate_diagnostic(
                target, original_row, original_row, reached_frontier=True,
                considered_same_target=True,
            ),
        )

        def stop(
            result: VisualPreclickRevalidationStatus, reason: str, *,
            fresh: Observation | None = None, context_stable: bool | None = None,
            attempts: tuple[VisualProviderAttempt, ...] = (),
            same_target: bool = False,
        ):
            failed = replace(
                diagnostic,
                fresh_snapshot_obtained=bool(
                    fresh is not None and fresh.observation_id
                    and fresh.observation_id != original_observation.observation_id
                ),
                trusted_context_stable=context_stable,
                same_target_found=same_target,
                result=result,
                provider_attempts=attempts[:4],
                failure_reason=reason,
            )
            self._active_visual_preclick_revalidation = failed
            return None, None, None, failed

        if budget.steps + 3 > self.budgets.max_steps:
            return stop(VisualPreclickRevalidationStatus.INCOMPLETE, "step_budget_exhausted")
        observer = getattr(self.computer, "observe_preclick_local", None)
        if not callable(observer):
            return stop(VisualPreclickRevalidationStatus.INCOMPLETE, "fresh_local_observation_unavailable")
        try:
            fresh_local = observer(original_observation)
        except KeyboardInterrupt:
            raise
        except Exception:
            return stop(VisualPreclickRevalidationStatus.INCOMPLETE, "fresh_local_observation_failed")
        budget.steps += 1
        trace.append(GenericTaskStep(
            budget.steps, GenericCapability.OBSERVE, fresh_local.observation_id or None,
        ))
        if (not fresh_local.observation_id or fresh_local.error
                or fresh_local.observation_id == original_observation.observation_id):
            return stop(
                VisualPreclickRevalidationStatus.INCOMPLETE, "fresh_local_observation_invalid",
                fresh=fresh_local,
            )
        context_stable = self._preclick_context_matches(original_observation, fresh_local)
        if not context_stable:
            return stop(
                VisualPreclickRevalidationStatus.CONTEXT_CHANGED, "trusted_context_changed",
                fresh=fresh_local, context_stable=False,
            )

        local_resolution, local_evidence = self._resolve(target, fresh_local)
        local_matches = [
            row for row in local_resolution.candidates
            if row.source == "UIA" and self._preclick_candidate_matches(target, original_row, row)
        ]
        if len(local_matches) == 1 and not local_matches[0].safety_eligible:
            return stop(
                VisualPreclickRevalidationStatus.INCOMPLETE,
                "matching_uia_target_failed_safety",
                fresh=fresh_local, context_stable=True, same_target=True,
            )
        if (len(local_matches) == 1 and local_resolution.status is TargetResolutionStatus.UNIQUE
                and local_resolution.selected_candidate_id == local_matches[0].candidate_id):
            fresh_row = local_matches[0]
            fresh_action = local_evidence.actions.get(fresh_row.candidate_id)
            if (isinstance(fresh_action, ClickAction) and fresh_row.admissible
                    and fresh_row.actionable and fresh_row.geometry_valid
                    and fresh_row.snapshot_valid):
                measured = self._preclick_geometry_diagnostic(
                    original_action, original_observation, fresh_action, fresh_local,
                    source="UIA",
                )
                if measured is not None and measured.fresh_snapshot_obtained:
                    self._active_visual_preclick_revalidation = measured
                    return fresh_action, fresh_local, fresh_row, measured

        if budget.steps + 3 > self.budgets.max_steps:
            return stop(
                VisualPreclickRevalidationStatus.INCOMPLETE, "step_budget_exhausted",
                fresh=fresh_local, context_stable=True,
            )
        if (budget.visual_grounding_calls >= self.budgets.visual_grounding_calls
                or getattr(self.computer, "visual_provider", True) is None):
            return stop(
                VisualPreclickRevalidationStatus.INCOMPLETE,
                "directed_visual_revalidation_unavailable",
                fresh=fresh_local, context_stable=True,
            )
        identity = self._diagnostic_text(original_row.primary_text, 100)
        role = self._diagnostic_text(
            original_row.semantic_role or target.desired_role or "actionable target", 60,
        )
        objective = (
            f'Find the same visible actionable target "{identity}" with semantic role "{role}".'
        )
        grounding = bounded_grounding_request(objective, max_elements=5)
        directed_observer = getattr(self.computer, "observe_preclick_directed", None)
        if not callable(directed_observer):
            return stop(
                VisualPreclickRevalidationStatus.INCOMPLETE,
                "context_bound_visual_observation_unavailable",
                fresh=fresh_local, context_stable=True,
            )
        budget.visual_grounding_calls += 1
        try:
            fresh_visual = directed_observer(grounding, fresh_local)
        except KeyboardInterrupt:
            raise
        except Exception:
            budget.steps += 1
            trace.append(GenericTaskStep(budget.steps, GenericCapability.OBSERVE))
            return stop(
                VisualPreclickRevalidationStatus.INCOMPLETE, "directed_visual_observation_failed",
                fresh=fresh_local, context_stable=True,
            )
        budget.steps += 1
        trace.append(GenericTaskStep(
            budget.steps, GenericCapability.OBSERVE, fresh_visual.observation_id or None,
        ))
        attempts = fresh_visual.visual_provider_attempts[:4]
        if fresh_visual.error == "trusted_context_changed":
            return stop(
                VisualPreclickRevalidationStatus.CONTEXT_CHANGED, "trusted_context_changed",
                fresh=fresh_visual, context_stable=False, attempts=attempts,
            )
        if not self._preclick_context_matches(original_observation, fresh_visual):
            return stop(
                VisualPreclickRevalidationStatus.CONTEXT_CHANGED, "trusted_context_changed",
                fresh=fresh_visual, context_stable=False, attempts=attempts,
            )
        if (fresh_visual.error or not fresh_visual.observation_id
                or fresh_visual.visual_provider_error is not None
                or fresh_visual.visual_grounding_status not in {
                    VisualGroundingStatus.SUCCESS_WITH_CANDIDATES,
                    VisualGroundingStatus.SUCCESS_EMPTY,
                }):
            return stop(
                VisualPreclickRevalidationStatus.INCOMPLETE, "directed_visual_result_incomplete",
                fresh=fresh_visual, context_stable=True, attempts=attempts,
            )
        binding = bind_directed_grounding(
            target, grounding, fresh_local, fresh_visual,
            expected_objective=grounding.objective,
        )
        if binding is None:
            return stop(
                VisualPreclickRevalidationStatus.INCOMPLETE, "directed_visual_binding_invalid",
                fresh=fresh_visual, context_stable=True, attempts=attempts,
            )
        fresh_resolution, fresh_evidence = self._resolve(
            target, fresh_visual, grounding_binding=binding,
        )
        directed_candidate_ids = {
            item.candidate_id for item in fresh_evidence.candidates
            if item.source == "VISUAL" and item.directed_grounding is not None
        }
        rows_by_id = {
            row.candidate_id: row for row in fresh_resolution.candidates
            if row.source == "VISUAL"
        }
        frontier_ids = set(fresh_resolution.frontier_candidate_ids)
        fresh_candidate_diagnostics: list[VisualPreclickCandidateDiagnostic] = []
        for raw_candidate in fresh_visual.visual_elements:
            row = rows_by_id.get(raw_candidate.id)
            if row is None:
                rejection_reason = "candidate_not_locally_resolved"
                considered_same_target = False
            elif raw_candidate.id not in directed_candidate_ids:
                rejection_reason = "directed_grounding_binding_missing"
                considered_same_target = False
            else:
                rejection_reason = self._preclick_candidate_rejection_reason(
                    target, original_row, row,
                )
                considered_same_target = rejection_reason is None
            fresh_candidate_diagnostics.append(self._preclick_candidate_diagnostic(
                target, original_row, row, raw=raw_candidate,
                reached_frontier=raw_candidate.id in frontier_ids,
                same_target_rejection_reason=rejection_reason,
                considered_same_target=considered_same_target,
            ))
        fresh_candidate_rows = [
            rows_by_id[candidate.id] for candidate in fresh_visual.visual_elements
            if candidate.id in rows_by_id and candidate.id in directed_candidate_ids
        ]
        diagnostic = replace(
            diagnostic,
            fresh_candidate_count=len(fresh_visual.visual_elements),
            admissible_fresh_candidate_count=sum(row.admissible for row in fresh_candidate_rows),
            frontier_fresh_candidate_count=sum(
                row.candidate_id in frontier_ids for row in fresh_candidate_rows
            ),
            fresh_candidates=tuple(fresh_candidate_diagnostics),
        )
        visual_matches = [
            row for row in fresh_candidate_rows
            if self._preclick_candidate_matches(target, original_row, row)
        ]
        if not visual_matches:
            has_identity_match = any(
                candidate.primary_identity_relation == IdentityEvidence.MATCH.value
                for candidate in fresh_candidate_diagnostics
            )
            has_unresolved_candidate = any(
                candidate.primary_identity_relation == "not_resolved"
                for candidate in fresh_candidate_diagnostics
            )
            rejected = has_identity_match or has_unresolved_candidate
            return stop(
                (VisualPreclickRevalidationStatus.REJECTED if rejected
                 else VisualPreclickRevalidationStatus.DISAPPEARED),
                ("fresh_candidate_rejected_by_local_matching" if rejected
                 else "provider_returned_no_plausible_target"),
                fresh=fresh_visual, context_stable=True, attempts=attempts,
            )
        if len(visual_matches) != 1:
            return stop(
                VisualPreclickRevalidationStatus.AMBIGUOUS, "multiple_matching_visual_targets",
                fresh=fresh_visual, context_stable=True, attempts=attempts, same_target=True,
            )
        fresh_row = visual_matches[0]
        fresh_action = fresh_evidence.actions.get(fresh_row.candidate_id)
        if (not isinstance(fresh_action, VisualClickAction) or not fresh_row.admissible
                or not fresh_row.actionable or not fresh_row.geometry_valid
                or not fresh_row.safety_eligible or not fresh_row.snapshot_valid):
            return stop(
                VisualPreclickRevalidationStatus.REJECTED,
                "same_target_failed_action_safety_or_geometry_validation",
                fresh=fresh_visual, context_stable=True, attempts=attempts, same_target=True,
            )
        if (fresh_resolution.status is not TargetResolutionStatus.UNIQUE
                or fresh_resolution.selected_candidate_id != fresh_row.candidate_id
                or fresh_row.candidate_id not in frontier_ids):
            return stop(
                VisualPreclickRevalidationStatus.AMBIGUOUS,
                "target_not_uniquely_resolved_in_frontier",
                fresh=fresh_visual, context_stable=True, attempts=attempts, same_target=True,
            )
        measured = self._preclick_geometry_diagnostic(
            original_action, original_observation, fresh_action, fresh_visual,
            source="visual", provider_attempts=attempts,
        )
        if measured is None or not measured.fresh_snapshot_obtained:
            return stop(
                VisualPreclickRevalidationStatus.INCOMPLETE, "fresh_target_geometry_unavailable",
                fresh=fresh_visual, context_stable=True, attempts=attempts, same_target=True,
            )
        measured = replace(
            measured,
            original_target=diagnostic.original_target,
            original_candidate=diagnostic.original_candidate,
            fresh_candidate_count=diagnostic.fresh_candidate_count,
            admissible_fresh_candidate_count=diagnostic.admissible_fresh_candidate_count,
            frontier_fresh_candidate_count=diagnostic.frontier_fresh_candidate_count,
            fresh_candidates=diagnostic.fresh_candidates,
        )
        self._active_visual_preclick_revalidation = measured
        return fresh_action, fresh_visual, fresh_row, measured

    def _execute_query_submit(
        self, action: QuerySubmitAction, observation: Observation, *,
        verified_search_observation: Observation | None = None,
    ) -> ActionResult:
        if verified_search_observation is not None:
            method = getattr(self.computer, "execute_query_submit_phase3", None)
            if not callable(method):
                return ActionResult(
                    False, action, "Verified visual query submission is unavailable.",
                    error="unsupported_action",
                )
            return method(action, observation, verified_search_observation)
        method = getattr(self.computer, "execute_generic_query_submit", None)
        if not callable(method):
            return ActionResult(False, action, "Verified query submission is unavailable.", error="unsupported_action")
        return method(action, observation)

    @staticmethod
    def _same_query_field_identity(before: UIElement, after: UIElement) -> bool:
        """Match a fresh UIA field by stable semantics, never by snapshot-local ID."""
        if before.control_type != after.control_type:
            return False
        if (before.parent_name != after.parent_name
                or before.parent_control_type != after.parent_control_type):
            return False
        if before.automation_id or after.automation_id:
            return bool(before.automation_id and before.automation_id == after.automation_id)
        return bool(before.name and before.name == after.name)

    @staticmethod
    def _uia_field_matches_visual_baseline(
        field: UIElement, visual_field: VisualElement,
        visual_observation: Observation,
    ) -> bool:
        metadata = visual_observation.screenshot
        rectangle = field.rectangle
        if (metadata is None or metadata.snapshot_id != visual_observation.observation_id
                or not isinstance(rectangle, Rect)):
            return False
        try:
            visual_rectangle = visual_rect_to_screen(visual_field.rectangle, metadata)
        except (ValueError, TypeError):
            return False
        width = max(0, min(rectangle.right, visual_rectangle.right)
                    - max(rectangle.left, visual_rectangle.left))
        height = max(0, min(rectangle.bottom, visual_rectangle.bottom)
                     - max(rectangle.top, visual_rectangle.top))
        intersection = width * height
        smaller = min(
            max(0, rectangle.right - rectangle.left) * max(0, rectangle.bottom - rectangle.top),
            max(0, visual_rectangle.right - visual_rectangle.left)
            * max(0, visual_rectangle.bottom - visual_rectangle.top),
        )
        return smaller > 0 and intersection / smaller >= 0.5

    @staticmethod
    def _trusted_context_facts(
        first: Observation, second: Observation,
    ) -> tuple[bool, bool, bool]:
        same_app = bool(
            first.application_id and first.application_id == second.application_id
            and isinstance(first.process_id, int) and first.process_id > 0
            and first.process_id == second.process_id
        )
        first_hwnd = GenericTaskDebugAgent._observation_hwnd(first)
        second_hwnd = GenericTaskDebugAgent._observation_hwnd(second)
        same_window = bool(
            isinstance(first_hwnd, int) and first_hwnd > 0
            and first_hwnd == second_hwnd
        )
        return same_app, same_window, bool(
            same_app and same_window and first.observation_id and second.observation_id
        )

    def _verify_query_field_continuity_after_type(
        self,
        selected_observation: Observation,
        typed_from: Observation,
        after_type: Observation,
        click_action: Action | None,
        click_result: ActionResult | None,
        request: str,
        literal: str,
        budget: _RunBudget,
        trace: list[GenericTaskStep],
    ) -> tuple[QuerySubmitContinuityDiagnostics, Observation | None]:
        """Prove the already selected query field still owns the literal before Enter."""
        query_diagnostics = self._active_query_field_diagnostics
        verification_diagnostics = (
            query_diagnostics.verification if query_diagnostics is not None else None
        )
        original_binding = self._active_visual_query_field_binding
        original_binding_present = original_binding is not None
        original_candidate_id = (
            original_binding.selected_candidate_id if original_binding is not None
            else query_diagnostics.selected_candidate_id if query_diagnostics is not None
            else None
        )
        original_snapshot_id = (
            original_binding.original_snapshot_id if original_binding is not None
            else query_diagnostics.selected_snapshot_id if query_diagnostics is not None
            else None
        )
        original_binding_unique = bool(
            original_binding and original_binding.unique
            and isinstance(original_binding.selected_candidate, VisualElement)
            and isinstance(original_binding.verified_candidate, VisualElement)
        )
        original_binding_source = (
            original_binding.verification_provenance if original_binding is not None
            else query_diagnostics.selected_source if query_diagnostics is not None else None
        )
        original_geometry_available = bool(
            original_binding is not None
            and _visual_query_field_geometry_available(
                original_binding.selected_candidate, original_binding.original_metadata,
                original_binding.original_snapshot_id,
            )
            and _visual_query_field_geometry_available(
                original_binding.verified_candidate, original_binding.verified_metadata,
                original_binding.verified_snapshot_id,
            )
        )
        source = (
            query_diagnostics.selected_source.casefold()
            if query_diagnostics is not None and query_diagnostics.selected_source else "unknown"
        )
        structural_complete = not after_type.truncated and after_type.inspection_errors == 0
        incompleteness_reasons = tuple(
            reason for condition, reason in (
                (after_type.truncated, "truncated"),
                (after_type.inspection_errors > 0, "inspection_errors"),
            ) if condition
        )
        visual_fallback_eligible = False
        literal_relation = "not_evaluated"
        field_continuity_relation = "not_evaluated"
        active_focused_relation = "not_evaluated"
        credential_safe: bool | None = None
        query_field_candidate_count = 0
        visual_candidate_diagnostics: tuple[QuerySubmitVisualCandidateDiagnostic, ...] = ()
        value_read_fallback_eligible = False
        value_read_fallback_attempted = False
        crop_valid: bool | None = None
        value_read_provider_attempts: tuple[VisualProviderAttempt, ...] = ()
        extracted_field_value: str | None = None
        final_submit_release = False
        fresh_candidate_count = 0
        continuity_comparison_started = False
        failure_stage = "preconditions"

        def make_diagnostics(*args: Any, **kwargs: Any) -> QuerySubmitContinuityDiagnostics:
            kwargs.setdefault("structural_observation_complete", structural_complete)
            kwargs.setdefault("incompleteness_reasons", incompleteness_reasons)
            kwargs.setdefault("visual_fallback_eligible", visual_fallback_eligible)
            kwargs.setdefault("literal_relation", literal_relation)
            kwargs.setdefault("field_continuity_relation", field_continuity_relation)
            kwargs.setdefault("active_focused_relation", active_focused_relation)
            kwargs.setdefault("credential_safe", credential_safe)
            kwargs.setdefault("final_continuity_result", args[0] if args else None)
            kwargs.setdefault("query_field_candidate_count", query_field_candidate_count)
            kwargs.setdefault("visual_candidates", visual_candidate_diagnostics)
            kwargs.setdefault("value_read_fallback_eligible", value_read_fallback_eligible)
            kwargs.setdefault("value_read_fallback_attempted", value_read_fallback_attempted)
            kwargs.setdefault("crop_valid", crop_valid)
            kwargs.setdefault("value_read_provider_attempts", value_read_provider_attempts)
            kwargs.setdefault("extracted_field_value", extracted_field_value)
            kwargs.setdefault("final_submit_release", final_submit_release)
            kwargs.setdefault("original_binding_present", original_binding_present)
            kwargs.setdefault("original_candidate_id", original_candidate_id)
            kwargs.setdefault("original_snapshot_id", original_snapshot_id)
            kwargs.setdefault("original_binding_unique", original_binding_unique)
            kwargs.setdefault("original_binding_source", original_binding_source)
            kwargs.setdefault("original_geometry_available", original_geometry_available)
            kwargs.setdefault("fresh_candidate_count", fresh_candidate_count)
            kwargs.setdefault("continuity_comparison_started", continuity_comparison_started)
            kwargs.setdefault("failure_stage", failure_stage)
            return QuerySubmitContinuityDiagnostics(*args, **kwargs)

        def annotate_query_candidates(
            *, identity_relation: str, acceptance: str,
            candidate_credential_safe: bool | None = None,
        ) -> None:
            nonlocal visual_candidate_diagnostics
            visual_candidate_diagnostics = tuple(
                replace(
                    item,
                    field_identity_relation=(
                        identity_relation if item.is_query_field else "not_applicable"
                    ),
                    candidate_acceptance=(
                        acceptance if item.is_query_field else "not_a_query_field"
                    ),
                    credential_safe=(
                        (candidate_credential_safe if candidate_credential_safe is not None
                         else item.credential_safe) if item.is_query_field
                        else item.credential_safe
                    ),
                )
                for item in visual_candidate_diagnostics
            )
        same_app, same_window, same_context = self._trusted_context_facts(
            typed_from, after_type,
        )
        selected_app, selected_window, selected_context = self._trusted_context_facts(
            selected_observation, typed_from,
        )
        if (after_type.error or not after_type.observation_id
                or after_type.observation_id == typed_from.observation_id):
            return make_diagnostics(
                "blocked", "fresh_post_type_observation_required", source,
                after_type.observation_id or None, same_app, same_window,
                None, False, None,
            ), None
        if (not selected_context or not same_context
                or not same_trusted_foreground_identity(selected_observation, typed_from)
                or not same_trusted_foreground_identity(typed_from, after_type)):
            failure_stage = "trusted_context"
            return make_diagnostics(
                "blocked", "trusted_foreground_changed", source,
                after_type.observation_id,
                bool(selected_app and same_app), bool(selected_window and same_window),
                None, False, None,
            ), None
        clicked_visual_target = None
        if isinstance(click_action, VisualClickAction):
            clicked_visual_target = (
                original_binding.selected_candidate if original_binding is not None else None
            )
        if has_credential_sensitive_evidence(
            request, clicked_visual_target,
            selected_observation, typed_from, after_type,
        ):
            credential_safe = False
            failure_stage = "credential_safety"
            return make_diagnostics(
                "blocked", "credential_sensitive_context", source,
                after_type.observation_id, same_app, same_window, None, False, False,
            ), None
        credential_safe = True

        # An incomplete post-type UIA snapshot remains a hard stop for the
        # UIA-only route. A field selected and strongly verified visually may
        # continue to the single fresh screenshot continuity check below.
        if not structural_complete and source != "visual":
            failure_stage = "structural_observation"
            return make_diagnostics(
                "blocked", "post_type_observation_incomplete", source,
                after_type.observation_id, same_app, same_window, None, False, None,
            ), None

        if (query_diagnostics is None
                or query_diagnostics.semantic_safety_eligible is not True
                or verification_diagnostics is None
                or verification_diagnostics.query_field_verification_method == "none"):
            failure_stage = "original_binding"
            return make_diagnostics(
                "blocked", "original_query_field_not_verified", source,
                after_type.observation_id, same_app, same_window, None, False, None,
            ), None
        if budget.query_submits >= self.budgets.query_submits:
            failure_stage = "submit_budget"
            return make_diagnostics(
                "blocked", "query_submit_budget_exhausted", source,
                after_type.observation_id, same_app, same_window, None, False, None,
            ), None

        if source == "uia":
            selected_field = next((
                item for item in selected_observation.elements
                if item.id == query_diagnostics.selected_candidate_id
            ), None)
            if (query_diagnostics.selected_snapshot_id != selected_observation.observation_id
                    or (click_action is not None and (
                        not isinstance(click_action, ClickAction)
                        or click_action.target_id != query_diagnostics.selected_candidate_id
                        or click_result is None or not click_result.success
                        or click_result.input_issued is not True
                    ))
                    or (click_action is None
                        and (selected_field is None or selected_field.focused is not True))):
                return make_diagnostics(
                    "blocked", "original_uia_query_field_binding_unavailable", source,
                    after_type.observation_id, same_app, same_window, False, False, None,
                ), None
            baseline_fields = [
                item for item in typed_from.elements if is_query_field(item, self.policy)
            ]
            current_fields = [
                item for item in after_type.elements if is_query_field(item, self.policy)
            ]
            baseline = self._focused_query_field(typed_from)
            current = self._focused_query_field(after_type)
            if selected_field is None or not is_query_field(selected_field, self.policy):
                reason = "selected_uia_query_field_unavailable"
                field_match = False
            elif len(baseline_fields) != 1 or baseline is None:
                reason = "original_uia_query_field_not_unique"
                field_match = False
            elif not self._same_query_field_identity(selected_field, baseline):
                reason = "original_uia_query_field_changed_before_typing"
                field_match = False
            elif len(current_fields) > 1:
                reason = "multiple_query_fields_after_typing"
                field_match = False
            elif len(current_fields) != 1 or current is None:
                reason = "focused_query_field_not_reestablished"
                field_match = False
            elif not self._same_query_field_identity(baseline, current):
                reason = "query_field_identity_changed_after_typing"
                field_match = False
            else:
                field_match = True
                if not self._query_value_matches(current, literal):
                    reason = "typed_literal_not_confirmed_in_query_field"
                else:
                    reason = "verified"
                    literal_relation = "exact"
                    field_continuity_relation = "same"
                    active_focused_relation = "active"
                    if structural_complete:
                        final_submit_release = True
                        return make_diagnostics(
                            "verified", reason, "uia", after_type.observation_id,
                            same_app, same_window, True, True, True,
                        ), after_type
            if reason == "typed_literal_not_confirmed_in_query_field":
                literal_relation = "mismatch"
                field_continuity_relation = "same" if field_match else "mismatch"
                active_focused_relation = (
                    "active" if current is not None and current.focused is True else "inactive"
                )
            if any(
                item.focused is True and item.control_type in {"Edit", "Document"}
                and item.is_password is not False for item in after_type.elements
            ):
                reason = "credential_sensitive_context"
            return make_diagnostics(
                "blocked", reason, "uia", after_type.observation_id,
                same_app, same_window, field_match,
                bool(current is not None and self._query_value_matches(current, literal)),
                bool(current is not None and current.focused is True),
            ), None

        if source != "visual":
            failure_stage = "original_binding"
            return make_diagnostics(
                "blocked", "original_query_field_source_unavailable", source,
                after_type.observation_id, same_app, same_window, None, False, None,
            ), None
        if (query_diagnostics.selected_candidate_id is None
                or query_diagnostics.selected_snapshot_id != selected_observation.observation_id
                or not isinstance(click_action, VisualClickAction)
                or click_result is None or not click_result.success
                or click_result.input_issued is not True
                or click_action.snapshot_id != selected_observation.observation_id
                or verification_diagnostics.query_field_verification_method != "strong_visual"
                or not verification_diagnostics.strong_visual_verification_used
                or verification_diagnostics.strong_visual_verification_result != "verified"):
            failure_stage = "original_binding"
            return make_diagnostics(
                "blocked", "original_visual_query_field_binding_unavailable", source,
                after_type.observation_id, same_app, same_window, False, False, None,
            ), None
        if original_binding is None:
            failure_stage = "original_binding"
            return make_diagnostics(
                "blocked", "original_visual_query_field_binding_missing", source,
                after_type.observation_id, same_app, same_window, False, False, None,
            ), None
        if not original_binding.unique:
            failure_stage = "original_binding"
            return make_diagnostics(
                "blocked", "original_visual_query_field_not_unique", source,
                after_type.observation_id, same_app, same_window, False, False, None,
            ), None
        if (not isinstance(original_binding.selected_candidate, VisualElement)
                or not isinstance(original_binding.verified_candidate, VisualElement)
                or not isinstance(original_binding.original_metadata, ScreenshotMetadata)
                or not isinstance(original_binding.verified_metadata, ScreenshotMetadata)):
            failure_stage = "original_binding"
            return make_diagnostics(
                "blocked", "original_visual_query_field_binding_inconsistent", source,
                after_type.observation_id, same_app, same_window, False, False, None,
            ), None
        normalized_original_role = " ".join(re.sub(
            r"[_-]+", " ", unicodedata.normalize(
                "NFKC", original_binding.selected_candidate.role,
            ).casefold(),
        ).split())
        binding_consistent = bool(
            original_binding.selected_candidate_id == query_diagnostics.selected_candidate_id
            == click_action.target_id
            and original_binding.selected_candidate.id == original_binding.selected_candidate_id
            and original_binding.original_snapshot_id == query_diagnostics.selected_snapshot_id
            == selected_observation.observation_id == click_action.snapshot_id
            and original_binding.original_metadata == selected_observation.screenshot
            and original_binding.verified_snapshot_id == typed_from.observation_id
            and original_binding.verified_metadata == typed_from.screenshot
            and original_binding.verified_candidate.id == original_binding.verified_candidate_id
            and original_binding.application_id == selected_observation.application_id
            == typed_from.application_id == after_type.application_id
            and original_binding.process_id == selected_observation.process_id
            == typed_from.process_id == after_type.process_id
            and original_binding.window_handle == self._observation_hwnd(selected_observation)
            == self._observation_hwnd(typed_from) == self._observation_hwnd(after_type)
            and original_binding.semantic_role == normalized_original_role
            and original_binding.verification_provenance
            == "strong_visual_unique_spatial_correspondence"
            and self._local_geometry_matches_visual_snapshot(selected_observation, typed_from)
        )
        if not binding_consistent:
            failure_stage = "original_binding"
            return make_diagnostics(
                "blocked", "original_visual_query_field_binding_inconsistent", source,
                after_type.observation_id, same_app, same_window, False, False, None,
            ), None
        if not original_geometry_available:
            failure_stage = "original_binding"
            return make_diagnostics(
                "blocked", "original_visual_query_field_geometry_unavailable", source,
                after_type.observation_id, same_app, same_window, False, False, None,
            ), None

        # Use the stored selected+verified provenance as the original side.
        # The post-click verification list is not re-filtered to rediscover it.
        baseline_field = original_binding.verified_candidate
        reason = "visual_submit_verification_required"
        field_match = True

        # A newly surfaced editable control could be a different input or a
        # credential field. Do not let a screenshot override that local fact.
        after_editable = [item for item in after_type.elements
                          if item.control_type in {"Edit", "Document"}
                          and item.visible is not False]
        if after_editable:
            local_fields = [item for item in after_type.elements
                            if is_query_field(item, self.policy)]
            current = self._focused_query_field(after_type)
            if (len(after_editable) != 1 or len(local_fields) != 1 or current is None):
                return make_diagnostics(
                    "blocked", "ambiguous_or_unreconciled_uia_editable_field", source,
                    after_type.observation_id, same_app, same_window, False, False, False,
                ), None
            if not self._uia_field_matches_visual_baseline(
                current, baseline_field, typed_from,
            ):
                return make_diagnostics(
                    "blocked", "uia_field_does_not_match_selected_visual_field", source,
                    after_type.observation_id, same_app, same_window, False,
                    self._query_value_matches(current, literal), True,
                ), None
            if not self._query_value_matches(current, literal):
                literal_relation = "mismatch"
                field_continuity_relation = "same"
                active_focused_relation = "active"
                return make_diagnostics(
                    "blocked", "typed_literal_not_confirmed_in_query_field", "visual_uia",
                    after_type.observation_id, same_app, same_window, True, False, True,
                ), None
            if structural_complete:
                literal_relation = "exact"
                field_continuity_relation = "same"
                active_focused_relation = "active"
                final_submit_release = True
                return make_diagnostics(
                    "verified", "verified", "visual_uia", after_type.observation_id,
                    same_app, same_window, True, True, True,
                ), after_type
        failure_stage = "fresh_observation"
        if getattr(self.computer, "visual_provider", None) is None:
            return make_diagnostics(
                "blocked", "visual_provider_unavailable", source,
                after_type.observation_id, same_app, same_window, True, False, None,
            ), None
        if budget.visual_grounding_calls >= self.budgets.visual_grounding_calls:
            return make_diagnostics(
                "blocked", "visual_grounding_budget_exhausted", source,
                after_type.observation_id, same_app, same_window, True, False, None,
            ), None
        if budget.steps + 2 > self.budgets.max_steps:
            return make_diagnostics(
                "blocked", "step_budget_exhausted", source,
                after_type.observation_id, same_app, same_window, True, False, None,
            ), None

        objective = f"Verify the active search field displays exactly: {literal}"
        if len(objective) > 240 or self.redactor.clean(objective) != objective:
            return make_diagnostics(
                "blocked", "visual_verification_objective_unavailable", source,
                after_type.observation_id, same_app, same_window, True, False, None,
            ), None
        visual_fallback_eligible = True
        grounding = VisualGroundingRequest(
            objective, max_elements=5, verification_only=True,
            query_field_continuity=True,
        )
        budget.visual_grounding_calls += 1
        try:
            verified = self.computer.observe_directed(grounding)
        except KeyboardInterrupt:
            raise
        except Exception:
            verified = None
        budget.steps += 1
        trace.append(GenericTaskStep(
            budget.steps, GenericCapability.OBSERVE,
            verified.observation_id if verified is not None else None,
            candidate_count=(len(verified.visual_elements) if verified is not None else 0),
            resolution_status="query_field_submit_continuity",
        ))
        if verified is None:
            literal_relation = "unavailable"
            field_continuity_relation = "unproven"
            active_focused_relation = "unavailable"
            return make_diagnostics(
                "blocked", "visual_provider_incomplete", source,
                after_type.observation_id, same_app, same_window, True, False, None,
                visual_verification_attempted=True,
            ), None
        visual_attempts = verified.visual_provider_attempts[:4]
        visual_count = len(verified.visual_elements)
        verified_app, verified_window, verified_context = self._trusted_context_facts(
            typed_from, verified,
        )
        if (verified.error or not verified.observation_id or not verified_context
                or not same_trusted_foreground_identity(after_type, verified)
                or verified.visual_provider_error is not None
                or verified.visual_grounding_status is not VisualGroundingStatus.SUCCESS_WITH_CANDIDATES
                or not verified.visual_directed_grounding
                or verified.screenshot is None
                or verified.screenshot.snapshot_id != verified.observation_id
                or not self._computer_visual_context_matches(verified)):
            literal_relation = "unavailable"
            field_continuity_relation = "unproven"
            active_focused_relation = "unavailable"
            return make_diagnostics(
                "blocked", "visual_observation_incomplete_or_context_changed", source,
                verified.observation_id or None, verified_app, verified_window,
                True, False, None, True, visual_count, visual_attempts,
            ), None
        fresh_query_fields = [
            item for item in verified.visual_elements
            if self._is_visual_query_continuity_field(item)
        ]
        query_field_candidate_count = len(fresh_query_fields)
        fresh_candidate_count = query_field_candidate_count
        visual_candidate_diagnostics = tuple(
            QuerySubmitVisualCandidateDiagnostic(
                candidate_id=self._diagnostic_text(item.id, 40),
                field_label=(
                    self._diagnostic_text(item.field_label, 100)
                    if item.is_query_field is True and item.field_label is not None else None
                ),
                field_value=(
                    self._diagnostic_text(item.field_value, 100)
                    if item.is_query_field is True and item.field_value is not None else None
                ),
                role=self._diagnostic_text(item.role, 60),
                activity=(item.activity if item.activity in {"active", "inactive"}
                          else "inactive" if item.activity == "not_active" else "unknown"),
                is_query_field=item.is_query_field is True,
                field_identity_relation="not_evaluated",
                literal_relation=(
                    "not_applicable" if item.is_query_field is not True
                    else "missing" if item.field_value is None or not item.field_value.strip()
                    else "exact" if self._preclick_text(item.field_value)
                    == self._preclick_text(literal)
                    else "mismatch"
                ),
                credential_safe=(item.credential_risk is False),
                candidate_acceptance=(
                    "pending" if self._is_visual_query_continuity_field(item)
                    else "not_a_query_field"
                ),
            )
            for item in verified.visual_elements[:MAX_DIAGNOSTIC_CANDIDATES]
        )
        if has_credential_sensitive_evidence(
            request, clicked_visual_target,
            selected_observation, typed_from, after_type, verified,
        ):
            credential_safe = False
            literal_relation = "unavailable"
            field_continuity_relation = "unproven"
            active_focused_relation = "unavailable"
            annotate_query_candidates(
                identity_relation="not_evaluated", acceptance="blocked_credential_risk",
                candidate_credential_safe=False,
            )
            return make_diagnostics(
                "blocked", "credential_sensitive_context", source,
                verified.observation_id, verified_app, verified_window,
                True, False, False, True, visual_count, visual_attempts,
            ), None

        fresh_fields = fresh_query_fields
        if len(fresh_fields) != 1:
            reason = "multiple_visual_query_fields" if len(fresh_fields) > 1 else "visual_query_field_missing"
            field_continuity_relation = "ambiguous" if len(fresh_fields) > 1 else "missing"
            literal_relation = "unavailable"
            active_focused_relation = "unavailable"
            annotate_query_candidates(
                identity_relation="ambiguous" if len(fresh_fields) > 1 else "missing",
                acceptance="blocked_ambiguous" if len(fresh_fields) > 1
                else "blocked_no_query_field",
            )
            return make_diagnostics(
                "blocked", reason, source, verified.observation_id,
                verified_app, verified_window, False, False, None,
                True, visual_count, visual_attempts,
            ), None
        fresh_field = fresh_fields[0]
        continuity_comparison_started = True
        failure_stage = "continuity_comparison"
        correspondence = verify_visual_field_correspondence(
            baseline_field, typed_from, verified,
        )
        candidate_identity_matches = bool(
            correspondence.verified
            and correspondence.compatible_candidate_count == 1
            and correspondence.selected_candidate_id == fresh_field.id
        )
        field_continuity_relation = (
            "same" if candidate_identity_matches else "mismatch"
        )
        value_known = isinstance(fresh_field.field_value, str) and bool(
            fresh_field.field_value.strip()
        )
        literal_confirmed = bool(
            value_known
            and self._preclick_text(fresh_field.field_value or "")
            == self._preclick_text(literal)
        )
        literal_relation = (
            "missing" if not value_known else "exact" if literal_confirmed else "mismatch"
        )
        active_focused_relation = (
            "active" if fresh_field.activity == "active" else "inactive"
        )
        annotate_query_candidates(
            identity_relation="same" if candidate_identity_matches else "mismatch",
            acceptance="pending",
            candidate_credential_safe=(fresh_field.credential_risk is False),
        )
        if not correspondence.verified or correspondence.compatible_candidate_count != 1:
            annotate_query_candidates(
                identity_relation="mismatch", acceptance="blocked_field_identity",
            )
            return make_diagnostics(
                "blocked", "visual_query_field_identity_changed", source,
                verified.observation_id, verified_app, verified_window,
                False, literal_confirmed,
                fresh_field.activity == "active", True, visual_count, visual_attempts,
            ), None
        if not candidate_identity_matches:
            annotate_query_candidates(
                identity_relation="mismatch", acceptance="blocked_field_identity",
            )
            return make_diagnostics(
                "blocked", "visual_query_field_identity_changed", source,
                verified.observation_id, verified_app, verified_window,
                False, literal_confirmed, fresh_field.activity == "active",
                True, visual_count, visual_attempts,
            ), None
        if fresh_field.credential_risk is not False:
            credential_safe = False if fresh_field.credential_risk is True else None
            annotate_query_candidates(
                identity_relation="same",
                acceptance="blocked_credential_risk" if fresh_field.credential_risk is True
                else "blocked_credential_risk_unknown",
                candidate_credential_safe=credential_safe,
            )
            return make_diagnostics(
                "blocked", "credential_sensitive_context" if fresh_field.credential_risk is True
                else "visual_credential_risk_unknown", source,
                verified.observation_id, verified_app, verified_window,
                True, literal_confirmed, fresh_field.activity == "active",
                True, visual_count, visual_attempts,
            ), None
        if fresh_field.activity != "active":
            annotate_query_candidates(
                identity_relation="same", acceptance="blocked_field_inactive",
                candidate_credential_safe=True,
            )
            return make_diagnostics(
                "blocked", "visual_query_field_not_active", source,
                verified.observation_id, verified_app, verified_window,
                True, literal_confirmed, False, True, visual_count, visual_attempts,
            ), None

        if not value_known:
            meta = verified.screenshot
            rect = fresh_field.rectangle
            crop_geometry_available = bool(
                fresh_field.source == "visual" and meta is not None
                and meta.snapshot_id == verified.observation_id
                and all(type(value) is int for value in (
                    rect.left, rect.top, rect.right, rect.bottom,
                ))
                and 0 <= rect.left < rect.right <= meta.pixel_width
                and 0 <= rect.top < rect.bottom <= meta.pixel_height
                and 24 <= rect.right - rect.left <= min(1200, round(meta.pixel_width * .95))
                and 8 <= rect.bottom - rect.top <= min(160, round(meta.pixel_height * .2))
            )
            value_reader = getattr(self.computer, "read_visual_field_value", None)
            value_read_fallback_eligible = bool(
                same_app and same_window and same_context and candidate_identity_matches
                and fresh_field.activity == "active" and credential_safe is True
                and fresh_field.credential_risk is False and crop_geometry_available
                and callable(value_reader)
                and getattr(self.computer, "visual_provider", None) is not None
                and budget.visual_grounding_calls < self.budgets.visual_grounding_calls
                and budget.steps + 2 <= self.budgets.max_steps
            )
            if value_read_fallback_eligible:
                failure_stage = "value_read_fallback"
                value_read_fallback_attempted = True
                budget.visual_grounding_calls += 1
                try:
                    value_read = value_reader(verified, fresh_field.id)
                except KeyboardInterrupt:
                    raise
                except Exception:
                    value_read = None
                budget.steps += 1
                trace.append(GenericTaskStep(
                    budget.steps, GenericCapability.OBSERVE, verified.observation_id,
                    resolution_status="query_field_value_read",
                ))
                if isinstance(value_read, VisualFieldValueRead):
                    value_read_provider_attempts = value_read.provider_attempts[:4]
                    crop_valid = value_read.crop_valid
                    extracted_field_value = (
                        self._diagnostic_text(value_read.field_value, 100)
                        if isinstance(value_read.field_value, str) else None
                    )
                    if value_read.credential_safe is not True:
                        credential_safe = value_read.credential_safe
                    if (value_read.context_stable is not True
                            or value_read.credential_safe is not True
                            or value_read.crop_valid is not True
                            or value_read.error is not None):
                        literal_relation = "unavailable"
                        annotate_query_candidates(
                            identity_relation="same",
                            acceptance=(
                                "blocked_value_read_context_changed"
                                if value_read.context_stable is not True else
                                "blocked_value_read_credential_risk"
                                if value_read.credential_safe is not True else
                                "blocked_value_read_crop_invalid"
                                if value_read.crop_valid is not True else
                                "blocked_value_read_provider_failure"
                            ),
                            candidate_credential_safe=credential_safe,
                        )
                        reason = (
                            "visual_value_read_context_changed"
                            if value_read.context_stable is not True else
                            "visual_value_read_credential_risk"
                            if value_read.credential_safe is not True else
                            "visual_value_read_crop_invalid"
                            if value_read.crop_valid is not True else
                            "visual_value_read_provider_failure"
                        )
                        return make_diagnostics(
                            "blocked", reason, source, verified.observation_id,
                            verified_app, verified_window, True, False, True,
                            True, visual_count, visual_attempts,
                        ), None
                    field_value = value_read.field_value
                    value_known = isinstance(field_value, str) and bool(field_value.strip())
                    literal_confirmed = bool(
                        value_known
                        and self._preclick_text(field_value or "")
                        == self._preclick_text(literal)
                    )
                    literal_relation = (
                        "missing" if not value_known else "exact" if literal_confirmed
                        else "mismatch"
                    )
                else:
                    literal_relation = "unavailable"
                    annotate_query_candidates(
                        identity_relation="same",
                        acceptance="blocked_value_read_provider_failure",
                    )
                    return make_diagnostics(
                        "blocked", "visual_value_read_provider_failure", source,
                        verified.observation_id, verified_app, verified_window,
                        True, False, True, True, visual_count, visual_attempts,
                    ), None

        failure_stage = "literal_confirmation"
        if not literal_confirmed:
            annotate_query_candidates(
                identity_relation="same",
                acceptance="blocked_literal_value_missing" if not value_known
                else "blocked_literal_value_mismatch",
                candidate_credential_safe=True,
            )
            return make_diagnostics(
                "blocked", "typed_literal_not_confirmed_visually", source,
                verified.observation_id, verified_app, verified_window,
                True, False, fresh_field.activity == "active",
                True, visual_count, visual_attempts,
            ), None
        if budget.query_submits >= self.budgets.query_submits:
            annotate_query_candidates(
                identity_relation="same", acceptance="blocked_submit_budget",
                candidate_credential_safe=True,
            )
            return make_diagnostics(
                "blocked", "query_submit_budget_exhausted", source,
                verified.observation_id, verified_app, verified_window,
                True, True, True, True, visual_count, visual_attempts,
            ), None
        annotate_query_candidates(
            identity_relation="same", acceptance="accepted",
            candidate_credential_safe=True,
        )
        failure_stage = "submit_release"
        final_submit_release = True
        return make_diagnostics(
            "verified", "verified", "visual", verified.observation_id,
            verified_app, verified_window, True, True, True,
            True, visual_count, visual_attempts,
        ), verified

    def _wait_for_application(
        self, candidate: ApplicationCandidate, initial_observation: Observation,
        *, open_app_action_started_at: float | None = None,
        open_app_action_elapsed_ms: int | None = None,
    ) -> ApplicationActivationWait:
        probe_factory = getattr(self.computer, "activation_target_probe", None)
        target_probe = None
        if callable(probe_factory):
            try:
                target_probe = probe_factory(candidate)
            except Exception:
                target_probe = None
        activate_window = getattr(self.computer, "activate_trusted_application_window", None)
        target_activator = (
            (lambda: activate_window(candidate))
            if target_probe is not None and callable(activate_window) else None
        )
        return wait_for_trusted_application_activation(
            self.computer.observe_local, candidate,
            initial_observation=initial_observation, launch_succeeded=True,
            timeout_seconds=self.budgets.app_activation_timeout_seconds,
            poll_interval_seconds=self.budgets.transition_poll_seconds,
            clock=self.clock, sleep_fn=self.sleep_fn,
            target_probe=target_probe,
            target_activator=target_activator,
            post_deadline_probe_seconds=3.0 if target_probe is not None else 0.0,
            open_app_action_started_at=open_app_action_started_at,
            open_app_action_elapsed_ms=open_app_action_elapsed_ms,
        )

    def _select_activate_and_stop(
        self, request: str, target: TargetSpec, resolution: TargetResolution,
        evidence_set: TargetEvidenceSet, observation: Observation,
        budget: _RunBudget, trace: list[GenericTaskStep], finish,
    ) -> GenericTaskDebugResult:
        self._active_budget, self._active_trace = budget, trace
        sufficiency = assess_target_evidence_sufficiency(
            target, resolution, evidence_set, observation,
            expected_grounding_objective=self._target_objective(target),
            activation_required=target.action_intent.casefold() in {"activate", "select"},
        )
        deterministic = sufficiency.status is TargetEvidenceSufficiency.SUFFICIENT
        jev_required = (
            not deterministic and resolution.status in {
                TargetResolutionStatus.UNIQUE, TargetResolutionStatus.CHOICE,
            }
        )
        diagnostic = GenericTargetDecisionDiagnostics(
            resolver_status=resolution.status.value,
            selection_mode="deterministic_local" if deterministic else "jev",
            evidence_sufficiency=sufficiency.status.value,
            evidence_sufficiency_reasons=tuple(reason.value for reason in sufficiency.reasons),
            admissible_candidate_count_total=len(resolution.admissible_candidate_ids),
            frontier_candidate_count=len(resolution.frontier_candidate_ids),
            jev_required=jev_required,
            jev_called=False,
            jev_provider_called=False if deterministic else None,
            jev_result_kind=None,
            jev_selected_candidate_id=None,
            jev_confidence=None,
            jev_release_result="not_released",
            stop_reason=None,
        )
        self._active_target_decision_diagnostics = diagnostic

        def update_diagnostic(**changes: object) -> None:
            nonlocal diagnostic
            diagnostic = replace(diagnostic, **changes)
            self._active_target_decision_diagnostics = diagnostic

        def diagnostic_candidate_id(value: object) -> str | None:
            return value if isinstance(value, str) and _SAFE_DIAGNOSTIC_CANDIDATE_ID.fullmatch(value) else None

        if budget.steps + 2 > self.budgets.max_steps:
            update_diagnostic(stop_reason="budget_exhausted")
            return finish(False, "budget_exhausted", "The overall step budget is exhausted.")
        if deterministic:
            candidate_id = resolution.selected_candidate_id
            if candidate_id is None:
                update_diagnostic(stop_reason="target_not_resolvable")
                return finish(False, "target_not_resolvable", "Sufficient evidence had no locally selected candidate.")
            final_confidence = None
        elif not jev_required:
            update_diagnostic(stop_reason="target_not_resolvable")
            return finish(False, "target_not_resolvable", "Only unique or bounded-choice targets can reach Jev.")
        else:
            update_diagnostic(jev_called=True)
            try:
                decision = self.decision_maker.decide_target_activation(target, resolution)
            except KeyboardInterrupt:
                raise
            except Exception:
                decision = TargetChoiceResult(
                    "error", None, None, "Target decision failed.", "decision_error",
                    "internal_error", None,
                )

            allowed_diagnostic_reasons = {
                "resolution_ineligible", "candidate_limit", "candidate_invalid",
                "evidence_indistinguishable", "model_stop",
                "model_confidence_below_threshold", "candidate_selected",
                "provider_error", "invalid_response", "internal_error",
            }
            decision_reason = decision.diagnostic_reason
            if decision_reason not in allowed_diagnostic_reasons:
                if decision.status == "error":
                    decision_reason = (
                        "invalid_response" if decision.error == "invalid_response"
                        else "internal_error" if decision.error == "decision_error"
                        else "provider_error"
                    )
                elif decision.status == "stop":
                    decision_reason = "model_stop"
                else:
                    decision_reason = "candidate_selected"
            raw_confidence = decision.confidence
            diagnostic_confidence = (
                float(raw_confidence)
                if isinstance(raw_confidence, (int, float))
                and not isinstance(raw_confidence, bool) and math.isfinite(raw_confidence)
                and 0.0 <= raw_confidence <= 1.0 else None
            )
            final_confidence = decision.confidence
            update_diagnostic(
                jev_provider_called=decision.provider_called,
                jev_result_kind=decision.status if decision.status in {"ready", "stop", "error"} else "invalid_result",
                jev_selected_candidate_id=diagnostic_candidate_id(
                    decision.candidate_id if decision.status == "ready"
                    else decision.proposed_candidate_id
                    if decision_reason == "model_confidence_below_threshold" else None
                ),
                jev_confidence=diagnostic_confidence,
                stop_reason=decision_reason,
            )
            if decision.status != "ready" or decision.candidate_id is None:
                update_diagnostic(stop_reason=decision_reason)
                return finish(False, "target_decision_stopped", decision.message)
            if (decision.confidence is None
                    or decision.confidence < max(.80, self.decision_maker.min_confidence)):
                update_diagnostic(stop_reason="low_confidence")
                return finish(False, "low_confidence", "Target decision confidence is below 0.80.")
            candidate_id = decision.candidate_id
        if candidate_id not in resolution.frontier_candidate_ids:
            update_diagnostic(stop_reason="unoffered_target")
            return finish(False, "unoffered_target", "The decision selected a candidate outside the local frontier.")
        row = next((item for item in resolution.candidates if item.candidate_id == candidate_id), None)
        action = evidence_set.actions.get(candidate_id)
        if (row is None or not row.admissible or not row.snapshot_valid
                or not row.safety_eligible or not row.actionable or action is None):
            update_diagnostic(stop_reason="target_revalidation_failed")
            return finish(False, "target_revalidation_failed", "The selected target failed local revalidation.")
        if isinstance(action, VisualClickAction) and action.snapshot_id != observation.observation_id:
            update_diagnostic(stop_reason="stale_target")
            return finish(False, "stale_target", "The visual target belongs to a different snapshot.")
        selected_action_is_visual = isinstance(action, VisualClickAction)
        if budget.steps + (3 if selected_action_is_visual else 2) > self.budgets.max_steps:
            update_diagnostic(stop_reason="budget_exhausted")
            return finish(False, "budget_exhausted", "The overall step budget is exhausted.")
        if budget.final_target_activations >= self.budgets.final_target_activations:
            update_diagnostic(stop_reason="budget_exhausted")
            return finish(False, "budget_exhausted", "The final target activation budget is exhausted.")
        if isinstance(action, VisualClickAction):
            if budget.visual_target_activations >= self.budgets.visual_target_activations:
                update_diagnostic(stop_reason="budget_exhausted")
                return finish(False, "budget_exhausted", "The visual activation budget is exhausted.")
            fresh_action, fresh_observation, fresh_row, revalidation = (
                self._revalidate_visual_target_before_click(
                    target, row, action, observation, budget, trace,
                )
            )
            self._active_visual_preclick_revalidation = revalidation
            if (fresh_action is None or fresh_observation is None or fresh_row is None
                    or revalidation.result not in {
                        VisualPreclickRevalidationStatus.STABLE,
                        VisualPreclickRevalidationStatus.MOVED,
                    }):
                update_diagnostic(stop_reason="visual_preclick_revalidation_failed")
                return finish(
                    False, "visual_preclick_revalidation_failed",
                    "Fresh evidence did not safely revalidate the visual target.",
                )
            action, observation, row = fresh_action, fresh_observation, fresh_row
        update_diagnostic(chosen_candidate_id=diagnostic_candidate_id(candidate_id))
        if selected_action_is_visual:
            budget.visual_target_activations += 1
        budget.final_target_activations += 1
        budget.steps += 1
        update_diagnostic(jev_release_result="released_to_local_safety_gate", stop_reason=None)
        result = self._execute_target_action(action, observation, row.semantic_role)
        if selected_action_is_visual:
            revalidation = self._active_visual_preclick_revalidation
            if revalidation is not None:
                self._active_visual_preclick_revalidation = replace(
                    revalidation,
                    click_released=result.input_issued is True,
                )
        if isinstance(action, VisualClickAction):
            self._active_visual_activation = result.visual_activation_diagnostic
        trace.append(GenericTaskStep(
            budget.steps, GenericCapability.ACTIVATE_TARGET, observation.observation_id,
            len(resolution.frontier_candidate_ids), resolution.status.value,
            action.kind, result.success,
        ))
        if not result.success:
            update_diagnostic(stop_reason="target_activation_failed")
            return finish(False, "target_activation_failed", "The target activation did not complete safely.")
        update_diagnostic(jev_release_result="action_succeeded")
        if selected_action_is_visual and self._active_visual_activation is not None:
            self._active_visual_activation = replace(
                self._active_visual_activation,
                post_action_observation_attempted=True,
            )
        try:
            post = self.computer.observe_local()
        except KeyboardInterrupt:
            raise
        except Exception:
            self._active_target_activation_postcondition = TargetActivationPostcondition(
                True, self.redactor.clean(target.primary_identity)[:120],
                self.redactor.clean(target.desired_role)[:80] if target.desired_role else None,
                False, False, (), ActivationEvidence.UNKNOWN, ActivationEvidence.UNKNOWN,
                False, ActivationEvidenceSource.UIA,
                ActivationPostconditionResult.INCOMPLETE, "fresh_observation_failed",
            )
            if selected_action_is_visual and self._active_visual_activation is not None:
                self._active_visual_activation = replace(
                    self._active_visual_activation,
                    failure_stage="post_action_observation",
                    failure_reason="local_observation_failed",
                )
            update_diagnostic(stop_reason="post_action_observation_failed")
            return finish(False, "post_action_observation_failed", "Fresh post-activation observation failed.")
        budget.steps += 1
        trace.append(GenericTaskStep(budget.steps, GenericCapability.OBSERVE, post.observation_id))
        if post.error or not post.observation_id:
            self._active_target_activation_postcondition = TargetActivationPostcondition(
                True, self.redactor.clean(target.primary_identity)[:120],
                self.redactor.clean(target.desired_role)[:80] if target.desired_role else None,
                False, False, (), ActivationEvidence.UNKNOWN, ActivationEvidence.UNKNOWN,
                False, ActivationEvidenceSource.UIA,
                ActivationPostconditionResult.INCOMPLETE, "fresh_observation_invalid",
            )
            if selected_action_is_visual and self._active_visual_activation is not None:
                self._active_visual_activation = replace(
                    self._active_visual_activation,
                    failure_stage="post_action_observation",
                    failure_reason="observation_invalid",
                )
            update_diagnostic(stop_reason="post_action_observation_failed")
            return finish(False, "post_action_observation_failed", "Fresh post-activation observation was invalid.")
        self._active_post_observation = True
        postcondition = self._activation_postcondition(target, observation, post, budget)
        self._active_target_activation_postcondition = postcondition
        self._active_post_foreground_stable = postcondition.trusted_context_stable
        if selected_action_is_visual and self._active_visual_activation is not None:
            self._active_visual_activation = replace(
                self._active_visual_activation,
                post_action_observation_obtained=True,
                failure_stage=(None if postcondition.result is ActivationPostconditionResult.VERIFIED
                               else "target_activation_postcondition"),
                failure_reason=(None if postcondition.result is ActivationPostconditionResult.VERIFIED
                                else postcondition.failure_reason),
            )
        if postcondition.result is not ActivationPostconditionResult.VERIFIED:
            reason = (
                "target_activation_postcondition_failed"
                if postcondition.result is ActivationPostconditionResult.CONTRADICTED
                else "target_activation_unverified"
            )
            update_diagnostic(stop_reason=reason)
            return finish(
                False, reason,
                "Fresh evidence did not verify that the requested target is active.",
            )
        # Only a fresh, trusted structural or directed visual active-identity
        # match can complete this generic target activation checkpoint.
        self._active_chosen_id, self._active_confidence = candidate_id, final_confidence
        return finish(True, "target_activated", "The requested target was activated; the controller stopped.")
