"""Experimental Jev-controlled grounding loop with visual execution blocked."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
import re
import time
from typing import Protocol
import unicodedata

from agent.application_activation import wait_for_trusted_application_activation
from agent.loop import AgentLimits, _RepeatGuard
from agent.visual_field_verification import (
    has_credential_sensitive_evidence, verify_visual_field_correspondence,
)
from computer.actions import (
    Action, FinishAction, OpenAppAction, QuerySubmitAction, TypeAction, VisualClickAction,
)
from computer.applications import ApplicationCandidate
from computer.models import (
    CaptureDiagnostics, Observation, Rect, VisualGroundingStatus, VisualPipelineDiagnostic,
    VisualReadinessResult, VisualRequestFingerprint,
)
from computer.results import ActionResult, LiteralInputDiagnostic
from computer.visual import (
    ScreenshotCapture, VisualGroundingRequest, bounded_grounding_request,
    visual_fallback_policy,
)
from decision.context import Redactor, phase2_type_literal, requested_target_spec
from decision.models import (
    ChoiceProbability, DecisionAttempt, HybridDecisionResult, OfferedApplication,
    DecisionEffect, ObservationPolicyDiagnostic, OptionFilterSummary, TaskProgress,
    VisualGroundingNeed,
)
from decision.target_resolution import (
    CandidateEvidence, TargetResolution, TargetResolutionStatus, TargetSpec,
    canonical_semantic_role, normalize_presentation_role, resolve_target,
)
from safety.interfaces import ActionPolicy, Confirmation
from safety.policy import BasicActionPolicy
from safety.policy import (
    Phase1VisualClickPolicy, Phase3ResultSelectionPolicy,
)


def _phase3_candidate_evidence(
    observation: Observation, redactor: Redactor,
) -> tuple[CandidateEvidence, ...]:
    """Convert separated provider fields plus local validation into resolver input."""

    metadata = observation.screenshot
    policy = Phase3ResultSelectionPolicy()
    result: list[CandidateEvidence] = []
    retained_ids: set[str] = set()
    for item in observation.visual_elements[:20]:
        rect = item.rectangle
        geometry_valid = bool(
            metadata and metadata.snapshot_id == observation.observation_id
            and isinstance(rect, Rect) and rect.left >= 0 and rect.top >= 0
            and rect.right > rect.left and rect.bottom > rect.top
            and rect.right <= metadata.pixel_width and rect.bottom <= metadata.pixel_height
        )
        action = VisualClickAction(observation.observation_id, item.id)
        safety_eligible = policy.validate(action, observation).disposition == "allow"
        result.append(CandidateEvidence(
            candidate_id=item.id,
            primary_text=" ".join(redactor.clean(item.label).split())[:240],
            secondary_text=(
                (" ".join(redactor.clean(item.parent).split())[:200],) if item.parent else ()
            ),
            semantic_role=canonical_semantic_role(redactor.clean(item.role)[:80]),
            provider_role=redactor.clean(item.role)[:80] or None,
            actionable=item.clickable is True,
            geometry_valid=geometry_valid,
            safety_eligible=safety_eligible,
            source=item.source[:40],
            snapshot_id=observation.observation_id,
            target_semantic_evidence=canonical_semantic_role(redactor.clean(item.role)[:80]),
            presentation_role=normalize_presentation_role(redactor.clean(item.role)[:80]),
        ))
        retained_ids.add(item.id)

    for index, item in enumerate(observation.visual_provider_candidates[:20], 1):
        if item.validated_visual_id and item.validated_visual_id in retained_ids:
            continue
        candidate_id = item.provider_candidate_id or f"provider_{index}"
        if candidate_id in retained_ids:
            continue
        result.append(CandidateEvidence(
            candidate_id=candidate_id,
            primary_text=" ".join(redactor.clean(item.label).split())[:240],
            secondary_text=(
                (" ".join(redactor.clean(item.parent).split())[:200],) if item.parent else ()
            ),
            semantic_role=canonical_semantic_role(redactor.clean(item.role)[:80]),
            provider_role=redactor.clean(item.role)[:80] or None,
            actionable=item.clickable is True,
            geometry_valid=item.geometry_valid,
            safety_eligible=False,
            source="visual",
            snapshot_id=observation.observation_id,
            target_semantic_evidence=canonical_semantic_role(redactor.clean(item.role)[:80]),
            presentation_role=normalize_presentation_role(redactor.clean(item.role)[:80]),
        ))
        retained_ids.add(candidate_id)
    return tuple(result)


class HybridComputer(Protocol):
    def observe_local(self) -> Observation: ...
    def observe_directed(self, grounding: VisualGroundingRequest) -> Observation: ...
    def observe_result_baseline(self) -> Observation: ...
    def observe_result_directed(
        self, grounding: VisualGroundingRequest, baseline: Observation,
    ) -> Observation: ...
    def execute(self, action: Action, observation: Observation | None = None) -> ActionResult: ...
    def set_observation_request(self, request: str) -> None: ...
    def execute_visual_click_phase1(
        self, action: VisualClickAction, observation: Observation,
        request: str, jev_confidence: float | None,
    ) -> ActionResult: ...
    def execute_type_phase2(
        self, action: TypeAction, observation: Observation, *, visual_verified: bool,
    ) -> ActionResult: ...
    def visual_context_matches(self, observation: Observation) -> bool: ...
    def execute_visual_click_phase3(
        self, action: VisualClickAction, observation: Observation,
    ) -> ActionResult: ...
    def execute_query_submit_phase3(
        self, action: QuerySubmitAction, observation: Observation,
        verified_search_observation: Observation,
    ) -> ActionResult: ...


class HybridDecisionMaker(Protocol):
    min_confidence: float

    def decide_hybrid(
        self, request: str, observation: Observation, history: Sequence[ActionResult] = (),
    ) -> HybridDecisionResult: ...
    def decide_result_selection(
        self, request: str, observation: Observation,
    ) -> HybridDecisionResult: ...


@dataclass(frozen=True, slots=True)
class HybridTraceStep:
    step: int
    observation_kind: str
    useful_uia_controls: int
    decision_type: str
    jev_latency_ms: int
    jev_confidence: float | None
    grounding_requested: bool
    grounding_objective_length: int | None
    grounding_objective: str | None
    visual_provider: str | None
    visual_provider_latency_ms: int | None
    returned_visual_candidates: int
    action_type: str | None
    executed: bool
    terminal_reason: str | None
    total_step_latency_ms: int
    offered_option_types: tuple[str, ...] = ()
    offered_grounding_needs: tuple[VisualGroundingNeed, ...] = ()
    offered_applications: tuple[OfferedApplication, ...] = ()
    selected_option: str | None = None
    observed_application_id: str | None = None
    observed_window_title: str | None = None
    target_application_id: str | None = None
    target_app_active: bool | None = None
    post_action_settle_ms: int = 0
    open_app_activation_wait_ms: int = 0
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
    task_progress: TaskProgress = field(default_factory=lambda: TaskProgress(False, False, True))
    choice_probabilities: tuple[ChoiceProbability, ...] = ()
    selected_option_probability: float | None = None
    decision_attempts: tuple[DecisionAttempt, ...] = ()
    option_filter_summary: OptionFilterSummary = field(default_factory=OptionFilterSummary)
    decision_effect: DecisionEffect = DecisionEffect.ACT
    release_policy: str = "action_confidence_gate"
    observation_policy: ObservationPolicyDiagnostic = field(
        default_factory=ObservationPolicyDiagnostic,
    )
    visual_pipeline: VisualPipelineDiagnostic | None = None
    visual_grounding_status: VisualGroundingStatus | None = None
    visual_rejection_summary: tuple[tuple[str, int], ...] = ()
    visual_request_fingerprint: VisualRequestFingerprint | None = None
    capture_diagnostics: CaptureDiagnostics | None = None
    visual_readiness: VisualReadinessResult | None = None
    visual_provider_call_count: int = 0
    visual_execution: VisualExecutionDiagnostic | None = None


@dataclass(frozen=True, slots=True)
class VisualEffectEvidence:
    previous_snapshot_id: str
    new_snapshot_id: str | None
    snapshot_changed: bool
    foreground_application_unchanged: bool | None
    window_title_changed: bool | None
    control_count_changed: bool | None
    focus_changed: bool | None
    editable_control_focused: bool | None
    observation_changed: bool


@dataclass(frozen=True, slots=True)
class VisualExecutionDiagnostic:
    mode: str = "phase1_single_click"
    initial_budget: int = 1
    remaining_budget: int = 1
    target_id: str | None = None
    target_role: str | None = None
    confidence: float | None = None
    snapshot_match: bool = False
    foreground_match: bool | None = None
    bounds_match: bool | None = None
    safety_disposition: str | None = None
    click_point_source: str = "local_bbox_center"
    visual_click_attempted: bool = False
    input_issued: bool = False
    post_click_observation_obtained: bool = False
    effect_evidence: VisualEffectEvidence | None = None


@dataclass(frozen=True, slots=True)
class PostClickTypeReadiness:
    ready: bool
    evidence: str
    verification_provider_called: bool
    foreground_match: bool
    target_consistency: bool | None


@dataclass(frozen=True, slots=True)
class Phase2TypeActionDiagnostic:
    literal_source: str = "deterministic_request_extraction"
    literal_length: int = 0
    input_issued: bool = False
    input_diagnostic: LiteralInputDiagnostic | None = None


@dataclass(frozen=True, slots=True)
class Phase2PostTypeDiagnostic:
    fresh_snapshot: bool = False
    observation_obtained: bool = False
    literal_match: bool | None = None


@dataclass(frozen=True, slots=True)
class Phase2ExecutionDiagnostic:
    visual_click_budget_initial: int = 1
    visual_click_budget_remaining: int = 0
    literal_type_budget_initial: int = 1
    literal_type_budget_remaining: int = 1
    clicked_target_id: str | None = None
    clicked_target_role: str | None = None
    type_readiness: PostClickTypeReadiness | None = None
    type_action: Phase2TypeActionDiagnostic | None = None
    post_type: Phase2PostTypeDiagnostic | None = None


@dataclass(frozen=True, slots=True)
class Phase3ExecutionDiagnostic:
    visual_click_budget_initial: int = 2
    visual_click_budget_remaining: int = 1
    literal_type_budget_initial: int = 1
    literal_type_budget_remaining: int = 0
    result_selection_budget_initial: int = 1
    result_selection_budget_remaining: int = 1
    query_input_attempted: bool = True
    query_required: bool = True
    query_input_issued: bool = True
    query_effect_verified: bool | None = None
    literal_length: int = 0
    requested_primary_identity_length: int = 0
    requested_qualifier_count: int = 0
    result_readiness: object | None = None
    provider_call_count: int = 0
    provider_raw_candidate_count: int | None = None
    provider_parsed_candidate_count: int | None = None
    provider_validated_candidate_count: int | None = None
    candidate_count: int = 0
    eligible_candidate_count: int = 0
    selected_target_id: str | None = None
    selected_role: str | None = None
    jev_confidence: float | None = None
    safety_disposition: str | None = None
    second_click_attempted: bool = False
    second_click_input_issued: bool = False
    post_result_snapshot_id: str | None = None
    post_result_fresh: bool = False
    foreground_application_unchanged: bool | None = None
    observation_changed: bool | None = None
    query_submit_budget_initial: int = 1
    query_submit_budget_remaining: int = 1
    query_submit_eligible: bool = False
    query_submit_eligibility_reason: str | None = None
    query_submit_release_policy: str = "bounded_query_submit_policy"
    query_submit_foreground_revalidated: bool = False
    query_submit_application_match: bool | None = None
    query_submit_hwnd_match: bool | None = None
    query_submit_pid_match: bool | None = None
    query_submit_bounds_compatible: bool | None = None
    query_submit_attempted: bool = False
    query_submitted: bool = False
    post_submit_observation_obtained: bool = False
    post_submit_snapshot_changed: bool | None = None
    post_submit_foreground_application_unchanged: bool | None = None
    post_submit_hwnd_unchanged: bool | None = None
    post_submit_pid_unchanged: bool | None = None
    post_submit_title_changed: bool | None = None
    post_submit_control_count_changed: bool | None = None
    post_submit_observation_changed: bool | None = None
    result_grounding_eligible: bool = False
    target_resolution: TargetResolution | None = None


@dataclass(frozen=True, slots=True)
class SelectedVisualTarget:
    id: str
    label: str
    role: str
    snapshot_id: str
    jev_confidence: float | None
    normal_safety_disposition: str
    normal_safety_would_allow: bool
    blocked_reason: str


@dataclass(frozen=True, slots=True)
class HybridDebugResult:
    success: bool
    stop_reason: str
    message: str
    steps: int
    total_run_latency_ms: int
    trace: tuple[HybridTraceStep, ...] = ()
    selected_visual_target: SelectedVisualTarget | None = None
    visual_execution: VisualExecutionDiagnostic | None = None
    phase2_execution: Phase2ExecutionDiagnostic | None = None
    phase3_execution: Phase3ExecutionDiagnostic | None = None


@dataclass
class HybridDebugAgent:
    computer: HybridComputer
    decision_maker: HybridDecisionMaker
    policy: ActionPolicy = field(default_factory=BasicActionPolicy)
    confirmation: Confirmation | None = None
    limits: AgentLimits = field(default_factory=AgentLimits)
    sleep_fn: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic
    include_objectives_in_trace: bool = True
    redactor: Redactor = field(default_factory=Redactor)
    open_app_activation_timeout_seconds: float = 3.0
    directed_capture_callback: Callable[[Observation, ScreenshotCapture], None] | None = None
    result_capture_callback: Callable[[Observation, ScreenshotCapture], None] | None = None
    open_app_poll_interval_seconds: float = 0.2
    visual_click_phase1_enabled: bool = False
    visual_type_phase2_enabled: bool = False
    visual_result_phase3_enabled: bool = False

    def __post_init__(self) -> None:
        if not 0 <= self.open_app_activation_timeout_seconds <= 15:
            raise ValueError("open-app activation timeout must be between 0 and 15 seconds")
        if not 0.01 <= self.open_app_poll_interval_seconds <= 1:
            raise ValueError("open-app poll interval must be between 0.01 and 1 second")

    def _wait_for_application(
        self, app_id: str, initial_observation: Observation | None = None,
    ) -> tuple[Observation | None, int]:
        catalog = getattr(self.policy, "app_catalog", None)
        candidate = catalog.resolve(app_id) if catalog is not None else None
        if candidate is None:
            # OpenAppAction is approved only by trusted application policy.
            # A minimal identity record preserves the historical ID-only wait.
            candidate = ApplicationCandidate(app_id, "unknown", "test")
        wait = wait_for_trusted_application_activation(
            self.computer.observe_local, candidate,
            initial_observation=initial_observation, launch_succeeded=True,
            timeout_seconds=self.open_app_activation_timeout_seconds,
            poll_interval_seconds=self.open_app_poll_interval_seconds,
            clock=self.clock, sleep_fn=self.sleep_fn,
        )
        return wait.observation, wait.diagnostics.activation_elapsed_ms

    def _finish(
        self, success: bool, reason: str, message: str, steps: int, started: float,
        trace: list[HybridTraceStep], selected: SelectedVisualTarget | None = None,
        visual_execution: VisualExecutionDiagnostic | None = None,
        phase2_execution: Phase2ExecutionDiagnostic | None = None,
        phase3_execution: Phase3ExecutionDiagnostic | None = None,
    ) -> HybridDebugResult:
        return HybridDebugResult(
            success, reason, message, steps,
            max(0, round((self.clock() - started) * 1000)), tuple(trace), selected,
            visual_execution,
            phase2_execution,
            phase3_execution,
        )

    def _observation_policy(
        self, decision: HybridDecisionResult, observation: Observation,
    ) -> ObservationPolicyDiagnostic:
        grounding = decision.grounding_need
        if decision.effect is not DecisionEffect.OBSERVE or grounding is None:
            return ObservationPolicyDiagnostic(False, "not_observation_decision")
        if (decision.observation_id != observation.observation_id
                or not observation.observation_id):
            return ObservationPolicyDiagnostic(False, "stale_observation")
        if not observation.app_name.strip():
            return ObservationPolicyDiagnostic(False, "untrusted_foreground_identity")
        if grounding not in decision.offered_grounding_needs:
            return ObservationPolicyDiagnostic(False, "grounding_not_locally_offered")
        if not (decision.selected_option or "").startswith("grounding_"):
            return ObservationPolicyDiagnostic(False, "invalid_grounding_selection")
        if observation.visual_directed_grounding:
            return ObservationPolicyDiagnostic(False, "grounding_requires_local_snapshot")
        fallback = visual_fallback_policy(observation, "")
        if not fallback.required:
            return ObservationPolicyDiagnostic(False, "local_observation_sufficient")
        if any(
            control.focused is True and control.control_type in {"Edit", "Document"}
            and control.is_password is not False
            for control in observation.elements
        ):
            return ObservationPolicyDiagnostic(False, "credential_sensitive_context")
        try:
            bounded = bounded_grounding_request(grounding.objective, grounding.max_candidates)
        except (TypeError, ValueError):
            return ObservationPolicyDiagnostic(False, "invalid_grounding_bounds")
        if (bounded.objective != grounding.objective
                or self.redactor.clean(grounding.objective) != grounding.objective):
            return ObservationPolicyDiagnostic(False, "unsafe_grounding_objective")
        return ObservationPolicyDiagnostic(True, "bounded_observation_allowed", False)

    def _run_phase2_type(
        self, request: str, clicked_observation: Observation, clicked_target,
        post_click: Observation,
    ) -> tuple[bool, str, str, Phase2ExecutionDiagnostic, Observation | None]:
        literal = phase2_type_literal(request)
        base = Phase2ExecutionDiagnostic(
            clicked_target_id=clicked_target.id, clicked_target_role=clicked_target.role,
        )
        if literal is None:
            readiness = PostClickTypeReadiness(False, "insufficient", False, False, None)
            return False, "visual_type_literal_unavailable", "No deterministic literal was available.", replace(
                base, type_readiness=readiness,
            ), None
        context_match = self.computer.visual_context_matches(clicked_observation)
        if has_credential_sensitive_evidence(
            request, clicked_target, clicked_observation, post_click,
        ):
            readiness = PostClickTypeReadiness(False, "insufficient", False, context_match, False)
            return False, "visual_type_safety_denied", "Credential-sensitive context.", replace(
                base, type_readiness=readiness,
            ), post_click
        focused = next((control for control in post_click.elements
                        if control.focused is True and control.enabled is True
                        and control.control_type in {"Edit", "Document"}
                        and control.is_password is False), None)
        type_observation = post_click
        visual_verified = False
        if (focused is not None and context_match
                and self.computer.visual_context_matches(clicked_observation)):
            readiness = PostClickTypeReadiness(True, "strong_local", False, True, True)
        else:
            try:
                verification = self.computer.observe_directed(bounded_grounding_request(
                    "Find the input control that was just activated and is ready for text entry.", 5,
                ))
            except KeyboardInterrupt:
                raise
            except Exception:
                verification = None
            if (verification is None or verification.error
                    or verification.visual_provider_error is not None
                    or not verification.visual_elements):
                readiness = PostClickTypeReadiness(False, "insufficient", True, False, False)
                return False, "visual_type_readiness_insufficient", (
                    "Visual typing readiness could not be established."
                ), replace(base, type_readiness=readiness), verification
            correspondence = verify_visual_field_correspondence(
                clicked_target, clicked_observation, verification,
            )
            same_context = (
                correspondence.window_geometry_stable
                and self.computer.visual_context_matches(clicked_observation)
            )
            if has_credential_sensitive_evidence(
                request, clicked_target, verification,
            ):
                readiness = PostClickTypeReadiness(False, "insufficient", True, same_context, False)
                return False, "visual_type_safety_denied", "Credential-sensitive context.", replace(
                    base, type_readiness=readiness,
                ), verification
            if not same_context or not correspondence.verified:
                readiness = PostClickTypeReadiness(
                    False, "insufficient", True, same_context,
                    correspondence.spatial_correspondence_result == "unique_match",
                )
                return False, "visual_type_readiness_insufficient", (
                    "Visual verification did not match the clicked input field."
                ), replace(base, type_readiness=readiness), verification
            readiness = PostClickTypeReadiness(True, "strong_visual", True, True, True)
            type_observation = verification
            visual_verified = True
        if any(control.focused is True and control.control_type in {"Edit", "Document"}
               and control.is_password is not False for control in type_observation.elements):
            readiness = replace(readiness, ready=False, evidence="insufficient")
            return False, "visual_type_safety_denied", "Credential-sensitive context.", replace(
                base, type_readiness=readiness,
            ), type_observation
        # Consume the type budget before entering the Windows executor.
        type_diag = Phase2TypeActionDiagnostic(literal_length=len(literal))
        executing = replace(
            base, literal_type_budget_remaining=0,
            type_readiness=readiness, type_action=type_diag,
        )
        try:
            result = self.computer.execute_type_phase2(
                TypeAction(literal), type_observation, visual_verified=visual_verified,
            )
        except KeyboardInterrupt:
            raise
        except Exception:
            result = ActionResult(
                False, TypeAction(literal), "Literal type input failed.",
                error="windows_operation_failed",
            )
        executing = replace(
            executing, type_action=replace(
                type_diag, input_issued=result.input_issued,
                input_diagnostic=result.literal_input_diagnostic,
            ),
        )
        settle_started = self.clock()
        try:
            self.sleep_fn(self.limits.settle_action_seconds)
            post_type = self.computer.observe_local()
        except KeyboardInterrupt:
            raise
        except Exception:
            post_type = None
        observed_values = [
            control.observed_text for control in post_type.elements
            if control.focused is True and control.control_type in {"Edit", "Document"}
            and control.is_password is False and control.observed_text is not None
        ] if post_type else []
        literal_match = (
            any(value in {literal, literal + "\r", literal + "\r\n"} for value in observed_values)
            if observed_values else None
        )
        post_diag = Phase2PostTypeDiagnostic(
            bool(post_type and post_type.observation_id != type_observation.observation_id),
            post_type is not None, literal_match,
        )
        finished = replace(executing, post_type=post_diag)
        if not result.success:
            return False, (
                "visual_type_context_changed"
                if result.error in {"stale_observation", "unsafe_target"}
                else "visual_type_input_failed"
            ), result.message, finished, post_type
        if post_type is None:
            return False, "visual_type_post_observation_failed", (
                "Literal input was issued but the post-type observation failed."
            ), finished, None
        return True, "visual_type_phase2_complete", (
            "Phase-2 literal type attempt completed; no submission or continuation occurred."
        ), finished, post_type

    @staticmethod
    def _semantic_text(value: str) -> str:
        normalized = unicodedata.normalize("NFKC", value).casefold()
        return " ".join(re.findall(r"[^\W_]+", normalized, flags=re.UNICODE))

    def _run_phase3_result(
        self, request: str, phase2: Phase2ExecutionDiagnostic, post_type: Observation,
        verified_search_observation: Observation,
    ) -> tuple[bool, str, str, Phase3ExecutionDiagnostic, Observation | None]:
        target_spec = requested_target_spec(request)
        literal_length = phase2.type_action.literal_length if phase2.type_action else 0
        base = Phase3ExecutionDiagnostic(
            query_input_issued=bool(phase2.type_action and phase2.type_action.input_issued),
            query_effect_verified=phase2.post_type.literal_match if phase2.post_type else None,
            literal_length=literal_length,
            requested_primary_identity_length=(
                len(target_spec.primary_identity) if target_spec else 0
            ),
            requested_qualifier_count=len(target_spec.qualifiers) if target_spec else 0,
        )
        if target_spec is None:
            return False, "result_target_unavailable", "A deterministic result target was unavailable.", base, None
        search_roles = {"search field", "search_field", "text field", "text_field", "edit"}
        readiness = phase2.type_readiness
        eligible = bool(
            phase2.type_action and phase2.type_action.input_issued
            and phase2.literal_type_budget_remaining == 0
            and readiness and readiness.ready and readiness.evidence in {"strong_local", "strong_visual"}
            and self._semantic_text(phase2.clicked_target_role or "") in {
                self._semantic_text(role) for role in search_roles
            }
        )
        if not eligible:
            return False, "query_submit_not_eligible", (
                "The typed context was not eligible for bounded query submission."
            ), replace(base, query_submit_eligibility_reason="unverified_search_query_context"), None
        try:
            baseline = self.computer.observe_result_baseline()
        except KeyboardInterrupt:
            raise
        except Exception:
            baseline = None
        if baseline is None or baseline.error or baseline.screenshot is None:
            return False, "query_submit_context_changed", (
                "The query context could not be recaptured before submission."
            ), replace(base, query_submit_eligibility_reason="baseline_unavailable"), baseline
        verified_meta = verified_search_observation.screenshot
        baseline_meta = baseline.screenshot
        app_match = bool(
            baseline.application_id
            and baseline.application_id == verified_search_observation.application_id
        )
        hwnd_match = bool(verified_meta and baseline_meta.window_handle == verified_meta.window_handle)
        pid_match = baseline.process_id == verified_search_observation.process_id
        bounds_match = bool(verified_meta and baseline_meta.window_bounds == verified_meta.window_bounds)
        credential_safe = not any(
            c.focused is True and c.control_type in {"Edit", "Document"}
            and c.is_password is not False for c in baseline.elements
        )
        context_ok = app_match and hwnd_match and pid_match and bounds_match and credential_safe
        diag = replace(
            base, query_submit_eligible=context_ok,
            query_submit_eligibility_reason=(
                "verified_search_query_context" if context_ok else "context_revalidation_failed"
            ),
            query_submit_foreground_revalidated=True,
            query_submit_application_match=app_match, query_submit_hwnd_match=hwnd_match,
            query_submit_pid_match=pid_match, query_submit_bounds_compatible=bounds_match,
        )
        if not context_ok:
            return False, "query_submit_context_changed", (
                "Foreground query context changed before submission."
            ), diag, baseline
        # Consume the one query-submit budget before invoking Windows input.
        diag = replace(
            diag, query_submit_budget_remaining=0, query_submit_attempted=True,
        )
        submit_action = QuerySubmitAction()
        try:
            submit = self.computer.execute_query_submit_phase3(
                submit_action, baseline, verified_search_observation,
            )
        except KeyboardInterrupt:
            raise
        except Exception:
            submit = ActionResult(False, submit_action, "Query submission failed.",
                                  error="windows_operation_failed")
        diag = replace(diag, query_submitted=submit.input_issued)
        if not submit.success:
            reason = (
                "query_submit_context_changed"
                if submit.error in {"stale_observation", "unsafe_target"}
                else "query_submit_input_failed"
            )
            return False, reason, submit.message, diag, baseline
        try:
            post_submit = self.computer.observe_result_baseline()
        except KeyboardInterrupt:
            raise
        except Exception:
            post_submit = None
        if post_submit is None:
            return False, "query_submit_post_observation_failed", (
                "Query was submitted but the fresh local observation failed."
            ), diag, None
        post_meta = post_submit.screenshot
        snapshot_changed = post_submit.observation_id != baseline.observation_id
        post_app_same = post_submit.application_id == baseline.application_id
        post_hwnd_same = bool(post_meta and post_meta.window_handle == baseline_meta.window_handle)
        post_pid_same = post_submit.process_id == baseline.process_id
        title_changed = post_submit.window_title != baseline.window_title
        controls_changed = len(post_submit.elements) != len(baseline.elements)
        observation_changed = snapshot_changed or title_changed or controls_changed
        diag = replace(
            diag, post_submit_observation_obtained=True,
            post_submit_snapshot_changed=snapshot_changed,
            post_submit_foreground_application_unchanged=post_app_same,
            post_submit_hwnd_unchanged=post_hwnd_same,
            post_submit_pid_unchanged=post_pid_same,
            post_submit_title_changed=title_changed,
            post_submit_control_count_changed=controls_changed,
            post_submit_observation_changed=observation_changed,
        )
        if not (post_app_same and post_pid_same):
            return False, "query_submit_context_changed", (
                "Foreground context changed after query submission."
            ), diag, post_submit
        identity = self.redactor.clean(target_spec.primary_identity)[:120]
        qualifier_text = "".join(
            f' with qualifier "{self.redactor.clean(value)[:60]}"'
            for value in target_spec.qualifiers[:2]
        )
        objective = bounded_grounding_request(
            f'Find visible actionable items matching identity "{identity}"{qualifier_text}.', 5,
        )
        try:
            observed = self.computer.observe_result_directed(objective, baseline)
        except KeyboardInterrupt:
            raise
        except Exception:
            observed = None
        if observed is None:
            return False, "result_grounding_failed", "Result observation failed.", diag, None
        if self.result_capture_callback is not None and observed.visual_request_fingerprint is not None:
            take_capture = getattr(self.computer, "take_debug_capture", None)
            capture = take_capture() if callable(take_capture) else None
            if capture is None:
                return False, "result_debug_screenshot_unavailable", (
                    "The exact result grounding capture was unavailable for saving."
                ), diag, observed
            try:
                self.result_capture_callback(observed, capture)
            except KeyboardInterrupt:
                raise
            except Exception:
                return False, "result_debug_screenshot_failed", (
                    "The requested exact result screenshot could not be saved."
                ), diag, observed
            finally:
                capture.discard()
        pipeline = observed.visual_pipeline
        diag = replace(
            diag, result_readiness=observed.result_readiness,
            provider_call_count=observed.visual_provider_call_count,
            candidate_count=len(observed.visual_elements),
            provider_raw_candidate_count=(
                pipeline.provider_raw_element_count if pipeline else None
            ),
            provider_parsed_candidate_count=pipeline.parsed_element_count if pipeline else None,
            provider_validated_candidate_count=(
                pipeline.validated_element_count if pipeline else None
            ),
            result_grounding_eligible=bool(
                submit.input_issued and observed.result_readiness
                and observed.result_readiness.ready
            ),
        )
        if observed.result_readiness is None or not observed.result_readiness.ready:
            return False, "result_readiness_timeout_after_submit", "Submitted results did not become locally ready.", diag, observed
        if observed.visual_provider_error is not None:
            return False, "result_grounding_provider_error", "Result grounding provider failed.", diag, observed
        policy = Phase3ResultSelectionPolicy()
        if not observed.observation_id:
            return False, "result_identity_mismatch", "A bound result observation is required.", diag, observed
        evidence = _phase3_candidate_evidence(observed, self.redactor)
        resolution = resolve_target(
            target_spec, evidence, expected_snapshot_id=observed.observation_id,
        )
        diag = replace(
            diag,
            target_resolution=resolution,
            eligible_candidate_count=len(resolution.admissible_candidate_ids),
        )
        if resolution.status is TargetResolutionStatus.NO_MATCH:
            if not observed.visual_elements:
                return False, "result_grounding_empty", "Visual grounding returned no candidates.", diag, observed
            return False, "result_identity_mismatch", "No candidate passed generic identity and safety checks.", diag, observed
        if resolution.status is TargetResolutionStatus.AMBIGUOUS:
            return False, "result_identity_ambiguous", "Target identity remains ambiguous.", diag, observed
        target_id = resolution.selected_candidate_id
        target = next((item for item in observed.visual_elements if item.id == target_id), None)
        if target is None:
            return False, "result_identity_mismatch", "The resolved candidate is not bound to this visual snapshot.", diag, observed
        eligible = (target,)
        eligible_observation = replace(observed, visual_elements=eligible)
        try:
            decision = self.decision_maker.decide_result_selection(request, eligible_observation)
        except KeyboardInterrupt:
            raise
        except Exception:
            decision = None
        if (decision is None or decision.status != "ready"
                or not isinstance(decision.action, VisualClickAction)
                or decision.confidence is None
                or decision.confidence < self.decision_maker.min_confidence):
            return False, "result_decision_rejected", "Jev did not release a confident result click.", diag, observed
        target = eligible[0]
        if decision.action.target_id != target.id or decision.action.snapshot_id != observed.observation_id:
            return False, "result_decision_rejected", "Jev selected an unavailable result.", diag, observed
        verdict = policy.validate(decision.action, eligible_observation)
        diag = replace(
            diag, selected_target_id=target.id, selected_role=target.role,
            jev_confidence=decision.confidence, safety_disposition=verdict.disposition,
        )
        if verdict.disposition != "allow":
            return False, "result_click_safety_denied", verdict.reason, diag, observed
        # Consume both remaining budgets before the only phase-3 OS input attempt.
        diag = replace(
            diag, visual_click_budget_remaining=0, result_selection_budget_remaining=0,
            second_click_attempted=True,
        )
        try:
            result = self.computer.execute_visual_click_phase3(decision.action, eligible_observation)
        except KeyboardInterrupt:
            raise
        except Exception:
            result = ActionResult(False, decision.action, "Result click failed.",
                                  error="windows_operation_failed")
        diag = replace(diag, second_click_input_issued=result.input_issued)
        try:
            self.sleep_fn(self.limits.settle_action_seconds)
            post = self.computer.observe_local()
        except KeyboardInterrupt:
            raise
        except Exception:
            post = None
        diag = replace(
            diag,
            post_result_snapshot_id=post.observation_id if post else None,
            post_result_fresh=bool(post and post.observation_id != observed.observation_id),
            foreground_application_unchanged=(
                post.application_id == observed.application_id if post else None
            ),
            observation_changed=(
                bool(post and (post.observation_id != observed.observation_id
                               or post.window_title != observed.window_title
                               or len(post.elements) != len(observed.elements))) if post else None
            ),
        )
        if not result.success:
            return False, "result_click_failed", result.message, diag, post
        if post is None:
            return False, "post_result_observation_failed", "Fresh result observation failed.", diag, None
        return True, "visual_result_phase3_complete", (
            "Bounded result selection completed; playback or intended effect was not asserted."
        ), diag, post

    def run(self, request: str) -> HybridDebugResult:
        started = self.clock()
        if not isinstance(request, str) or not request.strip() or self.redactor.clean(request) != request:
            return self._finish(
                False, "invalid_request", "A non-empty request without credentials is required.",
                0, started, [],
            )
        self.computer.set_observation_request(request)
        history: list[ActionResult] = []
        trace: list[HybridTraceStep] = []
        guard = _RepeatGuard(self.limits)
        observation: Observation | None = None
        cache: dict[tuple[str, str], Observation] = {}
        grounded_objective_by_snapshot: dict[str, str] = {}

        for step in range(1, self.limits.max_steps + 1):
            step_started = self.clock()
            if observation is None:
                try:
                    observation = self.computer.observe_local()
                except KeyboardInterrupt:
                    raise
                except Exception:
                    return self._finish(
                        False, "observation_failed", "Local observation failed.",
                        step - 1, started, trace,
                    )
            if observation.error:
                return self._finish(
                    False, "observation_failed", observation.error, step - 1, started, trace,
                )
            fallback = visual_fallback_policy(observation, request)
            kind = "hybrid" if observation.visual_directed_grounding else "local/UIA"
            recent = tuple(history[-self.limits.history_limit:])
            decision_started = self.clock()
            try:
                decision = self.decision_maker.decide_hybrid(request, observation, recent)
                if decision.status == "error" and decision.error in {"api_error", "invalid_response"}:
                    first_attempts = decision.decision_attempts
                    retry = self.decision_maker.decide_hybrid(request, observation, recent)
                    decision = replace(
                        retry, retry_attempted=True,
                        retry_result_category=(
                            retry.decision_error_category or retry.error or
                            ("success" if retry.status != "error" else "unknown_error")
                        ),
                        decision_attempts=first_attempts + tuple(
                            replace(item, attempt=index)
                            for index, item in enumerate(retry.decision_attempts, 2)
                        ),
                    )
            except KeyboardInterrupt:
                raise
            except Exception:
                return self._finish(
                    False, "decision_error", "Jev decision failed.", step, started, trace,
                )
            jev_latency = max(0, round((self.clock() - decision_started) * 1000))
            action_type = decision.action.kind if decision.action is not None else None
            selected_grounding = bool(
                decision.selected_option and decision.selected_option.startswith("grounding_")
            )
            decision_type = "visual_grounding_need" if (
                decision.grounding_need or selected_grounding
            ) else (
                action_type or decision.status
            )
            release_policy = {
                DecisionEffect.OBSERVE: "bounded_observation_policy",
                DecisionEffect.ACT: "action_confidence_gate",
                DecisionEffect.TERMINAL: "terminal",
            }[decision.effect]
            observation_policy = ObservationPolicyDiagnostic()

            def record(
                *, grounding: bool = False, objective: str | None = None,
                executed: bool = False, terminal: str | None = None,
                visual_observation: Observation | None = None,
                observed_state: Observation | None = None,
                target_application_id: str | None = None,
                post_action_settle_ms: int = 0,
                activation_wait_ms: int = 0,
                visual_execution: VisualExecutionDiagnostic | None = None,
            ) -> None:
                source = visual_observation or observed_state or observation
                assert source is not None
                pipeline = source.visual_pipeline
                if pipeline is not None:
                    pipeline = replace(
                        pipeline,
                        jev_visual_option_count=decision.offered_visual_option_count,
                    )
                grounding_status = source.visual_grounding_status
                if (pipeline is not None and pipeline.deduplicated_element_count
                        and not source.visual_elements):
                    grounding_status = VisualGroundingStatus.HANDOFF_EMPTY
                trace.append(HybridTraceStep(
                    step, kind, fallback.useful_named_interactive_controls,
                    decision_type, jev_latency, decision.confidence, grounding,
                    len(objective) if objective is not None else None,
                    self.redactor.clean(objective) if (
                        objective is not None and self.include_objectives_in_trace
                    ) else None,
                    source.visual_provider, source.visual_latency_ms,
                    len(source.visual_elements), action_type, executed, terminal,
                    max(0, round((self.clock() - step_started) * 1000)),
                    decision.offered_option_types,
                    decision.offered_grounding_needs,
                    decision.offered_applications,
                    decision.selected_option,
                    source.application_id,
                    self.redactor.clean(source.window_title)[:200],
                    target_application_id,
                    (source.application_id == target_application_id
                     if target_application_id is not None else None),
                    post_action_settle_ms,
                    activation_wait_ms,
                    decision.decision_error_category,
                    decision.http_status,
                    decision.provider_error_code,
                    decision.response_shape_summary,
                    decision.expected_primitive,
                    decision.offered_option_count,
                    decision.returned_option_id,
                    decision.returned_confidence_raw_type,
                    decision.retry_attempted,
                    decision.retry_result_category,
                    decision.task_progress,
                    decision.choice_probabilities,
                    decision.selected_option_probability,
                    decision.decision_attempts,
                    decision.option_filter_summary,
                    decision.effect,
                    release_policy,
                    observation_policy,
                    pipeline,
                    grounding_status,
                    source.visual_rejection_summary,
                    source.visual_request_fingerprint,
                    source.capture_diagnostics,
                    source.visual_readiness,
                    source.visual_provider_call_count,
                    visual_execution,
                ))

            if decision.status == "error":
                record(terminal="decision_error")
                return self._finish(False, "decision_error", decision.message, step, started, trace)
            if decision.status == "needs_human":
                reason = (
                    "needs_human" if decision.effect is DecisionEffect.TERMINAL
                    and decision.selected_option == "stop"
                    else "low_confidence" if decision.confidence is not None else "needs_human"
                )
                record(terminal=reason)
                return self._finish(False, reason, decision.message, step, started, trace)
            if decision.grounding_need is not None:
                grounding = bounded_grounding_request(
                    decision.grounding_need.objective, decision.grounding_need.max_candidates,
                )
                normalized = " ".join(grounding.objective.casefold().split())
                if grounded_objective_by_snapshot.get(observation.observation_id) == normalized:
                    record(grounding=True, objective=grounding.objective, terminal="repeated_grounding")
                    return self._finish(
                        False, "repeated_grounding",
                        "The same grounding need repeated without state progress.", step, started, trace,
                    )
                observation_policy = self._observation_policy(decision, observation)
                if not observation_policy.eligible:
                    record(terminal="observation_policy_rejected")
                    return self._finish(
                        False, "observation_policy_rejected",
                        "The bounded observation request failed local policy.",
                        step, started, trace,
                    )
                key = (observation.observation_id, normalized)
                grounded = cache.get(key)
                if grounded is None:
                    try:
                        grounded = self.computer.observe_directed(grounding)
                        if self.directed_capture_callback is not None:
                            take_capture = getattr(self.computer, "take_debug_capture", None)
                            capture = take_capture() if callable(take_capture) else None
                            if capture is not None:
                                try:
                                    self.directed_capture_callback(grounded, capture)
                                finally:
                                    capture.discard()
                    except KeyboardInterrupt:
                        raise
                    except Exception:
                        record(grounding=True, objective=grounding.objective,
                               terminal="visual_grounding_failed")
                        return self._finish(
                            False, "visual_grounding_failed", "Directed visual observation failed.",
                            step, started, trace,
                        )
                    if grounded.visual_readiness is not None and not grounded.visual_readiness.ready:
                        stop_reason = f"visual_readiness_{grounded.visual_readiness.reason}"
                        record(
                            grounding=True, objective=grounding.objective,
                            terminal=stop_reason, visual_observation=grounded,
                        )
                        return self._finish(
                            False, stop_reason,
                            "Directed visual observation stopped before the provider call.",
                            step, started, trace,
                        )
                    cache[key] = grounded
                if grounded.error or grounded.visual_provider_error is not None:
                    record(grounding=True, objective=grounding.objective,
                           terminal="visual_grounding_failed", visual_observation=grounded)
                    return self._finish(
                        False, "visual_grounding_failed", "Directed visual observation failed closed.",
                        step, started, trace,
                    )
                if not grounded.visual_elements:
                    record(
                        grounding=True, objective=grounding.objective,
                        terminal="visual_grounding_empty", visual_observation=grounded,
                    )
                    return self._finish(
                        False, "visual_grounding_empty",
                        "Directed visual observation completed without usable candidates.",
                        step, started, trace,
                    )
                grounded_objective_by_snapshot[grounded.observation_id] = normalized
                record(grounding=True, objective=grounding.objective, visual_observation=grounded)
                observation = grounded
                continue
            if decision.effect is DecisionEffect.OBSERVE:
                observation_policy = ObservationPolicyDiagnostic(
                    False, "missing_grounding_request",
                )
                record(terminal="observation_policy_rejected")
                return self._finish(
                    False, "observation_policy_rejected",
                    "The observation decision did not contain a grounding request.",
                    step, started, trace,
                )
            if decision.confidence is None or decision.confidence < self.limits.confidence_threshold:
                record(terminal="low_confidence")
                return self._finish(
                    False, "low_confidence", "Decision confidence is below the execution threshold.",
                    step, started, trace,
                )
            if decision.action is None:
                record(terminal="decision_error")
                return self._finish(
                    False, "decision_error", "Jev did not return an action or grounding need.",
                    step, started, trace,
                )
            action = decision.action
            if isinstance(action, VisualClickAction):
                target = next(
                    (item for item in observation.visual_elements
                     if item.id == action.target_id and action.snapshot_id == observation.observation_id),
                    None,
                )
                if target is None:
                    record(terminal="stale_visual_target")
                    return self._finish(
                        False, "stale_visual_target", "The selected visual target is stale or unresolved.",
                        step, started, trace,
                    )
                verdict = self.policy.validate(action, observation)
                selected = SelectedVisualTarget(
                    target.id, self.redactor.clean(target.label)[:160], target.role,
                    observation.observation_id, decision.confidence, verdict.disposition,
                    verdict.disposition == "allow", (
                        "phase1 single-click budget"
                        if self.visual_click_phase1_enabled
                        else "experimental visual execution disabled"
                    ),
                )
                if self.visual_click_phase1_enabled:
                    phase_verdict = Phase1VisualClickPolicy().validate(
                        action, observation, request, decision.confidence,
                    )
                    initial_diag = VisualExecutionDiagnostic(
                        target_id=target.id, target_role=target.role,
                        confidence=decision.confidence, snapshot_match=True,
                        safety_disposition=phase_verdict.disposition,
                    )
                    if phase_verdict.disposition != "allow":
                        record(
                            terminal="visual_click_safety_denied",
                            visual_execution=initial_diag,
                        )
                        return self._finish(
                            False, "visual_click_safety_denied", phase_verdict.reason,
                            step, started, trace, selected, initial_diag,
                        )
                    # Consume the entire run's budget before invoking OS input.
                    attempted = replace(
                        initial_diag, remaining_budget=0, visual_click_attempted=True,
                    )
                    executor = getattr(self.computer, "execute_visual_click_phase1", None)
                    try:
                        click_result = executor(
                            action, observation, request, decision.confidence,
                        ) if callable(executor) else ActionResult(
                            False, action, "Phase-1 executor is unavailable.",
                            error="unsupported_action",
                        )
                    except KeyboardInterrupt:
                        raise
                    except Exception:
                        click_result = ActionResult(
                            False, action, "Visual click input failed.",
                            error="windows_operation_failed",
                        )
                    attempted = replace(attempted, input_issued=click_result.input_issued)
                    if not click_result.success and click_result.error in {
                        "stale_observation", "unsafe_target", "policy_blocked",
                    }:
                        reason = (
                            "visual_click_safety_denied" if click_result.error == "policy_blocked"
                            else "visual_click_context_changed"
                        )
                        attempted = replace(
                            attempted, foreground_match=False, bounds_match=False,
                        )
                        record(terminal=reason, visual_execution=attempted)
                        return self._finish(
                            False, reason, click_result.message, step, started, trace,
                            selected, attempted,
                        )
                    settle_started = self.clock()
                    try:
                        self.sleep_fn(self.limits.settle_action_seconds)
                        post = self.computer.observe_local()
                    except KeyboardInterrupt:
                        raise
                    except Exception:
                        post = None
                    settle_ms = max(0, round((self.clock() - settle_started) * 1000))
                    before_focus = next((item.id for item in observation.elements
                                         if item.focused is True), None)
                    after_focus = next((item.id for item in post.elements
                                        if item.focused is True), None) if post else None
                    evidence = VisualEffectEvidence(
                        observation.observation_id,
                        post.observation_id if post else None,
                        bool(post and post.observation_id != observation.observation_id),
                        (post.application_id == observation.application_id) if post else None,
                        (post.window_title != observation.window_title) if post else None,
                        (len(post.elements) != len(observation.elements)) if post else None,
                        (after_focus != before_focus) if post else None,
                        (any(item.focused is True and item.control_type in {"Edit", "Document"}
                             and item.is_password is False for item in post.elements)
                         if post else None),
                        bool(post and (
                            post.observation_id != observation.observation_id
                            or post.window_title != observation.window_title
                            or len(post.elements) != len(observation.elements)
                            or after_focus != before_focus
                        )),
                    )
                    completed = replace(
                        attempted, foreground_match=click_result.success,
                        bounds_match=click_result.success,
                        post_click_observation_obtained=post is not None,
                        effect_evidence=evidence,
                    )
                    stop_reason = (
                        "visual_click_phase1_complete" if click_result.success and post is not None
                        else "visual_click_input_failed" if not click_result.success
                        else "post_click_observation_failed"
                    )
                    if (self.visual_type_phase2_enabled
                            and stop_reason == "visual_click_phase1_complete"):
                        phase2_success, phase2_reason, phase2_message, phase2, post_type = (
                            self._run_phase2_type(request, observation, target, post)
                        )
                        if (self.visual_result_phase3_enabled and phase2_success
                                and post_type is not None):
                            phase3_success, phase3_reason, phase3_message, phase3, post_result = (
                                self._run_phase3_result(
                                    request, phase2, post_type, observation,
                                )
                            )
                            record(
                                executed=click_result.input_issued,
                                terminal=phase3_reason,
                                observed_state=post_result or post_type,
                                post_action_settle_ms=settle_ms,
                                visual_execution=completed,
                            )
                            return self._finish(
                                phase3_success, phase3_reason, phase3_message,
                                step, started, trace, selected, completed, phase2, phase3,
                            )
                        record(
                            executed=click_result.input_issued,
                            terminal=phase2_reason,
                            observed_state=post_type or post,
                            post_action_settle_ms=settle_ms,
                            visual_execution=completed,
                        )
                        return self._finish(
                            phase2_success, phase2_reason, phase2_message,
                            step, started, trace, selected, completed, phase2,
                        )
                    record(
                        executed=click_result.input_issued, terminal=stop_reason,
                        observed_state=post, post_action_settle_ms=settle_ms,
                        visual_execution=completed,
                    )
                    return self._finish(
                        stop_reason == "visual_click_phase1_complete", stop_reason,
                        "Phase-1 visual click attempt finished; no further task steps will run.",
                        step, started, trace, selected, completed,
                    )
                record(terminal="visual_action_blocked_for_debug")
                return self._finish(
                    True, "visual_action_blocked_for_debug",
                    "Expected safety boundary reached; experimental visual execution is disabled.",
                    step, started, trace, selected,
                )
            verdict = self.policy.validate(action, observation)
            if verdict.disposition == "deny":
                record(terminal="safety_rejected")
                return self._finish(False, "safety_rejected", verdict.reason, step, started, trace)
            if verdict.disposition == "confirm":
                try:
                    approved = self.confirmation is not None and self.confirmation.confirm(
                        action, verdict.reason,
                    )
                except KeyboardInterrupt:
                    raise
                except Exception:
                    approved = False
                if not approved:
                    record(terminal="confirmation_required")
                    return self._finish(
                        False, "confirmation_required", verdict.reason, step, started, trace,
                    )
            if guard.repeated(action, observation):
                record(terminal="repeated_action")
                return self._finish(
                    False, "repeated_action", "Repeated action/state detected; stopping safely.",
                    step, started, trace,
                )
            if isinstance(action, FinishAction):
                record(terminal="finished")
                return self._finish(True, "finished", action.summary, step, started, trace)
            try:
                result = self.computer.execute(action, observation)
            except KeyboardInterrupt:
                raise
            except Exception:
                result = ActionResult(False, action, "Execution failed.", error="windows_operation_failed")
            history.append(result)
            if result.success and isinstance(action, TypeAction):
                result = replace(result, source_observation_id=observation.observation_id)
                history[-1] = result
            if not result.success:
                record(executed=True, terminal="execution_failed")
                return self._finish(
                    False, "execution_failed", result.error or result.message, step, started, trace,
                )
            cache.clear()
            grounded_objective_by_snapshot.clear()
            if isinstance(action, OpenAppAction):
                observation, wait_ms = self._wait_for_application(action.app_id, observation)
                record(
                    executed=True, observed_state=observation,
                    target_application_id=action.app_id,
                    post_action_settle_ms=wait_ms, activation_wait_ms=wait_ms,
                )
            else:
                settle_started = self.clock()
                try:
                    self.sleep_fn(self.limits.settle_action_seconds)
                except KeyboardInterrupt:
                    raise
                settle_ms = max(0, round((self.clock() - settle_started) * 1000))
                record(executed=True, post_action_settle_ms=settle_ms)
                observation = None
        return self._finish(
            False, "max_steps", "Maximum agent steps reached.", self.limits.max_steps,
            started, trace,
        )


class HybridClickDebugAgent(HybridDebugAgent):
    """Opt-in controller that may attempt one safe visual click and then stops."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.visual_click_phase1_enabled = True


class HybridTypeDebugAgent(HybridClickDebugAgent):
    """Opt-in controller for one visual click followed by one literal TypeAction."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.visual_type_phase2_enabled = True


class HybridResultDebugAgent(HybridTypeDebugAgent):
    """Opt-in bounded search-result selection: two clicks and one type maximum."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.visual_result_phase3_enabled = True
