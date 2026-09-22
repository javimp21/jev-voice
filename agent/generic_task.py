"""Small experimental controller for bounded generic target activation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
import re
import time
from typing import Protocol

from agent.application_activation import (
    ApplicationActivationDiagnostics, ApplicationActivationWait,
    wait_for_trusted_application_activation,
)
from computer.actions import Action, ClickAction, OpenAppAction, QuerySubmitAction, TypeAction, VisualClickAction
from computer.applications import ApplicationCandidate, ApplicationCatalog
from computer.models import Observation, UIElement
from computer.results import ActionResult
from computer.visual import VisualGroundingRequest, bounded_grounding_request
from decision.context import Redactor, requested_target_spec
from decision.models import TargetChoiceResult, VisualGroundingNeed
from decision.target_resolution import (
    TargetResolution, TargetResolutionStatus, TargetSpec, resolve_target,
)
from agent.target_evidence import (
    TargetEvidenceSet, adapt_observation_candidates, is_query_field,
    is_visual_query_field, query_literal_from_target,
)
from safety.policy import GenericTargetActivationPolicy


class GenericTaskComputer(Protocol):
    visual_provider: object | None

    def observe_local(self) -> Observation: ...
    def observe_directed(self, grounding: VisualGroundingRequest) -> Observation: ...
    def execute(self, action: Action, observation: Observation | None = None) -> ActionResult: ...
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
    steps: int = 0


def _requested_application_query(request: str) -> str | None:
    match = re.match(
        r"^\s*(?:open|launch|start|run)\s+(.+?)\s+(?:and\s+then|and|then)\s+"
        r"(?=(?:open|select|choose|find|locate|search|navigate|go)\b)",
        request, re.I,
    )
    return match.group(1).strip(" \t,;:'\"“”") if match else None


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
    _active_budget: _RunBudget = field(init=False, repr=False)
    _active_trace: list[GenericTaskStep] = field(init=False, repr=False)
    _active_chosen_id: str | None = field(init=False, default=None, repr=False)
    _active_confidence: float | None = field(init=False, default=None, repr=False)
    _active_post_observation: bool = field(init=False, default=False, repr=False)
    _active_post_foreground_stable: bool | None = field(init=False, default=None, repr=False)

    def run(self, request: str) -> GenericTaskDebugResult:
        budget = _RunBudget()
        trace: list[GenericTaskStep] = []
        self._active_budget, self._active_trace = budget, trace
        self._active_chosen_id = None
        self._active_confidence = None
        self._active_post_observation = False
        self._active_post_foreground_stable = None
        last_resolution: TargetResolution | None = None
        last_observation: Observation | None = None
        activation_diagnostics: ApplicationActivationDiagnostics | None = None

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
            )

        if not isinstance(request, str) or not request.strip() or self.redactor.clean(request) != request:
            return finish(False, "invalid_request", "A non-empty request without credentials is required.")
        target = requested_target_spec(request, experimental_generic=True)
        if target is None:
            return finish(False, "target_unavailable", "A bounded target specification could not be derived.")

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

        app_query = _requested_application_query(request)
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
                result = self.computer.execute(action, observation)
                trace.append(GenericTaskStep(
                    budget.steps, GenericCapability.OPEN_APPLICATION, observation.observation_id,
                    action_kind=action.kind, action_succeeded=result.success,
                ))
                if not result.success:
                    return finish(False, "application_launch_failed", "The trusted application did not launch.")
                activation = self._wait_for_application(candidate, observation)
                observation = activation.observation
                activation_diagnostics = activation.diagnostics
                budget.steps += 1
                last_observation = observation
                trace.append(GenericTaskStep(
                    budget.steps, GenericCapability.WAIT_FOR_TRANSITION,
                    observation.observation_id if observation else None,
                ))
                if observation is None or observation.error or observation.application_id != candidate.id:
                    return finish(False, "application_activation_timeout", "The requested application did not become foreground.")

        # Resolve current UIA first. A remote visual call happens only when the
        # current local evidence cannot safely resolve the target.
        observation, last_resolution, evidence_set = self._resolve_with_optional_visual(
            target, observation, allow_grounding=True, reason="target",
        )
        last_observation = observation
        if self._is_resolvable(last_resolution):
            return self._select_activate_and_stop(
                request, target, last_resolution, evidence_set, observation,
                budget, trace, finish,
            )
        if last_resolution.status is TargetResolutionStatus.AMBIGUOUS:
            return finish(False, "target_ambiguous", "Local and visual evidence does not distinguish the target.")

        # Search is considered only after direct activation could not resolve.
        capabilities = self.derive_capabilities(target, observation, last_resolution, budget)
        if not ({GenericCapability.FIND_QUERY_FIELD, GenericCapability.ENTER_LITERAL_QUERY} & set(capabilities)):
            return finish(False, "search_not_available", "No bounded search capability is available from the current state.")
        field_observation, query_field_action = self._find_query_field(observation, budget, trace)
        if field_observation is None:
            return finish(False, "query_field_unavailable", "No safe generic search field was available.")
        observation = field_observation
        last_observation = observation
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
            result = self._execute_target_action(query_field_action, observation, None)
            budget.steps += 1
            trace.append(GenericTaskStep(
                budget.steps, GenericCapability.FIND_QUERY_FIELD, observation.observation_id,
                action_kind=query_field_action.kind, action_succeeded=result.success,
            ))
            if not result.success:
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
        focused_field = self._focused_query_field(observation)
        if focused_field is None or observation.error:
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
        type_verdict = self.policy.validate(type_action, observation)
        if type_verdict.disposition != "allow":
            return finish(False, "type_safety_denied", type_verdict.reason)
        budget.literal_types += 1
        budget.steps += 1
        type_result = self.computer.execute(type_action, observation)
        trace.append(GenericTaskStep(
            budget.steps, GenericCapability.ENTER_LITERAL_QUERY, observation.observation_id,
            action_kind=type_action.kind, action_succeeded=type_result.success,
        ))
        if not type_result.success:
            return finish(False, "literal_type_failed", "The bounded literal query was not safely entered.")
        typed_from = observation

        try:
            observation = self.computer.observe_local()
        except KeyboardInterrupt:
            raise
        except Exception:
            return finish(False, "observation_failed", "Observation after literal typing failed.")
        budget.steps += 1
        last_observation = observation
        trace.append(GenericTaskStep(budget.steps, GenericCapability.OBSERVE, observation.observation_id))
        observation, last_resolution, evidence_set = self._resolve_with_optional_visual(
            target, observation, allow_grounding=True, reason="target",
        )
        last_observation = observation
        if self._is_resolvable(last_resolution):
            return self._select_activate_and_stop(
                request, target, last_resolution, evidence_set, observation,
                budget, trace, finish,
            )
        if last_resolution.status is TargetResolutionStatus.AMBIGUOUS:
            return finish(False, "target_ambiguous", "The visible target remains ambiguous after typing.")

        # Enter is allowed only when a fresh focused query field visibly contains
        # the deterministic literal and no target resolved after typing.
        current_field = self._focused_query_field(observation)
        if (current_field is None or not self._query_value_matches(current_field, literal)
                or not self._foreground_matches(typed_from, observation)):
            return finish(False, "query_submit_not_eligible", "The typed query context could not be revalidated.")
        capabilities = self.derive_capabilities(target, observation, last_resolution, budget)
        if GenericCapability.SUBMIT_QUERY not in capabilities:
            return finish(False, "query_submit_not_eligible", "Local capability policy did not authorize query submission.")
        if budget.query_submits >= self.budgets.query_submits:
            return finish(False, "budget_exhausted", "The query-submit budget is exhausted.")
        if budget.steps + 2 > self.budgets.max_steps:
            return finish(False, "budget_exhausted", "The overall step budget is exhausted.")
        submit_action = QuerySubmitAction()
        budget.query_submits += 1
        budget.steps += 1
        submit_result = self._execute_query_submit(submit_action, observation)
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
        observation, last_resolution, evidence_set = self._resolve_with_optional_visual(
            target, observation, allow_grounding=True, reason="target",
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
        resolution: TargetResolution, budget: _RunBudget,
    ) -> tuple[GenericCapability, ...]:
        """Expose only next steps justified by the current snapshot and budgets."""
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
        self, target: TargetSpec, observation: Observation,
    ) -> tuple[TargetResolution, TargetEvidenceSet]:
        evidence_set = adapt_observation_candidates(observation, self.policy, self.redactor)
        resolution = resolve_target(
            target, evidence_set.candidates,
            expected_snapshot_id=observation.observation_id,
            max_frontier_candidates=self.budgets.max_frontier_candidates,
            frontier_mode=True,
        )
        return resolution, evidence_set

    def _resolve_with_optional_visual(
        self, target: TargetSpec, observation: Observation, *,
        allow_grounding: bool, reason: str,
    ) -> tuple[Observation, TargetResolution, TargetEvidenceSet]:
        resolution, evidence_set = self._resolve(target, observation)
        capabilities = self.derive_capabilities(target, observation, resolution, self._active_budget)
        if self._is_resolvable(resolution) or not allow_grounding or GenericCapability.RESOLVE_TARGET not in capabilities:
            return observation, resolution, evidence_set
        need = VisualGroundingNeed(
            self._target_objective(target),
            "Local UIA evidence could not resolve the requested target safely.", 5,
        )
        grounding = bounded_grounding_request(need.objective, need.max_candidates)
        if self._active_budget.steps + 1 > self.budgets.max_steps:
            return observation, resolution, evidence_set
        self._active_budget.visual_grounding_calls += 1
        try:
            grounded = self.computer.observe_directed(grounding)
        except KeyboardInterrupt:
            raise
        except Exception:
            return observation, resolution, evidence_set
        self._active_budget.steps += 1
        self._active_trace.append(GenericTaskStep(
            self._active_budget.steps, GenericCapability.RESOLVE_TARGET, grounded.observation_id,
            resolution_status="visual_grounding", candidate_count=len(grounded.visual_elements),
        ))
        if grounded.error:
            return grounded, resolution, evidence_set
        resolution, evidence_set = self._resolve(target, grounded)
        return grounded, resolution, evidence_set

    def _target_objective(self, target: TargetSpec) -> str:
        identity = self.redactor.clean(target.primary_identity)[:120]
        role = self.redactor.clean(target.desired_role or "actionable target")[:50]
        return f'Find visible actionable elements corresponding to "{identity}" with semantic role "{role}".'

    def _find_query_field(
        self, observation: Observation, budget: _RunBudget, trace: list[GenericTaskStep],
    ) -> tuple[Observation | None, Action | None]:
        fields = [item for item in observation.elements if is_query_field(item, self.policy)]
        focused = [item for item in fields if item.focused is True]
        if len(focused) == 1:
            return observation, None
        if len(fields) == 1:
            return observation, ClickAction(fields[0].id)
        if len(fields) > 1:
            return None, None
        if (getattr(self.computer, "visual_provider", True) is None
                or budget.visual_grounding_calls >= self.budgets.visual_grounding_calls):
            return None, None
        request = VisualGroundingNeed(
            "Find one visible generic search or query text field for entering the requested literal.",
            "No uniquely eligible search field was exposed by UIA.", 5,
        )
        if budget.steps + 1 > self.budgets.max_steps:
            return None, None
        budget.visual_grounding_calls += 1
        grounding = bounded_grounding_request(request.objective, request.max_candidates)
        try:
            grounded = self.computer.observe_directed(grounding)
        except KeyboardInterrupt:
            raise
        except Exception:
            return None, None
        budget.steps += 1
        trace.append(GenericTaskStep(
            budget.steps, GenericCapability.FIND_QUERY_FIELD, grounded.observation_id,
            candidate_count=len(grounded.visual_elements),
        ))
        if grounded.error:
            return None, None
        fields_visual = [item for item in grounded.visual_elements if is_visual_query_field(item)]
        if len(fields_visual) != 1:
            return None, None
        return grounded, VisualClickAction(grounded.observation_id, fields_visual[0].id)

    def _focused_query_field(self, observation: Observation) -> UIElement | None:
        fields = [item for item in observation.elements
                  if item.focused is True and is_query_field(item, self.policy)]
        return fields[0] if len(fields) == 1 else None

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
        return True

    def _execute_target_action(
        self, action: Action, observation: Observation, semantic_role: str | None,
    ) -> ActionResult:
        verdict = self.policy.validate_candidate(action, observation, semantic_role=semantic_role)
        if verdict.disposition != "allow":
            return ActionResult(False, action, verdict.reason, error="policy_blocked")
        if isinstance(action, VisualClickAction):
            return self.computer.execute_generic_target_activation(action, observation)
        return self.computer.execute(action, observation)

    def _execute_query_submit(
        self, action: QuerySubmitAction, observation: Observation,
    ) -> ActionResult:
        method = getattr(self.computer, "execute_generic_query_submit", None)
        if not callable(method):
            return ActionResult(False, action, "Verified query submission is unavailable.", error="unsupported_action")
        return method(action, observation)

    def _wait_for_application(
        self, candidate: ApplicationCandidate, initial_observation: Observation,
    ) -> ApplicationActivationWait:
        return wait_for_trusted_application_activation(
            self.computer.observe_local, candidate,
            initial_observation=initial_observation, launch_succeeded=True,
            timeout_seconds=self.budgets.app_activation_timeout_seconds,
            poll_interval_seconds=self.budgets.transition_poll_seconds,
            clock=self.clock, sleep_fn=self.sleep_fn,
        )

    def _select_activate_and_stop(
        self, request: str, target: TargetSpec, resolution: TargetResolution,
        evidence_set: TargetEvidenceSet, observation: Observation,
        budget: _RunBudget, trace: list[GenericTaskStep], finish,
    ) -> GenericTaskDebugResult:
        self._active_budget, self._active_trace = budget, trace
        if budget.steps + 2 > self.budgets.max_steps:
            return finish(False, "budget_exhausted", "The overall step budget is exhausted.")
        if resolution.status not in {TargetResolutionStatus.UNIQUE, TargetResolutionStatus.CHOICE}:
            return finish(False, "target_not_resolvable", "Only unique or bounded-choice targets can reach Jev.")
        try:
            decision = self.decision_maker.decide_target_activation(target, resolution)
        except KeyboardInterrupt:
            raise
        except Exception:
            decision = TargetChoiceResult("error", None, None, "Target decision failed.", "decision_error")
        if decision.status != "ready" or decision.candidate_id is None:
            return finish(False, "target_decision_stopped", decision.message)
        if (decision.confidence is None
                or decision.confidence < max(.80, self.decision_maker.min_confidence)):
            return finish(False, "low_confidence", "Target decision confidence is below 0.80.")
        candidate_id = decision.candidate_id
        if candidate_id not in resolution.frontier_candidate_ids:
            return finish(False, "unoffered_target", "The decision selected a candidate outside the local frontier.")
        row = next((item for item in resolution.candidates if item.candidate_id == candidate_id), None)
        action = evidence_set.actions.get(candidate_id)
        if (row is None or not row.admissible or not row.snapshot_valid
                or not row.safety_eligible or not row.actionable or action is None):
            return finish(False, "target_revalidation_failed", "The selected target failed local revalidation.")
        if isinstance(action, VisualClickAction) and action.snapshot_id != observation.observation_id:
            return finish(False, "stale_target", "The visual target belongs to a different snapshot.")
        if budget.final_target_activations >= self.budgets.final_target_activations:
            return finish(False, "budget_exhausted", "The final target activation budget is exhausted.")
        if isinstance(action, VisualClickAction):
            if budget.visual_target_activations >= self.budgets.visual_target_activations:
                return finish(False, "budget_exhausted", "The visual activation budget is exhausted.")
            budget.visual_target_activations += 1
        budget.final_target_activations += 1
        budget.steps += 1
        result = self._execute_target_action(action, observation, row.semantic_role)
        trace.append(GenericTaskStep(
            budget.steps, GenericCapability.ACTIVATE_TARGET, observation.observation_id,
            len(resolution.frontier_candidate_ids), resolution.status.value,
            action.kind, result.success,
        ))
        if not result.success:
            return finish(False, "target_activation_failed", "The target activation did not complete safely.")
        try:
            post = self.computer.observe_local()
        except KeyboardInterrupt:
            raise
        except Exception:
            return finish(False, "post_action_observation_failed", "Fresh post-activation observation failed.")
        budget.steps += 1
        trace.append(GenericTaskStep(budget.steps, GenericCapability.OBSERVE, post.observation_id))
        if post.error or not post.observation_id:
            return finish(False, "post_action_observation_failed", "Fresh post-activation observation was invalid.")
        self._active_post_observation = True
        self._active_post_foreground_stable = self._foreground_matches(observation, post)
        # An input success plus one fresh, stable observation is the bounded
        # effect evidence for this checkpoint. No follow-up action is considered.
        self._active_chosen_id, self._active_confidence = candidate_id, decision.confidence
        return finish(True, "target_activated", "The requested target was activated; the controller stopped.")
