"""Bounded observe -> decide -> validate -> act orchestration."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
import time
from typing import Literal

from computer.actions import Action, ClickAction, FinishAction, OpenAppAction, PressKeyAction, TypeAction, VisualClickAction
from computer.interfaces import Computer
from computer.models import Observation
from computer.results import ActionResult
from decision.interfaces import DecisionMaker
from decision.context import completion_evidence
from decision.models import DecisionResult
from safety.interfaces import ActionPolicy, Confirmation
from safety.policy import BasicActionPolicy


StopReason = Literal[
    "finished", "needs_human", "low_confidence", "decision_error", "safety_rejected",
    "confirmation_required", "execution_failed", "observation_failed", "max_steps",
    "repeated_action", "interrupted", "invalid_request",
]

_RETRYABLE_DECISION_ERRORS = frozenset({"api_error", "invalid_response"})
_SAFE_DECISION_ERRORS = frozenset({"api_error", "invalid_response", "invalid_input", "configuration_error"})
_SAFE_DIAGNOSTICS = frozenset({
    "api_transport_or_http_failure", "decision_service_failure", "duplicate_response_fields",
    "non_finite_json_number", "response_size_limit", "unreadable_json", "missing_model_metadata",
    "unexpected_model_metadata", "unexpected_answers", "expected_choice_answer", "unoffered_option",
    "invalid_confidence_or_probability", "probability_option_mismatch",
    "invalid_probability_distribution", "selected_action_failed_policy", "response_contract_violation",
    "no_application_match",
})


@dataclass(frozen=True, slots=True)
class AgentLimits:
    """Small, bounded loop and settle-time limits."""

    max_steps: int = 8
    history_limit: int = 5
    repeat_limit: int = 2
    tab_repeat_limit: int = 3
    confidence_threshold: float = 0.8
    settle_open_seconds: float = 1.0
    settle_action_seconds: float = 0.2

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.history_limit <= 0 or self.repeat_limit <= 0 or self.tab_repeat_limit <= 0:
            raise ValueError("history and repetition limits must be positive")
        if not 0 < self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        for value in (self.settle_open_seconds, self.settle_action_seconds):
            if not 0 <= value <= 60:
                raise ValueError("settle delays must be between 0 and 60 seconds")


@dataclass(frozen=True, slots=True)
class AgentResult:
    success: bool
    stop_reason: StopReason
    message: str
    steps: int = 0
    history: tuple[ActionResult, ...] = ()
    last_observation: Observation | None = None
    last_decision: DecisionResult | None = None


def _action_signature(action: Action, observation: Observation | None = None) -> tuple[object, ...]:
    if isinstance(action, ClickAction):
        # Resolve the temporary ID only to describe the semantic target. The
        # ID itself must not make an otherwise identical click look new.
        control = next((item for item in observation.elements if item.id == action.target_id), None) if observation else None
        if control is None:
            return ("click", "unresolved")
        return ("click", control.name, control.control_type, control.automation_id)
    if isinstance(action, VisualClickAction):
        control = next((item for item in observation.visual_elements if item.id == action.target_id), None) if observation else None
        return ("visual_click", "unresolved") if control is None else (
            "visual_click", control.label, control.role,
        )
    if isinstance(action, TypeAction):
        return ("type", action.text)
    if isinstance(action, OpenAppAction):
        return ("open_app", action.app_id.lower())
    if isinstance(action, PressKeyAction):
        return ("press_key", tuple(key.lower() for key in action.keys))
    return ("finish", action.summary)


def _state_signature(observation: Observation) -> tuple[object, ...]:
    """Compact state for repetition checks; deliberately excludes snapshot IDs."""
    controls = tuple(
        (item.name, item.control_type, item.enabled, item.visible, item.focused)
        for item in observation.elements
    )
    visual = tuple((item.label, item.role, item.confidence, item.clickable)
                   for item in observation.visual_elements)
    return (observation.app_name, observation.window_title, controls, visual, observation.error)


@dataclass
class _RepeatGuard:
    limits: AgentLimits
    action_counts: dict[tuple[object, ...], int] = field(default_factory=dict)
    pair_counts: dict[tuple[tuple[object, ...], tuple[object, ...]], int] = field(default_factory=dict)

    def repeated(self, action: Action, observation: Observation) -> bool:
        action_key = _action_signature(action, observation)
        self.action_counts[action_key] = self.action_counts.get(action_key, 0) + 1
        pair_key = (action_key, _state_signature(observation))
        self.pair_counts[pair_key] = self.pair_counts.get(pair_key, 0) + 1
        action_limit = self.limits.tab_repeat_limit if isinstance(action, PressKeyAction) and tuple(
            key.lower() for key in action.keys
        ) == ("tab",) else self.limits.repeat_limit
        return (
            self.action_counts[action_key] > action_limit
            or self.pair_counts[pair_key] > self.limits.repeat_limit
        )


@dataclass
class Agent:
    """Run one bounded request with exactly one proposed action per step."""

    computer: Computer
    decision_maker: DecisionMaker
    policy: ActionPolicy = field(default_factory=BasicActionPolicy)
    confirmation: Confirmation | None = None
    limits: AgentLimits = field(default_factory=AgentLimits)
    sleep_fn: Callable[[float], None] = time.sleep
    reporter: Callable[[str, dict[str, object]], None] | None = None

    def _report(self, event: str, data: dict[str, object], debug: bool) -> None:
        if debug and self.reporter is not None:
            self.reporter(event, data)

    @staticmethod
    def _safe_diagnostic(decision: DecisionResult) -> str | None:
        return decision.diagnostic if decision.diagnostic in _SAFE_DIAGNOSTICS else (
            "other_error" if decision.diagnostic else None
        )

    @staticmethod
    def _safe_decision_error(decision: DecisionResult) -> str | None:
        return decision.error if decision.error in _SAFE_DECISION_ERRORS else (
            "other_error" if decision.error else None
        )

    def _result(self, success: bool, reason: StopReason, message: str, steps: int,
                history: Sequence[ActionResult], observation: Observation | None,
                decision: DecisionResult | None) -> AgentResult:
        return AgentResult(success, reason, message, steps, tuple(history), observation, decision)

    def run(self, request: str, *, dry_run: bool = False, debug: bool = False) -> AgentResult:
        if not isinstance(request, str) or not request.strip():
            return self._result(False, "invalid_request", "A non-empty request is required.", 0, (), None, None)
        history: list[ActionResult] = []
        set_request = getattr(self.computer, "set_observation_request", None)
        if callable(set_request):
            set_request(request)
        guard = _RepeatGuard(self.limits)
        last_observation: Observation | None = None
        last_decision: DecisionResult | None = None
        for step in range(1, self.limits.max_steps + 1):
            try:
                observation = self.computer.observe()
            except KeyboardInterrupt:
                return self._result(False, "interrupted", "Interrupted while observing.", step - 1, history, last_observation, last_decision)
            except Exception as exc:
                return self._result(False, "observation_failed", f"Observation failed: {exc}", step - 1, history, last_observation, last_decision)
            last_observation = observation
            focused = next((control for control in observation.elements if control.focused is True), None)
            focused_summary = None if focused is None else {
                "type": focused.control_type,
                "editable": focused.control_type in {"Edit", "Document"},
                "observed_text_present": focused.observed_text is not None,
                "observed_text_length": len(focused.observed_text) if focused.observed_text is not None else None,
                "observed_text_truncated": focused.observed_text_truncated,
            }
            self._report("observation", {
                "step": step, "control_count": len(observation.elements),
                "visual_control_count": len(observation.visual_elements),
                "has_error": bool(observation.error), "focused_control": focused_summary,
                "completion_evidence": completion_evidence(request, observation, history),
            }, debug)
            if observation.error:
                return self._result(False, "observation_failed", observation.error, step - 1, history, observation, last_decision)
            recent_history = tuple(history[-self.limits.history_limit:])
            try:
                decision = self.decision_maker.decide(request, observation, recent_history)
            except KeyboardInterrupt:
                return self._result(False, "interrupted", "Interrupted while deciding.", step - 1, history, observation, last_decision)
            except Exception as exc:
                return self._result(False, "decision_error", f"Decision failed: {exc}", step, history, observation, last_decision)
            last_decision = decision
            self._report("decision", {
                "step": step, "attempt": 1, "status": decision.status,
                "confidence": decision.confidence, "selected_option": decision.selected_option,
                "error": self._safe_decision_error(decision), "diagnostic": self._safe_diagnostic(decision),
            }, debug)
            if decision.status == "error" and decision.error in _RETRYABLE_DECISION_ERRORS:
                self._report("decision_retry", {
                    "step": step, "next_attempt": 2, "error": self._safe_decision_error(decision),
                    "diagnostic": self._safe_diagnostic(decision),
                }, debug)
                try:
                    decision = self.decision_maker.decide(request, observation, recent_history)
                except KeyboardInterrupt:
                    return self._result(False, "interrupted", "Interrupted while retrying decision.", step - 1, history, observation, last_decision)
                except Exception as exc:
                    return self._result(False, "decision_error", f"Decision retry failed: {exc}", step, history, observation, last_decision)
                last_decision = decision
                self._report("decision", {
                    "step": step, "attempt": 2, "status": decision.status,
                    "confidence": decision.confidence, "selected_option": decision.selected_option,
                    "error": self._safe_decision_error(decision), "diagnostic": self._safe_diagnostic(decision),
                }, debug)
            if decision.status == "error":
                return self._result(False, "decision_error", decision.message, step, history, observation, decision)
            if decision.status == "needs_human":
                reason: StopReason = "needs_human" if decision.selected_option == "stop" or decision.confidence is None else "low_confidence"
                return self._result(False, reason, decision.message, step, history, observation, decision)
            if decision.action is None:
                return self._result(False, "decision_error", "Decision did not provide an action.", step, history, observation, decision)
            action = decision.action
            if decision.confidence is None or decision.confidence < self.limits.confidence_threshold:
                return self._result(False, "low_confidence", "Decision confidence is below the execution threshold.", step, history, observation, decision)
            if dry_run:
                return self._result(True, "needs_human", "Dry run stopped after the first action proposal.", step, history, observation, decision)
            try:
                verdict = self.policy.validate(action, observation)
            except Exception as exc:
                return self._result(False, "safety_rejected", f"Safety validation failed closed: {exc}", step, history, observation, decision)
            if verdict.disposition not in {"allow", "deny", "confirm"}:
                return self._result(False, "safety_rejected", "Safety policy returned an invalid disposition.", step, history, observation, decision)
            if verdict.disposition == "deny":
                return self._result(False, "safety_rejected", verdict.reason, step, history, observation, decision)
            if verdict.disposition == "confirm":
                if self.confirmation is None:
                    return self._result(False, "confirmation_required", verdict.reason, step, history, observation, decision)
                try:
                    approved = self.confirmation.confirm(action, verdict.reason)
                except KeyboardInterrupt:
                    return self._result(False, "interrupted", "Interrupted while awaiting confirmation.", step, history, observation, decision)
                except Exception as exc:
                    return self._result(False, "confirmation_required", f"Confirmation failed: {exc}", step, history, observation, decision)
                if not approved:
                    return self._result(False, "confirmation_required", "Action was not confirmed.", step, history, observation, decision)
            if guard.repeated(action, observation):
                return self._result(False, "repeated_action", "Repeated action/state detected; stopping safely.", step, history, observation, decision)
            if isinstance(action, FinishAction):
                return self._result(True, "finished", action.summary, step, history, observation, decision)
            try:
                result = self.computer.execute(action, observation)
            except KeyboardInterrupt:
                return self._result(False, "interrupted", "Interrupted while executing.", step, history, observation, decision)
            except Exception as exc:
                result = ActionResult(False, action, "Execution failed.", error=str(exc))
            history.append(result)
            if len(history) > self.limits.history_limit:
                del history[:-self.limits.history_limit]
            safe_error = result.error if result.error in {
                None, "policy_blocked", "unsafe_target", "windows_operation_failed", "unsupported_action",
                "stale_observation",
            } else ("other_error" if result.error else None)
            self._report("execution", {"step": step, "action": action.kind, "success": result.success,
                                        "error": safe_error}, debug)
            if not result.success:
                return self._result(False, "execution_failed", result.error or result.message, step, history, observation, decision)
            try:
                self.sleep_fn(self.limits.settle_open_seconds if isinstance(action, OpenAppAction) else self.limits.settle_action_seconds)
            except KeyboardInterrupt:
                return self._result(False, "interrupted", "Interrupted while waiting for UI to settle.", step, history, observation, decision)
        return self._result(False, "max_steps", "Maximum agent steps reached.", self.limits.max_steps, history, last_observation, last_decision)
