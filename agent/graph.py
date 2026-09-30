"""Experimental LangGraph orchestration using the existing agent contracts.

This module deliberately keeps computer, Jev, and policy implementations in
runtime closures. Graph state contains only the bounded request lifecycle and
typed domain values; it never contains provider clients, screenshots, or UIA
wrappers.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from itertools import combinations
import math
import re
import time
from typing import Literal, TypedDict
from uuid import uuid4

from langgraph.graph import END, StateGraph

from agent.loop import (
    AgentLimits, AgentResult, StopReason, _RETRYABLE_DECISION_ERRORS, _RepeatGuard,
)
from computer.actions import (
    Action, ClickAction, FinishAction, OpenAppAction, PressKeyAction,
    QuerySubmitAction, TypeAction, VisualClickAction,
)
from computer.applications import (
    ApplicationCandidateDiagnostic, ApplicationIdentityComparisonDiagnostics,
    ApplicationIdentityEvidence, ApplicationMatch, compare_application_identity_evidence,
    diagnose_application_matches, normalize_app_identity, safe_application_id,
)
from computer.interfaces import Computer
from computer.models import Observation
from computer.results import ActionResult
from decision.interfaces import DecisionMaker
from decision.models import DecisionResult
from safety.interfaces import ActionPolicy, Confirmation, SafetyDecision
from safety.policy import BasicActionPolicy
from agent.telemetry import (
    AgentTelemetryEvent, JsonlTelemetrySink, TelemetryCollector, utc_timestamp,
)


GraphOutcome = Literal["continue", "success", "failure", "pending"]
RecoverableVerificationReason = Literal["post_action_observation_incomplete"]
TrustedCatalogMatchKind = Literal[
    "raw_exact", "canonical_exact", "normalized_exact", "unique_fuzzy",
    "ambiguous", "no_match",
]

_OPEN_APP_REQUEST = re.compile(r"^\s*(?:open|launch|start|run)\s+(.+?)\s*[.!?]?\s*$", re.I)
_SAFE_OPEN_APP_REJECTIONS = frozenset({
    "no_application_match", "low_confidence", "needs_human", "decision_error",
    "api_transport_or_http_failure", "invalid_response", "decision_service_failure",
    "invalid_input", "invalid_decision_result", "invalid_decision_status",
    "missing_or_invalid_action", "confidence_accepted",
})

_TELEMETRY_VALUES = frozenset({
    "action_executed", "action_failed", "confidence_below_threshold",
    "confirmation_declined", "confirmation_exception", "confirmation_required",
    "confirmation_unavailable", "continue_after_fresh_observation",
    "decision_after_replan", "decision_error", "decision_exception",
    "decision_needs_human", "decision_ready", "decision_retry_exception",
    "dry_run_proposal", "execution_exception", "execution_failed",
    "finish_action", "finished", "fresh_snapshot_missing",
    "graph_execution_failed", "interrupted", "invalid_action_result",
    "invalid_action_snapshot_binding", "invalid_decision_result",
    "invalid_decision_status", "invalid_execution_input", "invalid_finish_action",
    "invalid_observation", "invalid_policy_result", "invalid_request",
    "invalid_retry_result", "invalid_safety_input", "low_confidence",
    "max_steps_reached", "max_steps", "missing_fresh_observation",
    "missing_observation", "missing_or_invalid_action", "needs_human",
    "node_exception", "observation_complete", "observation_exception",
    "observation_failed", "observation_reported_error", "policy_denied",
    "policy_exception", "post_action_observation_complete",
    "post_action_observation_exception", "post_action_observation_incomplete",
    "recoverable_post_action_observation", "repeated_action",
    "replan_budget_exhausted", "replan_precondition_failed", "replan_requested",
    "replan_started", "replan_to_decide", "safety_allowed", "safety_rejected",
    "settle_wait_failed", "task_finished", "unknown",
})
_TELEMETRY_PROVIDERS = frozenset({"openai", "gemini", "openrouter", "deepseek"})


def _safe_telemetry_value(value: object) -> str | None:
    return value if isinstance(value, str) and value in _TELEMETRY_VALUES else None


def _safe_int(value: object, maximum: int = 100_000) -> int | None:
    return value if type(value) is int and 0 <= value <= maximum else None


def _safe_latency(value: object) -> int | None:
    return _safe_int(value, 600_000)


def _safe_visual_provider(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    candidate = value.casefold()
    return candidate if candidate in _TELEMETRY_PROVIDERS else None


@dataclass(frozen=True, slots=True)
class GraphNodeDiagnostic:
    """Safe trace entry; contains no request text, control labels, or geometry."""

    node: str
    step: int
    transition_reason: str
    observation_id: str | None = None
    action_kind: str | None = None
    success: bool | None = None
    stop_reason: str | None = None
    replan_count: int = 0
    max_replans: int = 1
    replan_reason: str | None = None
    failure_recoverable: bool | None = None
    open_app_decision: OpenAppDecisionDiagnostic | None = None


@dataclass(frozen=True, slots=True)
class OpenAppDecisionDiagnostic:
    """Allowlisted details for an explicit OPEN_APP decision only."""

    requested_action_kind: Literal["open_app"]
    requested_app_name_normalized: str | None
    trusted_catalog_candidate_count: int
    trusted_catalog_match_kind: TrustedCatalogMatchKind
    trusted_catalog_selected_app_id_present: bool
    candidate_count: int = 0
    matching_candidates: tuple[ApplicationCandidateDiagnostic, ...] = ()
    identity_comparisons: tuple[ApplicationIdentityComparisonDiagnostics, ...] = ()
    decision_action_kind: str | None = None
    decision_confidence: float | None = None
    decision_rejection_reason: str | None = None
    jev_was_called: bool | None = None
    catalog_resolution_before_jev: bool = False


def _requested_open_app_name(request: str) -> tuple[str | None, str | None]:
    match = _OPEN_APP_REQUEST.fullmatch(request)
    if match is None:
        return None, None
    raw_name = match.group(1).strip()
    if (not raw_name or len(raw_name) > 160
            or any(character in raw_name for character in "\\/:")
            or re.search(r"\b(?:api[ _-]?key|token|password|secret)\b", raw_name, re.I)):
        return raw_name, None
    normalized = normalize_app_identity(raw_name)
    if not normalized or len(normalized) > 80:
        return raw_name, None
    return raw_name, normalized


def _open_app_match_kind(
    matches: tuple[object, ...],
) -> tuple[TrustedCatalogMatchKind, bool]:
    if not matches:
        return "no_match", False
    match_kinds = {getattr(item, "match_kind", None) for item in matches}
    if len(match_kinds) != 1:
        return "ambiguous", False
    kind = next(iter(match_kinds))
    if kind not in {
        "raw_exact", "canonical_exact", "normalized_exact", "unique_fuzzy", "ambiguous",
    }:
        kind = "ambiguous" if len(matches) > 1 else "unique_fuzzy"
    if kind == "ambiguous" or len(matches) > 1:
        return "ambiguous", False
    return kind, bool(getattr(matches[0].candidate, "id", None))


def _open_app_decision_context(request: str, decision_maker: DecisionMaker) -> OpenAppDecisionDiagnostic | None:
    raw_name, normalized_name = _requested_open_app_name(request)
    if raw_name is None:
        return None
    catalog = getattr(decision_maker, "app_catalog", None)
    matches: tuple[object, ...] = ()
    catalog_resolved = False
    if catalog is not None:
        finder = getattr(catalog, "find", None)
        if callable(finder):
            catalog_resolved = True
            limit = getattr(decision_maker, "app_candidate_limit", 5)
            try:
                bounded_limit = limit if type(limit) is int and 1 <= limit <= 100 else 5
                matches = tuple(finder(request, bounded_limit))
            except Exception:
                matches = ()
    match_kind, selected_id_present = _open_app_match_kind(matches)
    matching_candidates = diagnose_application_matches(
        tuple(item for item in matches if isinstance(item, ApplicationMatch)),
    )
    identity_comparisons: list[ApplicationIdentityComparisonDiagnostics] = []
    physical_candidates: list[object] = []
    for item in matches:
        if isinstance(item, ApplicationMatch):
            physical_candidates.extend(item.equivalent_candidates or (item.candidate,))
    unique_candidates: list[object] = []
    seen_candidate_keys: set[tuple[str, str | None]] = set()
    for candidate in physical_candidates:
        key = (str(getattr(candidate, "id", "")), getattr(candidate, "identity_fingerprint", None))
        if key not in seen_candidate_keys:
            unique_candidates.append(candidate)
            seen_candidate_keys.add(key)
    comparer = getattr(catalog, "compare_identities", None)
    for left, right in list(combinations(unique_candidates, 2))[:10]:
        left_id = getattr(left, "id", "")
        right_id = getattr(right, "id", "")
        try:
            if callable(comparer):
                identity_comparisons.append(comparer(left_id, right_id))
            else:
                left_fingerprint = getattr(left, "identity_fingerprint", None)
                right_fingerprint = getattr(right, "identity_fingerprint", None)
                result = compare_application_identity_evidence(
                    ApplicationIdentityEvidence(catalog_identity_fingerprint=left_fingerprint)
                    if left_fingerprint else None,
                    ApplicationIdentityEvidence(catalog_identity_fingerprint=right_fingerprint)
                    if right_fingerprint else None,
                )
                identity_comparisons.append(ApplicationIdentityComparisonDiagnostics(
                    safe_application_id(str(left_id)), safe_application_id(str(right_id)),
                    result.comparison_attempted, result.equivalence_result,
                    result.equivalence_signal_kind, result.compared_identity_fields_present,
                ))
        except Exception:
            identity_comparisons.append(ApplicationIdentityComparisonDiagnostics(
                safe_application_id(str(left_id)), safe_application_id(str(right_id)),
                True, "UNKNOWN", "candidate_identity_unavailable",
                compare_application_identity_evidence(None, None).compared_identity_fields_present,
            ))
    return OpenAppDecisionDiagnostic(
        requested_action_kind="open_app",
        requested_app_name_normalized=normalized_name,
        trusted_catalog_candidate_count=len(matches),
        trusted_catalog_match_kind=match_kind,
        trusted_catalog_selected_app_id_present=selected_id_present,
        candidate_count=(len(matching_candidates) if matching_candidates else len(matches)),
        matching_candidates=matching_candidates,
        identity_comparisons=tuple(identity_comparisons),
        catalog_resolution_before_jev=catalog_resolved,
    )


def _complete_open_app_diagnostic(
    diagnostic: OpenAppDecisionDiagnostic | None,
    decision: DecisionResult,
    confidence_threshold: float,
) -> OpenAppDecisionDiagnostic | None:
    if diagnostic is None:
        return None
    confidence = decision.confidence
    safe_confidence = (
        float(confidence) if type(confidence) in {int, float}
        and math.isfinite(float(confidence)) and 0 <= confidence <= 1 else None
    )
    if decision.status == "error":
        rejection = decision.diagnostic if decision.diagnostic in _SAFE_OPEN_APP_REJECTIONS else "decision_error"
    elif decision.status == "needs_human":
        if decision.diagnostic in _SAFE_OPEN_APP_REJECTIONS:
            rejection = decision.diagnostic
        else:
            rejection = "low_confidence" if safe_confidence is not None else "needs_human"
    elif safe_confidence is not None and safe_confidence < confidence_threshold:
        rejection = "low_confidence"
    elif decision.status == "ready" and decision.action is not None:
        rejection = None
    else:
        rejection = "missing_or_invalid_action"
    provider_called = decision.provider_called
    if provider_called is None:
        provider_called = decision.model is not None
    action_kind = _action_kind(decision.action)
    if action_kind is None and isinstance(decision.selected_option, str):
        if decision.selected_option.startswith("open_"):
            action_kind = "open_app"
        elif decision.selected_option in {"finish", "stop"}:
            action_kind = decision.selected_option
    return replace(
        diagnostic,
        decision_action_kind=action_kind,
        decision_confidence=safe_confidence,
        decision_rejection_reason=rejection,
        jev_was_called=provider_called,
    )


class AgentState(TypedDict, total=False):
    """Explicit typed state passed between the experimental graph nodes."""

    request: str
    step: int
    dry_run: bool
    observation: Observation | None
    last_observation: Observation | None
    decision: DecisionResult | None
    last_decision: DecisionResult | None
    action: Action | None
    verdict: SafetyDecision | None
    action_result: ActionResult | None
    history: tuple[ActionResult, ...]
    success: bool
    stop_reason: StopReason
    message: str
    route: str
    transition_reason: str
    diagnostics: tuple[GraphNodeDiagnostic, ...]
    replan_count: int
    max_replans: int
    last_replan_reason: str | None


@dataclass(frozen=True, slots=True)
class GraphAgentResult:
    """Existing agent result plus a bounded, sanitized graph trace."""

    result: AgentResult
    diagnostics: tuple[GraphNodeDiagnostic, ...]
    transition_reason: str
    replan_count: int = 0
    max_replans: int = 1
    last_replan_reason: str | None = None


def _action_kind(action: Action | None) -> str | None:
    if isinstance(action, (ClickAction, VisualClickAction, TypeAction, OpenAppAction,
                           PressKeyAction, QuerySubmitAction, FinishAction)):
        return action.kind
    return None


def _append_diagnostic(
    state: AgentState,
    node: str,
    reason: str,
    *,
    observation: Observation | None = None,
    action: Action | None = None,
    success: bool | None = None,
    stop_reason: StopReason | None = None,
    failure_recoverable: bool | None = None,
    replan_reason: str | None = None,
    open_app_decision: OpenAppDecisionDiagnostic | None = None,
) -> tuple[GraphNodeDiagnostic, ...]:
    previous = state.get("diagnostics", ())
    # The topology performs at most six node visits per bounded action step,
    # plus the final node. Keep a hard cap even if a future edge is miswired.
    cap = max(8, state.get("step", 1) * 8 + 8)
    classified_recoverable = failure_recoverable
    if classified_recoverable is None and success is False:
        classified_recoverable = False
    item = GraphNodeDiagnostic(
        node=node,
        step=state.get("step", 1),
        transition_reason=reason,
        observation_id=observation.observation_id if observation else None,
        action_kind=_action_kind(action),
        success=success,
        stop_reason=stop_reason,
        replan_count=state.get("replan_count", 0),
        max_replans=state.get("max_replans", 1),
        replan_reason=(replan_reason if replan_reason is not None
                       else state.get("last_replan_reason")),
        failure_recoverable=classified_recoverable,
        open_app_decision=open_app_decision,
    )
    return (*previous[-(cap - 1):], item)


def _is_action(value: object) -> bool:
    return isinstance(value, (
        ClickAction, VisualClickAction, TypeAction, OpenAppAction,
        PressKeyAction, QuerySubmitAction, FinishAction,
    ))


@dataclass
class GraphAgent:
    """Incremental graph runner that mirrors the current bounded Agent loop."""

    computer: Computer
    decision_maker: DecisionMaker
    policy: ActionPolicy | None = None
    confirmation: Confirmation | None = None
    limits: AgentLimits = AgentLimits()
    max_replans: int = 1
    sleep_fn: Callable[[float], None] | None = None
    telemetry_collector: TelemetryCollector | None = None

    def __post_init__(self) -> None:
        if type(self.max_replans) is not int or not 0 <= self.max_replans <= 10:
            raise ValueError("max_replans must be an integer between 0 and 10")

    def run(self, request: str, *, dry_run: bool = False) -> GraphAgentResult:
        run_id = uuid4().hex
        started_at = utc_timestamp()
        run_started = time.perf_counter()
        collector = (
            self.telemetry_collector
            if self.telemetry_collector is not None else JsonlTelemetrySink()
        )

        def emit(event: AgentTelemetryEvent) -> None:
            try:
                collector.emit(event)
            except Exception:
                # Telemetry is a side channel and must never change agent behavior.
                pass

        emit(AgentTelemetryEvent(
            event_type="run_started", run_id=run_id, timestamp=started_at,
            started_at=started_at,
        ))

        def completed(result: GraphAgentResult) -> GraphAgentResult:
            ended_at = utc_timestamp()
            emit(AgentTelemetryEvent(
                event_type="run_completed", run_id=run_id, timestamp=ended_at,
                started_at=started_at, ended_at=ended_at,
                total_duration_ms=max(0, round((time.perf_counter() - run_started) * 1000)),
                success=result.result.success,
                stop_reason=_safe_telemetry_value(result.result.stop_reason),
                steps=result.result.steps, replan_count=result.replan_count,
                max_replans=result.max_replans,
            ))
            return result

        if not isinstance(request, str) or not request.strip():
            result = AgentResult(
                False, "invalid_request", "A non-empty request is required.",
            )
            return completed(GraphAgentResult(
                result, (), "invalid_request", 0, self.max_replans, None,
            ))

        policy = self.policy or BasicActionPolicy()
        sleep_fn = self.sleep_fn
        if sleep_fn is None:
            sleep_fn = time.sleep
        set_request = getattr(self.computer, "set_observation_request", None)
        if callable(set_request):
            set_request(request)

        repeat_guard = _RepeatGuard(self.limits)
        history_limit = self.limits.history_limit

        def instrument_node(
            node: str,
            function: Callable[[AgentState], dict[str, object]],
        ) -> Callable[[AgentState], dict[str, object]]:
            def invoke(state: AgentState) -> dict[str, object]:
                started = time.perf_counter()
                update: dict[str, object] | None = None
                failed = False
                try:
                    update = function(state)
                    return update
                except BaseException:
                    failed = True
                    raise
                finally:
                    elapsed_ms = max(0, round((time.perf_counter() - started) * 1000))
                    diagnostics = (
                        update.get("diagnostics") if update is not None
                        else state.get("diagnostics", ())
                    )
                    diagnostic = next(
                        (item for item in reversed(diagnostics or ()) if item.node == node),
                        None,
                    )
                    observation = (
                        update.get("observation") if update is not None else None
                    ) or state.get("observation")
                    action = (
                        update.get("action") if update is not None and "action" in update
                        else state.get("action")
                    )
                    action_kind = _action_kind(action) if _is_action(action) else None
                    transition = (
                        update.get("transition_reason") if update is not None
                        else None
                    ) or (diagnostic.transition_reason if diagnostic is not None else None)
                    if failed:
                        transition = "node_exception"
                    node_success = (
                        False if failed else (
                            update.get("success") if update is not None and "success" in update
                            else diagnostic.success if diagnostic is not None else None
                        )
                    )
                    stop_reason = (
                        update.get("stop_reason") if update is not None
                        else None
                    ) or (diagnostic.stop_reason if diagnostic is not None else None)
                    verdict = (
                        update.get("verdict") if update is not None and "verdict" in update
                        else state.get("verdict")
                    )
                    safety_outcome = getattr(verdict, "disposition", None)
                    if node == "SAFETY_CHECK" and safety_outcome not in {"allow", "deny", "confirm"}:
                        safety_outcome = (
                            "dry_run" if transition == "dry_run_proposal"
                            else "rejected" if node_success is False else "unavailable"
                        )
                    observation_complete = None
                    provider_used = None
                    provider_latency_ms = None
                    if isinstance(observation, Observation):
                        observation_complete = bool(
                            observation.error is None
                            and not observation.truncated
                            and observation.inspection_errors == 0
                        )
                        provider_used = _safe_visual_provider(
                            observation.selected_visual_provider or observation.visual_provider,
                        )
                        provider_latency_ms = _safe_latency(observation.visual_latency_ms)
                    emit(AgentTelemetryEvent(
                        event_type="node_completed", run_id=run_id,
                        timestamp=utc_timestamp(), node=node,
                        step=_safe_int(
                            update.get("step") if update is not None else state.get("step"),
                        ), duration_ms=elapsed_ms,
                        transition_reason=_safe_telemetry_value(transition),
                        action_kind=action_kind,
                        success=node_success if isinstance(node_success, bool) else None,
                        stop_reason=_safe_telemetry_value(stop_reason),
                        observation_complete=observation_complete,
                        provider_used=provider_used,
                        provider_latency_ms=provider_latency_ms,
                        recoverable_failure=(
                            diagnostic.failure_recoverable if diagnostic is not None else None
                        ),
                        safety_outcome=(
                            safety_outcome if safety_outcome in {
                                "allow", "deny", "confirm", "dry_run", "rejected", "unavailable",
                            } else None
                        ),
                    ))
            return invoke

        def observe(state: AgentState) -> dict[str, object]:
            try:
                observation = self.computer.observe()
            except KeyboardInterrupt:
                raise
            except Exception:
                return {
                    "route": "FAIL", "success": False,
                    "stop_reason": "observation_failed",
                    "message": "Observation failed.",
                    "transition_reason": "observation_exception",
                    "diagnostics": _append_diagnostic(
                        state, "OBSERVE", "observation_exception", success=False,
                        stop_reason="observation_failed",
                    ),
                }
            if not isinstance(observation, Observation) or observation.error:
                reason = "invalid_observation" if not isinstance(observation, Observation) else "observation_reported_error"
                return {
                    "observation": observation if isinstance(observation, Observation) else None,
                    "last_observation": observation if isinstance(observation, Observation) else None,
                    "route": "FAIL", "success": False,
                    "stop_reason": "observation_failed", "message": "Observation failed.",
                    "transition_reason": reason,
                    "diagnostics": _append_diagnostic(
                        state, "OBSERVE", reason,
                        observation=observation if isinstance(observation, Observation) else None,
                        success=False, stop_reason="observation_failed",
                    ),
                }
            return {
                "observation": observation, "last_observation": observation,
                "route": "DECIDE", "transition_reason": "observation_complete",
                "diagnostics": _append_diagnostic(
                    state, "OBSERVE", "observation_complete",
                    observation=observation, success=True,
                ),
            }

        def decide(state: AgentState) -> dict[str, object]:
            observation = state.get("observation")
            if observation is None or observation.error:
                return _stop_update(
                    state, "DECIDE", "missing_observation", "decision_error",
                    "A valid observation is required before deciding.",
                )
            prior_history = state.get("history", ())
            recent_history = prior_history[-history_limit:]
            replanning = state.get("transition_reason") == "replan_started"
            replan_reason = state.get("last_replan_reason") if replanning else None
            decision_request = request
            if replan_reason is not None:
                decision_request = (
                    f"{request}\n\nReplanning context: {replan_reason}. "
                    "Choose based on the current observation and prior action history; "
                    "do not retry an action automatically."
                )
            open_app_diagnostic = _open_app_decision_context(
                request, self.decision_maker,
            )
            try:
                decision = self.decision_maker.decide(
                    decision_request, observation, recent_history,
                )
            except KeyboardInterrupt:
                raise
            except Exception:
                return _stop_update(
                    state, "DECIDE", "decision_exception", "decision_error",
                    "Decision failed.", observation=observation,
                    open_app_decision=(
                        replace(
                            open_app_diagnostic, jev_was_called=True,
                            decision_rejection_reason="decision_error",
                        ) if open_app_diagnostic is not None else None
                    ),
                )
            if not isinstance(decision, DecisionResult):
                return _stop_update(
                    state, "DECIDE", "invalid_decision_result", "decision_error",
                    "Decision returned an invalid result.", observation=observation,
                    open_app_decision=open_app_diagnostic,
                )
            open_app_diagnostic = _complete_open_app_diagnostic(
                open_app_diagnostic, decision, self.limits.confidence_threshold,
            )
            if decision.status not in {"ready", "needs_human", "error"}:
                return _stop_update(
                    state, "DECIDE", "invalid_decision_status", "decision_error",
                    "Decision returned an invalid status.", observation=observation,
                    decision=decision, open_app_decision=open_app_diagnostic,
                )
            if decision.status == "error" and decision.error in _RETRYABLE_DECISION_ERRORS:
                try:
                    decision = self.decision_maker.decide(
                        decision_request, observation, recent_history,
                    )
                except KeyboardInterrupt:
                    raise
                except Exception:
                    return _stop_update(
                        state, "DECIDE", "decision_retry_exception", "decision_error",
                        "Decision failed.", observation=observation,
                        open_app_decision=(
                            replace(
                                open_app_diagnostic, jev_was_called=True,
                                decision_rejection_reason="decision_error",
                            ) if open_app_diagnostic is not None else None
                        ),
                    )
                if not isinstance(decision, DecisionResult):
                    return _stop_update(
                        state, "DECIDE", "invalid_retry_result", "decision_error",
                        "Decision returned an invalid result.", observation=observation,
                        open_app_decision=open_app_diagnostic,
                    )
                open_app_diagnostic = _complete_open_app_diagnostic(
                    open_app_diagnostic, decision, self.limits.confidence_threshold,
                )
            if decision.status == "error":
                return _stop_update(
                    state, "DECIDE", "decision_error", "decision_error",
                    "Decision could not produce a valid action.", observation=observation,
                    decision=decision, open_app_decision=open_app_diagnostic,
                )
            if decision.status == "needs_human":
                reason: StopReason = (
                    "needs_human"
                    if decision.selected_option == "stop" or decision.confidence is None
                    else "low_confidence"
                )
                return _stop_update(
                    state, "DECIDE", "decision_needs_human", reason,
                    "The decision requires human input.", observation=observation,
                    decision=decision, open_app_decision=open_app_diagnostic,
                )
            if not _is_action(decision.action):
                return _stop_update(
                    state, "DECIDE", "missing_or_invalid_action", "decision_error",
                    "Decision did not provide a valid action.", observation=observation,
                    decision=decision, open_app_decision=open_app_diagnostic,
                )
            if decision.confidence is None or decision.confidence < self.limits.confidence_threshold:
                return _stop_update(
                    state, "DECIDE", "confidence_below_threshold", "low_confidence",
                    "Decision confidence is below the execution threshold.", observation=observation,
                    decision=decision, action=decision.action,
                    open_app_decision=open_app_diagnostic,
                )
            action = decision.action
            decision_reason = "decision_after_replan" if replanning else "decision_ready"
            return {
                "decision": decision, "last_decision": decision,
                "action": action, "route": "SAFETY_CHECK",
                "transition_reason": decision_reason,
                "diagnostics": _append_diagnostic(
                    state, "DECIDE", decision_reason, observation=observation,
                    action=action, success=True, replan_reason=replan_reason,
                    open_app_decision=open_app_diagnostic,
                ),
            }

        def safety_check(state: AgentState) -> dict[str, object]:
            observation, action = state.get("observation"), state.get("action")
            decision = state.get("decision")
            if observation is None or action is None or decision is None:
                return _stop_update(
                    state, "SAFETY_CHECK", "invalid_safety_input", "safety_rejected",
                    "Safety validation failed closed.", observation=observation, action=action,
                )
            if state.get("dry_run", False):
                return {
                    "route": "FINISH", "success": True,
                    "stop_reason": "needs_human",
                    "message": "Dry run stopped after the first action proposal.",
                    "transition_reason": "dry_run_proposal",
                    "diagnostics": _append_diagnostic(
                        state, "SAFETY_CHECK", "dry_run_proposal",
                        observation=observation, action=action, success=True,
                        stop_reason="needs_human",
                    ),
                }
            try:
                verdict = policy.validate(action, observation)
            except KeyboardInterrupt:
                raise
            except Exception:
                return _stop_update(
                    state, "SAFETY_CHECK", "policy_exception", "safety_rejected",
                    "Safety validation failed closed.", observation=observation, action=action,
                )
            if (not isinstance(verdict, SafetyDecision)
                    or verdict.disposition not in {"allow", "deny", "confirm"}):
                return _stop_update(
                    state, "SAFETY_CHECK", "invalid_policy_result", "safety_rejected",
                    "Safety policy returned an invalid result.", observation=observation, action=action,
                )
            if verdict.disposition == "deny":
                return _stop_update(
                    state, "SAFETY_CHECK", "policy_denied", "safety_rejected",
                    "Safety policy denied the action.", observation=observation, action=action,
                )
            if verdict.disposition == "confirm":
                if self.confirmation is None:
                    return _stop_update(
                        state, "SAFETY_CHECK", "confirmation_unavailable", "confirmation_required",
                        "Human confirmation is required.", observation=observation, action=action,
                    )
                try:
                    approved = self.confirmation.confirm(action, verdict.reason)
                except KeyboardInterrupt:
                    raise
                except Exception:
                    return _stop_update(
                        state, "SAFETY_CHECK", "confirmation_exception", "confirmation_required",
                        "Human confirmation was unavailable.", observation=observation, action=action,
                    )
                if not isinstance(approved, bool) or not approved:
                    return _stop_update(
                        state, "SAFETY_CHECK", "confirmation_declined", "confirmation_required",
                        "Action was not confirmed.", observation=observation, action=action,
                    )
            if repeat_guard.repeated(action, observation):
                return _stop_update(
                    state, "SAFETY_CHECK", "repeated_action", "repeated_action",
                    "Repeated action/state detected; stopping safely.",
                    observation=observation, action=action,
                )
            route = "FINISH" if isinstance(action, FinishAction) else "EXECUTE"
            reason = "finish_action" if route == "FINISH" else "safety_allowed"
            return {
                "verdict": verdict, "route": route,
                "transition_reason": reason,
                "diagnostics": _append_diagnostic(
                    state, "SAFETY_CHECK", reason, observation=observation,
                    action=action, success=True,
                ),
            }

        def execute(state: AgentState) -> dict[str, object]:
            observation, action = state.get("observation"), state.get("action")
            if observation is None or action is None:
                return _stop_update(
                    state, "EXECUTE", "invalid_execution_input", "execution_failed",
                    "Execution failed closed.", observation=observation, action=action,
                )
            try:
                result = self.computer.execute(action, observation)
            except KeyboardInterrupt:
                raise
            except Exception:
                return _stop_update(
                    state, "EXECUTE", "execution_exception", "execution_failed",
                    "Execution failed.", observation=observation, action=action,
                )
            if not isinstance(result, ActionResult):
                return _stop_update(
                    state, "EXECUTE", "invalid_action_result", "execution_failed",
                    "Execution returned an invalid result.", observation=observation, action=action,
                )
            if not result.success:
                return {
                    "action_result": result, "route": "FAIL", "success": False,
                    "stop_reason": "execution_failed", "message": "Execution failed.",
                    "transition_reason": "action_failed",
                    "diagnostics": _append_diagnostic(
                        state, "EXECUTE", "action_failed", observation=observation,
                        action=action, success=False, stop_reason="execution_failed",
                    ),
                }
            if (result.source_observation_id is not None
                    and result.source_observation_id != observation.observation_id):
                return _stop_update(
                    state, "EXECUTE", "invalid_action_snapshot_binding", "execution_failed",
                    "Execution result was bound to a different observation.",
                    observation=observation, action=action,
                )
            history = (*state.get("history", ()), result)[-history_limit:]
            try:
                sleep_fn(
                    self.limits.settle_open_seconds
                    if isinstance(action, OpenAppAction)
                    else self.limits.settle_action_seconds
                )
            except KeyboardInterrupt:
                raise
            except Exception:
                return _stop_update(
                    state, "EXECUTE", "settle_wait_failed", "interrupted",
                    "Interrupted while waiting for the interface to settle.",
                    observation=observation, action=action,
                )
            return {
                "action_result": result, "history": history,
                "route": "REOBSERVE", "transition_reason": "action_executed",
                "diagnostics": _append_diagnostic(
                    state, "EXECUTE", "action_executed", observation=observation,
                    action=action, success=True,
                ),
            }

        def reobserve(state: AgentState) -> dict[str, object]:
            try:
                observation = self.computer.observe()
            except KeyboardInterrupt:
                raise
            except Exception:
                return _stop_update(
                    state, "REOBSERVE", "post_action_observation_exception",
                    "observation_failed", "Post-action observation failed.",
                    action=state.get("action"),
                )
            if not isinstance(observation, Observation) or observation.error:
                return _stop_update(
                    state, "REOBSERVE", "post_action_observation_incomplete",
                    "observation_failed", "Post-action observation failed.",
                    observation=observation if isinstance(observation, Observation) else None,
                    action=state.get("action"),
                )
            source_observation = state.get("observation")
            if (source_observation is None or not source_observation.observation_id
                    or not observation.observation_id
                    or observation.observation_id == source_observation.observation_id):
                return _stop_update(
                    state, "REOBSERVE", "fresh_snapshot_missing", "observation_failed",
                    "A distinct post-action observation is required.",
                    observation=observation, action=state.get("action"),
                )
            return {
                "observation": observation, "last_observation": observation,
                "route": "VERIFY_OR_CONTINUE",
                "transition_reason": "post_action_observation_complete",
                "diagnostics": _append_diagnostic(
                    state, "REOBSERVE", "post_action_observation_complete",
                    observation=observation, action=state.get("action"), success=True,
                ),
            }

        def verify_or_continue(state: AgentState) -> dict[str, object]:
            observation = state.get("observation")
            action = state.get("action")
            next_step = state.get("step", 1) + 1
            if next_step > self.limits.max_steps:
                return {
                    "step": self.limits.max_steps,
                    "route": "FAIL", "success": False, "stop_reason": "max_steps",
                    "message": "Maximum agent steps reached.",
                    "transition_reason": "max_steps_reached",
                    "diagnostics": _append_diagnostic(
                        state, "VERIFY_OR_CONTINUE", "max_steps_reached",
                        observation=observation, action=action,
                        success=False, stop_reason="max_steps",
                    ),
                }
            if observation is None or observation.error or not observation.observation_id:
                return _stop_update(
                    state, "VERIFY_OR_CONTINUE", "missing_fresh_observation",
                    "observation_failed", "A valid fresh observation is required.",
                    observation=observation, action=action,
                )
            # These are explicit structural completeness fields on Observation,
            # not text/error-string heuristics. The snapshot is valid and fresh,
            # but the UIA result may not prove the post-action state completely.
            if observation.truncated is True or observation.inspection_errors > 0:
                reason: RecoverableVerificationReason = "post_action_observation_incomplete"
                if state.get("replan_count", 0) >= state.get("max_replans", self.max_replans):
                    updated = {**state, "last_replan_reason": reason}
                    return {
                        "observation": observation,
                        "last_observation": observation,
                        "route": "FAIL", "success": False,
                        "stop_reason": "observation_failed",
                        "message": "Post-action observation remained incomplete.",
                        "transition_reason": "replan_budget_exhausted",
                        "step": next_step,
                        "last_replan_reason": reason,
                        "diagnostics": _append_diagnostic(
                            updated, "VERIFY_OR_CONTINUE", "replan_budget_exhausted",
                            observation=observation, action=action, success=False,
                            stop_reason="observation_failed", failure_recoverable=True,
                            replan_reason=reason,
                        ),
                    }
                updated = {**state, "step": next_step, "last_replan_reason": reason}
                return {
                    "step": next_step,
                    "last_replan_reason": reason,
                    "route": "REPLAN",
                    "transition_reason": "recoverable_post_action_observation",
                    "diagnostics": _append_diagnostic(
                        updated, "VERIFY_OR_CONTINUE", "replan_requested",
                        observation=observation, action=action, success=True,
                        failure_recoverable=True, replan_reason=reason,
                    ),
                }
            return {
                "step": next_step, "decision": None, "action": None,
                "verdict": None, "action_result": None,
                "route": "DECIDE", "transition_reason": "continue_after_fresh_observation",
                "diagnostics": _append_diagnostic(
                    state, "VERIFY_OR_CONTINUE", "continue_after_fresh_observation",
                    observation=observation, action=action, success=True,
                    failure_recoverable=False,
                ),
            }

        def replan(state: AgentState) -> dict[str, object]:
            observation = state.get("observation")
            reason = state.get("last_replan_reason")
            if (state.get("transition_reason") != "recoverable_post_action_observation"
                    or reason != "post_action_observation_incomplete"
                    or observation is None or observation.error or not observation.observation_id):
                return _stop_update(
                    state, "REPLAN", "replan_precondition_failed", "observation_failed",
                    "Replanning preconditions were not satisfied.",
                    observation=observation, action=state.get("action"),
                )
            replan_count = state.get("replan_count", 0)
            max_replans = state.get("max_replans", self.max_replans)
            if replan_count >= max_replans:
                return _stop_update(
                    state, "REPLAN", "replan_budget_exhausted", "observation_failed",
                    "Replanning budget is exhausted.",
                    observation=observation, action=state.get("action"),
                )
            replan_count += 1
            counted_state: AgentState = {**state, "replan_count": replan_count}
            return {
                # Keep the successful prior ActionResult in history for Jev,
                # while clearing the stale pending action/result fields.
                "decision": None, "action": None, "verdict": None,
                "action_result": None, "replan_count": replan_count,
                "route": "DECIDE", "transition_reason": "replan_started",
                "diagnostics": _append_diagnostic(
                    counted_state, "REPLAN", "replan_to_decide",
                    observation=observation, success=True, failure_recoverable=True,
                    replan_reason=reason,
                ),
            }

        def finish(state: AgentState) -> dict[str, object]:
            if state.get("transition_reason") == "dry_run_proposal":
                return _terminal_update(state, "FINISH", "dry_run_proposal")
            action = state.get("action")
            if not isinstance(action, FinishAction):
                return _stop_update(
                    state, "FINISH", "invalid_finish_action", "decision_error",
                    "Finish node received an invalid action.",
                    observation=state.get("observation"), action=action,
                )
            update = _terminal_update(
                state, "FINISH", "task_finished", action=action,
                success=True, stop_reason="finished",
            )
            update["success"] = True
            update["stop_reason"] = "finished"
            update["message"] = action.summary
            return update

        def fail(state: AgentState) -> dict[str, object]:
            return _terminal_update(state, "FAIL", state.get("transition_reason", "failed"))

        def select_route(state: AgentState) -> str:
            route = state.get("route", "FAIL")
            return route if route in {
                "DECIDE", "SAFETY_CHECK", "EXECUTE", "REOBSERVE",
                "VERIFY_OR_CONTINUE", "REPLAN", "FINISH", "FAIL",
            } else "FAIL"

        builder = StateGraph(AgentState)
        builder.add_node("OBSERVE", instrument_node("OBSERVE", observe))
        builder.add_node("DECIDE", instrument_node("DECIDE", decide))
        builder.add_node("SAFETY_CHECK", instrument_node("SAFETY_CHECK", safety_check))
        builder.add_node("EXECUTE", instrument_node("EXECUTE", execute))
        builder.add_node("REOBSERVE", instrument_node("REOBSERVE", reobserve))
        builder.add_node("VERIFY_OR_CONTINUE", instrument_node("VERIFY_OR_CONTINUE", verify_or_continue))
        builder.add_node("REPLAN", instrument_node("REPLAN", replan))
        builder.add_node("FINISH", instrument_node("FINISH", finish))
        builder.add_node("FAIL", instrument_node("FAIL", fail))
        builder.set_entry_point("OBSERVE")
        for node in (
            "OBSERVE", "DECIDE", "SAFETY_CHECK", "EXECUTE", "REOBSERVE",
            "VERIFY_OR_CONTINUE", "REPLAN",
        ):
            builder.add_conditional_edges(node, select_route, {
                "DECIDE": "DECIDE", "SAFETY_CHECK": "SAFETY_CHECK", "EXECUTE": "EXECUTE",
                "REOBSERVE": "REOBSERVE", "VERIFY_OR_CONTINUE": "VERIFY_OR_CONTINUE",
                "REPLAN": "REPLAN", "FINISH": "FINISH", "FAIL": "FAIL",
            })
        builder.add_edge("FINISH", END)
        builder.add_edge("FAIL", END)
        app = builder.compile()
        initial: AgentState = {
            "request": request, "step": 1, "dry_run": dry_run,
            "observation": None, "last_observation": None,
            "decision": None, "last_decision": None, "action": None,
            "history": (), "success": False, "message": "",
            "route": "OBSERVE", "transition_reason": "run_started",
            "diagnostics": (), "replan_count": 0,
            "max_replans": self.max_replans, "last_replan_reason": None,
        }
        try:
            final_state = app.invoke(
                initial,
                config={"recursion_limit": self.limits.max_steps * 8 + 16},
            )
        except KeyboardInterrupt:
            final_state = {
                **initial, "success": False, "stop_reason": "interrupted",
                "message": "Interrupted.", "transition_reason": "interrupted",
                "diagnostics": (),
            }
        except Exception:
            final_state = {
                **initial, "success": False, "stop_reason": "decision_error",
                "message": "Graph orchestration failed closed.",
                "transition_reason": "graph_execution_failed", "diagnostics": (),
            }
        reason = final_state.get("stop_reason", "decision_error")
        last_observation = final_state.get("last_observation")
        last_decision = final_state.get("last_decision")
        diagnostics = final_state.get("diagnostics", ())
        steps = final_state.get("step", 0)
        if (final_state.get("stop_reason") == "observation_failed" and steps == 1
                and not final_state.get("history")
                and diagnostics and diagnostics[0].node == "OBSERVE"):
            steps = 0
        result = AgentResult(
            success=final_state.get("success", False),
            stop_reason=reason,
            message=final_state.get("message", "Graph orchestration stopped."),
            steps=steps,
            history=final_state.get("history", ()),
            last_observation=last_observation,
            last_decision=last_decision,
        )
        transition_reason = final_state.get("transition_reason", "unknown")
        return completed(GraphAgentResult(
            result, diagnostics, transition_reason,
            replan_count=final_state.get("replan_count", 0),
            max_replans=final_state.get("max_replans", self.max_replans),
            last_replan_reason=final_state.get("last_replan_reason"),
        ))


def _stop_update(
    state: AgentState,
    node: str,
    reason: str,
    stop_reason: StopReason,
    message: str,
    *,
    observation: Observation | None = None,
    decision: DecisionResult | None = None,
    action: Action | None = None,
    failure_recoverable: bool = False,
    open_app_decision: OpenAppDecisionDiagnostic | None = None,
) -> dict[str, object]:
    return {
        "observation": observation if observation is not None else state.get("observation"),
        "last_observation": observation if observation is not None else state.get("last_observation"),
        "decision": decision if decision is not None else state.get("decision"),
        "last_decision": decision if decision is not None else state.get("last_decision"),
        "action": action if action is not None else state.get("action"),
        "route": "FAIL", "success": False, "stop_reason": stop_reason,
        "message": message, "transition_reason": reason,
        "diagnostics": _append_diagnostic(
            state, node, reason, observation=observation or state.get("observation"),
            action=action or state.get("action"), success=False, stop_reason=stop_reason,
            failure_recoverable=failure_recoverable,
            open_app_decision=open_app_decision,
        ),
    }


def _terminal_update(
    state: AgentState,
    node: str,
    reason: str,
    *,
    action: Action | None = None,
    success: bool | None = None,
    stop_reason: StopReason | None = None,
) -> dict[str, object]:
    return {
        "route": "END", "transition_reason": reason,
        "diagnostics": _append_diagnostic(
            state, node, reason, observation=state.get("observation"),
            action=action or state.get("action"),
            success=state.get("success") if success is None else success,
            stop_reason=state.get("stop_reason") if stop_reason is None else stop_reason,
        ),
    }
