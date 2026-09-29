"""Experimental LangGraph orchestration using the existing agent contracts.

This module deliberately keeps computer, Jev, and policy implementations in
runtime closures. Graph state contains only the bounded request lifecycle and
typed domain values; it never contains provider clients, screenshots, or UIA
wrappers.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, TypedDict

from langgraph.graph import END, StateGraph

from agent.loop import (
    AgentLimits, AgentResult, StopReason, _RETRYABLE_DECISION_ERRORS, _RepeatGuard,
)
from computer.actions import (
    Action, ClickAction, FinishAction, OpenAppAction, PressKeyAction,
    QuerySubmitAction, TypeAction, VisualClickAction,
)
from computer.interfaces import Computer
from computer.models import Observation
from computer.results import ActionResult
from decision.interfaces import DecisionMaker
from decision.models import DecisionResult
from safety.interfaces import ActionPolicy, Confirmation, SafetyDecision
from safety.policy import BasicActionPolicy


GraphOutcome = Literal["continue", "success", "failure", "pending"]
RecoverableVerificationReason = Literal["post_action_observation_incomplete"]


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

    def __post_init__(self) -> None:
        if type(self.max_replans) is not int or not 0 <= self.max_replans <= 10:
            raise ValueError("max_replans must be an integer between 0 and 10")

    def run(self, request: str, *, dry_run: bool = False) -> GraphAgentResult:
        if not isinstance(request, str) or not request.strip():
            result = AgentResult(
                False, "invalid_request", "A non-empty request is required.",
            )
            return GraphAgentResult(
                result, (), "invalid_request", 0, self.max_replans, None,
            )

        policy = self.policy or BasicActionPolicy()
        sleep_fn = self.sleep_fn
        if sleep_fn is None:
            import time
            sleep_fn = time.sleep
        set_request = getattr(self.computer, "set_observation_request", None)
        if callable(set_request):
            set_request(request)

        repeat_guard = _RepeatGuard(self.limits)
        history_limit = self.limits.history_limit

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
                )
            if not isinstance(decision, DecisionResult):
                return _stop_update(
                    state, "DECIDE", "invalid_decision_result", "decision_error",
                    "Decision returned an invalid result.", observation=observation,
                )
            if decision.status not in {"ready", "needs_human", "error"}:
                return _stop_update(
                    state, "DECIDE", "invalid_decision_status", "decision_error",
                    "Decision returned an invalid status.", observation=observation,
                    decision=decision,
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
                    )
                if not isinstance(decision, DecisionResult):
                    return _stop_update(
                        state, "DECIDE", "invalid_retry_result", "decision_error",
                        "Decision returned an invalid result.", observation=observation,
                    )
            if decision.status == "error":
                return _stop_update(
                    state, "DECIDE", "decision_error", "decision_error",
                    "Decision could not produce a valid action.", observation=observation,
                    decision=decision,
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
                    decision=decision,
                )
            if not _is_action(decision.action):
                return _stop_update(
                    state, "DECIDE", "missing_or_invalid_action", "decision_error",
                    "Decision did not provide a valid action.", observation=observation,
                    decision=decision,
                )
            if decision.confidence is None or decision.confidence < self.limits.confidence_threshold:
                return _stop_update(
                    state, "DECIDE", "confidence_below_threshold", "low_confidence",
                    "Decision confidence is below the execution threshold.", observation=observation,
                    decision=decision, action=decision.action,
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
        builder.add_node("OBSERVE", observe)
        builder.add_node("DECIDE", decide)
        builder.add_node("SAFETY_CHECK", safety_check)
        builder.add_node("EXECUTE", execute)
        builder.add_node("REOBSERVE", reobserve)
        builder.add_node("VERIFY_OR_CONTINUE", verify_or_continue)
        builder.add_node("REPLAN", replan)
        builder.add_node("FINISH", finish)
        builder.add_node("FAIL", fail)
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
        return GraphAgentResult(
            result, diagnostics, transition_reason,
            replan_count=final_state.get("replan_count", 0),
            max_replans=final_state.get("max_replans", self.max_replans),
            last_replan_reason=final_state.get("last_replan_reason"),
        )


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
