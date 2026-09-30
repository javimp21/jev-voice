"""Experimental System-2 plan runner above the existing bounded LangGraph."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from collections.abc import Callable, Sequence
import time
from typing import Literal
from uuid import uuid4

from agent.graph import GraphAgent, GraphAgentResult
from agent.loop import AgentLimits
from agent.planner import (
    MAX_PLAN_STEPS, Plan, PlanStatus, PlanStep, PlanStepKind, PlanStepStatus,
    Planner, PlannerCallError, PlannerConfigurationError, PlannerContext,
    PlannerCurrentState, PlannerFailureReason, PlanUpdate, PlanValidationDiagnostic,
    ProviderErrorDiagnostic,
    PlanValidationError,
    validate_plan, validate_plan_update,
)
from agent.telemetry import (
    AgentTelemetryEvent, JsonlTelemetrySink, TelemetryCollector, utc_timestamp,
)
from computer.actions import (
    Action, ClickAction, FinishAction, OpenAppAction, PressKeyAction,
    QuerySubmitAction, TypeAction, VisualClickAction,
)
from computer.interfaces import Computer
from computer.applications import ApplicationCatalog, normalize_app_name
from computer.models import Observation
from computer.results import ActionResult
from decision.interfaces import DecisionMaker
from decision.models import DecisionResult
from safety.interfaces import ActionPolicy, Confirmation
from safety.policy import ALLOWED_KEYS


PlannerStopReason = Literal[
    "finished", "dry_run", "planner_configuration_error", "planner_error",
    "plan_validation_failed", "safety_rejected", "confirmation_required",
    "decision_error", "execution_failed", "observation_failed", "max_steps",
    "low_confidence", "repeated_action", "interrupted", "needs_human",
    "planner_replan_budget_exhausted",
]


@dataclass(frozen=True, slots=True)
class PlannerAgentResult:
    success: bool
    stop_reason: PlannerStopReason
    message: str
    plan: Plan | None = field(default=None, repr=False)
    steps_completed: int = 0
    steps_attempted: int = 0
    total_steps: int = 0
    planner_replan_count: int = 0
    max_planner_replans: int = 1
    local_replan_count: int = 0
    planner_latency_ms: int | None = None
    error_category: str | None = None
    validation_diagnostic: PlanValidationDiagnostic | None = None
    provider_diagnostic: ProviderErrorDiagnostic | None = None


class _PlanStepDecisionMaker:
    """Narrow Jev's local choice to the current typed subgoal."""

    def __init__(
        self, decision_maker: DecisionMaker, step: PlanStep,
        app_catalog: ApplicationCatalog | None,
    ) -> None:
        self._decision_maker = decision_maker
        self._step = step
        self._app_catalog = app_catalog

    def decide(
        self, _request: str, observation: Observation,
        history: Sequence[ActionResult] = (),
    ) -> DecisionResult:
        if self._step.kind is PlanStepKind.FINISH:
            return DecisionResult(
                "ready", FinishAction("Planned steps complete."), 1.0,
                "The explicit plan finish step was reached.",
                observation_id=observation.observation_id,
                selected_option="planner_finish",
            )
        instruction = _step_instruction(self._step)
        decision = self._decision_maker.decide(instruction, observation, history)
        if not isinstance(decision, DecisionResult):
            return DecisionResult(
                "error", None, None, "The local decision result was invalid.",
                error="invalid_response", diagnostic="planner_step_invalid_decision",
            )
        if decision.status != "ready" or decision.action is None:
            return decision
        finish_evidence = (
            self._step.kind is PlanStepKind.FINISH
            or _finish_evidence_is_fresh(observation, history)
        )
        if not _action_allowed_for_step(
            self._step, decision.action, self._app_catalog,
            allow_finish=finish_evidence,
        ):
            return DecisionResult(
                "error", None, None, "The selected action does not match the current plan step.",
                observation_id=observation.observation_id,
                error="invalid_input", diagnostic="planner_step_action_mismatch",
            )
        return decision


def _action_allowed_for_step(
    step: PlanStep, action: Action, app_catalog: ApplicationCatalog | None,
    *, allow_finish: bool = False,
) -> bool:
    if isinstance(action, FinishAction):
        # Jev may confirm a non-finish subgoal only after an action has run and
        # the graph has obtained a new observation; it cannot skip the action.
        return step.kind is PlanStepKind.FINISH or allow_finish
    if step.kind is PlanStepKind.OPEN_APP:
        if not isinstance(action, OpenAppAction) or app_catalog is None or step.target is None:
            return False
        candidates = app_catalog.discover()
        exact_ids = {
            candidate.id for candidate in candidates
            if normalize_app_name(candidate.display_name) == normalize_app_name(step.target)
        }
        return len(exact_ids) == 1 and action.app_id in exact_ids
    if step.kind is PlanStepKind.ACTIVATE_TARGET:
        return isinstance(action, (ClickAction, VisualClickAction))
    if step.kind is PlanStepKind.SEARCH:
        if isinstance(action, TypeAction):
            return action.text == step.payload
        if isinstance(action, PressKeyAction):
            return tuple(key.casefold() for key in action.keys) in ALLOWED_KEYS
        return isinstance(action, (ClickAction, VisualClickAction, QuerySubmitAction))
    if step.kind is PlanStepKind.PRESS_KEY:
        return (
            isinstance(action, PressKeyAction)
            and "+".join(item.casefold() for item in action.keys) == (step.target or "")
        )
    if step.kind is PlanStepKind.TYPE_TEXT:
        if isinstance(action, TypeAction):
            return action.text == step.payload
        if isinstance(action, PressKeyAction):
            return tuple(key.casefold() for key in action.keys) in ALLOWED_KEYS
        return isinstance(action, (ClickAction, VisualClickAction))
    return step.kind is PlanStepKind.FINISH and isinstance(action, FinishAction)


def _step_instruction(step: PlanStep) -> str:
    descriptions = {
        PlanStepKind.OPEN_APP: f"Open the requested application {step.target!r}.",
        PlanStepKind.ACTIVATE_TARGET: f"Activate the visible semantic target {step.target!r}.",
        PlanStepKind.SEARCH: f"Search using the exact user-provided text {step.payload!r}.",
        PlanStepKind.PRESS_KEY: f"Press the allowlisted key combination {step.target!r}.",
        PlanStepKind.TYPE_TEXT: f"Type the exact user-provided text {step.payload!r}.",
        PlanStepKind.FINISH: "Finish the plan.",
    }
    return (
        "Complete only this current planned subgoal using one currently available safe action at a time. "
        "Do not execute later plan steps or invent text. Fresh UI evidence is required before completion. "
        + descriptions[step.kind]
    )


@dataclass(slots=True)
class _PlannerChildCollector:
    collector: TelemetryCollector
    planner_run_id: str

    def emit(self, event: AgentTelemetryEvent) -> None:
        from dataclasses import replace as dataclass_replace
        self.collector.emit(dataclass_replace(event, planner_run_id=self.planner_run_id))


@dataclass
class PlannerAgent:
    """Plan and execute high-level steps through isolated GraphAgent runs."""

    planner: Planner
    computer: Computer
    decision_maker: DecisionMaker
    policy: ActionPolicy | None = None
    confirmation: Confirmation | None = None
    limits: AgentLimits = AgentLimits()
    max_planner_replans: int = 1
    telemetry_collector: TelemetryCollector | None = None
    sleep_fn: Callable[[float], None] | None = None
    app_catalog: ApplicationCatalog | None = None

    def __post_init__(self) -> None:
        if type(self.max_planner_replans) is not int or not 0 <= self.max_planner_replans <= 3:
            raise ValueError("max_planner_replans must be between 0 and 3")
        if self.limits.max_steps > MAX_PLAN_STEPS:
            raise ValueError("planner max_steps cannot exceed the plan limit")

    def run(
        self, task: str, *, dry_run: bool = False, initial_plan: Plan | None = None,
    ) -> PlannerAgentResult:
        planner_run_id = uuid4().hex
        started_at = utc_timestamp()
        run_started = time.perf_counter()
        collector = (
            self.telemetry_collector
            if self.telemetry_collector is not None else JsonlTelemetrySink()
        )
        planner_replans = 0
        local_replans = 0
        completed_count = 0
        attempted_count = 0
        total_steps = 0
        total_planner_latency = 0
        plan: Plan | None = None

        def emit(event_type: str, **fields: object) -> None:
            try:
                collector.emit(AgentTelemetryEvent(
                    event_type=event_type, run_id=planner_run_id,
                    planner_run_id=planner_run_id, timestamp=utc_timestamp(), **fields,
                ))
            except Exception:
                # Telemetry failures never change whether or how an action is run.
                pass

        emit("run_started", started_at=started_at)

        def finish(
            success: bool, stop_reason: PlannerStopReason, message: str,
            *, error_category: str | None = None,
            validation_diagnostic: PlanValidationDiagnostic | None = None,
            provider_diagnostic: ProviderErrorDiagnostic | None = None,
        ) -> PlannerAgentResult:
            ended_at = utc_timestamp()
            emit(
                "run_completed", started_at=started_at, ended_at=ended_at,
                total_duration_ms=max(0, round((time.perf_counter() - run_started) * 1000)),
                success=success, stop_reason=stop_reason,
                steps=total_steps, planner_replan_count=planner_replans,
                max_replans=self.max_planner_replans,
                plan_step_count=len(plan.steps) if plan is not None else 0,
            )
            return PlannerAgentResult(
                success, stop_reason, message, plan,
                completed_count, attempted_count, total_steps,
                planner_replans, self.max_planner_replans, local_replans,
                total_planner_latency or None, error_category, validation_diagnostic,
                provider_diagnostic,
            )

        if not isinstance(task, str) or not task.strip():
            return finish(False, "plan_validation_failed", "A non-empty task is required.",
                          error_category="invalid_task")
        try:
            if initial_plan is None:
                planner_started = time.perf_counter()
                try:
                    candidate = self.planner.plan(task, PlannerContext(self.limits.max_steps))
                finally:
                    latency = max(0, round((time.perf_counter() - planner_started) * 1000))
                    total_planner_latency += latency
                    emit(
                        "planner_called", planner_called=True,
                        planner_latency_ms=latency,
                        planner_replan_count=planner_replans,
                    )
            else:
                candidate = initial_plan
            plan = validate_plan(candidate, task, max_steps=self.limits.max_steps)
        except PlannerConfigurationError:
            emit("plan_validation", plan_validation_result="rejected")
            return finish(False, "planner_configuration_error", "Planner configuration is unavailable.",
                          error_category="configuration_error")
        except PlannerCallError as exc:
            emit("plan_validation", plan_validation_result="rejected")
            return finish(False, "planner_error", "The planner did not return a usable structured plan.",
                          error_category=exc.category, provider_diagnostic=exc.diagnostic)
        except PlanValidationError as exc:
            diagnostic = exc.diagnostic
            emit(
                "plan_validation", plan_validation_result="rejected",
                validation_stage=diagnostic.validation_stage,
                validation_code=diagnostic.validation_code,
                validation_reason_category=diagnostic.reason_category,
                validation_step_index=diagnostic.step_index,
                validation_step_kind=diagnostic.step_kind,
                validation_field_name=diagnostic.field_name,
                validation_field_path=diagnostic.field_path,
                validation_error_type=diagnostic.error_type,
            )
            return finish(False, "plan_validation_failed", "The plan failed local validation.",
                          error_category=exc.category, validation_diagnostic=diagnostic)
        except Exception:
            emit("plan_validation", plan_validation_result="rejected")
            return finish(False, "planner_error", "Planning failed closed.",
                          error_category="planner_error")

        emit(
            "plan_validation", plan_validation_result="accepted",
            plan_step_count=len(plan.steps),
        )
        if dry_run:
            return finish(True, "dry_run", "A valid plan was created; no action was executed.")

        plan = plan.model_copy(update={"status": PlanStatus.RUNNING})
        cursor = 0
        while cursor < len(plan.steps):
            if total_steps >= self.limits.max_steps:
                plan = plan.model_copy(update={"status": PlanStatus.FAILED})
                return finish(False, "max_steps", "The planner execution step budget was exhausted.")
            step = plan.steps[cursor]
            attempted_count += 1
            emit(
                "plan_step_started", current_plan_step=step.step_id,
                plan_step_kind=step.kind.value, plan_step_count=len(plan.steps),
                planner_replan_count=planner_replans,
            )
            step_decider = _PlanStepDecisionMaker(
                self.decision_maker, step, self.app_catalog,
            )
            child_collector = _PlannerChildCollector(collector, planner_run_id)
            remaining = self.limits.max_steps - total_steps
            graph_limits = replace(self.limits, max_steps=remaining)
            graph_kwargs: dict[str, object] = {
                "policy": self.policy,
                "confirmation": self.confirmation,
                "limits": graph_limits,
                "telemetry_collector": child_collector,
            }
            if self.sleep_fn is not None:
                graph_kwargs["sleep_fn"] = self.sleep_fn
            graph = GraphAgent(self.computer, step_decider, **graph_kwargs)  # type: ignore[arg-type]
            graph_result = graph.run(_step_instruction(step))
            total_steps += max(1, graph_result.result.steps)
            local_replans += graph_result.replan_count
            if graph_result.result.success and graph_result.result.stop_reason == "finished":
                updated_step = step.model_copy(update={
                    "status": PlanStepStatus.COMPLETE,
                    "attempts": step.attempts + 1,
                    "last_failure_reason": None,
                })
                plan = _replace_step(plan, cursor, updated_step)
                completed_count += 1
                cursor += 1
                continue

            failure_reason = _map_graph_failure(graph_result)
            updated_failed = step.model_copy(update={
                "status": PlanStepStatus.FAILED,
                "attempts": step.attempts + 1,
                "last_failure_reason": PlannerFailureReason.STEP_EXECUTION_FAILED,
            })
            plan = _replace_step(plan, cursor, updated_failed, status=PlanStatus.FAILED)
            if failure_reason is None:
                return finish(
                    False, _safe_stop_reason(graph_result),
                    "The current plan step did not complete; execution stopped safely.",
                    error_category="step_failed",
                )
            if planner_replans >= self.max_planner_replans:
                return finish(
                    False, "planner_replan_budget_exhausted",
                    "The planner replan budget was exhausted.",
                    error_category="planner_replan_budget_exhausted",
                )
            fresh = _observe_fresh(self.computer, graph_result.result.last_observation)
            if fresh is None:
                return finish(
                    False, "observation_failed",
                    "A complete fresh observation is required before replanning.",
                    error_category="fresh_observation_unavailable",
                )
            completed_ids = tuple(
                item.step_id for item in plan.steps[:cursor]
                if item.status is PlanStepStatus.COMPLETE
            )
            current_state = PlannerCurrentState(
                completed_ids, step.step_id, _observation_complete(fresh),
                max(0, self.limits.max_steps - cursor - 1),
            )
            planner_replans += 1
            planner_started = time.perf_counter()
            try:
                update_candidate = self.planner.replan(
                    plan, current_state, failure_reason,
                )
            except PlannerCallError as exc:
                latency = max(0, round((time.perf_counter() - planner_started) * 1000))
                total_planner_latency += latency
                emit(
                    "planner_called", planner_called=True, planner_latency_ms=latency,
                    planner_replan_count=planner_replans,
                    planner_replan_reason=failure_reason.value,
                )
                emit("plan_validation", plan_validation_result="rejected")
                return finish(False, "planner_error", "Planner replan failed closed.",
                              error_category=exc.category,
                              provider_diagnostic=exc.diagnostic)
            except Exception:
                latency = max(0, round((time.perf_counter() - planner_started) * 1000))
                total_planner_latency += latency
                emit(
                    "planner_called", planner_called=True, planner_latency_ms=latency,
                    planner_replan_count=planner_replans,
                    planner_replan_reason=failure_reason.value,
                )
                emit("plan_validation", plan_validation_result="rejected")
                return finish(False, "planner_error", "Planner replan failed closed.",
                              error_category="planner_error")
            latency = max(0, round((time.perf_counter() - planner_started) * 1000))
            total_planner_latency += latency
            emit(
                "planner_called", planner_called=True, planner_latency_ms=latency,
                planner_replan_count=planner_replans,
                planner_replan_reason=failure_reason.value,
            )
            try:
                update = validate_plan_update(
                    update_candidate, task, max_steps=self.limits.max_steps,
                    existing_steps=plan.steps[:cursor + 1],
                )
                combined = [*plan.steps[:cursor + 1], *update.steps]
                if len(combined) > self.limits.max_steps:
                    raise PlanValidationError(
                        "invalid_plan_size", validation_stage="replan",
                        validation_code="too_many_steps", reason_category="bounds",
                        field_name="steps",
                    )
                if any(item.kind is PlanStepKind.FINISH for item in combined[:-1]):
                    finish_index = next(
                        index for index, item in enumerate(combined[:-1])
                        if item.kind is PlanStepKind.FINISH
                    )
                    raise PlanValidationError(
                        "invalid_step_fields", validation_stage="replan",
                        validation_code="premature_finish_step",
                        reason_category="required_step", step_index=finish_index,
                        step_kind=PlanStepKind.FINISH.value, field_name="kind",
                    )
                plan = Plan(
                    objective=plan.objective, steps=combined,
                    max_steps=self.limits.max_steps, status=PlanStatus.RUNNING,
                )
            except PlanValidationError as exc:
                diagnostic = exc.diagnostic
                emit(
                    "plan_validation", plan_validation_result="rejected",
                    validation_stage=diagnostic.validation_stage,
                    validation_code=diagnostic.validation_code,
                    validation_reason_category=diagnostic.reason_category,
                    validation_step_index=diagnostic.step_index,
                    validation_step_kind=diagnostic.step_kind,
                    validation_field_name=diagnostic.field_name,
                    validation_field_path=diagnostic.field_path,
                    validation_error_type=diagnostic.error_type,
                )
                return finish(False, "plan_validation_failed", "The replacement plan failed validation.",
                              error_category=exc.category,
                              validation_diagnostic=diagnostic)
            emit(
                "plan_validation", plan_validation_result="accepted",
                plan_step_count=len(plan.steps),
            )
            emit(
                "planner_replanned", planner_replan_count=planner_replans,
                planner_replan_reason=failure_reason.value,
                plan_step_count=len(plan.steps),
            )
            cursor += 1

        plan = plan.model_copy(update={"status": PlanStatus.COMPLETE})
        return finish(True, "finished", "All validated plan steps completed.")


def _replace_step(
    plan: Plan, index: int, step: PlanStep, *, status: PlanStatus | None = None,
) -> Plan:
    updated = list(plan.steps)
    updated[index] = step
    return plan.model_copy(update={"steps": updated, "status": status or PlanStatus.RUNNING})


def _observation_complete(observation: Observation) -> bool:
    return bool(
        observation.error is None and not observation.truncated
        and observation.inspection_errors == 0
    )


def _finish_evidence_is_fresh(
    observation: Observation, history: Sequence[ActionResult],
) -> bool:
    if not history or not _observation_complete(observation) or not observation.observation_id:
        return False
    prior_snapshot = history[-1].source_observation_id
    return prior_snapshot is None or prior_snapshot != observation.observation_id


def _observe_fresh(computer: Computer, previous: Observation | None) -> Observation | None:
    try:
        observation = computer.observe()
    except KeyboardInterrupt:
        raise
    except Exception:
        return None
    if (not isinstance(observation, Observation) or observation.error
            or not observation.observation_id or not _observation_complete(observation)):
        return None
    if previous is not None and observation.observation_id == previous.observation_id:
        return None
    return observation


def _map_graph_failure(result: GraphAgentResult) -> PlannerFailureReason | None:
    # Safety, trust, decision, and missing-observation failures remain terminal.
    if result.result.stop_reason == "execution_failed":
        return PlannerFailureReason.STEP_EXECUTION_FAILED
    return None


def _safe_stop_reason(result: GraphAgentResult) -> PlannerStopReason:
    allowed = {
        "safety_rejected", "confirmation_required", "decision_error",
        "execution_failed", "observation_failed", "max_steps", "low_confidence",
        "repeated_action", "interrupted", "needs_human",
    }
    reason = result.result.stop_reason
    return reason if reason in allowed else "decision_error"  # type: ignore[return-value]
