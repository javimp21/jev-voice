"""Local FastAPI wrapper around the experimental LangGraph agent runtime."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import os
from typing import Annotated, Literal, Protocol

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent.graph import GraphAgent, GraphAgentResult, GraphNodeDiagnostic
from agent.loop import AgentLimits
from computer.windows import ObservationOptions
from computer.windows_actions import WindowsComputer
from computer.windows_apps import WindowsApplicationCatalog
from computer.windows_capture import WindowsWindowCapture
from computer.visual_providers import (
    VisualProviderConfigurationError, visual_provider_from_environment,
)
from decision.client import ConfigurationError
from decision.jev import JevDecisionMaker
from safety.policy import AutonomousActionPolicy


MAX_API_STEPS = 20
MAX_API_REPLANS = 10
MAX_RETURNED_DIAGNOSTICS = 200

StopReasonValue = Literal[
    "finished", "needs_human", "low_confidence", "decision_error", "safety_rejected",
    "confirmation_required", "execution_failed", "observation_failed", "max_steps",
    "repeated_action", "interrupted", "invalid_request", "configuration_error",
    "runtime_error",
]
GraphNodeValue = Literal[
    "OBSERVE", "DECIDE", "SAFETY_CHECK", "EXECUTE", "REOBSERVE",
    "VERIFY_OR_CONTINUE", "REPLAN", "FINISH", "FAIL", "UNKNOWN",
]
GraphActionKind = Literal[
    "click", "visual_click", "type", "open_app", "press_key", "query_submit", "finish",
]

_STOP_REASONS = frozenset(StopReasonValue.__args__)
_GRAPH_NODES = frozenset({
    "OBSERVE", "DECIDE", "SAFETY_CHECK", "EXECUTE", "REOBSERVE",
    "VERIFY_OR_CONTINUE", "REPLAN", "FINISH", "FAIL",
})
_ACTION_KINDS = frozenset(GraphActionKind.__args__)
_TRANSITION_REASONS = frozenset({
    "run_started", "observation_complete", "observation_exception", "invalid_observation",
    "observation_reported_error", "missing_observation", "decision_exception",
    "decision_retry_exception", "invalid_decision_result", "invalid_decision_status",
    "decision_error", "decision_needs_human", "missing_or_invalid_action",
    "confidence_below_threshold", "decision_ready", "decision_after_replan",
    "dry_run_proposal", "invalid_safety_input", "policy_exception", "invalid_policy_result",
    "policy_denied", "confirmation_unavailable", "confirmation_exception",
    "confirmation_declined", "repeated_action", "safety_allowed", "finish_action",
    "invalid_execution_input", "execution_exception", "invalid_action_result", "action_failed",
    "action_executed", "settle_wait_failed", "post_action_observation_exception",
    "post_action_observation_complete", "post_action_observation_incomplete",
    "invalid_action_snapshot_binding", "fresh_snapshot_missing", "missing_fresh_observation",
    "max_steps_reached", "recoverable_post_action_observation", "replan_requested",
    "continue_after_fresh_observation", "replan_precondition_failed", "replan_budget_exhausted",
    "replan_to_decide", "replan_started", "invalid_finish_action", "task_finished",
    "graph_execution_failed", "recursion_limit", "interrupted", "invalid_request",
    "other", "failed", "finished", "unknown",
})
_REPLAN_REASONS = frozenset({"post_action_observation_incomplete"})


class AgentRunRequest(BaseModel):
    """Bounded user input for one graph run."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    command: Annotated[str, Field(min_length=1, max_length=2000)]
    dry_run: bool = True
    max_steps: Annotated[int | None, Field(strict=True, ge=1, le=MAX_API_STEPS)] = None
    max_replans: Annotated[int | None, Field(strict=True, ge=0, le=MAX_API_REPLANS)] = None

    @field_validator("command")
    @classmethod
    def command_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("command must not be blank")
        return value.strip()


class GraphDiagnosticResponse(BaseModel):
    """Allowlisted graph metadata; never includes observation contents."""

    node: GraphNodeValue
    step: Annotated[int, Field(ge=0, le=MAX_API_STEPS)]
    transition_reason: str = Field(max_length=48)
    observation_id_present: bool
    action_kind: GraphActionKind | None = None
    success: bool | None = None
    stop_reason: StopReasonValue | None = None
    replan_count: Annotated[int, Field(ge=0, le=MAX_API_REPLANS)]
    max_replans: Annotated[int, Field(ge=0, le=MAX_API_REPLANS)]
    replan_reason: str | None = Field(default=None, max_length=48)
    failure_recoverable: bool | None = None


class AgentRunResponse(BaseModel):
    """Safe summary of a graph run."""

    success: bool
    stop_reason: StopReasonValue
    message: str = Field(max_length=80)
    steps: Annotated[int, Field(ge=0, le=MAX_API_STEPS)]
    replan_count: Annotated[int, Field(ge=0, le=MAX_API_REPLANS)]
    max_replans: Annotated[int, Field(ge=0, le=MAX_API_REPLANS)]
    last_replan_reason: str | None = Field(default=None, max_length=48)
    graph_diagnostics: list[GraphDiagnosticResponse] = Field(max_length=MAX_RETURNED_DIAGNOSTICS)


class HealthResponse(BaseModel):
    status: Literal["ok"]


class GraphRuntime(Protocol):
    def run(self, request: str, *, dry_run: bool = False) -> GraphAgentResult: ...


RuntimeFactory = Callable[[AgentRunRequest], GraphRuntime]


def _load_project_environment() -> None:
    load_dotenv(
        dotenv_path=Path(__file__).resolve().parent / ".env",
        override=False,
        verbose=False,
    )


def _configured_max_steps() -> int:
    raw = os.environ.get("AGENT_MAX_STEPS", "8")
    try:
        value = int(raw)
    except ValueError:
        raise ConfigurationError("Agent step configuration is invalid.") from None
    if not 1 <= value <= MAX_API_STEPS:
        raise ConfigurationError("Agent step configuration is outside the service limit.")
    return value


def build_graph_agent(request: AgentRunRequest) -> GraphAgent:
    """Construct existing dependencies; all orchestration stays in GraphAgent."""

    _load_project_environment()
    catalog = WindowsApplicationCatalog()
    visual_provider = visual_provider_from_environment()
    decision_maker = JevDecisionMaker.from_environment(catalog)
    computer = WindowsComputer(
        ObservationOptions(),
        app_catalog=catalog,
        capture_service=WindowsWindowCapture() if visual_provider is not None else None,
        visual_provider=visual_provider,
    )
    limits = AgentLimits(
        max_steps=request.max_steps if request.max_steps is not None else _configured_max_steps(),
        confidence_threshold=float(getattr(decision_maker, "min_confidence", 0.8)),
    )
    return GraphAgent(
        computer,
        decision_maker,
        policy=AutonomousActionPolicy(catalog),
        limits=limits,
        max_replans=request.max_replans if request.max_replans is not None else 1,
    )


def _bounded_int(value: object, *, default: int, maximum: int) -> int:
    if type(value) is not int:
        return default
    return min(max(value, 0), maximum)


def _safe_stop_reason(value: object, *, default: StopReasonValue = "runtime_error") -> StopReasonValue:
    return value if isinstance(value, str) and value in _STOP_REASONS else default  # type: ignore[return-value]


def _safe_diagnostic(item: GraphNodeDiagnostic) -> GraphDiagnosticResponse:
    node = item.node if item.node in _GRAPH_NODES else "UNKNOWN"
    transition_reason = (
        item.transition_reason if item.transition_reason in _TRANSITION_REASONS else "other"
    )
    action_kind = item.action_kind if item.action_kind in _ACTION_KINDS else None
    stop_reason = (
        _safe_stop_reason(item.stop_reason) if item.stop_reason is not None else None
    )
    replan_reason = item.replan_reason if item.replan_reason in _REPLAN_REASONS else None
    return GraphDiagnosticResponse(
        node=node,
        step=_bounded_int(item.step, default=0, maximum=MAX_API_STEPS),
        transition_reason=transition_reason,
        observation_id_present=bool(item.observation_id),
        action_kind=action_kind,
        success=item.success if type(item.success) is bool else None,
        stop_reason=stop_reason,
        replan_count=_bounded_int(item.replan_count, default=0, maximum=MAX_API_REPLANS),
        max_replans=_bounded_int(item.max_replans, default=1, maximum=MAX_API_REPLANS),
        replan_reason=replan_reason,
        failure_recoverable=(
            item.failure_recoverable if type(item.failure_recoverable) is bool else None
        ),
    )


def _map_result(result: GraphAgentResult) -> AgentRunResponse:
    graph_result = result.result
    stop_reason = _safe_stop_reason(graph_result.stop_reason)
    success = graph_result.success if type(graph_result.success) is bool else False
    return AgentRunResponse(
        success=success,
        stop_reason=stop_reason,
        message="Agent run completed." if success else "Agent run stopped safely.",
        steps=_bounded_int(graph_result.steps, default=0, maximum=MAX_API_STEPS),
        replan_count=_bounded_int(result.replan_count, default=0, maximum=MAX_API_REPLANS),
        max_replans=_bounded_int(result.max_replans, default=1, maximum=MAX_API_REPLANS),
        last_replan_reason=(
            result.last_replan_reason
            if result.last_replan_reason in _REPLAN_REASONS else None
        ),
        graph_diagnostics=[
            _safe_diagnostic(item)
            for item in result.diagnostics[-MAX_RETURNED_DIAGNOSTICS:]
            if isinstance(item, GraphNodeDiagnostic)
        ],
    )


def _failure_response(
    stop_reason: Literal["configuration_error", "runtime_error"],
    *,
    max_replans: int = 1,
) -> AgentRunResponse:
    message = (
        "Agent configuration is unavailable."
        if stop_reason == "configuration_error"
        else "Agent runtime failed safely."
    )
    return AgentRunResponse(
        success=False,
        stop_reason=stop_reason,
        message=message,
        steps=0,
        replan_count=0,
        max_replans=_bounded_int(max_replans, default=1, maximum=MAX_API_REPLANS),
        graph_diagnostics=[],
    )


def create_app(runtime_factory: RuntimeFactory | None = None) -> FastAPI:
    factory = runtime_factory or build_graph_agent
    app = FastAPI(title="voice-jev local agent service", version="0.1.0")

    @app.exception_handler(RequestValidationError)
    async def safe_validation_error(
        request: Request, exc: RequestValidationError,
    ) -> JSONResponse:
        # FastAPI normally echoes invalid input values; omit them so request
        # text and accidental secrets never appear in validation responses.
        field_names = {"command", "dry_run", "max_steps", "max_replans"}
        message_by_type = {
            "missing": "Field is required.",
            "string_too_short": "Command must not be empty.",
            "string_too_long": "Command exceeds the service limit.",
            "value_error": "Request value is invalid.",
            "less_than_equal": "Value exceeds the service limit.",
            "greater_than_equal": "Value is below the service limit.",
            "int_type": "Value must be an integer.",
            "bool_type": "Value must be a boolean.",
            "extra_forbidden": "Unknown request field.",
            "json_invalid": "Request body is not valid JSON.",
        }
        errors = []
        for error in exc.errors():
            error_type = error.get("type")
            safe_type = error_type if error_type in message_by_type else "value_error"
            location = [
                part if part == "body" or part in field_names or type(part) is int else "body"
                for part in error.get("loc", ())
            ]
            errors.append({
                "loc": location,
                "msg": message_by_type[safe_type],
                "type": safe_type,
            })
        return JSONResponse(status_code=422, content={"detail": errors})

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.post("/agent/run", response_model=AgentRunResponse)
    def run_agent(body: AgentRunRequest) -> AgentRunResponse:
        configured_replans = body.max_replans if body.max_replans is not None else 1
        try:
            runtime = factory(body)
        except (ConfigurationError, VisualProviderConfigurationError):
            return _failure_response("configuration_error", max_replans=configured_replans)
        except Exception:
            return _failure_response("runtime_error", max_replans=configured_replans)
        try:
            result = runtime.run(body.command, dry_run=body.dry_run)
            return _map_result(result)
        except Exception:
            return _failure_response("runtime_error", max_replans=configured_replans)

    return app


app = create_app()
