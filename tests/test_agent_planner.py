"""Offline tests for typed planning, bounded graph delegation, and telemetry."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from agent.loop import AgentLimits
from agent.planner import (
    OpenAIPlanner, Plan, PlanStep, PlanStepKind, PlanStepStatus, PlanStatus,
    PlannerCallError, PlannerConfigurationError, PlannerContext,
    PlannerCurrentState, PlannerFailureReason, PlanUpdate, PlanValidationError,
    ProviderPlan, ProviderPlanStep, ProviderPlanUpdate, validate_plan,
    validate_plan_update,
)
from agent.planner_agent import PlannerAgent, PlannerAgentResult
from agent.telemetry import InMemoryTelemetryCollector
from computer.actions import FinishAction, OpenAppAction, PressKeyAction, TypeAction
from computer.applications import ApplicationCandidate, MemoryApplicationCatalog
from computer.models import Observation
from computer.results import ActionResult
from decision.models import DecisionResult
from safety.interfaces import SafetyDecision


def _step(
    step_id: str, kind: PlanStepKind, *, target: str | None = None,
    payload: str | None = None,
) -> PlanStep:
    return PlanStep(
        step_id=step_id, kind=kind, target=target, payload=payload,
        status=PlanStepStatus.PENDING, attempts=0, last_failure_reason=None,
    )


def _plan(task: str, steps: list[PlanStep], *, max_steps: int = 8) -> Plan:
    return Plan(
        objective=task[:100], steps=steps, max_steps=max_steps,
        status=PlanStatus.PLANNED,
    )


def _open_finish(task: str, target: str = "Notepad") -> Plan:
    return _plan(task, [
        _step("step_1", PlanStepKind.OPEN_APP, target=target),
        _step("step_2", PlanStepKind.FINISH),
    ])


@dataclass
class FakeComputer:
    observations: list[Observation | Exception]
    execution_successes: list[bool] | None = None

    def __post_init__(self) -> None:
        self.observed: list[Observation] = []
        self.executed: list[tuple[object, Observation | None]] = []

    def observe(self) -> Observation:
        item = self.observations.pop(0)
        if isinstance(item, Exception):
            raise item
        self.observed.append(item)
        return item

    def execute(self, action, observation=None) -> ActionResult:
        self.executed.append((action, observation))
        succeeded = True
        if self.execution_successes:
            succeeded = self.execution_successes.pop(0)
        return ActionResult(
            succeeded, action, "ok" if succeeded else "failed",
            error=None if succeeded else "windows_operation_failed",
        )


class FakePlanner:
    def __init__(self, plan_result, updates=()) -> None:
        self.plan_result = plan_result
        self.updates = list(updates)
        self.plan_calls: list[tuple[str, PlannerContext]] = []
        self.replan_calls: list[tuple[Plan, PlannerCurrentState, PlannerFailureReason]] = []

    def plan(self, task: str, context: PlannerContext):
        self.plan_calls.append((task, context))
        if isinstance(self.plan_result, Exception):
            raise self.plan_result
        return self.plan_result

    def replan(self, plan: Plan, current_state: PlannerCurrentState, failure: PlannerFailureReason):
        self.replan_calls.append((plan, current_state, failure))
        update = self.updates.pop(0)
        if isinstance(update, Exception):
            raise update
        return update


class FakeDecisionMaker:
    def __init__(self, actions) -> None:
        self.actions = list(actions)
        self.requests: list[str] = []

    def decide(self, request, observation, history=()):
        self.requests.append(request)
        action = self.actions.pop(0)
        return DecisionResult("ready", action, 0.95, "selected")


class AllowPolicy:
    def validate(self, action, observation):
        return SafetyDecision("allow", "fake allow")


class DenyPolicy:
    def validate(self, action, observation):
        return SafetyDecision("deny", "fake rejection")


class AllowThenDenyPolicy:
    def __init__(self) -> None:
        self.calls = 0

    def validate(self, action, observation):
        self.calls += 1
        return SafetyDecision("allow" if self.calls == 1 else "deny", "scripted")


class Confirm:
    def confirm(self, action, reason):
        return True


def _catalog() -> MemoryApplicationCatalog:
    return MemoryApplicationCatalog([
        ApplicationCandidate("notepad", "Notepad", "test", launch_policy="allow"),
        ApplicationCandidate("calculator", "Calculator", "test", launch_policy="allow"),
    ])


def _observations(count: int) -> list[Observation]:
    return [Observation("test.exe", "Test", observation_id=f"o-{i}") for i in range(count)]


def _agent(planner, computer, decision_maker, **kwargs) -> PlannerAgent:
    kwargs.setdefault("telemetry_collector", InMemoryTelemetryCollector())
    return PlannerAgent(
        planner, computer, decision_maker, policy=AllowPolicy(),
        confirmation=Confirm(), limits=AgentLimits(max_steps=8, settle_open_seconds=0,
                                                    settle_action_seconds=0),
        sleep_fn=lambda _seconds: None, app_catalog=_catalog(), **kwargs,
    )


def test_two_step_plan_completes_through_langgraph() -> None:
    task = "Open Notepad and type hello"
    plan = _plan(task, [
        _step("step_1", PlanStepKind.OPEN_APP, target="Notepad"),
        _step("step_2", PlanStepKind.TYPE_TEXT, payload="hello"),
        _step("step_3", PlanStepKind.FINISH),
    ])
    computer = FakeComputer(_observations(5))
    decisions = FakeDecisionMaker([
        OpenAppAction("notepad"), FinishAction("done"),
        TypeAction("hello"), FinishAction("done"),
    ])

    result = _agent(FakePlanner(plan), computer, decisions).run(task)

    assert result.success and result.stop_reason == "finished"
    assert result.steps_completed == 3
    assert [item[0] for item in computer.executed] == [OpenAppAction("notepad"), TypeAction("hello")]
    assert result.planner_replan_count == 0


def test_three_step_plan_completes_with_closed_step_kinds() -> None:
    task = "Open Notepad, type hello, then press ctrl+a"
    plan = _plan(task, [
        _step("step_1", PlanStepKind.OPEN_APP, target="Notepad"),
        _step("step_2", PlanStepKind.TYPE_TEXT, payload="hello"),
        _step("step_3", PlanStepKind.PRESS_KEY, target="ctrl+a"),
        _step("step_4", PlanStepKind.FINISH),
    ])
    computer = FakeComputer(_observations(7))
    decisions = FakeDecisionMaker([
        OpenAppAction("notepad"), FinishAction("done"),
        TypeAction("hello"), FinishAction("done"),
        PressKeyAction(("ctrl", "a")), FinishAction("done"),
    ])

    result = _agent(FakePlanner(plan), computer, decisions).run(task)

    assert result.success
    assert result.steps_completed == 4
    assert len(computer.executed) == 3


@pytest.mark.parametrize("candidate", [
    {"objective": "bad", "steps": [], "max_steps": 8, "status": "planned"},
    {"objective": "unsafe", "steps": [
        {"step_id": "step_1", "kind": "RUN_SHELL", "target": "cmd", "payload": None,
         "status": "pending", "attempts": 0, "last_failure_reason": None},
    ], "max_steps": 8, "status": "planned"},
])
def test_invalid_or_unsupported_plan_fails_before_observing_or_acting(candidate) -> None:
    task = "Open Notepad"
    computer = FakeComputer([])
    result = _agent(FakePlanner(candidate), computer, FakeDecisionMaker([])).run(task)

    assert not result.success
    assert result.stop_reason == "plan_validation_failed"
    assert computer.observed == []
    assert computer.executed == []


def test_step_execution_failure_replans_only_after_fresh_observation() -> None:
    task = "Open Notepad; if unavailable, open Calculator"
    first = _open_finish(task)
    replacement = PlanUpdate(steps=[
        _step("step_3", PlanStepKind.OPEN_APP, target="Calculator"),
        _step("step_4", PlanStepKind.FINISH),
    ])
    computer = FakeComputer(_observations(5), [False, True])
    decisions = FakeDecisionMaker([
        OpenAppAction("notepad"), OpenAppAction("calculator"), FinishAction("done"),
    ])

    result = _agent(FakePlanner(first, [replacement]), computer, decisions).run(task)

    assert result.success
    assert result.planner_replan_count == 1
    assert len(computer.observed) == 5
    assert len(computer.executed) == 2
    assert result.plan is not None
    assert result.plan.steps[0].status is PlanStepStatus.FAILED
    assert result.plan.steps[2].status is PlanStepStatus.COMPLETE


def test_planner_replan_budget_exhaustion_stops() -> None:
    task = "Open Notepad; if unavailable, open Calculator"
    replacement = PlanUpdate(steps=[
        _step("step_3", PlanStepKind.OPEN_APP, target="Calculator"),
        _step("step_4", PlanStepKind.FINISH),
    ])
    computer = FakeComputer(_observations(3), [False, False])
    decisions = FakeDecisionMaker([OpenAppAction("notepad"), OpenAppAction("calculator")])

    result = _agent(
        FakePlanner(_open_finish(task), [replacement]), computer, decisions,
        max_planner_replans=1,
    ).run(task)

    assert not result.success
    assert result.stop_reason == "planner_replan_budget_exhausted"
    assert result.planner_replan_count == 1
    assert len(computer.executed) == 2
    assert len(decisions.actions) == 0


def test_terminal_safety_rejection_does_not_trigger_planner_replan() -> None:
    task = "Open Notepad; if unavailable, open Calculator"
    planner = FakePlanner(_open_finish(task))
    computer = FakeComputer(_observations(1))
    result = PlannerAgent(
        planner, computer, FakeDecisionMaker([OpenAppAction("notepad")]),
        policy=DenyPolicy(), limits=AgentLimits(max_steps=8), app_catalog=_catalog(),
        telemetry_collector=InMemoryTelemetryCollector(),
    ).run(task)
    assert result.stop_reason == "safety_rejected"
    assert planner.replan_calls == []
    assert computer.executed == []


def test_planner_replan_requires_complete_fresh_observation() -> None:
    task = "Open Notepad; if unavailable, open Calculator"
    planner = FakePlanner(_open_finish(task), [PlanUpdate(steps=[
        _step("step_3", PlanStepKind.OPEN_APP, target="Calculator"),
        _step("step_4", PlanStepKind.FINISH),
    ])])
    computer = FakeComputer([_observations(1)[0], RuntimeError("no fresh state")], [False])
    result = _agent(
        planner, computer, FakeDecisionMaker([OpenAppAction("notepad")]),
    ).run(task)
    assert result.stop_reason == "observation_failed"
    assert result.planner_replan_count == 0
    assert planner.replan_calls == []


def test_replanned_action_still_passes_the_existing_safety_policy() -> None:
    task = "Open Notepad; if unavailable, open Calculator"
    planner = FakePlanner(_open_finish(task), [PlanUpdate(steps=[
        _step("step_3", PlanStepKind.OPEN_APP, target="Calculator"),
        _step("step_4", PlanStepKind.FINISH),
    ])])
    computer = FakeComputer(_observations(4), [False])
    policy = AllowThenDenyPolicy()
    result = PlannerAgent(
        planner, computer,
        FakeDecisionMaker([OpenAppAction("notepad"), OpenAppAction("calculator")]),
        policy=policy, limits=AgentLimits(max_steps=8), app_catalog=_catalog(),
        telemetry_collector=InMemoryTelemetryCollector(),
    ).run(task)
    assert result.stop_reason == "safety_rejected"
    assert result.planner_replan_count == 1
    assert policy.calls == 2
    assert len(computer.executed) == 1


def test_planner_provider_invalid_structured_response_fails_closed() -> None:
    task = "Open Notepad"
    result = _agent(
        FakePlanner(PlannerCallError("invalid_structured_response")),
        FakeComputer([]), FakeDecisionMaker([]),
    ).run(task)
    assert result.stop_reason == "planner_error"
    assert result.error_category == "structured_output_error"


def test_explicit_finish_step_uses_observation_and_never_executes() -> None:
    task = "Finish this task"
    plan = _plan(task, [_step("step_1", PlanStepKind.FINISH)])
    computer = FakeComputer(_observations(1))
    result = _agent(FakePlanner(plan), computer, FakeDecisionMaker([])).run(task)
    assert result.success
    assert computer.observed
    assert computer.executed == []


def test_early_finish_cannot_skip_a_non_finish_plan_step() -> None:
    task = "Open Notepad"
    result = _agent(
        FakePlanner(_open_finish(task)), FakeComputer(_observations(1)),
        FakeDecisionMaker([FinishAction("skip")]),
    ).run(task)
    assert not result.success
    assert result.stop_reason == "decision_error"


def test_existing_local_graph_replan_is_separate_from_planner_replan() -> None:
    task = "Type hello"
    plan = _plan(task, [
        _step("step_1", PlanStepKind.TYPE_TEXT, payload="hello"),
        _step("step_2", PlanStepKind.FINISH),
    ])
    computer = FakeComputer([
        _observations(1)[0],
        Observation("test.exe", "Test", observation_id="incomplete", truncated=True),
        Observation("test.exe", "Test", observation_id="replanned"),
        Observation("test.exe", "Test", observation_id="finish"),
    ])
    decisions = FakeDecisionMaker([
        TypeAction("hello"), PressKeyAction(("tab",)), FinishAction("done"),
    ])
    planner = FakePlanner(plan)

    result = _agent(planner, computer, decisions).run(task)

    assert result.success
    assert result.local_replan_count == 1
    assert result.planner_replan_count == 0
    assert planner.replan_calls == []
    assert len(computer.executed) == 2


def test_incomplete_post_action_observation_cannot_mark_a_plan_step_complete() -> None:
    task = "Type hello"
    plan = _plan(task, [
        _step("step_1", PlanStepKind.TYPE_TEXT, payload="hello"),
        _step("step_2", PlanStepKind.FINISH),
    ])
    computer = FakeComputer([
        Observation("test.exe", "Test", observation_id="before"),
        Observation("test.exe", "Test", observation_id="partial", truncated=True),
    ])
    result = _agent(
        FakePlanner(plan), computer,
        FakeDecisionMaker([TypeAction("hello"), FinishAction("premature")]),
    ).run(task)
    assert not result.success
    assert result.stop_reason == "decision_error"
    assert result.local_replan_count == 1
    assert result.planner_replan_count == 0
    assert len(computer.executed) == 1


def test_planner_telemetry_is_typed_and_excludes_task_and_literals() -> None:
    task = "Type the secret phrase sunset purple"
    plan = _plan(task, [
        _step("step_1", PlanStepKind.TYPE_TEXT, payload="sunset purple"),
        _step("step_2", PlanStepKind.FINISH),
    ])
    collector = InMemoryTelemetryCollector()
    result = _agent(
        FakePlanner(plan), FakeComputer([]), FakeDecisionMaker([]),
        telemetry_collector=collector,
    ).run(task, dry_run=True)

    records = [str(event) for event in collector.events]
    assert result.success and result.stop_reason == "dry_run"
    assert any(event.event_type == "planner_called" for event in collector.events)
    assert any(event.event_type == "plan_validation" for event in collector.events)
    assert "sunset purple" not in " ".join(records)
    assert task not in " ".join(records)
    assert all(event.planner_run_id for event in collector.events)


def test_plan_validation_rejects_shell_paths_duplicate_patterns_and_untrusted_text() -> None:
    task = "Open Notepad"
    unsafe = _plan(task, [
        _step("step_1", PlanStepKind.OPEN_APP, target="C:\\Windows\\notepad.exe"),
        _step("step_2", PlanStepKind.FINISH),
    ])
    with pytest.raises(PlanValidationError):
        validate_plan(unsafe, task, max_steps=8)

    untrusted = _plan(task, [
        _step("step_1", PlanStepKind.TYPE_TEXT, payload="invented text"),
        _step("step_2", PlanStepKind.FINISH),
    ])
    with pytest.raises(PlanValidationError):
        validate_plan(untrusted, task, max_steps=8)

    case_changed = _plan("Type Straße", [
        _step("step_1", PlanStepKind.TYPE_TEXT, payload="STRASSE"),
        _step("step_2", PlanStepKind.FINISH),
    ])
    with pytest.raises(PlanValidationError):
        validate_plan(case_changed, "Type Straße", max_steps=8)


def test_plan_validation_diagnostic_identifies_rule_without_rejected_content() -> None:
    task = "Type the private task marker TASK_SECRET_9841"
    rejected_literal = "REJECTED_LITERAL_7285"
    candidate = _plan(task, [
        _step("step_1", PlanStepKind.TYPE_TEXT, payload=rejected_literal),
        _step("step_2", PlanStepKind.FINISH),
    ])

    with pytest.raises(PlanValidationError) as caught:
        validate_plan(candidate, task, max_steps=8)

    diagnostic = caught.value.diagnostic.as_dict()
    assert diagnostic == {
        "validation_stage": "steps",
        "validation_code": "payload_not_grounded_in_task",
        "reason_category": "literal_grounding",
        "step_index": 0,
        "step_kind": "TYPE_TEXT",
        "field_name": "payload",
        "field_path": "steps.0.payload",
    }
    rendered = str(caught.value) + str(diagnostic)
    assert task not in rendered
    assert "TASK_SECRET_9841" not in rendered
    assert rejected_literal not in rendered


def test_pydantic_diagnostic_exposes_only_safe_schema_location() -> None:
    task = "Open Notepad and remember TASK_SECRET_517"
    rejected_literal = "MODEL_SECRET_661"
    candidate = {
        "objective": task,
        "steps": [
            {
                "step_id": "step_1", "kind": f"NOT_A_KIND_{rejected_literal}",
                "target": None, "payload": rejected_literal, "status": "pending",
                "attempts": 0, "last_failure_reason": None,
            },
        ],
        "max_steps": 8, "status": "planned",
    }

    with pytest.raises(PlanValidationError) as caught:
        validate_plan(candidate, task, max_steps=8)

    diagnostic = caught.value.diagnostic.as_dict()
    assert diagnostic["validation_stage"] == "schema"
    assert diagnostic["validation_code"] == "invalid_step_kind"
    assert diagnostic["field_name"] == "kind"
    assert diagnostic["field_path"] == "steps.0.kind"
    rendered = str(caught.value) + str(diagnostic)
    assert task not in rendered
    assert "TASK_SECRET_517" not in rendered
    assert rejected_literal not in rendered
    assert f"NOT_A_KIND_{rejected_literal}" not in rendered


class FakeResponses:
    def __init__(self, parsed=None, error: Exception | None = None) -> None:
        self.parsed = parsed
        self.error = error
        self.calls: list[dict[str, object]] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(output_parsed=self.parsed)


def test_openai_adapter_uses_responses_native_typed_parse_and_sanitizes_errors() -> None:
    task = "Open Notepad"
    plan = _open_finish(task)
    provider_plan = ProviderPlan(objective=plan.objective, steps=[
        ProviderPlanStep(kind=PlanStepKind.OPEN_APP, target="Notepad", payload=None),
        ProviderPlanStep(kind=PlanStepKind.FINISH, target=None, payload=None),
    ])
    responses = FakeResponses(parsed=provider_plan)
    adapter = OpenAIPlanner("test-model", SimpleNamespace(responses=responses))

    assert adapter.plan(task, PlannerContext(8)) == plan
    call = responses.calls[0]
    assert call["model"] == "test-model"
    assert call["text_format"] is ProviderPlan
    assert "step_id" not in ProviderPlanStep.model_fields
    assert call["max_output_tokens"] == 2_000

    secret = "sk-live-secret-must-not-escape"
    failing = OpenAIPlanner(
        "test-model",
        SimpleNamespace(responses=FakeResponses(error=RuntimeError(secret))),
    )
    with pytest.raises(PlannerCallError) as caught:
        failing.plan(task, PlannerContext(8))
    assert caught.value.category == "unknown_provider_error"
    assert secret not in str(caught.value)


def test_planner_model_configuration_defaults_to_luna_and_allows_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import openai

    client_options: list[dict[str, object]] = []

    def fake_openai(**kwargs):
        client_options.append(kwargs)
        return SimpleNamespace(responses=FakeResponses())

    monkeypatch.setattr(openai, "OpenAI", fake_openai)
    monkeypatch.setenv("PLANNER_API_KEY", "test-only-planner-key")
    monkeypatch.setenv("PLANNER_TIMEOUT_SECONDS", "30")
    monkeypatch.delenv("PLANNER_MODEL", raising=False)

    default_adapter = OpenAIPlanner.from_environment()
    assert default_adapter.model == "gpt-6-luna"

    monkeypatch.setenv("PLANNER_MODEL", "gpt-6-astra")
    override_adapter = OpenAIPlanner.from_environment()
    assert override_adapter.model == "gpt-6-astra"
    assert len(client_options) == 2


def test_openai_adapter_classifies_structured_parse_error_without_raw_values() -> None:
    # Construct a Pydantic failure separately so the mocked SDK exercises the
    # same exception shape returned by typed response parsing.
    try:
        ProviderPlan.model_validate({
            "objective": "private objective",
            "steps": [{
                "kind": "INVALID_MODEL_KIND_SECRET",
                "target": None, "payload": "PRIVATE_LITERAL_SECRET",
            }],
        })
    except Exception as exc:
        schema_error = exc
        pass
    else:
        raise AssertionError("test fixture must fail Pydantic validation")
    responses = FakeResponses(error=schema_error)
    adapter = OpenAIPlanner("test-model", SimpleNamespace(responses=responses))

    with pytest.raises(PlannerCallError) as caught:
        adapter.plan("Open Notepad", PlannerContext(8))

    diagnostic = str(caught.value) + str(caught.value.diagnostic.as_dict())
    assert caught.value.category == "structured_output_error"
    assert caught.value.diagnostic.provider_stage == "response_parse"
    assert caught.value.diagnostic.structured_output_error
    assert "INVALID_MODEL_KIND_SECRET" not in diagnostic
    assert "PRIVATE_LITERAL_SECRET" not in diagnostic


def _sdk_status_error(status: int, code: str | None):
    import openai

    response = SimpleNamespace(
        status_code=status,
        request=SimpleNamespace(),
        headers={"x-request-id": "req_safe12345678"},
    )
    error_type = {
        400: openai.BadRequestError,
        401: openai.AuthenticationError,
        403: openai.PermissionDeniedError,
        404: openai.NotFoundError,
        429: openai.RateLimitError,
        500: openai.InternalServerError,
    }.get(status, openai.APIStatusError)
    return error_type(
        "PRIVATE_TASK prompt=TASK_MARKER key=sk-sensitive raw provider body",
        response=response,
        body={
            "code": code,
            "type": "private raw error type",
            "message": "PRIVATE_PROVIDER_MESSAGE TASK_MARKER sk-sensitive",
        },
    )


@pytest.mark.parametrize(
    ("status", "code", "category", "flag"),
    [
        (400, "invalid_request_error", "bad_request", None),
        (401, "invalid_api_key", "authentication_error", "authentication_error"),
        (403, "permission_denied", "authentication_error", "authentication_error"),
        (404, "model_not_found", "model_not_found", "model_not_found"),
        (404, None, "unknown_provider_error", None),
        (429, "rate_limit_exceeded", "rate_limited", "rate_limited"),
        (500, "server_error", "server_error", None),
    ],
)
def test_openai_status_errors_map_to_safe_provider_categories(
    status, code, category, flag,
) -> None:
    error = _sdk_status_error(status, code)
    adapter = OpenAIPlanner(
        "gpt-6-astra", SimpleNamespace(responses=FakeResponses(error=error)),
    )

    with pytest.raises(PlannerCallError) as caught:
        adapter.plan("TASK_MARKER Open Notepad", PlannerContext(8))

    diagnostic = caught.value.diagnostic.as_dict()
    assert caught.value.category == category
    assert diagnostic["category"] == category
    assert diagnostic["http_status"] == status
    assert diagnostic["request_id"] == "req_safe12345678"
    assert diagnostic["model_name"] == "gpt-6-astra"
    assert diagnostic.get("provider_error_code") == code
    if flag is not None:
        assert diagnostic[flag] is True


def test_timeout_and_connection_errors_are_distinguished() -> None:
    import openai

    timeout = openai.APITimeoutError(request=SimpleNamespace())
    connection = openai.APIConnectionError(request=SimpleNamespace())
    categories = []
    for error in (timeout, connection):
        adapter = OpenAIPlanner(
            "gpt-6-astra", SimpleNamespace(responses=FakeResponses(error=error)),
        )
        with pytest.raises(PlannerCallError) as caught:
            adapter.plan("Open Notepad", PlannerContext(8))
        categories.append((caught.value.category, caught.value.diagnostic.as_dict()))

    assert categories[0][0] == "timeout"
    assert categories[0][1]["timeout"] is True
    assert categories[0][1]["connection_error"] is False
    assert categories[1][0] == "connection_error"
    assert categories[1][1]["connection_error"] is True


def test_api_schema_failure_and_invalid_structured_response_are_identified() -> None:
    import openai

    response = SimpleNamespace(
        status_code=400, request=SimpleNamespace(),
        headers={"x-request-id": "req_schema123456"},
    )
    schema_error = openai.BadRequestError(
        "raw schema error with TASK_SECRET and sk-secret",
        response=response,
        body={"code": "invalid_json_schema", "message": "raw response body"},
    )
    adapter = OpenAIPlanner(
        "gpt-6-astra", SimpleNamespace(responses=FakeResponses(error=schema_error)),
    )
    with pytest.raises(PlannerCallError) as caught_schema:
        adapter.plan("TASK_SECRET Open Notepad", PlannerContext(8))
    assert caught_schema.value.category == "structured_output_error"
    assert caught_schema.value.diagnostic.structured_output_error
    assert caught_schema.value.diagnostic.provider_error_code == "invalid_json_schema"

    response_error = openai.APIResponseValidationError(
        response=SimpleNamespace(
            status_code=200, request=SimpleNamespace(),
            headers={"x-request-id": "req_parse123456"},
        ),
        body={"message": "TASK_SECRET sk-secret raw payload"},
    )
    adapter = OpenAIPlanner(
        "gpt-6-astra", SimpleNamespace(responses=FakeResponses(error=response_error)),
    )
    with pytest.raises(PlannerCallError) as caught_parse:
        adapter.plan("TASK_SECRET Open Notepad", PlannerContext(8))
    assert caught_parse.value.category == "structured_output_error"
    assert caught_parse.value.diagnostic.provider_error_code == "api_response_validation_error"
    assert caught_parse.value.diagnostic.request_id == "req_parse123456"

    missing = OpenAIPlanner(
        "gpt-6-astra", SimpleNamespace(responses=FakeResponses(parsed=None)),
    )
    with pytest.raises(PlannerCallError) as caught_missing:
        missing.plan("Open Notepad", PlannerContext(8))
    assert caught_missing.value.category == "structured_output_error"
    assert caught_missing.value.diagnostic.provider_error_code == "missing_structured_output"


def test_provider_diagnostics_never_expose_task_key_or_raw_response() -> None:
    task = "TASK_SECRET_7932 Open Notepad"
    key = "sk-test-secret-92831"
    error = _sdk_status_error(400, "invalid_request_error")
    error.message = f"{task} {key} RAW_PROVIDER_RESPONSE"
    error.body = {"message": f"{task} {key} RAW_PROVIDER_RESPONSE"}
    adapter = OpenAIPlanner(
        "gpt-6-astra", SimpleNamespace(responses=FakeResponses(error=error)),
    )

    with pytest.raises(PlannerCallError) as caught:
        adapter.plan(task, PlannerContext(8))

    serialized = str(caught.value) + str(caught.value.diagnostic.as_dict())
    assert task not in serialized
    assert key not in serialized
    assert "RAW_PROVIDER_RESPONSE" not in serialized
    assert "PRIVATE_PROVIDER_MESSAGE" not in serialized


def test_provider_model_name_is_omitted_when_it_does_not_match_safe_model_shape() -> None:
    error = _sdk_status_error(400, "invalid_request_error")
    adapter = OpenAIPlanner(
        "sk-sensitive-model-field", SimpleNamespace(responses=FakeResponses(error=error)),
    )

    with pytest.raises(PlannerCallError) as caught:
        adapter.plan("Open Notepad", PlannerContext(8))

    assert "model_name" not in caught.value.diagnostic.as_dict()


def test_planner_cli_emits_only_safe_provider_diagnostic(monkeypatch, capsys) -> None:
    import json
    import main as cli
    import agent.planner as planner_module

    task = "TASK_PRIVATE_314 Open Notepad"
    key = "sk-test-private-156"
    error = _sdk_status_error(401, "invalid_api_key")
    error.message = f"{task} {key} RAW_BODY"
    error.body = {"message": f"{task} {key} RAW_BODY"}
    adapter = OpenAIPlanner(
        "gpt-6-astra", SimpleNamespace(responses=FakeResponses(error=error)),
    )
    monkeypatch.setattr(
        planner_module.OpenAIPlanner, "from_environment",
        classmethod(lambda _cls: adapter),
    )
    monkeypatch.setattr(cli, "JsonlTelemetrySink", lambda: InMemoryTelemetryCollector())

    assert cli.main(["run-agent-planner-debug", task, "--dry-run"]) == 1
    output = capsys.readouterr().err
    payload = json.loads(output)
    assert payload["stop_reason"] == "planner_error"
    assert payload["error_category"] == "authentication_error"
    assert payload["provider_diagnostic"]["http_status"] == 401
    assert payload["provider_diagnostic"]["authentication_error"] is True
    assert payload["provider_diagnostic"]["request_id"] == "req_safe12345678"
    assert task not in output
    assert key not in output
    assert "RAW_BODY" not in output


def test_planner_provider_check_calls_only_typed_adapter_and_emits_minimal_success(
    monkeypatch, capsys,
) -> None:
    import json
    import main as cli
    import agent.planner as planner_module

    provider_plan = ProviderPlan(
        objective="finish",
        steps=[ProviderPlanStep(kind=PlanStepKind.FINISH, target=None, payload=None)],
    )
    responses = FakeResponses(parsed=provider_plan)
    adapter = OpenAIPlanner("gpt-6-astra", SimpleNamespace(responses=responses))
    monkeypatch.setattr(
        planner_module.OpenAIPlanner, "from_environment",
        classmethod(lambda _cls: adapter),
    )
    monkeypatch.setattr(
        cli, "WindowsComputer",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Windows must not be used")),
    )
    monkeypatch.setattr(
        cli, "JevDecisionMaker",
        SimpleNamespace(from_environment=lambda: (_ for _ in ()).throw(
            AssertionError("Jev must not be used")
        )),
    )

    assert cli.main(["planner-provider-check"]) == 0
    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload == {
        "success": True,
        "model_name": "gpt-6-astra",
        "parsed_schema": True,
    }
    assert len(responses.calls) == 1
    assert responses.calls[0]["model"] == "gpt-6-astra"
    assert responses.calls[0]["text_format"] is ProviderPlan
    assert responses.calls[0]["max_output_tokens"] == 2_000
    assert responses.calls[0]["input"]
    assert all("Open Notepad" not in str(item) for item in responses.calls[0]["input"])


def test_planner_provider_check_sanitizes_failure_and_request_id(monkeypatch, capsys) -> None:
    import json
    import main as cli
    import agent.planner as planner_module

    private_prompt = "PRIVATE_DIAGNOSTIC_PROMPT_382"
    private_key = "sk-private-diagnostic-912"
    error = _sdk_status_error(401, "invalid_api_key")
    error.message = f"{private_prompt} {private_key} PRIVATE_RAW_BODY"
    error.body = {"message": f"{private_prompt} {private_key} PRIVATE_RAW_BODY"}
    adapter = OpenAIPlanner(
        "gpt-6-astra", SimpleNamespace(responses=FakeResponses(error=error)),
    )
    monkeypatch.setattr(
        planner_module.OpenAIPlanner, "from_environment",
        classmethod(lambda _cls: adapter),
    )

    assert cli.main(["planner-provider-check"]) == 1
    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload == {
        "success": False,
        "model_name": "gpt-6-astra",
        "provider_stage": "request",
        "error_category": "authentication_error",
        "structured_output_error": False,
        "http_status": 401,
        "provider_error_code": "invalid_api_key",
        "request_id": "req_safe12345678",
    }
    for private_value in (private_prompt, private_key, "PRIVATE_RAW_BODY"):
        assert private_value not in output


def test_planner_provider_check_reports_missing_configuration_safely(
    monkeypatch, capsys,
) -> None:
    import json
    import main as cli
    import agent.planner as planner_module

    monkeypatch.setattr(
        planner_module.OpenAIPlanner, "from_environment",
        classmethod(lambda _cls: (_ for _ in ()).throw(
            PlannerConfigurationError("configuration contains private details")
        )),
    )

    assert cli.main(["planner-provider-check"]) == 1
    output = capsys.readouterr().out
    assert json.loads(output) == {
        "success": False,
        "provider_stage": "configuration",
        "error_category": "configuration_error",
        "structured_output_error": False,
    }
    assert "private details" not in output


def test_planner_provider_check_sanitizes_client_initialization_error(
    monkeypatch, capsys,
) -> None:
    import json
    import main as cli
    import agent.planner as planner_module

    monkeypatch.setattr(
        planner_module.OpenAIPlanner, "from_environment",
        classmethod(lambda _cls: (_ for _ in ()).throw(
            RuntimeError("PRIVATE_API_KEY_AND_LOCAL_CONFIGURATION")
        )),
    )

    assert cli.main(["planner-provider-check"]) == 1
    output = capsys.readouterr().out
    assert json.loads(output) == {
        "success": False,
        "provider_stage": "client_initialization",
        "error_category": "unknown_provider_error",
        "structured_output_error": False,
    }
    assert "PRIVATE_API_KEY_AND_LOCAL_CONFIGURATION" not in output


def test_planner_cli_reports_safe_validation_rule_without_task_or_payload(
    monkeypatch, capsys,
) -> None:
    import json
    import main as cli
    import agent.planner as planner_module

    task = "Type TASK_PRIVATE_115"
    rejected_literal = "REJECTED_PRIVATE_991"
    candidate = _plan(task, [
        _step("step_1", PlanStepKind.TYPE_TEXT, payload=rejected_literal),
        _step("step_2", PlanStepKind.FINISH),
    ])
    monkeypatch.setattr(
        planner_module.OpenAIPlanner, "from_environment",
        classmethod(lambda _cls: FakePlanner(candidate)),
    )
    monkeypatch.setattr(cli, "JsonlTelemetrySink", lambda: InMemoryTelemetryCollector())

    assert cli.main(["run-agent-planner-debug", task, "--dry-run"]) == 1
    payload = json.loads(capsys.readouterr().err)
    assert payload["validation_diagnostic"]["validation_code"] == "payload_not_grounded_in_task"
    assert payload["validation_diagnostic"]["field_path"] == "steps.0.payload"
    rendered = str(payload)
    assert task not in rendered
    assert "TASK_PRIVATE_115" not in rendered
    assert rejected_literal not in rendered


def test_planner_cli_reports_only_allowlisted_provider_error_category(
    monkeypatch, capsys,
) -> None:
    import json
    import main as cli
    import agent.planner as planner_module

    monkeypatch.setattr(
        planner_module.OpenAIPlanner, "from_environment",
        classmethod(lambda _cls: FakePlanner(PlannerCallError("provider_error"))),
    )
    monkeypatch.setattr(cli, "JsonlTelemetrySink", lambda: InMemoryTelemetryCollector())

    assert cli.main(["run-agent-planner-debug", "Open Notepad", "--dry-run"]) == 1
    payload = json.loads(capsys.readouterr().err)
    assert payload["stop_reason"] == "planner_error"
    assert payload["error_category"] == "unknown_provider_error"
    assert payload["provider_diagnostic"]["category"] == "unknown_provider_error"
    assert "validation_diagnostic" not in payload
    assert "Open Notepad" not in str(payload)


def test_openai_adapter_replan_uses_typed_plan_update_schema() -> None:
    task = "Open Notepad and type hello; if typing fails, open Calculator"
    plan = _plan(task, [
        _step("step_1", PlanStepKind.OPEN_APP, target="Notepad").model_copy(update={
            "status": PlanStepStatus.COMPLETE, "attempts": 1,
        }),
        _step("step_2", PlanStepKind.TYPE_TEXT, payload="hello").model_copy(update={
            "status": PlanStepStatus.FAILED, "attempts": 1,
            "last_failure_reason": PlannerFailureReason.STEP_EXECUTION_FAILED,
        }),
        _step("step_3", PlanStepKind.FINISH),
    ])
    update = ProviderPlanUpdate(steps=[
        ProviderPlanStep(kind=PlanStepKind.OPEN_APP, target="Calculator", payload=None),
        ProviderPlanStep(kind=PlanStepKind.FINISH, target=None, payload=None),
    ])
    responses = FakeResponses(parsed=update)
    adapter = OpenAIPlanner("test-model", SimpleNamespace(responses=responses))
    materialized = adapter.replan(
        plan, PlannerCurrentState(("step_1",), "step_2", True, 5),
        PlannerFailureReason.STEP_EXECUTION_FAILED,
    )
    assert responses.calls[0]["text_format"] is ProviderPlanUpdate
    assert [step.step_id for step in materialized.steps] == ["step_3", "step_4"]
    request_content = responses.calls[0]["input"][1]["content"]
    assert '"step_id"' not in request_content
    validated = validate_plan_update(
        materialized, task, max_steps=8, existing_steps=plan.steps[:2],
    )
    combined = [*plan.steps[:2], *validated.steps]
    assert combined[0].status is PlanStepStatus.COMPLETE
    assert combined[1].status is PlanStepStatus.FAILED
    assert combined[1].last_failure_reason is PlannerFailureReason.STEP_EXECUTION_FAILED
    assert [step.step_id for step in combined] == ["step_1", "step_2", "step_3", "step_4"]


def test_provider_plan_ids_are_deterministic_and_ordered() -> None:
    task = "Open Notepad and type hello"
    provider_plan = ProviderPlan(objective=task, steps=[
        ProviderPlanStep(kind=PlanStepKind.OPEN_APP, target="Notepad", payload=None),
        ProviderPlanStep(kind=PlanStepKind.TYPE_TEXT, target=None, payload="hello"),
        ProviderPlanStep(kind=PlanStepKind.FINISH, target=None, payload=None),
    ])
    first = OpenAIPlanner(
        "test-model", SimpleNamespace(responses=FakeResponses(parsed=provider_plan)),
    ).plan(task, PlannerContext(8))
    second = OpenAIPlanner(
        "test-model", SimpleNamespace(responses=FakeResponses(parsed=provider_plan)),
    ).plan(task, PlannerContext(8))

    assert [step.step_id for step in first.steps] == ["step_1", "step_2", "step_3"]
    assert [step.step_id for step in second.steps] == ["step_1", "step_2", "step_3"]
    assert [step.kind for step in first.steps] == [step.kind for step in provider_plan.steps]
    assert first.max_steps == 8
    assert first.status is PlanStatus.PLANNED
    assert "step_id" not in ProviderPlan.model_fields


def test_provider_plan_duplicate_semantics_still_fail_local_validation() -> None:
    task = "Open Notepad twice"
    provider_plan = ProviderPlan(objective=task, steps=[
        ProviderPlanStep(kind=PlanStepKind.OPEN_APP, target="Notepad", payload=None),
        ProviderPlanStep(kind=PlanStepKind.OPEN_APP, target="Notepad", payload=None),
        ProviderPlanStep(kind=PlanStepKind.FINISH, target=None, payload=None),
    ])
    internal = OpenAIPlanner(
        "test-model", SimpleNamespace(responses=FakeResponses(parsed=provider_plan)),
    ).plan(task, PlannerContext(8))

    with pytest.raises(PlanValidationError) as caught:
        validate_plan(internal, task, max_steps=8)
    assert [step.step_id for step in internal.steps] == ["step_1", "step_2", "step_3"]
    assert caught.value.diagnostic.validation_code == "duplicate_step_pattern"


def test_provider_materialized_plan_keeps_existing_execution_path() -> None:
    task = "Open Notepad"
    provider_plan = ProviderPlan(objective=task, steps=[
        ProviderPlanStep(kind=PlanStepKind.OPEN_APP, target="Notepad", payload=None),
        ProviderPlanStep(kind=PlanStepKind.FINISH, target=None, payload=None),
    ])
    planner = OpenAIPlanner(
        "test-model", SimpleNamespace(responses=FakeResponses(parsed=provider_plan)),
    )
    computer = FakeComputer(_observations(3))

    result = _agent(
        planner, computer,
        FakeDecisionMaker([OpenAppAction("notepad"), FinishAction("done")]),
    ).run(task)

    assert result.success
    assert result.plan is not None
    assert [step.step_id for step in result.plan.steps] == ["step_1", "step_2"]
    assert [item[0] for item in computer.executed] == [OpenAppAction("notepad")]


def test_planner_cli_dry_run_is_redacted_and_does_not_construct_windows_or_jev(
    monkeypatch, capsys,
) -> None:
    import json
    import main as cli
    import agent.planner as planner_module

    task = "Type the secret phrase sunset purple"
    fake_planner = FakePlanner(_plan(task, [
        _step("step_1", PlanStepKind.TYPE_TEXT, payload="sunset purple"),
        _step("step_2", PlanStepKind.FINISH),
    ]))
    monkeypatch.setattr(
        planner_module.OpenAIPlanner, "from_environment",
        classmethod(lambda _cls: fake_planner),
    )
    collector = InMemoryTelemetryCollector()
    monkeypatch.setattr(cli, "JsonlTelemetrySink", lambda: collector)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("planner dry-run must not construct Windows or Jev runtime")

    monkeypatch.setattr(cli, "WindowsApplicationCatalog", forbidden)
    monkeypatch.setattr(cli.JevDecisionMaker, "from_environment", forbidden)
    monkeypatch.setattr(cli, "visual_provider_from_environment", forbidden)

    assert cli.main(["run-agent-planner-debug", task, "--dry-run"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["stop_reason"] == "dry_run"
    assert output["plan"] == [
        {"step_id": "step_1", "kind": "TYPE_TEXT"},
        {"step_id": "step_2", "kind": "FINISH"},
    ]
    assert "sunset purple" not in str(output)
    assert collector.events


def test_planner_cli_live_run_requires_confirmation_before_windows_setup(
    monkeypatch, capsys,
) -> None:
    import main as cli
    import agent.planner as planner_module

    task = "Open Notepad"
    fake_planner = FakePlanner(_open_finish(task))
    monkeypatch.setattr(
        planner_module.OpenAIPlanner, "from_environment",
        classmethod(lambda _cls: fake_planner),
    )
    monkeypatch.setattr(cli, "JsonlTelemetrySink", lambda: InMemoryTelemetryCollector())
    monkeypatch.setattr("builtins.input", lambda: "n")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Windows runtime must be built only after affirmative confirmation")

    monkeypatch.setattr(cli, "WindowsApplicationCatalog", forbidden)
    assert cli.main(["run-agent-planner-debug", task]) == 1
    assert "cancelled" in capsys.readouterr().err


def test_planner_result_summary_omits_objective_target_and_literal() -> None:
    import main as cli

    task = "Open Notepad and type a private phrase"
    plan = _plan(task, [
        _step("step_1", PlanStepKind.OPEN_APP, target="Notepad"),
        _step("step_2", PlanStepKind.TYPE_TEXT, payload="a private phrase"),
        _step("step_3", PlanStepKind.FINISH),
    ])
    result = PlannerAgentResult(True, "finished", "ok", plan=plan)
    output = str(cli._planner_result_payload(result))
    assert "Notepad" not in output
    assert "private phrase" not in output
    assert task not in output
    assert "TYPE_TEXT" in output


def test_planner_environment_fails_without_separate_key(monkeypatch) -> None:
    monkeypatch.delenv("PLANNER_API_KEY", raising=False)
    with pytest.raises(PlannerConfigurationError):
        OpenAIPlanner.from_environment()
