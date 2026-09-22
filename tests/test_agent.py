"""Deterministic tests for the bounded generic agent loop."""

from dataclasses import dataclass, replace

import pytest

from agent.loop import Agent, AgentLimits
from computer.actions import ClickAction, FinishAction, OpenAppAction, PressKeyAction, TypeAction, VisualClickAction
from computer.applications import ApplicationCandidate, MemoryApplicationCatalog
from computer.models import Observation, Rect, ScreenshotMetadata, UIElement, VisualElement
from computer.results import ActionResult
from decision.models import DecisionResult
from safety.interfaces import SafetyDecision
from safety.policy import BasicActionPolicy

APP_ID = "app_1111111111111111"


def _observation(label: str = "Editor") -> Observation:
    return Observation(
        "notepad.exe", label,
        elements=(UIElement("c1", "Document", "Document", enabled=True, visible=True, focused=True),),
        process_id=123,
    )


@dataclass
class FakeComputer:
    observations: list[Observation]
    execution_results: list[ActionResult] | None = None

    def __post_init__(self) -> None:
        self.observed: list[Observation] = []
        self.executed: list[tuple[object, Observation | None]] = []

    def observe(self) -> Observation:
        observation = self.observations.pop(0)
        self.observed.append(observation)
        return observation

    def execute(self, action, observation=None) -> ActionResult:
        self.executed.append((action, observation))
        if self.execution_results:
            return self.execution_results.pop(0)
        return ActionResult(True, action, "ok")


class ScriptedDecision:
    def __init__(self, decisions: list[DecisionResult]) -> None:
        self.decisions = decisions
        self.calls: list[tuple[str, Observation, tuple[ActionResult, ...]]] = []

    def decide(self, request, observation, history=()):
        self.calls.append((request, observation, tuple(history)))
        return self.decisions.pop(0)


def _ready(action, confidence: float = 0.95) -> DecisionResult:
    return DecisionResult("ready", action, confidence, "selected", selected_option=action.kind)


def _decision_error(code: str = "invalid_response", diagnostic: str = "unexpected_answers") -> DecisionResult:
    return DecisionResult("error", None, None, "safe failure", error=code, diagnostic=diagnostic)


def test_successful_loop_uses_fresh_observations_and_history() -> None:
    observations = [_observation("one"), _observation("two"), _observation("three")]
    open_action = OpenAppAction(APP_ID)
    type_action = TypeAction("Hello {ENTER} literally")
    computer = FakeComputer(observations.copy())
    decision = ScriptedDecision([_ready(open_action), _ready(type_action), _ready(FinishAction("done"))])
    catalog = MemoryApplicationCatalog((
        ApplicationCandidate(APP_ID, "Notepad", "test", launch_policy="allow"),
    ))
    result = Agent(
        computer, decision, policy=BasicActionPolicy(catalog),
        limits=AgentLimits(settle_open_seconds=0, settle_action_seconds=0),
    ).run("write a greeting")

    assert result.success and result.stop_reason == "finished"
    assert result.steps == 3
    assert [item[0] for item in computer.executed] == [open_action, type_action]
    assert computer.executed[0][1] is observations[0]
    assert computer.executed[1][1] is observations[1]
    assert decision.calls[1][2][0].action == open_action
    assert decision.calls[2][2][-1].action == type_action


def test_visual_click_is_followed_by_a_fresh_observation() -> None:
    metadata = ScreenshotMetadata(
        "visual-one", 1, Rect(0, 0, 100, 100), Rect(0, 0, 100, 100),
        100, 100, 96, 96, 1, 1,
    )
    before = Observation(
        "app.exe", "before", observation_id="visual-one", screenshot=metadata,
        visual_elements=(VisualElement("v1", "Search", "button", Rect(1, 1, 20, 20), .95, True),),
    )
    after = Observation("app.exe", "after", observation_id="visual-two")
    action = VisualClickAction("visual-one", "v1")
    computer = FakeComputer([before, after])
    decision = ScriptedDecision([_ready(action), _ready(FinishAction("done"))])
    result = Agent(
        computer, decision, limits=AgentLimits(settle_action_seconds=0),
    ).run("click Search")
    assert result.success
    assert computer.executed == [(action, before)]
    assert decision.calls[1][1] is after


def test_dry_run_proposes_one_action_without_execution() -> None:
    computer = FakeComputer([_observation()])
    decision = ScriptedDecision([_ready(TypeAction("secret"))])
    result = Agent(computer, decision).run("type", dry_run=True)

    assert result.success and result.stop_reason == "needs_human"
    assert computer.executed == []


def test_finish_is_terminal_and_does_not_execute() -> None:
    computer = FakeComputer([_observation()])
    decision = ScriptedDecision([_ready(FinishAction("task complete"))])
    result = Agent(computer, decision).run("finish")

    assert result.success and result.stop_reason == "finished"
    assert computer.executed == []


def test_low_confidence_stops_before_execution() -> None:
    computer = FakeComputer([_observation()])
    decision = ScriptedDecision([_ready(TypeAction("hello"), confidence=0.79)])
    result = Agent(computer, decision).run("type")

    assert result.stop_reason == "low_confidence"
    assert computer.executed == []
    assert len(decision.calls) == 1


def test_missing_confidence_fails_closed() -> None:
    computer = FakeComputer([_observation()])
    decision = ScriptedDecision([DecisionResult("ready", TypeAction("hello"), None, "missing")])
    result = Agent(computer, decision).run("type")

    assert result.stop_reason == "low_confidence"
    assert computer.executed == []


def test_needs_human_and_decision_errors_are_terminal() -> None:
    observation = _observation()
    for response, reason in (
        (DecisionResult("needs_human", None, None, "stop", selected_option="stop"), "needs_human"),
        (DecisionResult("error", None, None, "API unavailable", error="transport_error"), "decision_error"),
    ):
        computer = FakeComputer([observation])
        result = Agent(computer, ScriptedDecision([response])).run("request")
        assert result.stop_reason == reason
        assert computer.executed == []


@pytest.mark.parametrize("error", ["invalid_response", "api_error"])
def test_retryable_decision_error_retries_once_and_success_continues(error: str) -> None:
    action = TypeAction("hello")
    observation = _observation("before")
    after = _observation("after")
    computer = FakeComputer([observation, after])
    decision = ScriptedDecision([_decision_error(error), _ready(action), _ready(FinishAction("done"))])

    result = Agent(computer, decision, limits=AgentLimits(settle_action_seconds=0)).run("type hello")

    assert result.success and result.stop_reason == "finished"
    assert len(decision.calls) == 3
    assert decision.calls[0][1] is observation and decision.calls[1][1] is observation
    assert decision.calls[0][2] == decision.calls[1][2]
    assert [executed[0] for executed in computer.executed] == [action]


def test_second_invalid_decision_response_fails_closed() -> None:
    computer = FakeComputer([_observation()])
    decision = ScriptedDecision([_decision_error(), _decision_error("invalid_response", "wrong_model")])

    result = Agent(computer, decision).run("request")

    assert result.stop_reason == "decision_error"
    assert len(decision.calls) == 2
    assert computer.executed == []


def test_stop_is_never_retried() -> None:
    computer = FakeComputer([_observation()])
    decision = ScriptedDecision([
        DecisionResult("needs_human", None, 0.99, "stop", selected_option="stop"),
    ])

    result = Agent(computer, decision).run("request")

    assert result.stop_reason == "needs_human"
    assert len(decision.calls) == 1


def test_no_action_executes_between_decision_attempts() -> None:
    computer = FakeComputer([_observation()])

    class InspectingDecision:
        def __init__(self) -> None:
            self.calls = 0

        def decide(self, request, observation, history=()):
            self.calls += 1
            assert computer.executed == []
            return _decision_error() if self.calls == 1 else _ready(FinishAction("done"))

    result = Agent(computer, InspectingDecision()).run("request")

    assert result.success and result.stop_reason == "finished"
    assert computer.executed == []


def test_allowlist_and_invalid_click_id_are_safety_rejected() -> None:
    cases = [OpenAppAction("cmd"), PressKeyAction(("ctrl", "v")), ClickAction("c7")]
    for action in cases:
        computer = FakeComputer([_observation()])
        result = Agent(computer, ScriptedDecision([_ready(action)])).run("request")
        assert result.stop_reason == "safety_rejected"
        assert computer.executed == []


def test_execution_failure_is_recorded_without_retry() -> None:
    action = TypeAction("hello")
    computer = FakeComputer([_observation()], [ActionResult(False, action, "failed", error="unsafe_target")])
    result = Agent(computer, ScriptedDecision([_ready(action), _ready(FinishAction("never"))])).run("type")

    assert result.stop_reason == "execution_failed"
    assert len(result.history) == 1
    assert len(computer.executed) == 1


def test_observation_failure_and_keyboard_interrupt_stop_cleanly() -> None:
    class BrokenComputer(FakeComputer):
        def observe(self):
            raise RuntimeError("UIA unavailable")

    broken = BrokenComputer([])
    result = Agent(broken, ScriptedDecision([])).run("request")
    assert result.stop_reason == "observation_failed"

    class InterruptedComputer(FakeComputer):
        def observe(self):
            raise KeyboardInterrupt

    interrupted = Agent(InterruptedComputer([]), ScriptedDecision([])).run("request")
    assert interrupted.stop_reason == "interrupted"


def test_repetition_guard_stops_before_second_duplicate_when_configured() -> None:
    action = TypeAction("same")
    computer = FakeComputer([_observation("same"), _observation("same")])
    decision = ScriptedDecision([_ready(action), _ready(action)])
    result = Agent(
        computer, decision,
        limits=AgentLimits(repeat_limit=1, settle_action_seconds=0),
    ).run("repeat")

    assert result.stop_reason == "repeated_action"
    assert len(computer.executed) == 1


def test_click_repetition_ignores_snapshot_local_control_id() -> None:
    first = Observation(
        "app", "window",
        elements=(UIElement("c1", "Save", "Button", automation_id="save", enabled=True, visible=True),),
    )
    second = Observation(
        "app", "window",
        elements=(UIElement("c9", "Save", "Button", automation_id="save", enabled=True, visible=True),),
    )
    computer = FakeComputer([first, second])
    action_one = ClickAction("c1")
    action_two = ClickAction("c9")
    result = Agent(
        computer, ScriptedDecision([_ready(action_one), _ready(action_two)]),
        limits=AgentLimits(repeat_limit=1, settle_action_seconds=0),
    ).run("click save")

    assert result.stop_reason == "repeated_action"
    assert len(computer.executed) == 1


def test_max_steps_is_bounded() -> None:
    action_one = TypeAction("one")
    action_two = TypeAction("two")
    computer = FakeComputer([_observation("one"), _observation("two")])
    result = Agent(
        computer, ScriptedDecision([_ready(action_one), _ready(action_two)]),
        limits=AgentLimits(max_steps=2, settle_action_seconds=0),
    ).run("bounded")

    assert result.stop_reason == "max_steps"
    assert result.steps == 2


def test_confirmation_policy_fails_closed_without_consent() -> None:
    class ConfirmPolicy:
        def validate(self, action, observation):
            return SafetyDecision("confirm", "user confirmation required")

    computer = FakeComputer([_observation()])
    result = Agent(computer, ScriptedDecision([_ready(TypeAction("hello"))]), policy=ConfirmPolicy()).run("type")
    assert result.stop_reason == "confirmation_required"
    assert computer.executed == []


def test_debug_report_contains_no_request_or_literal_text() -> None:
    secret = "private-token-123"
    computer = FakeComputer([_observation()])
    events: list[dict[str, object]] = []
    decision = ScriptedDecision([_ready(TypeAction(secret))])
    result = Agent(computer, decision, reporter=lambda _event, data: events.append(data)).run(
        secret, dry_run=True, debug=True,
    )

    assert result.success
    assert secret not in repr(events)


def test_retry_debug_does_not_expose_diagnostic_secrets() -> None:
    secret = "private-response-secret"
    computer = FakeComputer([_observation()])
    events: list[tuple[str, dict[str, object]]] = []
    decision = ScriptedDecision([
        DecisionResult("error", None, None, "safe", error="invalid_response", diagnostic=secret),
        _ready(FinishAction("done")),
    ])

    result = Agent(computer, decision, reporter=lambda event, data: events.append((event, data))).run(
        "request", debug=True,
    )

    assert result.success
    assert any(event == "decision_retry" for event, _ in events)
    assert secret not in repr(events)


def test_debug_does_not_expose_unrelated_observed_text() -> None:
    secret = "unrelated-private-document-content"
    observation = _observation()
    observation = replace(
        observation,
        elements=(replace(observation.elements[0], observed_text=secret),),
    )
    events: list[dict[str, object]] = []

    result = Agent(
        FakeComputer([observation]), ScriptedDecision([_ready(FinishAction("done"))]),
        reporter=lambda _event, data: events.append(data),
    ).run("request", debug=True)

    assert result.success
    assert secret not in repr(events)
    observation_event = events[0]
    assert observation_event["focused_control"]["observed_text_length"] == len(secret)
