"""Synthetic integration tests for generic bounded target activation."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.generic_task import GenericTaskDebugAgent, wait_for_local_transition
from computer.actions import ClickAction, OpenAppAction, QuerySubmitAction, TypeAction, VisualClickAction
from computer.applications import ApplicationCandidate, MemoryApplicationCatalog
from computer.models import Observation, Rect, ScreenshotMetadata, UIElement, VisualElement
from computer.results import ActionResult
from decision.context import requested_target_spec
from decision.models import TargetChoiceResult
from decision.target_resolution import (
    CandidateEvidence, TargetResolutionStatus, TargetSpec, resolve_target,
)
from safety.policy import GenericTargetActivationPolicy


APP_ID = "app_1234567890abcdef"


def observation(
    snapshot: str,
    *,
    elements: tuple[UIElement, ...] = (),
    visual: tuple[VisualElement, ...] = (),
    app_id: str = APP_ID,
    process_id: int = 42,
    foreground_hwnd: int | None = 100,
) -> Observation:
    metadata = None
    if visual:
        metadata = ScreenshotMetadata(
            snapshot, 77, Rect(0, 0, 800, 600), Rect(0, 0, 800, 600),
            800, 600, 96, 96, 1.0, 1.0,
        )
    return Observation(
        "test.exe", "Test Window", elements, process_id=process_id,
        observation_id=snapshot, application_id=app_id, visual_elements=visual,
        screenshot=metadata, visual_provider="fixture" if visual else None,
        foreground_hwnd=foreground_hwnd,
    )


def conversation(control_id: str = "c1", name: str = "Pablo García", *, parent: str = "Conversations") -> UIElement:
    return UIElement(
        control_id, name, "ListItem", rectangle=Rect(10, 10, 300, 70),
        enabled=True, visible=True, parent_name=parent,
    )


def query_field(snapshot: str, value: str = "", *, focused: bool = True) -> UIElement:
    return UIElement(
        "c1", "Search", "Edit", automation_id="query_input",
        rectangle=Rect(5, 5, 300, 40), enabled=True, visible=True,
        focused=focused, is_password=False, observed_text=value,
    )


class FakeComputer:
    def __init__(self, locals_: list[Observation], directed: list[Observation] | None = None,
                 *, visual_provider: object | None = None) -> None:
        self.locals = list(locals_)
        self.directed = list(directed or [])
        self.visual_provider = visual_provider
        self.directed_calls = []
        self.executed = []
        self.visual_activations = []
        self.query_submits = []

    def observe_local(self) -> Observation:
        if not self.locals:
            raise RuntimeError("no synthetic observation remains")
        return self.locals.pop(0)

    def observe_directed(self, grounding) -> Observation:
        self.directed_calls.append(grounding)
        if not self.directed:
            raise RuntimeError("no synthetic visual observation remains")
        return self.directed.pop(0)

    def execute(self, action, observation=None) -> ActionResult:
        self.executed.append((action, observation))
        return ActionResult(True, action, "synthetic action", input_issued=True)

    def execute_generic_target_activation(self, action, observation) -> ActionResult:
        self.visual_activations.append((action, observation))
        return ActionResult(True, action, "synthetic visual activation", input_issued=True)

    def execute_generic_query_submit(self, action, observation) -> ActionResult:
        self.query_submits.append((action, observation))
        return ActionResult(True, action, "synthetic query submit", input_issued=True)


class FakeDecisionMaker:
    min_confidence = .8

    def __init__(self, candidate_id: str | None = None, confidence: float = .95) -> None:
        self.candidate_id = candidate_id
        self.confidence = confidence
        self.calls = []

    def decide_target_activation(self, target, resolution) -> TargetChoiceResult:
        self.calls.append((target, resolution))
        candidate_id = self.candidate_id or resolution.frontier_candidate_ids[0]
        return TargetChoiceResult("ready", candidate_id, self.confidence, "synthetic decision")


def chat_catalog() -> MemoryApplicationCatalog:
    candidate = ApplicationCandidate(
        APP_ID, "WhatsApp", "test", launch_policy="allow",
    )
    return MemoryApplicationCatalog((candidate,))


def test_visible_uia_chat_launches_if_needed_without_grounding_type_or_enter() -> None:
    computer = FakeComputer([
        observation("before", app_id=""),
        observation("opened", elements=(conversation(),)),
        observation("after", elements=(conversation(),)),
    ], visual_provider=object())
    decision = FakeDecisionMaker("c1")
    result = GenericTaskDebugAgent(
        computer, decision, app_catalog=chat_catalog(),
        policy=GenericTargetActivationPolicy(chat_catalog()),
    ).run("Open WhatsApp and open the chat with Pablo García")

    assert result.success and result.stop_reason == "target_activated"
    assert result.app_launches == 1
    assert [action.kind for action, _ in computer.executed] == ["open_app", "click"]
    assert computer.executed[1][1].observation_id == "opened"
    assert result.application_activation.identity_match
    assert result.application_activation.activation_attempts == 1
    assert result.application_activation.requested_app_id == APP_ID
    assert not computer.directed_calls and not computer.visual_activations
    assert all(not isinstance(action, TypeAction) for action, _ in computer.executed)
    assert not computer.query_submits
    assert result.post_action_observation_obtained


def test_generic_activation_timeout_reports_diagnostics_before_any_target_work() -> None:
    from agent.generic_task import GenericTaskBudgets

    class FakeClock:
        value = 0.0

        def __call__(self):
            return self.value

        def sleep(self, seconds):
            self.value += seconds

    clock = FakeClock()
    computer = FakeComputer([
        observation("before", app_id="old-app"),
        observation("intermediate", app_id="", process_id=50, foreground_hwnd=500),
        observation("intermediate-2", app_id="", process_id=50, foreground_hwnd=500),
        observation("intermediate-3", app_id="", process_id=50, foreground_hwnd=500),
    ], visual_provider=object())
    decision = FakeDecisionMaker("c1")
    catalog = chat_catalog()
    agent = GenericTaskDebugAgent(
        computer, decision, app_catalog=catalog,
        policy=GenericTargetActivationPolicy(catalog),
        budgets=GenericTaskBudgets(
            app_activation_timeout_seconds=.2, transition_poll_seconds=.1,
        ),
        clock=clock, sleep_fn=clock.sleep,
    )

    result = agent.run("Open WhatsApp and open the chat with Pablo García")

    assert not result.success and result.stop_reason == "application_activation_timeout"
    diagnostic = result.application_activation
    assert diagnostic is not None
    assert diagnostic.requested_display_name == "WhatsApp"
    assert diagnostic.requested_launch_kind == "test"
    assert diagnostic.launch_succeeded
    assert diagnostic.activation_timeout_ms == 200
    assert diagnostic.activation_poll_interval_ms == 100
    assert diagnostic.activation_attempts == 3
    assert diagnostic.identity_match is False
    assert diagnostic.identity_match_method == "none"
    assert diagnostic.initial_foreground.trusted_app_id == "old-app"
    assert diagnostic.final_foreground.trusted_app_id is None
    assert diagnostic.observed_foregrounds[-1].hwnd == 500
    assert [action.kind for action, _ in computer.executed] == ["open_app"]
    assert not decision.calls and not computer.directed_calls
    assert not computer.visual_activations and not computer.query_submits
    assert result.literal_types == result.query_submits == result.final_target_activations == 0


def test_visible_visual_chat_uses_same_resolver_and_snapshot_bound_click() -> None:
    visual = VisualElement(
        "v1", "Pablo García", "conversation", Rect(20, 80, 300, 140),
        None, True, parent="Conversations",
    )
    computer = FakeComputer(
        [observation("initial"), observation("after")],
        [observation("visual", visual=(visual,))], visual_provider=object(),
    )
    decision = FakeDecisionMaker("v1")
    result = GenericTaskDebugAgent(computer, decision).run(
        "Open WhatsApp and open the chat with Pablo García",
    )

    assert result.success
    assert len(computer.directed_calls) == 1
    assert len(decision.calls) == 1
    action, bound_observation = computer.visual_activations[0]
    assert action == VisualClickAction("visual", "v1")
    assert bound_observation.observation_id == action.snapshot_id
    assert not computer.query_submits


def test_query_results_after_typing_skip_enter_and_activate_visible_target() -> None:
    search = query_field("before", focused=True)
    after_type = observation("typed", elements=(
        query_field("typed", "Pablo García"), conversation("c2"),
    ))
    computer = FakeComputer([
        observation("initial", elements=(search,)), after_type,
        observation("after", elements=(conversation("c3"),)),
    ])
    decision = FakeDecisionMaker("c2")
    result = GenericTaskDebugAgent(computer, decision).run(
        "Open WhatsApp and open the chat with Pablo García",
    )

    assert result.success
    assert result.literal_types == 1 and result.query_submits == 0
    typed = next(action for action, _ in computer.executed if isinstance(action, TypeAction))
    assert typed.text == "Pablo García"
    assert not computer.query_submits


def test_search_submits_once_only_after_typed_value_is_reobserved_and_transition_seen() -> None:
    pre = observation("initial", elements=(query_field("initial", focused=True),))
    typed = observation("typed", elements=(query_field("typed", "Pablo García", focused=True),))
    result_state = observation("results", elements=(
        query_field("results", "Pablo García", focused=True), conversation("c2"),
    ))
    post = observation("after", elements=(conversation("c3"),))
    computer = FakeComputer([pre, typed, result_state, post])
    decision = FakeDecisionMaker("c2")
    result = GenericTaskDebugAgent(computer, decision).run(
        "Open WhatsApp and open the chat with Pablo García",
    )

    assert result.success
    assert result.literal_types == 1 and result.query_submits == 1
    assert len(computer.query_submits) == 1
    assert computer.query_submits[0][0] == QuerySubmitAction()
    assert sum(isinstance(action, TypeAction) for action, _ in computer.executed) == 1


def test_visual_search_field_is_clicked_then_verified_before_literal_type() -> None:
    search = VisualElement(
        "v1", "Search", "search field", Rect(20, 20, 350, 60), None, True,
    )
    computer = FakeComputer(
        [
            observation("initial"),
            observation("focused", elements=(query_field("focused", focused=True),)),
            observation("typed", elements=(query_field("typed", "Pablo García"), conversation("c2"))),
            observation("after"),
        ],
        [observation("target-ground", visual=(search,)), observation("field-ground", visual=(search,))],
        visual_provider=object(),
    )
    result = GenericTaskDebugAgent(computer, FakeDecisionMaker("c2")).run(
        "Open WhatsApp and open the chat with Pablo García",
    )

    assert result.success
    assert result.visual_grounding_calls == 2
    assert result.query_field_activations == 1
    assert result.visual_target_activations == 1
    assert computer.visual_activations[0][0] == VisualClickAction("field-ground", "v1")
    assert any(isinstance(action, TypeAction) for action, _ in computer.executed)


def test_indistinguishable_duplicates_stop_without_calling_jev_or_activating() -> None:
    computer = FakeComputer([observation(
        "duplicates", elements=(conversation("c1"), conversation("c2")),
    )])
    decision = FakeDecisionMaker()
    result = GenericTaskDebugAgent(computer, decision).run(
        "Open WhatsApp and open the chat with Pablo García",
    )

    assert not result.success and result.target_resolution_status == "ambiguous"
    assert not decision.calls and not computer.executed


def test_bounded_choice_uses_only_frontier_and_confidence_gate() -> None:
    request = "Open WhatsApp and open the chat with Pablo García from Work"
    spec = requested_target_spec(request, experimental_generic=True)
    assert spec is not None and spec.qualifiers == ("Work",)
    candidates = (
        CandidateEvidence(
            "c1", "Pablo García", (), "conversation", True, True, True,
            "UIA", "snapshot", "ListItem",
        ),
        CandidateEvidence(
            "c2", "Pablo García", ("Work contact",), "contact", True, True, True,
            "UIA", "snapshot", "ListItem",
        ),
    )
    resolution = resolve_target(
        spec, candidates, expected_snapshot_id="snapshot", frontier_mode=True,
    )
    assert resolution.status is TargetResolutionStatus.CHOICE
    assert resolution.frontier_candidate_ids == ("c1", "c2")
    assert resolution.evidence_distinguishable

    low_computer = FakeComputer([observation("choice", elements=(
        conversation("c1"), conversation("c2", parent="Work contact"),
    ))])
    low_decision = FakeDecisionMaker("c2", confidence=.79)
    low_result = GenericTaskDebugAgent(low_computer, low_decision).run(request)
    assert not low_result.success and low_result.stop_reason == "low_confidence"
    assert not low_computer.executed

    high_computer = FakeComputer([
        observation("choice", elements=(
            conversation("c1"), conversation("c2", parent="Work contact"),
        )),
        observation("after"),
    ])
    high_result = GenericTaskDebugAgent(
        high_computer, FakeDecisionMaker("c2", confidence=.80),
    ).run(request)
    assert high_result.success and high_result.chosen_candidate_id == "c2"
    assert isinstance(high_computer.executed[0][0], ClickAction)

    unoffered_computer = FakeComputer([observation("choice", elements=(
        conversation("c1"), conversation("c2", parent="Work contact"),
    ))])
    unoffered = GenericTaskDebugAgent(
        unoffered_computer, FakeDecisionMaker("c999"),
    ).run(request)
    assert unoffered.stop_reason == "unoffered_target"
    assert not unoffered_computer.executed


def test_explicit_identity_contradiction_is_removed_before_jev() -> None:
    wrong = UIElement(
        "c1", "Pablo López", "ListItem", rectangle=Rect(10, 10, 300, 70),
        enabled=True, visible=True, parent_name="Conversations",
    )
    computer = FakeComputer([observation("wrong", elements=(wrong,))])
    decision = FakeDecisionMaker()
    result = GenericTaskDebugAgent(computer, decision).run(
        "Open WhatsApp and open the chat with Pablo García",
    )
    assert not result.success
    assert result.target_resolution_status == "no_match"
    assert not decision.calls and not computer.executed


def test_file_domain_uses_the_same_generic_controller_and_resolver() -> None:
    file_row = UIElement(
        "c1", "factura septiembre.pdf", "ListItem",
        rectangle=Rect(10, 10, 300, 70), enabled=True, visible=True,
        parent_name="Files",
    )
    computer = FakeComputer([
        observation("file", elements=(file_row,)), observation("after"),
    ])
    decision = FakeDecisionMaker("c1")
    spec = requested_target_spec("Open factura septiembre.pdf", experimental_generic=True)
    assert spec is not None and spec.desired_role == "file"
    result = GenericTaskDebugAgent(computer, decision).run("Open factura septiembre.pdf")
    assert result.success and result.target_resolution_status == "unique"
    assert isinstance(computer.executed[0][0], ClickAction)


def test_stale_visual_snapshot_and_consequential_target_are_blocked() -> None:
    target = VisualElement("v1", "Pablo García", "conversation", Rect(5, 5, 80, 40), None, True)
    current = observation("current", visual=(target,))
    verdict = GenericTargetActivationPolicy().validate_candidate(
        VisualClickAction("old", "v1"), current,
    )
    assert verdict.disposition == "deny"

    dangerous = UIElement(
        "c1", "Delete Pablo García", "Button", enabled=True, visible=True,
    )
    computer = FakeComputer([observation("dangerous", elements=(dangerous,))])
    decision = FakeDecisionMaker()
    result = GenericTaskDebugAgent(computer, decision).run(
        "Open WhatsApp and open the chat with Pablo García",
    )
    assert not result.success and not decision.calls and not computer.executed


@pytest.mark.parametrize("role", [
    "conversation", "contact", "file", "folder", "navigation destination",
    "button", "list item", "card",
])
def test_generic_activation_policy_has_bounded_cross_domain_roles(role: str) -> None:
    element = VisualElement("v1", "Target", role, Rect(10, 10, 80, 50), None, True)
    current = observation("roles", visual=(element,))
    verdict = GenericTargetActivationPolicy().validate_candidate(
        VisualClickAction("roles", "v1"), current,
    )
    assert verdict.disposition == "allow"


def test_transition_wait_is_local_bounded_and_foreground_stable() -> None:
    baseline = observation("base", elements=(query_field("base", focused=True),))
    same_a = observation("same-a", elements=(query_field("same-a", focused=True),))
    same_b = observation("same-b", elements=(query_field("same-b", focused=True),))
    computer = FakeComputer([same_a, same_b], visual_provider=object())
    current = [0.0]

    def clock() -> float:
        return current[0]

    def sleep(seconds: float) -> None:
        current[0] += seconds

    timeout = wait_for_local_transition(
        computer, baseline, timeout_seconds=.4, poll_seconds=.2,
        clock=clock, sleep_fn=sleep,
    )
    assert not timeout.changed and timeout.reason == "transition_timeout"
    assert not computer.directed_calls

    changed_foreground = observation("other-window", app_id="other-app")
    foreground_computer = FakeComputer([changed_foreground], visual_provider=object())
    stopped = wait_for_local_transition(
        foreground_computer, baseline, timeout_seconds=1, poll_seconds=.2,
    )
    assert not stopped.changed and stopped.reason == "foreground_changed"
    assert not foreground_computer.directed_calls


def test_generic_source_has_no_application_or_example_specific_branches() -> None:
    root = Path(__file__).parents[1]
    paths = (
        root / "agent" / "generic_task.py",
        root / "agent" / "target_evidence.py",
        root / "decision" / "target_resolution.py",
        root / "safety" / "policy.py",
    )
    forbidden = (
        "spotify", "whatsapp", "chrome", "californication", "red hot chili peppers",
        "pablo garcía", "factura septiembre.pdf",
    )
    for path in paths:
        text = path.read_text(encoding="utf-8").casefold()
        assert not any(value in text for value in forbidden), path.name
