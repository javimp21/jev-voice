"""Experimental hybrid controller tests; no desktop or network access."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import Mock

import pytest

from agent.hybrid_debug import (
    HybridClickDebugAgent, HybridDebugAgent, HybridResultDebugAgent, HybridTypeDebugAgent,
)
from agent.loop import AgentLimits
from computer.actions import (
    ClickAction, FinishAction, OpenAppAction, PressKeyAction, QuerySubmitAction, TypeAction,
    VisualClickAction,
)
from computer.applications import ApplicationCandidate, MemoryApplicationCatalog
from computer.models import (
    Observation, ProviderErrorDiagnostic, Rect, ScreenshotMetadata, UIElement, VisualElement,
    VisualCandidateProviderDiagnostic, VisualGroundingStatus, VisualPipelineDiagnostic, VisualReadinessReason,
    ResultReadinessResult, VisualReadinessResult,
)
from computer.results import ActionResult
from computer.visual import VisualGroundingRequest
from decision.jev import JevDecisionMaker, decision_effect
from decision.models import DecisionEffect, HybridDecisionResult, VisualGroundingNeed
from decision.target_resolution import IdentityEvidence, TargetResolutionStatus
from safety.policy import AutonomousActionPolicy, BasicActionPolicy


APP_ID = "app_1111111111111111"


def local(snapshot: str = "local-1", *, elements: tuple[UIElement, ...] = ()) -> Observation:
    return Observation("spotify.exe", "Spotify", elements, observation_id=snapshot)


def hybrid(snapshot: str = "hybrid-1") -> Observation:
    metadata = ScreenshotMetadata(
        snapshot, 7, Rect(0, 0, 1000, 500), Rect(0, 0, 1000, 500),
        1000, 500, 96, 96, 1, 1,
    )
    return Observation(
        "spotify.exe", "Spotify", process_id=42, observation_id=snapshot,
        application_id=APP_ID, screenshot=metadata,
        visual_elements=(VisualElement(
            "v1", "Californication, Song, Red Hot Chili Peppers", "card",
            Rect(100, 100, 400, 180), None, True,
        ),), visual_provider="gemini", visual_latency_ms=2200,
        visual_directed_grounding=True, visual_returned_elements=1,
    )


def phase_hybrid(snapshot: str = "hybrid-1") -> Observation:
    return replace(hybrid(snapshot), visual_elements=(VisualElement(
        "v1", "Search", "search field", Rect(100, 100, 400, 180), None, True,
    ),))


def need(objective: str = "Find a visible music search field.", confidence: float = .95):
    grounding = VisualGroundingNeed(objective, "UIA is insufficient", 5)
    return HybridDecisionResult(
        "grounding", None, grounding, confidence, "ground", selected_option="grounding_1",
        offered_grounding_needs=(grounding,), effect=DecisionEffect.OBSERVE,
    )


def ready(action, confidence: float = .95):
    return HybridDecisionResult(
        "ready", action, None, confidence, "ready", selected_option=action.kind,
        effect=(DecisionEffect.TERMINAL if isinstance(action, FinishAction) else DecisionEffect.ACT),
    )


class Decisions:
    min_confidence = .8

    def __init__(self, values):
        self.values = list(values)
        self.calls = []
        self.result_decision = None
        self.result_calls = []

    def decide_hybrid(self, request, observation, history=()):
        self.calls.append((request, observation, tuple(history)))
        value = self.values.pop(0)
        if value.grounding_need is not None and not value.observation_id:
            value = replace(value, observation_id=observation.observation_id)
        return value

    def decide_result_selection(self, request, observation):
        self.result_calls.append((request, observation))
        return self.result_decision or ready(
            VisualClickAction(observation.observation_id, observation.visual_elements[0].id)
        )


class Computer:
    def __init__(self, locals_, directed=None):
        self.locals = list(locals_)
        self.directed_result = directed or hybrid()
        self.directed_calls: list[VisualGroundingRequest] = []
        self.executed = []
        self.request = ""
        self.last_local = None
        self.phase_clicks = []
        self.phase_result = None
        self.type_calls = []
        self.type_result = None
        self.context_match = True
        self.result_directed_result = None
        self.result_directed_calls = []
        self.result_clicks = []
        self.result_click_result = None
        self.result_baselines = [phase_hybrid("pre-submit"), phase_hybrid("post-submit")]
        self.query_submits = []
        self.query_submit_result = None

    def set_observation_request(self, request):
        self.request = request

    def observe_local(self):
        if self.locals:
            self.last_local = self.locals.pop(0)
        if self.last_local is None:
            raise RuntimeError("no local observation")
        return self.last_local

    def observe_directed(self, grounding):
        self.directed_calls.append(grounding)
        result = self.directed_result
        if isinstance(result, list):
            return result.pop(0)
        return result

    def observe_result_baseline(self):
        return self.result_baselines.pop(0)

    def observe_result_directed(self, grounding, baseline):
        self.result_directed_calls.append(grounding)
        return self.result_directed_result or hybrid("result-1")

    def execute_query_submit_phase3(self, action, observation, verified_search_observation):
        self.query_submits.append((action, observation, verified_search_observation))
        if isinstance(self.query_submit_result, Exception):
            raise self.query_submit_result
        return self.query_submit_result or ActionResult(
            True, action, "submitted", source_observation_id=observation.observation_id,
            input_issued=True,
        )

    def execute(self, action, observation=None):
        self.executed.append((action, observation))
        return ActionResult(True, action, "done")

    def execute_visual_click_phase1(self, action, observation, request, jev_confidence):
        self.phase_clicks.append((action, observation, request, jev_confidence))
        if isinstance(self.phase_result, Exception):
            raise self.phase_result
        return self.phase_result or ActionResult(
            True, action, "clicked", source_observation_id=observation.observation_id,
            input_issued=True,
        )

    def visual_context_matches(self, observation):
        return self.context_match

    def execute_type_phase2(self, action, observation, *, visual_verified):
        self.type_calls.append((action, observation, visual_verified))
        if isinstance(self.type_result, Exception):
            raise self.type_result
        return self.type_result or ActionResult(
            True, action, "typed", source_observation_id=observation.observation_id,
            input_issued=True,
        )

    def execute_visual_click_phase3(self, action, observation):
        self.result_clicks.append((action, observation))
        if isinstance(self.result_click_result, Exception):
            raise self.result_click_result
        return self.result_click_result or ActionResult(
            True, action, "result clicked", source_observation_id=observation.observation_id,
            input_issued=True,
        )


def clock():
    value = 0.0

    def tick():
        nonlocal value
        value += .01
        return value

    return tick


def reply(payload, choice, confidence=.95):
    options = payload["questions"]["next_action"]["criteria"]
    return {"model": "jev-1.13.0", "answers": {"next_action": {
        "type": "choice", "choice": choice, "confidence": confidence,
        "probabilities": {key: float(key == choice) for key in options},
    }}}


def test_jev_can_emit_typed_grounding_only_in_experimental_schema() -> None:
    client = Mock()
    client.evaluate.side_effect = lambda payload: reply(payload, "grounding_1")
    maker = JevDecisionMaker(client)
    result = maker.decide_hybrid("play Californication", local())
    assert result.status == "grounding"
    assert isinstance(result.grounding_need, VisualGroundingNeed)
    hybrid_options = client.evaluate.call_args.args[0]["questions"]["next_action"]["criteria"]
    assert "grounding_1" in hybrid_options

    client.evaluate.side_effect = lambda payload: reply(payload, "stop")
    maker.decide("play Californication", local())
    normal_options = client.evaluate.call_args.args[0]["questions"]["next_action"]["criteria"]
    assert not any(key.startswith("grounding_") for key in normal_options)


def test_request_target_grounding_is_concise_and_excludes_prior_open_step() -> None:
    client = Mock()
    client.evaluate.side_effect = lambda payload: reply(payload, "grounding_2")
    result = JevDecisionMaker(client).decide_hybrid(
        "play Californication by Red Hot Chili Peppers", local("fresh"),
        (ActionResult(
            True, TypeAction("Californication"), "typed", source_observation_id="old",
        ),),
    )
    assert result.grounding_need is not None
    assert "Californication by Red Hot Chili Peppers" in result.grounding_need.objective
    assert "Open Spotify" not in result.grounding_need.objective


def test_jev_does_not_offer_grounding_until_open_app_has_executed() -> None:
    catalog = MemoryApplicationCatalog((ApplicationCandidate(
        APP_ID, "Spotify", "test", launch_policy="allow",
    ),))
    client = Mock()
    client.evaluate.side_effect = lambda payload: reply(payload, "stop")
    maker = JevDecisionMaker(client, app_catalog=catalog)
    request = "Open Spotify and play Californication"
    maker.decide_hybrid(request, local())
    options = client.evaluate.call_args.args[0]["questions"]["next_action"]["criteria"]
    assert f"open_{APP_ID}" in options
    assert not any(key.startswith("grounding_") for key in options)

    opened = ActionResult(True, OpenAppAction(APP_ID), "opened")
    maker.decide_hybrid(
        request, replace(local("after"), application_id=APP_ID), (opened,),
    )
    options = client.evaluate.call_args.args[0]["questions"]["next_action"]["criteria"]
    assert any(key.startswith("grounding_") for key in options)
    target = options["grounding_1"]["observation_request"]
    assert "search" in target.casefold()


@pytest.mark.parametrize(("foreground_id", "title", "open_expected"), [
    ("", "Terminal", True),
    ("app_2222222222222222", "Other application", True),
    ("app_2222222222222222", "Spotify - similar title only", True),
    (APP_ID, "Arbitrary localized title", False),
])
def test_open_app_options_use_trusted_foreground_identity(
    foreground_id: str, title: str, open_expected: bool,
) -> None:
    catalog = MemoryApplicationCatalog((ApplicationCandidate(
        APP_ID, "Spotify", "test", launch_policy="allow",
    ),))
    client = Mock()
    client.evaluate.side_effect = lambda payload: reply(payload, "stop")
    maker = JevDecisionMaker(client, app_catalog=catalog)
    observation = replace(
        local(), application_id=foreground_id, window_title=title,
    )
    maker.decide_hybrid("Open Spotify and play Californication", observation)
    options = client.evaluate.call_args.args[0]["questions"]["next_action"]["criteria"]
    assert (f"open_{APP_ID}" in options) is open_expected
    if open_expected:
        assert not any(key.startswith("grounding_") for key in options)
    else:
        assert "grounding_1" in options
        assert options["grounding_1"]["observation_request"] == (
            "Find the visible control used to search for playable content."
        )


def test_confirmation_restricted_app_does_not_fall_through_to_grounding() -> None:
    catalog = MemoryApplicationCatalog((ApplicationCandidate(
        APP_ID, "Protected Player", "test", launch_policy="confirm",
    ),))
    maker = JevDecisionMaker(Mock(), app_catalog=catalog)
    _payload, candidates, _stats = maker._prepare(
        "Open Protected Player and play a song", local(), (), include_grounding=True,
    )
    assert not any(isinstance(value, VisualGroundingNeed) for value in candidates.values())
    assert not any(isinstance(value, OpenAppAction) for value in candidates.values())


def test_hybrid_result_reports_exact_safe_option_inventory() -> None:
    catalog = MemoryApplicationCatalog((ApplicationCandidate(
        APP_ID, "Spotify", "test", launch_policy="allow",
    ),))
    client = Mock()
    client.evaluate.side_effect = lambda payload: reply(payload, "stop")
    result = JevDecisionMaker(client, app_catalog=catalog).decide_hybrid(
        "Open Spotify and play Californication",
        replace(local(), application_id=APP_ID),
    )
    assert "visual_grounding_need" in result.offered_option_types
    assert "open_app" not in result.offered_option_types
    assert result.offered_applications == ()
    assert result.offered_grounding_needs
    question = client.evaluate.call_args.args[0]["questions"]["next_action"]
    assert question["type"] == "choice"
    assert isinstance(question["criteria"]["grounding_1"], dict)


def test_low_confidence_preserves_offered_and_selected_grounding_diagnostics() -> None:
    client = Mock()
    client.evaluate.side_effect = lambda payload: reply(
        payload, "grounding_1", confidence=.61,
    )
    result = JevDecisionMaker(client).decide_hybrid("play a requested item", local())
    assert result.status == "grounding"
    assert result.confidence == .61
    assert result.selected_option == "grounding_1"
    assert "visual_grounding_need" in result.offered_option_types
    assert result.offered_grounding_needs[0].objective.startswith("Find the visible control")


def test_low_confidence_trace_shows_grounding_was_offered_and_selected() -> None:
    decision = HybridDecisionResult(
        "needs_human", None, None, .61, "low", selected_option="grounding_1",
        offered_option_types=("press_key", "visual_grounding_need", "finish", "stop"),
        offered_grounding_needs=(VisualGroundingNeed(
            "Find the visible control used to search for playable content.",
            "UIA is insufficient.", 5,
        ),),
    )
    result = HybridDebugAgent(Computer([local()]), Decisions([decision]), clock=clock()).run("play")
    assert result.stop_reason == "low_confidence"
    assert result.trace[0].decision_type == "visual_grounding_need"
    assert result.trace[0].selected_option == "grounding_1"
    assert "visual_grounding_need" in result.trace[0].offered_option_types
    assert result.trace[0].grounding_requested is False


def test_grounding_options_skipped_when_uia_or_focused_editor_is_sufficient() -> None:
    client = Mock()
    client.evaluate.side_effect = lambda payload: reply(payload, "stop")
    maker = JevDecisionMaker(client)
    useful = tuple(
        UIElement(f"c{index}", name, "Button", enabled=True, visible=True)
        for index, name in enumerate(("Search", "Library", "Home"), 1)
    )
    structural = tuple(
        UIElement(f"c{index}", f"Section {index}", "Text", visible=True)
        for index in range(4, 10)
    )
    maker.decide_hybrid("find music", local(elements=useful + structural))
    options = client.evaluate.call_args.args[0]["questions"]["next_action"]["criteria"]
    assert not any(key.startswith("grounding_") for key in options)

    editor = UIElement(
        "c1", "", "Edit", enabled=True, visible=True, focused=True, is_password=False,
    )
    maker.decide_hybrid('type "Californication"', local(elements=(editor,)))
    options = client.evaluate.call_args.args[0]["questions"]["next_action"]["criteria"]
    assert "type_1" in options
    assert not any(key.startswith("grounding_") for key in options)


def test_open_app_executes_then_grounding_returns_fresh_candidates_and_visual_stops() -> None:
    catalog = MemoryApplicationCatalog((ApplicationCandidate(
        APP_ID, "Spotify", "test", launch_policy="allow",
    ),))
    computer = Computer([
        local("before"), replace(local("after"), application_id=APP_ID),
    ])
    decisions = Decisions([
        ready(OpenAppAction(APP_ID)), need(),
        ready(VisualClickAction("hybrid-1", "v1")),
    ])
    agent = HybridDebugAgent(
        computer, decisions, policy=AutonomousActionPolicy(catalog),
        limits=AgentLimits(settle_open_seconds=0, settle_action_seconds=0), clock=clock(),
    )
    result = agent.run("Open Spotify and play Californication by Red Hot Chili Peppers")
    assert result.success and result.stop_reason == "visual_action_blocked_for_debug"
    assert [item[0].kind for item in computer.executed] == ["open_app"]
    assert len(computer.directed_calls) == 1
    assert computer.directed_calls[0].objective == "Find a visible music search field."
    assert result.selected_visual_target is not None
    assert result.selected_visual_target.id == "v1"
    assert result.selected_visual_target.snapshot_id == "hybrid-1"
    assert result.selected_visual_target.blocked_reason == "experimental visual execution disabled"
    assert result.trace[1].grounding_requested is True
    assert result.trace[1].returned_visual_candidates == 1


def test_real_trace_regression_open_once_then_ground_then_block_visual() -> None:
    catalog = MemoryApplicationCatalog((ApplicationCandidate(
        APP_ID, "Spotify", "test", launch_policy="allow",
    ),))
    before = local("before")
    old_foreground = replace(local("launching"), application_id="app_2222222222222222")
    active = replace(local("active"), application_id=APP_ID)
    grounded = replace(hybrid("fresh-visual"), application_id=APP_ID)
    computer = Computer([before, old_foreground, active], directed=grounded)
    call_number = 0

    def evaluate(payload):
        nonlocal call_number
        call_number += 1
        options = payload["questions"]["next_action"]["criteria"]
        if call_number == 1:
            assert f"open_{APP_ID}" in options
            assert not any(key.startswith("grounding_") for key in options)
            return reply(payload, f"open_{APP_ID}")
        if call_number == 2:
            assert f"open_{APP_ID}" not in options
            assert "grounding_1" in options
            return reply(payload, "grounding_1")
        assert "visual_v1" in options
        return reply(payload, "visual_v1")

    maker = JevDecisionMaker(Mock(evaluate=Mock(side_effect=evaluate)), app_catalog=catalog)
    result = HybridDebugAgent(
        computer, maker, policy=AutonomousActionPolicy(catalog),
        limits=AgentLimits(settle_open_seconds=0, settle_action_seconds=0), clock=clock(),
        sleep_fn=lambda _seconds: None, open_app_poll_interval_seconds=.01,
    ).run("Open Spotify and play Californication by Red Hot Chili Peppers")
    assert result.success and result.stop_reason == "visual_action_blocked_for_debug"
    assert [action.kind for action, _observation in computer.executed] == ["open_app"]
    assert len(computer.directed_calls) == 1
    assert result.selected_visual_target is not None
    assert result.selected_visual_target.snapshot_id == "fresh-visual"
    assert result.selected_visual_target.id == "v1"
    assert result.trace[0].offered_option_types.count("open_app") == 1
    assert result.trace[0].target_app_active is True
    assert result.trace[0].observed_application_id == APP_ID
    assert "open_app" not in result.trace[1].offered_option_types
    assert "visual_grounding_need" in result.trace[1].offered_option_types
    assert result.trace[1].offered_grounding_needs[0].max_candidates == 5
    assert result.trace[2].offered_option_types.count("visual_click") == 1


def test_open_app_waits_for_trusted_identity_without_extra_decision_steps() -> None:
    catalog = MemoryApplicationCatalog((ApplicationCandidate(
        APP_ID, "Spotify", "test", launch_policy="allow",
    ),))
    old = replace(
        local("old"), application_id="app_2222222222222222", process_id=111,
        foreground_hwnd=101,
    )
    active = replace(
        local("active"), application_id=APP_ID, window_title="Spotify active",
        process_id=222, foreground_hwnd=202,
    )
    computer = Computer([local("before"), old, active])
    decisions = Decisions([ready(OpenAppAction(APP_ID)), ready(FinishAction("done"))])
    result = HybridDebugAgent(
        computer, decisions, policy=AutonomousActionPolicy(catalog), clock=clock(),
        sleep_fn=lambda _seconds: None, open_app_poll_interval_seconds=.01,
    ).run("Open Spotify")
    assert result.stop_reason == "finished"
    assert len(decisions.calls) == 2
    assert decisions.calls[1][1].observation_id == "active"
    assert decisions.calls[1][1].process_id == 222
    assert [action.kind for action, _ in computer.executed] == ["open_app"]
    assert result.trace[0].observed_application_id == APP_ID
    assert result.trace[0].observed_window_title == "Spotify active"
    assert result.trace[0].target_application_id == APP_ID
    assert result.trace[0].target_app_active is True
    assert result.trace[0].open_app_activation_wait_ms >= 0


def test_open_app_activation_timeout_preserves_actual_foreground_state() -> None:
    catalog = MemoryApplicationCatalog((ApplicationCandidate(
        APP_ID, "Spotify", "test", launch_policy="allow",
    ),))
    old = replace(local("still-old"), application_id="app_2222222222222222")
    computer = Computer([local("before"), old])
    decisions = Decisions([ready(OpenAppAction(APP_ID)), ready(FinishAction("stop"))])
    result = HybridDebugAgent(
        computer, decisions, policy=AutonomousActionPolicy(catalog), clock=clock(),
        sleep_fn=lambda _seconds: None, open_app_activation_timeout_seconds=.03,
        open_app_poll_interval_seconds=.01,
    ).run("Open Spotify")
    assert result.stop_reason == "finished"
    assert decisions.calls[1][1].observation_id == "still-old"
    assert result.trace[0].target_app_active is False
    assert result.trace[0].observed_application_id == "app_2222222222222222"


def test_initial_query_grounding_choice_id_is_opaque_and_result_is_pruned() -> None:
    client = Mock()
    client.evaluate.side_effect = lambda payload: reply(payload, "grounding_1")
    button = UIElement("c1", "Home", "Button", enabled=True, visible=True)
    result = JevDecisionMaker(client).decide_hybrid(
        "play Californication by Red Hot Chili Peppers", local(elements=(button,)),
    )
    criteria = client.evaluate.call_args.args[0]["questions"]["next_action"]["criteria"]
    grounding_ids = [key for key in criteria if key.startswith("grounding_")]
    assert grounding_ids == ["grounding_1"]
    assert all(len(key) <= 32 and "Californication" not in key for key in grounding_ids)
    assert not any(key.startswith("click_") for key in criteria)
    assert not any(key.startswith("key_") for key in criteria)
    assert {"finish", "stop"}.issubset(criteria)
    assert len(criteria) == len(set(criteria))
    assert result.grounding_need == result.offered_grounding_needs[0]
    assert result.task_progress.query_required is True
    assert result.task_progress.query_entered_or_submitted is False
    assert result.task_progress.result_grounding_eligible is False
    summary = result.option_filter_summary
    assert summary.uia_click_candidates_seen == 1
    assert summary.uia_click_options_offered == 0
    assert summary.press_key_capabilities_seen == 6
    assert summary.press_key_options_offered == 0
    assert summary.filtered_reasons["insufficient_semantic_relevance"] == 1


def test_semantically_relevant_uia_click_may_be_offered() -> None:
    search = UIElement("c1", "Search", "Button", enabled=True, visible=True)
    _result, criteria = _grounding_options(
        "search for query", local(elements=(search,)),
    )
    assert "click_c1" in criteria


def test_focused_editor_with_literal_offers_type_and_editing_keys() -> None:
    editor = UIElement(
        "c1", "Search", "Edit", enabled=True, visible=True, focused=True,
        is_password=False, observed_text="old query",
    )
    result, criteria = _grounding_options('type "new query"', local(elements=(editor,)))
    assert "type_1" in criteria
    assert "key_ctrl_a" in criteria
    assert "key_ctrl_c" in criteria
    assert result.option_filter_summary.type_options_offered == 1


def test_no_focused_editor_omits_type_and_ctrl_a() -> None:
    _result, criteria = _grounding_options('type "new query"', local())
    assert "type_1" not in criteria
    assert "key_ctrl_a" not in criteria


def test_enter_requires_plausible_submit_context() -> None:
    notes = UIElement(
        "c1", "Notes", "Edit", enabled=True, visible=True, focused=True,
        is_password=False,
    )
    _result, criteria = _grounding_options('type "hello"', local(elements=(notes,)))
    assert "key_enter" not in criteria

    search = replace(notes, name="Search")
    _result, criteria = _grounding_options('search for "hello"', local(elements=(search,)))
    assert "key_enter" in criteria


def test_escape_requires_modal_or_dismiss_context() -> None:
    _result, criteria = _grounding_options("find settings", local())
    assert "key_escape" not in criteria
    dialog = replace(local(), control_type="Dialog")
    _result, criteria = _grounding_options("find settings", dialog)
    assert "key_escape" in criteria


def test_tab_navigation_is_omitted_without_bounded_recovery_evidence() -> None:
    focused = UIElement(
        "c1", "Search", "Button", enabled=True, visible=True, focused=True,
    )
    _result, criteria = _grounding_options("search for query", local(elements=(focused,)))
    assert "key_tab" not in criteria
    assert "key_shift_tab" not in criteria


def test_spotify_like_generic_state_contains_no_irrelevant_local_actions() -> None:
    structural = tuple(
        UIElement(f"c{index}", "", "Button", enabled=True, visible=True)
        for index in range(1, 4)
    )
    result, criteria = _grounding_options(
        "play a requested song", local(elements=structural), choice="grounding_1",
    )
    assert {key for key in criteria if key.startswith("grounding_")} == {"grounding_1"}
    assert "stop" in criteria and "finish" in criteria
    assert not any(key.startswith("click_") for key in criteria)
    assert not any(key.startswith("key_") for key in criteria)
    assert result.offered_option_types == ("finish", "stop", "visual_grounding_need")
    assert result.option_filter_summary.uia_click_candidates_seen == 3
    assert result.option_filter_summary.uia_click_options_offered == 0


def _grounding_options(
    request: str, observation: Observation, history=(), choice: str = "stop",
):
    client = Mock()
    client.evaluate.side_effect = lambda payload: reply(payload, choice)
    result = JevDecisionMaker(client).decide_hybrid(request, observation, history)
    criteria = client.evaluate.call_args.args[0]["questions"]["next_action"]["criteria"]
    return result, criteria


@pytest.mark.parametrize("user_request", [
    "search for ambient music",
    "find product headphones",
    "find file report.txt",
])
def test_query_stage_pruning_is_application_independent(user_request: str) -> None:
    result, criteria = _grounding_options(user_request, local())
    assert "grounding_1" in criteria
    assert "grounding_2" not in criteria
    assert result.task_progress.query_required is True
    assert result.task_progress.result_grounding_eligible is False


def test_successful_validated_type_on_prior_snapshot_enables_result_grounding() -> None:
    history = (ActionResult(
        True, TypeAction("Californication"), "typed", source_observation_id="before-type",
    ),)
    result, criteria = _grounding_options(
        "play Californication by Red Hot Chili Peppers", local("after-type"), history,
        choice="grounding_2",
    )
    assert "grounding_1" not in criteria
    assert "grounding_2" in criteria
    assert result.grounding_need is not None
    assert result.task_progress.query_entered_or_submitted is True
    assert result.task_progress.result_grounding_eligible is True


@pytest.mark.parametrize("history", [
    (ActionResult(True, TypeAction("query"), "proposed only"),),
    (ActionResult(
        False, TypeAction("query"), "failed", source_observation_id="before-type",
    ),),
    (ActionResult(
        True, VisualClickAction("visual", "v1"), "blocked",
        source_observation_id="visual",
    ),),
])
def test_untrusted_or_failed_events_do_not_advance_query_progress(history) -> None:
    result, criteria = _grounding_options("search for query", local("fresh"), history)
    assert "grounding_1" in criteria
    assert "grounding_2" not in criteria
    assert result.task_progress.query_entered_or_submitted is False


def test_visual_search_candidate_does_not_advance_query_progress() -> None:
    result, criteria = _grounding_options("search for query", hybrid("visual-search"))
    assert result.task_progress.query_entered_or_submitted is False
    assert result.task_progress.result_grounding_eligible is False
    assert "grounding_2" not in criteria


def test_grounding_remains_a_typed_choice_and_is_not_auto_selected() -> None:
    result, criteria = _grounding_options("search for query", local(), choice="stop")
    assert "grounding_1" in criteria
    assert result.selected_option == "stop"
    assert result.grounding_need is None


def test_safe_choice_probability_diagnostics_are_bounded_to_offered_ids() -> None:
    client = Mock()
    client.evaluate.side_effect = lambda payload: reply(payload, "grounding_1", confidence=.67)
    result = JevDecisionMaker(client).decide_hybrid("search for query", local())
    offered = set(client.evaluate.call_args.args[0]["questions"]["next_action"]["criteria"])
    assert result.status == "grounding"
    assert result.confidence == .67
    assert 0 < len(result.choice_probabilities) <= 50
    assert {item.option_id for item in result.choice_probabilities}.issubset(offered)
    assert all(type(item.probability) is float for item in result.choice_probabilities)

    computer = Computer([local()])
    stopped = HybridDebugAgent(computer, JevDecisionMaker(client), clock=clock()).run(
        "search for query",
    )
    assert stopped.stop_reason == "repeated_grounding"
    assert stopped.trace[0].choice_probabilities
    assert stopped.trace[0].task_progress.result_grounding_eligible is False
    assert len(computer.directed_calls) == 1


def test_unknown_probability_id_fails_closed_and_is_not_traced() -> None:
    def response(payload):
        value = reply(payload, "grounding_1")
        value["answers"]["next_action"]["probabilities"]["unknown_provider_id"] = 0.0
        return value

    result = JevDecisionMaker(Mock(evaluate=Mock(side_effect=response))).decide_hybrid(
        "search for query", local(),
    )
    assert result.status == "error"
    assert result.decision_error_category == "serialization_schema_mismatch"
    assert result.choice_probabilities == ()
    assert "unknown_provider_id" not in repr(result.response_shape_summary)


@pytest.mark.parametrize(("answer", "category"), [
    ({"unexpected": {}}, "unexpected_response_envelope"),
    ({"model": "jev-1.13.0", "answers": {}}, "missing_choice_result"),
    ({"model": "jev-1.13.0", "answers": {"next_action": {
        "type": "text", "choice": "grounding_1", "confidence": "secret-value",
    }}}, "wrong_primitive_type"),
])
def test_hybrid_malformed_response_exposes_only_safe_shape(answer, category) -> None:
    result = JevDecisionMaker(Mock(evaluate=Mock(return_value=answer))).decide_hybrid(
        "play music", local(),
    )
    assert result.status == "error"
    assert result.decision_error_category == category
    assert result.expected_primitive == "choice"
    assert result.offered_option_count > 0
    assert "secret-value" not in repr(result.response_shape_summary)


def test_hybrid_retry_diagnostics_preserve_first_failure_and_result() -> None:
    calls = 0

    def evaluate(payload):
        nonlocal calls
        calls += 1
        value = reply(payload, "finish")
        if calls == 1:
            value["answers"]["next_action"]["probabilities"]["stop"] = .2
        return value

    result = HybridDebugAgent(
        Computer([local()]), JevDecisionMaker(Mock(evaluate=Mock(side_effect=evaluate))),
        clock=clock(),
    ).run("finish")
    assert result.success
    assert result.trace[0].retry_attempted is True
    assert result.trace[0].decision_error_category is None
    assert result.trace[0].retry_result_category == "success"
    assert len(result.trace[0].decision_attempts) == 2
    first, second = result.trace[0].decision_attempts
    assert first.attempt == 1 and first.result == "invalid_response"
    assert first.error_category == "invalid_probability_distribution"
    assert first.probability_sum == pytest.approx(1.2)
    assert second.attempt == 2 and second.result == "success"
    assert second.error_category is None
    assert second.probability_sum == pytest.approx(1.0)


def test_selected_probability_is_diagnostic_but_confidence_still_gates() -> None:
    def evaluate(payload):
        value = reply(payload, "grounding_1", confidence=.78)
        probabilities = value["answers"]["next_action"]["probabilities"]
        probabilities.update({key: 0.0 for key in probabilities})
        probabilities["grounding_1"] = .80
        probabilities["stop"] = .20
        return value

    computer = Computer([local()])
    result = HybridDebugAgent(
        computer, JevDecisionMaker(Mock(evaluate=Mock(side_effect=evaluate))), clock=clock(),
    ).run("search for query")
    assert len(computer.directed_calls) == 1
    assert result.trace[0].jev_confidence == .78
    assert result.trace[0].selected_option_probability == .80
    assert result.trace[0].release_policy == "bounded_observation_policy"
    assert result.trace[0].observation_policy.eligible is True


def test_small_floating_point_probability_error_is_accepted() -> None:
    def evaluate(payload):
        value = reply(payload, "grounding_1")
        value["answers"]["next_action"]["probabilities"]["stop"] = 0.0000001
        return value

    result = JevDecisionMaker(Mock(evaluate=Mock(side_effect=evaluate))).decide_hybrid(
        "search for query", local(),
    )
    assert result.status == "grounding"
    assert result.decision_attempts[0].probability_sum == pytest.approx(1.0000001)


def test_materially_invalid_probability_total_is_rejected() -> None:
    def evaluate(payload):
        value = reply(payload, "grounding_1")
        value["answers"]["next_action"]["probabilities"]["stop"] = .01
        return value

    result = JevDecisionMaker(Mock(evaluate=Mock(side_effect=evaluate))).decide_hybrid(
        "search for query", local(),
    )
    assert result.status == "error"
    assert result.decision_error_category == "invalid_probability_distribution"


@pytest.mark.parametrize("bad", [-.1, 1.1, float("nan"), float("inf")])
def test_unsafe_probability_values_remain_rejected(bad: float) -> None:
    def evaluate(payload):
        value = reply(payload, "grounding_1")
        value["answers"]["next_action"]["probabilities"]["stop"] = bad
        return value

    result = JevDecisionMaker(Mock(evaluate=Mock(side_effect=evaluate))).decide_hybrid(
        "search for query", local(),
    )
    assert result.status == "error"
    assert result.choice_probabilities == ()


def test_visual_click_never_reaches_executor_even_if_policy_would_allow() -> None:
    visual = hybrid()
    visual = replace(visual, visual_elements=(replace(visual.visual_elements[0], confidence=.99),))
    computer = Computer([local()], directed=visual)
    decisions = Decisions([need(), ready(VisualClickAction("hybrid-1", "v1"))])
    result = HybridDebugAgent(
        computer, decisions, policy=BasicActionPolicy(), clock=clock(),
    ).run("play Californication")
    assert result.stop_reason == "visual_action_blocked_for_debug"
    assert computer.executed == []
    assert result.selected_visual_target is not None
    assert result.selected_visual_target.normal_safety_would_allow is True


@pytest.mark.parametrize(("candidate", "expected"), [
    (VisualGroundingNeed("Find search.", "UIA insufficient."), DecisionEffect.OBSERVE),
    (OpenAppAction(APP_ID), DecisionEffect.ACT),
    (TypeAction("text"), DecisionEffect.ACT),
    (ClickAction("c1"), DecisionEffect.ACT),
    (PressKeyAction(("tab",)), DecisionEffect.ACT),
    (VisualClickAction("snapshot", "v1"), DecisionEffect.ACT),
    (FinishAction("done"), DecisionEffect.TERMINAL),
    (None, DecisionEffect.TERMINAL),
])
def test_decision_effect_is_explicitly_typed(candidate, expected) -> None:
    assert decision_effect(candidate) is expected


@pytest.mark.parametrize("action", [
    OpenAppAction(APP_ID), TypeAction("text"), ClickAction("c1"),
    PressKeyAction(("tab",)), VisualClickAction("local-1", "v1"),
])
def test_every_act_below_confidence_gate_remains_blocked(action) -> None:
    computer = Computer([local()])
    result = HybridDebugAgent(
        computer, Decisions([ready(action, .79)]), clock=clock(),
    ).run("perform action")
    assert result.stop_reason == "low_confidence"
    assert computer.executed == []


def test_act_at_confidence_gate_may_reach_normal_safety_and_executor() -> None:
    editor = UIElement(
        "c1", "Editor", "Edit", enabled=True, visible=True, focused=True,
        is_password=False,
    )
    computer = Computer([local(elements=(editor,)), local("fresh")])
    result = HybridDebugAgent(
        computer, Decisions([ready(TypeAction("text"), .80), ready(FinishAction("done"))]),
        limits=AgentLimits(settle_action_seconds=0), clock=clock(),
    ).run('type "text"')
    assert result.stop_reason == "finished"
    assert [action.kind for action, _ in computer.executed] == ["type"]


def test_low_confidence_observe_runs_once_then_high_visual_act_is_blocked() -> None:
    calls = 0

    def evaluate(payload):
        nonlocal calls
        calls += 1
        options = payload["questions"]["next_action"]["criteria"]
        if "visual_v1" in options:
            return reply(payload, "visual_v1", .90)
        return reply(payload, "grounding_1", .54)

    computer = Computer([local()], directed=hybrid())
    result = HybridDebugAgent(
        computer, JevDecisionMaker(Mock(evaluate=Mock(side_effect=evaluate))), clock=clock(),
    ).run("search for query")
    assert result.stop_reason == "visual_action_blocked_for_debug"
    assert len(computer.directed_calls) == 1
    assert computer.executed == []
    assert result.trace[0].decision_effect is DecisionEffect.OBSERVE
    assert result.trace[0].release_policy == "bounded_observation_policy"
    assert result.trace[0].observation_policy.eligible is True
    assert result.trace[1].decision_effect is DecisionEffect.ACT
    assert result.trace[1].release_policy == "action_confidence_gate"


def test_multiple_directed_candidates_survive_into_jev_visual_options() -> None:
    second = replace(
        hybrid("snapshot-b"),
        visual_elements=(
            replace(hybrid().visual_elements[0], id="v1"),
            replace(hybrid().visual_elements[0], id="v2", label="Library"),
        ),
        visual_grounding_status=VisualGroundingStatus.SUCCESS_WITH_CANDIDATES,
        visual_pipeline=VisualPipelineDiagnostic(5, 2, 2, 2, 2, 2),
    )
    computer = Computer([local("snapshot-a")], directed=second)
    call = 0

    def evaluate(payload):
        nonlocal call
        call += 1
        options = payload["questions"]["next_action"]["criteria"]
        if call == 1:
            return reply(payload, "grounding_1", .54)
        assert {"visual_v1", "visual_v2"}.issubset(options)
        return reply(payload, "visual_v2", .90)

    result = HybridDebugAgent(
        computer, JevDecisionMaker(Mock(evaluate=Mock(side_effect=evaluate))), clock=clock(),
    ).run("search for query")
    assert result.stop_reason == "visual_action_blocked_for_debug"
    assert len(computer.directed_calls) == 1
    assert result.selected_visual_target is not None
    assert result.selected_visual_target.id == "v2"
    assert result.selected_visual_target.snapshot_id == "snapshot-b"
    assert result.trace[1].visual_pipeline is not None
    assert result.trace[1].visual_pipeline.jev_visual_option_count == 2
    assert computer.executed == []


def test_valid_empty_directed_result_stops_without_second_provider_call() -> None:
    empty = replace(
        hybrid("snapshot-b"), visual_elements=(), visual_returned_elements=0,
        visual_grounding_status=VisualGroundingStatus.SUCCESS_EMPTY,
        visual_pipeline=VisualPipelineDiagnostic(5, 0, 0, 0, 0, 0),
    )
    computer = Computer([local("snapshot-a")], directed=empty)
    result = HybridDebugAgent(
        computer, Decisions([need(confidence=.54)]), clock=clock(),
    ).run("search for query")
    assert result.stop_reason == "visual_grounding_empty"
    assert len(computer.directed_calls) == 1
    assert result.trace[0].visual_grounding_status is VisualGroundingStatus.SUCCESS_EMPTY
    assert result.trace[0].terminal_reason == "visual_grounding_empty"
    assert computer.executed == []


def test_visual_readiness_timeout_is_distinct_from_grounding_empty() -> None:
    timed_out = replace(
        hybrid("snapshot-b"), visual_elements=(), visual_provider=None,
        visual_latency_ms=None, visual_readiness=VisualReadinessResult(
            False, VisualReadinessReason.TIMEOUT, 4, 3000,
        ),
    )
    computer = Computer([local("snapshot-a")], directed=timed_out)
    result = HybridDebugAgent(
        computer, Decisions([need(confidence=.54)]), clock=clock(),
    ).run("search for query")
    assert result.stop_reason == "visual_readiness_timeout"
    assert result.trace[0].terminal_reason == "visual_readiness_timeout"
    assert result.trace[0].visual_provider_latency_ms is None
    assert result.trace[0].visual_provider_call_count == 0
    assert computer.executed == []


def test_phase1_executes_one_visual_click_observes_fresh_state_and_stops() -> None:
    post = local("post-click", elements=(UIElement(
        "c1", "Search", "Edit", enabled=True, visible=True, focused=True,
        is_password=False,
    ),))
    computer = Computer([local("before"), post], directed=phase_hybrid("visual-snapshot"))
    decisions = Decisions([
        need(), ready(VisualClickAction("visual-snapshot", "v1"), .80),
    ])
    result = HybridClickDebugAgent(
        computer, decisions, limits=AgentLimits(settle_action_seconds=0), clock=clock(),
        sleep_fn=lambda _seconds: None,
    ).run("focus the music search field")
    assert result.success and result.stop_reason == "visual_click_phase1_complete"
    assert len(computer.phase_clicks) == 1
    assert len(decisions.calls) == 2
    assert result.visual_execution.remaining_budget == 0
    assert result.visual_execution.input_issued is True
    assert result.visual_execution.post_click_observation_obtained is True
    assert result.visual_execution.effect_evidence.new_snapshot_id == "post-click"
    assert result.visual_execution.effect_evidence.snapshot_changed is True
    assert result.visual_execution.effect_evidence.editable_control_focused is True


def test_phase1_confidence_below_threshold_never_clicks() -> None:
    computer = Computer([local()], directed=phase_hybrid())
    result = HybridClickDebugAgent(
        computer, Decisions([need(), ready(VisualClickAction("hybrid-1", "v1"), .79)]),
        clock=clock(),
    ).run("focus search")
    assert result.stop_reason == "low_confidence"
    assert computer.phase_clicks == []


@pytest.mark.parametrize("action", [
    VisualClickAction("older", "v1"), VisualClickAction("hybrid-1", "missing"),
])
def test_phase1_stale_or_missing_target_never_clicks(action) -> None:
    computer = Computer([local()], directed=phase_hybrid())
    result = HybridClickDebugAgent(
        computer, Decisions([need(), ready(action)]), clock=clock(),
    ).run("focus search")
    assert result.stop_reason == "stale_visual_target"
    assert computer.phase_clicks == []


def test_phase1_consequential_target_is_denied() -> None:
    dangerous = replace(hybrid(), visual_elements=(VisualElement(
        "v1", "Confirm purchase", "button", Rect(100, 100, 300, 180), None, True,
    ),))
    computer = Computer([local()], directed=dangerous)
    result = HybridClickDebugAgent(
        computer, Decisions([need(), ready(VisualClickAction("hybrid-1", "v1"))]),
        clock=clock(),
    ).run("open the menu")
    assert result.stop_reason == "visual_click_safety_denied"
    assert computer.phase_clicks == []


def test_phase1_executor_exception_consumes_budget_without_retry() -> None:
    computer = Computer([local("before"), local("after")], directed=phase_hybrid())
    computer.phase_result = RuntimeError("input failed")
    decisions = Decisions([need(), ready(VisualClickAction("hybrid-1", "v1"))])
    result = HybridClickDebugAgent(
        computer, decisions, limits=AgentLimits(settle_action_seconds=0),
        clock=clock(), sleep_fn=lambda _seconds: None,
    ).run("focus search")
    assert result.stop_reason == "visual_click_input_failed"
    assert result.visual_execution.remaining_budget == 0
    assert len(computer.phase_clicks) == 1 and len(decisions.calls) == 2


def test_phase2_strong_local_types_once_observes_and_stops() -> None:
    focused = UIElement(
        "c1", "Search", "Edit", enabled=True, visible=True, focused=True,
        is_password=False,
    )
    post_click = local("post-click", elements=(focused,))
    post_type = local("post-type", elements=(replace(
        focused, observed_text="Californication",
    ),))
    computer = Computer(
        [local("before"), post_click, post_type], directed=phase_hybrid("visual"),
    )
    result = HybridTypeDebugAgent(
        computer, Decisions([need(), ready(VisualClickAction("visual", "v1"), .98)]),
        limits=AgentLimits(settle_action_seconds=0), clock=clock(),
        sleep_fn=lambda _seconds: None,
    ).run("Open Spotify and play Californication by Red Hot Chili Peppers")
    assert result.success and result.stop_reason == "visual_type_phase2_complete"
    assert len(computer.phase_clicks) == 1 and len(computer.type_calls) == 1
    assert computer.type_calls[0][0] == TypeAction("Californication")
    assert computer.type_calls[0][2] is False
    assert result.phase2_execution.literal_type_budget_remaining == 0
    assert result.phase2_execution.type_readiness.evidence == "strong_local"
    assert result.phase2_execution.post_type.fresh_snapshot is True
    assert result.phase2_execution.post_type.literal_match is True
    assert computer.query_submits == []


def test_phase2_visual_verification_matching_target_releases_type() -> None:
    clicked = phase_hybrid("clicked")
    verified = phase_hybrid("verified")
    computer = Computer(
        [local("before"), local("post-click"), local("post-type")],
        directed=[clicked, verified],
    )
    result = HybridTypeDebugAgent(
        computer, Decisions([need(), ready(VisualClickAction("clicked", "v1"), .98)]),
        limits=AgentLimits(settle_action_seconds=0), clock=clock(),
        sleep_fn=lambda _seconds: None,
    ).run("Open Spotify and play Californication by Red Hot Chili Peppers")
    assert result.stop_reason == "visual_type_phase2_complete"
    assert len(computer.directed_calls) == 2
    assert len(computer.type_calls) == 1 and computer.type_calls[0][2] is True
    assert result.phase2_execution.type_readiness.evidence == "strong_visual"
    assert result.phase2_execution.type_readiness.verification_provider_called is True


@pytest.mark.parametrize("verification", [
    replace(phase_hybrid("verified"), visual_elements=(replace(
        phase_hybrid("verified").visual_elements[0], rectangle=Rect(700, 300, 900, 400),
    ),)),
    replace(phase_hybrid("verified"), visual_elements=()),
    replace(phase_hybrid("verified"), visual_elements=(),
            visual_provider_error=ProviderErrorDiagnostic("timeout")),
])
def test_phase2_visual_verification_failure_never_types(verification) -> None:
    computer = Computer(
        [local("before"), local("post-click")],
        directed=[phase_hybrid("clicked"), verification],
    )
    result = HybridTypeDebugAgent(
        computer, Decisions([need(), ready(VisualClickAction("clicked", "v1"))]),
        limits=AgentLimits(settle_action_seconds=0), clock=clock(),
        sleep_fn=lambda _seconds: None,
    ).run("Open Spotify and play Californication by Red Hot Chili Peppers")
    assert result.stop_reason == "visual_type_readiness_insufficient"
    assert computer.type_calls == []


def test_phase2_context_or_credentials_prevent_type() -> None:
    password = UIElement(
        "c1", "", "Edit", enabled=True, visible=True, focused=True, is_password=True,
    )
    computer = Computer(
        [local("before"), local("post-click", elements=(password,))],
        directed=[phase_hybrid("clicked"), phase_hybrid("verified")],
    )
    computer.context_match = False
    result = HybridTypeDebugAgent(
        computer, Decisions([need(), ready(VisualClickAction("clicked", "v1"))]),
        limits=AgentLimits(settle_action_seconds=0), clock=clock(),
        sleep_fn=lambda _seconds: None,
    ).run("Open Spotify and play Californication by Red Hot Chili Peppers")
    assert not result.success and computer.type_calls == []


def test_phase2_type_exception_consumes_budget_without_retry_or_enter() -> None:
    focused = UIElement(
        "c1", "Search", "Edit", enabled=True, visible=True, focused=True,
        is_password=False,
    )
    computer = Computer(
        [local("before"), local("post-click", elements=(focused,)), local("post-type")],
        directed=phase_hybrid("clicked"),
    )
    computer.type_result = RuntimeError("type failed")
    result = HybridTypeDebugAgent(
        computer, Decisions([need(), ready(VisualClickAction("clicked", "v1"))]),
        limits=AgentLimits(settle_action_seconds=0), clock=clock(),
        sleep_fn=lambda _seconds: None,
    ).run("Open Spotify and play Californication by Red Hot Chili Peppers")
    assert result.stop_reason == "visual_type_input_failed"
    assert result.phase2_execution.literal_type_budget_remaining == 0
    assert len(computer.type_calls) == 1
    assert all(not isinstance(action, PressKeyAction) for action, _obs in computer.executed)


def _phase3_run(result_observation=None, confidence=.95):
    focused = UIElement(
        "c1", "Search", "Edit", enabled=True, visible=True, focused=True, is_password=False,
    )
    computer = Computer([
        local("before"), local("post-click", elements=(focused,)),
        local("post-type", elements=(replace(focused, observed_text="Californication"),)),
        local("post-result"),
    ], directed=phase_hybrid("search-snapshot"))
    grounded = result_observation or hybrid("result-snapshot")
    computer.result_directed_result = (
        grounded if grounded.result_readiness is not None else replace(
            grounded, result_readiness=ResultReadinessResult(
                True, "changed_and_quiet_stable", 3, 1600, True, True, True, True,
            ),
        )
    )
    decisions = Decisions([
        need(), ready(VisualClickAction("search-snapshot", "v1"), .98),
    ])
    decisions.result_decision = ready(
        VisualClickAction("result-snapshot", "v1"), confidence,
    )
    result = HybridResultDebugAgent(
        computer, decisions, limits=AgentLimits(settle_action_seconds=0),
        clock=clock(), sleep_fn=lambda _seconds: None,
    ).run("Open Spotify and play Californication by Red Hot Chili Peppers")
    return result, computer, decisions


def test_phase3_exact_result_executes_once_observes_and_stops() -> None:
    result, computer, decisions = _phase3_run()
    assert result.success and result.stop_reason == "visual_result_phase3_complete"
    assert len(computer.phase_clicks) == 1
    assert len(computer.type_calls) == 1
    assert len(computer.query_submits) == 1
    assert len(computer.result_clicks) == 1
    assert len(computer.result_directed_calls) == 1
    assert len(decisions.result_calls) == 1
    assert result.phase3_execution.visual_click_budget_remaining == 0
    assert result.phase3_execution.literal_type_budget_remaining == 0
    assert result.phase3_execution.result_selection_budget_remaining == 0
    assert result.phase3_execution.post_result_fresh is True
    assert all(not isinstance(action, PressKeyAction) for action, _ in computer.executed)


def test_jev_result_choice_exposes_only_bounded_result_finish_and_stop() -> None:
    client = Mock()
    client.evaluate.side_effect = lambda payload: reply(payload, "visual_v1", .80)
    maker = JevDecisionMaker(client)
    result = maker.decide_result_selection(
        "play Californication by Red Hot Chili Peppers", hybrid("result-snapshot"),
    )
    assert result.status == "ready" and result.confidence == .80
    criteria = client.evaluate.call_args.args[0]["questions"]["next_action"]["criteria"]
    assert set(criteria) == {"visual_v1", "finish", "stop"}
    assert isinstance(result.action, VisualClickAction)


def test_phase3_low_jev_confidence_never_clicks_result() -> None:
    result, computer, _ = _phase3_run(confidence=.79)
    assert result.stop_reason == "result_decision_rejected"
    assert computer.result_clicks == []


def test_phase3_query_submit_is_exactly_one_enter() -> None:
    result, computer, _ = _phase3_run()
    assert result.success
    assert len(computer.query_submits) == 1
    action = computer.query_submits[0][0]
    assert action == QuerySubmitAction() and action.key == "enter"
    assert result.phase3_execution.query_submit_budget_remaining == 0
    assert result.phase3_execution.query_submit_release_policy == "bounded_query_submit_policy"


def test_phase3_query_submit_failure_consumes_budget_without_grounding_or_retry() -> None:
    focused = UIElement("c1", "Search", "Edit", enabled=True, visible=True,
                        focused=True, is_password=False)
    computer = Computer([local("before"), local("pc", elements=(focused,)), local("pt")],
                        directed=phase_hybrid("search-snapshot"))
    computer.query_submit_result = RuntimeError("input failed")
    decisions = Decisions([need(), ready(VisualClickAction("search-snapshot", "v1"))])
    result = HybridResultDebugAgent(
        computer, decisions, limits=AgentLimits(settle_action_seconds=0),
        clock=clock(), sleep_fn=lambda _: None,
    ).run("Open Spotify and play Californication by Red Hot Chili Peppers")
    assert result.stop_reason == "query_submit_input_failed"
    assert len(computer.query_submits) == 1
    assert computer.result_directed_calls == [] and computer.result_clicks == []
    assert result.phase3_execution.query_submit_budget_remaining == 0


@pytest.mark.parametrize("change", ["hwnd", "pid", "application", "credential"])
def test_phase3_context_change_prevents_enter(change) -> None:
    focused = UIElement("c1", "Search", "Edit", enabled=True, visible=True,
                        focused=True, is_password=False)
    computer = Computer([local("before"), local("pc", elements=(focused,)), local("pt")],
                        directed=phase_hybrid("search-snapshot"))
    baseline = phase_hybrid("pre-submit")
    if change == "hwnd":
        baseline = replace(baseline, screenshot=replace(baseline.screenshot, window_handle=99))
    elif change == "pid":
        baseline = replace(baseline, process_id=99)
    elif change == "application":
        baseline = replace(baseline, application_id="different")
    else:
        baseline = replace(baseline, elements=(UIElement(
            "c1", "", "Edit", enabled=True, visible=True, focused=True, is_password=True,
        ),))
    computer.result_baselines[0] = baseline
    result = HybridResultDebugAgent(
        computer, Decisions([need(), ready(VisualClickAction("search-snapshot", "v1"))]),
        limits=AgentLimits(settle_action_seconds=0), clock=clock(), sleep_fn=lambda _: None,
    ).run("Open Spotify and play Californication by Red Hot Chili Peppers")
    assert result.stop_reason == "query_submit_context_changed"
    assert computer.query_submits == [] and computer.result_directed_calls == []


def test_phase3_readiness_timeout_after_submit_never_clicks_result() -> None:
    timed_out = replace(
        hybrid("result-snapshot"), visual_elements=(), visual_provider_call_count=0,
        result_readiness=ResultReadinessResult(
            False, "timeout_before_quiet_stable", 8, 5000,
            False, True, False, True,
        ),
    )
    result, computer, decisions = _phase3_run(timed_out)
    assert result.stop_reason == "result_readiness_timeout_after_submit"
    assert len(computer.query_submits) == 1
    assert computer.result_clicks == [] and decisions.result_calls == []
    assert result.phase3_execution.provider_call_count == 0


def test_phase3_non_search_visual_target_never_submits() -> None:
    focused = UIElement("c1", "Search", "Edit", enabled=True, visible=True,
                        focused=True, is_password=False)
    target = replace(phase_hybrid("search-snapshot"), visual_elements=(replace(
        phase_hybrid("search-snapshot").visual_elements[0], role="button",
    ),))
    computer = Computer([local("before"), local("pc", elements=(focused,)), local("pt")],
                        directed=target)
    result = HybridResultDebugAgent(
        computer, Decisions([need(), ready(VisualClickAction("search-snapshot", "v1"))]),
        limits=AgentLimits(settle_action_seconds=0), clock=clock(), sleep_fn=lambda _: None,
    ).run("Open Spotify and play Californication by Red Hot Chili Peppers")
    assert result.stop_reason == "query_submit_not_eligible"
    assert computer.query_submits == []


def test_phase3_type_without_input_issued_never_submits() -> None:
    focused = UIElement("c1", "Search", "Edit", enabled=True, visible=True,
                        focused=True, is_password=False)
    computer = Computer([local("before"), local("pc", elements=(focused,)), local("pt")],
                        directed=phase_hybrid("search-snapshot"))
    computer.type_result = ActionResult(
        True, TypeAction("Californication"), "reported success without input", input_issued=False,
    )
    result = HybridResultDebugAgent(
        computer, Decisions([need(), ready(VisualClickAction("search-snapshot", "v1"))]),
        limits=AgentLimits(settle_action_seconds=0), clock=clock(), sleep_fn=lambda _: None,
    ).run("Open Spotify and play Californication by Red Hot Chili Peppers")
    assert result.stop_reason == "query_submit_not_eligible"
    assert computer.query_submits == []


@pytest.mark.parametrize("elements,reason", [
    ((), "result_grounding_empty"),
    ((VisualElement("v1", "Different Song", "card", Rect(1, 1, 20, 20), None, True),),
     "result_identity_mismatch"),
    ((VisualElement("v1", "Californication", "card", Rect(1, 1, 20, 20), None, True,
                    parent="Different Artist"),), "result_identity_mismatch"),
    ((VisualElement("v1", "Buy Californication", "card", Rect(1, 1, 20, 20), None, True,
                    parent="Red Hot Chili Peppers"),), "result_identity_mismatch"),
])
def test_phase3_rejects_empty_mismatch_conflict_and_consequential(elements, reason) -> None:
    observed = replace(hybrid("result-snapshot"), visual_elements=elements)
    result, computer, _ = _phase3_run(observed)
    assert result.stop_reason == reason
    assert computer.result_clicks == []


def test_phase3_missing_creator_is_allowed_only_when_unique() -> None:
    one = replace(hybrid("result-snapshot"), visual_elements=(VisualElement(
        "v1", "Californication", "song_result", Rect(10, 10, 100, 50), None, True,
    ),))
    result, computer, _ = _phase3_run(one)
    assert result.success and len(computer.result_clicks) == 1
    assert result.phase3_execution.target_resolution.candidates[0].qualifier_evidence == (
        IdentityEvidence.ABSENT,
    )

    two = replace(one, visual_elements=(
        one.visual_elements[0], replace(one.visual_elements[0], id="v2", rectangle=Rect(10, 60, 100, 100)),
    ))
    result, computer, decisions = _phase3_run(two)
    assert result.stop_reason == "result_identity_ambiguous"
    assert computer.result_clicks == []
    assert decisions.result_calls == []


def test_phase3_result_click_exception_consumes_budgets_without_retry() -> None:
    result_observation = hybrid("result-snapshot")
    result, computer, decisions = _phase3_run(result_observation)
    # Re-run with a failing executor to assert attempt accounting.
    focused = UIElement("c1", "Search", "Edit", enabled=True, visible=True,
                        focused=True, is_password=False)
    computer = Computer([local("before"), local("pc", elements=(focused,)), local("pt"), local("pr")],
                        directed=phase_hybrid("search-snapshot"))
    computer.result_directed_result = replace(
        result_observation, result_readiness=ResultReadinessResult(
            True, "changed_and_quiet_stable", 3, 1600, True, True, True, True,
        ),
    )
    computer.result_click_result = RuntimeError("failed")
    decisions = Decisions([need(), ready(VisualClickAction("search-snapshot", "v1"))])
    decisions.result_decision = ready(VisualClickAction("result-snapshot", "v1"))
    result = HybridResultDebugAgent(
        computer, decisions, limits=AgentLimits(settle_action_seconds=0),
        clock=clock(), sleep_fn=lambda _: None,
    ).run("Open Spotify and play Californication by Red Hot Chili Peppers")
    assert result.stop_reason == "result_click_failed"
    assert len(computer.result_clicks) == 1
    assert result.phase3_execution.visual_click_budget_remaining == 0
    assert result.phase3_execution.result_selection_budget_remaining == 0


def _phase3_candidate_observation(
    label="Californication", parent="Red Hot Chili Peppers", role="song result",
):
    base = hybrid("result-snapshot")
    item = VisualElement("v1", label, role, Rect(100, 100, 400, 180), None, True, parent)
    provider_item = VisualCandidateProviderDiagnostic(
        label[:160], parent[:120], role[:40], True, True, True,
        provider_candidate_id="p1", validated_visual_id="v1",
    )
    return replace(
        base, visual_elements=(item,),
        visual_provider_candidates=(provider_item,),
        visual_pipeline=VisualPipelineDiagnostic(5, 3, 2, 1, 1, 1),
    )


def test_phase3_diagnostics_preserve_label_parent_role_and_objective() -> None:
    result, _computer, _decisions = _phase3_run(_phase3_candidate_observation())
    resolution = result.phase3_execution.target_resolution
    candidate = resolution.candidates[0]
    assert result.success
    assert candidate.primary_text == "Californication"
    assert candidate.secondary_text == ("Red Hot Chili Peppers",)
    assert candidate.provider_role == "song result"
    assert candidate.primary_identity is IdentityEvidence.MATCH
    assert candidate.qualifier_evidence == (IdentityEvidence.MATCH,)
    assert candidate.geometry_valid and candidate.actionable and candidate.safety_eligible
    assert candidate.admissible is True
    assert resolution.target.primary_identity == "Californication"
    assert resolution.target.qualifiers == ("Red Hot Chili Peppers",)
    assert resolution.status is TargetResolutionStatus.UNIQUE
    assert resolution.resolution_reason == "one_candidate_has_uniquely_strongest_identity_and_role_evidence"
    execution = result.phase3_execution
    assert execution.provider_raw_candidate_count == 3
    assert execution.provider_parsed_candidate_count == 2
    assert execution.provider_validated_candidate_count == 1
    assert execution.candidate_count == 1 and execution.eligible_candidate_count == 1


def test_phase3_candidate_diagnostics_explain_title_and_creator_rejections() -> None:
    wrong_title, computer, decisions = _phase3_run(_phase3_candidate_observation("Other Song"))
    row = wrong_title.phase3_execution.target_resolution.candidates[0]
    assert wrong_title.stop_reason == "result_identity_mismatch"
    assert computer.result_clicks == []
    assert decisions.result_calls == []
    assert row.primary_identity is IdentityEvidence.MISMATCH
    assert "primary_identity_mismatch" in row.rejection_reasons

    wrong_creator, computer, decisions = _phase3_run(
        _phase3_candidate_observation(parent="Other Artist"),
    )
    row = wrong_creator.phase3_execution.target_resolution.candidates[0]
    assert wrong_creator.stop_reason == "result_identity_mismatch"
    assert computer.result_clicks == []
    assert decisions.result_calls == []
    assert row.qualifier_evidence == (IdentityEvidence.MISMATCH,)
    assert "qualifier_mismatch" in row.rejection_reasons


def test_phase3_reports_provider_candidate_rejected_by_bbox_validation() -> None:
    rejected = _phase3_candidate_observation(parent="")
    result, computer, _ = _phase3_run(replace(
        rejected, visual_elements=(),
        visual_provider_candidates=(replace(
            rejected.visual_provider_candidates[0], geometry_valid=False,
            rejection_reason="invalid_or_outside_bbox", validated_visual_id=None,
        ),),
        visual_pipeline=VisualPipelineDiagnostic(5, 1, 1, 0, 0, 0),
    ))
    row = result.phase3_execution.target_resolution.candidates[0]
    assert result.stop_reason == "result_grounding_empty"
    assert computer.result_clicks == []
    assert row.geometry_valid is False
    assert row.primary_identity is IdentityEvidence.MATCH
    assert row.qualifier_evidence == (IdentityEvidence.ABSENT,)
    assert "invalid_geometry" in row.rejection_reasons
    assert "safety_rejected" in row.rejection_reasons
    assert row.admissible is False


def test_phase3_missing_creator_diagnostic_is_null_and_candidate_text_is_bounded_redacted() -> None:
    missing = _phase3_candidate_observation(parent="")
    result, _computer, _ = _phase3_run(missing)
    row = result.phase3_execution.target_resolution.candidates[0]
    assert result.success
    assert row.qualifier_evidence == (IdentityEvidence.ABSENT,)

    secret = "sk-test-placeholder-for-redaction"
    long_observation = _phase3_candidate_observation(
        label="Californication " + secret + " x" * 200,
    )
    result, _computer, _ = _phase3_run(long_observation)
    row = result.phase3_execution.target_resolution.candidates[0]
    assert len(row.primary_text) <= 240
    assert len(row.secondary_text) <= 1 and all(len(value) <= 200 for value in row.secondary_text)
    assert secret not in row.primary_text
    assert "[REDACTED]" in row.primary_text


def test_low_confidence_visual_act_after_observation_remains_blocked() -> None:
    decisions = Decisions([
        need(confidence=.54), ready(VisualClickAction("hybrid-1", "v1"), .79),
    ])
    computer = Computer([local()], directed=hybrid())
    result = HybridDebugAgent(computer, decisions, clock=clock()).run("search for query")
    assert result.stop_reason == "low_confidence"
    assert len(computer.directed_calls) == 1
    assert computer.executed == []


def test_observe_is_suppressed_in_focused_credential_context() -> None:
    password = UIElement(
        "c1", "", "Edit", enabled=True, visible=True, focused=True, is_password=True,
    )
    computer = Computer([local(elements=(password,))])
    result = HybridDebugAgent(
        computer, Decisions([need(confidence=.54)]), clock=clock(),
    ).run("search for query")
    assert result.stop_reason == "observation_policy_rejected"
    assert result.trace[0].observation_policy.reason == "credential_sensitive_context"
    assert computer.directed_calls == []
    assert computer.executed == []


def test_stop_terminal_choice_does_not_trigger_observation() -> None:
    client = Mock()
    client.evaluate.side_effect = lambda payload: reply(payload, "stop", .54)
    computer = Computer([local()])
    result = HybridDebugAgent(computer, JevDecisionMaker(client), clock=clock()).run(
        "search for query",
    )
    assert result.stop_reason == "needs_human"
    assert result.trace[0].decision_effect is DecisionEffect.TERMINAL
    assert result.trace[0].release_policy == "terminal"
    assert computer.directed_calls == []


def test_stale_visual_id_fails_before_execution() -> None:
    computer = Computer([local()], directed=hybrid())
    decisions = Decisions([need(), ready(VisualClickAction("older", "v1"))])
    result = HybridDebugAgent(computer, decisions, clock=clock()).run("play music")
    assert result.stop_reason == "stale_visual_target"
    assert computer.executed == []


def test_repeated_identical_grounding_stops_and_reuses_current_result() -> None:
    computer = Computer([local()], directed=hybrid())
    decisions = Decisions([need(), need()])
    result = HybridDebugAgent(computer, decisions, clock=clock()).run("play music")
    assert result.stop_reason == "repeated_grounding"
    assert len(computer.directed_calls) == 1


def test_changed_objective_gets_a_fresh_directed_snapshot() -> None:
    computer = Computer(
        [local()], directed=[hybrid("hybrid-1"), hybrid("hybrid-2")],
    )
    decisions = Decisions([need("Find search."), need("Find matching song."), ready(FinishAction("done"))])
    result = HybridDebugAgent(computer, decisions, clock=clock()).run("play music")
    assert result.stop_reason == "observation_policy_rejected"
    assert [call.objective for call in computer.directed_calls] == ["Find search."]


def test_executed_action_invalidates_grounding_reuse_and_forces_local_observation() -> None:
    editor = UIElement(
        "c1", "Search", "Edit", enabled=True, visible=True, focused=True, is_password=False,
    )
    computer = Computer([local("one", elements=(editor,)), local("two")])
    decisions = Decisions([ready(TypeAction("hello")), need(), ready(FinishAction("done"))])
    result = HybridDebugAgent(
        computer, decisions, limits=AgentLimits(settle_action_seconds=0), clock=clock(),
    ).run('type "hello"')
    assert result.stop_reason == "finished"
    assert len(computer.executed) == 1
    assert decisions.calls[1][1].observation_id == "two"
    assert decisions.calls[1][2][0].source_observation_id == "one"
    assert len(computer.directed_calls) == 1


def test_low_confidence_and_max_steps_remain_bounded() -> None:
    computer = Computer([local()])
    low = HybridDebugAgent(computer, Decisions([need(confidence=.79)]), clock=clock()).run("play")
    assert len(computer.directed_calls) == 1

    computer = Computer([local()], directed=hybrid())
    bounded = HybridDebugAgent(
        computer, Decisions([need()]), limits=AgentLimits(max_steps=1), clock=clock(),
    ).run("play")
    assert bounded.stop_reason == "max_steps" and len(computer.directed_calls) == 1


@pytest.mark.parametrize("category", ["timeout", "malformed_response"])
def test_visual_provider_failures_fail_closed(category: str) -> None:
    failed = replace(
        hybrid(), visual_elements=(),
        visual_provider_error=ProviderErrorDiagnostic(category, message="safe failure"),
    )
    computer = Computer([local()], directed=failed)
    result = HybridDebugAgent(computer, Decisions([need()]), clock=clock()).run("play")
    assert result.stop_reason == "visual_grounding_failed"
    assert computer.executed == []


def test_trace_has_safe_metrics_without_pixels_coordinates_or_secrets(monkeypatch) -> None:
    monkeypatch.setenv("TEST_API_KEY", "super-private-key")
    computer = Computer([local()], directed=hybrid())
    result = HybridDebugAgent(
        computer, Decisions([need(), ready(VisualClickAction("hybrid-1", "v1"))]),
        clock=clock(),
    ).run("play music")
    serialized = repr(result)
    assert result.total_run_latency_ms > 0
    assert all(step.total_step_latency_ms >= 0 and step.jev_latency_ms >= 0 for step in result.trace)
    assert "super-private-key" not in serialized
    assert "base64" not in serialized.casefold()
    assert "rectangle" not in serialized.casefold()
