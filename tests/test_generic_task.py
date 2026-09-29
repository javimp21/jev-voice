"""Synthetic integration tests for generic bounded target activation."""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest

from agent.generic_task import (
    GenericCapability, GenericTaskBudgets, GenericTaskDebugAgent, GenericTaskOperation,
    _ReadinessAssessment, _RunBudget, decompose_generic_task,
    wait_for_local_transition,
)
from agent.activation_postcondition import (
    ActivationEvidence, ActivationPostconditionResult,
    evaluate_target_activation_postcondition,
)
from computer.actions import (
    ClickAction, OpenAppAction, QuerySubmitAction, TypeAction, VisualClickAction,
)
from computer.applications import (
    ApplicationCandidate, MemoryApplicationCatalog, TrustedApplicationRuntimeState,
    TrustedWindowActivationResult,
)
from computer.models import (
    Observation, ProviderErrorDiagnostic, Rect, ScreenshotMetadata, UIElement, VisualElement,
    VisualProviderAttempt,
    VisualGroundingStatus, VisualPipelineDiagnostic, VisualRegion, VisualSelectionState,
)
from computer.results import ActionResult, VisualActivationDiagnostic
from computer.visual import VisualFieldValueRead
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
            snapshot, foreground_hwnd or 77, Rect(0, 0, 800, 600), Rect(0, 0, 800, 600),
            800, 600, 96, 96, 1.0, 1.0,
        )
    return Observation(
        "test.exe", "Test Window", elements, process_id=process_id,
        observation_id=snapshot, application_id=app_id, visual_elements=visual,
        screenshot=metadata, visual_provider="fixture" if visual else None,
        visual_directed_grounding=bool(visual),
        visual_requested_max_elements=5 if visual else None,
        visual_returned_elements=len(visual) if visual else None,
        visual_provider_call_count=1 if visual else 0,
        visual_grounding_status=(
            VisualGroundingStatus.SUCCESS_WITH_CANDIDATES if visual else None
        ),
        visual_pipeline=(
            VisualPipelineDiagnostic(5, len(visual), len(visual), len(visual),
                                    len(visual), len(visual), None)
            if visual else None
        ),
        foreground_hwnd=foreground_hwnd,
    )


def conversation(control_id: str = "c1", name: str = "Pablo García", *, parent: str = "Conversations") -> UIElement:
    return UIElement(
        control_id, name, "ListItem", rectangle=Rect(10, 10, 300, 70),
        enabled=True, visible=True, parent_name=parent,
    )


def active_conversation(
    snapshot: str, name: str = "Pablo García", *, parent: str = "Conversations",
    app_id: str = APP_ID, process_id: int = 42, foreground_hwnd: int | None = 100,
) -> Observation:
    return observation(snapshot, elements=(replace(
        conversation(name=name, parent=parent), selected=True,
    ),), app_id=app_id, process_id=process_id, foreground_hwnd=foreground_hwnd)


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
        self.observed_local = []
        self.executed = []
        self.visual_activations = []
        self.query_submits = []
        self.phase3_query_submits = []
        self.phase2_type_calls = []
        self.context_matches = True
        self.visual_click_result = None
        self.on_type = None
        self.on_visual_click = None
        self.postcondition_groundings = []
        self.postcondition_visual_result = None
        self.on_postcondition_visual = None
        self.preclick_local_observations = []
        self.preclick_directed_observations = []
        self.preclick_local_calls = []
        self.preclick_directed_calls = []
        self._preclick_original = None
        self.field_value_reads = []
        self.field_value_result = VisualFieldValueRead(
            crop_valid=True, context_stable=True, credential_safe=True,
            provider_attempts=(VisualProviderAttempt(
                "fixture", "", 0, "value_read_empty",
            ),),
        )

    @staticmethod
    def _copy_for_preclick(source: Observation, suffix: str, *, directed: bool) -> Observation:
        snapshot_id = f"{source.observation_id}-{suffix}"
        screenshot = source.screenshot
        if screenshot is not None:
            screenshot = replace(screenshot, snapshot_id=snapshot_id)
        if directed:
            return replace(
                source, observation_id=snapshot_id, screenshot=screenshot,
                visual_provider=source.visual_provider or "fixture",
                visual_directed_grounding=True,
                visual_requested_max_elements=5,
                visual_grounding_status=(
                    VisualGroundingStatus.SUCCESS_WITH_CANDIDATES
                    if source.visual_elements else VisualGroundingStatus.SUCCESS_EMPTY
                ),
                visual_provider_error=None,
                visual_provider_attempts=(),
                visual_provider_call_count=1,
            )
        return replace(
            source, observation_id=snapshot_id, visual_elements=(), screenshot=None,
            visual_provider=None, visual_model=None, visual_latency_ms=None,
            visual_provider_error=None, visual_provider_attempts=(),
            visual_directed_grounding=False, visual_requested_max_elements=None,
            visual_grounding_status=None, visual_provider_call_count=0,
        )

    def observe_preclick_local(self, expected_context: Observation) -> Observation:
        self.preclick_local_calls.append(expected_context)
        self._preclick_original = expected_context
        if self.preclick_local_observations:
            return self.preclick_local_observations.pop(0)
        return self._copy_for_preclick(expected_context, "preclick-uia", directed=False)

    def observe_preclick_directed(self, grounding, expected_context: Observation) -> Observation:
        self.preclick_directed_calls.append((grounding, expected_context))
        if self.preclick_directed_observations:
            return self.preclick_directed_observations.pop(0)
        original = self._preclick_original or expected_context
        copied = self._copy_for_preclick(original, "preclick-visual", directed=True)
        return replace(
            copied,
            visual_requested_max_elements=grounding.max_elements,
            visual_provider_attempts=original.visual_provider_attempts,
        )

    def observe_local(self) -> Observation:
        if not self.locals:
            raise RuntimeError("no synthetic observation remains")
        result = self.locals.pop(0)
        self.observed_local.append(result)
        return result

    def observe_directed(self, grounding) -> Observation:
        self.directed_calls.append(grounding)
        if not self.directed:
            raise RuntimeError("no synthetic visual observation remains")
        return self.directed.pop(0)

    def read_visual_field_value(self, observation, field_id) -> VisualFieldValueRead:
        self.field_value_reads.append((observation, field_id))
        return self.field_value_result

    def execute(self, action, observation=None) -> ActionResult:
        self.executed.append((action, observation))
        return ActionResult(True, action, "synthetic action", input_issued=True)

    def execute_type_phase2(self, action, observation, *, visual_verified: bool) -> ActionResult:
        self.phase2_type_calls.append((action, observation, visual_verified))
        self.executed.append((action, observation))
        if self.on_type is not None:
            self.on_type(action, observation, visual_verified)
        return ActionResult(True, action, "synthetic verified literal input", input_issued=True)

    def visual_context_matches(self, _observation) -> bool:
        return self.context_matches

    def activation_postcondition_context_matches(self, before, after) -> bool:
        return bool(
            self.context_matches and before.application_id == after.application_id
            and before.process_id == after.process_id
            and before.foreground_hwnd == after.foreground_hwnd
        )

    def verify_activation_postcondition_visual(self, observation, grounding) -> Observation:
        self.postcondition_groundings.append(grounding)
        if self.on_postcondition_visual is not None:
            self.on_postcondition_visual()
        if isinstance(self.postcondition_visual_result, Exception):
            raise self.postcondition_visual_result
        if self.postcondition_visual_result is not None:
            return self.postcondition_visual_result
        return replace(
            observation,
            visual_provider_error=ProviderErrorDiagnostic("timeout"),
            visual_grounding_status=VisualGroundingStatus.PROVIDER_ERROR,
        )

    def execute_generic_target_activation(self, action, observation) -> ActionResult:
        self.visual_activations.append((action, observation))
        if self.on_visual_click is not None:
            self.on_visual_click(action, observation)
        if self.visual_click_result is not None:
            return self.visual_click_result
        return ActionResult(True, action, "synthetic visual activation", input_issued=True)

    def execute_generic_query_submit(self, action, observation) -> ActionResult:
        self.query_submits.append((action, observation))
        return ActionResult(True, action, "synthetic query submit", input_issued=True)

    def execute_query_submit_phase3(
        self, action, observation, verified_search_observation,
    ) -> ActionResult:
        self.phase3_query_submits.append((action, observation, verified_search_observation))
        self.query_submits.append((action, observation))
        return ActionResult(True, action, "synthetic visually verified query submit", input_issued=True)


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


class FakeReadinessClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds


def readiness_visual(
    snapshot: str, *, candidate: bool = False, foreground_hwnd: int = 100,
) -> Observation:
    elements = (VisualElement(
        "v1", "Gamma", "button", Rect(10, 10, 100, 50), None, True,
    ),) if candidate else ()
    return replace(
        observation(snapshot, visual=elements, foreground_hwnd=foreground_hwnd),
        screenshot=ScreenshotMetadata(
            snapshot, foreground_hwnd, Rect(0, 0, 800, 600), Rect(0, 0, 800, 600),
            800, 600, 96, 96, 1.0, 1.0,
        ),
        visual_directed_grounding=True,
        visual_requested_max_elements=5,
        visual_returned_elements=len(elements),
        visual_provider_call_count=1,
        visual_grounding_status=(
            VisualGroundingStatus.SUCCESS_WITH_CANDIDATES
            if candidate else VisualGroundingStatus.SUCCESS_EMPTY
        ),
        visual_pipeline=VisualPipelineDiagnostic(
            5, 1 if candidate else 0, 1 if candidate else 0,
            1 if candidate else 0, 1 if candidate else 0,
            1 if candidate else 0, None,
        ),
    )


def failed_readiness_visual(snapshot: str, *, category: str = "timeout") -> Observation:
    category_name = {
        "timeout": "timeout", "rate_limited": "rate_limit",
        "invalid_response": "invalid_response", "server_error": "server",
    }.get(category, "unknown")
    return replace(
        readiness_visual(snapshot),
        visual_provider_error=ProviderErrorDiagnostic(
            category, provider_name="gemini", provider_model="gemini-3.5-flash-lite",
            provider_error_category=category_name,
        ),
        visual_grounding_status=VisualGroundingStatus.PROVIDER_ERROR,
    )


@pytest.mark.parametrize("capability", [
    GenericCapability.FIND_QUERY_FIELD, GenericCapability.RESOLVE_TARGET,
])
def test_readiness_retry_uses_fresh_observation_and_stops_when_candidate_appears(
    capability: GenericCapability,
) -> None:
    clock = FakeReadinessClock()
    activated = observation("activated", app_id=APP_ID)
    fresh = observation("fresh-2", app_id=APP_ID)
    computer = FakeComputer([fresh])
    agent = GenericTaskDebugAgent(computer, FakeDecisionMaker(), clock=clock, sleep_fn=clock.sleep)
    budget = _RunBudget()
    attempts = 0
    diagnostics = []

    def discover(source: Observation):
        nonlocal attempts
        attempts += 1
        visual = (failed_readiness_visual("visual-1") if attempts == 1
                  else readiness_visual("visual-2", candidate=True))
        candidate_found = attempts == 2
        selected = None if not candidate_found else visual
        value = None if not candidate_found else VisualClickAction(visual.observation_id, "v1")
        return selected, value, _ReadinessAssessment(
            visual.observation_id, 0, 0 if not candidate_found else 1,
            False, candidate_found, candidate_found,
            "discovery_incomplete" if not candidate_found else "candidate_found", visual,
        )

    selected, action, stop_reason = agent._run_readiness_discovery(
        capability, activated, activation_observation=activated,
        activated_app_id=APP_ID, activation_verified=True, budget=budget,
        trace=[], diagnostics=diagnostics, discover=discover,
    )

    assert stop_reason is None
    assert attempts == 2 and budget.readiness_retries == 1
    assert selected.observation_id == "visual-2"
    assert action == VisualClickAction("visual-2", "v1")
    assert diagnostics[0].retry_performed and diagnostics[0].max_attempts == 2
    assert computer.executed == []
    assert clock.value == pytest.approx(.2)


def test_perception_retry_is_bounded_to_exactly_two_attempts() -> None:
    clock = FakeReadinessClock()
    activated = observation("activated", app_id=APP_ID)
    computer = FakeComputer([observation("fresh-2", app_id=APP_ID)])
    agent = GenericTaskDebugAgent(computer, FakeDecisionMaker(), clock=clock, sleep_fn=clock.sleep)
    budget = _RunBudget()
    diagnostics = []
    calls = 0

    def discover(_source: Observation):
        nonlocal calls
        calls += 1
        visual = (failed_readiness_visual("visual-1") if calls == 1
                  else failed_readiness_visual("visual-2"))
        candidate_found = calls == 2
        if candidate_found:
            visual = readiness_visual("visual-2", candidate=True)
        return (
            None if not candidate_found else visual,
            None if not candidate_found else VisualClickAction(visual.observation_id, "v1"),
            _ReadinessAssessment(
                visual.observation_id, 0, 1 if candidate_found else 0,
                False, candidate_found, candidate_found,
                "candidate_found" if candidate_found else "discovery_incomplete", visual,
            ),
        )

    selected, action, stop_reason = agent._run_readiness_discovery(
        GenericCapability.FIND_QUERY_FIELD, activated,
        activation_observation=activated, activated_app_id=APP_ID,
        activation_verified=True, budget=budget, trace=[],
        diagnostics=diagnostics, discover=discover,
    )

    assert stop_reason is None and calls == 2
    assert budget.readiness_retries == 1
    assert action == VisualClickAction("visual-2", "v1")
    detail = diagnostics[0]
    assert detail.applicable and detail.success
    assert detail.attempts == detail.max_attempts == 2
    assert [item.observation_id for item in detail.attempt_diagnostics] == [
        "activated", "fresh-2",
    ]
    assert len({item.observation_id for item in detail.attempt_diagnostics}) == 2
    assert [item.visual_observation_id for item in detail.attempt_diagnostics] == [
        "visual-1", "visual-2",
    ]
    assert all(item.foreground_identity_stable for item in detail.attempt_diagnostics)
    assert selected.screenshot is not None
    assert selected.screenshot.snapshot_id == "visual-2"
    assert computer.executed == []


def test_complete_empty_discovery_is_not_retried() -> None:
    clock = FakeReadinessClock()
    activated = observation("activated", app_id=APP_ID)
    computer = FakeComputer([observation("unused", app_id=APP_ID)])
    agent = GenericTaskDebugAgent(computer, FakeDecisionMaker(), clock=clock, sleep_fn=clock.sleep)
    budget = _RunBudget()
    diagnostics = []
    calls = 0

    def discover(_source: Observation):
        nonlocal calls
        calls += 1
        visual = readiness_visual(f"visual-{calls}")
        return None, None, _ReadinessAssessment(
            visual.observation_id, 0, 0, False, False, False,
            "discovery_incomplete", visual,
        )

    selected, value, stop_reason = agent._run_readiness_discovery(
        GenericCapability.FIND_QUERY_FIELD, activated,
        activation_observation=activated, activated_app_id=APP_ID,
        activation_verified=True, budget=budget, trace=[],
        diagnostics=diagnostics, discover=discover,
    )

    assert stop_reason is None and calls == 1
    assert selected.observation_id == "activated" and value is None
    assert not diagnostics[0].applicable and not diagnostics[0].success
    assert diagnostics[0].reason == "complete_semantic_no_match"
    assert diagnostics[0].retry_result == "not_attempted"
    assert diagnostics[0].attempts == 1 and diagnostics[0].max_attempts == 2
    assert len(computer.locals) == 1
    assert computer.executed == []


def test_readiness_retry_stops_when_a_fresh_local_probe_is_incomplete() -> None:
    clock = FakeReadinessClock()
    activated = observation("activated", app_id=APP_ID)
    incomplete = replace(observation("truncated", app_id=APP_ID), truncated=True)
    computer = FakeComputer([incomplete])
    agent = GenericTaskDebugAgent(computer, FakeDecisionMaker(), clock=clock, sleep_fn=clock.sleep)
    budget = _RunBudget()
    diagnostics = []
    calls = 0

    def discover(_source: Observation):
        nonlocal calls
        calls += 1
        visual = failed_readiness_visual(f"visual-{calls}")
        return None, None, _ReadinessAssessment(
            visual.observation_id, 0, 0, False, False, False,
            "discovery_incomplete", visual,
        )

    _, _, stop_reason = agent._run_readiness_discovery(
        GenericCapability.FIND_QUERY_FIELD, activated,
        activation_observation=activated, activated_app_id=APP_ID,
        activation_verified=True, budget=budget, trace=[],
        diagnostics=diagnostics, discover=discover,
    )

    assert stop_reason == "observation_incomplete"
    assert calls == 2 and budget.readiness_retries == 1
    assert diagnostics[0].attempt_diagnostics[-1].observation_id == "truncated"
    assert diagnostics[0].attempt_diagnostics[-1].foreground_identity_stable
    assert "structural_observation_truncated" in diagnostics[0].attempt_diagnostics[-1].incomplete_reasons
    assert computer.executed == []


def test_readiness_retry_rejects_a_reused_visual_snapshot_id() -> None:
    activated = observation("activated", app_id=APP_ID)
    computer = FakeComputer([observation("fresh-2", app_id=APP_ID)])
    agent = GenericTaskDebugAgent(computer, FakeDecisionMaker())
    budget = _RunBudget()
    calls = 0

    def discover(_source: Observation):
        nonlocal calls
        calls += 1
        visual = (failed_readiness_visual("same-visual-snapshot") if calls == 1
                  else readiness_visual("same-visual-snapshot", candidate=True))
        return None, None, _ReadinessAssessment(
            visual.observation_id, 0, 1 if calls == 2 else 0,
            False, calls == 2, calls == 2,
            "candidate_found" if calls == 2 else "discovery_incomplete", visual,
        )

    _, _, stop_reason = agent._run_readiness_discovery(
        GenericCapability.FIND_QUERY_FIELD, activated,
        activation_observation=activated, activated_app_id=APP_ID,
        activation_verified=True, budget=budget, trace=[], diagnostics=[],
        discover=discover,
    )

    assert stop_reason == "stale_observation"
    assert calls == 2 and budget.readiness_retries == 1
    assert computer.executed == []


@pytest.mark.parametrize(("changed", "expected_stop"), [
    (observation("changed-app", app_id="other"), "trusted_identity_changed"),
    (observation("window-gone", app_id=""), "trusted_identity_changed"),
    (replace(observation("changed-window", app_id=APP_ID), foreground_hwnd=999), "foreground_changed"),
    (observation("activated", app_id=APP_ID), "stale_observation"),
])
def test_readiness_retry_stops_on_changed_or_stale_foreground(
    changed: Observation, expected_stop: str,
) -> None:
    clock = FakeReadinessClock()
    activated = observation("activated", app_id=APP_ID)
    computer = FakeComputer([changed])
    agent = GenericTaskDebugAgent(computer, FakeDecisionMaker(), clock=clock, sleep_fn=clock.sleep)
    budget = _RunBudget()
    diagnostics = []
    calls = 0

    def discover(_source: Observation):
        nonlocal calls
        calls += 1
        visual = (failed_readiness_visual(f"visual-{calls}") if calls == 1
                  else readiness_visual(f"visual-{calls}"))
        return None, None, _ReadinessAssessment(
            visual.observation_id, 0, 0, True, True, False,
            "empty_discovery", visual,
        )

    _, _, stop_reason = agent._run_readiness_discovery(
        GenericCapability.RESOLVE_TARGET, activated,
        activation_observation=activated, activated_app_id=APP_ID,
        activation_verified=True, budget=budget, trace=[],
        diagnostics=diagnostics, discover=discover,
    )

    assert stop_reason == expected_stop
    assert calls == 1
    assert diagnostics[0].attempt_diagnostics[-1].foreground_identity_stable is (
        expected_stop == "stale_observation"
    )
    assert computer.executed == []


@pytest.mark.parametrize(("usable", "reason"), [
    (False, "unsafe_candidate"),
    (False, "ambiguous_candidates"),
    (False, "visible_candidates_no_match"),
])
def test_readiness_retry_does_not_retry_rejected_or_visible_candidates(
    usable: bool, reason: str,
) -> None:
    activated = observation("activated", app_id=APP_ID)
    computer = FakeComputer([observation("unused", app_id=APP_ID)])
    agent = GenericTaskDebugAgent(computer, FakeDecisionMaker())
    budget = _RunBudget()
    calls = 0

    def discover(source: Observation):
        nonlocal calls
        calls += 1
        visual = readiness_visual("visual-candidate", candidate=True)
        return None, None, _ReadinessAssessment(
            visual.observation_id, 1, 1, False, True, usable,
            reason, visual,
        )

    _, _, stop_reason = agent._run_readiness_discovery(
        GenericCapability.RESOLVE_TARGET, activated,
        activation_observation=activated, activated_app_id=APP_ID,
        activation_verified=True, budget=budget, trace=[], diagnostics=[],
        discover=discover,
    )

    assert stop_reason is None
    assert calls == 1 and budget.readiness_retries == 0
    assert computer.executed == []


def test_perception_retry_is_applicable_without_a_just_activated_marker() -> None:
    activated = observation("foreground", app_id=APP_ID)
    computer = FakeComputer([observation("fresh", app_id=APP_ID)])
    agent = GenericTaskDebugAgent(computer, FakeDecisionMaker())
    budget = _RunBudget()
    diagnostics = []
    calls = 0

    def discover(source: Observation):
        nonlocal calls
        calls += 1
        visual = (failed_readiness_visual("visual-1") if calls == 1
                  else readiness_visual("visual-2", candidate=True))
        candidate_found = calls == 2
        return (visual if candidate_found else None,
                VisualClickAction("visual-2", "v1") if candidate_found else None,
                _ReadinessAssessment(
            visual.observation_id, 0, int(candidate_found), False,
            candidate_found, candidate_found,
            "candidate_found" if candidate_found else "discovery_incomplete", visual,
        )
        )

    agent._run_readiness_discovery(
        GenericCapability.FIND_QUERY_FIELD, activated,
        activation_observation=None, activated_app_id=None,
        activation_verified=False, budget=budget, trace=[],
        diagnostics=diagnostics, discover=discover,
    )

    assert diagnostics[0].applicable and diagnostics[0].retry_performed
    assert diagnostics[0].reason == "recoverable_incomplete_perception"
    assert budget.readiness_retries == 1 and calls == 2


def test_readiness_retry_requires_a_stable_window_handle_anchor() -> None:
    unanchored = observation("unanchored", app_id=APP_ID, foreground_hwnd=None)
    computer = FakeComputer([])
    agent = GenericTaskDebugAgent(computer, FakeDecisionMaker())
    diagnostics = []

    agent._run_readiness_discovery(
        GenericCapability.FIND_QUERY_FIELD, unanchored,
        activation_observation=unanchored, activated_app_id=APP_ID,
        activation_verified=True, budget=_RunBudget(), trace=[],
        diagnostics=diagnostics,
        discover=lambda source: (
            None, None, _ReadinessAssessment(
                "visual-1", 0, 0, False, False, False,
                "discovery_incomplete", failed_readiness_visual("visual-1"),
            ),
        ),
    )

    assert not diagnostics[0].applicable
    assert diagnostics[0].reason == "foreground_identity_not_stable"
    assert computer.locals == []


def test_search_recovers_from_incomplete_query_discovery_then_keeps_action_limits() -> None:
    candidate = ApplicationCandidate(APP_ID, "Alpha", "test", launch_policy="allow")
    catalog = MemoryApplicationCatalog((candidate,))
    field = VisualElement(
        "v-search", "Search", "search field", Rect(20, 20, 350, 60), None, True,
    )
    field_grounding = replace(
        readiness_visual("field-visual", candidate=True),
        visual_elements=(field,),
        visual_pipeline=VisualPipelineDiagnostic(5, 1, 1, 1, 1, 1, None),
    )
    verification = replace(
        observation("verify-field", visual=(replace(field, id="v-verified"),),
                    app_id=APP_ID, foreground_hwnd=100),
        screenshot=ScreenshotMetadata(
            "verify-field", 100, Rect(0, 0, 800, 600), Rect(0, 0, 800, 600),
            800, 600, 96, 96, 1.0, 1.0,
        ),
    )
    clicked = observation(
        "after-click", elements=(UIElement(
            "c-pane", "Search Area", "Pane", enabled=True, visible=True,
            focused=True, is_password=False,
        ),), app_id=APP_ID, foreground_hwnd=100,
    )
    typed = observation(
        "typed", elements=(query_field("typed", "Beta", focused=True),),
        app_id=APP_ID, foreground_hwnd=100,
    )
    results = observation("results", elements=(
        query_field("results", "Beta", focused=True),
        UIElement(
            "c2", "Beta", "Button", rectangle=Rect(10, 50, 250, 90),
            enabled=True, visible=True,
        ),
    ), app_id=APP_ID, foreground_hwnd=100)
    computer = FakeComputer(
        [
            observation("prelaunch", app_id="other-app"),
            observation("activated", app_id=APP_ID),
            observation("fresh-after-empty", app_id=APP_ID),
            clicked, typed, results,
        ],
        [failed_readiness_visual("failed-visual-1"), field_grounding, verification],
        visual_provider=object(),
    )
    decision = FakeDecisionMaker("c2")
    agent = GenericTaskDebugAgent(
        computer, decision, app_catalog=catalog,
        policy=GenericTargetActivationPolicy(catalog),
    )

    result = agent.run("Open Alpha and search for Beta")

    assert result.success and result.stop_reason == "search_completed"
    assert result.perception_retry[0].capability == GenericCapability.FIND_QUERY_FIELD.value
    assert result.perception_retry[0].applicable and result.perception_retry[0].success
    assert result.perception_retry[0].attempts == 2
    assert [item.observation_id for item in result.perception_retry[0].attempt_diagnostics] == [
        "activated", "fresh-after-empty",
    ]
    assert [item.visual_observation_id for item in result.perception_retry[0].attempt_diagnostics] == [
        "failed-visual-1", "field-visual",
    ]
    assert result.query_field_diagnostics.verification.strong_visual_verification_used
    assert computer.visual_activations == [(VisualClickAction("field-visual", "v-search"), field_grounding)]
    assert computer.phase2_type_calls[0][0] == TypeAction("Beta")
    assert len(computer.query_submits) == 1
    assert [action.kind for action, _ in computer.executed].count("open_app") == 1
    assert sum(isinstance(action, TypeAction) for action, _ in computer.executed) == 1
    assert result.query_field_activations == 1
    assert result.visual_target_activations == 1
    assert result.query_submits == 1
    assert result.final_target_activations == 0
    assert not decision.calls


def test_search_all_empty_readiness_observations_still_stops_as_query_field_unavailable() -> None:
    candidate = ApplicationCandidate(APP_ID, "Alpha", "test", launch_policy="allow")
    catalog = MemoryApplicationCatalog((candidate,))
    computer = FakeComputer(
        [
            observation("prelaunch", app_id="other-app"),
            observation("activated", app_id=APP_ID),
            observation("fresh-2", app_id=APP_ID),
            observation("fresh-3", app_id=APP_ID),
        ],
        [
            readiness_visual("empty-visual-1"),
            readiness_visual("empty-visual-2"),
            readiness_visual("empty-visual-3"),
        ],
        visual_provider=object(),
    )
    agent = GenericTaskDebugAgent(
        computer, FakeDecisionMaker(), app_catalog=catalog,
        policy=GenericTargetActivationPolicy(catalog),
    )

    result = agent.run("Open Alpha and search for Beta")

    assert not result.success and result.stop_reason == "query_field_unavailable"
    diagnostic = next(item for item in result.perception_retry
                      if item.capability == GenericCapability.FIND_QUERY_FIELD.value)
    assert not diagnostic.applicable and not diagnostic.success
    assert diagnostic.attempts == 1 and diagnostic.max_attempts == 2
    assert diagnostic.reason == "complete_semantic_no_match"
    assert diagnostic.retry_result == "not_attempted"
    assert len({item.observation_id for item in diagnostic.attempt_diagnostics}) == 1
    assert not computer.visual_activations
    assert not computer.phase2_type_calls
    assert not computer.query_submits
    assert [action.kind for action, _ in computer.executed].count("open_app") == 1
    assert not any(isinstance(action, TypeAction) for action, _ in computer.executed)


def test_query_readiness_does_not_retry_provider_candidates_rejected_by_validation() -> None:
    rejected = replace(
        readiness_visual("provider-rejected"),
        visual_grounding_status=VisualGroundingStatus.VALIDATION_EMPTY,
        visual_pipeline=VisualPipelineDiagnostic(5, 1, 1, 0, 0, 0, None),
    )
    computer = FakeComputer([], [rejected], visual_provider=object())
    agent = GenericTaskDebugAgent(computer, FakeDecisionMaker())
    source = observation("query-source")

    selected, action = agent._find_query_field(source, _RunBudget(), [])
    assessment = agent._query_field_readiness_assessment(source, selected)

    assert selected is None and action is None
    assert not assessment.empty and assessment.complete
    assert assessment.stop_reason == "candidates_present"


@pytest.mark.parametrize(("structural_id", "visual_id"), [
    ("structural-1", "visual-1"),
    ("same-id", "same-id"),
])
def test_query_readiness_accepts_successful_empty_visual_result_with_fresh_ids(
    structural_id: str, visual_id: str,
) -> None:
    grounded = readiness_visual(visual_id)
    computer = FakeComputer([], [grounded], visual_provider=object())
    agent = GenericTaskDebugAgent(computer, FakeDecisionMaker())
    source = observation(structural_id)

    selected, action = agent._find_query_field(source, _RunBudget(), [])
    assessment = agent._query_field_readiness_assessment(source, selected)

    assert selected is None and action is None
    assert assessment.complete and assessment.empty
    assert assessment.stop_reason == "empty_discovery"
    assert assessment.visual_observation is grounded


@pytest.mark.parametrize(("case", "provider_result"), [
    ("provider_error", "error"),
    ("invalid_response", "invalid"),
    ("screenshot_missing", "unavailable"),
    ("snapshot_mismatch", "invalid"),
    ("visual_incomplete", "invalid"),
    ("foreground_changed", "invalid"),
])
def test_query_readiness_classifies_visual_failure_as_incomplete(
    case: str, provider_result: str,
) -> None:
    grounded = readiness_visual("visual-state")
    if case == "provider_error":
        grounded = replace(
            grounded, visual_grounding_status=VisualGroundingStatus.PROVIDER_ERROR,
            visual_provider_error=ProviderErrorDiagnostic("timeout"),
        )
    elif case == "invalid_response":
        grounded = replace(
            grounded, visual_grounding_status=VisualGroundingStatus.PARSE_ERROR,
            visual_provider_error=ProviderErrorDiagnostic("malformed_response"),
        )
    elif case == "screenshot_missing":
        grounded = replace(grounded, screenshot=None)
    elif case == "snapshot_mismatch":
        grounded = replace(
            grounded,
            screenshot=replace(grounded.screenshot, snapshot_id="stale-capture"),
        )
    elif case == "visual_incomplete":
        grounded = replace(grounded, truncated=True)
    else:
        grounded = replace(grounded, foreground_hwnd=999)

    computer = FakeComputer([], [grounded], visual_provider=object())
    agent = GenericTaskDebugAgent(computer, FakeDecisionMaker())
    source = observation("structural-state")
    selected, action = agent._find_query_field(source, _RunBudget(), [])
    assessment = agent._query_field_readiness_assessment(source, selected)
    attempt = agent._readiness_attempt_diagnostic(
        GenericCapability.FIND_QUERY_FIELD, source, grounded,
        observation("activated", app_id=APP_ID), assessment,
        fresh_structural=True, fresh_visual=True,
    )

    assert selected is None and action is None
    assert not assessment.complete and not assessment.empty
    assert attempt.visual_provider_result == provider_result
    assert attempt.discovery_classification == "incomplete"
    if case == "foreground_changed":
        assert not attempt.same_window


def test_readiness_attempt_reports_structural_incomplete_reason_and_distinct_ids() -> None:
    clock = FakeReadinessClock()
    structural = replace(observation("structural-1"), inspection_errors=1)
    activated = observation("activated")
    computer = FakeComputer([])
    agent = GenericTaskDebugAgent(computer, FakeDecisionMaker(), clock=clock, sleep_fn=clock.sleep)
    diagnostics = []

    agent._run_readiness_discovery(
        GenericCapability.FIND_QUERY_FIELD, structural,
        activation_observation=activated, activated_app_id=APP_ID,
        activation_verified=True, budget=_RunBudget(), trace=[],
        diagnostics=diagnostics,
        discover=lambda source: (
            None, None, _ReadinessAssessment(
                "visual-1", 0, 0, False, False, False,
                "discovery_incomplete", readiness_visual("visual-1"),
            ),
        ),
    )

    attempt = diagnostics[0].attempt_diagnostics[0]
    assert attempt.observation_id == "structural-1"
    assert attempt.visual_observation_id == "visual-1"
    assert not attempt.structural_observation_complete
    assert attempt.visual_observation_complete
    assert "structural_observation_inspection_errors" in attempt.incomplete_reasons
    assert "structural_observation_id_mismatch" not in attempt.incomplete_reasons


def test_perception_retry_wait_is_bounded_after_initial_discovery() -> None:
    clock = FakeReadinessClock()
    activated = observation("activated", app_id=APP_ID)
    computer = FakeComputer([observation("fresh-2", app_id=APP_ID)])
    agent = GenericTaskDebugAgent(computer, FakeDecisionMaker(), clock=clock, sleep_fn=clock.sleep)
    budget = _RunBudget()
    calls = 0

    def discover(source: Observation):
        nonlocal calls
        calls += 1
        if calls == 1:
            clock.value += 1.637
        visual = (failed_readiness_visual("visual-1") if calls == 1
                  else readiness_visual("visual-2", candidate=True))
        candidate_found = calls == 2
        assessment = _ReadinessAssessment(
            visual.observation_id, 0, int(candidate_found), False,
            candidate_found, candidate_found,
            "candidate_found" if candidate_found else "discovery_incomplete", visual,
        )
        return (visual if candidate_found else None,
                VisualClickAction("visual-2", "v1") if candidate_found else None,
                assessment)

    _, _, stop_reason = agent._run_readiness_discovery(
        GenericCapability.FIND_QUERY_FIELD, activated,
        activation_observation=activated, activated_app_id=APP_ID,
        activation_verified=True, budget=budget, trace=[], diagnostics=[],
        discover=discover,
    )

    assert stop_reason is None
    assert calls == 2 and budget.readiness_retries == 1
    assert budget.readiness_wait_elapsed_seconds == pytest.approx(.2)
    assert clock.value == pytest.approx(1.837)


def test_readiness_inter_attempt_wait_budget_is_bounded_separately_from_discovery() -> None:
    clock = FakeReadinessClock()
    activated = observation("activated", app_id=APP_ID)
    computer = FakeComputer([
        observation("fresh-2", app_id=APP_ID),
        observation("fresh-3", app_id=APP_ID),
    ])
    agent = GenericTaskDebugAgent(
        computer, FakeDecisionMaker(),
        budgets=replace(
            GenericTaskBudgets(), readiness_wait_budget_seconds=.3,
            readiness_poll_seconds=.5,
        ),
        clock=clock, sleep_fn=clock.sleep,
    )
    budget = _RunBudget()
    diagnostics = []
    calls = 0

    def discover(_source: Observation):
        nonlocal calls
        calls += 1
        clock.value += .5
        visual = (failed_readiness_visual("visual-failed") if calls == 1
                  else readiness_visual("visual-candidate", candidate=True))
        candidate_found = calls == 2
        return (visual if candidate_found else None,
                VisualClickAction("visual-candidate", "v1") if candidate_found else None,
                _ReadinessAssessment(
            visual.observation_id, 0, int(candidate_found), False,
            candidate_found, candidate_found,
            "candidate_found" if candidate_found else "discovery_incomplete", visual,
        ))

    agent._run_readiness_discovery(
        GenericCapability.FIND_QUERY_FIELD, activated,
        activation_observation=activated, activated_app_id=APP_ID,
        activation_verified=True, budget=budget, trace=[], diagnostics=diagnostics,
        discover=discover,
    )

    assert budget.readiness_wait_elapsed_seconds == pytest.approx(.3)
    assert diagnostics[0].readiness_wait_elapsed_ms == 300
    assert diagnostics[0].initial_discovery_elapsed_ms == 500
    assert diagnostics[0].retry_discovery_elapsed_ms == 500
    assert diagnostics[0].total_elapsed_ms == 1300


def test_target_resolution_recovers_from_incomplete_visual_failure() -> None:
    clock = FakeReadinessClock()
    activated = observation("activated", app_id=APP_ID)
    candidate_grounding = replace(
        readiness_visual("target-visual-2", candidate=True),
        visual_requested_max_elements=5,
    )
    computer = FakeComputer(
        [observation("target-local-2", app_id=APP_ID)],
        [failed_readiness_visual("target-visual-1"), candidate_grounding],
        visual_provider=object(),
    )
    agent = GenericTaskDebugAgent(
        computer, FakeDecisionMaker(), clock=clock, sleep_fn=clock.sleep,
    )
    budget = _RunBudget()
    agent._active_budget = budget
    agent._active_trace = []
    agent._active_resolution_diagnostics = []
    target = TargetSpec("Gamma")

    def discover(source: Observation):
        grounded, resolution, evidence = agent._resolve_with_optional_visual(
            target, source, allow_grounding=True, reason="readiness-test",
        )
        assessment = agent._target_readiness_assessment(source, grounded, resolution)
        return grounded, (resolution, evidence), assessment

    selected, (resolution, _), stop_reason = agent._run_readiness_discovery(
        GenericCapability.RESOLVE_TARGET, observation("target-local-1", app_id=APP_ID),
        activation_observation=activated, activated_app_id=APP_ID,
        activation_verified=True, budget=budget, trace=agent._active_trace,
        diagnostics=[], discover=discover,
    )

    assert stop_reason is None
    assert resolution.status in {TargetResolutionStatus.UNIQUE, TargetResolutionStatus.CHOICE}
    assert selected.observation_id == "target-visual-2"
    assert selected.screenshot is not None
    assert selected.screenshot.snapshot_id == selected.observation_id
    assert budget.readiness_retries == 1
    assert computer.executed == []


def test_target_resolution_does_not_retry_when_visible_evidence_misses_requested_target() -> None:
    unrelated = UIElement(
        "c-other", "Other item", "Button", rectangle=Rect(10, 10, 100, 40),
        enabled=True, visible=True,
    )
    local = observation("target-local", elements=(unrelated,), app_id=APP_ID)
    activated = observation("activated", app_id=APP_ID)
    computer = FakeComputer([], [readiness_visual("target-visual")], visual_provider=object())
    agent = GenericTaskDebugAgent(computer, FakeDecisionMaker())
    budget = _RunBudget()
    agent._active_budget = budget
    agent._active_trace = []
    agent._active_resolution_diagnostics = []
    target = TargetSpec("Gamma")

    def discover(source: Observation):
        grounded, resolution, evidence = agent._resolve_with_optional_visual(
            target, source, allow_grounding=True, reason="visible-miss-test",
        )
        return grounded, (resolution, evidence), agent._target_readiness_assessment(
            source, grounded, resolution,
        )

    _, (resolution, _), stop_reason = agent._run_readiness_discovery(
        GenericCapability.RESOLVE_TARGET, local,
        activation_observation=activated, activated_app_id=APP_ID,
        activation_verified=True, budget=budget, trace=agent._active_trace,
        diagnostics=[], discover=discover,
    )

    assert stop_reason is None
    assert resolution.status is TargetResolutionStatus.NO_MATCH
    assert budget.readiness_retries == 0
    assert computer.locals == []
    assert computer.executed == []


@pytest.mark.parametrize(("command", "application", "operation", "payload"), [
    ("Open Alpha and search for Beta", "Alpha", GenericTaskOperation.SEARCH, "Beta"),
    ("Open Alpha then search for Beta", "Alpha", GenericTaskOperation.SEARCH, "Beta"),
    ("Open Alpha and search Beta", "Alpha", GenericTaskOperation.SEARCH, "Beta"),
    ("Open Alpha and find Beta", "Alpha", GenericTaskOperation.SEARCH, "Beta"),
    ("Search for Beta in Alpha", "Alpha", GenericTaskOperation.SEARCH, "Beta"),
    ("Search Beta in Alpha", "Alpha", GenericTaskOperation.SEARCH, "Beta"),
    ("Open Alpha and search for War and Peace", "Alpha", GenericTaskOperation.SEARCH, "War and Peace"),
    ("Open Alpha and search for Search Party", "Alpha", GenericTaskOperation.SEARCH, "Search Party"),
    ("Open Alpha and search for Alpha Centauri", "Alpha", GenericTaskOperation.SEARCH, "Alpha Centauri"),
    ("Open Alpha", "Alpha", GenericTaskOperation.OPEN_ONLY, None),
])
def test_generic_task_decomposes_application_and_operation_payload(
    command: str, application: str, operation: GenericTaskOperation,
    payload: str | None,
) -> None:
    intent = decompose_generic_task(command)
    assert intent.source_language == "en"
    assert intent.application_request == application
    assert intent.operation is operation
    assert intent.operation_target == payload


def test_generic_task_decomposes_app_target_and_preserves_conversation_role() -> None:
    direct = decompose_generic_task("Open Alpha and open Gamma")
    assert direct.application_request == "Alpha"
    assert direct.operation is GenericTaskOperation.ACTIVATE_TARGET
    assert direct.operation_target == "open Gamma"
    target = requested_target_spec(direct.operation_target, experimental_generic=True)
    assert target is not None and target.primary_identity == "Gamma"

    chat = decompose_generic_task("Open Alpha and open the conversation with Gamma")
    assert chat.application_request == "Alpha"
    assert chat.operation is GenericTaskOperation.ACTIVATE_TARGET
    chat_target = requested_target_spec(chat.operation_target, experimental_generic=True)
    assert chat_target is not None
    assert chat_target.primary_identity == "Gamma"
    assert chat_target.desired_role == "conversation"

    explicit_play = decompose_generic_task("Open Alpha and play Gamma")
    assert explicit_play.operation is GenericTaskOperation.ACTIVATE_TARGET
    assert explicit_play.application_request == "Alpha"
    assert explicit_play.operation_target == "play Gamma"


@pytest.mark.parametrize(("command", "application", "operation", "target"), [
    ("Abre WhatsApp", "WhatsApp", GenericTaskOperation.OPEN_ONLY, None),
    ("Abre WhatsApp y abre el chat con Iago", "WhatsApp",
     GenericTaskOperation.ACTIVATE_TARGET, "open the conversation with Iago"),
    ("Abre WhatsApp y abre la conversación con Iago", "WhatsApp",
     GenericTaskOperation.ACTIVATE_TARGET, "open the conversation with Iago"),
    ("Abre Spotify y busca Californication", "Spotify",
     GenericTaskOperation.SEARCH, "Californication"),
    ("Abre Spotify y busca por Californication", "Spotify",
     GenericTaskOperation.SEARCH, "Californication"),
    ("Busca Californication en Spotify", "Spotify",
     GenericTaskOperation.SEARCH, "Californication"),
    ("Busca War and Peace en Spotify", "Spotify",
     GenericTaskOperation.SEARCH, "War and Peace"),
    ("Abre Alpha y abre Gamma", "Alpha",
     GenericTaskOperation.ACTIVATE_TARGET, "open Gamma"),
])
def test_spanish_generic_decomposition_maps_into_shared_typed_intents(
    command: str, application: str, operation: GenericTaskOperation,
    target: str | None,
) -> None:
    intent = decompose_generic_task(command)

    assert intent.source_language == "es"
    assert intent.application_request == application
    assert intent.operation is operation
    assert intent.operation_target == target
    if command.endswith("Iago"):
        target_spec = requested_target_spec(intent.operation_target, experimental_generic=True)
        assert target_spec is not None
        assert target_spec.primary_identity == "Iago"
        assert target_spec.desired_role == "conversation"


@pytest.mark.parametrize("command", [
    "Abre WhatsApp y busca",
    "Busca Californication en Spotify en Alpha",
    "Abre Alpha y abre Gamma y abre Beta",
    "Busca Californication",
])
def test_ambiguous_or_incomplete_spanish_commands_fail_closed(command: str) -> None:
    intent = decompose_generic_task(command)
    assert intent.source_language == "es"
    assert intent.operation is GenericTaskOperation.UNSUPPORTED


def test_decomposition_diagnostics_expose_language_without_rewriting_literals() -> None:
    candidate = ApplicationCandidate(APP_ID, "Spotify", "test", launch_policy="allow")
    catalog = MemoryApplicationCatalog((candidate,))
    computer = FakeComputer([
        observation("before", elements=(query_field("before", focused=True),)),
    ])
    result = GenericTaskDebugAgent(
        computer, FakeDecisionMaker(), app_catalog=catalog,
    ).run("Abre Spotify y busca War and Peace")

    assert result.task_decomposition is not None
    assert result.task_decomposition.source_language == "es"
    assert result.task_decomposition.application_text == "Spotify"
    assert result.task_decomposition.operation_target_text == "War and Peace"
    assert result.task_decomposition.decomposition_method == "app_then_search"


@pytest.mark.parametrize("command", [
    "Open Alpha and search for",
    "Open Alpha and search for Beta and play it",
    "Open Alpha and type Beta",
    "Search Alpha in Beta in Gamma",
])
def test_ambiguous_or_unsupported_compound_requests_fail_closed(command: str) -> None:
    assert decompose_generic_task(command).operation is GenericTaskOperation.UNSUPPORTED


def test_search_operation_types_exact_query_and_stops_without_activating_result() -> None:
    candidate = ApplicationCandidate(APP_ID, "Alpha", "test", launch_policy="allow")
    catalog = MemoryApplicationCatalog((candidate,))
    before = observation("before", elements=(query_field("before", focused=True),))
    after_type = observation(
        "typed", elements=(query_field("typed", "War and Peace", focused=True),),
    )
    results = observation("results", elements=(
        query_field("results", "War and Peace", focused=True),
        UIElement(
            "c2", "War and Peace", "Button", rectangle=Rect(10, 50, 250, 90),
            enabled=True, visible=True,
        ),
    ))
    computer = FakeComputer([before, after_type, results])
    decision = FakeDecisionMaker("c2")

    result = GenericTaskDebugAgent(
        computer, decision, app_catalog=catalog,
    ).run("Open Alpha and search for War and Peace")

    assert result.success and result.stop_reason == "search_completed"
    assert result.task_decomposition is not None
    assert result.task_decomposition.application_text == "Alpha"
    assert result.task_decomposition.operation == "search"
    assert result.task_decomposition.operation_target_text == "War and Peace"
    assert result.task_decomposition.decomposition_method == "app_then_search"
    typed = [action for action, _ in computer.executed if isinstance(action, TypeAction)]
    assert typed == [TypeAction("War and Peace")]
    assert len(computer.query_submits) == 1
    assert result.final_target_activations == 0
    assert not decision.calls
    assert not computer.visual_activations
    assert not any(isinstance(action, ClickAction) for action, _ in computer.executed)
    assert result.target_resolution_diagnostics[-1].attempt_stage == "search-results-after-submit"
    assert result.target_resolution_diagnostics[-1].target_spec.primary_identity == "War and Peace"


def test_open_only_resolves_only_the_application_and_does_not_resolve_ui_target() -> None:
    candidate = ApplicationCandidate(APP_ID, "Alpha", "test", launch_policy="allow")
    catalog = MemoryApplicationCatalog((candidate,))
    computer = FakeComputer([observation("foreground", elements=(conversation(),))])
    result = GenericTaskDebugAgent(
        computer, FakeDecisionMaker("c1"), app_catalog=catalog,
    ).run("Open Alpha")

    assert result.success and result.stop_reason == "application_opened"
    assert result.task_decomposition is not None
    assert result.task_decomposition.application_text == "Alpha"
    assert result.task_decomposition.operation == "open_only"
    assert result.task_decomposition.operation_target_text is None
    assert not result.target_resolution_diagnostics
    assert not computer.executed


@pytest.mark.parametrize(("typed_value", "typed_app_id"), [
    ("Wrong", APP_ID),
    ("Beta", "different-app"),
])
def test_search_submit_still_requires_reobserved_literal_and_same_foreground(
    typed_value: str, typed_app_id: str,
) -> None:
    candidate = ApplicationCandidate(APP_ID, "Alpha", "test", launch_policy="allow")
    catalog = MemoryApplicationCatalog((candidate,))
    computer = FakeComputer([
        observation("before", elements=(query_field("before", focused=True),)),
        observation(
            "after-type", elements=(query_field("after-type", typed_value, focused=True),),
            app_id=typed_app_id,
        ),
    ])

    result = GenericTaskDebugAgent(
        computer, FakeDecisionMaker(), app_catalog=catalog,
    ).run("Open Alpha and search for Beta")

    assert not result.success and result.stop_reason == "query_submit_not_eligible"
    assert not computer.query_submits


def _run_californication_search(locals_: list[Observation], *,
                                directed: list[Observation] | None = None,
                                visual_provider: object | None = None,
                                field_value_result: VisualFieldValueRead | None = None,
                                agent_type: type[GenericTaskDebugAgent] = GenericTaskDebugAgent):
    candidate = ApplicationCandidate(APP_ID, "Alpha", "test", launch_policy="allow")
    computer = FakeComputer(
        locals_, directed, visual_provider=visual_provider,
    )
    if field_value_result is not None:
        computer.field_value_result = field_value_result
    result = agent_type(
        computer, FakeDecisionMaker(),
        app_catalog=MemoryApplicationCatalog((candidate,)),
    ).run("Open Alpha and search for Californication")
    return result, computer


def _agent_with_visual_binding_mutation(mutation: str) -> type[GenericTaskDebugAgent]:
    class MutatedBindingAgent(GenericTaskDebugAgent):
        def _verify_query_field_continuity_after_type(
            self, *args: Any, **kwargs: Any,
        ) -> Any:
            binding = self._active_visual_query_field_binding
            if mutation == "missing":
                self._active_visual_query_field_binding = None
            elif binding is not None and mutation == "ambiguous":
                self._active_visual_query_field_binding = replace(binding, unique=False)
            elif binding is not None and mutation == "candidate":
                self._active_visual_query_field_binding = replace(
                    binding, selected_candidate=None,
                )
            elif binding is not None and mutation == "snapshot":
                self._active_visual_query_field_binding = replace(
                    binding, original_snapshot_id="inconsistent-snapshot",
                )
            return super()._verify_query_field_continuity_after_type(*args, **kwargs)

    return MutatedBindingAgent


def _californication_results(snapshot: str) -> Observation:
    return observation(snapshot, elements=(UIElement(
        "c2", "Californication", "Button", rectangle=Rect(10, 60, 260, 100),
        enabled=True, visible=True, is_password=False,
    ),))


def test_search_submit_accepts_same_trusted_uia_field_with_exact_literal() -> None:
    result, computer = _run_californication_search([
        observation("before", elements=(query_field("before", focused=True),)),
        observation("typed", elements=(query_field(
            "typed", "Californication", focused=True,
        ),)),
        _californication_results("results"),
    ])

    assert result.success and result.stop_reason == "search_completed"
    assert result.query_submits == 1 and len(computer.query_submits) == 1
    assert result.query_submit_continuity is not None
    continuity = result.query_submit_continuity
    assert continuity.result == "verified" and continuity.source == "uia"
    assert continuity.reason == "verified"
    assert continuity.same_trusted_app and continuity.same_window
    assert continuity.field_identity_match and continuity.literal_confirmed
    assert continuity.field_focused_or_active is True
    assert continuity.structural_observation_complete is True
    assert not continuity.visual_fallback_eligible
    assert not continuity.visual_verification_attempted
    assert not result.target_resolution_diagnostics or all(
        item.attempt_stage != "target-resolution-after-type"
        for item in result.target_resolution_diagnostics
    )
    assert result.final_target_activations == 0 and not computer.visual_activations


@pytest.mark.parametrize("changed", ["app", "window"])
def test_search_submit_blocks_if_trusted_app_or_window_changes_after_typing(changed: str) -> None:
    after_type = observation(
        "typed",
        elements=(query_field("typed", "Californication", focused=True),),
        app_id="another-app" if changed == "app" else APP_ID,
        foreground_hwnd=200 if changed == "window" else 100,
    )
    result, computer = _run_californication_search([
        observation("before", elements=(query_field("before", focused=True),)),
        after_type,
    ])

    assert not result.success and result.stop_reason == "query_submit_not_eligible"
    assert result.query_submits == 0 and not computer.query_submits
    assert result.query_submit_continuity is not None
    assert result.query_submit_continuity.reason == "trusted_foreground_changed"


def test_search_submit_blocks_when_original_field_is_lost_for_another_editable() -> None:
    other_field = UIElement(
        "c9", "Message", "Edit", automation_id="message_input",
        rectangle=Rect(5, 5, 300, 40), enabled=True, visible=True,
        focused=True, is_password=False, observed_text="Californication",
    )
    result, computer = _run_californication_search([
        observation("before", elements=(query_field("before", focused=True),)),
        observation("typed", elements=(other_field,)),
    ])

    assert not result.success and result.stop_reason == "query_submit_not_eligible"
    assert not computer.query_submits
    assert result.query_submit_continuity is not None
    assert result.query_submit_continuity.reason == "focused_query_field_not_reestablished"


def test_search_submit_does_not_accept_query_text_outside_original_field() -> None:
    elsewhere = UIElement(
        "c2", "Recent query", "Text", rectangle=Rect(10, 60, 260, 100),
        enabled=True, visible=True, focused=False, observed_text="Californication",
    )
    result, computer = _run_californication_search([
        observation("before", elements=(query_field("before", focused=True),)),
        observation("typed", elements=(elsewhere,)),
    ])

    assert not result.success and result.stop_reason == "query_submit_not_eligible"
    assert not computer.query_submits
    assert result.query_submit_continuity is not None
    assert result.query_submit_continuity.reason == "focused_query_field_not_reestablished"
    assert not result.query_submit_continuity.literal_confirmed


def test_search_submit_accepts_unique_visual_continuity_of_active_same_field() -> None:
    original_field = VisualElement(
        "v-search", "Search", "search field", Rect(20, 20, 350, 60), None, True,
    )
    pre_type_field = VisualElement(
        "v-focused", "Search", "search field", Rect(20, 20, 350, 60), None, True,
    )
    post_type_field = VisualElement(
        "v-query", "Californication", "search field", Rect(20, 20, 350, 60), None, True,
        activity="active",
        field_label="Search", field_value="Californication",
        is_query_field=True, credential_risk=False,
    )
    local = [
        observation("initial", app_id=APP_ID, foreground_hwnd=77),
        observation("after-click", app_id=APP_ID, foreground_hwnd=77),
        observation("after-type", app_id=APP_ID, foreground_hwnd=77),
        _californication_results("results"),
    ]
    directed = [
        observation("field-selected", visual=(original_field,),
                    app_id=APP_ID, foreground_hwnd=77),
        observation("field-focused", visual=(pre_type_field,),
                    app_id=APP_ID, foreground_hwnd=77),
        observation("query-continuity", visual=(post_type_field,),
                    app_id=APP_ID, foreground_hwnd=77),
    ]
    result, computer = _run_californication_search(
        local, directed=directed, visual_provider=object(),
    )

    assert result.success and result.stop_reason == "search_completed"
    assert result.query_submits == 1 and len(computer.query_submits) == 1
    assert len(computer.phase3_query_submits) == 1
    assert computer.phase3_query_submits[0][1] is computer.phase3_query_submits[0][2]
    assert [action for action, _ in computer.executed if isinstance(action, TypeAction)] == [
        TypeAction("Californication"),
    ]
    assert len(computer.directed_calls) == 3
    assert computer.directed_calls[-1].verification_only is True
    assert "Californication" in computer.directed_calls[-1].objective
    assert result.visual_grounding_calls == 3
    assert result.query_submit_continuity is not None
    assert result.query_submit_continuity.result == "verified"
    assert result.query_submit_continuity.source == "visual"
    assert result.query_submit_continuity.literal_confirmed
    assert result.query_submit_continuity.field_focused_or_active is True
    assert result.query_submit_continuity.visual_verification_attempted


def _incomplete_visual_search(*, fresh_label: str = "¿Qué quieres reproducir?",
                              fresh_value: str | None = "Californication",
                              fresh_rect: Rect = Rect(20, 20, 350, 60),
                              fresh_activity: str | None = "active",
                              after_type_app: str = APP_ID,
                              after_type_hwnd: int = 77,
                              extra_fresh: tuple[VisualElement, ...] = (),
                              fresh_credential_risk: bool | None = False,
                              credential_risk: bool = False,
                              provider_failure: bool = False,
                              field_value_result: VisualFieldValueRead | None = None,
                              focused_role: str = "search field",
                              selected_fields: tuple[VisualElement, ...] | None = None,
                              agent_type: type[GenericTaskDebugAgent] = GenericTaskDebugAgent):
    selected_field = VisualElement(
        "v-search", "Search", "search field", Rect(20, 20, 350, 60), None, True,
    )
    focused_field = replace(selected_field, id="v-focused", role=focused_role)
    fresh_field = VisualElement(
        "v-query", fresh_label, "search field", fresh_rect, None, True,
        activity=fresh_activity,
        field_label=fresh_label, field_value=fresh_value,
        is_query_field=True, credential_risk=fresh_credential_risk,
    )
    after_elements: tuple[UIElement, ...] = ()
    if credential_risk:
        after_elements = (replace(
            query_field("after-type", "Californication", focused=True),
            is_password=True,
        ),)
    after_type = replace(
        observation(
            "after-type", elements=after_elements, app_id=after_type_app,
            foreground_hwnd=after_type_hwnd,
        ),
        inspection_errors=1,
    )
    fresh = observation(
        "query-continuity", visual=(fresh_field, *extra_fresh), app_id=APP_ID,
        foreground_hwnd=77,
    )
    if provider_failure:
        fresh = replace(
            fresh,
            visual_provider_error=ProviderErrorDiagnostic("timeout"),
            visual_grounding_status=VisualGroundingStatus.PROVIDER_ERROR,
        )
    local = [
        observation("initial", app_id=APP_ID, foreground_hwnd=77),
        observation("after-click", app_id=APP_ID, foreground_hwnd=77),
        after_type,
        _californication_results("results"),
    ]
    directed = [
        observation("field-selected", visual=selected_fields or (selected_field,),
                    app_id=APP_ID, foreground_hwnd=77),
        observation("field-focused", visual=(focused_field,),
                    app_id=APP_ID, foreground_hwnd=77),
        fresh,
    ]
    return _run_californication_search(
        local, directed=directed, visual_provider=object(),
        field_value_result=field_value_result, agent_type=agent_type,
    )


def test_visual_binding_survives_post_click_candidate_role_filtering() -> None:
    result, computer = _incomplete_visual_search(focused_role="editable field")

    assert result.success and result.query_submits == 1
    # The fresh structural post-type observation has no visual candidates.
    assert computer.observed_local[2].visual_elements == ()
    continuity = result.query_submit_continuity
    assert continuity is not None
    assert continuity.original_binding_present is True
    assert continuity.original_candidate_id == "v-search"
    assert continuity.original_snapshot_id == "field-selected"
    assert continuity.original_binding_unique is True
    assert continuity.original_binding_source == "strong_visual_unique_spatial_correspondence"
    assert continuity.original_geometry_available is True
    assert continuity.fresh_candidate_count == 1
    assert continuity.continuity_comparison_started is True


def test_missing_original_visual_binding_blocks_without_reconstructing_it() -> None:
    result, computer = _incomplete_visual_search(
        agent_type=_agent_with_visual_binding_mutation("missing"),
    )

    assert not result.success and result.query_submits == 0
    continuity = result.query_submit_continuity
    assert continuity is not None
    assert continuity.reason == "original_visual_query_field_binding_missing"
    assert continuity.original_binding_present is False
    assert continuity.original_candidate_id == "v-search"
    assert continuity.original_snapshot_id == "field-selected"
    assert continuity.continuity_comparison_started is False
    assert continuity.failure_stage == "original_binding"
    assert len(computer.directed_calls) == 2


def test_missing_selected_candidate_from_stored_visual_provenance_blocks() -> None:
    result, computer = _incomplete_visual_search(
        agent_type=_agent_with_visual_binding_mutation("candidate"),
    )

    assert not result.success and result.query_submits == 0
    continuity = result.query_submit_continuity
    assert continuity is not None
    assert continuity.reason == "original_visual_query_field_binding_inconsistent"
    assert continuity.original_binding_present is True
    assert continuity.original_binding_unique is False
    assert continuity.original_candidate_id == "v-search"
    assert continuity.original_geometry_available is False
    assert continuity.failure_stage == "original_binding"
    assert not continuity.continuity_comparison_started
    assert len(computer.directed_calls) == 2


def test_ambiguous_original_visual_binding_fails_closed() -> None:
    result, computer = _incomplete_visual_search(
        agent_type=_agent_with_visual_binding_mutation("ambiguous"),
    )

    assert not result.success and result.query_submits == 0
    continuity = result.query_submit_continuity
    assert continuity is not None
    assert continuity.reason == "original_visual_query_field_not_unique"
    assert continuity.original_binding_present is True
    assert continuity.original_binding_unique is False
    assert continuity.failure_stage == "original_binding"
    assert not continuity.continuity_comparison_started
    assert len(computer.directed_calls) == 2


def test_inconsistent_original_visual_snapshot_binding_fails_closed() -> None:
    result, computer = _incomplete_visual_search(
        agent_type=_agent_with_visual_binding_mutation("snapshot"),
    )

    assert not result.success and result.query_submits == 0
    continuity = result.query_submit_continuity
    assert continuity is not None
    assert continuity.reason == "original_visual_query_field_binding_inconsistent"
    assert continuity.original_binding_present is True
    assert continuity.original_candidate_id == "v-search"
    assert continuity.original_snapshot_id == "inconsistent-snapshot"
    assert continuity.failure_stage == "original_binding"
    assert not continuity.continuity_comparison_started
    assert len(computer.directed_calls) == 2


def test_visual_binding_can_reach_value_read_when_fresh_value_is_missing() -> None:
    result, computer = _incomplete_visual_search(
        fresh_value=None, focused_role="editable field",
        field_value_result=VisualFieldValueRead(
            "Californication", crop_valid=True, context_stable=True,
            credential_safe=True,
        ),
    )

    assert result.success and result.query_submits == 1
    continuity = result.query_submit_continuity
    assert continuity is not None
    assert continuity.original_binding_present and continuity.original_binding_unique
    assert continuity.continuity_comparison_started
    assert continuity.value_read_fallback_eligible
    assert continuity.value_read_fallback_attempted
    assert continuity.literal_relation == "exact"


def test_ambiguous_visual_query_discovery_never_creates_original_binding() -> None:
    first = VisualElement("v-search-a", "Search", "search field", Rect(20, 20, 350, 60), None, True)
    second = VisualElement("v-search-b", "Search in page", "search field", Rect(20, 80, 350, 120), None, True)
    result, computer = _incomplete_visual_search(selected_fields=(first, second))

    assert not result.success and result.query_submits == 0
    assert not computer.visual_activations
    diagnostics = result.query_field_diagnostics
    assert diagnostics is not None
    assert diagnostics.selection_reason == "multiple_visual_query_fields"
    assert diagnostics.selected_candidate_id is None
    assert result.query_submit_continuity is None


def test_incomplete_post_type_uia_uses_one_fresh_visual_continuity_check() -> None:
    result, computer = _incomplete_visual_search()

    assert result.success and result.query_submits == 1
    assert len(computer.directed_calls) == 3
    continuity = result.query_submit_continuity
    assert continuity is not None
    assert continuity.result == "verified"
    assert continuity.structural_observation_complete is False
    assert continuity.incompleteness_reasons == ("inspection_errors",)
    assert continuity.visual_fallback_eligible
    assert continuity.visual_verification_attempted
    assert continuity.visual_candidate_count == 1
    assert continuity.literal_relation == "exact"
    assert continuity.field_continuity_relation == "same"
    assert continuity.active_focused_relation == "active"
    assert continuity.credential_safe is True
    assert continuity.final_continuity_result == "verified"


def test_incomplete_post_type_visual_continuity_uses_field_value_not_placeholder() -> None:
    result, computer = _incomplete_visual_search()

    assert result.success and result.query_submits == 1
    assert len(computer.directed_calls) == 3
    continuity = result.query_submit_continuity
    assert continuity is not None
    assert continuity.result == "verified"
    assert continuity.literal_relation == "exact"
    assert continuity.visual_verification_attempted
    assert continuity.query_field_candidate_count == 1
    assert continuity.visual_candidates[0].field_label == "¿Qué quieres reproducir?"
    assert continuity.visual_candidates[0].field_value == "Californication"
    assert continuity.visual_candidates[0].literal_relation == "exact"
    assert computer.field_value_reads == []
    assert continuity.value_read_fallback_eligible is False
    assert continuity.value_read_fallback_attempted is False
    assert continuity.final_submit_release is True


def test_incomplete_post_type_visual_continuity_blocks_wrong_field_value() -> None:
    result, computer = _incomplete_visual_search(fresh_value="Californicatio")

    assert not result.success and result.query_submits == 0
    assert len(computer.directed_calls) == 3
    continuity = result.query_submit_continuity
    assert continuity is not None
    assert continuity.reason == "typed_literal_not_confirmed_visually"
    assert continuity.literal_relation == "mismatch"
    assert continuity.visual_candidates[0].candidate_acceptance == "blocked_literal_value_mismatch"


def test_incomplete_post_type_visual_continuity_blocks_missing_field_value() -> None:
    result, computer = _incomplete_visual_search(fresh_value=None)

    assert not result.success and result.query_submits == 0
    continuity = result.query_submit_continuity
    assert continuity is not None
    assert continuity.reason == "typed_literal_not_confirmed_visually"
    assert continuity.literal_relation == "missing"
    assert continuity.visual_candidates[0].field_value is None
    assert continuity.visual_candidates[0].candidate_acceptance == "blocked_literal_value_missing"
    assert len(computer.directed_calls) == 3
    assert continuity.value_read_fallback_eligible is True
    assert continuity.value_read_fallback_attempted is True
    assert continuity.crop_valid is True
    assert continuity.extracted_field_value is None
    assert continuity.final_submit_release is False
    assert len(computer.field_value_reads) == 1


def test_missing_field_value_uses_one_bounded_read_and_exact_match_releases_submit() -> None:
    result, computer = _incomplete_visual_search(
        fresh_value=None, field_value_result=VisualFieldValueRead(
        "Californication", crop_valid=True, context_stable=True, credential_safe=True,
        provider_attempts=(VisualProviderAttempt("fixture", "crop-reader", 12, "success"),),
        ),
    )

    assert result.success and result.query_submits == 1
    assert len(computer.field_value_reads) == 1
    continuity = result.query_submit_continuity
    assert continuity is not None
    assert continuity.value_read_fallback_eligible
    assert continuity.value_read_fallback_attempted
    assert continuity.crop_valid is True
    assert continuity.extracted_field_value == "Californication"
    assert continuity.literal_relation == "exact"
    assert continuity.value_read_provider_attempts[0].result_class == "success"
    assert continuity.final_submit_release is True


def test_cropped_field_value_mismatch_blocks_submit() -> None:
    result, computer = _incomplete_visual_search(
        fresh_value=None,
        field_value_result=VisualFieldValueRead(
            "Californicatio", crop_valid=True, context_stable=True, credential_safe=True,
        ),
    )

    assert not result.success and result.query_submits == 0
    continuity = result.query_submit_continuity
    assert continuity is not None
    assert continuity.extracted_field_value == "Californicatio"
    assert continuity.literal_relation == "mismatch"
    assert continuity.crop_valid is True
    assert continuity.final_submit_release is False
    assert len(computer.field_value_reads) == 1


def test_cropped_field_value_missing_blocks_submit() -> None:
    result, computer = _incomplete_visual_search(fresh_value=None)

    assert not result.success and result.query_submits == 0
    continuity = result.query_submit_continuity
    assert continuity is not None
    assert continuity.literal_relation == "missing"
    assert continuity.crop_valid is True
    assert continuity.final_submit_release is False
    assert len(computer.field_value_reads) == 1


def test_cropped_value_provider_failure_blocks_submit() -> None:
    result, computer = _incomplete_visual_search(
        fresh_value=None,
        field_value_result=VisualFieldValueRead(
            "Californication", crop_valid=True, context_stable=True, credential_safe=True,
            error=ProviderErrorDiagnostic("timeout"),
            provider_attempts=(VisualProviderAttempt(
                "fixture", "crop-reader", 20_000, "provider_failure", "timeout",
            ),),
        ),
    )

    assert not result.success and result.query_submits == 0
    continuity = result.query_submit_continuity
    assert continuity is not None
    assert continuity.reason == "visual_value_read_provider_failure"
    assert continuity.literal_relation == "unavailable"
    assert continuity.value_read_provider_attempts[0].result_class == "provider_failure"
    assert not continuity.final_submit_release
    assert len(computer.field_value_reads) == 1


def test_context_change_before_crop_blocks_submit() -> None:
    result, computer = _incomplete_visual_search(
        fresh_value=None,
        field_value_result=VisualFieldValueRead(
            "Californication", crop_valid=True, context_stable=False, credential_safe=True,
            reason="window_geometry_changed",
        ),
    )

    assert not result.success and result.query_submits == 0
    continuity = result.query_submit_continuity
    assert continuity is not None
    assert continuity.reason == "visual_value_read_context_changed"
    assert continuity.literal_relation == "unavailable"
    assert not continuity.final_submit_release
    assert len(computer.field_value_reads) == 1


def test_credential_risk_appearing_before_crop_blocks_submit() -> None:
    result, computer = _incomplete_visual_search(
        fresh_value=None,
        field_value_result=VisualFieldValueRead(
            "Californication", crop_valid=True, context_stable=True, credential_safe=False,
        ),
    )

    assert not result.success and result.query_submits == 0
    continuity = result.query_submit_continuity
    assert continuity is not None
    assert continuity.reason == "visual_value_read_credential_risk"
    assert continuity.credential_safe is False
    assert not continuity.final_submit_release
    assert len(computer.field_value_reads) == 1


def test_unbounded_field_geometry_blocks_value_read_before_attempt() -> None:
    result, computer = _incomplete_visual_search(
        fresh_value=None,
        field_value_result=VisualFieldValueRead(
            "Californication", crop_valid=False, context_stable=True,
            credential_safe=True, reason="crop_invalid",
        ),
    )

    assert not result.success and result.query_submits == 0
    continuity = result.query_submit_continuity
    assert continuity is not None
    assert continuity.reason == "visual_value_read_crop_invalid"
    assert continuity.value_read_fallback_eligible
    assert continuity.value_read_fallback_attempted
    assert continuity.crop_valid is False
    assert len(computer.field_value_reads) == 1


def test_query_literal_in_nearby_result_is_not_field_value_evidence() -> None:
    nearby_result = VisualElement(
        "v-result", "Californication", "list item", Rect(20, 100, 300, 150), None, True,
        is_query_field=False, credential_risk=False,
    )
    result, _computer = _incomplete_visual_search(
        fresh_value=None, extra_fresh=(nearby_result,),
    )

    assert not result.success and result.query_submits == 0
    continuity = result.query_submit_continuity
    assert continuity is not None
    assert continuity.query_field_candidate_count == 1
    assert continuity.visual_candidates[0].field_value is None
    assert continuity.visual_candidates[1].is_query_field is False
    assert continuity.visual_candidates[1].field_value is None
    assert continuity.reason == "typed_literal_not_confirmed_visually"


def test_query_literal_in_field_label_without_field_value_is_not_confirmation() -> None:
    result, _computer = _incomplete_visual_search(
        fresh_label="Californication", fresh_value=None,
    )

    assert not result.success and result.query_submits == 0
    continuity = result.query_submit_continuity
    assert continuity is not None
    assert continuity.visual_candidates[0].field_label == "Californication"
    assert continuity.visual_candidates[0].field_value is None
    assert continuity.literal_relation == "missing"


def test_visual_query_continuity_normalizes_value_only_for_comparison() -> None:
    result, computer = _incomplete_visual_search(
        fresh_value="  CALIFORNICATION  ",
    )

    assert result.success and result.query_submits == 1
    typed_actions = [action for action, _ in computer.executed if isinstance(action, TypeAction)]
    assert typed_actions == [TypeAction("Californication")]
    continuity = result.query_submit_continuity
    assert continuity is not None and continuity.literal_relation == "exact"


def test_visual_query_continuity_blocks_credential_risk_even_when_value_matches() -> None:
    result, computer = _incomplete_visual_search(fresh_credential_risk=True)

    assert not result.success and result.query_submits == 0
    continuity = result.query_submit_continuity
    assert continuity is not None
    assert continuity.reason == "credential_sensitive_context"
    assert continuity.credential_safe is False
    assert continuity.visual_candidates[0].field_value == "Californication"
    assert continuity.visual_candidates[0].candidate_acceptance == "blocked_credential_risk"
    assert len(computer.directed_calls) == 3


def test_visual_query_continuity_blocks_multiple_fields_even_with_same_value() -> None:
    second_field = VisualElement(
        "v-second", "Secondary search", "search field", Rect(420, 20, 750, 60), None, True,
        activity="active", field_label="Secondary search",
        field_value="Californication", is_query_field=True, credential_risk=False,
    )
    result, computer = _incomplete_visual_search(extra_fresh=(second_field,))

    assert not result.success and result.query_submits == 0
    continuity = result.query_submit_continuity
    assert continuity is not None
    assert continuity.reason == "multiple_visual_query_fields"
    assert continuity.query_field_candidate_count == 2
    assert all(item.candidate_acceptance == "blocked_ambiguous"
               for item in continuity.visual_candidates if item.is_query_field)
    assert len(computer.directed_calls) == 3


def test_query_submit_visual_candidate_diagnostics_are_bounded_redacted_and_coordinate_free() -> None:
    token = "sk-test-placeholder-for-redaction"
    extra = VisualElement(
        "v-status", "Californication " + token + (" x" * 100), "text",
        Rect(360, 20, 500, 60), None, False,
    )
    result, _computer = _incomplete_visual_search(
        fresh_value=token + (" x" * 100), extra_fresh=(extra,),
    )

    continuity = result.query_submit_continuity
    assert continuity is not None
    assert continuity.visual_candidate_count == 2
    assert continuity.query_field_candidate_count == 1
    assert [item.is_query_field for item in continuity.visual_candidates] == [True, False]
    assert continuity.visual_candidates[0].literal_relation == "mismatch"
    summary = continuity.visual_candidates[0]
    assert token not in (summary.field_value or "")
    assert "[REDACTED]" in (summary.field_value or "")
    assert len(summary.field_value or "") <= 100
    assert continuity.visual_candidates[1].field_value is None
    assert continuity.visual_candidates[1].literal_relation == "not_applicable"
    serialized = str(asdict(continuity))
    assert all(key not in serialized for key in ("left", "top", "right", "bottom"))


def test_incomplete_post_type_visual_continuity_blocks_inactive_field() -> None:
    result, computer = _incomplete_visual_search(fresh_activity="not_active")

    assert not result.success and result.query_submits == 0
    continuity = result.query_submit_continuity
    assert continuity is not None
    assert continuity.reason == "visual_query_field_not_active"
    assert continuity.active_focused_relation == "inactive"
    assert continuity.visual_verification_attempted


@pytest.mark.parametrize(("changed", "kwargs"), [
    ("app", {"after_type_app": "different-app"}),
    ("window", {"after_type_hwnd": 88}),
])
def test_incomplete_post_type_visual_continuity_blocks_changed_trusted_context(
    changed: str, kwargs: dict[str, object],
) -> None:
    result, computer = _incomplete_visual_search(**kwargs)

    assert not result.success and result.query_submits == 0
    assert len(computer.directed_calls) == 2
    continuity = result.query_submit_continuity
    assert continuity is not None
    assert continuity.reason == "trusted_foreground_changed"
    assert not continuity.visual_verification_attempted
    assert not continuity.visual_fallback_eligible
    assert continuity.failure_stage == "trusted_context"
    assert not continuity.continuity_comparison_started


def test_incomplete_post_type_visual_continuity_blocks_credential_risk() -> None:
    result, computer = _incomplete_visual_search(credential_risk=True)

    assert not result.success and result.query_submits == 0
    assert len(computer.directed_calls) == 2
    continuity = result.query_submit_continuity
    assert continuity is not None
    assert continuity.reason == "credential_sensitive_context"
    assert continuity.credential_safe is False
    assert not continuity.visual_verification_attempted


def test_incomplete_post_type_visual_provider_failure_blocks_enter() -> None:
    result, computer = _incomplete_visual_search(provider_failure=True)

    assert not result.success and result.query_submits == 0
    assert len(computer.directed_calls) == 3
    continuity = result.query_submit_continuity
    assert continuity is not None
    assert continuity.reason == "visual_observation_incomplete_or_context_changed"
    assert continuity.visual_verification_attempted
    assert continuity.visual_candidate_count == 1
    assert continuity.literal_relation == "unavailable"
    assert continuity.final_continuity_result == "blocked"


def test_search_submit_blocks_password_field() -> None:
    password = replace(
        query_field("typed", "Californication", focused=True), is_password=True,
    )
    result, computer = _run_californication_search([
        observation("before", elements=(query_field("before", focused=True),)),
        observation("typed", elements=(password,)),
    ])

    assert not result.success and result.stop_reason == "query_submit_not_eligible"
    assert not computer.query_submits
    assert result.query_submit_continuity is not None
    assert result.query_submit_continuity.reason == "credential_sensitive_context"


def test_search_submit_blocks_multiple_query_fields() -> None:
    second = replace(
        query_field("typed-second", "Californication", focused=False),
        id="c2", automation_id="secondary_search_query",
    )
    result, computer = _run_californication_search([
        observation("before", elements=(query_field("before", focused=True),)),
        observation("typed", elements=(
            query_field("typed", "Californication", focused=True), second,
        )),
    ])

    assert not result.success and result.stop_reason == "query_submit_not_eligible"
    assert not computer.query_submits
    assert result.query_submit_continuity is not None
    assert result.query_submit_continuity.reason == "multiple_query_fields_after_typing"


def test_search_submit_blocks_literal_mismatch_in_original_field() -> None:
    result, computer = _run_californication_search([
        observation("before", elements=(query_field("before", focused=True),)),
        observation("typed", elements=(query_field("typed", "Californiacation", focused=True),)),
    ])

    assert not result.success and result.stop_reason == "query_submit_not_eligible"
    assert not computer.query_submits
    assert result.query_submit_continuity is not None
    assert result.query_submit_continuity.reason == "typed_literal_not_confirmed_in_query_field"
    assert not result.query_submit_continuity.literal_confirmed


def test_visible_uia_chat_launches_if_needed_without_grounding_type_or_enter() -> None:
    computer = FakeComputer([
        observation("before", app_id=""),
        observation("opened", elements=(conversation(),)),
        active_conversation("after"),
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
    computer.activation_target_probe = (
        lambda _candidate: lambda: TrustedApplicationRuntimeState()
    )
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
    assert result.activation_lifecycle is not None
    assert result.activation_lifecycle.timing is not None
    assert result.activation_lifecycle.timing.open_app_action_elapsed_ms == 0
    assert result.activation_lifecycle.timing.activation_wait_started_since_launch_ms == 0
    assert diagnostic.initial_foreground.trusted_app_id == "old-app"
    assert diagnostic.final_foreground.trusted_app_id is None
    assert diagnostic.observed_foregrounds[-1].trusted_app_id is None
    assert [action.kind for action, _ in computer.executed] == ["open_app"]
    assert not decision.calls and not computer.directed_calls
    assert not computer.visual_activations and not computer.query_submits
    assert result.literal_types == result.query_submits == result.final_target_activations == 0


def test_generic_open_app_requests_only_the_bound_trusted_window_activation() -> None:
    computer = FakeComputer([
        observation("background", app_id="old-app"),
        observation("fresh-foreground", app_id=APP_ID),
    ], visual_provider=object())
    catalog = chat_catalog()
    candidate = catalog.resolve(APP_ID)
    assert candidate is not None
    probe_calls: list[str] = []
    activation_calls: list[str] = []
    stable_background = TrustedApplicationRuntimeState(
        process_observed=True, window_observed=True, visible=True, minimized=False,
        window_foreground=False, trusted_identity_match=True, probe_complete=True,
        window_stable=True,
    )

    def make_probe(requested):
        assert requested.id == APP_ID

        def probe():
            probe_calls.append(requested.id)
            return stable_background

        return probe

    def activate(requested):
        activation_calls.append(requested.id)
        return TrustedWindowActivationResult(
            True, "eligible", True, False, True,
            "activate", True, foreground_verified=True,
        )

    computer.activation_target_probe = make_probe
    computer.activate_trusted_application_window = activate
    decision = FakeDecisionMaker("c1")
    agent = GenericTaskDebugAgent(
        computer, decision, app_catalog=catalog,
        policy=GenericTargetActivationPolicy(catalog),
    )

    result = agent._wait_for_application(candidate, observation("initial", app_id="old-app"))

    assert result.reason == "activated"
    assert result.observation.observation_id == "fresh-foreground"
    assert activation_calls == [APP_ID]
    assert probe_calls == [APP_ID]
    assert not decision.calls and not computer.directed_calls
    assert not computer.visual_activations and not computer.query_submits
    assert result.lifecycle.explicit_activation.success


def test_generic_post_deadline_probe_is_read_only_and_cannot_continue_the_agent() -> None:
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
        observation("during-1", app_id="old-app"),
        observation("during-2", app_id="old-app"),
        observation("during-3", app_id="old-app"),
    ], visual_provider=object())
    probe_calls = 0
    late_target = TrustedApplicationRuntimeState(
        process_observed=True, window_observed=True, foreground_observed=True,
        visible=True, minimized=False, window_foreground=True,
        trusted_identity_match=True,
    )

    def make_probe(candidate):
        assert candidate.id == APP_ID

        def probe():
            nonlocal probe_calls
            probe_calls += 1
            return (TrustedApplicationRuntimeState() if probe_calls <= 3 else late_target)

        return probe

    computer.activation_target_probe = make_probe
    decision = FakeDecisionMaker("c1")
    catalog = chat_catalog()
    result = GenericTaskDebugAgent(
        computer, decision, app_catalog=catalog,
        policy=GenericTargetActivationPolicy(catalog),
        budgets=GenericTaskBudgets(
            app_activation_timeout_seconds=.2, transition_poll_seconds=.1,
        ),
        clock=clock, sleep_fn=clock.sleep,
    ).run("Open WhatsApp and open the chat with Pablo García")

    assert result.stop_reason == "application_activation_timeout"
    assert result.activation_lifecycle is not None
    assert not result.activation_lifecycle.target_process_observed
    assert result.activation_lifecycle.post_deadline_probe.performed
    assert result.activation_lifecycle.post_deadline_probe.target_process_observed
    assert result.activation_lifecycle.post_deadline_probe.target_foreground_observed
    assert result.application_activation.activation_elapsed_ms == 200
    assert result.application_activation.final_foreground.trusted_app_id == "old-app"
    assert [action.kind for action, _ in computer.executed] == ["open_app"]
    assert not decision.calls
    assert not computer.directed_calls
    assert not computer.visual_activations
    assert not computer.query_submits
    assert result.literal_types == result.query_submits == result.final_target_activations == 0


def test_visible_visual_chat_uses_same_resolver_and_snapshot_bound_click() -> None:
    visual = VisualElement(
        "v1", "Pablo García", "conversation", Rect(20, 80, 300, 140),
        None, True, parent="Conversations",
    )
    computer = FakeComputer(
        [observation("initial"), active_conversation("after", "Pablo García")],
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
    assert action == VisualClickAction("visual-preclick-visual", "v1")
    assert bound_observation.observation_id == action.snapshot_id
    assert not computer.query_submits


def test_query_results_after_typing_skip_enter_and_activate_visible_target() -> None:
    search = query_field("before", focused=True)
    after_type = observation("typed", elements=(
        query_field("typed", "Pablo García"), conversation("c2"),
    ))
    computer = FakeComputer([
        observation("initial", elements=(search,)), after_type,
        active_conversation("after", "Pablo García"),
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


def test_domain_unknown_list_item_after_typing_skips_enter_and_activates() -> None:
    initial = observation("initial", elements=(query_field("initial", focused=True),))
    typed = observation("typed", elements=(
        query_field("typed", "Alex", focused=True),
        UIElement(
            "c2", "Alex", "ListItem", rectangle=Rect(10, 50, 250, 90),
            enabled=True, visible=True, focused=False, is_password=False,
        ),
    ))
    computer = FakeComputer([initial, typed, active_conversation("after", "Alex")])
    decision = FakeDecisionMaker("c2")

    result = GenericTaskDebugAgent(computer, decision).run("Select Alex")

    assert result.success and result.target_resolution_status == "unique"
    assert result.literal_types == 1 and result.query_submits == 0
    assert result.final_target_activations == 1
    assert [action for action, _ in computer.executed if isinstance(action, TypeAction)] == [
        TypeAction("Alex"),
    ]
    assert not computer.query_submits
    assert any(isinstance(action, ClickAction) and action.target_id == "c2"
               for action, _ in computer.executed)
    after_type = result.target_resolution_diagnostics[-1]
    assert after_type.attempt_stage == "target-resolution-after-type"
    matching = next(row for row in after_type.frontier_evidence if row.candidate_id == "c2")
    assert matching.target_semantic_evidence is None
    assert matching.presentation_role == "list_item"
    assert matching.admissible


def test_search_submits_once_only_after_typed_value_is_reobserved_and_transition_seen() -> None:
    pre = observation("initial", elements=(query_field("initial", focused=True),))
    typed = observation("typed", elements=(query_field("typed", "Pablo García", focused=True),))
    result_state = observation("results", elements=(
        query_field("results", "Pablo García", focused=True), conversation("c2"),
    ))
    post = active_conversation("after", "Pablo García")
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
            observation("initial", foreground_hwnd=77),
            observation("focused", elements=(query_field("focused", focused=True),),
                        foreground_hwnd=77),
            observation("typed", elements=(query_field("typed", "Pablo García"), conversation("c2")),
                        foreground_hwnd=77),
            active_conversation("after", "Pablo García", foreground_hwnd=77),
        ],
        [observation("target-ground", visual=(search,), foreground_hwnd=77),
         observation("field-ground", visual=(search,), foreground_hwnd=77)],
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
    assert result.query_field_diagnostics.verification.query_field_verification_method == "strong_local"
    assert len(computer.directed_calls) == 2
    assert computer.phase2_type_calls == []


def _strong_visual_query_flow(
    verification: Observation | None = None,
    *,
    post_click: Observation | None = None,
) -> tuple[GenericTaskDebugAgent, FakeComputer]:
    clicked = VisualElement(
        "v-search", "Search", "search field", Rect(20, 20, 350, 60), None, True,
    )
    verify_observation = verification or observation(
        "verify-field", visual=(VisualElement(
            "v-verified", "Search", "search field", Rect(20, 20, 350, 60), None, True,
        ),), foreground_hwnd=77,
    )
    after_click = post_click or observation(
        "after-click", elements=(UIElement(
            "c-pane", "Search Area", "Pane", enabled=True, visible=True,
            focused=True, is_password=False,
        ),), foreground_hwnd=77,
    )
    computer = FakeComputer(
        [
            observation("initial", foreground_hwnd=77), after_click,
            observation("typed", elements=(conversation("c2", "Iago"),),
                        foreground_hwnd=77),
            active_conversation("after", "Iago", foreground_hwnd=77),
        ],
        [
            readiness_visual("target-ground", foreground_hwnd=77),
            observation("clicked-query-field", visual=(clicked,), foreground_hwnd=77),
            verify_observation,
        ],
        visual_provider=object(),
    )
    return GenericTaskDebugAgent(computer, FakeDecisionMaker("c2")), computer


def test_strong_visual_query_field_verification_releases_exact_literal_and_skips_enter() -> None:
    agent, computer = _strong_visual_query_flow()
    type_budget_at_input: list[int] = []
    computer.on_type = lambda _action, _observation, visual_verified: (
        type_budget_at_input.append(agent._active_budget.literal_types)
        if visual_verified else None
    )

    result = agent.run("Open WhatsApp and open the chat with Iago")

    assert result.success and result.stop_reason == "target_activated"
    assert result.literal_types == 1
    assert result.visual_grounding_calls == 3
    assert len(computer.directed_calls) == 3
    assert len(computer.phase2_type_calls) == 1
    typed_action, typed_observation, visual_verified = computer.phase2_type_calls[0]
    assert typed_action == TypeAction("Iago")
    assert typed_observation.observation_id == "verify-field"
    assert visual_verified is True
    assert type_budget_at_input == [1]
    assert result.post_action_observation_obtained
    assert computer.query_submits == []
    verification = result.query_field_diagnostics.verification
    assert verification.query_field_verification_method == "strong_visual"
    assert verification.strong_visual_verification_available
    assert verification.strong_visual_verification_attempted
    assert verification.strong_visual_verification_used
    assert verification.strong_visual_verification_result == "verified"
    assert verification.strong_visual_grounding_objective == (
        "Verify the visible text/search/query field that was just activated and is "
        "appropriate for entering the requested literal."
    )
    assert verification.strong_visual_candidate_count == 1
    assert verification.strong_visual_candidate_ids == ("v-verified",)
    assert verification.spatial_correspondence_result == "unique_match"
    assert verification.semantic_correspondence_result == "compatible"
    assert verification.credential_safety_result == "clear"
    assert verification.window_geometry_stable is True


def test_fresh_strong_local_query_field_prevents_visual_verification_call() -> None:
    focused_field = query_field("after-click", focused=True)
    agent, computer = _strong_visual_query_flow(post_click=observation(
        "after-click", elements=(focused_field,), foreground_hwnd=77,
    ))

    result = agent.run("Open WhatsApp and open the chat with Iago")

    assert result.success and result.literal_types == 1
    assert len(computer.directed_calls) == 2
    assert not computer.phase2_type_calls
    assert any(isinstance(action, TypeAction) for action, _ in computer.executed)
    assert result.query_field_diagnostics.verification.query_field_verification_method == "strong_local"


@pytest.mark.parametrize(
    ("verification", "expected_spatial", "expected_semantic"),
    [
        (observation("different-field", visual=(VisualElement(
            "v-different", "Other Search", "search field", Rect(500, 300, 750, 360), None, True,
        ),), foreground_hwnd=77), "no_spatial_match", "compatible"),
        (observation("wrong-role", visual=(VisualElement(
            "v-button", "Open", "button", Rect(20, 20, 350, 60), None, True,
        ),), foreground_hwnd=77), "not_evaluated", "no_compatible_candidate"),
        (observation("multiple-fields", visual=(
            VisualElement("v-match-1", "Search", "search field", Rect(20, 20, 350, 60), None, True),
            VisualElement("v-match-2", "Query", "text field", Rect(20, 20, 350, 60), None, True),
        ), foreground_hwnd=77), "multiple_matches", "compatible"),
    ],
)
def test_strong_visual_requires_unique_spatially_corresponding_text_field(
    verification, expected_spatial, expected_semantic,
) -> None:
    agent, computer = _strong_visual_query_flow(verification)

    result = agent.run("Open WhatsApp and open the chat with Iago")

    assert not result.success and result.stop_reason == "query_field_not_verified"
    assert result.literal_types == 0
    assert not any(isinstance(action, TypeAction) for action, _ in computer.executed)
    diagnostics = result.query_field_diagnostics.verification
    assert diagnostics.query_field_verification_method == "none"
    assert diagnostics.spatial_correspondence_result == expected_spatial
    assert diagnostics.semantic_correspondence_result == expected_semantic
    assert not diagnostics.strong_visual_verification_used


def test_strong_visual_rejects_changed_window_geometry() -> None:
    verification = observation("moved-window", visual=(VisualElement(
        "v-new", "Search", "search field", Rect(20, 20, 350, 60), None, True,
    ),), foreground_hwnd=77)
    verification = replace(
        verification,
        screenshot=replace(
            verification.screenshot, window_bounds=Rect(0, 0, 801, 600),
        ),
    )
    agent, computer = _strong_visual_query_flow(verification)

    result = agent.run("Open WhatsApp and open the chat with Iago")

    assert not result.success and result.literal_types == 0
    diagnostics = result.query_field_diagnostics.verification
    assert diagnostics.window_geometry_stable is False
    assert diagnostics.strong_visual_verification_result == "window_geometry_changed"
    assert computer.phase2_type_calls == []


def test_password_or_credential_evidence_blocks_strong_visual_verification() -> None:
    password = UIElement(
        "c-password", "Access token", "Edit", enabled=True, visible=True,
        focused=False, is_password=True,
    )
    agent, computer = _strong_visual_query_flow(post_click=observation(
        "after-click", elements=(password,), foreground_hwnd=77,
    ))

    result = agent.run("Open WhatsApp and open the chat with Iago")

    assert not result.success and result.literal_types == 0
    assert len(computer.directed_calls) == 2
    diagnostics = result.query_field_diagnostics.verification
    assert diagnostics.credential_safety_result == "blocked"
    assert diagnostics.strong_visual_verification_result == "credential_sensitive_context"
    assert not diagnostics.strong_visual_verification_attempted


def test_credential_evidence_in_fresh_visual_response_blocks_and_is_reported() -> None:
    credential_labeled = observation("credential-verify", visual=(VisualElement(
        "v-secret", "Access token", "search field", Rect(20, 20, 350, 60), None, True,
    ),), foreground_hwnd=77)
    agent, computer = _strong_visual_query_flow(credential_labeled)

    result = agent.run("Open WhatsApp and open the chat with Iago")

    assert not result.success and result.literal_types == 0
    diagnostics = result.query_field_diagnostics.verification
    assert diagnostics.strong_visual_verification_result == "credential_sensitive_context"
    assert diagnostics.credential_safety_result == "blocked"
    assert not diagnostics.strong_visual_verification_used
    assert not computer.phase2_type_calls


def test_foreground_change_blocks_strong_visual_verification() -> None:
    agent, computer = _strong_visual_query_flow(post_click=observation(
        "other-foreground", app_id="other_app_1234567890", process_id=55,
        foreground_hwnd=98,
    ))

    result = agent.run("Open WhatsApp and open the chat with Iago")

    assert not result.success and result.literal_types == 0
    assert len(computer.directed_calls) == 2
    diagnostics = result.query_field_diagnostics.verification
    assert diagnostics.foreground_stable is False
    assert diagnostics.strong_visual_verification_result == "foreground_context_changed"


def test_stale_visual_click_fails_before_verification() -> None:
    agent, computer = _strong_visual_query_flow()
    computer.visual_click_result = ActionResult(
        False, VisualClickAction("clicked-query-field", "v-search"),
        "stale visual click", error="stale_observation", input_issued=False,
    )

    result = agent.run("Open WhatsApp and open the chat with Iago")

    assert not result.success and result.stop_reason == "query_field_activation_failed"
    assert result.literal_types == 0
    assert len(computer.directed_calls) == 2


def test_already_used_query_field_verification_budget_blocks_another_provider_call() -> None:
    agent, computer = _strong_visual_query_flow()

    def consume_verification_budget(_action, _observation) -> None:
        agent._active_budget.query_field_visual_verification_calls = 1

    computer.on_visual_click = consume_verification_budget
    result = agent.run("Open WhatsApp and open the chat with Iago")

    assert not result.success and result.literal_types == 0
    assert len(computer.directed_calls) == 2
    assert len(computer.visual_activations) == 1
    diagnostics = result.query_field_diagnostics.verification
    assert diagnostics.strong_visual_verification_result == "verification_budget_exhausted"
    assert not diagnostics.strong_visual_verification_attempted
    assert not computer.phase2_type_calls


def test_exhausted_visual_grounding_budget_blocks_query_field_verification() -> None:
    from agent.generic_task import GenericTaskBudgets

    agent, computer = _strong_visual_query_flow()
    agent.budgets = GenericTaskBudgets(visual_grounding_calls=2)

    result = agent.run("Open WhatsApp and open the chat with Iago")

    assert not result.success and result.literal_types == 0
    assert len(computer.directed_calls) == 2
    diagnostics = result.query_field_diagnostics.verification
    assert not diagnostics.strong_visual_verification_available
    assert not diagnostics.strong_visual_verification_attempted
    assert diagnostics.strong_visual_verification_result == "visual_grounding_budget_exhausted"


def test_empty_or_failed_strong_visual_provider_response_never_types() -> None:
    empty = readiness_visual("empty-verify", foreground_hwnd=77)
    failed = replace(
        observation("failed-verify", foreground_hwnd=77),
        visual_provider_error=ProviderErrorDiagnostic("timeout"),
    )
    for verification, expected in ((empty, "provider_empty"), (failed, "provider_error")):
        agent, computer = _strong_visual_query_flow(verification)
        result = agent.run("Open WhatsApp and open the chat with Iago")
        assert not result.success and result.literal_types == 0
        assert not any(isinstance(action, TypeAction) for action, _ in computer.executed)
        diagnostics = result.query_field_diagnostics.verification
        assert diagnostics.strong_visual_verification_attempted
        assert diagnostics.strong_visual_verification_result == expected
        assert computer.phase2_type_calls == []


def test_target_resolution_diagnostics_bound_uia_and_visual_rejections() -> None:
    blocked_uia = UIElement(
        "c1", "Iago", "Button", rectangle=Rect(10, 10, 100, 40),
        enabled=True, visible=True, parent_name="Delete",
    )
    local = observation("initial", elements=(blocked_uia,))
    visible_chats = tuple(
        VisualElement(
            f"v{index}", f"Iago chat {index}", "conversation",
            Rect(5, 10 + index * 50, 250, 45 + index * 50), None, True,
        )
        for index in range(7)
    )
    blocked_visual = VisualElement(
        "v-delete", "Delete Iago", "conversation", Rect(5, 400, 250, 440), None, True,
    )
    grounded = replace(
        observation("target-ground", elements=(blocked_uia,),
                    visual=(*visible_chats, blocked_visual)),
        visual_pipeline=VisualPipelineDiagnostic(
            provider_requested_max_elements=5, provider_raw_element_count=18,
            parsed_element_count=9, validated_element_count=8,
            deduplicated_element_count=8, observation_visual_control_count=8,
        ),
    )
    computer = FakeComputer([local], [grounded], visual_provider=object())
    decision = FakeDecisionMaker()

    result = GenericTaskDebugAgent(computer, decision).run(
        "Open WhatsApp and open the chat with Iago",
    )

    assert not result.success and result.stop_reason == "target_ambiguous"
    assert not decision.calls and not computer.executed and not computer.visual_activations
    assert len(result.target_resolution_diagnostics) == 1
    diagnostic = result.target_resolution_diagnostics[0]
    assert diagnostic.attempt_stage == "target-resolution"
    assert diagnostic.target_spec.primary_identity.casefold() == "iago"
    assert diagnostic.target_spec.desired_role == "conversation"
    assert diagnostic.uia.raw_candidate_count == 1
    assert diagnostic.uia.adapted_candidate_count == 1
    assert diagnostic.uia.rejected_candidate_count == 1
    assert dict(diagnostic.uia.rejection_reason_counts)["safety_rejected"] == 1
    assert diagnostic.visual.grounding_called
    assert "Iago" in diagnostic.visual.grounding_objective
    assert diagnostic.visual.raw_candidate_count == 8
    assert diagnostic.visual.provider_raw_element_count == 18
    assert diagnostic.visual.parsed_element_count == 9
    assert diagnostic.visual.validated_element_count == 8
    assert diagnostic.visual.rejected_candidate_count == 1
    assert dict(diagnostic.visual.rejection_reason_counts)["safety_rejected"] == 1
    assert diagnostic.status == "ambiguous"
    assert diagnostic.frontier_candidate_count == 7
    assert len(diagnostic.frontier_evidence) == 5
    assert len(diagnostic.near_matches) <= 5
    assert any("safety_rejected" in item.rejection_reasons
               for item in diagnostic.near_matches)
    assert all(
        item.primary_text is None or len(item.primary_text) <= 100
        for item in (*diagnostic.frontier_evidence, *diagnostic.near_matches)
    )
    serialized = asdict(diagnostic)
    all_keys = set()
    def collect_keys(value: object) -> None:
        if isinstance(value, dict):
            all_keys.update(value)
            for child in value.values():
                collect_keys(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                collect_keys(child)
    collect_keys(serialized)
    assert not all_keys.intersection({"rectangle", "coordinates", "click_center", "x", "y"})


def test_visual_query_field_failure_reports_selection_and_fresh_uia_facts() -> None:
    search = VisualElement(
        "v-search", "Search", "search field", Rect(20, 20, 350, 60), None, True,
    )
    after_click = observation(
        "after-click", elements=(query_field("after-click", focused=False),),
        foreground_hwnd=77,
    )
    computer = FakeComputer(
        [observation("initial", foreground_hwnd=77), after_click],
        [readiness_visual("target-ground", foreground_hwnd=77),
         observation("field-ground", visual=(search,), foreground_hwnd=77)],
        visual_provider=object(),
    )

    result = GenericTaskDebugAgent(computer, FakeDecisionMaker()).run(
        "Open WhatsApp and open the chat with Iago",
    )

    assert not result.success and result.stop_reason == "query_field_not_verified"
    assert result.literal_types == 0
    assert not any(isinstance(action, TypeAction) for action, _ in computer.executed)
    assert len(computer.visual_activations) == 1
    diagnostics = result.query_field_diagnostics
    assert diagnostics is not None
    assert diagnostics.grounding_called
    assert diagnostics.grounding_objective
    assert diagnostics.selected_candidate_id == "v-search"
    assert diagnostics.selected_source == "VISUAL"
    assert diagnostics.selected_snapshot_id == "field-ground"
    assert diagnostics.selection_reason == "unique_visual_query_field"
    assert diagnostics.semantic_safety_eligible is True
    assert diagnostics.visual_candidates[0].label == "Search"
    assert diagnostics.visual_candidates[0].role == "search field"
    assert diagnostics.visual_candidates[0].clickable is True
    assert diagnostics.visual_candidates[0].considered_query_field
    assert diagnostics.visual_candidates[0].consideration_reason == "eligible_visual_query_field"
    assert diagnostics.visual_candidates[0].semantic_safety_eligible is True
    assert diagnostics.visual_click_snapshot_id == "field-ground"
    assert diagnostics.visual_click_succeeded is True
    verification = diagnostics.verification
    assert verification is not None
    assert verification.foreground_stable is True
    assert verification.same_hwnd is True
    assert verification.same_pid is True
    assert verification.same_trusted_app_id is True
    assert verification.fresh_observation_id == "after-click"
    assert verification.fresh_observation_distinct_from_click is True
    assert verification.focused_control_present is False
    assert verification.focused_control_count == 0
    assert verification.focused_control_id is None
    assert verification.clicked_visual_target_id == "v-search"
    assert verification.clicked_visual_target_role == "search field"
    assert verification.visual_focus_verification_attempted
    assert verification.visual_focus_verification_result == "provider_error"
    assert verification.query_field_verification_method == "none"
    assert verification.query_field_verification_failure_reason == "provider_error"
    assert verification.strong_visual_verification_available
    assert verification.strong_visual_verification_attempted
    assert verification.strong_visual_verification_result == "provider_error"
    assert not verification.strong_visual_verification_used


def test_strong_local_query_field_verification_is_reported_separately() -> None:
    initial = observation("initial", elements=(query_field("initial", focused=True),))
    typed = observation("typed", elements=(
        query_field("typed", "Iago", focused=True), conversation("c2", "Iago"),
    ))
    computer = FakeComputer([initial, typed, active_conversation("after", "Iago")])

    result = GenericTaskDebugAgent(computer, FakeDecisionMaker("c2")).run(
        "Open WhatsApp and open the chat with Iago",
    )

    assert result.success
    diagnostics = result.query_field_diagnostics
    assert diagnostics is not None and diagnostics.verification is not None
    assert diagnostics.selected_source == "UIA"
    assert diagnostics.selection_reason == "already_focused_uia_query_field"
    assert diagnostics.verification.query_field_verification_method == "strong_local"
    assert diagnostics.verification.query_field_verification_failure_reason is None
    assert diagnostics.verification.focused_control_present is True
    assert diagnostics.verification.focused_control_id == "c1"
    assert diagnostics.verification.focused_control_editable is True
    assert diagnostics.verification.focused_control_password is False
    assert diagnostics.verification.focused_control_enabled is True
    assert diagnostics.verification.focused_control_visible is True
    assert diagnostics.verification.uia_value_pattern_available is None
    assert diagnostics.verification.uia_text_pattern_available is None
    assert not diagnostics.verification.visual_focus_verification_attempted
    assert not diagnostics.verification.strong_visual_verification_available
    assert result.literal_types == 1


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
    assert low_result.target_decision is not None
    assert low_result.target_decision.resolver_status == "choice"
    assert low_result.target_decision.jev_called
    assert low_result.target_decision.jev_result_kind == "ready"
    assert low_result.target_decision.jev_selected_candidate_id == "c2"
    assert low_result.target_decision.jev_confidence == .79
    assert low_result.target_decision.stop_reason == "low_confidence"
    assert not low_computer.executed

    high_computer = FakeComputer([
        observation("choice", elements=(
            conversation("c1"), conversation("c2", parent="Work contact"),
        )),
        active_conversation("after", "Pablo García", parent="Work contact"),
    ])
    high_result = GenericTaskDebugAgent(
        high_computer, FakeDecisionMaker("c2", confidence=.80),
    ).run(request)
    assert high_result.success and high_result.chosen_candidate_id == "c2"
    assert high_result.target_decision is not None
    assert high_result.target_decision.resolver_status == "choice"
    assert high_result.target_decision.jev_release_result == "action_succeeded"
    assert high_result.target_decision.stop_reason is None
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
        observation("file", elements=(file_row,)),
        active_conversation("after", "factura septiembre.pdf", parent="Files"),
    ])
    decision = FakeDecisionMaker("c1")
    spec = requested_target_spec("Open factura septiembre.pdf", experimental_generic=True)
    assert spec is not None and spec.desired_role == "file"
    result = GenericTaskDebugAgent(computer, decision).run("Open factura septiembre.pdf")
    assert result.success and result.target_resolution_status == "unique"
    assert result.target_decision is not None
    assert result.target_decision.resolver_status == "unique"
    assert result.target_decision.selection_mode == "deterministic_local"
    assert result.target_decision.evidence_sufficiency == "sufficient"
    assert not result.target_decision.jev_required and not result.target_decision.jev_called
    assert result.target_decision.jev_provider_called is False
    assert result.target_decision.jev_selected_candidate_id is None
    assert result.target_decision.jev_confidence is None
    assert result.target_decision.chosen_candidate_id == "c1"
    assert result.target_decision.jev_release_result == "action_succeeded"
    assert not decision.calls
    assert result.final_target_activations == 1
    assert result.post_action_observation_obtained
    assert isinstance(computer.executed[0][0], ClickAction)


def test_fresh_directed_visual_evidence_skips_jev_but_uses_normal_activation_path() -> None:
    grounded = replace(observation("grounded", visual=(VisualElement(
        "v1", "Iago", "list item", Rect(10, 20, 180, 70), None, True,
    ),)), visual_directed_grounding=True, visual_requested_max_elements=5)
    computer = FakeComputer(
        [observation("before"), active_conversation("after", "Iago")], [grounded],
        visual_provider=object(),
    )
    decision = FakeDecisionMaker("v1")

    result = GenericTaskDebugAgent(computer, decision).run("Select Iago")

    assert result.success and result.target_resolution_status == "unique"
    assert not decision.calls
    diagnostics = result.target_decision
    assert diagnostics is not None
    assert diagnostics.selection_mode == "deterministic_local"
    assert diagnostics.evidence_sufficiency == "sufficient"
    assert diagnostics.admissible_candidate_count_total == 1
    assert diagnostics.frontier_candidate_count == 1
    assert not diagnostics.jev_required and not diagnostics.jev_called
    assert diagnostics.jev_provider_called is False
    assert diagnostics.jev_confidence is None
    assert diagnostics.chosen_candidate_id == "v1"
    assert result.chosen_candidate_id == "v1"
    assert result.final_target_activations == 1
    assert result.visual_target_activations == 1
    assert result.post_action_observation_obtained
    assert computer.visual_activations[0][0] == VisualClickAction("grounded-preclick-visual", "v1")


def test_deterministic_selection_keeps_budget_and_executor_failure_gates() -> None:
    grounded = replace(observation("grounded", visual=(VisualElement(
        "v1", "Iago", "list item", Rect(10, 20, 180, 70), None, True,
    ),)), visual_directed_grounding=True, visual_requested_max_elements=5)
    computer = FakeComputer(
        [observation("before")], [grounded], visual_provider=object(),
    )
    computer.visual_click_result = ActionResult(
        False, VisualClickAction("grounded", "v1"), "synthetic failure",
        error="windows_operation_failed", input_issued=False,
    )
    decision = FakeDecisionMaker("v1")

    result = GenericTaskDebugAgent(computer, decision).run("Select Iago")

    assert not result.success and result.stop_reason == "target_activation_failed"
    assert not decision.calls
    assert result.final_target_activations == 1
    assert result.visual_target_activations == 1
    assert result.chosen_candidate_id is None
    assert result.target_decision is not None
    assert result.target_decision.chosen_candidate_id == "v1"
    assert result.target_decision.jev_confidence is None
    assert result.target_decision.stop_reason == "target_activation_failed"
    assert result.query_field_activations == 0 and result.query_submits == 0


def test_generic_visual_activation_includes_executor_and_post_observation_diagnostics() -> None:
    grounded = replace(observation("grounded", visual=(VisualElement(
        "v1", "Iago", "list item", Rect(10, 20, 180, 70), None, True,
    ),)), visual_directed_grounding=True, visual_requested_max_elements=5)
    activation = VisualActivationDiagnostic(
        "v1", snapshot_binding_valid=True, candidate_lookup_succeeded=True,
        provenance_valid=True, provider_name="openai", preflight_started=True,
        preflight_succeeded=True, geometry_resolution_succeeded=True,
        resolved_point_inside_candidate=True, resolved_point_inside_bound_window=True,
        foreground_stable_before_input=True, input_attempted=True,
        input_result="succeeded", snapshot_consumed=True,
    )
    computer = FakeComputer(
        [observation("before"), active_conversation("after", "Iago")], [grounded],
        visual_provider=object(),
    )
    computer.visual_click_result = ActionResult(
        True, VisualClickAction("grounded", "v1"), "synthetic visual click",
        input_issued=True, visual_activation_diagnostic=activation,
    )

    result = GenericTaskDebugAgent(computer, FakeDecisionMaker("v1")).run("Select Iago")

    assert result.success and result.stop_reason == "target_activated"
    diagnostic = result.visual_activation
    assert diagnostic is not None
    assert diagnostic.provider_name == "openai" and diagnostic.provider_execution_agnostic
    assert diagnostic.post_action_observation_attempted
    assert diagnostic.post_action_observation_obtained
    assert diagnostic.failure_stage is None and diagnostic.failure_reason is None


def test_generic_activation_fails_explicitly_if_foreground_changes_after_click() -> None:
    grounded = replace(observation("grounded", visual=(VisualElement(
        "v1", "Iago", "list item", Rect(10, 20, 180, 70), None, True,
    ),)), visual_directed_grounding=True, visual_requested_max_elements=5)
    activation = VisualActivationDiagnostic(
        "v1", snapshot_binding_valid=True, candidate_lookup_succeeded=True,
        provenance_valid=True, provider_name="gemini", preflight_started=True,
        preflight_succeeded=True, geometry_resolution_succeeded=True,
        resolved_point_inside_candidate=True, resolved_point_inside_bound_window=True,
        foreground_stable_before_input=True, input_attempted=True,
        input_result="succeeded", snapshot_consumed=True,
    )
    after = replace(observation("after"), foreground_hwnd=101)
    computer = FakeComputer(
        [observation("before"), after], [grounded], visual_provider=object(),
    )
    computer.visual_click_result = ActionResult(
        True, VisualClickAction("grounded", "v1"), "synthetic visual click",
        input_issued=True, visual_activation_diagnostic=activation,
    )

    result = GenericTaskDebugAgent(computer, FakeDecisionMaker("v1")).run("Select Iago")

    assert not result.success and result.stop_reason == "target_activation_postcondition_failed"
    assert result.visual_activation is not None
    assert result.visual_activation.post_action_observation_obtained
    assert result.visual_activation.failure_stage == "target_activation_postcondition"
    assert result.visual_activation.failure_reason == "trusted_context_changed"
    assert result.target_activation_postcondition is not None
    assert not result.target_activation_postcondition.trusted_context_stable
    assert result.query_field_activations == 0 and result.query_submits == 0


def _activation_computer(after: Observation, *, visual_provider: object | None = None):
    before = observation("before", elements=(conversation(name="Alpha"),))
    return FakeComputer([before, after], visual_provider=visual_provider)


def _activation_fixture(after: Observation, *, visual_provider: object | None = None):
    computer = _activation_computer(after, visual_provider=visual_provider)
    result = GenericTaskDebugAgent(computer, FakeDecisionMaker("c1")).run("Select Alpha")
    return result, computer


def _visual_postcondition_observation(
    snapshot: str, elements: tuple[VisualElement, ...],
) -> Observation:
    base = observation(snapshot)
    metadata = ScreenshotMetadata(
        snapshot, 100, Rect(0, 0, 800, 600), Rect(0, 0, 800, 600),
        800, 600, 96, 96, 1.0, 1.0,
    )
    return replace(
        base, screenshot=metadata, visual_elements=elements,
        visual_grounding_status=(
            VisualGroundingStatus.SUCCESS_WITH_CANDIDATES if elements
            else VisualGroundingStatus.SUCCESS_EMPTY
        ),
        visual_directed_grounding=True,
    )


def _visual_postcondition_candidate(
    candidate_id: str,
    label: str,
    role: str,
    rectangle: Rect,
    *,
    region: VisualRegion,
    activity: str = "unknown",
    selection_state: VisualSelectionState = VisualSelectionState.UNKNOWN,
) -> VisualElement:
    return VisualElement(
        candidate_id, label, role, rectangle, None, False,
        activity=activity, selection_state=selection_state, region=region,
    )


def _evaluate_iago_visual_postcondition(
    candidates: tuple[VisualElement, ...], *, provider_incomplete: bool = False,
):
    target = TargetSpec("Iago", (), "conversation", "activate")
    visual = _visual_postcondition_observation("visual-postcondition", candidates)
    if provider_incomplete:
        visual = replace(
            visual, visual_grounding_status=VisualGroundingStatus.PROVIDER_ERROR,
            visual_provider_error=ProviderErrorDiagnostic("timeout"),
        )
    return evaluate_target_activation_postcondition(
        target, visual, trusted_context_stable=True,
        visual_verification_called=True,
        visual_grounding_objective=(
            'Determine whether "Iago" is active/open/selected, not merely visible.'
        ),
        provider_incomplete=provider_incomplete,
    )


def test_visual_postcondition_sidebar_identity_alone_is_insufficient() -> None:
    result = _evaluate_iago_visual_postcondition((
        _visual_postcondition_candidate(
            "v1", "Iago", "list item", Rect(20, 120, 200, 165),
            region=VisualRegion.NAVIGATION, activity="not_active",
        ),
    ))

    assert result.result is ActivationPostconditionResult.INSUFFICIENT
    assert result.target_identity_evidence is ActivationEvidence.MATCH
    assert result.active_state_evidence is ActivationEvidence.UNKNOWN
    assert result.visual_candidate_count == 1
    candidate = result.visual_candidate_diagnostics[0]
    assert candidate.identity_text == "Iago"
    assert candidate.presentation_role == "list_item"
    assert candidate.region == "navigation"
    assert candidate.geometry_region_bucket == "navigation"
    assert candidate.activity_evidence == "not_active"
    assert candidate.active_state_relation is ActivationEvidence.UNKNOWN
    assert candidate.authority_class == "non_authoritative"
    assert candidate.rejection_reason == "navigation_identity_is_not_active_context_evidence"


def test_visual_postcondition_identity_in_active_header_verifies() -> None:
    result = _evaluate_iago_visual_postcondition((
        _visual_postcondition_candidate(
            "v1", "Iago", "text", Rect(420, 55, 700, 120),
            region=VisualRegion.HEADER,
        ),
    ))

    assert result.result is ActivationPostconditionResult.VERIFIED
    assert result.target_identity_evidence is ActivationEvidence.MATCH
    assert result.active_state_evidence is ActivationEvidence.MATCH
    assert result.visual_candidate_diagnostics[0].authority_class == "authoritative"


def test_visual_postcondition_other_identity_in_active_detail_contradicts() -> None:
    result = _evaluate_iago_visual_postcondition((
        _visual_postcondition_candidate(
            "v1", "Iago", "list item", Rect(20, 120, 200, 165),
            region=VisualRegion.NAVIGATION, activity="not_active",
        ),
        _visual_postcondition_candidate(
            "v2", "Marta", "text", Rect(420, 230, 700, 280),
            region=VisualRegion.DETAIL, activity="active",
        ),
    ))

    assert result.result is ActivationPostconditionResult.CONTRADICTED
    assert result.active_state_evidence is ActivationEvidence.MISMATCH
    detail = next(item for item in result.visual_candidate_diagnostics
                  if item.candidate_id == "v2")
    assert detail.region == "detail"
    assert detail.identity_relation is ActivationEvidence.MISMATCH
    assert detail.authority_class == "authoritative"


def test_visual_postcondition_selected_row_and_matching_detail_verify() -> None:
    result = _evaluate_iago_visual_postcondition((
        _visual_postcondition_candidate(
            "v1", "Iago", "list item", Rect(20, 120, 200, 165),
            region=VisualRegion.NAVIGATION, activity="not_active",
            selection_state=VisualSelectionState.SELECTED,
        ),
        _visual_postcondition_candidate(
            "v2", "Iago", "text", Rect(420, 230, 700, 280),
            region=VisualRegion.DETAIL, activity="active",
        ),
    ))

    assert result.result is ActivationPostconditionResult.VERIFIED
    row = next(item for item in result.visual_candidate_diagnostics
               if item.candidate_id == "v1")
    assert row.selected_state == "selected"
    assert row.authority_class == "authoritative"
    assert row.active_state_relation is ActivationEvidence.MATCH


def test_visual_postcondition_selected_row_conflicting_detail_contradicts() -> None:
    result = _evaluate_iago_visual_postcondition((
        _visual_postcondition_candidate(
            "v1", "Iago", "list item", Rect(20, 120, 200, 165),
            region=VisualRegion.NAVIGATION,
            selection_state=VisualSelectionState.SELECTED,
        ),
        _visual_postcondition_candidate(
            "v2", "Marta", "text", Rect(420, 230, 700, 280),
            region=VisualRegion.DETAIL, activity="active",
        ),
    ))

    assert result.result is ActivationPostconditionResult.CONTRADICTED
    row = next(item for item in result.visual_candidate_diagnostics
               if item.candidate_id == "v1")
    assert row.authority_class == "authoritative"
    assert row.active_state_relation is ActivationEvidence.MISMATCH
    assert row.rejection_reason == "selected_row_conflicts_with_active_detail_identity"


def test_visual_postcondition_identity_without_active_or_region_evidence_is_insufficient() -> None:
    result = _evaluate_iago_visual_postcondition((
        _visual_postcondition_candidate(
            "v1", "Iago", "text", Rect(420, 380, 700, 440),
            region=VisualRegion.CONTENT,
        ),
    ))

    assert result.result is ActivationPostconditionResult.INSUFFICIENT
    assert result.target_identity_evidence is ActivationEvidence.MATCH
    assert result.active_state_evidence is ActivationEvidence.UNKNOWN
    candidate = result.visual_candidate_diagnostics[0]
    assert candidate.region == "content"
    assert candidate.authority_class == "non_authoritative"
    assert candidate.rejection_reason == "content_identity_requires_selected_row_or_detail_context"


def test_visual_postcondition_provider_failure_is_incomplete() -> None:
    result = _evaluate_iago_visual_postcondition((), provider_incomplete=True)

    assert result.result is ActivationPostconditionResult.INCOMPLETE
    assert result.visual_candidate_count == 0
    assert result.visual_candidate_diagnostics == ()
    assert result.failure_reason == "visual_verification_incomplete"


def test_semantic_postcondition_verifies_selected_requested_identity() -> None:
    result, computer = _activation_fixture(active_conversation("after", "Alpha"))

    assert result.success and result.stop_reason == "target_activated"
    assert result.target_activation_postcondition is not None
    assert result.target_activation_postcondition.result is ActivationPostconditionResult.VERIFIED
    assert result.target_activation_postcondition.target_identity_evidence.value == "match"
    assert result.target_activation_postcondition.active_state_evidence.value == "match"
    assert not computer.postcondition_groundings


def test_visible_sidebar_neighbor_does_not_override_authoritative_active_target() -> None:
    after = observation("after", elements=(
        replace(conversation("c1", "Alpha"), selected=True),
        conversation("c2", "Beta"),
    ))

    result, computer = _activation_fixture(after)

    assert result.success and result.stop_reason == "target_activated"
    postcondition = result.target_activation_postcondition
    assert postcondition is not None
    assert postcondition.result is ActivationPostconditionResult.VERIFIED
    assert postcondition.target_identity_evidence is ActivationEvidence.MATCH
    assert postcondition.active_state_evidence is ActivationEvidence.MATCH
    diagnostic = postcondition.structural_postcondition_evidence
    assert diagnostic.ignored_non_authoritative_count >= 1
    beta = next(item for item in diagnostic.evidence_items if item.identity_text == "Beta")
    assert beta.presentation_role == "navigation_sidebar"
    assert beta.identity_relation is ActivationEvidence.MISMATCH
    assert beta.active_state_relation is ActivationEvidence.UNKNOWN
    assert beta.authority_class == "non_authoritative_visible"
    assert not computer.postcondition_groundings


def test_sidebar_mismatch_without_active_identity_is_insufficient_and_allows_visual() -> None:
    after = observation("after", elements=(conversation("c1", "Beta"),))
    computer = _activation_computer(after, visual_provider=object())
    computer.postcondition_visual_result = replace(
        after, visual_grounding_status=VisualGroundingStatus.SUCCESS_EMPTY,
        visual_directed_grounding=True,
    )

    result = GenericTaskDebugAgent(computer, FakeDecisionMaker("c1")).run("Select Alpha")

    assert not result.success and result.stop_reason == "target_activation_unverified"
    postcondition = result.target_activation_postcondition
    assert postcondition is not None
    assert postcondition.result is ActivationPostconditionResult.INSUFFICIENT
    assert postcondition.target_identity_evidence is ActivationEvidence.UNKNOWN
    assert postcondition.active_state_evidence is ActivationEvidence.UNKNOWN
    diagnostic = postcondition.structural_postcondition_evidence
    assert diagnostic.contradictory_identity_count >= 1
    assert diagnostic.authoritative_contradiction_count == 0
    assert diagnostic.ignored_non_authoritative_count >= 1
    assert len(computer.postcondition_groundings) == 1
    assert computer.postcondition_groundings[0].verification_only is True


def test_current_entity_window_title_can_contradict_requested_target() -> None:
    after = replace(observation("after"), window_title="Chat with Beta")

    diagnostic = evaluate_target_activation_postcondition(
        requested_target_spec("Select Alpha"), after,
        trusted_context_stable=True,
    )

    assert diagnostic.result is ActivationPostconditionResult.CONTRADICTED
    assert diagnostic.target_identity_evidence is ActivationEvidence.MISMATCH
    title = diagnostic.structural_postcondition_evidence.evidence_items[0]
    assert title.evidence_kind == "window_title"
    assert title.presentation_role == "current_surface_title"
    assert title.authority_class == "authoritative_active_entity"
    assert title.active_state_relation is ActivationEvidence.MISMATCH


def test_conflicting_authoritative_uia_facts_are_insufficient_and_allow_visual() -> None:
    after = observation("after", elements=(
        replace(conversation("c1", "Alpha"), selected=True),
        UIElement(
            "c2", "Beta", "Text", visible=True,
            parent_name="Conversation details", parent_control_type="Pane",
        ),
    ))
    computer = _activation_computer(after, visual_provider=object())
    computer.postcondition_visual_result = replace(
        after, visual_grounding_status=VisualGroundingStatus.SUCCESS_EMPTY,
        visual_directed_grounding=True,
    )

    result = GenericTaskDebugAgent(computer, FakeDecisionMaker("c1")).run("Select Alpha")

    postcondition = result.target_activation_postcondition
    assert postcondition is not None
    assert postcondition.result is ActivationPostconditionResult.INSUFFICIENT
    assert postcondition.active_state_evidence is ActivationEvidence.UNKNOWN
    assert postcondition.structural_postcondition_evidence.conflicting_authoritative_evidence
    assert len(computer.postcondition_groundings) == 1
    assert len(computer.executed) == 1


def test_unrelated_focused_control_is_not_authoritative_and_diagnostic_is_redacted() -> None:
    after = observation("after", elements=(UIElement(
        "c9", "Beta", "Edit", visible=True, focused=True,
        parent_name="Unrelated form", parent_control_type="Pane",
    ),))

    diagnostic = evaluate_target_activation_postcondition(
        requested_target_spec("Select Alpha"), after,
        trusted_context_stable=True,
        clean_text=lambda value: "[REDACTED]" if value == "Beta" else value,
    )

    assert diagnostic.result is ActivationPostconditionResult.INSUFFICIENT
    item = next(
        item for item in diagnostic.structural_postcondition_evidence.evidence_items
        if item.control_id == "c9"
    )
    assert item.control_type == "Edit" and item.focused is True
    assert item.presentation_role == "unrelated_control"
    assert item.authority_class == "non_authoritative_visible"
    assert item.identity_relation is ActivationEvidence.MISMATCH
    assert item.active_state_relation is ActivationEvidence.UNKNOWN
    assert item.identity_text == "[REDACTED]"
    assert "hwnd" not in repr(asdict(diagnostic)).casefold()


def test_wrong_neighbor_selected_after_successful_click_fails_without_retry() -> None:
    result, computer = _activation_fixture(active_conversation("after", "Beta"))

    assert not result.success
    assert result.stop_reason == "target_activation_postcondition_failed"
    assert result.target_activation_postcondition is not None
    assert result.target_activation_postcondition.result is ActivationPostconditionResult.CONTRADICTED
    assert result.target_activation_postcondition.target_identity_evidence.value == "mismatch"
    evidence = result.target_activation_postcondition.structural_postcondition_evidence
    assert evidence.authoritative_contradiction_count >= 1
    beta = next(item for item in evidence.evidence_items if item.identity_text == "Beta")
    assert beta.selected is True
    assert beta.authority_class == "authoritative_active_entity"
    assert len(computer.executed) == 1
    assert not computer.query_submits and not computer.postcondition_groundings


def test_visible_requested_identity_with_different_selected_identity_fails() -> None:
    after = observation("after", elements=(
        conversation("c1", "Alpha"),
        replace(conversation("c2", "Beta"), selected=True),
    ))
    result, computer = _activation_fixture(after)

    assert not result.success and result.stop_reason == "target_activation_postcondition_failed"
    assert result.target_activation_postcondition is not None
    assert result.target_activation_postcondition.target_identity_evidence.value == "mismatch"
    assert len(computer.executed) == 1


def test_visible_but_unselected_identity_does_not_verify_activation() -> None:
    result, computer = _activation_fixture(observation(
        "after", elements=(conversation("c1", "Alpha"),),
    ))

    assert not result.success and result.stop_reason == "target_activation_unverified"
    assert result.target_activation_postcondition is not None
    assert result.target_activation_postcondition.result is ActivationPostconditionResult.INSUFFICIENT
    assert result.target_activation_postcondition.target_identity_evidence.value == "match"
    assert result.target_activation_postcondition.active_state_evidence.value == "unknown"
    assert not computer.postcondition_groundings
    assert len(computer.executed) == 1


def test_visual_postcondition_can_verify_active_identity_once() -> None:
    after = observation("after")
    visual = replace(_visual_postcondition_observation("after", (VisualElement(
        "v1", "Alpha", "text", Rect(420, 60, 700, 120), None, False,
        activity="active", region=VisualRegion.HEADER,
    ),)), visual_provider_attempts=(VisualProviderAttempt(
            "gemini", "gemini-3.5-flash-lite", 20, "success_with_candidates",
        ),))
    computer = _activation_computer(after, visual_provider=object())
    computer.postcondition_visual_result = visual

    result = GenericTaskDebugAgent(computer, FakeDecisionMaker("c1")).run("Select Alpha")

    assert result.success and result.stop_reason == "target_activated"
    assert len(computer.postcondition_groundings) == 1
    assert computer.postcondition_groundings[0].verification_only is True
    assert result.visual_grounding_calls == 1
    assert result.target_activation_postcondition is not None
    assert result.target_activation_postcondition.result is ActivationPostconditionResult.VERIFIED
    assert result.target_activation_postcondition.source.value == "visual"
    assert "active, open, or selected entity, not merely visible" in (
        result.target_activation_postcondition.visual_grounding_objective or ""
    )
    assert len(computer.executed) == 1


def test_misleading_sidebar_identity_does_not_override_visual_active_target() -> None:
    after = observation("after", elements=(conversation("c1", "Beta"),))
    visual = replace(_visual_postcondition_observation("after", (VisualElement(
        "v1", "Alpha", "text", Rect(420, 60, 700, 120), None, False,
        activity="active", region=VisualRegion.HEADER,
    ),)), visual_provider_attempts=(VisualProviderAttempt(
            "gemini", "gemini-3.5-flash-lite", 20, "success_with_candidates",
        ),))
    computer = _activation_computer(after, visual_provider=object())
    computer.postcondition_visual_result = visual

    result = GenericTaskDebugAgent(computer, FakeDecisionMaker("c1")).run("Select Alpha")

    assert result.success and result.stop_reason == "target_activated"
    assert result.target_activation_postcondition is not None
    assert result.target_activation_postcondition.result is ActivationPostconditionResult.VERIFIED
    assert result.target_activation_postcondition.source.value == "visual"
    assert len(computer.postcondition_groundings) == 1
    assert len(computer.executed) == 1


def test_visual_postcondition_identifying_neighbor_fails_closed() -> None:
    after = observation("after")
    visual = _visual_postcondition_observation("after", (VisualElement(
        "v1", "Beta", "text", Rect(420, 60, 700, 120), None, False,
        activity="active", region=VisualRegion.HEADER,
    ),))
    computer = _activation_computer(after, visual_provider=object())
    computer.postcondition_visual_result = visual

    result = GenericTaskDebugAgent(computer, FakeDecisionMaker("c1")).run("Select Alpha")

    assert not result.success and result.stop_reason == "target_activation_postcondition_failed"
    assert result.target_activation_postcondition is not None
    assert result.target_activation_postcondition.result is ActivationPostconditionResult.CONTRADICTED
    assert len(computer.postcondition_groundings) == 1
    assert len(computer.executed) == 1


def test_trusted_context_is_rechecked_after_remote_visual_verification() -> None:
    after = observation("after")
    visual = _visual_postcondition_observation("after", (VisualElement(
        "v1", "Alpha", "text", Rect(420, 60, 700, 120), None, False,
        activity="active", region=VisualRegion.HEADER,
    ),))
    computer = _activation_computer(after, visual_provider=object())
    computer.postcondition_visual_result = visual
    computer.on_postcondition_visual = lambda: setattr(computer, "context_matches", False)

    result = GenericTaskDebugAgent(computer, FakeDecisionMaker("c1")).run("Select Alpha")

    assert not result.success and result.stop_reason == "target_activation_postcondition_failed"
    assert result.target_activation_postcondition is not None
    assert not result.target_activation_postcondition.trusted_context_stable
    assert len(computer.postcondition_groundings) == 1
    assert len(computer.executed) == 1


def test_visual_failover_attempts_are_bounded_and_reported() -> None:
    after = observation("after")
    visual = replace(_visual_postcondition_observation("after", (VisualElement(
        "v1", "Alpha", "text", Rect(420, 60, 700, 120), None, False,
        activity="active", region=VisualRegion.HEADER,
    ),)), visual_provider_attempts=(
            VisualProviderAttempt("gemini", "gemini-3.5-flash-lite", 30,
                                  "provider_failure", "timeout"),
            VisualProviderAttempt("openai", "gpt-5.6-luna", 40,
                                  "success_with_candidates"),
        ),
        provider_failover_used=True,
        provider_failover_reason="timeout",
    )
    computer = _activation_computer(after, visual_provider=object())
    computer.postcondition_visual_result = visual

    result = GenericTaskDebugAgent(computer, FakeDecisionMaker("c1")).run("Select Alpha")

    assert result.success
    assert result.target_activation_postcondition is not None
    assert [attempt.provider for attempt in result.target_activation_postcondition.visual_provider_attempts] == [
        "gemini", "openai",
    ]
    assert len(computer.postcondition_groundings) == 1
    assert len(computer.executed) == 1


def test_visual_success_empty_is_insufficient_and_provider_failures_are_incomplete() -> None:
    after = observation("after")
    computer = _activation_computer(after, visual_provider=object())
    computer.postcondition_visual_result = replace(
        after, visual_grounding_status=VisualGroundingStatus.SUCCESS_EMPTY,
        visual_directed_grounding=True,
    )
    result = GenericTaskDebugAgent(computer, FakeDecisionMaker("c1")).run("Select Alpha")
    assert not result.success and result.stop_reason == "target_activation_unverified"
    assert result.target_activation_postcondition is not None
    assert result.target_activation_postcondition.result is ActivationPostconditionResult.INSUFFICIENT
    assert len(computer.postcondition_groundings) == 1

    failed_computer = _activation_computer(after, visual_provider=object())
    failed_computer.postcondition_visual_result = replace(
        after,
        visual_grounding_status=VisualGroundingStatus.PROVIDER_ERROR,
        visual_provider_error=ProviderErrorDiagnostic("timeout"),
        visual_provider_attempts=(
            VisualProviderAttempt("gemini", "gemini-3.5-flash-lite", 20,
                                  "provider_failure", "timeout"),
            VisualProviderAttempt("openai", "gpt-5.6-luna", 20,
                                  "provider_failure", "timeout"),
        ),
    )
    failed = GenericTaskDebugAgent(failed_computer, FakeDecisionMaker("c1")).run("Select Alpha")
    assert not failed.success and failed.stop_reason == "target_activation_unverified"
    assert failed.target_activation_postcondition is not None
    assert failed.target_activation_postcondition.result is ActivationPostconditionResult.INCOMPLETE
    assert len(failed.target_activation_postcondition.visual_provider_attempts) == 2
    assert len(failed_computer.executed) == 1


@pytest.mark.parametrize("changed", ["app", "process", "window"])
def test_changed_trusted_context_prevents_semantic_success(changed: str) -> None:
    kwargs = {
        "app_id": "other_app" if changed == "app" else APP_ID,
        "process_id": 99 if changed == "process" else 42,
        "foreground_hwnd": 101 if changed == "window" else 100,
    }
    result, computer = _activation_fixture(active_conversation("after", "Alpha", **kwargs))

    assert not result.success and result.stop_reason == "target_activation_postcondition_failed"
    assert result.target_activation_postcondition is not None
    assert result.target_activation_postcondition.trusted_context_stable is False
    assert result.target_activation_postcondition.result is ActivationPostconditionResult.CONTRADICTED
    assert len(computer.executed) == 1
    assert not computer.postcondition_groundings


@pytest.mark.parametrize(("decision", "expected_reason", "result_kind", "selected_id"), [
    (TargetChoiceResult(
        "stop", None, .94, "Jev selected stop.", diagnostic_reason="model_stop",
        provider_called=True,
    ), "model_stop", "stop", None),
    (TargetChoiceResult(
        "stop", None, .72, "Jev confidence was below threshold.",
        diagnostic_reason="model_confidence_below_threshold", provider_called=True,
        proposed_candidate_id="c1",
    ), "model_confidence_below_threshold", "stop", "c1"),
    (TargetChoiceResult(
        "error", None, None, "Invalid model response.", "invalid_response",
        "invalid_response", True,
    ), "invalid_response", "error", None),
    (TargetChoiceResult(
        "error", None, None, "Provider failed.", "timeout", "provider_error", True,
    ), "provider_error", "error", None),
])
def test_target_decision_failure_diagnostics_do_not_release_or_expose_messages(
    decision: TargetChoiceResult, expected_reason: str, result_kind: str,
    selected_id: str | None,
) -> None:
    class FixedDecisionMaker(FakeDecisionMaker):
        def decide_target_activation(self, target, resolution) -> TargetChoiceResult:
            self.calls.append((target, resolution))
            return decision

    uncertain_target = UIElement(
        "c1", "Pablo García active", "ListItem",
        rectangle=Rect(10, 10, 300, 70), enabled=True, visible=True,
        parent_name="Results",
    )
    computer = FakeComputer([observation("unique", elements=(uncertain_target,))])
    result = GenericTaskDebugAgent(computer, FixedDecisionMaker()).run(
        "Open WhatsApp and open the chat with Pablo García",
    )

    assert not result.success and result.stop_reason == "target_decision_stopped"
    diagnostics = result.target_decision
    assert diagnostics is not None
    assert diagnostics.resolver_status == "unique"
    assert diagnostics.jev_required and diagnostics.jev_called
    assert diagnostics.jev_provider_called is True
    assert diagnostics.jev_result_kind == result_kind
    assert diagnostics.jev_selected_candidate_id == selected_id
    assert diagnostics.jev_confidence == decision.confidence
    assert diagnostics.jev_release_result == "not_released"
    assert diagnostics.stop_reason == expected_reason
    assert not computer.executed and not computer.visual_activations
    assert "Provider failed." not in str(asdict(diagnostics))


def test_target_decision_diagnostics_bound_untrusted_ids_and_confidence() -> None:
    class FixedDecisionMaker(FakeDecisionMaker):
        def decide_target_activation(self, target, resolution) -> TargetChoiceResult:
            self.calls.append((target, resolution))
            return TargetChoiceResult(
                "ready", "API_KEY_secret_value", float("nan"), "safe fixed response",
                diagnostic_reason="candidate_selected", provider_called=True,
            )

    uncertain_target = UIElement(
        "c1", "Pablo García active", "ListItem",
        rectangle=Rect(10, 10, 300, 70), enabled=True, visible=True,
        parent_name="Results",
    )
    computer = FakeComputer([observation("unique", elements=(uncertain_target,))])
    result = GenericTaskDebugAgent(computer, FixedDecisionMaker()).run(
        "Open WhatsApp and open the chat with Pablo García",
    )
    diagnostics = result.target_decision

    assert diagnostics is not None
    serialized = str(asdict(diagnostics))
    assert "API_KEY_secret_value" not in serialized
    assert "nan" not in serialized.casefold()
    assert diagnostics.jev_selected_candidate_id is None
    assert diagnostics.jev_confidence is None
    assert not computer.executed


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


def _visual_target_run(
    original: VisualElement,
    fresh: Observation,
    *,
    context_matches: bool = True,
    request: str | None = None,
) -> tuple[object, FakeComputer]:
    grounded = observation("grounded", visual=(original,))
    computer = FakeComputer(
        [observation("before"), active_conversation("after", original.label)],
        [grounded], visual_provider=object(),
    )
    computer.context_matches = context_matches
    computer.preclick_directed_observations.append(fresh)
    result = GenericTaskDebugAgent(computer, FakeDecisionMaker(original.id)).run(
        request or f"Select {original.label}",
    )
    return result, computer


def test_visual_preclick_revalidation_stable_rebinds_to_fresh_snapshot() -> None:
    original = VisualElement("v1", "Iago", "conversation", Rect(40, 180, 300, 230), None, True)
    fresh = observation("fresh-stable", visual=(replace(original),))

    result, computer = _visual_target_run(original, fresh)

    diagnostic = result.visual_preclick_revalidation
    assert result.success and diagnostic is not None
    assert diagnostic.result.value == "STABLE"
    assert diagnostic.source == "visual"
    assert diagnostic.rebound_to_fresh_snapshot and diagnostic.click_released
    assert diagnostic.geometry_changed is False
    assert diagnostic.fresh_candidate_count == 1
    assert diagnostic.admissible_fresh_candidate_count == 1
    assert diagnostic.frontier_fresh_candidate_count == 1
    assert diagnostic.original_target is not None
    assert diagnostic.original_target.primary_identity == "Iago"
    assert diagnostic.original_candidate is not None
    assert diagnostic.original_candidate.provider_role == "conversation"
    assert diagnostic.fresh_candidates[0].primary_identity_relation == "match"
    assert diagnostic.fresh_candidates[0].considered_same_target
    assert diagnostic.fresh_candidates[0].same_target_rejection_reason is None
    assert diagnostic.original_candidate.semantic_role == "conversation"
    assert diagnostic.fresh_candidates[0].semantic_role == "conversation"
    assert len(computer.preclick_directed_calls) == 1
    action, clicked_observation = computer.visual_activations[0]
    assert action == VisualClickAction("fresh-stable", "v1")
    assert clicked_observation is fresh


@pytest.mark.parametrize(
    ("fresh_top", "expected_bucket"),
    [(220, "16_40_px"), (120, "41_100_px")],
)
def test_visual_preclick_revalidation_uses_fresh_geometry_after_vertical_shift(
    fresh_top: int, expected_bucket: str,
) -> None:
    original = VisualElement("v1", "Iago", "conversation", Rect(40, 180, 300, 230), None, True)
    moved = VisualElement("v1", "Iago", "conversation", Rect(40, fresh_top, 300, fresh_top + 50), None, True)
    # A neighbor occupies the original row after the shift. The executor only
    # receives the current candidate rectangle from the fresh observation.
    neighbor = VisualElement(
        "v2", "Neighbor", "conversation", Rect(40, 180, 300, 230), None, True,
    )
    fresh = observation("fresh-moved", visual=(moved, neighbor))

    result, computer = _visual_target_run(original, fresh)

    diagnostic = result.visual_preclick_revalidation
    assert result.success and diagnostic is not None
    assert diagnostic.result.value == "MOVED"
    assert diagnostic.geometry_changed is True
    assert diagnostic.displacement_bucket == expected_bucket
    assert diagnostic.rebound_to_fresh_snapshot and diagnostic.click_released
    serialized = str(asdict(diagnostic))
    assert all(key not in serialized for key in ("left", "top", "right", "bottom"))
    assert len(computer.visual_activations) == 1
    action, clicked_observation = computer.visual_activations[0]
    assert action == VisualClickAction("fresh-moved", "v1")
    assert clicked_observation.visual_elements[0].rectangle == moved.rectangle
    assert clicked_observation.visual_elements[0].rectangle != original.rectangle


def test_visual_preclick_revalidation_uses_unique_fresh_uia_target_without_visual_call() -> None:
    original = VisualElement("v1", "Iago", "conversation", Rect(40, 180, 300, 230), None, True)
    grounded = observation("grounded", visual=(original,))
    fresh_local = observation(
        "fresh-uia", elements=(conversation("c7", "Iago"),),
    )
    computer = FakeComputer(
        [observation("before"), active_conversation("after", "Iago")],
        [grounded], visual_provider=object(),
    )
    computer.preclick_local_observations.append(fresh_local)

    result = GenericTaskDebugAgent(computer, FakeDecisionMaker("v1")).run("Select Iago")

    diagnostic = result.visual_preclick_revalidation
    assert result.success and diagnostic is not None
    assert diagnostic.source == "UIA" and diagnostic.rebound_to_fresh_snapshot
    assert diagnostic.result.value == "MOVED"
    assert not computer.preclick_directed_calls
    assert computer.executed[0][0] == ClickAction("c7")
    assert computer.executed[0][1] is fresh_local


def test_visual_preclick_revalidation_disappearance_and_ambiguity_fail_closed() -> None:
    original = VisualElement("v1", "Iago", "conversation", Rect(40, 180, 300, 230), None, True)
    disappeared = observation("fresh-gone", visual=(VisualElement(
        "v2", "Other", "conversation", Rect(20, 80, 300, 130), None, True,
    ),))
    result_gone, computer_gone = _visual_target_run(original, disappeared)
    assert not result_gone.success
    assert result_gone.visual_preclick_revalidation is not None
    assert result_gone.visual_preclick_revalidation.result.value == "DISAPPEARED"
    assert result_gone.visual_preclick_revalidation.fresh_candidate_count == 1
    assert result_gone.visual_preclick_revalidation.fresh_candidates[0].primary_identity_relation == "mismatch"
    assert result_gone.visual_preclick_revalidation.fresh_candidates[0].same_target_rejection_reason == "primary_identity_mismatch"
    assert not computer_gone.visual_activations

    ambiguous = observation("fresh-ambiguous", visual=(
        original,
        VisualElement("v2", "Iago", "conversation", Rect(40, 260, 300, 310), None, True),
    ))
    result_ambiguous, computer_ambiguous = _visual_target_run(original, ambiguous)
    assert not result_ambiguous.success
    assert result_ambiguous.visual_preclick_revalidation is not None
    assert result_ambiguous.visual_preclick_revalidation.result.value == "AMBIGUOUS"
    assert result_ambiguous.visual_preclick_revalidation.fresh_candidate_count == 2
    assert result_ambiguous.visual_preclick_revalidation.frontier_fresh_candidate_count == 2
    assert all(row.considered_same_target for row in result_ambiguous.visual_preclick_revalidation.fresh_candidates)
    assert not computer_ambiguous.visual_activations


def test_visual_preclick_revalidation_reports_secondary_preview_and_same_identity() -> None:
    original = VisualElement(
        "v1", "Iago", "conversation", Rect(40, 180, 300, 230), None, True,
    )
    fresh_candidate = replace(original, parent="Iago sent: Nos vemos mañana")
    result, computer = _visual_target_run(
        original, observation("fresh-preview", visual=(fresh_candidate,)),
    )

    diagnostic = result.visual_preclick_revalidation
    assert result.success and diagnostic is not None
    assert diagnostic.result.value in {"STABLE", "MOVED"}
    assert diagnostic.fresh_candidates[0].primary_text == "Iago"
    assert diagnostic.fresh_candidates[0].secondary_text == ("Iago sent: Nos vemos mañana",)
    assert diagnostic.fresh_candidates[0].primary_identity_relation == "match"
    assert diagnostic.fresh_candidates[0].considered_same_target
    assert len(computer.visual_activations) == 1


def test_visual_preclick_role_wording_change_is_diagnosed_without_relaxing_match() -> None:
    original = VisualElement("v1", "Iago", "conversation", Rect(40, 180, 300, 230), None, True)
    fresh_candidate = replace(original, role="list item")
    result, computer = _visual_target_run(
        original, observation("fresh-role-wording", visual=(fresh_candidate,)),
    )

    diagnostic = result.visual_preclick_revalidation
    assert not result.success and diagnostic is not None
    assert diagnostic.result.value == "REJECTED"
    candidate = diagnostic.fresh_candidates[0]
    assert candidate.primary_identity_relation == "match"
    assert candidate.provider_role == "list item"
    assert candidate.presentation_role == "list_item"
    assert candidate.presentation_compatibility == "compatible"
    assert candidate.considered_same_target is False
    assert candidate.same_target_rejection_reason == "semantic_role_continuity_unproven"
    assert not computer.visual_activations


def test_visual_preclick_unknown_semantic_role_continuity_accepts_compatible_presentation() -> None:
    original = VisualElement("v1", "Iago", "list item", Rect(40, 180, 300, 230), None, True)
    fresh = observation("fresh-null-role", visual=(replace(original),))
    result, computer = _visual_target_run(
        original, fresh,
        request="Open WhatsApp and open the chat with Iago",
    )

    diagnostic = result.visual_preclick_revalidation
    assert result.success and diagnostic is not None
    assert diagnostic.result.value in {"STABLE", "MOVED"}
    assert diagnostic.original_target is not None
    assert diagnostic.original_target.desired_role == "conversation"
    assert diagnostic.original_candidate is not None
    assert diagnostic.original_candidate.semantic_role is None
    assert diagnostic.original_candidate.presentation_role == "list_item"
    assert diagnostic.original_candidate.presentation_compatibility == "compatible"
    candidate = diagnostic.fresh_candidates[0]
    assert candidate.semantic_role is None
    assert candidate.role_compatibility == "unknown"
    assert candidate.presentation_role == "list_item"
    assert candidate.presentation_compatibility == "compatible"
    assert candidate.considered_same_target
    assert candidate.same_target_rejection_reason is None
    assert len(computer.visual_activations) == 1


def test_visual_preclick_unknown_semantic_role_with_incompatible_presentation_rejects() -> None:
    original = VisualElement("v1", "Iago", "list item", Rect(40, 180, 300, 230), None, True)
    fresh_candidate = replace(original, role="text field")
    result, computer = _visual_target_run(
        original, observation("fresh-incompatible-role", visual=(fresh_candidate,)),
        request="Open WhatsApp and open the chat with Iago",
    )

    diagnostic = result.visual_preclick_revalidation
    assert not result.success and diagnostic is not None
    assert diagnostic.result.value == "REJECTED"
    candidate = diagnostic.fresh_candidates[0]
    assert candidate.primary_identity_relation == "match"
    assert candidate.semantic_role is None
    assert candidate.presentation_compatibility == "incompatible"
    assert candidate.same_target_rejection_reason == "presentation_role_incompatible"
    assert not computer.visual_activations


def test_visual_preclick_known_semantic_role_conflict_rejects() -> None:
    original = VisualElement("v1", "Iago", "conversation", Rect(40, 180, 300, 230), None, True)
    fresh_candidate = replace(original, role="contact")
    result, computer = _visual_target_run(
        original, observation("fresh-known-role-conflict", visual=(fresh_candidate,)),
    )

    diagnostic = result.visual_preclick_revalidation
    assert not result.success and diagnostic is not None
    assert diagnostic.result.value == "REJECTED"
    candidate = diagnostic.fresh_candidates[0]
    assert candidate.semantic_role == "contact"
    assert candidate.same_target_rejection_reason == "semantic_role_mismatch"
    assert not computer.visual_activations


def test_visual_preclick_qualifier_mismatch_rejects_same_identity() -> None:
    target = TargetSpec(
        "Iago", qualifiers=("Work",), desired_role="conversation", action_intent="select",
    )
    original = VisualElement(
        "v1", "Iago", "list item", Rect(40, 180, 300, 230), None, True, parent="Work",
    )
    fresh_candidate = replace(original, parent="Family")
    agent = GenericTaskDebugAgent(
        FakeComputer([], visual_provider=object()), FakeDecisionMaker(),
    )
    original_resolution, _ = agent._resolve(
        target, observation("qualified-original", visual=(original,)),
    )
    fresh_resolution, _ = agent._resolve(
        target, observation("qualified-fresh", visual=(fresh_candidate,)),
    )

    reason = agent._preclick_candidate_rejection_reason(
        target, original_resolution.candidates[0], fresh_resolution.candidates[0],
    )

    assert original_resolution.candidates[0].qualifier_evidence[0].value == "match"
    assert fresh_resolution.candidates[0].primary_identity.value == "match"
    assert fresh_resolution.candidates[0].qualifier_evidence[0].value == "mismatch"
    assert reason == "qualifier_mismatch"


def test_visual_preclick_same_identity_but_non_actionable_is_rejected_without_click() -> None:
    original = VisualElement("v1", "Iago", "conversation", Rect(40, 180, 300, 230), None, True)
    fresh_candidate = replace(original, clickable=False)
    result, computer = _visual_target_run(
        original, observation("fresh-disabled", visual=(fresh_candidate,)),
    )

    diagnostic = result.visual_preclick_revalidation
    assert not result.success and diagnostic is not None
    assert diagnostic.result.value == "REJECTED"
    candidate = diagnostic.fresh_candidates[0]
    assert candidate.primary_identity_relation == "match"
    assert candidate.considered_same_target
    assert candidate.actionable is False
    assert candidate.admissible is False
    assert not computer.visual_activations


def test_visual_preclick_zero_provider_candidates_is_disappeared() -> None:
    original = VisualElement("v1", "Iago", "conversation", Rect(40, 180, 300, 230), None, True)
    grounded = observation("grounded-empty-test", visual=(original,))
    screenshot = replace(grounded.screenshot, snapshot_id="fresh-zero")
    fresh_empty = replace(
        observation("fresh-zero"),
        screenshot=screenshot,
        visual_provider="fixture",
        visual_directed_grounding=True,
        visual_requested_max_elements=5,
        visual_grounding_status=VisualGroundingStatus.SUCCESS_EMPTY,
    )
    computer = FakeComputer(
        [observation("before"), active_conversation("after", "Iago")],
        [grounded], visual_provider=object(),
    )
    computer.preclick_directed_observations.append(fresh_empty)

    result = GenericTaskDebugAgent(computer, FakeDecisionMaker("v1")).run("Select Iago")

    diagnostic = result.visual_preclick_revalidation
    assert not result.success and diagnostic is not None
    assert diagnostic.result.value == "DISAPPEARED"
    assert diagnostic.fresh_candidate_count == 0
    assert diagnostic.admissible_fresh_candidate_count == 0
    assert diagnostic.frontier_fresh_candidate_count == 0
    assert diagnostic.fresh_candidates == ()
    assert not computer.visual_activations


def test_visual_preclick_revalidation_context_change_and_provider_failure_fail_closed() -> None:
    original = VisualElement("v1", "Iago", "conversation", Rect(40, 180, 300, 230), None, True)
    fresh = observation("fresh-context", visual=(replace(original),))
    result_context, computer_context = _visual_target_run(
        original, fresh, context_matches=False,
    )
    assert not result_context.success
    assert result_context.visual_preclick_revalidation is not None
    assert result_context.visual_preclick_revalidation.result.value == "CONTEXT_CHANGED"
    assert not computer_context.preclick_directed_calls
    assert not computer_context.visual_activations

    failed = replace(
        observation("fresh-provider-failed"),
        visual_directed_grounding=True,
        visual_requested_max_elements=5,
        visual_provider_error=ProviderErrorDiagnostic("timeout"),
        visual_grounding_status=VisualGroundingStatus.PROVIDER_ERROR,
        visual_provider_attempts=(VisualProviderAttempt("gemini", "test", 10, "timeout"),),
    )
    result_failed, computer_failed = _visual_target_run(original, failed)
    assert not result_failed.success
    assert result_failed.visual_preclick_revalidation is not None
    assert result_failed.visual_preclick_revalidation.result.value == "INCOMPLETE"
    assert len(result_failed.visual_preclick_revalidation.provider_attempts) == 1
    assert not computer_failed.visual_activations


def test_visual_preclick_click_failure_is_not_retried() -> None:
    original = VisualElement("v1", "Iago", "conversation", Rect(40, 180, 300, 230), None, True)
    fresh = observation("fresh-click-fails", visual=(replace(original),))
    grounded = observation("grounded", visual=(original,))
    failing = FakeComputer(
        [observation("before"), active_conversation("after", "Iago")],
        [grounded], visual_provider=object(),
    )
    failing.preclick_directed_observations.append(fresh)
    failing.visual_click_result = ActionResult(
        False, VisualClickAction("fresh-click-fails", "v1"), "synthetic failure",
        error="windows_operation_failed", input_issued=False,
    )
    failed_result = GenericTaskDebugAgent(failing, FakeDecisionMaker("v1")).run("Select Iago")

    assert not failed_result.success and failed_result.stop_reason == "target_activation_failed"
    assert len(failing.visual_activations) == 1
    assert failed_result.visual_preclick_revalidation is not None
    assert failed_result.visual_preclick_revalidation.click_released is False
