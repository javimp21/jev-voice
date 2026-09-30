"""Offline contract tests for the experimental LangGraph orchestration."""

from dataclasses import dataclass

import pytest

from agent.graph import AgentState, GraphAgent
from agent.loop import Agent, AgentLimits
from computer.applications import ApplicationCandidate, MemoryApplicationCatalog
from computer.actions import FinishAction, OpenAppAction, PressKeyAction, TypeAction
from computer.models import Observation
from computer.results import ActionResult
from decision.models import DecisionResult
from safety.interfaces import SafetyDecision
from safety.policy import AutonomousActionPolicy, BasicActionPolicy


def _observation(identity: str) -> Observation:
    return Observation("test.exe", "Test Window", observation_id=identity)


def _incomplete_observation(identity: str) -> Observation:
    return Observation(
        "test.exe", "Test Window", observation_id=identity, truncated=True,
    )


@dataclass
class FakeComputer:
    observations: list[Observation | Exception]

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
        return ActionResult(True, action, "ok")


class FailingComputer(FakeComputer):
    def execute(self, action, observation=None) -> ActionResult:
        self.executed.append((action, observation))
        return ActionResult(False, action, "failed", error="windows_operation_failed")


class ScriptedDecision:
    def __init__(self, results: list[DecisionResult]) -> None:
        self.results = results
        self.calls: list[tuple[str, Observation, tuple[ActionResult, ...]]] = []

    def decide(self, request, observation, history=()) -> DecisionResult:
        self.calls.append((request, observation, tuple(history)))
        return self.results.pop(0)


class DenyPolicy:
    def __init__(self) -> None:
        self.calls = 0

    def validate(self, action, observation) -> SafetyDecision:
        self.calls += 1
        return SafetyDecision("deny", "test denial")


class AllowThenDenyPolicy:
    def __init__(self) -> None:
        self.calls: list[object] = []

    def validate(self, action, observation) -> SafetyDecision:
        self.calls.append(action)
        if len(self.calls) == 2:
            return SafetyDecision("deny", "second proposal denied")
        return SafetyDecision("allow", "first proposal allowed")


def _ready(action) -> DecisionResult:
    return DecisionResult("ready", action, 0.95, "selected")


def test_state_type_and_graph_starts_with_observe() -> None:
    computer = FakeComputer([_observation("obs-1")])
    agent = GraphAgent(computer, ScriptedDecision([_ready(FinishAction("done"))]))

    result = agent.run("finish task")

    assert result.result.success is True
    assert result.result.stop_reason == "finished"
    assert result.diagnostics[0].node == "OBSERVE"
    assert issubclass(AgentState, dict)
    assert "observation" in AgentState.__annotations__


def test_finish_routes_through_safety_then_finish_without_computer_action() -> None:
    finish = FinishAction("done")
    computer = FakeComputer([_observation("obs-1")])

    result = GraphAgent(
        computer, ScriptedDecision([_ready(finish)]),
        policy=BasicActionPolicy(), sleep_fn=lambda _seconds: None,
    ).run("finish task")

    assert result.result.success is True
    assert result.result.stop_reason == "finished"
    assert result.result.message == "done"
    assert [item.node for item in result.diagnostics] == [
        "OBSERVE", "DECIDE", "SAFETY_CHECK", "FINISH",
    ]
    assert computer.executed == []


def test_action_execution_is_followed_by_fresh_observation_before_next_decision() -> None:
    first, second = _observation("before"), _observation("after")
    computer = FakeComputer([first, second])
    decision = ScriptedDecision([_ready(TypeAction("literal")), _ready(FinishAction("complete"))])

    result = GraphAgent(
        computer, decision, policy=BasicActionPolicy(), sleep_fn=lambda _seconds: None,
    ).run("type a value")

    assert result.result.success is True
    assert len(computer.observed) == 2
    assert [call[1].observation_id for call in decision.calls] == ["before", "after"]
    assert computer.executed[0][1] is first
    assert [item.node for item in result.diagnostics] == [
        "OBSERVE", "DECIDE", "SAFETY_CHECK", "EXECUTE", "REOBSERVE",
        "VERIFY_OR_CONTINUE", "DECIDE", "SAFETY_CHECK", "FINISH",
    ]


def test_safety_denial_stops_before_computer_execution() -> None:
    computer = FakeComputer([_observation("obs-1")])
    policy = DenyPolicy()

    result = GraphAgent(
        computer, ScriptedDecision([_ready(TypeAction("no action"))]), policy=policy,
    ).run("type a value")

    assert result.result.success is False
    assert result.result.stop_reason == "safety_rejected"
    assert policy.calls == 1
    assert computer.executed == []
    assert result.diagnostics[-1].node == "FAIL"


def test_observation_failure_fails_closed() -> None:
    computer = FakeComputer([RuntimeError("private detail")])

    result = GraphAgent(computer, ScriptedDecision([])).run("request")

    assert result.result.stop_reason == "observation_failed"
    assert result.result.steps == 0
    assert result.diagnostics[0].node == "OBSERVE"
    assert "private detail" not in result.result.message


@pytest.mark.parametrize(
    "decision",
    [
        DecisionResult("ready", None, 0.95, "missing action"),
        DecisionResult("unexpected", TypeAction("x"), 0.95, "invalid status"),  # type: ignore[arg-type]
        DecisionResult("ready", TypeAction("x"), None, "missing confidence"),
    ],
)
def test_invalid_or_incomplete_decision_fails_closed(decision: DecisionResult) -> None:
    computer = FakeComputer([_observation("obs-1")])

    result = GraphAgent(computer, ScriptedDecision([decision])).run("request")

    assert result.result.success is False
    assert result.result.stop_reason in {"decision_error", "low_confidence"}
    assert computer.executed == []


def test_post_action_observation_failure_blocks_continuation() -> None:
    computer = FakeComputer([_observation("before"), RuntimeError("post-action failure")])

    result = GraphAgent(
        computer, ScriptedDecision([_ready(TypeAction("value"))]),
        policy=BasicActionPolicy(), sleep_fn=lambda _seconds: None,
    ).run("type a value")

    assert result.result.stop_reason == "observation_failed"
    assert len(computer.executed) == 1
    assert result.diagnostics[-1].node == "FAIL"
    assert not any(item.node == "DECIDE" for item in result.diagnostics[5:])


def test_max_step_bound_is_enforced_after_fresh_post_action_observation() -> None:
    computer = FakeComputer([_observation(f"obs-{i}") for i in range(1, 4)])
    decision = ScriptedDecision([_ready(TypeAction("value")), _ready(TypeAction("value"))])

    result = GraphAgent(
        computer, decision, policy=BasicActionPolicy(),
        limits=AgentLimits(max_steps=2, repeat_limit=5, settle_action_seconds=0),
        sleep_fn=lambda _seconds: None,
    ).run("type a value")

    assert result.result.stop_reason == "max_steps"
    assert result.result.steps == 2
    assert len(computer.executed) == 2
    assert len(computer.observed) == 3
    assert result.diagnostics[-1].node == "FAIL"


def test_max_steps_is_terminal_even_when_post_action_observation_is_incomplete() -> None:
    computer = FakeComputer([_observation("before"), _incomplete_observation("after")])

    result = GraphAgent(
        computer, ScriptedDecision([_ready(TypeAction("value"))]),
        policy=BasicActionPolicy(), limits=AgentLimits(max_steps=1),
        sleep_fn=lambda _seconds: None,
    ).run("perform task")

    assert result.result.stop_reason == "max_steps"
    assert result.replan_count == 0
    assert "REPLAN" not in [item.node for item in result.diagnostics]


def test_dry_run_finishes_after_one_proposal_without_safety_or_execution() -> None:
    computer = FakeComputer([_observation("obs-1")])
    policy = DenyPolicy()

    result = GraphAgent(
        computer, ScriptedDecision([_ready(TypeAction("secret literal"))]),
        policy=policy,
    ).run("secret request", dry_run=True)

    assert result.result.success is True
    assert result.result.stop_reason == "needs_human"
    assert policy.calls == 0
    assert computer.executed == []
    assert [item.node for item in result.diagnostics] == [
        "OBSERVE", "DECIDE", "SAFETY_CHECK", "FINISH",
    ]


def test_recoverable_post_action_observation_incompleteness_replans_once() -> None:
    before = _observation("before")
    incomplete = _incomplete_observation("after-incomplete")
    computer = FakeComputer([before, incomplete])
    decision = ScriptedDecision([
        _ready(TypeAction("first action")), _ready(FinishAction("done")),
    ])

    result = GraphAgent(
        computer, decision, policy=BasicActionPolicy(), sleep_fn=lambda _seconds: None,
    ).run("perform task")

    assert result.result.success is True
    assert result.replan_count == 1
    assert result.max_replans == 1
    assert result.last_replan_reason == "post_action_observation_incomplete"
    assert [item.node for item in result.diagnostics] == [
        "OBSERVE", "DECIDE", "SAFETY_CHECK", "EXECUTE", "REOBSERVE",
        "VERIFY_OR_CONTINUE", "REPLAN", "DECIDE", "SAFETY_CHECK", "FINISH",
    ]
    assert decision.calls[1][0].endswith(
        "Replanning context: post_action_observation_incomplete. "
        "Choose based on the current observation and prior action history; "
        "do not retry an action automatically."
    )
    assert decision.calls[1][1] is incomplete
    assert decision.calls[1][2][-1].action == TypeAction("first action")
    assert len(decision.calls) == 2
    assert result.diagnostics[6].action_kind is None
    assert result.diagnostics[6].replan_count == 1
    assert result.diagnostics[6].failure_recoverable is True
    assert len(computer.executed) == 1


def test_structural_inspection_errors_are_a_typed_recoverable_outcome() -> None:
    incomplete = Observation(
        "test.exe", "Test Window", observation_id="after-errors",
        inspection_errors=1,
    )
    computer = FakeComputer([_observation("before"), incomplete])
    decision = ScriptedDecision([
        _ready(TypeAction("first action")), _ready(FinishAction("done")),
    ])

    result = GraphAgent(
        computer, decision, policy=BasicActionPolicy(), sleep_fn=lambda _seconds: None,
    ).run("perform task")

    assert result.replan_count == 1
    assert "REPLAN" in [item.node for item in result.diagnostics]
    assert result.diagnostics[5].failure_recoverable is True


def test_replan_budget_exhaustion_fails_without_a_second_replan() -> None:
    computer = FakeComputer([
        _observation("before"),
        _incomplete_observation("partial-1"),
        _incomplete_observation("partial-2"),
    ])
    limits = AgentLimits(max_steps=3, repeat_limit=5, settle_action_seconds=0)
    decision = ScriptedDecision([
        _ready(TypeAction("same")), _ready(TypeAction("same")),
    ])

    result = GraphAgent(
        computer, decision, policy=BasicActionPolicy(), limits=limits,
        sleep_fn=lambda _seconds: None,
    ).run("perform task")

    assert result.result.success is False
    assert result.result.stop_reason == "observation_failed"
    assert result.transition_reason == "replan_budget_exhausted"
    assert result.replan_count == result.max_replans == 1
    assert result.last_replan_reason == "post_action_observation_incomplete"
    assert [item.node for item in result.diagnostics].count("REPLAN") == 1
    assert result.diagnostics[-2].failure_recoverable is True
    assert result.diagnostics[-1].node == "FAIL"
    assert len(computer.executed) == 2


def test_terminal_safety_rejection_never_enters_replan() -> None:
    computer = FakeComputer([_observation("before")])

    result = GraphAgent(
        computer, ScriptedDecision([_ready(TypeAction("blocked"))]),
        policy=DenyPolicy(),
    ).run("perform task")

    assert result.result.stop_reason == "safety_rejected"
    assert result.replan_count == 0
    assert "REPLAN" not in [item.node for item in result.diagnostics]
    assert computer.executed == []


def test_confirmation_required_action_never_enters_replan() -> None:
    computer = FakeComputer([_observation("before")])
    result = GraphAgent(
        computer, ScriptedDecision([_ready(PressKeyAction(("enter",)))]),
        policy=AutonomousActionPolicy(MemoryApplicationCatalog(())),
    ).run("press enter")

    assert result.result.stop_reason == "confirmation_required"
    assert result.replan_count == 0
    assert "REPLAN" not in [item.node for item in result.diagnostics]
    assert computer.executed == []


def test_missing_fresh_observation_never_enters_replan() -> None:
    computer = FakeComputer([_observation("before"), RuntimeError("observe failed")])

    result = GraphAgent(
        computer, ScriptedDecision([_ready(TypeAction("value"))]),
        policy=BasicActionPolicy(), sleep_fn=lambda _seconds: None,
    ).run("perform task")

    assert result.result.stop_reason == "observation_failed"
    assert result.replan_count == 0
    assert "REPLAN" not in [item.node for item in result.diagnostics]


def test_executor_hard_failure_never_enters_replan() -> None:
    computer = FailingComputer([_observation("before")])

    result = GraphAgent(
        computer, ScriptedDecision([_ready(TypeAction("value"))]),
        policy=BasicActionPolicy(), sleep_fn=lambda _seconds: None,
    ).run("perform task")

    assert result.result.stop_reason == "execution_failed"
    assert result.replan_count == 0
    assert "REPLAN" not in [item.node for item in result.diagnostics]


def test_replan_does_not_execute_and_same_action_still_passes_safety() -> None:
    action = TypeAction("same action")
    computer = FakeComputer([_observation("before"), _incomplete_observation("after")])
    policy = AllowThenDenyPolicy()

    result = GraphAgent(
        computer, ScriptedDecision([_ready(action), _ready(action)]),
        policy=policy, sleep_fn=lambda _seconds: None,
    ).run("perform task")

    assert result.result.stop_reason == "safety_rejected"
    assert result.replan_count == 1
    assert policy.calls == [action, action]
    assert len(computer.executed) == 1
    nodes = [item.node for item in result.diagnostics]
    replan_index = nodes.index("REPLAN")
    assert nodes[replan_index + 1] == "DECIDE"
    assert "EXECUTE" not in nodes[replan_index + 1:nodes.index("FAIL")]


def test_reobserve_requires_a_distinct_snapshot_binding() -> None:
    same_snapshot = _observation("same")
    computer = FakeComputer([same_snapshot, same_snapshot])

    result = GraphAgent(
        computer, ScriptedDecision([_ready(TypeAction("value"))]),
        policy=BasicActionPolicy(), sleep_fn=lambda _seconds: None,
    ).run("perform task")

    assert result.result.stop_reason == "observation_failed"
    assert result.transition_reason == "fresh_snapshot_missing"
    assert result.replan_count == 0
    assert "REPLAN" not in [item.node for item in result.diagnostics]


def test_non_graph_agent_loop_keeps_its_original_flow() -> None:
    computer = FakeComputer([_observation("before"), _observation("after")])
    decision = ScriptedDecision([
        _ready(TypeAction("literal")), _ready(FinishAction("done")),
    ])

    result = Agent(
        computer, decision, policy=BasicActionPolicy(),
        limits=AgentLimits(max_steps=2, settle_action_seconds=0),
        sleep_fn=lambda _seconds: None,
    ).run("perform task")

    assert result.success is True
    assert result.stop_reason == "finished"
    assert len(computer.executed) == 1
    assert len(decision.calls) == 2


def _open_app_candidate(
    app_id: str,
    name: str,
    *,
    identity_fingerprint: str | None = None,
    source: str = "test",
    launch_target_type: str = "other",
) -> ApplicationCandidate:
    return ApplicationCandidate(
        app_id, name, source, publisher="PRIVATE_PUBLISHER",
        process_names=(r"C:\Users\private\Apps\notepad++.exe",),
        identity_fingerprint=identity_fingerprint,
        launch_target_type=launch_target_type,
    )


def _run_open_app_diagnostic(
    request: str,
    candidates: tuple[ApplicationCandidate, ...],
    decision_result: DecisionResult | None = None,
) -> tuple[object, object]:
    catalog = MemoryApplicationCatalog(candidates)
    scripted = ScriptedDecision([decision_result or DecisionResult(
        "ready", OpenAppAction(candidates[0].id), 0.95, "PRIVATE_PROMPT",
        model="jev-latest", provider_called=True,
    )])
    scripted.app_catalog = catalog
    scripted.app_candidate_limit = 5
    result = GraphAgent(
        FakeComputer([_observation("open-app-diagnostic")]), scripted,
        policy=DenyPolicy(), sleep_fn=lambda _seconds: None,
    ).run(request)
    decide_entry = next(item for item in result.diagnostics if item.node == "DECIDE")
    return result, decide_entry.open_app_decision


def test_open_app_exact_unique_catalog_match_is_reported_before_jev() -> None:
    candidate = _open_app_candidate("trusted-notepad-plus-plus", "Notepad++")
    plain_notepad = _open_app_candidate("trusted-notepad", "Notepad")
    result, diagnostic = _run_open_app_diagnostic(
        "Open Notepad++", (plain_notepad, candidate),
    )

    assert result.result.stop_reason == "safety_rejected"
    assert diagnostic.requested_action_kind == "open_app"
    assert diagnostic.requested_app_name_normalized == "notepad++"
    assert diagnostic.trusted_catalog_candidate_count == 1
    assert diagnostic.trusted_catalog_match_kind == "raw_exact"
    assert diagnostic.trusted_catalog_selected_app_id_present is True
    assert diagnostic.catalog_resolution_before_jev is True
    assert diagnostic.jev_was_called is True


def test_open_app_normalized_unique_catalog_match_is_reported() -> None:
    candidate = _open_app_candidate("trusted-notepad", "Notepad™")
    _result, diagnostic = _run_open_app_diagnostic("Open Notepad", (candidate,))

    assert diagnostic.trusted_catalog_candidate_count == 1
    assert diagnostic.trusted_catalog_match_kind == "canonical_exact"
    assert diagnostic.trusted_catalog_selected_app_id_present is True


def test_open_app_ambiguous_catalog_match_is_reported_without_selection() -> None:
    candidates = (
        _open_app_candidate("trusted-notepad-one", "Notepad"),
        _open_app_candidate("trusted-notepad-two", "Notepad"),
    )
    _result, diagnostic = _run_open_app_diagnostic("Open Notepad", candidates)

    assert diagnostic.trusted_catalog_candidate_count == 2
    assert diagnostic.trusted_catalog_match_kind == "ambiguous"
    assert diagnostic.trusted_catalog_selected_app_id_present is False
    assert diagnostic.candidate_count == 2
    assert diagnostic.trusted_catalog_candidate_count == 2


def test_open_app_diagnostics_distinguish_proven_duplicate_representations_safely() -> None:
    candidates = (
        _open_app_candidate(
            "app_aaaaaaaaaaaaaaaa", "Notepad++", identity_fingerprint="shortcut:private-target-hash",
            source="start_menu", launch_target_type="shortcut",
        ),
        _open_app_candidate(
            "app_bbbbbbbbbbbbbbbb", "Notepad++", identity_fingerprint="shortcut:private-target-hash",
            source="start_menu", launch_target_type="shortcut",
        ),
    )
    _result, diagnostic = _run_open_app_diagnostic("Open Notepad++", candidates)

    assert diagnostic.candidate_count == 2
    assert diagnostic.trusted_catalog_candidate_count == 1
    assert diagnostic.trusted_catalog_match_kind == "raw_exact"
    assert diagnostic.trusted_catalog_selected_app_id_present is True
    assert len(diagnostic.matching_candidates) == 2
    assert all(item.canonical_name == "notepad++" for item in diagnostic.matching_candidates)
    assert all(item.identity_fingerprint_present for item in diagnostic.matching_candidates)
    assert {item.identity_equivalence_group for item in diagnostic.matching_candidates} == {
        "identity_group_1",
    }
    assert all(item.duplicate_of_another_candidate for item in diagnostic.matching_candidates)
    assert all(item.launch_target_type == "shortcut" for item in diagnostic.matching_candidates)

    import json
    from dataclasses import asdict
    serialized = json.dumps(asdict(diagnostic))
    for forbidden in ("private-target-hash", "C:\\Users\\private", "notepad++.exe", "PRIVATE_PUBLISHER"):
        assert forbidden not in serialized


def test_open_app_no_catalog_match_is_reported() -> None:
    no_match = DecisionResult(
        "needs_human", None, None, "PRIVATE_PROMPT",
        diagnostic="no_application_match", provider_called=False,
    )
    _result, diagnostic = _run_open_app_diagnostic(
        "Open Notepad++", (_open_app_candidate("trusted-calculator", "Calculator"),),
        no_match,
    )

    assert diagnostic.trusted_catalog_candidate_count == 0
    assert diagnostic.trusted_catalog_match_kind == "no_match"
    assert diagnostic.trusted_catalog_selected_app_id_present is False
    assert diagnostic.jev_was_called is False
    assert diagnostic.decision_rejection_reason == "no_application_match"


def test_jev_low_confidence_with_unique_catalog_match_is_diagnosed_without_behavior_change() -> None:
    candidate = _open_app_candidate("trusted-notepad-plus-plus", "Notepad++")
    low_confidence = DecisionResult(
        "ready", OpenAppAction(candidate.id), 0.72, "PRIVATE_PROMPT",
        model="jev-latest", provider_called=True,
    )
    result, diagnostic = _run_open_app_diagnostic(
        "Open Notepad++", (candidate,), low_confidence,
    )

    assert result.result.stop_reason == "low_confidence"
    assert diagnostic.trusted_catalog_candidate_count == 1
    assert diagnostic.trusted_catalog_match_kind == "raw_exact"
    assert diagnostic.decision_action_kind == "open_app"
    assert diagnostic.decision_confidence == 0.72
    assert diagnostic.decision_rejection_reason == "low_confidence"
    assert diagnostic.jev_was_called is True


def test_open_app_diagnostic_contains_no_paths_secrets_or_prompt_text() -> None:
    candidate = _open_app_candidate(
        r"C:\private\catalog\SECRET_TOKEN_338", "Notepad++",
    )
    private_prompt = "PRIVATE_USER_TASK_AND_API_KEY sk-private-901"
    response = DecisionResult(
        "ready", OpenAppAction(candidate.id), 0.72, private_prompt,
        model="jev-latest", provider_called=True,
    )
    _result, diagnostic = _run_open_app_diagnostic(
        "Open Notepad++", (candidate,), response,
    )
    from dataclasses import asdict
    import json

    serialized = json.dumps(asdict(diagnostic))
    for forbidden in (
        "C:\\private", "SECRET_TOKEN_338", "notepad++.exe",
        "PRIVATE_PUBLISHER", private_prompt, "sk-private-901",
    ):
        assert forbidden not in serialized
