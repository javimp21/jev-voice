"""Deterministic, fully offline scenarios for the experimental LangGraph agent."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from collections.abc import Sequence

from agent.graph import GraphAgent
from agent.loop import AgentLimits
from agent.telemetry import InMemoryTelemetryCollector
from computer.actions import Action, FinishAction, OpenAppAction, TypeAction
from computer.models import Observation
from computer.results import ActionResult
from decision.models import DecisionResult
from safety.interfaces import SafetyDecision


@dataclass(frozen=True, slots=True)
class EvalScenario:
    """Inputs and expected graph behavior for one isolated evaluation."""

    name: str
    fake_observations: tuple[Observation, ...]
    fake_decisions: tuple[DecisionResult, ...]
    fake_safety_outcomes: tuple[str, ...]
    fake_executor_outcomes: tuple[bool, ...]
    max_steps: int
    max_replans: int
    expected_success: bool
    expected_stop_reason: str
    expected_steps: int
    expected_replans: int
    expected_nodes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EvalScenarioResult:
    name: str
    passed: bool
    expected_success: bool
    actual_success: bool
    expected_stop_reason: str
    actual_stop_reason: str
    expected_steps: int
    actual_steps: int
    expected_replans: int
    actual_replans: int
    expected_nodes: tuple[str, ...]
    actual_nodes: tuple[str, ...]
    mismatches: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EvalReport:
    total_scenarios: int
    passed: int
    failed: int
    success_rate: float
    average_steps: float
    average_replans: float
    stop_reason_distribution: dict[str, int]
    transition_counts: dict[str, int]
    scenarios: tuple[EvalScenarioResult, ...]


class _FakeComputer:
    def __init__(
        self,
        observations: tuple[Observation, ...],
        outcomes: tuple[bool, ...],
    ) -> None:
        self._observations = list(observations)
        self._executor_outcomes = list(outcomes)
        self.observe_calls = 0
        self.execute_calls = 0

    def observe(self) -> Observation:
        self.observe_calls += 1
        if self._observations:
            return self._observations.pop(0)
        return Observation("eval.exe", "Eval", error="fake_observation_exhausted")

    def execute(
        self, action: Action, observation: Observation | None = None,
    ) -> ActionResult:
        self.execute_calls += 1
        succeeded = self._executor_outcomes.pop(0) if self._executor_outcomes else True
        return ActionResult(
            succeeded, action, "offline eval result",
            error=None if succeeded else "fake_executor_failure",
            source_observation_id=observation.observation_id if observation else None,
        )


class _FakeDecisionMaker:
    def __init__(self, decisions: tuple[DecisionResult, ...]) -> None:
        self._decisions = list(decisions)

    def decide(
        self, _request: str, _observation: Observation,
        _history: tuple[ActionResult, ...] = (),
    ) -> DecisionResult:
        if self._decisions:
            return self._decisions.pop(0)
        return DecisionResult("error", None, None, "offline eval exhausted", error="invalid_input")


class _FakePolicy:
    def __init__(self, outcomes: tuple[str, ...]) -> None:
        self._outcomes = list(outcomes)

    def validate(
        self, _action: Action, _observation: Observation,
    ) -> SafetyDecision:
        outcome = self._outcomes.pop(0) if self._outcomes else "allow"
        if outcome not in {"allow", "deny", "confirm"}:
            outcome = "deny"
        return SafetyDecision(outcome, "offline eval policy")


def _obs(identity: str, *, incomplete: bool = False) -> Observation:
    return Observation(
        "eval.exe", "Eval Window", observation_id=identity,
        truncated=incomplete,
    )


def _ready(action: Action) -> DecisionResult:
    return DecisionResult("ready", action, 0.99, "offline", selected_option="eval")


_TYPE = TypeAction("eval-only literal")
_FINISH = FinishAction("eval complete")


DEFAULT_EVAL_SCENARIOS: tuple[EvalScenario, ...] = (
    EvalScenario(
        "open_app_success", (_obs("open-before"), _obs("open-after")),
        (_ready(OpenAppAction("notepad")), _ready(_FINISH)), ("allow", "allow"), (True,),
        8, 1, True, "finished", 2, 0,
        ("OBSERVE", "DECIDE", "SAFETY_CHECK", "EXECUTE", "REOBSERVE",
         "VERIFY_OR_CONTINUE", "DECIDE", "SAFETY_CHECK", "FINISH"),
    ),
    EvalScenario(
        "safety_rejection", (_obs("safety"),), (_ready(_TYPE),), ("deny",), (),
        8, 1, False, "safety_rejected", 1, 0,
        ("OBSERVE", "DECIDE", "SAFETY_CHECK", "FAIL"),
    ),
    EvalScenario(
        "executor_failure", (_obs("executor"),), (_ready(_TYPE),), ("allow",), (False,),
        8, 1, False, "execution_failed", 1, 0,
        ("OBSERVE", "DECIDE", "SAFETY_CHECK", "EXECUTE", "FAIL"),
    ),
    EvalScenario(
        "recoverable_incomplete_observation_then_success",
        (_obs("replan-before"), _obs("replan-incomplete", incomplete=True)),
        (_ready(_TYPE), _ready(_FINISH)), ("allow", "allow"), (True,),
        8, 1, True, "finished", 2, 1,
        ("OBSERVE", "DECIDE", "SAFETY_CHECK", "EXECUTE", "REOBSERVE",
         "VERIFY_OR_CONTINUE", "REPLAN", "DECIDE", "SAFETY_CHECK", "FINISH"),
    ),
    EvalScenario(
        "recoverable_observation_then_decision_failure",
        (_obs("decision-before"), _obs("decision-incomplete", incomplete=True)),
        (_ready(_TYPE), DecisionResult("error", None, None, "offline error", error="invalid_input")),
        ("allow",), (True,), 8, 1, False, "decision_error", 2, 1,
        ("OBSERVE", "DECIDE", "SAFETY_CHECK", "EXECUTE", "REOBSERVE",
         "VERIFY_OR_CONTINUE", "REPLAN", "DECIDE", "FAIL"),
    ),
    EvalScenario(
        "replan_budget_exhausted",
        (_obs("budget-before"), _obs("budget-incomplete-1", incomplete=True),
         _obs("budget-incomplete-2", incomplete=True)),
        (_ready(_TYPE), _ready(_TYPE)), ("allow", "allow"), (True, True),
        3, 1, False, "observation_failed", 3, 1,
        ("OBSERVE", "DECIDE", "SAFETY_CHECK", "EXECUTE", "REOBSERVE",
         "VERIFY_OR_CONTINUE", "REPLAN", "DECIDE", "SAFETY_CHECK", "EXECUTE",
         "REOBSERVE", "VERIFY_OR_CONTINUE", "FAIL"),
    ),
    EvalScenario(
        "duplicate_snapshot_failure", (_obs("duplicate"), _obs("duplicate")),
        (_ready(_TYPE),), ("allow",), (True,), 8, 1, False, "observation_failed", 1, 0,
        ("OBSERVE", "DECIDE", "SAFETY_CHECK", "EXECUTE", "REOBSERVE", "FAIL"),
    ),
    EvalScenario(
        "finish_without_execution", (_obs("finish"),), (_ready(_FINISH),), ("allow",), (),
        8, 1, True, "finished", 1, 0,
        ("OBSERVE", "DECIDE", "SAFETY_CHECK", "FINISH"),
    ),
    EvalScenario(
        "max_steps_reached", (_obs("max-before"), _obs("max-after")), (_ready(_TYPE),),
        ("allow",), (True,), 1, 1, False, "max_steps", 1, 0,
        ("OBSERVE", "DECIDE", "SAFETY_CHECK", "EXECUTE", "REOBSERVE",
         "VERIFY_OR_CONTINUE", "FAIL"),
    ),
    EvalScenario(
        "invalid_decision", (_obs("invalid-decision"),),
        (DecisionResult("ready", None, 0.99, "invalid offline action"),), (), (),
        8, 1, False, "decision_error", 1, 0,
        ("OBSERVE", "DECIDE", "FAIL"),
    ),
)


def run_evaluations(
    scenarios: Sequence[EvalScenario] = DEFAULT_EVAL_SCENARIOS,
) -> EvalReport:
    """Run scenarios against the real graph and exclusively fake dependencies."""

    results: list[EvalScenarioResult] = []
    stop_reasons: Counter[str] = Counter()
    transitions: Counter[str] = Counter()
    total_steps = 0
    total_replans = 0
    for scenario in scenarios:
        computer = _FakeComputer(scenario.fake_observations, scenario.fake_executor_outcomes)
        collector = InMemoryTelemetryCollector()
        result = GraphAgent(
            computer, _FakeDecisionMaker(scenario.fake_decisions),
            policy=_FakePolicy(scenario.fake_safety_outcomes),
            limits=AgentLimits(max_steps=scenario.max_steps, settle_open_seconds=0,
                               settle_action_seconds=0),
            max_replans=scenario.max_replans, sleep_fn=lambda _seconds: None,
            telemetry_collector=collector,
        ).run("offline deterministic evaluation")
        nodes = tuple(item.node for item in result.diagnostics)
        mismatches: list[str] = []
        if result.result.success != scenario.expected_success:
            mismatches.append("success_mismatch")
        if result.result.stop_reason != scenario.expected_stop_reason:
            mismatches.append("stop_reason_mismatch")
        if result.result.steps != scenario.expected_steps:
            mismatches.append("steps_mismatch")
        if result.replan_count != scenario.expected_replans:
            mismatches.append("replan_count_mismatch")
        if result.max_replans != scenario.max_replans:
            mismatches.append("max_replans_mismatch")
        if nodes != scenario.expected_nodes:
            mismatches.append("node_sequence_mismatch")
        results.append(EvalScenarioResult(
            scenario.name, not mismatches, scenario.expected_success, result.result.success,
            scenario.expected_stop_reason, result.result.stop_reason,
            scenario.expected_steps, result.result.steps,
            scenario.expected_replans, result.replan_count,
            scenario.expected_nodes, nodes, tuple(mismatches),
        ))
        stop_reasons[result.result.stop_reason] += 1
        transitions.update(f"{left}->{right}" for left, right in zip(nodes, nodes[1:]))
        total_steps += result.result.steps
        total_replans += result.replan_count

    total = len(results)
    passed = sum(item.passed for item in results)
    return EvalReport(
        total, passed, total - passed,
        round(passed / total, 3) if total else 1.0,
        round(total_steps / total, 3) if total else 0.0,
        round(total_replans / total, 3) if total else 0.0,
        dict(sorted(stop_reasons.items())), dict(sorted(transitions.items())), tuple(results),
    )


def format_eval_report(report: EvalReport) -> str:
    """Render concise human-readable output without action or request content."""

    lines = [
        f"Agent evals: {report.total_scenarios} scenarios; {report.passed} passed; {report.failed} failed",
        f"Pass rate: {report.success_rate:.1%}; average steps: {report.average_steps:.3f}; "
        f"average replans: {report.average_replans:.3f}",
        "Stop reasons:",
    ]
    lines.extend(f"  {reason}: {count}" for reason, count in report.stop_reason_distribution.items())
    lines.append("Scenarios:")
    lines.extend(
        f"  {'PASS' if item.passed else 'FAIL'} {item.name}: "
        f"{item.actual_stop_reason}; steps={item.actual_steps}; replans={item.actual_replans}"
        + (f"; mismatches={','.join(item.mismatches)}" if item.mismatches else "")
        for item in report.scenarios
    )
    return "\n".join(lines)
