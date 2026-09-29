"""Privacy and persistence tests for structured graph telemetry."""

from dataclasses import asdict
import json
from pathlib import Path

from agent.graph import GraphAgent
from agent.telemetry import InMemoryTelemetryCollector, JsonlTelemetrySink
from computer.models import Observation
from computer.results import ActionResult
from computer.actions import Action, TypeAction
from decision.models import DecisionResult
from safety.interfaces import SafetyDecision


class _Computer:
    def __init__(self) -> None:
        self.calls = 0

    def observe(self) -> Observation:
        self.calls += 1
        return Observation(
            "test.exe", "Test", observation_id=f"obs-{self.calls}",
            visual_provider="gemini", visual_latency_ms=321,
        )

    def execute(self, action: Action, observation: Observation | None = None) -> ActionResult:
        return ActionResult(True, action, "ok")


class _Decision:
    def __init__(self, action: Action) -> None:
        self.action = action

    def decide(
        self, _request: str, _observation: Observation,
        _history: tuple[ActionResult, ...] = (),
    ) -> DecisionResult:
        return DecisionResult("ready", self.action, 0.99, "selected")


def test_telemetry_marks_incomplete_observation_and_replan_reason() -> None:
    from computer.actions import FinishAction

    class ReplanComputer(_Computer):
        def __init__(self) -> None:
            super().__init__()
            self.observations = [
                Observation("test.exe", "Test", observation_id="before"),
                Observation("test.exe", "Test", observation_id="after", truncated=True),
            ]

        def observe(self) -> Observation:
            self.calls += 1
            return self.observations.pop(0)

    class Decisions:
        def __init__(self) -> None:
            self.actions = [TypeAction("hidden"), FinishAction("done")]

        def decide(
            self, _request: str, _observation: Observation,
            _history: tuple[ActionResult, ...] = (),
        ) -> DecisionResult:
            return DecisionResult("ready", self.actions.pop(0), 0.99, "selected")

    class Allow:
        def validate(self, _action: Action, _observation: Observation) -> SafetyDecision:
            return SafetyDecision("allow", "allowed")

    collector = InMemoryTelemetryCollector()
    result = GraphAgent(
        ReplanComputer(), Decisions(), policy=Allow(), max_replans=1,
        sleep_fn=lambda _seconds: None, telemetry_collector=collector,
    ).run("private request")

    assert result.result.success is True
    verify = next(
        event for event in collector.events
        if event.event_type == "node_completed" and event.node == "VERIFY_OR_CONTINUE"
    )
    replan = next(
        event for event in collector.events
        if event.event_type == "node_completed" and event.node == "REPLAN"
    )
    assert verify.observation_complete is False
    assert verify.recoverable_failure is True
    assert verify.transition_reason == "recoverable_post_action_observation"
    assert replan.transition_reason == "replan_started"
    assert replan.safety_outcome is None


def test_graph_emits_run_and_node_telemetry_with_safe_metadata() -> None:
    collector = InMemoryTelemetryCollector()
    action = TypeAction("PRIVATE_TYPED_VALUE")
    result = GraphAgent(
        _Computer(), _Decision(action), telemetry_collector=collector,
    ).run("PRIVATE_COMMAND", dry_run=True)

    assert result.result.success is True
    assert [event.event_type for event in collector.events] == [
        "run_started", "node_completed", "node_completed", "node_completed",
        "node_completed", "run_completed",
    ]
    start, *middle, end = collector.events
    nodes = [event for event in middle if event.event_type == "node_completed"]
    assert [event.node for event in nodes] == ["OBSERVE", "DECIDE", "SAFETY_CHECK", "FINISH"]
    assert all(event.run_id == start.run_id == end.run_id for event in collector.events)
    assert start.started_at is not None
    assert end.started_at == start.started_at
    assert end.ended_at is not None and end.total_duration_ms is not None
    assert end.stop_reason == "needs_human"
    assert nodes[0].observation_complete is True
    assert nodes[0].provider_used == "gemini"
    assert nodes[0].provider_latency_ms == 321
    assert nodes[1].action_kind == "type"
    assert all(event.duration_ms is not None for event in nodes)

    serialized = json.dumps([asdict(event) for event in collector.events])
    for forbidden in ("PRIVATE_COMMAND", "PRIVATE_TYPED_VALUE", "screenshot", "coordinates", "prompt"):
        assert forbidden not in serialized


def test_jsonl_sink_writes_valid_one_event_per_line(tmp_path: Path) -> None:
    collector = InMemoryTelemetryCollector()
    action = TypeAction("hidden literal")
    GraphAgent(
        _Computer(), _Decision(action), telemetry_collector=collector,
    ).run("hidden request", dry_run=True)
    path = tmp_path / "telemetry.jsonl"
    sink = JsonlTelemetrySink(path)
    for event in collector.events:
        sink.emit(event)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(collector.events)
    records = [json.loads(line) for line in lines]
    assert records[0]["event_type"] == "run_started"
    assert records[-1]["event_type"] == "run_completed"
    assert all(record["run_id"] == records[0]["run_id"] for record in records)
    contents = "\n".join(lines)
    assert "hidden request" not in contents
    assert "hidden literal" not in contents
