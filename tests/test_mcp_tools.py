from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from uuid import uuid4

import pytest

from computer.actions import (
    Action, ClickAction, FinishAction, OpenAppAction, PressKeyAction,
    TypeAction, VisualClickAction,
)
from computer.applications import ApplicationCandidate, MemoryApplicationCatalog
from computer.mcp_tools import ComputerToolService
from computer.models import Observation, Rect, UIElement
from computer.results import ActionResult
from safety.interfaces import SafetyDecision
from safety.policy import BasicActionPolicy


def make_observation(
    observation_id: str | None = None,
    *,
    name: str = "Save",
    observed_text: str | None = None,
    elements: tuple[UIElement, ...] | None = None,
) -> Observation:
    return Observation(
        app_name="notepad.exe",
        window_title="Notes - private",
        elements=elements if elements is not None else (
            UIElement(
                id="c1", name=name, control_type="Button", rectangle=Rect(10, 20, 80, 45),
                enabled=True, visible=True, focused=False,
            ),
            UIElement(
                id="c2", name="Editable area", control_type="Edit", observed_text=observed_text,
                is_password=False, enabled=True, visible=True, focused=True,
            ),
        ),
        truncated=False,
        inspection_errors=0,
        observation_id=observation_id or uuid4().hex,
        application_id="app_0123456789abcdef",
    )


@dataclass
class FakeComputer:
    observations: list[Observation]
    execute_result: ActionResult | None = None
    execute_error: Exception | None = None
    execute_calls: list[tuple[Action, Observation | None]] = field(default_factory=list)
    observe_calls: int = 0

    def observe(self) -> Observation:
        self.observe_calls += 1
        if self.observations:
            return self.observations.pop(0)
        return Observation(app_name="", window_title="", error="unavailable")

    def execute(self, action: Action, observation: Observation | None = None) -> ActionResult:
        self.execute_calls.append((action, observation))
        if self.execute_error:
            raise self.execute_error
        assert self.execute_result is not None
        return self.execute_result


@dataclass
class FakePolicy:
    disposition: str = "allow"
    calls: list[tuple[object, Observation | None]] = field(default_factory=list)

    def validate(self, action: Action, observation: Observation | None) -> SafetyDecision:
        self.calls.append((action, observation))
        return SafetyDecision(self.disposition, "test-only reason")  # type: ignore[arg-type]


def service_for(
    observations: list[Observation] | None = None,
    *,
    policy: FakePolicy | None = None,
    catalog: MemoryApplicationCatalog | None = None,
    execute_result: ActionResult | None = None,
    execute_error: Exception | None = None,
) -> tuple[ComputerToolService, FakeComputer, FakePolicy]:
    default_action = ClickAction("c1")
    computer = FakeComputer(
        observations or [make_observation(), make_observation()],
        execute_result=execute_result or ActionResult(True, default_action, "private result text"),
        execute_error=execute_error,
    )
    selected_policy = policy or FakePolicy()
    selected_catalog = catalog or MemoryApplicationCatalog(())
    return ComputerToolService(computer, selected_policy, selected_catalog), computer, selected_policy


def test_observe_calls_existing_observer_and_returns_bounded_redacted_summary(monkeypatch) -> None:
    secret = "MCP_SECRET_SENTINEL"
    monkeypatch.setenv("EXAMPLE_API_KEY", secret)
    observation = make_observation(
        name=f"Save {secret}",
        observed_text="TYPED_LITERAL_SENTINEL",
        elements=(
            UIElement(
                id="c1", name=f"Save {secret}", control_type="Button",
                rectangle=Rect(10, 20, 80, 45), enabled=True, visible=True,
            ),
            UIElement(
                id="c2", name="Password", control_type="Edit", is_password=True,
                observed_text="PASSWORD_SENTINEL", visible=True,
            ),
        ),
    )
    service, computer, _ = service_for([observation])

    summary = service.observe()
    serialized = summary.model_dump_json()

    assert computer.observe_calls == 1
    assert summary.observation_id == observation.observation_id
    assert summary.trusted_application_id == "app_0123456789abcdef"
    assert len(summary.elements) == 1
    assert summary.elements[0].name == "Save [REDACTED]"
    for hidden in (secret, "TYPED_LITERAL_SENTINEL", "PASSWORD_SENTINEL", "rectangle", "screenshot"):
        assert hidden not in serialized


def test_open_app_maps_through_catalog_and_existing_safety_before_execution() -> None:
    candidate = ApplicationCandidate(
        "app_0123456789abcdef", "Notepad", "test", launch_policy="allow",
    )
    catalog = MemoryApplicationCatalog((candidate,), {candidate.id: lambda: None})
    fresh = make_observation("1" * 32)
    service, computer, policy = service_for(
        [fresh], catalog=catalog,
        execute_result=ActionResult(True, OpenAppAction(candidate.id), "private result text"),
    )

    result = service.open_app("notepad")

    assert result.success is True
    assert result.action_kind == "open_app"
    assert len(policy.calls) == 1
    action, _ = computer.execute_calls[0]
    assert isinstance(action, OpenAppAction)
    assert action.app_id == candidate.id
    assert result.fresh_observation_id == fresh.observation_id


def test_safety_rejection_prevents_executor_call() -> None:
    policy = FakePolicy(disposition="deny")
    service, computer, _ = service_for(policy=policy)
    observation = service.observe()

    result = service.click(observation.observation_id or "", "c1")

    assert result.error_category == "safety_rejected"
    assert computer.execute_calls == []
    assert len(policy.calls) == 1


def test_confirmation_required_is_not_silently_approved() -> None:
    service, computer, _ = service_for(policy=FakePolicy(disposition="confirm"))
    observation = service.observe()

    result = service.press_key(observation.observation_id or "", "enter")

    assert result.error_category == "confirmation_required"
    assert result.requires_confirmation is True
    assert computer.execute_calls == []


def test_unsupported_key_combination_is_denied_by_existing_key_policy() -> None:
    first, fresh = make_observation(), make_observation()
    computer = FakeComputer(
        [first, fresh], execute_result=ActionResult(
            True, PressKeyAction(("alt", "f4")), "private result",
        ),
    )
    policy = BasicActionPolicy(MemoryApplicationCatalog(()))
    service = ComputerToolService(computer, policy, MemoryApplicationCatalog(()))
    snapshot = service.observe()

    # The MCP tool schema itself only lists allowlisted combinations; this
    # direct boundary test proves an unsupported caller still fails closed.
    result = service.perform_action(
        PressKeyAction(("alt", "f4")), observation_id=snapshot.observation_id,
    )

    assert result.error_category == "safety_rejected"
    assert computer.execute_calls == []


def test_executor_success_returns_safe_result_and_fresh_snapshot_id() -> None:
    first, fresh = make_observation(), make_observation()
    action = ClickAction("c1")
    service, computer, policy = service_for(
        [first, fresh], execute_result=ActionResult(True, action, "private text"),
    )
    summary = service.observe()

    result = service.click(summary.observation_id or "", "c1")

    assert result.model_dump() == {
        "success": True,
        "action_kind": "click",
        "error_category": None,
        "requires_confirmation": False,
        "fresh_observation_id": fresh.observation_id,
    }
    assert computer.execute_calls == [(action, first)]
    assert policy.calls == [(action, first)]
    assert computer.observe_calls == 2


def test_stale_or_invalid_snapshot_binding_is_rejected() -> None:
    first, latest = make_observation(), make_observation()
    service, computer, policy = service_for([first, latest])
    service.observe()
    latest_summary = service.observe()

    result = service.click(first.observation_id, "c1")

    assert latest_summary.observation_id == latest.observation_id
    assert result.error_category == "invalid_snapshot_binding"
    assert computer.execute_calls == []
    assert policy.calls == []


def test_unsupported_existing_action_is_rejected_before_safety_or_executor() -> None:
    service, computer, policy = service_for()

    result = service.perform_action(FinishAction("done"))

    assert result.action_kind == "unsupported"
    assert result.error_category == "unsupported_action"
    assert policy.calls == []
    assert computer.execute_calls == []


def test_open_app_cannot_map_arbitrary_shell_to_trusted_catalog() -> None:
    restricted = ApplicationCandidate(
        "app_aaaaaaaaaaaaaaaa", "PowerShell", "test", launch_policy="deny",
    )
    catalog = MemoryApplicationCatalog((restricted,), {restricted.id: lambda: pytest.fail("launch")})
    service, computer, policy = service_for(catalog=catalog)

    result = service.open_app("PowerShell")

    assert result.error_category == "application_not_found"
    assert computer.execute_calls == []
    assert policy.calls == []


def test_ambiguous_application_match_is_rejected_before_policy_or_executor() -> None:
    candidates = (
        ApplicationCandidate("app_aaaaaaaaaaaaaaaa", "Notes", "test", launch_policy="allow"),
        ApplicationCandidate("app_bbbbbbbbbbbbbbbb", "Notes", "test", launch_policy="allow"),
    )
    catalog = MemoryApplicationCatalog(candidates)
    service, computer, policy = service_for(catalog=catalog)

    result = service.open_app("Notes")

    assert result.error_category == "ambiguous_application"
    assert policy.calls == []
    assert computer.execute_calls == []


def test_type_text_uses_literal_action_and_does_not_return_literal() -> None:
    literal = "TEXT_SENTINEL_WITH_[key]"
    first, fresh = make_observation(), make_observation()
    action = TypeAction(literal)
    service, computer, _ = service_for(
        [first, fresh], execute_result=ActionResult(True, action, literal),
    )
    snapshot = service.observe()

    result = service.type_text(snapshot.observation_id or "", literal)

    assert computer.execute_calls[0][0] == action
    assert literal not in result.model_dump_json()


def test_failed_execution_is_not_retried_and_observes_once_afterward() -> None:
    first, fresh = make_observation(), make_observation()
    action = ClickAction("c1")
    failure = ActionResult(False, action, "private error detail", error="windows_operation_failed")
    service, computer, _ = service_for([first, fresh], execute_result=failure)
    snapshot = service.observe()

    result = service.click(snapshot.observation_id or "", "c1")

    assert result.success is False
    assert result.error_category == "executor_failure"
    assert len(computer.execute_calls) == 1
    assert computer.observe_calls == 2


def test_mcp_server_exposes_tools_over_in_memory_sdk_client() -> None:
    pytest.importorskip("mcp")
    from mcp import Client
    from mcp_server import create_mcp_server

    observation = make_observation()
    service, computer, _ = service_for([observation])
    server = create_mcp_server(service)

    async def exercise() -> tuple[set[str], dict[str, object]]:
        async with Client(server, raise_exceptions=True) as client:
            tools = await client.list_tools()
            result = await client.call_tool("observe", {})
            assert result.is_error is False
            assert result.structured_content is not None
            return {tool.name for tool in tools.tools}, result.structured_content

    names, output = asyncio.run(exercise())

    assert names == {"observe", "open_app", "click", "type_text", "press_key"}
    assert output["observation_id"] == observation.observation_id
    assert computer.observe_calls == 1
