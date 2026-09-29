from __future__ import annotations

from dataclasses import dataclass

from fastapi.testclient import TestClient

import api
from agent.graph import GraphAgentResult, GraphNodeDiagnostic
from agent.loop import AgentResult
from computer.actions import TypeAction
from computer.results import ActionResult


def graph_result(
    *,
    success: bool = True,
    stop_reason: str = "finished",
    message: str = "private provider prompt, typed literal, screenshot, coordinates, API key",
    diagnostics: tuple[GraphNodeDiagnostic, ...] = (),
    history: tuple[ActionResult, ...] = (),
) -> GraphAgentResult:
    result = AgentResult(
        success=success,
        stop_reason=stop_reason,  # type: ignore[arg-type]
        message=message,
        steps=1,
        history=history,
    )
    return GraphAgentResult(
        result,
        diagnostics,
        transition_reason="finished",
        replan_count=1,
        max_replans=1,
        last_replan_reason="post_action_observation_incomplete",
    )


@dataclass
class FakeRuntime:
    result: GraphAgentResult | None = None
    error: Exception | None = None
    calls: list[tuple[str, bool]] | None = None

    def run(self, request: str, *, dry_run: bool = False) -> GraphAgentResult:
        if self.calls is not None:
            self.calls.append((request, dry_run))
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result


def test_health_is_local_and_does_not_construct_runtime() -> None:
    def forbidden_factory(_: api.AgentRunRequest) -> FakeRuntime:
        raise AssertionError("health must not create the runtime")

    client = TestClient(api.create_app(forbidden_factory))
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_valid_dry_run_uses_typed_response_and_existing_runtime() -> None:
    calls: list[tuple[str, bool]] = []
    runtime = FakeRuntime(graph_result(), calls=calls)
    client = TestClient(api.create_app(lambda _: runtime))

    response = client.post("/agent/run", json={"command": "open notepad"})

    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "stop_reason": "finished",
        "message": "Agent run completed.",
        "steps": 1,
        "replan_count": 1,
        "max_replans": 1,
        "last_replan_reason": "post_action_observation_incomplete",
        "graph_diagnostics": [],
    }
    assert calls == [("open notepad", True)]
    assert api.AgentRunResponse.model_validate(response.json())


def test_invalid_empty_command_returns_422_without_echoing_input() -> None:
    secret = "INVALID_COMMAND_SENTINEL"
    client = TestClient(api.create_app(lambda _: FakeRuntime(graph_result())))

    response = client.post("/agent/run", json={"command": "   "})

    assert response.status_code == 422
    assert secret not in response.text
    assert "input" not in response.text


def test_validation_errors_do_not_echo_overlong_command_or_extra_values() -> None:
    secret = "REQUEST_SECRET_SENTINEL"
    client = TestClient(api.create_app(lambda _: FakeRuntime(graph_result())))

    response = client.post("/agent/run", json={
        "command": secret + ("x" * 2000), "unexpected": secret,
    })

    assert response.status_code == 422
    assert secret not in response.text


def test_maximum_step_and_replan_limits_are_validated() -> None:
    client = TestClient(api.create_app(lambda _: FakeRuntime(graph_result())))
    assert client.post("/agent/run", json={"command": "x", "max_steps": 21}).status_code == 422
    assert client.post("/agent/run", json={"command": "x", "max_replans": 11}).status_code == 422
    assert client.post("/agent/run", json={"command": "x", "max_steps": 0}).status_code == 422
    assert client.post("/agent/run", json={"command": "x", "max_replans": -1}).status_code == 422


def test_runtime_failures_return_safe_structured_response() -> None:
    secret = "sk-test-private-runtime-detail"
    client = TestClient(api.create_app(lambda _: FakeRuntime(error=RuntimeError(secret))))

    response = client.post("/agent/run", json={"command": "safe request"})

    assert response.status_code == 200
    assert response.json()["success"] is False
    assert response.json()["stop_reason"] == "runtime_error"
    assert response.json()["message"] == "Agent runtime failed safely."
    assert secret not in response.text


def test_graph_agent_failure_result_is_mapped_to_safe_response() -> None:
    result = graph_result(
        success=False,
        stop_reason="safety_rejected",
        message="unsafe details and private request text",
        diagnostics=(GraphNodeDiagnostic(
            node="SAFETY_CHECK",
            step=1,
            transition_reason="policy_denied",
            observation_id="opaque-observation-id",
            action_kind="click",
            success=False,
            stop_reason="safety_rejected",
            replan_count=0,
            max_replans=1,
            failure_recoverable=False,
        ),),
    )
    client = TestClient(api.create_app(lambda _: FakeRuntime(result)))

    response = client.post("/agent/run", json={"command": "private request"})

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is False
    assert body["stop_reason"] == "safety_rejected"
    assert body["message"] == "Agent run stopped safely."
    assert body["graph_diagnostics"][0] == {
        "node": "SAFETY_CHECK",
        "step": 1,
        "transition_reason": "policy_denied",
        "observation_id_present": True,
        "action_kind": "click",
        "success": False,
        "stop_reason": "safety_rejected",
        "replan_count": 0,
        "max_replans": 1,
        "replan_reason": None,
        "failure_recoverable": False,
    }
    assert "private request" not in response.text


def test_configuration_failure_is_classified_without_exposing_details() -> None:
    def fail(_: api.AgentRunRequest) -> FakeRuntime:
        raise api.ConfigurationError("API key=private-secret")

    response = TestClient(api.create_app(fail)).post(
        "/agent/run", json={"command": "safe request"},
    )
    assert response.status_code == 200
    assert response.json()["stop_reason"] == "configuration_error"
    assert "private-secret" not in response.text


def test_failed_runtime_response_preserves_requested_replan_budget() -> None:
    def fail(_: api.AgentRunRequest) -> FakeRuntime:
        raise RuntimeError("private failure detail")

    response = TestClient(api.create_app(fail)).post(
        "/agent/run", json={"command": "safe request", "max_replans": 0},
    )
    assert response.status_code == 200
    assert response.json()["max_replans"] == 0


def test_response_redacts_prompts_literals_screenshots_coordinates_and_secrets() -> None:
    secret = "API_KEY_PRIVATE_SENTINEL"
    prompt = "PROVIDER_PROMPT_SENTINEL"
    literal = "TYPED_LITERAL_SENTINEL"
    screenshot = "SCREENSHOT_BASE64_SENTINEL"
    coordinates = "COORDINATES_SENTINEL"
    unsafe_diagnostic = GraphNodeDiagnostic(
        node=secret,
        step=1,
        transition_reason=prompt,
        observation_id=screenshot,
        action_kind=literal,
        success=True,
        stop_reason=secret,
        replan_count=0,
        max_replans=1,
        replan_reason=coordinates,
        failure_recoverable=True,
    )
    action_history = (ActionResult(
        success=True,
        action=TypeAction(literal),
        message=literal,
    ),)
    runtime = FakeRuntime(graph_result(
        message=f"{secret} {prompt} {literal} {screenshot} {coordinates}",
        diagnostics=(unsafe_diagnostic,),
        history=action_history,
    ))
    response = TestClient(api.create_app(lambda _: runtime)).post(
        "/agent/run", json={"command": "safe request"},
    )

    assert response.status_code == 200
    for value in (secret, prompt, literal, screenshot, coordinates):
        assert value not in response.text
    diagnostic = response.json()["graph_diagnostics"][0]
    assert diagnostic["node"] == "UNKNOWN"
    assert diagnostic["transition_reason"] == "other"
    assert diagnostic["observation_id_present"] is True
    assert diagnostic["action_kind"] is None
    assert diagnostic["replan_reason"] is None


def test_agent_runtime_factory_constructs_existing_graph_agent(monkeypatch) -> None:
    instances: dict[str, object] = {}

    class FakeCatalog:
        def __init__(self) -> None:
            instances["catalog"] = self

    class FakeDecisionMaker:
        min_confidence = 0.73

        @classmethod
        def from_environment(cls, catalog):
            instances["decision_catalog"] = catalog
            return cls()

    class FakeComputer:
        def __init__(self, options, *, app_catalog, capture_service, visual_provider):
            instances["computer_args"] = (options, app_catalog, capture_service, visual_provider)

    class FakePolicy:
        def __init__(self, catalog):
            instances["policy_catalog"] = catalog

    class FakeGraphAgent:
        def __init__(self, computer, decision_maker, *, policy, limits, max_replans):
            instances["graph_args"] = (computer, decision_maker, policy, limits, max_replans)

    monkeypatch.setattr(api, "WindowsApplicationCatalog", FakeCatalog)
    monkeypatch.setattr(api, "JevDecisionMaker", FakeDecisionMaker)
    monkeypatch.setattr(api, "WindowsComputer", FakeComputer)
    monkeypatch.setattr(api, "AutonomousActionPolicy", FakePolicy)
    monkeypatch.setattr(api, "GraphAgent", FakeGraphAgent)
    monkeypatch.setattr(api, "visual_provider_from_environment", lambda: None)
    monkeypatch.setattr(api, "_load_project_environment", lambda: None)
    monkeypatch.setenv("AGENT_MAX_STEPS", "9")
    monkeypatch.delenv("VISUAL_PROVIDER", raising=False)

    runtime = api.build_graph_agent(api.AgentRunRequest(
        command="dry-run request", max_replans=2,
    ))

    assert isinstance(runtime, FakeGraphAgent)
    computer, decision, policy, limits, max_replans = instances["graph_args"]
    assert isinstance(computer, FakeComputer)
    assert isinstance(decision, FakeDecisionMaker)
    assert isinstance(policy, FakePolicy)
    assert limits.max_steps == 9
    assert limits.confidence_threshold == 0.73
    assert max_replans == 2
    assert instances["decision_catalog"] is instances["catalog"]
    assert instances["policy_catalog"] is instances["catalog"]


def test_explicit_bounds_are_passed_to_existing_graph_runtime() -> None:
    captured: list[tuple[str, bool]] = []
    factory_requests: list[api.AgentRunRequest] = []
    result = graph_result()
    runtime = FakeRuntime(result, calls=captured)
    def factory(request: api.AgentRunRequest) -> FakeRuntime:
        factory_requests.append(request)
        return runtime

    client = TestClient(api.create_app(factory))

    response = client.post("/agent/run", json={
        "command": "bounded request", "dry_run": False,
        "max_steps": 3, "max_replans": 0,
    })

    assert response.status_code == 200
    assert captured == [("bounded request", False)]
    assert factory_requests[0].max_steps == 3
    assert factory_requests[0].max_replans == 0


def test_cli_graph_command_remains_available() -> None:
    import main

    assert callable(main.main)
