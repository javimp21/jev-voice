"""Jev tests use local responses only; never require a TypeSafe account."""

from dataclasses import asdict, replace
import json
from typing import Any
from unittest.mock import Mock

import pytest

from computer.actions import ClickAction, FinishAction, OpenAppAction, PressKeyAction, TypeAction
from computer.applications import ApplicationCandidate, MemoryApplicationCatalog
from computer.models import Observation, Rect, UIElement
from computer.results import ActionResult
from decision.client import APIError, ConfigurationError, InvalidResponse, JevSettings, TypeSafeHTTPClient
import decision.client as transport
from decision.context import Redactor, compact_state, literal_texts, phase2_type_literal
from decision.jev import JevDecisionMaker
from decision.models import DecisionResult
from decision.target_resolution import CandidateEvidence, TargetSpec, resolve_target
from safety.policy import ALLOWED_KEYS
import main as cli

NOTEPAD_ID = "app_1111111111111111"
TEST_CATALOG = MemoryApplicationCatalog((
    ApplicationCandidate(NOTEPAD_ID, "Notepad", "test", launch_policy="allow", process_names=("notepad.exe",)),
    ApplicationCandidate("app_2222222222222222", "Calculator", "test", launch_policy="allow"),
    ApplicationCandidate("app_3333333333333333", "File Explorer", "test", launch_policy="allow"),
))


@pytest.fixture
def observation() -> Observation:
    return Observation("Notepad", "Untitled", (
        UIElement("c1", "Text editor", "Edit", enabled=True, visible=True, focused=True, is_password=False),
        UIElement("c2", "Search", "Button", enabled=True, visible=True, is_password=False),
        UIElement("c3", "Unavailable", "Button", enabled=False, visible=True),
        UIElement("c4", "Hidden", "Button", enabled=True, visible=False),
    ), process_id=123, observation_id="local-snapshot")


def reply(payload: dict[str, Any], choice: str, confidence: object = 0.95) -> dict[str, Any]:
    criteria = payload["questions"]["next_action"]["criteria"]
    return {"model": "jev-1.13.0", "answers": {"next_action": {
        "type": "choice", "choice": choice, "confidence": confidence,
        "probabilities": {key: float(key == choice) for key in criteria},
    }}, "usage": {"input_tokens": 100, "output_tokens": 10}}


def maker(choice: str, confidence: object = 0.95) -> tuple[JevDecisionMaker, Mock]:
    client = Mock()
    client.evaluate.side_effect = lambda payload: reply(payload, choice, confidence)
    return JevDecisionMaker(client, app_catalog=TEST_CATALOG), client


def _target_choice_resolution():
    target = TargetSpec("Pablo García", ("Work",), "conversation")
    candidates = (
        CandidateEvidence(
            "c1", "Pablo García", (), "conversation", True, True, True,
            "UIA", "snapshot", "ListItem",
        ),
        CandidateEvidence(
            "c2", "Pablo García", ("Work contact",), "contact", True, True, True,
            "UIA", "snapshot", "ListItem",
        ),
        CandidateEvidence(
            "c3", "Pablo García", (), "contact", True, True, True,
            "UIA", "snapshot", "ListItem",
        ),
    )
    return target, resolve_target(
        target, candidates, expected_snapshot_id="snapshot", frontier_mode=True,
    )


def test_target_activation_jev_receives_only_the_bounded_semantic_frontier() -> None:
    target, resolution = _target_choice_resolution()
    decision, client = maker("target_2")

    result = decision.decide_target_activation(target, resolution)

    assert result.status == "ready" and result.candidate_id == "c2"
    payload = client.evaluate.call_args.args[0]
    criteria = payload["questions"]["next_action"]["criteria"]
    assert set(criteria) == {"target_1", "target_2", "stop"}
    assert {criteria[key]["candidate_id"] for key in ("target_1", "target_2")} == {"c1", "c2"}
    assert criteria["target_1"]["target_semantic_evidence"] == "conversation"
    assert criteria["target_1"]["presentation_role"] == "list_item"
    serialized = json.dumps(payload).casefold()
    for forbidden in ("screenshot", "coordinates", "hwnd", "process_id", "automation_id"):
        assert forbidden not in serialized
    assert "work contact" in serialized


def test_target_activation_jev_rejects_unoffered_choice_and_low_confidence() -> None:
    target, resolution = _target_choice_resolution()
    decision, client = maker("unoffered")

    bad = decision.decide_target_activation(target, resolution)
    assert bad.status == "error" and bad.error == "invalid_response"
    assert bad.diagnostic_reason == "invalid_response" and bad.provider_called is True

    low, _ = maker("target_1", confidence=.79)
    stopped = low.decide_target_activation(target, resolution)
    assert stopped.status == "stop" and stopped.candidate_id is None
    assert stopped.diagnostic_reason == "model_confidence_below_threshold"
    assert stopped.provider_called is True
    assert stopped.proposed_candidate_id == "c1"

    explicit_stop, _ = maker("stop", confidence=.95)
    stopped = explicit_stop.decide_target_activation(target, resolution)
    assert stopped.status == "stop" and stopped.diagnostic_reason == "model_stop"
    assert stopped.provider_called is True
    assert stopped.proposed_candidate_id is None


def test_target_activation_jev_provider_failure_is_bounded() -> None:
    target, resolution = _target_choice_resolution()
    decision, client = maker("target_1")
    client.evaluate.side_effect = APIError(
        "private provider detail", category="timeout", http_status=408,
    )

    result = decision.decide_target_activation(target, resolution)

    assert result.status == "error" and result.error == "timeout"
    assert result.diagnostic_reason == "provider_error" and result.provider_called is True
    assert "private provider detail" not in repr(result)


def test_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("JEV_MIN_CONFIDENCE", raising=False)
    with pytest.raises(ConfigurationError, match="TYPESAFE_API_KEY"):
        JevDecisionMaker.from_environment()


def test_default_jev_min_confidence_remains_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-only-key")
    monkeypatch.delenv("JEV_MIN_CONFIDENCE", raising=False)

    assert JevSettings.from_environment().min_confidence == 0.8


@pytest.mark.parametrize("value", ["nan", "inf", "0", "-1", "1.1", "not a number"])
def test_invalid_threshold(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key-only")
    monkeypatch.setenv("JEV_MIN_CONFIDENCE", value)
    with pytest.raises(ConfigurationError, match="CONFIDENCE"):
        JevSettings.from_environment()


def test_settings_and_key_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "private-test-credential")
    monkeypatch.setenv("JEV_MIN_CONFIDENCE", "0.9")
    monkeypatch.setenv("JEV_MODEL", "jev-1.13.0")
    settings = JevSettings.from_environment()
    assert settings.min_confidence == 0.9
    assert settings.model == "jev-1.13.0"
    assert "private-test-credential" not in repr(settings)


def test_compact_state_excludes_windows_metadata_and_raw_errors(observation: Observation) -> None:
    control = replace(observation.elements[0], rectangle=Rect(1, 2, 3, 4), automation_id="internal-id")
    history = [ActionResult(False, ClickAction("c77"), "secret error detail", error="unsafe_target")] * 8
    state = compact_state("click Search", observation, [control], history, Redactor())
    assert len(state["history"]) == 5
    assert state["history"][0]["error"] == "unsafe_target"
    assert set(state["controls"][0]) == {
        "id", "name", "type", "enabled", "focused", "editable",
        "observed_text_present", "observed_text_length", "observed_text_truncated",
        "matches_requested_literal",
    }
    serialized = json.dumps(state)
    for forbidden in ("rectangle", "process_id", "automation_id", "internal-id", "secret error detail", "local-snapshot"):
        assert forbidden not in serialized


def test_type_success_and_fresh_matching_value_becomes_completion_evidence(
    observation: Observation,
) -> None:
    literal = "Hello from Jev"
    current = replace(
        observation,
        elements=(replace(observation.elements[0], observed_text=literal), *observation.elements[1:]),
    )
    history = (ActionResult(True, TypeAction(literal), "typed"),)

    state = compact_state(f"write '{literal}'", current, current.elements, history, Redactor())

    assert state["completion_evidence"]["type"] == {
        "previous_type_action_succeeded": True,
        "fresh_focused_editable_observed": True,
        "observed_value_matches_requested_literal": True,
    }
    assert state["focused_controls"][0]["matches_requested_literal"] is True
    assert literal not in json.dumps(state["focused_controls"])


def test_text_pattern_terminal_line_break_still_matches_literal(observation: Observation) -> None:
    literal = "Hello from Jev"
    current = replace(
        observation,
        elements=(replace(observation.elements[0], observed_text=literal + "\r\n"), *observation.elements[1:]),
    )
    history = (ActionResult(True, TypeAction(literal), "typed"),)

    state = compact_state(f"write '{literal}'", current, current.elements, history, Redactor())

    assert state["completion_evidence"]["type"]["observed_value_matches_requested_literal"] is True


def test_action_history_does_not_fabricate_observed_type_state(observation: Observation) -> None:
    literal = "Hello from Jev"
    history = (ActionResult(True, TypeAction(literal), "typed"),)

    state = compact_state(f"write '{literal}'", observation, observation.elements, history, Redactor())

    assert state["completion_evidence"]["type"]["previous_type_action_succeeded"] is True
    assert state["completion_evidence"]["type"]["fresh_focused_editable_observed"] is False
    assert state["completion_evidence"]["type"]["observed_value_matches_requested_literal"] is False


def test_open_app_success_and_matching_foreground_becomes_evidence(observation: Observation) -> None:
    current = replace(observation, app_name="Notepad.exe", window_title="Untitled", application_id=NOTEPAD_ID)
    history = (ActionResult(True, OpenAppAction(NOTEPAD_ID), "opened"),)

    state = compact_state("Open Notepad", current, current.elements, history, Redactor())

    assert state["completion_evidence"]["open_app"] == {
        "previous_open_action_succeeded": True,
        "requested_app_id": NOTEPAD_ID,
        "foreground_matches_requested_app": True,
    }


def test_credential_filtering(observation: Observation, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTHER_API_KEY", "test-private-key")
    controls = [replace(observation.elements[1], name="test-private-key")]
    state = compact_state("click Search", replace(observation, window_title="password=hunter2"),
                          controls, (), Redactor())
    assert "test-private-key" not in json.dumps(state)
    assert "hunter2" not in json.dumps(state)


def test_sensitive_request_never_sent(observation: Observation) -> None:
    client = Mock()
    decision = JevDecisionMaker(client, secrets=("private-credential",))
    result = decision.decide('type "private-credential"', observation)
    assert result.error == "invalid_input"
    client.evaluate.assert_not_called()
    assert "private-credential" not in json.dumps(asdict(result))


def test_password_controls_never_sent(observation: Observation) -> None:
    decision, client = maker("stop")
    password = replace(observation.elements[0], name="secret-value", is_password=True)
    decision.decide("click Search", replace(observation, elements=(password, observation.elements[1])))
    payload = client.evaluate.call_args.args[0]
    assert "secret-value" not in json.dumps(payload)
    assert "click_c1" not in payload["questions"]["next_action"]["criteria"]


def test_unknown_password_state_is_omitted(observation: Observation) -> None:
    decision, client = maker("stop")
    control = replace(observation.elements[0], name="unknown-sensitive-content", is_password=None)
    decision.decide("click editor", replace(observation, elements=(control,)))
    assert "unknown-sensitive-content" not in json.dumps(client.evaluate.call_args.args[0])


def test_response_cannot_override_literal_payload(observation: Observation) -> None:
    def malicious(payload: dict[str, Any]) -> object:
        data = reply(payload, "type_1")
        data["answers"]["next_action"]["text"] = "invented prose"
        return data

    decision = JevDecisionMaker(Mock(evaluate=Mock(side_effect=malicious)))
    assert decision.decide("write hello world", observation).action == TypeAction("hello world")


@pytest.mark.parametrize("user_request,choice,action", [
    ("click Search", "click_c2", ClickAction("c2")),
    ("open Notepad", f"open_{NOTEPAD_ID}", OpenAppAction(NOTEPAD_ID)),
    ("press Tab", "key_tab", PressKeyAction(("tab",))),
    ("write hello world", "type_1", TypeAction("hello world")),
    ("done", "finish", FinishAction("Task complete according to the current observation and history.")),
])
def test_action_mapping(observation: Observation, user_request: str, choice: str, action: object) -> None:
    decision, _ = maker(choice)
    result = decision.decide(user_request, observation)
    assert result.status == "ready"
    assert result.action == action
    assert result.confidence == 0.95
    assert result.observation_id == "local-snapshot"
    assert json.loads(json.dumps(asdict(result)))["model"] == "jev-1.13.0"


def test_exact_allowlists_and_current_ids(observation: Observation) -> None:
    decision, client = maker("stop")
    decision.decide("click Search", observation)
    criteria = client.evaluate.call_args.args[0]["questions"]["next_action"]["criteria"]
    assert not {key for key in criteria if key.startswith("open_")}
    assert {tuple(key.removeprefix("key_").split("_")) for key in criteria if key.startswith("key_")} == ALLOWED_KEYS
    assert {key for key in criteria if key.startswith("click_")} == {"click_c1", "click_c2"}


@pytest.mark.parametrize("choice", ["click_c999", "click_c3", "click_c4", "open_cmd", "key_alt_f4", "type_invented"])
def test_unoffered_choice_rejected(observation: Observation, choice: str) -> None:
    decision, _ = maker(choice)
    result = decision.decide("click Search", observation)
    assert result.action is None
    assert result.error == "invalid_response"


def test_invalid_response_has_sanitized_diagnostic(observation: Observation) -> None:
    client = Mock(evaluate=Mock(return_value={"answers": {"private": "secret-value"}}))

    result = JevDecisionMaker(client).decide("click Search", observation)

    assert result.error == "invalid_response"
    assert result.diagnostic == "missing_model_metadata"
    assert "secret-value" not in json.dumps(asdict(result))


@pytest.mark.parametrize("user_request,expected", [
    ("write hello world", ("hello world",)),
    ("Open Notepad and write hello world", ("hello world",)),
    ("search for Adele", ("Adele",)),
    ("Open Chrome and go to wikipedia.org", ("wikipedia.org",)),
    ("search for Daft Punk and then type live", ("Daft Punk", "live")),
    ('write "a short story"', ("a short story",)),
    ("type '{ENTER}+^%世界'", ("{ENTER}+^%世界",)),
    ("write hello and then press Enter", ("hello",)),
    ("write me a 500 word essay about Rome", ()),
    ("write an essay about Rome", ()),
    ("generate a poem about Rome", ()),
    ('write "unfinished', ()),
])
def test_literal_texts(user_request: str, expected: tuple[str, ...]) -> None:
    result = literal_texts(user_request)
    assert result == expected
    assert all(value in user_request for value in result)


def test_phase2_literal_is_deterministic_source_slice() -> None:
    request = "Open Spotify and play Californication by Red Hot Chili Peppers"
    assert phase2_type_literal(request) == "Californication"
    assert phase2_type_literal('type "Hello world"') == "Hello world"
    assert phase2_type_literal("Open Spotify") is None


def test_text_requires_focused_editor(observation: Observation) -> None:
    decision, client = maker("stop")
    elements = tuple(replace(control, focused=False) for control in observation.elements)
    decision.decide("write hello", replace(observation, elements=elements))
    assert not any(key.startswith("type_") for key in client.evaluate.call_args.args[0]["questions"]["next_action"]["criteria"])


def test_generation_request_has_no_text_action(observation: Observation) -> None:
    decision, client = maker("stop")
    result = decision.decide("write me a 500 word essay about Rome", observation)
    assert result.status == "needs_human" and result.action is None
    assert not any(key.startswith("type_") for key in client.evaluate.call_args.args[0]["questions"]["next_action"]["criteria"])


@pytest.mark.parametrize("confidence,expected", [(0.79, "needs_human"), (0.8, "ready"), (0, "needs_human")])
def test_confidence_gate(observation: Observation, confidence: float, expected: str) -> None:
    decision, _ = maker("click_c2", confidence)
    result = decision.decide("click Search", observation)
    assert result.status == expected
    assert (result.action is None) == (expected != "ready")
    # Probability is 1 in this fixture: it must NOT substitute for confidence.
    assert result.probabilities["click_c2"] == 1


@pytest.mark.parametrize("confidence", [None, "0.9", True, -0.1, 1.1, float("nan"), float("inf")])
def test_invalid_confidence(observation: Observation, confidence: object) -> None:
    decision, _ = maker("click_c2", confidence)
    assert decision.decide("click Search", observation).error == "invalid_response"


@pytest.mark.parametrize("variant", ["missing_answers", "wrong_type", "missing_probability", "invalid_probability", "wrong_sum", "wrong_winner", "wrong_model"])
def test_malformed_response(observation: Observation, variant: str) -> None:
    def malformed(payload: dict[str, Any]) -> object:
        data = reply(payload, "click_c2")
        answer = data["answers"]["next_action"]
        if variant == "missing_answers":
            del data["answers"]
        elif variant == "wrong_type":
            answer["type"] = "score"
        elif variant == "missing_probability":
            del answer["probabilities"]["stop"]
        elif variant == "invalid_probability":
            answer["probabilities"]["stop"] = True
        elif variant == "wrong_sum":
            answer["probabilities"]["stop"] = 0.3
        elif variant == "wrong_winner":
            answer["probabilities"]["stop"] = 1
            answer["probabilities"]["click_c2"] = 0
        else:
            data["model"] = "private error detail"
        return data

    client = Mock(evaluate=Mock(side_effect=malformed))
    assert JevDecisionMaker(client).decide("click Search", observation).error == "invalid_response"


def test_network_failure_does_not_leak_response(observation: Observation) -> None:
    client = Mock(evaluate=Mock(side_effect=APIError("private-key and private response")))
    result = JevDecisionMaker(client).decide("click Search", observation)
    assert result.action is None and result.error == "api_error"
    assert "private" not in json.dumps(asdict(result))


def test_invalid_observation_never_calls_api(observation: Observation) -> None:
    decision, client = maker("stop")
    result = decision.decide("click Search", replace(observation, elements=(observation.elements[0],) * 2))
    assert result.error == "invalid_input"
    client.evaluate.assert_not_called()


def test_control_limit(observation: Observation) -> None:
    decision, client = maker("stop")
    elements = tuple(replace(observation.elements[1], id=f"c{i}") for i in range(1, 201))
    decision.decide("click Search", replace(observation, elements=elements))
    payload = client.evaluate.call_args.args[0]
    assert len(payload["state"]["controls"]) == 80
    assert payload["state"]["observation_incomplete"]
    assert len(payload["questions"]["next_action"]["criteria"]) < 255


def test_http_endpoint_auth_and_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    connection = Mock()
    connection.getresponse.return_value.status = 200
    connection.getresponse.return_value.read.return_value = b'{"answers": {}}'
    constructor = Mock(return_value=connection)
    monkeypatch.setattr(transport, "HTTPSConnection", constructor)
    payload = {"model": "jev-latest", "state": {}, "questions": {}}
    assert TypeSafeHTTPClient("test-only-key").evaluate(payload) == {"answers": {}}
    constructor.assert_called_once_with("api.typesafe.ai", timeout=15)
    args, kwargs = connection.request.call_args
    assert args == ("POST", "/v1/systemone")
    assert kwargs["headers"]["Authorization"] == "Bearer test-only-key"
    assert json.loads(kwargs["body"]) == payload
    connection.close.assert_called_once_with()


def test_transport_timeout_is_sanitized(monkeypatch: pytest.MonkeyPatch) -> None:
    connection = Mock()
    connection.request.side_effect = TimeoutError("sensitive transport details")
    monkeypatch.setattr(transport, "HTTPSConnection", Mock(return_value=connection))
    with pytest.raises(APIError, match="timed out") as caught:
        TypeSafeHTTPClient("test-only-key").evaluate({})
    assert "sensitive" not in str(caught.value)
    connection.close.assert_called_once_with()


def test_payload_budget_fails_before_api(observation: Observation) -> None:
    decision, client = maker("stop")
    elements = tuple(replace(observation.elements[1], id=f"c{i}", name="界" * 160) for i in range(1, 81))
    result = decision.decide("click Search", replace(observation, elements=elements))
    assert result.error == "invalid_input"
    client.evaluate.assert_not_called()


@pytest.mark.parametrize("status", [301, 401, 422, 429, 529])
def test_http_failures_no_redirect_or_retry(monkeypatch: pytest.MonkeyPatch, status: int) -> None:
    connection = Mock()
    connection.getresponse.return_value.status = status
    monkeypatch.setattr(transport, "HTTPSConnection", Mock(return_value=connection))
    with pytest.raises(APIError):
        TypeSafeHTTPClient("test-only-key").evaluate({})
    connection.request.assert_called_once()
    connection.getresponse.return_value.read.assert_not_called()


@pytest.mark.parametrize("body", [b'not JSON', b'{"x":NaN}', b'{"x":1,"x":2}', b'x' * 262_145],
                         ids=["invalid_json", "nan", "duplicate_keys", "oversized"])
def test_http_malformed_json(monkeypatch: pytest.MonkeyPatch, body: bytes) -> None:
    connection = Mock()
    connection.getresponse.return_value.status = 200
    connection.getresponse.return_value.read.return_value = body
    monkeypatch.setattr(transport, "HTTPSConnection", Mock(return_value=connection))
    with pytest.raises(InvalidResponse):
        TypeSafeHTTPClient("test-only-key").evaluate({})


def test_decide_cli_never_constructs_executor(
    observation: Observation, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    decision, _ = maker("click_c2")
    monkeypatch.setattr(cli.JevDecisionMaker, "from_environment", lambda _catalog=None: decision)
    monkeypatch.setattr(cli, "WindowsObserver", Mock(return_value=Mock(observe=Mock(return_value=observation))))
    executor = Mock(side_effect=AssertionError("decide must not execute"))
    monkeypatch.setattr(cli, "WindowsComputer", executor)
    assert cli.main(["decide", "click Search", "--delay", "0"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["action"] == {"kind": "click", "target_id": "c2"}
    assert data["confidence"] == 0.95
    executor.assert_not_called()


def test_cli_configuration_error_before_observation(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> None:
    monkeypatch.setattr(cli, "__file__", str(tmp_path / "main.py"))
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("JEV_MIN_CONFIDENCE", raising=False)
    observer = Mock(side_effect=AssertionError("should validate configuration first"))
    monkeypatch.setattr(cli, "WindowsObserver", observer)
    assert cli.main(["decide", "click Search", "--delay", "0"]) == 1
    observer.assert_not_called()
