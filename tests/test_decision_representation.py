"""Representation regressions, not claims about live model accuracy."""

from dataclasses import replace
import json
from typing import Any
from unittest.mock import Mock

import pytest

from computer.actions import ClickAction
from computer.models import Observation, Rect, UIElement
from decision.client import APIError
from decision.context import select_controls
from decision.jev import JevDecisionMaker
import main as cli


def control(control_id: str, name: str, kind: str, **fields: Any) -> UIElement:
    return UIElement(control_id, name, kind, enabled=True, visible=True, is_password=False, **fields)


@pytest.fixture
def noisy_observation() -> Observation:
    return Observation("Example editor", "Untitled", (
        control("c10", "Archivo", "MenuItem", rectangle=Rect(0, 0, 80, 40)),
        control("c11", "Archivo", "Text", rectangle=Rect(5, 5, 60, 30)),
        control("c12", "\ue700", "Text"),
        control("c13", "Editar", "MenuItem", rectangle=Rect(80, 0, 160, 40)),
        control("c14", "", "Pane"),
        control("c15", "\U000f0001", "Text"),
        control("c16", "Ready", "Text"),
        control("c1", "Editor de texto", "Document", focused=True),
    ), observation_id="test-snapshot")


def choose(payload: dict[str, Any], option: str) -> dict[str, Any]:
    return {"model": "jev-1.13.0", "answers": {"next_action": {
        "type": "choice", "choice": option, "confidence": 0.95,
        "probabilities": {key: float(key == option) for key in payload["questions"]["next_action"]["criteria"]},
    }}}


def test_semantic_options_and_local_mapping(noisy_observation: Observation) -> None:
    client = Mock(evaluate=Mock(side_effect=lambda payload: choose(payload, "click_c1")))
    decision = JevDecisionMaker(client).decide("click the text editor", noisy_observation)
    payload = client.evaluate.call_args.args[0]
    choices = payload["questions"]["next_action"]["criteria"]
    assert choices["click_c1"] == (
        'Click "Editor de texto" [Document, focused, enabled] (c1). Focus this document/text-editing area.'
    )
    assert '"Archivo" [MenuItem' in choices["click_c10"]
    assert '"Editar" [MenuItem' in choices["click_c13"]
    assert decision.action == ClickAction("c1")
    assert payload["state"]["focused_controls"] == [{
        "id": "c1", "name": "Editor de texto", "type": "Document", "enabled": True, "focused": True,
        "editable": True, "observed_text_present": False, "observed_text_length": None,
        "observed_text_truncated": False, "matches_requested_literal": False,
    }]


@pytest.mark.parametrize("name,kind", [("Editor de texto", "Document"), ("Search input", "Edit"), ("文書", "Document")])
def test_semantics_generalize_without_name_or_app_rules(name: str, kind: str) -> None:
    observation = Observation("Unrelated app", "Different title", (control("c73", name, kind, focused=True),))
    client = Mock(evaluate=Mock(side_effect=lambda payload: choose(payload, "click_c73")))
    assert JevDecisionMaker(client).decide("click the input", observation).action == ClickAction("c73")
    description = client.evaluate.call_args.args[0]["questions"]["next_action"]["criteria"]["click_c73"]
    assert name in description and kind in description and "focused" in description


def test_one_bounded_parent_is_sent_with_semantic_control() -> None:
    child = control(
        "c1", "Daft Punk", "ListItem", parent_name="Search results",
        parent_control_type="List",
    )
    observation = Observation("music.exe", "Music", (child,))
    client = Mock(evaluate=Mock(side_effect=lambda payload: choose(payload, "click_c1")))
    JevDecisionMaker(client).decide("click Daft Punk", observation)
    payload = client.evaluate.call_args.args[0]
    assert payload["state"]["controls"][0]["parent"] == {
        "name": "Search results", "type": "List",
    }
    assert "Parent: \"Search results\" [List]" in payload["questions"]["next_action"]["criteria"]["click_c1"]
    assert "grandparent" not in json.dumps(payload).casefold()


def test_filtering_counts_and_raw_observation_preserved(noisy_observation: Observation) -> None:
    original = noisy_observation.elements
    selection = select_controls(noisy_observation)
    assert [item.id for item in selection.controls] == ["c1", "c10", "c13", "c16"]
    assert selection.presentation_omitted == 4
    assert selection.budget_omitted == 0
    assert noisy_observation.elements is original
    assert len(noisy_observation.elements) == 8


def test_presentation_omissions_do_not_signal_missing_task_evidence(noisy_observation: Observation) -> None:
    maker = JevDecisionMaker(Mock())
    payload, candidates, stats = maker._prepare("click the text editor", noisy_observation, ())
    assert not payload["state"]["observation_incomplete"]
    assert stats == {
        "observed_controls": 8, "eligible_controls": 8, "selected_controls": 4,
        "presentation_omitted": 4, "privacy_omitted": 0, "budget_omitted": 0,
        "eligible_click_options": 3, "selected_click_options": 3, "total_options": 11,
        "selected_visual_click_options": 0,
        "trusted_catalog_candidate_count": 0, "trusted_catalog_match_kind": "no_match",
    }
    assert set(key for key in candidates if key.startswith("click_")) == {"click_c1", "click_c10", "click_c13"}


def test_unknown_ancestry_deprioritizes_instead_of_deleting_duplicate_text() -> None:
    observation = Observation("", "", (
        control("c1", "Search", "Text"), control("c2", "Search", "Button"),
        control("c3", "Unique status", "Text"),
    ))
    assert [item.id for item in select_controls(observation).controls] == ["c2", "c3", "c1"]


def test_uncontained_duplicate_label_is_retained() -> None:
    observation = Observation("", "", (
        control("c1", "Search", "Text", rectangle=Rect(500, 500, 600, 600)),
        control("c2", "Search", "Button", rectangle=Rect(0, 0, 100, 40)),
    ))
    assert {item.id for item in select_controls(observation).controls} == {"c1", "c2"}


@pytest.mark.parametrize("glyph", ["\ue700", "\U000f0001", "\U00100001", " \ue700\u200d \uf123 "])
def test_icon_font_text_removed_but_actionable_icons_retained(glyph: str) -> None:
    observation = Observation("", "", (control("c1", glyph, "Text"), control("c2", glyph, "Button")))
    assert [item.id for item in select_controls(observation).controls] == ["c2"]


def test_mixed_text_and_focused_text_retained() -> None:
    observation = Observation("", "", (
        control("c1", "Search \ue700", "Text"), control("c2", "\ue700", "Text", focused=True),
    ))
    assert [item.id for item in select_controls(observation).controls] == ["c2", "c1"]


@pytest.mark.parametrize("kind", [
    "Button", "MenuItem", "Document", "Edit", "Hyperlink", "ListItem", "TabItem",
    "CheckBox", "RadioButton", "ComboBox", "Slider", "Spinner", "TreeItem",
])
def test_interactive_roles_survive_filtering(kind: str) -> None:
    observation = Observation("", "", (control("c1", "", kind),))
    assert select_controls(observation).controls == observation.elements


def test_duplicate_actionable_controls_are_not_merged() -> None:
    observation = Observation("", "", (control("c1", "Open", "Button"), control("c2", "Open", "Button")))
    assert len(select_controls(observation).controls) == 2


def test_unknown_and_identifiable_controls_remain_available() -> None:
    observation = Observation("", "", (
        control("c1", "Custom widget", "Custom"), control("c2", "", "Pane", automation_id="widget"),
    ))
    assert select_controls(observation).controls == observation.elements


def test_focused_and_actionable_controls_rank_before_budget() -> None:
    labels = tuple(control(f"c{i}", f"Status {i}", "Text") for i in range(1, 100))
    focused = control("c100", "Input", "Edit", focused=True)
    button = control("c101", "Accept", "Button")
    selection = select_controls(Observation("", "", (*labels, button, focused)), limit=2)
    assert selection.controls == (focused, button)
    assert selection.budget_omitted == 99


def test_finish_stop_and_focus_semantics(noisy_observation: Observation) -> None:
    payload, _, _ = JevDecisionMaker(Mock())._prepare("click the text editor", noisy_observation, ())
    question = payload["questions"]["next_action"]
    assert "ALREADY been completed" in question["criteria"]["finish"]
    assert "not evidence that a requested click was performed" in question["criteria"]["finish"]
    assert "NO safe supported offered action can progress" in question["criteria"]["stop"]
    assert "existing focus alone does not satisfy a requested click" in question["instructions"]


def test_no_raw_uia_objects_or_geometry_in_payload(noisy_observation: Observation) -> None:
    raw = Mock()
    raw.__repr__ = Mock(side_effect=AssertionError("must not serialize a native object"))
    injected = replace(noisy_observation.elements[-1], rectangle=raw, automation_id=raw)
    observation = replace(noisy_observation, elements=(injected,))
    payload, _, _ = JevDecisionMaker(Mock())._prepare("click the text editor", observation, ())
    serialized = json.dumps(payload)
    assert "rectangle" not in serialized and "automation_id" not in serialized
    assert raw not in payload["state"]["controls"][0].values()


def test_debug_payload_matches_sent_request_and_cannot_mutate_it(noisy_observation: Observation) -> None:
    client = Mock(evaluate=Mock(side_effect=lambda payload: choose(payload, "click_c1")))
    debug: list[dict[str, Any]] = []

    def capture(data: dict[str, Any]) -> None:
        debug.append(json.loads(json.dumps(data)))
        data["payload"]["questions"]["next_action"]["criteria"].clear()

    JevDecisionMaker(client).decide("click the text editor", noisy_observation, debug_context=capture)
    assert debug[0]["payload"] == client.evaluate.call_args.args[0]
    assert debug[0]["selection"]["selected_controls"] == 4


def test_debug_is_sanitized_even_on_api_failure(noisy_observation: Observation, monkeypatch: pytest.MonkeyPatch) -> None:
    key = "test-secret-api-key-not-real"
    monkeypatch.setenv("TYPESAFE_API_KEY", key)
    secret_control = replace(noisy_observation.elements[-1], name=f"Editor {key}")
    observation = replace(noisy_observation, window_title="password=hidden-value", elements=(secret_control,))
    client = Mock(evaluate=Mock(side_effect=APIError(key)))
    debug = Mock()
    result = JevDecisionMaker(client, secrets=(key,)).decide("click editor", observation, debug_context=debug)
    output = json.dumps(debug.call_args.args[0])
    assert key not in output and "hidden-value" not in output and "Authorization" not in output
    assert "[REDACTED]" in output
    assert key not in result.message


def test_cli_debug_to_stderr_result_to_stdout_and_no_execution(
    noisy_observation: Observation, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    client = Mock(evaluate=Mock(side_effect=lambda payload: choose(payload, "click_c1")))
    maker = JevDecisionMaker(client)
    monkeypatch.setattr(cli.JevDecisionMaker, "from_environment", lambda _catalog=None: maker)
    monkeypatch.setattr(cli, "WindowsObserver", Mock(return_value=Mock(observe=Mock(return_value=noisy_observation))))
    executor = Mock(side_effect=AssertionError("must not execute"))
    monkeypatch.setattr(cli, "WindowsComputer", executor)
    assert cli.main(["decide", "click the text editor", "--delay", "0", "--debug-context"]) == 0
    output = capsys.readouterr()
    assert json.loads(output.err)["payload"] == client.evaluate.call_args.args[0]
    assert json.loads(output.out)["action"]["target_id"] == "c1"
    executor.assert_not_called()
