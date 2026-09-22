"""Deterministic tests: no real Windows input, process launches, or clipboard."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, dataclass, field, replace
import json
import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from computer.actions import ClickAction, FinishAction, OpenAppAction, PressKeyAction, TypeAction
from computer.applications import ApplicationCandidate, MemoryApplicationCatalog
from computer.models import Observation, Rect, UIElement
from computer import windows, windows_actions as actions
from computer.windows_actions import WindowsComputer
from safety.interfaces import SafetyDecision
from safety.policy import ALLOWED_KEYS, BasicActionPolicy

APP_ID = "app_1111111111111111"


@dataclass
class Node:
    runtime_id: tuple[int, ...]
    name: str
    control_type: str
    parent: Node | None = None
    handle: int = 100
    process_id: int = 42
    automation_id: str = ""
    visible: bool = True
    enabled: bool = True
    rectangle: Rect = Rect(0, 0, 100, 100)
    children: list[Node] = field(default_factory=list)
    element: Any = field(default_factory=lambda: Mock(CurrentIsPassword=False))

    def iter_children(self) -> Iterator[Node]:
        return iter(self.children)


@dataclass
class Rig:
    computer: WindowsComputer
    observation: Observation
    root: Node
    edit: Node
    button: Node
    focus: Any
    wrappers: dict[tuple[int, ...], Mock]
    launch: Mock
    send_key: Mock
    catalog: MemoryApplicationCatalog


@pytest.fixture
def rig(monkeypatch: pytest.MonkeyPatch) -> Rig:
    root = Node((1,), "Test window", "Window")
    edit = Node((2,), "Editor", "Edit", root)
    button = Node((3,), "Ordinary button", "Button", root)
    root.children = [edit, button]
    focus = SimpleNamespace(node=edit)
    wrappers = {node.runtime_id: Mock() for node in (root, edit, button)}
    wrappers[(2,)].iface_value.CurrentIsReadOnly = False
    monkeypatch.setattr(windows, "_foreground", lambda: root)
    monkeypatch.setattr(actions, "_foreground", lambda: root)
    monkeypatch.setattr(actions, "_focused", lambda: focus.node)
    monkeypatch.setattr(actions, "_wrapper", lambda node: wrappers[node.runtime_id])
    launch, send_key = Mock(), Mock()
    monkeypatch.setattr(actions, "_send_key", send_key)
    catalog = MemoryApplicationCatalog((
        ApplicationCandidate(APP_ID, "Test Editor", "test", launch_policy="allow", process_names=("editor.exe",)),
    ), {APP_ID: launch})
    computer = WindowsComputer(app_catalog=catalog)
    return Rig(computer, computer.observe(), root, edit, button, focus, wrappers, launch, send_key, catalog)


def test_trusted_catalog_application_launch(rig: Rig) -> None:
    result = rig.computer.execute(OpenAppAction(APP_ID))
    assert result.success
    rig.launch.assert_called_once_with()


@pytest.mark.parametrize("app_id", ["cmd", "powershell", "notepad & calc", "notepad.exe", "C:\\evil.exe", "", "app_deadbeefdeadbeef"])
def test_unsupported_application_never_launches(rig: Rig, app_id: str) -> None:
    result = rig.computer.execute(OpenAppAction(app_id))
    assert not result.success
    assert result.error == "policy_blocked"
    rig.launch.assert_not_called()


def test_model_strings_cannot_become_launch_paths_or_arguments(rig: Rig) -> None:
    for injected in ("C:\\evil.exe", "spotify.exe --delete", "cmd /c calc", "https://example.com"):
        assert not rig.computer.execute(OpenAppAction(injected)).success
    rig.launch.assert_not_called()


@pytest.mark.parametrize("keys", sorted(ALLOWED_KEYS))
def test_key_allowlist(rig: Rig, keys: tuple[str, ...]) -> None:
    result = rig.computer.execute(PressKeyAction(tuple(key.upper() for key in keys)), rig.observation)
    assert result.success
    rig.send_key.assert_called_once_with(keys)


@pytest.mark.parametrize("keys", [("alt", "f4"), ("ctrl", "s"), ("{ENTER}",), ("tab 20",), (), ("^a",)])
def test_unsupported_keys_never_send(rig: Rig, keys: tuple[str, ...]) -> None:
    assert not rig.computer.execute(PressKeyAction(keys), rig.observation).success
    rig.send_key.assert_not_called()


def test_key_translation_uses_only_internal_expressions(monkeypatch: pytest.MonkeyPatch) -> None:
    send = Mock()
    monkeypatch.setitem(sys.modules, "pywinauto.keyboard", SimpleNamespace(send_keys=send))
    actions._send_key(("shift", "tab"))
    send.assert_called_once_with("+{TAB}", pause=0)
    with pytest.raises(KeyError):
        actions._send_key(("{PAUSE 10}",))
    assert send.call_count == 1


def test_click_resolves_exact_retained_control(rig: Rig) -> None:
    result = rig.computer.execute(ClickAction("c2"), rig.observation)
    assert result.success
    rig.wrappers[(3,)].iface_invoke.Invoke.assert_called_once_with()
    rig.wrappers[(3,)].click_input.assert_not_called()
    rig.wrappers[(2,)].iface_invoke.Invoke.assert_not_called()


def test_click_editor_uses_uia_focus(rig: Rig) -> None:
    result = rig.computer.execute(ClickAction("c1"), rig.observation)
    assert result.success
    rig.edit.element.SetFocus.assert_called_once_with()
    rig.wrappers[(2,)].click_input.assert_not_called()


@pytest.mark.parametrize("kind, pattern, method", [
    ("CheckBox", "iface_toggle", "Toggle"),
    ("TabItem", "iface_selection_item", "Select"),
])
def test_semantic_click_patterns(rig: Rig, kind: str, pattern: str, method: str) -> None:
    rig.button.control_type = kind
    observation = rig.computer.observe()
    assert rig.computer.execute(ClickAction("c2"), observation).success
    getattr(getattr(rig.wrappers[(3,)], pattern), method).assert_called_once_with()


@pytest.mark.parametrize("control_id", ["c0", "c9", "", "100,200"])
def test_invalid_ids_fail_closed(rig: Rig, control_id: str) -> None:
    assert not rig.computer.execute(ClickAction(control_id), rig.observation).success
    rig.wrappers[(3,)].iface_invoke.Invoke.assert_not_called()


def test_new_observation_invalidates_old_even_with_same_ids(rig: Rig) -> None:
    old = rig.observation
    newer = rig.computer.observe()
    assert old.observation_id != newer.observation_id
    assert not rig.computer.execute(ClickAction("c2"), old).success
    rig.wrappers[(3,)].iface_invoke.Invoke.assert_not_called()


def test_foreign_executor_snapshot_is_rejected(rig: Rig) -> None:
    other = WindowsComputer()
    other.observe()
    assert not other.execute(ClickAction("c2"), rig.observation).success


def test_copied_snapshot_is_rejected(rig: Rig) -> None:
    assert not rig.computer.execute(ClickAction("c2"), replace(rig.observation)).success


def test_ui_action_requires_explicit_snapshot(rig: Rig) -> None:
    assert not rig.computer.execute(TypeAction("hello")).success


def test_snapshot_is_single_use(rig: Rig) -> None:
    assert rig.computer.execute(ClickAction("c2"), rig.observation).success
    assert not rig.computer.execute(ClickAction("c2"), rig.observation).success
    assert rig.wrappers[(3,)].iface_invoke.Invoke.call_count == 1


@pytest.mark.parametrize("attribute,value", [
    ("runtime_id", (999,)), ("process_id", 99), ("name", "Different"),
    ("automation_id", "different"), ("visible", False), ("enabled", False),
])
def test_changed_control_is_rejected(rig: Rig, attribute: str, value: object) -> None:
    setattr(rig.button, attribute, value)
    assert not rig.computer.execute(ClickAction("c2"), rig.observation).success
    rig.wrappers[(3,)].iface_invoke.Invoke.assert_not_called()


def test_reparented_control_is_rejected(rig: Rig) -> None:
    rig.button.parent = Node((99,), "Other", "Window")
    assert not rig.computer.execute(ClickAction("c2"), rig.observation).success


def test_window_change_is_rejected(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(actions, "_foreground", lambda: Node((99,), "Other", "Window"))
    assert not rig.computer.execute(ClickAction("c2"), rig.observation).success


def test_unavailable_runtime_identity_is_not_bound(rig: Rig) -> None:
    rig.button.runtime_id = ()
    observation = rig.computer.observe()
    assert not rig.computer.execute(ClickAction("c2"), observation).success


def test_control_changed_during_observation_is_not_bound(rig: Rig) -> None:
    # children() is queried after the control description has been emitted.
    def children() -> Iterator[Node]:
        rig.button.name = "A different action"
        return iter(())

    rig.button.iter_children = children  # type: ignore[method-assign]
    observation = rig.computer.observe()
    assert observation.elements[1].name == "Ordinary button"
    assert not rig.computer.execute(ClickAction("c2"), observation).success
    rig.wrappers[(3,)].iface_invoke.Invoke.assert_not_called()


def test_expired_snapshot(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    now = actions.time.monotonic()
    monkeypatch.setattr(actions.time, "monotonic", lambda: now + 121)
    assert not rig.computer.execute(ClickAction("c2"), rig.observation).success


def test_text_is_literal_native_value(rig: Rig) -> None:
    text = "Hello {ENTER} +^%() 世界\nsecond line"
    result = rig.computer.execute(TypeAction(text), rig.observation)
    assert result.success
    rig.wrappers[(2,)].iface_value.SetValue.assert_called_once_with(text)
    rig.send_key.assert_not_called()


@pytest.mark.parametrize("reason", ["readonly", "password", "not_editable", "focus_changed"])
def test_unsafe_text_target(rig: Rig, reason: str) -> None:
    if reason == "readonly":
        rig.wrappers[(2,)].iface_value.CurrentIsReadOnly = True
    elif reason == "password":
        rig.edit.element.CurrentIsPassword = True
    elif reason == "not_editable":
        rig.focus.node = rig.button
        rig.observation = rig.computer.observe()
    else:
        rig.focus.node = rig.button
    assert not rig.computer.execute(TypeAction("hello"), rig.observation).success
    rig.wrappers[(2,)].iface_value.SetValue.assert_not_called()


def test_key_focus_drift_is_rejected(rig: Rig) -> None:
    rig.focus.node = rig.button
    assert not rig.computer.execute(PressKeyAction(("tab",)), rig.observation).success
    rig.send_key.assert_not_called()


def test_native_failure_never_falls_back_or_retries(rig: Rig) -> None:
    rig.wrappers[(3,)].iface_invoke.Invoke.side_effect = RuntimeError("private details")
    result = rig.computer.execute(ClickAction("c2"), rig.observation)
    assert not result.success
    assert result.error == "windows_operation_failed"
    assert "private details" not in result.message
    rig.wrappers[(3,)].click_input.assert_not_called()
    assert not rig.computer.execute(ClickAction("c2"), rig.observation).success
    assert rig.wrappers[(3,)].iface_invoke.Invoke.call_count == 1


def test_finish_is_typed_and_has_no_windows_effect(monkeypatch: pytest.MonkeyPatch) -> None:
    foreground = Mock(side_effect=AssertionError("must not inspect"))
    monkeypatch.setattr(actions, "_foreground", foreground)
    action = FinishAction("Done")
    result = WindowsComputer().execute(action)
    assert result.success and result.completed
    assert result.action is action
    assert result.error is None and not result.requires_confirmation
    assert json.loads(json.dumps(asdict(result)))["action"]["kind"] == "finish"
    foreground.assert_not_called()


def test_confirmation_policy_never_executes(rig: Rig) -> None:
    rig.computer.policy = Mock(validate=Mock(return_value=SafetyDecision("confirm", "Needs consent")))
    result = rig.computer.execute(ClickAction("c2"), rig.observation)
    assert not result.success and result.requires_confirmation
    rig.wrappers[(3,)].iface_invoke.Invoke.assert_not_called()


def test_custom_policy_cannot_relax_allowlist(rig: Rig) -> None:
    rig.computer.policy = Mock(validate=Mock(return_value=SafetyDecision("allow", "Anything")))
    assert not rig.computer.execute(OpenAppAction("cmd")).success
    rig.launch.assert_not_called()


def test_safety_policy_requires_known_enabled_target() -> None:
    observation = Observation("", "", (UIElement("c1", "Button", "Button"),))
    assert BasicActionPolicy().validate(ClickAction("c1"), observation).disposition == "deny"
    assert BasicActionPolicy().validate(TypeAction("\x00"), observation).disposition == "deny"
