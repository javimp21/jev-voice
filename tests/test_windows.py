"""Mock the UIA boundary so observation tests never need an interactive desktop."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, replace
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock, PropertyMock

import pytest

from computer import windows
from computer.models import Observation, Rect, ScreenshotMetadata, UIElement, VisualElement
from computer.visual import ScreenshotCapture, VisualFieldValueRead, VisualObservation
from computer.windows import ObservationOptions, WindowsObserver
from computer.windows_actions import WindowsComputer
from main import main


def test_foreground_uses_uia_and_current_handle(monkeypatch: pytest.MonkeyPatch) -> None:
    info = node("Target")
    desktop = Mock()
    desktop.return_value.window.return_value.wrapper_object.return_value.element_info = info
    foreground = Mock(return_value=123)
    monkeypatch.setattr(windows.sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "win32gui", SimpleNamespace(GetForegroundWindow=foreground))
    monkeypatch.setitem(sys.modules, "pywinauto", SimpleNamespace(Desktop=desktop))
    assert windows._foreground() is info
    foreground.assert_called_once_with()
    desktop.assert_called_once_with(backend="uia")
    desktop.return_value.window.assert_called_once_with(handle=123)


def test_observation_reads_focus_and_password_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    control = node("Editor", "Edit")
    control.element = SimpleNamespace(CurrentHasKeyboardFocus=1, CurrentIsPassword=0)
    result = observe(monkeypatch, node(children=(control,)))
    assert result.elements[0].focused is True
    assert result.elements[0].is_password is False


def test_missing_foreground_does_not_inspect_desktop(monkeypatch: pytest.MonkeyPatch) -> None:
    desktop = Mock()
    monkeypatch.setattr(windows.sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "win32gui", SimpleNamespace(GetForegroundWindow=lambda: 0))
    monkeypatch.setitem(sys.modules, "pywinauto", SimpleNamespace(Desktop=desktop))
    assert WindowsObserver().observe().error
    desktop.assert_not_called()


def node(
    name: str = "", kind: str = "Pane", *, visible: bool = True,
    enabled: bool = True, automation_id: str = "", children: tuple[Mock, ...] = (),
) -> Mock:
    result = Mock()
    result.name = name
    result.control_type = kind
    result.visible = visible
    result.enabled = enabled
    result.automation_id = automation_id
    result.process_id = 42
    result.handle = 123
    result.rectangle = Rect(-10, 20, 100, 60)
    result.iter_children.side_effect = lambda: iter(children)
    return result


def observe(monkeypatch: pytest.MonkeyPatch, root: Mock, **limits: int) -> Observation:
    monkeypatch.setattr(windows, "_foreground", lambda: root)
    return WindowsObserver(ObservationOptions(**limits)).observe()


def test_serialization_and_window_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    root = node("Editor 世界", "Window", children=(
        node("Save", "Button", automation_id="save", enabled=False),
    ))
    result = observe(monkeypatch, root)
    data = json.loads(json.dumps(asdict(result)))
    assert data["window_title"] == "Editor 世界"
    assert data["process_id"] == 42
    assert data["control_type"] == "Window"
    assert data["foreground_hwnd"] == 123
    assert data["elements"] == [{
        "id": "c1", "name": "Save", "control_type": "Button",
        "automation_id": "save", "rectangle": {
            "left": -10, "top": 20, "right": 100, "bottom": 60,
        }, "enabled": False, "visible": True, "focused": None, "is_password": None,
        "observed_text": None, "observed_text_truncated": False,
        "parent_name": "Editor 世界", "parent_control_type": "Window",
        "selected": None,
    }]
    assert not result.truncated
    assert result.inspection_errors == 0


def test_value_reader_sends_only_fresh_locally_bound_field_crop(monkeypatch: pytest.MonkeyPatch) -> None:
    from PIL import Image

    bounds = Rect(0, 0, 200, 100)
    source_meta = ScreenshotMetadata("source", 77, bounds, bounds, 200, 100, 96, 96, 1, 1)
    field = VisualElement(
        "v1", "Search", "search field", Rect(10, 10, 100, 30), None, True,
        activity="active", field_label="Search", field_value=None,
        is_query_field=True, credential_risk=False,
    )
    source = Observation(
        "Spotify.exe", "Spotify", process_id=42, observation_id="source",
        application_id="app-spotify", visual_elements=(field,), screenshot=source_meta,
    )

    class CaptureService:
        def __init__(self) -> None:
            self.calls = []

        def current_window_bounds(self, handle):
            assert handle == 77
            return bounds

        def capture(self, snapshot_id, expected_handle, process_name, sensitive_regions=()):
            self.calls.append((snapshot_id, expected_handle, tuple(sensitive_regions)))
            return ScreenshotCapture(
                replace(source_meta, snapshot_id=snapshot_id),
                Image.new("RGB", (200, 100), "white"),
            )

    class Provider:
        name = "fixture"
        model = "fixture-model"

        def __init__(self) -> None:
            self.call = None

        def observe(self, screenshot, window, objective, grounding):
            self.call = (screenshot.image.size, objective, grounding, window)
            return VisualObservation(
                (), "fixture", "fixture-model", 2, directed_grounding=True,
                field_value="Californication",
            )

    capture_service = CaptureService()
    provider = Provider()
    computer = WindowsComputer(capture_service=capture_service, visual_provider=provider)
    computer._session = SimpleNamespace(
        observation=source, handle=77,
        root=SimpleNamespace(identity=SimpleNamespace(process_id=42)),
    )
    monkeypatch.setattr(computer, "_check_window", lambda _session: None)

    result = computer.read_visual_field_value(source, field.id)

    assert isinstance(result, VisualFieldValueRead)
    assert result.field_value == "Californication"
    assert result.crop_valid and result.context_stable and result.credential_safe
    assert len(capture_service.calls) == 1
    assert capture_service.calls[0][0] != source.observation_id
    assert provider.call is not None
    crop_size, objective, grounding, passed_window = provider.call
    assert crop_size == (94, 24)
    assert grounding.field_value_only is True
    assert "Search" not in objective and "Californication" not in objective
    assert passed_window is source


def test_value_reader_refuses_unbound_snapshot_and_password_risk(monkeypatch: pytest.MonkeyPatch) -> None:
    bounds = Rect(0, 0, 200, 100)
    metadata = ScreenshotMetadata("source", 77, bounds, bounds, 200, 100, 96, 96, 1, 1)
    field = VisualElement(
        "v1", "Search", "search field", Rect(10, 10, 100, 30), None, True,
        activity="active", is_query_field=True, credential_risk=False,
    )
    source = Observation(
        "Spotify.exe", "Spotify", process_id=42, observation_id="source",
        application_id="app-spotify", visual_elements=(field,), screenshot=metadata,
    )
    capture_service = Mock()
    capture_service.current_window_bounds.return_value = bounds
    provider = Mock()
    computer = WindowsComputer(capture_service=capture_service, visual_provider=provider)
    computer._session = SimpleNamespace(
        observation=replace(source), handle=77,
        root=SimpleNamespace(identity=SimpleNamespace(process_id=42)),
    )

    unbound = computer.read_visual_field_value(source, field.id)
    assert not unbound.crop_valid and not unbound.context_stable
    capture_service.capture.assert_not_called()

    password = replace(source, elements=(UIElement(
        "c1", "Password", "Edit", rectangle=Rect(10, 10, 100, 30),
        visible=True, is_password=True,
    ),))
    computer._session.observation = password
    monkeypatch.setattr(computer, "_check_window", lambda _session: None)
    blocked = computer.read_visual_field_value(password, field.id)
    assert blocked.credential_safe is False
    assert blocked.reason == "credential_control_present"
    capture_service.capture.assert_not_called()


def test_observation_includes_best_effort_process_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(windows, "_process_name", lambda process_id: "Notepad.exe" if process_id == 42 else "")

    result = observe(monkeypatch, node("Untitled", "Window"))

    assert result.app_name == "Notepad.exe"


def test_process_name_reads_basename_and_closes_handle(monkeypatch: pytest.MonkeyPatch) -> None:
    opened = object()
    close = Mock()
    monkeypatch.setattr(windows.sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "win32con", SimpleNamespace(
        PROCESS_QUERY_INFORMATION=1, PROCESS_VM_READ=2,
    ))
    monkeypatch.setitem(sys.modules, "win32api", SimpleNamespace(
        OpenProcess=lambda access, inherit, pid: opened, CloseHandle=close,
    ))
    monkeypatch.setitem(sys.modules, "win32process", SimpleNamespace(
        GetModuleFileNameEx=lambda handle, module: r"C:\Program Files\Example\Music.exe",
    ))
    assert windows._process_name(42) == "Music.exe"
    close.assert_called_once_with(opened)


def test_filtering_and_contiguous_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    hidden = node("Hidden", visible=False, children=(node("Also hidden"),))
    root = node(children=(
        hidden, node("   "), node(children=(node("Save", "Button"),)),
        node(kind="Button"), node(automation_id="identified"), node(kind="Text"),
    ))
    result = observe(monkeypatch, root)
    assert [control.id for control in result.elements] == ["c1", "c2", "c3"]
    assert [control.name for control in result.elements] == ["Save", "", ""]
    hidden.iter_children.assert_not_called()
    assert observe(monkeypatch, root).elements == result.elements


def test_ids_restart_for_new_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    assert observe(monkeypatch, node(children=(node("A"),))).elements[0].id == "c1"
    assert observe(monkeypatch, node(children=(node("B"),))).elements[0].id == "c1"


def test_depth_limit_does_not_expand_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    boundary = node("Parent", children=(node("Too deep"),))
    result = observe(monkeypatch, node(children=(boundary,)), max_depth=1)
    assert [control.name for control in result.elements] == ["Parent"]
    assert result.truncated
    boundary.iter_children.assert_not_called()


def test_hierarchy_records_only_immediate_parent(monkeypatch: pytest.MonkeyPatch) -> None:
    leaf = node("Search", "Edit")
    middle = node("Navigation", "Group", children=(leaf,))
    result = observe(monkeypatch, node("Application", "Window", children=(middle,)))
    by_name = {control.name: control for control in result.elements}
    assert by_name["Navigation"].parent_name == "Application"
    assert by_name["Search"].parent_name == "Navigation"
    assert by_name["Search"].parent_control_type == "Group"


def test_zero_depth_reads_only_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    root = node("Root")
    result = observe(monkeypatch, root, max_depth=0)
    assert result.window_title == "Root"
    assert result.elements == ()
    root.iter_children.assert_not_called()


def test_control_limit_stops_before_next_sibling(monkeypatch: pytest.MonkeyPatch) -> None:
    def siblings() -> Iterator[Mock]:
        yield node("First")
        pytest.fail("Read beyond the output limit")

    root = node()
    root.iter_children.side_effect = siblings
    result = observe(monkeypatch, root, max_controls=1)
    assert len(result.elements) == 1
    assert result.truncated


def test_node_budget_includes_filtered_nodes(monkeypatch: pytest.MonkeyPatch) -> None:
    def siblings() -> Iterator[Mock]:
        yield node(visible=False)
        yield node()
        pytest.fail("Read beyond the traversal budget")

    root = node()
    root.iter_children.side_effect = siblings
    result = observe(monkeypatch, root, max_nodes=2)
    assert not result.elements
    assert result.truncated


def test_text_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    result = observe(monkeypatch, node("long title", children=(
        node("abcdefgh", automation_id="abcdefgh"),
    )), max_text_length=4)
    assert result.window_title == "long"
    assert result.elements[0].name == "abcd"
    assert result.elements[0].automation_id == "abcd"
    assert result.truncated


def test_editable_text_pattern_value_appears_safely(monkeypatch: pytest.MonkeyPatch) -> None:
    control = node("Editor", "Document")
    control.element = SimpleNamespace(CurrentHasKeyboardFocus=1, CurrentIsPassword=0)
    text_range = Mock(GetText=Mock(return_value="Hello from Jev"))
    monkeypatch.setattr(windows, "_uia_wrapper", lambda _node: SimpleNamespace(
        iface_text=SimpleNamespace(DocumentRange=text_range),
    ))

    result = observe(monkeypatch, node(children=(control,)))

    assert result.elements[0].observed_text == "Hello from Jev"
    assert not result.elements[0].observed_text_truncated
    text_range.GetText.assert_called_once_with(501)


def test_password_and_unknown_password_values_are_never_read(monkeypatch: pytest.MonkeyPatch) -> None:
    password = node("Password", "Edit")
    password.element = SimpleNamespace(CurrentHasKeyboardFocus=1, CurrentIsPassword=1)
    unknown = node("Unknown", "Edit")
    unknown.element = SimpleNamespace(CurrentHasKeyboardFocus=0, CurrentIsPassword=None)
    wrapper = Mock(side_effect=AssertionError("password content must not be read"))
    monkeypatch.setattr(windows, "_uia_wrapper", wrapper)

    result = observe(monkeypatch, node(children=(password, unknown)))

    assert all(control.observed_text is None for control in result.elements)
    wrapper.assert_not_called()


def test_editable_value_pattern_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    control = node("Editor", "Edit")
    control.element = SimpleNamespace(CurrentHasKeyboardFocus=1, CurrentIsPassword=0)
    wrapper = SimpleNamespace(iface_value=SimpleNamespace(CurrentValue="abcdefgh"))
    monkeypatch.setattr(windows, "_uia_wrapper", lambda _node: wrapper)

    result = observe(monkeypatch, node(children=(control,)), max_observed_text_length=4)

    assert result.elements[0].observed_text == "abcd"
    assert result.elements[0].observed_text_truncated
    assert result.truncated


def test_editable_credentials_are_redacted_before_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "test-secret-value"
    monkeypatch.setenv("EXAMPLE_SECRET", secret)
    control = node("Editor", "Edit")
    control.element = SimpleNamespace(CurrentHasKeyboardFocus=1, CurrentIsPassword=0)
    monkeypatch.setattr(windows, "_uia_wrapper", lambda _node: SimpleNamespace(
        iface_value=SimpleNamespace(CurrentValue=f"prefix {secret}"),
    ))

    result = observe(monkeypatch, node(children=(control,)))

    assert secret not in (result.elements[0].observed_text or "")
    assert "[REDACTED]" in (result.elements[0].observed_text or "")


def test_property_failures_preserve_control_and_siblings(monkeypatch: pytest.MonkeyPatch) -> None:
    broken = node("Save", "Button")
    for attribute in ("name", "rectangle", "enabled"):
        setattr(type(broken), attribute, PropertyMock(side_effect=RuntimeError("gone")))
    root = node(children=(broken, node("Next")))
    type(root).process_id = PropertyMock(side_effect=RuntimeError("denied"))
    result = observe(monkeypatch, root)
    assert len(result.elements) == 2
    assert result.elements[0].name == ""
    assert result.elements[0].rectangle is None
    assert result.elements[0].enabled is None
    assert result.process_id is None
    assert result.inspection_errors == 4


def test_visibility_failure_still_allows_children(monkeypatch: pytest.MonkeyPatch) -> None:
    broken = node("Unknown", children=(node("Child"),))
    type(broken).visible = PropertyMock(side_effect=RuntimeError("gone"))
    result = observe(monkeypatch, node(children=(broken, node("Sibling"))))
    assert [control.name for control in result.elements] == ["Child", "Sibling"]
    assert result.inspection_errors == 1


def test_iteration_failure_preserves_partial_branch_and_ancestor_siblings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken_children() -> Iterator[Mock]:
        yield node("Before failure")
        raise RuntimeError("disappeared")

    branch = node()
    branch.iter_children.side_effect = broken_children
    result = observe(monkeypatch, node(children=(branch, node("After failure"))))
    assert [control.name for control in result.elements] == ["Before failure", "After failure"]
    assert result.inspection_errors == 1


def test_foreground_failure_is_structured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(windows, "_foreground", Mock(side_effect=RuntimeError("private data")))
    result = WindowsObserver().observe()
    assert result.error
    assert result.inspection_errors == 1
    assert result.elements == ()
    assert "private data" not in json.dumps(asdict(result))


@pytest.mark.parametrize("limits", [
    {"max_depth": -1}, {"max_controls": 0}, {"max_nodes": 0}, {"max_text_length": 0},
])
def test_invalid_limits(limits: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        ObservationOptions(**limits)


def test_cli_json(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(windows, "_foreground", lambda: node("Desktop app", "Window"))
    assert main(["observe", "--max-controls", "2"]) == 0
    assert json.loads(capsys.readouterr().out)["window_title"] == "Desktop app"


def test_cli_failure_exit(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(windows, "_foreground", Mock(side_effect=RuntimeError))
    assert main(["observe"]) == 1
    assert json.loads(capsys.readouterr().out)["error"]
