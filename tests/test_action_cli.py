"""Test interactive snapshot lifetime without touching the desktop."""

from unittest.mock import Mock
import json

from PIL import Image
import pytest

from computer.actions import ClickAction, OpenAppAction, PressKeyAction, TypeAction
from computer.applications import ApplicationCandidate, MemoryApplicationCatalog
from computer.models import (
    Observation, ProviderErrorDiagnostic, Rect, ScreenshotMetadata, UIElement, VisualElement,
    VisualRequestFingerprint,
)
from computer.visual import ScreenshotCapture, VisualObservation
from computer.results import ActionResult, LiteralInputDiagnostic
from decision.models import DecisionResult
import main as cli
from agent.generic_task import GenericTaskDebugResult
from agent.hybrid_debug import (
    HybridDebugResult, HybridResultDebugAgent, Phase2ExecutionDiagnostic,
    VisualExecutionDiagnostic,
)

APP_ID = "app_1111111111111111"


@pytest.mark.parametrize("argv,action", [
    (["open-app", "notepad"], OpenAppAction(APP_ID)),
    (["type-text", "literal {ENTER}", "--delay", "0"], TypeAction("literal {ENTER}")),
    (["press-key", "shift+tab", "--delay", "0"], PressKeyAction(("shift", "tab"))),
])
def test_action_commands(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    argv: list[str], action: OpenAppAction | TypeAction | PressKeyAction,
) -> None:
    observation = Observation("", "Target")
    computer = Mock()
    computer.observe.return_value = observation
    computer.execute.return_value = ActionResult(True, action, "Done")
    monkeypatch.setattr(cli, "WindowsComputer", Mock(return_value=computer))
    catalog = MemoryApplicationCatalog((
        ApplicationCandidate(APP_ID, "Notepad", "test", launch_policy="allow"),
    ))
    monkeypatch.setattr(cli, "WindowsApplicationCatalog", Mock(return_value=catalog))
    assert cli.main(argv) == 0
    assert json.loads(capsys.readouterr().out)["success"]
    if isinstance(action, OpenAppAction):
        computer.observe.assert_not_called()
        computer.execute.assert_called_once_with(action)
    else:
        computer.execute.assert_called_once_with(action, observation)


def test_debug_type_literal_is_explicit_bounded_and_independent(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    diagnostic = LiteralInputDiagnostic(
        5, 5, 10, 10, 10, None, None, 40, 123, 42, True,
    )
    probe = Mock(return_value=diagnostic)
    monkeypatch.setattr(cli, "debug_type_literal", probe)
    monkeypatch.setattr(cli, "WindowsComputer", Mock(side_effect=AssertionError("no agent")))
    monkeypatch.setattr("builtins.input", lambda: "yes")
    assert cli.main(["debug-type-literal", "Hello", "--delay", "0"]) == 0
    captured = capsys.readouterr()
    assert "EXPLICIT WINDOWS LITERAL INPUT DEBUG" in captured.err
    assert "NO HOTKEY SYNTAX / NO CLIPBOARD" in captured.err
    payload = json.loads(captured.out)
    assert payload["diagnostic"]["input_struct_size"] == 40
    assert "Hello" not in captured.out
    probe.assert_called_once_with("Hello")


def test_inspect_and_click_keeps_original_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    observation = Observation("", "Target")
    computer = Mock()
    computer.observe.return_value = observation
    computer.execute.return_value = ActionResult(True, ClickAction("c7"), "Done")
    monkeypatch.setattr(cli, "WindowsComputer", Mock(return_value=computer))
    monkeypatch.setattr("builtins.input", lambda: "c7")
    sleep = Mock()
    monkeypatch.setattr(cli.time, "sleep", sleep)
    assert cli.main(["inspect-and-click"]) == 0
    computer.observe.assert_called_once_with()
    assert computer.execute.call_args.args[1] is observation
    computer.execute.assert_called_once_with(ClickAction("c7"), observation)
    assert sleep.call_count == 2


def test_cancel_never_executes(monkeypatch: pytest.MonkeyPatch) -> None:
    computer = Mock()
    computer.observe.return_value = Observation("", "Target")
    monkeypatch.setattr(cli, "WindowsComputer", Mock(return_value=computer))
    monkeypatch.setattr("builtins.input", lambda: "")
    assert cli.main(["inspect-and-click", "--delay", "0"]) == 1
    computer.execute.assert_not_called()


def test_action_failure_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    computer = Mock()
    monkeypatch.setattr(cli, "WindowsComputer", Mock(return_value=computer))
    monkeypatch.setattr(cli, "WindowsApplicationCatalog", Mock(return_value=MemoryApplicationCatalog(())))
    assert cli.main(["open-app", "cmd"]) == 1
    computer.execute.assert_not_called()


def test_application_catalog_commands_emit_sanitized_metadata(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    private_path = r"C:\private\Spotify.exe"
    catalog = MemoryApplicationCatalog((ApplicationCandidate(
        APP_ID, "Spotify", "test", publisher="Example Publisher",
        launch_policy="allow", process_names=(private_path,),
    ),))
    monkeypatch.setattr(cli, "WindowsApplicationCatalog", Mock(return_value=catalog))
    assert cli.main(["list-apps"]) == 0
    listed = capsys.readouterr().out
    assert private_path not in listed
    assert json.loads(listed)[0]["display_name"] == "Spotify"
    assert cli.main(["find-app", "Open Spotify", "--limit", "1"]) == 0
    found = capsys.readouterr().out
    assert private_path not in found
    assert json.loads(found)[0]["id"] == APP_ID


def test_inspect_app_never_prints_editable_or_password_contents(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "must-not-appear"
    observation = Observation("browser.exe", "Safe title", (
        UIElement("c1", "Search", "Edit", enabled=True, visible=True, focused=True,
                  is_password=False, observed_text=secret, parent_name="Navigation", parent_control_type="Group"),
        UIElement("c2", "Password", "Edit", enabled=True, visible=True,
                  is_password=True, observed_text=secret),
    ), process_id=123, application_id=APP_ID)
    observer = Mock(observe=Mock(return_value=observation))
    monkeypatch.setattr(cli, "WindowsObserver", Mock(return_value=observer))
    monkeypatch.setattr(cli, "WindowsApplicationCatalog", Mock(return_value=MemoryApplicationCatalog((
        ApplicationCandidate(APP_ID, "Browser", "test", launch_policy="allow"),
    ))))
    assert cli.main(["inspect-app", "--delay", "0"]) == 0
    output = capsys.readouterr().out
    payload = json.loads(output)
    assert secret not in output
    assert payload["focused_observed_text_present"] is True
    assert payload["focused_observed_text_length"] == len(secret)
    assert [item["name"] for item in payload["interactive_named_controls"]] == ["Search"]


def test_inspect_hybrid_reports_geometry_without_pixels(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    metadata = ScreenshotMetadata(
        "snapshot", 9, Rect(-20, 10, 180, 110), Rect(-20, 10, 180, 110),
        400, 200, 192, 192, 2, 2,
    )
    observation = Observation(
        "app.exe", "Target", (
            UIElement("c1", "Search", "Button", enabled=True, visible=True),
        ), observation_id="snapshot", screenshot=metadata,
        visual_fallback_reason="sparse_uia",
    )
    observer = Mock(observe=Mock(return_value=observation), take_debug_capture=Mock(return_value=None))
    monkeypatch.setattr(cli, "WindowsObserver", Mock(return_value=observer))
    monkeypatch.setattr(cli, "WindowsWindowCapture", Mock(return_value=Mock()))
    monkeypatch.setattr(cli, "WindowsApplicationCatalog", Mock(return_value=MemoryApplicationCatalog(())))
    monkeypatch.setattr(cli, "visual_provider_from_environment", Mock(return_value=None))
    assert cli.main(["inspect-hybrid", "--delay", "0"]) == 0
    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload["screenshot"]["captured"] is True
    assert payload["screenshot"]["dimensions"] == [400, 200]
    assert payload["visual_provider_available"] is False
    assert "base64" not in output and "bytes" not in output


def test_inspect_hybrid_reports_sanitized_real_visual_grounding(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    metadata = ScreenshotMetadata(
        "snapshot", 9, Rect(0, 0, 200, 100), Rect(0, 0, 200, 100),
        200, 100, 96, 96, 1, 1,
    )
    observation = Observation(
        "app.exe", "Target", observation_id="snapshot", screenshot=metadata,
        visual_elements=(VisualElement(
            "v1", "Buscar", "navigation item", Rect(10, 20, 80, 45), None, True,
        ),), visual_provider="openai", visual_model="gpt-6-astra",
        visual_latency_ms=850, visual_usage=(("input_tokens", 123),),
    )
    observer = Mock(observe=Mock(return_value=observation), take_debug_capture=Mock(return_value=None))
    provider = Mock(name="openai", model="gpt-6-astra")
    provider.name = "openai"
    monkeypatch.setattr(cli, "visual_provider_from_environment", Mock(return_value=provider))
    monkeypatch.setattr(cli, "WindowsObserver", Mock(return_value=observer))
    monkeypatch.setattr(cli, "WindowsWindowCapture", Mock(return_value=Mock()))
    monkeypatch.setattr(cli, "WindowsApplicationCatalog", Mock(return_value=MemoryApplicationCatalog(())))
    assert cli.main(["inspect-hybrid", "--delay", "0"]) == 0
    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload["visual_provider_available"] is True
    assert payload["visual_model"] == "gpt-6-astra"
    assert payload["provider_latency_ms"] == 850
    assert payload["usage"] == {"input_tokens": 123}
    assert payload["visual_controls"] == [{
        "id": "v1", "label": "Buscar", "role": "navigation item",
        "confidence": None, "clickable": True, "parent": None,
        "rect": {"left": 10, "top": 20, "right": 80, "bottom": 45},
    }]
    assert "base64" not in output and "secret" not in output


def test_inspect_hybrid_reports_visual_timeout_without_crashing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    observation = Observation(
        "spotify.exe", "Spotify", observation_id="snapshot",
        visual_provider="openrouter", visual_model="inclusionai/ling-3.0-flash-vl:free",
        visual_latency_ms=20_001,
        visual_provider_error=ProviderErrorDiagnostic(
            "timeout", message="The visual provider request timed out.",
        ),
    )
    observer = Mock(observe=Mock(return_value=observation), take_debug_capture=Mock(return_value=None))
    provider = Mock(model="inclusionai/ling-3.0-flash-vl:free")
    provider.name = "openrouter"
    monkeypatch.setattr(cli, "visual_provider_from_environment", Mock(return_value=provider))
    monkeypatch.setattr(cli, "WindowsObserver", Mock(return_value=observer))
    monkeypatch.setattr(cli, "WindowsWindowCapture", Mock(return_value=Mock()))
    monkeypatch.setattr(cli, "WindowsApplicationCatalog", Mock(return_value=MemoryApplicationCatalog(())))
    assert cli.main(["inspect-hybrid", "--delay", "0"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["provider_latency_ms"] == 20_001
    assert payload["visual_provider_error"]["category"] == "timeout"


def test_directed_hybrid_cli_is_bounded_non_acting_and_reports_metrics(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    observation = Observation(
        "spotify.exe", "Spotify", observation_id="snapshot",
        visual_directed_grounding=True, visual_requested_max_elements=5,
        visual_returned_elements=1, screenshot_capture_ms=25,
        visual_request_build_ms=12, visual_latency_ms=700,
        visual_response_parse_ms=4, visual_total_observation_ms=741,
    )
    observer = Mock(observe=Mock(return_value=observation), take_debug_capture=Mock(return_value=None))
    observer_type = Mock(return_value=observer)
    monkeypatch.setenv("VISUAL_DIRECTED_MAX_ELEMENTS", "5")
    monkeypatch.setattr(cli, "WindowsObserver", observer_type)
    monkeypatch.setattr(cli, "WindowsWindowCapture", Mock(return_value=Mock()))
    monkeypatch.setattr(cli, "WindowsApplicationCatalog", Mock(return_value=MemoryApplicationCatalog(())))
    monkeypatch.setattr(cli, "visual_provider_from_environment", Mock(return_value=None))
    monkeypatch.setattr(cli, "WindowsComputer", Mock(side_effect=AssertionError("must not act")))
    assert cli.main([
        "inspect-hybrid-directed", "--objective", "Find the search field", "--delay", "0",
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["directed_grounding"] is True
    assert payload["requested_max_elements"] == 5
    assert payload["returned_visual_elements"] == 1
    assert payload["timing"] == {
        "screenshot_capture_ms": 25, "request_build_ms": 12,
        "provider_latency_ms": 700, "response_parse_ms": 4,
        "total_visual_observation_ms": 741,
    }
    serialized = json.dumps(payload)
    assert "Find the search field" not in serialized
    grounding = observer_type.call_args.kwargs["visual_grounding"]
    assert grounding.objective == "Find the search field"


def test_visual_diagnostic_cli_captures_once_calls_12_times_and_never_acts(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    metadata = ScreenshotMetadata(
        "one-capture", 9, Rect(0, 0, 200, 100), Rect(0, 0, 200, 100),
        200, 100, 96, 96, 1, 1,
    )
    capture = ScreenshotCapture(metadata, Image.new("RGB", (200, 100), "black"))
    observation = Observation("spotify.exe", "Spotify", observation_id="one-capture", screenshot=metadata)
    observer = Mock(observe=Mock(return_value=observation), take_debug_capture=Mock(return_value=capture))
    provider = Mock()
    provider.name = "gemini"
    provider.model = "gemini-3.5-flash-lite"
    provider.request_fingerprint.side_effect = lambda _shot, request: VisualRequestFingerprint(
        "gemini", provider.model, True, request.max_elements, 800, "application/json",
        "visual_elements", "1", (200, 100), 123, "a" * 64, len(request.objective),
    )
    provider.observe.return_value = VisualObservation((), "gemini", provider.model)
    monkeypatch.setattr(cli, "WindowsObserver", Mock(return_value=observer))
    monkeypatch.setattr(cli, "WindowsWindowCapture", Mock(return_value=Mock()))
    monkeypatch.setattr(cli, "WindowsApplicationCatalog", Mock(return_value=MemoryApplicationCatalog(())))
    monkeypatch.setattr(cli, "visual_provider_from_environment", Mock(return_value=provider))
    monkeypatch.setattr(cli, "WindowsComputer", Mock(side_effect=AssertionError("must not act")))

    assert cli.main(["diagnose-visual-grounding", "--delay", "0"]) == 0
    captured = capsys.readouterr()
    assert "VISUAL DIAGNOSTIC ONLY / NO COMPUTER ACTIONS WILL EXECUTE" in captured.err
    report = json.loads(captured.out)
    assert report["screenshot_capture_count"] == 1
    assert report["provider_call_count"] == 12
    assert provider.observe.call_count == 12
    observer.observe.assert_called_once_with()


def test_hybrid_debug_cli_announces_disabled_visual_execution(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    provider = Mock(name="gemini", model="gemini-3.5-flash-lite")
    provider.name = "gemini"
    decision = Mock(min_confidence=.8)
    computer = Mock()
    result = HybridDebugResult(
        True, "visual_action_blocked_for_debug", "expected boundary", 2, 123,
    )
    hybrid_agent = Mock(run=Mock(return_value=result))
    monkeypatch.setattr(cli, "visual_provider_from_environment", Mock(return_value=provider))
    monkeypatch.setattr(cli, "WindowsApplicationCatalog", Mock(return_value=MemoryApplicationCatalog(())))
    monkeypatch.setattr(cli, "WindowsWindowCapture", Mock(return_value=Mock()))
    monkeypatch.setattr(cli, "WindowsComputer", Mock(return_value=computer))
    monkeypatch.setattr(cli.JevDecisionMaker, "from_environment", Mock(return_value=decision))
    monkeypatch.setattr(cli, "HybridDebugAgent", Mock(return_value=hybrid_agent))
    monkeypatch.setattr("builtins.input", lambda: "yes")
    assert cli.main([
        "run-agent-hybrid-debug", "Open Spotify and play Californication", "--delay", "0",
    ]) == 0
    captured = capsys.readouterr()
    assert "EXPERIMENTAL HYBRID DEBUG" in captured.err
    assert "VISUAL EXECUTION: DISABLED" in captured.err
    assert json.loads(captured.out)["stop_reason"] == "visual_action_blocked_for_debug"
    hybrid_agent.run.assert_called_once_with("Open Spotify and play Californication")
    assert cli.WindowsComputer.call_args.kwargs["retain_debug_capture"] is False
    assert cli.HybridDebugAgent.call_args.kwargs["directed_capture_callback"] is None


def test_hybrid_click_debug_is_separate_confirmed_single_click_mode(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    provider = Mock()
    provider.name = "gemini"
    provider.model = "gemini-3.5-flash-lite"
    decision = Mock(min_confidence=.8)
    result = HybridDebugResult(
        True, "visual_click_phase1_complete", "stopped", 2, 100,
        visual_execution=VisualExecutionDiagnostic(
            remaining_budget=0, visual_click_attempted=True, input_issued=True,
            post_click_observation_obtained=True,
        ),
    )
    agent = Mock(run=Mock(return_value=result))
    monkeypatch.setattr(cli, "visual_provider_from_environment", Mock(return_value=provider))
    monkeypatch.setattr(cli, "WindowsApplicationCatalog", Mock(return_value=MemoryApplicationCatalog(())))
    monkeypatch.setattr(cli, "WindowsWindowCapture", Mock(return_value=Mock()))
    monkeypatch.setattr(cli, "WindowsComputer", Mock(return_value=Mock()))
    monkeypatch.setattr(cli.JevDecisionMaker, "from_environment", Mock(return_value=decision))
    click_type = Mock(return_value=agent)
    monkeypatch.setattr(cli, "HybridClickDebugAgent", click_type)
    monkeypatch.setattr("builtins.input", lambda: "yes")
    assert cli.main([
        "run-agent-hybrid-click-debug", "focus Spotify search", "--delay", "0",
    ]) == 0
    captured = capsys.readouterr()
    assert "EXPERIMENTAL HYBRID CLICK EXECUTION" in captured.err
    assert "AT MOST ONE VISUAL CLICK MAY EXECUTE" in captured.err
    assert "NO VISUAL TYPING" in captured.err
    assert json.loads(captured.out)["stop_reason"] == "visual_click_phase1_complete"
    click_type.assert_called_once()


def test_hybrid_type_debug_is_separate_confirmed_bounded_mode(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    provider = Mock()
    provider.name = "gemini"
    provider.model = "gemini-3.5-flash-lite"
    decision = Mock(min_confidence=.8)
    result = HybridDebugResult(
        True, "visual_type_phase2_complete", "stopped", 2, 100,
        phase2_execution=Phase2ExecutionDiagnostic(literal_type_budget_remaining=0),
    )
    agent = Mock(run=Mock(return_value=result))
    monkeypatch.setattr(cli, "visual_provider_from_environment", Mock(return_value=provider))
    monkeypatch.setattr(cli, "WindowsApplicationCatalog", Mock(return_value=MemoryApplicationCatalog(())))
    monkeypatch.setattr(cli, "WindowsWindowCapture", Mock(return_value=Mock()))
    monkeypatch.setattr(cli, "WindowsComputer", Mock(return_value=Mock()))
    monkeypatch.setattr(cli.JevDecisionMaker, "from_environment", Mock(return_value=decision))
    type_agent = Mock(return_value=agent)
    monkeypatch.setattr(cli, "HybridTypeDebugAgent", type_agent)
    monkeypatch.setattr("builtins.input", lambda: "yes")
    assert cli.main([
        "run-agent-hybrid-type-debug",
        "Open Spotify and play Californication by Red Hot Chili Peppers", "--delay", "0",
    ]) == 0
    captured = capsys.readouterr()
    assert "EXPERIMENTAL HYBRID CLICK + TYPE EXECUTION" in captured.err
    assert "AT MOST ONE VISUAL CLICK AND ONE LITERAL TYPE MAY EXECUTE" in captured.err
    assert "NO RESULT SELECTION" in captured.err
    assert json.loads(captured.out)["stop_reason"] == "visual_type_phase2_complete"
    type_agent.assert_called_once()


def test_hybrid_debug_explicitly_saves_exact_masked_provider_png(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path,
) -> None:
    from computer.visual_providers.common import png_fingerprint

    metadata = ScreenshotMetadata(
        "visual", 9, Rect(0, 0, 20, 10), Rect(0, 0, 20, 10),
        20, 10, 96, 96, 1, 1, masked_regions=1,
    )
    capture = ScreenshotCapture(metadata, Image.new("RGB", (20, 10), "navy"))
    length, digest = png_fingerprint(capture)
    observation = Observation(
        "spotify.exe", "Spotify Premium", observation_id="visual", screenshot=metadata,
        visual_request_fingerprint=VisualRequestFingerprint(
            "gemini", "gemini-3.5-flash-lite", True, 5, 800,
            "application/json", "visual_elements", "1", (20, 10),
            length, digest, 61,
        ),
    )
    provider = Mock()
    provider.name = "gemini"
    provider.model = "gemini-3.5-flash-lite"
    decision = Mock(min_confidence=.8)
    computer = Mock(take_debug_capture=Mock(return_value=capture))

    class FakeAgent:
        def __init__(self, *args, **kwargs):
            self.callback = kwargs["directed_capture_callback"]

        def run(self, _request):
            self.callback(observation, computer.take_debug_capture())
            return HybridDebugResult(True, "visual_action_blocked_for_debug", "done", 1, 10)

    monkeypatch.setattr(cli, "visual_provider_from_environment", Mock(return_value=provider))
    monkeypatch.setattr(cli, "WindowsApplicationCatalog", Mock(return_value=MemoryApplicationCatalog(())))
    monkeypatch.setattr(cli, "WindowsWindowCapture", Mock(return_value=Mock()))
    computer_type = Mock(return_value=computer)
    monkeypatch.setattr(cli, "WindowsComputer", computer_type)
    monkeypatch.setattr(cli.JevDecisionMaker, "from_environment", Mock(return_value=decision))
    monkeypatch.setattr(cli, "HybridDebugAgent", FakeAgent)
    monkeypatch.setattr("builtins.input", lambda: "yes")
    path = tmp_path / "exact-masked.png"
    assert cli.main([
        "run-agent-hybrid-debug", "request", "--delay", "0",
        "--save-visual-debug-screenshot", str(path),
    ]) == 0
    saved = path.read_bytes()
    assert len(saved) == length
    import hashlib
    assert hashlib.sha256(saved).hexdigest() == digest
    assert computer_type.call_args.kwargs["retain_debug_capture"] is True
    assert computer_type.call_args.kwargs["collect_provider_candidate_diagnostics"] is False
    assert "may contain sensitive UI contents" in capsys.readouterr().err


def test_result_screenshot_save_is_exact_hashed_and_optional(tmp_path) -> None:
    from computer.visual_providers.common import png_fingerprint, png_bytes

    metadata = ScreenshotMetadata(
        "result", 9, Rect(0, 0, 20, 10), Rect(0, 0, 20, 10),
        20, 10, 96, 96, 1, 1,
    )
    capture = ScreenshotCapture(metadata, Image.new("RGB", (20, 10), "navy"))
    length, digest = png_fingerprint(capture)
    observation = Observation(
        "spotify.exe", "Spotify", observation_id="result", screenshot=metadata,
        visual_request_fingerprint=VisualRequestFingerprint(
            "gemini", "gemini-3.5-flash-lite", True, 5, 800,
            "application/json", "visual_elements", "1", (20, 10),
            length, digest, 110,
        ),
    )
    assert HybridResultDebugAgent.__dataclass_fields__["result_capture_callback"].default is None
    path = tmp_path / "result.png"
    report = cli._save_exact_result_capture(observation, capture, path)
    saved = path.read_bytes()
    import hashlib
    assert saved == png_bytes(capture)
    assert hashlib.sha256(saved).hexdigest() == digest
    assert report["sha256"] == digest and report["byte_length"] == length
    assert report["matches_request_fingerprint"] is True
    with pytest.raises(FileExistsError):
        cli._save_exact_result_capture(observation, capture, path)


def test_result_screenshot_save_rejects_non_directed_fingerprint(tmp_path) -> None:
    metadata = ScreenshotMetadata(
        "result", 9, Rect(0, 0, 20, 10), Rect(0, 0, 20, 10),
        20, 10, 96, 96, 1, 1,
    )
    capture = ScreenshotCapture(metadata, Image.new("RGB", (20, 10), "navy"))
    observation = Observation(
        "app.exe", "Window", visual_request_fingerprint=VisualRequestFingerprint(
            "gemini", "model", False, 5, 800, "application/json", "schema", "1",
            (20, 10), 0, "0" * 64, 0,
        ),
    )
    with pytest.raises(ValueError):
        cli._save_exact_result_capture(observation, capture, tmp_path / "bad.png")
    assert not (tmp_path / "bad.png").exists()


def test_result_debug_cli_saves_only_result_grounding_capture(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path,
) -> None:
    from computer.visual_providers.common import png_fingerprint

    metadata = ScreenshotMetadata(
        "result", 9, Rect(0, 0, 20, 10), Rect(0, 0, 20, 10),
        20, 10, 96, 96, 1, 1,
    )
    capture = ScreenshotCapture(metadata, Image.new("RGB", (20, 10), "navy"))
    length, digest = png_fingerprint(capture)
    observation = Observation(
        "spotify.exe", "Spotify", observation_id="result", screenshot=metadata,
        visual_request_fingerprint=VisualRequestFingerprint(
            "gemini", "gemini-3.5-flash-lite", True, 5, 800,
            "application/json", "visual_elements", "1", (20, 10), length, digest, 110,
        ),
    )
    provider = Mock(name="gemini", model="gemini-3.5-flash-lite")
    decision = Mock(min_confidence=.8)
    computer = Mock(take_debug_capture=Mock(return_value=capture))

    class FakeAgent:
        def __init__(self, *args, **kwargs):
            self.result_callback = kwargs["result_capture_callback"]
            self.directed_callback = kwargs["directed_capture_callback"]

        def run(self, _request):
            assert self.directed_callback is None
            self.result_callback(observation, computer.take_debug_capture())
            return HybridDebugResult(True, "result_identity_mismatch", "stopped", 1, 10)

    monkeypatch.setattr(cli, "visual_provider_from_environment", Mock(return_value=provider))
    monkeypatch.setattr(cli, "WindowsApplicationCatalog", Mock(return_value=MemoryApplicationCatalog(())))
    monkeypatch.setattr(cli, "WindowsWindowCapture", Mock(return_value=Mock()))
    computer_type = Mock(return_value=computer)
    monkeypatch.setattr(cli, "WindowsComputer", computer_type)
    monkeypatch.setattr(cli.JevDecisionMaker, "from_environment", Mock(return_value=decision))
    monkeypatch.setattr(cli, "HybridResultDebugAgent", FakeAgent)
    monkeypatch.setattr("builtins.input", lambda: "yes")
    path = tmp_path / "phase3-result.png"
    assert cli.main([
        "run-agent-hybrid-result-debug", "request", "--delay", "0",
        "--save-result-debug-screenshot", str(path),
    ]) == 0
    import hashlib
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert len(path.read_bytes()) == length
    assert computer_type.call_args.kwargs["retain_debug_capture"] is True
    assert computer_type.call_args.kwargs["collect_provider_candidate_diagnostics"] is True
    assert "RESULT GROUNDING SCREENSHOT" in capsys.readouterr().err


def test_run_agent_dry_run_never_prompts_or_executes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("OPEN_APP_ACTIVATION_TIMEOUT_SECONDS", "invalid-for-run-agent")
    observation = Observation("notepad.exe", "Target")
    computer = Mock()
    computer.observe.return_value = observation
    computer.execute.side_effect = AssertionError("dry-run must not execute")
    decision = Mock()
    decision.min_confidence = 0.8
    decision.decide.return_value = DecisionResult(
        "ready", TypeAction("literal"), 0.95, "proposal", selected_option="type_1",
    )
    monkeypatch.setattr(cli, "WindowsComputer", Mock(return_value=computer))
    monkeypatch.setattr(cli.JevDecisionMaker, "from_environment", lambda _catalog=None: decision)
    monkeypatch.setattr("builtins.input", lambda: (_ for _ in ()).throw(AssertionError("no prompt in dry-run")))

    secret = "private-token-123"
    decision.decide.return_value = DecisionResult(
        "ready", TypeAction(secret), 0.95, "proposal", selected_option="type_1",
    )
    assert cli.main(["run-agent", secret, "--dry-run", "--debug", "--delay", "0"]) == 0
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["stop_reason"] == "needs_human"
    assert secret not in captured.out + captured.err
    computer.execute.assert_not_called()


def test_run_agent_requires_explicit_yes_for_real_run(monkeypatch: pytest.MonkeyPatch) -> None:
    decision = Mock()
    monkeypatch.setattr(cli.JevDecisionMaker, "from_environment", lambda _catalog=None: decision)
    computer = Mock()
    monkeypatch.setattr(cli, "WindowsComputer", Mock(return_value=computer))
    monkeypatch.setattr("builtins.input", lambda: "no")

    assert cli.main(["run-agent", "request", "--delay", "0"]) == 1
    computer.observe.assert_not_called()


def test_generic_debug_cli_is_separate_and_requires_confirmation(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    catalog = MemoryApplicationCatalog(())
    decision = Mock(min_confidence=.8)
    computer = Mock()
    result = GenericTaskDebugResult(True, "target_activated", "stopped", 4)
    agent = Mock(run=Mock(return_value=result))
    agent_factory = Mock(return_value=agent)
    monkeypatch.setattr(cli, "visual_provider_from_environment", Mock(return_value=None))
    monkeypatch.setattr(cli, "WindowsApplicationCatalog", Mock(return_value=catalog))
    monkeypatch.setattr(cli, "WindowsComputer", Mock(return_value=computer))
    monkeypatch.setattr(cli.JevDecisionMaker, "from_environment", Mock(return_value=decision))
    monkeypatch.setattr(cli, "GenericTaskDebugAgent", agent_factory)
    monkeypatch.setattr("builtins.input", lambda: "yes")
    monkeypatch.setenv("OPEN_APP_ACTIVATION_TIMEOUT_SECONDS", "4.5")

    request = "Open WhatsApp and open the chat with Pablo García"
    assert cli.main(["run-agent-generic-debug", request, "--delay", "0"]) == 0
    captured = capsys.readouterr()
    assert "EXPERIMENTAL GENERIC TARGET ACTIVATION" in captured.err
    assert "CURRENT STATE DRIVES ACTION SELECTION" in captured.err
    assert "NO CONSEQUENTIAL ACTIONS" in captured.err
    assert "Continue? [y/N]" in captured.err
    assert json.loads(captured.out)["stop_reason"] == "target_activated"
    agent.run.assert_called_once_with(request)
    assert agent_factory.call_args.kwargs["budgets"].app_activation_timeout_seconds == 4.5
