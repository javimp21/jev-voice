"""Deterministic hybrid-observation and visual-action safety tests."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import json
import sys
import ctypes
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image

from computer.actions import ClickAction, TypeAction, VisualClickAction
from computer.models import (
    CaptureDiagnostics, Observation, Rect, ScreenshotMetadata, UIElement, VisualElement,
    VisualGroundingStatus, VisualRequestFingerprint,
)
from computer.visual import (
    FakeVisualObserver, ScreenshotCapture, VisualCandidate, VisualGroundingRequest, VisualObservation,
    VisualProviderFailure, crop_screenshot_to_field,
    deduplicate_visual_elements, deduplicate_visual_elements_within,
    validate_visual_candidates, visual_click_point, visual_fallback_policy, visual_rect_to_screen,
    ResultReadinessOptions, VisualReadinessOptions, bounded_grounding_request, visual_frame_is_informative,
    result_readiness_richness_ratios, result_simplification_signal_count,
    visual_readiness_frame, visual_frames_meaningfully_differ,
)
from computer import windows, windows_actions
from computer.windows import WindowsObserver
from computer.windows_actions import WindowsComputer, _visual_click_geometry_diagnostics
from computer.windows_capture import CaptureUnavailable, WindowsWindowCapture
from computer.windows_capture import save_debug_overlay, save_debug_screenshot
from decision.jev import JevDecisionMaker
from safety.policy import AutonomousActionPolicy, BasicActionPolicy, Phase1VisualClickPolicy
from computer.visual_providers.common import (
    directed_visual_prompt, strict_visual_field_value_json, visual_schema,
)


def test_result_readiness_frame_comparison_is_bounded_and_deterministic() -> None:
    base = visual_readiness_frame(CaptureDiagnostics(
        1, 2, Rect(0, 0, 100, 100), Rect(0, 0, 100, 100), Rect(0, 0, 100, 100),
        100, 100, 10_000, 0, 0.0, "test", True, False, 500,
        (0, 0, 0), (255, 255, 255), 100.0, 20.0, 0.0, 0.0,
    ))
    same = replace(base, png_byte_length=10_100)
    changed = replace(base, png_byte_length=12_000)
    assert visual_frames_meaningfully_differ(base, same) is False
    assert visual_frames_meaningfully_differ(base, changed) is True


def metadata(*, snapshot: str = "snap", bounds: Rect = Rect(-200, 100, 300, 500),
             pixels: tuple[int, int] = (1000, 800)) -> ScreenshotMetadata:
    width, height = pixels
    return ScreenshotMetadata(
        snapshot, 99, bounds, bounds, width, height, 192, 192,
        width / (bounds.right - bounds.left), height / (bounds.bottom - bounds.top),
    )


def visual(element_id: str = "v1", *, label: str = "Search", role: str = "button",
           rect: Rect = Rect(20, 40, 120, 100), confidence: float | None = 0.95) -> VisualElement:
    return VisualElement(element_id, label, role, rect, confidence, True)


def observation_with_visual(element: VisualElement | None = None) -> Observation:
    meta = metadata()
    return Observation(
        "app.exe", "Window", observation_id=meta.snapshot_id,
        screenshot=meta, visual_elements=(element or visual(),),
    )


def test_visual_candidate_validation_assigns_snapshot_local_ids() -> None:
    candidates = (
        VisualCandidate("Search", "button", Rect(1, 2, 40, 30), .9),
        VisualCandidate("Result", "list item", Rect(1, 40, 80, 70), .8),
    )
    first = validate_visual_candidates(candidates, metadata(snapshot="one"))
    second = validate_visual_candidates(candidates, metadata(snapshot="two"))
    assert [item.id for item in first] == ["v1", "v2"]
    assert [item.id for item in second] == ["v1", "v2"]
    assert BasicActionPolicy().validate(
        VisualClickAction("one", "v1"), replace(observation_with_visual(first[0]), observation_id="two"),
    ).disposition == "deny"


def test_field_value_crop_is_bounded_and_derived_from_local_rectangle() -> None:
    bounds = Rect(0, 0, 1000, 800)
    meta = ScreenshotMetadata("crop-source", 99, bounds, bounds, 1000, 800, 96, 96, 1, 1)
    source = ScreenshotCapture(meta, Image.new("RGB", (1000, 800), "red"))
    crop = crop_screenshot_to_field(source, Rect(100, 200, 300, 240))

    assert crop.image.size == (210, 50)
    assert crop.metadata.capture_bounds == Rect(95, 195, 305, 245)
    assert crop.metadata.snapshot_id == source.metadata.snapshot_id
    assert crop.image.getpixel((0, 0)) == (255, 0, 0)
    crop.discard()
    source.discard()


def test_field_value_crop_fails_closed_for_password_overlap_or_unbounded_geometry() -> None:
    bounds = Rect(0, 0, 1000, 800)
    meta = ScreenshotMetadata("crop-source", 99, bounds, bounds, 1000, 800, 96, 96, 1, 1)
    source = ScreenshotCapture(meta, Image.new("RGB", (1000, 800), "white"))
    with pytest.raises(PermissionError):
        crop_screenshot_to_field(source, Rect(100, 200, 300, 240),
                                 sensitive_regions=(Rect(90, 190, 150, 230),))
    with pytest.raises(ValueError):
        crop_screenshot_to_field(source, Rect(0, 0, 990, 700))
    source.discard()


def test_value_only_visual_response_is_machine_readable_and_fail_closed() -> None:
    request = VisualGroundingRequest(
        "Read the current visible value inside the supplied text field crop.", 1,
        verification_only=True, query_field_continuity=True, field_value_only=True,
    )
    schema = visual_schema(include_field_value_only=True)
    prompt = directed_visual_prompt(
        1, request.objective, verification_only=True,
        query_field_continuity=True, field_value_only=True,
    )
    assert schema["required"] == ["field_value"]
    assert set(schema["properties"]) == {"field_value"}
    assert "Do not identify" in prompt and "nearby results" in prompt
    assert strict_visual_field_value_json('{"field_value":"Californication"}') == "Californication"
    assert strict_visual_field_value_json('{"field_value":null}') is None
    for invalid in (
        '{"field_value":"Californication","label":"Search"}',
        '{"field_value":123}', '{"field_value":"x","field_value":"y"}',
        'value is Californication',
    ):
        with pytest.raises(VisualProviderFailure, match="malformed_response"):
            strict_visual_field_value_json(invalid)


def test_verification_candidate_requires_typed_activity_and_rejects_unknown_values() -> None:
    from computer.visual_providers.common import candidate_from_normalized

    base = {
        "label": "Alpha", "role": "list_item",
        "box": {"left": 10, "top": 10, "right": 100, "bottom": 80},
        "clickable": False, "activity": "not_active",
        "selection_state": "not_selected", "region": "navigation",
    }
    candidate = candidate_from_normalized(
        base, 1000, 800, parent_required=False, verification_only=True,
    )
    assert candidate.activity == "not_active"
    assert candidate.selection_state.value == "not_selected"
    assert candidate.region.value == "navigation"
    with pytest.raises(VisualProviderFailure, match="invalid_response"):
        candidate_from_normalized(
            {key: value for key, value in base.items() if key != "activity"},
            1000, 800, parent_required=False, verification_only=True,
        )
    with pytest.raises(VisualProviderFailure, match="invalid_response"):
        candidate_from_normalized(
            {**base, "activity": "maybe"}, 1000, 800,
            parent_required=False, verification_only=True,
        )


def test_query_field_continuity_schema_separates_label_and_current_value() -> None:
    from computer.visual_providers.common import (
        candidate_from_normalized, directed_visual_prompt, visual_schema,
    )

    schema = visual_schema(include_query_field_continuity=True)
    item_schema = schema["properties"]["elements"]["items"]
    assert set(item_schema["properties"]) == {
        "field_label", "field_value", "role", "box", "clickable", "activity",
        "is_query_field", "credential_risk",
    }
    assert item_schema["properties"]["field_label"]["type"] == ["string", "null"]
    assert item_schema["properties"]["field_value"]["type"] == ["string", "null"]
    assert item_schema["properties"]["credential_risk"]["type"] == ["boolean", "null"]
    assert "field_label" in item_schema["required"]
    assert "field_value" in item_schema["required"]

    prompt = directed_visual_prompt(
        5, "Verify Californication in the active search field.",
        verification_only=True, query_field_continuity=True,
    )
    assert "placeholder" in prompt and "field_value separately" in prompt
    assert "nearby results" in prompt and "search history" in prompt

    candidate = candidate_from_normalized({
        "field_label": "¿Qué quieres reproducir?",
        "field_value": "Californication",
        "role": "search_field",
        "box": {"left": 10, "top": 10, "right": 200, "bottom": 60},
        "clickable": True,
        "activity": "active",
        "is_query_field": True,
        "credential_risk": False,
    }, 800, 600, verification_only=True, query_field_continuity=True)
    assert candidate.label == "¿Qué quieres reproducir?"
    assert candidate.field_label == "¿Qué quieres reproducir?"
    assert candidate.field_value == "Californication"
    assert candidate.is_query_field is True
    assert candidate.credential_risk is False
    validated = validate_visual_candidates((candidate,), metadata())
    assert validated[0].field_label == "¿Qué quieres reproducir?"
    assert validated[0].field_value == "Californication"
    assert validated[0].is_query_field is True
    assert validated[0].credential_risk is False


def test_query_field_parser_rejects_values_on_non_query_candidates() -> None:
    from computer.visual_providers.common import candidate_from_normalized

    with pytest.raises(VisualProviderFailure, match="invalid_response"):
        candidate_from_normalized({
            "field_label": None, "field_value": "Californication",
            "role": "list_item",
            "box": {"left": 10, "top": 10, "right": 200, "bottom": 60},
            "clickable": True, "activity": "not_active",
            "is_query_field": False, "credential_risk": False,
        }, 800, 600, verification_only=True, query_field_continuity=True)


@pytest.mark.parametrize("rect", [
    Rect(-1, 0, 10, 10), Rect(0, 0, 0, 10), Rect(0, 0, 1001, 10), Rect(0, 799, 10, 801),
])
def test_invalid_or_outside_visual_rectangles_are_rejected(rect: Rect) -> None:
    result = validate_visual_candidates((VisualCandidate("Bad", "button", rect, .9),), metadata())
    assert result == ()


def test_visual_target_and_confidence_must_be_valid() -> None:
    observation = observation_with_visual(visual(confidence=.74))
    policy = BasicActionPolicy(visual_min_confidence=.75)
    assert policy.validate(VisualClickAction("snap", "v1"), observation).disposition == "deny"
    assert policy.validate(VisualClickAction("snap", "v9"), observation).disposition == "deny"
    assert policy.validate(VisualClickAction("older", "v1"), observation).disposition == "deny"


def test_coordinate_conversion_handles_negative_monitors_and_dpi_scale() -> None:
    meta = metadata()
    element = visual(rect=Rect(200, 100, 400, 300))
    assert visual_rect_to_screen(element.rectangle, meta) == Rect(-100, 150, 0, 250)
    assert visual_click_point(element, meta) == (-50, 200)


def test_dedup_prefers_overlapping_equivalent_uia_control() -> None:
    meta = metadata(bounds=Rect(0, 0, 500, 400))
    uia = UIElement("c1", "Search", "Button", rectangle=Rect(10, 10, 60, 40),
                    enabled=True, visible=True)
    duplicate = visual(rect=Rect(20, 20, 120, 80))
    adjacent = visual("v2", rect=Rect(140, 20, 240, 80))
    assert deduplicate_visual_elements((uia,), (duplicate,), meta) == ()
    retained = deduplicate_visual_elements((uia,), (duplicate, adjacent), meta)
    assert len(retained) == 1 and retained[0].rectangle == adjacent.rectangle
    assert retained[0].id == "v1"


def test_similar_label_dedup_is_bounded_by_overlap() -> None:
    meta = metadata(bounds=Rect(0, 0, 500, 400))
    uia = UIElement("c1", "Search button", "Button", rectangle=Rect(10, 10, 60, 40))
    matching = visual(label="Search", rect=Rect(20, 20, 120, 80))
    assert deduplicate_visual_elements((uia,), (matching,), meta) == ()


def test_visual_to_visual_dedup_prefers_labeled_actionable_region() -> None:
    icon = visual("v1", label="", role="icon button", rect=Rect(20, 20, 40, 40), confidence=None)
    button = visual("v2", label="Search", role="button", rect=Rect(10, 10, 80, 50), confidence=None)
    adjacent = visual("v3", label="Search", role="button", rect=Rect(90, 10, 160, 50), confidence=None)
    retained = deduplicate_visual_elements_within((icon, button, adjacent))
    assert [(item.id, item.label, item.rectangle) for item in retained] == [
        ("v1", "Search", button.rectangle), ("v2", "Search", adjacent.rectangle),
    ]


def test_fallback_is_generic_for_sparse_and_rich_uia() -> None:
    sparse = Observation("anything.exe", "Any", (
        UIElement("c1", "", "Pane", enabled=True, visible=True),
    ))
    rich = Observation("anything.exe", "Any", tuple(
        UIElement(f"c{i}", name, kind, enabled=True, visible=True, is_password=False)
        for i, (name, kind) in enumerate((
            ("Back", "Button"), ("Address", "Edit"), ("Search", "Edit"),
            ("New", "Button"), ("Copy", "Button"), ("Folder", "ListItem"),
            ("View", "Button"), ("Sort", "Button"), ("Refresh", "Button"),
        ), 1)
    ))
    assert visual_fallback_policy(sparse).required
    assert visual_fallback_policy(rich, "search for invoice.pdf").required is False
    assert visual_fallback_policy(replace(sparse, app_name="spotify.exe")) == visual_fallback_policy(sparse)


def test_rich_uia_does_not_capture_or_call_remote_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    root = Node()
    children = []
    for index in range(9):
        child = Node(f"Control {index}", "Edit" if index == 0 else "Button")
        child.runtime_id = (index + 2,)
        child.parent = root
        children.append(child)
    root.iter_children = lambda: iter(children)  # type: ignore[method-assign]
    monkeypatch.setattr(windows, "_foreground", lambda: root)
    capture_service = Mock()
    provider = FakeVisualObserver((VisualCandidate("Unused", "button", Rect(1, 1, 10, 10), .9),))
    observation = WindowsObserver(
        capture_service=capture_service, visual_provider=provider,
    ).observe()
    assert observation.visual_fallback_reason is None
    capture_service.capture.assert_not_called()
    assert provider.calls == []


def test_activation_visual_verification_reuses_fresh_uia_session_and_masks_password_regions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Node()
    password = Node("Password", "Edit", runtime_id=(2,), parent=root,
                    rectangle=Rect(-180, 120, -80, 155))
    password.element = SimpleNamespace(
        CurrentHasKeyboardFocus=0, CurrentIsPassword=1, CurrentIsSelected=1,
    )
    root.iter_children = lambda: iter((password,))  # type: ignore[method-assign]
    monkeypatch.setattr(windows, "_foreground", lambda: root)
    monkeypatch.setattr(windows_actions, "_foreground", lambda: root)
    monkeypatch.setattr(windows_actions, "_focused", Mock(side_effect=RuntimeError))
    capture_service = FakeCapture()
    calls = []

    class Provider:
        name = "gemini"
        model = "gemini-3.5-flash-lite"
        pricing_class = "account_tier_dependent"

        def observe(self, screenshot, window, original_request, grounding=None):
            calls.append((screenshot, window, original_request, grounding))
            return VisualObservation((VisualCandidate(
                "Alpha", "list item", Rect(100, 100, 250, 180), None, False,
                activity="active",
            ),), self.name, self.model, directed_grounding=True)

    computer = WindowsComputer(capture_service=capture_service, visual_provider=Provider())
    fresh = computer.observe_local()
    assert fresh.elements[0].selected is True
    grounding = VisualGroundingRequest("Verify Alpha is the active entity.", 5, True)

    verified = computer.verify_activation_postcondition_visual(fresh, grounding)

    assert len(calls) == 1
    assert calls[0][1] is fresh
    assert calls[0][3].verification_only is True
    assert capture_service.sensitive == (password.rectangle,)
    assert verified.observation_id == fresh.observation_id
    assert verified.visual_elements[0].activity == "active"
    assert verified.visual_execution_authorized is False
    assert calls[0][0].image is None


def test_bilingual_consequential_labels_and_safe_navigation() -> None:
    for label in ("Delete", "Send", "Purchase", "Eliminar", "Borrar", "Enviar", "Comprar", "Pagar", "Publicar"):
        obs = observation_with_visual(visual(label=label))
        assert AutonomousActionPolicy().validate(VisualClickAction("snap", "v1"), obs).disposition == "confirm"
    for label in ("Search", "Buscar", "Your Library", "Atrás", "Account overview"):
        obs = observation_with_visual(visual(label=label))
        assert AutonomousActionPolicy().validate(VisualClickAction("snap", "v1"), obs).disposition == "allow"
    # Word boundaries avoid obvious substring false positives.
    obs = observation_with_visual(visual(label="Sender details"))
    assert AutonomousActionPolicy().validate(VisualClickAction("snap", "v1"), obs).disposition == "allow"


@dataclass
class Node:
    name: str = "Window"
    control_type: str = "Window"
    process_id: int = 42
    handle: int = 99
    visible: bool = True
    enabled: bool = True
    automation_id: str = ""
    rectangle: Rect = Rect(-200, 100, 300, 500)
    runtime_id: tuple[int, ...] = (1,)
    parent: object | None = None
    element: object = field(default_factory=lambda: SimpleNamespace(
        CurrentHasKeyboardFocus=0, CurrentIsPassword=0,
    ))

    def iter_children(self):
        return iter(())


class FakeCapture:
    def __init__(self, *, current: Rect | None = None) -> None:
        self.meta = metadata()
        self.current = current or self.meta.window_bounds
        self.image = Mock()
        self.image.size = (self.meta.pixel_width, self.meta.pixel_height)
        self.sensitive: tuple[Rect, ...] = ()
        self.virtual = Rect(-1000, -1000, 2000, 2000)

    def capture(self, snapshot_id, expected_handle, process_name, sensitive_regions=()):
        self.sensitive = tuple(sensitive_regions)
        self.meta = replace(self.meta, snapshot_id=snapshot_id, window_handle=expected_handle)
        diagnostics = CaptureDiagnostics(
            expected_handle, 42, self.meta.window_bounds, self.meta.capture_bounds,
            self.virtual, self.meta.pixel_width, self.meta.pixel_height, 100_000,
            len(self.sensitive), 0, "fake", True, False, 500, (0, 0, 0),
            (255, 255, 255), 50, 20, 0, 0,
        )
        return ScreenshotCapture(self.meta, self.image, diagnostics=diagnostics)

    def current_window_bounds(self, handle):
        return self.current

    def current_virtual_screen_bounds(self):
        return self.virtual


def hybrid_computer(
    monkeypatch: pytest.MonkeyPatch, *, moved: bool = False, confidence: float = .95,
    observation_only: bool = False,
):
    root = Node()
    monkeypatch.setattr(windows, "_foreground", lambda: root)
    monkeypatch.setattr(windows_actions, "_foreground", lambda: root)
    monkeypatch.setattr(windows_actions, "_focused", Mock(side_effect=RuntimeError))
    capture = FakeCapture(current=Rect(0, 0, 10, 10) if moved else None)
    base_provider = FakeVisualObserver((
        VisualCandidate("Search", "button", Rect(20, 40, 120, 100), confidence),
    ))
    if observation_only:
        class Provider:
            name = "openai"
            model = "gpt-6-astra"

            def observe(self, screenshot, window, original_request):
                result = base_provider.observe(screenshot, window, original_request)
                return VisualObservation(result.candidates, "openai", self.model, execution_authorized=False)

        provider = Provider()
    else:
        provider = base_provider
    click = Mock()
    monkeypatch.setattr(windows_actions, "_click_point", click)
    computer = WindowsComputer(capture_service=capture, visual_provider=provider)
    computer.set_observation_request("click Search")
    return computer, computer.observe(), capture, provider, click


def test_fake_provider_to_semantic_jev_option_has_no_coordinates(monkeypatch: pytest.MonkeyPatch) -> None:
    computer, observation, _capture, provider, _click = hybrid_computer(monkeypatch)

    def evaluate(payload):
        serialized = json.dumps(payload)
        assert "visual_v1" in payload["questions"]["next_action"]["criteria"]
        assert all(term not in serialized for term in ('"left"', '"right"', '"top"', '"bottom"', "image"))
        keys = payload["questions"]["next_action"]["criteria"]
        return {"model": "jev-1.13.0", "answers": {"next_action": {
            "type": "choice", "choice": "visual_v1", "confidence": .95,
            "probabilities": {key: float(key == "visual_v1") for key in keys},
        }}}

    decision = JevDecisionMaker(Mock(evaluate=Mock(side_effect=evaluate))).decide(
        "click Search", observation,
    )
    assert decision.action == VisualClickAction(observation.observation_id, "v1")
    assert provider.calls == [(observation.observation_id, "click Search")]


def test_phase1_executor_issues_one_local_center_click(monkeypatch: pytest.MonkeyPatch) -> None:
    computer, observation, _capture, _provider, click = hybrid_computer(monkeypatch)
    action = VisualClickAction(observation.observation_id, "v1")
    result = computer.execute_visual_click_phase1(action, observation, "focus Search", .80)
    assert result.success and result.input_issued is True
    assert result.source_observation_id == observation.observation_id
    click.assert_called_once()
    # The consumed snapshot cannot issue input a second time.
    second = computer.execute_visual_click_phase1(action, observation, "focus Search", .99)
    assert not second.success
    click.assert_called_once()


@pytest.mark.parametrize("provider_name", ["openai", "gemini"])
def test_generic_visual_target_executor_is_provider_agnostic_and_reports_local_binding(
    monkeypatch: pytest.MonkeyPatch, provider_name: str,
) -> None:
    root = Node()
    monkeypatch.setattr(windows, "_foreground", lambda: root)
    monkeypatch.setattr(windows_actions, "_foreground", lambda: root)
    monkeypatch.setattr(windows_actions, "_focused", Mock(side_effect=RuntimeError))
    capture = FakeCapture()
    from computer.visual_providers.common import candidate_from_normalized
    candidate = candidate_from_normalized({
        "label": "Iago", "role": "list_item",
        "box": {"left": 20, "top": 50, "right": 120, "bottom": 125},
        "clickable": True, "parent": "Chats",
    }, 1000, 800)
    assert candidate.rectangle == Rect(20, 40, 120, 100)

    class Provider:
        name = provider_name
        model = "mock-visual-model"
        pricing_class = "free"

        def observe(self, screenshot, window, original_request):
            return VisualObservation((candidate,), provider_name, self.model, execution_authorized=False)

    click = Mock()
    monkeypatch.setattr(windows_actions, "_click_point", click)
    computer = WindowsComputer(capture_service=capture, visual_provider=Provider())
    computer.set_observation_request("Open chat with Iago")
    observation = computer.observe()
    assert [item.id for item in observation.visual_elements] == ["v1"]
    assert observation.visual_elements[0].label == "Iago"
    action = VisualClickAction(observation.observation_id, "v1")

    result = computer.execute_generic_target_activation(action, observation)

    assert result.success and result.input_issued
    click.assert_called_once_with(-165, 135)
    diagnostic = result.visual_activation_diagnostic
    assert diagnostic is not None
    assert diagnostic.candidate_id == "v1"
    assert diagnostic.candidate_lookup_succeeded
    assert diagnostic.provenance_valid
    assert diagnostic.snapshot_binding_valid
    assert diagnostic.normalized_box_width_bucket == "small"
    assert diagnostic.normalized_box_height_bucket == "small"
    assert diagnostic.click_point_relative_bucket == "center:center"
    assert diagnostic.candidate_box_inside_bound_window is True
    assert diagnostic.candidate_box_aspect_bucket == "wide"
    assert diagnostic.overlaps_actionable_candidate is False
    assert diagnostic.nearest_actionable_neighbor_distance_bucket is None
    assert not {"x", "y", "coordinates", "rectangle"}.intersection(asdict(diagnostic))
    assert diagnostic.provider_name == provider_name
    assert diagnostic.provider_execution_agnostic
    assert diagnostic.preflight_started and diagnostic.preflight_succeeded
    assert diagnostic.geometry_resolution_succeeded
    assert diagnostic.resolved_point_inside_candidate
    assert diagnostic.resolved_point_inside_bound_window
    assert diagnostic.foreground_stable_before_input
    assert diagnostic.input_attempted and diagnostic.input_result == "succeeded"
    assert diagnostic.snapshot_consumed
    assert diagnostic.failure_stage is None and diagnostic.failure_reason is None
    # Raw window handles are not part of the new activation diagnostic.
    assert "window_handle" not in diagnostic.__dataclass_fields__


def test_visual_click_geometry_diagnostic_reports_overlap_and_neighbor_distance_as_buckets() -> None:
    target = visual(rect=Rect(20, 40, 120, 100))
    meta = metadata()
    screen_rect = visual_rect_to_screen(target.rectangle, meta)
    click_point = visual_click_point(target, meta)
    observation = replace(
        observation_with_visual(target),
        elements=(UIElement(
            "c1", "Neighbor", "Button", rectangle=Rect(-185, 125, -145, 145),
            enabled=True, visible=True,
        ),),
        visual_elements=(target, VisualElement(
            "v2", "Other", "button", Rect(121, 40, 170, 100), .9, True,
        )),
    )

    diagnostics = _visual_click_geometry_diagnostics(
        observation, target, meta, screen_rect, click_point,
    )

    assert diagnostics["overlaps_actionable_candidate"] is True
    assert diagnostics["nearest_actionable_neighbor_distance_bucket"] == "overlap"
    assert diagnostics["candidate_box_inside_bound_window"] is True
    assert diagnostics["click_point_relative_bucket"] == "center:center"


def test_generic_visual_target_diagnostics_identify_preflight_and_mouse_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    computer, observation, capture, _provider, click = hybrid_computer(monkeypatch)
    capture.current = Rect(-200, 100, 301, 500)
    stale = computer.execute_generic_target_activation(
        VisualClickAction(observation.observation_id, "v1"), observation,
    )
    assert not stale.success and stale.error == "stale_observation"
    assert stale.visual_activation_diagnostic is not None
    assert stale.visual_activation_diagnostic.failure_stage == "window_geometry_preflight"
    assert stale.visual_activation_diagnostic.failure_reason == "window_bounds_changed"
    assert not stale.visual_activation_diagnostic.input_attempted
    click.assert_not_called()

    computer, observation, _capture, _provider, click = hybrid_computer(monkeypatch)
    click.side_effect = OSError("private desktop detail must not escape")
    failed = computer.execute_generic_target_activation(
        VisualClickAction(observation.observation_id, "v1"), observation,
    )
    assert not failed.success and failed.error == "windows_operation_failed"
    diagnostic = failed.visual_activation_diagnostic
    assert diagnostic is not None
    assert diagnostic.failure_stage == "os_mouse_input"
    assert diagnostic.failure_reason == "mouse_input_failed_or_unknown"
    assert diagnostic.input_attempted and diagnostic.input_result == "failed_or_unknown"
    assert "private desktop" not in str(diagnostic)
    click.assert_called_once()


def test_generic_visual_target_rejects_wrong_and_consumed_snapshots_without_click(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    computer, observation, _capture, _provider, click = hybrid_computer(monkeypatch)
    wrong = computer.execute_generic_target_activation(
        VisualClickAction("other-snapshot", "v1"), observation,
    )
    assert not wrong.success
    assert wrong.visual_activation_diagnostic is not None
    assert not wrong.visual_activation_diagnostic.snapshot_binding_valid
    assert wrong.visual_activation_diagnostic.failure_stage == "safety_policy"
    click.assert_not_called()

    computer, observation, _capture, _provider, click = hybrid_computer(monkeypatch)
    action = VisualClickAction(observation.observation_id, "v1")
    assert computer.execute_generic_target_activation(action, observation).success
    second = computer.execute_generic_target_activation(action, observation)
    assert not second.success
    assert second.visual_activation_diagnostic is not None
    assert second.visual_activation_diagnostic.failure_stage == "snapshot_binding"
    assert second.visual_activation_diagnostic.snapshot_consumed
    click.assert_called_once()


def test_generic_visual_target_rejects_changed_trusted_foreground_before_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    computer, observation, _capture, _provider, click = hybrid_computer(monkeypatch)
    changed = Node(process_id=77, runtime_id=(77,))
    monkeypatch.setattr(windows_actions, "_foreground", lambda: changed)

    result = computer.execute_generic_target_activation(
        VisualClickAction(observation.observation_id, "v1"), observation,
    )

    assert not result.success and result.error == "unsafe_target"
    diagnostic = result.visual_activation_diagnostic
    assert diagnostic is not None
    assert diagnostic.failure_stage == "foreground_preflight"
    assert diagnostic.failure_reason == "foreground_or_trusted_window_changed"
    assert diagnostic.foreground_stable_before_input is False
    assert not diagnostic.input_attempted
    click.assert_not_called()


@pytest.mark.parametrize("change", ["hwnd", "pid", "bounds", "virtual"])
def test_phase1_executor_rejects_changed_window_context(
    monkeypatch: pytest.MonkeyPatch, change: str,
) -> None:
    computer, observation, capture, _provider, click = hybrid_computer(monkeypatch)
    root = Node()
    if change == "hwnd":
        root.handle = 100
        root.runtime_id = (100,)
    elif change == "pid":
        root.process_id = 77
    elif change == "bounds":
        capture.current = Rect(-200, 100, 301, 500)
    else:
        capture.virtual = Rect(-900, -900, 1900, 1900)
    monkeypatch.setattr(windows_actions, "_foreground", lambda: root)
    result = computer.execute_visual_click_phase1(
        VisualClickAction(observation.observation_id, "v1"), observation,
        "focus Search", .95,
    )
    assert not result.success
    assert result.error in {"stale_observation", "unsafe_target"}
    click.assert_not_called()


def test_phase1_policy_rejects_outside_consequential_and_credentials() -> None:
    base = observation_with_visual(visual(label="Search", role="search field", confidence=None))
    policy = Phase1VisualClickPolicy()
    action = VisualClickAction(base.observation_id, "v1")
    assert policy.validate(action, base, "focus search", .80).disposition == "allow"
    outside = replace(base, visual_elements=(replace(
        base.visual_elements[0], rectangle=Rect(0, 0, 5000, 5000),
    ),))
    assert policy.validate(action, outside, "focus search", .99).disposition == "deny"
    dangerous = replace(base, visual_elements=(replace(
        base.visual_elements[0], label="Confirm purchase",
    ),))
    assert policy.validate(action, dangerous, "focus search", .99).disposition == "deny"
    password = replace(base, elements=(UIElement(
        "c1", "", "Edit", rectangle=Rect(0, 0, 100, 100), enabled=True,
        visible=True, focused=True, is_password=True,
    ),))
    assert policy.validate(action, password, "focus search", .99).disposition == "deny"


def test_phase2_visual_verified_type_uses_literal_windows_executor_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    computer, observation, _capture, _provider, _click = hybrid_computer(monkeypatch)
    clicked = computer.execute_visual_click_phase1(
        VisualClickAction(observation.observation_id, "v1"), observation,
        "focus Search", .95,
    )
    assert clicked.success
    verification = computer.observe_directed(bounded_grounding_request(
        "Find the input control that was just activated and is ready for text entry.", 5,
    ))
    literal_input = Mock()
    monkeypatch.setattr(windows_actions, "_type_literal_unicode", literal_input)
    result = computer.execute_type_phase2(
        TypeAction("Californication"), verification, visual_verified=True,
    )
    assert result.success and result.input_issued is True
    literal_input.assert_called_once_with(
        "Californication", foreground_hwnd=99, foreground_pid=42,
    )
    # The type snapshot is consumed before input and cannot be reused.
    assert not computer.execute_type_phase2(
        TypeAction("again"), verification, visual_verified=True,
    ).success
    literal_input.assert_called_once()


@pytest.mark.parametrize("text,expected_units", [
    ("Californication", tuple(ord(char) for char in "Californication")),
    ("canción", tuple(ord(char) for char in "canción")),
    ("😀", (0xD83D, 0xDE00)),
])
def test_literal_send_input_builds_unicode_down_up_events(text, expected_units) -> None:
    captured = {}

    def send(count, events, size):
        captured["count"] = count
        captured["size"] = size
        captured["units"] = tuple(events[index].ki.wScan for index in range(count))
        captured["flags"] = tuple(events[index].ki.dwFlags for index in range(count))
        return count

    diagnostic = windows_actions._type_literal_unicode(text, send_input=send)
    expanded = tuple(unit for unit in expected_units for _ in range(2))
    assert captured["units"] == expanded
    assert captured["flags"] == tuple(flag for _ in expected_units for flag in (0x0004, 0x0006))
    assert captured["size"] == ctypes.sizeof(windows_actions._INPUT)
    assert diagnostic.utf16_code_unit_count == len(expected_units)
    assert diagnostic.input_event_count == len(expected_units) * 2


@pytest.mark.parametrize("returned", [0, 1])
def test_literal_send_input_zero_or_partial_fails_closed_without_retry(returned) -> None:
    calls = []
    with pytest.raises(windows_actions.LiteralInputFailure) as failure:
        windows_actions._type_literal_unicode(
            "A", send_input=lambda count, events, size: calls.append(count) or returned,
            get_last_error=lambda: 87,
        )
    assert calls == [2]
    diagnostic = failure.value.diagnostic
    assert diagnostic.send_input_requested_count == 2
    assert diagnostic.send_input_returned_count == returned
    assert diagnostic.last_error == 87
    assert diagnostic.failure_stage == "send_input_partial_or_failed"


def test_send_input_abi_matches_native_layout_and_declares_signature(monkeypatch) -> None:
    expected = 40 if ctypes.sizeof(ctypes.c_void_p) == 8 else 28
    assert ctypes.sizeof(windows_actions._INPUT) == expected
    function = Mock()
    library = Mock(SendInput=function)
    win_dll = Mock(return_value=library)
    monkeypatch.setattr(windows_actions.ctypes, "WinDLL", win_dll)
    loaded, _get_error = windows_actions._windows_send_input_api()
    assert loaded is function
    win_dll.assert_called_once_with("user32", use_last_error=True)
    assert function.argtypes == (
        windows_actions.wintypes.UINT,
        ctypes.POINTER(windows_actions._INPUT), ctypes.c_int,
    )
    assert function.restype is windows_actions.wintypes.UINT


def test_phase2_context_change_prevents_send_input(monkeypatch: pytest.MonkeyPatch) -> None:
    computer, _observation, _capture, _provider, _click = hybrid_computer(monkeypatch)
    verification = computer.observe_directed(bounded_grounding_request(
        "Find the input control that was just activated and is ready for text entry.", 5,
    ))
    changed = Node(handle=100, process_id=77, runtime_id=(100,))
    monkeypatch.setattr(windows_actions, "_foreground", lambda: changed)
    literal_input = Mock(side_effect=AssertionError("SendInput must not run"))
    monkeypatch.setattr(windows_actions, "_type_literal_unicode", literal_input)
    result = computer.execute_type_phase2(
        TypeAction("Californication"), verification, visual_verified=True,
    )
    assert not result.success and result.error == "stale_observation"
    literal_input.assert_not_called()


def test_visual_click_resolves_once_and_moved_window_fails_stale(monkeypatch: pytest.MonkeyPatch) -> None:
    computer, observation, _capture, _provider, click = hybrid_computer(monkeypatch)
    action = VisualClickAction(observation.observation_id, "v1")
    result = computer.execute(action, observation)
    assert result.success
    click.assert_called_once_with(-165, 135)
    assert not computer.execute(action, observation).success

    moved_computer, moved_observation, _capture, _provider, moved_click = hybrid_computer(monkeypatch, moved=True)
    moved = moved_computer.execute(
        VisualClickAction(moved_observation.observation_id, "v1"), moved_observation,
    )
    assert not moved.success and moved.error == "stale_observation"
    moved_click.assert_not_called()


def test_real_provider_visual_click_is_observation_only(monkeypatch: pytest.MonkeyPatch) -> None:
    computer, observation, _capture, _provider, click = hybrid_computer(
        monkeypatch, observation_only=True,
    )
    result = computer.execute(
        VisualClickAction(observation.observation_id, "v1"), observation,
    )
    assert not result.success and result.error == "policy_blocked"
    assert "observation-only" in result.message
    click.assert_not_called()


def test_provider_image_is_discarded_and_debug_payload_contains_no_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    _computer, observation, capture, _provider, _click = hybrid_computer(monkeypatch)
    capture.image.close.assert_called_once_with()
    payload, _candidates, _stats = JevDecisionMaker(Mock())._prepare("click Search", observation, ())
    serialized = json.dumps(payload)
    assert "base64" not in serialized and "screenshot bytes" not in serialized


@pytest.mark.parametrize("code", ["api_error", "timeout", "invalid_response"])
def test_provider_failure_preserves_uia_observation_safely(
    monkeypatch: pytest.MonkeyPatch, code: str,
) -> None:
    root = Node()
    monkeypatch.setattr(windows, "_foreground", lambda: root)
    capture = FakeCapture()

    class BrokenProvider:
        name = "openai"
        model = "gpt-6-astra"

        def observe(self, screenshot, window, original_request):
            raise VisualProviderFailure(code)

    observation = WindowsObserver(
        capture_service=capture, visual_provider=BrokenProvider(),
    ).observe()
    assert observation.error is None
    assert observation.screenshot is not None
    assert observation.visual_elements == ()
    assert observation.visual_provider == "openai"
    assert observation.visual_provider_error == code
    assert isinstance(observation.visual_latency_ms, int)
    assert code in {"api_error", "timeout", "invalid_response"}


def test_password_rectangles_are_passed_to_capture_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    password = Node("Password", "Edit", rectangle=Rect(-100, 150, 0, 190))
    password.runtime_id = (2,)
    password.parent = None
    password.element = SimpleNamespace(CurrentHasKeyboardFocus=0, CurrentIsPassword=1)
    root = Node()
    password.parent = root
    root.iter_children = lambda: iter((password,))  # type: ignore[method-assign]
    monkeypatch.setattr(windows, "_foreground", lambda: root)
    capture = FakeCapture()
    observer = WindowsObserver(
        capture_service=capture, visual_provider=FakeVisualObserver(()),
    )
    observer.observe()
    assert capture.sensitive == (password.rectangle,)


def test_directed_pipeline_preserves_multiple_candidates_on_fresh_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Node()
    monkeypatch.setattr(windows, "_foreground", lambda: root)
    capture = FakeCapture()
    provider = FakeVisualObserver((
        VisualCandidate("Search", "text field", Rect(20, 40, 120, 100)),
        VisualCandidate("Library", "button", Rect(140, 40, 240, 100)),
    ))
    computer = WindowsComputer(
        capture_service=capture, visual_provider=provider,
        collect_provider_candidate_diagnostics=True,
    )
    computer.set_observation_request("search for music")
    local_snapshot = computer.observe_local()
    directed = computer.observe_directed(bounded_grounding_request("Find search.", 5))
    assert directed.observation_id != local_snapshot.observation_id
    assert [item.id for item in directed.visual_elements] == ["v1", "v2"]
    assert directed.screenshot is not None
    assert directed.screenshot.snapshot_id == directed.observation_id
    assert directed.visual_grounding_status is VisualGroundingStatus.SUCCESS_WITH_CANDIDATES
    assert directed.visual_pipeline is not None
    assert directed.visual_pipeline.provider_requested_max_elements == 5
    assert directed.visual_pipeline.provider_raw_element_count == 2
    assert directed.visual_pipeline.parsed_element_count == 2
    assert directed.visual_pipeline.validated_element_count == 2
    assert directed.visual_pipeline.deduplicated_element_count == 2
    assert directed.visual_pipeline.observation_visual_control_count == 2
    assert [item.provider_candidate_id for item in directed.visual_provider_candidates] == ["p1", "p2"]
    assert [item.label for item in directed.visual_provider_candidates] == ["Search", "Library"]


def test_provider_candidate_text_diagnostics_are_off_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Node()
    monkeypatch.setattr(windows, "_foreground", lambda: root)
    provider = FakeVisualObserver((
        VisualCandidate("Visible Search Result", "card", Rect(20, 40, 120, 100)),
    ))
    observer = WindowsObserver(
        capture_service=FakeCapture(), visual_provider=provider,
        visual_grounding=bounded_grounding_request("Find this result.", 5),
    )
    result = observer.observe()
    assert len(result.visual_elements) == 1
    assert result.visual_provider_candidates == ()


def test_visual_pipeline_reports_validation_rejection_without_sensitive_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Node()
    monkeypatch.setattr(windows, "_foreground", lambda: root)
    provider = FakeVisualObserver((
        VisualCandidate("bad", "button", Rect(-1, 0, 20, 20)),
    ))
    observer = WindowsObserver(
        capture_service=FakeCapture(), visual_provider=provider,
        visual_grounding=bounded_grounding_request("Find search.", 5),
    )
    observer.set_observation_request("search for music")
    result = observer.observe()
    assert result.visual_elements == ()
    assert result.visual_grounding_status is VisualGroundingStatus.VALIDATION_EMPTY
    assert result.visual_pipeline is not None
    assert result.visual_pipeline.provider_raw_element_count == 1
    assert result.visual_pipeline.validated_element_count == 0
    assert dict(result.visual_rejection_summary)["invalid_bbox"] == 1
    assert "bad" not in repr(result.visual_rejection_summary)


def test_windows_capture_rejects_secure_or_changed_foreground(monkeypatch: pytest.MonkeyPatch) -> None:
    import computer.windows_capture as capture_module
    monkeypatch.setattr(capture_module.sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "win32gui", SimpleNamespace(GetForegroundWindow=lambda: 8))
    service = WindowsWindowCapture(desktop_check=lambda: False)
    with pytest.raises(CaptureUnavailable):
        service.capture("snap", 8, "app.exe")

    service = WindowsWindowCapture(desktop_check=lambda: True)
    with pytest.raises(CaptureUnavailable):
        service.capture("snap", 7, "app.exe")


def test_foreground_capture_clips_multi_monitor_bounds_scales_and_masks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import computer.windows_capture as capture_module
    from PIL import Image

    monkeypatch.setattr(capture_module.sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "win32gui", SimpleNamespace(GetForegroundWindow=lambda: 99))
    monkeypatch.setattr(capture_module, "_window_rect", lambda _handle: Rect(-200, 0, 100, 100))
    monkeypatch.setattr(capture_module, "_virtual_screen", lambda: Rect(-100, -50, 500, 500))
    monkeypatch.setattr(capture_module, "_dpi", lambda _handle: (144, 144))
    captured_bounds: list[Rect] = []

    def grab(bounds: Rect):
        captured_bounds.append(bounds)
        return Image.new("RGB", (400, 200), "white")

    service = WindowsWindowCapture(grabber=grab, desktop_check=lambda: True)
    result = service.capture("snap", 99, "safe.exe", (Rect(-90, 10, -50, 30),))
    assert captured_bounds == [Rect(-100, 0, 100, 100)]
    assert result.metadata.window_bounds == Rect(-200, 0, 100, 100)
    assert result.metadata.capture_bounds == Rect(-100, 0, 100, 100)
    assert (result.metadata.scale_x, result.metadata.scale_y) == (2, 2)
    assert (result.metadata.dpi_x, result.metadata.dpi_y) == (144, 144)
    assert result.metadata.masked_regions == 1
    assert result.diagnostics is not None
    assert result.diagnostics.capture_backend == "pillow_imagegrab_screen_region"
    assert result.diagnostics.mask_region_count == 1
    assert result.diagnostics.masked_area_percent == 4.0
    assert result.diagnostics.near_white_percent < 100
    assert result.image.getpixel((30, 30)) == (0, 0, 0)
    result.discard()


def test_capture_content_metrics_distinguish_black_white_and_rich_images() -> None:
    from PIL import Image
    from computer.windows_capture import _content_metrics

    black = _content_metrics(Image.new("RGB", (64, 64), "black"))
    white = _content_metrics(Image.new("RGB", (64, 64), "white"))
    rich_image = Image.new("RGB", (64, 64))
    rich_image.putdata([(x * 4, y * 4, (x + y) * 2) for y in range(64) for x in range(64)])
    rich = _content_metrics(rich_image)
    assert black[0] == 1 and black[5] == 100.0 and black[6] == 0.0
    assert white[0] == 1 and white[5] == 0.0 and white[6] == 100.0
    assert rich[0] > 100 and rich[4] > 20 and rich[5] < 5 and rich[6] < 5
    rich_image.close()


def test_debug_screenshot_requires_explicit_new_png(tmp_path) -> None:
    from PIL import Image

    capture = ScreenshotCapture(metadata(), Image.new("RGB", (10, 10), "white"))
    path = tmp_path / "capture.png"
    save_debug_screenshot(capture, path)
    assert path.read_bytes().startswith(b"\x89PNG")
    from computer.visual_providers.common import png_fingerprint
    length, digest = png_fingerprint(capture)
    import hashlib
    assert len(path.read_bytes()) == length
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    with pytest.raises(FileExistsError):
        save_debug_screenshot(capture, path)
    with pytest.raises(ValueError):
        save_debug_screenshot(capture, tmp_path / "capture.jpg")
    overlay = tmp_path / "overlay.png"
    save_debug_overlay(capture, (visual(rect=Rect(1, 1, 9, 9)),), overlay)
    assert overlay.read_bytes().startswith(b"\x89PNG")
    with pytest.raises(FileExistsError):
        save_debug_overlay(capture, (visual(),), overlay)
    capture.discard()


def test_directed_observation_resolves_fresh_foreground_hwnd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stale = Node(name="Spotify", handle=10, runtime_id=(10,))
    current = Node(name="Spotify Premium", handle=20, runtime_id=(20,))
    foreground = Mock(side_effect=[stale, current])
    monkeypatch.setattr(windows, "_foreground", foreground)
    monkeypatch.setattr(windows_actions, "_focused", Mock(side_effect=RuntimeError))
    capture = FakeCapture()
    handles: list[int] = []
    original_capture = capture.capture

    def record_capture(snapshot_id, expected_handle, process_name, sensitive_regions=()):
        handles.append(expected_handle)
        return original_capture(snapshot_id, expected_handle, process_name, sensitive_regions)

    capture.capture = record_capture  # type: ignore[method-assign]
    computer = WindowsComputer(
        capture_service=capture, visual_provider=FakeVisualObserver(()),
    )
    computer.set_observation_request("search for music")
    first = computer.observe_local()
    directed = computer.observe_directed(bounded_grounding_request("Find search", 5))
    assert first.window_title == "Spotify"
    assert directed.window_title == "Spotify Premium"
    assert handles == [20]
    assert foreground.call_count == 2


def readiness_capture_diagnostics(*, informative: bool) -> CaptureDiagnostics:
    return CaptureDiagnostics(
        99, 42, Rect(0, 0, 100, 100), Rect(0, 0, 100, 100),
        Rect(0, 0, 1920, 1080), 100, 100,
        50_000 if informative else 100,
        0, 0.0, "test_screen_region", True, False,
        500 if informative else 1, (5, 5, 5),
        (240, 240, 240) if informative else (5, 5, 5),
        20.0, 25.0 if informative else 0.0, 0.0, 0.0,
    )


_READINESS_STATE_STATS = {
    "autocomplete": (40_000, 300, 20.0),
    "blank": (15_000, 150, 10.0),
    "loading": (20_000, 190, 12.0),
    "results": (74_000, 590, 45.0),
    "results_late": (87_000, 720, 60.0),
    "results_latest": (101_000, 850, 75.0),
    "changed_a": (53_000, 410, 28.0),
    "changed_b": (66_000, 520, 39.0),
    "changed_c": (79_000, 640, 51.0),
}
_READINESS_STATE_COLORS = {
    state: (index * 25 % 256, index * 41 % 256, index * 59 % 256)
    for index, state in enumerate(_READINESS_STATE_STATS, 1)
}


def readiness_state_diagnostics(state: str) -> CaptureDiagnostics:
    byte_length, unique_colors, stddev = _READINESS_STATE_STATS[state]
    return replace(
        readiness_capture_diagnostics(informative=True),
        png_byte_length=byte_length,
        sampled_unique_colors=unique_colors,
        luminance_stddev=stddev,
    )


def readiness_state_color(state: str) -> tuple[int, int, int]:
    return _READINESS_STATE_COLORS[state]


class ReadinessCapture:
    def __init__(self, frames: list[bool | str]) -> None:
        self.frames = frames
        self.calls = 0
        self.captures: list[ScreenshotCapture] = []

    def capture(self, snapshot_id, expected_handle, process_name, sensitive_regions=()):
        from PIL import Image
        state = self.frames[min(self.calls, len(self.frames) - 1)]
        self.calls += 1
        meta = ScreenshotMetadata(
            snapshot_id, expected_handle, Rect(0, 0, 100, 100), Rect(0, 0, 100, 100),
            100, 100, 96, 96, 1, 1,
        )
        if type(state) is bool:
            diagnostics = readiness_capture_diagnostics(informative=state)
            color = "navy" if state else "white"
        else:
            diagnostics = readiness_state_diagnostics(state)
            color = readiness_state_color(state)
        capture = ScreenshotCapture(meta, Image.new("RGB", (100, 100), color), diagnostics=diagnostics)
        self.captures.append(capture)
        return capture

    def current_window_bounds(self, handle):
        return Rect(0, 0, 100, 100)


class ReadinessProvider:
    name = "gemini"
    model = "test"
    pricing_class = "test"

    def __init__(self) -> None:
        self.payloads: list[bytes] = []

    def observe(self, screenshot, window, original_request, grounding=None):
        from computer.visual_providers.common import png_bytes
        self.payloads.append(png_bytes(screenshot))
        return VisualObservation((), self.name, self.model, requested_max_elements=5)

    def request_fingerprint(self, screenshot, grounding):
        from computer.visual_providers.common import png_fingerprint
        length, digest = png_fingerprint(screenshot)
        return VisualRequestFingerprint(
            self.name, self.model, True, grounding.max_elements, 800,
            "application/json", "visual_elements", "1", (100, 100),
            length, digest, len(grounding.objective),
        )


class ReadinessClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def readiness_computer(
    monkeypatch, frames, *, foreground=None, timeout=.5, result_timeout=5,
    quiet_ms=1200, poll_interval=.2, grace_ms=2500, simplification_ratio=.75,
    simplification_min_signals=2, retain_debug_capture=False,
):
    root = Node(name="Spotify Premium", handle=99, runtime_id=(99,))
    monkeypatch.setattr(windows, "_foreground", foreground or (lambda: root))
    monkeypatch.setattr(windows_actions, "_focused", Mock(side_effect=RuntimeError))
    clock = ReadinessClock()
    capture = ReadinessCapture(frames)
    provider = ReadinessProvider()
    computer = WindowsComputer(
        capture_service=capture, visual_provider=provider,
        visual_readiness_options=VisualReadinessOptions(timeout, .1),
        result_readiness_options=ResultReadinessOptions(
            result_timeout, poll_interval, quiet_ms, grace_ms,
            simplification_ratio, simplification_min_signals,
        ),
        retain_debug_capture=retain_debug_capture,
        readiness_clock=clock, readiness_sleep=clock.sleep,
    )
    computer.set_observation_request("search for music")
    return computer, capture, provider, root


def result_readiness_baseline(state: str = "autocomplete") -> Observation:
    return Observation(
        "spotify.exe", "Spotify", process_id=42, observation_id="pre-submit",
        capture_diagnostics=readiness_state_diagnostics(state),
    )


def test_visual_readiness_initially_informative_calls_provider_once(monkeypatch) -> None:
    computer, capture, provider, _root = readiness_computer(monkeypatch, [True])
    result = computer.observe_directed(bounded_grounding_request("Find search", 5))
    assert result.visual_readiness.ready is True
    assert result.visual_readiness.reason == "initially_ready"
    assert result.visual_readiness.attempts == 1
    assert capture.calls == 1 and len(provider.payloads) == 1
    assert result.visual_provider_call_count == 1
    assert len(provider.payloads[0]) == result.visual_request_fingerprint.encoded_image_byte_length


def test_opt_in_debug_capture_is_retained_for_typed_provider_failure(monkeypatch) -> None:
    from computer.visual_providers.common import png_bytes

    computer, _capture, provider, _root = readiness_computer(
        monkeypatch, [True], retain_debug_capture=True,
    )
    sent: list[bytes] = []

    def fail_after_send(screenshot, _window, _original_request, _grounding=None):
        sent.append(png_bytes(screenshot))
        raise VisualProviderFailure("timeout")

    provider.observe = fail_after_send
    result = computer.observe_directed(bounded_grounding_request("Find search", 5))

    assert result.visual_provider_error.category == "timeout"
    retained = computer.take_debug_capture()
    assert retained is not None
    assert png_bytes(retained) == sent[0]
    retained.discard()
    assert computer.take_debug_capture() is None


def test_visual_readiness_uniform_then_informative_reuses_final_capture(monkeypatch) -> None:
    computer, capture, provider, _root = readiness_computer(monkeypatch, [False, False, True])
    result = computer.observe_directed(bounded_grounding_request("Find search", 5))
    assert result.visual_readiness.reason == "became_ready"
    assert result.visual_readiness.attempts == 3
    assert capture.calls == 3 and len(provider.payloads) == 1
    import hashlib
    assert hashlib.sha256(provider.payloads[0]).hexdigest() == result.visual_request_fingerprint.screenshot_sha256


def test_visual_readiness_timeout_never_calls_provider(monkeypatch) -> None:
    computer, capture, provider, _root = readiness_computer(monkeypatch, [False], timeout=.25)
    result = computer.observe_directed(bounded_grounding_request("Find search", 5))
    assert result.visual_readiness.ready is False
    assert result.visual_readiness.reason == "timeout"
    assert capture.calls >= 2 and provider.payloads == []
    assert result.visual_latency_ms is None
    assert result.visual_provider_call_count == 0


def test_visual_readiness_foreground_change_never_calls_provider(monkeypatch) -> None:
    root = Node(name="Spotify Premium", handle=99, runtime_id=(99,))
    changed = Node(name="Other", handle=100, process_id=77, runtime_id=(100,))
    foreground = Mock(side_effect=[root, changed])
    computer, _capture, provider, _root = readiness_computer(
        monkeypatch, [False, True], foreground=foreground,
    )
    result = computer.observe_directed(bounded_grounding_request("Find search", 5))
    assert result.visual_readiness.reason == "foreground_changed"
    assert provider.payloads == []


def test_preclick_directed_observation_does_not_upload_after_trusted_context_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_root = Node(name="Spotify", handle=99, process_id=42, runtime_id=(99,))
    changed_root = Node(name="Other", handle=100, process_id=77, runtime_id=(100,))
    foreground = Mock(side_effect=[expected_root, changed_root])
    monkeypatch.setattr(windows, "_foreground", foreground)
    monkeypatch.setattr(windows_actions, "_foreground", lambda: expected_root)

    class Catalog:
        def identify(self, app_name, package_family):
            return "trusted_spotify_app"

    class Provider:
        name = "gemini"
        model = "test-model"
        pricing_class = "free"

        def __init__(self):
            self.calls = []

        def observe(self, screenshot, window, original_request, grounding=None):
            self.calls.append((screenshot, window, grounding))
            return VisualObservation((), self.name, self.model, directed_grounding=True)

    provider = Provider()
    capture = FakeCapture()
    computer = WindowsComputer(
        app_catalog=Catalog(), capture_service=capture, visual_provider=provider,
    )
    expected = Observation(
        "spotify.exe", "Spotify", process_id=42, application_id="trusted_spotify_app",
        observation_id="before", foreground_hwnd=99,
    )

    result = computer.observe_preclick_directed(
        bounded_grounding_request("Find the same visible actionable target", 5), expected,
    )

    assert result.error == "trusted_context_changed"
    assert provider.calls == []
    assert result.visual_provider_call_count == 0


def test_visual_readiness_uses_combined_structure_signals() -> None:
    dark_rich = replace(
        readiness_capture_diagnostics(informative=True),
        mean_luminance=8.0, near_black_percent=80.0,
    )
    bright_uniform = replace(
        readiness_capture_diagnostics(informative=False),
        mean_luminance=255.0, near_white_percent=100.0,
    )
    assert visual_frame_is_informative(visual_readiness_frame(dark_rich))
    assert not visual_frame_is_informative(visual_readiness_frame(bright_uniform))


def test_result_readiness_uses_pre_submit_baseline_and_sends_stable_final_once(monkeypatch) -> None:
    computer, capture, provider, _root = readiness_computer(monkeypatch, [True, True])
    computer.result_readiness_options = ResultReadinessOptions(.5, .1, 100)
    baseline = Observation(
        "spotify.exe", "Spotify", process_id=42, observation_id="pre-submit",
        screenshot=ScreenshotMetadata(
            "pre-submit", 99, Rect(0, 0, 100, 100), Rect(0, 0, 100, 100),
            100, 100, 96, 96, 1, 1,
        ), capture_diagnostics=readiness_capture_diagnostics(informative=False),
    )
    result = computer.observe_result_directed(
        bounded_grounding_request("Find submitted result", 5), baseline,
    )
    assert result.result_readiness.ready is True
    assert result.result_readiness.after_query_submit is True
    assert result.result_readiness.meaningful_change_seen is True
    assert result.result_readiness.stable is True
    assert result.result_readiness.reason == "changed_and_quiet_stable"
    assert result.result_readiness.first_transition_elapsed_ms == 0
    assert result.result_readiness.last_meaningful_change_elapsed_ms == 0
    assert result.result_readiness.required_quiet_ms == 100
    assert result.result_readiness.observed_quiet_ms == 100
    assert capture.calls == 2 and len(provider.payloads) == 1
    import hashlib
    assert hashlib.sha256(provider.payloads[0]).hexdigest() == result.visual_request_fingerprint.screenshot_sha256


def test_result_readiness_unchanged_frame_times_out_before_provider(monkeypatch) -> None:
    computer, capture, provider, _root = readiness_computer(monkeypatch, [True], timeout=.25)
    computer.result_readiness_options = ResultReadinessOptions(.25, .1)
    baseline = Observation(
        "spotify.exe", "Spotify", process_id=42, observation_id="pre-submit",
        capture_diagnostics=readiness_capture_diagnostics(informative=True),
    )
    result = computer.observe_result_directed(
        bounded_grounding_request("Find submitted result", 5), baseline,
    )
    assert result.result_readiness.ready is False
    assert result.result_readiness.reason == "timeout_before_quiet_stable"
    assert result.visual_provider_call_count == 0
    assert provider.payloads == [] and capture.calls >= 2


def test_result_readiness_richness_ratios_and_simplification_require_multiple_signals() -> None:
    baseline = visual_readiness_frame(readiness_state_diagnostics("autocomplete"))
    simplified = visual_readiness_frame(readiness_state_diagnostics("blank"))
    ratios = result_readiness_richness_ratios(baseline, simplified)
    assert ratios.png_complexity == .375
    assert ratios.unique_colors == .5
    assert ratios.luminance_stddev == .5
    assert result_simplification_signal_count(ratios, .75) == 3
    assert result_simplification_signal_count(ratios, .3) == 0


def test_result_readiness_defaults_to_seven_seconds_with_quiet_and_grace() -> None:
    options = ResultReadinessOptions()
    assert options.timeout_seconds == 7
    assert options.settle_quiet_ms == 1200
    assert options.transition_grace_ms == 2500
    assert options.simplification_ratio_threshold == .75
    assert options.simplification_min_signals == 2


def test_result_readiness_fast_final_transition_waits_for_quiet_interval(monkeypatch) -> None:
    computer, capture, provider, _root = readiness_computer(
        monkeypatch, ["results"] * 10,
    )
    result = computer.observe_result_directed(
        bounded_grounding_request("Find submitted result", 5),
        result_readiness_baseline(),
    )
    readiness = result.result_readiness
    assert readiness.ready and readiness.reason == "changed_and_quiet_stable"
    assert readiness.first_transition_elapsed_ms == 0
    assert readiness.last_meaningful_change_elapsed_ms == 0
    assert readiness.observed_quiet_ms == 1200
    assert readiness.quiet_timer_reset_count == 0
    assert readiness.state == "ready"
    assert not readiness.transitional_simplification
    assert not readiness.awaiting_followup_transition
    assert capture.calls == 7 and len(provider.payloads) == 1


def test_result_readiness_waits_past_stable_blank_then_uses_final_result_frame(
    monkeypatch, tmp_path,
) -> None:
    from computer.visual_providers.common import png_bytes
    from main import _save_exact_result_capture

    computer, capture, provider, _root = readiness_computer(
        monkeypatch,
        ["blank"] * 10 + ["results"] * 10,
        retain_debug_capture=True,
    )
    result = computer.observe_result_directed(
        bounded_grounding_request("Find submitted result", 5),
        result_readiness_baseline(),
    )
    readiness = result.result_readiness
    assert readiness.ready and readiness.reason == "changed_and_quiet_stable"
    assert readiness.state == "ready"
    assert readiness.first_transition_elapsed_ms == 0
    assert readiness.last_meaningful_change_elapsed_ms == 2000
    assert readiness.quiet_timer_reset_count == 1
    assert readiness.required_quiet_ms == 1200
    assert readiness.observed_quiet_ms == 1200
    assert readiness.baseline_changed and readiness.recent_frame_changed is False
    assert readiness.transitional_simplification is False
    assert readiness.simplification_signal_count == 0
    assert readiness.followup_transition_seen
    assert not readiness.awaiting_followup_transition
    assert readiness.transition_grace_elapsed_ms == 800
    assert readiness.richness_ratios.png_complexity > .75
    assert readiness.richness_ratios.unique_colors > .75
    assert readiness.richness_ratios.luminance_stddev > .75
    assert readiness.elapsed_ms == 3200
    assert capture.calls == 17 and len(provider.payloads) == 1

    accepted_capture = computer.take_debug_capture()
    assert accepted_capture is capture.captures[-1]
    expected_provider_bytes = png_bytes(accepted_capture)
    assert provider.payloads[0] == expected_provider_bytes
    import hashlib
    digest = hashlib.sha256(expected_provider_bytes).hexdigest()
    fingerprint = result.visual_request_fingerprint
    assert fingerprint.screenshot_sha256 == digest
    assert fingerprint.encoded_image_byte_length == len(expected_provider_bytes)

    destination = tmp_path / "accepted-result.png"
    saved = _save_exact_result_capture(result, accepted_capture, destination)
    assert destination.read_bytes() == provider.payloads[0]
    assert saved["sha256"] == digest
    assert saved["byte_length"] == len(provider.payloads[0])
    assert saved["matches_request_fingerprint"] is True
    accepted_capture.discard()


def test_result_readiness_resets_quiet_timer_for_each_async_transition(monkeypatch) -> None:
    computer, capture, provider, _root = readiness_computer(
        monkeypatch,
        ["blank"] * 7 + ["loading", "loading", "results"] + ["results"] * 8,
    )
    result = computer.observe_result_directed(
        bounded_grounding_request("Find submitted result", 5),
        result_readiness_baseline(),
    )
    readiness = result.result_readiness
    assert readiness.ready is True
    assert readiness.followup_transition_seen
    assert readiness.transition_grace_elapsed_ms == 200
    assert readiness.first_transition_elapsed_ms == 0
    assert readiness.last_meaningful_change_elapsed_ms == 1800
    assert readiness.quiet_timer_reset_count == 2
    assert readiness.observed_quiet_ms == 1200
    assert readiness.elapsed_ms == 3000
    assert capture.calls == 16 and len(provider.payloads) == 1


def test_result_readiness_persistent_simplified_candidate_fails_closed_at_grace(
    monkeypatch,
) -> None:
    computer, capture, provider, _root = readiness_computer(
        monkeypatch, ["blank"] * 30, result_timeout=7, grace_ms=1000,
    )
    result = computer.observe_result_directed(
        bounded_grounding_request("Find submitted result", 5),
        result_readiness_baseline(),
    )
    readiness = result.result_readiness
    assert not readiness.ready
    assert readiness.reason == "timeout_transitional_simplification"
    assert readiness.state == "timeout"
    assert readiness.transitional_simplification
    assert readiness.simplification_signal_count == 3
    assert readiness.awaiting_followup_transition
    assert not readiness.followup_transition_seen
    assert readiness.transition_grace_ms == 1000
    assert readiness.transition_grace_elapsed_ms == 1000
    assert readiness.elapsed_ms == 2200
    assert result.visual_provider_call_count == 0
    assert provider.payloads == []
    assert capture.calls == 12


def test_result_readiness_change_just_before_quiet_threshold_restarts_timer(monkeypatch) -> None:
    computer, capture, provider, _root = readiness_computer(
        monkeypatch,
        ["blank", "blank", "blank", "blank", "blank", "results"] + ["results"] * 9,
    )
    result = computer.observe_result_directed(
        bounded_grounding_request("Find submitted result", 5),
        result_readiness_baseline(),
    )
    readiness = result.result_readiness
    assert readiness.ready is True
    assert readiness.last_meaningful_change_elapsed_ms == 1000
    assert readiness.quiet_timer_reset_count == 1
    assert readiness.elapsed_ms == 2200
    assert readiness.observed_quiet_ms == 1200
    assert capture.calls == 12 and len(provider.payloads) == 1


def test_result_readiness_without_baseline_transition_times_out_without_provider(monkeypatch) -> None:
    computer, capture, provider, _root = readiness_computer(
        monkeypatch, ["autocomplete"] * 10, result_timeout=.5,
    )
    result = computer.observe_result_directed(
        bounded_grounding_request("Find submitted result", 5),
        result_readiness_baseline(),
    )
    readiness = result.result_readiness
    assert readiness.ready is False
    assert readiness.reason == "timeout_before_quiet_stable"
    assert readiness.meaningful_change_seen is False
    assert readiness.informative_frame_seen is True
    assert readiness.stable_frame_seen is True
    assert readiness.first_transition_elapsed_ms is None
    assert provider.payloads == [] and result.visual_provider_call_count == 0
    assert capture.calls == 3


def test_result_readiness_changed_but_never_stable_times_out_without_provider(monkeypatch) -> None:
    computer, capture, provider, _root = readiness_computer(
        monkeypatch,
        ["blank", "loading", "results", "results_late", "results_latest", "changed_a"],
        result_timeout=.5,
    )
    result = computer.observe_result_directed(
        bounded_grounding_request("Find submitted result", 5),
        result_readiness_baseline(),
    )
    readiness = result.result_readiness
    assert readiness.ready is False
    assert readiness.reason == "timeout_before_quiet_stable"
    assert readiness.meaningful_change_seen is True
    assert readiness.stable is False and readiness.stable_frame_seen is False
    assert readiness.last_meaningful_change_elapsed_ms == 400
    assert provider.payloads == [] and result.visual_provider_call_count == 0
    assert capture.calls == 3


def test_result_readiness_timeout_before_quiet_threshold_never_calls_provider(monkeypatch) -> None:
    computer, capture, provider, _root = readiness_computer(
        monkeypatch, ["blank"] * 10, result_timeout=.8,
    )
    result = computer.observe_result_directed(
        bounded_grounding_request("Find submitted result", 5),
        result_readiness_baseline(),
    )
    readiness = result.result_readiness
    assert readiness.ready is False
    assert readiness.reason == "timeout_before_quiet_stable"
    assert readiness.required_quiet_ms == 1200
    assert readiness.observed_quiet_ms == 800
    assert readiness.informative_frame_seen and readiness.stable_frame_seen
    assert provider.payloads == [] and result.visual_provider_call_count == 0
    assert capture.calls == 4


def test_jev_rejects_invented_visual_coordinate_choice(monkeypatch: pytest.MonkeyPatch) -> None:
    _computer, observation, _capture, _provider, _click = hybrid_computer(monkeypatch)

    def malicious(payload):
        keys = payload["questions"]["next_action"]["criteria"]
        return {"model": "jev-1.13.0", "answers": {"next_action": {
            "type": "choice", "choice": "visual_100_200", "confidence": .99,
            "probabilities": {key: (1.0 if key == "finish" else 0.0) for key in keys},
        }}}

    result = JevDecisionMaker(Mock(evaluate=Mock(side_effect=malicious))).decide(
        "click Search", observation,
    )
    assert result.action is None and result.error == "invalid_response"
