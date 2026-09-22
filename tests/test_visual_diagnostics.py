"""Repeated visual diagnostics are read-only and reuse one in-memory capture."""

import json
from unittest.mock import Mock

from PIL import Image

from computer.models import Observation, Rect, ScreenshotMetadata
from computer.visual import ScreenshotCapture, VisualProviderFailure
from computer.visual_diagnostics import (
    DEFAULT_DIAGNOSTIC_OBJECTIVES, diagnose_visual_grounding,
)
from computer.visual_providers.gemini import GeminiVisualObserver


def _capture() -> ScreenshotCapture:
    metadata = ScreenshotMetadata(
        "same-snapshot", 7, Rect(0, 0, 200, 100), Rect(0, 0, 200, 100),
        200, 100, 96, 96, 1, 1, masked_regions=1,
    )
    return ScreenshotCapture(metadata, Image.new("RGB", (200, 100), "navy"))


def _response(elements: list[dict[str, object]]) -> dict[str, object]:
    return {"candidates": [{"finishReason": "STOP", "content": {"parts": [{
        "text": json.dumps({"elements": elements}),
    }]}}], "usageMetadata": {"totalTokenCount": 12}}


def _element() -> dict[str, object]:
    return {
        "label": "Search", "role": "search_field", "clickable": True,
        "box": {"left": 100, "top": 100, "right": 400, "bottom": 300},
    }


def test_default_diagnostic_makes_12_independent_calls_over_identical_image() -> None:
    transport = Mock()
    transport.create.side_effect = [_response([]), _response([_element()])] * 6
    provider = GeminiVisualObserver("secret", transport=transport)
    shot = _capture()
    report = diagnose_visual_grounding(provider, shot, Observation("app", "Window"))

    assert report["objective_count"] == len(DEFAULT_DIAGNOSTIC_OBJECTIVES) == 4
    assert report["provider_call_count"] == 12
    assert transport.create.call_count == 12
    uploaded = [call.args[0]["contents"][0]["parts"][1]["inlineData"]["data"]
                for call in transport.create.call_args_list]
    assert len(set(uploaded)) == 1
    fingerprints = [item["request_fingerprint"]["screenshot_sha256"]
                    for item in report["results"]]
    assert len(set(fingerprints)) == 1
    assert {item["status"] for item in report["results"]} == {
        "success_empty", "success_with_candidates",
    }
    serialized = json.dumps(report)
    assert "secret" not in serialized and uploaded[0] not in serialized
    assert report["computer_actions_enabled"] is False
    shot.discard()


def test_provider_error_does_not_abort_remaining_diagnostic_calls() -> None:
    transport = Mock()
    transport.create.side_effect = [VisualProviderFailure("timeout"), _response([])]
    provider = GeminiVisualObserver("secret", transport=transport)
    shot = _capture()
    report = diagnose_visual_grounding(
        provider, shot, Observation("app", "Window"), ("Find search",), runs=2,
    )
    assert transport.create.call_count == 2
    assert report["results"][0]["provider_error"]["category"] == "timeout"
    assert report["results"][1]["status"] == "success_empty"
    shot.discard()
