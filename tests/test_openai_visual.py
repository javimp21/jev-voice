"""Mocked OpenAI visual provider contract tests; never call a paid API."""

from __future__ import annotations

import json
from unittest.mock import Mock

import pytest
from PIL import Image

from computer.models import Observation, Rect, ScreenshotMetadata
from computer.visual import ScreenshotCapture, VisualProviderFailure, validate_visual_candidates
from computer.visual_providers import VisualProviderConfigurationError, visual_provider_from_environment
from computer.visual_providers.openai import OpenAIVisualObserver


def capture(width: int = 1000, height: int = 500) -> ScreenshotCapture:
    metadata = ScreenshotMetadata(
        "snapshot", 10, Rect(0, 0, width, height), Rect(0, 0, width, height),
        width, height, 96, 96, 1, 1,
    )
    return ScreenshotCapture(metadata, Image.new("RGB", (width, height), "white"))


def element(*, label="Buscar", role="navigation_item", left=100, top=200,
            right=300, bottom=400, clickable=True, parent="Navigation"):
    return {
        "label": label, "role": role,
        "box": {"left": left, "top": top, "right": right, "bottom": bottom},
        "clickable": clickable, "parent": parent,
    }


def response(elements, *, usage=None):
    value = {
        "status": "completed",
        "output": [{"type": "message", "content": [{
            "type": "output_text", "text": json.dumps({"elements": elements}, ensure_ascii=False),
        }]}],
    }
    if usage is not None:
        value["usage"] = usage
    return value


def observer(result, *, max_elements=40, clock=None):
    transport = Mock()
    transport.create.return_value = result
    times = iter(clock or (1.0, 1.85))
    return OpenAIVisualObserver(
        "secret-api-key", max_elements=max_elements, transport=transport,
        clock=lambda: next(times),
    ), transport


def test_request_contains_png_and_strict_schema_without_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "secret-api-key"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    provider, transport = observer(response([element()]))
    screenshot = capture()
    result = provider.observe(screenshot, Observation("app.exe", "Private title"), f"find {secret}")
    payload, passed_key, timeout = transport.create.call_args.args
    serialized = json.dumps(payload)
    image = payload["input"][0]["content"][1]
    assert image["type"] == "input_image"
    assert image["image_url"].startswith("data:image/png;base64,")
    assert image["detail"] == "high"
    assert payload["text"]["format"]["type"] == "json_schema"
    assert payload["text"]["format"]["strict"] is True
    assert payload["store"] is False
    assert secret not in serialized and "Private title" not in serialized
    assert passed_key == secret and timeout == 20
    assert result.execution_authorized is False
    screenshot.discard()


def test_normalized_boxes_convert_to_capture_pixels_and_preserve_unicode() -> None:
    provider, _transport = observer(response([element(label="Tu biblioteca á", left=100, top=200,
                                                       right=300, bottom=400)]))
    screenshot = capture()
    result = provider.observe(screenshot, Observation("", ""), "inspect")
    candidate = result.candidates[0]
    assert candidate.label == "Tu biblioteca á"
    assert candidate.rectangle == Rect(100, 100, 300, 200)
    assert candidate.confidence is None
    screenshot.discard()


@pytest.mark.parametrize("bad", [
    element(left=500, right=500),
    element(left=-1),
    element(right=1001),
    {"label": "x"},
    {**element(), "unexpected": True},
])
def test_malformed_or_out_of_range_provider_elements_fail_closed(bad) -> None:
    provider, _transport = observer(response([bad]))
    screenshot = capture()
    with pytest.raises(VisualProviderFailure, match="invalid_response"):
        provider.observe(screenshot, Observation("", ""), "inspect")
    screenshot.discard()


@pytest.mark.parametrize("raw", [
    {"status": "completed", "output": []},
    {"status": "failed", "output": []},
    {"status": "completed", "output": [{"type": "message", "content": [{"type": "refusal"}]}]},
])
def test_response_envelope_validation(raw) -> None:
    provider, _transport = observer(raw)
    screenshot = capture()
    with pytest.raises(VisualProviderFailure):
        provider.observe(screenshot, Observation("", ""), "inspect")
    screenshot.discard()


def test_max_elements_latency_and_sanitized_usage() -> None:
    raw_usage = {"input_tokens": 120, "output_tokens": 30, "total_tokens": 150,
                 "private": "do-not-log", "cached_tokens": -1}
    provider, _transport = observer(response([element(label=f"Item {i}") for i in range(6)], usage=raw_usage),
                                    max_elements=3, clock=(10.0, 10.321))
    screenshot = capture()
    result = provider.observe(screenshot, Observation("", ""), "inspect")
    assert len(result.candidates) == 3
    assert result.latency_ms == 321
    assert dict(result.usage) == {"input_tokens": 120, "output_tokens": 30, "total_tokens": 150}
    assert "private" not in repr(result)
    screenshot.discard()


def test_long_labels_are_bounded_and_secrets_redacted_after_generic_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "private-token-value"
    monkeypatch.setenv("VISIBLE_SECRET", secret)
    provider, _transport = observer(response([element(label=secret + ("A" * 300))]))
    screenshot = capture()
    result = provider.observe(screenshot, Observation("", ""), "inspect")

    class Redactor:
        def clean(self, value):
            return value.replace(secret, "[REDACTED]")

    validated = validate_visual_candidates(result.candidates, screenshot.metadata, redactor=Redactor())
    assert len(validated[0].label) <= 160
    assert secret not in validated[0].label
    assert "[REDACTED]" in validated[0].label
    screenshot.discard()


@pytest.mark.parametrize("failure", [VisualProviderFailure("api_error"), TimeoutError()])
def test_api_and_timeout_failures_are_sanitized(failure) -> None:
    provider, transport = observer(response([]))
    transport.create.side_effect = failure
    screenshot = capture()
    with pytest.raises(VisualProviderFailure) as caught:
        provider.observe(screenshot, Observation("", ""), "inspect")
    assert caught.value.code in {"api_error", "timeout"}
    assert "secret-api-key" not in str(caught.value)
    screenshot.discard()


def test_provider_environment_is_optional_and_actions_cannot_be_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("VISUAL_PROVIDER", "OPENAI_API_KEY", "VISUAL_ACTIONS_ENABLED"):
        monkeypatch.delenv(name, raising=False)
    assert visual_provider_from_environment() is None
    monkeypatch.setenv("VISUAL_PROVIDER", "openai")
    with pytest.raises(VisualProviderConfigurationError):
        visual_provider_from_environment()
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("VISUAL_ACTIONS_ENABLED", "true")
    with pytest.raises(VisualProviderConfigurationError, match="cannot be enabled"):
        visual_provider_from_environment()
