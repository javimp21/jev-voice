"""Deterministic tests for one-shot Gemini to OpenAI visual failover."""

from __future__ import annotations

import json
from unittest.mock import Mock

from PIL import Image
import pytest

from computer.models import Observation, Rect, ScreenshotMetadata
from computer.visual import (
    ScreenshotCapture, VisualGroundingRequest, VisualProviderFailure, bounded_grounding_request,
)
from computer.visual_providers.failover import GeminiVisualGrounder
from computer.visual_providers.openai import OpenAIVisualGrounder


def capture() -> ScreenshotCapture:
    return ScreenshotCapture(
        ScreenshotMetadata(
            "snapshot-1", 100, Rect(0, 0, 800, 600), Rect(0, 0, 800, 600),
            800, 600, 96, 96, 1.0, 1.0,
        ),
        Image.new("RGB", (800, 600), "white"),
    )


def candidate(label: str = "Iago") -> dict[str, object]:
    return {
        "label": label, "role": "list_item",
        "box": {"left": 100, "top": 100, "right": 400, "bottom": 220},
        "clickable": True,
    }


def gemini_response(elements: list[dict[str, object]]) -> dict[str, object]:
    return {"candidates": [{
        "finishReason": "STOP",
        "content": {"parts": [{"text": json.dumps({"elements": elements})}]},
    }]}


def openai_response(elements: list[dict[str, object]]) -> dict[str, object]:
    return {"status": "completed", "output": [{
        "type": "message", "content": [{
            "type": "output_text", "text": json.dumps({"elements": elements}),
        }],
    }]}


def configured(primary_result, fallback_result=None):
    primary_transport = Mock()
    if isinstance(primary_result, BaseException):
        primary_transport.create.side_effect = primary_result
    else:
        primary_transport.create.return_value = primary_result
    fallback_transport = Mock()
    if isinstance(fallback_result, BaseException):
        fallback_transport.create.side_effect = fallback_result
    else:
        fallback_transport.create.return_value = fallback_result
    fallback = OpenAIVisualGrounder(
        "openai-key", model="gpt-5.6-luna", transport=fallback_transport,
    )
    provider = GeminiVisualGrounder(
        "gemini-key", model="gemini-3.5-flash-lite",
        openai_fallback=fallback, transport=primary_transport,
    )
    return provider, primary_transport, fallback_transport


def test_gemini_success_with_candidates_does_not_call_openai() -> None:
    provider, primary, fallback = configured(gemini_response([candidate()]))
    shot = capture()
    result = provider.observe(
        shot, Observation("app", "Window"), "ignored",
        bounded_grounding_request("Find Iago's conversation", 3),
    )
    assert primary.create.call_count == 1 and fallback.create.call_count == 0
    assert result.provider == "gemini" and not result.provider_failover_used
    assert result.provider_attempts[0].result_class == "success_with_candidates"
    assert result.candidates[0].confidence is None and not result.execution_authorized
    shot.discard()


def test_gemini_valid_empty_result_does_not_call_openai() -> None:
    provider, primary, fallback = configured(gemini_response([]), openai_response([candidate()]))
    shot = capture()
    result = provider.observe(
        shot, Observation("app", "Window"), "",
        bounded_grounding_request("Find the conversation", 3),
    )
    assert result.candidates == ()
    assert result.provider_attempts[0].result_class == "success_empty"
    assert primary.create.call_count == 1 and fallback.create.call_count == 0
    shot.discard()


@pytest.mark.parametrize(("failure", "failover_reason"), [
    (VisualProviderFailure("timeout"), "timeout"),
    (VisualProviderFailure("network_error"), "connection"),
    (VisualProviderFailure("rate_limited", http_status=429), "rate_limit"),
    (VisualProviderFailure("server_error", http_status=500), "server"),
    (VisualProviderFailure("malformed_response"), "invalid_response"),
    (VisualProviderFailure("invalid_response"), "invalid_response"),
])
def test_recoverable_gemini_failure_calls_openai_once_with_same_request(
    failure: VisualProviderFailure, failover_reason: str,
) -> None:
    provider, primary, fallback = configured(failure, openai_response([candidate()]))
    shot = capture()
    grounding = bounded_grounding_request("Find the conversation Iago", 3)

    result = provider.observe(shot, Observation("app", "Window"), "ignored text", grounding)

    gemini_payload = primary.create.call_args.args[0]
    openai_payload = fallback.create.call_args.args[0]
    gemini_prompt = gemini_payload["contents"][0]["parts"][0]["text"]
    openai_content = openai_payload["input"][0]["content"]
    openai_prompt = openai_content[0]["text"]
    assert primary.create.call_count == 1 and fallback.create.call_count == 1
    assert gemini_prompt == openai_prompt
    assert grounding.objective in gemini_prompt
    assert "at most 3" in gemini_prompt
    assert gemini_payload["contents"][0]["parts"][1]["inlineData"]["data"] == (
        openai_content[1]["image_url"].split(",", 1)[1]
    )
    assert result.provider == "openai" and result.model == "gpt-5.6-luna"
    assert result.provider_failover_used is True
    assert result.provider_failover_reason == failover_reason
    assert [item.provider for item in result.provider_attempts] == ["gemini", "openai"]
    assert [item.result_class for item in result.provider_attempts] == [
        "provider_failure", "success_with_candidates",
    ]
    assert all(item.elapsed_ms >= 0 for item in result.provider_attempts)
    assert result.candidates[0].confidence is None and not result.execution_authorized
    shot.discard()


def test_verification_only_shape_survives_existing_gemini_to_openai_failover() -> None:
    active = {
        **candidate(), "activity": "active", "selection_state": "selected",
        "region": "header",
    }
    provider, primary, fallback = configured(
        VisualProviderFailure("timeout"), openai_response([active]),
    )
    shot = capture()
    grounding = VisualGroundingRequest("Verify Iago is active.", 3, verification_only=True)

    result = provider.observe(shot, Observation("app", "Window"), "", grounding)

    gemini_payload = primary.create.call_args.args[0]
    openai_payload = fallback.create.call_args.args[0]
    gemini_schema = gemini_payload["generationConfig"]["responseJsonSchema"]
    openai_schema = openai_payload["text"]["format"]["schema"]
    assert "activity" in gemini_schema["properties"]["elements"]["items"]["required"]
    assert "activity" in openai_schema["properties"]["elements"]["items"]["required"]
    assert "selection_state" in gemini_schema["properties"]["elements"]["items"]["required"]
    assert "region" in openai_schema["properties"]["elements"]["items"]["required"]
    assert result.candidates[0].activity == "active"
    assert [attempt.provider for attempt in result.provider_attempts] == ["gemini", "openai"]
    assert result.provider_failover_used is True
    shot.discard()


def test_openai_success_empty_is_accepted_as_empty_not_provider_failure() -> None:
    provider, _primary, fallback = configured(
        VisualProviderFailure("timeout"), openai_response([]),
    )
    shot = capture()
    result = provider.observe(
        shot, Observation("app", "Window"), "",
        bounded_grounding_request("Find a search field", 2),
    )
    assert result.provider == "openai" and result.candidates == ()
    assert result.provider_attempts[-1].result_class == "success_empty"
    assert result.provider_failover_used and fallback.create.call_count == 1
    shot.discard()


def test_both_provider_failures_return_safe_incomplete_failure_diagnostics() -> None:
    provider, primary, fallback = configured(
        VisualProviderFailure("timeout"),
        VisualProviderFailure("server_error", http_status=503, provider_code="unavailable"),
    )
    shot = capture()
    with pytest.raises(VisualProviderFailure) as caught:
        provider.observe(
            shot, Observation("app", "Window"), "",
            bounded_grounding_request("Find Iago", 5),
        )
    error = caught.value
    assert primary.create.call_count == fallback.create.call_count == 1
    assert error.selected_visual_provider is None
    assert error.provider_failover_used is True
    assert error.provider_failover_reason == "timeout"
    assert [item.provider for item in error.provider_attempts] == ["gemini", "openai"]
    assert error.diagnostic.provider_name == "openai"
    assert error.diagnostic.provider_model == "gpt-5.6-luna"
    assert error.diagnostic.http_status == 503
    assert "openai-key" not in repr(error.diagnostic)
    shot.discard()


def test_nonrecoverable_gemini_authentication_error_does_not_fallback() -> None:
    provider, primary, fallback = configured(
        VisualProviderFailure("authentication_error", http_status=401),
        openai_response([candidate()]),
    )
    shot = capture()
    with pytest.raises(VisualProviderFailure) as caught:
        provider.observe(
            shot, Observation("app", "Window"), "",
            bounded_grounding_request("Find Iago", 5),
        )
    assert caught.value.diagnostic.category == "authentication_error"
    assert caught.value.provider_attempts[0].provider == "gemini"
    assert not caught.value.provider_failover_used
    assert primary.create.call_count == 1 and fallback.create.call_count == 0
    shot.discard()
