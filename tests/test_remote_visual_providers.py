"""Deterministic OpenRouter/DeepSeek visual adapter tests; no live API calls."""

from __future__ import annotations

import json
from unittest.mock import Mock

from PIL import Image
import pytest

from computer.models import Observation, Rect, ScreenshotMetadata
from computer.visual import ScreenshotCapture, VisualGroundingRequest, VisualProviderFailure
from computer.visual_providers import (
    VisualProviderConfigurationError, visual_provider_from_environment,
)
from computer.visual_providers.common import (
    candidate_from_normalized, strict_visual_json, visual_schema,
)
from computer.visual_providers.deepseek import DEEPSEEK_VISUAL_MODEL, DeepSeekVisualObserver
from computer.visual_providers.openai import OpenAIVisualObserver
from computer.visual_providers.openrouter import (
    OPENROUTER_FREE_VISUAL_MODEL, OPENROUTER_LING_VISUAL_MODEL,
    SUPPORTED_FREE_VISUAL_MODELS, OpenRouterVisualObserver,
)


def capture(width: int = 1000, height: int = 500) -> ScreenshotCapture:
    metadata = ScreenshotMetadata(
        "snapshot", 10, Rect(0, 0, width, height), Rect(0, 0, width, height),
        width, height, 96, 96, 1, 1,
    )
    return ScreenshotCapture(metadata, Image.new("RGB", (width, height), "white"))


def element(*, label: str = "Buscar", left: int = 100, top: int = 200,
            right: int = 300, bottom: int = 400) -> dict[str, object]:
    return {
        "label": label, "role": "navigation_item",
        "box": {"left": left, "top": top, "right": right, "bottom": bottom},
        "clickable": True, "parent": "Navigation",
    }


def response(elements: list[dict[str, object]]) -> dict[str, object]:
    return {
        "choices": [{"message": {"content": json.dumps(
            {"elements": elements}, ensure_ascii=False,
        )}}],
        "usage": {"prompt_tokens": 40, "completion_tokens": 20, "total_tokens": 60,
                  "private": "not-diagnostic"},
    }


def tool_response(elements: list[dict[str, object]]) -> dict[str, object]:
    return {
        "choices": [{"message": {"content": None, "tool_calls": [{
            "type": "function",
            "function": {
                "name": "report_visible_ui_elements",
                "arguments": json.dumps({"elements": elements}, ensure_ascii=False),
            },
        }]}}],
        "usage": {"prompt_tokens": 40, "completion_tokens": 20, "total_tokens": 60},
    }


def field_value_response(value: str | None) -> dict[str, object]:
    return {"choices": [{"message": {"content": json.dumps({"field_value": value})}}]}


def field_value_tool_response(value: str | None) -> dict[str, object]:
    return {"choices": [{"message": {"content": None, "tool_calls": [{
        "type": "function", "function": {
            "name": "report_visible_field_value",
            "arguments": json.dumps({"field_value": value}),
        },
    }]}}]}


def make(provider_type, result, **kwargs):
    transport = Mock()
    transport.create.return_value = result
    provider = provider_type(
        "provider-secret", transport=transport, clock=iter((1.0, 1.25)).__next__, **kwargs,
    )
    return provider, transport


@pytest.mark.parametrize("provider_type", [OpenRouterVisualObserver, DeepSeekVisualObserver])
def test_chat_adapters_send_png_and_return_observation_only(provider_type) -> None:
    provider, transport = make(provider_type, response([element()]))
    shot = capture()
    result = provider.observe(shot, Observation("", ""), "inspect")
    payload, key, timeout = transport.create.call_args.args
    image = payload["messages"][0]["content"][1]
    assert image["type"] == "image_url"
    assert image["image_url"]["url"].startswith("data:image/png;base64,")
    assert payload["response_format"] == {"type": "json_object"}
    assert key == "provider-secret" and timeout == 20
    assert result.execution_authorized is False
    assert result.candidates[0].rectangle == Rect(100, 100, 300, 200)
    assert result.candidates[0].confidence is None
    assert dict(result.usage) == {"prompt_tokens": 40, "completion_tokens": 20, "total_tokens": 60}
    assert "provider-secret" not in repr(result)
    assert "base64" not in repr(result)
    shot.discard()


def test_openrouter_is_pinned_free_without_model_fallback() -> None:
    provider, transport = make(OpenRouterVisualObserver, response([]))
    shot = capture()
    result = provider.observe(shot, Observation("", ""), "inspect")
    payload = transport.create.call_args.args[0]
    assert payload["model"] == OPENROUTER_FREE_VISUAL_MODEL
    assert payload["provider"] == {
        "allow_fallbacks": False, "require_parameters": True, "data_collection": "deny",
    }
    assert "models" not in payload
    assert result.pricing_class == "free"
    shot.discard()


def test_openrouter_explicit_allow_omits_only_data_collection_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_CREDENTIAL", "private-visible-token")
    deny, deny_transport = make(OpenRouterVisualObserver, response([]))
    allow, allow_transport = make(
        OpenRouterVisualObserver, response([]), data_collection_policy="allow",
    )
    deny_shot, allow_shot = capture(), capture()
    deny_shot.image.putpixel((0, 0), (0, 0, 0))
    allow_shot.image.putpixel((0, 0), (0, 0, 0))
    request = "inspect private-visible-token"
    deny_result = deny.observe(deny_shot, Observation("", ""), request)
    allow_result = allow.observe(allow_shot, Observation("", ""), request)
    deny_payload = deny_transport.create.call_args.args[0]
    allow_payload = allow_transport.create.call_args.args[0]

    assert deny_payload["provider"]["data_collection"] == "deny"
    assert "data_collection" not in allow_payload["provider"]
    assert allow_payload["provider"]["allow_fallbacks"] is False
    assert "models" not in allow_payload
    assert allow_payload["model"] == OPENROUTER_FREE_VISUAL_MODEL
    assert allow_payload["messages"] == deny_payload["messages"]
    assert "private-visible-token" not in json.dumps(allow_payload)
    assert allow_result.execution_authorized is False
    assert deny_result.execution_authorized is False
    deny_shot.discard()
    allow_shot.discard()


def test_openrouter_data_collection_policy_fails_closed() -> None:
    with pytest.raises(ValueError, match="data_collection_policy"):
        OpenRouterVisualObserver("key", data_collection_policy="sometimes")
    with pytest.raises(ValueError, match="supported free image model"):
        OpenRouterVisualObserver(
            "key", model="paid/model", data_collection_policy="allow",
        )


def test_ling_uses_one_forced_schema_tool_without_parallel_calls_or_fallback() -> None:
    provider, transport = make(
        OpenRouterVisualObserver, tool_response([element(label="Tu biblioteca")]),
        model=OPENROUTER_LING_VISUAL_MODEL,
    )
    shot = capture()
    result = provider.observe(shot, Observation("", ""), "inspect")
    payload = transport.create.call_args.args[0]
    assert payload["model"] == "inclusionai/ling-3.0-flash-vl:free"
    assert "response_format" not in payload
    assert payload["tool_choice"] == {
        "type": "function", "function": {"name": "report_visible_ui_elements"},
    }
    assert "parallel_tool_calls" not in payload
    assert len(payload["tools"]) == 1
    assert payload["tools"][0]["function"]["parameters"]["additionalProperties"] is False
    assert payload["provider"]["allow_fallbacks"] is False
    assert payload["provider"]["require_parameters"] is True
    assert payload["provider"]["data_collection"] == "deny"
    assert "models" not in payload
    assert result.candidates[0].label == "Tu biblioteca"
    assert result.candidates[0].confidence is None
    assert result.execution_authorized is False
    shot.discard()


def test_openrouter_verification_returns_typed_active_context_fields() -> None:
    raw = element(label="Iago")
    raw.pop("parent")
    raw.update({
        "activity": "unknown", "selection_state": "selected", "region": "navigation",
    })
    provider, transport = make(
        OpenRouterVisualObserver, tool_response([raw]), model=OPENROUTER_LING_VISUAL_MODEL,
    )
    shot = capture()
    grounding = VisualGroundingRequest(
        'Determine whether "Iago" is active, not merely visible.', 5,
        verification_only=True,
    )

    result = provider.observe(shot, Observation("app", "Window"), "ignored", grounding)

    payload = transport.create.call_args.args[0]
    prompt = payload["messages"][0]["content"][0]["text"]
    schema = payload["tools"][0]["function"]["parameters"]["properties"]["elements"]["items"]
    assert "not merely visible" in prompt
    assert "activity" in schema["required"]
    assert "selection_state" in schema["required"]
    assert "region" in schema["required"]
    assert "parent" not in schema["properties"]
    assert result.candidates[0].activity == "unknown"
    assert result.candidates[0].selection_state.value == "selected"
    assert result.candidates[0].region.value == "navigation"
    assert result.directed_grounding and not result.execution_authorized
    shot.discard()


def test_ling_is_allowlisted_alongside_gemma() -> None:
    assert SUPPORTED_FREE_VISUAL_MODELS == {
        OPENROUTER_FREE_VISUAL_MODEL, OPENROUTER_LING_VISUAL_MODEL,
    }


@pytest.mark.parametrize("bad", [
    {"choices": [{"message": {"content": '{"elements": []}'}}]},
    {"choices": [{"message": {"tool_calls": []}}]},
    {"choices": [{"message": {"tool_calls": [
        {"function": {"name": "report_visible_ui_elements", "arguments": '{"elements":[]}'}},
        {"function": {"name": "report_visible_ui_elements", "arguments": '{"elements":[]}'}},
    ]}}]},
    {"choices": [{"message": {"tool_calls": [{"function": {
        "name": "wrong_tool", "arguments": '{"elements": []}',
    }}]}}]},
    {"choices": [{"message": {"tool_calls": [{"function": {
        "name": "report_visible_ui_elements", "arguments": "```json\n{}\n```",
    }}]}}]},
    {"choices": [{"message": {"tool_calls": [{"function": {
        "name": "report_visible_ui_elements", "arguments": '{"elements":[],"extra":1}',
    }}]}}]},
])
def test_ling_rejects_missing_wrong_or_malformed_tool_output(bad) -> None:
    provider, _transport = make(
        OpenRouterVisualObserver, bad, model=OPENROUTER_LING_VISUAL_MODEL,
    )
    shot = capture()
    with pytest.raises(VisualProviderFailure, match="malformed_response"):
        provider.observe(shot, Observation("", ""), "inspect")
    shot.discard()


@pytest.mark.parametrize("model", ["openrouter/free", "text-only/model:free", "paid/model"])
def test_openrouter_rejects_unverified_or_nonfree_model(model: str) -> None:
    with pytest.raises(ValueError, match="supported free image model"):
        OpenRouterVisualObserver("key", model=model)


def test_deepseek_uses_only_current_documented_visual_model() -> None:
    provider, transport = make(DeepSeekVisualObserver, response([]))
    shot = capture()
    result = provider.observe(shot, Observation("", ""), "inspect")
    payload = transport.create.call_args.args[0]
    assert payload["model"] == DEEPSEEK_VISUAL_MODEL
    assert payload["thinking"] == {"type": "disabled"}
    assert result.pricing_class == "very_low_cost"
    shot.discard()
    with pytest.raises(ValueError, match="deepseek-flash"):
        DeepSeekVisualObserver("key", model="deepseek-chat")


def test_deepseek_value_only_contract_returns_no_candidates() -> None:
    provider, transport = make(DeepSeekVisualObserver, field_value_response("Californication"))
    shot = capture()
    grounding = VisualGroundingRequest(
        "Read the current visible value inside the supplied text field crop.", 1,
        verification_only=True, query_field_continuity=True, field_value_only=True,
    )

    result = provider.observe(shot, Observation("", ""), "", grounding)

    prompt = transport.create.call_args.args[0]["messages"][0]["content"][0]["text"]
    assert "Do not identify" in prompt and "nearby results" in prompt
    assert result.field_value == "Californication" and result.candidates == ()
    shot.discard()


def test_openrouter_gemma_value_only_contract_is_strict_json() -> None:
    provider, transport = make(OpenRouterVisualObserver, field_value_response("Californication"))
    shot = capture()
    grounding = VisualGroundingRequest(
        "Read the current visible value inside the supplied text field crop.", 1,
        verification_only=True, query_field_continuity=True, field_value_only=True,
    )

    result = provider.observe(shot, Observation("", ""), "", grounding)

    payload = transport.create.call_args.args[0]
    prompt = payload["messages"][0]["content"][0]["text"]
    assert payload["response_format"] == {"type": "json_object"}
    assert "Do not identify" in prompt and result.field_value == "Californication"
    assert result.candidates == ()
    shot.discard()


def test_openrouter_ling_value_only_uses_forced_value_tool_schema() -> None:
    provider, transport = make(
        OpenRouterVisualObserver, field_value_tool_response("Californication"),
        model=OPENROUTER_LING_VISUAL_MODEL,
    )
    shot = capture()
    grounding = VisualGroundingRequest(
        "Read the current visible value inside the supplied text field crop.", 1,
        verification_only=True, query_field_continuity=True, field_value_only=True,
    )

    result = provider.observe(shot, Observation("", ""), "", grounding)

    payload = transport.create.call_args.args[0]
    function = payload["tools"][0]["function"]
    assert function["name"] == "report_visible_field_value"
    assert set(function["parameters"]["properties"]) == {"field_value"}
    assert payload["tool_choice"]["function"]["name"] == "report_visible_field_value"
    assert "parallel_tool_calls" not in payload
    assert payload["provider"]["require_parameters"] is True
    assert payload["provider"]["allow_fallbacks"] is False
    assert result.field_value == "Californication" and result.candidates == ()
    shot.discard()


@pytest.mark.parametrize("provider_type", [OpenRouterVisualObserver, DeepSeekVisualObserver])
def test_unicode_is_preserved_and_invalid_boxes_fail_closed(provider_type) -> None:
    provider, _transport = make(provider_type, response([element(label="Música española")]))
    shot = capture()
    assert provider.observe(shot, Observation("", ""), "inspect").candidates[0].label == "Música española"
    shot.discard()
    for bad in (element(left=-1), element(right=1001), element(left=500, right=500)):
        provider, _transport = make(provider_type, response([bad]))
        shot = capture()
        with pytest.raises(VisualProviderFailure, match="(?:invalid|malformed)_response"):
            provider.observe(shot, Observation("", ""), "inspect")
        shot.discard()


@pytest.mark.parametrize("provider_type", [OpenRouterVisualObserver, DeepSeekVisualObserver])
@pytest.mark.parametrize("failure,code", [
    (VisualProviderFailure("api_error"), "api_error"), (TimeoutError(), "timeout"),
])
def test_provider_and_timeout_failures_are_sanitized(provider_type, failure, code: str) -> None:
    provider, transport = make(provider_type, response([]))
    transport.create.side_effect = failure
    shot = capture()
    with pytest.raises(VisualProviderFailure) as caught:
        provider.observe(shot, Observation("", ""), "inspect")
    assert caught.value.code == code
    assert "provider-secret" not in str(caught.value)
    shot.discard()


@pytest.mark.parametrize("provider_type", [OpenRouterVisualObserver, DeepSeekVisualObserver])
@pytest.mark.parametrize("bad", [
    {}, {"choices": []}, {"choices": [{"message": {"content": "```json\n{}\n```"}}]},
    {"choices": [{"message": {"content": '{"elements":[],"extra":1}'}}]},
])
def test_malformed_chat_response_fails_closed_without_prose_recovery(provider_type, bad) -> None:
    provider, _transport = make(provider_type, bad)
    shot = capture()
    with pytest.raises(VisualProviderFailure, match="(?:invalid|malformed)_response"):
        provider.observe(shot, Observation("", ""), "inspect")
    shot.discard()


def test_shared_schema_and_conversion_are_strict() -> None:
    schema = visual_schema()
    assert schema["additionalProperties"] is False
    assert schema["properties"]["elements"]["items"]["properties"]["box"]["additionalProperties"] is False
    assert strict_visual_json('{"elements":[]}') == {"elements": []}
    raw_candidate = element()
    candidate = candidate_from_normalized(raw_candidate, 200, 100)
    assert candidate.rectangle == Rect(20, 20, 60, 40)
    assert candidate.role == raw_candidate["role"].replace("_", " ")
    assert candidate.provider_role == raw_candidate["role"]
    with pytest.raises(VisualProviderFailure):
        strict_visual_json('{"elements":[],"elements":[]}')


def test_directed_visual_schema_omits_parent_and_result_roles_are_generic() -> None:
    directed = visual_schema(include_parent=False)["properties"]["elements"]["items"]
    generic = visual_schema(include_parent=True)["properties"]["elements"]["items"]
    assert "parent" not in directed["properties"]
    assert "parent" in generic["properties"]
    assert "parent" not in directed["required"]
    roles = directed["properties"]["role"]["enum"]
    assert "card" in roles and "list_item" in roles
    assert "song_result" not in roles


def test_provider_switching_is_environment_only(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "VISUAL_PROVIDER", "OPENROUTER_API_KEY", "OPENROUTER_VISUAL_MODEL",
        "OPENROUTER_DATA_COLLECTION",
        "DEEPSEEK_API_KEY", "DEEPSEEK_VISUAL_MODEL", "OPENAI_API_KEY", "VISUAL_MODEL",
        "VISUAL_ACTIONS_ENABLED",
        "VISUAL_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)
    assert visual_provider_from_environment() is None
    cases = (
        ("openrouter", "OPENROUTER_API_KEY", OpenRouterVisualObserver, OPENROUTER_FREE_VISUAL_MODEL),
        ("deepseek", "DEEPSEEK_API_KEY", DeepSeekVisualObserver, DEEPSEEK_VISUAL_MODEL),
        ("openai", "OPENAI_API_KEY", OpenAIVisualObserver, "gpt-6-astra"),
    )
    for provider_name, key_name, provider_type, expected_model in cases:
        monkeypatch.setenv("VISUAL_PROVIDER", provider_name)
        monkeypatch.setenv(key_name, "test-key")
        provider = visual_provider_from_environment()
        assert isinstance(provider, provider_type)
        assert provider.model == expected_model
        monkeypatch.delenv(key_name)

    monkeypatch.setenv("VISUAL_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_VISUAL_MODEL", OPENROUTER_LING_VISUAL_MODEL)
    provider = visual_provider_from_environment()
    assert isinstance(provider, OpenRouterVisualObserver)
    assert provider.model == OPENROUTER_LING_VISUAL_MODEL
    assert provider.data_collection_policy == "deny"
    assert provider.timeout == 20

    monkeypatch.setenv("OPENROUTER_DATA_COLLECTION", "allow")
    monkeypatch.setenv("VISUAL_TIMEOUT_SECONDS", "12.5")
    provider = visual_provider_from_environment()
    assert isinstance(provider, OpenRouterVisualObserver)
    assert provider.data_collection_policy == "allow"
    assert provider.timeout == 12.5


def test_environment_rejects_unknown_provider_model_and_real_actions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VISUAL_PROVIDER", "unknown")
    with pytest.raises(VisualProviderConfigurationError):
        visual_provider_from_environment()
    monkeypatch.setenv("VISUAL_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_VISUAL_MODEL", "openrouter/free")
    with pytest.raises(VisualProviderConfigurationError, match="supported free image"):
        visual_provider_from_environment()
    monkeypatch.setenv("OPENROUTER_VISUAL_MODEL", OPENROUTER_FREE_VISUAL_MODEL)
    monkeypatch.setenv("OPENROUTER_DATA_COLLECTION", "invalid")
    with pytest.raises(VisualProviderConfigurationError, match="data_collection_policy"):
        visual_provider_from_environment()
    monkeypatch.setenv("OPENROUTER_DATA_COLLECTION", "allow")
    monkeypatch.setenv("VISUAL_TIMEOUT_SECONDS", "invalid")
    with pytest.raises(VisualProviderConfigurationError, match="VISUAL_TIMEOUT_SECONDS"):
        visual_provider_from_environment()
    monkeypatch.setenv("VISUAL_TIMEOUT_SECONDS", "20")
    monkeypatch.setenv("VISUAL_ACTIONS_ENABLED", "true")
    with pytest.raises(VisualProviderConfigurationError, match="cannot be enabled"):
        visual_provider_from_environment()
