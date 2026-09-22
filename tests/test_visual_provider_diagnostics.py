"""Safe remote-provider HTTP diagnostics and metadata-only connectivity checks."""

from __future__ import annotations

from dataclasses import asdict
from io import BytesIO
import json
import socket
import threading
import time
from unittest.mock import Mock
import urllib.error

import pytest

import main as cli
from computer.visual import VisualProviderFailure
from computer.visual_providers.common import HTTPSJSONTransport
from computer.visual_providers.openrouter import (
    OPENROUTER_FREE_VISUAL_MODEL, OPENROUTER_LING_VISUAL_MODEL,
    OpenRouterVisualObserver, openrouter_http_error,
)


def http_error(status: int, body: bytes) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "https://openrouter.ai/api/v1/chat/completions", status, "error", None, BytesIO(body),
    )


def invoke_http_error(monkeypatch: pytest.MonkeyPatch, status: int, payload: object) -> VisualProviderFailure:
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
    monkeypatch.setattr("urllib.request.urlopen", Mock(side_effect=http_error(status, body)))
    transport = HTTPSJSONTransport(
        "https://openrouter.ai/api/v1/chat/completions", error_parser=openrouter_http_error,
    )
    with pytest.raises(VisualProviderFailure) as caught:
        transport.create({"safe": True}, "secret-key", 5)
    return caught.value


@pytest.mark.parametrize(("status", "expected"), [
    (400, "invalid_request"), (401, "authentication_error"),
    (402, "payment_required"), (403, "permission_error"),
    (404, "model_not_found"), (408, "timeout"), (429, "rate_limited"),
    (500, "server_error"), (502, "provider_unavailable"),
    (503, "provider_unavailable"), (529, "provider_unavailable"),
])
def test_openrouter_http_status_categories(
    monkeypatch: pytest.MonkeyPatch, status: int, expected: str,
) -> None:
    failure = invoke_http_error(
        monkeypatch, status, {"error": {"code": status, "message": "generic failure"}},
    )
    diagnostic = asdict(failure.diagnostic)
    assert diagnostic["category"] == expected
    assert diagnostic["http_status"] == status
    assert diagnostic["provider_code"] == str(status)
    assert len(diagnostic["message"]) <= 160


@pytest.mark.parametrize(("message", "expected"), [
    ("No endpoints found that satisfy data policy", "no_eligible_provider"),
    ("Unknown model selected", "model_not_found"),
    ("response_format is not supported by any endpoint", "unsupported_response_format"),
    ("Upstream provider overloaded", "provider_unavailable"),
])
def test_openrouter_safe_message_classification(
    monkeypatch: pytest.MonkeyPatch, message: str, expected: str,
) -> None:
    failure = invoke_http_error(
        monkeypatch, 404 if expected != "unsupported_response_format" else 400,
        {"error": {"type": "routing_error", "message": message}},
    )
    assert failure.code == expected
    assert failure.diagnostic.provider_code == "routing_error"
    assert message not in failure.diagnostic.message


def test_malformed_error_json_uses_status_without_exposing_body(monkeypatch: pytest.MonkeyPatch) -> None:
    failure = invoke_http_error(monkeypatch, 401, b"not-json secret-key Bearer private")
    assert failure.code == "authentication_error"
    assert failure.diagnostic.provider_code is None
    assert "secret" not in repr(failure.diagnostic).casefold()
    assert "bearer" not in repr(failure.diagnostic).casefold()


def test_provider_error_message_cannot_echo_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "sk-test-placeholder-for-redaction"
    failure = invoke_http_error(
        monkeypatch, 400,
        {"error": {"code": "bad_request", "message": f"invalid payload {secret} data:image/png;base64,AAAA"}},
    )
    serialized = json.dumps(asdict(failure.diagnostic))
    assert secret not in serialized
    assert "base64" not in serialized
    assert failure.code == "invalid_request"


@pytest.mark.parametrize(("exception", "expected"), [
    (socket.timeout(), "timeout"),
    (urllib.error.URLError("offline"), "network_error"),
])
def test_transport_timeout_and_network_categories(
    monkeypatch: pytest.MonkeyPatch, exception: Exception, expected: str,
) -> None:
    monkeypatch.setattr("urllib.request.urlopen", Mock(side_effect=exception))
    transport = HTTPSJSONTransport("https://openrouter.ai/api/v1/chat/completions")
    with pytest.raises(VisualProviderFailure) as caught:
        transport.create({}, "secret", 5)
    assert caught.value.code == expected


def test_success_with_malformed_json_is_malformed_response(monkeypatch: pytest.MonkeyPatch) -> None:
    response = BytesIO(b"not json")
    response.__enter__ = Mock(return_value=response)  # type: ignore[attr-defined]
    response.__exit__ = Mock(return_value=False)  # type: ignore[attr-defined]
    monkeypatch.setattr("urllib.request.urlopen", Mock(return_value=response))
    with pytest.raises(VisualProviderFailure, match="malformed_response"):
        HTTPSJSONTransport("https://example.invalid").create({}, "secret", 5)


class StalledBody:
    def __init__(self) -> None:
        self.closed = threading.Event()

    def read1(self, _size: int) -> bytes:
        self.closed.wait(5)
        return b""

    def close(self) -> None:
        self.closed.set()


class SlowChunks:
    def __init__(self) -> None:
        self.closed = False

    def read1(self, _size: int) -> bytes:
        time.sleep(0.015)
        return b" " if not self.closed else b""

    def close(self) -> None:
        self.closed = True


def test_total_deadline_covers_response_header_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    release = threading.Event()
    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: release.wait(5))
    started = time.monotonic()
    with pytest.raises(VisualProviderFailure, match="timeout") as caught:
        HTTPSJSONTransport("https://example.invalid").create({}, "secret", 0.03)
    assert caught.value.diagnostic.category == "timeout"
    assert time.monotonic() - started < 0.5
    release.set()


def test_total_deadline_interrupts_stalled_chunked_body(monkeypatch: pytest.MonkeyPatch) -> None:
    response = StalledBody()
    monkeypatch.setattr("urllib.request.urlopen", Mock(return_value=response))
    with pytest.raises(VisualProviderFailure, match="timeout"):
        HTTPSJSONTransport("https://example.invalid").create({}, "secret", 0.03)
    assert response.closed.wait(0.5)


def test_slow_chunks_cannot_extend_total_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    response = SlowChunks()
    monkeypatch.setattr("urllib.request.urlopen", Mock(return_value=response))
    started = time.monotonic()
    with pytest.raises(VisualProviderFailure, match="timeout"):
        HTTPSJSONTransport("https://example.invalid").create({}, "secret", 0.04)
    assert time.monotonic() - started < 0.5


def test_successful_chunked_response_inside_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    response = BytesIO(b'{"ok":true}')
    monkeypatch.setattr("urllib.request.urlopen", Mock(return_value=response))
    assert HTTPSJSONTransport("https://example.invalid").create({}, "secret", 0.5) == {
        "ok": True,
    }


def test_oversized_response_remains_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    response = BytesIO(b"{" + b" " * 1_000_001)
    monkeypatch.setattr("urllib.request.urlopen", Mock(return_value=response))
    with pytest.raises(VisualProviderFailure, match="malformed_response"):
        HTTPSJSONTransport("https://example.invalid").create({}, "secret", 0.5)


def test_keyboard_interrupt_is_never_converted_to_provider_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("urllib.request.urlopen", Mock(side_effect=KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        HTTPSJSONTransport("https://example.invalid").create({}, "secret", 0.5)


def eligible_model(
    model: str = OPENROUTER_FREE_VISUAL_MODEL,
    parameters: list[str] | None = None,
) -> dict[str, object]:
    return {
        "id": model,
        "canonical_slug": model,
        "architecture": {"input_modalities": ["text", "image"], "output_modalities": ["text"]},
        "supported_parameters": parameters or ["temperature", "max_tokens", "response_format"],
        "pricing": {"prompt": "0", "completion": "0"},
    }


def endpoint_metadata(
    parameters: list[str], *, status: int = 0,
    pricing: dict[str, str] | None = None,
) -> dict[str, object]:
    return {"data": {"endpoints": [{
        "name": "NovitaAI: Ling endpoint",
        "provider_name": "NovitaAI",
        "status": status,
        "pricing": pricing or {"prompt": "0", "completion": "0"},
        "supported_parameters": parameters,
    }]}}


def test_connectivity_check_uses_metadata_only_and_reports_capabilities() -> None:
    generation = Mock()
    metadata = Mock()
    metadata.get.side_effect = [
        {"data": [eligible_model()]},
        endpoint_metadata(["temperature", "max_tokens", "response_format"]),
    ]
    provider = OpenRouterVisualObserver(
        "secret", transport=generation, metadata_transport=metadata,
    )
    report = provider.check_connectivity()
    assert report.connectivity is True
    assert report.model_available is True
    assert report.image_input_supported is True
    assert report.json_mode_supported is True
    assert report.tool_calling_supported is False
    assert report.machine_readable_output_supported is True
    assert report.machine_readable_strategy == "json_object"
    assert report.free_pricing is True
    assert report.account_policy_eligible is True
    assert report.data_collection_policy == "deny"
    assert report.metadata_generation_performed is False
    assert report.endpoint_metadata_available is True
    assert report.endpoints[0].eligible is None
    assert report.endpoints[0].image_input_supported is None
    assert report.endpoints[0].data_collection_policy is None
    assert "not verifiable" in report.request_privacy_policy
    generation.create.assert_not_called()
    endpoint, key, timeout = metadata.get.call_args_list[0].args
    assert "/api/v1/models/user?" in endpoint
    assert key == "secret" and timeout == 20
    endpoint, key, timeout = metadata.get.call_args_list[1].args
    assert endpoint.endswith(
        "/models/google/gemma-4-26b-a4b-it%3Afree/endpoints"
    )
    assert key == "secret" and timeout == 20
    assert "secret" not in json.dumps(asdict(report))


def test_ling_connectivity_accepts_documented_tool_calling_instead_of_response_format() -> None:
    metadata = Mock()
    metadata.get.side_effect = [
        {"data": [eligible_model(
            OPENROUTER_LING_VISUAL_MODEL,
            ["temperature", "max_tokens", "tools", "tool_choice"],
        )]},
        endpoint_metadata(["temperature", "max_tokens", "tools", "tool_choice"]),
    ]
    report = OpenRouterVisualObserver(
        "secret", model=OPENROUTER_LING_VISUAL_MODEL, metadata_transport=metadata,
    ).check_connectivity()
    assert report.model_available is True
    assert report.image_input_supported is True
    assert report.json_mode_supported is False
    assert report.tool_calling_supported is True
    assert report.machine_readable_output_supported is True
    assert report.machine_readable_strategy == "forced_tool_call"
    assert report.free_pricing is True
    assert report.request_required_parameters == (
        "max_tokens", "temperature", "tool_choice", "tools",
    )
    endpoint = report.endpoints[0]
    assert endpoint.provider_name == "NovitaAI"
    assert endpoint.available is True and endpoint.free is True
    assert endpoint.tools_supported is True
    assert endpoint.tool_choice_supported is True
    assert endpoint.parallel_tool_calls_supported is False
    assert endpoint.response_format_supported is False
    assert endpoint.eligible is None
    assert endpoint.missing_required_parameters == ()
    assert endpoint.excluded_by_require_parameters is False
    assert report.require_parameters is True
    assert report.allow_fallbacks is False
    assert endpoint.ineligible_reasons == ("endpoint_data_collection_policy_not_exposed",)


def test_ling_endpoint_diagnostic_is_compatible_when_exact_request_parameters_exist() -> None:
    metadata = Mock()
    parameters = ["temperature", "max_tokens", "tools", "tool_choice"]
    metadata.get.side_effect = [
        {"data": [eligible_model(OPENROUTER_LING_VISUAL_MODEL, parameters)]},
        endpoint_metadata(parameters),
    ]
    report = OpenRouterVisualObserver(
        "secret", model=OPENROUTER_LING_VISUAL_MODEL,
        data_collection_policy="allow", metadata_transport=metadata,
    ).check_connectivity()
    endpoint = report.endpoints[0]
    assert endpoint.eligible is True
    assert endpoint.excluded_by_require_parameters is False
    assert endpoint.ineligible_reasons == ()
    assert endpoint.image_input_supported is None
    assert report.image_input_supported is True


def test_endpoint_diagnostic_reports_paid_unavailable_and_missing_parameters() -> None:
    metadata = Mock()
    metadata.get.side_effect = [
        {"data": [eligible_model()]},
        endpoint_metadata(
            ["temperature"], status=2,
            pricing={"prompt": "0.1", "completion": "0.2"},
        ),
    ]
    report = OpenRouterVisualObserver(
        "secret", data_collection_policy="allow", metadata_transport=metadata,
    ).check_connectivity()
    endpoint = report.endpoints[0]
    assert endpoint.eligible is False
    assert endpoint.available is False
    assert endpoint.free is False
    assert endpoint.ineligible_reasons == (
        "missing_required_parameter:max_tokens",
        "missing_required_parameter:response_format",
        "endpoint_unavailable",
        "endpoint_not_free",
    )


def test_endpoint_metadata_failure_is_sanitized_without_losing_model_diagnostic() -> None:
    metadata = Mock()
    metadata.get.side_effect = [
        {"data": [eligible_model()]},
        VisualProviderFailure("permission_error", http_status=403),
    ]
    report = OpenRouterVisualObserver("secret", metadata_transport=metadata).check_connectivity()
    assert report.model_available is True
    assert report.endpoint_metadata_available is False
    assert report.endpoints == ()
    assert report.endpoint_metadata_error is not None
    assert report.endpoint_metadata_error.category == "permission_error"
    assert "secret" not in json.dumps(asdict(report))


def test_connectivity_reports_explicit_allow_privacy_consequence() -> None:
    metadata = Mock()
    metadata.get.side_effect = [
        {"data": [eligible_model()]},
        endpoint_metadata(["temperature", "max_tokens", "response_format"]),
    ]
    report = OpenRouterVisualObserver(
        "secret", data_collection_policy="allow", metadata_transport=metadata,
    ).check_connectivity()
    assert report.data_collection_policy == "allow"
    assert "process or retain data" in report.request_privacy_policy


def test_connectivity_check_reports_no_eligible_model_without_generation() -> None:
    generation = Mock()
    metadata = Mock()
    metadata.get.return_value = {"data": []}
    report = OpenRouterVisualObserver(
        "secret", transport=generation, metadata_transport=metadata,
    ).check_connectivity()
    assert report.connectivity is True
    assert report.model_available is False
    assert report.account_policy_eligible is False
    assert report.error is not None and report.error.category == "no_eligible_provider"
    generation.create.assert_not_called()


def test_connectivity_check_preserves_safe_http_failure() -> None:
    metadata = Mock()
    metadata.get.side_effect = VisualProviderFailure(
        "authentication_error", http_status=401, provider_code="invalid_key",
    )
    report = OpenRouterVisualObserver("secret", metadata_transport=metadata).check_connectivity()
    assert report.connectivity is False
    assert report.error is not None
    assert asdict(report.error) == {
        "category": "authentication_error", "http_status": 401,
        "provider_code": "invalid_key",
        "message": "The visual provider rejected the API credentials.",
    }


def test_check_visual_provider_cli_prints_only_safe_report(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    report = {
        "provider": "openrouter", "configured": True, "api_key_present": True,
        "model": OPENROUTER_FREE_VISUAL_MODEL, "connectivity": True,
        "model_available": True, "metadata_generation_performed": False, "error": None,
    }
    checker = Mock(return_value=report)
    monkeypatch.setattr(cli, "check_visual_provider_from_environment", checker)
    assert cli.main(["check-visual-provider"]) == 0
    output = capsys.readouterr().out
    assert json.loads(output) == report
    assert "base64" not in output.casefold() and "authorization" not in output.casefold()
    checker.assert_called_once_with()
