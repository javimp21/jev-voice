"""Observation-only OpenRouter adapter pinned to a free multimodal model."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import json
import re
import socket
import time
from urllib.parse import quote

from computer.models import Observation, ProviderErrorDiagnostic
from computer.visual import (
    ScreenshotCapture, VisualGroundingRequest, VisualObservation, VisualProviderFailure,
)
from computer.visual_providers.common import (
    HTTPSJSONTransport, JSONTransport, MetadataTransport, candidate_from_normalized,
    chat_output_text, generic_http_error, png_data_url, strict_visual_json, token_usage,
    redact_secrets, visual_prompt, visual_schema,
)


OPENROUTER_FREE_VISUAL_MODEL = "google/gemma-4-26b-a4b-it:free"
OPENROUTER_LING_VISUAL_MODEL = "inclusionai/ling-3.0-flash-vl:free"
_MACHINE_READABLE_STRATEGIES = {
    OPENROUTER_FREE_VISUAL_MODEL: "json_object",
    OPENROUTER_LING_VISUAL_MODEL: "forced_tool_call",
}
SUPPORTED_FREE_VISUAL_MODELS = frozenset(_MACHINE_READABLE_STRATEGIES)
_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
_USER_MODELS_ENDPOINT = "https://openrouter.ai/api/v1/models/user?limit=1000&output_modalities=text"
_MODEL_ENDPOINTS_ROOT = "https://openrouter.ai/api/v1/models"
_VISUAL_TOOL_NAME = "report_visible_ui_elements"


def _provider_code(value: object) -> str | None:
    if type(value) is int:
        value = str(value)
    if (isinstance(value, str) and redact_secrets(value) == value
            and re.fullmatch(r"[A-Za-z0-9_.:-]{1,64}", value)):
        return value
    return None


def openrouter_http_error(status: int, body: bytes) -> VisualProviderFailure:
    """Classify an OpenRouter error without retaining or exposing its raw body."""
    code: str | None = None
    message = ""
    try:
        parsed = json.loads(body, parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()))
        error = parsed.get("error") if isinstance(parsed, dict) else None
        if isinstance(error, dict):
            code = _provider_code(error.get("type")) or _provider_code(error.get("code"))
            if isinstance(error.get("message"), str):
                message = error["message"].casefold()[:2000]
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError):
        pass
    signal = " ".join(filter(None, (code.casefold() if code else "", message)))
    if "no endpoint" in signal or "no eligible provider" in signal or "no available provider" in signal:
        category = "no_eligible_provider"
    elif ("response_format" in signal or "response format" in signal
          or "structured output" in signal or "json mode" in signal):
        category = "unsupported_response_format"
    elif "model" in signal and ("not found" in signal or "unknown" in signal):
        category = "model_not_found"
    elif "provider" in signal and ("unavailable" in signal or "overload" in signal):
        category = "provider_unavailable"
    else:
        generic = generic_http_error(status)
        category = generic.code
    return VisualProviderFailure(category, http_status=status, provider_code=code)


@dataclass(frozen=True, slots=True)
class OpenRouterEndpointCompatibility:
    provider_name: str
    endpoint_name: str
    available: bool | None
    free: bool | None
    image_input_supported: bool | None
    tools_supported: bool
    tool_choice_supported: bool
    parallel_tool_calls_supported: bool
    response_format_supported: bool
    supported_parameters: tuple[str, ...]
    data_collection_policy: str | None
    eligible: bool | None
    missing_required_parameters: tuple[str, ...]
    excluded_by_require_parameters: bool
    ineligible_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class OpenRouterConnectivity:
    provider: str
    configured: bool
    api_key_present: bool
    model: str
    connectivity: bool
    model_available: bool | None
    image_input_supported: bool | None
    json_mode_supported: bool | None
    tool_calling_supported: bool | None
    machine_readable_output_supported: bool | None
    machine_readable_strategy: str
    free_pricing: bool | None
    account_policy_eligible: bool | None
    data_collection_policy: str
    request_privacy_policy: str
    metadata_generation_performed: bool = False
    error: ProviderErrorDiagnostic | None = None
    endpoint_metadata_available: bool = False
    request_required_parameters: tuple[str, ...] = ()
    endpoints: tuple[OpenRouterEndpointCompatibility, ...] = ()
    endpoint_metadata_error: ProviderErrorDiagnostic | None = None
    require_parameters: bool = True
    allow_fallbacks: bool = False


def _zero_price(pricing: object) -> bool | None:
    if not isinstance(pricing, dict):
        return None
    try:
        return all(Decimal(str(pricing[key])) == 0 for key in ("prompt", "completion"))
    except (KeyError, InvalidOperation, ValueError):
        return None


def _safe_metadata_label(value: object) -> str:
    if not isinstance(value, str):
        return ""
    cleaned = redact_secrets(value).strip()
    return cleaned[:120] if cleaned == value else "[REDACTED]"


def _safe_parameters(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(sorted({item for item in value
                         if isinstance(item, str)
                         and re.fullmatch(r"[a-z][a-z0-9_]{0,63}", item)}))


def _endpoint_is_free(pricing: object) -> bool | None:
    if not isinstance(pricing, dict):
        return None
    values: list[Decimal] = []
    try:
        for key in ("prompt", "completion", "request", "image"):
            if key in pricing:
                values.append(Decimal(str(pricing[key])))
    except (InvalidOperation, ValueError):
        return None
    return bool(values) and all(value == 0 for value in values)


class OpenRouterVisualObserver:
    """Ground visible UI through one explicit free model; never route to another model."""

    name = "openrouter"
    pricing_class = "free"

    def __init__(
        self, api_key: str, *, model: str = OPENROUTER_FREE_VISUAL_MODEL,
        max_elements: int = 40, timeout: float = 20,
        data_collection_policy: str = "deny",
        transport: JSONTransport | None = None,
        metadata_transport: MetadataTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required")
        if model not in SUPPORTED_FREE_VISUAL_MODELS:
            raise ValueError(
                "OpenRouter visual model must be an explicitly supported free image model: "
                + ", ".join(sorted(SUPPORTED_FREE_VISUAL_MODELS))
            )
        if not 1 <= max_elements <= 100:
            raise ValueError("max_elements must be between 1 and 100")
        if not 1 <= timeout <= 120:
            raise ValueError("timeout must be between 1 and 120 seconds")
        if data_collection_policy not in {"deny", "allow"}:
            raise ValueError("data_collection_policy must be 'deny' or 'allow'")
        self._api_key = api_key
        self.model = model
        self.max_elements = max_elements
        self.timeout = timeout
        self.data_collection_policy = data_collection_policy
        self.transport = transport or HTTPSJSONTransport(_ENDPOINT, error_parser=openrouter_http_error)
        self.metadata_transport = metadata_transport or HTTPSJSONTransport(
            _ENDPOINT, error_parser=openrouter_http_error,
        )
        self._clock = clock

    def _request_parameter_fields(self) -> dict[str, object]:
        fields: dict[str, object] = {"temperature": 0, "max_tokens": 8000}
        if _MACHINE_READABLE_STRATEGIES[self.model] == "json_object":
            fields["response_format"] = {"type": "json_object"}
        else:
            fields["tools"] = [{
                "type": "function",
                "function": {
                    "name": _VISUAL_TOOL_NAME,
                    "description": "Report useful visible interactive UI elements and their normalized boxes.",
                    "parameters": visual_schema(),
                },
            }]
            fields["tool_choice"] = {
                "type": "function", "function": {"name": _VISUAL_TOOL_NAME},
            }
        return fields

    def _required_parameters(self) -> tuple[str, ...]:
        return tuple(sorted(self._request_parameter_fields()))

    def _endpoints_url(self) -> str:
        author, slug = self.model.split("/", 1)
        return f"{_MODEL_ENDPOINTS_ROOT}/{quote(author, safe='')}/{quote(slug, safe='')}/endpoints"

    def _endpoint_compatibility(
        self, endpoint: dict[str, object], *, model_has_image: bool,
    ) -> OpenRouterEndpointCompatibility:
        parameters = _safe_parameters(endpoint.get("supported_parameters"))
        missing = tuple(parameter for parameter in self._required_parameters()
                        if parameter not in parameters)
        status = endpoint.get("status")
        available = status == 0 if type(status) is int else None
        free = _endpoint_is_free(endpoint.get("pricing"))
        reasons = [f"missing_required_parameter:{parameter}" for parameter in missing]
        if available is False:
            reasons.append("endpoint_unavailable")
        if free is False:
            reasons.append("endpoint_not_free")
        if not model_has_image:
            reasons.append("model_does_not_support_image_input")
        eligibility_unknown = available is None or free is None
        if self.data_collection_policy == "deny":
            reasons.append("endpoint_data_collection_policy_not_exposed")
            eligibility_unknown = True
        eligible = False if any(reason != "endpoint_data_collection_policy_not_exposed"
                                for reason in reasons) else (None if eligibility_unknown else True)
        return OpenRouterEndpointCompatibility(
            provider_name=_safe_metadata_label(endpoint.get("provider_name")),
            endpoint_name=_safe_metadata_label(endpoint.get("name")),
            available=available,
            free=free,
            image_input_supported=None,
            tools_supported="tools" in parameters,
            tool_choice_supported="tool_choice" in parameters,
            parallel_tool_calls_supported="parallel_tool_calls" in parameters,
            response_format_supported="response_format" in parameters,
            supported_parameters=parameters,
            data_collection_policy=None,
            eligible=eligible,
            missing_required_parameters=missing,
            excluded_by_require_parameters=bool(missing),
            ineligible_reasons=tuple(reasons),
        )

    def check_connectivity(self) -> OpenRouterConnectivity:
        try:
            response = self.metadata_transport.get(_USER_MODELS_ENDPOINT, self._api_key, self.timeout)
        except VisualProviderFailure as exc:
            return OpenRouterConnectivity(
                self.name, True, True, self.model, False, None, None, None, None, None,
                _MACHINE_READABLE_STRATEGIES[self.model], None, None,
                self.data_collection_policy,
                "not_checked_due_to_metadata_error", error=exc.diagnostic,
            )
        data = response.get("data")
        if not isinstance(data, list) or any(not isinstance(item, dict) for item in data):
            failure = VisualProviderFailure("malformed_response")
            return OpenRouterConnectivity(
                self.name, True, True, self.model, True, None, None, None, None, None,
                _MACHINE_READABLE_STRATEGIES[self.model], None, None,
                self.data_collection_policy,
                "not_verifiable_from_malformed_metadata", error=failure.diagnostic,
            )
        model = next((item for item in data if item.get("id") == self.model
                      or item.get("canonical_slug") == self.model), None)
        if model is None:
            failure = VisualProviderFailure("no_eligible_provider")
            return OpenRouterConnectivity(
                self.name, True, True, self.model, True, False, None, None, None, None,
                _MACHINE_READABLE_STRATEGIES[self.model], None, False,
                self.data_collection_policy,
                "model_absent_from_user_filtered_catalog", error=failure.diagnostic,
            )
        architecture = model.get("architecture")
        modalities = architecture.get("input_modalities") if isinstance(architecture, dict) else None
        parameters = model.get("supported_parameters")
        json_mode = isinstance(parameters, list) and "response_format" in parameters
        tool_mode = (isinstance(parameters, list) and "tools" in parameters
                     and "tool_choice" in parameters)
        strategy = _MACHINE_READABLE_STRATEGIES[self.model]
        machine_readable = json_mode if strategy == "json_object" else tool_mode
        privacy_description = (
            "deny restricts routing to providers that do not collect user data; "
            "endpoint eligibility is not verifiable from model metadata"
            if self.data_collection_policy == "deny" else
            "allow omits the per-request data-collection filter; the underlying "
            "OpenRouter provider may process or retain data under its own policy"
        )
        endpoint_diagnostics: tuple[OpenRouterEndpointCompatibility, ...] = ()
        endpoint_error: ProviderErrorDiagnostic | None = None
        endpoint_metadata_available = False
        try:
            endpoint_response = self.metadata_transport.get(
                self._endpoints_url(), self._api_key, self.timeout,
            )
            endpoint_data = endpoint_response.get("data")
            raw_endpoints = endpoint_data.get("endpoints") if isinstance(endpoint_data, dict) else None
            if not isinstance(raw_endpoints, list) or any(
                not isinstance(endpoint, dict) for endpoint in raw_endpoints
            ):
                raise VisualProviderFailure("malformed_response")
            endpoint_diagnostics = tuple(
                self._endpoint_compatibility(
                    endpoint, model_has_image=isinstance(modalities, list) and "image" in modalities,
                )
                for endpoint in raw_endpoints
            )
            endpoint_metadata_available = True
        except VisualProviderFailure as exc:
            endpoint_error = exc.diagnostic
        return OpenRouterConnectivity(
            self.name, True, True, self.model, True, True,
            isinstance(modalities, list) and "image" in modalities,
            json_mode, tool_mode, machine_readable, strategy,
            _zero_price(model.get("pricing")), True,
            self.data_collection_policy, privacy_description,
            endpoint_metadata_available=endpoint_metadata_available,
            request_required_parameters=self._required_parameters(),
            endpoints=endpoint_diagnostics,
            endpoint_metadata_error=endpoint_error,
        )

    @staticmethod
    def _tool_arguments(response: dict[str, object]) -> str:
        choices = response.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            raise VisualProviderFailure("malformed_response")
        message = choices[0].get("message")
        calls = message.get("tool_calls") if isinstance(message, dict) else None
        if not isinstance(calls, list) or len(calls) != 1 or not isinstance(calls[0], dict):
            raise VisualProviderFailure("malformed_response")
        function = calls[0].get("function")
        if (not isinstance(function, dict) or function.get("name") != _VISUAL_TOOL_NAME
                or not isinstance(function.get("arguments"), str)):
            raise VisualProviderFailure("malformed_response")
        return function["arguments"]

    def observe(
        self, screenshot: ScreenshotCapture, window: Observation, original_request: str,
        grounding: VisualGroundingRequest | None = None,
    ) -> VisualObservation:
        provider_routing: dict[str, object] = {
            "allow_fallbacks": False,
            "require_parameters": True,
        }
        if self.data_collection_policy == "deny":
            provider_routing["data_collection"] = "deny"
        payload: dict[str, object] = {
            "model": self.model,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": visual_prompt(self.max_elements, original_request)},
                {"type": "image_url", "image_url": {"url": png_data_url(screenshot)}},
            ]}],
            "provider": provider_routing,
        }
        payload.update(self._request_parameter_fields())
        strategy = _MACHINE_READABLE_STRATEGIES[self.model]
        started = self._clock()
        try:
            response = self.transport.create(payload, self._api_key, self.timeout)
        except VisualProviderFailure:
            raise
        except (TimeoutError, socket.timeout) as exc:
            raise VisualProviderFailure("timeout") from exc
        except Exception as exc:
            raise VisualProviderFailure("unknown_api_error") from exc
        latency_ms = max(0, round((self._clock() - started) * 1000))
        try:
            output = (chat_output_text(response) if strategy == "json_object"
                      else self._tool_arguments(response))
            parsed = strict_visual_json(output)
            candidates = tuple(
                candidate_from_normalized(raw, screenshot.metadata.pixel_width, screenshot.metadata.pixel_height)
                for raw in parsed["elements"][:self.max_elements]
            )
        except VisualProviderFailure as exc:
            if exc.code == "invalid_response":
                raise VisualProviderFailure("malformed_response") from exc
            raise
        return VisualObservation(
            candidates, self.name, self.model, latency_ms,
            token_usage(response, ("prompt_tokens", "completion_tokens", "total_tokens")),
            False, self.pricing_class,
        )
