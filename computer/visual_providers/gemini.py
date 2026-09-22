"""Observation-only Google Gemini visual grounding adapter."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
import re
import socket
import time

from computer.models import Observation, ProviderErrorDiagnostic, VisualRequestFingerprint
from computer.visual import (
    ScreenshotCapture, VisualGroundingRequest, VisualObservation, VisualProviderFailure,
)
from computer.visual_providers.common import (
    HTTPSJSONTransport, JSONTransport, MetadataTransport, candidate_from_normalized,
    directed_visual_prompt, generic_http_error, png_base64, png_fingerprint, strict_visual_json,
    visual_prompt, visual_schema,
)


GEMINI_VISUAL_MODEL = "gemini-3.5-flash-lite"
GEMINI_GENERIC_MAX_OUTPUT_TOKENS = 8000
GEMINI_DIRECTED_MAX_OUTPUT_TOKENS = 800
GEMINI_VISUAL_SCHEMA_NAME = "visual_elements"
GEMINI_VISUAL_SCHEMA_VERSION = "1"
_API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"


def _provider_code(value: object) -> str | None:
    if type(value) is int:
        value = str(value)
    if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1,64}", value):
        return value
    return None


def gemini_http_error(status: int, body: bytes) -> VisualProviderFailure:
    """Map Gemini errors without retaining provider messages or response bodies."""
    code: str | None = None
    message = ""
    try:
        parsed = json.loads(body, parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()))
        error = parsed.get("error") if isinstance(parsed, dict) else None
        if isinstance(error, dict):
            code = _provider_code(error.get("status")) or _provider_code(error.get("code"))
            if isinstance(error.get("message"), str):
                message = error["message"].casefold()[:2000]
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError):
        pass
    signal = " ".join(filter(None, (code.casefold() if code else "", message)))
    if "api key not valid" in signal or "api_key_invalid" in signal or "unauthenticated" in signal:
        category = "authentication_error"
    elif status == 429 or "resource_exhausted" in signal or "quota" in signal:
        category = "rate_limited"
    elif "permission_denied" in signal:
        category = "permission_error"
    elif status == 404 or ("model" in signal and "not found" in signal):
        category = "model_not_found"
    else:
        category = generic_http_error(status).code
    return VisualProviderFailure(category, http_status=status, provider_code=code)


@dataclass(frozen=True, slots=True)
class GeminiConnectivity:
    provider: str
    configured: bool
    api_key_present: bool
    model: str
    connectivity: bool
    model_available: bool | None
    image_input_supported: bool | None
    structured_output_supported: bool | None
    machine_readable_output_supported: bool | None
    pricing_class: str
    metadata_generation_performed: bool = False
    error: ProviderErrorDiagnostic | None = None


def _response_text(response: dict[str, object]) -> str:
    candidates = response.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 1 or not isinstance(candidates[0], dict):
        raise VisualProviderFailure("malformed_response")
    candidate = candidates[0]
    if candidate.get("finishReason") != "STOP":
        raise VisualProviderFailure("malformed_response")
    content = candidate.get("content")
    parts = content.get("parts") if isinstance(content, dict) else None
    if not isinstance(parts, list) or len(parts) != 1 or not isinstance(parts[0], dict):
        raise VisualProviderFailure("malformed_response")
    text = parts[0].get("text")
    if not isinstance(text, str):
        raise VisualProviderFailure("malformed_response")
    return text


def _usage(response: dict[str, object]) -> tuple[tuple[str, int], ...]:
    raw = response.get("usageMetadata")
    if not isinstance(raw, dict):
        return ()
    mapping = {
        "promptTokenCount": "input_tokens",
        "candidatesTokenCount": "output_tokens",
        "totalTokenCount": "total_tokens",
    }
    return tuple((safe, value) for source, safe in mapping.items()
                 if type(value := raw.get(source)) is int and 0 <= value <= 100_000_000)


def safe_request_shape(payload: dict[str, object]) -> dict[str, object]:
    """Return field names only; never return prompts, pixels, headers, or values."""
    contents = payload.get("contents")
    content = contents[0] if isinstance(contents, list) and contents else None
    parts = content.get("parts") if isinstance(content, dict) else None
    generation = payload.get("generationConfig")
    inline_data = None
    if isinstance(parts, list):
        inline_part = next((part for part in parts
                            if isinstance(part, dict) and "inlineData" in part), None)
        inline_data = inline_part.get("inlineData") if isinstance(inline_part, dict) else None
    return {
        "top_level_fields": tuple(sorted(payload)),
        "generation_config_fields": (
            tuple(sorted(generation)) if isinstance(generation, dict) else ()
        ),
        "content_fields": tuple(sorted(content)) if isinstance(content, dict) else (),
        "part_fields": tuple(
            tuple(sorted(part)) for part in parts if isinstance(part, dict)
        ) if isinstance(parts, list) else (),
        "inline_data_fields": (
            tuple(sorted(inline_data)) if isinstance(inline_data, dict) else ()
        ),
    }


class GeminiVisualObserver:
    """Ground foreground-window pixels through Gemini without authorizing actions."""

    name = "gemini"
    pricing_class = "account_tier_dependent"

    def __init__(
        self, api_key: str, *, model: str = GEMINI_VISUAL_MODEL,
        max_elements: int = 40, timeout: float = 20,
        directed_max_output_tokens: int = GEMINI_DIRECTED_MAX_OUTPUT_TOKENS,
        transport: JSONTransport | None = None,
        metadata_transport: MetadataTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required")
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,100}", model):
            raise ValueError("Gemini visual model must be a valid model ID")
        if not 1 <= max_elements <= 100:
            raise ValueError("max_elements must be between 1 and 100")
        if not 1 <= timeout <= 120:
            raise ValueError("timeout must be between 1 and 120 seconds")
        if not 128 <= directed_max_output_tokens <= GEMINI_GENERIC_MAX_OUTPUT_TOKENS:
            raise ValueError("directed_max_output_tokens must be between 128 and 8000")
        self._api_key = api_key
        self.model = model
        self.max_elements = max_elements
        self.timeout = timeout
        self.directed_max_output_tokens = directed_max_output_tokens
        endpoint = f"{_API_ROOT}/{model}:generateContent"
        self.transport = transport or HTTPSJSONTransport(
            endpoint, error_parser=gemini_http_error,
            auth_header="x-goog-api-key", auth_scheme="",
        )
        self.metadata_transport = metadata_transport or HTTPSJSONTransport(
            endpoint, error_parser=gemini_http_error,
            auth_header="x-goog-api-key", auth_scheme="",
        )
        self._clock = clock

    def request_fingerprint(
        self, screenshot: ScreenshotCapture, grounding: VisualGroundingRequest,
    ) -> VisualRequestFingerprint:
        encoded_length, digest = png_fingerprint(screenshot)
        return VisualRequestFingerprint(
            self.name, self.model, True, grounding.max_elements,
            self.directed_max_output_tokens, "application/json",
            GEMINI_VISUAL_SCHEMA_NAME, GEMINI_VISUAL_SCHEMA_VERSION,
            (screenshot.metadata.pixel_width, screenshot.metadata.pixel_height),
            encoded_length, digest, len(grounding.objective),
        )

    def check_connectivity(self) -> GeminiConnectivity:
        try:
            response = self.metadata_transport.get(
                f"{_API_ROOT}/{self.model}", self._api_key, self.timeout,
            )
        except VisualProviderFailure as exc:
            return GeminiConnectivity(
                self.name, True, True, self.model, False, None, None, None, None,
                self.pricing_class, error=exc.diagnostic,
            )
        expected_name = f"models/{self.model}"
        if response.get("name") != expected_name:
            failure = VisualProviderFailure("malformed_response")
            return GeminiConnectivity(
                self.name, True, True, self.model, True, None, None, None, None,
                self.pricing_class, error=failure.diagnostic,
            )
        methods = response.get("supportedGenerationMethods")
        available = isinstance(methods, list) and "generateContent" in methods
        return GeminiConnectivity(
            self.name, True, True, self.model, True, available,
            None, None, None, self.pricing_class,
        )

    def observe(
        self, screenshot: ScreenshotCapture, window: Observation, original_request: str,
        grounding: VisualGroundingRequest | None = None,
    ) -> VisualObservation:
        directed = grounding is not None
        requested_max = grounding.max_elements if grounding is not None else self.max_elements
        build_started = self._clock()
        payload: dict[str, object] = {
            "contents": [{"role": "user", "parts": [
                {"text": (
                    directed_visual_prompt(requested_max, grounding.objective)
                    if grounding is not None else visual_prompt(requested_max, original_request)
                )},
                {"inlineData": {"mimeType": "image/png", "data": png_base64(screenshot)}},
            ]}],
            "generationConfig": {
                "maxOutputTokens": (
                    self.directed_max_output_tokens if directed
                    else GEMINI_GENERIC_MAX_OUTPUT_TOKENS
                ),
                "responseMimeType": "application/json",
                "responseJsonSchema": visual_schema(include_parent=not directed),
            },
        }
        request_build_ms = max(0, round((self._clock() - build_started) * 1000))
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
        parse_started = self._clock()
        try:
            parsed = strict_visual_json(_response_text(response))
            raw_elements = parsed["elements"]
            candidates = tuple(
                candidate_from_normalized(
                    raw, screenshot.metadata.pixel_width, screenshot.metadata.pixel_height,
                    parent_required=not directed,
                )
                for raw in raw_elements[:requested_max]
            )
        except VisualProviderFailure as exc:
            if exc.code == "invalid_response":
                raise VisualProviderFailure("malformed_response") from exc
            raise
        response_parse_ms = max(0, round((self._clock() - parse_started) * 1000))
        return VisualObservation(
            candidates, self.name, self.model, latency_ms, _usage(response),
            False, self.pricing_class,
            request_build_ms=request_build_ms, response_parse_ms=response_parse_ms,
            requested_max_elements=requested_max,
            returned_visual_elements=len(candidates), directed_grounding=directed,
            raw_element_count=len(raw_elements), parsed_element_count=len(candidates),
        )
