"""Observation-only OpenAI Responses API adapter for grounded visible UI elements."""

from __future__ import annotations

from collections.abc import Callable
import socket
import time
from typing import Any

from computer.models import Observation
from computer.visual import (
    ScreenshotCapture, VisualGroundingRequest, VisualObservation, VisualProviderFailure,
)
from computer.visual_providers.common import (
    HTTPSJSONTransport, JSONTransport, candidate_from_normalized, directed_visual_prompt,
    png_data_url, strict_visual_json, token_usage, visual_prompt, visual_schema,
)


_ENDPOINT = "https://api.openai.com/v1/responses"
OPENAI_VISUAL_MODEL = "gpt-5.6-luna"
OPENAI_DIRECTED_MAX_OUTPUT_TOKENS = 800


def _output_text(response: dict[str, Any]) -> str:
    if response.get("status") != "completed" or not isinstance(response.get("output"), list):
        raise VisualProviderFailure("invalid_response")
    texts: list[str] = []
    for item in response["output"]:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "output_text" and isinstance(part.get("text"), str):
                texts.append(part["text"])
            elif isinstance(part, dict) and part.get("type") == "refusal":
                raise VisualProviderFailure("refusal")
    if len(texts) != 1:
        raise VisualProviderFailure("invalid_response")
    return texts[0]


class OpenAIVisualObserver:
    """Ground visible interactive regions; never select or execute an action."""

    name = "openai"
    pricing_class = "paid"

    def __init__(
        self, api_key: str, *, model: str = "gpt-6-astra", max_elements: int = 40,
        timeout: float = 20, transport: JSONTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required")
        if not model or len(model) > 100:
            raise ValueError("model must be a nonempty ID of at most 100 characters")
        if not 1 <= max_elements <= 100:
            raise ValueError("max_elements must be between 1 and 100")
        if not 1 <= timeout <= 120:
            raise ValueError("timeout must be between 1 and 120 seconds")
        self._api_key = api_key
        self.model = model
        self.max_elements = max_elements
        self.timeout = timeout
        self.transport = transport or HTTPSJSONTransport(_ENDPOINT)
        self._clock = clock

    def observe(
        self, screenshot: ScreenshotCapture, window: Observation, original_request: str,
        grounding: VisualGroundingRequest | None = None,
    ) -> VisualObservation:
        directed = grounding is not None
        requested_max = grounding.max_elements if grounding is not None else self.max_elements
        prompt = (
            directed_visual_prompt(
                requested_max, grounding.objective,
                verification_only=grounding.verification_only,
            )
            if grounding is not None else visual_prompt(requested_max, original_request)
        )
        payload = {
            "model": self.model,
            "store": False,
            "reasoning": {"effort": "low"},
            "max_output_tokens": OPENAI_DIRECTED_MAX_OUTPUT_TOKENS if directed else 8000,
            "input": [{
                "role": "user", "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": png_data_url(screenshot), "detail": "high"},
                ],
            }],
            "text": {"format": {
                "type": "json_schema", "name": "visible_ui_elements", "strict": True,
                "schema": visual_schema(
                    include_parent=not directed,
                    include_activity=bool(grounding and grounding.verification_only),
                ),
            }},
        }
        started = self._clock()
        try:
            response = self.transport.create(payload, self._api_key, self.timeout)
            latency_ms = max(0, round((self._clock() - started) * 1000))
            parsed = strict_visual_json(_output_text(response))
            candidates = tuple(
                candidate_from_normalized(
                    raw, screenshot.metadata.pixel_width, screenshot.metadata.pixel_height,
                    parent_required=not directed,
                    verification_only=bool(grounding and grounding.verification_only),
                )
                for raw in parsed["elements"][:requested_max]
            )
        except VisualProviderFailure as exc:
            raise exc.with_provider_context(self.name, self.model) from exc
        except (TimeoutError, socket.timeout) as exc:
            raise VisualProviderFailure("timeout").with_provider_context(
                self.name, self.model,
            ) from exc
        except Exception as exc:
            raise VisualProviderFailure("unknown_api_error").with_provider_context(
                self.name, self.model,
            ) from exc
        return VisualObservation(
            candidates, self.name, self.model, latency_ms,
            token_usage(response, ("input_tokens", "output_tokens", "total_tokens")),
            False, self.pricing_class,
            requested_max_elements=requested_max,
            returned_visual_elements=len(candidates), directed_grounding=directed,
            raw_element_count=len(parsed["elements"]),
            parsed_element_count=len(candidates),
        )


class OpenAIVisualGrounder(OpenAIVisualObserver):
    """Responses API grounder used as the one-shot Gemini provider fallback."""

    def __init__(
        self, api_key: str, *, model: str = OPENAI_VISUAL_MODEL, max_elements: int = 40,
        timeout: float = 20, transport: JSONTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(
            api_key, model=model, max_elements=max_elements, timeout=timeout,
            transport=transport, clock=clock,
        )
