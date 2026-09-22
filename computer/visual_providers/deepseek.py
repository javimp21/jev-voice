"""Observation-only DeepSeek multimodal Chat Completions adapter."""

from __future__ import annotations

from collections.abc import Callable
import socket
import time

from computer.models import Observation
from computer.visual import (
    ScreenshotCapture, VisualGroundingRequest, VisualObservation, VisualProviderFailure,
)
from computer.visual_providers.common import (
    HTTPSJSONTransport, JSONTransport, candidate_from_normalized, chat_output_text,
    png_data_url, strict_visual_json, token_usage, visual_prompt,
)


DEEPSEEK_VISUAL_MODEL = "deepseek-flash"
_ENDPOINT = "https://api.deepseek.com/chat/completions"


class DeepSeekVisualObserver:
    """Ground visible UI with DeepSeek's documented image-capable Flash model."""

    name = "deepseek"
    pricing_class = "very_low_cost"

    def __init__(
        self, api_key: str, *, model: str = DEEPSEEK_VISUAL_MODEL,
        max_elements: int = 40, timeout: float = 20,
        transport: JSONTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required")
        if model != DEEPSEEK_VISUAL_MODEL:
            raise ValueError("DeepSeek visual model must be 'deepseek-flash'.")
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
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": visual_prompt(self.max_elements, original_request)},
                {"type": "image_url", "image_url": {
                    "url": png_data_url(screenshot), "detail": "high",
                }},
            ]}],
            "response_format": {"type": "json_object"},
            "thinking": {"type": "disabled"},
            "temperature": 0,
            "max_tokens": 8000,
        }
        started = self._clock()
        try:
            response = self.transport.create(payload, self._api_key, self.timeout)
        except VisualProviderFailure:
            raise
        except (TimeoutError, socket.timeout) as exc:
            raise VisualProviderFailure("timeout") from exc
        except Exception as exc:
            raise VisualProviderFailure("api_error") from exc
        latency_ms = max(0, round((self._clock() - started) * 1000))
        parsed = strict_visual_json(chat_output_text(response))
        candidates = tuple(
            candidate_from_normalized(raw, screenshot.metadata.pixel_width, screenshot.metadata.pixel_height)
            for raw in parsed["elements"][:self.max_elements]
        )
        return VisualObservation(
            candidates, self.name, self.model, latency_ms,
            token_usage(response, ("prompt_tokens", "completion_tokens", "total_tokens")),
            False, self.pricing_class,
        )
