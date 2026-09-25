"""One-shot, observation-only Gemini to OpenAI visual-provider failover."""

from __future__ import annotations

from dataclasses import replace
import time
from collections.abc import Callable

from computer.models import VisualProviderAttempt
from computer.visual import (
    Observation, ScreenshotCapture, VisualGroundingRequest, VisualObservation,
    VisualProviderFailure,
)
from computer.visual_providers.gemini import GeminiVisualObserver
from computer.visual_providers.openai import OpenAIVisualGrounder


def _recoverable_failover_reason(failure: VisualProviderFailure) -> str | None:
    if failure.code == "timeout":
        return "timeout"
    if failure.code in {"network_error", "connection_error"}:
        return "connection"
    if failure.code in {"rate_limited", "rate_limit", "quota_exceeded", "insufficient_quota"}:
        return "rate_limit"
    if failure.code in {"server_error", "provider_unavailable"}:
        return "server"
    if failure.code in {"malformed_response", "invalid_response"}:
        return "invalid_response"
    return None


def _attempt(
    provider: str, model: str, elapsed_ms: int, *,
    result_class: str, error_category: str | None = None,
) -> VisualProviderAttempt:
    return VisualProviderAttempt(
        provider[:40], model[:100], max(0, min(int(elapsed_ms), 3_600_000)),
        result_class[:40], error_category[:40] if error_category else None,
    )


class GeminiVisualGrounder(GeminiVisualObserver):
    """Gemini primary that may call OpenAI once after a recoverable provider failure."""

    def __init__(
        self, api_key: str, *, openai_fallback: OpenAIVisualGrounder | None = None,
        clock: Callable[[], float] = time.monotonic, **kwargs,
    ) -> None:
        super().__init__(api_key, clock=clock, **kwargs)
        self.openai_fallback = openai_fallback
        self._failover_clock = clock

    def observe(
        self, screenshot: ScreenshotCapture, window: Observation, original_request: str,
        grounding: VisualGroundingRequest | None = None,
    ) -> VisualObservation:
        attempts: list[VisualProviderAttempt] = []
        started = self._failover_clock()
        try:
            primary = super().observe(screenshot, window, original_request, grounding)
        except KeyboardInterrupt:
            raise
        except VisualProviderFailure as failure:
            elapsed = max(0, round((self._failover_clock() - started) * 1000))
            failure = failure.with_provider_context(self.name, self.model)
            reason = _recoverable_failover_reason(failure)
            attempts.append(_attempt(
                self.name, self.model, elapsed, result_class="provider_failure",
                error_category=failure.diagnostic.provider_error_category,
            ))
            if reason is None or self.openai_fallback is None:
                raise failure.with_provider_context(
                    self.name, self.model, provider_attempts=tuple(attempts),
                    selected_visual_provider=None, provider_failover_used=False,
                    provider_failover_reason=None,
                ) from failure
        else:
            attempts.append(_attempt(
                primary.provider, primary.model, primary.latency_ms or 0,
                result_class=("success_with_candidates" if primary.candidates else "success_empty"),
            ))
            return replace(primary, provider_attempts=tuple(attempts))

        fallback = self.openai_fallback
        assert fallback is not None
        # Pass the exact same capture object, window, request, and grounding spec.
        fallback_started = self._failover_clock()
        try:
            secondary = fallback.observe(screenshot, window, original_request, grounding)
        except KeyboardInterrupt:
            raise
        except VisualProviderFailure as failure:
            elapsed = max(0, round((self._failover_clock() - fallback_started) * 1000))
            failure = failure.with_provider_context(fallback.name, fallback.model)
            attempts.append(_attempt(
                fallback.name, fallback.model, elapsed, result_class="provider_failure",
                error_category=failure.diagnostic.provider_error_category,
            ))
            raise failure.with_provider_context(
                fallback.name, fallback.model, provider_attempts=tuple(attempts),
                selected_visual_provider=None, provider_failover_used=True,
                provider_failover_reason=reason,
            ) from failure
        except Exception as exc:
            elapsed = max(0, round((self._failover_clock() - fallback_started) * 1000))
            failure = VisualProviderFailure("unknown_api_error").with_provider_context(
                fallback.name, fallback.model,
            )
            attempts.append(_attempt(
                fallback.name, fallback.model, elapsed, result_class="provider_failure",
                error_category=failure.diagnostic.provider_error_category,
            ))
            raise failure.with_provider_context(
                fallback.name, fallback.model, provider_attempts=tuple(attempts),
                selected_visual_provider=None, provider_failover_used=True,
                provider_failover_reason=reason,
            ) from exc

        elapsed = max(0, round((self._failover_clock() - fallback_started) * 1000))
        attempts.append(_attempt(
            secondary.provider, secondary.model, elapsed,
            result_class=("success_with_candidates" if secondary.candidates else "success_empty"),
        ))
        return replace(
            secondary, provider_attempts=tuple(attempts),
            provider_failover_used=True, provider_failover_reason=reason,
        )
