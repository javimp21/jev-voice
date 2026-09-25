"""Optional concrete visual providers and environment configuration."""

from __future__ import annotations

import os
from dataclasses import asdict

from computer.visual import VisualObserver
from computer.visual_providers.deepseek import DEEPSEEK_VISUAL_MODEL, DeepSeekVisualObserver
from computer.visual_providers.failover import GeminiVisualGrounder
from computer.visual_providers.gemini import (
    GEMINI_INTERACTIVE_TIMEOUT_SECONDS, GEMINI_VISUAL_MODEL, GeminiVisualObserver,
)
from computer.visual_providers.openai import (
    OPENAI_VISUAL_MODEL, OpenAIVisualGrounder, OpenAIVisualObserver,
)
from computer.visual_providers.openrouter import (
    OPENROUTER_FREE_VISUAL_MODEL, OpenRouterVisualObserver,
)


class VisualProviderConfigurationError(ValueError):
    pass


def _max_elements() -> int:
    try:
        value = int(os.environ.get("VISUAL_MAX_ELEMENTS", "40"))
    except ValueError as exc:
        raise VisualProviderConfigurationError("VISUAL_MAX_ELEMENTS must be an integer.") from exc
    if not 1 <= value <= 100:
        raise VisualProviderConfigurationError("VISUAL_MAX_ELEMENTS must be between 1 and 100.")
    return value


def directed_max_elements_from_environment() -> int:
    try:
        value = int(os.environ.get("VISUAL_DIRECTED_MAX_ELEMENTS", "5"))
    except ValueError as exc:
        raise VisualProviderConfigurationError(
            "VISUAL_DIRECTED_MAX_ELEMENTS must be an integer."
        ) from exc
    if not 1 <= value <= 100:
        raise VisualProviderConfigurationError(
            "VISUAL_DIRECTED_MAX_ELEMENTS must be between 1 and 100."
        )
    return value


def _gemini_directed_max_output_tokens() -> int:
    try:
        value = int(os.environ.get("GEMINI_VISUAL_DIRECTED_MAX_OUTPUT_TOKENS", "800"))
    except ValueError as exc:
        raise VisualProviderConfigurationError(
            "GEMINI_VISUAL_DIRECTED_MAX_OUTPUT_TOKENS must be an integer."
        ) from exc
    if not 128 <= value <= 8000:
        raise VisualProviderConfigurationError(
            "GEMINI_VISUAL_DIRECTED_MAX_OUTPUT_TOKENS must be between 128 and 8000."
        )
    return value


def _timeout_seconds() -> float:
    try:
        value = float(os.environ.get("VISUAL_TIMEOUT_SECONDS", "20"))
    except ValueError as exc:
        raise VisualProviderConfigurationError("VISUAL_TIMEOUT_SECONDS must be a number.") from exc
    if not 1 <= value <= 120:
        raise VisualProviderConfigurationError(
            "VISUAL_TIMEOUT_SECONDS must be between 1 and 120."
        )
    return value


def _gemini_timeout_seconds() -> float:
    """Respect a shorter global timeout and cap the interactive Gemini attempt."""
    return min(_timeout_seconds(), GEMINI_INTERACTIVE_TIMEOUT_SECONDS)


def visual_provider_from_environment() -> VisualObserver | None:
    provider = os.environ.get("VISUAL_PROVIDER", "").strip().casefold()
    if not provider:
        return None
    if provider not in {"openrouter", "deepseek", "openai", "gemini"}:
        raise VisualProviderConfigurationError(
            "VISUAL_PROVIDER must be unset, 'openrouter', 'deepseek', 'openai', or 'gemini'."
        )
    enabled = os.environ.get("VISUAL_ACTIONS_ENABLED", "false").strip().casefold()
    if enabled not in {"", "0", "false", "no"}:
        raise VisualProviderConfigurationError(
            "Real-provider visual actions cannot be enabled in this observation-only release."
        )
    max_elements = _max_elements()
    timeout = _timeout_seconds()
    if provider == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            raise VisualProviderConfigurationError(
                "OPENROUTER_API_KEY is required when VISUAL_PROVIDER=openrouter."
            )
        model = os.environ.get("OPENROUTER_VISUAL_MODEL", OPENROUTER_FREE_VISUAL_MODEL).strip()
        data_collection_policy = os.environ.get(
            "OPENROUTER_DATA_COLLECTION", "deny",
        ).strip().casefold()
        try:
            return OpenRouterVisualObserver(
                api_key, model=model, max_elements=max_elements,
                data_collection_policy=data_collection_policy, timeout=timeout,
            )
        except ValueError as exc:
            raise VisualProviderConfigurationError(str(exc)) from exc
    if provider == "deepseek":
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            raise VisualProviderConfigurationError(
                "DEEPSEEK_API_KEY is required when VISUAL_PROVIDER=deepseek."
            )
        model = os.environ.get("DEEPSEEK_VISUAL_MODEL", DEEPSEEK_VISUAL_MODEL).strip()
        try:
            return DeepSeekVisualObserver(
                api_key, model=model, max_elements=max_elements, timeout=timeout,
            )
        except ValueError as exc:
            raise VisualProviderConfigurationError(str(exc)) from exc
    if provider == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise VisualProviderConfigurationError(
                "GEMINI_API_KEY is required when VISUAL_PROVIDER=gemini."
            )
        model = os.environ.get("GEMINI_VISUAL_MODEL", GEMINI_VISUAL_MODEL).strip()
        openai_fallback = None
        openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        if openai_api_key:
            openai_model = os.environ.get("OPENAI_VISUAL_MODEL", OPENAI_VISUAL_MODEL).strip()
            try:
                openai_fallback = OpenAIVisualGrounder(
                    openai_api_key, model=openai_model, max_elements=max_elements,
                    timeout=timeout,
                )
            except ValueError as exc:
                raise VisualProviderConfigurationError(str(exc)) from exc
        try:
            return GeminiVisualGrounder(
                api_key, openai_fallback=openai_fallback,
                model=model, max_elements=max_elements, timeout=_gemini_timeout_seconds(),
                directed_max_output_tokens=_gemini_directed_max_output_tokens(),
            )
        except ValueError as exc:
            raise VisualProviderConfigurationError(str(exc)) from exc
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise VisualProviderConfigurationError("OPENAI_API_KEY is required when VISUAL_PROVIDER=openai.")
    model = os.environ.get("VISUAL_MODEL", "gpt-6-astra").strip()
    try:
        return OpenAIVisualObserver(
            api_key, model=model, max_elements=max_elements, timeout=timeout,
        )
    except ValueError as exc:
        raise VisualProviderConfigurationError(str(exc)) from exc


def check_visual_provider_from_environment() -> dict[str, object]:
    """Run metadata-only checks; never upload pixels or request a generation."""
    provider_name = os.environ.get("VISUAL_PROVIDER", "").strip().casefold()
    if not provider_name:
        return {
            "provider": None, "configured": False, "api_key_present": False,
            "model": None, "connectivity": False,
            "metadata_generation_performed": False,
            "error": {"category": "invalid_request", "http_status": None,
                      "provider_code": None, "message": "VISUAL_PROVIDER is not configured."},
        }
    key_variable = {
        "openrouter": "OPENROUTER_API_KEY", "deepseek": "DEEPSEEK_API_KEY",
        "openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY",
    }.get(provider_name)
    key_present = bool(key_variable and os.environ.get(key_variable, ""))
    model = {
        "openrouter": os.environ.get("OPENROUTER_VISUAL_MODEL", OPENROUTER_FREE_VISUAL_MODEL),
        "deepseek": os.environ.get("DEEPSEEK_VISUAL_MODEL", DEEPSEEK_VISUAL_MODEL),
        "openai": os.environ.get("VISUAL_MODEL", "gpt-6-astra"),
        "gemini": os.environ.get("GEMINI_VISUAL_MODEL", GEMINI_VISUAL_MODEL),
    }.get(provider_name)
    data_collection_policy = (
        os.environ.get("OPENROUTER_DATA_COLLECTION", "deny").strip().casefold()
        if provider_name == "openrouter" else None
    )
    if not key_present:
        return {
            "provider": provider_name, "configured": provider_name in {"openrouter", "deepseek", "openai", "gemini"},
            "api_key_present": False, "model": model, "connectivity": False,
            **({"data_collection_policy": data_collection_policy} if provider_name == "openrouter" else {}),
            "metadata_generation_performed": False,
            "error": {"category": "authentication_error", "http_status": None,
                      "provider_code": None, "message": "The configured provider API key is absent."},
        }
    try:
        provider = visual_provider_from_environment()
    except VisualProviderConfigurationError as exc:
        return {
            "provider": provider_name, "configured": False, "api_key_present": True,
            "model": model, "connectivity": False, "metadata_generation_performed": False,
            **({"data_collection_policy": data_collection_policy} if provider_name == "openrouter" else {}),
            "error": {"category": "invalid_request", "http_status": None,
                      "provider_code": None, "message": str(exc)[:160]},
        }
    if isinstance(provider, OpenRouterVisualObserver):
        return asdict(provider.check_connectivity())
    if isinstance(provider, GeminiVisualObserver):
        return asdict(provider.check_connectivity())
    return {
        "provider": provider_name, "configured": True, "api_key_present": True,
        "model": model, "connectivity": None, "model_available": None,
        "image_input_supported": None, "json_mode_supported": None,
        "account_policy_eligible": None,
        "request_privacy_policy": "metadata-only connectivity check is not implemented for this provider",
        "metadata_generation_performed": False, "error": None,
    }
