"""OpenAI transcription API adapter; provider exceptions stay private."""

from __future__ import annotations

from io import BytesIO
import re
import time
from typing import Any

from voice.models import (
    AudioCaptureResult, ProviderErrorDiagnostics, SpeechTranscript, VoiceInputError,
)


DEFAULT_OPENAI_STT_MODEL = "gpt-transcribe"
_SAFE_ERROR_CODE = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_SAFE_REQUEST_ID = re.compile(r"req_[A-Za-z0-9_-]{1,120}\Z")
_SAFE_EXCEPTION_TYPE = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,79}\Z")


def _safe_provider_code(value: object) -> str | None:
    if not isinstance(value, str) or not _SAFE_ERROR_CODE.fullmatch(value):
        return None
    if value.startswith(("sk_", "rk_", "token_", "secret_")):
        return None
    return value


def _safe_request_id(value: object) -> str | None:
    if isinstance(value, str) and _SAFE_REQUEST_ID.fullmatch(value):
        return value
    return None


def _provider_error_diagnostics(error: Exception) -> ProviderErrorDiagnostics:
    error_type = type(error).__name__
    if not _SAFE_EXCEPTION_TYPE.fullmatch(error_type):
        error_type = "ProviderError"

    status = getattr(error, "status_code", None)
    if isinstance(status, bool) or not isinstance(status, int) or not 100 <= status <= 599:
        status = None

    error_code = _safe_provider_code(getattr(error, "code", None))
    request_id = _safe_request_id(getattr(error, "request_id", None))
    code_lower = (error_code or "").casefold()
    type_lower = error_type.casefold()
    if status == 429:
        category = "quota" if "quota" in code_lower else "rate_limit"
    elif status == 401:
        category = "authentication"
    elif status == 403:
        category = "permission"
    elif status in {400, 422}:
        category = "invalid_request"
    elif status == 408 or "timeout" in type_lower:
        category = "timeout"
    elif status is not None and status >= 500:
        category = "server"
    elif "connection" in type_lower or isinstance(error, ConnectionError):
        category = "connection"
    else:
        category = "unknown"
    return ProviderErrorDiagnostics(
        provider_error_type=error_type,
        provider_status_code=status,
        provider_error_code=error_code,
        provider_request_id=request_id,
        provider_error_category=category,
    )


class OpenAISpeechToText:
    """Use OpenAI's audio transcription endpoint for multilingual speech."""

    name = "openai"

    def __init__(self, client: Any, *, model: str = DEFAULT_OPENAI_STT_MODEL) -> None:
        self._client = client
        self.model = model

    def transcribe(self, audio: AudioCaptureResult) -> SpeechTranscript:
        started = time.perf_counter()
        try:
            response = self._client.audio.transcriptions.create(
                model=self.model,
                file=("voice-command.wav", BytesIO(audio.audio_bytes), "audio/wav"),
                response_format="json",
            )
            text = response.text
            if not isinstance(text, str):
                raise ValueError("invalid transcription response")
            languages = getattr(response, "languages", None)
            language = None
            if isinstance(languages, (list, tuple)) and len(languages) == 1:
                language_item = languages[0]
                candidate = (
                    language_item.get("code") if isinstance(language_item, dict)
                    else getattr(language_item, "code", None)
                )
                if isinstance(candidate, str) and 2 <= len(candidate) <= 16:
                    language = candidate
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            raise VoiceInputError(
                "stt_provider_error",
                "Speech transcription failed.",
                provider_diagnostics=_provider_error_diagnostics(exc),
            ) from None
        return SpeechTranscript(
            text=text,
            provider=self.name,
            model=self.model,
            language_if_known=language,
            duration_ms=round((time.perf_counter() - started) * 1000),
        )


def _create_openai_client(api_key: str, timeout_seconds: float) -> Any:
    try:
        from openai import OpenAI
    except ImportError:
        raise VoiceInputError(
            "configuration_error", "The OpenAI SDK is missing; install the project dependencies.",
        ) from None
    return OpenAI(api_key=api_key, timeout=timeout_seconds, max_retries=1)
