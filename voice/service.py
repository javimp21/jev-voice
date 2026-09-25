"""One-shot voice capture, transcription, and minimal transcript cleanup."""

from __future__ import annotations

import os
import time
from typing import Mapping

from voice.audio import WindowsPushToTalkMicrophone
from voice.interfaces import Microphone, SpeechToText
from voice.models import (
    AudioCaptureResult, SpeechTranscript, VoiceInputError, VoiceInputResult,
)
from voice.openai_stt import DEFAULT_OPENAI_STT_MODEL, OpenAISpeechToText, _create_openai_client


MAX_VOICE_COMMAND_LENGTH = 4000


def normalize_transcript(text: str) -> str:
    """Trim and collapse whitespace only; preserve language and wording."""
    if not isinstance(text, str):
        raise VoiceInputError("empty_transcript", "Speech transcription was empty.")
    normalized = " ".join(text.split())
    if not normalized:
        raise VoiceInputError("empty_transcript", "Speech transcription was empty.")
    if len(normalized) > MAX_VOICE_COMMAND_LENGTH:
        raise VoiceInputError("transcript_too_long", "The spoken command exceeded 4000 characters.")
    return normalized


class VoiceInputService:
    def __init__(self, microphone: Microphone, provider: SpeechToText) -> None:
        self._microphone = microphone
        self._provider = provider

    def capture_and_transcribe(self) -> VoiceInputResult:
        capture_started = time.perf_counter()
        audio = self._microphone.record()
        capture_wall_time_ms = round((time.perf_counter() - capture_started) * 1000)
        if not isinstance(audio, AudioCaptureResult) or not audio.audio_bytes:
            raise VoiceInputError("empty_audio", "No speech audio was captured.")
        capture_duration_ms = audio.duration_ms
        sample_rate = audio.sample_rate
        channels = audio.channels
        transcription_started = time.perf_counter()
        try:
            transcript = self._provider.transcribe(audio)
        except KeyboardInterrupt:
            raise
        except VoiceInputError:
            raise
        except Exception:
            # Provider exceptions can embed request metadata or credentials.
            raise VoiceInputError("stt_provider_error", "Speech transcription failed.") from None
        finally:
            # No audio bytes are stored on the service or result object.
            del audio
        elapsed_ms = round((time.perf_counter() - transcription_started) * 1000)
        if not isinstance(transcript, SpeechTranscript):
            raise VoiceInputError("invalid_transcript", "Speech transcription returned an invalid result.")
        normalized_text = normalize_transcript(transcript.text)
        normalized = SpeechTranscript(
            text=normalized_text,
            provider=transcript.provider,
            model=transcript.model,
            language_if_known=transcript.language_if_known,
            duration_ms=transcript.duration_ms,
        )
        return VoiceInputResult(
            transcript=normalized,
            capture_duration_ms=capture_duration_ms,
            capture_wall_time_ms=capture_wall_time_ms,
            sample_rate=sample_rate,
            channels=channels,
            transcription_elapsed_ms=elapsed_ms,
            total_voice_input_latency_ms=round((time.perf_counter() - transcription_started) * 1000),
        )


def voice_input_from_environment(
    environment: Mapping[str, str] | None = None,
) -> VoiceInputService:
    env = os.environ if environment is None else environment
    provider_name = env.get("VOICE_STT_PROVIDER", "openai").strip().casefold()
    if provider_name != "openai":
        raise VoiceInputError(
            "configuration_error", "VOICE_STT_PROVIDER must be set to the supported provider 'openai'.",
        )
    api_key = env.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise VoiceInputError("configuration_error", "Set OPENAI_API_KEY before using voice transcription.")
    try:
        max_duration = float(env.get("VOICE_MAX_DURATION_SECONDS", "15"))
        timeout = float(env.get("VOICE_STT_TIMEOUT_SECONDS", "20"))
    except ValueError:
        raise VoiceInputError(
            "configuration_error", "Voice duration and timeout settings must be numbers.",
        ) from None
    if not 1 <= max_duration <= 60 or not 1 <= timeout <= 120:
        raise VoiceInputError(
            "configuration_error", "Voice duration must be 1-60 seconds and STT timeout 1-120 seconds.",
        )
    client = _create_openai_client(api_key, timeout)
    return VoiceInputService(
        WindowsPushToTalkMicrophone(max_duration_seconds=max_duration),
        OpenAISpeechToText(client, model=DEFAULT_OPENAI_STT_MODEL),
    )
