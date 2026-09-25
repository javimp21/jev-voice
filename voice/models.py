"""Typed data shared by the audio-capture and speech-recognition adapters."""

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ProviderErrorDiagnostics:
    provider_error_type: str
    provider_status_code: int | None = None
    provider_error_code: str | None = None
    provider_request_id: str | None = None
    provider_error_category: str | None = None

    def as_dict(self) -> dict[str, str | int]:
        values: dict[str, str | int | None] = {
            "provider_error_type": self.provider_error_type,
            "provider_status_code": self.provider_status_code,
            "provider_error_code": self.provider_error_code,
            "provider_request_id": self.provider_request_id,
            "provider_error_category": self.provider_error_category,
        }
        return {key: value for key, value in values.items() if value is not None}


@dataclass(frozen=True, slots=True)
class AudioCaptureResult:
    """One PCM/WAV recording held only in memory."""

    audio_bytes: bytes = field(repr=False, compare=False)
    duration_ms: int
    sample_rate: int
    channels: int


@dataclass(frozen=True, slots=True)
class SpeechTranscript:
    text: str
    provider: str
    model: str
    language_if_known: str | None
    duration_ms: int


@dataclass(frozen=True, slots=True)
class VoiceInputResult:
    """Safe voice diagnostics and transcript; deliberately contains no audio."""

    transcript: SpeechTranscript
    capture_duration_ms: int
    capture_wall_time_ms: int
    sample_rate: int
    channels: int
    transcription_elapsed_ms: int
    total_voice_input_latency_ms: int

    def diagnostics(self) -> dict[str, object]:
        return {
            "capture_success": True,
            "capture_duration_ms": self.capture_duration_ms,
            "capture_wall_time_ms": self.capture_wall_time_ms,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "stt_provider": self.transcript.provider,
            "stt_model": self.transcript.model,
            "language_if_known": self.transcript.language_if_known,
            "transcript_length": len(self.transcript.text),
            "transcript": self.transcript.text,
            "transcription_elapsed_ms": self.transcription_elapsed_ms,
            "total_voice_input_latency_ms": self.total_voice_input_latency_ms,
        }


class VoiceInputError(RuntimeError):
    """A failure with a bounded, safe category and message."""

    def __init__(
        self,
        category: str,
        message: str,
        *,
        provider_diagnostics: ProviderErrorDiagnostics | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.safe_message = message
        self.provider_diagnostics = provider_diagnostics
