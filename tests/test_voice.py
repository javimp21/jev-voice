from __future__ import annotations

import struct
from io import BytesIO
import json
import sys
from types import SimpleNamespace
import wave

import pytest

from voice.audio import WindowsPushToTalkMicrophone
from voice.models import (
    AudioCaptureResult, ProviderErrorDiagnostics, SpeechTranscript, VoiceInputError,
)
from voice.openai_stt import DEFAULT_OPENAI_STT_MODEL, OpenAISpeechToText
from voice.service import VoiceInputService, normalize_transcript, voice_input_from_environment


class FakeMicrophone:
    def __init__(self, result: AudioCaptureResult | None = None, error: Exception | None = None) -> None:
        self.result = result or AudioCaptureResult(b"private-audio", 250, 16_000, 1)
        self.error = error

    def record(self) -> AudioCaptureResult:
        if self.error:
            raise self.error
        return self.result


class FakeSpeechProvider:
    def __init__(self, transcript: SpeechTranscript | None = None, error: Exception | None = None) -> None:
        self.transcript = transcript or SpeechTranscript(
            "  abre   WhatsApp y abre el chat con Iago  ",
            "test-provider", "test-model", None, 4,
        )
        self.error = error
        self.received_audio: AudioCaptureResult | None = None

    def transcribe(self, audio: AudioCaptureResult) -> SpeechTranscript:
        self.received_audio = audio
        if self.error:
            raise self.error
        return self.transcript


class FakeVoiceProviderService:
    def __init__(self, error: VoiceInputError) -> None:
        self.error = error

    def capture_and_transcribe(self):
        raise self.error


def test_capture_result_repr_does_not_include_audio_bytes() -> None:
    audio = AudioCaptureResult(b"API_KEY_LIKE_AUDIO_BYTES", 20, 16_000, 1)
    assert "API_KEY_LIKE_AUDIO_BYTES" not in repr(audio)


def test_voice_service_success_collapses_whitespace_without_rewriting_spanish() -> None:
    provider = FakeSpeechProvider()
    result = VoiceInputService(FakeMicrophone(), provider).capture_and_transcribe()
    assert result.transcript.text == "abre WhatsApp y abre el chat con Iago"
    assert result.transcript.provider == "test-provider"
    assert result.transcript.language_if_known is None
    assert result.diagnostics()["transcript"] == "abre WhatsApp y abre el chat con Iago"
    assert not hasattr(result, "audio_bytes")


def test_english_transcript_is_preserved_without_translation() -> None:
    transcript = SpeechTranscript("Open Notepad and type hello", "fake", "model", None, 1)
    result = VoiceInputService(FakeMicrophone(), FakeSpeechProvider(transcript)).capture_and_transcribe()
    assert result.transcript.text == "Open Notepad and type hello"


def test_normalization_only_trims_and_collapses_whitespace() -> None:
    assert normalize_transcript("\n  abre  WhatsApp\tahora.  ") == "abre WhatsApp ahora."


@pytest.mark.parametrize("text", ["", " \n\t "])
def test_empty_transcript_fails_closed(text: str) -> None:
    provider = FakeSpeechProvider(SpeechTranscript(text, "fake", "model", None, 1))
    with pytest.raises(VoiceInputError, match="empty") as caught:
        VoiceInputService(FakeMicrophone(), provider).capture_and_transcribe()
    assert caught.value.category == "empty_transcript"


def test_overlong_transcript_fails_closed() -> None:
    provider = FakeSpeechProvider(SpeechTranscript("x" * 4001, "fake", "model", None, 1))
    with pytest.raises(VoiceInputError) as caught:
        VoiceInputService(FakeMicrophone(), provider).capture_and_transcribe()
    assert caught.value.category == "transcript_too_long"


def test_capture_failure_is_preserved_as_typed_error() -> None:
    error = VoiceInputError("permission_denied", "Microphone access was denied or unavailable.")
    with pytest.raises(VoiceInputError) as caught:
        VoiceInputService(FakeMicrophone(error=error), FakeSpeechProvider()).capture_and_transcribe()
    assert caught.value.category == "permission_denied"


def test_empty_audio_fails_before_provider_call() -> None:
    provider = FakeSpeechProvider()
    microphone = FakeMicrophone(AudioCaptureResult(b"", 0, 16_000, 1))
    with pytest.raises(VoiceInputError) as caught:
        VoiceInputService(microphone, provider).capture_and_transcribe()
    assert caught.value.category == "empty_audio"
    assert provider.received_audio is None


def test_stt_provider_failure_does_not_retain_audio_or_expose_exception() -> None:
    secret = "sk-secret-provider-detail"
    microphone = FakeMicrophone()
    provider = FakeSpeechProvider(error=RuntimeError(secret))
    with pytest.raises(VoiceInputError) as caught:
        VoiceInputService(microphone, provider).capture_and_transcribe()
    assert caught.value.category == "stt_provider_error"
    assert secret not in str(caught.value)
    assert not hasattr(VoiceInputService, "last_audio")


def test_no_recording_files_are_created_on_success_or_provider_failure(tmp_path) -> None:
    success = VoiceInputService(FakeMicrophone(), FakeSpeechProvider())
    success.capture_and_transcribe()
    assert list(tmp_path.iterdir()) == []

    failing = VoiceInputService(
        FakeMicrophone(), FakeSpeechProvider(error=RuntimeError("provider failure")),
    )
    with pytest.raises(VoiceInputError):
        failing.capture_and_transcribe()
    assert list(tmp_path.iterdir()) == []


def test_openai_stt_uses_wav_json_and_preserves_provider_text() -> None:
    calls: list[dict[str, object]] = []
    wav_buffer = BytesIO()
    with wave.open(wav_buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(struct.pack("<" + "h" * 160, *([500] * 160)))
    wav_payload = wav_buffer.getvalue()

    class Transcriptions:
        def create(self, **kwargs):
            calls.append(kwargs)
            uploaded = kwargs["file"]
            assert uploaded[0] == "voice-command.wav"
            assert uploaded[1].tell() == 0
            assert uploaded[1].read() == wav_payload
            assert uploaded[2] == "audio/wav"
            return SimpleNamespace(text="  abre WhatsApp  ", languages=[SimpleNamespace(code="es")])

    client = SimpleNamespace(audio=SimpleNamespace(transcriptions=Transcriptions()))
    provider = OpenAISpeechToText(client)
    result = provider.transcribe(AudioCaptureResult(wav_payload, 10, 16_000, 1))
    assert result.text == "  abre WhatsApp  "
    assert result.provider == "openai"
    assert result.model == DEFAULT_OPENAI_STT_MODEL
    assert result.language_if_known == "es"
    assert calls[0]["response_format"] == "json"
    assert calls[0]["model"] == "gpt-transcribe"


def test_openai_response_handler_accepts_the_installed_sdk_transcription_type() -> None:
    from openai.types.audio.transcription import Transcription

    response = Transcription(text="hello world", languages=[{"code": "en"}])
    client = SimpleNamespace(audio=SimpleNamespace(
        transcriptions=SimpleNamespace(create=lambda **_kwargs: response),
    ))
    transcript = OpenAISpeechToText(client).transcribe(AudioCaptureResult(b"wav", 1, 16_000, 1))
    assert transcript.text == "hello world"
    assert transcript.language_if_known == "en"


def test_openai_provider_hides_provider_error_details() -> None:
    secret = "sk-secret-error-contents"
    client = SimpleNamespace(audio=SimpleNamespace(
        transcriptions=SimpleNamespace(create=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError(secret))),
    ))
    with pytest.raises(VoiceInputError) as caught:
        OpenAISpeechToText(client).transcribe(AudioCaptureResult(b"wav", 1, 16_000, 1))
    assert caught.value.category == "stt_provider_error"
    assert secret not in str(caught.value)


@pytest.mark.parametrize(("error_name", "status", "code", "category"), [
    ("BadRequestError", 400, "invalid_request", "invalid_request"),
    ("AuthenticationError", 401, "invalid_api_key", "authentication"),
    ("RateLimitError", 429, "insufficient_quota", "quota"),
    ("RateLimitError", 429, "rate_limit_exceeded", "rate_limit"),
    ("APITimeoutError", None, None, "timeout"),
    ("APIConnectionError", None, None, "connection"),
    ("InternalServerError", 500, "server_error", "server"),
])
def test_openai_error_diagnostics_are_actionable_and_bounded(
    error_name: str, status: int | None, code: str | None, category: str,
) -> None:
    error_type = type(error_name, (Exception,), {})
    error = error_type("private response detail")
    error.status_code = status
    error.code = code
    error.request_id = "req_request123"
    client = SimpleNamespace(audio=SimpleNamespace(
        transcriptions=SimpleNamespace(create=lambda **_kwargs: (_ for _ in ()).throw(error)),
    ))
    with pytest.raises(VoiceInputError) as caught:
        OpenAISpeechToText(client).transcribe(AudioCaptureResult(b"private audio", 1, 16_000, 1))
    diagnostic = caught.value.provider_diagnostics
    assert caught.value.category == "stt_provider_error"
    assert caught.value.safe_message == "Speech transcription failed."
    assert diagnostic is not None
    assert diagnostic.provider_error_type == error_name
    assert diagnostic.provider_status_code == status
    assert diagnostic.provider_error_code == code
    assert diagnostic.provider_request_id == "req_request123"
    assert diagnostic.provider_error_category == category
    assert "private response detail" not in repr(diagnostic)
    assert "private audio" not in repr(diagnostic)


def test_openai_diagnostics_never_surface_secrets_or_raw_audio() -> None:
    secret = "sk-proj-super-secret-value"
    raw_audio_marker = "RAW_AUDIO_MUST_NOT_APPEAR"
    error_type = type("BadRequestError", (Exception,), {})
    error = error_type(f"{secret} {raw_audio_marker}")
    error.status_code = 400
    error.code = secret
    error.request_id = secret
    error.body = {"message": secret, "audio": raw_audio_marker}
    client = SimpleNamespace(audio=SimpleNamespace(
        transcriptions=SimpleNamespace(create=lambda **_kwargs: (_ for _ in ()).throw(error)),
    ))
    with pytest.raises(VoiceInputError) as caught:
        OpenAISpeechToText(client).transcribe(AudioCaptureResult(raw_audio_marker.encode(), 1, 16_000, 1))
    rendered = repr(caught.value.provider_diagnostics)
    assert secret not in rendered
    assert raw_audio_marker not in rendered
    assert caught.value.provider_diagnostics is not None
    assert caught.value.provider_diagnostics.provider_error_code is None
    assert caught.value.provider_diagnostics.provider_request_id is None


def test_successful_transcription_does_not_gain_error_diagnostics() -> None:
    provider = FakeSpeechProvider(SpeechTranscript("hello world", "openai", "gpt-transcribe", "en", 2))
    result = VoiceInputService(FakeMicrophone(), provider).capture_and_transcribe()
    assert result.transcript.text == "hello world"
    assert result.transcript.model == "gpt-transcribe"
    assert result.transcript.language_if_known == "en"


def test_openai_error_diagnostics_reach_cli_without_changing_user_message(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    import main as cli

    diagnostic = ProviderErrorDiagnostics(
        "BadRequestError", 400, "invalid_request", "req_cli123", "invalid_request",
    )
    monkeypatch.setattr(
        cli, "voice_input_from_environment",
        lambda: FakeVoiceProviderService(VoiceInputError(
            "stt_provider_error", "Speech transcription failed.", provider_diagnostics=diagnostic,
        )),
    )
    assert cli.main(["voice-transcribe"]) == 1
    payload = json.loads(capsys.readouterr().err)
    assert payload == {
        "success": False,
        "error": "stt_provider_error",
        "message": "Speech transcription failed.",
        "provider_error_type": "BadRequestError",
        "provider_status_code": 400,
        "provider_error_code": "invalid_request",
        "provider_request_id": "req_cli123",
        "provider_error_category": "invalid_request",
    }


def test_environment_requires_key_and_does_not_echo_it(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(VoiceInputError) as caught:
        voice_input_from_environment({"VOICE_STT_PROVIDER": "openai"})
    assert caught.value.category == "configuration_error"
    assert "OPENAI_API_KEY" in str(caught.value)
    assert "sk-" not in str(caught.value)


def test_environment_rejects_unknown_provider_and_invalid_bounds() -> None:
    with pytest.raises(VoiceInputError) as unsupported:
        voice_input_from_environment({"VOICE_STT_PROVIDER": "unknown", "OPENAI_API_KEY": "secret"})
    assert unsupported.value.category == "configuration_error"
    with pytest.raises(VoiceInputError) as invalid:
        voice_input_from_environment({
            "OPENAI_API_KEY": "secret", "VOICE_STT_TIMEOUT_SECONDS": "0",
        })
    assert invalid.value.category == "configuration_error"


def test_environment_limits_openai_sdk_to_one_retry_and_configured_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import voice.service as service

    received: dict[str, object] = {}
    monkeypatch.setattr(
        service, "_create_openai_client",
        lambda key, timeout: received.update(key=key, timeout=timeout) or object(),
    )
    result = voice_input_from_environment({
        "OPENAI_API_KEY": "secret-key-is-never-printed",
        "VOICE_STT_PROVIDER": "openai",
        "VOICE_MAX_DURATION_SECONDS": "10",
        "VOICE_STT_TIMEOUT_SECONDS": "12",
    })
    assert isinstance(result, VoiceInputService)
    assert received == {"key": "secret-key-is-never-printed", "timeout": 12.0}
    assert "secret-key-is-never-printed" not in repr(result)


def test_openai_sdk_is_configured_for_one_retry_and_bounded_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import voice.openai_stt as openai_stt

    client_args: dict[str, object] = {}
    monkeypatch.setitem(
        sys.modules, "openai",
        SimpleNamespace(OpenAI=lambda **kwargs: client_args.update(kwargs) or object()),
    )
    client = openai_stt._create_openai_client("secret", 12.0)
    assert client is not None
    assert client_args == {"api_key": "secret", "timeout": 12.0, "max_retries": 1}


class FakeConsole:
    def __init__(self, keys: list[str]) -> None:
        self.keys = list(keys)

    def kbhit(self) -> bool:
        return bool(self.keys)

    def getwch(self) -> str:
        return self.keys.pop(0)


class FakeCallbackStop(Exception):
    pass


class FakeStream:
    def __init__(self, callback, data: bytes, active: bool = True) -> None:
        self.callback = callback
        self.data = data
        self.active = active
        self.closed = False

    def __enter__(self):
        try:
            self.callback(self.data, len(self.data) // 2, None, None)
        except FakeCallbackStop:
            pass
        return self

    def __exit__(self, *_args) -> None:
        self.closed = True


class FakeSoundDevice:
    CallbackStop = FakeCallbackStop

    def __init__(self, data: bytes, error: Exception | None = None, device=None) -> None:
        self.data = data
        self.error = error
        self.device = device if device is not None else {"max_input_channels": 1}
        self.stream: FakeStream | None = None

    def query_devices(self, *, kind: str):
        assert kind == "input"
        if self.error:
            raise self.error
        return self.device

    def RawInputStream(self, **kwargs):
        if self.error:
            raise self.error
        self.stream = FakeStream(kwargs["callback"], self.data)
        return self.stream


def _capture(
    sounddevice: FakeSoundDevice, *, keys: list[str] | None = None,
    max_duration_seconds: float = 2,
) -> AudioCaptureResult:
    import voice.audio as audio_module

    monkey = pytest.MonkeyPatch()
    monkey.setattr(audio_module.platform, "system", lambda: "Windows")
    try:
        return WindowsPushToTalkMicrophone(
            max_duration_seconds=max_duration_seconds,
            sounddevice_module=sounddevice,
            console_module=FakeConsole(keys if keys is not None else ["\r"]),
            input_fn=lambda _prompt: "",
            print_fn=lambda *_args, **_kwargs: None,
            sleep_fn=lambda _delay: None,
            monotonic_fn=lambda: 1.0,
        ).record()
    finally:
        monkey.undo()


def test_windows_capture_writes_a_standard_mono_wav_in_memory() -> None:
    pcm = struct.pack("<" + "h" * 320, *([500] * 320))
    sounddevice = FakeSoundDevice(pcm)
    result = _capture(sounddevice)
    assert result.duration_ms == 20
    assert result.sample_rate == 16_000 and result.channels == 1
    assert result.audio_bytes[:4] == b"RIFF"
    with wave.open(BytesIO(result.audio_bytes), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getframerate() == 16_000
        assert wav.getsampwidth() == 2
        assert wav.getnframes() == 320
        assert wav.readframes(320) == pcm
    assert sounddevice.stream is not None and sounddevice.stream.closed


def test_windows_capture_reports_no_microphone_and_permission_denied() -> None:
    with pytest.raises(VoiceInputError) as no_device:
        _capture(FakeSoundDevice(b"", device={"max_input_channels": 0}))
    assert no_device.value.category == "no_microphone"

    with pytest.raises(VoiceInputError) as denied:
        _capture(FakeSoundDevice(b"", error=RuntimeError("microphone permission denied")))
    assert denied.value.category == "permission_denied"


def test_windows_capture_rejects_silence_as_empty_audio() -> None:
    with pytest.raises(VoiceInputError) as caught:
        _capture(FakeSoundDevice(bytes(320)))
    assert caught.value.category == "empty_audio"


def test_windows_capture_max_duration_discards_partial_audio() -> None:
    pcm = struct.pack("<" + "h" * 16_000, *([500] * 16_000))
    sounddevice = FakeSoundDevice(pcm)
    with pytest.raises(VoiceInputError) as caught:
        _capture(sounddevice, max_duration_seconds=1)
    assert caught.value.category == "timeout"
    assert sounddevice.stream is not None and sounddevice.stream.closed
