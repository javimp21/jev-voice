"""Windows push-to-talk PCM capture with no persistent audio files."""

from __future__ import annotations

from array import array
from io import BytesIO
import importlib
import platform
import time
import wave
from collections.abc import Callable
from typing import Any

from voice.models import AudioCaptureResult, VoiceInputError


class WindowsPushToTalkMicrophone:
    """Capture one mono utterance; Enter starts and Enter stops recording."""

    def __init__(
        self,
        *,
        max_duration_seconds: float = 15,
        sample_rate: int = 16_000,
        channels: int = 1,
        sounddevice_module: Any | None = None,
        console_module: Any | None = None,
        input_fn: Callable[[str], str] | None = None,
        print_fn: Callable[..., object] | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        monotonic_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        if not 1 <= max_duration_seconds <= 60:
            raise ValueError("max_duration_seconds must be between 1 and 60")
        if sample_rate != 16_000 or channels != 1:
            raise ValueError("capture must use 16 kHz mono audio")
        self.max_duration_seconds = max_duration_seconds
        self.sample_rate = sample_rate
        self.channels = channels
        self._sounddevice = sounddevice_module
        self._console = console_module
        self._input = input_fn
        self._print = print_fn
        self._sleep = sleep_fn
        self._monotonic = monotonic_fn

    def record(self) -> AudioCaptureResult:
        if platform.system() != "Windows" and self._console is None:
            raise VoiceInputError("capture_failed", "Push-to-talk capture is available on Windows only.")
        try:
            sounddevice = self._sounddevice or importlib.import_module("sounddevice")
            console = self._console or importlib.import_module("msvcrt")
        except ImportError:
            raise VoiceInputError(
                "capture_failed", "Audio capture dependencies are unavailable; install the project dependencies.",
            ) from None
        input_fn = self._input or input
        print_fn = self._print or print

        try:
            device = sounddevice.query_devices(kind="input")
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            raise self._capture_error(exc) from None
        try:
            input_channels = int(device.get("max_input_channels", 0)) if isinstance(device, dict) else 0
        except (TypeError, ValueError):
            input_channels = 0
        if input_channels < 1:
            raise VoiceInputError("no_microphone", "No usable microphone was found.")

        try:
            input_fn("Press Enter to begin one voice command...")
        except KeyboardInterrupt:
            raise
        except EOFError:
            raise VoiceInputError("capture_failed", "Push-to-talk input was closed.") from None

        print_fn(
            f"Recording. Press Enter to stop (maximum {self.max_duration_seconds:g} seconds).",
            flush=True,
        )
        max_bytes = int(self.max_duration_seconds * self.sample_rate * self.channels * 2)
        chunks: list[bytes] = []
        recorded_bytes = 0
        timed_out = False

        def on_audio(indata: Any, frames: int, _time_info: Any, _status: Any) -> None:
            nonlocal recorded_bytes, timed_out
            del frames
            raw = bytes(indata)
            remaining = max_bytes - recorded_bytes
            if remaining <= 0:
                timed_out = True
                raise sounddevice.CallbackStop
            accepted = raw[:remaining]
            chunks.append(accepted)
            recorded_bytes += len(accepted)
            if len(raw) > remaining or recorded_bytes >= max_bytes:
                timed_out = True
                raise sounddevice.CallbackStop

        start = self._monotonic()
        try:
            stream = sounddevice.RawInputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="int16",
                blocksize=0,
                callback=on_audio,
            )
            with stream as active_stream:
                while True:
                    if timed_out or self._monotonic() - start >= self.max_duration_seconds:
                        raise VoiceInputError(
                            "timeout", "Maximum recording duration reached; the partial command was discarded.",
                        )
                    if not getattr(active_stream, "active", True):
                        raise VoiceInputError("capture_failed", "Microphone capture stopped unexpectedly.")
                    if console.kbhit():
                        key = console.getwch()
                        if key == "\x03":
                            raise KeyboardInterrupt
                        if key in {"\r", "\n"}:
                            break
                    self._sleep(0.02)
        except (KeyboardInterrupt, VoiceInputError):
            raise
        except Exception as exc:
            raise self._capture_error(exc) from None

        pcm = b"".join(chunks)
        if not pcm or not any(array("h", pcm)):
            raise VoiceInputError("empty_audio", "No speech audio was captured.")
        frame_count = len(pcm) // (2 * self.channels)
        audio = BytesIO()
        with wave.open(audio, "wb") as wav:
            wav.setnchannels(self.channels)
            wav.setsampwidth(2)
            wav.setframerate(self.sample_rate)
            wav.writeframes(pcm)
        return AudioCaptureResult(
            audio_bytes=audio.getvalue(),
            duration_ms=round(frame_count * 1000 / self.sample_rate),
            sample_rate=self.sample_rate,
            channels=self.channels,
        )

    @staticmethod
    def _capture_error(error: Exception) -> VoiceInputError:
        message = str(error).casefold()
        if any(term in message for term in ("permission", "access denied", "device unavailable")):
            return VoiceInputError("permission_denied", "Microphone access was denied or unavailable.")
        if any(term in message for term in ("no default input", "no input device", "no devices")):
            return VoiceInputError("no_microphone", "No usable microphone was found.")
        return VoiceInputError("capture_failed", "Microphone capture failed.")
