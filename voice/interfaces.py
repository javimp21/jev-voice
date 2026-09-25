"""Provider-independent contracts for one-shot voice input."""

from typing import Protocol

from voice.models import AudioCaptureResult, SpeechTranscript


class Microphone(Protocol):
    def record(self) -> AudioCaptureResult:
        """Record one bounded utterance and return it in memory."""
        ...


class SpeechToText(Protocol):
    def transcribe(self, audio: AudioCaptureResult) -> SpeechTranscript:
        """Convert one utterance into a transcript without rewriting it."""
        ...
