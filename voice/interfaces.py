"""Recording and transcription are independent responsibilities."""

from pathlib import Path
from typing import Protocol


class Microphone(Protocol):
    def record(self) -> Path:
        """Record one utterance to a local audio file and return its path.

        The eventual caller owns deletion; format and stop behavior remain
        adapter decisions. No recording is implemented at this stage.
        """
        ...


class SpeechToText(Protocol):
    def transcribe(self, recording: Path) -> str:
        """Convert a recorded utterance into the user's request."""
        ...
