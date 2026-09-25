from __future__ import annotations

import json
from unittest.mock import Mock

import pytest

from agent.generic_task import GenericTaskDebugResult
from computer.applications import MemoryApplicationCatalog
from decision.models import DecisionResult
from voice.models import AudioCaptureResult, SpeechTranscript, VoiceInputError, VoiceInputResult
from voice.service import VoiceInputService
import main as cli


def _voice_result(text: str = "Open WhatsApp and open the chat with Iago") -> VoiceInputResult:
    return VoiceInputResult(
        SpeechTranscript(text, "openai", "gpt-transcribe", None, 50),
        capture_duration_ms=900,
        capture_wall_time_ms=1000,
        sample_rate=16_000,
        channels=1,
        transcription_elapsed_ms=250,
        total_voice_input_latency_ms=251,
    )


class FakeVoiceService:
    def __init__(self, result: VoiceInputResult | None = None, error: Exception | None = None) -> None:
        self.result = result or _voice_result()
        self.error = error

    def capture_and_transcribe(self) -> VoiceInputResult:
        if self.error:
            raise self.error
        return self.result


class FakeMicrophone:
    def record(self) -> AudioCaptureResult:
        return AudioCaptureResult(b"fake wav", 100, 16_000, 1)


class FakeTranscriber:
    def __init__(self, text: str) -> None:
        self.text = text

    def transcribe(self, _audio: AudioCaptureResult) -> SpeechTranscript:
        return SpeechTranscript(self.text, "fake", "test-model", None, 10)


def test_voice_transcribe_never_constructs_or_runs_computer_agent(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "voice_input_from_environment", lambda: FakeVoiceService())
    computer = Mock()
    agent = Mock()
    monkeypatch.setattr(cli, "WindowsComputer", computer)
    monkeypatch.setattr(cli, "GenericTaskDebugAgent", agent)

    assert cli.main(["voice-transcribe"]) == 0
    captured = capsys.readouterr()
    result = json.loads(captured.out.split("\n", maxsplit=1)[1])
    assert captured.out.startswith("Transcript: Open WhatsApp and open the chat with Iago\n")
    assert result["voice_input"]["stt_provider"] == "openai"
    assert result["voice_input"]["transcript_length"] == len(
        "Open WhatsApp and open the chat with Iago",
    )
    computer.assert_not_called()
    agent.assert_not_called()


def test_voice_debug_displays_then_passes_exact_normalized_transcript_through_confirmation(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    expected = "abre WhatsApp y abre el chat con Iago"
    monkeypatch.setattr(
        cli, "voice_input_from_environment",
        lambda: VoiceInputService(FakeMicrophone(), FakeTranscriber(f"  {expected}  ")),
    )
    monkeypatch.setattr(cli, "visual_provider_from_environment", Mock(return_value=None))
    catalog = MemoryApplicationCatalog(())
    monkeypatch.setattr(cli, "WindowsApplicationCatalog", Mock(return_value=catalog))
    decision = Mock(min_confidence=0.8)
    monkeypatch.setattr(cli.JevDecisionMaker, "from_environment", Mock(return_value=decision))
    computer = Mock()
    monkeypatch.setattr(cli, "WindowsComputer", Mock(return_value=computer))
    agent = Mock(run=Mock(return_value=GenericTaskDebugResult(True, "target_activated", "stopped", 4)))
    agent_factory = Mock(return_value=agent)
    monkeypatch.setattr(cli, "GenericTaskDebugAgent", agent_factory)
    monkeypatch.setattr("builtins.input", lambda: "yes")

    assert cli.main(["run-agent-voice-debug"]) == 0
    captured = capsys.readouterr()
    assert f"Transcript: {expected}" in captured.out
    assert "Continue? [y/N]" in captured.err
    agent.run.assert_called_once_with(expected)
    output = json.loads(captured.out.split("\n", maxsplit=1)[1])
    assert output["voice_input"]["transcript"] == expected
    assert isinstance(output["voice_input"]["handoff_to_agent_ms"], int)
    assert agent_factory.call_count == 1


def test_voice_debug_cancellation_prevents_generic_agent_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "voice_input_from_environment", lambda: FakeVoiceService())
    monkeypatch.setattr(cli, "visual_provider_from_environment", Mock(return_value=None))
    monkeypatch.setattr(cli, "WindowsApplicationCatalog", Mock(return_value=MemoryApplicationCatalog(())))
    monkeypatch.setattr(cli.JevDecisionMaker, "from_environment", Mock(return_value=Mock()))
    monkeypatch.setattr(cli, "WindowsComputer", Mock(return_value=Mock()))
    agent_factory = Mock()
    monkeypatch.setattr(cli, "GenericTaskDebugAgent", agent_factory)
    monkeypatch.setattr("builtins.input", lambda: "n")
    assert cli.main(["run-agent-voice-debug"]) == 1
    agent_factory.assert_not_called()


def test_voice_transcribe_error_is_safe_and_does_not_leak_provider_secret(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "sk-provider-secret-detail"
    monkeypatch.setattr(
        cli, "voice_input_from_environment",
        lambda: FakeVoiceService(error=VoiceInputError("stt_provider_error", "Speech transcription failed.")),
    )
    assert cli.main(["voice-transcribe"]) == 1
    captured = capsys.readouterr()
    assert secret not in captured.out + captured.err
    assert json.loads(captured.err)["error"] == "stt_provider_error"


def test_voice_debug_uses_the_original_generic_confirmation_and_safety_policy(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """A denied existing confirmation leaves the generic agent unconstructed."""
    monkeypatch.setattr(cli, "voice_input_from_environment", lambda: FakeVoiceService())
    monkeypatch.setattr(cli, "visual_provider_from_environment", Mock(return_value=None))
    monkeypatch.setattr(cli, "WindowsApplicationCatalog", Mock(return_value=MemoryApplicationCatalog(())))
    monkeypatch.setattr(cli.JevDecisionMaker, "from_environment", Mock(return_value=Mock()))
    monkeypatch.setattr(cli, "WindowsComputer", Mock(return_value=Mock()))
    agent_factory = Mock()
    monkeypatch.setattr(cli, "GenericTaskDebugAgent", agent_factory)
    monkeypatch.setattr("builtins.input", lambda: "no")
    assert cli.main(["run-agent-voice-debug"]) == 1
    captured = capsys.readouterr()
    assert "one trusted app launch" in captured.err
    assert "Continue? [y/N]" in captured.err
    agent_factory.assert_not_called()
