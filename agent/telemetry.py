"""Small, privacy-safe telemetry primitives for the LangGraph runtime."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from threading import Lock
from typing import Literal, Protocol


EventType = Literal["run_started", "node_completed", "run_completed"]


@dataclass(frozen=True, slots=True)
class AgentTelemetryEvent:
    """Allowlisted runtime metadata; never includes request or UI content."""

    event_type: EventType
    run_id: str
    timestamp: str
    started_at: str | None = None
    ended_at: str | None = None
    total_duration_ms: int | None = None
    node: str | None = None
    step: int | None = None
    duration_ms: int | None = None
    transition_reason: str | None = None
    action_kind: str | None = None
    success: bool | None = None
    stop_reason: str | None = None
    steps: int | None = None
    replan_count: int | None = None
    max_replans: int | None = None
    observation_complete: bool | None = None
    provider_used: str | None = None
    provider_latency_ms: int | None = None
    recoverable_failure: bool | None = None
    safety_outcome: str | None = None


class TelemetryCollector(Protocol):
    """Receives typed events. Implementations must not mutate them."""

    def emit(self, event: AgentTelemetryEvent) -> None: ...


class InMemoryTelemetryCollector:
    """Injectable collector for tests and local diagnostics."""

    def __init__(self) -> None:
        self.events: list[AgentTelemetryEvent] = []

    def emit(self, event: AgentTelemetryEvent) -> None:
        self.events.append(event)


_JSONL_LOCK = Lock()


class JsonlTelemetrySink:
    """Append one JSON event per line to a local file."""

    def __init__(self, path: Path | str = Path(".local") / "agent-telemetry.jsonl") -> None:
        self.path = Path(path)

    def emit(self, event: AgentTelemetryEvent) -> None:
        record = json.dumps(asdict(event), ensure_ascii=True, separators=(",", ":"))
        with _JSONL_LOCK:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(record + "\n")


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp with millisecond precision."""

    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
