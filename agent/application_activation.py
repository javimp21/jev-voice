"""Shared bounded polling for trusted foreground application activation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import re
import time

from computer.applications import ApplicationCandidate
from computer.models import Observation
from decision.context import Redactor


@dataclass(frozen=True, slots=True)
class ForegroundIdentity:
    hwnd: int | None
    pid: int | None
    trusted_app_id: str | None


@dataclass(frozen=True, slots=True)
class ApplicationActivationDiagnostics:
    requested_app_id: str
    requested_display_name: str
    requested_launch_kind: str
    launch_succeeded: bool
    activation_timeout_ms: int
    activation_poll_interval_ms: int
    activation_attempts: int
    activation_elapsed_ms: int
    initial_foreground: ForegroundIdentity | None
    observed_foregrounds: tuple[ForegroundIdentity, ...]
    final_foreground: ForegroundIdentity | None
    identity_match: bool
    identity_match_method: str


@dataclass(frozen=True, slots=True)
class ApplicationActivationWait:
    observation: Observation | None
    diagnostics: ApplicationActivationDiagnostics
    reason: str


def _bounded_app_id(value: str) -> str | None:
    return value[:80] if value and re.fullmatch(r"[A-Za-z0-9_-]{1,80}", value) else None


def _foreground_identity(observation: Observation | None) -> ForegroundIdentity | None:
    if observation is None:
        return None
    hwnd = observation.foreground_hwnd
    if hwnd is None and observation.screenshot is not None:
        hwnd = observation.screenshot.window_handle
    return ForegroundIdentity(
        hwnd=hwnd if isinstance(hwnd, int) and hwnd > 0 else None,
        pid=(observation.process_id
             if isinstance(observation.process_id, int) and observation.process_id > 0 else None),
        trusted_app_id=_bounded_app_id(observation.application_id),
    )


def _foreground_signature(observation: Observation | None) -> tuple[object, ...] | None:
    identity = _foreground_identity(observation)
    return ((identity.hwnd, identity.pid, identity.trusted_app_id) if identity else None)


def _identity_match_method(
    observation: Observation | None, candidate: ApplicationCandidate,
) -> str:
    if observation is None or observation.application_id != candidate.id:
        return "none"
    package_family = observation.package_family_name.casefold()
    if candidate.package_family and package_family == candidate.package_family.casefold():
        return "package_identity"
    process_name = Path(observation.app_name).name.casefold()
    if process_name and process_name in {
        Path(value).name.casefold() for value in candidate.process_names
    }:
        return "executable_identity"
    # application_id is assigned by the existing trusted ApplicationCatalog matcher.
    return "app_id"


def wait_for_trusted_application_activation(
    observe_local: Callable[[], Observation],
    candidate: ApplicationCandidate,
    *,
    initial_observation: Observation | None,
    launch_succeeded: bool,
    timeout_seconds: float,
    poll_interval_seconds: float,
    clock: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
    max_recorded_transitions: int = 8,
) -> ApplicationActivationWait:
    """Poll the existing catalog-derived application ID; never compare PIDs for equality."""
    if not 0 <= timeout_seconds <= 15:
        raise ValueError("Application activation timeout must be between zero and 15 seconds.")
    if not 0.01 <= poll_interval_seconds <= 1:
        raise ValueError("Application activation polling interval must be between 10 ms and one second.")
    if not 1 <= max_recorded_transitions <= 8:
        raise ValueError("At most eight foreground transitions may be recorded.")

    started = clock()
    deadline = started + timeout_seconds
    latest: Observation | None = None
    attempts = 0
    transitions: list[ForegroundIdentity] = []
    previous_signature = _foreground_signature(initial_observation)
    reason = "activation_timeout"

    if launch_succeeded:
        while True:
            try:
                latest = observe_local()
            except KeyboardInterrupt:
                raise
            except Exception:
                reason = "observation_failed"
                break
            attempts += 1
            signature = _foreground_signature(latest)
            if (signature is not None and signature != previous_signature
                    and len(transitions) < max_recorded_transitions):
                identity = _foreground_identity(latest)
                if identity is not None:
                    transitions.append(identity)
            previous_signature = signature
            # The catalog-derived app ID is authoritative. Launcher and final
            # foreground PIDs may differ for packaged apps and activation brokers.
            if latest.application_id == candidate.id:
                reason = "activated"
                break
            remaining = deadline - clock()
            if remaining <= 0:
                break
            sleep_fn(min(poll_interval_seconds, remaining))

    elapsed_ms = max(0, round((clock() - started) * 1000))
    matched = bool(latest and latest.application_id == candidate.id)
    requested_id = _bounded_app_id(candidate.id) or "unavailable"
    requested_name = Redactor().clean(candidate.display_name)
    requested_name = re.sub(r"[\x00-\x1f\x7f]", "", requested_name).strip()[:120]
    if any(separator in requested_name for separator in ("/", "\\")):
        requested_name = "[redacted]"
    launch_kind = candidate.source if candidate.source in {"start_menu", "packaged", "test"} else "unknown"
    diagnostics = ApplicationActivationDiagnostics(
        requested_app_id=requested_id,
        requested_display_name=requested_name or "unknown",
        requested_launch_kind=launch_kind,
        launch_succeeded=launch_succeeded,
        activation_timeout_ms=round(timeout_seconds * 1000),
        activation_poll_interval_ms=round(poll_interval_seconds * 1000),
        activation_attempts=attempts,
        activation_elapsed_ms=elapsed_ms,
        initial_foreground=_foreground_identity(initial_observation),
        observed_foregrounds=tuple(transitions),
        final_foreground=_foreground_identity(latest if latest is not None else initial_observation),
        identity_match=matched,
        identity_match_method=_identity_match_method(latest, candidate),
    )
    return ApplicationActivationWait(latest, diagnostics, reason)
