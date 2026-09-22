"""Synthetic activation polling tests; no Windows desktop is inspected."""

from __future__ import annotations

from dataclasses import asdict
import json

from agent.application_activation import wait_for_trusted_application_activation
from computer.applications import ApplicationCandidate
from computer.models import Observation


APP_ID = "app_1234567890abcdef"


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds


class ObservationSequence:
    def __init__(self, observations: list[Observation]) -> None:
        self.observations = list(observations)
        self.calls = 0

    def __call__(self) -> Observation:
        self.calls += 1
        if not self.observations:
            raise AssertionError("unexpected activation poll")
        return self.observations.pop(0)


def foreground(
    snapshot: str, *, app_id: str = "", pid: int | None = 10, hwnd: int | None = 100,
    app_name: str = "broker.exe", package: str = "", title: str = "private title",
) -> Observation:
    return Observation(
        app_name, title, process_id=pid, observation_id=snapshot,
        application_id=app_id, package_family_name=package, foreground_hwnd=hwnd,
    )


def candidate(**overrides) -> ApplicationCandidate:
    values = dict(
        id=APP_ID, display_name="Trusted App", source="start_menu",
        launch_policy="allow", process_names=("trusted.exe",),
    )
    values.update(overrides)
    return ApplicationCandidate(**values)


def wait(sequence, clock, initial, *, timeout=1.0, candidate_=None):
    return wait_for_trusted_application_activation(
        sequence, candidate_ or candidate(), initial_observation=initial,
        launch_succeeded=True, timeout_seconds=timeout, poll_interval_seconds=.2,
        clock=clock, sleep_fn=clock.sleep,
    )


def test_immediate_trusted_foreground_activation() -> None:
    clock = FakeClock()
    initial = foreground("before", app_id="other-app")
    active = foreground("active", app_id=APP_ID, pid=11, hwnd=101, app_name="trusted.exe")
    sequence = ObservationSequence([active])

    result = wait(sequence, clock, initial)

    assert result.reason == "activated"
    assert result.observation is active
    assert result.diagnostics.activation_attempts == 1
    assert result.diagnostics.identity_match
    assert result.diagnostics.identity_match_method == "executable_identity"


def test_delayed_activation_after_multiple_local_polls() -> None:
    clock = FakeClock()
    initial = foreground("before", app_id="old-app")
    sequence = ObservationSequence([
        foreground("starting", app_id="old-app", pid=10, hwnd=100),
        foreground("loading", app_id="", pid=20, hwnd=200),
        foreground("active", app_id=APP_ID, pid=30, hwnd=300, app_name="trusted.exe"),
    ])

    result = wait(sequence, clock, initial)

    assert result.reason == "activated"
    assert result.diagnostics.activation_attempts == 3
    assert result.diagnostics.activation_elapsed_ms == 400
    assert [item.hwnd for item in result.diagnostics.observed_foregrounds] == [200, 300]


def test_launcher_pid_may_differ_when_trusted_application_identity_matches() -> None:
    clock = FakeClock()
    initial = foreground("launcher", pid=111, hwnd=100, app_id="")
    active = foreground("final-window", app_id=APP_ID, pid=999, hwnd=200, app_name="trusted.exe")

    result = wait(ObservationSequence([active]), clock, initial)

    assert result.reason == "activated"
    assert result.diagnostics.initial_foreground.pid == 111
    assert result.diagnostics.final_foreground.pid == 999
    assert result.diagnostics.identity_match


def test_packaged_application_matches_trusted_package_identity() -> None:
    clock = FakeClock()
    packaged = candidate(
        source="packaged", package_family="Vendor.Package_123", process_names=(),
    )
    active = foreground(
        "packaged-window", app_id=APP_ID, pid=55, app_name="ApplicationFrameHost.exe",
        package="Vendor.Package_123",
    )

    result = wait(
        ObservationSequence([active]), clock, foreground("launcher", pid=22),
        candidate_=packaged,
    )

    assert result.diagnostics.identity_match
    assert result.diagnostics.identity_match_method == "package_identity"


def test_wrong_trusted_application_remains_foreground_until_timeout() -> None:
    clock = FakeClock()
    initial = foreground("before", app_id="old-app")
    wrong = foreground("wrong", app_id="other-app", pid=20, hwnd=200)
    sequence = ObservationSequence([wrong, wrong, wrong, wrong, wrong, wrong])

    result = wait(sequence, clock, initial, timeout=.4)

    assert result.reason == "activation_timeout"
    assert result.observation is wrong
    assert result.diagnostics.activation_attempts == 3
    assert not result.diagnostics.identity_match
    assert result.diagnostics.identity_match_method == "none"


def test_untrusted_foreground_times_out_without_inventing_identity() -> None:
    clock = FakeClock()
    initial = foreground("before", app_id="old-app")
    untrusted = foreground("untrusted", app_id="", pid=77, hwnd=700)

    result = wait(
        ObservationSequence([untrusted, untrusted]), clock, initial, timeout=.2,
    )

    assert result.reason == "activation_timeout"
    assert result.diagnostics.final_foreground.trusted_app_id is None
    assert not result.diagnostics.identity_match


def test_activation_diagnostics_are_bounded_and_do_not_include_titles_or_paths() -> None:
    clock = FakeClock()
    initial = foreground("initial", app_id="old-app", title="secret title")
    steps = [foreground(
        f"transition-{index}", app_id=f"unknown-{index}", pid=100 + index,
        hwnd=200 + index, title=f"C:\\private\\secret-{index}.txt",
    ) for index in range(40)]
    sequence = ObservationSequence(steps)
    unsafe_display = candidate(display_name="C:\\private\\Trusted App.exe")

    result = wait(
        sequence, clock, initial, timeout=5.0, candidate_=unsafe_display,
    )
    serialized = json.dumps(asdict(result.diagnostics))

    assert result.diagnostics.activation_attempts == 26
    assert len(result.diagnostics.observed_foregrounds) <= 8
    assert result.diagnostics.requested_display_name == "[redacted]"
    assert "private" not in serialized
    assert "secret title" not in serialized
    assert "window_title" not in serialized


def test_diagnostic_records_package_and_executable_match_semantics() -> None:
    clock = FakeClock()
    package_candidate = candidate(source="packaged", package_family="Vendor.Package_123")
    active = foreground(
        "package", app_id=APP_ID, package="Vendor.Package_123", app_name="host.exe",
    )
    result = wait(
        ObservationSequence([active]), clock, foreground("old"), candidate_=package_candidate,
    )
    assert result.diagnostics.identity_match_method == "package_identity"
