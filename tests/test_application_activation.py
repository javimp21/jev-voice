"""Synthetic activation polling tests; no Windows desktop is inspected."""

from __future__ import annotations

from dataclasses import asdict
import json

from agent.application_activation import wait_for_trusted_application_activation
from computer.applications import (
    ActivationWindowCandidateDiagnostics, ApplicationCandidate,
    EligibleWindowDiagnostics, EligibleWindowFacts, EligibleWindowRelationship,
    SimpleActivationAttempt, ThreadInputFallbackAttempt,
    TrustedApplicationRuntimeState, TrustedWindowActivationResult,
)
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


def wait(
    sequence, clock, initial, *, timeout=1.0, candidate_=None, target_probe=None,
    target_activator=None, post_deadline_probe_seconds=0.0,
):
    return wait_for_trusted_application_activation(
        sequence, candidate_ or candidate(), initial_observation=initial,
        launch_succeeded=True, timeout_seconds=timeout, poll_interval_seconds=.2,
        clock=clock, sleep_fn=clock.sleep, target_probe=target_probe,
        target_activator=target_activator,
        post_deadline_probe_seconds=post_deadline_probe_seconds,
    )


def eligible_diagnostic(
    diagnostic_id: str, *, area: str, index: int, stable: bool = False,
) -> EligibleWindowDiagnostics:
    return EligibleWindowDiagnostics(
        diagnostic_id=diagnostic_id,
        eligible_candidate_index=index,
        facts=EligibleWindowFacts(
            visible=True,
            minimized=False,
            enabled=True,
            cloaked_state="uncloaked",
            owner_present=False,
            root_owner_relationship="self",
            tool_window=False,
            app_window=False,
            has_nonzero_client_area=True,
            client_area_bucket=area,
            window_area_bucket=area,
            foreground=False,
            stable_across_probes=stable,
            explicit_activation_eligible=True,
            eligibility_reason="eligible",
            z_order_bucket="back",
        ),
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
    assert [item.trusted_app_id for item in result.diagnostics.observed_foregrounds] == [None, APP_ID]


def test_launcher_pid_may_differ_when_trusted_application_identity_matches() -> None:
    clock = FakeClock()
    initial = foreground("launcher", pid=111, hwnd=100, app_id="")
    active = foreground("final-window", app_id=APP_ID, pid=999, hwnd=200, app_name="trusted.exe")

    result = wait(ObservationSequence([active]), clock, initial)

    assert result.reason == "activated"
    assert result.diagnostics.initial_foreground.trusted_app_id is None
    assert result.diagnostics.final_foreground.trusted_app_id == APP_ID
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
    assert '"hwnd"' not in serialized and '"pid"' not in serialized


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


def test_lifecycle_reports_launch_acceptance_without_target_appearance() -> None:
    clock = FakeClock()
    state = TrustedApplicationRuntimeState()
    result = wait(
        ObservationSequence([foreground("old"), foreground("old-2")]),
        clock, foreground("initial"), timeout=.2, target_probe=lambda: state,
        post_deadline_probe_seconds=.2,
    )

    lifecycle = result.lifecycle
    assert result.reason == "activation_timeout"
    assert lifecycle is not None
    assert lifecycle.launch_request_accepted
    assert lifecycle.target_probe_available
    assert not lifecycle.target_process_observed
    assert not lifecycle.target_window_observed
    assert not lifecycle.target_foreground_observed
    assert lifecycle.first_target_process_elapsed_ms is None
    assert lifecycle.first_target_window_elapsed_ms is None
    assert lifecycle.first_target_foreground_elapsed_ms is None
    assert lifecycle.target_window_state is None
    assert lifecycle.post_deadline_probe.performed
    assert lifecycle.post_deadline_probe.delay_ms == 200
    assert not lifecycle.post_deadline_probe.target_process_observed


def test_lifecycle_distinguishes_process_without_window() -> None:
    clock = FakeClock()
    states = iter((
        TrustedApplicationRuntimeState(process_observed=True, trusted_identity_match=True),
        TrustedApplicationRuntimeState(process_observed=True, trusted_identity_match=True),
    ))
    result = wait(
        ObservationSequence([foreground("old"), foreground("old-2")]),
        clock, foreground("initial"), timeout=.2,
        target_probe=lambda: next(states),
    )

    lifecycle = result.lifecycle
    assert lifecycle is not None
    assert lifecycle.target_process_observed
    assert not lifecycle.target_window_observed
    assert lifecycle.first_target_process_elapsed_ms == 0
    assert lifecycle.first_target_window_elapsed_ms is None
    assert lifecycle.target_window_state is None


def test_lifecycle_marks_target_absence_inconclusive_when_probe_is_incomplete() -> None:
    clock = FakeClock()
    incomplete = TrustedApplicationRuntimeState(probe_complete=False)
    result = wait(
        ObservationSequence([foreground("old"), foreground("old-2")]),
        clock, foreground("initial"), timeout=.2,
        target_probe=lambda: incomplete,
    )

    lifecycle = result.lifecycle
    assert lifecycle is not None
    assert lifecycle.target_probe_available
    assert not lifecycle.target_probe_complete
    assert not lifecycle.target_process_observed
    assert not lifecycle.target_window_observed


def test_lifecycle_reports_sanitized_window_resolution_and_rejections() -> None:
    clock = FakeClock()
    candidate_diagnostic = ActivationWindowCandidateDiagnostics(
        candidate_index=1,
        trusted_identity_match=True,
        visible=False,
        minimized=False,
        enabled=True,
        cloaked_if_available=False,
        owner_present=False,
        root_owner_relationship="self",
        tool_window=False,
        app_window=False,
        has_nonzero_client_area=True,
        client_area_bucket="large",
        window_area_bucket="large",
        foreground=False,
        stable_across_probes=False,
        z_order_bucket="back",
        activation_candidate=True,
        explicit_activation_eligible=False,
        rejection_reason="target_not_visible",
    )
    state = TrustedApplicationRuntimeState(
        process_observed=True,
        window_observed=True,
        trusted_identity_match=True,
        probe_complete=True,
        window_resolution_status="none",
        matching_trusted_window_count=2,
        eligible_window_count=0,
        enumeration_complete=True,
        rejection_reason_counts=(
            ("target_not_visible", 2),
            ("C:\\private\\window-title", 999),
        ),
        window_candidates=(candidate_diagnostic,),
    )

    result = wait(
        ObservationSequence([foreground("old")]),
        clock,
        foreground("initial"),
        timeout=0,
        target_probe=lambda: state,
        target_activator=lambda: (_ for _ in ()).throw(AssertionError("must fail closed")),
    )

    lifecycle = result.lifecycle
    assert lifecycle is not None
    assert lifecycle.window_resolution_status == "none"
    assert lifecycle.matching_trusted_window_count == 2
    assert lifecycle.eligible_window_count == 0
    assert lifecycle.enumeration_complete
    assert lifecycle.rejection_reason_counts == (("target_not_visible", 2),)
    assert lifecycle.matching_window_candidates[0].rejection_reason == "target_not_visible"
    assert lifecycle.explicit_activation is not None
    assert lifecycle.explicit_activation.eligibility_reason == "no_eligible_window"
    assert "private" not in json.dumps(asdict(lifecycle))


def test_eligible_window_history_correlates_initial_repeated_and_post_deadline_probes() -> None:
    clock = FakeClock()
    relation = EligibleWindowRelationship(
        first_diagnostic_id="ew1",
        second_diagnostic_id="ew2",
        relationship="independent",
    )
    post_relation = EligibleWindowRelationship(
        first_diagnostic_id="ew1",
        second_diagnostic_id="ew2",
        relationship="one_owned_by_other",
    )

    def state(
        diagnostics: tuple[EligibleWindowDiagnostics, ...],
        relationships: tuple[EligibleWindowRelationship, ...] = (),
    ) -> TrustedApplicationRuntimeState:
        return TrustedApplicationRuntimeState(
            process_observed=True,
            window_observed=True,
            trusted_identity_match=True,
            probe_complete=True,
            window_ambiguous=True,
            window_resolution_status="ambiguous",
            matching_trusted_window_count=17,
            eligible_window_count=2,
            enumeration_complete=True,
            eligible_window_diagnostics=diagnostics,
            eligible_window_relationships=relationships,
        )

    states = iter((
        state((eligible_diagnostic("ew1", area="large", index=1),)),
        state((
            eligible_diagnostic("ew1", area="medium", index=1, stable=True),
            eligible_diagnostic("ew2", area="small", index=2),
        ), (relation,)),
        state((
            eligible_diagnostic("ew1", area="large", index=1, stable=True),
            eligible_diagnostic("ew2", area="medium", index=2, stable=True),
        ), (post_relation,)),
        state((
            eligible_diagnostic("ew1", area="large", index=1, stable=True),
            eligible_diagnostic("ew2", area="large", index=2, stable=True),
        ), (post_relation,)),
    ))
    activation_calls: list[str] = []
    result = wait(
        ObservationSequence([foreground("old"), foreground("old-2")]),
        clock,
        foreground("initial"),
        timeout=.2,
        target_probe=lambda: next(states),
        target_activator=lambda: activation_calls.append("activate")
        or TrustedWindowActivationResult(True, "eligible"),
        post_deadline_probe_seconds=.2,
    )

    lifecycle = result.lifecycle
    assert lifecycle is not None
    assert lifecycle.window_resolution_status == "ambiguous"
    assert not lifecycle.eligible_window_diagnostics_truncated
    assert [item.diagnostic_id for item in lifecycle.eligible_window_diagnostics] == [
        "ew1", "ew2",
    ]
    first, second = lifecycle.eligible_window_diagnostics
    assert first.present_in_initial_probe and first.present_in_repeated_probe
    assert first.present_in_post_deadline_probe
    assert first.initial_probe_facts.client_area_bucket == "large"
    assert first.latest_repeated_probe_facts.client_area_bucket == "medium"
    assert not second.present_in_initial_probe
    assert second.present_in_repeated_probe and second.present_in_post_deadline_probe
    assert second.latest_repeated_probe_facts.client_area_bucket == "small"
    assert second.latest_post_deadline_probe_facts.client_area_bucket == "large"
    assert second.observed_probe_count == 3
    ownership = lifecycle.eligible_window_relationships[0]
    assert ownership.present_in_repeated_probe and ownership.present_in_post_deadline_probe
    assert ownership.latest_repeated_probe_relationship == "independent"
    assert ownership.latest_post_deadline_probe_relationship == "one_owned_by_other"
    assert lifecycle.explicit_activation is not None
    assert not lifecycle.explicit_activation.attempted
    assert not lifecycle.explicit_activation.budget_consumed
    assert activation_calls == []


def test_lifecycle_reports_background_and_minimized_target_windows() -> None:
    clock = FakeClock()
    background = TrustedApplicationRuntimeState(
        process_observed=True, window_observed=True, visible=True,
        minimized=False, window_foreground=False, trusted_identity_match=True,
    )
    result = wait(
        ObservationSequence([foreground("old"), foreground("old-2")]),
        clock, foreground("initial"), timeout=.2, target_probe=lambda: background,
    )
    lifecycle = result.lifecycle
    assert lifecycle is not None
    assert lifecycle.target_window_observed
    assert not lifecycle.target_foreground_observed
    assert lifecycle.target_window_state is not None
    assert lifecycle.target_window_state.visible is True
    assert lifecycle.target_window_state.minimized is False
    assert lifecycle.target_window_state.foreground is False
    assert lifecycle.target_window_state.trusted_identity_match

    minimized = TrustedApplicationRuntimeState(
        process_observed=True, window_observed=True, visible=True,
        minimized=True, window_foreground=False, trusted_identity_match=True,
    )
    minimized_result = wait(
        ObservationSequence([foreground("old"), foreground("old-2")]),
        FakeClock(), foreground("initial"), timeout=.2,
        target_probe=lambda: minimized,
    )
    minimized_state = minimized_result.lifecycle.target_window_state
    assert minimized_state is not None
    assert minimized_state.visible is True
    assert minimized_state.minimized is True
    assert minimized_state.foreground is False


def test_launch_timing_records_window_milestones_and_immediate_eligible_attempt() -> None:
    clock = FakeClock()
    clock.value = 0.05  # OpenAppAction returned after 50 ms.
    first_probe = TrustedApplicationRuntimeState(
        process_observed=True,
        trusted_identity_match=True,
        probe_complete=True,
        enumeration_complete=True,
        primary_surface_candidate_count=0,
        primary_surface_resolution_status="none",
    )
    primary_facts = EligibleWindowFacts(
        visible=True,
        minimized=False,
        enabled=True,
        cloaked_state="uncloaked",
        owner_present=False,
        root_owner_relationship="self",
        tool_window=False,
        app_window=False,
        has_nonzero_client_area=True,
        client_area_bucket="large",
        window_area_bucket="large",
        foreground=False,
        stable_across_probes=True,
        explicit_activation_eligible=True,
        eligibility_reason="eligible",
    )
    second_probe = TrustedApplicationRuntimeState(
        process_observed=True,
        window_observed=True,
        visible=True,
        minimized=False,
        window_foreground=False,
        trusted_identity_match=True,
        probe_complete=True,
        window_stable=True,
        window_resolution_status="unique",
        matching_trusted_window_count=1,
        eligible_window_count=1,
        enumeration_complete=True,
        eligible_window_diagnostics=(EligibleWindowDiagnostics(
            "ew1", 1, primary_facts, surface_class="primary",
        ),),
        primary_surface_candidate_count=1,
        primary_surface_resolution_status="unique",
        primary_surface_facts=primary_facts,
    )
    probes = iter((first_probe, second_probe))
    activation_calls: list[float] = []

    result = wait_for_trusted_application_activation(
        ObservationSequence([
            foreground("old-1", app_id="other-app"),
            foreground("old-2", app_id="other-app"),
            foreground("old-after-attempt", app_id="other-app"),
        ]),
        candidate(),
        initial_observation=foreground("initial", app_id="other-app"),
        launch_succeeded=True,
        timeout_seconds=0.4,
        poll_interval_seconds=0.2,
        clock=clock,
        sleep_fn=clock.sleep,
        target_probe=lambda: next(probes),
        target_activator=lambda: activation_calls.append(clock()) or TrustedWindowActivationResult(
            True, "eligible", mechanism="activate", budget_consumed=True,
            foreground_verified=False, failure_reason="os_activation_rejected",
        ),
        open_app_action_started_at=0.0,
        open_app_action_elapsed_ms=50,
    )

    assert result.lifecycle is not None
    timing = result.lifecycle.timing
    assert timing is not None
    assert timing.open_app_action_elapsed_ms == 50
    assert timing.activation_wait_started_since_launch_ms == 50
    assert timing.first_process_since_launch_ms == 50
    assert timing.first_primary_window_since_launch_ms == 250
    assert timing.first_visible_primary_window_since_launch_ms == 250
    assert timing.first_stable_eligible_window_since_launch_ms == 250
    assert timing.explicit_activation_attempt_since_launch_ms == 250
    assert timing.eligible_to_attempt_delay_ms == 0
    assert len(activation_calls) == 1


def test_launch_timing_records_request_observation_probe_and_foreground_transition() -> None:
    clock = FakeClock()
    clock.value = 0.02  # The launcher returned 20 ms after its request started.
    observations = iter((
        foreground("after-launch-old", app_id="other-app"),
        foreground("after-launch-target", app_id=APP_ID, pid=20, hwnd=200),
    ))

    def observe() -> Observation:
        clock.value += 0.03
        return next(observations)

    probe_states = iter((
        TrustedApplicationRuntimeState(
            process_observed=True, trusted_identity_match=True, probe_complete=True,
        ),
        TrustedApplicationRuntimeState(
            process_observed=True, window_observed=True, foreground_observed=True,
            visible=True, minimized=False, window_foreground=True,
            trusted_identity_match=True, probe_complete=True,
        ),
    ))

    def probe() -> TrustedApplicationRuntimeState:
        clock.value += 0.04
        return next(probe_states)

    result = wait_for_trusted_application_activation(
        observe, candidate(), initial_observation=foreground("before", app_id="other-app"),
        launch_succeeded=True, timeout_seconds=1, poll_interval_seconds=.2,
        clock=clock, sleep_fn=clock.sleep, target_probe=probe,
        target_activator=lambda: pytest.fail("passive foreground must skip activation"),
        open_app_action_started_at=0.0, open_app_action_elapsed_ms=20,
    )

    assert result.lifecycle is not None and result.lifecycle.timing is not None
    timing = result.lifecycle.timing
    assert timing.launch_request_started_since_launch_ms == 0
    assert timing.launch_request_finished_since_launch_ms == 20
    assert timing.post_launch_setup_delay_ms == 0
    assert timing.foreground_identity_before_launch.trusted_app_id == "other-app"
    assert timing.first_post_launch_foreground_identity.trusted_app_id == "other-app"
    assert timing.first_post_launch_observation_started_since_launch_ms == 20
    assert timing.first_post_launch_observation_finished_since_launch_ms == 50
    assert timing.first_post_launch_target_probe_started_since_launch_ms == 50
    assert timing.first_post_launch_target_probe_finished_since_launch_ms == 90
    assert timing.first_foreground_since_launch_ms == 360
    assert len(timing.foreground_transitions_since_launch) == 1
    transition = timing.foreground_transitions_since_launch[0]
    assert transition.elapsed_since_launch_ms == 320
    assert transition.identity.trusted_app_id == APP_ID


def test_lifecycle_records_foreground_success_without_pid_equality() -> None:
    clock = FakeClock()
    active = foreground("active", app_id=APP_ID, pid=999, hwnd=200, app_name="trusted.exe")
    target = TrustedApplicationRuntimeState(
        process_observed=True, window_observed=True, foreground_observed=True,
        visible=True, minimized=False, window_foreground=True,
        trusted_identity_match=True,
    )
    result = wait(
        ObservationSequence([active]), clock, foreground("launcher", pid=111),
        target_probe=lambda: target,
    )

    assert result.reason == "activated"
    assert result.diagnostics.initial_foreground.trusted_app_id is None
    assert result.diagnostics.final_foreground.trusted_app_id == APP_ID
    lifecycle = result.lifecycle
    assert lifecycle is not None
    assert lifecycle.launch_request_accepted
    assert lifecycle.target_process_observed
    assert lifecycle.target_window_observed
    assert lifecycle.target_foreground_observed
    assert lifecycle.first_target_process_elapsed_ms == 0
    assert lifecycle.first_target_window_elapsed_ms == 0
    assert lifecycle.first_target_foreground_elapsed_ms == 0
    assert lifecycle.target_window_state.foreground is True
    assert not lifecycle.post_deadline_probe.performed


def test_post_deadline_target_appearance_is_diagnostic_only() -> None:
    clock = FakeClock()
    old = foreground("still-old", app_id="other-app", pid=111, hwnd=100)
    late_target = TrustedApplicationRuntimeState(
        process_observed=True, window_observed=True, foreground_observed=True,
        visible=True, minimized=False, window_foreground=True,
        trusted_identity_match=True,
    )
    calls = 0

    def probe() -> TrustedApplicationRuntimeState:
        nonlocal calls
        calls += 1
        return TrustedApplicationRuntimeState() if calls <= 2 else late_target

    result = wait(
        ObservationSequence([old, old, old]), clock, foreground("initial"),
        timeout=.2, target_probe=probe, post_deadline_probe_seconds=.3,
    )

    assert result.reason == "activation_timeout"
    assert result.observation is old
    assert result.diagnostics.activation_elapsed_ms == 200
    lifecycle = result.lifecycle
    assert lifecycle is not None
    assert not lifecycle.target_process_observed
    assert not lifecycle.target_window_observed
    assert not lifecycle.target_foreground_observed
    assert lifecycle.post_deadline_probe.performed
    assert lifecycle.post_deadline_probe.delay_ms == 300
    assert lifecycle.post_deadline_probe.target_process_observed
    assert lifecycle.post_deadline_probe.target_window_observed
    assert lifecycle.post_deadline_probe.target_foreground_observed


def test_lifecycle_diagnostics_are_bounded_and_contain_no_title_or_path() -> None:
    clock = FakeClock()
    private_title = "Private Conversation - Alice"
    private_path = "C:\\Users\\someone\\Documents\\private.txt"
    window_state = TrustedApplicationRuntimeState(
        process_observed=True, window_observed=True, visible=True,
        minimized=False, window_foreground=False, trusted_identity_match=True,
    )
    result = wait(
        ObservationSequence([
            foreground("old", title=private_title),
            foreground("old-2", title=private_path),
            foreground("old-3", title=private_title),
        ]),
        clock, foreground("initial", title=private_title), timeout=.2,
        target_probe=lambda: window_state, post_deadline_probe_seconds=.3,
    )
    serialized = json.dumps(asdict(result.lifecycle))

    assert result.diagnostics.activation_attempts <= 3
    assert result.lifecycle.post_deadline_probe.delay_ms <= 300
    assert private_title not in serialized
    assert private_path not in serialized
    assert "hwnd" not in serialized and '"pid"' not in serialized

    try:
        wait(
            ObservationSequence([foreground("old")]), FakeClock(), foreground("initial"),
            target_probe=lambda: window_state, post_deadline_probe_seconds=3.01,
        )
    except ValueError as exc:
        assert "three seconds" in str(exc)
    else:
        raise AssertionError("post-deadline probe must be bounded to three seconds")


def test_passive_foreground_activation_never_uses_explicit_fallback() -> None:
    clock = FakeClock()
    target = TrustedApplicationRuntimeState(
        process_observed=True, window_observed=True, foreground_observed=True,
        visible=True, minimized=False, window_foreground=True,
        trusted_identity_match=True, window_stable=True,
    )
    calls: list[str] = []
    active = foreground("active", app_id=APP_ID, app_name="trusted.exe")

    result = wait(
        ObservationSequence([active]), clock, foreground("before"),
        target_probe=lambda: target,
        target_activator=lambda: calls.append("activate") or TrustedWindowActivationResult(
            True, "eligible", True, False, True, "activate", True,
            foreground_verified=True,
        ),
    )

    assert result.reason == "activated"
    assert calls == []
    explicit = result.lifecycle.explicit_activation
    assert explicit is not None
    assert not explicit.attempted and not explicit.budget_consumed
    assert explicit.eligibility_reason == "passive_activation_observed"


def test_fallback_rejects_unstable_invisible_ambiguous_and_untrusted_windows() -> None:
    cases = (
        (TrustedApplicationRuntimeState(
            process_observed=True, window_observed=True, visible=True,
            window_foreground=False, trusted_identity_match=True,
        ), "target_window_unstable"),
        (TrustedApplicationRuntimeState(
            process_observed=True, window_observed=True, visible=False, minimized=False,
            window_foreground=False, trusted_identity_match=True, window_stable=True,
        ), "target_not_visible"),
        (TrustedApplicationRuntimeState(
            process_observed=True, window_observed=True, window_ambiguous=True,
            trusted_identity_match=True, probe_complete=True,
        ), "target_window_ambiguous"),
        (TrustedApplicationRuntimeState(
            process_observed=True, window_observed=True, visible=True,
            window_stable=True, trusted_identity_match=False,
        ), "trusted_identity_mismatch"),
    )

    for state, expected_reason in cases:
        calls: list[str] = []
        result = wait(
            ObservationSequence([foreground("old")]), FakeClock(), foreground("initial"),
            timeout=0, target_probe=lambda state=state: state,
            target_activator=lambda: calls.append("activate") or TrustedWindowActivationResult(
                True, "eligible", True, False, True, "activate", True,
            ),
        )

        explicit = result.lifecycle.explicit_activation
        assert explicit is not None
        assert not explicit.attempted and not explicit.budget_consumed
        assert explicit.eligibility_reason == expected_reason
        assert calls == []


def test_stable_trusted_background_window_activates_once_and_requires_fresh_observation() -> None:
    clock = FakeClock()
    stable_background = TrustedApplicationRuntimeState(
        process_observed=True, window_observed=True, visible=True, minimized=False,
        window_foreground=False, trusted_identity_match=True, window_stable=True,
    )
    old = foreground("old", app_id="old-app")
    active = foreground("fresh-active", app_id=APP_ID, app_name="trusted.exe")
    sequence = ObservationSequence([old, active])
    calls: list[str] = []

    result = wait(
        sequence, clock, foreground("initial", app_id="old-app"),
        target_probe=lambda: stable_background,
        target_activator=lambda: calls.append("activate") or TrustedWindowActivationResult(
            True, "eligible", True, False, True, "activate", True,
            foreground_verified=True,
        ),
    )

    explicit = result.lifecycle.explicit_activation
    assert result.reason == "activated"
    assert calls == ["activate"]
    assert explicit is not None
    assert explicit.eligible and explicit.attempted and explicit.budget_consumed
    assert explicit.mechanism == "activate"
    assert explicit.os_call_reported_success is True
    assert explicit.fresh_verification_obtained
    assert explicit.foreground_after and explicit.trusted_identity_after
    assert explicit.success and explicit.failure_reason is None
    assert explicit.window_state_before.visible is True
    assert explicit.window_state_before.minimized is False
    assert explicit.window_state_before.trusted_identity_match


def test_activation_api_success_without_fresh_foreground_verification_fails_closed() -> None:
    clock = FakeClock()
    stable_background = TrustedApplicationRuntimeState(
        process_observed=True, window_observed=True, visible=True,
        minimized=False, window_foreground=False, trusted_identity_match=True,
        window_stable=True,
    )
    old = foreground("still-old", app_id="old-app")
    calls: list[str] = []
    result = wait(
        ObservationSequence([old, old, old, old]), clock, foreground("initial"),
        timeout=.4, target_probe=lambda: stable_background,
        target_activator=lambda: calls.append("activate") or TrustedWindowActivationResult(
            True, "eligible", True, False, True, "activate", True,
            foreground_verified=True,
        ),
    )

    explicit = result.lifecycle.explicit_activation
    assert result.reason == "activation_timeout"
    assert calls == ["activate"]
    assert explicit is not None
    assert explicit.attempted and explicit.budget_consumed
    assert explicit.os_call_reported_success is True
    assert explicit.fresh_verification_obtained
    assert explicit.foreground_after is False
    assert explicit.trusted_identity_after is False
    assert not explicit.success
    assert explicit.failure_reason == "trusted_foreground_not_observed"


def test_os_activation_rejection_is_verified_once_without_retrying() -> None:
    clock = FakeClock()
    stable_background = TrustedApplicationRuntimeState(
        process_observed=True, window_observed=True, visible=True,
        minimized=False, window_foreground=False, trusted_identity_match=True,
        window_stable=True,
    )
    old = foreground("still-old", app_id="old-app")
    calls: list[str] = []
    result = wait(
        ObservationSequence([old, old, old, old]), clock, foreground("initial"),
        timeout=.4, target_probe=lambda: stable_background,
        target_activator=lambda: calls.append("activate") or TrustedWindowActivationResult(
            True, "eligible", True, False, True,
            "activate", False, "foreground_not_selected", foreground_verified=False,
        ),
    )

    explicit = result.lifecycle.explicit_activation
    assert calls == ["activate"]
    assert explicit is not None
    assert explicit.budget_consumed and explicit.attempted
    assert explicit.os_call_reported_success is False
    assert explicit.failure_reason == "foreground_not_selected"
    assert explicit.selected_window_verified is False
    assert not explicit.success


def test_fresh_exact_window_verification_overrides_false_api_return() -> None:
    clock = FakeClock()
    stable_background = TrustedApplicationRuntimeState(
        process_observed=True, window_observed=True, visible=True, minimized=False,
        window_foreground=False, trusted_identity_match=True, window_stable=True,
    )
    active = foreground("fresh-active", app_id=APP_ID, app_name="trusted.exe")
    result = wait(
        ObservationSequence([foreground("old", app_id="old-app"), active]),
        clock, foreground("initial", app_id="old-app"),
        target_probe=lambda: stable_background,
        target_activator=lambda: TrustedWindowActivationResult(
            True, "eligible", True, False, True, "activate", False,
            foreground_verified=True,
        ),
    )

    explicit = result.lifecycle.explicit_activation
    assert result.reason == "activated"
    assert explicit is not None
    assert explicit.os_call_reported_success is False
    assert explicit.selected_window_verified is True
    assert explicit.success


def test_thread_input_fallback_diagnostics_are_safely_exposed_and_gate_detach_failure() -> None:
    stable_background = TrustedApplicationRuntimeState(
        process_observed=True, window_observed=True, visible=True, minimized=False,
        window_foreground=False, trusted_identity_match=True, window_stable=True,
    )
    active = foreground("fresh-active", app_id=APP_ID, app_name="trusted.exe")
    fallback = ThreadInputFallbackAttempt(
        eligible=True,
        reason="detach_failed",
        foreground_hwnd_present=True,
        selected_thread_resolved=True,
        foreground_thread_resolved=True,
        current_thread_resolved=True,
        attach_attempted=True,
        attach_succeeded=True,
        incidental_foreground_changed_before_activation=True,
        selected_target_still_valid_before_activation=True,
        set_foreground_attempted=True,
        set_foreground_return=True,
        detach_succeeded=False,
        verified=True,
        failure_reason="detach_failed",
    )

    result = wait(
        ObservationSequence([foreground("old"), active]), FakeClock(),
        foreground("initial", app_id="old-app"),
        target_probe=lambda: stable_background,
        target_activator=lambda: TrustedWindowActivationResult(
            True, "eligible", True, False, True, "activate", False,
            "detach_failed", True, True,
            activation_strategy="attach_thread_input",
            simple_attempt=SimpleActivationAttempt(False, False),
            fallback_attempt=fallback,
        ),
    )

    explicit = result.lifecycle.explicit_activation
    assert result.reason == "activation_timeout"
    assert explicit is not None
    assert explicit.activation_strategy == "attach_thread_input"
    assert explicit.simple_attempt == SimpleActivationAttempt(False, False)
    assert explicit.fallback_attempt is not None
    assert explicit.fallback_attempt.attach_attempted
    assert explicit.fallback_attempt.attach_succeeded is True
    assert explicit.fallback_attempt.incidental_foreground_changed_before_activation is True
    assert explicit.fallback_attempt.selected_target_still_valid_before_activation is True
    assert explicit.fallback_attempt.set_foreground_attempted is True
    assert explicit.fallback_attempt.detach_succeeded is False
    assert explicit.fallback_attempt.failure_reason == "detach_failed"
    assert not explicit.success
    serialized = json.dumps(asdict(result.lifecycle))
    assert "foreground_hwnd_present" in serialized
    assert '"hwnd"' not in serialized and '"pid"' not in serialized


def test_same_app_observation_cannot_override_selected_window_verification_failure() -> None:
    clock = FakeClock()
    stable_background = TrustedApplicationRuntimeState(
        process_observed=True, window_observed=True, visible=True, minimized=False,
        window_foreground=False, trusted_identity_match=True, window_stable=True,
    )
    same_app_but_different_window = foreground(
        "fresh-same-app", app_id=APP_ID, app_name="trusted.exe",
    )
    result = wait(
        ObservationSequence([
            foreground("old", app_id="old-app"), same_app_but_different_window,
        ]),
        clock, foreground("initial", app_id="old-app"), timeout=.1,
        target_probe=lambda: stable_background,
        target_activator=lambda: TrustedWindowActivationResult(
            True, "eligible", True, False, True, "activate", True,
            "foreground_not_selected", foreground_verified=False,
        ),
    )

    explicit = result.lifecycle.explicit_activation
    assert result.reason == "activation_timeout"
    assert explicit is not None
    assert explicit.trusted_identity_after is True
    assert explicit.foreground_after is True
    assert explicit.selected_window_verified is False
    assert explicit.failure_reason == "foreground_not_selected"
    assert not explicit.success


def test_same_app_observation_cannot_override_preflight_window_binding_failure() -> None:
    clock = FakeClock()
    stable_background = TrustedApplicationRuntimeState(
        process_observed=True, window_observed=True, visible=True, minimized=False,
        window_foreground=False, trusted_identity_match=True, window_stable=True,
    )
    same_app = foreground("fresh-same-app", app_id=APP_ID, app_name="trusted.exe")
    result = wait(
        ObservationSequence([foreground("old", app_id="old-app"), same_app]),
        clock, foreground("initial", app_id="old-app"), timeout=.1,
        target_probe=lambda: stable_background,
        target_activator=lambda: TrustedWindowActivationResult(
            False, "stale_window", trusted_identity_match=True,
            failure_reason="stale_window", budget_consumed=False,
        ),
    )

    explicit = result.lifecycle.explicit_activation
    assert result.reason == "activation_timeout"
    assert explicit is not None
    assert not explicit.attempted
    assert explicit.eligibility_reason == "stale_window"
    assert not explicit.success


def test_activation_diagnostics_sanitize_unexpected_adapter_strings() -> None:
    clock = FakeClock()
    background = TrustedApplicationRuntimeState(
        process_observed=True, window_observed=True, visible=True,
        minimized=False, window_foreground=False, trusted_identity_match=True,
        window_stable=True,
    )
    private = "C:\\Users\\Alice\\private.txt"
    result = wait(
        ObservationSequence([foreground("old"), foreground("old")]),
        clock, foreground("initial"), timeout=.1,
        target_probe=lambda: background,
        target_activator=lambda: TrustedWindowActivationResult(
            True, private, True, False, True, private, False, private,
        ),
    )

    serialized = json.dumps(asdict(result.lifecycle))
    explicit = result.lifecycle.explicit_activation
    assert explicit is not None
    assert explicit.eligibility_reason == "activation_error"
    assert explicit.mechanism is None and explicit.failure_reason is None
    assert private not in serialized
    assert '"hwnd"' not in serialized and '"pid"' not in serialized


def test_window_candidate_diagnostics_are_sanitized_before_lifecycle_serialization() -> None:
    clock = FakeClock()
    private = "C:\\Users\\Alice\\private-window-title"
    diagnostic = ActivationWindowCandidateDiagnostics(
        candidate_index=1,
        trusted_identity_match=True,
        visible=True,
        minimized=False,
        enabled=True,
        cloaked_if_available=False,
        owner_present=False,
        root_owner_relationship=private,  # type: ignore[arg-type]
        tool_window=False,
        app_window=True,
        has_nonzero_client_area=True,
        client_area_bucket=private,  # type: ignore[arg-type]
        window_area_bucket="large",
        foreground=False,
        stable_across_probes=True,
        z_order_bucket=private,  # type: ignore[arg-type]
        activation_candidate=True,
    )
    state = TrustedApplicationRuntimeState(
        process_observed=True,
        window_observed=True,
        window_ambiguous=True,
        trusted_identity_match=True,
        window_candidates=(diagnostic,),
    )

    activation_calls: list[str] = []
    result = wait(
        ObservationSequence([foreground("old")]), clock, foreground("initial"),
        timeout=0, target_probe=lambda: state,
        target_activator=lambda: activation_calls.append("activate")
        or TrustedWindowActivationResult(True, "eligible"),
    )

    serialized = json.dumps(asdict(result.lifecycle))
    assert private not in serialized
    explicit = result.lifecycle.explicit_activation
    assert explicit is not None
    assert explicit.eligibility_reason == "target_window_ambiguous"
    assert not explicit.attempted and not explicit.budget_consumed
    assert activation_calls == []
    candidate_diagnostic = result.lifecycle.matching_window_candidates[0]
    assert candidate_diagnostic.root_owner_relationship == "unavailable"
    assert candidate_diagnostic.client_area_bucket == "unavailable"
    assert candidate_diagnostic.z_order_bucket == "unavailable"


def test_explicit_activation_uses_unique_primary_surface_when_base_has_tool_window() -> None:
    primary_facts = EligibleWindowFacts(
        visible=True,
        minimized=False,
        enabled=True,
        cloaked_state="uncloaked",
        owner_present=False,
        root_owner_relationship="self",
        tool_window=False,
        app_window=False,
        has_nonzero_client_area=True,
        client_area_bucket="large",
        window_area_bucket="large",
        foreground=False,
        stable_across_probes=True,
        explicit_activation_eligible=True,
        eligibility_reason="eligible",
    )
    tool_facts = EligibleWindowFacts(
        **{
            **asdict(primary_facts),
            "tool_window": True,
            "stable_across_probes": True,
        }
    )
    state = TrustedApplicationRuntimeState(
        process_observed=True,
        window_observed=True,
        trusted_identity_match=True,
        probe_complete=True,
        window_ambiguous=True,
        window_resolution_status="ambiguous",
        matching_trusted_window_count=2,
        eligible_window_count=2,
        enumeration_complete=True,
        eligible_window_diagnostics=(
            EligibleWindowDiagnostics(
                "ew1", 1, primary_facts, surface_class="primary",
            ),
            EligibleWindowDiagnostics(
                "ew2", 2, tool_facts, surface_class="tool",
            ),
        ),
        base_eligible_window_count=2,
        primary_surface_candidate_count=1,
        tool_surface_candidate_count=1,
        primary_surface_resolution_status="unique",
        primary_surface_facts=primary_facts,
    )
    active = foreground("primary-activated", app_id=APP_ID, app_name="trusted.exe")
    calls: list[str] = []

    result = wait(
        ObservationSequence([foreground("old"), active]),
        FakeClock(),
        foreground("initial", app_id="other-app"),
        timeout=.2,
        target_probe=lambda: state,
        target_activator=lambda: calls.append("activate") or TrustedWindowActivationResult(
            True, "eligible", True, False, True, "activate", True,
        ),
    )

    lifecycle = result.lifecycle
    assert lifecycle is not None
    assert lifecycle.window_resolution_status == "ambiguous"
    assert lifecycle.base_eligible_window_count == 2
    assert lifecycle.primary_surface_candidate_count == 1
    assert lifecycle.tool_surface_candidate_count == 1
    assert lifecycle.primary_surface_resolution == "unique"
    assert [item.surface_class for item in lifecycle.eligible_window_diagnostics] == [
        "primary", "tool",
    ]
    assert calls == ["activate"]
    assert lifecycle.explicit_activation is not None
    assert lifecycle.explicit_activation.eligible
