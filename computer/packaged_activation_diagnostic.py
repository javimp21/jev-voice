"""Manual, one-mechanism-at-a-time packaged-app activation diagnostics."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
import time
from typing import Literal

from agent.application_activation import (
    ActivationTimingDiagnostics, wait_for_trusted_application_activation,
)
from computer.applications import ApplicationCandidate, TrustedApplicationRuntimeState
from computer.models import Observation
from computer.windows_apps import PackagedLaunchDiagnosticResult, WindowsApplicationCatalog


Mechanism = Literal["shell", "activation-manager"]


@dataclass(frozen=True, slots=True)
class PackagedActivationDiagnostic:
    success: bool
    mechanism: Mechanism
    trusted_app_id: str | None
    launch_kind: str
    activation_request_succeeded: bool
    activation_call_elapsed_ms: int | None
    returned_pid_present: bool
    activation_hresult: int | None
    target_process_observed_before_request: bool
    target_window_observed_before_request: bool
    target_foreground_observed_before_request: bool
    target_process_observed: bool
    target_window_observed: bool
    primary_surface_unique: bool
    primary_window_visible: bool | None
    primary_window_stable: bool
    automatic_foreground_observed: bool
    automatic_foreground_elapsed_ms: int | None
    post_launch_setforeground_required: bool | None
    explicit_activation_attempted: bool
    explicit_foreground_attempted: bool
    explicit_simple_setforeground_return: bool | None
    explicit_fallback_setforeground_attempted: bool
    explicit_fallback_setforeground_return: bool | None
    explicit_set_foreground_returned_successfully: bool | None
    explicit_foreground_succeeded: bool | None
    final_foreground_verified: bool
    failure_stage: str | None
    failure_reason: str | None
    timing: ActivationTimingDiagnostics | None = None


def _failure(
    mechanism: Mechanism,
    *,
    stage: str,
    reason: str,
    app_id: str | None = None,
    launch_kind: str = "unknown",
) -> PackagedActivationDiagnostic:
    return PackagedActivationDiagnostic(
        success=False, mechanism=mechanism, trusted_app_id=app_id,
        launch_kind=launch_kind, activation_request_succeeded=False,
        activation_call_elapsed_ms=None, returned_pid_present=False,
        activation_hresult=None, target_process_observed_before_request=False,
        target_window_observed_before_request=False,
        target_foreground_observed_before_request=False,
        target_process_observed=False, target_window_observed=False,
        primary_surface_unique=False, primary_window_visible=None,
        primary_window_stable=False, automatic_foreground_observed=False,
        automatic_foreground_elapsed_ms=None,
        post_launch_setforeground_required=None,
        explicit_activation_attempted=False,
        explicit_foreground_attempted=False,
        explicit_simple_setforeground_return=None,
        explicit_fallback_setforeground_attempted=False,
        explicit_fallback_setforeground_return=None,
        explicit_set_foreground_returned_successfully=None,
        explicit_foreground_succeeded=None, final_foreground_verified=False,
        failure_stage=stage, failure_reason=reason,
    )


def run_packaged_activation_diagnostic(
    query: str,
    mechanism: Mechanism,
    catalog: WindowsApplicationCatalog,
    *,
    computer_factory: Callable[..., object],
    timeout_seconds: float = 8.0,
    poll_interval_seconds: float = 0.2,
    activation_manager: Callable[[str], PackagedLaunchDiagnosticResult] | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> PackagedActivationDiagnostic:
    """Launch one uniquely resolved packaged app and monitor its trusted surface."""
    if mechanism not in {"shell", "activation-manager"}:
        raise ValueError("Unsupported packaged activation diagnostic mechanism.")
    if not 0.1 <= timeout_seconds <= 15:
        raise ValueError("Diagnostic timeout must be between 0.1 and 15 seconds.")
    if not 0.01 <= poll_interval_seconds <= 1:
        raise ValueError("Diagnostic poll interval must be between 10 ms and one second.")

    matches = catalog.find(query, 5)
    if not matches or (len(matches) > 1 and matches[0].score == matches[1].score):
        return _failure(mechanism, stage="resolve", reason="no_unique_application_match")
    candidate: ApplicationCandidate = matches[0].candidate
    if candidate.source != "packaged":
        return _failure(
            mechanism, stage="resolve", reason="not_a_packaged_application",
            app_id=candidate.id, launch_kind=candidate.source,
        )
    if candidate.launch_policy != "allow":
        return _failure(
            mechanism, stage="resolve", reason="application_launch_not_allowed",
            app_id=candidate.id, launch_kind="packaged",
        )

    computer = computer_factory(app_catalog=catalog)
    target_probe = computer.activation_target_probe(candidate)
    if not callable(target_probe):
        return _failure(
            mechanism, stage="setup", reason="trusted_window_probe_unavailable",
            app_id=candidate.id, launch_kind="packaged",
        )
    try:
        initial_observation: Observation = computer.observe_local()
        before_request: TrustedApplicationRuntimeState = target_probe()
    except KeyboardInterrupt:
        raise
    except Exception:
        return _failure(
            mechanism, stage="setup", reason="local_observation_failed",
            app_id=candidate.id, launch_kind="packaged",
        )
    if initial_observation.error:
        return _failure(
            mechanism, stage="setup", reason="local_observation_failed",
            app_id=candidate.id, launch_kind="packaged",
        )

    if mechanism == "activation-manager" and activation_manager is None:
        from computer.windows_activation_manager import activate_application

        activation_manager = activate_application

    request_started_at = clock()
    try:
        request = catalog.launch_packaged_for_diagnostic(
            candidate.id, mechanism, activation_manager=activation_manager,
        )
    except KeyboardInterrupt:
        raise
    except ValueError:
        return _failure(
            mechanism, stage="launch", reason="trusted_application_binding_unavailable",
            app_id=candidate.id, launch_kind="packaged",
        )
    except Exception:
        return _failure(
            mechanism, stage="launch", reason="launch_request_failed",
            app_id=candidate.id, launch_kind="packaged",
        )

    if not request.request_succeeded:
        result = _failure(
            mechanism, stage="launch",
            reason=request.failure_reason or "launch_request_failed",
            app_id=candidate.id, launch_kind="packaged",
        )
        return replace(
            result,
            activation_call_elapsed_ms=request.request_elapsed_ms,
            returned_pid_present=request.returned_pid_present,
            activation_hresult=request.hresult,
        )

    try:
        wait_result = wait_for_trusted_application_activation(
            computer.observe_local, candidate,
            initial_observation=initial_observation,
            launch_succeeded=True, timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
            clock=clock, sleep_fn=sleep_fn, target_probe=target_probe,
            target_activator=lambda: computer.activate_trusted_application_window(candidate),
            post_deadline_probe_seconds=0,
            open_app_action_started_at=request_started_at,
            open_app_action_elapsed_ms=request.request_elapsed_ms,
        )
    except KeyboardInterrupt:
        raise
    except Exception:
        return _failure(
            mechanism, stage="monitor", reason="activation_observation_failed",
            app_id=candidate.id, launch_kind="packaged",
        )

    lifecycle = wait_result.lifecycle
    explicit = lifecycle.explicit_activation if lifecycle is not None else None
    timing = lifecycle.timing if lifecycle is not None else None
    primary_unique = bool(
        lifecycle is not None
        and lifecycle.primary_surface_resolution == "unique"
        and lifecycle.primary_surface_candidate_count == 1
    )
    primary_visible = (
        lifecycle.target_window_state.visible
        if lifecycle is not None and lifecycle.target_window_state is not None else None
    )
    primary_stable = bool(
        timing is not None and timing.first_stable_eligible_window_since_launch_ms is not None
    )
    explicit_activation_attempted = bool(explicit and explicit.attempted)
    explicit_set_foreground_attempted = bool(
        explicit_activation_attempted
        and explicit is not None
        and (
            explicit.mechanism in {"activate", "restore_then_activate"}
            or (
                explicit.fallback_attempt is not None
                and explicit.fallback_attempt.set_foreground_attempted
            )
        )
    )
    automatic_foreground = bool(
        lifecycle is not None
        and lifecycle.target_foreground_observed
        and not before_request.foreground_observed
        and not explicit_activation_attempted
    )
    automatic_elapsed = (
        timing.first_foreground_since_launch_ms
        if automatic_foreground and timing is not None else None
    )
    final_foreground = bool(
        wait_result.observation is not None
        and wait_result.observation.application_id == candidate.id
    )
    explicit_success = (
        explicit.success if explicit is not None and explicit_activation_attempted else None
    )
    simple_set_foreground_return = (
        explicit.simple_attempt.set_foreground_return
        if explicit is not None and explicit.simple_attempt is not None else None
    )
    fallback_set_foreground_attempted = bool(
        explicit is not None and explicit.fallback_attempt is not None
        and explicit.fallback_attempt.set_foreground_attempted
    )
    fallback_set_foreground_return = (
        explicit.fallback_attempt.set_foreground_return
        if explicit is not None and explicit.fallback_attempt is not None else None
    )
    explicit_set_foreground_return: bool | None = None
    if explicit is not None and explicit_set_foreground_attempted:
        fallback = explicit.fallback_attempt
        if fallback is not None and fallback.set_foreground_attempted:
            explicit_set_foreground_return = fallback.set_foreground_return
        elif explicit.simple_attempt is not None:
            explicit_set_foreground_return = explicit.simple_attempt.set_foreground_return
        else:
            explicit_set_foreground_return = explicit.os_call_reported_success
    if automatic_foreground:
        foreground_required: bool | None = False
    elif explicit_set_foreground_attempted:
        foreground_required = True
    else:
        foreground_required = None

    if final_foreground:
        failure_stage = failure_reason = None
    elif explicit_activation_attempted:
        failure_stage = "foreground_activation"
        failure_reason = (
            explicit.failure_reason if explicit and explicit.failure_reason
            else "foreground_not_verified"
        )
    elif not primary_unique:
        failure_stage = "target_window_resolution"
        failure_reason = (
            lifecycle.primary_surface_resolution if lifecycle is not None
            else "target_window_not_observed"
        )
    else:
        failure_stage = "foreground_observation"
        failure_reason = wait_result.reason

    return PackagedActivationDiagnostic(
        success=final_foreground, mechanism=mechanism,
        trusted_app_id=candidate.id, launch_kind="packaged",
        activation_request_succeeded=True,
        activation_call_elapsed_ms=request.request_elapsed_ms,
        returned_pid_present=request.returned_pid_present,
        activation_hresult=request.hresult,
        target_process_observed_before_request=before_request.process_observed,
        target_window_observed_before_request=before_request.window_observed,
        target_foreground_observed_before_request=before_request.foreground_observed,
        target_process_observed=bool(lifecycle and lifecycle.target_process_observed),
        target_window_observed=bool(lifecycle and lifecycle.target_window_observed),
        primary_surface_unique=primary_unique,
        primary_window_visible=primary_visible,
        primary_window_stable=primary_stable,
        automatic_foreground_observed=automatic_foreground,
        automatic_foreground_elapsed_ms=automatic_elapsed,
        post_launch_setforeground_required=foreground_required,
        explicit_activation_attempted=explicit_activation_attempted,
        explicit_foreground_attempted=explicit_set_foreground_attempted,
        explicit_simple_setforeground_return=simple_set_foreground_return,
        explicit_fallback_setforeground_attempted=fallback_set_foreground_attempted,
        explicit_fallback_setforeground_return=fallback_set_foreground_return,
        explicit_set_foreground_returned_successfully=explicit_set_foreground_return,
        explicit_foreground_succeeded=explicit_success,
        final_foreground_verified=final_foreground,
        failure_stage=failure_stage, failure_reason=failure_reason,
        timing=timing,
    )


__all__ = ["PackagedActivationDiagnostic", "run_packaged_activation_diagnostic"]
