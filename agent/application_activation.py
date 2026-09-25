"""Shared bounded polling for trusted foreground application activation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import re
import time

from computer.applications import (
    ActivationWindowCandidateDiagnostics, ApplicationCandidate,
    EligibleWindowDiagnostics, EligibleWindowFacts, EligibleWindowRelationship,
    SimpleActivationAttempt, ThreadInputFallbackAttempt,
    TrustedApplicationRuntimeState, TrustedWindowActivationResult,
    WindowResolutionStatus,
)
from computer.models import Observation
from decision.context import Redactor


@dataclass(frozen=True, slots=True)
class ForegroundIdentity:
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
class ActivationTargetWindowState:
    visible: bool | None
    minimized: bool | None
    foreground: bool | None
    trusted_identity_match: bool


@dataclass(frozen=True, slots=True)
class PostDeadlineProbeDiagnostics:
    performed: bool = False
    delay_ms: int = 0
    target_process_observed: bool = False
    target_window_observed: bool = False
    target_foreground_observed: bool = False


@dataclass(frozen=True, slots=True)
class ActivationLifecycleDiagnostics:
    launch_request_accepted: bool
    target_probe_available: bool
    target_probe_complete: bool
    target_process_observed: bool
    target_window_observed: bool
    target_foreground_observed: bool
    first_target_process_elapsed_ms: int | None
    first_target_window_elapsed_ms: int | None
    first_target_foreground_elapsed_ms: int | None
    target_window_state: ActivationTargetWindowState | None
    matching_window_candidates: tuple[ActivationWindowCandidateDiagnostics, ...]
    candidate_diagnostics_truncated: bool
    post_deadline_probe: PostDeadlineProbeDiagnostics
    explicit_activation: ExplicitActivationDiagnostics | None = None
    window_resolution_status: WindowResolutionStatus = "incomplete"
    matching_trusted_window_count: int = 0
    eligible_window_count: int = 0
    enumeration_complete: bool = False
    rejection_reason_counts: tuple[tuple[str, int], ...] = ()
    eligible_window_diagnostics: tuple[EligibleWindowDiagnostics, ...] = ()
    eligible_window_diagnostics_truncated: bool = False
    eligible_window_relationships: tuple[EligibleWindowRelationship, ...] = ()
    eligible_window_relationships_truncated: bool = False
    base_eligible_window_count: int = 0
    primary_surface_candidate_count: int = 0
    tool_surface_candidate_count: int = 0
    primary_surface_resolution: WindowResolutionStatus = "incomplete"


@dataclass(frozen=True, slots=True)
class ExplicitActivationDiagnostics:
    eligible: bool
    eligibility_reason: str
    attempted: bool
    budget_consumed: bool
    window_state_before: ActivationTargetWindowState | None
    mechanism: str | None
    os_call_reported_success: bool | None
    fresh_verification_obtained: bool
    foreground_after: bool | None
    trusted_identity_after: bool | None
    success: bool
    failure_reason: str | None
    selected_window_verified: bool | None = None
    activation_strategy: str | None = None
    simple_attempt: SimpleActivationAttempt | None = None
    fallback_attempt: ThreadInputFallbackAttempt | None = None


@dataclass(frozen=True, slots=True)
class ApplicationActivationWait:
    observation: Observation | None
    diagnostics: ApplicationActivationDiagnostics
    reason: str
    lifecycle: ActivationLifecycleDiagnostics | None = None


def _bounded_app_id(value: str) -> str | None:
    return value[:80] if value and re.fullmatch(r"[A-Za-z0-9_-]{1,80}", value) else None


def _foreground_identity(observation: Observation | None) -> ForegroundIdentity | None:
    if observation is None:
        return None
    return ForegroundIdentity(
        trusted_app_id=_bounded_app_id(observation.application_id),
    )


def _foreground_signature(observation: Observation | None) -> tuple[object, ...] | None:
    identity = _foreground_identity(observation)
    if identity is None:
        return None
    hwnd = observation.foreground_hwnd if observation is not None else None
    if hwnd is None and observation is not None and observation.screenshot is not None:
        hwnd = observation.screenshot.window_handle
    process_id = observation.process_id if observation is not None else None
    return (
        hwnd if isinstance(hwnd, int) and hwnd > 0 else None,
        process_id if isinstance(process_id, int) and process_id > 0 else None,
        identity.trusted_app_id,
    )


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


def _sanitize_window_candidates(
    values: tuple[ActivationWindowCandidateDiagnostics, ...],
) -> tuple[ActivationWindowCandidateDiagnostics, ...]:
    """Copy only bounded, typed, coarse structural diagnostics to the generic layer."""
    area_buckets = {"zero", "small", "medium", "large", "unavailable"}
    z_order_buckets = {"front", "middle", "back", "unavailable"}
    root_relationships = {"self", "other", "unavailable"}
    rejection_reasons = {
        "trusted_identity_mismatch", "window_missing", "target_not_visible",
        "cloaking_unknown", "target_cloaked", "client_area_unknown", "zero_client_area",
    }
    safe: list[ActivationWindowCandidateDiagnostics] = []
    for value in values[:8]:
        if not isinstance(value, ActivationWindowCandidateDiagnostics):
            continue
        if type(value.candidate_index) is not int or not 1 <= value.candidate_index <= 8:
            continue
        if value.trusted_identity_match is not True:
            continue
        safe.append(ActivationWindowCandidateDiagnostics(
            candidate_index=value.candidate_index,
            trusted_identity_match=True,
            visible=value.visible if type(value.visible) is bool else None,
            minimized=value.minimized if type(value.minimized) is bool else None,
            enabled=value.enabled if type(value.enabled) is bool else None,
            cloaked_if_available=(
                value.cloaked_if_available
                if type(value.cloaked_if_available) is bool else None
            ),
            owner_present=value.owner_present if type(value.owner_present) is bool else None,
            root_owner_relationship=(
                value.root_owner_relationship
                if value.root_owner_relationship in root_relationships else "unavailable"
            ),
            tool_window=value.tool_window if type(value.tool_window) is bool else None,
            app_window=value.app_window if type(value.app_window) is bool else None,
            has_nonzero_client_area=(
                value.has_nonzero_client_area
                if type(value.has_nonzero_client_area) is bool else None
            ),
            client_area_bucket=(
                value.client_area_bucket if value.client_area_bucket in area_buckets
                else "unavailable"
            ),
            window_area_bucket=(
                value.window_area_bucket if value.window_area_bucket in area_buckets
                else "unavailable"
            ),
            foreground=value.foreground if type(value.foreground) is bool else False,
            stable_across_probes=(
                value.stable_across_probes
                if type(value.stable_across_probes) is bool else False
            ),
            z_order_bucket=(
                value.z_order_bucket if value.z_order_bucket in z_order_buckets
                else "unavailable"
            ),
            activation_candidate=(
                value.activation_candidate
                if type(value.activation_candidate) is bool else False
            ),
            explicit_activation_eligible=(
                value.explicit_activation_eligible
                if type(value.explicit_activation_eligible) is bool else False
            ),
            rejection_reason=(
                value.rejection_reason if value.rejection_reason in rejection_reasons else None
            ),
        ))
    return tuple(safe)


def _sanitize_eligible_facts(value: object) -> EligibleWindowFacts | None:
    if not isinstance(value, EligibleWindowFacts):
        return None
    if (value.explicit_activation_eligible is not True
            or value.eligibility_reason != "eligible"
            or value.rejection_reason is not None
            or value.visible is not True
            or value.cloaked_state != "uncloaked"
            or value.has_nonzero_client_area is not True):
        return None
    area_buckets = {"zero", "small", "medium", "large", "unavailable"}
    z_order_buckets = {"front", "middle", "back", "unavailable"}
    root_relationships = {"self", "other", "unavailable"}
    return EligibleWindowFacts(
        visible=True,
        minimized=value.minimized if type(value.minimized) is bool else None,
        enabled=value.enabled if type(value.enabled) is bool else None,
        cloaked_state="uncloaked",
        owner_present=value.owner_present if type(value.owner_present) is bool else None,
        root_owner_relationship=(
            value.root_owner_relationship
            if value.root_owner_relationship in root_relationships else "unavailable"
        ),
        tool_window=value.tool_window if type(value.tool_window) is bool else None,
        app_window=value.app_window if type(value.app_window) is bool else None,
        has_nonzero_client_area=True,
        client_area_bucket=(
            value.client_area_bucket
            if value.client_area_bucket in area_buckets else "unavailable"
        ),
        window_area_bucket=(
            value.window_area_bucket
            if value.window_area_bucket in area_buckets else "unavailable"
        ),
        foreground=value.foreground if type(value.foreground) is bool else False,
        stable_across_probes=(
            value.stable_across_probes
            if type(value.stable_across_probes) is bool else False
        ),
        explicit_activation_eligible=True,
        eligibility_reason="eligible",
        rejection_reason=None,
        z_order_bucket=(
            value.z_order_bucket
            if value.z_order_bucket in z_order_buckets else "unavailable"
        ),
    )


def _valid_window_diagnostic_id(value: object) -> bool:
    return isinstance(value, str) and len(value) <= 11 and re.fullmatch(r"ew[1-9]\d{0,8}", value) is not None


def _sanitize_eligible_window_diagnostics(
    values: tuple[EligibleWindowDiagnostics, ...],
) -> tuple[EligibleWindowDiagnostics, ...]:
    safe: list[EligibleWindowDiagnostics] = []
    if not isinstance(values, tuple):
        return ()
    for value in values[:8]:
        if not isinstance(value, EligibleWindowDiagnostics):
            continue
        if (not _valid_window_diagnostic_id(value.diagnostic_id)
                or type(value.eligible_candidate_index) is not int
                or not 1 <= value.eligible_candidate_index <= 512):
            continue
        facts = _sanitize_eligible_facts(value.facts)
        if facts is None:
            continue
        initial_facts = _sanitize_eligible_facts(value.initial_probe_facts)
        repeated_facts = _sanitize_eligible_facts(value.latest_repeated_probe_facts)
        post_facts = _sanitize_eligible_facts(value.latest_post_deadline_probe_facts)
        safe.append(EligibleWindowDiagnostics(
            diagnostic_id=value.diagnostic_id,
            eligible_candidate_index=value.eligible_candidate_index,
            facts=facts,
            present_in_initial_probe=(
                value.present_in_initial_probe is True and initial_facts is not None
            ),
            present_in_repeated_probe=(
                value.present_in_repeated_probe is True and repeated_facts is not None
            ),
            present_in_post_deadline_probe=(
                value.present_in_post_deadline_probe is True and post_facts is not None
            ),
            observed_probe_count=(
                value.observed_probe_count
                if type(value.observed_probe_count) is int
                and 0 <= value.observed_probe_count <= 512 else 0
            ),
            initial_probe_facts=initial_facts,
            latest_repeated_probe_facts=repeated_facts,
            latest_post_deadline_probe_facts=post_facts,
            surface_class=(
                value.surface_class
                if value.surface_class in {"primary", "tool", "unknown"}
                and (
                    value.surface_class == "unknown" or
                    value.surface_class == ("tool" if facts.tool_window is True else "primary")
                ) and (facts.tool_window is not None or value.surface_class == "unknown")
                else "unknown"
            ),
        ))
    return tuple(safe)


def _sanitize_eligible_window_relationships(
    values: tuple[EligibleWindowRelationship, ...], valid_ids: set[str],
) -> tuple[EligibleWindowRelationship, ...]:
    if not isinstance(values, tuple):
        return ()
    allowed = {"independent", "one_owned_by_other", "same_root_owner", "unknown"}
    safe: list[EligibleWindowRelationship] = []
    for value in values[:28]:
        if not isinstance(value, EligibleWindowRelationship):
            continue
        if (value.first_diagnostic_id not in valid_ids
                or value.second_diagnostic_id not in valid_ids
                or value.first_diagnostic_id == value.second_diagnostic_id
                or value.relationship not in allowed):
            continue
        safe.append(EligibleWindowRelationship(
            first_diagnostic_id=value.first_diagnostic_id,
            second_diagnostic_id=value.second_diagnostic_id,
            relationship=value.relationship,
            present_in_initial_probe=value.present_in_initial_probe is True,
            present_in_repeated_probe=value.present_in_repeated_probe is True,
            present_in_post_deadline_probe=value.present_in_post_deadline_probe is True,
            observed_probe_count=(
                value.observed_probe_count
                if type(value.observed_probe_count) is int
                and 0 <= value.observed_probe_count <= 512 else 0
            ),
            initial_probe_relationship=(
                value.initial_probe_relationship
                if value.initial_probe_relationship in allowed else None
            ),
            latest_repeated_probe_relationship=(
                value.latest_repeated_probe_relationship
                if value.latest_repeated_probe_relationship in allowed else None
            ),
            latest_post_deadline_probe_relationship=(
                value.latest_post_deadline_probe_relationship
                if value.latest_post_deadline_probe_relationship in allowed else None
            ),
        ))
    return tuple(safe)


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
    target_probe: Callable[[], TrustedApplicationRuntimeState] | None = None,
    target_activator: Callable[[], TrustedWindowActivationResult] | None = None,
    post_deadline_probe_seconds: float = 0,
) -> ApplicationActivationWait:
    """Poll passively, with one optional verified-window activation inside the deadline."""
    if not 0 <= timeout_seconds <= 15:
        raise ValueError("Application activation timeout must be between zero and 15 seconds.")
    if not 0.01 <= poll_interval_seconds <= 1:
        raise ValueError("Application activation polling interval must be between 10 ms and one second.")
    if not 1 <= max_recorded_transitions <= 8:
        raise ValueError("At most eight foreground transitions may be recorded.")
    if not 0 <= post_deadline_probe_seconds <= 3:
        raise ValueError("Post-deadline diagnostic probe must be between zero and three seconds.")

    started = clock()
    deadline = started + timeout_seconds
    latest: Observation | None = None
    attempts = 0
    transitions: list[ForegroundIdentity] = []
    previous_signature = _foreground_signature(initial_observation)
    reason = "activation_timeout"
    target_process_observed = False
    target_window_observed = False
    target_foreground_observed = False
    first_target_process_elapsed_ms: int | None = None
    first_target_window_elapsed_ms: int | None = None
    first_target_foreground_elapsed_ms: int | None = None
    target_window_state: ActivationTargetWindowState | None = None
    matching_window_candidates: tuple[ActivationWindowCandidateDiagnostics, ...] = ()
    candidate_diagnostics_truncated = False
    window_resolution_status: WindowResolutionStatus = "incomplete"
    matching_trusted_window_count = 0
    eligible_window_count = 0
    primary_surface_candidate_count = 0
    tool_surface_candidate_count = 0
    primary_surface_resolution: WindowResolutionStatus = "incomplete"
    enumeration_complete = False
    rejection_reason_counts: tuple[tuple[str, int], ...] = ()
    eligible_window_records: dict[str, EligibleWindowDiagnostics] = {}
    eligible_relationship_records: dict[tuple[str, str], EligibleWindowRelationship] = {}
    eligible_window_diagnostics_truncated = False
    eligible_window_relationships_truncated = False
    successful_target_probe_count = 0
    target_probe_available = False
    target_probe_complete = True
    last_target_state: TrustedApplicationRuntimeState | None = None
    explicit_attempted = False
    explicit_budget_consumed = False
    explicit_diagnostics: ExplicitActivationDiagnostics | None = None

    def read_target_state() -> TrustedApplicationRuntimeState | None:
        nonlocal target_probe_available, target_probe_complete, last_target_state
        if target_probe is None:
            return None
        try:
            state = target_probe()
        except KeyboardInterrupt:
            raise
        except Exception:
            target_probe_complete = False
            return None
        if not isinstance(state, TrustedApplicationRuntimeState):
            target_probe_complete = False
            return None
        target_probe_available = True
        target_probe_complete = target_probe_complete and state.probe_complete
        last_target_state = state
        return state

    def elapsed_ms() -> int:
        return max(0, round((clock() - started) * 1000))

    def record_eligible_window_diagnostics(
        state: TrustedApplicationRuntimeState | None, phase: str,
    ) -> None:
        nonlocal eligible_window_diagnostics_truncated
        nonlocal eligible_window_relationships_truncated
        if state is None:
            return
        raw = state.eligible_window_diagnostics
        eligible_window_diagnostics_truncated |= (
            state.eligible_window_diagnostics_truncated is True
            or (isinstance(raw, tuple) and len(raw) > 8)
        )
        sanitized = _sanitize_eligible_window_diagnostics(raw)
        raw_ids = {item.diagnostic_id for item in sanitized}
        for item in sanitized:
            existing = eligible_window_records.get(item.diagnostic_id)
            if existing is None and len(eligible_window_records) >= 8:
                eligible_window_diagnostics_truncated = True
                continue
            initial_facts = existing.initial_probe_facts if existing is not None else None
            repeated_facts = (
                existing.latest_repeated_probe_facts if existing is not None else None
            )
            post_facts = (
                existing.latest_post_deadline_probe_facts if existing is not None else None
            )
            if phase == "initial" and initial_facts is None:
                initial_facts = item.facts
            elif phase == "repeated":
                repeated_facts = item.facts
            elif phase == "post_deadline":
                post_facts = item.facts
            previous_count = existing.observed_probe_count if existing is not None else 0
            eligible_window_records[item.diagnostic_id] = EligibleWindowDiagnostics(
                diagnostic_id=item.diagnostic_id,
                eligible_candidate_index=item.eligible_candidate_index,
                facts=item.facts,
                present_in_initial_probe=initial_facts is not None,
                present_in_repeated_probe=repeated_facts is not None,
                present_in_post_deadline_probe=post_facts is not None,
                observed_probe_count=min(512, previous_count + 1),
                initial_probe_facts=initial_facts,
                latest_repeated_probe_facts=repeated_facts,
                latest_post_deadline_probe_facts=post_facts,
                surface_class=item.surface_class,
            )

        relationships = _sanitize_eligible_window_relationships(
            state.eligible_window_relationships, raw_ids,
        )
        for relation in relationships:
            ids = tuple(sorted((relation.first_diagnostic_id, relation.second_diagnostic_id)))
            if ids not in eligible_relationship_records and len(eligible_relationship_records) >= 28:
                eligible_window_relationships_truncated = True
                continue
            previous = eligible_relationship_records.get(ids)
            initial_relationship = (
                previous.initial_probe_relationship if previous is not None else None
            )
            repeated_relationship = (
                previous.latest_repeated_probe_relationship if previous is not None else None
            )
            post_relationship = (
                previous.latest_post_deadline_probe_relationship if previous is not None else None
            )
            if phase == "initial" and initial_relationship is None:
                initial_relationship = relation.relationship
            elif phase == "repeated":
                repeated_relationship = relation.relationship
            elif phase == "post_deadline":
                post_relationship = relation.relationship
            eligible_relationship_records[ids] = EligibleWindowRelationship(
                first_diagnostic_id=ids[0],
                second_diagnostic_id=ids[1],
                relationship=relation.relationship,
                present_in_initial_probe=(
                    (previous.present_in_initial_probe if previous else False)
                    or phase == "initial"
                ),
                present_in_repeated_probe=(
                    (previous.present_in_repeated_probe if previous else False)
                    or phase == "repeated"
                ),
                present_in_post_deadline_probe=(
                    (previous.present_in_post_deadline_probe if previous else False)
                    or phase == "post_deadline"
                ),
                observed_probe_count=min(
                    512, (previous.observed_probe_count if previous else 0) + 1,
                ),
                initial_probe_relationship=initial_relationship,
                latest_repeated_probe_relationship=repeated_relationship,
                latest_post_deadline_probe_relationship=post_relationship,
            )

    def note_activation_state(state: TrustedApplicationRuntimeState | None) -> None:
        nonlocal successful_target_probe_count
        nonlocal target_process_observed, target_window_observed
        nonlocal target_foreground_observed, first_target_process_elapsed_ms
        nonlocal first_target_window_elapsed_ms, first_target_foreground_elapsed_ms
        nonlocal target_window_state
        nonlocal matching_window_candidates, candidate_diagnostics_truncated
        nonlocal window_resolution_status, matching_trusted_window_count
        nonlocal eligible_window_count, enumeration_complete, rejection_reason_counts
        nonlocal primary_surface_candidate_count, tool_surface_candidate_count
        nonlocal primary_surface_resolution
        elapsed = elapsed_ms()
        if state is not None:
            phase = "initial" if successful_target_probe_count == 0 else "repeated"
            record_eligible_window_diagnostics(state, phase)
            successful_target_probe_count += 1
            matching_window_candidates = _sanitize_window_candidates(state.window_candidates)
            candidate_diagnostics_truncated = (
                state.candidate_diagnostics_truncated is True
            )
            statuses = {"unique", "none", "ambiguous", "incomplete"}
            if state.window_resolution_status in statuses:
                window_resolution_status = state.window_resolution_status  # type: ignore[assignment]
            matching_trusted_window_count = (
                state.matching_trusted_window_count
                if type(state.matching_trusted_window_count) is int
                and 0 <= state.matching_trusted_window_count <= 512 else 0
            )
            eligible_window_count = (
                state.eligible_window_count
                if type(state.eligible_window_count) is int
                and 0 <= state.eligible_window_count <= 512 else 0
            )
            primary_surface_candidate_count = (
                state.primary_surface_candidate_count
                if type(state.primary_surface_candidate_count) is int
                and 0 <= state.primary_surface_candidate_count <= 512 else 0
            )
            tool_surface_candidate_count = (
                state.tool_surface_candidate_count
                if type(state.tool_surface_candidate_count) is int
                and 0 <= state.tool_surface_candidate_count <= 512 else 0
            )
            if state.primary_surface_resolution_status in statuses:
                primary_surface_resolution = state.primary_surface_resolution_status  # type: ignore[assignment]
            enumeration_complete = state.enumeration_complete is True
            allowed_reasons = {
                "trusted_identity_mismatch", "window_missing", "target_not_visible",
                "cloaking_unknown", "target_cloaked", "client_area_unknown", "zero_client_area",
            }
            if isinstance(state.rejection_reason_counts, tuple):
                counts: dict[str, int] = {}
                for reason, count in state.rejection_reason_counts[:7]:
                    if reason in allowed_reasons and type(count) is int and 0 < count <= 512:
                        counts[reason] = min(512, counts.get(reason, 0) + count)
                rejection_reason_counts = tuple(sorted(counts.items()))

        if state is not None and state.trusted_identity_match:
            process_seen = state.process_observed
            window_seen = state.window_observed
            foreground_seen = state.foreground_observed
            if process_seen and first_target_process_elapsed_ms is None:
                first_target_process_elapsed_ms = elapsed
            if window_seen and first_target_window_elapsed_ms is None:
                first_target_window_elapsed_ms = elapsed
            if foreground_seen and first_target_foreground_elapsed_ms is None:
                first_target_foreground_elapsed_ms = elapsed
            target_process_observed |= process_seen
            target_window_observed |= window_seen
            target_foreground_observed |= foreground_seen
            target_window_state = (
                ActivationTargetWindowState(
                    state.visible, state.minimized, state.window_foreground, True,
                )
                if window_seen and not state.window_ambiguous
                and (state.window_resolution_status in {None, "unique"}) else None
            )

    def target_eligibility(
        state: TrustedApplicationRuntimeState | None,
    ) -> tuple[bool, str, ActivationTargetWindowState | None]:
        if state is None:
            return False, "probe_unavailable", None
        if (target_activator is not None
                and state.primary_surface_resolution_status is not None):
            status = state.primary_surface_resolution_status
            if (status not in {"unique", "none", "ambiguous", "incomplete"}
                    or state.enumeration_complete is not True or not state.probe_complete
                    or status == "incomplete"):
                return False, "probe_incomplete", None
            if status == "none":
                return False, "no_eligible_window", None
            if status == "ambiguous":
                return False, "target_window_ambiguous", None
            facts = _sanitize_eligible_facts(state.primary_surface_facts)
            if (type(state.primary_surface_candidate_count) is not int
                    or state.primary_surface_candidate_count != 1 or facts is None
                    or facts.tool_window is not False):
                return False, "probe_incomplete", None
            window_state = ActivationTargetWindowState(
                facts.visible, facts.minimized, facts.foreground,
                state.trusted_identity_match,
            )
            if not state.trusted_identity_match:
                return False, "trusted_identity_mismatch", window_state
            if facts.visible is not True:
                return False, "target_not_visible", window_state
            if facts.cloaked_state != "uncloaked":
                return False, (
                    "target_cloaked" if facts.cloaked_state == "cloaked"
                    else "cloaking_unknown"
                ), window_state
            if facts.has_nonzero_client_area is not True:
                return False, (
                    "zero_client_area" if facts.has_nonzero_client_area is False
                    else "client_area_unknown"
                ), window_state
            if facts.foreground:
                return False, "already_foreground", window_state
            if not facts.stable_across_probes:
                return False, "target_window_unstable", window_state
            return True, "eligible", window_state
        if state.window_resolution_status is not None:
            status = state.window_resolution_status
            if (status not in {"unique", "none", "ambiguous", "incomplete"}
                    or state.enumeration_complete is not True or not state.probe_complete
                    or status == "incomplete"):
                return False, "probe_incomplete", None
            if status == "none":
                reason = (
                    "target_window_missing"
                    if state.matching_trusted_window_count == 0 else "no_eligible_window"
                )
                return False, reason, None
            if status == "ambiguous":
                return False, "target_window_ambiguous", None
            if state.eligible_window_count != 1:
                return False, "probe_incomplete", None
        window_state = (
            ActivationTargetWindowState(
                state.visible, state.minimized, state.window_foreground,
                state.trusted_identity_match,
            )
            if state.window_observed and not state.window_ambiguous else None
        )
        if not state.probe_complete:
            return False, "probe_incomplete", window_state
        if state.window_ambiguous:
            return False, "target_window_ambiguous", None
        if not state.trusted_identity_match:
            return False, "trusted_identity_mismatch", window_state
        if not state.window_observed:
            return False, "target_window_missing", window_state
        if state.foreground_observed or state.window_foreground is True:
            return False, "already_foreground", window_state
        if state.visible is not True:
            return False, "target_not_visible", window_state
        if not state.window_stable:
            return False, "target_window_unstable", window_state
        return True, "eligible", window_state

    def sanitized_explicit_diagnostics(
        *, eligible: bool, eligibility_reason: str, attempted: bool,
        budget_consumed: bool, window_state_before: ActivationTargetWindowState | None,
        mechanism: str | None, os_call_reported_success: bool | None,
        fresh_verification_obtained: bool, foreground_after: bool | None,
        trusted_identity_after: bool | None, success: bool, failure_reason: str | None,
        selected_window_verified: bool | None = None,
        activation_strategy: str | None = None,
        simple_attempt: SimpleActivationAttempt | None = None,
        fallback_attempt: ThreadInputFallbackAttempt | None = None,
    ) -> ExplicitActivationDiagnostics:
        allowed_reasons = {
            "eligible", "probe_unavailable", "probe_incomplete", "target_window_missing",
            "no_eligible_window", "target_window_ambiguous", "target_window_unstable",
            "target_not_visible", "cloaking_unknown", "target_cloaked",
            "client_area_unknown", "zero_client_area", "already_foreground",
            "trusted_identity_mismatch", "stale_window",
            "window_state_changed", "system_surface", "restore_not_started",
            "os_activation_rejected", "activation_error", "deadline_expired",
            "passive_activation_observed", "foreground_not_selected",
            "foreground_verification_unavailable", "selected_window_changed",
            "foreground_window_missing", "foreground_window_invalid",
            "selected_thread_unresolved", "foreground_thread_unresolved",
            "current_thread_unresolved", "foreground_thread_is_current",
            "attach_failed", "set_foreground_failed", "detach_failed",
            "selected_target_changed", "selected_target_revalidation_failed",
            "post_activation_verification_failed", "simple_restore_failed",
        }
        allowed_mechanisms = {
            "activate", "restore_then_activate", "restore",
        }
        allowed_failures = {
            "activation_error", "activation_budget_consumed", "restore_not_started",
            "os_activation_rejected", "fresh_observation_failed",
            "trusted_foreground_not_observed", "activation_not_eligible",
            "foreground_not_selected", "foreground_verification_unavailable",
            "target_window_missing", "trusted_identity_mismatch", "target_not_visible",
            "stale_window", "selected_window_changed", "foreground_window_missing",
            "foreground_window_invalid", "selected_thread_unresolved",
            "foreground_thread_unresolved", "current_thread_unresolved",
            "foreground_thread_is_current", "attach_failed", "set_foreground_failed",
            "detach_failed", "selected_target_changed",
            "selected_target_revalidation_failed", "post_activation_verification_failed",
            "simple_restore_failed",
            "trusted_identity_mismatch", "target_window_missing", "target_not_visible",
        }
        fallback_reasons = {
            "simple_verified", "simple_foreground_unverified", "selected_window_changed",
            "selected_thread_unresolved", "foreground_window_missing",
            "foreground_window_invalid", "foreground_thread_unresolved",
            "current_thread_unresolved", "foreground_thread_is_current",
            "attach_failed", "set_foreground_failed", "detach_failed",
            "selected_target_changed", "selected_target_revalidation_failed",
            "post_activation_verification_failed", "selected_primary_surface_changed",
            "target_not_visible", "target_cloaked", "cloaking_unknown",
            "client_area_unknown", "zero_client_area", "simple_restore_failed",
            "foreground_verification_unavailable",
            "fallback_attempted", "verified",
            "selected_window_changed", "trusted_identity_mismatch", "target_window_missing",
        }
        fallback_failures = allowed_failures | {
            "attach_failed", "set_foreground_failed", "detach_failed",
            "selected_target_changed", "selected_target_revalidation_failed",
            "post_activation_verification_failed", "simple_restore_failed",
        }
        safe_simple = (
            SimpleActivationAttempt(
                set_foreground_return=(
                    simple_attempt.set_foreground_return
                    if type(simple_attempt.set_foreground_return) is bool else None
                ),
                verified=(simple_attempt.verified
                          if type(simple_attempt.verified) is bool else None),
            ) if isinstance(simple_attempt, SimpleActivationAttempt) else None
        )
        safe_fallback = None
        if isinstance(fallback_attempt, ThreadInputFallbackAttempt):
            safe_fallback = ThreadInputFallbackAttempt(
                eligible=fallback_attempt.eligible is True,
                reason=(fallback_attempt.reason
                        if fallback_attempt.reason in fallback_reasons
                        else "post_activation_verification_failed"),
                foreground_hwnd_present=(
                    fallback_attempt.foreground_hwnd_present
                    if type(fallback_attempt.foreground_hwnd_present) is bool else None
                ),
                selected_thread_resolved=(
                    fallback_attempt.selected_thread_resolved
                    if type(fallback_attempt.selected_thread_resolved) is bool else None
                ),
                foreground_thread_resolved=(
                    fallback_attempt.foreground_thread_resolved
                    if type(fallback_attempt.foreground_thread_resolved) is bool else None
                ),
                current_thread_resolved=(
                    fallback_attempt.current_thread_resolved
                    if type(fallback_attempt.current_thread_resolved) is bool else None
                ),
                incidental_foreground_changed_before_activation=(
                    fallback_attempt.incidental_foreground_changed_before_activation
                    if type(fallback_attempt.incidental_foreground_changed_before_activation)
                    is bool else None
                ),
                selected_target_still_valid_before_activation=(
                    fallback_attempt.selected_target_still_valid_before_activation
                    if type(fallback_attempt.selected_target_still_valid_before_activation)
                    is bool else None
                ),
                attach_attempted=fallback_attempt.attach_attempted is True,
                attach_succeeded=(
                    fallback_attempt.attach_succeeded
                    if type(fallback_attempt.attach_succeeded) is bool else None
                ),
                set_foreground_attempted=fallback_attempt.set_foreground_attempted is True,
                set_foreground_return=(
                    fallback_attempt.set_foreground_return
                    if type(fallback_attempt.set_foreground_return) is bool else None
                ),
                bring_to_top_used=fallback_attempt.bring_to_top_used is True,
                set_active_used=fallback_attempt.set_active_used is True,
                detach_succeeded=(
                    fallback_attempt.detach_succeeded
                    if type(fallback_attempt.detach_succeeded) is bool else None
                ),
                verified=(fallback_attempt.verified
                          if type(fallback_attempt.verified) is bool else None),
                failure_reason=(
                    fallback_attempt.failure_reason
                    if fallback_attempt.failure_reason in fallback_failures else None
                ),
            )
        return ExplicitActivationDiagnostics(
            eligible=eligible,
            eligibility_reason=(eligibility_reason if eligibility_reason in allowed_reasons
                                else "activation_error"),
            attempted=attempted,
            budget_consumed=budget_consumed,
            window_state_before=window_state_before,
            mechanism=mechanism if mechanism in allowed_mechanisms else None,
            os_call_reported_success=os_call_reported_success,
            fresh_verification_obtained=fresh_verification_obtained,
            foreground_after=foreground_after,
            trusted_identity_after=trusted_identity_after,
            success=success,
            failure_reason=failure_reason if failure_reason in allowed_failures else None,
            selected_window_verified=selected_window_verified,
            activation_strategy=(
                activation_strategy
                if activation_strategy in {"simple", "attach_thread_input"} else None
            ),
            simple_attempt=safe_simple,
            fallback_attempt=safe_fallback,
        )

    def note_post_deadline_state(
        state: TrustedApplicationRuntimeState | None,
        seen_process: bool, seen_window: bool, seen_foreground: bool,
    ) -> tuple[bool, bool, bool]:
        if state is None or not state.trusted_identity_match:
            return seen_process, seen_window, seen_foreground
        return (
            seen_process or state.process_observed,
            seen_window or state.window_observed,
            seen_foreground or state.foreground_observed,
        )

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
            target_state = read_target_state()
            note_activation_state(target_state)
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
                # The normal foreground observer uses the same trusted catalog
                # identity matcher and is authoritative even if a window probe
                # could not read visibility metadata.
                target_process_observed = True
                target_window_observed = True
                target_foreground_observed = True
                if first_target_process_elapsed_ms is None:
                    first_target_process_elapsed_ms = elapsed_ms()
                if first_target_window_elapsed_ms is None:
                    first_target_window_elapsed_ms = elapsed_ms()
                if first_target_foreground_elapsed_ms is None:
                    first_target_foreground_elapsed_ms = elapsed_ms()
                if target_window_state is None:
                    target_window_state = ActivationTargetWindowState(
                        None, None, True, True,
                    )
                break

            if target_activator is not None and not explicit_attempted:
                eligible, eligibility_reason, before_state = target_eligibility(target_state)
                if eligible:
                    if clock() >= deadline:
                        explicit_diagnostics = sanitized_explicit_diagnostics(
                            eligible=False, eligibility_reason="deadline_expired",
                            attempted=False, budget_consumed=False,
                            window_state_before=before_state, mechanism=None,
                            os_call_reported_success=None,
                            fresh_verification_obtained=False, foreground_after=None,
                            trusted_identity_after=None, success=False,
                            failure_reason=None,
                        )
                    else:
                        # Mark the one-shot attempt before entering the adapter so
                        # no later poll can retry. The Windows adapter consumes its
                        # OS-call budget only after its own resolve/revalidate phase.
                        explicit_attempted = True
                        activation_result: TrustedWindowActivationResult
                        try:
                            activation_result = target_activator()
                        except KeyboardInterrupt:
                            raise
                        except Exception:
                            activation_result = TrustedWindowActivationResult(
                                True, "eligible", failure_reason="activation_error",
                            )
                        if not isinstance(activation_result, TrustedWindowActivationResult):
                            activation_result = TrustedWindowActivationResult(
                                True, "eligible", failure_reason="activation_error",
                            )
                        explicit_budget_consumed = (
                            activation_result.budget_consumed is True
                            or (
                                activation_result.budget_consumed is None
                                and activation_result.mechanism in {
                                    "activate", "restore_then_activate", "restore",
                                }
                            )
                        )
                        activation_mechanism_attempted = activation_result.mechanism in {
                            "activate", "restore_then_activate", "restore",
                        }

                        fresh_observation: Observation | None = None
                        try:
                            fresh_observation = observe_local()
                        except KeyboardInterrupt:
                            raise
                        except Exception:
                            pass
                        if fresh_observation is not None:
                            attempts += 1
                            latest = fresh_observation
                            signature = _foreground_signature(latest)
                            if (signature is not None and signature != previous_signature
                                    and len(transitions) < max_recorded_transitions):
                                identity = _foreground_identity(latest)
                                if identity is not None:
                                    transitions.append(identity)
                            previous_signature = signature

                        trusted_after = (
                            latest.application_id == candidate.id
                            if fresh_observation is not None else None
                        )
                        foreground_after = trusted_after
                        fallback_cleanup_complete = not (
                            activation_result.fallback_attempt is not None
                            and activation_result.fallback_attempt.attach_succeeded is True
                        ) or (
                            activation_result.fallback_attempt is not None
                            and activation_result.fallback_attempt.detach_succeeded is True
                        )
                        verified_success = (
                            trusted_after is True
                            and activation_result.foreground_verified is True
                            and fallback_cleanup_complete
                        )
                        if verified_success:
                            reason = "activated"
                            target_process_observed = True
                            target_window_observed = True
                            target_foreground_observed = True
                            if first_target_process_elapsed_ms is None:
                                first_target_process_elapsed_ms = elapsed_ms()
                            if first_target_window_elapsed_ms is None:
                                first_target_window_elapsed_ms = elapsed_ms()
                            if first_target_foreground_elapsed_ms is None:
                                first_target_foreground_elapsed_ms = elapsed_ms()
                            target_window_state = ActivationTargetWindowState(
                                activation_result.visible, activation_result.minimized,
                                True, True,
                            )
                        failure = None
                        if not verified_success:
                            if fresh_observation is None:
                                failure = "fresh_observation_failed"
                            elif (activation_result.fallback_attempt is not None
                                  and activation_result.fallback_attempt.attach_succeeded is True
                                  and activation_result.fallback_attempt.detach_succeeded is not True):
                                failure = "detach_failed"
                            elif not activation_result.eligible:
                                failure = (
                                    activation_result.failure_reason
                                    or "activation_not_eligible"
                                )
                            elif (activation_mechanism_attempted
                                  and activation_result.foreground_verified is False):
                                failure = (
                                    activation_result.failure_reason
                                    or "foreground_not_selected"
                                )
                            elif (activation_mechanism_attempted
                                  and activation_result.foreground_verified is None):
                                failure = "foreground_verification_unavailable"
                            else:
                                failure = (
                                    activation_result.failure_reason
                                    or "trusted_foreground_not_observed"
                                )
                        explicit_diagnostics = sanitized_explicit_diagnostics(
                            eligible=activation_result.eligible,
                            eligibility_reason=activation_result.eligibility_reason,
                            attempted=activation_mechanism_attempted,
                            budget_consumed=explicit_budget_consumed,
                            window_state_before=(before_state if activation_result.visible is None
                                                 else ActivationTargetWindowState(
                                                     activation_result.visible,
                                                     activation_result.minimized,
                                                     False,
                                                     activation_result.trusted_identity_match,
                                                 )),
                            mechanism=activation_result.mechanism,
                            os_call_reported_success=activation_result.os_call_reported_success,
                            fresh_verification_obtained=fresh_observation is not None,
                            foreground_after=foreground_after,
                            trusted_identity_after=trusted_after,
                            success=verified_success,
                            failure_reason=failure,
                            selected_window_verified=activation_result.foreground_verified,
                            activation_strategy=activation_result.activation_strategy,
                            simple_attempt=activation_result.simple_attempt,
                            fallback_attempt=activation_result.fallback_attempt,
                        )
                        if verified_success:
                            break
                        if (not activation_result.eligible
                                and activation_result.eligibility_reason != "already_foreground"):
                            # The preflight scan no longer confirmed the bound
                            # primary HWND. Do not accept a replacement window later.
                            reason = "activation_timeout"
                            break
                        if activation_mechanism_attempted:
                            # The adapter checked the exact bound HWND after its
                            # one-shot OS attempt. Do not let a later same-app
                            # observation from another window override that result.
                            reason = "activation_timeout"
                            break
            remaining = deadline - clock()
            if remaining <= 0:
                break
            sleep_fn(min(poll_interval_seconds, remaining))

    activation_elapsed_ms = max(0, round((clock() - started) * 1000))
    post_deadline = PostDeadlineProbeDiagnostics()
    if (launch_succeeded and reason == "activation_timeout" and target_probe is not None
            and target_probe_available
            and post_deadline_probe_seconds > 0):
        probe_started = clock()
        probe_deadline = probe_started + post_deadline_probe_seconds
        post_process = post_window = post_foreground = False
        while True:
            state = read_target_state()
            record_eligible_window_diagnostics(state, "post_deadline")
            post_process, post_window, post_foreground = note_post_deadline_state(
                state, post_process, post_window, post_foreground,
            )
            remaining = probe_deadline - clock()
            if remaining <= 0:
                break
            sleep_fn(min(poll_interval_seconds, remaining))
        post_deadline = PostDeadlineProbeDiagnostics(
            performed=True,
            delay_ms=max(0, round((clock() - probe_started) * 1000)),
            target_process_observed=post_process,
            target_window_observed=post_window,
            target_foreground_observed=post_foreground,
        )

    if target_activator is not None and explicit_diagnostics is None:
        eligible, eligibility_reason, before_state = target_eligibility(last_target_state)
        if latest is not None and latest.application_id == candidate.id:
            eligibility_reason = "passive_activation_observed"
            eligible = False
        explicit_diagnostics = sanitized_explicit_diagnostics(
            eligible=eligible,
            eligibility_reason=eligibility_reason,
            attempted=explicit_attempted,
            budget_consumed=explicit_budget_consumed,
            window_state_before=before_state,
            mechanism=None,
            os_call_reported_success=None,
            fresh_verification_obtained=False,
            foreground_after=None,
            trusted_identity_after=None,
            success=False,
            failure_reason=None,
        )

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
        activation_elapsed_ms=activation_elapsed_ms,
        initial_foreground=_foreground_identity(initial_observation),
        observed_foregrounds=tuple(transitions),
        final_foreground=_foreground_identity(latest if latest is not None else initial_observation),
        identity_match=matched,
        identity_match_method=_identity_match_method(latest, candidate),
    )
    lifecycle = None
    if target_probe is not None or post_deadline_probe_seconds > 0 or target_activator is not None:
        lifecycle = ActivationLifecycleDiagnostics(
            launch_request_accepted=launch_succeeded,
            target_probe_available=target_probe_available,
            target_probe_complete=target_probe_available and target_probe_complete,
            target_process_observed=target_process_observed,
            target_window_observed=target_window_observed,
            target_foreground_observed=target_foreground_observed,
            first_target_process_elapsed_ms=first_target_process_elapsed_ms,
            first_target_window_elapsed_ms=first_target_window_elapsed_ms,
            first_target_foreground_elapsed_ms=first_target_foreground_elapsed_ms,
            target_window_state=target_window_state,
            matching_window_candidates=matching_window_candidates,
            candidate_diagnostics_truncated=candidate_diagnostics_truncated,
            post_deadline_probe=post_deadline,
            explicit_activation=explicit_diagnostics,
            window_resolution_status=window_resolution_status,
            matching_trusted_window_count=matching_trusted_window_count,
            eligible_window_count=eligible_window_count,
            enumeration_complete=enumeration_complete,
            rejection_reason_counts=rejection_reason_counts,
            eligible_window_diagnostics=tuple(eligible_window_records.values()),
            eligible_window_diagnostics_truncated=eligible_window_diagnostics_truncated,
            eligible_window_relationships=tuple(eligible_relationship_records.values()),
            eligible_window_relationships_truncated=eligible_window_relationships_truncated,
            base_eligible_window_count=eligible_window_count,
            primary_surface_candidate_count=primary_surface_candidate_count,
            tool_surface_candidate_count=tool_surface_candidate_count,
            primary_surface_resolution=primary_surface_resolution,
        )
    return ApplicationActivationWait(latest, diagnostics, reason, lifecycle)
