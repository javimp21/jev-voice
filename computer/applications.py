"""Platform-neutral trusted application catalog contracts and matching."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import re
from typing import Literal, Protocol
import unicodedata


LaunchPolicy = Literal["allow", "confirm", "deny"]
ApplicationSource = Literal["start_menu", "packaged", "test"]
WindowAreaBucket = Literal["zero", "small", "medium", "large", "unavailable"]
WindowZOrderBucket = Literal["front", "middle", "back", "unavailable"]
RootOwnerRelationship = Literal["self", "other", "unavailable"]
WindowCloakedState = Literal["uncloaked", "cloaked", "unknown"]
EligibleWindowOwnershipRelationship = Literal[
    "independent", "one_owned_by_other", "same_root_owner", "unknown",
]
WindowResolutionStatus = Literal["unique", "none", "ambiguous", "incomplete"]
EligibleWindowSurfaceClass = Literal["primary", "tool", "unknown"]
WindowRejectionReason = Literal[
    "trusted_identity_mismatch", "window_missing", "target_not_visible",
    "cloaking_unknown", "target_cloaked", "client_area_unknown", "zero_client_area",
]
WindowActivationEligibilityReason = Literal[
    "eligible",
    "probe_incomplete",
    "target_window_missing",
    "no_eligible_window",
    "target_window_ambiguous",
    "target_window_unstable",
    "target_not_visible",
    "cloaking_unknown",
    "target_cloaked",
    "client_area_unknown",
    "zero_client_area",
    "already_foreground",
    "trusted_identity_mismatch",
    "stale_window",
    "window_state_changed",
    "system_surface",
    "restore_not_started",
    "os_activation_rejected",
    "activation_error",
]
WindowActivationMechanism = Literal[
    "activate",
    "restore_then_activate",
    "restore",
    "already_foreground",
]
WindowActivationStrategy = Literal["simple", "attach_thread_input"]


@dataclass(frozen=True, slots=True)
class ApplicationCandidate:
    """Sanitized public identity; trusted launch references remain catalog-private."""

    id: str
    display_name: str
    source: ApplicationSource
    publisher: str = ""
    launch_policy: LaunchPolicy = "confirm"
    process_names: tuple[str, ...] = ()
    package_family: str = ""


@dataclass(frozen=True, slots=True)
class ApplicationMatch:
    candidate: ApplicationCandidate
    score: int


@dataclass(frozen=True, slots=True)
class ActivationWindowCandidateDiagnostics:
    """Sanitized, coarse structural facts about one trusted top-level window."""

    candidate_index: int
    trusted_identity_match: bool
    visible: bool | None
    minimized: bool | None
    enabled: bool | None
    cloaked_if_available: bool | None
    owner_present: bool | None
    root_owner_relationship: RootOwnerRelationship
    tool_window: bool | None
    app_window: bool | None
    has_nonzero_client_area: bool | None
    client_area_bucket: WindowAreaBucket
    window_area_bucket: WindowAreaBucket
    foreground: bool
    stable_across_probes: bool
    z_order_bucket: WindowZOrderBucket
    activation_candidate: bool
    explicit_activation_eligible: bool = False
    rejection_reason: WindowRejectionReason | None = None


@dataclass(frozen=True, slots=True)
class EligibleWindowFacts:
    """Sanitized structural facts from one eligible-window probe."""

    visible: bool | None
    minimized: bool | None
    enabled: bool | None
    cloaked_state: WindowCloakedState
    owner_present: bool | None
    root_owner_relationship: RootOwnerRelationship
    tool_window: bool | None
    app_window: bool | None
    has_nonzero_client_area: bool | None
    client_area_bucket: WindowAreaBucket
    window_area_bucket: WindowAreaBucket
    foreground: bool
    stable_across_probes: bool
    explicit_activation_eligible: bool
    eligibility_reason: Literal["eligible"]
    rejection_reason: None = None
    z_order_bucket: WindowZOrderBucket = "unavailable"


@dataclass(frozen=True, slots=True)
class EligibleWindowDiagnostics:
    """Run-local correlation and bounded phase snapshots for an eligible window."""

    diagnostic_id: str
    eligible_candidate_index: int
    facts: EligibleWindowFacts
    present_in_initial_probe: bool = False
    present_in_repeated_probe: bool = False
    present_in_post_deadline_probe: bool = False
    observed_probe_count: int = 0
    initial_probe_facts: EligibleWindowFacts | None = None
    latest_repeated_probe_facts: EligibleWindowFacts | None = None
    latest_post_deadline_probe_facts: EligibleWindowFacts | None = None
    surface_class: EligibleWindowSurfaceClass = "unknown"


@dataclass(frozen=True, slots=True)
class EligibleWindowRelationship:
    """Coarse ownership relation for a pair of eligible windows."""

    first_diagnostic_id: str
    second_diagnostic_id: str
    relationship: EligibleWindowOwnershipRelationship
    present_in_initial_probe: bool = False
    present_in_repeated_probe: bool = False
    present_in_post_deadline_probe: bool = False
    observed_probe_count: int = 0
    initial_probe_relationship: EligibleWindowOwnershipRelationship | None = None
    latest_repeated_probe_relationship: EligibleWindowOwnershipRelationship | None = None
    latest_post_deadline_probe_relationship: EligibleWindowOwnershipRelationship | None = None


@dataclass(frozen=True, slots=True)
class TrustedApplicationRuntimeState:
    """Bounded identity-only evidence for a catalog-trusted running application."""

    process_observed: bool = False
    window_observed: bool = False
    foreground_observed: bool = False
    visible: bool | None = None
    minimized: bool | None = None
    window_foreground: bool | None = None
    trusted_identity_match: bool = False
    # False means the probe cannot safely conclude that an unobserved target is absent.
    probe_complete: bool = True
    # True only when the same unique eligible trusted window was found consecutively.
    window_stable: bool = False
    # True means multiple eligible windows matched; callers must not choose one.
    window_ambiguous: bool = False
    window_resolution_status: WindowResolutionStatus | None = None
    matching_trusted_window_count: int | None = None
    eligible_window_count: int | None = None
    enumeration_complete: bool | None = None
    rejection_reason_counts: tuple[tuple[str, int], ...] = ()
    # Sanitized diagnostics. Resolver inputs always come from the full internal scan.
    window_candidates: tuple[ActivationWindowCandidateDiagnostics, ...] = ()
    candidate_diagnostics_truncated: bool = False
    eligible_window_diagnostics: tuple[EligibleWindowDiagnostics, ...] = ()
    eligible_window_diagnostics_truncated: bool = False
    eligible_window_relationships: tuple[EligibleWindowRelationship, ...] = ()
    # Separate explicit-activation selection metadata; base eligibility above is unchanged.
    base_eligible_window_count: int | None = None
    primary_surface_candidate_count: int | None = None
    tool_surface_candidate_count: int | None = None
    primary_surface_resolution_status: WindowResolutionStatus | None = None
    primary_surface_facts: EligibleWindowFacts | None = None


@dataclass(frozen=True, slots=True)
class SimpleActivationAttempt:
    """Safe result fields for the first foreground attempt."""

    set_foreground_return: bool | None
    verified: bool | None


@dataclass(frozen=True, slots=True)
class ThreadInputFallbackAttempt:
    """Sanitized diagnostics; raw HWNDs and thread IDs are intentionally absent."""

    eligible: bool
    reason: str
    foreground_hwnd_present: bool | None = None
    selected_thread_resolved: bool | None = None
    foreground_thread_resolved: bool | None = None
    current_thread_resolved: bool | None = None
    incidental_foreground_changed_before_activation: bool | None = None
    selected_target_still_valid_before_activation: bool | None = None
    attach_attempted: bool = False
    attach_succeeded: bool | None = None
    set_foreground_attempted: bool = False
    set_foreground_return: bool | None = None
    bring_to_top_used: bool = False
    set_active_used: bool = False
    detach_succeeded: bool | None = None
    verified: bool | None = None
    failure_reason: str | None = None


@dataclass(frozen=True, slots=True)
class TrustedWindowActivationResult:
    """Sanitized result of one local, catalog-bound window activation attempt."""

    eligible: bool
    eligibility_reason: WindowActivationEligibilityReason
    visible: bool | None = None
    minimized: bool | None = None
    trusted_identity_match: bool = False
    mechanism: WindowActivationMechanism | None = None
    os_call_reported_success: bool | None = None
    failure_reason: str | None = None
    budget_consumed: bool | None = None
    foreground_verified: bool | None = None
    activation_strategy: WindowActivationStrategy | None = None
    simple_attempt: SimpleActivationAttempt | None = None
    fallback_attempt: ThreadInputFallbackAttempt | None = None


class ApplicationCatalog(Protocol):
    def discover(self) -> tuple[ApplicationCandidate, ...]: ...
    def find(self, query: str, limit: int = 5) -> tuple[ApplicationMatch, ...]: ...
    def resolve(self, app_id: str) -> ApplicationCandidate | None: ...
    def launch(self, app_id: str) -> None: ...
    def identify(self, process_name: str, package_family: str = "") -> str | None: ...


def normalize_app_name(value: str) -> str:
    # Decorative symbols (for example the trademark sign) carry no useful
    # identity signal. Remove them before NFKC can expand them into letters.
    without_symbols = "".join(character for character in value
                              if not unicodedata.category(character).startswith("S"))
    normalized = unicodedata.normalize("NFKC", without_symbols).casefold()
    return " ".join(re.findall(r"[\w]+", normalized, flags=re.UNICODE))


_QUERY_STOPWORDS = frozenset({
    "open", "launch", "start", "run", "and", "then", "search", "for", "go", "to", "the",
    "app", "application", "write", "type", "enter", "find", "navigate", "in", "with", "a", "an",
})


def _score(query: str, candidate: ApplicationCandidate) -> int:
    query_name = normalize_app_name(query)
    name = normalize_app_name(candidate.display_name)
    if not query_name or not name:
        return 0
    if query_name == name:
        return 100
    query_tokens = set(query_name.split()) - _QUERY_STOPWORDS
    name_tokens = set(name.split())
    if not query_tokens or not name_tokens:
        return 0
    if re.search(rf"(?<!\w){re.escape(name)}(?!\w)", query_name):
        return 95
    overlap = query_tokens & name_tokens
    if not overlap:
        return 0
    if name_tokens <= query_tokens:
        return 90
    if query_tokens <= name_tokens:
        return 85
    coverage = len(overlap) / len(name_tokens)
    return 70 + round(15 * coverage) if coverage >= 0.5 else 0


def match_applications(
    query: str, candidates: Sequence[ApplicationCandidate], limit: int = 5,
) -> tuple[ApplicationMatch, ...]:
    if limit < 1:
        raise ValueError("Application candidate limit must be positive.")
    matches = [ApplicationMatch(candidate, _score(query, candidate)) for candidate in candidates]
    plausible = [match for match in matches if match.score >= 70 and match.candidate.launch_policy != "deny"]
    plausible.sort(key=lambda match: (-match.score, normalize_app_name(match.candidate.display_name), match.candidate.id))
    return tuple(plausible[:limit])


class MemoryApplicationCatalog:
    """Deterministic catalog used by tests and embedders with trusted callbacks."""

    def __init__(
        self, candidates: Sequence[ApplicationCandidate],
        launchers: dict[str, Callable[[], None]] | None = None,
    ) -> None:
        self._candidates = tuple(candidates)
        self._by_id = {candidate.id: candidate for candidate in self._candidates}
        self._launchers = launchers or {}

    def discover(self) -> tuple[ApplicationCandidate, ...]:
        return self._candidates

    def find(self, query: str, limit: int = 5) -> tuple[ApplicationMatch, ...]:
        return match_applications(query, self._candidates, limit)

    def resolve(self, app_id: str) -> ApplicationCandidate | None:
        return self._by_id.get(app_id)

    def launch(self, app_id: str) -> None:
        candidate = self.resolve(app_id)
        if candidate is None or candidate.launch_policy != "allow" or app_id not in self._launchers:
            raise ValueError("Application ID is unavailable or not approved for launch.")
        self._launchers[app_id]()

    def identify(self, process_name: str, package_family: str = "") -> str | None:
        process = normalize_app_name(process_name.removesuffix(".exe"))
        package = package_family.casefold()
        matches = [candidate.id for candidate in self._candidates if (
            package and candidate.package_family.casefold() == package
        ) or any(normalize_app_name(name.removesuffix(".exe")) == process for name in candidate.process_names)]
        return matches[0] if len(matches) == 1 else None
