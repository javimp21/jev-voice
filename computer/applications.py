"""Platform-neutral trusted application catalog contracts and matching."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
import re
from typing import Literal, Protocol
import unicodedata


LaunchPolicy = Literal["allow", "confirm", "deny"]
ApplicationSource = Literal["start_menu", "packaged", "test", "equivalent"]
ApplicationLaunchTargetType = Literal["shortcut", "packaged", "executable", "other"]
ApplicationEquivalenceResult = Literal["PROVEN_EQUIVALENT", "PROVEN_DISTINCT", "UNKNOWN"]
ApplicationEquivalenceSignal = Literal[
    "same_app_user_model_id", "same_trusted_executable_identity", "same_catalog_identity",
    "different_package_identity", "identity_metadata_conflict",
    "insufficient_identity_evidence", "candidate_identity_unavailable",
]


@dataclass(frozen=True, slots=True)
class ComparedIdentityFieldsPresent:
    app_user_model_id: bool = False
    package_identity: bool = False
    executable_identity: bool = False
    shortcut_arguments: bool = False
    publisher_product: bool = False
    catalog_identity: bool = False


@dataclass(frozen=True, slots=True)
class ApplicationIdentityEvidence:
    """Hashed identity facts from trusted local metadata; raw values stay private."""

    app_user_model_id_fingerprint: str | None = field(default=None, repr=False)
    package_identity_fingerprint: str | None = field(default=None, repr=False)
    executable_identity_fingerprint: str | None = field(default=None, repr=False)
    shortcut_arguments_fingerprint: str | None = field(default=None, repr=False)
    publisher_product_fingerprint: str | None = field(default=None, repr=False)
    catalog_identity_fingerprint: str | None = field(default=None, repr=False)


@dataclass(frozen=True, slots=True)
class ApplicationIdentityEquivalence:
    comparison_attempted: bool
    equivalence_result: ApplicationEquivalenceResult
    equivalence_signal_kind: ApplicationEquivalenceSignal
    compared_identity_fields_present: ComparedIdentityFieldsPresent
    equivalence_fingerprint: str | None = field(default=None, repr=False)


@dataclass(frozen=True, slots=True)
class ApplicationIdentityComparisonDiagnostics:
    left_app_id: str
    right_app_id: str
    comparison_attempted: bool
    equivalence_result: ApplicationEquivalenceResult
    equivalence_signal_kind: ApplicationEquivalenceSignal
    compared_identity_fields_present: ComparedIdentityFieldsPresent


def safe_application_id(value: str) -> str:
    """Return an opaque bounded app identifier suitable for diagnostics."""
    if re.fullmatch(r"app_[a-f0-9]{16}", value):
        return value
    return "app_" + sha256(value.encode("utf-8", "replace")).hexdigest()[:16]


def compare_application_identity_evidence(
    left: ApplicationIdentityEvidence | None,
    right: ApplicationIdentityEvidence | None,
) -> ApplicationIdentityEquivalence:
    """Compare only exact trusted identifiers; missing evidence stays UNKNOWN."""
    if left is None or right is None:
        return ApplicationIdentityEquivalence(
            True, "UNKNOWN", "candidate_identity_unavailable", ComparedIdentityFieldsPresent(),
        )
    fields = ComparedIdentityFieldsPresent(
        app_user_model_id=bool(left.app_user_model_id_fingerprint and right.app_user_model_id_fingerprint),
        package_identity=bool(left.package_identity_fingerprint and right.package_identity_fingerprint),
        executable_identity=bool(left.executable_identity_fingerprint and right.executable_identity_fingerprint),
        shortcut_arguments=bool(left.shortcut_arguments_fingerprint and right.shortcut_arguments_fingerprint),
        publisher_product=bool(left.publisher_product_fingerprint and right.publisher_product_fingerprint),
        catalog_identity=bool(left.catalog_identity_fingerprint and right.catalog_identity_fingerprint),
    )
    same_app_id = (
        fields.app_user_model_id
        and left.app_user_model_id_fingerprint == right.app_user_model_id_fingerprint
    )
    same_package = (
        fields.package_identity
        and left.package_identity_fingerprint == right.package_identity_fingerprint
    )
    different_package = fields.package_identity and not same_package
    different_shortcut_arguments = (
        fields.shortcut_arguments
        and left.shortcut_arguments_fingerprint != right.shortcut_arguments_fingerprint
    )
    if different_shortcut_arguments:
        return ApplicationIdentityEquivalence(
            True, "UNKNOWN", "identity_metadata_conflict", fields,
        )
    if same_app_id and different_package:
        return ApplicationIdentityEquivalence(
            True, "UNKNOWN", "identity_metadata_conflict", fields,
        )
    if same_app_id:
        fingerprint = left.app_user_model_id_fingerprint
        return ApplicationIdentityEquivalence(
            True, "PROVEN_EQUIVALENT", "same_app_user_model_id", fields,
            f"app_user_model_id:{fingerprint}",
        )
    if different_package:
        return ApplicationIdentityEquivalence(
            True, "PROVEN_DISTINCT", "different_package_identity", fields,
        )
    if (fields.executable_identity
            and left.executable_identity_fingerprint == right.executable_identity_fingerprint):
        fingerprint = left.executable_identity_fingerprint
        return ApplicationIdentityEquivalence(
            True, "PROVEN_EQUIVALENT", "same_trusted_executable_identity", fields,
            f"executable_identity:{fingerprint}",
        )
    if (fields.catalog_identity
            and left.catalog_identity_fingerprint == right.catalog_identity_fingerprint):
        fingerprint = left.catalog_identity_fingerprint
        return ApplicationIdentityEquivalence(
            True, "PROVEN_EQUIVALENT", "same_catalog_identity", fields,
            f"catalog_identity:{fingerprint}",
        )
    return ApplicationIdentityEquivalence(
        True, "UNKNOWN", "insufficient_identity_evidence", fields,
    )
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
    identity_fingerprint: str | None = field(default=None, repr=False, compare=False)
    launch_target_type: ApplicationLaunchTargetType = "other"
    identity_aliases: tuple[str, ...] = field(default=(), repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class ApplicationMatch:
    candidate: ApplicationCandidate
    score: int
    match_kind: Literal[
        "raw_exact", "canonical_exact", "normalized_exact", "unique_fuzzy", "ambiguous",
    ] = "unique_fuzzy"
    equivalent_candidates: tuple[ApplicationCandidate, ...] = field(default=(), repr=False)


@dataclass(frozen=True, slots=True)
class ApplicationCandidateDiagnostic:
    """Bounded safe metadata for one physical catalog representation."""

    app_id: str
    canonical_name: str
    source_kind: str
    identity_fingerprint_present: bool
    identity_equivalence_group: str | None
    launch_target_type: ApplicationLaunchTargetType
    duplicate_of_another_candidate: bool


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


_DECORATIVE_NAME_SYMBOLS = frozenset({"™", "®", "©", "℠"})
_IDENTITY_PUNCTUATION = frozenset("+#.-_&!'/()")
_APPLICATION_VERB = re.compile(r"^\s*(?:open|launch|start|run)\s+", re.I)


def normalize_app_identity(value: str) -> str:
    """Canonical app identity that preserves product-name punctuation."""
    without_decorative = "".join(
        character for character in value if character not in _DECORATIVE_NAME_SYMBOLS
    )
    normalized = unicodedata.normalize("NFKC", without_decorative).casefold()
    output: list[str] = []
    for character in normalized:
        if character in _DECORATIVE_NAME_SYMBOLS:
            continue
        if character.isspace():
            output.append(" ")
        elif character.isalnum() or character == "_" or character in _IDENTITY_PUNCTUATION:
            output.append(character)
        else:
            output.append(" ")
    return " ".join("".join(output).split())


def normalize_app_search(value: str) -> str:
    """Loose search form; punctuation and symbols are intentionally discarded."""
    # Decorative symbols (for example the trademark sign) carry no useful
    # identity signal. Remove them before NFKC can expand them into letters.
    without_symbols = "".join(character for character in value
                              if not unicodedata.category(character).startswith("S"))
    normalized = unicodedata.normalize("NFKC", without_symbols).casefold()
    return " ".join(re.findall(r"[\w]+", normalized, flags=re.UNICODE))


def normalize_app_name(value: str) -> str:
    """Backward-compatible alias for the loose application search form."""
    return normalize_app_search(value)


_QUERY_STOPWORDS = frozenset({
    "open", "launch", "start", "run", "and", "then", "search", "for", "go", "to", "the",
    "app", "application", "write", "type", "enter", "find", "navigate", "in", "with", "a", "an",
})


def _score(query: str, candidate: ApplicationCandidate) -> int:
    query_name = normalize_app_search(query)
    names = (candidate.display_name, *candidate.identity_aliases)
    return max((_score_name(query_name, normalize_app_search(name)) for name in names), default=0)


def _score_name(query_name: str, name: str) -> int:
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


def logical_application_id(identity_fingerprint: str) -> str:
    """Create an opaque ID shared only by candidates with proven identity."""
    digest = sha256(f"logical-application\0{identity_fingerprint}".encode("utf-8")).hexdigest()[:16]
    return f"app_{digest}"


def _candidate_identity_groups(
    candidates: Sequence[ApplicationCandidate],
) -> tuple[tuple[ApplicationCandidate, tuple[ApplicationCandidate, ...]], ...]:
    groups: list[list[ApplicationCandidate]] = []
    fingerprint_indexes: dict[str, int] = {}
    for candidate in candidates:
        fingerprint = candidate.identity_fingerprint
        if fingerprint:
            index = fingerprint_indexes.get(fingerprint)
            if index is not None:
                groups[index].append(candidate)
                continue
            fingerprint_indexes[fingerprint] = len(groups)
        groups.append([candidate])

    result: list[tuple[ApplicationCandidate, tuple[ApplicationCandidate, ...]]] = []
    policy_rank = {"deny": 0, "confirm": 1, "allow": 2}
    for members in groups:
        original_members = tuple(members)
        if len(members) == 1:
            result.append((members[0], original_members))
            continue
        fingerprints = {item.identity_fingerprint for item in members}
        if len(fingerprints) != 1 or None in fingerprints:
            # Defensive fail-closed behavior if a malformed group is ever supplied.
            result.extend((item, (item,)) for item in members)
            continue
        sources = {item.source for item in members}
        families = {item.package_family for item in members if item.package_family}
        target_types = {item.launch_target_type for item in members}
        aliases = tuple(dict.fromkeys(
            name for item in members for name in (item.display_name, *item.identity_aliases)
            if name.strip()
        ))
        logical = ApplicationCandidate(
            id=logical_application_id(next(iter(fingerprints))),
            display_name=members[0].display_name,
            source=next(iter(sources)) if len(sources) == 1 else "equivalent",
            publisher=(members[0].publisher if len({item.publisher for item in members}) == 1 else ""),
            launch_policy=min((item.launch_policy for item in members), key=policy_rank.__getitem__),
            process_names=tuple(dict.fromkeys(name for item in members for name in item.process_names)),
            package_family=next(iter(families)) if len(families) == 1 else "",
            identity_fingerprint=next(iter(fingerprints)),
            launch_target_type=next(iter(target_types)) if len(target_types) == 1 else "other",
            identity_aliases=aliases,
        )
        result.append((logical, original_members))
    return tuple(result)


def group_application_candidates(
    candidates: Sequence[ApplicationCandidate],
) -> tuple[tuple[ApplicationCandidate, tuple[ApplicationCandidate, ...]], ...]:
    """Build logical candidates only for entries sharing a trusted fingerprint."""
    return _candidate_identity_groups(candidates)


def diagnose_application_matches(
    matches: Sequence[ApplicationMatch],
) -> tuple[ApplicationCandidateDiagnostic, ...]:
    """Return bounded, path-free diagnostics for the physical matched entries."""
    originals: list[ApplicationCandidate] = []
    for match in matches:
        members = match.equivalent_candidates or (match.candidate,)
        originals.extend(members)

    fingerprint_counts: dict[str, int] = {}
    for candidate in originals:
        if candidate.identity_fingerprint:
            fingerprint_counts[candidate.identity_fingerprint] = (
                fingerprint_counts.get(candidate.identity_fingerprint, 0) + 1
            )
    duplicate_fingerprints = sorted(
        fingerprint for fingerprint, count in fingerprint_counts.items() if count > 1
    )
    group_ids = {
        fingerprint: f"identity_group_{index + 1}"
        for index, fingerprint in enumerate(duplicate_fingerprints)
    }
    diagnostics: list[ApplicationCandidateDiagnostic] = []
    for candidate in originals:
        display_name = candidate.display_name
        normalized = normalize_app_identity(display_name)
        unsafe_name = (
            len(display_name) > 160
            or "\\" in display_name or "/" in display_name
            or re.search(r"\b(?:api[ _-]?key|token|password|secret|credential)\b", display_name, re.I)
        )
        canonical_name = "[redacted]" if unsafe_name else normalized[:80]
        app_id = candidate.id
        if not re.fullmatch(r"app_[a-f0-9]{16}", app_id):
            app_id = safe_application_id(app_id)
        source_kind = candidate.source if candidate.source in {
            "start_menu", "packaged", "test", "equivalent",
        } else "other"
        fingerprint = candidate.identity_fingerprint
        diagnostics.append(ApplicationCandidateDiagnostic(
            app_id=app_id,
            canonical_name=canonical_name,
            source_kind=source_kind,
            identity_fingerprint_present=bool(fingerprint),
            identity_equivalence_group=group_ids.get(fingerprint) if fingerprint else None,
            launch_target_type=candidate.launch_target_type,
            duplicate_of_another_candidate=bool(fingerprint and fingerprint_counts.get(fingerprint, 0) > 1),
        ))
    return tuple(diagnostics)


def match_applications(
    query: str, candidates: Sequence[ApplicationCandidate], limit: int = 5,
) -> tuple[ApplicationMatch, ...]:
    if limit < 1:
        raise ValueError("Application candidate limit must be positive.")
    groups = tuple(
        (logical, members)
        for logical, members in _candidate_identity_groups(candidates)
        if logical.launch_policy != "deny"
    )
    eligible = tuple(logical for logical, _members in groups)
    query_tail = _APPLICATION_VERB.sub("", query, count=1).strip()
    raw_query = " ".join(query_tail.split())

    def exact_matches(
        kind: Literal["raw_exact", "canonical_exact", "normalized_exact"],
        query_name: str,
        *, case_insensitive: bool = False,
    ) -> tuple[ApplicationMatch, ...]:
        comparison_query = query_name.casefold() if case_insensitive else query_name
        found: list[tuple[ApplicationCandidate, tuple[ApplicationCandidate, ...]]] = []
        for logical, members in groups:
            names = tuple(dict.fromkeys((logical.display_name, *logical.identity_aliases)))
            if any(
                ((name.strip().casefold() if case_insensitive else name.strip()) == comparison_query)
                if kind == "raw_exact"
                else ((normalize_app_identity(name) if kind == "canonical_exact"
                       else normalize_app_search(name)) == query_name)
                for name in names
            ):
                found.append((logical, members))
        if not found:
            return ()
        result_kind = kind if len(found) == 1 else "ambiguous"
        return tuple(
            ApplicationMatch(logical, 100, result_kind, members)
            for logical, members in found[:limit]
        )

    raw_matches = exact_matches(
        "raw_exact", raw_query, case_insensitive=True,
    )
    if raw_matches:
        return raw_matches

    identity_query = normalize_app_identity(query_tail)
    identity_matches = exact_matches(
        "canonical_exact", identity_query,
    )
    if identity_matches:
        return identity_matches

    search_query = normalize_app_search(query_tail)
    search_matches = exact_matches("normalized_exact", search_query)
    if search_matches:
        return search_matches

    plausible = [
        ApplicationMatch(candidate, score, equivalent_candidates=members)
        for candidate, members in groups
        if (score := _score(query, candidate)) >= 70
    ]
    plausible.sort(key=lambda match: (
        -match.score, normalize_app_identity(match.candidate.display_name), match.candidate.id,
    ))
    if len(plausible) > 1:
        plausible = [
            ApplicationMatch(
                item.candidate, item.score, "ambiguous", item.equivalent_candidates,
            ) for item in plausible
        ]
    elif plausible:
        item = plausible[0]
        plausible = [ApplicationMatch(
            item.candidate, item.score, "unique_fuzzy", item.equivalent_candidates,
        )]
    return tuple(plausible[:limit])


class MemoryApplicationCatalog:
    """Deterministic catalog used by tests and embedders with trusted callbacks."""

    def __init__(
        self, candidates: Sequence[ApplicationCandidate],
        launchers: dict[str, Callable[[], None]] | None = None,
    ) -> None:
        self._candidates = tuple(candidates)
        groups = group_application_candidates(self._candidates)
        self._logical_members = {logical.id: members for logical, members in groups}
        self._by_id = {candidate.id: candidate for candidate in self._candidates}
        self._by_id.update({logical.id: logical for logical, _members in groups})
        self._launchers = launchers or {}

    def discover(self) -> tuple[ApplicationCandidate, ...]:
        return self._candidates

    def find(self, query: str, limit: int = 5) -> tuple[ApplicationMatch, ...]:
        return match_applications(query, self._candidates, limit)

    def resolve(self, app_id: str) -> ApplicationCandidate | None:
        return self._by_id.get(app_id)

    def launch(self, app_id: str) -> None:
        candidate = self.resolve(app_id)
        members = self._logical_members.get(app_id, ())
        launcher_id = next((item.id for item in members if item.id in self._launchers), None)
        if candidate is None or candidate.launch_policy != "allow" or launcher_id is None:
            raise ValueError("Application ID is unavailable or not approved for launch.")
        self._launchers[launcher_id]()

    def identify(self, process_name: str, package_family: str = "") -> str | None:
        process = normalize_app_identity(process_name.removesuffix(".exe"))
        package = package_family.casefold()
        matches = [candidate.id for candidate in self._candidates if (
            package and candidate.package_family.casefold() == package
        ) or any(normalize_app_identity(name.removesuffix(".exe")) == process
                 for name in candidate.process_names)]
        return matches[0] if len(matches) == 1 else None
