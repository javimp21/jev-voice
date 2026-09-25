"""Provider-neutral semantic evidence and deterministic target resolution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import hashlib
import json
import re
import unicodedata


class IdentityEvidence(StrEnum):
    MATCH = "match"
    MISMATCH = "mismatch"
    ABSENT = "absent"


class RoleCompatibility(StrEnum):
    EXACT = "exact"
    COMPATIBLE = "compatible"
    CONTAINER = "container"
    INCOMPATIBLE = "incompatible"
    UNKNOWN = "unknown"


class DomainSemanticCompatibility(StrEnum):
    """How candidate meaning compares with the requested domain entity kind."""

    EXACT = "exact"
    COMPATIBLE = "compatible"
    CONTRADICTORY = "contradictory"
    UNKNOWN = "unknown"


class PresentationRole(StrEnum):
    """Normalized UI presentation role, separate from target/domain meaning."""

    BUTTON = "button"
    LINK = "link"
    LIST_ITEM = "list_item"
    ROW = "row"
    CARD = "card"
    TREE_ITEM = "tree_item"
    TAB = "tab"
    MENU_ITEM = "menu_item"
    TEXT_FIELD = "text_field"
    SEARCH_FIELD = "search_field"
    CHECKBOX = "checkbox"
    TOGGLE = "toggle"
    CONTAINER = "container"
    LABEL = "label"
    UNKNOWN = "unknown"


class PresentationCompatibility(StrEnum):
    """Structural suitability of a UI presentation for the requested action."""

    COMPATIBLE = "compatible"
    INCOMPATIBLE = "incompatible"
    UNKNOWN = "unknown"


class TargetResolutionStatus(StrEnum):
    UNIQUE = "unique"
    CHOICE = "choice"
    AMBIGUOUS = "ambiguous"
    NO_MATCH = "no_match"


@dataclass(frozen=True, slots=True)
class TargetSpec:
    """Bounded intent; ``desired_role`` is a domain/target semantic role."""

    primary_identity: str
    qualifiers: tuple[str, ...] = ()
    desired_role: str | None = None
    action_intent: str = "activate"

    @property
    def domain_semantic_role(self) -> str | None:
        """Descriptive alias clarifying that ``desired_role`` is not a UI role."""
        return self.desired_role

    def __post_init__(self) -> None:
        if not _valid_text(self.primary_identity, 200):
            raise ValueError("primary_identity must contain 1 to 200 safe characters")
        if type(self.qualifiers) is not tuple or len(self.qualifiers) > 8:
            raise ValueError("qualifiers must be a tuple containing at most 8 values")
        if any(not _valid_text(value, 200) for value in self.qualifiers):
            raise ValueError("each qualifier must contain 1 to 200 safe characters")
        if self.desired_role is not None and not _valid_text(self.desired_role, 80):
            raise ValueError("desired_role must be None or contain at most 80 safe characters")
        if not _valid_text(self.action_intent, 40):
            raise ValueError("action_intent must contain 1 to 40 safe characters")


@dataclass(frozen=True, slots=True)
class DirectedGroundingEvidence:
    """Local binding between a directed request and one returned candidate.

    The fingerprints are hashes of locally constructed request data. The
    provider does not create or control this record.
    """

    target_spec_fingerprint: str
    objective_fingerprint: str
    snapshot_id: str
    previous_snapshot_id: str
    candidate_id: str
    max_elements: int

    def __post_init__(self) -> None:
        for name, value in (
            ("target_spec_fingerprint", self.target_spec_fingerprint),
            ("objective_fingerprint", self.objective_fingerprint),
        ):
            if not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError(f"{name} must be a SHA-256 hex digest")
        if not _valid_text(self.snapshot_id, 128):
            raise ValueError("snapshot_id must contain 1 to 128 safe characters")
        if not isinstance(self.previous_snapshot_id, str) or len(self.previous_snapshot_id) > 128:
            raise ValueError("previous_snapshot_id must contain at most 128 characters")
        if not _valid_text(self.candidate_id, 80):
            raise ValueError("candidate_id must contain 1 to 80 safe characters")
        if type(self.max_elements) is not int or not 1 <= self.max_elements <= 100:
            raise ValueError("max_elements must be between 1 and 100")


@dataclass(frozen=True, slots=True)
class CandidateEvidence:
    """Domain meaning, presentation, and authorization facts for one snapshot.

    ``semantic_role`` remains as a compatibility alias for older callers.
    New adapters should set ``target_semantic_evidence`` and
    ``presentation_role`` independently.
    """

    candidate_id: str
    primary_text: str
    secondary_text: tuple[str, ...]
    semantic_role: str | None
    actionable: bool
    geometry_valid: bool
    safety_eligible: bool
    source: str
    snapshot_id: str
    provider_role: str | None = None
    target_semantic_evidence: str | None = None
    presentation_role: PresentationRole | str | None = None
    directed_grounding: DirectedGroundingEvidence | None = None

    def __post_init__(self) -> None:
        if not _valid_text(self.candidate_id, 80):
            raise ValueError("candidate_id must contain 1 to 80 safe characters")
        if not isinstance(self.primary_text, str) or len(self.primary_text) > 240 or "\x00" in self.primary_text:
            raise ValueError("primary_text must be a string of at most 240 characters")
        if type(self.secondary_text) is not tuple or len(self.secondary_text) > 8:
            raise ValueError("secondary_text must be a tuple containing at most 8 values")
        if any(not isinstance(value, str) or len(value) > 200 or "\x00" in value
               for value in self.secondary_text):
            raise ValueError("secondary_text values must be strings of at most 200 characters")
        for name, value in (
            ("semantic_role", self.semantic_role), ("provider_role", self.provider_role),
            ("target_semantic_evidence", self.target_semantic_evidence),
        ):
            if value is not None and (not isinstance(value, str) or len(value) > 80 or "\x00" in value):
                raise ValueError(f"{name} must be None or a string of at most 80 characters")
        if self.presentation_role is not None and not isinstance(
            self.presentation_role, (str, PresentationRole),
        ):
            raise ValueError("presentation_role must be a normalized role or bounded string")
        if isinstance(self.presentation_role, str) and (
            len(self.presentation_role) > 80 or "\x00" in self.presentation_role
        ):
            raise ValueError("presentation_role must contain at most 80 safe characters")
        if self.directed_grounding is not None and not isinstance(
            self.directed_grounding, DirectedGroundingEvidence,
        ):
            raise ValueError("directed_grounding must be typed local evidence")
        if any(type(value) is not bool for value in (
            self.actionable, self.geometry_valid, self.safety_eligible,
        )):
            raise ValueError("candidate validation flags must be booleans")
        if not _valid_text(self.source, 40) or not _valid_text(self.snapshot_id, 128):
            raise ValueError("source and snapshot_id must be bounded nonempty strings")


@dataclass(frozen=True, slots=True)
class CandidateResolution:
    candidate_id: str
    primary_text: str
    secondary_text: tuple[str, ...]
    semantic_role: str | None
    provider_role: str | None
    source: str
    snapshot_id: str
    primary_identity: IdentityEvidence
    qualifier_evidence: tuple[IdentityEvidence, ...]
    role_compatibility: RoleCompatibility
    actionable: bool
    geometry_valid: bool
    safety_eligible: bool
    snapshot_valid: bool
    admissible: bool
    rejection_reasons: tuple[str, ...] = ()
    target_semantic_evidence: str | None = None
    domain_semantic_compatibility: DomainSemanticCompatibility = DomainSemanticCompatibility.UNKNOWN
    presentation_role: PresentationRole = PresentationRole.UNKNOWN
    presentation_compatibility: PresentationCompatibility = PresentationCompatibility.UNKNOWN


@dataclass(frozen=True, slots=True)
class TargetResolution:
    status: TargetResolutionStatus
    target: TargetSpec
    candidates: tuple[CandidateResolution, ...]
    admissible_candidate_ids: tuple[str, ...]
    strongest_candidate_ids: tuple[str, ...]
    selected_candidate_id: str | None
    resolution_reason: str
    frontier_candidate_ids: tuple[str, ...] = ()
    evidence_distinguishable: bool = False


def _valid_text(value: str, maximum: int) -> bool:
    return (
        isinstance(value, str) and bool(value.strip()) and len(value) <= maximum
        and "\x00" not in value and "\r" not in value and "\n" not in value
    )


def _normalize(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", unicodedata.normalize("NFKC", value).casefold())
    without_marks = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(re.findall(r"[^\W_]+", without_marks, flags=re.UNICODE))


def target_spec_fingerprint(target: TargetSpec) -> str:
    """Return a stable opaque fingerprint without exposing target text."""
    payload = json.dumps(
        (target.primary_identity, target.qualifiers, target.desired_role, target.action_intent),
        ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def grounding_objective_fingerprint(objective: str) -> str:
    """Return a stable opaque fingerprint for the exact bounded objective."""
    return hashlib.sha256(objective.encode("utf-8")).hexdigest()


def exact_identity_match(text: str, requested: str) -> bool:
    """Whether normalized identity text is exactly equal, without phrase matching."""
    return bool(_normalize(text)) and _normalize(text) == _normalize(requested)


def _contains_identity(text: str, requested: str) -> bool:
    normalized_text = _normalize(text)
    normalized_requested = _normalize(requested)
    if not normalized_text or not normalized_requested:
        return False
    return f" {normalized_requested} " in f" {normalized_text} "


def identity_evidence(requested: str, values: tuple[str, ...]) -> IdentityEvidence:
    """Match a phrase, report contradiction when identity-bearing text exists, else absence."""

    present = tuple(value for value in values if _normalize(value))
    if any(_contains_identity(value, requested) for value in present):
        return IdentityEvidence.MATCH
    return IdentityEvidence.MISMATCH if present else IdentityEvidence.ABSENT


def _qualifier_evidence(
    requested: str, primary_text: str, secondary_text: tuple[str, ...],
) -> IdentityEvidence:
    explicit_secondary = tuple(value for value in secondary_text if _normalize(value))
    if explicit_secondary:
        return (
            IdentityEvidence.MATCH
            if any(_contains_identity(value, requested) for value in explicit_secondary)
            else IdentityEvidence.MISMATCH
        )
    return (
        IdentityEvidence.MATCH if _contains_identity(primary_text, requested)
        else IdentityEvidence.ABSENT
    )


_DOMAIN_ROLE_ALIASES = {
    "file": "file",
    "file result": "file",
    "document": "file",
    "document result": "file",
    "conversation": "conversation",
    "conversation thread": "conversation",
    "chat": "conversation",
    "contact": "contact",
    "person": "contact",
    "folder": "container",
    "directory": "container",
    "container": "container",
    "navigation destination": "navigation_destination",
    "media": "media",
    "media item": "media",
    "song": "media",
    "song result": "media",
    "track": "media",
    "album": "media",
    "audio": "media",
    "video": "media",
    "image": "media",
    "setting": "setting",
    "settings": "setting",
    "preference": "setting",
}

_PRESENTATION_ROLE_ALIASES: dict[str, PresentationRole] = {
    "button": PresentationRole.BUTTON,
    "push button": PresentationRole.BUTTON,
    "command button": PresentationRole.BUTTON,
    "radio button": PresentationRole.BUTTON,
    "link": PresentationRole.LINK,
    "hyperlink": PresentationRole.LINK,
    "anchor": PresentationRole.LINK,
    "list item": PresentationRole.LIST_ITEM,
    "listitem": PresentationRole.LIST_ITEM,
    "list view item": PresentationRole.LIST_ITEM,
    "option": PresentationRole.LIST_ITEM,
    "result item": PresentationRole.LIST_ITEM,
    "row": PresentationRole.ROW,
    "table row": PresentationRole.ROW,
    "data row": PresentationRole.ROW,
    "data item": PresentationRole.ROW,
    "card": PresentationRole.CARD,
    "tile": PresentationRole.CARD,
    "tree item": PresentationRole.TREE_ITEM,
    "treeitem": PresentationRole.TREE_ITEM,
    "tab": PresentationRole.TAB,
    "tab item": PresentationRole.TAB,
    "tabitem": PresentationRole.TAB,
    "page tab": PresentationRole.TAB,
    "menu item": PresentationRole.MENU_ITEM,
    "menuitem": PresentationRole.MENU_ITEM,
    "text field": PresentationRole.TEXT_FIELD,
    "text box": PresentationRole.TEXT_FIELD,
    "editable field": PresentationRole.TEXT_FIELD,
    "input field": PresentationRole.TEXT_FIELD,
    "edit": PresentationRole.TEXT_FIELD,
    "search field": PresentationRole.SEARCH_FIELD,
    "search box": PresentationRole.SEARCH_FIELD,
    "search input": PresentationRole.SEARCH_FIELD,
    "checkbox": PresentationRole.CHECKBOX,
    "check box": PresentationRole.CHECKBOX,
    "check box control": PresentationRole.CHECKBOX,
    "toggle": PresentationRole.TOGGLE,
    "toggle button": PresentationRole.TOGGLE,
    "switch": PresentationRole.TOGGLE,
    "container": PresentationRole.CONTAINER,
    "pane": PresentationRole.CONTAINER,
    "panel": PresentationRole.CONTAINER,
    "group": PresentationRole.CONTAINER,
    "list": PresentationRole.CONTAINER,
    "region": PresentationRole.CONTAINER,
    "label": PresentationRole.LABEL,
    "text": PresentationRole.LABEL,
    "static text": PresentationRole.LABEL,
    "heading": PresentationRole.LABEL,
}
_ACTIVATABLE_PRESENTATION_ROLES = frozenset({
    PresentationRole.BUTTON, PresentationRole.LINK, PresentationRole.LIST_ITEM,
    PresentationRole.ROW, PresentationRole.CARD, PresentationRole.TREE_ITEM,
    PresentationRole.TAB, PresentationRole.MENU_ITEM, PresentationRole.CHECKBOX,
    PresentationRole.TOGGLE,
})


def normalize_presentation_role(
    role: str | PresentationRole | None,
) -> PresentationRole:
    if isinstance(role, PresentationRole):
        return role
    if not role:
        return PresentationRole.UNKNOWN
    normalized = _normalize(role.replace("_", " ").replace("-", " "))
    return _PRESENTATION_ROLE_ALIASES.get(normalized, PresentationRole.UNKNOWN)


def _canonical_domain_role(role: str | None, *, allow_unlisted: bool = False) -> str | None:
    if not role:
        return None
    normalized = _normalize(role.replace("_", " ").replace("-", " "))
    if not normalized or normalized in {"unknown", "unspecified", "none"}:
        return None
    if normalized.endswith(" like"):
        normalized = normalized[:-5].strip()
    if normalize_presentation_role(normalized) is not PresentationRole.UNKNOWN:
        return None
    known = _DOMAIN_ROLE_ALIASES.get(normalized)
    if known is not None:
        return known
    return normalized.replace(" ", "_") if allow_unlisted else None


def canonical_semantic_role(role: str | None) -> str | None:
    """Compatibility helper that recognizes domain semantics only."""
    return _canonical_domain_role(role)


def role_compatibility(
    desired_role: str | None, candidate_role: str | None,
) -> RoleCompatibility:
    desired = canonical_semantic_role(desired_role)
    candidate = canonical_semantic_role(candidate_role)
    if desired is None or candidate is None:
        return RoleCompatibility.UNKNOWN
    if desired == candidate:
        return RoleCompatibility.EXACT
    if desired == "conversation" and candidate == "contact":
        return RoleCompatibility.COMPATIBLE
    if candidate == "container":
        return RoleCompatibility.CONTAINER
    return RoleCompatibility.INCOMPATIBLE


def domain_semantic_compatibility(
    desired_role: str | None, candidate_role: str | None, *, explicit: bool = False,
) -> DomainSemanticCompatibility:
    desired = _canonical_domain_role(desired_role, allow_unlisted=True)
    candidate = _canonical_domain_role(candidate_role, allow_unlisted=explicit)
    if desired is None or candidate is None:
        return DomainSemanticCompatibility.UNKNOWN
    if desired == candidate:
        return DomainSemanticCompatibility.EXACT
    if desired == "conversation" and candidate == "contact":
        return DomainSemanticCompatibility.COMPATIBLE
    return DomainSemanticCompatibility.CONTRADICTORY


def presentation_role_compatibility(
    action_intent: str, presentation_role: str | PresentationRole | None,
) -> PresentationCompatibility:
    role = normalize_presentation_role(presentation_role)
    intent = _normalize(action_intent)
    if role in {
        PresentationRole.TEXT_FIELD, PresentationRole.SEARCH_FIELD,
        PresentationRole.CONTAINER, PresentationRole.LABEL,
    }:
        return PresentationCompatibility.INCOMPATIBLE
    if intent not in {"activate", "select"}:
        return PresentationCompatibility.UNKNOWN
    if role in _ACTIVATABLE_PRESENTATION_ROLES:
        return PresentationCompatibility.COMPATIBLE
    return PresentationCompatibility.UNKNOWN


def _candidate_domain_semantics(candidate: CandidateEvidence) -> tuple[str | None, bool]:
    if candidate.target_semantic_evidence is not None:
        return candidate.target_semantic_evidence, True
    return candidate.semantic_role, False


def _candidate_presentation_role(candidate: CandidateEvidence) -> PresentationRole:
    if candidate.presentation_role is not None:
        return normalize_presentation_role(candidate.presentation_role)
    role = normalize_presentation_role(candidate.provider_role)
    if role is not PresentationRole.UNKNOWN:
        return role
    return normalize_presentation_role(candidate.semantic_role)


def _overall_role_compatibility(
    domain: DomainSemanticCompatibility,
    presentation: PresentationRole,
    presentation_compatibility: PresentationCompatibility,
) -> RoleCompatibility:
    if presentation is PresentationRole.CONTAINER:
        return RoleCompatibility.CONTAINER
    if presentation_compatibility is PresentationCompatibility.INCOMPATIBLE:
        return RoleCompatibility.INCOMPATIBLE
    if domain is DomainSemanticCompatibility.EXACT:
        return RoleCompatibility.EXACT
    if domain is DomainSemanticCompatibility.COMPATIBLE:
        return RoleCompatibility.COMPATIBLE
    if domain is DomainSemanticCompatibility.CONTRADICTORY:
        return RoleCompatibility.INCOMPATIBLE
    # Generic UI presentation must not rank otherwise equivalent entities.
    return RoleCompatibility.UNKNOWN


_ROLE_STRENGTH = {
    RoleCompatibility.EXACT: 4,
    RoleCompatibility.COMPATIBLE: 3,
    RoleCompatibility.CONTAINER: 2,
    RoleCompatibility.UNKNOWN: 1,
    RoleCompatibility.INCOMPATIBLE: 0,
}


def resolve_target(
    target: TargetSpec,
    candidates: tuple[CandidateEvidence, ...],
    *,
    expected_snapshot_id: str,
    max_candidates: int = 100,
    max_frontier_candidates: int = 5,
    frontier_mode: bool = False,
) -> TargetResolution:
    """Reject contradictions and return a bounded Pareto frontier of candidates.

    Identity, safety, geometry, and snapshot validity are hard gates. Remaining
    candidates are compared on positive qualifier evidence and role strength.
    Coordinates, candidate ordering, source, and application identity never
    contribute to preference.
    """

    if not _valid_text(expected_snapshot_id, 128):
        raise ValueError("expected_snapshot_id must contain 1 to 128 safe characters")
    if type(candidates) is not tuple or len(candidates) > max_candidates:
        raise ValueError(f"candidates must be a tuple containing at most {max_candidates} values")
    if (max_candidates < 1 or max_frontier_candidates < 1
            or len({item.candidate_id for item in candidates}) != len(candidates)):
        raise ValueError("candidate IDs must be unique and the candidate limit positive")

    diagnostics: list[CandidateResolution] = []
    ranks: dict[str, tuple[int, int]] = {}
    admissible_ids: list[str] = []
    for candidate in candidates:
        primary = identity_evidence(target.primary_identity, (candidate.primary_text,))
        qualifiers = tuple(
            _qualifier_evidence(qualifier, candidate.primary_text, candidate.secondary_text)
            for qualifier in target.qualifiers
        )
        semantic_value, semantic_is_explicit = _candidate_domain_semantics(candidate)
        semantic_role = (
            semantic_value if semantic_is_explicit
            or canonical_semantic_role(semantic_value) is not None else None
        )
        domain_compatibility = domain_semantic_compatibility(
            target.desired_role, semantic_value, explicit=semantic_is_explicit,
        )
        presentation = _candidate_presentation_role(candidate)
        presentation_compatibility = presentation_role_compatibility(
            target.action_intent, presentation,
        )
        role = _overall_role_compatibility(
            domain_compatibility, presentation, presentation_compatibility,
        )
        snapshot_valid = candidate.snapshot_id == expected_snapshot_id
        reasons: list[str] = []
        if not candidate.actionable:
            reasons.append("not_actionable")
        if not candidate.geometry_valid:
            reasons.append("invalid_geometry")
        if not candidate.safety_eligible:
            reasons.append("safety_rejected")
        if not snapshot_valid:
            reasons.append("stale_snapshot")
        if primary is IdentityEvidence.ABSENT:
            reasons.append("primary_identity_absent")
        elif primary is IdentityEvidence.MISMATCH:
            reasons.append("primary_identity_mismatch")
        if IdentityEvidence.MISMATCH in qualifiers:
            reasons.append("qualifier_mismatch")
        if domain_compatibility is DomainSemanticCompatibility.CONTRADICTORY:
            reasons.append("contradictory_domain_semantics")
        if presentation_compatibility is PresentationCompatibility.INCOMPATIBLE:
            reasons.append("incompatible_presentation_role")
        if role is RoleCompatibility.INCOMPATIBLE:
            reasons.append("incompatible_role")
        if role is RoleCompatibility.CONTAINER:
            reasons.append("container_is_not_direct_target")
        admissible = not reasons
        diagnostics.append(CandidateResolution(
            candidate.candidate_id, candidate.primary_text, candidate.secondary_text,
            semantic_role, candidate.provider_role, candidate.source,
            candidate.snapshot_id, primary, qualifiers, role,
            candidate.actionable, candidate.geometry_valid, candidate.safety_eligible,
            snapshot_valid, admissible, tuple(reasons), semantic_role,
            domain_compatibility, presentation, presentation_compatibility,
        ))
        if admissible:
            admissible_ids.append(candidate.candidate_id)
            ranks[candidate.candidate_id] = (
                sum(item is IdentityEvidence.MATCH for item in qualifiers),
                _ROLE_STRENGTH[role],
            )

    if not admissible_ids:
        return TargetResolution(
            TargetResolutionStatus.NO_MATCH, target, tuple(diagnostics), (), (), None,
            "no_candidate_has_valid_positive_identity_and_safe_action_evidence", (), False,
        )

    if not frontier_mode:
        best_rank = max(ranks.values())
        strongest = tuple(
            candidate_id for candidate_id in admissible_ids if ranks[candidate_id] == best_rank
        )
        if len(strongest) > 1:
            return TargetResolution(
                TargetResolutionStatus.AMBIGUOUS, target, tuple(diagnostics),
                tuple(admissible_ids), strongest, None,
                "multiple_candidates_have_equal_strongest_identity_and_role_evidence",
                strongest, False,
            )
        return TargetResolution(
            TargetResolutionStatus.UNIQUE, target, tuple(diagnostics),
            tuple(admissible_ids), strongest, strongest[0],
            "one_candidate_has_uniquely_strongest_identity_and_role_evidence",
            strongest, False,
        )

    def dominates(left: tuple[int, int], right: tuple[int, int]) -> bool:
        return left[0] >= right[0] and left[1] >= right[1] and left != right

    frontier = tuple(
        candidate_id for candidate_id in admissible_ids
        if not any(dominates(ranks[other], ranks[candidate_id])
                   for other in admissible_ids if other != candidate_id)
    )
    if len(frontier) == 1:
        return TargetResolution(
            TargetResolutionStatus.UNIQUE, target, tuple(diagnostics),
            tuple(admissible_ids), frontier, frontier[0],
            "one_candidate_dominates_or_matches_all_admissible_evidence", frontier, False,
        )

    by_id = {item.candidate_id: item for item in diagnostics}
    signatures = {
        (
            tuple(value.value for value in by_id[candidate_id].qualifier_evidence),
            by_id[candidate_id].role_compatibility.value,
        )
        for candidate_id in frontier
    }
    distinguishable = len(signatures) > 1
    if distinguishable and len(frontier) <= max_frontier_candidates:
        status = TargetResolutionStatus.CHOICE
        reason = "non_dominated_candidates_have_distinct_bounded_semantic_evidence"
    else:
        status = TargetResolutionStatus.AMBIGUOUS
        reason = (
            "candidate_frontier_exceeds_bounded_choice_limit" if distinguishable
            else "non_dominated_candidates_have_indistinguishable_semantic_evidence"
        )
    return TargetResolution(
        status, target, tuple(diagnostics), tuple(admissible_ids), frontier, None,
        reason, frontier, distinguishable,
    )
