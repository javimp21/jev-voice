"""Provider-neutral semantic evidence and deterministic target resolution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
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


class TargetResolutionStatus(StrEnum):
    UNIQUE = "unique"
    CHOICE = "choice"
    AMBIGUOUS = "ambiguous"
    NO_MATCH = "no_match"


@dataclass(frozen=True, slots=True)
class TargetSpec:
    """Bounded user intent, independent of application and provider."""

    primary_identity: str
    qualifiers: tuple[str, ...] = ()
    desired_role: str | None = None
    action_intent: str = "activate"

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
class CandidateEvidence:
    """Separated semantic fields and local authorization facts for one snapshot."""

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
        for name, value in (("semantic_role", self.semantic_role), ("provider_role", self.provider_role)):
            if value is not None and (not isinstance(value, str) or len(value) > 80 or "\x00" in value):
                raise ValueError(f"{name} must be None or a string of at most 80 characters")
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


_ROLE_ALIASES = {
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
    "navigation item": "navigation_destination",
    "menu item": "navigation_destination",
    "tab": "navigation_destination",
    "tab item": "navigation_destination",
    "actionable item": "actionable_item",
    "result": "actionable_item",
    "result item": "actionable_item",
    "card": "actionable_item",
    "list item": "actionable_item",
    "link": "actionable_item",
    "button": "actionable_item",
}


def canonical_semantic_role(role: str | None) -> str | None:
    if not role:
        return None
    normalized = _normalize(role.replace("_", " ").replace("-", " "))
    return _ROLE_ALIASES.get(normalized)


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
        role = role_compatibility(target.desired_role, candidate.semantic_role)
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
        if role is RoleCompatibility.INCOMPATIBLE:
            reasons.append("incompatible_role")
        if role is RoleCompatibility.CONTAINER:
            reasons.append("container_is_not_direct_target")
        admissible = not reasons
        diagnostics.append(CandidateResolution(
            candidate.candidate_id, candidate.primary_text, candidate.secondary_text,
            candidate.semantic_role, candidate.provider_role, candidate.source,
            candidate.snapshot_id, primary, qualifiers, role,
            candidate.actionable, candidate.geometry_valid, candidate.safety_eligible,
            snapshot_valid, admissible, tuple(reasons),
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
