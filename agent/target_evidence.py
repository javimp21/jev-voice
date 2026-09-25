"""Adapt one bounded UIA/visual observation into resolver evidence and actions."""

from __future__ import annotations

from dataclasses import dataclass
import re

from computer.actions import Action, ClickAction, VisualClickAction
from computer.models import Observation, Rect
from decision.context import Redactor
from decision.target_resolution import (
    CandidateEvidence, DirectedGroundingEvidence, TargetSpec,
    canonical_semantic_role, grounding_objective_fingerprint,
    normalize_presentation_role, target_spec_fingerprint,
)
from safety.policy import GenericTargetActivationPolicy, generic_uia_semantic_role
from computer.visual import VisualGroundingRequest


MAX_UIA_TARGET_CANDIDATES = 40
MAX_VISUAL_TARGET_CANDIDATES = 20
_STRUCTURAL_PARENT_WORDS = frozenset({
    "chat", "chats", "conversation", "conversations", "contact", "contacts",
    "file", "files", "folder", "folders", "document", "documents", "results",
    "result", "list", "pane", "panel", "navigation", "menu",
})


@dataclass(frozen=True, slots=True)
class TargetEvidenceSet:
    candidates: tuple[CandidateEvidence, ...]
    actions: dict[str, Action]
    conflicting_source_candidate_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DirectedGroundingBinding:
    """Controller-owned context for exactly one directed observation call."""

    target_spec_fingerprint: str
    objective_fingerprint: str
    snapshot_id: str
    previous_snapshot_id: str
    max_elements: int


def bind_directed_grounding(
    target: TargetSpec,
    request: VisualGroundingRequest,
    previous_observation: Observation,
    returned_observation: Observation,
    *,
    expected_objective: str,
) -> DirectedGroundingBinding | None:
    """Bind returned visual evidence only when the exact directed call is verifiable."""
    snapshot_id = returned_observation.observation_id
    metadata = returned_observation.screenshot
    if (
        request.objective != expected_objective
        or not previous_observation.observation_id
        or not snapshot_id
        or snapshot_id == previous_observation.observation_id
        or returned_observation.error
        or returned_observation.visual_directed_grounding is not True
        or returned_observation.visual_requested_max_elements != request.max_elements
        or metadata is None
        or metadata.snapshot_id != snapshot_id
    ):
        return None
    return DirectedGroundingBinding(
        target_spec_fingerprint(target), grounding_objective_fingerprint(request.objective),
        snapshot_id, previous_observation.observation_id, request.max_elements,
    )


def _clean(value: str, redactor: Redactor, maximum: int) -> str:
    return " ".join(redactor.clean(value).split())[:maximum]


def _valid_rect(rectangle: Rect | None) -> bool:
    return rectangle is None or (
        rectangle.right > rectangle.left and rectangle.bottom > rectangle.top
    )


def _identity_context(value: str, redactor: Redactor) -> str:
    cleaned = _clean(value, redactor, 200)
    words = set(re.findall(r"[^\W_]+", cleaned.casefold(), flags=re.UNICODE))
    return "" if words and words <= _STRUCTURAL_PARENT_WORDS else cleaned


def adapt_observation_candidates(
    observation: Observation,
    policy: GenericTargetActivationPolicy,
    redactor: Redactor | None = None,
    *,
    grounding_binding: DirectedGroundingBinding | None = None,
) -> TargetEvidenceSet:
    """Create source-bound resolver evidence without dropping execution actions."""
    cleaner = redactor or Redactor()
    evidence: list[CandidateEvidence] = []
    actions: dict[str, Action] = {}
    snapshot_id = observation.observation_id
    if not snapshot_id or observation.error:
        return TargetEvidenceSet((), {})

    for control in observation.elements[:MAX_UIA_TARGET_CANDIDATES]:
        primary = _clean(control.name, cleaner, 240)
        if not primary and control.observed_text is not None and not control.observed_text_truncated:
            primary = _clean(control.observed_text, cleaner, 240)
        if not primary:
            continue
        action = ClickAction(control.id)
        domain_role = generic_uia_semantic_role(control)
        presentation_role = normalize_presentation_role(control.control_type)
        verdict = policy.validate_candidate(action, observation, semantic_role=domain_role)
        secondary = _identity_context(control.parent_name, cleaner)
        item = CandidateEvidence(
            candidate_id=control.id,
            primary_text=primary,
            secondary_text=(secondary,) if secondary else (),
            semantic_role=domain_role,
            actionable=(control.visible is True and control.enabled is True),
            geometry_valid=_valid_rect(control.rectangle),
            safety_eligible=verdict.disposition == "allow",
            source="UIA",
            snapshot_id=snapshot_id,
            provider_role=control.control_type[:80] or None,
            target_semantic_evidence=domain_role,
            presentation_role=presentation_role,
        )
        evidence.append(item)
        actions[item.candidate_id] = action

    metadata = observation.screenshot
    for element in observation.visual_elements[:MAX_VISUAL_TARGET_CANDIDATES]:
        primary = _clean(element.label, cleaner, 240)
        if not primary:
            continue
        action = VisualClickAction(snapshot_id, element.id)
        geometry_valid = bool(
            metadata is not None and metadata.snapshot_id == snapshot_id
            and element.rectangle.left >= 0 and element.rectangle.top >= 0
            and element.rectangle.right > element.rectangle.left
            and element.rectangle.bottom > element.rectangle.top
            and element.rectangle.right <= metadata.pixel_width
            and element.rectangle.bottom <= metadata.pixel_height
        )
        verdict = policy.validate_candidate(action, observation)
        secondary = _identity_context(element.parent, cleaner)
        domain_role = canonical_semantic_role(element.role[:80])
        presentation_role = normalize_presentation_role(element.role[:80])
        item = CandidateEvidence(
            candidate_id=element.id,
            primary_text=primary,
            secondary_text=(secondary,) if secondary else (),
            semantic_role=domain_role,
            actionable=element.clickable is True,
            geometry_valid=geometry_valid,
            safety_eligible=verdict.disposition == "allow",
            source="VISUAL",
            snapshot_id=snapshot_id,
            provider_role=element.role[:80] or None,
            target_semantic_evidence=domain_role,
            presentation_role=presentation_role,
            directed_grounding=(
                DirectedGroundingEvidence(
                    grounding_binding.target_spec_fingerprint,
                    grounding_binding.objective_fingerprint,
                    grounding_binding.snapshot_id,
                    grounding_binding.previous_snapshot_id,
                    element.id,
                    grounding_binding.max_elements,
                )
                if grounding_binding is not None
                and grounding_binding.snapshot_id == snapshot_id
                and observation.visual_directed_grounding is True
                and observation.visual_requested_max_elements == grounding_binding.max_elements
                and metadata is not None and metadata.snapshot_id == snapshot_id
                else None
            ),
        )
        evidence.append(item)
        actions[item.candidate_id] = action

    # IDs are snapshot-local but must still be unique across the mixed sources.
    seen: set[str] = set()
    bounded: list[CandidateEvidence] = []
    bounded_actions: dict[str, Action] = {}
    by_id: dict[str, CandidateEvidence] = {}
    conflicts: set[str] = set()
    for item in evidence:
        if item.candidate_id in seen:
            previous = by_id[item.candidate_id]
            if previous.source != item.source or previous != item:
                conflicts.add(item.candidate_id)
            continue
        seen.add(item.candidate_id)
        bounded.append(item)
        bounded_actions[item.candidate_id] = actions[item.candidate_id]
        by_id[item.candidate_id] = item
    return TargetEvidenceSet(tuple(bounded), bounded_actions, tuple(sorted(conflicts)))


def query_literal_from_target(target) -> str:
    """Construct a literal only from text already present in the parsed request."""
    values = (target.primary_identity, *target.qualifiers)
    literal = " ".join(value.strip() for value in values if value.strip())
    if (not literal or len(literal) > 240 or "\x00" in literal
            or any("\r" in value or "\n" in value for value in values)):
        raise ValueError("A bounded literal query could not be derived from the target.")
    return literal


def is_query_field(control, policy: GenericTargetActivationPolicy) -> bool:
    return policy.is_query_field(control)


def is_visual_query_field(element) -> bool:
    if not element.clickable or not element.label.strip():
        return False
    role = re.sub(r"[_-]+", " ", element.role.casefold()).strip()
    return role in {"search field", "search box", "text field", "text box", "edit"}
