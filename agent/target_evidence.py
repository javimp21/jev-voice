"""Adapt one bounded UIA/visual observation into resolver evidence and actions."""

from __future__ import annotations

from dataclasses import dataclass
import re

from computer.actions import Action, ClickAction, VisualClickAction
from computer.models import Observation, Rect
from decision.context import Redactor
from decision.target_resolution import CandidateEvidence
from safety.policy import GenericTargetActivationPolicy, generic_uia_semantic_role


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
        role = generic_uia_semantic_role(control)
        verdict = policy.validate_candidate(action, observation, semantic_role=role)
        secondary = _identity_context(control.parent_name, cleaner)
        item = CandidateEvidence(
            candidate_id=control.id,
            primary_text=primary,
            secondary_text=(secondary,) if secondary else (),
            semantic_role=role,
            actionable=(control.visible is True and control.enabled is True),
            geometry_valid=_valid_rect(control.rectangle),
            safety_eligible=verdict.disposition == "allow",
            source="UIA",
            snapshot_id=snapshot_id,
            provider_role=control.control_type[:80] or None,
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
        item = CandidateEvidence(
            candidate_id=element.id,
            primary_text=primary,
            secondary_text=(secondary,) if secondary else (),
            semantic_role=element.role[:80] or None,
            actionable=element.clickable is True,
            geometry_valid=geometry_valid,
            safety_eligible=verdict.disposition == "allow",
            source="VISUAL",
            snapshot_id=snapshot_id,
            provider_role=element.role[:80] or None,
        )
        evidence.append(item)
        actions[item.candidate_id] = action

    # IDs are snapshot-local but must still be unique across the mixed sources.
    seen: set[str] = set()
    bounded: list[CandidateEvidence] = []
    bounded_actions: dict[str, Action] = {}
    for item in evidence:
        if item.candidate_id in seen:
            continue
        seen.add(item.candidate_id)
        bounded.append(item)
        bounded_actions[item.candidate_id] = actions[item.candidate_id]
    return TargetEvidenceSet(tuple(bounded), bounded_actions)


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
