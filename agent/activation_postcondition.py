"""Deterministic semantic verification for generic target activation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import re
from collections.abc import Callable

from computer.models import Observation, VisualProviderAttempt, UIElement
from decision.target_resolution import (
    IdentityEvidence, TargetSpec, canonical_semantic_role, identity_evidence,
    role_compatibility,
)
from safety.policy import generic_uia_semantic_role


class ActivationEvidence(StrEnum):
    MATCH = "match"
    MISMATCH = "mismatch"
    UNKNOWN = "unknown"


class ActivationPostconditionResult(StrEnum):
    VERIFIED = "verified"
    CONTRADICTED = "contradicted"
    INSUFFICIENT = "insufficient"
    INCOMPLETE = "incomplete"


class ActivationEvidenceSource(StrEnum):
    UIA = "uia"
    VISUAL = "visual"
    COMBINED = "combined"


@dataclass(frozen=True, slots=True)
class TargetActivationPostcondition:
    attempted: bool
    target_identity: str
    desired_role: str | None
    structural_evidence_available: bool
    visual_verification_called: bool
    visual_provider_attempts: tuple[VisualProviderAttempt, ...]
    target_identity_evidence: ActivationEvidence
    active_state_evidence: ActivationEvidence
    trusted_context_stable: bool
    source: ActivationEvidenceSource
    result: ActivationPostconditionResult
    failure_reason: str | None = None


_ENTITY_CONTROL_TYPES = frozenset({
    "Button", "DataItem", "Hyperlink", "ListItem", "MenuItem", "TabItem", "TreeItem",
})
_ACTIVE_CONTEXT_MARKERS = frozenset({
    "active", "current", "detail", "details", "document", "header", "heading",
    "opened", "page", "profile", "recipient", "selected", "subject", "title",
})
_ACTIVE_TITLE = re.compile(r"\b(?:chat|conversation|document|profile)\s+(?:with|:)", re.I)


def _map_identity(value: IdentityEvidence) -> ActivationEvidence:
    if value is IdentityEvidence.MATCH:
        return ActivationEvidence.MATCH
    if value is IdentityEvidence.MISMATCH:
        return ActivationEvidence.MISMATCH
    return ActivationEvidence.UNKNOWN


def _entity_role_matches(control: UIElement, target: TargetSpec) -> bool:
    if target.desired_role is None:
        return control.control_type in _ENTITY_CONTROL_TYPES
    role = generic_uia_semantic_role(control) or canonical_semantic_role(control.control_type)
    if role is None:
        return False
    return role_compatibility(target.desired_role, role).value in {"exact", "compatible"}


def _active_context_control(control: UIElement) -> bool:
    context = " ".join((control.parent_name, control.parent_control_type, control.automation_id))
    words = set(re.findall(r"\w+", context.casefold().replace("_", " ").replace("-", " ")))
    if words & _ACTIVE_CONTEXT_MARKERS:
        return True
    # A self-identifying header/title is useful; a row under a generic
    # Conversations/Contacts list is not active evidence.
    own = " ".join((control.name, control.control_type, control.automation_id))
    own_words = set(re.findall(r"\w+", own.casefold().replace("_", " ").replace("-", " ")))
    return bool(own_words & (_ACTIVE_CONTEXT_MARKERS - {"selected", "active", "current"}))


def _uia_active_evidence(target: TargetSpec, observation: Observation) -> ActivationEvidence:
    evidence: list[ActivationEvidence] = []

    title = observation.window_title.strip()
    title_match = _map_identity(identity_evidence(target.primary_identity, (title,)))
    if title_match is ActivationEvidence.MATCH:
        evidence.append(title_match)
    elif title and _ACTIVE_TITLE.search(title):
        # A named current conversation/document title that names another
        # identity is explicit negative evidence. A generic app title is not.
        evidence.append(ActivationEvidence.MISMATCH)

    for control in observation.elements:
        if control.visible is False:
            continue
        identity_values = tuple(value for value in (control.name, control.observed_text or "") if value.strip())
        if not identity_values:
            continue
        is_selected_or_focused = control.selected is True or control.focused is True
        is_active_entity = (
            is_selected_or_focused
            and control.control_type in _ENTITY_CONTROL_TYPES
            and _entity_role_matches(control, target)
        )
        if is_active_entity or _active_context_control(control):
            evidence.append(_map_identity(identity_evidence(target.primary_identity, identity_values)))

    if ActivationEvidence.MISMATCH in evidence:
        return ActivationEvidence.MISMATCH
    if ActivationEvidence.MATCH in evidence:
        return ActivationEvidence.MATCH
    return ActivationEvidence.UNKNOWN


def _visual_active_evidence(target: TargetSpec, observation: Observation) -> ActivationEvidence:
    active = [item for item in observation.visual_elements if item.activity == "active"]
    if not active:
        return ActivationEvidence.UNKNOWN
    evidence: list[ActivationEvidence] = []
    desired_role = canonical_semantic_role(target.desired_role)
    for item in active:
        role = canonical_semantic_role(item.role)
        if desired_role is not None and role is not None:
            if role_compatibility(desired_role, role).value not in {"exact", "compatible"}:
                continue
        evidence.append(_map_identity(identity_evidence(
            target.primary_identity, tuple(value for value in (item.label, item.parent) if value.strip()),
        )))
    if ActivationEvidence.MISMATCH in evidence:
        return ActivationEvidence.MISMATCH
    if ActivationEvidence.MATCH in evidence:
        return ActivationEvidence.MATCH
    return ActivationEvidence.UNKNOWN


def evaluate_target_activation_postcondition(
    target: TargetSpec,
    observation: Observation,
    *,
    trusted_context_stable: bool,
    visual_verification_called: bool = False,
    visual_provider_attempts: tuple[VisualProviderAttempt, ...] = (),
    provider_incomplete: bool = False,
    clean_text: Callable[[str], str] = lambda value: value,
) -> TargetActivationPostcondition:
    """Combine typed active-entity evidence; visible identity alone never verifies."""

    uia_active = _uia_active_evidence(target, observation)
    visual_active = _visual_active_evidence(target, observation) if visual_verification_called else ActivationEvidence.UNKNOWN
    source = (
        ActivationEvidenceSource.COMBINED
        if visual_verification_called and uia_active is not ActivationEvidence.UNKNOWN
        and visual_active is not ActivationEvidence.UNKNOWN
        else ActivationEvidenceSource.VISUAL
        if visual_active is not ActivationEvidence.UNKNOWN
        else ActivationEvidenceSource.UIA
    )
    identity_evidence_value = (
        ActivationEvidence.MISMATCH
        if ActivationEvidence.MISMATCH in {uia_active, visual_active}
        else ActivationEvidence.MATCH
        if ActivationEvidence.MATCH in {uia_active, visual_active}
        else ActivationEvidence.UNKNOWN
    )
    active_evidence_value = identity_evidence_value

    failure: str | None = None
    if not trusted_context_stable:
        result = ActivationPostconditionResult.CONTRADICTED
        failure = "trusted_context_changed"
    elif ActivationEvidence.MISMATCH in {identity_evidence_value, active_evidence_value}:
        result = ActivationPostconditionResult.CONTRADICTED
        failure = "active_identity_mismatch"
    elif (identity_evidence_value is ActivationEvidence.MATCH
          and active_evidence_value is ActivationEvidence.MATCH):
        result = ActivationPostconditionResult.VERIFIED
    elif provider_incomplete:
        result = ActivationPostconditionResult.INCOMPLETE
        failure = "visual_verification_incomplete"
    else:
        result = ActivationPostconditionResult.INSUFFICIENT
        failure = "active_identity_not_established"

    return TargetActivationPostcondition(
        attempted=True,
        target_identity=clean_text(target.primary_identity)[:120],
        desired_role=(clean_text(target.desired_role)[:80] if target.desired_role else None),
        structural_evidence_available=bool(observation.elements or observation.window_title.strip()),
        visual_verification_called=visual_verification_called,
        visual_provider_attempts=visual_provider_attempts[:4],
        target_identity_evidence=identity_evidence_value,
        active_state_evidence=active_evidence_value,
        trusted_context_stable=trusted_context_stable,
        source=source,
        result=result,
        failure_reason=failure,
    )
