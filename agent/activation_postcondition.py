"""Deterministic semantic verification for generic target activation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
import re

from computer.models import (
    Observation, UIElement, VisualElement, VisualProviderAttempt, VisualRegion,
    VisualSelectionState,
)
from decision.target_resolution import (
    IdentityEvidence, TargetSpec, canonical_semantic_role, identity_evidence,
    normalize_presentation_role, PresentationRole, role_compatibility,
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
class StructuralPostconditionEvidenceItem:
    """Bounded UIA fact used, or deliberately ignored, by the verifier."""

    evidence_kind: str
    control_id: str | None
    control_type: str
    identity_text: str
    automation_id: str
    parent_name: str
    parent_control_type: str
    selected: bool | None
    focused: bool | None
    presentation_role: str
    identity_relation: ActivationEvidence
    active_state_relation: ActivationEvidence
    authority_class: str


@dataclass(frozen=True, slots=True)
class StructuralPostconditionEvidence:
    evidence_items: tuple[StructuralPostconditionEvidenceItem, ...] = ()
    positive_identity_count: int = 0
    contradictory_identity_count: int = 0
    authoritative_contradiction_count: int = 0
    ignored_non_authoritative_count: int = 0
    truncated: bool = False
    conflicting_authoritative_evidence: bool = False


@dataclass(frozen=True, slots=True)
class VisualPostconditionCandidateDiagnostic:
    candidate_id: str | None
    identity_text: str
    provider_role: str
    presentation_role: str
    region: str
    provider_region: str
    geometry_region_bucket: str
    activity_evidence: str
    selected_state: str
    identity_relation: ActivationEvidence
    active_state_relation: ActivationEvidence
    authority_class: str
    rejection_reason: str | None = None


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
    structural_postcondition_evidence: StructuralPostconditionEvidence = StructuralPostconditionEvidence()
    visual_grounding_objective: str | None = None
    visual_candidate_count: int = 0
    visual_candidate_diagnostics: tuple[VisualPostconditionCandidateDiagnostic, ...] = ()


_ENTITY_CONTROL_TYPES = frozenset({
    "Button", "DataItem", "Hyperlink", "ListItem", "MenuItem", "TabItem", "TreeItem",
})
_COLLECTION_CONTEXT_MARKERS = frozenset({
    "chats", "contacts", "conversations", "list", "menu", "navigation",
    "sidebar", "tabs",
})
_CURRENT_DETAIL_MARKERS = frozenset({
    "content", "detail", "details", "header", "heading", "opened", "recipient",
    "subject", "title",
})
_ACTIVE_TITLE = re.compile(r"\b(?:chat|conversation|document|profile)\s+(?:with|:)", re.I)
_MAX_EVIDENCE_ITEMS = 32


def _map_identity(value: IdentityEvidence) -> ActivationEvidence:
    if value is IdentityEvidence.MATCH:
        return ActivationEvidence.MATCH
    if value is IdentityEvidence.MISMATCH:
        return ActivationEvidence.MISMATCH
    return ActivationEvidence.UNKNOWN


def _entity_role_matches(control: UIElement, target: TargetSpec) -> bool:
    if target.desired_role is None:
        # Avoid treating arbitrary focused buttons as the requested entity.
        return control.control_type in {"DataItem", "ListItem", "TabItem", "TreeItem"}
    role = generic_uia_semantic_role(control) or canonical_semantic_role(control.control_type)
    if role is None:
        return False
    return role_compatibility(target.desired_role, role).value in {"exact", "compatible"}


def _words(*values: str) -> set[str]:
    return set(re.findall(
        r"\w+", " ".join(values).casefold().replace("_", " ").replace("-", " "),
    ))


def _presentation_role(control: UIElement) -> str:
    parent_words = _words(control.parent_name)
    parent_type_words = _words(control.parent_control_type)
    context_words = _words(control.parent_name, control.automation_id)
    if context_words & _CURRENT_DETAIL_MARKERS:
        return "active_detail"
    if (parent_words & _COLLECTION_CONTEXT_MARKERS
            or parent_type_words & {"list", "listview", "tree", "treeview", "menu", "tabs"}):
        return "navigation_sidebar"
    if (control.control_type in {"Header", "TitleBar"}
            or control.parent_control_type in {"Header", "TitleBar"}):
        return "header_title"
    if control.control_type in _ENTITY_CONTROL_TYPES:
        return "identity_bearing_control"
    if control.control_type in {"Text", "Document", "Custom"}:
        return "content_text"
    return "unrelated_control"


def _active_detail_authority(control: UIElement, presentation_role: str) -> bool:
    if presentation_role == "active_detail":
        return control.control_type in {"Text", "Header", "TitleBar", "Document", "Custom"}
    # A child of a UIA header/title surface can identify the active entity, but
    # generic section headers and navigation collections are not authoritative.
    parent_words = _words(control.parent_name)
    if parent_words & _COLLECTION_CONTEXT_MARKERS:
        return False
    return (
        control.parent_control_type in {"Header", "TitleBar"}
        and control.control_type in {"Text", "Header", "TitleBar"}
    )


def _merge_evidence(*values: ActivationEvidence) -> ActivationEvidence:
    known = {value for value in values if value is not ActivationEvidence.UNKNOWN}
    # Opposing strong facts do not establish which surface is current.
    if len(known) > 1:
        return ActivationEvidence.UNKNOWN
    if ActivationEvidence.MISMATCH in known:
        return ActivationEvidence.MISMATCH
    if ActivationEvidence.MATCH in known:
        return ActivationEvidence.MATCH
    return ActivationEvidence.UNKNOWN


def _uia_evidence(
    target: TargetSpec,
    observation: Observation,
    clean_text: Callable[[str], str],
) -> tuple[ActivationEvidence, ActivationEvidence, StructuralPostconditionEvidence]:
    visible_identity: list[ActivationEvidence] = []
    active_identity: list[ActivationEvidence] = []
    diagnostic_items: list[StructuralPostconditionEvidenceItem] = []
    positive_count = 0
    contradictory_count = 0
    authoritative_contradiction_count = 0
    ignored_non_authoritative_count = 0
    authoritative_relations: list[ActivationEvidence] = []

    def add_item(
        *, evidence_kind: str, control: UIElement | None, text: str,
        relation: ActivationEvidence, presentation_role: str,
        authority_class: str, active_relation: ActivationEvidence,
    ) -> None:
        nonlocal positive_count, contradictory_count
        nonlocal authoritative_contradiction_count, ignored_non_authoritative_count
        if relation is ActivationEvidence.MATCH:
            positive_count += 1
        elif relation is ActivationEvidence.MISMATCH:
            contradictory_count += 1
            if authority_class == "authoritative_active_entity":
                authoritative_contradiction_count += 1
            else:
                ignored_non_authoritative_count += 1
        if authority_class == "authoritative_active_entity" and relation is not ActivationEvidence.UNKNOWN:
            authoritative_relations.append(relation)
        diagnostic_items.append(StructuralPostconditionEvidenceItem(
            evidence_kind=evidence_kind,
            control_id=control.id if control is not None else None,
            control_type=(clean_text(control.control_type)[:40] if control is not None else "Window"),
            identity_text=clean_text(text)[:100],
            automation_id=(clean_text(control.automation_id)[:80] if control is not None else ""),
            parent_name=(clean_text(control.parent_name)[:100] if control is not None else ""),
            parent_control_type=(clean_text(control.parent_control_type)[:40] if control is not None else ""),
            selected=control.selected if control is not None else None,
            focused=control.focused if control is not None else None,
            presentation_role=presentation_role,
            identity_relation=relation,
            active_state_relation=active_relation,
            authority_class=authority_class,
        ))

    title = observation.window_title.strip()
    title_relation = _map_identity(identity_evidence(target.primary_identity, (title,)))
    title_is_entity_title = bool(title and _ACTIVE_TITLE.search(title))
    if title_relation is ActivationEvidence.MATCH:
        visible_identity.append(title_relation)
    if title_is_entity_title:
        active_identity.append(title_relation)
    if title and (title_is_entity_title or title_relation is ActivationEvidence.MATCH):
        add_item(
            evidence_kind="window_title", control=None, text=title,
            relation=title_relation,
            presentation_role=("current_surface_title" if title_is_entity_title else "window_title"),
            authority_class=("authoritative_active_entity" if title_is_entity_title
                             else "non_authoritative_visible"),
            active_relation=(title_relation if title_is_entity_title else ActivationEvidence.UNKNOWN),
        )

    for control in observation.elements:
        if control.visible is not True:
            continue
        presentation = _presentation_role(control)
        identity_candidate = (
            control.control_type in _ENTITY_CONTROL_TYPES
            or presentation in {"active_detail", "header_title"}
            or (presentation == "navigation_sidebar" and control.control_type in {
                "Text", "DataItem", "ListItem", "TabItem", "TreeItem",
            })
            or control.selected is True
            or control.focused is True
        )
        if not identity_candidate:
            continue
        identity_values = tuple(
            (kind, value) for kind, value in (
                ("control_name", control.name),
                ("observed_text", control.observed_text or ""),
            ) if value.strip()
        )
        if not identity_values:
            continue

        field_relations = tuple(
            (kind, value, _map_identity(identity_evidence(target.primary_identity, (value,))))
            for kind, value in identity_values
        )
        control_relation = _merge_evidence(*(relation for _, _, relation in field_relations))
        if control_relation is ActivationEvidence.MATCH and (
            control.control_type in _ENTITY_CONTROL_TYPES
            or presentation in {"active_detail", "header_title", "navigation_sidebar"}
        ):
            visible_identity.append(control_relation)

        selected_entity = (
            control.selected is True
            and control.control_type in _ENTITY_CONTROL_TYPES
            and _entity_role_matches(control, target)
        )
        focused_entity = (
            control.focused is True
            and control.selected is not True
            and control.control_type in _ENTITY_CONTROL_TYPES
            and _entity_role_matches(control, target)
            and presentation != "navigation_sidebar"
        )
        active_detail = _active_detail_authority(control, presentation)
        authoritative = selected_entity or focused_entity or active_detail
        authority_class = (
            "authoritative_active_entity" if authoritative else "non_authoritative_visible"
        )
        active_relation = control_relation if authoritative else ActivationEvidence.UNKNOWN
        if authoritative and control_relation is not ActivationEvidence.UNKNOWN:
            active_identity.append(control_relation)
        for kind, value, relation in field_relations:
            add_item(
                evidence_kind=kind, control=control, text=value,
                relation=relation, presentation_role=presentation,
                authority_class=authority_class,
                active_relation=(relation if authoritative else ActivationEvidence.UNKNOWN),
            )

    active = _merge_evidence(*active_identity)
    # Visible contrary identities are useful diagnostics, not proof that the
    # requested entity is inactive. Preserve positive visibility without
    # promoting a non-active row to an authoritative mismatch.
    positive_visible = tuple(value for value in visible_identity if value is ActivationEvidence.MATCH)
    identity = active if active is not ActivationEvidence.UNKNOWN else _merge_evidence(*positive_visible)
    diagnostic_items.sort(key=lambda item: (
        0 if (item.authority_class == "authoritative_active_entity"
              and item.active_state_relation is ActivationEvidence.MISMATCH) else
        1 if (item.authority_class == "authoritative_active_entity"
              and item.active_state_relation is ActivationEvidence.MATCH) else
        2 if (item.presentation_role == "navigation_sidebar"
              and item.identity_relation is ActivationEvidence.MISMATCH) else
        3 if item.identity_relation is ActivationEvidence.MATCH else 4,
    ))
    evidence = StructuralPostconditionEvidence(
        evidence_items=tuple(diagnostic_items[:_MAX_EVIDENCE_ITEMS]),
        positive_identity_count=positive_count,
        contradictory_identity_count=contradictory_count,
        authoritative_contradiction_count=authoritative_contradiction_count,
        ignored_non_authoritative_count=ignored_non_authoritative_count,
        truncated=(len(diagnostic_items) > _MAX_EVIDENCE_ITEMS),
        conflicting_authoritative_evidence=(
            ActivationEvidence.MATCH in authoritative_relations
            and ActivationEvidence.MISMATCH in authoritative_relations
        ),
    )
    return identity, active, evidence


def _visual_evidence(
    target: TargetSpec, observation: Observation,
    clean_text: Callable[[str], str],
) -> tuple[
    ActivationEvidence, ActivationEvidence,
    tuple[VisualPostconditionCandidateDiagnostic, ...],
]:
    visible_identity: list[ActivationEvidence] = []
    active_identity: list[ActivationEvidence] = []
    facts: list[dict[str, object]] = []
    metadata = observation.screenshot

    for item in observation.visual_elements:
        identity_values = tuple(value for value in (item.label, item.parent) if value.strip())
        identity = _merge_evidence(*(
            _map_identity(identity_evidence(target.primary_identity, (value,)))
            for value in identity_values
        ))
        if identity is ActivationEvidence.MATCH:
            visible_identity.append(identity)

        presentation = normalize_presentation_role(item.role).value
        provider_region = (
            item.region.value if isinstance(item.region, VisualRegion)
            else VisualRegion.UNKNOWN.value
        )
        geometry_region = "unknown"
        region = "unknown"
        if (metadata is not None and metadata.snapshot_id == observation.observation_id
                and metadata.pixel_width > 0 and metadata.pixel_height > 0
                and item.rectangle.left >= 0 and item.rectangle.top >= 0
                and item.rectangle.right > item.rectangle.left
                and item.rectangle.bottom > item.rectangle.top
                and item.rectangle.right <= metadata.pixel_width
                and item.rectangle.bottom <= metadata.pixel_height):
            center_x = (item.rectangle.left + item.rectangle.right) / (2 * metadata.pixel_width)
            center_y = (item.rectangle.top + item.rectangle.bottom) / (2 * metadata.pixel_height)
            top = item.rectangle.top / metadata.pixel_height
            bottom = item.rectangle.bottom / metadata.pixel_height
            nav_role = normalize_presentation_role(item.role) in {
                PresentationRole.LIST_ITEM, PresentationRole.ROW,
                PresentationRole.TREE_ITEM, PresentationRole.TAB,
                PresentationRole.MENU_ITEM,
            }
            if nav_role and center_x <= 0.38 and center_y >= 0.08:
                geometry_region = "navigation"
            elif center_x >= 0.32 and top <= 0.22 and bottom <= 0.36:
                geometry_region = "header"
            elif center_x >= 0.38 and 0.16 <= center_y <= 0.56:
                geometry_region = "main_area"
            elif center_x >= 0.38 and center_y > 0.56:
                geometry_region = "content"

            if geometry_region == "navigation" or provider_region == VisualRegion.NAVIGATION.value:
                region = VisualRegion.NAVIGATION.value
            elif (provider_region == VisualRegion.HEADER.value
                  and geometry_region in {"header", "main_area"}):
                region = VisualRegion.HEADER.value
            elif (provider_region == VisualRegion.DETAIL.value
                  and geometry_region in {"main_area", "content"}):
                region = VisualRegion.DETAIL.value
            elif (provider_region == VisualRegion.CONTENT.value
                  and geometry_region in {"main_area", "content"}):
                region = VisualRegion.CONTENT.value
            elif provider_region == VisualRegion.UNKNOWN.value:
                region = (
                    VisualRegion.HEADER.value if geometry_region == "header"
                    else VisualRegion.CONTENT.value if geometry_region == "content"
                    else VisualRegion.UNKNOWN.value
                )

        activity = item.activity if item.activity in {"active", "not_active", "unknown"} else "unknown"
        selected_state = (
            item.selection_state.value
            if isinstance(item.selection_state, VisualSelectionState)
            else VisualSelectionState.UNKNOWN.value
        )
        rejection: str | None = None
        authority = "non_authoritative"
        active_relation = ActivationEvidence.UNKNOWN
        if region in {VisualRegion.DETAIL.value, VisualRegion.HEADER.value}:
            if activity == "not_active":
                rejection = "provider_marks_detail_candidate_not_active"
                if identity is ActivationEvidence.MATCH:
                    authority = "authoritative"
                    active_relation = ActivationEvidence.MISMATCH
            elif identity is ActivationEvidence.UNKNOWN:
                rejection = "active_region_has_no_target_identity"
            else:
                authority = "authoritative"
                active_relation = identity
        elif region == VisualRegion.NAVIGATION.value:
            rejection = (
                "selected_navigation_row_requires_corresponding_active_detail"
                if selected_state == VisualSelectionState.SELECTED.value
                else "navigation_identity_is_not_active_context_evidence"
            )
        elif region == VisualRegion.CONTENT.value:
            rejection = "content_identity_requires_selected_row_or_detail_context"
        elif geometry_region == "main_area" and provider_region in {
            VisualRegion.HEADER.value, VisualRegion.DETAIL.value,
        }:
            rejection = "provider_region_conflicts_with_geometry"
        else:
            rejection = "visual_region_not_authoritative"

        facts.append({
            "item": item,
            "identity": identity,
            "region": region,
            "provider_region": provider_region,
            "geometry_region": geometry_region,
            "activity": activity,
            "selected_state": selected_state,
            "active_relation": active_relation,
            "authority": authority,
            "rejection": rejection,
            "presentation": presentation,
        })

    # A selected navigation row becomes authoritative only when the current
    # main/detail surface corroborates that same identity.
    current_surfaces = [fact for fact in facts if (
        fact["region"] in {VisualRegion.HEADER.value, VisualRegion.DETAIL.value}
        or (fact["region"] == VisualRegion.CONTENT.value and fact["activity"] == "active")
    )]
    for fact in facts:
        if (fact["region"] != VisualRegion.NAVIGATION.value
                or fact["selected_state"] != VisualSelectionState.SELECTED.value):
            continue
        item = fact["item"]
        assert isinstance(item, VisualElement)
        surface_relations = [surface["identity"] for surface in current_surfaces
                             if surface["identity"] is not ActivationEvidence.UNKNOWN]
        if not surface_relations:
            continue
        if fact["identity"] is ActivationEvidence.MATCH:
            if ActivationEvidence.MATCH in surface_relations:
                fact["authority"] = "authoritative"
                fact["active_relation"] = ActivationEvidence.MATCH
                fact["rejection"] = None
            elif ActivationEvidence.MISMATCH in surface_relations:
                fact["authority"] = "authoritative"
                fact["active_relation"] = ActivationEvidence.MISMATCH
                fact["rejection"] = "selected_row_conflicts_with_active_detail_identity"

    diagnostics: list[VisualPostconditionCandidateDiagnostic] = []
    for fact in facts:
        item = fact["item"]
        assert isinstance(item, VisualElement)
        active_relation = fact["active_relation"]
        if fact["authority"] == "authoritative" and isinstance(active_relation, ActivationEvidence):
            active_identity.append(active_relation)
        candidate_id = item.id if re.fullmatch(r"v\d{1,5}", item.id) else None
        diagnostics.append(VisualPostconditionCandidateDiagnostic(
            candidate_id=candidate_id,
            identity_text=clean_text(item.label)[:120],
            provider_role=clean_text(item.role)[:60],
            presentation_role=str(fact["presentation"]),
            region=str(fact["region"]),
            provider_region=clean_text(str(fact["provider_region"]))[:20],
            geometry_region_bucket=str(fact["geometry_region"]),
            activity_evidence=str(fact["activity"]),
            selected_state=str(fact["selected_state"]),
            identity_relation=fact["identity"],
            active_state_relation=active_relation,
            authority_class=str(fact["authority"]),
            rejection_reason=(clean_text(str(fact["rejection"]))[:80]
                              if fact["rejection"] else None),
        ))
    return (
        _merge_evidence(*visible_identity),
        _merge_evidence(*active_identity),
        tuple(diagnostics[:_MAX_EVIDENCE_ITEMS]),
    )


def evaluate_target_activation_postcondition(
    target: TargetSpec,
    observation: Observation,
    *,
    trusted_context_stable: bool,
    visual_verification_called: bool = False,
    visual_provider_attempts: tuple[VisualProviderAttempt, ...] = (),
    visual_grounding_objective: str | None = None,
    provider_incomplete: bool = False,
    clean_text: Callable[[str], str] = lambda value: value,
) -> TargetActivationPostcondition:
    """Combine typed active-entity evidence; visible identity alone never verifies."""

    uia_identity, uia_active, structural_evidence = _uia_evidence(
        target, observation, clean_text,
    )
    if visual_verification_called:
        visual_identity, visual_active, visual_diagnostics = _visual_evidence(
            target, observation, clean_text,
        )
    else:
        visual_identity, visual_active, visual_diagnostics = (
            ActivationEvidence.UNKNOWN, ActivationEvidence.UNKNOWN, (),
        )
    source = (
        ActivationEvidenceSource.COMBINED
        if visual_verification_called and (uia_identity is not ActivationEvidence.UNKNOWN
                                           or uia_active is not ActivationEvidence.UNKNOWN)
        and (visual_identity is not ActivationEvidence.UNKNOWN
             or visual_active is not ActivationEvidence.UNKNOWN)
        else ActivationEvidenceSource.VISUAL
        if visual_identity is not ActivationEvidence.UNKNOWN
        or visual_active is not ActivationEvidence.UNKNOWN
        else ActivationEvidenceSource.UIA
    )
    identity_evidence_value = _merge_evidence(uia_identity, visual_identity)
    active_evidence_value = _merge_evidence(uia_active, visual_active)

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
        structural_postcondition_evidence=structural_evidence,
        visual_grounding_objective=(
            clean_text(visual_grounding_objective)[:240]
            if visual_grounding_objective else None
        ),
        visual_candidate_count=len(observation.visual_elements) if visual_verification_called else 0,
        visual_candidate_diagnostics=visual_diagnostics,
    )
