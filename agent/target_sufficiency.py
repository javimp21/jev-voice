"""Strict local evidence gate, separate from target resolution and action safety."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from agent.target_evidence import TargetEvidenceSet
from computer.actions import ClickAction, VisualClickAction
from computer.models import Observation
from decision.target_resolution import (
    DomainSemanticCompatibility, IdentityEvidence, PresentationCompatibility,
    TargetResolution, TargetResolutionStatus, TargetSpec, exact_identity_match,
    grounding_objective_fingerprint, target_spec_fingerprint,
)


class TargetEvidenceSufficiency(StrEnum):
    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"


class TargetEvidenceSufficiencyReason(StrEnum):
    ACTIVATION_NOT_REQUIRED = "activation_not_required"
    RESOLVER_NOT_UNIQUE = "resolver_not_unique"
    FRONTIER_NOT_SINGLE = "frontier_not_single"
    ADMISSIBLE_COUNT_NOT_ONE = "admissible_count_not_one"
    IDENTITY_NOT_MATCH = "identity_not_match"
    IDENTITY_CONTRADICTION = "identity_contradiction"
    DUPLICATE_IDENTITY_EVIDENCE = "duplicate_identity_evidence"
    QUALIFIER_NOT_MATCH = "qualifier_not_match"
    NOT_ACTIONABLE = "not_actionable"
    INVALID_GEOMETRY = "invalid_geometry"
    NOT_SAFE = "not_safe"
    STALE_SNAPSHOT = "stale_snapshot"
    PRESENTATION_NOT_COMPATIBLE = "presentation_not_compatible"
    PRESENTATION_ROLE_INVALID = "presentation_role_invalid"
    DOMAIN_SEMANTIC_CONTRADICTION = "domain_semantic_contradiction"
    DOMAIN_SEMANTIC_UNVERIFIED = "domain_semantic_unverified"
    DIRECTED_PROVENANCE_MISSING = "directed_provenance_missing"
    DIRECTED_PROVENANCE_MISMATCH = "directed_provenance_mismatch"
    SOURCE_CONFLICT = "source_conflict"
    MATCHING_EVIDENCE_UNSAFE_OR_STALE = "matching_evidence_unsafe_or_stale"
    ACTION_NOT_BOUND = "action_not_bound"
    RESOLVER_SELECTION_MISMATCH = "resolver_selection_mismatch"


@dataclass(frozen=True, slots=True)
class TargetEvidenceSufficiencyResult:
    status: TargetEvidenceSufficiency
    reasons: tuple[TargetEvidenceSufficiencyReason, ...]


def assess_target_evidence_sufficiency(
    target: TargetSpec,
    resolution: TargetResolution,
    evidence_set: TargetEvidenceSet,
    observation: Observation,
    *,
    expected_grounding_objective: str,
    activation_required: bool,
) -> TargetEvidenceSufficiencyResult:
    """Allow deterministic selection only when exactly one fully supported target remains."""
    reasons: list[TargetEvidenceSufficiencyReason] = []

    def add(reason: TargetEvidenceSufficiencyReason) -> None:
        if reason not in reasons:
            reasons.append(reason)

    if not activation_required or target.action_intent.casefold() not in {"activate", "select"}:
        add(TargetEvidenceSufficiencyReason.ACTIVATION_NOT_REQUIRED)
    if resolution.status is not TargetResolutionStatus.UNIQUE:
        add(TargetEvidenceSufficiencyReason.RESOLVER_NOT_UNIQUE)
    if len(resolution.frontier_candidate_ids) != 1:
        add(TargetEvidenceSufficiencyReason.FRONTIER_NOT_SINGLE)
    if len(resolution.admissible_candidate_ids) != 1:
        add(TargetEvidenceSufficiencyReason.ADMISSIBLE_COUNT_NOT_ONE)
    if evidence_set.conflicting_source_candidate_ids:
        add(TargetEvidenceSufficiencyReason.SOURCE_CONFLICT)

    matching_rows = tuple(
        row for row in resolution.candidates
        if row.primary_identity is IdentityEvidence.MATCH
    )
    if len(matching_rows) > 1:
        add(TargetEvidenceSufficiencyReason.DUPLICATE_IDENTITY_EVIDENCE)
    if any(
        row.primary_identity is IdentityEvidence.MATCH
        and (not row.snapshot_valid or not row.actionable or not row.geometry_valid
             or not row.safety_eligible)
        for row in resolution.candidates
    ):
        add(TargetEvidenceSufficiencyReason.MATCHING_EVIDENCE_UNSAFE_OR_STALE)
    if any(
        row.primary_identity is IdentityEvidence.MATCH
        and any(value is not IdentityEvidence.MATCH for value in row.qualifier_evidence)
        for row in resolution.candidates
    ):
        add(TargetEvidenceSufficiencyReason.QUALIFIER_NOT_MATCH)
    if any(
        row.primary_identity is IdentityEvidence.MATCH
        and row.domain_semantic_compatibility is DomainSemanticCompatibility.CONTRADICTORY
        for row in resolution.candidates
    ):
        add(TargetEvidenceSufficiencyReason.DOMAIN_SEMANTIC_CONTRADICTION)
    if not resolution.admissible_candidate_ids:
        if any(row.primary_identity is IdentityEvidence.MISMATCH for row in resolution.candidates):
            add(TargetEvidenceSufficiencyReason.IDENTITY_CONTRADICTION)
        elif any(row.primary_identity is IdentityEvidence.ABSENT for row in resolution.candidates):
            add(TargetEvidenceSufficiencyReason.IDENTITY_NOT_MATCH)

    candidate_id = (
        resolution.admissible_candidate_ids[0]
        if len(resolution.admissible_candidate_ids) == 1 else None
    )
    row = next((item for item in resolution.candidates
                if item.candidate_id == candidate_id), None)
    candidate = next((item for item in evidence_set.candidates
                      if item.candidate_id == candidate_id), None)
    action = evidence_set.actions.get(candidate_id) if candidate_id is not None else None
    if row is None or candidate is None:
        add(TargetEvidenceSufficiencyReason.ACTION_NOT_BOUND)
    else:
        if resolution.selected_candidate_id != candidate_id:
            add(TargetEvidenceSufficiencyReason.RESOLVER_SELECTION_MISMATCH)
        if not row.admissible:
            add(TargetEvidenceSufficiencyReason.ACTION_NOT_BOUND)
        if row.primary_identity is not IdentityEvidence.MATCH:
            add(TargetEvidenceSufficiencyReason.IDENTITY_NOT_MATCH)
        if IdentityEvidence.MISMATCH in row.qualifier_evidence or any(
            value is not IdentityEvidence.MATCH for value in row.qualifier_evidence
        ):
            add(TargetEvidenceSufficiencyReason.QUALIFIER_NOT_MATCH)
        if not row.actionable:
            add(TargetEvidenceSufficiencyReason.NOT_ACTIONABLE)
        if not row.geometry_valid:
            add(TargetEvidenceSufficiencyReason.INVALID_GEOMETRY)
        if not row.safety_eligible:
            add(TargetEvidenceSufficiencyReason.NOT_SAFE)
        if not row.snapshot_valid or row.snapshot_id != observation.observation_id:
            add(TargetEvidenceSufficiencyReason.STALE_SNAPSHOT)
        if row.presentation_compatibility is not PresentationCompatibility.COMPATIBLE:
            add(TargetEvidenceSufficiencyReason.PRESENTATION_NOT_COMPATIBLE)
        if row.presentation_role.value == "unknown":
            add(TargetEvidenceSufficiencyReason.PRESENTATION_ROLE_INVALID)
        action_is_bound = (
            isinstance(action, ClickAction) and row.source.upper() == "UIA"
            and action.target_id == row.candidate_id
        ) or (
            isinstance(action, VisualClickAction) and row.source.upper() == "VISUAL"
            and action.target_id == row.candidate_id
            and action.snapshot_id == observation.observation_id
        )
        if not action_is_bound:
            add(TargetEvidenceSufficiencyReason.ACTION_NOT_BOUND)
        if row.domain_semantic_compatibility is DomainSemanticCompatibility.CONTRADICTORY:
            add(TargetEvidenceSufficiencyReason.DOMAIN_SEMANTIC_CONTRADICTION)

        domain_is_known_compatible = row.domain_semantic_compatibility in {
            DomainSemanticCompatibility.EXACT,
            DomainSemanticCompatibility.COMPATIBLE,
        }
        if row.source.upper() == "VISUAL":
            provenance = candidate.directed_grounding
            provenance_valid = bool(
                provenance is not None
                and provenance.target_spec_fingerprint == target_spec_fingerprint(target)
                and provenance.objective_fingerprint == grounding_objective_fingerprint(
                    expected_grounding_objective,
                )
                and provenance.snapshot_id == observation.observation_id == row.snapshot_id
                and provenance.previous_snapshot_id
                and provenance.previous_snapshot_id != observation.observation_id
                and provenance.candidate_id == row.candidate_id == candidate.candidate_id
                and provenance.max_elements == observation.visual_requested_max_elements
                and observation.visual_directed_grounding is True
                and observation.screenshot is not None
                and observation.screenshot.snapshot_id == observation.observation_id
            )
            if not provenance_valid:
                add(
                    TargetEvidenceSufficiencyReason.DIRECTED_PROVENANCE_MISMATCH
                    if provenance is not None
                    else TargetEvidenceSufficiencyReason.DIRECTED_PROVENANCE_MISSING
                )
        elif not domain_is_known_compatible and not exact_identity_match(
            candidate.primary_text, target.primary_identity,
        ):
            # UIA exact identity plus a current actionable UIA presentation is
            # locally verifiable generic evidence when the domain role is unknown.
            add(TargetEvidenceSufficiencyReason.DOMAIN_SEMANTIC_UNVERIFIED)

    if len(reasons) > 12:
        reasons = reasons[:12]
    return TargetEvidenceSufficiencyResult(
        TargetEvidenceSufficiency.INSUFFICIENT if reasons
        else TargetEvidenceSufficiency.SUFFICIENT,
        tuple(reasons),
    )
