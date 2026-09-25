"""Local target-evidence sufficiency tests independent of Jev and Windows."""

from __future__ import annotations

from dataclasses import replace

import pytest

from agent.target_evidence import (
    TargetEvidenceSet, adapt_observation_candidates, bind_directed_grounding,
)
from agent.target_sufficiency import (
    TargetEvidenceSufficiency, TargetEvidenceSufficiencyReason,
    assess_target_evidence_sufficiency,
)
from computer.actions import ClickAction, VisualClickAction
from computer.models import Observation, Rect, ScreenshotMetadata, UIElement, VisualElement
from computer.visual import VisualGroundingRequest
from decision.target_resolution import (
    CandidateEvidence, DirectedGroundingEvidence, TargetResolutionStatus,
    TargetSpec, grounding_objective_fingerprint, resolve_target,
    target_spec_fingerprint,
)
from safety.policy import GenericTargetActivationPolicy


SNAPSHOT = "snapshot-current"
OBJECTIVE = 'Find visible actionable elements corresponding to "Alex" with semantic role "contact".'


def evidence(
    candidate_id: str = "c1", *, text: str = "Alex", secondary: tuple[str, ...] = (),
    role: str | None = "contact", source: str = "UIA", snapshot: str = SNAPSHOT,
    actionable: bool = True, geometry: bool = True, safe: bool = True,
    presentation: str = "list item", provenance: DirectedGroundingEvidence | None = None,
) -> CandidateEvidence:
    return CandidateEvidence(
        candidate_id, text, secondary, role, actionable, geometry, safe, source,
        snapshot, presentation_role=presentation, directed_grounding=provenance,
    )


def observation(*, visual: bool = False) -> Observation:
    metadata = ScreenshotMetadata(
        SNAPSHOT, 7, Rect(0, 0, 400, 300), Rect(0, 0, 400, 300),
        400, 300, 96, 96, 1.0, 1.0,
    ) if visual else None
    return Observation(
        "test.exe", "Test", process_id=42, observation_id=SNAPSHOT,
        screenshot=metadata,
        visual_directed_grounding=visual,
        visual_requested_max_elements=5 if visual else None,
    )


def sufficiency(
    candidates: tuple[CandidateEvidence, ...], *, target: TargetSpec | None = None,
    obs: Observation | None = None, conflicts: tuple[str, ...] = (),
    objective: str = OBJECTIVE, activation_required: bool = True,
):
    current_target = target or TargetSpec("Alex", desired_role="contact")
    current_observation = obs or observation(
        visual=any(candidate.source == "VISUAL" for candidate in candidates),
    )
    actions = {
        item.candidate_id: (
            VisualClickAction(item.snapshot_id, item.candidate_id)
            if item.source == "VISUAL" else ClickAction(item.candidate_id)
        )
        for item in candidates
    }
    evidence_set = TargetEvidenceSet(candidates, actions, conflicts)
    resolution = resolve_target(
        current_target, candidates, expected_snapshot_id=current_observation.observation_id,
        frontier_mode=True,
    )
    result = assess_target_evidence_sufficiency(
        current_target, resolution, evidence_set, current_observation,
        expected_grounding_objective=objective, activation_required=activation_required,
    )
    return result, resolution


def valid_visual_provenance(
    target: TargetSpec | None = None, *, candidate_id: str = "v1",
    snapshot: str = SNAPSHOT, previous: str = "snapshot-before",
    objective: str = OBJECTIVE, max_elements: int = 5,
) -> DirectedGroundingEvidence:
    current_target = target or TargetSpec("Alex", desired_role="contact")
    return DirectedGroundingEvidence(
        target_spec_fingerprint(current_target), grounding_objective_fingerprint(objective),
        snapshot, previous, candidate_id, max_elements,
    )


def test_one_strong_uia_candidate_is_sufficient() -> None:
    result, resolution = sufficiency((evidence(),))
    assert resolution.status is TargetResolutionStatus.UNIQUE
    assert len(resolution.admissible_candidate_ids) == 1
    assert result.status is TargetEvidenceSufficiency.SUFFICIENT
    assert result.reasons == ()


def test_exact_uia_identity_can_supply_generic_local_link_when_domain_role_unknown() -> None:
    result, _ = sufficiency((evidence(role=None),))
    assert result.status is TargetEvidenceSufficiency.SUFFICIENT


def test_one_fresh_bound_directed_visual_candidate_is_sufficient() -> None:
    target = TargetSpec("Alex", desired_role="contact")
    result, _ = sufficiency((evidence(
        "v1", role=None, source="VISUAL", provenance=valid_visual_provenance(target),
    ),), target=target, obs=observation(visual=True))
    assert result.status is TargetEvidenceSufficiency.SUFFICIENT


def test_unique_pareto_winner_with_multiple_admissible_candidates_is_insufficient() -> None:
    target = TargetSpec("Alex", qualifiers=("Work",), desired_role="contact")
    candidates = (
        evidence("c1", secondary=("Work contact",), role="contact"),
        evidence("c2", role=None),
    )
    result, resolution = sufficiency(candidates, target=target)
    assert resolution.status is TargetResolutionStatus.UNIQUE
    assert resolution.frontier_candidate_ids == ("c1",)
    assert len(resolution.admissible_candidate_ids) == 2
    assert result.status is TargetEvidenceSufficiency.INSUFFICIENT
    assert TargetEvidenceSufficiencyReason.ADMISSIBLE_COUNT_NOT_ONE in result.reasons


def test_choice_remains_insufficient() -> None:
    result, resolution = sufficiency((evidence("c1"), evidence("c2")))
    assert resolution.status is TargetResolutionStatus.AMBIGUOUS
    assert result.status is TargetEvidenceSufficiency.INSUFFICIENT
    assert TargetEvidenceSufficiencyReason.RESOLVER_NOT_UNIQUE in result.reasons


@pytest.mark.parametrize(("candidate", "target", "expected_reason"), [
    (evidence(secondary=()), TargetSpec("Alex", ("Work",), "contact"),
     TargetEvidenceSufficiencyReason.QUALIFIER_NOT_MATCH),
    (evidence(secondary=("Personal",)), TargetSpec("Alex", ("Work",), "contact"),
     TargetEvidenceSufficiencyReason.QUALIFIER_NOT_MATCH),
    (evidence(text=""), TargetSpec("Alex", desired_role="contact"),
     TargetEvidenceSufficiencyReason.IDENTITY_NOT_MATCH),
    (evidence(text="Other"), TargetSpec("Alex", desired_role="contact"),
     TargetEvidenceSufficiencyReason.IDENTITY_CONTRADICTION),
    (evidence(role="file"), TargetSpec("Alex", desired_role="contact"),
     TargetEvidenceSufficiencyReason.DOMAIN_SEMANTIC_CONTRADICTION),
    (evidence(actionable=False), TargetSpec("Alex", desired_role="contact"),
     TargetEvidenceSufficiencyReason.ADMISSIBLE_COUNT_NOT_ONE),
    (evidence(geometry=False), TargetSpec("Alex", desired_role="contact"),
     TargetEvidenceSufficiencyReason.ADMISSIBLE_COUNT_NOT_ONE),
    (evidence(safe=False), TargetSpec("Alex", desired_role="contact"),
     TargetEvidenceSufficiencyReason.ADMISSIBLE_COUNT_NOT_ONE),
    (evidence(snapshot="old-snapshot"), TargetSpec("Alex", desired_role="contact"),
     TargetEvidenceSufficiencyReason.ADMISSIBLE_COUNT_NOT_ONE),
    (evidence(presentation="custom role"), TargetSpec("Alex", desired_role="contact"),
     TargetEvidenceSufficiencyReason.PRESENTATION_NOT_COMPATIBLE),
])
def test_strict_invariants_fail_closed(
    candidate: CandidateEvidence, target: TargetSpec,
    expected_reason: TargetEvidenceSufficiencyReason,
) -> None:
    result, _ = sufficiency((candidate,), target=target)
    assert result.status is TargetEvidenceSufficiency.INSUFFICIENT
    assert expected_reason in result.reasons


def test_visual_unknown_domain_without_directed_provenance_is_insufficient() -> None:
    result, _ = sufficiency(
        (evidence("v1", role=None, source="VISUAL"),), obs=observation(visual=True),
    )
    assert result.status is TargetEvidenceSufficiency.INSUFFICIENT
    assert TargetEvidenceSufficiencyReason.DIRECTED_PROVENANCE_MISSING in result.reasons


@pytest.mark.parametrize("provenance", [
    valid_visual_provenance(target=TargetSpec("Other", desired_role="contact")),
    valid_visual_provenance(objective="a different directed objective"),
    valid_visual_provenance(snapshot="old-snapshot"),
    valid_visual_provenance(previous=SNAPSHOT),
    valid_visual_provenance(max_elements=4),
])
def test_visual_provenance_mismatch_is_insufficient(
    provenance: DirectedGroundingEvidence,
) -> None:
    result, _ = sufficiency(
        (evidence("v1", role=None, source="VISUAL", provenance=provenance),),
        obs=observation(visual=True),
    )
    assert result.status is TargetEvidenceSufficiency.INSUFFICIENT
    assert TargetEvidenceSufficiencyReason.DIRECTED_PROVENANCE_MISMATCH in result.reasons


def test_source_conflict_and_duplicate_identity_candidates_are_insufficient() -> None:
    conflict, _ = sufficiency((evidence(),), conflicts=("c1",))
    duplicates, _ = sufficiency((evidence("c1"), evidence("c2")))
    assert TargetEvidenceSufficiencyReason.SOURCE_CONFLICT in conflict.reasons
    assert TargetEvidenceSufficiencyReason.DUPLICATE_IDENTITY_EVIDENCE in duplicates.reasons


def test_adapter_reports_conflicting_uia_and_visual_evidence_for_same_id() -> None:
    metadata = ScreenshotMetadata(
        SNAPSHOT, 7, Rect(0, 0, 400, 300), Rect(0, 0, 400, 300),
        400, 300, 96, 96, 1.0, 1.0,
    )
    current = Observation(
        "test.exe", "Test", (
            UIElement("v1", "Alex", "ListItem", enabled=True, visible=True,
                      parent_name="Contacts"),
        ), process_id=42, observation_id=SNAPSHOT,
        visual_elements=(VisualElement(
            "v1", "Alex other", "list item", Rect(10, 20, 100, 70), None, True,
        ),), screenshot=metadata,
    )
    adapted = adapt_observation_candidates(current, GenericTargetActivationPolicy())
    result, _ = sufficiency(tuple(adapted.candidates), obs=current,
                            conflicts=adapted.conflicting_source_candidate_ids)
    assert adapted.conflicting_source_candidate_ids == ("v1",)
    assert TargetEvidenceSufficiencyReason.SOURCE_CONFLICT in result.reasons


def test_directed_binding_requires_exact_objective_and_fresh_bound_snapshot() -> None:
    target = TargetSpec("Alex", desired_role="contact")
    previous = replace(observation(), observation_id="snapshot-before")
    returned = observation(visual=True)
    request = VisualGroundingRequest(OBJECTIVE, 5)
    binding = bind_directed_grounding(
        target, request, previous, returned, expected_objective=OBJECTIVE,
    )
    assert binding is not None
    assert bind_directed_grounding(
        target, request, previous, returned, expected_objective="different objective",
    ) is None
    assert bind_directed_grounding(
        target, request, returned, returned, expected_objective=OBJECTIVE,
    ) is None


def test_unknown_domain_and_nonexact_uia_identity_are_insufficient() -> None:
    result, _ = sufficiency((evidence(text="Alex active", role=None),))
    assert result.status is TargetEvidenceSufficiency.INSUFFICIENT
    assert TargetEvidenceSufficiencyReason.DOMAIN_SEMANTIC_UNVERIFIED in result.reasons


def test_non_activation_intent_cannot_be_marked_sufficient() -> None:
    target = TargetSpec("Alex", desired_role="contact", action_intent="observe")
    result, _ = sufficiency((evidence(),), target=target, activation_required=False)
    assert result.status is TargetEvidenceSufficiency.INSUFFICIENT
    assert TargetEvidenceSufficiencyReason.ACTIVATION_NOT_REQUIRED in result.reasons
