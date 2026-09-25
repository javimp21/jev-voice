"""Pure deterministic target resolver tests; no provider or desktop access."""

from dataclasses import fields
from pathlib import Path

import pytest

from agent.target_evidence import adapt_observation_candidates
from computer.models import Observation, Rect, ScreenshotMetadata, UIElement, VisualElement
from decision.context import requested_target_spec
from decision.target_resolution import (
    CandidateEvidence,
    DomainSemanticCompatibility,
    IdentityEvidence,
    PresentationCompatibility,
    PresentationRole,
    RoleCompatibility,
    TargetResolutionStatus,
    TargetSpec,
    identity_evidence,
    normalize_presentation_role,
    presentation_role_compatibility,
    resolve_target,
    role_compatibility,
)
from safety.policy import GenericTargetActivationPolicy


SNAPSHOT = "snapshot-current"


def evidence(
    candidate_id: str,
    primary: str,
    *secondary: str,
    role: str | None = None,
    presentation: str | PresentationRole | None = None,
    target_semantics: str | None = None,
    source: str = "visual",
    actionable: bool = True,
    geometry: bool = True,
    safe: bool = True,
    snapshot: str = SNAPSHOT,
) -> CandidateEvidence:
    return CandidateEvidence(
        candidate_id, primary, tuple(secondary), role, actionable, geometry, safe,
        source, snapshot, role, target_semantics, presentation,
    )


def test_target_spec_validates_bounded_typed_fields() -> None:
    target = TargetSpec("Bluetooth", desired_role="navigation destination")
    assert target.domain_semantic_role == target.desired_role
    with pytest.raises(ValueError):
        TargetSpec(" ")
    with pytest.raises(ValueError):
        TargetSpec("target", qualifiers=["invalid tuple"])  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        TargetSpec("target", qualifiers=("x" * 201,))
    with pytest.raises(ValueError):
        TargetSpec("target", action_intent="\x00")


def test_candidate_evidence_validates_bounds_and_snapshot_binding() -> None:
    assert evidence("v1", "Target").snapshot_id == SNAPSHOT
    with pytest.raises(ValueError):
        evidence("", "Target")
    with pytest.raises(ValueError):
        CandidateEvidence("v1", "Target", ("x" * 201,), None, True, True, True, "visual", SNAPSHOT)
    with pytest.raises(ValueError):
        CandidateEvidence("v1", "Target", (), None, 1, True, True, "visual", SNAPSHOT)  # type: ignore[arg-type]


@pytest.mark.parametrize(("values", "expected"), [
    (("Red Hot Chili Peppers",), IdentityEvidence.MATCH),
    (("KZCO",), IdentityEvidence.MISMATCH),
    ((), IdentityEvidence.ABSENT),
])
def test_qualifier_evidence_distinguishes_match_mismatch_and_absence(values, expected) -> None:
    assert identity_evidence("Red Hot Chili Peppers", values) is expected


def test_primary_identity_match_mismatch_and_absence() -> None:
    target = TargetSpec("Pablo García")
    matching = resolve_target(
        target, (evidence("c1", "PABLO GARCIA · active"),), expected_snapshot_id=SNAPSHOT,
    )
    mismatching = resolve_target(
        target, (evidence("c1", "Pablo López"),), expected_snapshot_id=SNAPSHOT,
    )
    absent = resolve_target(
        target, (evidence("c1", ""),), expected_snapshot_id=SNAPSHOT,
    )
    assert matching.candidates[0].primary_identity is IdentityEvidence.MATCH
    assert mismatching.candidates[0].primary_identity is IdentityEvidence.MISMATCH
    assert absent.candidates[0].primary_identity is IdentityEvidence.ABSENT
    assert matching.status is TargetResolutionStatus.UNIQUE
    assert mismatching.status is absent.status is TargetResolutionStatus.NO_MATCH


@pytest.mark.parametrize(("desired", "candidate", "expected"), [
    ("file", "document result", RoleCompatibility.EXACT),
    ("conversation", "contact", RoleCompatibility.COMPATIBLE),
    ("file", "conversation", RoleCompatibility.INCOMPATIBLE),
    ("navigation destination", "custom widget", RoleCompatibility.UNKNOWN),
    (None, "video", RoleCompatibility.UNKNOWN),
    ("file", "folder", RoleCompatibility.CONTAINER),
])
def test_generic_role_compatibility(desired, candidate, expected) -> None:
    assert role_compatibility(desired, candidate) is expected


@pytest.mark.parametrize(("role", "expected"), [
    ("push button", PresentationRole.BUTTON),
    ("Hyperlink", PresentationRole.LINK),
    ("ListItem", PresentationRole.LIST_ITEM),
    ("table-row", PresentationRole.ROW),
    ("tile", PresentationRole.CARD),
    ("Tree Item", PresentationRole.TREE_ITEM),
    ("TabItem", PresentationRole.TAB),
    ("MenuItem", PresentationRole.MENU_ITEM),
    ("Edit", PresentationRole.TEXT_FIELD),
    ("search box", PresentationRole.SEARCH_FIELD),
    ("check box", PresentationRole.CHECKBOX),
    ("toggle button", PresentationRole.TOGGLE),
    ("Pane", PresentationRole.CONTAINER),
    ("static text", PresentationRole.LABEL),
    ("custom provider role", PresentationRole.UNKNOWN),
])
def test_provider_presentation_roles_normalize_to_bounded_taxonomy(role, expected) -> None:
    assert normalize_presentation_role(role) is expected


@pytest.mark.parametrize("role", [
    "button", "link", "list_item", "row", "card", "tree_item", "menu_item",
])
def test_activatable_presentation_roles_are_structurally_compatible(role: str) -> None:
    assert presentation_role_compatibility(
        "activate", role,
    ) is PresentationCompatibility.COMPATIBLE


@pytest.mark.parametrize("role", ["text_field", "search_field", "label", "container"])
def test_non_entity_presentation_roles_are_structurally_incompatible(role: str) -> None:
    assert presentation_role_compatibility(
        "activate", role,
    ) is PresentationCompatibility.INCOMPATIBLE


@pytest.mark.parametrize(("target_role", "presentation"), [
    ("conversation", "list_item"),
    ("file", "row"),
    ("media", "card"),
    ("setting", "button"),
    ("contact", "row"),
])
def test_unknown_domain_semantics_remain_admissible_across_generic_presentations(
    target_role: str, presentation: str,
) -> None:
    target = TargetSpec("Alex", desired_role=target_role)
    candidate = evidence(
        "candidate", "Alex", presentation=presentation,
    )

    result = resolve_target(target, (candidate,), expected_snapshot_id=SNAPSHOT)

    row = result.candidates[0]
    assert row.primary_identity is IdentityEvidence.MATCH
    assert row.target_semantic_evidence is None
    assert row.domain_semantic_compatibility is DomainSemanticCompatibility.UNKNOWN
    assert row.presentation_compatibility is PresentationCompatibility.COMPATIBLE
    assert row.admissible and result.status is TargetResolutionStatus.UNIQUE


@pytest.mark.parametrize("presentation", ["text_field", "search_field"])
def test_text_entry_presentations_do_not_satisfy_domain_target_activation(
    presentation: str,
) -> None:
    result = resolve_target(
        TargetSpec("report.pdf", desired_role="file"),
        (evidence("field", "report.pdf", presentation=presentation),),
        expected_snapshot_id=SNAPSHOT,
    )

    assert result.status is TargetResolutionStatus.NO_MATCH
    assert "incompatible_presentation_role" in result.candidates[0].rejection_reasons


@pytest.mark.parametrize("presentation", ["label", "container"])
def test_non_actionable_labels_and_containers_are_rejected(presentation: str) -> None:
    result = resolve_target(
        TargetSpec("Alex", desired_role="contact"),
        (evidence("passive", "Alex", presentation=presentation, actionable=False),),
        expected_snapshot_id=SNAPSHOT,
    )

    assert result.status is TargetResolutionStatus.NO_MATCH
    assert "not_actionable" in result.candidates[0].rejection_reasons


def test_actionable_label_is_still_not_a_domain_target() -> None:
    result = resolve_target(
        TargetSpec("Alex", desired_role="contact"),
        (evidence("label", "Alex", presentation="label"),),
        expected_snapshot_id=SNAPSHOT,
    )

    assert result.status is TargetResolutionStatus.NO_MATCH
    assert "incompatible_presentation_role" in result.candidates[0].rejection_reasons


def test_explicit_contradictory_domain_semantics_are_rejected() -> None:
    result = resolve_target(
        TargetSpec("Alex", desired_role="conversation"),
        (evidence(
            "file-result", "Alex", presentation="list item",
            target_semantics="file",
        ),),
        expected_snapshot_id=SNAPSHOT,
    )

    assert result.status is TargetResolutionStatus.NO_MATCH
    assert result.candidates[0].domain_semantic_compatibility is DomainSemanticCompatibility.CONTRADICTORY
    assert "contradictory_domain_semantics" in result.candidates[0].rejection_reasons


def test_explicit_conversation_like_semantics_are_retained_separately_from_presentation() -> None:
    result = resolve_target(
        TargetSpec("Alex", desired_role="conversation"),
        (evidence(
            "candidate", "Alex", presentation="list item",
            target_semantics="conversation-like",
        ),),
        expected_snapshot_id=SNAPSHOT,
    )

    row = result.candidates[0]
    assert result.status is TargetResolutionStatus.UNIQUE
    assert row.target_semantic_evidence == "conversation-like"
    assert row.domain_semantic_compatibility is DomainSemanticCompatibility.EXACT
    assert row.presentation_role is PresentationRole.LIST_ITEM
    assert row.admissible


def test_presentation_role_and_source_do_not_distinguish_duplicate_domain_targets() -> None:
    target = TargetSpec("Alex", desired_role="contact")
    result = resolve_target(
        target,
        (
            evidence("uia", "Alex", presentation="list item", source="UIA"),
            evidence("visual", "Alex", presentation="card", source="VISUAL"),
        ),
        expected_snapshot_id=SNAPSHOT, frontier_mode=True,
    )

    assert result.status is TargetResolutionStatus.AMBIGUOUS
    assert not result.evidence_distinguishable
    assert result.frontier_candidate_ids == ("uia", "visual")


@pytest.mark.parametrize("source", ["UIA", "VISUAL"])
def test_uia_and_visual_list_item_roles_share_domain_neutral_resolution(source: str) -> None:
    if source == "UIA":
        observation = Observation(
            "example.exe", "Example", (
                UIElement("c1", "Alex", "ListItem", enabled=True, visible=True),
            ), process_id=42, observation_id=SNAPSHOT, application_id="app_example",
        )
    else:
        metadata = ScreenshotMetadata(
            SNAPSHOT, 11, Rect(0, 0, 400, 300), Rect(0, 0, 400, 300),
            400, 300, 96, 96, 1.0, 1.0,
        )
        observation = Observation(
            "example.exe", "Example", (), process_id=42,
            observation_id=SNAPSHOT, application_id="app_example",
            visual_elements=(VisualElement(
                "v1", "Alex", "list item", Rect(10, 10, 120, 55), None, True,
            ),), screenshot=metadata,
        )

    adapted = adapt_observation_candidates(observation, GenericTargetActivationPolicy())
    result = resolve_target(
        TargetSpec("Alex", desired_role="contact"), adapted.candidates,
        expected_snapshot_id=SNAPSHOT,
    )

    row = result.candidates[0]
    assert result.status is TargetResolutionStatus.UNIQUE
    assert row.target_semantic_evidence is None
    assert row.presentation_role is PresentationRole.LIST_ITEM
    assert row.presentation_compatibility is PresentationCompatibility.COMPATIBLE


def test_uia_tab_presentation_does_not_fabricate_navigation_domain_semantics() -> None:
    observation = Observation(
        "example.exe", "Example", (
            UIElement("c1", "Bluetooth", "TabItem", enabled=True, visible=True),
        ), process_id=42, observation_id=SNAPSHOT, application_id="app_example",
    )
    adapted = adapt_observation_candidates(observation, GenericTargetActivationPolicy())
    result = resolve_target(
        TargetSpec("Bluetooth", desired_role="navigation destination"),
        adapted.candidates, expected_snapshot_id=SNAPSHOT,
    )

    row = result.candidates[0]
    assert result.status is TargetResolutionStatus.UNIQUE
    assert row.target_semantic_evidence is None
    assert row.presentation_role is PresentationRole.TAB
    assert row.domain_semantic_compatibility is DomainSemanticCompatibility.UNKNOWN


def test_unique_ambiguous_and_no_match_states() -> None:
    target = TargetSpec("invoice.pdf", desired_role="file")
    unique = resolve_target(
        target,
        (evidence("f1", "invoice.pdf", role="document"),),
        expected_snapshot_id=SNAPSHOT,
    )
    ambiguous = resolve_target(
        target,
        (evidence("f1", "invoice.pdf", role="file"),
         evidence("f2", "invoice.pdf", role="document")),
        expected_snapshot_id=SNAPSHOT,
    )
    no_match = resolve_target(
        target, (evidence("f3", "august.pdf", role="file"),),
        expected_snapshot_id=SNAPSHOT,
    )
    assert unique.status is TargetResolutionStatus.UNIQUE
    assert unique.selected_candidate_id == "f1"
    assert ambiguous.status is TargetResolutionStatus.AMBIGUOUS
    assert ambiguous.selected_candidate_id is None
    assert no_match.status is TargetResolutionStatus.NO_MATCH


def test_positive_qualifier_evidence_beats_missing_evidence() -> None:
    target = TargetSpec("Pablo García", ("Work account",), "conversation")
    result = resolve_target(
        target,
        (evidence("missing", "Pablo García", role="conversation"),
         evidence("positive", "Pablo García", "Work account", role="conversation")),
        expected_snapshot_id=SNAPSHOT,
    )
    assert result.status is TargetResolutionStatus.UNIQUE
    assert result.selected_candidate_id == "positive"
    assert result.admissible_candidate_ids == ("missing", "positive")
    assert result.candidates[0].qualifier_evidence == (IdentityEvidence.ABSENT,)
    assert result.candidates[1].qualifier_evidence == (IdentityEvidence.MATCH,)


def test_frontier_choice_requires_non_dominated_distinct_evidence_and_is_opt_in() -> None:
    target = TargetSpec("Pablo García", ("Work",), "conversation")
    candidates = (
        evidence("conversation", "Pablo García", role="conversation"),
        evidence("contact", "Pablo García", "Work contact", role="contact"),
        evidence("dominated", "Pablo García", role="contact"),
    )
    legacy = resolve_target(target, candidates, expected_snapshot_id=SNAPSHOT)
    assert legacy.status is TargetResolutionStatus.UNIQUE
    assert legacy.selected_candidate_id == "contact"

    frontier = resolve_target(
        target, candidates, expected_snapshot_id=SNAPSHOT, frontier_mode=True,
    )
    assert frontier.status is TargetResolutionStatus.CHOICE
    assert frontier.frontier_candidate_ids == ("conversation", "contact")
    assert frontier.evidence_distinguishable
    assert "dominated" not in frontier.frontier_candidate_ids


def test_frontier_does_not_treat_source_or_duplicate_order_as_distinguishing_evidence() -> None:
    target = TargetSpec("Pablo García", desired_role="conversation")
    duplicates = (
        evidence("uia", "Pablo García", role="conversation"),
        CandidateEvidence(
            "visual", "Pablo García", (), "conversation", True, True, True,
            "VISUAL", SNAPSHOT, "conversation",
        ),
    )
    result = resolve_target(
        target, duplicates, expected_snapshot_id=SNAPSHOT, frontier_mode=True,
    )
    assert result.status is TargetResolutionStatus.AMBIGUOUS
    assert not result.evidence_distinguishable


def test_provider_order_does_not_change_winner_and_position_is_not_evidence() -> None:
    target = TargetSpec("Bluetooth", desired_role="navigation destination")
    candidates = (
        evidence("other", "Personalization", role="navigation item"),
        evidence("target", "Bluetooth y dispositivos", role="navigation item"),
    )
    forward = resolve_target(target, candidates, expected_snapshot_id=SNAPSHOT)
    reverse = resolve_target(target, tuple(reversed(candidates)), expected_snapshot_id=SNAPSHOT)
    assert forward.status is reverse.status is TargetResolutionStatus.UNIQUE
    assert forward.selected_candidate_id == reverse.selected_candidate_id == "target"
    assert "rectangle" not in {item.name for item in fields(CandidateEvidence)}


def test_media_candidates_reject_explicit_qualifier_contradiction() -> None:
    target = TargetSpec("Californication", ("Red Hot Chili Peppers",), None, "activate")
    result = resolve_target(
        target,
        (evidence("correct", "Californication", "Red Hot Chili Peppers", role="video"),
         evidence("wrong-qualifier", "Californication", "KZCO", role="video"),
         evidence("artist", "Red Hot Chili Peppers", role="artist")),
        expected_snapshot_id=SNAPSHOT,
    )
    assert result.status is TargetResolutionStatus.UNIQUE
    assert result.selected_candidate_id == "correct"
    assert result.candidates[1].qualifier_evidence == (IdentityEvidence.MISMATCH,)
    assert result.candidates[2].primary_identity is IdentityEvidence.MISMATCH


def test_explicit_secondary_contradiction_overrides_conflicting_primary_metadata() -> None:
    target = TargetSpec("Californication", ("Red Hot Chili Peppers",))
    result = resolve_target(
        target,
        (evidence(
            "conflict", "Californication by Red Hot Chili Peppers", "KZCO", role="video",
        ),),
        expected_snapshot_id=SNAPSHOT,
    )
    assert result.status is TargetResolutionStatus.NO_MATCH
    assert result.candidates[0].qualifier_evidence == (IdentityEvidence.MISMATCH,)


def test_media_items_with_equal_identity_but_unknown_domain_roles_remain_ambiguous() -> None:
    target = TargetSpec("Californication", ("Red Hot Chili Peppers",), None, "activate")
    result = resolve_target(
        target,
        (evidence("video", "Californication", "Red Hot Chili Peppers", role="music video"),
         evidence("album", "Californication", "Red Hot Chili Peppers", role="album")),
        expected_snapshot_id=SNAPSHOT,
    )
    assert result.status is TargetResolutionStatus.AMBIGUOUS
    assert result.strongest_candidate_ids == ("video", "album")
    assert all(row.role_compatibility is RoleCompatibility.UNKNOWN for row in result.candidates)


def test_chat_candidates_use_identity_and_generic_role_evidence() -> None:
    target = TargetSpec("Pablo García", desired_role="conversation")
    result = resolve_target(
        target,
        (evidence("chat", "Pablo García", role="conversation"),
         evidence("other-person", "Pablo López", role="conversation"),
         evidence("profile", "Pablo García", role="profile action")),
        expected_snapshot_id=SNAPSHOT,
    )
    assert result.status is TargetResolutionStatus.UNIQUE
    assert result.selected_candidate_id == "chat"
    assert result.candidates[2].role_compatibility is RoleCompatibility.UNKNOWN


def test_file_candidates_reject_other_file_and_container() -> None:
    target = TargetSpec("factura septiembre.pdf", desired_role="file", action_intent="select")
    result = resolve_target(
        target,
        (evidence("right", "factura septiembre.pdf", role="file"),
         evidence("other-file", "factura agosto.pdf", role="file"),
         evidence("folder", "facturas", role="folder")),
        expected_snapshot_id=SNAPSHOT,
    )
    assert result.status is TargetResolutionStatus.UNIQUE
    assert result.selected_candidate_id == "right"
    assert result.candidates[2].role_compatibility is RoleCompatibility.INCOMPATIBLE
    assert "contradictory_domain_semantics" in result.candidates[2].rejection_reasons


def test_settings_candidates_preserve_generic_navigation_target() -> None:
    target = TargetSpec("Bluetooth", desired_role="navigation destination")
    result = resolve_target(
        target,
        (evidence("bluetooth", "Bluetooth y dispositivos", role="navigation item"),
         evidence("network", "Red e Internet", role="navigation item"),
         evidence("personalization", "Personalización", role="navigation item")),
        expected_snapshot_id=SNAPSHOT,
    )
    assert result.status is TargetResolutionStatus.UNIQUE
    assert result.selected_candidate_id == "bluetooth"
    assert result.candidates[0].role_compatibility is RoleCompatibility.UNKNOWN
    assert result.candidates[0].target_semantic_evidence is None


def test_resolver_rejects_invalid_unsafe_and_stale_candidates() -> None:
    target = TargetSpec("Target")
    candidates = (
        evidence("stale", "Target", snapshot="old"),
        evidence("unsafe", "Target", safe=False),
        evidence("geometry", "Target", geometry=False),
        evidence("not-actionable", "Target", actionable=False),
    )
    result = resolve_target(target, candidates, expected_snapshot_id=SNAPSHOT)
    assert result.status is TargetResolutionStatus.NO_MATCH
    assert {reason for row in result.candidates for reason in row.rejection_reasons} == {
        "stale_snapshot", "safety_rejected", "invalid_geometry", "not_actionable",
    }


@pytest.mark.parametrize(("request_text", "primary", "qualifiers", "role"), [
    ("Open Spotify and play Californication by Red Hot Chili Peppers",
     "Californication", ("Red Hot Chili Peppers",), None),
    ("Open WhatsApp and open the chat with Pablo García",
     "Pablo García", (), "conversation"),
    ("Open Explorer and find factura septiembre.pdf",
     "factura septiembre.pdf", (), "file"),
    ("Open factura septiembre.pdf", "factura septiembre.pdf", (), "file"),
    ("Open Settings and open Bluetooth",
     "Bluetooth", (), None),
    ("Open Chrome and select the GitHub tab",
     "GitHub", (), "navigation destination"),
])
def test_request_parser_extracts_generic_identity_without_app_fields(
    request_text, primary, qualifiers, role,
) -> None:
    target = requested_target_spec(request_text, experimental_generic=True)
    assert target is not None
    assert target.primary_identity == primary
    assert target.qualifiers == qualifiers
    assert target.desired_role == role
    expected_intent = "select" if any(
        phrase in request_text.casefold() for phrase in ("find ", "select ", "choose ", "locate ")
    ) else "activate"
    assert target.action_intent == expected_intent
    assert not hasattr(target, "creator")
    assert not hasattr(target, "content_type")


def test_generic_debug_parser_refinements_are_opt_in() -> None:
    legacy_chat = requested_target_spec(
        "Open WhatsApp and open the chat with Pablo García from Work",
    )
    generic_chat = requested_target_spec(
        "Open WhatsApp and open the chat with Pablo García from Work",
        experimental_generic=True,
    )
    legacy_file = requested_target_spec("Open factura septiembre.pdf")
    generic_file = requested_target_spec(
        "Open factura septiembre.pdf", experimental_generic=True,
    )

    assert legacy_chat is not None and legacy_chat.primary_identity == "Pablo García from Work"
    assert legacy_chat.qualifiers == ()
    assert generic_chat is not None and generic_chat.primary_identity == "Pablo García"
    assert generic_chat.qualifiers == ("Work",)
    assert legacy_file is not None and legacy_file.desired_role is None
    assert generic_file is not None and generic_file.desired_role == "file"


def test_generic_resolver_has_no_application_or_demo_domain_branches() -> None:
    source = (Path(__file__).parents[1] / "decision" / "target_resolution.py").read_text(
        encoding="utf-8",
    ).casefold()
    for forbidden in (
        "spotify", "whatsapp", "explorer", "chrome", "californication",
        "red hot chili peppers",
    ):
        assert forbidden not in source
