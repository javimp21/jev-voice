"""Generic application discovery, matching, launch, and safety tests."""

from pathlib import Path
from unittest.mock import Mock

from computer.actions import ClickAction, OpenAppAction, PressKeyAction
from computer.applications import (
    ApplicationCandidate, ApplicationIdentityEvidence, MemoryApplicationCatalog,
    compare_application_identity_evidence, diagnose_application_matches,
    normalize_app_identity, normalize_app_name,
)
from computer.models import Observation, UIElement
from computer.windows_apps import PackagedMetadata, ShortcutMetadata, WindowsApplicationCatalog
from decision.jev import JevDecisionMaker
from safety.policy import AutonomousActionPolicy, BasicActionPolicy


def candidate(
    app_id: str,
    name: str,
    *,
    policy: str = "allow",
    process: str = "",
    source: str = "test",
    identity_fingerprint: str | None = None,
    launch_target_type: str = "other",
) -> ApplicationCandidate:
    return ApplicationCandidate(
        app_id, name, source, launch_policy=policy,
        process_names=(process,) if process else (),
        identity_fingerprint=identity_fingerprint,
        launch_target_type=launch_target_type,
    )


def test_application_name_normalization() -> None:
    assert normalize_app_name("  Spótify—Music™  ") == "spótify music"
    assert normalize_app_name("GOOGLE.Chrome") == "google chrome"
    assert normalize_app_identity("Notepad++") == "notepad++"
    assert normalize_app_identity("Notepad") == "notepad"
    assert normalize_app_identity("Notepad++") != normalize_app_identity("Notepad")
    assert normalize_app_identity("C#") != normalize_app_identity("C")
    assert normalize_app_identity("Node.js") != normalize_app_identity("Node js")


def test_exact_partial_ambiguous_and_no_match() -> None:
    catalog = MemoryApplicationCatalog((
        candidate("app_1111111111111111", "Spotify"),
        candidate("app_2222222222222222", "Spotify Music"),
        candidate("app_3333333333333333", "Google Chrome"),
    ))
    exact = catalog.find("Spotify")
    assert exact[0].candidate.display_name == "Spotify" and exact[0].score == 100
    assert [match.candidate.display_name for match in catalog.find("Open Chrome and go to wikipedia.org")] == ["Google Chrome"]
    ambiguous = catalog.find("Open Spotify Music")
    assert [(match.candidate.display_name, match.match_kind) for match in ambiguous] == [
        ("Spotify Music", "raw_exact"),
    ]
    assert catalog.find("Open an unrelated nonexistent program") == ()


def test_notepad_plus_plus_exact_match_suppresses_the_looser_notepad_match() -> None:
    notepad = candidate("app_notepad", "Notepad")
    notepad_plus = candidate("app_notepad_plus", "Notepad++")
    catalog = MemoryApplicationCatalog((notepad, notepad_plus))

    matches = catalog.find("Open Notepad++")

    assert [(match.candidate.display_name, match.match_kind) for match in matches] == [
        ("Notepad++", "raw_exact"),
    ]


def test_same_trusted_identity_representations_collapse_to_one_logical_candidate() -> None:
    first = candidate(
        "app_aaaaaaaaaaaaaaaa", "Notepad++", source="start_menu",
        identity_fingerprint="shortcut:trusted-hash", launch_target_type="shortcut",
    )
    second = candidate(
        "app_bbbbbbbbbbbbbbbb", "Notepad++", source="packaged",
        identity_fingerprint="shortcut:trusted-hash", launch_target_type="shortcut",
    )
    first_launcher = Mock()
    second_launcher = Mock()
    catalog = MemoryApplicationCatalog(
        (first, second), {first.id: first_launcher, second.id: second_launcher},
    )

    matches = catalog.find("Open Notepad++")

    assert len(matches) == 1
    assert matches[0].candidate.id.startswith("app_")
    assert matches[0].match_kind == "raw_exact"
    assert {item.id for item in matches[0].equivalent_candidates} == {first.id, second.id}
    assert matches[0].candidate.source == "equivalent"
    assert BasicActionPolicy(catalog).validate(OpenAppAction(matches[0].candidate.id), None).disposition == "allow"
    catalog.launch(matches[0].candidate.id)
    assert first_launcher.call_count + second_launcher.call_count == 1


def test_same_display_name_with_different_trusted_identities_remains_ambiguous() -> None:
    catalog = MemoryApplicationCatalog((
        candidate("app_aaaaaaaaaaaaaaaa", "Notepad++", source="start_menu",
                  identity_fingerprint="exe:one", launch_target_type="shortcut"),
        candidate("app_bbbbbbbbbbbbbbbb", "Notepad++", source="packaged",
                  identity_fingerprint="packaged:two", launch_target_type="packaged"),
    ))

    matches = catalog.find("Open Notepad++")

    assert len(matches) == 2
    assert all(match.match_kind == "ambiguous" for match in matches)


def test_no_source_priority_is_used_for_genuinely_different_app_identities() -> None:
    catalog = MemoryApplicationCatalog((
        candidate("app_aaaaaaaaaaaaaaaa", "Example App", source="start_menu",
                  identity_fingerprint="shortcut:one", launch_target_type="shortcut"),
        candidate("app_bbbbbbbbbbbbbbbb", "Example App", source="packaged",
                  identity_fingerprint="packaged:two", launch_target_type="packaged"),
    ))

    matches = catalog.find("Open Example App")

    assert {match.candidate.source for match in matches} == {"start_menu", "packaged"}
    assert all(match.match_kind == "ambiguous" for match in matches)


def test_same_app_user_model_id_proves_equivalence() -> None:
    result = compare_application_identity_evidence(
        ApplicationIdentityEvidence(
            app_user_model_id_fingerprint="aumid:same",
            package_identity_fingerprint="package:same",
        ),
        ApplicationIdentityEvidence(
            app_user_model_id_fingerprint="aumid:same",
            package_identity_fingerprint="package:same",
        ),
    )

    assert result.equivalence_result == "PROVEN_EQUIVALENT"
    assert result.equivalence_signal_kind == "same_app_user_model_id"
    assert result.compared_identity_fields_present.app_user_model_id is True
    assert result.compared_identity_fields_present.package_identity is True


def test_same_trusted_executable_identity_proves_equivalence_when_package_ids_do_not_conflict() -> None:
    result = compare_application_identity_evidence(
        ApplicationIdentityEvidence(executable_identity_fingerprint="exe:same"),
        ApplicationIdentityEvidence(executable_identity_fingerprint="exe:same"),
    )

    assert result.equivalence_result == "PROVEN_EQUIVALENT"
    assert result.equivalence_signal_kind == "same_trusted_executable_identity"


def test_different_package_identity_proves_distinct_even_with_same_executable() -> None:
    result = compare_application_identity_evidence(
        ApplicationIdentityEvidence(
            package_identity_fingerprint="package:first",
            executable_identity_fingerprint="exe:shared",
        ),
        ApplicationIdentityEvidence(
            package_identity_fingerprint="package:second",
            executable_identity_fingerprint="exe:shared",
        ),
    )

    assert result.equivalence_result == "PROVEN_DISTINCT"
    assert result.equivalence_signal_kind == "different_package_identity"


def test_insufficient_cross_source_identity_evidence_is_unknown() -> None:
    result = compare_application_identity_evidence(
        ApplicationIdentityEvidence(catalog_identity_fingerprint="shortcut:one"),
        ApplicationIdentityEvidence(catalog_identity_fingerprint="packaged:two"),
    )

    assert result.equivalence_result == "UNKNOWN"
    assert result.equivalence_signal_kind == "insufficient_identity_evidence"


def test_same_aumid_with_conflicting_package_identity_remains_unknown() -> None:
    result = compare_application_identity_evidence(
        ApplicationIdentityEvidence(
            app_user_model_id_fingerprint="aumid:same",
            package_identity_fingerprint="package:first",
        ),
        ApplicationIdentityEvidence(
            app_user_model_id_fingerprint="aumid:same",
            package_identity_fingerprint="package:second",
        ),
    )

    assert result.equivalence_result == "UNKNOWN"
    assert result.equivalence_signal_kind == "identity_metadata_conflict"


def test_notepad_and_case_insensitive_raw_exact_still_select_the_named_app() -> None:
    notepad = candidate("app_notepad", "Notepad")
    notepad_plus = candidate("app_notepad_plus", "Notepad++")
    catalog = MemoryApplicationCatalog((notepad, notepad_plus))

    plain = catalog.find("Open Notepad")
    casefolded = catalog.find("oPeN nOtEpAd++")

    assert [(match.candidate.display_name, match.match_kind) for match in plain] == [
        ("Notepad", "raw_exact"),
    ]
    assert [(match.candidate.display_name, match.match_kind) for match in casefolded] == [
        ("Notepad++", "raw_exact"),
    ]


def test_canonical_exact_and_loose_normalized_exact_are_distinct_stages() -> None:
    catalog = MemoryApplicationCatalog((
        candidate("app_notepad", "Notepad™"),
        candidate("app_google_chrome", "Google Chrome"),
    ))

    canonical = catalog.find("Open Notepad")
    normalized = catalog.find("Open Google.Chrome")

    assert [(match.candidate.display_name, match.match_kind) for match in canonical] == [
        ("Notepad™", "canonical_exact"),
    ]
    assert [(match.candidate.display_name, match.match_kind) for match in normalized] == [
        ("Google Chrome", "normalized_exact"),
    ]


def test_fuzzy_match_kind_is_used_only_after_exact_stages() -> None:
    catalog = MemoryApplicationCatalog((candidate("app_google_chrome", "Google Chrome"),))

    matches = catalog.find("Open Chrome and go to wikipedia.org")

    assert [(match.candidate.display_name, match.match_kind) for match in matches] == [
        ("Google Chrome", "unique_fuzzy"),
    ]


def test_ambiguous_loose_application_match_fails_closed_before_jev() -> None:
    client = Mock(evaluate=Mock(side_effect=AssertionError("ambiguous app must not reach Jev")))
    catalog = MemoryApplicationCatalog((
        candidate("app_notepad", "Notepad"),
        candidate("app_notepad_plus", "Notepad++"),
    ))

    result = JevDecisionMaker(client, app_catalog=catalog).decide(
        "Open Notepad+", Observation("explorer.exe", "Desktop"),
    )

    assert result.status == "needs_human"
    assert result.diagnostic == "ambiguous_application_match"
    assert result.provider_called is False
    client.evaluate.assert_not_called()


def test_genuine_duplicate_name_ambiguity_still_does_not_call_jev() -> None:
    client = Mock(evaluate=Mock(side_effect=AssertionError("ambiguity must not reach Jev")))
    catalog = MemoryApplicationCatalog((
        candidate("app_aaaaaaaaaaaaaaaa", "Notepad++", identity_fingerprint="exe:one"),
        candidate("app_bbbbbbbbbbbbbbbb", "Notepad++", identity_fingerprint="exe:two"),
    ))

    result = JevDecisionMaker(client, app_catalog=catalog).decide(
        "Open Notepad++", Observation("explorer.exe", "Desktop"),
    )

    assert result.status == "needs_human"
    assert result.diagnostic == "ambiguous_application_match"
    assert result.provider_called is False
    client.evaluate.assert_not_called()


def test_resolved_app_still_uses_existing_trusted_safety_and_launcher() -> None:
    launched = Mock()
    app = candidate("app_notepad_plus", "Notepad++")
    catalog = MemoryApplicationCatalog((
        candidate("app_notepad", "Notepad"), app,
    ), {app.id: launched})

    match = catalog.find("Open Notepad++")[0]
    action = OpenAppAction(match.candidate.id)

    assert BasicActionPolicy(catalog).validate(action, None).disposition == "allow"
    catalog.launch(match.candidate.id)
    launched.assert_called_once_with()


def test_candidate_budget_is_enforced() -> None:
    apps = tuple(candidate(f"app_{index:016x}", f"Editor {index}") for index in range(20))
    catalog = MemoryApplicationCatalog(apps)
    assert len(catalog.find("Open Editor", limit=3)) == 3


def test_trusted_id_maps_to_private_launcher_and_unknown_strings_fail() -> None:
    launch = Mock()
    app_id = "app_1111111111111111"
    catalog = MemoryApplicationCatalog((candidate(app_id, "Spotify", process="spotify.exe"),), {app_id: launch})
    catalog.launch(app_id)
    launch.assert_called_once_with()
    for injected in ("C:\\Spotify.exe", "spotify --argument", "cmd /c spotify", "https://spotify.com"):
        assert BasicActionPolicy(catalog).validate(OpenAppAction(injected), None).disposition == "deny"


def test_start_menu_and_packaged_discovery_use_opaque_ids(tmp_path: Path) -> None:
    spotify = tmp_path / "Spotify.lnk"
    terminal = tmp_path / "Windows Terminal.lnk"
    spotify.touch()
    terminal.touch()
    launched = Mock()

    def read_shortcut(path: Path) -> ShortcutMetadata:
        return ShortcutMetadata("C:\\Apps\\Spotify.exe" if path == spotify else "C:\\Windows\\System32\\wt.exe")

    packaged_launch = Mock()
    catalog = WindowsApplicationCatalog(
        start_menu_roots=(tmp_path,), shortcut_reader=read_shortcut,
        shortcut_launcher=launched,
        packaged_reader=lambda: (PackagedMetadata(
            "WhatsApp", "family_123!App", package_family="family_123", launcher=packaged_launch,
        ),),
    )
    apps = catalog.discover()
    by_name = {app.display_name: app for app in apps}

    assert set(by_name) == {"Spotify", "Windows Terminal", "WhatsApp"}
    assert all(app.id.startswith("app_") and "\\" not in app.id for app in apps)
    assert by_name["Windows Terminal"].launch_policy == "deny"
    assert by_name["Spotify"].launch_policy == "allow"
    assert by_name["WhatsApp"].source == "packaged"
    assert catalog.find("Open Spotify")[0].candidate.display_name == "Spotify"
    assert catalog.find("Open WhatsApp")[0].candidate.display_name == "WhatsApp"
    catalog.launch(by_name["Spotify"].id)
    launched.assert_called_once_with(spotify.resolve())


def test_duplicate_notepad_plus_plus_shortcuts_with_same_target_form_one_logical_app(
    tmp_path: Path,
) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first_shortcut = first_dir / "Notepad++.lnk"
    second_shortcut = second_dir / "Notepad++.lnk"
    first_shortcut.touch()
    second_shortcut.touch()
    launched = Mock()
    catalog = WindowsApplicationCatalog(
        start_menu_roots=(tmp_path,),
        shortcut_reader=lambda _path: ShortcutMetadata(
            r"C:\Program Files\Notepad++\notepad++.exe", arguments="--single-instance",
        ),
        shortcut_launcher=launched,
        packaged_reader=lambda: (),
    )

    physical = catalog.discover()
    matches = catalog.find("Open Notepad++")
    metadata = diagnose_application_matches(matches)

    assert len(physical) == 2
    assert len(matches) == 1
    assert matches[0].match_kind == "raw_exact"
    assert matches[0].candidate.id != physical[0].id
    assert catalog.resolve(matches[0].candidate.id) is not None
    assert catalog.identify("notepad++.exe") == matches[0].candidate.id
    assert len(metadata) == 2
    assert {item.canonical_name for item in metadata} == {"notepad++"}
    assert all(item.identity_fingerprint_present for item in metadata)
    assert all(item.identity_equivalence_group == "identity_group_1" for item in metadata)
    assert all(item.launch_target_type == "shortcut" for item in metadata)
    assert all(item.duplicate_of_another_candidate for item in metadata)
    assert all(item.app_id.startswith("app_") for item in metadata)
    assert BasicActionPolicy(catalog).validate(OpenAppAction(matches[0].candidate.id), None).disposition == "allow"
    catalog.launch(matches[0].candidate.id)
    launched.assert_called_once()


def test_same_name_notepad_plus_plus_shortcuts_to_different_executables_stay_ambiguous(
    tmp_path: Path,
) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    (first_dir / "Notepad++.lnk").touch()
    (second_dir / "Notepad++.lnk").touch()
    targets = {
        first_dir / "Notepad++.lnk": r"C:\Program Files\Notepad++\notepad++.exe",
        second_dir / "Notepad++.lnk": r"D:\Portable\Notepad++\notepad++.exe",
    }
    catalog = WindowsApplicationCatalog(
        start_menu_roots=(tmp_path,),
        shortcut_reader=lambda path: ShortcutMetadata(targets[path]),
        shortcut_launcher=Mock(), packaged_reader=lambda: (),
    )

    matches = catalog.find("Open Notepad++")
    metadata = diagnose_application_matches(matches)

    assert len(matches) == 2
    assert all(match.match_kind == "ambiguous" for match in matches)
    assert len(metadata) == 2
    assert all(item.identity_fingerprint_present for item in metadata)
    assert all(item.identity_equivalence_group is None for item in metadata)
    assert all(not item.duplicate_of_another_candidate for item in metadata)


def test_start_menu_and_packaged_entries_with_same_aumid_are_deduplicated(tmp_path: Path) -> None:
    shortcut = tmp_path / "Notepad++.lnk"
    shortcut.touch()
    packaged_launch = Mock()
    private_aumid = "private.family.identity!NotepadPlusPlus"
    private_target = r"C:\Users\private\SECRET\NotepadPlusPlus.exe"
    private_arguments = "--token private-arguments-sentinel"
    catalog = WindowsApplicationCatalog(
        start_menu_roots=(tmp_path,),
        shortcut_reader=lambda _path: ShortcutMetadata(
            private_target, private_arguments, app_user_model_id=private_aumid,
        ),
        shortcut_launcher=Mock(),
        packaged_reader=lambda: (PackagedMetadata(
            "Notepad++", private_aumid, package_family="private.family.identity",
            launcher=packaged_launch,
        ),),
    )

    matches = catalog.find("Open Notepad++")
    diagnostic = catalog.compare_identities(
        next(item.id for item in catalog.discover() if item.source == "start_menu"),
        next(item.id for item in catalog.discover() if item.source == "packaged"),
    )

    assert len(matches) == 1
    assert matches[0].match_kind == "raw_exact"
    assert diagnostic.comparison_attempted is True
    assert diagnostic.equivalence_result == "PROVEN_EQUIVALENT"
    assert diagnostic.equivalence_signal_kind == "same_app_user_model_id"
    assert diagnostic.compared_identity_fields_present.app_user_model_id is True
    from dataclasses import asdict
    import json
    safe_json = json.dumps(asdict(diagnostic))
    for forbidden in (private_aumid, private_target, private_arguments, "private-arguments-sentinel"):
        assert forbidden not in safe_json


def test_same_target_with_insufficient_package_association_stays_unknown_and_ambiguous(
    tmp_path: Path,
) -> None:
    shortcut = tmp_path / "Notepad++.lnk"
    shortcut.touch()
    catalog = WindowsApplicationCatalog(
        start_menu_roots=(tmp_path,),
        shortcut_reader=lambda _path: ShortcutMetadata(r"C:\Apps\NotepadPlusPlus.exe"),
        shortcut_launcher=Mock(),
        packaged_reader=lambda: (PackagedMetadata(
            "Notepad++", "family.package!NotepadPlusPlus", package_family="family.package",
            launcher=Mock(),
        ),),
    )

    start_menu_id = next(item.id for item in catalog.discover() if item.source == "start_menu")
    packaged_id = next(item.id for item in catalog.discover() if item.source == "packaged")
    diagnostic = catalog.compare_identities(start_menu_id, packaged_id)

    assert diagnostic.equivalence_result == "UNKNOWN"
    assert diagnostic.equivalence_signal_kind == "insufficient_identity_evidence"
    assert len(catalog.find("Open Notepad++")) == 2
    assert all(match.match_kind == "ambiguous" for match in catalog.find("Open Notepad++"))


def test_transitive_equivalence_does_not_collapse_unknown_pair() -> None:
    candidates = [
        candidate("candidate-a", "Example", source="start_menu"),
        candidate("candidate-b", "Example", source="packaged"),
        candidate("candidate-c", "Example", source="start_menu"),
    ]
    catalog = WindowsApplicationCatalog(start_menu_roots=(), packaged_reader=lambda: ())
    catalog._identity_evidence_by_id.update({
        "candidate-a": ApplicationIdentityEvidence(
            app_user_model_id_fingerprint="aumid:shared",
            package_identity_fingerprint="package:first",
        ),
        "candidate-b": ApplicationIdentityEvidence(
            app_user_model_id_fingerprint="aumid:shared",
            package_identity_fingerprint="package:second",
        ),
        "candidate-c": ApplicationIdentityEvidence(
            app_user_model_id_fingerprint="aumid:shared",
        ),
    })

    grouped = catalog._apply_proven_equivalence(candidates)

    # A-C and B-C individually share an AUMID, but A-B has contradictory
    # package metadata. An indirect path is not enough to prove the full group.
    assert [item.identity_fingerprint for item in grouped] == [None, None, None]


def test_restricted_system_utility_cannot_launch(tmp_path: Path) -> None:
    shortcut = tmp_path / "Friendly Tool.lnk"
    shortcut.touch()
    launch = Mock()
    catalog = WindowsApplicationCatalog(
        start_menu_roots=(tmp_path,),
        shortcut_reader=lambda _path: ShortcutMetadata("C:\\Windows\\System32\\powershell.exe"),
        shortcut_launcher=launch, packaged_reader=lambda: (),
    )
    restricted = catalog.discover()[0]
    assert restricted.launch_policy == "deny"
    assert BasicActionPolicy(catalog).validate(OpenAppAction(restricted.id), None).disposition == "deny"
    launch.assert_not_called()


def test_all_management_console_shortcuts_are_restricted(tmp_path: Path) -> None:
    shortcut = tmp_path / "Event Viewer.lnk"
    shortcut.touch()
    catalog = WindowsApplicationCatalog(
        start_menu_roots=(tmp_path,),
        shortcut_reader=lambda _path: ShortcutMetadata(r"C:\Windows\System32\eventvwr.msc"),
        shortcut_launcher=Mock(), packaged_reader=lambda: (),
    )
    assert catalog.discover()[0].launch_policy == "deny"


def test_generic_foreground_identity_uses_process_or_package() -> None:
    process_app = candidate("app_1111111111111111", "Spotify", process="Spotify.exe")
    package_app = ApplicationCandidate(
        "app_2222222222222222", "WhatsApp", "test", launch_policy="allow", package_family="family_123",
    )
    catalog = MemoryApplicationCatalog((process_app, package_app))
    assert catalog.identify("spotify.exe") == process_app.id
    assert catalog.identify("ApplicationFrameHost.exe", "family_123") == package_app.id
    assert catalog.identify("unknown.exe") is None


def test_jev_sees_only_opaque_discovered_candidate() -> None:
    app_id = "app_1111111111111111"
    catalog = MemoryApplicationCatalog((candidate(app_id, "Spotify", process="C:\\private\\Spotify.exe"),))

    def evaluate(payload):
        criteria = payload["questions"]["next_action"]["criteria"]
        option = f"open_{app_id}"
        assert option in criteria
        serialized = repr(payload)
        assert "C:\\private" not in serialized and "--" not in criteria[option]
        return {"model": "jev-1.13.0", "answers": {"next_action": {
            "type": "choice", "choice": option, "confidence": 0.99,
            "probabilities": {key: float(key == option) for key in criteria},
        }}}

    result = JevDecisionMaker(Mock(evaluate=Mock(side_effect=evaluate)), app_catalog=catalog).decide(
        "Open Spotify and search for Daft Punk", Observation("", "Desktop"),
    )
    assert result.action == OpenAppAction(app_id)


def test_explicit_app_request_without_match_stops_before_api() -> None:
    client = Mock(evaluate=Mock(side_effect=AssertionError("API must not choose an unrelated action")))
    catalog = MemoryApplicationCatalog((candidate("app_1111111111111111", "Calculator"),))
    result = JevDecisionMaker(client, app_catalog=catalog).decide(
        "Open a definitely missing program", Observation("explorer.exe", "Desktop"),
    )
    assert result.status == "needs_human"
    assert result.diagnostic == "no_application_match"
    client.evaluate.assert_not_called()


def test_autonomous_policy_requires_confirmation_at_consequential_boundary() -> None:
    observation = Observation("browser.exe", "Page", (
        UIElement("c1", "Search", "Button", enabled=True, visible=True),
        UIElement("c2", "Submit form", "Button", enabled=True, visible=True),
    ))
    policy = AutonomousActionPolicy()
    assert policy.validate(ClickAction("c1"), observation).disposition == "allow"
    assert policy.validate(ClickAction("c2"), observation).disposition == "confirm"
    assert policy.validate(PressKeyAction(("enter",)), observation).disposition == "confirm"
