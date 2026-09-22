"""Generic application discovery, matching, launch, and safety tests."""

from pathlib import Path
from unittest.mock import Mock

from computer.actions import ClickAction, OpenAppAction, PressKeyAction
from computer.applications import ApplicationCandidate, MemoryApplicationCatalog, normalize_app_name
from computer.models import Observation, UIElement
from computer.windows_apps import PackagedMetadata, ShortcutMetadata, WindowsApplicationCatalog
from decision.jev import JevDecisionMaker
from safety.policy import AutonomousActionPolicy, BasicActionPolicy


def candidate(app_id: str, name: str, *, policy: str = "allow", process: str = "") -> ApplicationCandidate:
    return ApplicationCandidate(
        app_id, name, "test", launch_policy=policy, process_names=(process,) if process else (),
    )


def test_application_name_normalization() -> None:
    assert normalize_app_name("  Spótify—Music™  ") == "spótify music"
    assert normalize_app_name("GOOGLE.Chrome") == "google chrome"


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
    assert {match.candidate.display_name for match in ambiguous} == {"Spotify", "Spotify Music"}
    assert catalog.find("Open an unrelated nonexistent program") == ()


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
    catalog.launch(by_name["Spotify"].id)
    launched.assert_called_once_with(spotify.resolve())


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
