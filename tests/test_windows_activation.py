"""Mocked Windows process/window activation; no desktop or input is used."""

from __future__ import annotations

from dataclasses import asdict
import json
import sys
from types import SimpleNamespace

from computer.applications import ApplicationCandidate, MemoryApplicationCatalog
from computer.windows_activation import (
    MAX_ACTIVATION_WINDOWS, MAX_DIAGNOSTIC_CANDIDATES,
    MAX_ELIGIBLE_DIAGNOSTIC_CANDIDATES,
    WindowsTrustedApplicationProbe,
)


APP_ID = "app_1234567890abcdef"


def candidate() -> ApplicationCandidate:
    return ApplicationCandidate(
        APP_ID, "Trusted Packaged App", "packaged", launch_policy="allow",
        package_family="Vendor.Package_123",
    )


class FakeWindows:
    def __init__(
        self,
        handles: list[int],
        *,
        process_ids: dict[int, int] | None = None,
        thread_ids: dict[int, int] | None = None,
        foreground: int = 999,
        visible: dict[int, bool] | None = None,
        minimized: dict[int, bool] | None = None,
        classes: dict[int, str] | None = None,
        enabled: dict[int, bool] | None = None,
        cloaked: dict[int, bool | None] | None = None,
        owners: dict[int, int | None] | None = None,
        roots: dict[int, int | None] | None = None,
        styles: dict[int, int] | None = None,
        client_rects: dict[int, tuple[int, int, int, int]] | None = None,
        window_rects: dict[int, tuple[int, int, int, int]] | None = None,
        z_previous: dict[int, int | None] | None = None,
        enumeration_result: bool = True,
        set_foreground_result: bool = True,
        fallback_set_foreground_result: bool | None = None,
        restore_result: bool = True,
        foreground_after_activation: int | None = None,
        fallback_foreground_after_activation: int | None = None,
        after_foreground_call=None,
        after_attach=None,
        package_identity_valid: bool = True,
        attach_result: bool = True,
        detach_result: bool = True,
        current_thread: int = 10,
        raise_on_foreground_call: int | None = None,
        attach_raises: bool = False,
        detach_raises: bool = False,
    ) -> None:
        self.handles = handles
        self.process_ids = process_ids or {handle: 77 for handle in handles}
        self.thread_ids = thread_ids or {}
        self.foreground = foreground
        self.visible = visible or {}
        self.minimized = minimized or {}
        self.classes = classes or {}
        self.enabled = enabled or {}
        self.cloaked = cloaked or {}
        self.owners = owners or {}
        self.roots = roots or {}
        self.styles = styles or {}
        self.client_rects = client_rects or {}
        self.window_rects = window_rects or {}
        self.z_previous = z_previous or {}
        self.enumeration_result = enumeration_result
        self.set_foreground_result = set_foreground_result
        self.fallback_set_foreground_result = fallback_set_foreground_result
        self.restore_result = restore_result
        self.foreground_after_activation = foreground_after_activation
        self.fallback_foreground_after_activation = fallback_foreground_after_activation
        self.after_foreground_call = after_foreground_call
        self.after_attach = after_attach
        self.package_identity_valid = package_identity_valid
        self.attach_result = attach_result
        self.detach_result = detach_result
        self.current_thread = current_thread
        self.raise_on_foreground_call = raise_on_foreground_call
        self.attach_raises = attach_raises
        self.detach_raises = detach_raises
        self.calls: list[tuple[object, ...]] = []
        self.guard = None

    def enum_windows(self, callback, context) -> bool:
        for handle in tuple(self.handles):
            if not callback(handle, context):
                break
        return self.enumeration_result

    def is_window(self, handle: int) -> bool:
        return handle in self.handles

    def get_window(self, handle: int, command: int) -> int | None:
        if command == 4:  # GW_OWNER
            return self.owners.get(handle, 0)
        if command == 3:  # GW_HWNDPREV
            return self.z_previous.get(handle)
        return None

    def get_ancestor(self, handle: int, _command: int) -> int | None:
        return self.roots.get(handle, handle)

    def get_window_long(self, handle: int, _index: int) -> int:
        return self.styles.get(handle, 0)

    def get_client_rect(self, handle: int) -> tuple[int, int, int, int]:
        return self.client_rects.get(handle, (0, 0, 780, 560))

    def get_window_rect(self, handle: int) -> tuple[int, int, int, int]:
        return self.window_rects.get(handle, (0, 0, 800, 600))

    def set_foreground_window(self, handle: int) -> bool:
        assert self.guard is None or self.guard._activation_consumed
        call_index = sum(call[0] == "foreground" for call in self.calls)
        self.calls.append(("foreground", handle))
        if self.raise_on_foreground_call == call_index + 1:
            raise RuntimeError("simulated activation failure")
        result = (
            self.set_foreground_result if call_index == 0
            else (self.fallback_set_foreground_result
                  if self.fallback_set_foreground_result is not None
                  else self.set_foreground_result)
        )
        foreground_after = (
            self.foreground_after_activation if call_index == 0
            else self.fallback_foreground_after_activation
        )
        if foreground_after is not None:
            self.foreground = foreground_after
        elif result:
            self.foreground = handle
        if self.after_foreground_call is not None:
            self.after_foreground_call(self)
        return result

    def show_window_async(self, handle: int, command: int) -> bool:
        assert self.guard is None or self.guard._activation_consumed
        self.calls.append(("restore", handle, command))
        return self.restore_result

    def current_thread_id(self) -> int:
        return self.current_thread

    def attach_thread_input(
        self, attach_thread_id: int, target_thread_id: int, attach: bool,
    ) -> bool:
        self.calls.append(("attach" if attach else "detach", attach_thread_id, target_thread_id))
        if (attach and self.attach_raises) or (not attach and self.detach_raises):
            raise RuntimeError("simulated thread-input API failure")
        if attach:
            if self.attach_result and self.after_attach is not None:
                self.after_attach(self)
            return self.attach_result
        return self.detach_result


def install_fake_windows(monkeypatch, fake: FakeWindows) -> None:
    fake_gui = SimpleNamespace(
        GetForegroundWindow=lambda: fake.foreground,
        EnumWindows=fake.enum_windows,
        IsWindow=fake.is_window,
        IsWindowVisible=lambda handle: fake.visible.get(handle, True),
        IsIconic=lambda handle: fake.minimized.get(handle, False),
        IsWindowEnabled=lambda handle: fake.enabled.get(handle, True),
        GetClassName=lambda handle: fake.classes.get(handle, "TrustedAppWindow"),
        GetWindow=fake.get_window,
        GetAncestor=fake.get_ancestor,
        GetWindowLong=fake.get_window_long,
        GetClientRect=fake.get_client_rect,
        GetWindowRect=fake.get_window_rect,
    )
    fake_process = SimpleNamespace(
        GetWindowThreadProcessId=lambda handle: (
            fake.thread_ids.get(handle, 1), fake.process_ids.get(handle, 9),
        ),
    )
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "win32gui", fake_gui)
    monkeypatch.setitem(sys.modules, "win32process", fake_process)
    monkeypatch.setitem(sys.modules, "win32con", SimpleNamespace(
        SW_RESTORE=9,
        GW_OWNER=4,
        GW_HWNDPREV=3,
        GA_ROOTOWNER=3,
        GWL_EXSTYLE=-20,
        WS_EX_TOOLWINDOW=0x80,
        WS_EX_APPWINDOW=0x40000,
    ))
    monkeypatch.setattr(
        "computer.windows_activation._dwm_cloaked",
        lambda handle: fake.cloaked.get(handle, False),
    )
    monkeypatch.setattr(
        "computer.windows._package_family_name",
        lambda process_id: (
            "Vendor.Package_123"
            if process_id == 77 and fake.package_identity_valid else ""
        ),
    )


def make_probe(
    fake: FakeWindows,
    app: ApplicationCandidate | None = None,
    *,
    with_activation_api: bool = False,
):
    return WindowsTrustedApplicationProbe(
        MemoryApplicationCatalog((app or candidate(),)), app or candidate(),
        activation_api=fake if with_activation_api else None,
    )


def test_windows_probe_matches_trusted_identity_without_exposing_window_data(monkeypatch) -> None:
    app = candidate()
    fake = FakeWindows(list(range(100, 140)), process_ids={117: 77}, foreground=117)
    install_fake_windows(monkeypatch, fake)
    package_reads: list[int] = []
    monkeypatch.setattr(
        "computer.windows._package_family_name",
        lambda process_id: package_reads.append(process_id)
        or ("Vendor.Package_123" if process_id == 77 else ""),
    )
    probe = make_probe(fake, app)

    state = probe()
    serialized = json.dumps(asdict(state))

    assert state.process_observed and state.window_observed and state.foreground_observed
    assert state.visible is True and state.minimized is False
    assert state.trusted_identity_match and state.probe_complete
    assert not state.window_stable
    assert len(package_reads) == 2
    assert "hwnd" not in serialized and '"pid"' not in serialized
    assert "window_title" not in serialized and "Vendor.Package_123" not in serialized
    assert not hasattr(sys.modules["win32gui"], "GetWindowText")
    assert not hasattr(sys.modules["win32gui"], "SendInput")


def test_stability_requires_the_same_unique_trusted_window_in_two_probes(monkeypatch) -> None:
    fake = FakeWindows([117])
    install_fake_windows(monkeypatch, fake)
    probe = make_probe(fake)

    first = probe()
    second = probe()

    assert not first.window_stable
    assert second.window_stable
    assert not second.window_ambiguous


def test_multiple_trusted_windows_are_ambiguous_and_never_chosen(monkeypatch) -> None:
    fake = FakeWindows([117, 118])
    install_fake_windows(monkeypatch, fake)
    probe = make_probe(fake)

    state = probe()
    result = probe.activate(candidate())

    assert state.window_ambiguous and state.window_observed
    assert state.window_resolution_status == "ambiguous"
    assert state.eligible_window_count == 2
    assert not state.window_stable
    assert not result.eligible
    assert result.eligibility_reason == "target_window_ambiguous"
    assert fake.calls == []


def test_unique_visible_background_window_gets_one_foreground_call(monkeypatch) -> None:
    fake = FakeWindows([117])
    install_fake_windows(monkeypatch, fake)
    probe = make_probe(fake, with_activation_api=True)
    fake.guard = probe
    probe()
    assert probe().window_stable

    result = probe.activate(candidate())

    assert result.eligible
    assert result.mechanism == "activate"
    assert result.os_call_reported_success is True
    assert fake.calls == [("foreground", 117)]
    assert not hasattr(result, "hwnd") and not hasattr(result, "process_id")
    assert not hasattr(fake, "send_keys") and not hasattr(fake, "click_input")
    assert result.activation_strategy == "simple"
    assert result.simple_attempt is not None and result.simple_attempt.verified is True
    assert result.fallback_attempt is not None
    assert result.fallback_attempt.reason == "simple_verified"


def test_unique_primary_already_foreground_is_verified_without_os_call(monkeypatch) -> None:
    fake = FakeWindows([117], foreground=117)
    install_fake_windows(monkeypatch, fake)
    probe = make_probe(fake, with_activation_api=True)
    probe()
    probe()

    result = probe.activate(candidate())

    assert not result.eligible
    assert result.eligibility_reason == "already_foreground"
    assert result.foreground_verified is True
    assert fake.calls == []


def test_minimized_window_uses_bounded_restore_then_activation(monkeypatch) -> None:
    fake = FakeWindows([117], visible={117: True}, minimized={117: True})
    install_fake_windows(monkeypatch, fake)
    probe = make_probe(fake, with_activation_api=True)
    fake.guard = probe
    probe()
    probe()

    result = probe.activate(candidate())

    assert result.eligible and result.minimized
    assert result.mechanism == "restore_then_activate"
    assert result.foreground_verified is True
    assert fake.calls == [("restore", 117, 9), ("foreground", 117)]


def test_failed_restore_does_not_call_foreground_or_retry(monkeypatch) -> None:
    fake = FakeWindows([117], minimized={117: True}, restore_result=False)
    install_fake_windows(monkeypatch, fake)
    probe = make_probe(fake, with_activation_api=True)
    probe()
    probe()

    result = probe.activate(candidate())
    second = probe.activate(candidate())

    assert result.eligible
    assert result.mechanism == "restore"
    assert result.failure_reason == "restore_not_started"
    assert not second.eligible
    assert fake.calls == [("restore", 117, 9)]


def test_hidden_minimized_window_fails_closed_before_restore(monkeypatch) -> None:
    fake = FakeWindows([117], visible={117: False}, minimized={117: True})
    install_fake_windows(monkeypatch, fake)
    probe = make_probe(fake, with_activation_api=True)
    probe()
    probe()

    result = probe.activate(candidate())

    assert not result.eligible
    assert result.eligibility_reason == "no_eligible_window"
    assert fake.calls == []


def test_disappeared_window_fails_closed_before_activation(monkeypatch) -> None:
    fake = FakeWindows([117])
    install_fake_windows(monkeypatch, fake)
    probe = make_probe(fake)
    probe()
    probe()
    fake.handles.clear()

    result = probe.activate(candidate())

    assert not result.eligible
    assert result.eligibility_reason == "target_window_missing"
    assert fake.calls == []


def test_replaced_handle_or_changed_identity_fails_closed(monkeypatch) -> None:
    fake = FakeWindows([117], process_ids={117: 77, 118: 77})
    install_fake_windows(monkeypatch, fake)
    probe = make_probe(fake)
    probe()
    probe()
    fake.handles[:] = [118]

    replaced = probe.activate(candidate())
    assert not replaced.eligible
    assert replaced.eligibility_reason == "stale_window"
    assert fake.calls == []

    fake = FakeWindows([117])
    install_fake_windows(monkeypatch, fake)
    probe = make_probe(fake)
    probe()
    probe()
    fake.process_ids[117] = 9
    changed = probe.activate(candidate())
    assert not changed.eligible
    assert changed.eligibility_reason == "target_window_missing"
    assert fake.calls == []


def test_unrelated_and_known_system_windows_are_not_candidates(monkeypatch) -> None:
    fake = FakeWindows([117], process_ids={117: 9})
    install_fake_windows(monkeypatch, fake)
    probe = make_probe(fake)
    unrelated = probe()
    assert not unrelated.trusted_identity_match
    assert not probe.activate(candidate()).eligible

    fake = FakeWindows([117], classes={117: "Progman"})
    install_fake_windows(monkeypatch, fake)
    probe = make_probe(fake)
    system_surface = probe()
    assert not system_surface.trusted_identity_match
    assert not probe.activate(candidate()).eligible
    assert fake.calls == []


def test_os_rejection_is_reported_and_activation_budget_cannot_repeat(monkeypatch) -> None:
    fake = FakeWindows([117], set_foreground_result=False)
    install_fake_windows(monkeypatch, fake)
    probe = make_probe(fake, with_activation_api=True)
    fake.guard = probe
    probe()
    probe()

    result = probe.activate(candidate())
    second = probe.activate(candidate())

    assert result.eligible
    assert not result.os_call_reported_success
    assert result.failure_reason == "foreground_not_selected"
    assert result.foreground_verified is False
    assert not second.eligible
    assert second.failure_reason == "activation_budget_consumed"
    assert fake.calls == [("foreground", 117)]


def test_os_false_return_is_successfully_verified_when_exact_selected_hwnd_is_foreground(
    monkeypatch,
) -> None:
    fake = FakeWindows(
        [117], set_foreground_result=False, foreground_after_activation=117,
    )
    install_fake_windows(monkeypatch, fake)
    probe = make_probe(fake, with_activation_api=True)
    probe()
    probe()

    result = probe.activate(candidate())

    assert result.eligible
    assert result.os_call_reported_success is False
    assert result.foreground_verified is True
    assert result.failure_reason is None
    assert result.activation_strategy == "simple"
    assert result.simple_attempt is not None
    assert result.simple_attempt.set_foreground_return is False
    assert result.simple_attempt.verified is True
    assert result.fallback_attempt is not None
    assert not result.fallback_attempt.attach_attempted


def _activate_stable_window(monkeypatch, fake: FakeWindows):
    install_fake_windows(monkeypatch, fake)
    probe = make_probe(fake, with_activation_api=True)
    probe()
    probe()
    return probe, probe.activate(candidate())


def _fallback_fake(**kwargs) -> FakeWindows:
    """Use a valid unrelated foreground HWND without adding another app window."""
    handles = kwargs.pop("handles", [117, 200])
    options = {
        "process_ids": {117: 77, 200: 9},
        "thread_ids": {117: 1, 200: 20},
        "foreground": 200,
        "set_foreground_result": False,
    }
    options.update(kwargs)
    return FakeWindows(handles, **options)


def test_failed_simple_activation_uses_one_scoped_thread_input_fallback(monkeypatch) -> None:
    fake = _fallback_fake(
        fallback_set_foreground_result=True,
        fallback_foreground_after_activation=117,
    )

    _probe, result = _activate_stable_window(monkeypatch, fake)

    assert result.activation_strategy == "attach_thread_input"
    assert result.simple_attempt is not None
    assert result.simple_attempt.set_foreground_return is False
    assert result.simple_attempt.verified is False
    assert result.foreground_verified is True
    assert result.failure_reason is None
    fallback = result.fallback_attempt
    assert fallback is not None
    assert fallback.eligible and fallback.reason == "verified"
    assert fallback.foreground_hwnd_present is True
    assert fallback.selected_thread_resolved is True
    assert fallback.foreground_thread_resolved is True
    assert fallback.current_thread_resolved is True
    assert fallback.incidental_foreground_changed_before_activation is False
    assert fallback.selected_target_still_valid_before_activation is True
    assert fallback.attach_attempted and fallback.attach_succeeded is True
    assert fallback.set_foreground_attempted
    assert fallback.set_foreground_return is True
    assert fallback.detach_succeeded is True and fallback.verified is True
    assert fake.calls == [
        ("foreground", 117), ("attach", 10, 20), ("foreground", 117), ("detach", 10, 20),
    ]


def test_fallback_false_return_succeeds_when_fresh_exact_hwnd_verification_passes(
    monkeypatch,
) -> None:
    fake = _fallback_fake(
        fallback_set_foreground_result=False,
        fallback_foreground_after_activation=117,
    )

    _probe, result = _activate_stable_window(monkeypatch, fake)

    assert result.os_call_reported_success is False
    assert result.foreground_verified is True
    assert result.failure_reason is None
    assert result.fallback_attempt is not None
    assert result.fallback_attempt.set_foreground_return is False
    assert result.fallback_attempt.verified is True
    assert result.fallback_attempt.detach_succeeded is True
    assert result.fallback_attempt.set_foreground_attempted


def test_fallback_true_return_without_exact_verification_fails_closed(monkeypatch) -> None:
    fake = _fallback_fake(
        fallback_set_foreground_result=True,
        fallback_foreground_after_activation=200,
    )

    _probe, result = _activate_stable_window(monkeypatch, fake)

    assert result.fallback_attempt is not None
    assert result.fallback_attempt.set_foreground_attempted is True
    assert result.fallback_attempt.set_foreground_return is True
    assert result.fallback_attempt.verified is False
    assert result.fallback_attempt.failure_reason == "post_activation_verification_failed"
    assert result.foreground_verified is False
    assert result.failure_reason == "post_activation_verification_failed"
    assert result.fallback_attempt.detach_succeeded is True


def test_failed_detach_prevents_activation_success_even_if_hwnd_is_foreground(monkeypatch) -> None:
    fake = _fallback_fake(
        fallback_set_foreground_result=True,
        fallback_foreground_after_activation=117,
        detach_result=False,
    )

    _probe, result = _activate_stable_window(monkeypatch, fake)

    assert result.foreground_verified is True
    assert result.failure_reason == "detach_failed"
    assert result.fallback_attempt is not None
    assert result.fallback_attempt.verified is True
    assert result.fallback_attempt.detach_succeeded is False
    assert result.fallback_attempt.failure_reason == "detach_failed"


def test_fallback_fails_closed_when_selected_window_disappears(monkeypatch) -> None:
    def remove_selected(fake: FakeWindows) -> None:
        fake.handles.remove(117)

    fake = _fallback_fake(after_foreground_call=remove_selected)

    _probe, result = _activate_stable_window(monkeypatch, fake)

    assert result.foreground_verified is False
    assert result.failure_reason == "target_window_missing"
    assert result.fallback_attempt is not None
    assert not result.fallback_attempt.attach_attempted
    assert result.fallback_attempt.reason == "selected_target_changed"
    assert not any(call[0] in {"attach", "detach"} for call in fake.calls)


def test_fallback_fails_closed_when_selected_trusted_identity_changes(monkeypatch) -> None:
    def change_identity(fake: FakeWindows) -> None:
        fake.process_ids[117] = 9

    fake = _fallback_fake(after_foreground_call=change_identity)

    _probe, result = _activate_stable_window(monkeypatch, fake)

    assert result.foreground_verified is False
    assert result.fallback_attempt is not None
    assert result.fallback_attempt.reason == "selected_target_changed"
    assert not result.fallback_attempt.attach_attempted
    assert not any(call[0] in {"attach", "detach"} for call in fake.calls)


def test_fallback_requires_a_current_valid_foreground_window(monkeypatch) -> None:
    def remove_foreground(fake: FakeWindows) -> None:
        fake.foreground = 999

    fake = _fallback_fake(after_foreground_call=remove_foreground)

    _probe, result = _activate_stable_window(monkeypatch, fake)

    assert result.fallback_attempt is not None
    assert result.fallback_attempt.foreground_hwnd_present is True
    assert result.fallback_attempt.reason == "foreground_window_invalid"
    assert not result.fallback_attempt.attach_attempted
    assert not any(call[0] in {"attach", "detach"} for call in fake.calls)


def test_fallback_requires_resolvable_selected_foreground_and_current_threads(monkeypatch) -> None:
    cases = (
        (_fallback_fake(thread_ids={117: 0, 200: 20}), "selected_thread_unresolved"),
        (_fallback_fake(thread_ids={117: 1, 200: 0}), "foreground_thread_unresolved"),
        (_fallback_fake(current_thread=0), "current_thread_unresolved"),
    )
    for fake, expected in cases:
        _probe, result = _activate_stable_window(monkeypatch, fake)
        fallback = result.fallback_attempt
        assert fallback is not None
        assert fallback.reason == expected
        assert not fallback.attach_attempted
        assert not any(call[0] == "attach" for call in fake.calls)


def test_attach_thread_input_failure_does_not_call_foreground_or_claim_detach(
    monkeypatch,
) -> None:
    fake = _fallback_fake(attach_result=False)

    _probe, result = _activate_stable_window(monkeypatch, fake)

    fallback = result.fallback_attempt
    assert fallback is not None
    assert fallback.attach_attempted and fallback.attach_succeeded is False
    assert fallback.failure_reason == "attach_failed"
    assert fallback.detach_succeeded is None
    assert [call[0] for call in fake.calls] == ["foreground", "attach"]


def test_incidental_foreground_change_after_attach_is_diagnostic_not_a_veto(
    monkeypatch,
) -> None:
    def change_incidental_foreground(fake: FakeWindows) -> None:
        fake.foreground = 201

    fake = _fallback_fake(
        handles=[117, 200, 201],
        process_ids={117: 77, 200: 9, 201: 9},
        thread_ids={117: 1, 200: 20, 201: 21},
        after_attach=change_incidental_foreground,
        fallback_set_foreground_result=True,
        fallback_foreground_after_activation=117,
    )

    _probe, result = _activate_stable_window(monkeypatch, fake)

    assert result.foreground_verified is True
    fallback = result.fallback_attempt
    assert fallback is not None
    assert fallback.incidental_foreground_changed_before_activation is True
    assert fallback.selected_target_still_valid_before_activation is True
    assert fallback.set_foreground_attempted is True
    assert fallback.detach_succeeded is True and fallback.verified is True
    assert len([call for call in fake.calls if call[0] == "attach"]) == 1
    assert len([call for call in fake.calls if call[0] == "foreground"]) == 2


def test_selected_target_disappearance_after_attach_blocks_set_foreground(monkeypatch) -> None:
    def remove_selected(fake: FakeWindows) -> None:
        fake.handles.remove(117)

    fake = _fallback_fake(after_attach=remove_selected)
    _probe, result = _activate_stable_window(monkeypatch, fake)

    fallback = result.fallback_attempt
    assert fallback is not None
    assert fallback.reason == "selected_target_changed"
    assert fallback.selected_target_still_valid_before_activation is False
    assert fallback.set_foreground_attempted is False
    assert fallback.detach_succeeded is True
    assert result.failure_reason == "selected_target_changed"
    assert len([call for call in fake.calls if call[0] == "foreground"]) == 1
    assert fake.calls[-1] == ("detach", 10, 20)


def test_selected_pid_change_after_attach_blocks_set_foreground(monkeypatch) -> None:
    def change_pid(fake: FakeWindows) -> None:
        fake.process_ids[117] = 99

    fake = _fallback_fake(after_attach=change_pid)
    _probe, result = _activate_stable_window(monkeypatch, fake)

    fallback = result.fallback_attempt
    assert fallback is not None
    assert fallback.reason == "selected_target_changed"
    assert fallback.selected_target_still_valid_before_activation is False
    assert not fallback.set_foreground_attempted
    assert result.failure_reason == "selected_target_changed"
    assert len([call for call in fake.calls if call[0] == "foreground"]) == 1


def test_selected_thread_change_after_attach_blocks_set_foreground(monkeypatch) -> None:
    def change_thread(fake: FakeWindows) -> None:
        fake.thread_ids[117] = 3

    fake = _fallback_fake(after_attach=change_thread)
    _probe, result = _activate_stable_window(monkeypatch, fake)

    fallback = result.fallback_attempt
    assert fallback is not None
    assert fallback.reason == "selected_target_changed"
    assert fallback.selected_target_still_valid_before_activation is False
    assert not fallback.set_foreground_attempted


def test_selected_trusted_identity_change_after_attach_blocks_set_foreground(monkeypatch) -> None:
    fake = _fallback_fake(
        after_attach=lambda current: setattr(current, "package_identity_valid", False),
    )
    _probe, result = _activate_stable_window(monkeypatch, fake)

    fallback = result.fallback_attempt
    assert fallback is not None
    assert fallback.reason == "selected_target_changed"
    assert fallback.selected_target_still_valid_before_activation is False
    assert not fallback.set_foreground_attempted


def test_selected_primary_surface_change_after_attach_blocks_set_foreground(monkeypatch) -> None:
    def mark_selected_as_tool_window(fake: FakeWindows) -> None:
        fake.styles[117] = 0x80

    fake = _fallback_fake(after_attach=mark_selected_as_tool_window)
    _probe, result = _activate_stable_window(monkeypatch, fake)

    fallback = result.fallback_attempt
    assert fallback is not None
    assert fallback.reason == "selected_target_revalidation_failed"
    assert fallback.selected_target_still_valid_before_activation is False
    assert not fallback.set_foreground_attempted
    assert result.failure_reason == "selected_target_revalidation_failed"


def test_new_matching_window_is_never_substituted_after_selected_target_disappears(
    monkeypatch,
) -> None:
    def replace_target(fake: FakeWindows) -> None:
        fake.handles.remove(117)
        fake.visible[118] = True

    fake = _fallback_fake(
        handles=[117, 118, 200],
        process_ids={117: 77, 118: 77, 200: 9},
        thread_ids={117: 1, 118: 2, 200: 20},
        visible={118: False},
        after_attach=replace_target,
    )
    _probe, result = _activate_stable_window(monkeypatch, fake)

    assert result.fallback_attempt is not None
    assert result.fallback_attempt.reason == "selected_target_changed"
    assert not result.fallback_attempt.set_foreground_attempted
    assert all(call[1] != 118 for call in fake.calls if call[0] == "foreground")
    assert fake.calls[-1] == ("detach", 10, 20)


def test_detach_is_attempted_in_finally_after_fallback_exception(monkeypatch) -> None:
    def raise_on_fallback(fake: FakeWindows) -> None:
        if sum(call[0] == "foreground" for call in fake.calls) == 2:
            fake.foreground = 200
            raise RuntimeError("simulated SetForegroundWindow wrapper failure")

    fake = _fallback_fake(after_foreground_call=raise_on_fallback)

    _probe, result = _activate_stable_window(monkeypatch, fake)

    assert result.fallback_attempt is not None
    assert result.fallback_attempt.failure_reason == "set_foreground_failed"
    assert result.fallback_attempt.detach_succeeded is True
    assert fake.calls[-1] == ("detach", 10, 20)


def test_attach_exception_still_attempts_paired_detach(monkeypatch) -> None:
    fake = _fallback_fake(attach_raises=True)

    _probe, result = _activate_stable_window(monkeypatch, fake)

    assert result.fallback_attempt is not None
    assert result.fallback_attempt.attach_succeeded is None
    assert result.fallback_attempt.failure_reason == "attach_failed"
    assert result.fallback_attempt.detach_succeeded is True
    assert fake.calls[-1] == ("detach", 10, 20)


def test_keyboard_interrupt_propagates_after_fallback_detach(monkeypatch) -> None:
    def interrupt_on_fallback(fake: FakeWindows) -> None:
        if sum(call[0] == "foreground" for call in fake.calls) == 2:
            raise KeyboardInterrupt

    fake = _fallback_fake(after_foreground_call=interrupt_on_fallback)
    install_fake_windows(monkeypatch, fake)
    probe = make_probe(fake, with_activation_api=True)
    probe()
    probe()

    try:
        probe.activate(candidate())
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError("KeyboardInterrupt must propagate")

    assert fake.calls[-1] == ("detach", 10, 20)


def test_alternate_matching_window_is_never_substituted_after_selected_disappears(
    monkeypatch,
) -> None:
    def replace_selected_with_another_primary(fake: FakeWindows) -> None:
        fake.handles.remove(117)
        fake.visible[118] = True

    fake = FakeWindows(
        [117, 118], visible={118: False},
        thread_ids={117: 1, 118: 2}, foreground=999,
        after_foreground_call=replace_selected_with_another_primary,
    )
    _probe, result = _activate_stable_window(monkeypatch, fake)

    assert result.eligible and result.budget_consumed
    assert result.foreground_verified is False
    assert result.failure_reason == "target_window_missing"
    assert result.fallback_attempt is not None
    assert result.fallback_attempt.reason == "selected_target_changed"
    assert not result.fallback_attempt.attach_attempted
    assert all(call[1] != 118 for call in fake.calls if call[0] == "foreground")
    assert len([call for call in fake.calls if call[0] == "foreground"]) == 1


def test_tool_window_foreground_never_replaces_exact_selected_primary(monkeypatch) -> None:
    fake = FakeWindows(
        [117, 118], styles={118: 0x80},
        thread_ids={117: 1, 118: 20}, foreground=118,
        set_foreground_result=False,
        fallback_set_foreground_result=True,
        fallback_foreground_after_activation=117,
    )

    _probe, result = _activate_stable_window(monkeypatch, fake)

    assert result.foreground_verified is True
    assert result.activation_strategy == "attach_thread_input"
    assert [call[1] for call in fake.calls if call[0] == "foreground"] == [117, 117]
    assert fake.calls[1] == ("attach", 10, 20)


def test_fallback_attempt_is_bounded_and_never_synthesizes_keyboard_input(monkeypatch) -> None:
    fake = _fallback_fake(
        fallback_set_foreground_result=False,
        fallback_foreground_after_activation=200,
    )
    probe, result = _activate_stable_window(monkeypatch, fake)
    second = probe.activate(candidate())

    assert result.fallback_attempt is not None
    assert result.fallback_attempt.attach_attempted
    assert result.fallback_attempt.set_foreground_return is False
    assert result.fallback_attempt.verified is False
    assert result.foreground_verified is False
    assert result.failure_reason == "set_foreground_failed"
    assert not second.eligible
    assert len([call for call in fake.calls if call[0] == "attach"]) == 1
    assert len([call for call in fake.calls if call[0] == "foreground"]) <= 2
    assert all(call[0] in {"foreground", "attach", "detach", "restore"} for call in fake.calls)


def test_activation_fallback_diagnostics_do_not_expose_hwnds_or_thread_ids(monkeypatch) -> None:
    fake = _fallback_fake(
        fallback_set_foreground_result=True,
        fallback_foreground_after_activation=117,
    )

    _probe, result = _activate_stable_window(monkeypatch, fake)
    serialized = json.dumps(asdict(result))

    for private_value in ("117", "200", "77", "10", "20"):
        assert private_value not in serialized
    assert "foreground_hwnd_present" in serialized
    assert '"pid"' not in serialized


def test_os_true_return_fails_when_fresh_foreground_is_unrelated(monkeypatch) -> None:
    fake = FakeWindows([117], foreground_after_activation=999)
    install_fake_windows(monkeypatch, fake)
    probe = make_probe(fake, with_activation_api=True)
    probe()
    probe()

    result = probe.activate(candidate())

    assert result.os_call_reported_success is True
    assert result.foreground_verified is False
    assert result.failure_reason == "foreground_not_selected"


def test_selected_window_disappearing_after_os_call_fails_verification(monkeypatch) -> None:
    def remove_selected(fake: FakeWindows) -> None:
        fake.handles.remove(117)

    fake = FakeWindows([117], after_foreground_call=remove_selected)
    install_fake_windows(monkeypatch, fake)
    probe = make_probe(fake, with_activation_api=True)
    probe()
    probe()

    result = probe.activate(candidate())

    assert result.foreground_verified is False
    assert result.failure_reason == "target_window_missing"


def test_trusted_identity_change_after_os_call_fails_verification(monkeypatch) -> None:
    def change_identity(fake: FakeWindows) -> None:
        fake.process_ids[117] = 9

    fake = FakeWindows([117], after_foreground_call=change_identity)
    install_fake_windows(monkeypatch, fake)
    probe = make_probe(fake, with_activation_api=True)
    probe()
    probe()

    result = probe.activate(candidate())

    assert result.foreground_verified is False
    assert result.failure_reason == "trusted_identity_mismatch"


def test_windows_probe_bounds_window_enumeration_and_identity_checks(monkeypatch) -> None:
    candidate_ = candidate()
    fake = FakeWindows(list(range(1, MAX_ACTIVATION_WINDOWS + 100)))
    install_fake_windows(monkeypatch, fake)
    callback_count = 0
    old_enum = sys.modules["win32gui"].EnumWindows

    def counted_enum(callback, context):
        nonlocal callback_count

        def count(handle, _context):
            nonlocal callback_count
            callback_count += 1
            return callback(handle, _context)

        old_enum(count, context)

    sys.modules["win32gui"].EnumWindows = counted_enum
    identity_reads: list[int] = []
    monkeypatch.setattr(
        "computer.windows._package_family_name",
        lambda process_id: identity_reads.append(process_id) or "",
    )

    state = make_probe(fake, candidate_)()

    assert not state.process_observed and not state.window_observed
    assert not state.probe_complete
    assert len(identity_reads) <= MAX_ACTIVATION_WINDOWS
    assert callback_count <= MAX_ACTIVATION_WINDOWS + 1


def test_one_visible_window_and_hidden_helper_resolve_unique(monkeypatch) -> None:
    fake = FakeWindows([117, 118], visible={118: False}, z_previous={118: 117})
    install_fake_windows(monkeypatch, fake)

    probe = make_probe(fake)
    first = probe()
    second = probe()

    assert first.window_resolution_status == "unique"
    assert first.matching_trusted_window_count == 2
    assert first.eligible_window_count == 1
    assert first.enumeration_complete
    assert [item.visible for item in first.window_candidates] == [True, False]
    assert all(item.activation_candidate for item in first.window_candidates)
    assert [item.explicit_activation_eligible for item in first.window_candidates] == [True, False]
    assert [item.rejection_reason for item in first.window_candidates] == [None, "target_not_visible"]
    assert [item.stable_across_probes for item in first.window_candidates] == [False, False]
    assert second.window_stable
    assert [item.stable_across_probes for item in second.window_candidates] == [True, False]
    assert [item.z_order_bucket for item in second.window_candidates] == ["front", "back"]


def test_cloaked_window_is_rejected_from_eligibility(monkeypatch) -> None:
    fake = FakeWindows([117, 118], cloaked={118: True}, z_previous={118: 117})
    install_fake_windows(monkeypatch, fake)

    state = make_probe(fake)()

    assert state.window_resolution_status == "unique"
    assert state.eligible_window_count == 1
    assert [item.cloaked_if_available for item in state.window_candidates] == [False, True]
    assert [item.activation_candidate for item in state.window_candidates] == [True, True]
    assert [item.rejection_reason for item in state.window_candidates] == [None, "target_cloaked"]


def test_unknown_cloaking_state_fails_closed_for_candidate(monkeypatch) -> None:
    fake = FakeWindows([117], cloaked={117: None})
    install_fake_windows(monkeypatch, fake)

    state = make_probe(fake)()

    assert state.window_resolution_status == "none"
    assert state.eligible_window_count == 0
    assert state.rejection_reason_counts == (("cloaking_unknown", 1),)
    assert state.window_candidates[0].rejection_reason == "cloaking_unknown"


def test_candidate_diagnostics_identify_owned_dialog_relationship(monkeypatch) -> None:
    fake = FakeWindows(
        [117, 118], owners={118: 117}, roots={117: 117, 118: 117},
        z_previous={118: 117},
    )
    install_fake_windows(monkeypatch, fake)

    state = make_probe(fake)()

    assert state.window_ambiguous
    main, dialog = state.window_candidates
    assert not main.owner_present and main.root_owner_relationship == "self"
    assert dialog.owner_present and dialog.root_owner_relationship == "other"


def test_two_genuine_visible_windows_remain_ambiguous_with_relative_order(monkeypatch) -> None:
    fake = FakeWindows([117, 118], z_previous={118: 117})
    install_fake_windows(monkeypatch, fake)

    state = make_probe(fake)()

    assert state.window_ambiguous
    assert all(item.visible and item.enabled for item in state.window_candidates)
    assert all(item.has_nonzero_client_area for item in state.window_candidates)
    assert [item.z_order_bucket for item in state.window_candidates] == ["front", "back"]


def test_candidate_diagnostics_report_tool_window_but_keep_existing_candidate(monkeypatch) -> None:
    fake = FakeWindows([117], styles={117: 0x80})
    install_fake_windows(monkeypatch, fake)

    state = make_probe(fake)()

    diagnostic = state.window_candidates[0]
    assert diagnostic.tool_window is True
    assert diagnostic.app_window is False
    assert diagnostic.activation_candidate
    assert diagnostic.explicit_activation_eligible


def test_candidate_diagnostics_report_zero_client_area_and_minimized_state(monkeypatch) -> None:
    fake = FakeWindows(
        [117], minimized={117: True}, client_rects={117: (0, 0, 0, 0)},
    )
    install_fake_windows(monkeypatch, fake)

    diagnostic = make_probe(fake)().window_candidates[0]

    assert diagnostic.minimized
    assert diagnostic.has_nonzero_client_area is False
    assert diagnostic.client_area_bucket == "zero"
    assert diagnostic.window_area_bucket == "medium"


def test_zero_client_area_window_is_rejected_from_activation(monkeypatch) -> None:
    fake = FakeWindows([117], client_rects={117: (0, 0, 0, 0)})
    install_fake_windows(monkeypatch, fake)

    state = make_probe(fake)()

    assert state.window_resolution_status == "none"
    assert state.rejection_reason_counts == (("zero_client_area", 1),)
    assert state.window_candidates[0].rejection_reason == "zero_client_area"


def test_small_visible_client_area_is_eligible_and_disabled_state_is_not_rejection(
    monkeypatch,
) -> None:
    fake = FakeWindows(
        [117], enabled={117: False}, client_rects={117: (0, 0, 1, 1)},
    )
    install_fake_windows(monkeypatch, fake)

    state = make_probe(fake)()

    assert state.window_resolution_status == "unique"
    assert state.window_candidates[0].enabled is False
    assert state.window_candidates[0].client_area_bucket == "small"
    assert state.window_candidates[0].explicit_activation_eligible


def test_optional_window_fact_failures_do_not_abort_trusted_probe(monkeypatch) -> None:
    fake = FakeWindows([117])
    install_fake_windows(monkeypatch, fake)

    def unavailable(_handle: int) -> object:
        raise OSError("mocked optional Win32 query failure")

    sys.modules["win32gui"].IsWindowEnabled = unavailable
    sys.modules["win32gui"].GetClientRect = unavailable

    state = make_probe(fake)()

    assert state.probe_complete and state.window_observed
    assert len(state.window_candidates) == 1
    assert state.window_candidates[0].enabled is None
    assert state.window_candidates[0].has_nonzero_client_area is None
    assert state.window_candidates[0].client_area_bucket == "unavailable"


def test_incomplete_enumeration_retains_bounded_candidates_but_not_stability(monkeypatch) -> None:
    fake = FakeWindows([117, 118], enumeration_result=False, z_previous={118: 117})
    install_fake_windows(monkeypatch, fake)
    probe = make_probe(fake)
    probe()

    state = probe()

    assert state.window_resolution_status == "incomplete"
    assert not state.probe_complete
    assert all(not item.stable_across_probes for item in state.window_candidates)


def test_window_candidate_diagnostics_are_limited_to_eight(monkeypatch) -> None:
    handles = list(range(100, 100 + MAX_DIAGNOSTIC_CANDIDATES + 2))
    fake = FakeWindows(handles)
    install_fake_windows(monkeypatch, fake)

    state = make_probe(fake)()

    assert state.window_ambiguous
    assert len(state.window_candidates) == MAX_DIAGNOSTIC_CANDIDATES
    assert state.candidate_diagnostics_truncated
    assert [item.candidate_index for item in state.window_candidates] == list(range(1, 9))


def test_full_resolver_finds_unique_eligible_window_after_diagnostic_limit(monkeypatch) -> None:
    handles = list(range(100, 100 + MAX_DIAGNOSTIC_CANDIDATES + 1))
    fake = FakeWindows(handles, visible={handle: False for handle in handles[:-1]})
    install_fake_windows(monkeypatch, fake)

    state = make_probe(fake)()

    assert state.candidate_diagnostics_truncated
    assert len(state.window_candidates) == MAX_DIAGNOSTIC_CANDIDATES
    assert state.window_resolution_status == "unique"
    assert state.matching_trusted_window_count == MAX_DIAGNOSTIC_CANDIDATES + 1
    assert state.eligible_window_count == 1


def test_full_resolver_finds_second_eligible_window_after_diagnostic_limit(monkeypatch) -> None:
    handles = list(range(100, 100 + MAX_DIAGNOSTIC_CANDIDATES + 2))
    hidden = {handle: False for handle in handles[:-2]}
    fake = FakeWindows(handles, visible=hidden)
    install_fake_windows(monkeypatch, fake)

    state = make_probe(fake)()

    assert state.candidate_diagnostics_truncated
    assert state.window_resolution_status == "ambiguous"
    assert state.eligible_window_count == 2


def test_eligible_diagnostics_include_candidates_after_general_cap_without_activating(
    monkeypatch,
) -> None:
    handles = list(range(100, 117))
    visible = {handle: False for handle in handles}
    visible[115] = True
    visible[116] = True
    fake = FakeWindows(
        handles,
        visible=visible,
        foreground=999,
        owners={116: 115},
        roots={115: 115, 116: 115},
        z_previous={116: 115},
        classes={
            115: "C:\\private\\HiddenWindowClassOne",
            116: "C:\\private\\HiddenWindowClassTwo",
        },
        process_ids={handle: 77 for handle in handles},
        client_rects={115: (0, 0, 901, 701), 116: (0, 0, 302, 202)},
        window_rects={115: (5, 7, 906, 708), 116: (13, 17, 315, 219)},
    )
    install_fake_windows(monkeypatch, fake)
    probe = make_probe(fake, with_activation_api=True)

    state = probe()
    activation = probe.activate(candidate())
    serialized = json.dumps(asdict(state))

    assert state.matching_trusted_window_count == 17
    assert state.eligible_window_count == 2
    assert state.window_resolution_status == "ambiguous"
    assert state.window_ambiguous
    assert len(state.window_candidates) == MAX_DIAGNOSTIC_CANDIDATES
    assert state.candidate_diagnostics_truncated
    assert len(state.eligible_window_diagnostics) == 2
    assert not state.eligible_window_diagnostics_truncated
    first, second = state.eligible_window_diagnostics
    assert [item.eligible_candidate_index for item in state.eligible_window_diagnostics] == [1, 2]
    assert first.diagnostic_id == "ew1" and second.diagnostic_id == "ew2"
    assert first.facts.explicit_activation_eligible and second.facts.explicit_activation_eligible
    assert first.facts.stable_across_probes is False
    assert second.facts.stable_across_probes is False
    assert first.facts.z_order_bucket in {"front", "middle", "back", "unavailable"}
    assert state.eligible_window_relationships[0].relationship == "one_owned_by_other"
    for private_value in (
        "115", "116", "77", "HiddenWindowClassOne", "HiddenWindowClassTwo",
        "private", "901", "701", "906", "708", "302", "202", "315", "219",
    ):
        assert private_value not in serialized
    assert '"hwnd"' not in serialized and '"pid"' not in serialized
    assert not activation.eligible
    assert activation.eligibility_reason == "target_window_ambiguous"
    assert fake.calls == []


def test_eligible_diagnostic_array_has_its_own_cap(monkeypatch) -> None:
    handles = list(range(100, 100 + MAX_ELIGIBLE_DIAGNOSTIC_CANDIDATES + 2))
    fake = FakeWindows(handles)
    install_fake_windows(monkeypatch, fake)

    state = make_probe(fake)()

    assert state.window_resolution_status == "ambiguous"
    assert state.eligible_window_count == MAX_ELIGIBLE_DIAGNOSTIC_CANDIDATES + 2
    assert len(state.window_candidates) == MAX_DIAGNOSTIC_CANDIDATES
    assert state.candidate_diagnostics_truncated
    assert len(state.eligible_window_diagnostics) == MAX_ELIGIBLE_DIAGNOSTIC_CANDIDATES
    assert state.eligible_window_diagnostics_truncated
    assert len(state.eligible_window_relationships) == (
        MAX_ELIGIBLE_DIAGNOSTIC_CANDIDATES * (MAX_ELIGIBLE_DIAGNOSTIC_CANDIDATES - 1) // 2
    )


def test_eligible_diagnostic_ids_correlate_only_current_probe_candidates(monkeypatch) -> None:
    fake = FakeWindows([117, 118], z_previous={118: 117})
    install_fake_windows(monkeypatch, fake)
    probe = make_probe(fake)

    first = probe()
    second = probe()

    first_ids = [item.diagnostic_id for item in first.eligible_window_diagnostics]
    second_ids = [item.diagnostic_id for item in second.eligible_window_diagnostics]
    assert first_ids == second_ids == ["ew1", "ew2"]
    assert [item.facts.stable_across_probes for item in first.eligible_window_diagnostics] == [
        False, False,
    ]
    assert [item.facts.stable_across_probes for item in second.eligible_window_diagnostics] == [
        True, True,
    ]
    assert second.eligible_window_relationships[0].relationship == "independent"


def test_eligible_window_relationships_report_shared_root_or_unknown(monkeypatch) -> None:
    fake = FakeWindows([117, 118], roots={117: 117, 118: 117})
    install_fake_windows(monkeypatch, fake)

    same_root = make_probe(fake)()

    assert same_root.eligible_window_relationships[0].relationship == "same_root_owner"

    fake = FakeWindows([117, 118], owners={117: None, 118: None})
    install_fake_windows(monkeypatch, fake)

    unknown = make_probe(fake)()

    assert unknown.eligible_window_relationships[0].relationship == "unknown"


def test_hidden_large_tool_and_zero_area_helpers_do_not_create_ambiguity(monkeypatch) -> None:
    handles = [117, 118, 119, 120, 121]
    fake = FakeWindows(
        handles,
        visible={118: False, 119: False, 120: False, 121: False},
        styles={120: 0x80},
        client_rects={121: (0, 0, 0, 0)},
        window_rects={119: (0, 0, 4000, 3000)},
    )
    install_fake_windows(monkeypatch, fake)

    state = make_probe(fake)()

    assert state.window_resolution_status == "unique"
    assert state.matching_trusted_window_count == 5
    assert state.eligible_window_count == 1
    assert state.window_candidates[2].window_area_bucket == "large"
    assert state.window_candidates[3].tool_window is True
    assert state.window_candidates[4].has_nonzero_client_area is False


def test_zero_eligible_windows_resolve_none(monkeypatch) -> None:
    fake = FakeWindows([117, 118], visible={117: False, 118: False})
    install_fake_windows(monkeypatch, fake)

    state = make_probe(fake)()

    assert state.window_resolution_status == "none"
    assert state.eligible_window_count == 0
    assert not state.window_ambiguous


def test_candidate_diagnostics_serialization_excludes_identifiers_and_private_metadata(
    monkeypatch,
) -> None:
    fake = FakeWindows(
        [117], process_ids={117: 77}, classes={117: "C:\\private\\InternalWindowClass"},
        window_rects={117: (123, 456, 923, 1056)},
    )
    install_fake_windows(monkeypatch, fake)

    serialized = json.dumps(asdict(make_probe(fake)()))

    for private_value in (
        "117", "77", "InternalWindowClass", "private", "Vendor.Package_123",
        "123", "456", "923", "1056", "800", "600",
    ):
        assert private_value not in serialized


def test_pre_activation_revalidation_rejects_hidden_or_cloaked_selected_window(
    monkeypatch,
) -> None:
    for mutation, expected in (
        (lambda fake: fake.visible.update({117: False}), "target_not_visible"),
        (lambda fake: fake.cloaked.update({117: True}), "target_cloaked"),
        (lambda fake: fake.cloaked.update({117: None}), "cloaking_unknown"),
    ):
        fake = FakeWindows([117])
        install_fake_windows(monkeypatch, fake)
        probe = make_probe(fake, with_activation_api=True)
        probe()
        probe()
        mutation(fake)

        result = probe.activate(candidate())

        assert not result.eligible
        assert result.eligibility_reason == expected
        assert result.budget_consumed is False
        assert fake.calls == []


def test_second_eligible_window_appearing_in_pre_activation_scan_fails_closed(
    monkeypatch,
) -> None:
    fake = FakeWindows([117, 118])
    install_fake_windows(monkeypatch, fake)
    probe = make_probe(fake, with_activation_api=True)
    probe()
    probe()
    fake.visible[118] = True

    result = probe.activate(candidate())

    assert not result.eligible
    assert result.eligibility_reason == "target_window_ambiguous"
    assert result.budget_consumed is False
    assert fake.calls == []


def test_revalidated_activation_consumes_budget_before_os_call(monkeypatch) -> None:
    fake = FakeWindows([117])
    install_fake_windows(monkeypatch, fake)
    probe = make_probe(fake, with_activation_api=True)
    fake.guard = probe
    probe()
    probe()

    result = probe.activate(candidate())

    assert result.budget_consumed is True
    assert fake.calls == [("foreground", 117)]


def test_primary_surface_is_unique_with_one_tool_window_and_preserves_base_ambiguity(
    monkeypatch,
) -> None:
    fake = FakeWindows([117, 118], styles={118: 0x80})
    install_fake_windows(monkeypatch, fake)
    probe = make_probe(fake, with_activation_api=True)
    fake.guard = probe

    probe()
    state = probe()
    result = probe.activate(candidate())

    assert state.window_resolution_status == "ambiguous"
    assert state.eligible_window_count == state.base_eligible_window_count == 2
    assert state.primary_surface_candidate_count == 1
    assert state.tool_surface_candidate_count == 1
    assert state.primary_surface_resolution_status == "unique"
    assert all(item.facts.explicit_activation_eligible for item in state.eligible_window_diagnostics)
    assert [item.surface_class for item in state.eligible_window_diagnostics] == [
        "primary", "tool",
    ]
    assert result.eligible
    assert fake.calls == [("foreground", 117)]


def test_primary_surface_stays_unique_with_many_tool_windows(monkeypatch) -> None:
    fake = FakeWindows([117, 118, 119, 120], styles={118: 0x80, 119: 0x80, 120: 0x80})
    install_fake_windows(monkeypatch, fake)
    probe = make_probe(fake, with_activation_api=True)
    probe()
    state = probe()

    result = probe.activate(candidate())

    assert state.window_resolution_status == "ambiguous"
    assert state.base_eligible_window_count == 4
    assert state.primary_surface_candidate_count == 1
    assert state.tool_surface_candidate_count == 3
    assert state.primary_surface_resolution_status == "unique"
    assert result.eligible
    assert fake.calls == [("foreground", 117)]


def test_multiple_primary_surfaces_remain_ambiguous_even_with_tool_window(monkeypatch) -> None:
    fake = FakeWindows([117, 118, 119], styles={119: 0x80})
    install_fake_windows(monkeypatch, fake)
    probe = make_probe(fake, with_activation_api=True)
    probe()
    state = probe()

    result = probe.activate(candidate())

    assert state.base_eligible_window_count == 3
    assert state.primary_surface_candidate_count == 2
    assert state.tool_surface_candidate_count == 1
    assert state.primary_surface_resolution_status == "ambiguous"
    assert not result.eligible
    assert result.eligibility_reason == "target_window_ambiguous"
    assert fake.calls == []


def test_tool_windows_alone_never_become_primary(monkeypatch) -> None:
    for handles in ([117], [117, 118, 119]):
        fake = FakeWindows(handles, styles={handle: 0x80 for handle in handles})
        install_fake_windows(monkeypatch, fake)
        probe = make_probe(fake, with_activation_api=True)
        probe()
        state = probe()

        result = probe.activate(candidate())

        assert state.window_resolution_status == ("unique" if len(handles) == 1 else "ambiguous")
        assert state.base_eligible_window_count == len(handles)
        assert state.primary_surface_candidate_count == 0
        assert state.tool_surface_candidate_count == len(handles)
        assert state.primary_surface_resolution_status == "none"
        assert not result.eligible
        assert result.eligibility_reason == "no_eligible_window"
        assert fake.calls == []


def test_minimized_primary_with_tool_window_keeps_restore_then_activate_path(monkeypatch) -> None:
    fake = FakeWindows(
        [117, 118], minimized={117: True}, styles={118: 0x80},
    )
    install_fake_windows(monkeypatch, fake)
    probe = make_probe(fake, with_activation_api=True)
    probe()
    state = probe()

    result = probe.activate(candidate())

    assert state.primary_surface_resolution_status == "unique"
    assert state.primary_surface_facts is not None
    assert state.primary_surface_facts.minimized is True
    assert result.eligible and result.minimized
    assert result.mechanism == "restore_then_activate"
    assert fake.calls == [("restore", 117, 9), ("foreground", 117)]


def test_selected_primary_disappearing_before_activation_fails_closed(monkeypatch) -> None:
    fake = FakeWindows([117, 118], styles={118: 0x80})
    install_fake_windows(monkeypatch, fake)
    probe = make_probe(fake, with_activation_api=True)
    probe()
    probe()
    fake.handles.remove(117)

    result = probe.activate(candidate())

    assert not result.eligible
    assert result.eligibility_reason in {"target_window_missing", "stale_window"}
    assert fake.calls == []


def test_selected_primary_becoming_tool_window_fails_closed(monkeypatch) -> None:
    fake = FakeWindows([117, 118], styles={118: 0x80})
    install_fake_windows(monkeypatch, fake)
    probe = make_probe(fake, with_activation_api=True)
    probe()
    probe()
    fake.styles[117] = 0x80

    result = probe.activate(candidate())

    assert not result.eligible
    assert result.eligibility_reason == "no_eligible_window"
    assert fake.calls == []


def test_second_primary_appearing_during_rescan_prevents_os_call(monkeypatch) -> None:
    fake = FakeWindows([117, 118], visible={118: False})
    install_fake_windows(monkeypatch, fake)
    probe = make_probe(fake, with_activation_api=True)
    probe()
    probe()
    fake.visible[118] = True

    result = probe.activate(candidate())

    assert not result.eligible
    assert result.eligibility_reason == "target_window_ambiguous"
    assert fake.calls == []


def test_foreground_tool_window_does_not_change_primary_surface_identity(monkeypatch) -> None:
    fake = FakeWindows([117, 118], styles={118: 0x80}, foreground=118)
    install_fake_windows(monkeypatch, fake)
    probe = make_probe(fake, with_activation_api=True)
    probe()
    state = probe()

    assert state.foreground_observed
    assert state.primary_surface_resolution_status == "unique"
    assert state.primary_surface_facts is not None
    assert state.primary_surface_facts.foreground is False
    assert [item.surface_class for item in state.eligible_window_diagnostics] == [
        "primary", "tool",
    ]
    result = probe.activate(candidate())
    assert result.eligible
    assert fake.calls == [("foreground", 117)]


def test_unknown_tool_style_classification_fails_closed(monkeypatch) -> None:
    fake = FakeWindows([117, 118])
    install_fake_windows(monkeypatch, fake)
    gui = sys.modules["win32gui"]

    def get_style(handle: int, index: int) -> int:
        if handle == 118:
            raise OSError("style unavailable")
        return fake.get_window_long(handle, index)

    gui.GetWindowLong = get_style
    probe = make_probe(fake, with_activation_api=True)
    probe()
    state = probe()

    result = probe.activate(candidate())

    assert state.window_resolution_status == "ambiguous"
    assert state.eligible_window_count == 2
    assert state.primary_surface_candidate_count == 1
    assert state.tool_surface_candidate_count == 0
    assert state.primary_surface_resolution_status == "incomplete"
    assert [item.surface_class for item in state.eligible_window_diagnostics] == [
        "primary", "unknown",
    ]
    assert not result.eligible
    assert result.eligibility_reason == "probe_incomplete"
    assert fake.calls == []


def test_primary_surface_diagnostics_are_bounded_and_keep_base_eligibility(monkeypatch) -> None:
    fake = FakeWindows(
        [117, 118], process_ids={117: 77, 118: 77},
        classes={117: "C:\\private\\PrimaryClass", 118: "C:\\private\\ToolClass"},
        styles={118: 0x80},
    )
    install_fake_windows(monkeypatch, fake)
    state = make_probe(fake)()
    serialized = json.dumps(asdict(state))

    assert state.window_resolution_status == "ambiguous"
    assert state.eligible_window_count == 2
    assert state.base_eligible_window_count == 2
    assert state.primary_surface_candidate_count == 1
    assert state.tool_surface_candidate_count == 1
    assert state.primary_surface_resolution_status == "unique"
    assert [item.surface_class for item in state.eligible_window_diagnostics] == [
        "primary", "tool",
    ]
    for private in ("117", "118", "77", "PrimaryClass", "ToolClass", "private"):
        assert private not in serialized
    assert '"hwnd"' not in serialized and '"pid"' not in serialized
