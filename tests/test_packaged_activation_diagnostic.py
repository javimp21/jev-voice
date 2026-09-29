"""Mocked tests for the manual packaged-app activation A/B diagnostic."""

from __future__ import annotations

from dataclasses import asdict
import json
from unittest.mock import Mock

import pytest

import main as cli
from computer.applications import (
    EligibleWindowFacts, SimpleActivationAttempt, TrustedApplicationRuntimeState,
    ThreadInputFallbackAttempt, TrustedWindowActivationResult,
)
from computer.models import Observation
from computer.packaged_activation_diagnostic import (
    PackagedActivationDiagnostic, run_packaged_activation_diagnostic,
)
from computer import windows_activation_manager
from computer.windows_activation_manager import activate_application
from computer.windows_apps import (
    PackagedLaunchDiagnosticResult, PackagedMetadata, ShortcutMetadata,
    WindowsApplicationCatalog,
)


TRUSTED_AUMID = "Vendor.Package_123!App"


def packaged_catalog(launcher: Mock | None = None) -> tuple[WindowsApplicationCatalog, str]:
    catalog = WindowsApplicationCatalog(
        start_menu_roots=(),
        packaged_reader=lambda: (PackagedMetadata(
            "WhatsApp", TRUSTED_AUMID, package_family="Vendor.Package_123",
            process_names=("WhatsApp.exe",), launcher=launcher or Mock(),
        ),),
    )
    candidate = catalog.discover()[0]
    return catalog, candidate.id


def test_catalog_binds_manager_request_to_trusted_aumid_and_keeps_production_launch() -> None:
    shell_launcher = Mock()
    catalog, app_id = packaged_catalog(shell_launcher)
    manager = Mock(return_value=PackagedLaunchDiagnosticResult(True, 12, True, 0))

    result = catalog.launch_packaged_for_diagnostic(
        app_id, "activation-manager", activation_manager=manager,
    )

    manager.assert_called_once_with(TRUSTED_AUMID)
    assert result.returned_pid_present
    assert TRUSTED_AUMID not in repr(result)
    shell_launcher.assert_not_called()

    catalog.launch(app_id)
    shell_launcher.assert_called_once_with()
    assert "activation-manager" not in repr(catalog.launch)


def test_shell_diagnostic_uses_same_catalog_launcher() -> None:
    launcher = Mock()
    catalog, app_id = packaged_catalog(launcher)

    result = catalog.launch_packaged_for_diagnostic(app_id, "shell")

    assert result.request_succeeded
    assert result.request_elapsed_ms is not None
    launcher.assert_called_once_with()


def test_diagnostic_rejects_unknown_or_nonpackaged_ids_without_calling_manager(tmp_path) -> None:
    catalog, app_id = packaged_catalog()
    manager = Mock()

    with pytest.raises(ValueError):
        catalog.launch_packaged_for_diagnostic("app_deadbeef", "activation-manager", activation_manager=manager)
    manager.assert_not_called()

    shortcut = tmp_path / "Editor.lnk"
    shortcut.touch()
    shortcut_catalog = WindowsApplicationCatalog(
        start_menu_roots=(tmp_path,),
        shortcut_reader=lambda _path: ShortcutMetadata("C:\\Apps\\editor.exe"),
        packaged_reader=lambda: (),
    )
    shortcut_id = shortcut_catalog.discover()[0].id
    with pytest.raises(ValueError):
        shortcut_catalog.launch_packaged_for_diagnostic(
            shortcut_id, "activation-manager", activation_manager=manager,
        )
    manager.assert_not_called()
    assert catalog.resolve(app_id) is not None


class FakeComRuntime:
    def __init__(self, *, hresult: int = 0, returned_pid: int = 77, error: Exception | None = None) -> None:
        self.hresult = hresult
        self.returned_pid = returned_pid
        self.error = error
        self.initialized = 0
        self.uninitialized = 0
        self.created = 0
        self.released = 0
        self.calls: list[tuple[object, str]] = []

    def initialize_com(self) -> None:
        self.initialized += 1

    def create_manager(self) -> object:
        self.created += 1
        return object()

    def activate_application(self, manager: object, app_user_model_id: str) -> tuple[int, int]:
        self.calls.append((manager, app_user_model_id))
        if self.error is not None:
            raise self.error
        return self.hresult, self.returned_pid

    def release_manager(self, _manager: object) -> None:
        self.released += 1

    def uninitialize(self) -> None:
        self.uninitialized += 1


def test_com_success_uses_windows_launch_and_reports_only_pid_presence() -> None:
    runtime = FakeComRuntime(returned_pid=12345)

    result = activate_application(TRUSTED_AUMID, runtime_factory=lambda: runtime)

    assert result.request_succeeded
    assert result.returned_pid_present
    assert result.hresult == 0
    assert runtime.calls[0][1] == TRUSTED_AUMID
    assert runtime.initialized == runtime.uninitialized == runtime.created == 1
    assert runtime.released == 1
    assert "12345" not in repr(result)


def test_com_success_without_pid_is_still_a_successful_request() -> None:
    runtime = FakeComRuntime(returned_pid=0)

    result = activate_application(TRUSTED_AUMID, runtime_factory=lambda: runtime)

    assert result.request_succeeded
    assert not result.returned_pid_present


def test_com_failure_exposes_hresult_category_without_exception_text() -> None:
    class FakeComError(Exception):
        hresult = -2147024891

    runtime = FakeComRuntime(error=FakeComError("private COM details"))

    result = activate_application(TRUSTED_AUMID, runtime_factory=lambda: runtime)

    assert not result.request_succeeded
    assert result.failure_reason == "activation_failed"
    assert result.hresult == 0x80070005
    assert "private COM details" not in repr(result)
    assert runtime.uninitialized == 1


def test_com_activation_failure_hresult_is_reported_as_unsigned_code() -> None:
    runtime = FakeComRuntime(hresult=-2147024891, returned_pid=12345)

    result = activate_application(TRUSTED_AUMID, runtime_factory=lambda: runtime)

    assert not result.request_succeeded
    assert result.hresult == 0x80070005
    assert not result.returned_pid_present


def test_com_adapter_uses_documented_class_interface_and_context_ids() -> None:
    assert windows_activation_manager._CLSID_APPLICATION_ACTIVATION_MANAGER == (
        "45BA127D-10A8-46EA-8AB7-56EA9078943C"
    )
    assert windows_activation_manager._IID_IAPPLICATION_ACTIVATION_MANAGER == (
        "2E941141-7F97-4756-BA1D-9DECDE894A3D"
    )
    assert windows_activation_manager._CLSCTX_LOCAL_SERVER == 4
    assert windows_activation_manager._COINIT_MULTITHREADED == 0
    assert windows_activation_manager._COINIT_DISABLE_OLE1DDE == 4
    assert windows_activation_manager._COM_INITIALIZATION_FLAGS == 4
    assert windows_activation_manager._ACTIVATE_APPLICATION_VTABLE_INDEX == 3
    iid = windows_activation_manager._guid(
        windows_activation_manager._IID_IAPPLICATION_ACTIVATION_MANAGER,
    )
    assert (iid.Data1, iid.Data2, iid.Data3) == (
        0x2E941141, 0x7F97, 0x4756,
    )
    assert bytes(iid.Data4) == bytes.fromhex("ba1d9decde894a3d")


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds


def primary_state(*, resolution: str = "unique", count: int = 1, foreground: bool = False):
    facts = EligibleWindowFacts(
        visible=True, minimized=False, enabled=True, cloaked_state="uncloaked",
        owner_present=False, root_owner_relationship="self", tool_window=False,
        app_window=False, has_nonzero_client_area=True, client_area_bucket="large",
        window_area_bucket="large", foreground=foreground,
        stable_across_probes=True, explicit_activation_eligible=True,
        eligibility_reason="eligible",
    )
    return TrustedApplicationRuntimeState(
        process_observed=True, window_observed=True, foreground_observed=foreground,
        visible=True, minimized=False, window_foreground=foreground,
        trusted_identity_match=True, probe_complete=True, window_stable=True,
        enumeration_complete=True, primary_surface_candidate_count=count,
        primary_surface_resolution_status=resolution,
        primary_surface_facts=facts if count == 1 and resolution == "unique" else None,
    )


class FakeComputer:
    def __init__(self, *, app_catalog, states, activation_result=None) -> None:
        self.app_catalog = app_catalog
        self.states = iter(states)
        self.activation_calls = 0
        self.activation_result = activation_result or TrustedWindowActivationResult(
            True, "eligible", mechanism="activate", budget_consumed=True,
            os_call_reported_success=False, foreground_verified=False,
            simple_attempt=SimpleActivationAttempt(False, False),
            failure_reason="os_activation_rejected",
        )
        self.observation = Observation("terminal.exe", "Terminal")

    def observe_local(self) -> Observation:
        return self.observation

    def activation_target_probe(self, _candidate):
        return lambda: next(self.states)

    def activate_trusted_application_window(self, _candidate):
        self.activation_calls += 1
        return self.activation_result


def test_diagnostic_requires_unique_primary_surface_before_one_activation_attempt() -> None:
    catalog, app_id = packaged_catalog()
    clock = FakeClock()
    computer = FakeComputer(
        app_catalog=catalog,
        states=[TrustedApplicationRuntimeState(), primary_state()],
    )
    result = run_packaged_activation_diagnostic(
        "WhatsApp", "shell", catalog,
        computer_factory=lambda **kwargs: computer,
        clock=clock, sleep_fn=clock.sleep,
    )

    assert result.trusted_app_id == app_id
    assert result.primary_surface_unique
    assert result.primary_window_stable
    assert result.explicit_foreground_attempted
    assert result.explicit_simple_setforeground_return is False
    assert not result.explicit_fallback_setforeground_attempted
    assert result.explicit_set_foreground_returned_successfully is False
    assert result.explicit_foreground_succeeded is False
    assert result.post_launch_setforeground_required is True
    assert result.timing is not None
    assert result.timing.launch_request_started_since_launch_ms == 0
    assert result.timing.launch_request_finished_since_launch_ms == result.activation_call_elapsed_ms
    assert result.timing.foreground_identity_before_launch.trusted_app_id is None
    assert result.timing.first_post_launch_observation_started_since_launch_ms is not None
    assert result.timing.first_post_launch_target_probe_started_since_launch_ms is not None
    assert computer.activation_calls == 1
    # FakeComputer exposes no keyboard or mouse API; the diagnostic only probes
    # and invokes the existing trusted-window activation method.
    assert not hasattr(computer, "click") and not hasattr(computer, "type_text")


def test_ambiguous_primary_surface_never_triggers_explicit_activation() -> None:
    catalog, _ = packaged_catalog()
    clock = FakeClock()
    ambiguous = primary_state(resolution="ambiguous", count=2)
    computer = FakeComputer(
        app_catalog=catalog,
        states=[TrustedApplicationRuntimeState(), ambiguous, ambiguous],
    )
    result = run_packaged_activation_diagnostic(
        "WhatsApp", "shell", catalog,
        computer_factory=lambda **kwargs: computer,
        timeout_seconds=0.1, poll_interval_seconds=0.05,
        clock=clock, sleep_fn=clock.sleep,
    )

    assert not result.primary_surface_unique
    assert not result.explicit_foreground_attempted
    assert result.failure_stage == "target_window_resolution"
    assert computer.activation_calls == 0


def test_restore_failure_is_not_misreported_as_setforeground_attempt() -> None:
    catalog, _ = packaged_catalog()
    clock = FakeClock()
    computer = FakeComputer(
        app_catalog=catalog,
        states=[TrustedApplicationRuntimeState(), primary_state()],
        activation_result=TrustedWindowActivationResult(
            True, "eligible", mechanism="restore", budget_consumed=True,
            foreground_verified=False,
            simple_attempt=SimpleActivationAttempt(None, False),
            failure_reason="restore_not_started",
        ),
    )

    result = run_packaged_activation_diagnostic(
        "WhatsApp", "shell", catalog,
        computer_factory=lambda **kwargs: computer,
        clock=clock, sleep_fn=clock.sleep,
    )

    assert result.explicit_activation_attempted
    assert not result.explicit_foreground_attempted
    assert result.explicit_simple_setforeground_return is None
    assert result.explicit_set_foreground_returned_successfully is None
    assert result.post_launch_setforeground_required is None


def test_fallback_setforeground_return_is_reported_separately_from_verification() -> None:
    catalog, _ = packaged_catalog()
    clock = FakeClock()
    computer = FakeComputer(
        app_catalog=catalog,
        states=[TrustedApplicationRuntimeState(), primary_state()],
        activation_result=TrustedWindowActivationResult(
            True, "eligible", mechanism="activate", budget_consumed=True,
            os_call_reported_success=False, foreground_verified=False,
            simple_attempt=SimpleActivationAttempt(False, False),
            fallback_attempt=ThreadInputFallbackAttempt(
                eligible=True, reason="verified", attach_attempted=True,
                attach_succeeded=True, set_foreground_attempted=True,
                set_foreground_return=True, detach_succeeded=True, verified=False,
            ),
            failure_reason="os_activation_rejected",
        ),
    )

    result = run_packaged_activation_diagnostic(
        "WhatsApp", "shell", catalog,
        computer_factory=lambda **kwargs: computer,
        clock=clock, sleep_fn=clock.sleep,
    )

    assert result.explicit_simple_setforeground_return is False
    assert result.explicit_fallback_setforeground_attempted
    assert result.explicit_fallback_setforeground_return is True
    assert result.explicit_set_foreground_returned_successfully is True
    assert result.explicit_foreground_succeeded is False


def test_automatic_foreground_is_separated_from_explicit_activation() -> None:
    catalog, app_id = packaged_catalog()
    clock = FakeClock()
    active_computer = FakeComputer(
        app_catalog=catalog,
        states=[TrustedApplicationRuntimeState(), primary_state(foreground=True)],
    )
    active_computer.observation = Observation(
        "WhatsApp.exe", "WhatsApp", application_id=app_id,
    )

    result = run_packaged_activation_diagnostic(
        "WhatsApp", "shell", catalog,
        computer_factory=lambda **kwargs: active_computer,
        clock=clock, sleep_fn=clock.sleep,
    )

    assert result.success
    assert result.automatic_foreground_observed
    assert result.automatic_foreground_elapsed_ms == 0
    assert result.post_launch_setforeground_required is False
    assert not result.explicit_foreground_attempted
    assert active_computer.activation_calls == 0


def test_diagnostic_result_schema_has_no_process_window_or_aumid_values() -> None:
    diagnostic = PackagedActivationDiagnostic(
        success=False, mechanism="activation-manager", trusted_app_id="app_abc",
        launch_kind="packaged", activation_request_succeeded=True,
        activation_call_elapsed_ms=5, returned_pid_present=True, activation_hresult=0,
        target_process_observed_before_request=False,
        target_window_observed_before_request=False,
        target_foreground_observed_before_request=False,
        target_process_observed=True, target_window_observed=True,
        primary_surface_unique=True, primary_window_visible=True,
        primary_window_stable=True, automatic_foreground_observed=False,
        automatic_foreground_elapsed_ms=None, post_launch_setforeground_required=True,
        explicit_activation_attempted=True,
        explicit_foreground_attempted=True,
        explicit_simple_setforeground_return=False,
        explicit_fallback_setforeground_attempted=False,
        explicit_fallback_setforeground_return=None,
        explicit_set_foreground_returned_successfully=False,
        explicit_foreground_succeeded=False, final_foreground_verified=False,
        failure_stage="foreground_activation", failure_reason="os_activation_rejected",
    )
    payload = asdict(diagnostic)
    serialized = repr(payload)

    assert payload["trusted_app_id"] == "app_abc"
    assert payload["returned_pid_present"] is True
    assert not {"hwnd", "pid", "aumid", "app_user_model_id"} & payload.keys()
    for private_value in (TRUSTED_AUMID, "12345", "987654"):
        assert private_value not in serialized


def test_cli_runs_exactly_the_selected_mechanism_and_prints_structured_result(
    monkeypatch, capsys,
) -> None:
    catalog, _ = packaged_catalog()
    expected = PackagedActivationDiagnostic(
        success=False, mechanism="activation-manager", trusted_app_id="app_abc",
        launch_kind="packaged", activation_request_succeeded=True,
        activation_call_elapsed_ms=5, returned_pid_present=True, activation_hresult=0,
        target_process_observed_before_request=False,
        target_window_observed_before_request=False,
        target_foreground_observed_before_request=False,
        target_process_observed=True, target_window_observed=True,
        primary_surface_unique=True, primary_window_visible=True,
        primary_window_stable=True, automatic_foreground_observed=False,
        automatic_foreground_elapsed_ms=None, post_launch_setforeground_required=True,
        explicit_activation_attempted=True,
        explicit_foreground_attempted=True,
        explicit_simple_setforeground_return=False,
        explicit_fallback_setforeground_attempted=False,
        explicit_fallback_setforeground_return=None,
        explicit_set_foreground_returned_successfully=False,
        explicit_foreground_succeeded=False, final_foreground_verified=False,
        failure_stage="foreground_activation", failure_reason="os_activation_rejected",
    )
    runner = Mock(return_value=expected)
    monkeypatch.setattr(cli, "WindowsApplicationCatalog", lambda: catalog)
    monkeypatch.setattr(cli, "run_packaged_activation_diagnostic", runner)

    exit_code = cli.main([
        "debug-packaged-activation", "WhatsApp", "--mechanism", "activation-manager",
    ])

    assert exit_code == 1
    runner.assert_called_once()
    assert runner.call_args.args[:2] == ("WhatsApp", "activation-manager")
    assert json.loads(capsys.readouterr().out)["mechanism"] == "activation-manager"
