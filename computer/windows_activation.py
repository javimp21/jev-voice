"""Catalog-bound Windows foreground activation for trusted OpenApp actions."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import sys
from typing import Any, Protocol

from computer.applications import (
    ActivationWindowCandidateDiagnostics,
    ApplicationCandidate,
    ApplicationCatalog,
    EligibleWindowDiagnostics,
    EligibleWindowFacts,
    EligibleWindowRelationship,
    SimpleActivationAttempt,
    ThreadInputFallbackAttempt,
    TrustedApplicationRuntimeState,
    TrustedWindowActivationResult,
    WindowRejectionReason,
    WindowResolutionStatus,
)


MAX_ACTIVATION_WINDOWS = 512
MAX_DIAGNOSTIC_CANDIDATES = 8
MAX_ELIGIBLE_DIAGNOSTIC_CANDIDATES = 8
_SYSTEM_WINDOW_CLASSES = frozenset({
    "progman", "workerw", "shell_traywnd", "shell_secondarytraywnd", "shell_defview",
})
_DWMWA_CLOAKED = 14


def _dwm_cloaked(handle: int) -> bool | None:
    """Return DWM cloak status when that optional API is available."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        dwmapi = ctypes.WinDLL("dwmapi", use_last_error=True)
        get_attribute = dwmapi.DwmGetWindowAttribute
        get_attribute.argtypes = (
            wintypes.HWND, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), wintypes.DWORD,
        )
        get_attribute.restype = ctypes.c_long
        value = wintypes.DWORD()
        result = get_attribute(
            handle, _DWMWA_CLOAKED, ctypes.byref(value), ctypes.sizeof(value),
        )
        return bool(value.value) if result == 0 else None
    except Exception:
        return None


def _area_bucket(rectangle: object) -> str:
    """Convert a Win32 rectangle into a coarse area category only."""
    if not isinstance(rectangle, (tuple, list)) or len(rectangle) != 4:
        return "unavailable"
    try:
        left, top, right, bottom = (int(value) for value in rectangle)
    except (TypeError, ValueError, OverflowError):
        return "unavailable"
    width = max(0, right - left)
    height = max(0, bottom - top)
    area = width * height
    if area == 0:
        return "zero"
    if area < 150_000:
        return "small"
    if area < 750_000:
        return "medium"
    return "large"


class _WindowActivationApi(Protocol):
    def show_window_async(self, handle: int, command: int) -> bool: ...
    def set_foreground_window(self, handle: int) -> bool: ...
    def current_thread_id(self) -> int: ...
    def attach_thread_input(
        self, attach_thread_id: int, target_thread_id: int, attach: bool,
    ) -> bool: ...


class _Win32WindowActivationApi:
    """Small ctypes binding for User32 APIs absent from this PyWin32 build."""

    def __init__(self) -> None:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.WinDLL("user32", use_last_error=True)
        self._show_window_async = user32.ShowWindowAsync
        self._show_window_async.argtypes = (wintypes.HWND, wintypes.INT)
        self._show_window_async.restype = wintypes.BOOL
        self._set_foreground_window = user32.SetForegroundWindow
        self._set_foreground_window.argtypes = (wintypes.HWND,)
        self._set_foreground_window.restype = wintypes.BOOL
        self._attach_thread_input = user32.AttachThreadInput
        self._attach_thread_input.argtypes = (wintypes.DWORD, wintypes.DWORD, wintypes.BOOL)
        self._attach_thread_input.restype = wintypes.BOOL
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._get_current_thread_id = kernel32.GetCurrentThreadId
        self._get_current_thread_id.argtypes = ()
        self._get_current_thread_id.restype = wintypes.DWORD

    def show_window_async(self, handle: int, command: int) -> bool:
        return bool(self._show_window_async(handle, command))

    def set_foreground_window(self, handle: int) -> bool:
        return bool(self._set_foreground_window(handle))

    def current_thread_id(self) -> int:
        return int(self._get_current_thread_id())

    def attach_thread_input(
        self, attach_thread_id: int, target_thread_id: int, attach: bool,
    ) -> bool:
        return bool(self._attach_thread_input(
            attach_thread_id, target_thread_id, bool(attach),
        ))


@dataclass(frozen=True, slots=True)
class _TrustedWindow:
    """Private HWND binding. This record never leaves this Windows module."""

    handle: int
    process_id: int
    trusted_identity_match: bool
    exists: bool
    visible: bool
    minimized: bool
    foreground: bool
    enabled: bool | None
    cloaked: bool | None
    owner_present: bool | None
    root_owner_relationship: str
    tool_window: bool | None
    app_window: bool | None
    has_nonzero_client_area: bool | None
    client_area_bucket: str
    window_area_bucket: str
    owner_handle: int | None
    root_owner_handle: int | None


@dataclass(frozen=True, slots=True)
class _ActivationWindowResolution:
    status: WindowResolutionStatus
    eligible_windows: tuple[_TrustedWindow, ...]
    rejection_reason_counts: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class _PrimarySurfaceResolution:
    status: WindowResolutionStatus
    primary_windows: tuple[_TrustedWindow, ...]
    tool_window_count: int
    classification_incomplete: bool


def _activation_window_rejection_reason(
    window: _TrustedWindow,
) -> WindowRejectionReason | None:
    """Apply the single hard structural eligibility rule for explicit activation."""
    if not window.trusted_identity_match:
        return "trusted_identity_mismatch"
    if not window.exists:
        return "window_missing"
    if window.visible is not True:
        return "target_not_visible"
    if window.cloaked is None:
        return "cloaking_unknown"
    if window.cloaked:
        return "target_cloaked"
    if window.has_nonzero_client_area is None:
        return "client_area_unknown"
    if not window.has_nonzero_client_area:
        return "zero_client_area"
    return None


def _resolve_activation_windows(
    matches: list[_TrustedWindow], *, enumeration_complete: bool,
) -> _ActivationWindowResolution:
    """Resolve the full internal trusted-window scan without ranking candidates."""
    eligible: list[_TrustedWindow] = []
    rejected = Counter()
    for window in matches:
        reason = _activation_window_rejection_reason(window)
        if reason is None:
            eligible.append(window)
        else:
            rejected[reason] += 1
    if not enumeration_complete:
        status: WindowResolutionStatus = "incomplete"
    elif not eligible:
        status = "none"
    elif len(eligible) == 1:
        status = "unique"
    else:
        status = "ambiguous"
    return _ActivationWindowResolution(
        status, tuple(eligible), tuple(sorted(rejected.items())),
    )


def _resolve_primary_surfaces(
    base_eligible_windows: tuple[_TrustedWindow, ...], *, enumeration_complete: bool,
) -> _PrimarySurfaceResolution:
    """Partition eligible windows by tool-window style without ranking them."""
    primary = tuple(window for window in base_eligible_windows if window.tool_window is False)
    tool_count = sum(window.tool_window is True for window in base_eligible_windows)
    classification_incomplete = any(window.tool_window is None for window in base_eligible_windows)
    if not enumeration_complete or classification_incomplete:
        status: WindowResolutionStatus = "incomplete"
    elif not primary:
        status = "none"
    elif len(primary) == 1:
        status = "unique"
    else:
        status = "ambiguous"
    return _PrimarySurfaceResolution(
        status, primary, tool_count, classification_incomplete,
    )


class WindowsTrustedApplicationProbe:
    """Read and activate only a unique top-level window for one catalog app.

    This enumerates bounded top-level windows on the current interactive desktop,
    not the system process table. Raw HWNDs and process IDs remain private here.
    """

    def __init__(
        self,
        app_catalog: ApplicationCatalog,
        candidate: ApplicationCandidate,
        *,
        activation_api: _WindowActivationApi | None = None,
    ) -> None:
        self._app_catalog = app_catalog
        self._candidate = candidate
        self._checked_process_ids: set[int] = set()
        self._trusted_process_ids: set[int] = set()
        self._previous_unique_handle: int | None = None
        self._previous_eligible_handles: set[int] = set()
        self._previous_resolution_status: WindowResolutionStatus = "incomplete"
        self._previous_primary_handle: int | None = None
        self._previous_primary_handles: set[int] = set()
        self._previous_primary_resolution_status: WindowResolutionStatus = "incomplete"
        self._diagnostic_ids_by_handle: dict[int, str] = {}
        self._next_diagnostic_id = 1
        self._activation_consumed = False
        self._activation_api = activation_api

    def _reset_process_identity_cache(self) -> None:
        self._checked_process_ids.clear()
        self._trusted_process_ids.clear()

    def _matches_process(self, process_id: int) -> bool:
        if process_id in self._checked_process_ids:
            return process_id in self._trusted_process_ids
        self._checked_process_ids.add(process_id)
        from computer.windows import _package_family_name, _process_name

        process_name = _process_name(process_id) if self._candidate.process_names else ""
        package_family = (
            _package_family_name(process_id) if self._candidate.package_family else ""
        )
        try:
            trusted = self._app_catalog.identify(process_name, package_family) == self._candidate.id
        except Exception:
            trusted = False
        if trusted:
            self._trusted_process_ids.add(process_id)
        return trusted

    @staticmethod
    def _optional_value(callback: Any, *args: object) -> Any:
        try:
            return callback(*args)
        except Exception:
            return None

    def _window_structure(
        self, handle: int, win32gui: Any, win32con: Any,
    ) -> tuple[bool | None, bool | None, bool | None, str, bool | None,
               bool | None, bool | None, str, str, int | None, int | None]:
        enabled_raw = self._optional_value(getattr(win32gui, "IsWindowEnabled", None), handle)
        enabled = bool(enabled_raw) if enabled_raw is not None else None
        cloaked = _dwm_cloaked(handle)

        owner_raw = self._optional_value(
            getattr(win32gui, "GetWindow", None), handle, getattr(win32con, "GW_OWNER", 4),
        )
        owner_present = bool(owner_raw) if owner_raw is not None else None
        owner_handle = (
            owner_raw if type(owner_raw) is int and owner_raw > 0 else None
        )

        root_raw = self._optional_value(
            getattr(win32gui, "GetAncestor", None), handle,
            getattr(win32con, "GA_ROOTOWNER", 3),
        )
        root_relationship = (
            "unavailable" if root_raw is None or not root_raw
            else "self" if root_raw == handle
            else "other"
        )
        root_owner_handle = (
            root_raw if type(root_raw) is int and root_raw > 0 else None
        )

        extended_style = self._optional_value(
            getattr(win32gui, "GetWindowLong", None), handle,
            getattr(win32con, "GWL_EXSTYLE", -20),
        )
        if extended_style is None:
            tool_window = app_window = None
        else:
            tool_window = bool(extended_style & getattr(win32con, "WS_EX_TOOLWINDOW", 0x80))
            app_window = bool(extended_style & getattr(win32con, "WS_EX_APPWINDOW", 0x40000))

        client_rect = self._optional_value(getattr(win32gui, "GetClientRect", None), handle)
        client_bucket = _area_bucket(client_rect)
        has_nonzero_client_area = (
            None if client_bucket == "unavailable" else client_bucket != "zero"
        )
        window_rect = self._optional_value(getattr(win32gui, "GetWindowRect", None), handle)
        window_bucket = _area_bucket(window_rect)
        return (
            enabled, cloaked, owner_present, root_relationship, tool_window, app_window,
            has_nonzero_client_area, client_bucket, window_bucket, owner_handle,
            root_owner_handle,
        )

    @staticmethod
    def _z_order_buckets(matches: list[_TrustedWindow], win32gui: Any, win32con: Any) -> dict[int, str]:
        """Bucket trusted candidates by relative Z order, without exposing other windows."""
        candidate_handles = {window.handle for window in matches}
        result: dict[int, str] = {}
        command = getattr(win32con, "GW_HWNDPREV", 3)
        for window in matches:
            above_count = 0
            visited = {window.handle}
            try:
                previous = win32gui.GetWindow(window.handle, command)
                steps = 0
                while previous:
                    if previous in visited or steps >= MAX_ACTIVATION_WINDOWS:
                        raise RuntimeError("Z-order traversal was incomplete.")
                    visited.add(previous)
                    steps += 1
                    if previous in candidate_handles:
                        above_count += 1
                    previous = win32gui.GetWindow(previous, command)
            except Exception:
                result[window.handle] = "unavailable"
                continue
            if above_count == 0:
                result[window.handle] = "front"
            elif above_count >= len(matches) - 1:
                result[window.handle] = "back"
            else:
                result[window.handle] = "middle"
        return result

    def _scan_windows(self) -> tuple[list[_TrustedWindow], bool, int, Any, Any]:
        if sys.platform != "win32":
            raise OSError("Windows activation diagnostics require Windows.")
        import win32gui
        import win32con
        import win32process

        self._reset_process_identity_cache()
        foreground_handle = win32gui.GetForegroundWindow()
        matches: list[_TrustedWindow] = []
        inspected_windows = 0
        scan_complete = True

        def inspect_window(handle: int, _context: object) -> bool:
            nonlocal inspected_windows, scan_complete
            inspected_windows += 1
            if inspected_windows > MAX_ACTIVATION_WINDOWS:
                scan_complete = False
                return False
            try:
                if not win32gui.IsWindow(handle):
                    return True
                _thread_id, process_id = win32process.GetWindowThreadProcessId(handle)
                if not self._matches_process(process_id):
                    return True
                class_name = win32gui.GetClassName(handle).casefold()
                # Do not treat the desktop, taskbar, or shell view as an app window.
                if class_name in _SYSTEM_WINDOW_CLASSES:
                    return True
                structure = self._window_structure(handle, win32gui, win32con)
                matches.append(_TrustedWindow(
                    handle=handle,
                    process_id=process_id,
                    trusted_identity_match=True,
                    exists=True,
                    visible=bool(win32gui.IsWindowVisible(handle)),
                    minimized=bool(win32gui.IsIconic(handle)),
                    foreground=handle == foreground_handle,
                    enabled=structure[0],
                    cloaked=structure[1],
                    owner_present=structure[2],
                    root_owner_relationship=structure[3],
                    tool_window=structure[4],
                    app_window=structure[5],
                    has_nonzero_client_area=structure[6],
                    client_area_bucket=structure[7],
                    window_area_bucket=structure[8],
                    owner_handle=structure[9],
                    root_owner_handle=structure[10],
                ))
            except Exception:
                # An inaccessible window makes uniqueness inconclusive.
                scan_complete = False
            return True

        enumeration_result = win32gui.EnumWindows(inspect_window, None)
        if enumeration_result is False:
            scan_complete = False
        return matches, scan_complete, foreground_handle, win32gui, win32process

    @staticmethod
    def _candidate_diagnostics(
        matches: list[_TrustedWindow],
        scan_complete: bool,
        previous_eligible_handles: set[int],
        z_order: dict[int, str],
    ) -> tuple[ActivationWindowCandidateDiagnostics, ...]:
        diagnostics: list[ActivationWindowCandidateDiagnostics] = []
        for index, window in enumerate(matches[:MAX_DIAGNOSTIC_CANDIDATES], start=1):
            rejection_reason = _activation_window_rejection_reason(window)
            explicitly_eligible = rejection_reason is None
            diagnostics.append(ActivationWindowCandidateDiagnostics(
                candidate_index=index,
                trusted_identity_match=window.trusted_identity_match,
                visible=window.visible,
                minimized=window.minimized,
                enabled=window.enabled,
                cloaked_if_available=window.cloaked,
                owner_present=window.owner_present,
                root_owner_relationship=window.root_owner_relationship,  # type: ignore[arg-type]
                tool_window=window.tool_window,
                app_window=window.app_window,
                has_nonzero_client_area=window.has_nonzero_client_area,
                client_area_bucket=window.client_area_bucket,  # type: ignore[arg-type]
                window_area_bucket=window.window_area_bucket,  # type: ignore[arg-type]
                foreground=window.foreground,
                stable_across_probes=(
                    scan_complete and explicitly_eligible
                    and window.handle in previous_eligible_handles
                ),
                z_order_bucket=z_order.get(window.handle, "unavailable"),  # type: ignore[arg-type]
                # Compatibility field: this still means a trusted process match.
                activation_candidate=True,
                explicit_activation_eligible=explicitly_eligible,
                rejection_reason=rejection_reason,
            ))
        return tuple(diagnostics)

    def _eligible_window_diagnostics(
        self,
        windows: tuple[_TrustedWindow, ...],
        scan_complete: bool,
        previous_eligible_handles: set[int],
        z_order: dict[int, str],
        currently_matching_handles: set[int],
    ) -> tuple[
        tuple[EligibleWindowDiagnostics, ...], bool,
        tuple[EligibleWindowRelationship, ...],
    ]:
        """Build a separately bounded, handle-free view of eligible candidates."""
        self._diagnostic_ids_by_handle = {
            handle: diagnostic_id
            for handle, diagnostic_id in self._diagnostic_ids_by_handle.items()
            if handle in currently_matching_handles
        }
        ids_by_handle: dict[int, str] = {}
        for window in windows:
            diagnostic_id = self._diagnostic_ids_by_handle.get(window.handle)
            if diagnostic_id is None:
                diagnostic_id = f"ew{self._next_diagnostic_id}"
                self._next_diagnostic_id += 1
                self._diagnostic_ids_by_handle[window.handle] = diagnostic_id
            ids_by_handle[window.handle] = diagnostic_id

        diagnostics: list[EligibleWindowDiagnostics] = []
        displayed_windows = windows[:MAX_ELIGIBLE_DIAGNOSTIC_CANDIDATES]
        for index, window in enumerate(displayed_windows, start=1):
            cloaked_state = (
                "unknown" if window.cloaked is None
                else "cloaked" if window.cloaked else "uncloaked"
            )
            diagnostics.append(EligibleWindowDiagnostics(
                diagnostic_id=ids_by_handle[window.handle],
                eligible_candidate_index=index,
                facts=EligibleWindowFacts(
                    visible=window.visible,
                    minimized=window.minimized,
                    enabled=window.enabled,
                    cloaked_state=cloaked_state,
                    owner_present=window.owner_present,
                    root_owner_relationship=window.root_owner_relationship,  # type: ignore[arg-type]
                    tool_window=window.tool_window,
                    app_window=window.app_window,
                    has_nonzero_client_area=window.has_nonzero_client_area,
                    client_area_bucket=window.client_area_bucket,  # type: ignore[arg-type]
                    window_area_bucket=window.window_area_bucket,  # type: ignore[arg-type]
                    foreground=window.foreground,
                    stable_across_probes=(
                        scan_complete and window.handle in previous_eligible_handles
                    ),
                    explicit_activation_eligible=True,
                    eligibility_reason="eligible",
                    z_order_bucket=z_order.get(window.handle, "unavailable"),  # type: ignore[arg-type]
                ),
                surface_class=(
                    "tool" if window.tool_window is True
                    else "primary" if window.tool_window is False else "unknown"
                ),
            ))
        relationships: list[EligibleWindowRelationship] = []
        for first_index, first in enumerate(displayed_windows):
            for second in displayed_windows[first_index + 1:]:
                if (first.owner_handle == second.handle
                        or second.owner_handle == first.handle):
                    relationship = "one_owned_by_other"
                elif (first.root_owner_handle is None or second.root_owner_handle is None):
                    relationship = "unknown"
                elif first.root_owner_handle == second.root_owner_handle:
                    relationship = "same_root_owner"
                elif first.owner_present is None or second.owner_present is None:
                    relationship = "unknown"
                else:
                    relationship = "independent"
                relationships.append(EligibleWindowRelationship(
                    first_diagnostic_id=ids_by_handle[first.handle],
                    second_diagnostic_id=ids_by_handle[second.handle],
                    relationship=relationship,
                ))
        return (
            tuple(diagnostics),
            len(windows) > MAX_ELIGIBLE_DIAGNOSTIC_CANDIDATES,
            tuple(relationships),
        )

    def __call__(self) -> TrustedApplicationRuntimeState:
        matches, scan_complete, _foreground_handle, _win32gui, _win32process = (
            self._scan_windows()
        )
        try:
            import win32con
            z_order = self._z_order_buckets(matches, _win32gui, win32con)
        except Exception:
            z_order = {window.handle: "unavailable" for window in matches}
        candidates = self._candidate_diagnostics(
            matches, scan_complete, self._previous_eligible_handles, z_order,
        )
        truncated = len(matches) > MAX_DIAGNOSTIC_CANDIDATES
        resolution = _resolve_activation_windows(
            matches, enumeration_complete=scan_complete,
        )
        primary_resolution = _resolve_primary_surfaces(
            resolution.eligible_windows, enumeration_complete=scan_complete,
        )
        previous_unique_handle = self._previous_unique_handle
        previous_eligible_handles = self._previous_eligible_handles
        previous_primary_handles = self._previous_primary_handles
        eligible_diagnostics, eligible_truncated, relationships = (
            self._eligible_window_diagnostics(
                resolution.eligible_windows, scan_complete, previous_eligible_handles,
                z_order, {window.handle for window in matches},
            )
        )
        eligible_handles = {window.handle for window in resolution.eligible_windows}
        self._previous_eligible_handles = eligible_handles if scan_complete else set()
        self._previous_resolution_status = resolution.status
        self._previous_unique_handle = (
            resolution.eligible_windows[0].handle
            if resolution.status == "unique" else None
        )
        primary_facts = None
        if primary_resolution.status == "unique":
            primary_window = primary_resolution.primary_windows[0]
            cloaked_state = (
                "unknown" if primary_window.cloaked is None
                else "cloaked" if primary_window.cloaked else "uncloaked"
            )
            primary_facts = EligibleWindowFacts(
                visible=primary_window.visible,
                minimized=primary_window.minimized,
                enabled=primary_window.enabled,
                cloaked_state=cloaked_state,
                owner_present=primary_window.owner_present,
                root_owner_relationship=primary_window.root_owner_relationship,  # type: ignore[arg-type]
                tool_window=False,
                app_window=primary_window.app_window,
                has_nonzero_client_area=primary_window.has_nonzero_client_area,
                client_area_bucket=primary_window.client_area_bucket,  # type: ignore[arg-type]
                window_area_bucket=primary_window.window_area_bucket,  # type: ignore[arg-type]
                foreground=primary_window.foreground,
                stable_across_probes=(
                    scan_complete and primary_window.handle in previous_primary_handles
                ),
                explicit_activation_eligible=True,
                eligibility_reason="eligible",
                z_order_bucket=z_order.get(primary_window.handle, "unavailable"),  # type: ignore[arg-type]
            )
        if scan_complete and not primary_resolution.classification_incomplete:
            self._previous_primary_handles = {
                window.handle for window in primary_resolution.primary_windows
            }
        else:
            self._previous_primary_handles = set()
        self._previous_primary_resolution_status = primary_resolution.status
        self._previous_primary_handle = (
            primary_resolution.primary_windows[0].handle
            if primary_resolution.status == "unique" else None
        )
        matching_count = len(matches)
        eligible_count = len(resolution.eligible_windows)
        common_diagnostics = dict(
            process_observed=bool(matches),
            window_observed=bool(matches),
            foreground_observed=any(window.foreground for window in matches),
            trusted_identity_match=bool(matches),
            probe_complete=scan_complete,
            window_ambiguous=resolution.status == "ambiguous",
            window_resolution_status=resolution.status,
            matching_trusted_window_count=matching_count,
            eligible_window_count=eligible_count,
            enumeration_complete=scan_complete,
            rejection_reason_counts=resolution.rejection_reason_counts,
            window_candidates=candidates,
            candidate_diagnostics_truncated=truncated,
            eligible_window_diagnostics=eligible_diagnostics,
            eligible_window_diagnostics_truncated=eligible_truncated,
            eligible_window_relationships=relationships,
            base_eligible_window_count=eligible_count,
            primary_surface_candidate_count=len(primary_resolution.primary_windows),
            tool_surface_candidate_count=primary_resolution.tool_window_count,
            primary_surface_resolution_status=primary_resolution.status,
            primary_surface_facts=primary_facts,
        )
        if resolution.status != "unique":
            return TrustedApplicationRuntimeState(
                visible=None,
                minimized=None,
                window_foreground=None,
                window_stable=False,
                **common_diagnostics,
            )

        window = resolution.eligible_windows[0]
        stable = previous_unique_handle == window.handle and scan_complete
        return TrustedApplicationRuntimeState(
            visible=window.visible,
            minimized=window.minimized,
            window_foreground=window.foreground,
            window_stable=stable,
            **common_diagnostics,
        )

    def activate(self, candidate: ApplicationCandidate) -> TrustedWindowActivationResult:
        """Re-resolve the unique eligible window, then make at most one OS attempt."""
        if self._activation_consumed:
            return TrustedWindowActivationResult(
                False, "activation_error", failure_reason="activation_budget_consumed",
                budget_consumed=True,
            )
        if candidate.id != self._candidate.id:
            return TrustedWindowActivationResult(
                False, "trusted_identity_mismatch", trusted_identity_match=False,
                budget_consumed=False,
            )
        stable_handle = self._previous_primary_handle
        if stable_handle is None:
            prior_reason = (
                "target_window_ambiguous"
                if self._previous_primary_resolution_status == "ambiguous"
                else "probe_incomplete"
                if self._previous_primary_resolution_status == "incomplete"
                else "no_eligible_window"
                if self._previous_primary_resolution_status == "none"
                else "target_window_unstable"
            )
            return TrustedWindowActivationResult(
                False, prior_reason, trusted_identity_match=True,
                budget_consumed=False,
            )

        try:
            matches, scan_complete, _foreground_handle, win32gui, win32process = (
                self._scan_windows()
            )
        except KeyboardInterrupt:
            raise
        except Exception:
            return TrustedWindowActivationResult(
                False, "probe_incomplete", trusted_identity_match=False,
                budget_consumed=False,
            )

        base_resolution = _resolve_activation_windows(
            matches, enumeration_complete=scan_complete,
        )
        resolution = _resolve_primary_surfaces(
            base_resolution.eligible_windows, enumeration_complete=scan_complete,
        )
        if resolution.status == "incomplete":
            return TrustedWindowActivationResult(
                False, "probe_incomplete", trusted_identity_match=False,
                budget_consumed=False,
            )
        if resolution.status == "none":
            selected_match = next((item for item in matches if item.handle == stable_handle), None)
            if selected_match is None:
                reason = "target_window_missing" if not matches else "stale_window"
                visible = minimized = None
            else:
                reason = (
                    _activation_window_rejection_reason(selected_match)
                    or "no_eligible_window"
                )
                visible, minimized = selected_match.visible, selected_match.minimized
            return TrustedWindowActivationResult(
                False, reason, visible, minimized, trusted_identity_match=bool(matches),
                budget_consumed=False,
            )
        if resolution.status == "ambiguous":
            return TrustedWindowActivationResult(
                False, "target_window_ambiguous", trusted_identity_match=True,
                budget_consumed=False,
            )

        window = resolution.primary_windows[0]
        if window.handle != stable_handle:
            return TrustedWindowActivationResult(
                False, "stale_window", window.visible, window.minimized, True,
                budget_consumed=False,
            )
        try:
            currently_exists = bool(win32gui.IsWindow(window.handle))
        except KeyboardInterrupt:
            raise
        except Exception:
            return TrustedWindowActivationResult(
                False, "probe_incomplete", window.visible, window.minimized, True,
                budget_consumed=False,
            )
        if not currently_exists:
            return TrustedWindowActivationResult(
                False, "stale_window", window.visible, window.minimized, True,
                budget_consumed=False,
            )

        # Re-read trusted identity and required structural facts immediately before
        # the OS call. Owner/style/area buckets remain diagnostic only.
        try:
            import win32con

            selected_thread_id, current_process_id = win32process.GetWindowThreadProcessId(window.handle)
            self._checked_process_ids.discard(current_process_id)
            self._trusted_process_ids.discard(current_process_id)
            identity_matches = (
                current_process_id == window.process_id
                and self._matches_process(current_process_id)
            )
            current_class = win32gui.GetClassName(window.handle).casefold()
            still_exists = bool(win32gui.IsWindow(window.handle))
            visible = bool(win32gui.IsWindowVisible(window.handle))
            minimized = bool(win32gui.IsIconic(window.handle))
            current_foreground = window.handle == win32gui.GetForegroundWindow()
            structure = self._window_structure(window.handle, win32gui, win32con)
        except KeyboardInterrupt:
            raise
        except Exception:
            return TrustedWindowActivationResult(
                False, "probe_incomplete", window.visible, window.minimized, False,
                budget_consumed=False,
            )

        if type(selected_thread_id) is not int or selected_thread_id <= 0:
            return TrustedWindowActivationResult(
                False, "probe_incomplete", visible, minimized, True,
                failure_reason="selected_thread_unresolved", budget_consumed=False,
                activation_strategy="simple",
                fallback_attempt=ThreadInputFallbackAttempt(
                    eligible=False, reason="selected_thread_unresolved",
                    selected_thread_resolved=False,
                    failure_reason="selected_thread_unresolved",
                ),
            )

        if not identity_matches:
            return TrustedWindowActivationResult(
                False, "trusted_identity_mismatch", visible, minimized, False,
                budget_consumed=False,
            )
        if current_class in _SYSTEM_WINDOW_CLASSES:
            return TrustedWindowActivationResult(
                False, "system_surface", visible, minimized, True,
                budget_consumed=False,
            )
        if not still_exists:
            return TrustedWindowActivationResult(
                False, "stale_window", visible, minimized, True,
                budget_consumed=False,
            )
        current_window = _TrustedWindow(
            handle=window.handle,
            process_id=current_process_id,
            trusted_identity_match=True,
            exists=True,
            visible=visible,
            minimized=minimized,
            foreground=current_foreground,
            enabled=structure[0],
            cloaked=structure[1],
            owner_present=structure[2],
            root_owner_relationship=structure[3],
            tool_window=structure[4],
            app_window=structure[5],
            has_nonzero_client_area=structure[6],
            client_area_bucket=structure[7],
            window_area_bucket=structure[8],
            owner_handle=structure[9],
            root_owner_handle=structure[10],
        )
        current_base_resolution = _resolve_activation_windows(
            [current_window], enumeration_complete=True,
        )
        current_resolution = _resolve_primary_surfaces(
            tuple(current_base_resolution.eligible_windows), enumeration_complete=True,
        )
        if current_resolution.status != "unique" or current_window.tool_window is not False:
            reason = (
                "target_window_ambiguous"
                if current_resolution.status == "ambiguous"
                else "probe_incomplete" if current_resolution.status == "incomplete"
                else _activation_window_rejection_reason(current_window) or "no_eligible_window"
            )
            return TrustedWindowActivationResult(
                False, reason, visible, minimized, True,
                budget_consumed=False,
            )
        if current_foreground:
            return TrustedWindowActivationResult(
                False, "already_foreground", visible, minimized, True,
                "already_foreground", None, None, False, True,
            )

        mechanism = "restore_then_activate" if minimized else "activate"
        try:
            activation_api = self._activation_api or _Win32WindowActivationApi()
        except KeyboardInterrupt:
            raise
        except Exception:
            return TrustedWindowActivationResult(
                True, "eligible", visible, minimized, True, None, None,
                "activation_error", False,
            )

        if minimized:
            try:
                # Consume only after the exact selected window was revalidated.
                self._activation_consumed = True
                restore_started = activation_api.show_window_async(
                    window.handle, win32con.SW_RESTORE,
                )
            except KeyboardInterrupt:
                raise
            except Exception:
                restore_started = False
            if not restore_started:
                return TrustedWindowActivationResult(
                    True, "eligible", visible, minimized, True,
                    "restore", False, "restore_not_started", True,
                    activation_strategy="simple",
                    simple_attempt=SimpleActivationAttempt(None, False),
                    fallback_attempt=ThreadInputFallbackAttempt(
                        eligible=False, reason="simple_restore_failed",
                        failure_reason="simple_restore_failed",
                    ),
                )

        # First and only simple attempt. Its return value is diagnostic; a fresh
        # exact-HWND observation decides whether a second strategy is warranted.
        try:
            if not self._activation_consumed:
                self._activation_consumed = True
            simple_set_foreground_return = bool(
                activation_api.set_foreground_window(window.handle)
            )
        except KeyboardInterrupt:
            raise
        except Exception:
            simple_set_foreground_return = None
        simple_verified, simple_verification_failure = self._verify_selected_foreground(
            window.handle, window.process_id, selected_thread_id,
            win32gui, win32process, win32con,
        )
        simple_attempt = SimpleActivationAttempt(
            simple_set_foreground_return, simple_verified,
        )
        if simple_verified is True:
            return TrustedWindowActivationResult(
                True, "eligible", visible, minimized, True, mechanism,
                simple_set_foreground_return, None, True, True,
                activation_strategy="simple",
                simple_attempt=simple_attempt,
                fallback_attempt=ThreadInputFallbackAttempt(
                    eligible=False, reason="simple_verified",
                ),
            )

        if simple_verified is None:
            return TrustedWindowActivationResult(
                True, "eligible", visible, minimized, True, mechanism,
                simple_set_foreground_return,
                simple_verification_failure or "foreground_verification_unavailable",
                True, None,
                activation_strategy="simple",
                simple_attempt=simple_attempt,
                fallback_attempt=ThreadInputFallbackAttempt(
                    eligible=False, reason="foreground_verification_unavailable",
                    selected_thread_resolved=True,
                    failure_reason="foreground_verification_unavailable",
                ),
            )

        fallback_attempt, fallback_verified, fallback_failure = (
            self._run_thread_input_fallback(
                window.handle,
                window.process_id,
                selected_thread_id,
                activation_api,
                win32gui,
                win32process,
                win32con,
            )
        )
        used_fallback = fallback_attempt.attach_attempted
        activation_strategy = "attach_thread_input" if used_fallback else "simple"
        detach_succeeded = fallback_attempt.detach_succeeded
        selected_verified = fallback_verified if used_fallback else simple_verified
        failure_reason = (
            fallback_failure if used_fallback else simple_verification_failure
        )
        if selected_verified is True and detach_succeeded is not False:
            failure_reason = None
        elif used_fallback and fallback_failure is not None:
            failure_reason = fallback_failure
        return TrustedWindowActivationResult(
            True,
            "eligible",
            visible,
            minimized,
            True,
            mechanism,
            simple_set_foreground_return,
            failure_reason,
            True,
            selected_verified,
            activation_strategy=activation_strategy,
            simple_attempt=simple_attempt,
            fallback_attempt=fallback_attempt,
        )

    def _bound_selected_window_status(
        self,
        handle: int,
        expected_process_id: int,
        expected_thread_id: int,
        win32gui: Any,
        win32process: Any,
        win32con: Any,
    ) -> tuple[bool, str | None, int | None]:
        """Revalidate only the previously selected primary HWND; never resolve a replacement."""
        if self._previous_primary_handle != handle:
            return False, "selected_window_changed", None
        try:
            if not bool(win32gui.IsWindow(handle)):
                return False, "target_window_missing", None
            thread_id_raw, process_id_raw = win32process.GetWindowThreadProcessId(handle)
            if type(thread_id_raw) is not int or thread_id_raw <= 0:
                return False, "selected_thread_unresolved", None
            if type(process_id_raw) is not int or process_id_raw <= 0:
                return False, "selected_target_revalidation_failed", thread_id_raw
            if thread_id_raw != expected_thread_id:
                return False, "selected_window_changed", thread_id_raw
            if process_id_raw != expected_process_id:
                return False, "trusted_identity_mismatch", thread_id_raw
            self._checked_process_ids.discard(process_id_raw)
            self._trusted_process_ids.discard(process_id_raw)
            if not self._matches_process(process_id_raw):
                return False, "trusted_identity_mismatch", thread_id_raw
            class_name = win32gui.GetClassName(handle).casefold()
            if class_name in _SYSTEM_WINDOW_CLASSES:
                return False, "selected_primary_surface_changed", thread_id_raw
            visible = bool(win32gui.IsWindowVisible(handle))
            structure = self._window_structure(handle, win32gui, win32con)
            current = _TrustedWindow(
                handle=handle,
                process_id=process_id_raw,
                trusted_identity_match=True,
                exists=True,
                visible=visible,
                minimized=bool(win32gui.IsIconic(handle)),
                foreground=handle == win32gui.GetForegroundWindow(),
                enabled=structure[0],
                cloaked=structure[1],
                owner_present=structure[2],
                root_owner_relationship=structure[3],
                tool_window=structure[4],
                app_window=structure[5],
                has_nonzero_client_area=structure[6],
                client_area_bucket=structure[7],
                window_area_bucket=structure[8],
                owner_handle=structure[9],
                root_owner_handle=structure[10],
            )
            if not visible:
                return False, "target_not_visible", thread_id_raw
            if current.tool_window is not False:
                return False, "selected_primary_surface_changed", thread_id_raw
            rejection = _activation_window_rejection_reason(current)
            if rejection is not None:
                return False, rejection, thread_id_raw
            return True, None, thread_id_raw
        except KeyboardInterrupt:
            raise
        except Exception:
            return False, "selected_window_changed", None

    def _verify_selected_foreground(
        self,
        handle: int,
        expected_process_id: int,
        expected_thread_id: int,
        win32gui: Any,
        win32process: Any,
        win32con: Any,
    ) -> tuple[bool | None, str | None]:
        """Freshly verify the exact selected HWND, primary status, and catalog identity."""
        try:
            if self._previous_primary_handle != handle:
                return False, "selected_window_changed"
            if not bool(win32gui.IsWindow(handle)):
                return False, "target_window_missing"
            if win32gui.GetForegroundWindow() != handle:
                return False, "foreground_not_selected"
            thread_id, process_id = win32process.GetWindowThreadProcessId(handle)
            if (type(thread_id) is not int or thread_id <= 0
                    or thread_id != expected_thread_id):
                return False, "selected_window_changed"
            if type(process_id) is not int or process_id <= 0:
                return False, "trusted_identity_mismatch"
            self._checked_process_ids.discard(process_id)
            self._trusted_process_ids.discard(process_id)
            if process_id != expected_process_id or not self._matches_process(process_id):
                return False, "trusted_identity_mismatch"
            if not bool(win32gui.IsWindowVisible(handle)):
                return False, "target_not_visible"
            structure = self._window_structure(handle, win32gui, win32con)
            if structure[4] is not False:
                return False, "selected_primary_surface_changed"
            if structure[1] is not False:
                return False, "target_cloaked" if structure[1] is True else "cloaking_unknown"
            if structure[6] is not True:
                return False, "client_area_unknown" if structure[6] is None else "zero_client_area"
            return True, None
        except KeyboardInterrupt:
            raise
        except Exception:
            return None, "foreground_verification_unavailable"

    def _run_thread_input_fallback(
        self,
        handle: int,
        expected_process_id: int,
        expected_thread_id: int,
        activation_api: _WindowActivationApi,
        win32gui: Any,
        win32process: Any,
        win32con: Any,
    ) -> tuple[ThreadInputFallbackAttempt, bool | None, str | None]:
        """Make one scoped foreground-thread attach attempt for the bound target HWND."""
        fields: dict[str, Any] = {
            "eligible": False,
            "reason": "simple_foreground_unverified",
            "foreground_hwnd_present": None,
            "selected_thread_resolved": None,
            "foreground_thread_resolved": None,
            "current_thread_resolved": None,
            "incidental_foreground_changed_before_activation": None,
            "selected_target_still_valid_before_activation": None,
            "attach_attempted": False,
            "attach_succeeded": None,
            "set_foreground_attempted": False,
            "set_foreground_return": None,
            "bring_to_top_used": False,
            "set_active_used": False,
            "detach_succeeded": None,
            "verified": None,
            "failure_reason": None,
        }

        def result(
            reason: str, *, verified: bool | None = None,
            failure_reason: str | None = None,
        ) -> tuple[ThreadInputFallbackAttempt, bool | None, str | None]:
            fields["reason"] = reason
            fields["verified"] = verified
            fields["failure_reason"] = failure_reason
            return ThreadInputFallbackAttempt(**fields), verified, failure_reason

        def selected_failure(reason: str | None) -> str:
            if reason in {
                "selected_window_changed", "target_window_missing",
                "trusted_identity_mismatch",
            }:
                return "selected_target_changed"
            return "selected_target_revalidation_failed"

        def revalidate_selected_target() -> tuple[bool, str | None]:
            selected_ok, selected_reason, repeated_thread_id = (
                self._bound_selected_window_status(
                    handle, expected_process_id, expected_thread_id,
                    win32gui, win32process, win32con,
                )
            )
            fields["selected_thread_resolved"] = repeated_thread_id is not None
            fields["selected_target_still_valid_before_activation"] = selected_ok
            if not selected_ok:
                return False, selected_failure(selected_reason)
            if repeated_thread_id != expected_thread_id:
                fields["selected_target_still_valid_before_activation"] = False
                return False, "selected_target_changed"
            return True, None

        def finish_before_activation(
            failure: str,
        ) -> tuple[ThreadInputFallbackAttempt, bool | None, str | None]:
            fields["reason"] = failure
            fields["failure_reason"] = failure
            fields["verified"] = False
            return ThreadInputFallbackAttempt(**fields), False, failure

        def post_verification_failure(reason: str | None) -> str:
            if reason in {
                "selected_window_changed", "target_window_missing",
                "trusted_identity_mismatch",
            }:
                return "selected_target_changed"
            if reason in {
                "selected_primary_surface_changed", "target_not_visible",
                "target_cloaked", "cloaking_unknown", "client_area_unknown",
                "zero_client_area",
            }:
                return "selected_target_revalidation_failed"
            return "post_activation_verification_failed"

        selected_ok, selected_reason, selected_thread_id = self._bound_selected_window_status(
            handle, expected_process_id, expected_thread_id,
            win32gui, win32process, win32con,
        )
        fields["selected_thread_resolved"] = selected_thread_id is not None
        if not selected_ok:
            failure = selected_failure(selected_reason)
            fields["selected_target_still_valid_before_activation"] = False
            return finish_before_activation(failure)

        try:
            foreground_handle = win32gui.GetForegroundWindow()
            foreground_present = type(foreground_handle) is int and foreground_handle > 0
            fields["foreground_hwnd_present"] = foreground_present
        except KeyboardInterrupt:
            raise
        except Exception:
            return result("foreground_window_invalid", failure_reason="foreground_window_invalid")
        if not foreground_present:
            return result("foreground_window_missing", failure_reason="foreground_window_missing")
        if foreground_handle == handle:
            return result("simple_foreground_unverified", failure_reason="foreground_not_selected")
        try:
            if not bool(win32gui.IsWindow(foreground_handle)):
                return result("foreground_window_invalid", failure_reason="foreground_window_invalid")
            foreground_thread_id_raw, foreground_process_id = (
                win32process.GetWindowThreadProcessId(foreground_handle)
            )
            foreground_thread_id = (
                foreground_thread_id_raw
                if type(foreground_thread_id_raw) is int and foreground_thread_id_raw > 0
                and type(foreground_process_id) is int and foreground_process_id > 0
                else None
            )
            fields["foreground_thread_resolved"] = foreground_thread_id is not None
        except KeyboardInterrupt:
            raise
        except Exception:
            foreground_thread_id = None
        if foreground_thread_id is None:
            return result("foreground_thread_unresolved",
                          failure_reason="foreground_thread_unresolved")
        try:
            current_thread_id_raw = activation_api.current_thread_id()
            current_thread_id = (
                current_thread_id_raw
                if type(current_thread_id_raw) is int and current_thread_id_raw > 0
                else None
            )
            fields["current_thread_resolved"] = current_thread_id is not None
        except KeyboardInterrupt:
            raise
        except Exception:
            current_thread_id = None
        if current_thread_id is None:
            return result("current_thread_unresolved",
                          failure_reason="current_thread_unresolved")
        if current_thread_id == foreground_thread_id:
            return result("foreground_thread_is_current",
                          failure_reason="foreground_thread_is_current")

        fields["eligible"] = True
        fields["reason"] = "fallback_attempted"
        fields["attach_attempted"] = True
        attached = False
        fallback_failure: str | None = None
        set_foreground_call_error = False
        try:
            attached = bool(activation_api.attach_thread_input(
                current_thread_id, foreground_thread_id, True,
            ))
            fields["attach_succeeded"] = attached
            if not attached:
                fallback_failure = "attach_failed"
            else:
                selected_ok, selected_failure_reason = revalidate_selected_target()
                if not selected_ok:
                    fallback_failure = selected_failure_reason
                elif bool(win32gui.IsIconic(handle)):
                    restored = bool(activation_api.show_window_async(handle, win32con.SW_RESTORE))
                    if not restored:
                        fallback_failure = "selected_target_revalidation_failed"
                    else:
                        selected_ok, selected_failure_reason = revalidate_selected_target()
                        if not selected_ok:
                            fallback_failure = selected_failure_reason
                if fallback_failure is None:
                    # This is diagnostic only. The sampled foreground owns the
                    # thread we attached to; a later incidental foreground
                    # change does not invalidate the exact selected target.
                    try:
                        fields["incidental_foreground_changed_before_activation"] = (
                            win32gui.GetForegroundWindow() != foreground_handle
                        )
                    except KeyboardInterrupt:
                        raise
                    except Exception:
                        fields["incidental_foreground_changed_before_activation"] = None

                    # Revalidate the target last, immediately before the single
                    # fallback activation call. Never substitute another HWND.
                    selected_ok, selected_failure_reason = revalidate_selected_target()
                    if not selected_ok:
                        fallback_failure = selected_failure_reason
                if fallback_failure is None:
                    fields["set_foreground_attempted"] = True
                    try:
                        fields["set_foreground_return"] = bool(
                            activation_api.set_foreground_window(handle)
                        )
                    except KeyboardInterrupt:
                        raise
                    except Exception:
                        set_foreground_call_error = True
                        fallback_failure = "set_foreground_failed"
        except KeyboardInterrupt:
            raise
        except Exception:
            fallback_failure = fallback_failure or (
                "attach_failed" if fields["attach_succeeded"] is None
                else "selected_target_revalidation_failed"
            )
        finally:
            # If the attach call itself raises, its state is unknown. A paired
            # detach is harmless when no attachment occurred and prevents a
            # successful native attach from being stranded by a wrapper error.
            if attached or fields["attach_succeeded"] is None:
                try:
                    fields["detach_succeeded"] = bool(activation_api.attach_thread_input(
                        current_thread_id, foreground_thread_id, False,
                    ))
                except KeyboardInterrupt:
                    raise
                except Exception:
                    fields["detach_succeeded"] = False

        # An unsuccessful or uncertain attach never proceeds to activation and
        # cannot be turned into success by an unrelated foreground race.
        if fields["attach_succeeded"] is not True:
            return finish_before_activation("attach_failed")
        if not fields["set_foreground_attempted"]:
            return finish_before_activation(
                fallback_failure or "selected_target_revalidation_failed",
            )
        verified, verification_failure = self._verify_selected_foreground(
            handle, expected_process_id, expected_thread_id,
            win32gui, win32process, win32con,
        )
        if fields["detach_succeeded"] is not True:
            return result("detach_failed", verified=verified, failure_reason="detach_failed")
        if verified is True:
            return result("verified", verified=True, failure_reason=None)
        if post_verification_failure(verification_failure) == "selected_target_changed":
            failure = "selected_target_changed"
        elif set_foreground_call_error or fields["set_foreground_return"] is False:
            failure = "set_foreground_failed"
        else:
            failure = "post_activation_verification_failed"
        return result(failure, verified=verified, failure_reason=failure)
