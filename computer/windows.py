"""Read-only, bounded Windows UI Automation observation via pywinauto."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Protocol, cast
from uuid import uuid4

from computer.applications import ApplicationCatalog
from computer.models import (
    Observation, Rect, ResultReadinessResult, UIElement, VisualCandidateProviderDiagnostic,
    VisualProviderAttempt,
    VisualGroundingStatus, VisualPipelineDiagnostic,
    VisualReadinessReason, VisualReadinessResult,
)
from computer.visual import (
    ScreenCapture, ScreenshotCapture, VisualGroundingRequest, VisualObserver,
    deduplicate_visual_elements,
    deduplicate_visual_elements_within,
    VisualProviderFailure, validate_visual_candidates_detailed, visual_fallback_policy,
    visual_request_fingerprint, ResultReadinessOptions, VisualReadinessOptions,
    visual_readiness_frame, visual_frame_is_informative, visual_frames_meaningfully_differ,
    result_readiness_richness_ratios, result_simplification_signal_count,
)


class _Bounds(Protocol):
    left: int
    top: int
    right: int
    bottom: int


class _Node(Protocol):
    """The small subset of UIAElementInfo used by this adapter."""

    name: str
    control_type: str
    automation_id: str
    rectangle: _Bounds
    enabled: bool
    visible: bool
    process_id: int
    element: Any

    def iter_children(self) -> Iterator[_Node]: ...


@dataclass(frozen=True, slots=True)
class ObservationOptions:
    """Window is depth zero; max_depth zero returns window metadata only."""

    max_depth: int = 6
    max_controls: int = 100
    max_nodes: int = 500
    max_text_length: int = 200
    max_observed_text_length: int = 500

    def __post_init__(self) -> None:
        if self.max_depth < 0:
            raise ValueError("max_depth must be non-negative")
        if min(self.max_controls, self.max_nodes, self.max_text_length, self.max_observed_text_length) < 1:
            raise ValueError("control, node, and text limits must be positive")


_USEFUL_UNNAMED = frozenset({
    "Button", "CheckBox", "ComboBox", "Edit", "Hyperlink", "ListItem",
    "MenuItem", "RadioButton", "ScrollBar", "Slider", "Spinner",
    "TabItem", "TreeItem",
})


def _foreground() -> _Node:
    """Snapshot the foreground HWND once; never activate or refocus a window."""
    if sys.platform != "win32":
        raise OSError("Windows UI Automation requires Windows")
    import win32gui
    from pywinauto import Desktop

    handle = win32gui.GetForegroundWindow()
    if not handle:
        raise RuntimeError("No foreground window")
    return cast(_Node, Desktop(backend="uia").window(handle=handle).wrapper_object().element_info)


def _uia_flag(node: _Node, name: str) -> bool | None:
    value = getattr(node.element, name)
    return bool(value) if type(value) in (bool, int) and value in (0, 1) else None


def _uia_wrapper(node: _Node) -> Any:
    from pywinauto.controls.uiawrapper import UIAWrapper
    return UIAWrapper(node)


def _redact_observed_text(value: str) -> str:
    """Best-effort credential removal before text enters an Observation."""
    secrets = [secret for name, secret in os.environ.items()
               if len(secret) >= 4 and re.search(r"(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)", name, re.I)]
    for secret in sorted(set(secrets), key=len, reverse=True):
        value = value.replace(secret, "[REDACTED]")
    value = re.sub(r"\bBearer\s+\S+", "Bearer [REDACTED]", value, flags=re.I)
    value = re.sub(r"\b(?:sk-[A-Za-z0-9_-]{8,}|(?:jv|ts)_(?:live|test)_[A-Za-z0-9_-]+)",
                   "[REDACTED]", value)
    return re.sub(r"\b(?:api[_ -]?key|password|secret|token)\s*[:=]\s*(?:\"[^\"]*\"|'[^']*'|\S+)",
                  "[REDACTED]", value, flags=re.I)


def _editable_text(node: _Node, limit: int) -> tuple[str | None, bool]:
    """Read at most a small editable value, preferring bounded TextPattern."""
    try:
        wrapper = _uia_wrapper(node)
    except Exception:
        return None, False
    raw: object = None
    try:
        raw = wrapper.iface_text.DocumentRange.GetText(limit + 1)
    except Exception:
        try:
            # ValuePattern has no bounded read. Slice immediately and never use
            # it for controls not explicitly known to be non-password editors.
            raw = wrapper.iface_value.CurrentValue
        except Exception:
            return None, False
    if not isinstance(raw, str) or "\x00" in raw:
        return None, False
    truncated = len(raw) > limit
    return _redact_observed_text(raw[:limit]), truncated


def _process_name(process_id: int | None) -> str:
    """Best-effort executable basename for foreground-app evidence."""
    if not isinstance(process_id, int) or process_id <= 0 or sys.platform != "win32":
        return ""
    try:
        import win32api
        import win32con
        import win32process
        handle = win32api.OpenProcess(
            win32con.PROCESS_QUERY_INFORMATION | win32con.PROCESS_VM_READ, False, process_id,
        )
        try:
            return Path(win32process.GetModuleFileNameEx(handle, 0)).name[:100]
        finally:
            win32api.CloseHandle(handle)
    except Exception:
        return ""


def _package_family_name(process_id: int | None) -> str:
    if not isinstance(process_id, int) or process_id <= 0 or sys.platform != "win32":
        return ""
    try:
        import ctypes
        from ctypes import wintypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetPackageFamilyName.argtypes = (
            wintypes.HANDLE, ctypes.POINTER(wintypes.UINT), wintypes.LPWSTR,
        )
        kernel32.GetPackageFamilyName.restype = wintypes.LONG
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(0x1000, False, process_id)
        if not handle:
            return ""
        try:
            length = wintypes.UINT(0)
            if kernel32.GetPackageFamilyName(handle, ctypes.byref(length), None) not in (0, 122):
                return ""
            buffer = ctypes.create_unicode_buffer(length.value)
            if kernel32.GetPackageFamilyName(handle, ctypes.byref(length), buffer) != 0:
                return ""
            return buffer.value[:200]
        finally:
            kernel32.CloseHandle(handle)
    except Exception:
        return ""


class WindowsObserver:
    """Compact snapshot with bounded, non-password editable text when available.

    UIA Name supplies labels/text. Bounds are descriptive only. IDs reset each
    observation. Windows imports are lazy to keep generic modules portable.
    """

    def __init__(
        self, options: ObservationOptions | None = None, app_catalog: ApplicationCatalog | None = None,
        *, capture_service: ScreenCapture | None = None,
        visual_provider: VisualObserver | None = None,
        capture_without_provider: bool = False,
        force_capture: bool = False,
        retain_debug_capture: bool = False,
        collect_provider_candidate_diagnostics: bool = False,
        visual_grounding: VisualGroundingRequest | None = None,
        visual_readiness_options: VisualReadinessOptions | None = None,
        result_readiness_options: ResultReadinessOptions | None = None,
        readiness_clock: Callable[[], float] = time.monotonic,
        readiness_sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.options = options or ObservationOptions()
        self.app_catalog = app_catalog
        self._root: _Node | None = None
        self._nodes: dict[str, _Node] = {}
        self.capture_service = capture_service
        self.visual_provider = visual_provider
        self.capture_without_provider = capture_without_provider
        self.force_capture = force_capture
        self.retain_debug_capture = retain_debug_capture
        self.collect_provider_candidate_diagnostics = collect_provider_candidate_diagnostics
        self.visual_grounding = visual_grounding
        self.visual_readiness_options = visual_readiness_options
        self.result_readiness_options = result_readiness_options
        self.result_readiness_enabled = False
        self.result_readiness_baseline = None
        self._readiness_clock = readiness_clock
        self._readiness_sleep = readiness_sleep
        self._observation_request = ""
        self._debug_capture: ScreenshotCapture | None = None

    def set_observation_request(self, request: str) -> None:
        self._observation_request = request[:4000]

    def take_debug_capture(self) -> ScreenshotCapture | None:
        capture, self._debug_capture = self._debug_capture, None
        return capture

    def observe(self) -> Observation:
        if self._debug_capture is not None:
            self._debug_capture.discard()
            self._debug_capture = None
        self._root = None
        self._nodes = {}
        options = self.options
        errors = 0
        truncated = False

        def read[T](getter: Callable[[], T], fallback: T) -> T:
            nonlocal errors
            try:
                return getter()
            except Exception:
                # Individual COM/provider failures must not discard the snapshot.
                # Exception messages may contain desktop content: do not echo them.
                errors += 1
                return fallback

        def text(getter: Callable[[], str]) -> str:
            nonlocal truncated
            value = (read(getter, "") or "").strip()
            if len(value) > options.max_text_length:
                truncated = True
                return value[:options.max_text_length]
            return value

        try:
            root = _foreground()
        except Exception:
            return Observation(
                app_name="", window_title="", inspection_errors=1,
                error="Foreground window unavailable; check desktop access and UIA availability.",
            )

        title = text(lambda: root.name)
        self._root = root
        window_type = text(lambda: root.control_type)
        process_id = read(lambda: root.process_id, None)
        try:
            raw_handle = root.handle
            foreground_hwnd = int(raw_handle) if raw_handle else None
        except Exception:
            foreground_hwnd = None
        app_name = _process_name(process_id)
        package_family = _package_family_name(process_id)
        application_id = read(lambda: self.app_catalog.identify(app_name, package_family) or "", "") \
            if self.app_catalog else ""
        controls: list[UIElement] = []

        def children(node: _Node) -> Iterator[_Node]:
            nonlocal errors
            try:
                yield from node.iter_children()
            except Exception:
                # Retain prior children and resume the ancestor's iterator.
                errors += 1

        def rectangle(node: _Node) -> Rect:
            bounds = node.rectangle
            return Rect(bounds.left, bounds.top, bounds.right, bounds.bottom)

        # Lazy sibling iterators: no descendants() or eager tree dump.
        stack: list[tuple[Iterator[_Node], int, str, str]] = []
        if read(lambda: root.visible, None) is not False:
            if options.max_depth:
                stack.append((children(root), 1, title, window_type))
            else:
                truncated = True
        visited = 0
        while stack:
            if visited >= options.max_nodes or len(controls) >= options.max_controls:
                truncated = True
                break
            siblings, depth, parent_name, parent_type = stack[-1]
            node = next(siblings, None)
            if node is None:
                stack.pop()
                continue
            visited += 1
            visible = read(lambda: node.visible, None)
            if visible is False:
                continue
            node_name = ""
            node_kind = ""
            # Unknown visibility: skip this node, but still inspect descendants.
            if visible is True:
                name = node_name = text(lambda: node.name)
                kind = node_kind = text(lambda: node.control_type)
                automation_id = text(lambda: node.automation_id)
                if name or automation_id or kind in _USEFUL_UNNAMED:
                    is_password = read(lambda: _uia_flag(node, "CurrentIsPassword"), None)
                    focused = read(lambda: _uia_flag(node, "CurrentHasKeyboardFocus"), None)
                    selected = read(lambda: _uia_flag(node, "CurrentIsSelected"), None)
                    observed_text, observed_text_truncated = (
                        _editable_text(node, options.max_observed_text_length)
                        if (kind in {"Edit", "Document"} and is_password is False
                            and focused is True) else (None, False)
                    )
                    if observed_text_truncated:
                        truncated = True
                    controls.append(UIElement(
                        id=f"c{len(controls) + 1}", name=name, control_type=kind,
                        automation_id=automation_id,
                        rectangle=read(lambda: rectangle(node), None),
                        enabled=read(lambda: node.enabled, None), visible=True,
                        focused=focused,
                        is_password=is_password, observed_text=observed_text,
                        observed_text_truncated=observed_text_truncated,
                        parent_name=parent_name, parent_control_type=parent_type,
                        selected=selected,
                    ))
                    self._nodes[controls[-1].id] = node
            if depth < options.max_depth:
                stack.append((children(node), depth + 1, node_name, node_kind))
            else:
                # Conservative: don't probe deeper just to prove truncation.
                truncated = True

        observation = Observation(
            app_name=app_name, window_title=title, elements=tuple(controls),
            process_id=process_id, control_type=window_type,
            truncated=truncated, inspection_errors=errors,
            observation_id=uuid4().hex,
            application_id=application_id or "", package_family_name=package_family,
            foreground_hwnd=foreground_hwnd if foreground_hwnd and foreground_hwnd > 0 else None,
        )
        fallback = visual_fallback_policy(observation, self._observation_request)
        observation = replace(observation, visual_fallback_reason=fallback.reason)
        request_safe = _redact_observed_text(self._observation_request) == self._observation_request
        should_capture = ((fallback.required or self.force_capture)
                          and (self.visual_provider is not None or self.capture_without_provider)
                          and request_safe)
        if not should_capture or self.capture_service is None or self._root is None:
            return observation
        sensitive = tuple(control.rectangle for control in observation.elements
                          if control.is_password is True and isinstance(control.rectangle, Rect))
        visual_started = time.monotonic()
        result_readiness_started = (
            self._readiness_clock()
            if self.result_readiness_enabled and self.visual_grounding is not None else None
        )
        capture_started = time.monotonic()
        try:
            capture = self.capture_service.capture(
                observation.observation_id, int(self._root.handle), app_name, sensitive,  # type: ignore[attr-defined]
            )
        except Exception:
            return observation
        screenshot_capture_ms = max(0, round((time.monotonic() - capture_started) * 1000))
        result_readiness = None
        if self.result_readiness_enabled and self.visual_grounding is not None:
            options = self.result_readiness_options or ResultReadinessOptions()
            result_started = (
                result_readiness_started if result_readiness_started is not None
                else self._readiness_clock()
            )
            attempts = 1
            current_frame = visual_readiness_frame(capture.diagnostics) if capture.diagnostics else None
            baseline_frame = self.result_readiness_baseline
            baseline_changed_now = bool(
                baseline_frame is not None and current_frame is not None
                and visual_frames_meaningfully_differ(baseline_frame, current_frame)
            )
            recent_frame_changed = False
            meaningful_change_seen = baseline_changed_now
            first_transition_elapsed_ms = 0 if baseline_changed_now else None
            last_meaningful_change_elapsed_ms = 0 if baseline_changed_now else None
            quiet_timer_reset_count = 0
            informative = bool(current_frame is not None and visual_frame_is_informative(current_frame))
            informative_frame_seen = informative
            richness_ratios = result_readiness_richness_ratios(baseline_frame, current_frame)
            simplification_signal_count = result_simplification_signal_count(
                richness_ratios, options.simplification_ratio_threshold,
            )
            transitional_simplification = (
                simplification_signal_count >= options.simplification_min_signals
            )
            stable_since_elapsed_ms = (
                0 if baseline_frame is not None and not baseline_changed_now else None
            )
            stable_frame_seen = stable_since_elapsed_ms is not None
            state = "settling_candidate" if meaningful_change_seen else "waiting_for_transition"
            awaiting_followup_transition = False
            followup_transition_seen = False
            transition_grace_started_elapsed_ms: int | None = None
            transition_grace_elapsed_ms = 0
            timeout_reason = "timeout_before_quiet_stable"
            deadline = result_started + options.timeout_seconds

            def readiness_result(ready: bool, reason: str) -> ResultReadinessResult:
                elapsed_ms = max(0, round((self._readiness_clock() - result_started) * 1000))
                quiet_ms = (
                    max(0, elapsed_ms - last_meaningful_change_elapsed_ms)
                    if last_meaningful_change_elapsed_ms is not None else 0
                )
                grace_elapsed_ms = transition_grace_elapsed_ms
                if transition_grace_started_elapsed_ms is not None:
                    grace_elapsed_ms = max(
                        0, elapsed_ms - transition_grace_started_elapsed_ms,
                    )
                stable = bool(
                    current_frame is not None and stable_since_elapsed_ms is not None
                )
                return ResultReadinessResult(
                    ready=ready, reason=reason, attempts=attempts, elapsed_ms=elapsed_ms,
                    meaningful_change_seen=meaningful_change_seen,
                    final_informative=informative, stable=stable, after_query_submit=True,
                    baseline=baseline_frame, final=current_frame,
                    first_transition_elapsed_ms=first_transition_elapsed_ms,
                    last_meaningful_change_elapsed_ms=last_meaningful_change_elapsed_ms,
                    required_quiet_ms=options.settle_quiet_ms,
                    observed_quiet_ms=quiet_ms,
                    quiet_timer_reset_count=quiet_timer_reset_count,
                    informative_frame_seen=informative_frame_seen,
                    stable_frame_seen=stable_frame_seen,
                    state="ready" if ready else "timeout",
                    baseline_changed=baseline_changed_now,
                    recent_frame_changed=recent_frame_changed,
                    richness_ratios=richness_ratios,
                    simplification_signal_count=simplification_signal_count,
                    transitional_simplification=transitional_simplification,
                    awaiting_followup_transition=awaiting_followup_transition,
                    followup_transition_seen=followup_transition_seen,
                    transition_grace_ms=options.transition_grace_ms,
                    transition_grace_elapsed_ms=grace_elapsed_ms,
                    simplification_ratio_threshold=options.simplification_ratio_threshold,
                    simplification_min_signals=options.simplification_min_signals,
                )

            ready = False
            if current_frame is not None:
                while self._readiness_clock() < deadline:
                    elapsed_ms = max(0, round((self._readiness_clock() - result_started) * 1000))
                    quiet_ms = (
                        max(0, elapsed_ms - last_meaningful_change_elapsed_ms)
                        if last_meaningful_change_elapsed_ms is not None else 0
                    )
                    stable = bool(current_frame is not None and stable_since_elapsed_ms is not None)
                    if (meaningful_change_seen and informative and stable
                            and quiet_ms >= options.settle_quiet_ms):
                        if transitional_simplification:
                            if not awaiting_followup_transition:
                                awaiting_followup_transition = True
                                transition_grace_started_elapsed_ms = elapsed_ms
                            state = "awaiting_followup_transition"
                        else:
                            ready = True
                            state = "ready"
                            break
                    if (awaiting_followup_transition
                            and transition_grace_started_elapsed_ms is not None
                            and elapsed_ms - transition_grace_started_elapsed_ms
                            >= options.transition_grace_ms):
                        timeout_reason = "timeout_transitional_simplification"
                        state = "timeout"
                        break
                    remaining = deadline - self._readiness_clock()
                    self._readiness_sleep(min(options.poll_interval_seconds, max(0, remaining)))
                    if self._readiness_clock() >= deadline:
                        break
                    try:
                        current = _foreground()
                        if (int(current.handle) != int(self._root.handle)  # type: ignore[attr-defined]
                                or current.process_id != self._root.process_id):
                            capture.discard()
                            return replace(observation, result_readiness=readiness_result(
                                False, "foreground_changed",
                            ))
                        replacement = self.capture_service.capture(
                            observation.observation_id, int(current.handle), app_name, sensitive,
                        )
                    except Exception:
                        capture.discard()
                        return replace(observation, result_readiness=readiness_result(
                            False, "capture_error",
                        ))
                    capture.discard()
                    capture = replacement
                    attempts += 1
                    frame = visual_readiness_frame(capture.diagnostics) if capture.diagnostics else None
                    prior_frame = current_frame
                    baseline_changed_now = bool(
                        baseline_frame is not None and frame is not None
                        and visual_frames_meaningfully_differ(baseline_frame, frame)
                    )
                    recent_frame_changed = bool(
                        prior_frame is not None and frame is not None
                        and visual_frames_meaningfully_differ(prior_frame, frame)
                    )
                    richness_ratios = result_readiness_richness_ratios(baseline_frame, frame)
                    simplification_signal_count = result_simplification_signal_count(
                        richness_ratios, options.simplification_ratio_threshold,
                    )
                    transitional_simplification = (
                        simplification_signal_count >= options.simplification_min_signals
                    )
                    elapsed_ms = max(0, round((self._readiness_clock() - result_started) * 1000))
                    new_first_transition = (
                        first_transition_elapsed_ms is None
                        and (baseline_changed_now or recent_frame_changed)
                    )
                    transition_seen = new_first_transition or (
                        recent_frame_changed and meaningful_change_seen
                    )
                    if transition_seen:
                        was_meaningful = meaningful_change_seen
                        meaningful_change_seen = True
                        if new_first_transition:
                            first_transition_elapsed_ms = elapsed_ms
                        elif last_meaningful_change_elapsed_ms is not None:
                            quiet_timer_reset_count += 1
                        if was_meaningful:
                            followup_transition_seen = True
                        if awaiting_followup_transition:
                            if transition_grace_started_elapsed_ms is not None:
                                transition_grace_elapsed_ms = max(
                                    0, elapsed_ms - transition_grace_started_elapsed_ms,
                                )
                            awaiting_followup_transition = False
                            transition_grace_started_elapsed_ms = None
                        state = (
                            "settling_final_candidate" if followup_transition_seen
                            else "settling_candidate"
                        )
                        last_meaningful_change_elapsed_ms = elapsed_ms
                        stable_since_elapsed_ms = None
                    informative = bool(frame is not None and visual_frame_is_informative(frame))
                    informative_frame_seen = informative_frame_seen or informative
                    if not transition_seen and prior_frame is not None \
                            and stable_since_elapsed_ms is None:
                        stable_since_elapsed_ms = elapsed_ms
                    if (stable_since_elapsed_ms is not None and prior_frame is not None
                            and frame is not None and not recent_frame_changed):
                        stable_frame_seen = True
                    current_frame = frame
            result_readiness = readiness_result(
                ready,
                "changed_and_quiet_stable" if ready else timeout_reason,
            )
            if not ready:
                capture.discard()
                return replace(
                    observation, screenshot=capture.metadata,
                    capture_diagnostics=capture.diagnostics,
                    visual_directed_grounding=True,
                    visual_requested_max_elements=self.visual_grounding.max_elements,
                    result_readiness=result_readiness,
                )
        readiness = None
        if self.visual_readiness_options is not None and self.visual_grounding is not None:
            readiness_started = self._readiness_clock()
            attempts = 1
            initial_frame = (
                visual_readiness_frame(capture.diagnostics)
                if capture.diagnostics is not None else None
            )

            def readiness_result(
                ready: bool, reason: VisualReadinessReason, final_frame=initial_frame,
            ) -> VisualReadinessResult:
                return VisualReadinessResult(
                    ready, reason, attempts,
                    max(0, round((self._readiness_clock() - readiness_started) * 1000)),
                    initial_frame, final_frame,
                )

            def failed_readiness(reason: VisualReadinessReason) -> Observation:
                nonlocal capture
                result = readiness_result(False, reason, (
                    visual_readiness_frame(capture.diagnostics)
                    if capture.diagnostics is not None else None
                ))
                failed = replace(
                    observation, screenshot=capture.metadata,
                    screenshot_capture_ms=max(
                        0, round((time.monotonic() - capture_started) * 1000),
                    ),
                    visual_requested_max_elements=self.visual_grounding.max_elements,
                    visual_directed_grounding=True,
                    visual_pipeline=VisualPipelineDiagnostic(
                        self.visual_grounding.max_elements,
                    ),
                    capture_diagnostics=capture.diagnostics,
                    visual_readiness=result,
                )
                if self.retain_debug_capture:
                    self._debug_capture = capture
                    capture = None
                else:
                    capture.discard()
                    capture = None
                return failed

            if initial_frame is None:
                return failed_readiness(VisualReadinessReason.CAPTURE_ERROR)
            if capture.diagnostics and capture.diagnostics.masked_area_percent >= 80:
                return failed_readiness(VisualReadinessReason.CREDENTIAL_SENSITIVE)
            if visual_frame_is_informative(initial_frame):
                readiness = readiness_result(
                    True, VisualReadinessReason.INITIALLY_READY, initial_frame,
                )
            else:
                deadline = readiness_started + self.visual_readiness_options.timeout_seconds
                while self._readiness_clock() < deadline:
                    remaining = deadline - self._readiness_clock()
                    self._readiness_sleep(min(
                        self.visual_readiness_options.poll_interval_seconds, max(0, remaining),
                    ))
                    if self._readiness_clock() >= deadline:
                        break
                    try:
                        current = _foreground()
                        if (int(current.handle) != int(self._root.handle)  # type: ignore[attr-defined]
                                or current.process_id != self._root.process_id):
                            return failed_readiness(VisualReadinessReason.FOREGROUND_CHANGED)
                        replacement = self.capture_service.capture(
                            observation.observation_id, int(current.handle), app_name, sensitive,
                        )
                    except Exception:
                        return failed_readiness(VisualReadinessReason.CAPTURE_ERROR)
                    capture.discard()
                    capture = replacement
                    attempts += 1
                    final_frame = (
                        visual_readiness_frame(capture.diagnostics)
                        if capture.diagnostics is not None else None
                    )
                    if final_frame is None:
                        return failed_readiness(VisualReadinessReason.CAPTURE_ERROR)
                    if capture.diagnostics and capture.diagnostics.masked_area_percent >= 80:
                        return failed_readiness(VisualReadinessReason.CREDENTIAL_SENSITIVE)
                    if visual_frame_is_informative(final_frame):
                        readiness = readiness_result(
                            True, VisualReadinessReason.BECAME_READY, final_frame,
                        )
                        break
                if readiness is None:
                    return failed_readiness(VisualReadinessReason.TIMEOUT)
        visual_elements = ()
        provider_name = self.visual_provider.name[:80] if self.visual_provider is not None else None
        provider_model = str(getattr(self.visual_provider, "model", ""))[:100] or None
        provider_latency = None
        provider_usage: tuple[tuple[str, int], ...] = ()
        provider_attempts: tuple[VisualProviderAttempt, ...] = ()
        selected_visual_provider: str | None = None
        provider_failover_used = False
        provider_failover_reason: str | None = None
        provider_pricing = str(getattr(self.visual_provider, "pricing_class", ""))[:40] or None
        execution_authorized = False
        request_build_ms = None
        response_parse_ms = None
        requested_max_elements = None
        returned_visual_elements = None
        raw_element_count = None
        parsed_element_count = None
        validated_element_count = None
        deduplicated_element_count = None
        rejection_summary: dict[str, int] = {}
        provider_candidate_diagnostics: tuple[VisualCandidateProviderDiagnostic, ...] = ()
        grounding_status = None
        directed_grounding = self.visual_grounding is not None
        request_fingerprint = None
        if self.visual_provider is not None and self.visual_grounding is not None:
            try:
                request_fingerprint = visual_request_fingerprint(
                    self.visual_provider, capture, self.visual_grounding,
                )
            except Exception:
                request_fingerprint = None
        provider_started: float | None = None
        provider_call_count = 0
        try:
            if self.visual_provider is not None and fallback.required:
                provider_started = time.monotonic()
                provider_call_count = 1
                if self.visual_grounding is None:
                    provider_result = self.visual_provider.observe(
                        capture, observation, self._observation_request,
                    )
                else:
                    provider_result = self.visual_provider.observe(
                        capture, observation, self._observation_request, self.visual_grounding,
                    )
                provider_name = provider_result.provider[:80]
                provider_model = provider_result.model[:100] or None
                provider_latency = provider_result.latency_ms
                provider_attempts = provider_result.provider_attempts or (
                    VisualProviderAttempt(
                        provider_name, provider_model or "",
                        max(0, provider_latency or 0),
                        "success_with_candidates" if provider_result.candidates else "success_empty",
                    ),
                )
                selected_visual_provider = provider_result.provider[:80]
                provider_failover_used = provider_result.provider_failover_used
                provider_failover_reason = provider_result.provider_failover_reason
                provider_call_count = max(1, len(provider_attempts))
                provider_usage = provider_result.usage
                provider_pricing = provider_result.pricing_class[:40]
                execution_authorized = provider_result.execution_authorized
                request_build_ms = provider_result.request_build_ms
                response_parse_ms = provider_result.response_parse_ms
                requested_max_elements = (
                    provider_result.requested_max_elements
                    if provider_result.requested_max_elements is not None
                    else self.visual_grounding.max_elements if self.visual_grounding else None
                )
                # Direction is a property of the locally constructed request,
                # never something provider output may enable or disable.
                directed_grounding = self.visual_grounding is not None
                raw_element_count = (
                    provider_result.raw_element_count
                    if provider_result.raw_element_count is not None
                    else len(provider_result.candidates)
                )
                parsed_element_count = (
                    provider_result.parsed_element_count
                    if provider_result.parsed_element_count is not None
                    else len(provider_result.candidates)
                )

                def provider_candidate_diagnostic(candidate) -> VisualCandidateProviderDiagnostic:
                    label = candidate.label if isinstance(candidate.label, str) else ""
                    parent = candidate.parent if isinstance(candidate.parent, str) else ""
                    provider_role = getattr(candidate, "provider_role", None)
                    role = (provider_role if isinstance(provider_role, str) else
                            candidate.role if isinstance(candidate.role, str) else "")
                    label = _redact_observed_text(" ".join(label.split())[:160])
                    parent = _redact_observed_text(" ".join(parent.split())[:120])
                    role = _redact_observed_text(" ".join(role.split())[:40])
                    rect = candidate.rectangle
                    geometry_valid = bool(
                        isinstance(rect, Rect) and rect.left >= 0 and rect.top >= 0
                        and rect.right > rect.left and rect.bottom > rect.top
                        and rect.right <= capture.metadata.pixel_width
                        and rect.bottom <= capture.metadata.pixel_height
                    )
                    reason = None if geometry_valid else "invalid_or_outside_bbox"
                    return VisualCandidateProviderDiagnostic(
                        label, parent, role,
                        candidate.clickable if type(candidate.clickable) is bool else None,
                        geometry_valid,
                        bool(role and len(role) <= 40 and "\x00" not in role), reason,
                    )

                class _Redactor:
                    @staticmethod
                    def clean(value: str) -> str:
                        return _redact_observed_text(value)

                validated, rejection_summary = validate_visual_candidates_detailed(
                    provider_result.candidates, capture.metadata, redactor=_Redactor(),
                )
                validated_element_count = len(validated)
                if self.collect_provider_candidate_diagnostics:
                    raw_candidate_diagnostics = tuple(
                        provider_candidate_diagnostic(item)
                        for item in provider_result.candidates[:100]
                    )
                    raw_records: list[VisualCandidateProviderDiagnostic] = []
                    accepted_cursor = 0
                    for index, (candidate, diagnostic) in enumerate(zip(
                        provider_result.candidates[:100], raw_candidate_diagnostics,
                    ), 1):
                        matched_id = None
                        if accepted_cursor < len(validated):
                            item = validated[accepted_cursor]
                            candidate_rect = getattr(candidate, "rectangle", None)
                            if (diagnostic.geometry_valid
                                    and item.label == diagnostic.label
                                    and item.role == diagnostic.role.replace("_", " ")
                                    and item.parent == diagnostic.parent
                                    and item.rectangle == candidate_rect):
                                matched_id = item.id
                                accepted_cursor += 1
                        rejection = diagnostic.rejection_reason
                        if matched_id is None and rejection is None:
                            rejection = "candidate_not_retained_by_validation"
                        raw_records.append(replace(
                            diagnostic, provider_candidate_id=f"p{index}",
                            validated_visual_id=matched_id, rejection_reason=rejection,
                        ))
                    provider_candidate_diagnostics = tuple(raw_records)
                within = deduplicate_visual_elements_within(validated)
                visual_elements = deduplicate_visual_elements(
                    observation.elements, within, capture.metadata,
                )
                deduplicated_element_count = len(visual_elements)
                rejection_summary["duplicate"] = max(
                    0, validated_element_count - deduplicated_element_count,
                )
                returned_visual_elements = len(visual_elements)
                if visual_elements:
                    grounding_status = VisualGroundingStatus.SUCCESS_WITH_CANDIDATES
                elif parsed_element_count == 0:
                    grounding_status = VisualGroundingStatus.SUCCESS_EMPTY
                elif validated_element_count == 0:
                    grounding_status = VisualGroundingStatus.VALIDATION_EMPTY
                else:
                    grounding_status = VisualGroundingStatus.DEDUP_EMPTY
            total_visual_ms = max(0, round((time.monotonic() - visual_started) * 1000))
            pipeline = VisualPipelineDiagnostic(
                requested_max_elements, raw_element_count, parsed_element_count,
                validated_element_count, deduplicated_element_count,
                len(visual_elements), None,
            )
            observation = replace(
                observation, screenshot=capture.metadata, visual_elements=visual_elements,
                visual_provider=provider_name, visual_model=provider_model,
                visual_latency_ms=provider_latency,
                visual_usage=provider_usage,
                visual_pricing_class=provider_pricing,
                visual_execution_authorized=execution_authorized,
                visual_provider_attempts=provider_attempts,
                selected_visual_provider=selected_visual_provider,
                provider_failover_used=provider_failover_used,
                provider_failover_reason=provider_failover_reason,
                screenshot_capture_ms=screenshot_capture_ms,
                visual_request_build_ms=request_build_ms,
                visual_response_parse_ms=response_parse_ms,
                visual_total_observation_ms=total_visual_ms,
                visual_requested_max_elements=requested_max_elements,
                visual_returned_elements=returned_visual_elements,
                visual_directed_grounding=directed_grounding,
                visual_grounding_status=grounding_status,
                visual_pipeline=pipeline,
                visual_rejection_summary=tuple(sorted(rejection_summary.items())),
                visual_request_fingerprint=request_fingerprint,
                capture_diagnostics=capture.diagnostics,
                visual_readiness=readiness,
                result_readiness=result_readiness,
                visual_provider_call_count=provider_call_count,
                visual_provider_candidates=provider_candidate_diagnostics,
            )
            if self.retain_debug_capture:
                self._debug_capture = capture
                capture = None
            return observation
        except VisualProviderFailure as exc:
            failure_latency = (None if provider_started is None else
                               max(0, round((time.monotonic() - provider_started) * 1000)))
            return replace(
                observation, screenshot=capture.metadata, visual_provider=provider_name,
                visual_model=provider_model, visual_pricing_class=provider_pricing,
                visual_latency_ms=failure_latency,
                visual_provider_error=exc.diagnostic,
                visual_provider_attempts=exc.provider_attempts,
                selected_visual_provider=exc.selected_visual_provider,
                provider_failover_used=exc.provider_failover_used,
                provider_failover_reason=exc.provider_failover_reason,
                screenshot_capture_ms=screenshot_capture_ms,
                visual_total_observation_ms=max(
                    0, round((time.monotonic() - visual_started) * 1000),
                ),
                visual_requested_max_elements=(
                    self.visual_grounding.max_elements if self.visual_grounding else None
                ),
                visual_directed_grounding=directed_grounding,
                visual_grounding_status=(
                    VisualGroundingStatus.PARSE_ERROR
                    if exc.code in {"malformed_response", "invalid_response"}
                    else VisualGroundingStatus.PROVIDER_ERROR
                ),
                visual_pipeline=VisualPipelineDiagnostic(
                    self.visual_grounding.max_elements if self.visual_grounding else None,
                ),
                visual_request_fingerprint=request_fingerprint,
                capture_diagnostics=capture.diagnostics,
                visual_readiness=readiness,
                result_readiness=result_readiness,
                visual_provider_call_count=max(1, len(exc.provider_attempts)),
                visual_provider_candidates=provider_candidate_diagnostics,
            )
        except Exception:
            failure = VisualProviderFailure("unknown_api_error").with_provider_context(
                provider_name or "unknown", provider_model or "",
                provider_attempts=(), selected_visual_provider=None,
            )
            return replace(
                observation, screenshot=capture.metadata, visual_provider=provider_name,
                visual_model=provider_model, visual_pricing_class=provider_pricing,
                visual_provider_error=failure.diagnostic,
                selected_visual_provider=None,
                screenshot_capture_ms=screenshot_capture_ms,
                visual_total_observation_ms=max(
                    0, round((time.monotonic() - visual_started) * 1000),
                ),
                visual_requested_max_elements=(
                    self.visual_grounding.max_elements if self.visual_grounding else None
                ),
                visual_directed_grounding=directed_grounding,
                visual_grounding_status=VisualGroundingStatus.PROVIDER_ERROR,
                visual_pipeline=VisualPipelineDiagnostic(
                    self.visual_grounding.max_elements if self.visual_grounding else None,
                ),
                visual_request_fingerprint=request_fingerprint,
                capture_diagnostics=capture.diagnostics,
                visual_readiness=readiness,
                result_readiness=result_readiness,
                visual_provider_call_count=provider_call_count,
                visual_provider_candidates=provider_candidate_diagnostics,
            )
        finally:
            if capture is not None:
                if self.retain_debug_capture:
                    # Keep the exact masked provider input available to an
                    # explicitly opted-in debug callback, including when the
                    # provider returns a typed error. The caller consumes it
                    # once via take_debug_capture(); normal runs still discard.
                    self._debug_capture = capture
                else:
                    capture.discard()
