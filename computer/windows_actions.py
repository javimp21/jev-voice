"""Windows-only action adapter with single-use observation bindings."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

from computer.actions import (
    Action, ClickAction, FinishAction, OpenAppAction, PressKeyAction, QuerySubmitAction,
    TypeAction, VisualClickAction,
)
from computer.applications import ApplicationCatalog
from computer.models import Observation, Rect
from computer.results import ActionResult, LiteralInputDiagnostic
from computer.windows import ObservationOptions, WindowsObserver, _foreground
from computer.visual import (
    ScreenCapture, VisualGroundingRequest, VisualObserver, VisualReadinessOptions,
    visual_click_point, visual_readiness_frame, visual_rect_to_screen,
)
from safety.interfaces import ActionPolicy
from safety.policy import BasicActionPolicy, GenericTargetActivationPolicy, Phase1VisualClickPolicy
_KEY_EXPRESSIONS = {
    ("enter",): "{ENTER}", ("escape",): "{ESC}", ("tab",): "{TAB}",
    ("shift", "tab"): "+{TAB}", ("ctrl", "a"): "^a", ("ctrl", "c"): "^c",
}


class UnsafeTarget(RuntimeError):
    """A target cannot be proved to still match its original observation."""


class StaleObservation(UnsafeTarget):
    """The captured window or visual snapshot changed and must be observed again."""


def debug_type_literal(text: str) -> LiteralInputDiagnostic:
    """Explicit standalone literal-input probe for a user-focused trusted window."""
    if not isinstance(text, str) or not text or "\x00" in text or len(text) > 500:
        raise ValueError("Debug literal must contain 1 to 500 characters without NUL.")
    current = _foreground()
    identity = _identity(current)
    if not current.handle or not current.visible or not current.enabled:  # type: ignore[attr-defined]
        raise UnsafeTarget("Foreground window is not safely available for literal input.")
    return _type_literal_unicode(
        text, foreground_hwnd=int(current.handle),  # type: ignore[attr-defined]
        foreground_pid=identity.process_id,
    )


def _wrapper(node: Any) -> Any:
    from pywinauto.controls.uiawrapper import UIAWrapper
    return UIAWrapper(node)


def _focused() -> Any:
    from pywinauto.uia_defines import IUIA
    from pywinauto.uia_element_info import UIAElementInfo
    return UIAElementInfo(IUIA().get_focused_element())


def _send_key(keys: tuple[str, ...]) -> None:
    from pywinauto.keyboard import send_keys
    # Only fixed internal strings reach the keyboard parser.
    send_keys(_KEY_EXPRESSIONS[keys], pause=0)


def _click_point(x: int, y: int) -> None:
    from pywinauto.mouse import click
    click(button="left", coords=(x, y))


import ctypes
from ctypes import wintypes


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = (
        ("dx", wintypes.LONG), ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD), ("dwExtraInfo", wintypes.WPARAM),
    )


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = (
        ("wVk", wintypes.WORD), ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD),
        ("dwExtraInfo", wintypes.WPARAM),
    )


class _HARDWAREINPUT(ctypes.Structure):
    _fields_ = (
        ("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    )


class _INPUTUNION(ctypes.Union):
    _fields_ = (("mi", _MOUSEINPUT), ("ki", _KEYBDINPUT), ("hi", _HARDWAREINPUT))


class _INPUT(ctypes.Structure):
    _anonymous_ = ("value",)
    _fields_ = (("type", wintypes.DWORD), ("value", _INPUTUNION))


class LiteralInputFailure(RuntimeError):
    def __init__(self, diagnostic: LiteralInputDiagnostic) -> None:
        super().__init__(diagnostic.failure_stage or "send_input_failed")
        self.diagnostic = diagnostic


def _windows_send_input_api():
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.SendInput.argtypes = (wintypes.UINT, ctypes.POINTER(_INPUT), ctypes.c_int)
    user32.SendInput.restype = wintypes.UINT
    return user32.SendInput, ctypes.get_last_error


def _type_literal_unicode(
    text: str, *, send_input=None, get_last_error=None,
    foreground_hwnd: int | None = None, foreground_pid: int | None = None,
) -> LiteralInputDiagnostic:
    """Send literal UTF-16 units through ABI-correct SendInput structures."""
    units = text.encode("utf-16-le")
    events: list[_INPUT] = []
    for index in range(0, len(units), 2):
        unit = int.from_bytes(units[index:index + 2], "little")
        events.extend((
            _INPUT(type=1, ki=_KEYBDINPUT(0, unit, 0x0004, 0, 0)),
            _INPUT(type=1, ki=_KEYBDINPUT(0, unit, 0x0004 | 0x0002, 0, 0)),
        ))
    if not events:
        return LiteralInputDiagnostic(
            len(text), 0, 0, 0, 0, None, None, ctypes.sizeof(_INPUT),
            foreground_hwnd, foreground_pid, True,
        )
    if send_input is None:
        send_input, get_last_error = _windows_send_input_api()
        ctypes.set_last_error(0)
    array = (_INPUT * len(events))(*events)
    sent = int(send_input(len(events), array, ctypes.sizeof(_INPUT)))
    error = int(get_last_error()) if sent != len(events) and get_last_error else None
    diagnostic = LiteralInputDiagnostic(
        len(text), len(units) // 2, len(events), len(events), sent, error,
        None if sent == len(events) else "send_input_partial_or_failed",
        ctypes.sizeof(_INPUT), foreground_hwnd, foreground_pid, True,
    )
    if sent != len(events):
        raise LiteralInputFailure(diagnostic)
    return diagnostic


@dataclass(frozen=True)
class _Identity:
    runtime_id: tuple[int, ...]
    process_id: int
    control_type: str
    automation_id: str
    name: str


def _identity(node: Any) -> _Identity:
    runtime_id = node.runtime_id
    if (not isinstance(runtime_id, (tuple, list)) or not runtime_id
            or any(not isinstance(part, int) for part in runtime_id)):
        raise UnsafeTarget("UIA runtime identity is unavailable.")
    pid = node.process_id
    if not isinstance(pid, int) or pid <= 0:
        raise UnsafeTarget("Process identity is unavailable.")
    return _Identity(tuple(runtime_id), pid, node.control_type, node.automation_id, node.name)


@dataclass(frozen=True)
class _Binding:
    node: Any  # Retained UIAElementInfo; never re-resolved by name or index.
    identity: _Identity


@dataclass
class _Session:
    observation: Observation
    root: _Binding
    handle: int
    controls: dict[str, _Binding]
    focused: _Binding | None
    created: float


class WindowsComputer(WindowsObserver):
    """Synchronous manual executor; deliberately not an agent loop.

    A UI action consumes its snapshot even on failure. Observe again before
    another action. No serialized or copied snapshot can authorize execution.
    Additional policy can restrict the baseline, never relax it.
    """

    def __init__(
        self, options: ObservationOptions | None = None, policy: ActionPolicy | None = None,
        app_catalog: ApplicationCatalog | None = None,
        *, capture_service: ScreenCapture | None = None,
        visual_provider: VisualObserver | None = None,
        visual_min_confidence: float = 0.75,
        retain_debug_capture: bool = False,
        collect_provider_candidate_diagnostics: bool = False,
        visual_readiness_options: VisualReadinessOptions | None = None,
        result_readiness_options=None,
        readiness_clock=time.monotonic,
        readiness_sleep=time.sleep,
    ) -> None:
        super().__init__(
            options, app_catalog, capture_service=capture_service, visual_provider=visual_provider,
            retain_debug_capture=retain_debug_capture,
            collect_provider_candidate_diagnostics=collect_provider_candidate_diagnostics,
            visual_readiness_options=visual_readiness_options,
            result_readiness_options=result_readiness_options,
            readiness_clock=readiness_clock, readiness_sleep=readiness_sleep,
        )
        self.policy = policy
        self.app_catalog = app_catalog
        self.visual_min_confidence = visual_min_confidence
        self._session: _Session | None = None

    def observe(self) -> Observation:
        self._session = None
        observation = super().observe()
        if observation.error or self._root is None:
            return observation
        try:
            root = _Binding(self._root, _identity(self._root))
            limit = self.options.max_text_length
            if (root.identity.name.strip()[:limit] != observation.window_title
                    or root.identity.control_type.strip()[:limit] != observation.control_type):
                return observation  # The window changed while it was being read.
            handle = self._root.handle  # type: ignore[attr-defined]
            if not handle:
                return observation
            bindings: dict[str, _Binding] = {}
            descriptions = {control.id: control for control in observation.elements}
            for control_id, node in self._nodes.items():
                try:
                    identity = _identity(node)
                    description = descriptions[control_id]
                    if (identity.name.strip()[:limit] != description.name
                            or identity.control_type.strip()[:limit] != description.control_type
                            or identity.automation_id.strip()[:limit] != description.automation_id):
                        continue  # Never bind a control that changed during traversal.
                    bindings[control_id] = _Binding(node, identity)
                except Exception:
                    continue  # Readable controls aren't necessarily safely actionable.
            focused = None
            try:
                focus_identity = _identity(_focused())
                focused = next((b for b in [root, *bindings.values()] if b.identity == focus_identity), None)
            except Exception:
                pass  # Click may still work; typing and keys require a bound focus.
            self._session = _Session(observation, root, handle, bindings, focused, time.monotonic())
        except Exception:
            self._session = None
        return observation

    def observe_local(self) -> Observation:
        """Observe without invoking the configured remote visual provider."""
        provider, grounding = self.visual_provider, self.visual_grounding
        self.visual_provider = None
        self.visual_grounding = None
        try:
            return self.observe()
        finally:
            self.visual_provider = provider
            self.visual_grounding = grounding

    def observe_directed(self, grounding: VisualGroundingRequest) -> Observation:
        """Create a fresh directed hybrid snapshot with normal local bindings."""
        if self.visual_provider is None:
            raise RuntimeError("A visual provider is required for directed observation.")
        previous = self.visual_grounding
        self.visual_grounding = grounding
        try:
            return self.observe()
        finally:
            self.visual_grounding = previous

    def observe_result_baseline(self) -> Observation:
        """Capture one masked post-type/pre-submit baseline without a provider call."""
        provider = self.visual_provider
        old_capture = self.capture_without_provider
        old_force = self.force_capture
        self.visual_provider = None
        self.capture_without_provider = True
        self.force_capture = True
        try:
            return self.observe()
        finally:
            self.visual_provider = provider
            self.capture_without_provider = old_capture
            self.force_capture = old_force

    def observe_result_directed(
        self, grounding: VisualGroundingRequest, baseline: Observation,
    ) -> Observation:
        """Poll masked local frames, then ground exactly the final accepted capture once."""
        previous = self.result_readiness_enabled
        previous_baseline = self.result_readiness_baseline
        self.result_readiness_baseline = (
            visual_readiness_frame(baseline.capture_diagnostics)
            if baseline.capture_diagnostics is not None else None
        )
        self.result_readiness_enabled = True
        try:
            return self.observe_directed(grounding)
        finally:
            self.result_readiness_enabled = previous
            self.result_readiness_baseline = previous_baseline

    def execute_query_submit_phase3(
        self, action: QuerySubmitAction, observation: Observation,
        verified_search_observation: Observation,
    ) -> ActionResult:
        """Consume one post-type snapshot and issue exactly one Enter after revalidation."""
        session = self._session
        self._session = None
        try:
            if action.key != "enter":
                raise UnsafeTarget("Only Enter is supported for bounded query submission.")
            if session is None or observation is not session.observation:
                raise UnsafeTarget("Missing, foreign, copied, consumed, or superseded observation.")
            current_meta = observation.screenshot
            verified_meta = verified_search_observation.screenshot
            if current_meta is None or verified_meta is None or self.capture_service is None:
                raise UnsafeTarget("Bound query context geometry is unavailable.")
            if (current_meta.window_handle != verified_meta.window_handle
                    or observation.process_id != verified_search_observation.process_id
                    or current_meta.window_bounds != verified_meta.window_bounds
                    or not observation.application_id
                    or observation.application_id != verified_search_observation.application_id):
                raise StaleObservation("Verified query context changed before submission.")
            if any(c.focused is True and c.control_type in {"Edit", "Document"}
                   and c.is_password is not False for c in observation.elements):
                raise UnsafeTarget("Credential-sensitive editor context is active.")
            self._check_window(session)
            if self.capture_service.current_window_bounds(session.handle) != current_meta.window_bounds:
                raise StaleObservation("Window geometry changed before query submission.")
            self._check_window(session)
            _send_key(("enter",))
            return ActionResult(
                True, action, "One bounded query-submit Enter was sent.",
                source_observation_id=observation.observation_id, input_issued=True,
            )
        except StaleObservation as exc:
            return ActionResult(False, action, str(exc), error="stale_observation")
        except UnsafeTarget as exc:
            return ActionResult(False, action, str(exc), error="unsafe_target")
        except Exception:
            return ActionResult(False, action, "Windows query submission failed.",
                                error="windows_operation_failed", input_issued=False)

    def execute_generic_query_submit(
        self, action: QuerySubmitAction, observation: Observation,
    ) -> ActionResult:
        """Submit one locally verified generic query from its exact fresh UIA snapshot."""
        session = self._session
        self._session = None
        try:
            if action.key != "enter":
                raise UnsafeTarget("Only the bounded Enter query-submit action is supported.")
            if session is None or observation is not session.observation:
                raise UnsafeTarget("Missing, foreign, copied, consumed, or superseded observation.")
            if observation.error or not observation.observation_id:
                raise UnsafeTarget("A fresh foreground observation is required.")
            if any(control.focused is True and control.control_type in {"Edit", "Document"}
                   and control.is_password is not False for control in observation.elements):
                raise UnsafeTarget("Credential-sensitive editor context is active.")
            binding = session.focused
            focused_control = next((control for control in observation.elements
                                    if control.focused is True and binding is not None
                                    and control.id in session.controls
                                    and session.controls[control.id].identity == binding.identity), None)
            if (focused_control is None
                    or not GenericTargetActivationPolicy.is_query_field(focused_control)):
                raise UnsafeTarget("Focused control is not a verified generic query field.")
            if binding is None:
                raise UnsafeTarget("Focused control identity is unavailable.")
            wrapper = self._check_focus(binding, session)
            if binding.node.element.CurrentIsPassword:
                raise UnsafeTarget("Password fields cannot submit a query.")
            value_pattern = wrapper.iface_value
            if value_pattern.CurrentIsReadOnly:
                raise UnsafeTarget("Query field is read-only.")
            if (focused_control.observed_text is None or focused_control.observed_text_truncated
                    or value_pattern.CurrentValue != focused_control.observed_text):
                raise StaleObservation("Query value changed after its local observation.")
            self._check_window(session)
            _send_key(("enter",))
            return ActionResult(
                True, action, "One locally verified query-submit Enter was sent.",
                source_observation_id=observation.observation_id, input_issued=True,
            )
        except StaleObservation as exc:
            return ActionResult(False, action, str(exc), error="stale_observation")
        except UnsafeTarget as exc:
            return ActionResult(False, action, str(exc), error="unsafe_target")
        except Exception:
            return ActionResult(False, action, "Generic query submission failed.",
                                error="windows_operation_failed", input_issued=False)

    def _check_window(self, session: _Session) -> None:
        current = _foreground()
        if time.monotonic() - session.created > 120:
            raise UnsafeTarget("Observation expired; observe again.")
        if current.handle != session.handle or _identity(current) != session.root.identity:  # type: ignore[attr-defined]
            raise UnsafeTarget("Foreground window changed; observe again.")
        if _identity(session.root.node) != session.root.identity:
            raise UnsafeTarget("Original window is stale.")
        if not current.visible or not current.enabled:
            raise UnsafeTarget("Window is not visible and enabled.")

    def _check_control(self, binding: _Binding, session: _Session) -> Any:
        node = binding.node
        if _identity(node) != binding.identity or not node.visible or not node.enabled:
            raise UnsafeTarget("Control changed, disappeared, or is unavailable.")
        parent = node
        for _ in range(128):
            if parent is None:
                break
            if _identity(parent) == session.root.identity:
                return _wrapper(node)
            parent = parent.parent
        raise UnsafeTarget("Control no longer belongs to the observed window.")

    def _check_focus(self, binding: _Binding, session: _Session) -> Any:
        wrapper = self._check_control(binding, session)
        if _identity(_focused()) != binding.identity:
            raise UnsafeTarget("Focused control changed; observe again.")
        return wrapper

    def _execute_visual_click(
        self, action: VisualClickAction, observation: Observation, session: _Session,
        *, require_phase1_geometry: bool,
    ) -> ActionResult:
        metadata = observation.screenshot
        element = next((item for item in observation.visual_elements
                        if item.id == action.target_id), None)
        if metadata is None or element is None:
            raise UnsafeTarget("Visual target or capture metadata is unavailable.")
        if (metadata.snapshot_id != action.snapshot_id
                or action.snapshot_id != observation.observation_id):
            raise StaleObservation("Visual target belongs to a different snapshot; observe again.")
        if metadata.window_handle != session.handle:
            raise StaleObservation("Visual capture belongs to a different window.")
        if self.capture_service is None:
            raise UnsafeTarget("No screen-capture geometry service is configured.")
        self._check_window(session)
        current_bounds = self.capture_service.current_window_bounds(session.handle)
        if current_bounds != metadata.window_bounds:
            raise StaleObservation("Window moved or resized; observe again.")
        virtual_bounds = None
        if require_phase1_geometry:
            diagnostics = observation.capture_diagnostics
            current_virtual = getattr(self.capture_service, "current_virtual_screen_bounds", None)
            if diagnostics is None or not callable(current_virtual):
                raise UnsafeTarget("Phase-1 capture context is unavailable.")
            virtual_bounds = current_virtual()
            if virtual_bounds != diagnostics.virtual_screen_bounds:
                raise StaleObservation("Virtual desktop geometry changed; observe again.")
            expected_capture = Rect(
                max(current_bounds.left, virtual_bounds.left),
                max(current_bounds.top, virtual_bounds.top),
                min(current_bounds.right, virtual_bounds.right),
                min(current_bounds.bottom, virtual_bounds.bottom),
            )
            if expected_capture != metadata.capture_bounds:
                raise StaleObservation("Capture bounds changed; observe again.")
        screen_rect = visual_rect_to_screen(element.rectangle, metadata)
        bounds = metadata.capture_bounds
        if (screen_rect.left < bounds.left or screen_rect.top < bounds.top
                or screen_rect.right > bounds.right or screen_rect.bottom > bounds.bottom
                or screen_rect.right <= screen_rect.left or screen_rect.bottom <= screen_rect.top):
            raise UnsafeTarget("Visual target lies outside the captured foreground window.")
        point = visual_click_point(element, metadata)
        if not (screen_rect.left <= point[0] < screen_rect.right
                and screen_rect.top <= point[1] < screen_rect.bottom
                and bounds.left <= point[0] < bounds.right
                and bounds.top <= point[1] < bounds.bottom):
            raise UnsafeTarget("Visual click point is invalid.")
        if virtual_bounds is not None and not (
            virtual_bounds.left <= point[0] < virtual_bounds.right
            and virtual_bounds.top <= point[1] < virtual_bounds.bottom
        ):
            raise UnsafeTarget("Visual click point is outside the virtual desktop.")
        self._check_window(session)
        _click_point(*point)
        return ActionResult(
            True, action, "One validated visual target click performed.",
            source_observation_id=observation.observation_id, input_issued=True,
        )

    def execute_visual_click_phase1(
        self, action: VisualClickAction, observation: Observation,
        request: str, jev_confidence: float | None,
    ) -> ActionResult:
        """Consume one bound snapshot and attempt one phase-1 visual click."""
        session = self._session
        self._session = None
        try:
            verdict = Phase1VisualClickPolicy().validate(action, observation, request, jev_confidence)
            if verdict.disposition != "allow":
                return ActionResult(False, action, verdict.reason, error="policy_blocked")
            if session is None or observation is not session.observation:
                raise UnsafeTarget("Missing, foreign, copied, consumed, or superseded observation.")
            return self._execute_visual_click(action, observation, session, require_phase1_geometry=True)
        except StaleObservation as exc:
            return ActionResult(False, action, str(exc), error="stale_observation")
        except UnsafeTarget as exc:
            return ActionResult(False, action, str(exc), error="unsafe_target")
        except Exception:
            return ActionResult(False, action, "Windows visual click input failed.",
                                error="windows_operation_failed", input_issued=False)

    def execute_visual_click_phase3(
        self, action: VisualClickAction, observation: Observation,
    ) -> ActionResult:
        """Consume one exact result snapshot and attempt one validated result click."""
        from safety.policy import Phase3ResultSelectionPolicy
        session = self._session
        self._session = None
        try:
            verdict = Phase3ResultSelectionPolicy().validate(action, observation)
            if verdict.disposition != "allow":
                return ActionResult(False, action, verdict.reason, error="policy_blocked")
            if session is None or observation is not session.observation:
                raise UnsafeTarget("Missing, foreign, copied, consumed, or superseded observation.")
            return self._execute_visual_click(action, observation, session, require_phase1_geometry=True)
        except StaleObservation as exc:
            return ActionResult(False, action, str(exc), error="stale_observation")
        except UnsafeTarget as exc:
            return ActionResult(False, action, str(exc), error="unsafe_target")
        except Exception:
            return ActionResult(False, action, "Windows result click input failed.",
                                error="windows_operation_failed", input_issued=False)

    def execute_generic_target_activation(
        self, action: VisualClickAction, observation: Observation,
    ) -> ActionResult:
        """Execute one generic visual target with the bounded generic safety gate."""
        session = self._session
        self._session = None
        try:
            verdict = GenericTargetActivationPolicy(self.app_catalog).validate_candidate(
                action, observation,
            )
            if verdict.disposition != "allow":
                return ActionResult(False, action, verdict.reason, error="policy_blocked")
            if session is None or observation is not session.observation:
                raise UnsafeTarget("Missing, foreign, copied, consumed, or superseded observation.")
            return self._execute_visual_click(action, observation, session, require_phase1_geometry=True)
        except StaleObservation as exc:
            return ActionResult(False, action, str(exc), error="stale_observation")
        except UnsafeTarget as exc:
            return ActionResult(False, action, str(exc), error="unsafe_target")
        except Exception:
            return ActionResult(False, action, "Generic visual target activation failed.",
                                error="windows_operation_failed", input_issued=False)
    def execute_type_phase2(
        self, action: TypeAction, observation: Observation, *, visual_verified: bool,
    ) -> ActionResult:
        """Consume the fresh post-click snapshot and issue one literal TypeAction."""
        session = self._session
        self._session = None
        literal_diagnostic = None
        try:
            verdict = BasicActionPolicy(self.app_catalog).validate(action, observation)
            if verdict.disposition != "allow":
                return ActionResult(False, action, verdict.reason, error="policy_blocked")
            if session is None or observation is not session.observation:
                raise UnsafeTarget("Missing, foreign, copied, consumed, or superseded observation.")
            if any(control.focused is True and control.control_type in {"Edit", "Document"}
                   and control.is_password is not False for control in observation.elements):
                raise UnsafeTarget("Credential-sensitive editor context is active.")
            if session.focused is not None and session.focused.identity.control_type in {"Edit", "Document"}:
                self._check_window(session)
                wrapper = self._check_focus(session.focused, session)
                if session.focused.node.element.CurrentIsPassword:
                    raise UnsafeTarget("Password fields are not supported.")
                value = wrapper.iface_value
                if value.CurrentIsReadOnly:
                    raise UnsafeTarget("Focused text control is read-only.")
                self._check_window(session)
                value.SetValue(action.text)
            elif visual_verified:
                current = _foreground()
                current_identity = _identity(current)
                if (int(current.handle) != session.handle  # type: ignore[attr-defined]
                        or current_identity.process_id != session.root.identity.process_id
                        or current_identity != session.root.identity):
                    raise StaleObservation("Foreground context changed immediately before typing.")
                if _identity(session.root.node) != session.root.identity:
                    raise StaleObservation("Original foreground context became stale before typing.")
                literal_diagnostic = _type_literal_unicode(
                    action.text, foreground_hwnd=session.handle,
                    foreground_pid=current_identity.process_id,
                )
            else:
                raise UnsafeTarget("Typing focus was not safely verified.")
            return ActionResult(
                True, action, "One literal TypeAction performed without submission.",
                source_observation_id=observation.observation_id, input_issued=True,
                literal_input_diagnostic=literal_diagnostic,
            )
        except LiteralInputFailure as exc:
            return ActionResult(
                False, action, "Windows literal type input failed.",
                error="windows_operation_failed", input_issued=False,
                literal_input_diagnostic=exc.diagnostic,
            )
        except StaleObservation as exc:
            units = len(action.text.encode("utf-16-le")) // 2
            return ActionResult(
                False, action, str(exc), error="stale_observation",
                literal_input_diagnostic=LiteralInputDiagnostic(
                    len(action.text), units, units * 2, 0, 0, None,
                    "foreground_context_changed", ctypes.sizeof(_INPUT),
                    session.handle if session is not None else None,
                    session.root.identity.process_id if session is not None else None,
                    False,
                ),
            )
        except UnsafeTarget as exc:
            return ActionResult(False, action, str(exc), error="unsafe_target")
        except Exception:
            return ActionResult(
                False, action, "Windows literal type input failed.",
                error="windows_operation_failed", input_issued=False,
            )

    def visual_context_matches(self, observation: Observation) -> bool:
        """Compare the current bound foreground session to an earlier visual snapshot."""
        session = self._session
        metadata = observation.screenshot
        if session is None or metadata is None or self.capture_service is None:
            return False
        try:
            self._check_window(session)
            return (
                session.handle == metadata.window_handle
                and session.root.identity.process_id == observation.process_id
                and self.capture_service.current_window_bounds(session.handle)
                == metadata.window_bounds
            )
        except Exception:
            return False

    def execute(self, action: Action, observation: Observation | None = None) -> ActionResult:
        session = self._session
        # Consume up front: no retries of potentially partially applied actions.
        self._session = None
        try:
            policies = [BasicActionPolicy(
                self.app_catalog, visual_min_confidence=self.visual_min_confidence,
            )]
            if self.policy is not None:
                policies.append(self.policy)
            for policy in policies:
                verdict = policy.validate(action, observation)
                if verdict.disposition != "allow":
                    return ActionResult(False, action, verdict.reason, error="policy_blocked",
                                        requires_confirmation=verdict.disposition == "confirm")
            if isinstance(action, FinishAction):
                return ActionResult(True, action, action.summary, completed=True)
            if isinstance(action, OpenAppAction):
                if self.app_catalog is None:
                    raise UnsafeTarget("No trusted application catalog is configured.")
                self.app_catalog.launch(action.app_id)
                return ActionResult(True, action, "Application launch requested; observe before interacting.")
            if session is None or observation is not session.observation:
                raise UnsafeTarget("Missing, foreign, copied, consumed, or superseded observation.")
            self._check_window(session)
            if isinstance(action, VisualClickAction):
                if not observation.visual_execution_authorized:
                    return ActionResult(
                        False, action,
                        "Real-provider visual actions are observation-only in this release.",
                        error="policy_blocked",
                    )
                return self._execute_visual_click(
                    action, observation, session, require_phase1_geometry=False,
                )
            if isinstance(action, ClickAction):
                binding = session.controls.get(action.target_id)
                if binding is None:
                    raise UnsafeTarget("Control has no safe binding in this observation.")
                wrapper = self._check_control(binding, session)
                self._check_window(session)
                kind = binding.identity.control_type
                if kind in {"Edit", "Document"}:
                    # Raw UIA SetFocus reports failure instead of pywinauto's warning.
                    binding.node.element.SetFocus()
                    self._check_focus(binding, session)
                elif kind == "CheckBox":
                    wrapper.iface_toggle.Toggle()
                elif kind in {"TabItem", "ListItem", "TreeItem", "RadioButton"}:
                    wrapper.iface_selection_item.Select()
                else:
                    wrapper.iface_invoke.Invoke()
                return ActionResult(True, action, "Semantic control action performed.")
            if session.focused is None:
                raise UnsafeTarget("Focused control was not safely captured; observe again.")
            wrapper = self._check_focus(session.focused, session)
            if isinstance(action, TypeAction):
                if session.focused.identity.control_type not in {"Edit", "Document"}:
                    raise UnsafeTarget("Focused control is not an editable text control.")
                if session.focused.node.element.CurrentIsPassword:
                    raise UnsafeTarget("Password fields are not supported.")
                value = wrapper.iface_value
                if value.CurrentIsReadOnly:
                    raise UnsafeTarget("Focused text control is read-only.")
                self._check_window(session)
                self._check_focus(session.focused, session)
                value.SetValue(action.text)  # Literal BSTR, never a key expression.
                return ActionResult(True, action, "Editable value replaced literally; no submit key sent.")
            if isinstance(action, PressKeyAction):
                self._check_window(session)
                self._check_focus(session.focused, session)
                _send_key(tuple(key.lower() for key in action.keys))
                return ActionResult(True, action, "Allowlisted key combination sent.")
            return ActionResult(False, action, "Unsupported action.", error="unsupported_action")
        except StaleObservation as exc:
            return ActionResult(False, action, str(exc), error="stale_observation")
        except UnsafeTarget as exc:
            return ActionResult(False, action, str(exc), error="unsafe_target")
        except Exception:
            # Provider failures can happen after partial effects. No fallback/retry.
            return ActionResult(False, action,
                                "Windows operation failed or its UIA pattern is unavailable; observe before retrying.",
                                error="windows_operation_failed")

