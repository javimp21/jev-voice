"""Thin, snapshot-bound MCP-facing adapters over existing computer contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from threading import RLock
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from computer.actions import (
    ClickAction, OpenAppAction, PressKeyAction, TypeAction,
)
from computer.applications import ApplicationCatalog, match_applications
from computer.interfaces import Computer
from computer.models import Observation, UIElement
from computer.results import ActionResult
from decision.context import Redactor
from safety.interfaces import ActionPolicy, SafetyDecision


MAX_SUMMARY_CONTROLS = 40
MAX_CONTROL_NAME = 120
MAX_WINDOW_TITLE = 160
MAX_APP_QUERY = 120
MAX_TEXT_INPUT = 4000
MAX_ID_LENGTH = 64
MAX_INSPECTION_ERRORS = 500

ActionKind = Literal["open_app", "click", "type", "press_key", "unsupported"]
ActionErrorCategory = Literal[
    "invalid_snapshot_binding", "unsupported_action", "application_not_found",
    "ambiguous_application", "safety_rejected", "confirmation_required",
    "executor_failure", "observation_failed", "invalid_action_input",
]
ObservationErrorCategory = Literal["observation_unavailable", "invalid_observation"]

_SUPPORTED_ACTION_TYPES = (OpenAppAction, ClickAction, TypeAction, PressKeyAction)
_ACTION_ERROR_MAP = {
    "policy_blocked": "safety_rejected",
    "unsafe_target": "safety_rejected",
    "stale_observation": "invalid_snapshot_binding",
    "unsupported_action": "unsupported_action",
    "windows_operation_failed": "executor_failure",
}
_OBSERVATION_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_APP_ID_RE = re.compile(r"^app_[0-9a-f]{16}$")
_CONTROL_ID_RE = re.compile(r"^c[1-9][0-9]{0,2}$")


class ElementSummary(BaseModel):
    """Bounded accessible-control summary with no value or geometry fields."""

    model_config = ConfigDict(extra="forbid")

    control_id: str = Field(pattern=r"^c[1-9][0-9]{0,2}$")
    name: str = Field(max_length=MAX_CONTROL_NAME)
    control_type: str = Field(max_length=40)
    enabled: bool | None
    visible: bool
    focused: bool | None


class ObservationSummary(BaseModel):
    """Safe observation projection; deliberately excludes raw observations."""

    model_config = ConfigDict(extra="forbid")

    observation_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    app_name: str = Field(max_length=80)
    trusted_application_id: str | None = Field(default=None, pattern=r"^app_[0-9a-f]{16}$")
    window_title: str = Field(max_length=MAX_WINDOW_TITLE)
    elements: list[ElementSummary] = Field(max_length=MAX_SUMMARY_CONTROLS)
    truncated: bool
    inspection_error_count: int = Field(ge=0, le=MAX_INSPECTION_ERRORS)
    error_category: ObservationErrorCategory | None = None


class ActionSummary(BaseModel):
    """Action outcome without action payloads, exception text, or UI state."""

    model_config = ConfigDict(extra="forbid")

    success: bool
    action_kind: ActionKind
    error_category: ActionErrorCategory | None = None
    requires_confirmation: bool = False
    fresh_observation_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")


@dataclass
class ComputerToolService:
    """Serialize snapshot use and delegate every effect to policy + Computer."""

    computer: Computer
    policy: ActionPolicy
    catalog: ApplicationCatalog
    _observation: Observation | None = field(default=None, init=False, repr=False)
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def observe(self) -> ObservationSummary:
        """Take one local UIA observation and retain its exact snapshot object."""

        with self._lock:
            try:
                observation = self.computer.observe()
            except KeyboardInterrupt:
                raise
            except Exception:
                self._observation = None
                return _unavailable_observation()
            return self._bind_observation(observation)

    def open_app(self, app_name: str) -> ActionSummary:
        """Resolve a unique trusted catalog candidate, then use OpenAppAction."""

        if not isinstance(app_name, str) or not app_name.strip() or len(app_name) > MAX_APP_QUERY:
            return _action_failure("open_app", "invalid_action_input")
        try:
            matches = match_applications(app_name.strip(), self.catalog.discover(), limit=5)
        except Exception:
            return _action_failure("open_app", "application_not_found")
        if not matches:
            return _action_failure("open_app", "application_not_found")
        top_score = matches[0].score
        best = tuple(match for match in matches if match.score == top_score)
        if len(best) != 1:
            return _action_failure("open_app", "ambiguous_application")
        return self.perform_action(OpenAppAction(best[0].candidate.id))

    def click(self, observation_id: str, target_id: str) -> ActionSummary:
        return self.perform_action(ClickAction(target_id), observation_id=observation_id)

    def type_text(self, observation_id: str, text: str) -> ActionSummary:
        if not isinstance(text, str) or len(text) > MAX_TEXT_INPUT:
            return _action_failure("type", "invalid_action_input")
        return self.perform_action(TypeAction(text), observation_id=observation_id)

    def press_key(
        self,
        observation_id: str,
        key_combo: Literal["enter", "escape", "tab", "shift+tab", "ctrl+a", "ctrl+c"],
    ) -> ActionSummary:
        return self.perform_action(
            PressKeyAction(tuple(key_combo.casefold().split("+"))),
            observation_id=observation_id,
        )

    def perform_action(
        self, action: object, *, observation_id: str | None = None,
    ) -> ActionSummary:
        """Preflight once, execute at most once, then observe once for rebinding."""

        kind = _action_kind(action)
        if not isinstance(action, _SUPPORTED_ACTION_TYPES):
            return _action_failure(kind, "unsupported_action")
        with self._lock:
            requires_snapshot = not isinstance(action, OpenAppAction)
            observation = self._observation
            if requires_snapshot:
                if (observation is None or not observation.observation_id
                        or not isinstance(observation_id, str)
                        or len(observation_id) > MAX_ID_LENGTH
                        or observation_id != observation.observation_id):
                    return _action_failure(kind, "invalid_snapshot_binding")
            try:
                verdict = self.policy.validate(action, observation)
            except KeyboardInterrupt:
                raise
            except Exception:
                return _action_failure(kind, "safety_rejected")
            if not isinstance(verdict, SafetyDecision) or verdict.disposition != "allow":
                if isinstance(verdict, SafetyDecision) and verdict.disposition == "confirm":
                    return _action_failure(
                        kind, "confirmation_required", requires_confirmation=True,
                    )
                return _action_failure(kind, "safety_rejected")

            # The Windows executor consumes its current UIA session on every
            # execute call. Drop our copy before entering it as an extra guard.
            self._observation = None
            try:
                result = self.computer.execute(action, observation)
            except KeyboardInterrupt:
                raise
            except Exception:
                fresh_id = self._observe_after_action()
                return _action_failure(kind, "executor_failure", fresh_id=fresh_id)
            fresh_id = self._observe_after_action()
            if not isinstance(result, ActionResult):
                return _action_failure(kind, "executor_failure", fresh_id=fresh_id)
            if result.success:
                return ActionSummary(
                    success=True,
                    action_kind=kind,
                    error_category="observation_failed" if fresh_id is None else None,
                    fresh_observation_id=fresh_id,
                )
            category = (
                "confirmation_required" if result.requires_confirmation
                else _ACTION_ERROR_MAP.get(result.error or "", "executor_failure")
            )
            return ActionSummary(
                success=False,
                action_kind=kind,
                error_category=category,
                requires_confirmation=result.requires_confirmation,
                fresh_observation_id=fresh_id,
            )

    def _bind_observation(self, observation: object) -> ObservationSummary:
        if not isinstance(observation, Observation):
            self._observation = None
            return _unavailable_observation("invalid_observation")
        if observation.error:
            self._observation = None
            return _observation_summary(observation, observation_id=None,
                                       error_category="observation_unavailable")
        if not _OBSERVATION_ID_RE.fullmatch(observation.observation_id):
            self._observation = None
            return _observation_summary(observation, observation_id=None,
                                       error_category="invalid_observation")
        self._observation = observation
        return _observation_summary(observation, observation_id=observation.observation_id)

    def _observe_after_action(self) -> str | None:
        try:
            observation = self.computer.observe()
        except KeyboardInterrupt:
            raise
        except Exception:
            self._observation = None
            return None
        summary = self._bind_observation(observation)
        return summary.observation_id


def _action_kind(action: object) -> ActionKind:
    if isinstance(action, OpenAppAction):
        return "open_app"
    if isinstance(action, ClickAction):
        return "click"
    if isinstance(action, TypeAction):
        return "type"
    if isinstance(action, PressKeyAction):
        return "press_key"
    return "unsupported"


def _action_failure(
    kind: ActionKind,
    category: ActionErrorCategory,
    *,
    requires_confirmation: bool = False,
    fresh_id: str | None = None,
) -> ActionSummary:
    return ActionSummary(
        success=False,
        action_kind=kind,
        error_category=category,
        requires_confirmation=requires_confirmation,
        fresh_observation_id=fresh_id,
    )


def _clean(value: str, limit: int, redactor: Redactor) -> tuple[str, bool]:
    value = re.sub(r"[\x00-\x1f\x7f]+", " ", redactor.clean(value))
    value = " ".join(value.split())
    return value[:limit], len(value) > limit


def _observation_summary(
    observation: Observation,
    *,
    observation_id: str | None,
    error_category: ObservationErrorCategory | None = None,
) -> ObservationSummary:
    redactor = Redactor()
    app_name, app_truncated = _clean(observation.app_name, 80, redactor)
    window_title, title_truncated = _clean(observation.window_title, MAX_WINDOW_TITLE, redactor)
    app_id = observation.application_id
    trusted_id = app_id if isinstance(app_id, str) and _APP_ID_RE.fullmatch(app_id) else None
    elements: list[ElementSummary] = []
    eligible = tuple(
        element for element in observation.elements
        if isinstance(element, UIElement)
        and element.visible is True
        and element.is_password is not True
        and (element.control_type not in {"Edit", "Document"} or element.is_password is False)
        and isinstance(element.id, str)
        and _CONTROL_ID_RE.fullmatch(element.id)
    )
    elements_truncated = len(eligible) > MAX_SUMMARY_CONTROLS
    for element in eligible[:MAX_SUMMARY_CONTROLS]:
        name, was_truncated = _clean(element.name, MAX_CONTROL_NAME, redactor)
        control_type, type_truncated = _clean(element.control_type, 40, redactor)
        elements_truncated |= was_truncated or type_truncated
        elements.append(ElementSummary(
            control_id=element.id,
            name=name,
            control_type=control_type,
            enabled=element.enabled if type(element.enabled) is bool else None,
            visible=True,
            focused=element.focused if type(element.focused) is bool else None,
        ))
    return ObservationSummary(
        observation_id=observation_id,
        app_name=app_name,
        trusted_application_id=trusted_id,
        window_title=window_title,
        elements=elements,
        truncated=(observation.truncated or app_truncated or title_truncated or elements_truncated),
        inspection_error_count=min(
            max(observation.inspection_errors, 0), MAX_INSPECTION_ERRORS,
        ),
        error_category=error_category,
    )


def _unavailable_observation(
    error_category: ObservationErrorCategory = "observation_unavailable",
) -> ObservationSummary:
    return ObservationSummary(
        observation_id=None,
        app_name="",
        trusted_application_id=None,
        window_title="",
        elements=[],
        truncated=False,
        inspection_error_count=1,
        error_category=error_category,
    )
