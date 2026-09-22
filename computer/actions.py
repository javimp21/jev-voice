"""Typed action requests; constructing an action never executes it."""

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True, slots=True)
class ClickAction:
    """Click an opaque element ID from the latest observation."""

    target_id: str
    kind: Literal["click"] = field(default="click", init=False)


@dataclass(frozen=True, slots=True)
class VisualClickAction:
    """Click a validated visual target from one exact screenshot snapshot."""

    snapshot_id: str
    target_id: str
    kind: Literal["visual_click"] = field(default="visual_click", init=False)


@dataclass(frozen=True, slots=True)
class TypeAction:
    """Replace the focused editable control's whole value literally; do not submit."""

    text: str
    kind: Literal["type"] = field(default="type", init=False)


@dataclass(frozen=True, slots=True)
class OpenAppAction:
    """Request an application by an opaque ID from the trusted local catalog."""

    app_id: str
    kind: Literal["open_app"] = field(default="open_app", init=False)


@dataclass(frozen=True, slots=True)
class PressKeyAction:
    """Press a chord of semantic key names, e.g. ('CTRL', 'L').

    A future Windows adapter must validate and translate these names rather
    than treating them as raw pywinauto keyboard syntax.
    """

    keys: tuple[str, ...]
    kind: Literal["press_key"] = field(default="press_key", init=False)


@dataclass(frozen=True, slots=True)
class QuerySubmitAction:
    """Submit one previously verified and deterministically populated query context."""

    key: Literal["enter"] = "enter"
    kind: Literal["query_submit"] = field(default="query_submit", init=False)


@dataclass(frozen=True, slots=True)
class FinishAction:
    """Signal completion to the orchestrator; this is not a desktop action."""

    summary: str
    kind: Literal["finish"] = field(default="finish", init=False)


type ComputerAction = (ClickAction | VisualClickAction | TypeAction | OpenAppAction
                       | PressKeyAction | QuerySubmitAction)
type Action = ComputerAction | FinishAction
