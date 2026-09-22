"""Verify the data contract without accessing Windows or external services."""

from dataclasses import FrozenInstanceError, asdict

import pytest

from computer.actions import (
    Action, ClickAction, FinishAction, OpenAppAction, PressKeyAction, TypeAction, VisualClickAction,
)
from computer.models import Observation, UIElement


@pytest.mark.parametrize(
    ("action", "kind", "payload"),
    [
        (ClickAction("button-1"), "click", {"target_id": "button-1"}),
        (VisualClickAction("snapshot-1", "v1"), "visual_click", {
            "snapshot_id": "snapshot-1", "target_id": "v1",
        }),
        (TypeAction("Hello {ENTER}\n世界"), "type", {"text": "Hello {ENTER}\n世界"}),
        (OpenAppAction("app_0123456789abcdef"), "open_app", {"app_id": "app_0123456789abcdef"}),
        (PressKeyAction(("CTRL", "L")), "press_key", {"keys": ("CTRL", "L")}),
        (FinishAction("Done"), "finish", {"summary": "Done"}),
    ],
)
def test_action_payload_and_immutability(
    action: Action, kind: str, payload: dict[str, object],
) -> None:
    assert asdict(action) == {"kind": kind, **payload}
    with pytest.raises(FrozenInstanceError):
        setattr(action, "kind", "changed")


def test_observation_contains_platform_neutral_elements() -> None:
    element = UIElement("button-1", "Save", "Button")
    observation = Observation("Editor", "Untitled", (element,))
    assert observation.elements[0].id == "button-1"
    assert Observation("Editor", "Empty").elements == ()
    with pytest.raises(FrozenInstanceError):
        setattr(element, "name", "changed")
