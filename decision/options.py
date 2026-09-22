"""Semantic Choice descriptions; option keys still map to local typed actions."""

import json
from collections.abc import Mapping

from computer.actions import Action, ClickAction, FinishAction, OpenAppAction, PressKeyAction, TypeAction, VisualClickAction
from computer.applications import ApplicationCandidate
from computer.models import UIElement, VisualElement
from decision.context import Redactor

_CLICK_EFFECTS = {
    "Document": "Focus this document/text-editing area.",
    "Edit": "Focus this editable text field.",
    "Button": "Activate this button.",
    "MenuItem": "Activate this menu item.",
    "Hyperlink": "Activate this link.",
    "CheckBox": "Toggle this check box.",
    "RadioButton": "Select this radio button.",
    "ListItem": "Select this list item.",
    "TreeItem": "Select this tree item.",
    "TabItem": "Select this tab.",
}
_KEY_EFFECTS = {
    ("enter",): "Confirm or submit in the focused control; may activate its default action.",
    ("escape",): "Dismiss or cancel the current popup or interaction.",
    ("tab",): "Move keyboard focus to the next control.",
    ("shift", "tab"): "Move keyboard focus to the previous control.",
    ("ctrl", "a"): "Select all in the focused control.",
    ("ctrl", "c"): "Copy the current selection to the clipboard.",
}


def describe_action(
    action: Action | None, controls: Mapping[str, UIElement], redactor: Redactor,
    applications: Mapping[str, ApplicationCandidate] | None = None,
    visual_controls: Mapping[str, VisualElement] | None = None,
) -> str:
    if action is None:
        return ("Stop and request human intervention: the task is unfinished and NO safe supported "
                "offered action can progress it. Use for missing targets, unavailable capabilities, or "
                "required user information. Do not use merely because a relevant control is already focused.")
    if isinstance(action, FinishAction):
        return ("Finish: ALL of the user's requested task has ALREADY been completed, with evidence "
                "from the observed outcome or successful relevant action history. A target merely being "
                "present, enabled, or focused is not evidence that a requested click was performed.")
    if isinstance(action, ClickAction):
        control = controls[action.target_id]
        label = json.dumps(redactor.clean(control.name)[:160] or "unnamed control", ensure_ascii=False)
        focus = "focused" if control.focused is True else "not focused" if control.focused is False else "focus unknown"
        kind = redactor.clean(control.control_type)[:40]
        effect = _CLICK_EFFECTS.get(control.control_type, "Activate this observed control.")
        parent = ""
        if control.parent_name or control.parent_control_type:
            parent_label = json.dumps(redactor.clean(control.parent_name)[:120] or "unnamed", ensure_ascii=False)
            parent = f" Parent: {parent_label} [{redactor.clean(control.parent_control_type)[:40]}]."
        return f"Click {label} [{kind}, {focus}, enabled] ({action.target_id}).{parent} {effect}"
    if isinstance(action, VisualClickAction):
        control = (visual_controls or {})[action.target_id]
        label = json.dumps(redactor.clean(control.label)[:160] or "unnamed visual control", ensure_ascii=False)
        role = redactor.clean(control.role)[:40]
        parent = f" Parent/group: {json.dumps(redactor.clean(control.parent)[:120], ensure_ascii=False)}." if control.parent else ""
        confidence = (f"confidence {control.confidence:.2f}"
                      if control.confidence is not None else "confidence unavailable")
        return (f"Click {label} [{role}, visual, {confidence}] "
                f"({action.target_id}).{parent} Activate this screenshot-local semantic target once.")
    if isinstance(action, OpenAppAction):
        candidate = (applications or {}).get(action.app_id)
        if candidate is None:
            raise ValueError("Unknown application action.")
        name = json.dumps(redactor.clean(candidate.display_name)[:160], ensure_ascii=False)
        return f"Open installed application {name}. Launch it; verify the fresh foreground identity before completion."
    if isinstance(action, PressKeyAction):
        return f"Press {'+'.join(action.keys)}. {_KEY_EFFECTS[action.keys]}"
    if isinstance(action, TypeAction):
        literal = json.dumps(redactor.clean(action.text), ensure_ascii=False)
        return f"Type the user's supplied literal text {literal}: replace the ENTIRE focused editable value, without submitting."
    raise ValueError("Unsupported action description.")
