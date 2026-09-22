"""Bounded state, credential filtering, and literal text candidates."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import os
import re
from typing import Any
import unicodedata

from computer.actions import ClickAction, OpenAppAction, PressKeyAction, TypeAction, VisualClickAction
from computer.models import Observation, Rect, UIElement
from computer.results import ActionResult
from decision.target_resolution import TargetSpec
from safety.policy import ALLOWED_KEYS, CLICK_TYPES

MAX_CONTROLS = 80
MAX_REQUEST = 4000
MAX_TEXT = 500

# Preference is based on UI semantics, never application names or task phrases.
_INTERACTIVE_TYPES = CLICK_TYPES | {"ComboBox", "Slider", "Spinner", "ScrollBar", "DataItem"}
_STRUCTURAL_TYPES = {"Pane", "Group", "Window"}


@dataclass(frozen=True, slots=True)
class ControlSelection:
    controls: tuple[UIElement, ...]
    eligible: tuple[UIElement, ...]
    presentation_omitted: int
    privacy_omitted: int
    budget_omitted: int


def _normalized_label(name: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", name).casefold().split())


def _icon_only(name: str) -> bool:
    characters = [char for char in name if not char.isspace() and unicodedata.category(char) != "Cf"]
    return bool(characters) and all(unicodedata.category(char) == "Co" for char in characters)


def _contains(parent: Rect | None, child: Rect | None) -> bool:
    # Bounds are used locally only; neither geometry nor UIA objects go to Jev.
    return (isinstance(parent, Rect) and isinstance(child, Rect)
            and child.right > child.left and child.bottom > child.top
            and parent.left <= child.left < child.right <= parent.right
            and parent.top <= child.top < child.bottom <= parent.bottom)


def select_controls(observation: Observation, limit: int = MAX_CONTROLS) -> ControlSelection:
    """Rank before bounding; preserve actionable/focused/unknown named controls.

    Without parent IDs, remove duplicate Text only with matching labels AND
    containment evidence. Otherwise keep the Text, but rank it last. Never
    deduplicate two actionable controls, even when their labels are identical.
    """
    if limit < 1:
        raise ValueError("Control limit must be positive.")
    visible = [control for control in observation.elements if control.visible is True]
    eligible = [control for control in visible if control.is_password is not True
                and (control.control_type not in {"Edit", "Document"} or control.is_password is False)]
    interactive = [control for control in eligible if control.control_type in _INTERACTIVE_TYPES]
    labels: dict[str, list[UIElement]] = {}
    for control in interactive:
        if label := _normalized_label(control.name):
            labels.setdefault(label, []).append(control)
    retained: list[UIElement] = []
    for control in eligible:
        if control.focused is not True and control.control_type not in _INTERACTIVE_TYPES:
            if control.control_type in _STRUCTURAL_TYPES and not control.name.strip() and not control.automation_id:
                continue
            if control.control_type == "Text":
                if not control.name.strip() or _icon_only(control.name):
                    continue
                matches = labels.get(_normalized_label(control.name), ())
                if any(_contains(parent.rectangle, control.rectangle) for parent in matches):
                    continue
        retained.append(control)

    def rank(control: UIElement) -> int:
        if control.focused is True:
            return 0
        if control.control_type in _INTERACTIVE_TYPES:
            return 1 if control.enabled is True else 2
        if control.control_type == "Text" and _normalized_label(control.name) in labels:
            return 4
        return 3  # Preserve unique text/status information and unknown controls.

    ordered = sorted(retained, key=rank)  # Stable: preserve snapshot order on ties.
    return ControlSelection(tuple(ordered[:limit]), tuple(eligible), len(eligible) - len(retained),
                            len(visible) - len(eligible), max(0, len(ordered) - limit))


def control_context(
    control: UIElement, redactor: Redactor, requested_literals: Sequence[str] = (),
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "id": control.id, "name": redactor.clean(control.name)[:160],
        "type": redactor.clean(control.control_type)[:40], "enabled": control.enabled,
        "focused": control.focused,
    }
    if control.parent_name or control.parent_control_type:
        result["parent"] = {
            "name": redactor.clean(control.parent_name)[:120],
            "type": redactor.clean(control.parent_control_type)[:40],
        }
    if control.control_type in {"Edit", "Document"}:
        result.update({
            "editable": True,
            "observed_text_present": control.observed_text is not None,
            "observed_text_length": len(control.observed_text) if control.observed_text is not None else None,
            "observed_text_truncated": control.observed_text_truncated,
            "matches_requested_literal": (
                control.observed_text is not None and not control.observed_text_truncated
                and any(_matches_observed_text(control.observed_text, literal)
                        for literal in requested_literals)
            ),
        })
    return result


class Redactor:
    """Best-effort credential detection; not a general-purpose DLP system."""

    def __init__(self, secrets: Sequence[str] = ()) -> None:
        values = [value for name, value in os.environ.items()
                  if re.search(r"(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)", name, re.I) and len(value) >= 4]
        self._secrets = tuple(sorted(set(values + [s for s in secrets if s]), key=len, reverse=True))

    def clean(self, text: str) -> str:
        for secret in self._secrets:
            text = text.replace(secret, "[REDACTED]")
        text = re.sub(r"\bBearer\s+\S+", "Bearer [REDACTED]", text, flags=re.I)
        text = re.sub(r"\b(?:sk-[A-Za-z0-9_-]{8,}|(?:jv|ts)_(?:live|test)_[A-Za-z0-9_-]+)", "[REDACTED]", text)
        return re.sub(r"\b(?:api[_ -]?key|password|secret|token)\s*[:=]\s*(?:\"[^\"]*\"|'[^']*'|\S+)",
                      "[REDACTED]", text, flags=re.I)


def literal_texts(request: str) -> tuple[str, ...]:
    """Conservative English entry phrases; candidates are exact source slices.

    No screen/history text, synthesis, translation, or escape decoding. Quote
    content to disambiguate literal prose from a request to generate prose.
    """
    next_action = r"(?:click|press|open|type|write|enter|search|go|navigate|visit|save|send|submit)"
    pattern = rf'''\b(?:type|write|enter|search\s+for|go\s+to|navigate\s+to|visit)\s+(?:the\s+text\s+)?(?:"([^"]+)"|“([^”]+)”|'([^']+)'|([^\r\n]+?)(?=\s+(?:and(?:\s+then)?|then)\s+{next_action}\b|$))'''
    candidates: list[str] = []
    for match in re.finditer(pattern, request, re.I):
        quoted = next((group for group in match.groups()[:3] if group is not None), None)
        value = quoted if quoted is not None else (match.group(4) or "")
        if quoted is None:
            value = re.split(r"\s+(?:and(?:\s+then)?|then)\s+(?=click|press|open|type|write|enter|search|go|navigate|visit|save|send|submit)",
                             value, maxsplit=1, flags=re.I)[0]
            if (value.startswith(('"', "'", "“")) or re.match(r"(?:me|us|a|an|some)\b", value, re.I)
                    or re.search(r"\b(?:essay|poem|story|paragraph|compose|generate|summarize|translate)\b", value, re.I)):
                continue
        value = value.strip()
        if value and len(value) <= MAX_TEXT and "\x00" not in value and value in request and value not in candidates:
            candidates.append(value)
        if len(candidates) == 8:
            break
    return tuple(candidates)


def phase2_type_literal(request: str) -> str | None:
    """Choose one bounded source slice for the click+type experiment."""
    existing = literal_texts(request)
    if existing:
        return existing[0]
    match = re.search(
        r"\bplay\s+(.+?)(?:\s+by\s+[^\r\n.!?]+)?(?:[.!?]|$)", request, re.I,
    )
    if not match:
        return None
    value = match.group(1).strip(" \t\"'“”")
    if (not value or len(value) > MAX_TEXT or "\x00" in value
            or value not in request or Redactor().clean(value) != value):
        return None
    return value


def requested_target_spec(
    request: str, *, experimental_generic: bool = False,
) -> TargetSpec | None:
    """Extract bounded intent; opt in to generic-debug-only parsing refinements."""
    clauses = re.split(
        r"\s+(?:and\s+)?(?=(?:then\s+)?(?:play|open|select|choose|find|locate|navigate\s+to|go\s+to)\b)",
        request, maxsplit=1, flags=re.I,
    )
    target_request = clauses[-1].strip()
    role: str | None = None
    match = re.search(
        r"\b(?:open|select|choose)\s+(?:(?:the|a|an)\s+)?(?:chat|conversation)\s+with\s+"
        r"(.+?)(?:[!?]|\.(?=\s|$)|$)", target_request, re.I,
    )
    if match:
        raw_identity = match.group(1).strip(" \t\"'“”")
        qualifier_match = (
            re.fullmatch(
                r"(.+?)\s+(?:from|in|for)\s+([^\r\n.!?]+)", raw_identity, re.I,
            )
            if experimental_generic else None
        )
        primary = (qualifier_match.group(1) if qualifier_match else raw_identity).strip(
            " \t\"'“”",
        )
        role = "conversation"
        qualifier = qualifier_match.group(2).strip(" \t\"'“”") if qualifier_match else None
    else:
        match = re.search(
            r"\b(?:find|locate|open|select)\s+(?:(?:the|a|an)\s+)?(?:file|document)\s+"
            r"(.+?)(?:[!?]|\.(?=\s|$)|$)", target_request, re.I,
        )
        if match:
            primary = match.group(1).strip(" \t\"'“”")
            role = "file"
            qualifier = None
        else:
            match = re.search(
                r"\b(?:find|locate)\s+(.+?)(?:[!?]|\.(?=\s|$)|$)",
                target_request, re.I,
            )
            if match:
                primary = match.group(1).strip(" \t\"'“”")
                role = (
                    "file" if experimental_generic
                    and re.search(r"\.[A-Za-z0-9]{1,8}$", primary) else None
                )
                qualifier = None
            else:
                tab_match = re.search(
                    r"\b(?:open|select|choose)\s+(?:(?:the|a|an)\s+)?(.+?)\s+tab(?:[.!?]|$)",
                    target_request, re.I,
                )
                if tab_match:
                    primary = tab_match.group(1).strip(" \t\"'“”")
                    role = "navigation destination"
                    qualifier = None
                    match = tab_match
                else:
                    action_matches = tuple(re.finditer(
                        r"\b(?:play|open|select|choose|navigate\s+to|go\s+to)\s+"
                        r"(.+?)(?:[!?]|\.(?=\s|$)|$)", target_request, re.I,
                    ))
                    if not action_matches:
                        return None
                    action_match = action_matches[-1]
                    text = action_match.group(1)
                    text = re.split(r"\s+(?:and\s+then|then)\s+", text, maxsplit=1, flags=re.I)[0]
                    qualifier_match = re.fullmatch(
                        r"(.+?)\s+by\s+([^\r\n.!?]+)", text, re.I,
                    )
                    primary = (qualifier_match.group(1) if qualifier_match else text).strip(
                        " \t\"'“”",
                    )
                    qualifier = (
                        qualifier_match.group(2).strip(" \t\"'“”")
                        if qualifier_match else None
                    )
                    role = (
                        "file" if experimental_generic
                        and re.search(r"\.[A-Za-z0-9]{1,8}$", primary)
                        else None
                    )
                    match = action_match
                if not match:
                    return None
            if not primary:
                return None

    if (not primary or len(primary) > 200 or primary not in request
            or qualifier is not None and (
                len(qualifier) > 200 or qualifier not in request
            )):
        return None
    redactor = Redactor()
    if redactor.clean(primary) != primary or qualifier and redactor.clean(qualifier) != qualifier:
        return None
    action_intent = (
        "select" if re.search(r"\b(?:find|locate|select|choose)\b", target_request, re.I)
        else "activate"
    )
    return TargetSpec(
        primary_identity=primary,
        qualifiers=(qualifier,) if qualifier else (),
        desired_role=role,
        action_intent=action_intent,
    )


def _matches_observed_text(observed: str, literal: str) -> bool:
    # Some TextPattern providers append one terminal line break to a document
    # range even when ValuePattern contains the exact SetValue payload.
    return observed == literal or (
        not literal.endswith(("\r", "\n")) and observed in {literal + "\r", literal + "\r\n"}
    )


def completion_evidence(
    request: str, observation: Observation, history: Sequence[ActionResult],
) -> dict[str, Any]:
    """Derive compact postconditions from action results plus the fresh UI state."""
    evidence: dict[str, Any] = {}
    literals = literal_texts(request)
    typed = next((result for result in reversed(history)
                  if result.success and isinstance(result.action, TypeAction)), None)
    if typed is not None:
        editors = [control for control in observation.elements
                   if control.focused is True and control.control_type in {"Edit", "Document"}
                   and control.is_password is False]
        observed = [control for control in editors if control.observed_text is not None]
        evidence["type"] = {
            "previous_type_action_succeeded": True,
            "fresh_focused_editable_observed": bool(observed),
            "observed_value_matches_requested_literal": any(
                not control.observed_text_truncated
                and _matches_observed_text(control.observed_text, typed.action.text)
                and typed.action.text in literals
                for control in observed
            ),
        }
    opened = next((result for result in reversed(history)
                   if result.success and isinstance(result.action, OpenAppAction)), None)
    if opened is not None:
        app_id = opened.action.app_id
        evidence["open_app"] = {
            "previous_open_action_succeeded": True,
            "requested_app_id": app_id if re.fullmatch(r"app_[0-9a-f]{16}", app_id) else "unsupported",
            "foreground_matches_requested_app": observation.application_id == app_id,
        }
    return evidence


def compact_state(
    request: str, observation: Observation, controls: Sequence[UIElement],
    history: Sequence[ActionResult], redactor: Redactor,
    *, selection_incomplete: bool = False,
) -> dict[str, Any]:
    requested_literals = literal_texts(request)
    attempts: list[dict[str, Any]] = []
    for result in history[-5:]:
        action = result.action
        entry: dict[str, Any] = {"kind": action.kind, "success": result.success,
                                 "completed": result.completed, "requires_confirmation": result.requires_confirmation}
        if isinstance(action, ClickAction):
            entry["previous_snapshot_target"] = action.target_id if re.fullmatch(r"c[1-9]\d{0,8}", action.target_id) else "omitted"
        elif isinstance(action, VisualClickAction):
            entry["previous_visual_target"] = action.target_id if re.fullmatch(r"v[1-9]\d{0,8}", action.target_id) else "omitted"
        elif isinstance(action, OpenAppAction):
            entry["app_id"] = action.app_id if re.fullmatch(r"app_[0-9a-f]{16}", action.app_id) else "unsupported"
        elif isinstance(action, PressKeyAction):
            keys = tuple(key.lower() for key in action.keys)
            entry["keys"] = keys if keys in ALLOWED_KEYS else "unsupported"
        elif isinstance(action, TypeAction):
            # Only content already in the current request is relevant to this task.
            entry["text"] = redactor.clean(action.text)[:MAX_TEXT] if action.text in request else "[previous text omitted]"
        entry["error"] = result.error if result.error in {
            None, "policy_blocked", "unsafe_target", "windows_operation_failed", "unsupported_action",
        } else "other_error"
        attempts.append(entry)
    return {
        "request": request,
        "window": {"app": redactor.clean(observation.app_name)[:100],
                   "title": redactor.clean(observation.window_title)[:160]},
        "focused_controls": [control_context(control, redactor, requested_literals)
                             for control in controls if control.focused is True],
        "controls": [control_context(control, redactor, requested_literals) for control in controls],
        "visual_controls": [
            {"id": control.id, "label": redactor.clean(control.label)[:160],
             "role": redactor.clean(control.role)[:40], "confidence": control.confidence,
             "clickable": control.clickable,
             **({"parent": redactor.clean(control.parent)[:120]} if control.parent else {})}
            for control in observation.visual_elements
        ],
        "capabilities": {
            "uia_controls": len(observation.elements),
            "visual_controls": len(observation.visual_elements),
            "visual_fallback_reason": observation.visual_fallback_reason,
            "screenshot_captured": observation.screenshot is not None,
            "visual_provider": redactor.clean(observation.visual_provider or "")[:80] or None,
            "visual_model": redactor.clean(observation.visual_model or "")[:100] or None,
        },
        "history": attempts,
        "completion_evidence": completion_evidence(request, observation, history),
        "observation_incomplete": observation.truncated or bool(observation.inspection_errors)
                                  or selection_incomplete,
    }
