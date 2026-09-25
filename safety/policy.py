"""Deterministic action-shape and consequential-effect safety policies."""

import re
import unicodedata

from computer.actions import Action, ClickAction, FinishAction, OpenAppAction, PressKeyAction, TypeAction, VisualClickAction
from computer.applications import ApplicationCatalog
from computer.models import Observation, Rect
from computer.visual import visual_rect_to_screen
from safety.interfaces import SafetyDecision

ALLOWED_KEYS = frozenset({
    ("enter",), ("escape",), ("tab",), ("shift", "tab"), ("ctrl", "a"), ("ctrl", "c"),
})
CLICK_TYPES = frozenset({
    "Button", "Hyperlink", "MenuItem", "Edit", "Document", "TabItem",
    "ListItem", "TreeItem", "RadioButton", "CheckBox",
})


class BasicActionPolicy:
    """Fail closed on unsupported requests; UI identity is checked by the adapter.

    Click/Enter effects depend on the application. This policy does not infer
    sending, deleting, purchasing, or form submission from labels. It is for
    supervised debugging only. Additional policies may deny or request consent.
    """

    def __init__(self, app_catalog: ApplicationCatalog | None = None, *, visual_min_confidence: float = 0.75) -> None:
        if not 0 <= visual_min_confidence <= 1:
            raise ValueError("visual_min_confidence must be between 0 and 1")
        self.app_catalog = app_catalog
        self.visual_min_confidence = visual_min_confidence

    def validate(self, action: Action, observation: Observation | None) -> SafetyDecision:
        reason: str | None = None
        if isinstance(action, FinishAction):
            return SafetyDecision("allow", "Completion does not interact with Windows.")
        if isinstance(action, OpenAppAction):
            candidate = self.app_catalog.resolve(action.app_id) if self.app_catalog else None
            if candidate is None:
                return SafetyDecision("deny", "Unknown application ID; paths, arguments, and commands are not accepted.")
            if candidate.launch_policy == "deny":
                return SafetyDecision("deny", "This application is restricted from autonomous launch.")
            if candidate.launch_policy == "confirm":
                return SafetyDecision("confirm", "This application requires human approval before launch.")
            return SafetyDecision("allow", "Trusted locally discovered application.")
        elif isinstance(action, PressKeyAction):
            if tuple(key.lower() for key in action.keys) not in ALLOWED_KEYS:
                reason = "Unsupported key combination."
        elif isinstance(action, TypeAction):
            if not isinstance(action.text, str) or "\x00" in action.text:
                reason = "Text must be a string without NUL characters."
        elif isinstance(action, ClickAction):
            control = next((item for item in observation.elements if item.id == action.target_id), None) if observation else None
            if control is None:
                reason = "Unresolved control ID."
            elif control.control_type not in CLICK_TYPES:
                reason = "Unsupported control type for semantic clicking."
            elif control.visible is not True or control.enabled is not True:
                reason = "Control was not known to be visible and enabled."
        elif isinstance(action, VisualClickAction):
            control = next((item for item in observation.visual_elements if item.id == action.target_id), None) if observation else None
            if observation is None or action.snapshot_id != observation.observation_id:
                reason = "Visual target belongs to a different observation."
            elif observation.screenshot is None or observation.screenshot.snapshot_id != action.snapshot_id:
                reason = "Visual capture metadata is unavailable or belongs to another snapshot."
            elif control is None:
                reason = "Unresolved visual control ID."
            elif (not isinstance(control.rectangle, Rect) or control.rectangle.left < 0
                  or control.rectangle.top < 0 or control.rectangle.right <= control.rectangle.left
                  or control.rectangle.bottom <= control.rectangle.top
                  or control.rectangle.right > observation.screenshot.pixel_width
                  or control.rectangle.bottom > observation.screenshot.pixel_height):
                reason = "Visual target rectangle is invalid or outside the capture."
            elif not control.clickable:
                reason = "Visual target is not classified as interactable."
            elif control.confidence is None:
                reason = "Visual target has no calibrated confidence for execution."
            elif control.confidence < self.visual_min_confidence:
                reason = "Visual target confidence is below the local threshold."
        else:
            reason = "Unsupported action."
        if reason:
            return SafetyDecision("deny", reason)
        if observation is None or observation.error:
            return SafetyDecision("deny", "A valid observation is required.")
        return SafetyDecision("allow", "Supported manual action; live target checks still required.")


_CONSEQUENTIAL_TERMS = (
    "send", "submit", "purchase", "buy", "pay", "delete", "remove", "upload", "post",
    "download", "install", "uninstall", "grant permission", "allow access", "account modification",
    "publish", "confirm order", "place order", "change password", "reset password", "sign in", "log in",
    "enviar", "confirmar", "comprar", "pagar", "eliminar", "borrar", "subir", "publicar",
    "cambiar contraseña", "restablecer contraseña", "iniciar sesión", "configuración de cuenta",
    "account settings", "security settings", "configuración de seguridad",
)

_CREDENTIAL_TERMS = (
    "password", "credential", "passcode", "api key", "access token",
    "contraseña", "credencial", "código de acceso",
)


def _normalized_words(label: str) -> str:
    normalized = unicodedata.normalize("NFKC", label).casefold()
    return " ".join(re.findall(r"\w+", normalized, flags=re.UNICODE))


def _consequential(label: str) -> bool:
    words = f" {_normalized_words(label)} "
    return any(f" {_normalized_words(term)} " in words for term in _CONSEQUENTIAL_TERMS)


class AutonomousActionPolicy(BasicActionPolicy):
    """Add a generic confirmation boundary for consequential UI effects."""

    def validate(self, action: Action, observation: Observation | None) -> SafetyDecision:
        baseline = super().validate(action, observation)
        if baseline.disposition != "allow":
            return baseline
        if isinstance(action, PressKeyAction) and tuple(key.lower() for key in action.keys) == ("enter",):
            return SafetyDecision("confirm", "Enter may submit or activate a consequential default action.")
        if isinstance(action, (ClickAction, VisualClickAction)) and observation is not None:
            if isinstance(action, ClickAction):
                control = next((item for item in observation.elements if item.id == action.target_id), None)
                label = control.name if control else ""
            else:
                visual = next((item for item in observation.visual_elements if item.id == action.target_id), None)
                label = visual.label if visual else ""
            if _consequential(label):
                return SafetyDecision("confirm", "The selected control may cause an external or destructive action.")
        return baseline


_PHASE1_VISUAL_ROLES = frozenset({
    "search field", "search_field", "text field", "text_field", "edit",
    "navigation item", "navigation_item", "tab", "menu item", "menu_item",
    "toggle", "checkbox", "icon button", "icon_button", "button",
})


class Phase1VisualClickPolicy:
    """Strict local gate for the single-click experimental milestone."""

    def validate(
        self, action: VisualClickAction, observation: Observation | None,
        request: str, jev_confidence: float | None,
    ) -> SafetyDecision:
        if jev_confidence is None or jev_confidence < .80:
            return SafetyDecision("deny", "Jev confidence is below the action threshold.")
        if observation is None or observation.error or not observation.observation_id:
            return SafetyDecision("deny", "A valid bound observation is required.")
        if action.snapshot_id != observation.observation_id:
            return SafetyDecision("deny", "Visual target belongs to a different observation.")
        metadata = observation.screenshot
        target = next((item for item in observation.visual_elements
                       if item.id == action.target_id), None)
        if metadata is None or metadata.snapshot_id != action.snapshot_id or target is None:
            return SafetyDecision("deny", "Visual target or snapshot is unavailable.")
        if target.source != "visual" or not target.clickable:
            return SafetyDecision("deny", "Target is not a clickable visual candidate.")
        rect = target.rectangle
        if (not isinstance(rect, Rect) or rect.left < 0 or rect.top < 0
                or rect.right <= rect.left or rect.bottom <= rect.top
                or rect.right > metadata.pixel_width or rect.bottom > metadata.pixel_height):
            return SafetyDecision("deny", "Visual target rectangle is outside the capture.")
        role = _normalized_words(target.role).replace(" ", "_")
        allowed_roles = {item.replace(" ", "_") for item in _PHASE1_VISUAL_ROLES}
        if role not in allowed_roles:
            return SafetyDecision("deny", "Visual target role is not eligible for phase 1.")
        if not target.label.strip():
            return SafetyDecision("deny", "Unlabeled visual targets are uncertain in phase 1.")
        if _consequential(target.label) or _consequential(request):
            return SafetyDecision("deny", "Target or request may cause a consequential action.")
        if any(term in _normalized_words(request) for term in _CREDENTIAL_TERMS):
            return SafetyDecision("deny", "Request is credential-sensitive.")
        if (observation.visual_readiness is not None
                and str(observation.visual_readiness.reason) == "credential_sensitive"):
            return SafetyDecision("deny", "Observation privacy context is unsafe.")
        if any(control.focused is True and control.control_type in {"Edit", "Document"}
               and control.is_password is not False for control in observation.elements):
            return SafetyDecision("deny", "Credential-sensitive editor context is active.")
        screen_rect = visual_rect_to_screen(rect, metadata)
        for control in observation.elements:
            if control.is_password is True and isinstance(control.rectangle, Rect):
                overlap_width = max(0, min(screen_rect.right, control.rectangle.right)
                                    - max(screen_rect.left, control.rectangle.left))
                overlap_height = max(0, min(screen_rect.bottom, control.rectangle.bottom)
                                     - max(screen_rect.top, control.rectangle.top))
                if overlap_width and overlap_height:
                    return SafetyDecision("deny", "Target overlaps a credential-sensitive region.")
        return SafetyDecision("allow", "Non-consequential phase-1 visual focus/navigation click.")


_RESULT_ROLES = frozenset({
    "song", "song_result", "album", "album_result", "artist", "artist_result",
    "video", "video_result", "document", "document_result", "content_result",
    "list_item", "card", "link", "navigation_item", "menu_item", "button",
    "other_interactive",
})


def result_role_supported(role: str) -> bool:
    return _normalized_words(role).replace(" ", "_") in _RESULT_ROLES


class Phase3ResultSelectionPolicy:
    """Narrow gate for one semantically prevalidated ordinary result click."""

    def validate(self, action: VisualClickAction, observation: Observation | None) -> SafetyDecision:
        if observation is None or action.snapshot_id != observation.observation_id:
            return SafetyDecision("deny", "Result target belongs to a different observation.")
        target = next((x for x in observation.visual_elements if x.id == action.target_id), None)
        if target is None or target.source != "visual" or not target.clickable:
            return SafetyDecision("deny", "A clickable visual result is required.")
        metadata = observation.screenshot
        rect = target.rectangle
        if (metadata is None or metadata.snapshot_id != action.snapshot_id
                or not isinstance(rect, Rect) or rect.left < 0 or rect.top < 0
                or rect.right <= rect.left or rect.bottom <= rect.top
                or rect.right > metadata.pixel_width or rect.bottom > metadata.pixel_height):
            return SafetyDecision("deny", "Result rectangle is outside its bound capture.")
        if not result_role_supported(target.role):
            return SafetyDecision("deny", "Target role is not an ordinary result type.")
        if not target.label.strip() or _consequential(" ".join((target.label, target.parent))):
            return SafetyDecision("deny", "Result is unlabeled, uncertain, or consequential.")
        if any(c.focused is True and c.control_type in {"Edit", "Document"}
               and c.is_password is not False for c in observation.elements):
            return SafetyDecision("deny", "Credential-sensitive editor context is active.")
        screen_rect = visual_rect_to_screen(rect, metadata)
        for control in observation.elements:
            if control.is_password is True and isinstance(control.rectangle, Rect):
                if (max(0, min(screen_rect.right, control.rectangle.right)
                        - max(screen_rect.left, control.rectangle.left))
                        and max(0, min(screen_rect.bottom, control.rectangle.bottom)
                                - max(screen_rect.top, control.rectangle.top))):
                    return SafetyDecision("deny", "Result overlaps a credential-sensitive region.")
        return SafetyDecision("allow", "Validated non-consequential result selection.")


_GENERIC_TARGET_ROLES = frozenset({
    "conversation", "contact", "file", "container", "navigation_destination",
    "actionable_item", "search_field", "text_field",
})
_GENERIC_ROLE_ALIASES = {
    "chat": "conversation", "conversation": "conversation", "conversation thread": "conversation",
    "contact": "contact", "person": "contact",
    "file": "file", "document": "file",
    "folder": "container", "directory": "container", "container": "container",
    "navigation destination": "navigation_destination", "navigation item": "navigation_destination",
    "menu item": "navigation_destination", "tab": "navigation_destination",
    "tab item": "navigation_destination",
    "button": "actionable_item", "list item": "actionable_item", "card": "actionable_item",
    "link": "actionable_item", "actionable item": "actionable_item",
    "search field": "search_field", "search box": "search_field",
    "text field": "text_field", "text box": "text_field", "edit": "text_field",
}
_SAFE_UIA_ACTIVATION_TYPES = frozenset({
    "Button", "Hyperlink", "MenuItem", "Edit", "Document", "TabItem", "ListItem",
    "TreeItem", "RadioButton", "CheckBox",
})
_QUERY_FIELD_WORDS = frozenset({
    "search", "find", "query", "buscar", "busqueda", "búsqueda",
})


def _canonical_generic_role(value: str | None) -> str | None:
    if not value:
        return None
    normalized = " ".join(re.findall(
        r"\w+", unicodedata.normalize("NFKC", value.replace("_", " ").replace("-", " ")).casefold(),
    ))
    return _GENERIC_ROLE_ALIASES.get(normalized)


def generic_uia_semantic_role(control) -> str | None:
    """Infer domain meaning from UIA context, never from the target's own UI role."""
    parent_words = set(_normalized_words(control.parent_name).split())
    if parent_words & {"conversation", "conversations", "chat", "chats"}:
        return "conversation"
    if parent_words & {"contact", "contacts", "person", "people"}:
        return "contact"
    if parent_words & {"file", "files", "document", "documents"}:
        return "file"
    if parent_words & {"folder", "folders", "directory", "directories"}:
        return "container"
    if parent_words & {"navigation", "menu", "tabs"}:
        return "navigation_destination"
    parent_role = _canonical_generic_role(control.parent_control_type)
    if parent_role in {"conversation", "contact", "file", "container"}:
        return parent_role
    return None


class GenericTargetActivationPolicy:
    """Bounded safety gate for one ordinary target activation in the generic debug mode."""

    def __init__(self, app_catalog: ApplicationCatalog | None = None) -> None:
        self.app_catalog = app_catalog
        self._basic = BasicActionPolicy(app_catalog)

    @staticmethod
    def is_query_field(control) -> bool:
        if (control.control_type not in {"Edit", "Document"}
                or control.visible is not True or control.enabled is not True
                or control.is_password is not False):
            return False
        words = set(_normalized_words(" ".join((
            control.name, control.parent_name, control.automation_id,
        ))).split())
        return bool(words & _QUERY_FIELD_WORDS)

    def validate_candidate(
        self, action: Action, observation: Observation | None,
        *, semantic_role: str | None = None,
    ) -> SafetyDecision:
        if observation is None or observation.error or not observation.observation_id:
            return SafetyDecision("deny", "A fresh observation is required.")
        if not isinstance(observation.process_id, int) or observation.process_id <= 0:
            return SafetyDecision("deny", "Foreground process identity is unavailable.")
        if any(
            item.focused is True and item.control_type in {"Edit", "Document"}
            and item.is_password is not False for item in observation.elements
        ):
            return SafetyDecision("deny", "Credential-sensitive editor context is active.")

        if isinstance(action, ClickAction):
            target = next((item for item in observation.elements if item.id == action.target_id), None)
            if target is None or target.control_type not in _SAFE_UIA_ACTIVATION_TYPES:
                return SafetyDecision("deny", "UIA target is not a supported ordinary control.")
            if target.visible is not True or target.enabled is not True:
                return SafetyDecision("deny", "UIA target is not known to be visible and enabled.")
            if target.is_password is True:
                return SafetyDecision("deny", "Password controls cannot be activated by this mode.")
            rect = target.rectangle
            if rect is not None and (
                rect.right <= rect.left or rect.bottom <= rect.top
            ):
                return SafetyDecision("deny", "UIA target geometry is invalid.")
            role = semantic_role or generic_uia_semantic_role(target)
            if role is not None and _canonical_generic_role(role) not in _GENERIC_TARGET_ROLES:
                return SafetyDecision("deny", "UIA target role is outside the bounded allowlist.")
            label = " ".join((target.name, target.observed_text or "", target.parent_name))
        elif isinstance(action, VisualClickAction):
            if action.snapshot_id != observation.observation_id:
                return SafetyDecision("deny", "Visual target belongs to a different snapshot.")
            target = next((item for item in observation.visual_elements if item.id == action.target_id), None)
            metadata = observation.screenshot
            if (target is None or target.source != "visual" or not target.clickable
                    or metadata is None or metadata.snapshot_id != action.snapshot_id):
                return SafetyDecision("deny", "A clickable target from the current visual snapshot is required.")
            rect = target.rectangle
            if (rect.left < 0 or rect.top < 0 or rect.right <= rect.left
                    or rect.bottom <= rect.top or rect.right > metadata.pixel_width
                    or rect.bottom > metadata.pixel_height):
                return SafetyDecision("deny", "Visual target geometry is outside the current capture.")
            role = _canonical_generic_role(target.role)
            if role not in _GENERIC_TARGET_ROLES:
                return SafetyDecision("deny", "Visual target role is outside the bounded allowlist.")
            if not target.label.strip():
                return SafetyDecision("deny", "Unlabeled targets cannot be activated.")
            screen_rect = visual_rect_to_screen(rect, metadata)
            for control in observation.elements:
                if control.is_password is True and isinstance(control.rectangle, Rect):
                    if (max(0, min(screen_rect.right, control.rectangle.right)
                            - max(screen_rect.left, control.rectangle.left))
                            and max(0, min(screen_rect.bottom, control.rectangle.bottom)
                                    - max(screen_rect.top, control.rectangle.top))):
                        return SafetyDecision("deny", "Visual target overlaps a credential-sensitive region.")
            label = " ".join((target.label, target.parent))
        else:
            return SafetyDecision("deny", "Only a bounded target activation is supported.")

        if _consequential(label):
            return SafetyDecision("deny", "Consequential controls are blocked in generic target mode.")
        if any(term in _normalized_words(label) for term in _CREDENTIAL_TERMS):
            return SafetyDecision("deny", "Credential-related targets are blocked.")
        return SafetyDecision("allow", "Target is fresh, actionable, and non-consequential.")

    def validate(self, action: Action, observation: Observation | None) -> SafetyDecision:
        if isinstance(action, (ClickAction, VisualClickAction)):
            return self.validate_candidate(action, observation)
        return self._basic.validate(action, observation)
