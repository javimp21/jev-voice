"""Shared provider-neutral checks for visually grounded text-entry fields."""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata

from computer.models import Observation, Rect, VisualElement


_TEXT_ENTRY_ROLES = frozenset({
    "edit", "text field", "text box", "search field", "search box",
    "query field", "editable field",
})
_CREDENTIAL_WORDS = re.compile(
    r"\b(?:password|credential|passcode|secret|token|api key|access key|"
    r"contraseña|credencial|código de acceso)\b",
    re.IGNORECASE,
)
_SAFE_CANDIDATE_ID = re.compile(r"[A-Za-z0-9_-]{1,80}\Z")


@dataclass(frozen=True, slots=True)
class VisualFieldCorrespondence:
    """Result values intentionally omit rectangles and screen coordinates."""

    window_geometry_stable: bool
    semantic_correspondence_result: str
    spatial_correspondence_result: str
    compatible_candidate_count: int
    spatial_candidate_ids: tuple[str, ...]
    selected_candidate_id: str | None

    @property
    def verified(self) -> bool:
        return (
            self.window_geometry_stable
            and self.semantic_correspondence_result == "compatible"
            and self.spatial_correspondence_result == "unique_match"
            and self.selected_candidate_id is not None
        )


def role_is_text_entry(role: str) -> bool:
    normalized = re.sub(r"[_-]+", " ", unicodedata.normalize("NFKC", role).casefold())
    return " ".join(normalized.split()) in _TEXT_ENTRY_ROLES


def same_trusted_foreground_identity(first: Observation, second: Observation) -> bool:
    """Require the same known trusted app, PID, and foreground HWND."""
    first_hwnd = (
        first.screenshot.window_handle if first.screenshot is not None
        else first.foreground_hwnd
    )
    second_hwnd = (
        second.screenshot.window_handle if second.screenshot is not None
        else second.foreground_hwnd
    )
    return bool(
        first.observation_id and second.observation_id
        and first.application_id and first.application_id == second.application_id
        and isinstance(first.process_id, int) and first.process_id > 0
        and first.process_id == second.process_id
        and isinstance(first_hwnd, int) and first_hwnd > 0 and first_hwnd == second_hwnd
    )


def window_geometry_is_stable(clicked: Observation, verification: Observation) -> bool:
    first = clicked.screenshot
    second = verification.screenshot
    if first is None or second is None:
        return False
    return bool(
        clicked.observation_id and verification.observation_id
        and clicked.observation_id != verification.observation_id
        and first.snapshot_id == clicked.observation_id
        and second.snapshot_id == verification.observation_id
        and clicked.application_id and clicked.application_id == verification.application_id
        and isinstance(clicked.process_id, int) and clicked.process_id > 0
        and clicked.process_id == verification.process_id
        and first.window_handle == second.window_handle
        and first.window_bounds == second.window_bounds
        and first.capture_bounds == second.capture_bounds
        and first.pixel_width == second.pixel_width
        and first.pixel_height == second.pixel_height
        and first.dpi_x == second.dpi_x and first.dpi_y == second.dpi_y
        and first.scale_x == second.scale_x and first.scale_y == second.scale_y
    )


def _spatially_corresponds(
    clicked_rectangle: Rect, candidate_rectangle: Rect,
    clicked: Observation, verification: Observation,
) -> bool:
    first_meta, second_meta = clicked.screenshot, verification.screenshot
    if (first_meta is None or second_meta is None or first_meta.pixel_width <= 0
            or first_meta.pixel_height <= 0 or second_meta.pixel_width <= 0
            or second_meta.pixel_height <= 0):
        return False

    def normalized(rectangle: Rect, width: int, height: int) -> tuple[float, float, float, float]:
        return (
            rectangle.left / width, rectangle.top / height,
            rectangle.right / width, rectangle.bottom / height,
        )

    first = normalized(clicked_rectangle, first_meta.pixel_width, first_meta.pixel_height)
    second = normalized(candidate_rectangle, second_meta.pixel_width, second_meta.pixel_height)
    overlap_width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    overlap_height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = overlap_width * overlap_height
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    smaller_area = min(first_area, second_area)
    return smaller_area > 0 and intersection / smaller_area >= 0.5


def verify_visual_field_correspondence(
    clicked_target: VisualElement,
    clicked_observation: Observation,
    verification_observation: Observation,
) -> VisualFieldCorrespondence:
    """Reuse Phase-2 overlap semantics and require one compatible matching field."""
    geometry_stable = window_geometry_is_stable(
        clicked_observation, verification_observation,
    )
    clicked_semantic = clicked_target.clickable is True and role_is_text_entry(clicked_target.role)
    compatible = tuple(
        item for item in verification_observation.visual_elements[:5]
        if item.clickable is True and role_is_text_entry(item.role)
    )
    matches = tuple(
        item for item in compatible
        if isinstance(item.rectangle, Rect)
        and _spatially_corresponds(
            clicked_target.rectangle, item.rectangle,
            clicked_observation, verification_observation,
        )
    ) if geometry_stable and clicked_semantic else ()

    if not clicked_semantic:
        semantic_result = "clicked_role_incompatible"
    elif compatible:
        semantic_result = "compatible"
    else:
        semantic_result = "no_compatible_candidate"

    if not geometry_stable:
        spatial_result = "geometry_unstable"
    elif not clicked_semantic:
        spatial_result = "not_evaluated"
    elif len(matches) == 1:
        spatial_result = "unique_match"
    elif len(matches) > 1:
        spatial_result = "multiple_matches"
    elif compatible:
        spatial_result = "no_spatial_match"
    else:
        spatial_result = "not_evaluated"

    safe_ids = tuple(
        item.id for item in matches[:5] if _SAFE_CANDIDATE_ID.fullmatch(item.id)
    )
    selected_id = matches[0].id if len(matches) == 1 else None
    return VisualFieldCorrespondence(
        geometry_stable, semantic_result, spatial_result, len(compatible),
        safe_ids, selected_id,
    )


def has_credential_sensitive_evidence(
    request: str,
    clicked_target: VisualElement | None,
    *observations: Observation,
) -> bool:
    """Fail closed on local password markers, masking, or credential semantics."""
    texts = [request]
    if clicked_target is not None:
        texts.extend((clicked_target.label, clicked_target.role, clicked_target.parent))
    for observation in observations:
        texts.extend((observation.app_name, observation.window_title))
        for control in observation.elements:
            texts.extend((control.name, control.automation_id, control.parent_name))
            if control.is_password is True:
                return True
            if (control.focused is True and control.control_type in {"Edit", "Document"}
                    and control.is_password is not False):
                return True
        for item in observation.visual_elements:
            texts.extend((item.label, item.role, item.parent))
        for item in observation.visual_provider_candidates:
            texts.extend((item.label, item.role, item.parent))
        if (observation.visual_readiness is not None
                and str(observation.visual_readiness.reason) == "credential_sensitive"):
            return True
        if (observation.screenshot is not None
                and observation.screenshot.masked_regions > 0):
            return True
        if (observation.capture_diagnostics is not None
                and observation.capture_diagnostics.mask_region_count > 0):
            return True
    return any(_CREDENTIAL_WORDS.search(text) for text in texts)


def safe_candidate_ids(observation: Observation, limit: int = 5) -> tuple[str, ...]:
    return tuple(
        item.id for item in observation.visual_elements[:limit]
        if _SAFE_CANDIDATE_ID.fullmatch(item.id)
    )
