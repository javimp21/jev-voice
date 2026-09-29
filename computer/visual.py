"""Provider-neutral visual observation, validation, fallback, and deduplication."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from difflib import SequenceMatcher
import math
import re
from typing import Any, Protocol
import unicodedata

from computer.models import (
    CaptureDiagnostics, Observation, ProviderErrorDiagnostic, Rect, ScreenshotMetadata,
    ResultReadinessRichnessRatios, UIElement, VisualElement, VisualProviderAttempt,
    VisualRegion, VisualSelectionState,
    VisualReadinessFrame, VisualRequestFingerprint,
)


@dataclass(slots=True)
class ScreenshotCapture:
    """Ephemeral pixels plus safe geometry; image is deliberately absent from repr."""

    metadata: ScreenshotMetadata
    image: Any
    encoded_png: bytes | None = field(default=None, repr=False)
    diagnostics: CaptureDiagnostics | None = None

    def discard(self) -> None:
        image, self.image = self.image, None
        self.encoded_png = None
        close = getattr(image, "close", None)
        if callable(close):
            close()


def crop_screenshot_to_field(
    screenshot: ScreenshotCapture, rectangle: Rect, *,
    sensitive_regions: tuple[Rect, ...] = (),
) -> ScreenshotCapture:
    """Create a small local crop bound to one already-validated field rectangle."""
    metadata = screenshot.metadata
    width, height = metadata.pixel_width, metadata.pixel_height
    if (screenshot.image is None or getattr(screenshot.image, "size", None) != (width, height)
            or not all(math.isfinite(scale) and scale > 0
                       for scale in (metadata.scale_x, metadata.scale_y))
            or not all(type(value) is int for value in (
                rectangle.left, rectangle.top, rectangle.right, rectangle.bottom,
            ))):
        raise ValueError("Field crop source geometry is unavailable.")
    field_width = rectangle.right - rectangle.left
    field_height = rectangle.bottom - rectangle.top
    if (rectangle.left < 0 or rectangle.top < 0 or rectangle.right > width
            or rectangle.bottom > height or field_width < 24 or field_height < 8
            or field_width > min(1200, round(width * .95))
            or field_height > min(160, round(height * .2))):
        raise ValueError("Field geometry cannot be safely bounded.")
    pad_x = min(8, max(2, field_width // 40))
    pad_y = min(5, max(2, field_height // 8))
    left, top = max(0, rectangle.left - pad_x), max(0, rectangle.top - pad_y)
    right, bottom = min(width, rectangle.right + pad_x), min(height, rectangle.bottom + pad_y)
    if right - left > 1216 or bottom - top > 176:
        raise ValueError("Field crop exceeds the bounded size.")
    screen_crop = Rect(
        metadata.capture_bounds.left + round(left / metadata.scale_x),
        metadata.capture_bounds.top + round(top / metadata.scale_y),
        metadata.capture_bounds.left + round(right / metadata.scale_x),
        metadata.capture_bounds.top + round(bottom / metadata.scale_y),
    )
    if any(
        min(screen_crop.right, region.right) > max(screen_crop.left, region.left)
        and min(screen_crop.bottom, region.bottom) > max(screen_crop.top, region.top)
        for region in sensitive_regions
    ):
        raise PermissionError("Sensitive UI geometry overlaps the field crop.")
    image = screenshot.image.crop((left, top, right, bottom))
    crop_width, crop_height = getattr(image, "size", (0, 0))
    screen_width = screen_crop.right - screen_crop.left
    screen_height = screen_crop.bottom - screen_crop.top
    if (crop_width < 1 or crop_height < 1 or screen_width < 1 or screen_height < 1
            or crop_width > 1216 or crop_height > 176):
        close = getattr(image, "close", None)
        if callable(close):
            close()
        raise ValueError("Field crop dimensions are invalid.")
    cropped_metadata = ScreenshotMetadata(
        metadata.snapshot_id, metadata.window_handle, metadata.window_bounds,
        screen_crop, crop_width, crop_height, metadata.dpi_x, metadata.dpi_y,
        crop_width / screen_width, crop_height / screen_height,
    )
    return ScreenshotCapture(cropped_metadata, image)


@dataclass(frozen=True, slots=True)
class VisualCandidate:
    """Untrusted provider result before local IDs and validation."""

    label: str
    role: str
    rectangle: Rect
    confidence: float | None = None
    clickable: bool = True
    parent: str = ""
    provider_role: str | None = None
    activity: str | None = None
    selection_state: VisualSelectionState = VisualSelectionState.UNKNOWN
    region: VisualRegion = VisualRegion.UNKNOWN
    field_label: str | None = None
    field_value: str | None = None
    is_query_field: bool | None = None
    credential_risk: bool | None = None


MAX_GROUNDING_OBJECTIVE_CHARS = 240
RESULT_SIMPLIFICATION_RATIO_THRESHOLD = 0.75
RESULT_SIMPLIFICATION_MIN_SIGNALS = 2


@dataclass(frozen=True, slots=True)
class VisualGroundingRequest:
    """Bounded, untrusted semantic target for observation only."""

    objective: str
    max_elements: int = 5
    verification_only: bool = False
    query_field_continuity: bool = False
    field_value_only: bool = False

    def __post_init__(self) -> None:
        if not self.objective.strip() or len(self.objective) > MAX_GROUNDING_OBJECTIVE_CHARS:
            raise ValueError(
                f"objective must contain 1 to {MAX_GROUNDING_OBJECTIVE_CHARS} characters"
            )
        if not 1 <= self.max_elements <= 100:
            raise ValueError("max_elements must be between 1 and 100")
        if type(self.verification_only) is not bool:
            raise ValueError("verification_only must be a boolean")
        if type(self.query_field_continuity) is not bool:
            raise ValueError("query_field_continuity must be a boolean")
        if type(self.field_value_only) is not bool:
            raise ValueError("field_value_only must be a boolean")
        if self.query_field_continuity and not self.verification_only:
            raise ValueError("query_field_continuity requires verification_only")
        if self.field_value_only and not (self.verification_only and self.query_field_continuity):
            raise ValueError("field_value_only requires visual query-field verification")


@dataclass(frozen=True, slots=True)
class VisualFieldValueRead:
    """Safe result of one locally cropped visual field-value extraction."""

    field_value: str | None = field(default=None, repr=False)
    crop_valid: bool = False
    context_stable: bool = False
    credential_safe: bool | None = None
    provider_attempts: tuple[VisualProviderAttempt, ...] = ()
    error: ProviderErrorDiagnostic | None = None
    reason: str | None = None


def bounded_grounding_request(objective: str, max_elements: int = 5) -> VisualGroundingRequest:
    """Normalize and cap CLI/task text before it reaches a remote adapter."""

    bounded = " ".join(objective.split())[:MAX_GROUNDING_OBJECTIVE_CHARS].strip()
    return VisualGroundingRequest(bounded, max_elements)


class VisualObserver(Protocol):
    name: str
    pricing_class: str

    def observe(
        self, screenshot: ScreenshotCapture, window: Observation, original_request: str,
        grounding: VisualGroundingRequest | None = None,
    ) -> VisualObservation: ...


def visual_request_fingerprint(
    provider: VisualObserver, screenshot: ScreenshotCapture,
    grounding: VisualGroundingRequest,
) -> VisualRequestFingerprint | None:
    """Obtain provider-specific safe shape metadata without exposing request contents."""
    method = getattr(provider, "request_fingerprint", None)
    result = method(screenshot, grounding) if callable(method) else None
    return result if isinstance(result, VisualRequestFingerprint) else None


class VisualProviderFailure(RuntimeError):
    """Sanitized provider failure that is safe to expose in diagnostics."""

    def __init__(
        self, code: str, *, http_status: int | None = None,
        provider_code: str | None = None, message: str | None = None,
        provider_name: str | None = None, provider_model: str | None = None,
        provider_error_type: str | None = None, provider_request_id: str | None = None,
        provider_attempts: tuple[VisualProviderAttempt, ...] = (),
        selected_visual_provider: str | None = None,
        provider_failover_used: bool = False,
        provider_failover_reason: str | None = None,
    ) -> None:
        provider_error_category = _provider_error_category(code)
        diagnostic = ProviderErrorDiagnostic(
            code, http_status, provider_code, message or _PROVIDER_ERROR_MESSAGES.get(
                code, "Remote visual provider request failed.",
            ), provider_name, provider_model, provider_error_type, provider_request_id,
            provider_error_category,
        )
        super().__init__(code)
        self.code = code
        self.diagnostic = diagnostic
        self.provider_attempts = provider_attempts
        self.selected_visual_provider = selected_visual_provider
        self.provider_failover_used = provider_failover_used
        self.provider_failover_reason = provider_failover_reason

    def with_provider_context(
        self, provider_name: str, provider_model: str, *,
        provider_error_type: str | None = None, provider_request_id: str | None = None,
        provider_attempts: tuple[VisualProviderAttempt, ...] | None = None,
        selected_visual_provider: str | None = None,
        provider_failover_used: bool | None = None,
        provider_failover_reason: str | None = None,
    ) -> VisualProviderFailure:
        return VisualProviderFailure(
            self.code, http_status=self.diagnostic.http_status,
            provider_code=self.diagnostic.provider_code,
            message=self.diagnostic.message,
            provider_name=provider_name, provider_model=provider_model,
            provider_error_type=(provider_error_type or self.diagnostic.provider_error_type),
            provider_request_id=(provider_request_id or self.diagnostic.provider_request_id),
            provider_attempts=(self.provider_attempts if provider_attempts is None else provider_attempts),
            selected_visual_provider=(self.selected_visual_provider if selected_visual_provider is None
                                      else selected_visual_provider),
            provider_failover_used=(self.provider_failover_used if provider_failover_used is None
                                    else provider_failover_used),
            provider_failover_reason=(self.provider_failover_reason if provider_failover_reason is None
                                      else provider_failover_reason),
        )


def _provider_error_category(code: str) -> str:
    if code in {"authentication_error", "permission_error"}:
        return "authentication"
    if code in {"payment_required", "quota_exceeded", "insufficient_quota"}:
        return "quota"
    if code in {"rate_limited", "rate_limit"}:
        return "rate_limit"
    if code == "timeout":
        return "timeout"
    if code in {"network_error", "connection_error"}:
        return "connection"
    if code == "invalid_request":
        return "invalid_request"
    if code in {"malformed_response", "invalid_response", "refusal"}:
        return "invalid_response"
    if code in {"server_error", "provider_unavailable"}:
        return "server"
    return "unknown"


_PROVIDER_ERROR_MESSAGES = {
    "authentication_error": "The visual provider rejected the API credentials.",
    "permission_error": "The API key or account is not permitted to perform this request.",
    "payment_required": "The provider requires available credit or billing access.",
    "rate_limited": "The provider rate limit was reached.",
    "model_not_found": "The configured visual model was not found.",
    "no_eligible_provider": "No eligible endpoint satisfies the selected model and routing/privacy policy.",
    "provider_unavailable": "The selected model provider is temporarily unavailable.",
    "invalid_request": "The provider rejected the request parameters.",
    "unsupported_response_format": "No eligible endpoint supports the required JSON response format.",
    "timeout": "The visual provider request timed out.",
    "network_error": "The visual provider could not be reached.",
    "server_error": "The visual provider returned a server error.",
    "malformed_response": "The visual provider returned a malformed response.",
    "unknown_api_error": "The visual provider returned an unclassified API error.",
    "invalid_response": "The visual provider returned an invalid response.",
    "refusal": "The visual provider refused the observation request.",
}


@dataclass(frozen=True, slots=True)
class VisualObservation:
    candidates: tuple[VisualCandidate, ...]
    provider: str
    model: str = ""
    latency_ms: int | None = None
    usage: tuple[tuple[str, int], ...] = ()
    execution_authorized: bool = False
    pricing_class: str = "unknown"
    request_build_ms: int | None = None
    response_parse_ms: int | None = None
    requested_max_elements: int | None = None
    returned_visual_elements: int | None = None
    directed_grounding: bool = False
    raw_element_count: int | None = None
    parsed_element_count: int | None = None
    provider_attempts: tuple[VisualProviderAttempt, ...] = ()
    provider_failover_used: bool = False
    provider_failover_reason: str | None = None
    field_value: str | None = field(default=None, repr=False)


class ScreenCapture(Protocol):
    def capture(
        self, snapshot_id: str, expected_handle: int, process_name: str,
        sensitive_regions: Sequence[Rect] = (),
    ) -> ScreenshotCapture: ...

    def current_window_bounds(self, handle: int) -> Rect: ...

    def current_virtual_screen_bounds(self) -> Rect: ...


@dataclass(frozen=True, slots=True)
class VisualReadinessOptions:
    timeout_seconds: float = 3.0
    poll_interval_seconds: float = .2

    def __post_init__(self) -> None:
        if not 0 <= self.timeout_seconds <= 15:
            raise ValueError("visual readiness timeout must be between 0 and 15 seconds")
        if not .05 <= self.poll_interval_seconds <= 2:
            raise ValueError("visual readiness poll interval must be between 50 and 2000 ms")


@dataclass(frozen=True, slots=True)
class ResultReadinessOptions:
    timeout_seconds: float = 7.0
    poll_interval_seconds: float = .2
    settle_quiet_ms: int = 1200
    transition_grace_ms: int = 2500
    simplification_ratio_threshold: float = RESULT_SIMPLIFICATION_RATIO_THRESHOLD
    simplification_min_signals: int = RESULT_SIMPLIFICATION_MIN_SIGNALS

    def __post_init__(self) -> None:
        if not 0 <= self.timeout_seconds <= 15:
            raise ValueError("result readiness timeout must be between 0 and 15 seconds")
        if not .05 <= self.poll_interval_seconds <= 2:
            raise ValueError("result readiness poll interval must be between 50 and 2000 ms")
        if not 0 <= self.settle_quiet_ms <= 10_000:
            raise ValueError("result readiness quiet time must be between 0 and 10000 ms")
        if not 0 <= self.transition_grace_ms <= 10_000:
            raise ValueError("result transition grace must be between 0 and 10000 ms")
        if not 0 < self.simplification_ratio_threshold <= 1:
            raise ValueError("result simplification ratio threshold must be in (0, 1]")
        if not 2 <= self.simplification_min_signals <= 3:
            raise ValueError("result simplification requires 2 or 3 independent signals")


def visual_frames_meaningfully_differ(a: VisualReadinessFrame, b: VisualReadinessFrame) -> bool:
    """Compare bounded render statistics without retaining or exposing pixels."""
    byte_delta = abs(a.png_byte_length - b.png_byte_length)
    return any((
        byte_delta >= max(1024, round(max(a.png_byte_length, b.png_byte_length) * .02)),
        abs(a.sampled_unique_colors - b.sampled_unique_colors) >= 16,
        abs(a.luminance_stddev - b.luminance_stddev) >= 1.0,
        abs(a.channel_range - b.channel_range) >= 8,
    ))


def visual_readiness_frame(diagnostics: CaptureDiagnostics) -> VisualReadinessFrame:
    megapixels = max(1e-6, diagnostics.capture_width * diagnostics.capture_height / 1_000_000)
    channel_range = max(
        maximum - minimum
        for minimum, maximum in zip(diagnostics.channel_min, diagnostics.channel_max)
    )
    return VisualReadinessFrame(
        diagnostics.png_byte_length, diagnostics.sampled_unique_colors,
        diagnostics.luminance_stddev,
        round(diagnostics.png_byte_length / megapixels, 3), channel_range,
    )


def result_readiness_richness_ratios(
    baseline: VisualReadinessFrame | None,
    candidate: VisualReadinessFrame | None,
) -> ResultReadinessRichnessRatios:
    """Compare bounded frame complexity signals without inspecting pixels."""
    if baseline is None or candidate is None:
        return ResultReadinessRichnessRatios()

    def ratio(current: float, original: float) -> float | None:
        return round(current / original, 3) if original > 0 else None

    return ResultReadinessRichnessRatios(
        png_complexity=ratio(
            candidate.encoded_bytes_per_megapixel,
            baseline.encoded_bytes_per_megapixel,
        ),
        unique_colors=ratio(candidate.sampled_unique_colors, baseline.sampled_unique_colors),
        luminance_stddev=ratio(candidate.luminance_stddev, baseline.luminance_stddev),
    )


def result_simplification_signal_count(
    ratios: ResultReadinessRichnessRatios, threshold: float,
) -> int:
    """Count independent richness ratios that fell below the configured threshold."""
    return sum(
        value is not None and value < threshold
        for value in (
            ratios.png_complexity, ratios.unique_colors, ratios.luminance_stddev,
        )
    )


def visual_frame_is_informative(frame: VisualReadinessFrame) -> bool:
    """Reject only frames where several independent low-structure signals agree."""
    suspicious_signals = sum((
        frame.sampled_unique_colors <= 64,
        frame.luminance_stddev <= 3.0,
        frame.encoded_bytes_per_megapixel <= 20_000,
        frame.channel_range <= 48,
    ))
    return suspicious_signals < 3


@dataclass(frozen=True, slots=True)
class VisualFallbackDecision:
    required: bool
    reason: str | None
    useful_named_interactive_controls: int


_INTERACTIVE = frozenset({
    "Button", "CheckBox", "ComboBox", "Document", "Edit", "Hyperlink", "ListItem",
    "MenuItem", "RadioButton", "Slider", "Spinner", "TabItem", "TreeItem",
})
_TEXT_GOAL = re.compile(r"\b(?:type|write|enter|search|go\s+to|navigate|visit)\b", re.I)


def visual_fallback_policy(observation: Observation, request: str = "") -> VisualFallbackDecision:
    useful = [control for control in observation.elements
              if control.visible is True and control.enabled is True
              and control.control_type in _INTERACTIVE and bool(control.name.strip())
              and control.is_password is not True]
    if not useful:
        return VisualFallbackDecision(True, "insufficient_named_interactive_controls", 0)
    if len(observation.elements) <= 8 or len(useful) < 3:
        return VisualFallbackDecision(True, "sparse_uia", len(useful))
    if (_TEXT_GOAL.search(request)
            and not any(control.control_type in {"Edit", "Document"} for control in useful)):
        return VisualFallbackDecision(True, "no_editable_controls", len(useful))
    if observation.truncated and len(useful) < 5:
        return VisualFallbackDecision(True, "truncated_uia", len(useful))
    return VisualFallbackDecision(False, None, len(useful))


def _clean_text(value: object, limit: int) -> str:
    if not isinstance(value, str) or "\x00" in value:
        return ""
    value = " ".join(value.split())
    return value[:limit]


def validate_visual_candidates(
    candidates: Sequence[VisualCandidate], metadata: ScreenshotMetadata,
    *, max_elements: int = 80, max_text: int = 160, redactor: Any = None,
) -> tuple[VisualElement, ...]:
    """Normalize an untrusted provider response and assign observation-local IDs."""
    validated, _rejections = validate_visual_candidates_detailed(
        candidates, metadata, max_elements=max_elements, max_text=max_text, redactor=redactor,
    )
    return validated


def validate_visual_candidates_detailed(
    candidates: Sequence[VisualCandidate], metadata: ScreenshotMetadata,
    *, max_elements: int = 80, max_text: int = 160, redactor: Any = None,
) -> tuple[tuple[VisualElement, ...], dict[str, int]]:
    """Validate candidates and return aggregate safe rejection reason counts."""
    if max_elements < 1 or max_text < 1:
        raise ValueError("Visual limits must be positive.")
    validated: list[VisualElement] = []
    rejected = {
        "invalid_bbox": 0, "invalid_label": 0, "unsupported_role": 0,
        "outside_capture": 0, "duplicate": 0, "snapshot_mismatch": 0,
        "other_validation_failure": 0,
    }
    for candidate in tuple(candidates)[:max_elements]:
        if not isinstance(candidate, VisualCandidate):
            rejected["other_validation_failure"] += 1
            continue
        rect = candidate.rectangle
        if (not isinstance(rect, Rect) or rect.left < 0 or rect.top < 0
                or rect.right <= rect.left or rect.bottom <= rect.top):
            rejected["invalid_bbox"] += 1
            continue
        if rect.right > metadata.pixel_width or rect.bottom > metadata.pixel_height:
            rejected["outside_capture"] += 1
            continue
        if (candidate.confidence is not None and (
                type(candidate.confidence) not in (float, int)
                or not math.isfinite(candidate.confidence) or not 0 <= candidate.confidence <= 1)):
            rejected["other_validation_failure"] += 1
            continue
        if candidate.activity not in {None, "active", "not_active", "unknown"}:
            rejected["other_validation_failure"] += 1
            continue
        if candidate.selection_state not in {
            VisualSelectionState.SELECTED,
            VisualSelectionState.NOT_SELECTED,
            VisualSelectionState.UNKNOWN,
        }:
            rejected["other_validation_failure"] += 1
            continue
        if candidate.region not in {
            VisualRegion.NAVIGATION, VisualRegion.DETAIL, VisualRegion.HEADER,
            VisualRegion.CONTENT, VisualRegion.UNKNOWN,
        }:
            rejected["other_validation_failure"] += 1
            continue
        if ((candidate.field_label is not None and not isinstance(candidate.field_label, str))
                or (candidate.field_value is not None and not isinstance(candidate.field_value, str))
                or (candidate.is_query_field is not None
                    and type(candidate.is_query_field) is not bool)
                or (candidate.credential_risk is not None
                    and type(candidate.credential_risk) is not bool)
                or (candidate.is_query_field is not True and candidate.field_value is not None)):
            rejected["other_validation_failure"] += 1
            continue
        label = _clean_text(candidate.label, max_text)
        field_label = (
            _clean_text(candidate.field_label, max_text)
            if candidate.field_label is not None else None
        )
        field_value = (
            _clean_text(candidate.field_value, max_text)
            if candidate.field_value is not None else None
        )
        raw_role = _clean_text(candidate.role, 40)
        if not label and not raw_role:
            rejected["invalid_label"] += 1
            continue
        role = raw_role or "unknown"
        parent = _clean_text(candidate.parent, 120)
        if redactor is not None:
            label, parent = redactor.clean(label), redactor.clean(parent)
            field_label = redactor.clean(field_label) if field_label is not None else None
            field_value = redactor.clean(field_value) if field_value is not None else None
        validated.append(VisualElement(
            f"v{len(validated) + 1}", label, role, rect,
            float(candidate.confidence) if candidate.confidence is not None else None,
            candidate.clickable is True, parent, activity=candidate.activity,
            selection_state=candidate.selection_state,
            region=candidate.region,
            field_label=field_label, field_value=field_value,
            is_query_field=candidate.is_query_field,
            credential_risk=candidate.credential_risk,
        ))
    return tuple(validated), rejected


def visual_rect_to_screen(rect: Rect, metadata: ScreenshotMetadata) -> Rect:
    """Convert capture-pixel-relative geometry to the capture's screen coordinate space."""
    if metadata.scale_x <= 0 or metadata.scale_y <= 0:
        raise ValueError("Invalid capture scale.")
    return Rect(
        metadata.capture_bounds.left + round(rect.left / metadata.scale_x),
        metadata.capture_bounds.top + round(rect.top / metadata.scale_y),
        metadata.capture_bounds.left + round(rect.right / metadata.scale_x),
        metadata.capture_bounds.top + round(rect.bottom / metadata.scale_y),
    )


def visual_click_point(element: VisualElement, metadata: ScreenshotMetadata) -> tuple[int, int]:
    rect = visual_rect_to_screen(element.rectangle, metadata)
    return ((rect.left + rect.right) // 2, (rect.top + rect.bottom) // 2)


def _normalized(value: str) -> str:
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", value).casefold()))


def _labels_similar(first: str, second: str) -> bool:
    first, second = _normalized(first), _normalized(second)
    role_words = {"button", "boton", "botón", "link", "item", "control"}
    first_semantic = " ".join(word for word in first.split() if word not in role_words)
    second_semantic = " ".join(word for word in second.split() if word not in role_words)
    return bool(first and second) and (
        first == second or (first_semantic and first_semantic == second_semantic)
        or SequenceMatcher(None, first, second).ratio() >= 0.88
    )


def _overlap_ratio(first: Rect, second: Rect) -> float:
    width = max(0, min(first.right, second.right) - max(first.left, second.left))
    height = max(0, min(first.bottom, second.bottom) - max(first.top, second.top))
    intersection = width * height
    if not intersection:
        return 0.0
    smaller = min((first.right - first.left) * (first.bottom - first.top),
                  (second.right - second.left) * (second.bottom - second.top))
    return intersection / smaller if smaller > 0 else 0.0


_ROLE_COMPATIBILITY = {
    "button": {"Button", "MenuItem", "Hyperlink"},
    "button like": {"Button", "MenuItem", "Hyperlink"},
    "edit": {"Edit", "Document", "ComboBox"},
    "text field": {"Edit", "Document", "ComboBox"},
    "list item": {"ListItem", "TreeItem", "DataItem"},
    "navigation item": {"ListItem", "TreeItem", "TabItem", "Hyperlink"},
}


def deduplicate_visual_elements(
    uia_controls: Sequence[UIElement], visual: Sequence[VisualElement], metadata: ScreenshotMetadata,
) -> tuple[VisualElement, ...]:
    """Prefer a same-label, compatible UIA target only when bounds strongly overlap."""
    retained: list[VisualElement] = []
    for element in visual:
        screen_rect = visual_rect_to_screen(element.rectangle, metadata)
        compatible = _ROLE_COMPATIBILITY.get(_normalized(element.role), set())
        duplicate = any(
            _labels_similar(element.label, control.name)
            and (not compatible or control.control_type in compatible)
            and _overlap_ratio(screen_rect, control.rectangle) >= 0.7
            for control in uia_controls if isinstance(control.rectangle, Rect)
        )
        if not duplicate:
            retained.append(element)
    return tuple(replace(element, id=f"v{index}") for index, element in enumerate(retained, 1))


def deduplicate_visual_elements_within(
    visual: Sequence[VisualElement],
) -> tuple[VisualElement, ...]:
    """Collapse nested/overlapping provider duplicates before UIA comparison."""
    retained: list[VisualElement] = []

    def compatible(first: VisualElement, second: VisualElement) -> bool:
        first_role, second_role = _normalized(first.role), _normalized(second.role)
        same_family = first_role == second_role or {
            first_role, second_role,
        } <= {"button", "icon button", "link", "navigation item", "menu item"}
        labels = _labels_similar(first.label, second.label) or not first.label or not second.label
        return same_family and labels and _overlap_ratio(first.rectangle, second.rectangle) >= 0.8

    def score(element: VisualElement) -> tuple[int, int, int, float, int]:
        area = ((element.rectangle.right - element.rectangle.left)
                * (element.rectangle.bottom - element.rectangle.top))
        return (int(element.clickable), int(bool(element.label)),
                int(_normalized(element.role) not in {"", "other interactive", "unknown"}),
                element.confidence if element.confidence is not None else -1.0, area)

    for element in visual:
        duplicate_index = next((index for index, prior in enumerate(retained)
                                if compatible(prior, element)), None)
        if duplicate_index is None:
            retained.append(element)
        elif score(element) > score(retained[duplicate_index]):
            retained[duplicate_index] = element
    return tuple(replace(element, id=f"v{index}") for index, element in enumerate(retained, 1))


class FakeVisualObserver:
    """Deterministic provider used by tests and local architecture experiments."""

    name = "fake"

    def __init__(self, candidates: Sequence[VisualCandidate]) -> None:
        self.candidates = tuple(candidates)
        self.calls: list[tuple[str, str]] = []

    def observe(
        self, screenshot: ScreenshotCapture, window: Observation, original_request: str,
        grounding: VisualGroundingRequest | None = None,
    ) -> VisualObservation:
        self.calls.append((screenshot.metadata.snapshot_id, original_request))
        return VisualObservation(
            self.candidates, self.name, model="deterministic-fixture", execution_authorized=True,
            pricing_class="local_test",
        )
