"""Minimal, immutable UI descriptions without Windows object references."""

from dataclasses import dataclass
from enum import StrEnum


class VisualGroundingStatus(StrEnum):
    SUCCESS_WITH_CANDIDATES = "success_with_candidates"
    SUCCESS_EMPTY = "success_empty"
    PROVIDER_ERROR = "provider_error"
    PARSE_ERROR = "parse_error"
    VALIDATION_EMPTY = "validation_empty"
    DEDUP_EMPTY = "dedup_empty"
    HANDOFF_EMPTY = "handoff_empty"


class VisualReadinessReason(StrEnum):
    INITIALLY_READY = "initially_ready"
    BECAME_READY = "became_ready"
    TIMEOUT = "timeout"
    FOREGROUND_CHANGED = "foreground_changed"
    CAPTURE_ERROR = "capture_error"
    CREDENTIAL_SENSITIVE = "credential_sensitive"
    INSUFFICIENT_VISUAL_INFORMATION = "insufficient_visual_information"


@dataclass(frozen=True, slots=True)
class VisualPipelineDiagnostic:
    provider_requested_max_elements: int | None = None
    provider_raw_element_count: int | None = None
    parsed_element_count: int | None = None
    validated_element_count: int | None = None
    deduplicated_element_count: int | None = None
    observation_visual_control_count: int = 0
    jev_visual_option_count: int | None = None


@dataclass(frozen=True, slots=True)
class VisualRequestFingerprint:
    """Safe request metadata; contains neither prompt text nor screenshot bytes."""

    provider: str
    model: str
    directed: bool
    max_elements: int
    max_output_tokens: int | None
    response_mime_type: str | None
    schema_name: str
    schema_version: str
    screenshot_dimensions: tuple[int, int]
    encoded_image_byte_length: int
    screenshot_sha256: str
    objective_length: int


@dataclass(frozen=True, slots=True)
class Rect:
    """UIA screen bounds for observation only; negative coordinates are valid."""

    left: int
    top: int
    right: int
    bottom: int


@dataclass(frozen=True, slots=True)
class UIElement:
    """An adapter-assigned ID is valid only within its observation."""

    id: str
    name: str
    control_type: str
    automation_id: str = ""
    rectangle: Rect | None = None
    enabled: bool | None = None
    visible: bool | None = None
    focused: bool | None = None
    is_password: bool | None = None
    observed_text: str | None = None
    observed_text_truncated: bool = False
    parent_name: str = ""
    parent_control_type: str = ""


@dataclass(frozen=True, slots=True)
class ScreenshotMetadata:
    """Geometry only. Screenshot pixels are never stored in an Observation."""

    snapshot_id: str
    window_handle: int
    window_bounds: Rect
    capture_bounds: Rect
    pixel_width: int
    pixel_height: int
    dpi_x: int
    dpi_y: int
    scale_x: float
    scale_y: float
    masked_regions: int = 0


@dataclass(frozen=True, slots=True)
class CaptureDiagnostics:
    """Safe local evidence about the exact masked pixels sent to vision."""

    foreground_hwnd: int
    foreground_pid: int | None
    window_bounds: Rect
    capture_bounds: Rect
    virtual_screen_bounds: Rect
    capture_width: int
    capture_height: int
    png_byte_length: int
    mask_region_count: int
    masked_area_percent: float
    capture_backend: str
    window_visible: bool
    window_minimized: bool
    sampled_unique_colors: int
    channel_min: tuple[int, int, int]
    channel_max: tuple[int, int, int]
    mean_luminance: float
    luminance_stddev: float
    near_black_percent: float
    near_white_percent: float


@dataclass(frozen=True, slots=True)
class VisualReadinessFrame:
    png_byte_length: int
    sampled_unique_colors: int
    luminance_stddev: float
    encoded_bytes_per_megapixel: float
    channel_range: int


@dataclass(frozen=True, slots=True)
class VisualReadinessResult:
    ready: bool
    reason: VisualReadinessReason
    attempts: int
    elapsed_ms: int
    initial: VisualReadinessFrame | None = None
    final: VisualReadinessFrame | None = None


@dataclass(frozen=True, slots=True)
class VisualCandidateProviderDiagnostic:
    """Bounded redacted semantic fields and validation status from one provider item."""

    label: str
    parent: str
    role: str
    clickable: bool | None
    geometry_valid: bool
    role_well_formed: bool
    rejection_reason: str | None = None
    provider_candidate_id: str = ""
    validated_visual_id: str | None = None


@dataclass(frozen=True, slots=True)
class ResultReadinessRichnessRatios:
    """Current visual richness relative to the pre-submit frame."""

    png_complexity: float | None = None
    unique_colors: float | None = None
    luminance_stddev: float | None = None


@dataclass(frozen=True, slots=True)
class ResultReadinessResult:
    """Safe diagnostics for the local-only wait before result grounding."""

    ready: bool
    reason: str
    attempts: int
    elapsed_ms: int
    meaningful_change_seen: bool = False
    final_informative: bool = False
    stable: bool = False
    after_query_submit: bool = False
    baseline: VisualReadinessFrame | None = None
    final: VisualReadinessFrame | None = None
    first_transition_elapsed_ms: int | None = None
    last_meaningful_change_elapsed_ms: int | None = None
    required_quiet_ms: int = 0
    observed_quiet_ms: int = 0
    quiet_timer_reset_count: int = 0
    informative_frame_seen: bool = False
    stable_frame_seen: bool = False
    state: str = "waiting_for_transition"
    baseline_changed: bool = False
    recent_frame_changed: bool = False
    richness_ratios: ResultReadinessRichnessRatios | None = None
    simplification_signal_count: int = 0
    transitional_simplification: bool = False
    awaiting_followup_transition: bool = False
    followup_transition_seen: bool = False
    transition_grace_ms: int = 2500
    transition_grace_elapsed_ms: int = 0
    simplification_ratio_threshold: float = 0.75
    simplification_min_signals: int = 2


@dataclass(frozen=True, slots=True)
class VisualElement:
    """Validated, snapshot-local visual target; rectangle is capture-pixel relative."""

    id: str
    label: str
    role: str
    rectangle: Rect
    confidence: float | None
    clickable: bool
    parent: str = ""
    source: str = "visual"


@dataclass(frozen=True, slots=True, eq=False)
class ProviderErrorDiagnostic:
    """Bounded remote-provider failure metadata safe for JSON diagnostics."""

    category: str
    http_status: int | None = None
    provider_code: str | None = None
    message: str = "Remote visual provider request failed."

    def __eq__(self, other: object) -> bool:
        # Preserve compatibility with callers that previously compared the string code.
        if isinstance(other, str):
            return self.category == other
        if isinstance(other, ProviderErrorDiagnostic):
            return (
                self.category, self.http_status, self.provider_code, self.message,
            ) == (
                other.category, other.http_status, other.provider_code, other.message,
            )
        return NotImplemented


@dataclass(frozen=True, slots=True)
class Observation:
    """A snapshot of the foreground application's accessible controls."""

    app_name: str
    window_title: str
    elements: tuple[UIElement, ...] = ()
    process_id: int | None = None
    control_type: str = ""
    truncated: bool = False
    inspection_errors: int = 0
    error: str | None = None
    observation_id: str = ""
    application_id: str = ""
    package_family_name: str = ""
    visual_elements: tuple[VisualElement, ...] = ()
    screenshot: ScreenshotMetadata | None = None
    visual_fallback_reason: str | None = None
    visual_provider: str | None = None
    visual_model: str | None = None
    visual_latency_ms: int | None = None
    visual_usage: tuple[tuple[str, int], ...] = ()
    visual_pricing_class: str | None = None
    visual_execution_authorized: bool = False
    visual_provider_error: ProviderErrorDiagnostic | None = None
    screenshot_capture_ms: int | None = None
    visual_request_build_ms: int | None = None
    visual_response_parse_ms: int | None = None
    visual_total_observation_ms: int | None = None
    visual_requested_max_elements: int | None = None
    visual_returned_elements: int | None = None
    visual_directed_grounding: bool = False
    visual_grounding_status: VisualGroundingStatus | None = None
    visual_pipeline: VisualPipelineDiagnostic | None = None
    visual_rejection_summary: tuple[tuple[str, int], ...] = ()
    visual_request_fingerprint: VisualRequestFingerprint | None = None
    capture_diagnostics: CaptureDiagnostics | None = None
    visual_readiness: VisualReadinessResult | None = None
    result_readiness: ResultReadinessResult | None = None
    visual_provider_candidates: tuple[VisualCandidateProviderDiagnostic, ...] = ()
    visual_provider_call_count: int = 0
    foreground_hwnd: int | None = None
