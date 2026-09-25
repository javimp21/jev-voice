"""Platform-neutral action outcomes. Failure never implies a safe retry."""

from dataclasses import dataclass
from typing import Literal

from computer.actions import Action


@dataclass(frozen=True, slots=True)
class LiteralInputDiagnostic:
    requested_character_count: int
    utf16_code_unit_count: int
    input_event_count: int
    send_input_requested_count: int
    send_input_returned_count: int
    last_error: int | None
    failure_stage: str | None
    input_struct_size: int
    foreground_hwnd: int | None = None
    foreground_pid: int | None = None
    context_revalidation_passed: bool = False


@dataclass(frozen=True, slots=True)
class VisualActivationDiagnostic:
    """Safe, bounded diagnostics for one locally validated visual activation."""

    candidate_id: str | None
    snapshot_binding_valid: bool | None = None
    candidate_lookup_succeeded: bool = False
    provenance_valid: bool | None = None
    provider_name: Literal["openai", "gemini", "other", "unknown"] = "unknown"
    provider_execution_agnostic: bool = True
    preflight_started: bool = False
    preflight_succeeded: bool = False
    preflight_failure_reason: str | None = None
    geometry_resolution_succeeded: bool = False
    resolved_point_inside_candidate: bool | None = None
    resolved_point_inside_bound_window: bool | None = None
    foreground_stable_before_input: bool | None = None
    input_attempted: bool = False
    input_result: Literal["not_attempted", "succeeded", "failed_or_unknown"] = "not_attempted"
    snapshot_consumed: bool = False
    post_action_observation_attempted: bool = False
    post_action_observation_obtained: bool = False
    normalized_box_width_bucket: str | None = None
    normalized_box_height_bucket: str | None = None
    click_point_relative_bucket: str | None = None
    candidate_box_inside_bound_window: bool | None = None
    candidate_box_aspect_bucket: str | None = None
    overlaps_actionable_candidate: bool | None = None
    nearest_actionable_neighbor_distance_bucket: str | None = None
    failure_stage: str | None = None
    failure_reason: str | None = None


@dataclass(frozen=True, slots=True)
class ActionResult:
    success: bool
    action: Action
    message: str
    error: str | None = None
    requires_confirmation: bool = False
    completed: bool = False
    source_observation_id: str | None = None
    input_issued: bool = False
    literal_input_diagnostic: LiteralInputDiagnostic | None = None
    visual_activation_diagnostic: VisualActivationDiagnostic | None = None
