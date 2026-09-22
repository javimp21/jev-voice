"""Platform-neutral action outcomes. Failure never implies a safe retry."""

from dataclasses import dataclass

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
