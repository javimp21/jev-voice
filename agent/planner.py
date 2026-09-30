"""Typed System-2 planning contracts, validation, and an OpenAI adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import json
import os
import re
from typing import Literal, Protocol
import unicodedata

from pydantic import BaseModel, ConfigDict, ValidationError


MAX_PLAN_STEPS = 8
MAX_PLAN_TASK_CHARS = 4_000
MAX_PLAN_OBJECTIVE_CHARS = 2_000
MAX_PLAN_TARGET_CHARS = 160
MAX_PLAN_PAYLOAD_CHARS = 1_000
DEFAULT_PLANNER_MODEL = "gpt-6-luna"
DEFAULT_PLANNER_TIMEOUT_SECONDS = 30


class PlanStepKind(StrEnum):
    OPEN_APP = "OPEN_APP"
    ACTIVATE_TARGET = "ACTIVATE_TARGET"
    SEARCH = "SEARCH"
    PRESS_KEY = "PRESS_KEY"
    TYPE_TEXT = "TYPE_TEXT"
    FINISH = "FINISH"


class PlanStepStatus(StrEnum):
    PENDING = "pending"
    COMPLETE = "complete"
    FAILED = "failed"


class PlanStatus(StrEnum):
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


class PlannerFailureReason(StrEnum):
    STEP_EXECUTION_FAILED = "step_execution_failed"


@dataclass(frozen=True, slots=True)
class PlannerContext:
    max_steps: int


@dataclass(frozen=True, slots=True)
class PlannerCurrentState:
    """Content-free facts offered for a bounded replan."""

    completed_step_ids: tuple[str, ...]
    failed_step_id: str
    observation_complete: bool
    remaining_plan_slots: int


class PlanStep(BaseModel):
    """A closed-kind high-level intent; never contains control IDs or geometry."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    step_id: str
    kind: PlanStepKind
    target: str | None
    payload: str | None
    status: PlanStepStatus
    attempts: int
    last_failure_reason: PlannerFailureReason | None


class Plan(BaseModel):
    """Strict model output and local progress state for one user task."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    objective: str
    steps: list[PlanStep]
    max_steps: int
    status: PlanStatus


class PlanUpdate(BaseModel):
    """Strict replacement for the remaining, not-yet-completed plan steps."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    steps: list[PlanStep]


class ProviderPlanStep(BaseModel):
    """Provider-facing semantic step; internal IDs and progress stay local."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    kind: PlanStepKind
    target: str | None
    payload: str | None


class ProviderPlan(BaseModel):
    """Structured initial plan output without runtime bookkeeping fields."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    objective: str
    steps: list[ProviderPlanStep]


class ProviderPlanUpdate(BaseModel):
    """Structured replacement steps without provider-controlled identifiers."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    steps: list[ProviderPlanStep]


def _materialize_steps(
    steps: list[ProviderPlanStep], *, first_id: int = 1,
) -> list[PlanStep]:
    """Create internal steps in provider order with canonical local IDs."""

    return [
        PlanStep(
            step_id=f"step_{first_id + index}", kind=step.kind,
            target=step.target, payload=step.payload,
            status=PlanStepStatus.PENDING, attempts=0,
            last_failure_reason=None,
        )
        for index, step in enumerate(steps)
    ]


def _materialize_plan(candidate: ProviderPlan, *, max_steps: int) -> Plan:
    return Plan(
        objective=candidate.objective,
        steps=_materialize_steps(candidate.steps),
        max_steps=max_steps,
        status=PlanStatus.PLANNED,
    )


def _materialize_plan_update(
    candidate: ProviderPlanUpdate, *, first_id: int,
) -> PlanUpdate:
    return PlanUpdate(steps=_materialize_steps(candidate.steps, first_id=first_id))


class Planner(Protocol):
    def plan(self, task: str, context: PlannerContext) -> Plan: ...

    def replan(
        self, plan: Plan, current_state: PlannerCurrentState,
        failure: PlannerFailureReason,
    ) -> PlanUpdate: ...


class PlannerConfigurationError(ValueError):
    """Safe planner configuration failure; never contains a credential."""


ProviderFailureCategory = Literal[
    "timeout", "connection_error", "authentication_error", "rate_limited",
    "bad_request", "model_not_found", "structured_output_error",
    "server_error", "unknown_provider_error",
]

_PROVIDER_FAILURE_CATEGORIES = frozenset({
    "timeout", "connection_error", "authentication_error", "rate_limited",
    "bad_request", "model_not_found", "structured_output_error",
    "server_error", "unknown_provider_error",
})
_SAFE_PROVIDER_CODES = frozenset({
    "invalid_api_key", "model_not_found", "rate_limit_exceeded",
    "insufficient_quota", "permission_denied", "invalid_request_error",
    "server_error", "server_is_overloaded", "credit_balance_exhausted",
    "organization_spend_limit_exceeded", "project_spend_limit_exceeded",
    "organization_usage_limit_exceeded", "slow_down", "invalid_json_schema",
    "invalid_response_format", "unsupported_response_format",
    "context_length_exceeded", "pydantic_validation_error",
    "api_response_validation_error", "missing_structured_output",
})


@dataclass(frozen=True, slots=True)
class ProviderErrorDiagnostic:
    """Bounded provider metadata; intentionally excludes messages and bodies."""

    category: ProviderFailureCategory
    provider_stage: Literal["request", "response_parse"]
    provider_error_code: str | None = None
    http_status: int | None = None
    timeout: bool = False
    connection_error: bool = False
    rate_limited: bool = False
    authentication_error: bool = False
    model_not_found: bool = False
    structured_output_error: bool = False
    request_id: str | None = None
    model_name: str | None = None

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "category": self.category,
            "provider_stage": self.provider_stage,
            "timeout": self.timeout,
            "connection_error": self.connection_error,
            "rate_limited": self.rate_limited,
            "authentication_error": self.authentication_error,
            "model_not_found": self.model_not_found,
            "structured_output_error": self.structured_output_error,
        }
        if self.provider_error_code is not None:
            result["provider_error_code"] = self.provider_error_code
        if self.http_status is not None:
            result["http_status"] = self.http_status
        if self.request_id is not None:
            result["request_id"] = self.request_id
        if self.model_name is not None:
            result["model_name"] = self.model_name
        return result


def _safe_provider_code(error: BaseException) -> str | None:
    try:
        code = getattr(error, "code", None)
    except Exception:
        return None
    if isinstance(code, str) and code in _SAFE_PROVIDER_CODES:
        return code
    return None


def _safe_request_id(error: BaseException) -> str | None:
    try:
        request_id = getattr(error, "request_id", None)
        if not isinstance(request_id, str):
            response = getattr(error, "response", None)
            headers = getattr(response, "headers", None)
            request_id = headers.get("x-request-id") if headers is not None else None
    except Exception:
        return None
    if isinstance(request_id, str) and re.fullmatch(r"req_[A-Za-z0-9_-]{8,72}", request_id):
        return request_id
    return None


def _safe_model_name(model_name: str) -> str | None:
    if re.fullmatch(r"gpt-[A-Za-z0-9][A-Za-z0-9._:-]{0,63}", model_name):
        return model_name
    return None


def _provider_error_diagnostic(
    error: BaseException, *, model_name: str, stage: str = "request",
) -> ProviderErrorDiagnostic:
    """Map SDK exception types/statuses without reading their messages or bodies."""

    try:
        import openai
    except ImportError:
        openai = None  # type: ignore[assignment]

    provider_stage: Literal["request", "response_parse"] = (
        "response_parse" if stage == "response_parse" else "request"
    )
    code = _safe_provider_code(error)
    try:
        status_value = getattr(error, "status_code", None)
    except Exception:
        status_value = None
    status = status_value if type(status_value) is int and 100 <= status_value <= 599 else None

    if isinstance(error, ValidationError) or (
        openai is not None and isinstance(error, openai.APIResponseValidationError)
    ):
        category: ProviderFailureCategory = "structured_output_error"
        provider_stage = "response_parse"
        if code is None:
            code = (
                "pydantic_validation_error" if isinstance(error, ValidationError)
                else "api_response_validation_error"
            )
    elif openai is not None and isinstance(error, openai.APITimeoutError) or status == 408:
        category = "timeout"
    elif openai is not None and isinstance(error, openai.APIConnectionError):
        category = "connection_error"
    elif openai is not None and isinstance(error, openai.APIStatusError):
        if status in {401, 403}:
            category = "authentication_error"
        elif status == 429:
            category = "rate_limited"
        elif status == 404 and code == "model_not_found":
            category = "model_not_found"
        elif status in {400, 422}:
            category = (
                "structured_output_error"
                if code in {"invalid_json_schema", "invalid_response_format", "unsupported_response_format"}
                else "bad_request"
            )
        elif status is not None and status >= 500:
            category = "server_error"
        else:
            category = "unknown_provider_error"
    else:
        category = "unknown_provider_error"

    return ProviderErrorDiagnostic(
        category=category,
        provider_stage=provider_stage,
        provider_error_code=code,
        http_status=status,
        timeout=category == "timeout",
        connection_error=category == "connection_error",
        rate_limited=category == "rate_limited",
        authentication_error=category == "authentication_error",
        model_not_found=category == "model_not_found",
        structured_output_error=category == "structured_output_error",
        request_id=_safe_request_id(error),
        model_name=_safe_model_name(model_name),
    )


class PlannerCallError(RuntimeError):
    """Sanitized structured-planning request failure."""

    def __init__(
        self, category: str | ProviderErrorDiagnostic,
        diagnostic: ProviderErrorDiagnostic | None = None,
    ) -> None:
        if isinstance(category, ProviderErrorDiagnostic):
            diagnostic = category
            safe_category = category.category
        elif category in _PROVIDER_FAILURE_CATEGORIES:
            safe_category = category
        elif category == "invalid_structured_response":
            safe_category = "structured_output_error"
        else:
            safe_category = "unknown_provider_error"
        super().__init__(safe_category)
        self.category: ProviderFailureCategory = safe_category  # type: ignore[assignment]
        self.diagnostic = diagnostic or ProviderErrorDiagnostic(
            category=self.category, provider_stage="response_parse"
            if self.category == "structured_output_error" else "request",
            timeout=self.category == "timeout",
            connection_error=self.category == "connection_error",
            rate_limited=self.category == "rate_limited",
            authentication_error=self.category == "authentication_error",
            model_not_found=self.category == "model_not_found",
            structured_output_error=self.category == "structured_output_error",
        )


@dataclass(frozen=True, slots=True)
class PlanValidationDiagnostic:
    """Allowlisted validation metadata with no rejected values or prompts."""

    validation_stage: str
    validation_code: str
    reason_category: str
    step_index: int | None = None
    step_kind: str | None = None
    field_name: str | None = None
    field_path: str | None = None
    error_type: str | None = None

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "validation_stage": self.validation_stage,
            "validation_code": self.validation_code,
            "reason_category": self.reason_category,
        }
        for name in (
            "step_index", "step_kind", "field_name", "field_path", "error_type",
        ):
            value = getattr(self, name)
            if value is not None:
                result[name] = value
        return result


class PlanValidationError(ValueError):
    """A plan is invalid or cannot safely be executed."""

    _CATEGORIES = {
            "invalid_plan_schema", "invalid_plan_status", "invalid_plan_size",
            "invalid_step_status", "invalid_step_id", "duplicate_step_id",
            "duplicate_step_pattern", "invalid_step_fields", "untrusted_literal",
            "unsafe_target", "unsafe_coordinates", "unsupported_key", "replan_step_collision",
        }
    _DEFAULT_DIAGNOSTICS = {
        "invalid_plan_schema": ("schema", "schema_validation_failed", "schema"),
        "invalid_plan_status": ("plan", "invalid_plan_status", "plan_invariant"),
        "invalid_plan_size": ("plan", "invalid_plan_size", "bounds"),
        "invalid_step_status": ("steps", "invalid_step_status", "step_state"),
        "invalid_step_id": ("steps", "invalid_step_id", "step_identity"),
        "duplicate_step_id": ("steps", "duplicate_step_id", "duplicate"),
        "duplicate_step_pattern": ("steps", "duplicate_step_pattern", "duplicate"),
        "invalid_step_fields": ("steps", "invalid_step_fields", "field_shape"),
        "untrusted_literal": ("steps", "payload_not_grounded_in_task", "literal_grounding"),
        "unsafe_target": ("steps", "shell_or_path_target_rejected", "unsafe_target"),
        "unsafe_coordinates": ("steps", "unsafe_coordinates", "unsafe_geometry"),
        "unsupported_key": ("steps", "unsupported_key", "allowlist"),
        "replan_step_collision": ("replan", "duplicate_step_pattern", "duplicate"),
    }
    _SAFE_STAGES = frozenset({"schema", "plan", "steps", "replan"})
    _SAFE_CODES = frozenset({
        "schema_validation_failed", "invalid_plan_status", "invalid_plan_size",
        "empty_plan", "too_many_steps", "missing_finish_step", "premature_finish_step",
        "missing_required_field", "invalid_step_kind", "invalid_step_status",
        "invalid_step_id", "duplicate_step_id", "duplicate_step_pattern",
        "invalid_step_fields", "payload_not_grounded_in_task",
        "shell_or_path_target_rejected", "unsafe_coordinates", "unsupported_key",
        "target_not_grounded_in_task", "payload_too_long", "target_too_long",
    })
    _SAFE_FIELDS = frozenset({
        "objective", "steps", "max_steps", "status", "step_id", "kind",
        "target", "payload", "attempts", "last_failure_reason", "unknown_field",
    })
    _SAFE_REASONS = frozenset({
        "schema", "plan_invariant", "bounds", "step_state", "step_identity",
        "duplicate", "field_shape", "literal_grounding", "unsafe_target",
        "unsafe_geometry", "allowlist", "required_step",
    })

    def __init__(
        self, category: str, *, validation_stage: str | None = None,
        validation_code: str | None = None, reason_category: str | None = None,
        step_index: int | None = None, step_kind: str | None = None,
        field_name: str | None = None, error_type: str | None = None,
    ) -> None:
        if category not in self._CATEGORIES:
            category = "invalid_plan_schema"
        super().__init__(category)
        self.category = category
        default_stage, default_code, default_reason = self._DEFAULT_DIAGNOSTICS[category]
        safe_stage = validation_stage if validation_stage in self._SAFE_STAGES else default_stage
        safe_code = validation_code if validation_code in self._SAFE_CODES else default_code
        safe_reason = reason_category if reason_category in self._SAFE_REASONS else default_reason
        safe_index = step_index if type(step_index) is int and 0 <= step_index < MAX_PLAN_STEPS else None
        safe_kind = step_kind if step_kind in {item.value for item in PlanStepKind} else None
        safe_field = field_name if field_name in self._SAFE_FIELDS else None
        path_parts: list[str] = []
        if safe_field == "steps":
            path_parts.append("steps")
        elif safe_index is not None:
            path_parts.extend(("steps", str(safe_index)))
            if safe_field is not None:
                path_parts.append(safe_field)
        elif safe_field is not None:
            path_parts.append(safe_field)
        safe_error_type = (
            error_type if isinstance(error_type, str)
            and re.fullmatch(r"[a-z_]{1,40}", error_type) else None
        )
        self.diagnostic = PlanValidationDiagnostic(
            safe_stage, safe_code, safe_reason, safe_index, safe_kind,
            safe_field, ".".join(path_parts) or None, safe_error_type,
        )


def _pydantic_diagnostic(
    error: ValidationError, *, stage: str,
) -> PlanValidationError:
    """Translate only Pydantic's safe type/location metadata into a diagnostic."""
    try:
        issues = error.errors(
            include_url=False, include_context=False, include_input=False,
        )
    except Exception:
        issues = []
    issue = issues[0] if issues else {}
    issue_type = issue.get("type") if isinstance(issue, dict) else None
    issue_type = (
        issue_type if isinstance(issue_type, str)
        and re.fullmatch(r"[a-z_]{1,40}", issue_type) else None
    )
    location = issue.get("loc", ()) if isinstance(issue, dict) else ()
    step_index: int | None = None
    safe_field: str | None = None
    for part in location if isinstance(location, (tuple, list)) else ():
        if type(part) is int and 0 <= part < MAX_PLAN_STEPS:
            step_index = part
        elif isinstance(part, str):
            if part in PlanValidationError._SAFE_FIELDS:
                safe_field = part
            elif safe_field is None:
                safe_field = "unknown_field"
    code = "schema_validation_failed"
    if (issue_type in {"enum", "literal_error", "is_instance_of"}
            and safe_field == "kind" and step_index is not None):
        code = "invalid_step_kind"
    elif (issue_type in {"enum", "literal_error", "is_instance_of"}
          and safe_field == "status" and step_index is not None):
        code = "invalid_step_status"
    elif (issue_type in {"enum", "literal_error", "is_instance_of"}
          and safe_field == "status"):
        code = "invalid_plan_status"
    elif issue_type == "missing":
        code = "missing_required_field"
    diagnostic_stage = stage if stage in {"schema", "plan", "replan"} else "plan"
    return PlanValidationError(
        "invalid_plan_schema", validation_stage=diagnostic_stage,
        validation_code=code, reason_category="schema",
        step_index=step_index, field_name=safe_field, error_type=issue_type,
    )


class _ResponsesResource(Protocol):
    def parse(self, **kwargs: object) -> object: ...


class StructuredResponsesClient(Protocol):
    responses: _ResponsesResource


_STEP_ID = re.compile(r"^step_([1-8])$")
_COORDINATE = re.compile(r"(?:\b[xy]\s*[:=]\s*-?\d+\b|\b\d{1,5}\s*,\s*\d{1,5}\b)", re.I)
_PATH_OR_ARGUMENT = re.compile(r"[\\/]|\s--?[A-Za-z]|^[A-Za-z]:")
_SHELL_APP_NAMES = frozenset({
    "cmd", "cmd.exe", "command prompt", "powershell", "powershell.exe",
    "windows powershell", "windows terminal", "terminal", "python", "python.exe",
    "python3", "wsl", "bash", "developer command prompt",
})
_ALLOWED_KEY_CHORDS = frozenset({
    "enter", "escape", "tab", "shift+tab", "ctrl+a", "ctrl+c",
})


def _normalized(value: str) -> str:
    folded = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(folded.split())


def _contains_user_literal(payload: str, task: str) -> bool:
    # Typed/search text must be copied exactly; case-folding could silently
    # transform identifiers, messages, or other case-sensitive user literals.
    return bool(payload and payload in task)


def _parse_model(
    model_type: type[Plan] | type[PlanUpdate], candidate: object, *,
    stage: str = "schema",
) -> Plan | PlanUpdate:
    try:
        if isinstance(candidate, model_type):
            return candidate
        return model_type.model_validate(candidate, strict=True)
    except ValidationError as exc:
        raise _pydantic_diagnostic(exc, stage=stage) from None
    except (TypeError, ValueError):
        raise PlanValidationError("invalid_plan_schema") from None


def _step_signature(step: PlanStep) -> tuple[str, str, str]:
    return (
        step.kind.value,
        _normalized(step.target or ""),
        _normalized(step.payload or ""),
    )


def _validate_step_sequence(
    steps: list[PlanStep], task: str, *, max_steps: int,
    existing_ids: set[str] | None = None,
    existing_signatures: set[tuple[str, str, str]] | None = None,
) -> None:
    if not steps:
        raise PlanValidationError(
            "invalid_plan_size", validation_stage="steps",
            validation_code="empty_plan", reason_category="required_step",
            field_name="steps",
        )
    if not 1 <= len(steps) <= min(MAX_PLAN_STEPS, max_steps):
        raise PlanValidationError(
            "invalid_plan_size", validation_stage="steps",
            validation_code="too_many_steps", reason_category="bounds",
            field_name="steps",
        )
    seen_ids = set(existing_ids or ())
    seen_signatures = set(existing_signatures or ())
    for step_index, step in enumerate(steps):
        match = _STEP_ID.fullmatch(step.step_id)
        if match is None:
            raise PlanValidationError(
                "invalid_step_id", step_index=step_index, step_kind=step.kind.value,
                field_name="step_id",
            )
        if step.step_id in seen_ids:
            raise PlanValidationError(
                "duplicate_step_id", step_index=step_index, step_kind=step.kind.value,
                field_name="step_id",
            )
        seen_ids.add(step.step_id)
        if (step.status is not PlanStepStatus.PENDING or step.attempts != 0
                or step.last_failure_reason is not None):
            raise PlanValidationError(
                "invalid_step_status", step_index=step_index, step_kind=step.kind.value,
                field_name="status",
            )
        target = step.target
        payload = step.payload
        if target is not None and _COORDINATE.search(target) is not None:
            raise PlanValidationError(
                "unsafe_coordinates", step_index=step_index, step_kind=step.kind.value,
                field_name="target",
            )
        if payload is not None and _COORDINATE.search(payload) is not None:
            raise PlanValidationError(
                "unsafe_coordinates", step_index=step_index, step_kind=step.kind.value,
                field_name="payload",
            )
        if target is not None and len(target) > MAX_PLAN_TARGET_CHARS:
            raise PlanValidationError(
                "invalid_step_fields", validation_code="target_too_long",
                reason_category="bounds", step_index=step_index,
                step_kind=step.kind.value, field_name="target",
            )
        if payload is not None and len(payload) > MAX_PLAN_PAYLOAD_CHARS:
            raise PlanValidationError(
                "invalid_step_fields", validation_code="payload_too_long",
                reason_category="bounds", step_index=step_index,
                step_kind=step.kind.value, field_name="payload",
            )
        if ((target is not None and "\x00" in target)
                or (payload is not None and "\x00" in payload)):
            raise PlanValidationError(
                "invalid_step_fields", step_index=step_index,
                step_kind=step.kind.value,
                field_name="target" if target is not None and "\x00" in target else "payload",
            )
        if target is not None and _PATH_OR_ARGUMENT.search(target):
            raise PlanValidationError(
                "unsafe_target", step_index=step_index, step_kind=step.kind.value,
                field_name="target",
            )
        if step.kind is PlanStepKind.OPEN_APP:
            if not target or payload is not None:
                raise PlanValidationError(
                    "invalid_step_fields", step_index=step_index, step_kind=step.kind.value,
                    field_name="target" if not target else "payload",
                )
            if _normalized(target) in _SHELL_APP_NAMES:
                raise PlanValidationError(
                    "unsafe_target", step_index=step_index, step_kind=step.kind.value,
                    field_name="target",
                )
            if not _contains_user_literal(target, task):
                raise PlanValidationError(
                    "unsafe_target", validation_code="target_not_grounded_in_task",
                    reason_category="literal_grounding", step_index=step_index,
                    step_kind=step.kind.value, field_name="target",
                )
        elif step.kind is PlanStepKind.ACTIVATE_TARGET:
            if not target or payload is not None:
                raise PlanValidationError(
                    "invalid_step_fields", step_index=step_index,
                    step_kind=step.kind.value,
                    field_name="target" if not target else "payload",
                )
        elif step.kind is PlanStepKind.SEARCH:
            if target is not None:
                raise PlanValidationError(
                    "invalid_step_fields", step_index=step_index, step_kind=step.kind.value,
                    field_name="target",
                )
            if not payload or not _contains_user_literal(payload, task):
                raise PlanValidationError(
                    "untrusted_literal", step_index=step_index, step_kind=step.kind.value,
                    field_name="payload",
                )
        elif step.kind is PlanStepKind.PRESS_KEY:
            if (not target or payload is not None
                    or _normalized(target) not in _ALLOWED_KEY_CHORDS):
                raise PlanValidationError(
                    "unsupported_key", step_index=step_index, step_kind=step.kind.value,
                    field_name="target",
                )
        elif step.kind is PlanStepKind.TYPE_TEXT:
            if target is not None:
                raise PlanValidationError(
                    "invalid_step_fields", step_index=step_index, step_kind=step.kind.value,
                    field_name="target",
                )
            if not payload or not _contains_user_literal(payload, task):
                raise PlanValidationError(
                    "untrusted_literal", step_index=step_index, step_kind=step.kind.value,
                    field_name="payload",
                )
        elif step.kind is PlanStepKind.FINISH:
            if target is not None or payload is not None:
                raise PlanValidationError(
                    "invalid_step_fields", step_index=step_index, step_kind=step.kind.value,
                    field_name="target" if target is not None else "payload",
                )
        signature = _step_signature(step)
        if signature in seen_signatures:
            raise PlanValidationError(
                "duplicate_step_pattern", step_index=step_index, step_kind=step.kind.value,
                field_name="kind",
            )
        seen_signatures.add(signature)


def validate_plan(candidate: object, task: str, *, max_steps: int) -> Plan:
    """Parse and validate a complete untrusted planner response."""

    if not isinstance(task, str) or not task.strip() or len(task) > MAX_PLAN_TASK_CHARS:
        raise PlanValidationError("invalid_plan_status")
    parsed = _parse_model(Plan, candidate)
    assert isinstance(parsed, Plan)
    if (parsed.status is not PlanStatus.PLANNED or not parsed.objective.strip()
            or len(parsed.objective) > MAX_PLAN_OBJECTIVE_CHARS):
        raise PlanValidationError(
            "invalid_plan_status", field_name="objective" if not parsed.objective.strip()
            or len(parsed.objective) > MAX_PLAN_OBJECTIVE_CHARS else "status",
        )
    if not isinstance(max_steps, int) or not 1 <= max_steps <= MAX_PLAN_STEPS:
        raise PlanValidationError("invalid_plan_size")
    if parsed.max_steps < len(parsed.steps) or parsed.max_steps > max_steps:
        raise PlanValidationError(
            "invalid_plan_size", validation_code="too_many_steps",
            reason_category="bounds", field_name="max_steps",
        )
    _validate_step_sequence(parsed.steps, task, max_steps=max_steps)
    if parsed.steps[-1].kind is not PlanStepKind.FINISH:
        raise PlanValidationError(
            "invalid_step_fields", validation_stage="steps",
            validation_code="missing_finish_step", reason_category="required_step",
            step_index=len(parsed.steps) - 1, step_kind=parsed.steps[-1].kind.value,
            field_name="kind",
        )
    if any(step.kind is PlanStepKind.FINISH for step in parsed.steps[:-1]):
        finish_index = next(
            index for index, step in enumerate(parsed.steps[:-1])
            if step.kind is PlanStepKind.FINISH
        )
        raise PlanValidationError(
            "invalid_step_fields", validation_stage="steps",
            validation_code="premature_finish_step", reason_category="required_step",
            step_index=finish_index, step_kind=PlanStepKind.FINISH.value,
            field_name="kind",
        )
    return parsed


def validate_plan_update(
    candidate: object, task: str, *, max_steps: int,
    existing_steps: list[PlanStep],
) -> PlanUpdate:
    """Validate only replacement steps and reject retries of prior step patterns."""

    parsed = _parse_model(PlanUpdate, candidate, stage="replan")
    assert isinstance(parsed, PlanUpdate)
    used_slots = max_steps - len(existing_steps)
    existing_ids = {step.step_id for step in existing_steps}
    signatures = {_step_signature(step) for step in existing_steps}
    _validate_step_sequence(
        parsed.steps, task, max_steps=used_slots,
        existing_ids=existing_ids, existing_signatures=signatures,
    )
    if parsed.steps[-1].kind is not PlanStepKind.FINISH:
        raise PlanValidationError(
            "invalid_step_fields", validation_stage="replan",
            validation_code="missing_finish_step", reason_category="required_step",
            step_index=len(parsed.steps) - 1, step_kind=parsed.steps[-1].kind.value,
            field_name="kind",
        )
    if any(step.kind is PlanStepKind.FINISH for step in parsed.steps[:-1]):
        finish_index = next(
            index for index, step in enumerate(parsed.steps[:-1])
            if step.kind is PlanStepKind.FINISH
        )
        raise PlanValidationError(
            "invalid_step_fields", validation_stage="replan",
            validation_code="premature_finish_step", reason_category="required_step",
            step_index=finish_index, step_kind=PlanStepKind.FINISH.value,
            field_name="kind",
        )
    return parsed


_PLAN_INSTRUCTIONS = (
    "Create a short executable intent plan, not UI clicks or tool calls. "
    "Use only the closed step kinds. Targets are semantic names, never control IDs or coordinates. "
    "For SEARCH and TYPE_TEXT, copy the literal exactly from the user's task; never invent text. "
    "OPEN_APP must name an application mentioned by the user and must not name a shell or interpreter. "
    "Return only the required typed provider plan. Do not include step IDs or runtime progress fields. "
    "Do not include prose, code, commands, coordinates, tool names, credentials, or generated message content."
)


_REPLAN_INSTRUCTIONS = (
    "Replace only the remaining plan steps after the failed step. Do not repeat completed or failed "
    "step patterns. Do not invent SEARCH or TYPE_TEXT literals. Return only ordered semantic steps; "
    "do not include step IDs or runtime progress fields."
)


@dataclass(slots=True)
class OpenAIPlanner:
    """OpenAI Responses API adapter using native Pydantic Structured Outputs."""

    model: str
    client: StructuredResponsesClient = field(repr=False)

    @classmethod
    def from_environment(cls) -> OpenAIPlanner:
        api_key = os.environ.get("PLANNER_API_KEY", "").strip()
        if not api_key:
            raise PlannerConfigurationError("Set PLANNER_API_KEY to enable the planner.")
        model = os.environ.get("PLANNER_MODEL", DEFAULT_PLANNER_MODEL).strip()
        if not model or len(model) > 128 or any(char.isspace() for char in model):
            raise PlannerConfigurationError("PLANNER_MODEL is invalid.")
        try:
            timeout = float(os.environ.get(
                "PLANNER_TIMEOUT_SECONDS", str(DEFAULT_PLANNER_TIMEOUT_SECONDS),
            ))
        except ValueError:
            raise PlannerConfigurationError("PLANNER_TIMEOUT_SECONDS is invalid.") from None
        if not 1 <= timeout <= 90:
            raise PlannerConfigurationError("PLANNER_TIMEOUT_SECONDS must be between 1 and 90.")
        from openai import OpenAI
        return cls(
            model=model,
            client=OpenAI(api_key=api_key, timeout=timeout, max_retries=0),
        )

    def plan(self, task: str, context: PlannerContext) -> Plan:
        if not isinstance(task, str) or not task.strip() or len(task) > MAX_PLAN_TASK_CHARS:
            raise PlanValidationError("invalid_plan_status")
        if type(context.max_steps) is not int or not 1 <= context.max_steps <= MAX_PLAN_STEPS:
            raise PlanValidationError("invalid_plan_size")
        parsed = self._structured_call(
            ProviderPlan,
            _PLAN_INSTRUCTIONS,
            {"task": task, "max_steps": context.max_steps},
        )
        if not isinstance(parsed, ProviderPlan):
            raise PlannerCallError("invalid_structured_response")
        return _materialize_plan(parsed, max_steps=context.max_steps)

    def replan(
        self, plan: Plan, current_state: PlannerCurrentState,
        failure: PlannerFailureReason,
    ) -> PlanUpdate:
        failed_indexes = [
            index for index, step in enumerate(plan.steps)
            if step.step_id == current_state.failed_step_id
        ]
        if len(failed_indexes) != 1:
            raise PlanValidationError("invalid_step_id")
        failed_index = failed_indexes[0]
        parsed = self._structured_call(
            ProviderPlanUpdate,
            _REPLAN_INSTRUCTIONS,
            {
                "plan": [
                    {
                        "kind": step.kind.value,
                        "target": step.target,
                        "payload": step.payload,
                        "status": step.status.value,
                        "attempts": step.attempts,
                        "last_failure_reason": (
                            step.last_failure_reason.value
                            if step.last_failure_reason is not None else None
                        ),
                    }
                    for step in plan.steps
                ],
                "current_state": {
                    "completed_step_count": len(current_state.completed_step_ids),
                    "failed_step_position": failed_index,
                    "observation_complete": current_state.observation_complete,
                    "remaining_plan_slots": current_state.remaining_plan_slots,
                },
                "failure": failure.value,
            },
        )
        if not isinstance(parsed, ProviderPlanUpdate):
            raise PlannerCallError("invalid_structured_response")
        return _materialize_plan_update(parsed, first_id=failed_index + 2)

    def _structured_call(
        self, model_type: type[ProviderPlan] | type[ProviderPlanUpdate],
        instructions: str, user_payload: dict[str, object],
    ) -> ProviderPlan | ProviderPlanUpdate:
        try:
            response = self.client.responses.parse(
                model=self.model,
                input=[
                    {"role": "system", "content": instructions},
                    {
                        "role": "user",
                        "content": json.dumps(user_payload, ensure_ascii=False, separators=(",", ":")),
                    },
                ],
                text_format=model_type,
                max_output_tokens=2_000,
            )
        except ValidationError as exc:
            raise PlannerCallError(_provider_error_diagnostic(
                exc, model_name=self.model, stage="response_parse",
            )) from None
        except Exception as exc:
            raise PlannerCallError(_provider_error_diagnostic(
                exc, model_name=self.model,
            )) from None
        parsed = getattr(response, "output_parsed", None)
        if not isinstance(parsed, model_type):
            raise PlannerCallError(ProviderErrorDiagnostic(
                category="structured_output_error",
                provider_stage="response_parse",
                provider_error_code="missing_structured_output",
                structured_output_error=True,
                model_name=_safe_model_name(self.model),
            ))
        return parsed
