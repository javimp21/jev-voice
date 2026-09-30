"""Supervised Windows debugging and a bounded Jev request loop."""

import argparse
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import TYPE_CHECKING

from dotenv import load_dotenv

from agent.loop import Agent, AgentLimits, AgentResult
from agent.hybrid_debug import (
    HybridClickDebugAgent, HybridDebugAgent, HybridResultDebugAgent, HybridTypeDebugAgent,
)
from agent.generic_task import (
    GenericTaskBudgets, GenericTaskDebugAgent, GenericVisualDebugScreenshotDiagnostic,
)
from agent.evals import format_eval_report, run_evaluations
from agent.telemetry import JsonlTelemetrySink
from computer.actions import ClickAction, OpenAppAction, PressKeyAction, TypeAction
from computer.applications import ApplicationCandidate
from computer.windows_actions import LiteralInputFailure, WindowsComputer, debug_type_literal
from computer.windows import ObservationOptions, WindowsObserver
from computer.windows_apps import WindowsApplicationCatalog
from computer.packaged_activation_diagnostic import run_packaged_activation_diagnostic
from computer.windows_capture import WindowsWindowCapture, save_debug_overlay, save_debug_screenshot
from computer.visual import (
    ResultReadinessOptions, VisualReadinessOptions, bounded_grounding_request,
    visual_fallback_policy,
)
from computer.visual_providers.common import png_bytes
from computer.visual_diagnostics import (
    DEFAULT_DIAGNOSTIC_OBJECTIVES, diagnose_visual_grounding,
)
from computer.visual_providers import (
    VisualProviderConfigurationError, check_visual_provider_from_environment,
    directed_max_elements_from_environment,
    visual_provider_from_environment,
)
from decision.client import ConfigurationError
from decision.context import Redactor
from decision.jev import JevDecisionMaker
from decision.models import DecisionResult
from safety.policy import (
    AutonomousActionPolicy, CLICK_TYPES, GenericTargetActivationPolicy,
)
from voice.models import VoiceInputError
from voice.service import voice_input_from_environment

if TYPE_CHECKING:
    from agent.graph import GraphAgentResult


_VISUAL_PROVIDER_KEY_VARIABLES = {
    "openrouter": "OPENROUTER_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
}
_SAFE_VISUAL_PROVIDER_NAMES = frozenset(_VISUAL_PROVIDER_KEY_VARIABLES)


def _accepts_keyword(method: object, parameter_name: str) -> bool:
    if not callable(method):
        return False
    try:
        parameters = inspect.signature(method).parameters
    except (TypeError, ValueError):
        return False
    parameter = parameters.get(parameter_name)
    return bool(
        parameter is not None and parameter.kind in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
        or any(item.kind is inspect.Parameter.VAR_KEYWORD for item in parameters.values())
    )


def _generic_visual_configuration(
    provider: object | None,
    *,
    configured_provider: str | None,
    computer: object | None = None,
    configuration_error: bool = False,
    mode_enabled: bool = True,
) -> dict[str, object]:
    """Expose only bounded visual setup metadata; never serialize credentials."""
    configured_name = (configured_provider or "").strip().casefold()
    provider_name = (
        configured_name if configured_name in _SAFE_VISUAL_PROVIDER_NAMES else None
    )
    if provider_name is None and provider is not None:
        candidate_name = getattr(provider, "name", None)
        if isinstance(candidate_name, str) and candidate_name.casefold() in _SAFE_VISUAL_PROVIDER_NAMES:
            provider_name = candidate_name.casefold()
    provider_configured = bool(configured_name or provider is not None)
    key_variable = _VISUAL_PROVIDER_KEY_VARIABLES.get(provider_name or "")
    api_key_present = bool(key_variable and os.environ.get(key_variable, ""))

    capture_service = getattr(computer, "capture_service", None)
    directed_observer = getattr(computer, "observe_directed", None)
    visual_observer_present = capture_service is not None
    provider_observe = getattr(provider, "observe", None)
    directed_grounding_supported = bool(
        visual_observer_present and callable(directed_observer)
        and _accepts_keyword(provider_observe, "grounding")
    )

    reason: str | None = None
    if not mode_enabled:
        reason = "disabled_by_mode"
    elif not provider_configured:
        reason = "provider_missing"
    elif provider_name is None:
        reason = "configuration_error"
    elif not api_key_present:
        reason = "api_key_missing"
    elif configuration_error:
        reason = "configuration_error"
    elif not visual_observer_present:
        reason = "observer_missing"
    elif not directed_grounding_supported:
        reason = "directed_grounding_unsupported"

    model_name: str | None = None
    model = getattr(provider, "model", None) if provider is not None else None
    if (provider_name is not None and isinstance(model, str)
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/+-]{0,119}", model)):
        model_name = model
    return {
        "visual_observer_present": visual_observer_present,
        "directed_grounding_supported": directed_grounding_supported,
        "provider_configured": provider_configured,
        "provider_name": provider_name,
        "model_name": model_name,
        "api_key_present": api_key_present,
        "visual_unavailable_reason": reason,
    }


def _candidate_json(candidate: ApplicationCandidate, score: int | None = None) -> dict[str, object]:
    redactor = Redactor()
    result: dict[str, object] = {
        "id": candidate.id, "display_name": redactor.clean(candidate.display_name)[:160],
        "publisher": redactor.clean(candidate.publisher)[:160], "source": candidate.source,
        "launch_policy": candidate.launch_policy,
    }
    if score is not None:
        result["score"] = score
    return result


def _safe_agent_result(result: AgentResult) -> dict[str, object]:
    """Return debug-safe loop metadata without labels, requests, or literals."""
    safe_errors = {None, "policy_blocked", "unsafe_target", "stale_observation", "windows_operation_failed", "unsupported_action"}
    return {
        "success": result.success,
        "stop_reason": result.stop_reason,
        "message": "Agent completed." if result.success else "Agent stopped.",
        "steps": result.steps,
        "history": [
            {"success": item.success, "action": {"kind": item.action.kind},
             "error": item.error if item.error in safe_errors else ("other_error" if item.error else None)}
            for item in result.history
        ],
        "last_observation": None if result.last_observation is None else {
            "control_count": len(result.last_observation.elements),
            "has_error": bool(result.last_observation.error),
        },
        "last_decision": None if result.last_decision is None else {
            "status": result.last_decision.status,
            "confidence": result.last_decision.confidence,
            "selected_option": result.last_decision.selected_option,
            "error": result.last_decision.error if result.last_decision.error in {
                None, "api_error", "invalid_response", "invalid_input", "configuration_error",
            } else ("other_error" if result.last_decision.error else None),
            "diagnostic": Agent._safe_diagnostic(result.last_decision),
        },
    }


def _safe_graph_agent_result(graph_result: "GraphAgentResult") -> dict[str, object]:
    """Serialize only the graph trace metadata safe for local debug output."""
    result = graph_result.result
    return {
        "success": result.success,
        "stop_reason": result.stop_reason,
        "message": "Graph run completed." if result.success else "Graph run stopped.",
        "steps": result.steps,
        "replan_count": graph_result.replan_count,
        "max_replans": graph_result.max_replans,
        "last_replan_reason": graph_result.last_replan_reason,
        "history": [
            {"success": item.success, "action_kind": item.action.kind,
             "error": item.error if item.error in {
                 None, "policy_blocked", "unsafe_target", "stale_observation",
                 "windows_operation_failed", "unsupported_action",
             } else ("other_error" if item.error else None)}
            for item in result.history
        ],
        "graph": {
            "node_sequence": [item.node for item in graph_result.diagnostics],
            "transition_reason": graph_result.transition_reason,
            "transitions": [asdict(item) for item in graph_result.diagnostics],
        },
    }


def _save_exact_result_capture(observation, capture, path: Path) -> dict[str, object]:
    """Save only the exact masked PNG paired with one result grounding request."""
    fingerprint = observation.visual_request_fingerprint
    if fingerprint is None or not fingerprint.directed:
        raise ValueError("A directed result grounding fingerprint is required.")
    encoded = png_bytes(capture)
    import hashlib
    digest = hashlib.sha256(encoded).hexdigest()
    if (len(encoded) != fingerprint.encoded_image_byte_length
            or digest != fingerprint.screenshot_sha256):
        raise ValueError("Result screenshot does not match the provider request fingerprint.")
    save_debug_screenshot(capture, path)
    saved = path.read_bytes()
    saved_digest = hashlib.sha256(saved).hexdigest()
    if (len(saved) != fingerprint.encoded_image_byte_length
            or saved_digest != fingerprint.screenshot_sha256):
        raise ValueError("Saved result screenshot does not match the provider request fingerprint.")
    return {
        "path": str(path), "sha256": saved_digest, "byte_length": len(saved),
        "matches_request_fingerprint": True,
    }


def _save_exact_generic_capture(
    observation, capture, path: Path, stage: str,
) -> GenericVisualDebugScreenshotDiagnostic:
    """Persist only an exact, fingerprint-bound PNG from generic directed grounding."""
    fingerprint = observation.visual_request_fingerprint
    if fingerprint is None or not fingerprint.directed:
        return GenericVisualDebugScreenshotDiagnostic(
            stage, False, error="directed_request_fingerprint_unavailable",
        )
    encoded = png_bytes(capture)
    digest = hashlib.sha256(encoded).hexdigest()
    if (len(encoded) != fingerprint.encoded_image_byte_length
            or digest != fingerprint.screenshot_sha256):
        return GenericVisualDebugScreenshotDiagnostic(
            stage, False, error="capture_does_not_match_provider_input",
        )
    try:
        save_debug_screenshot(capture, path)
    except FileExistsError:
        return GenericVisualDebugScreenshotDiagnostic(
            stage, False, error="path_exists_or_parent_missing",
        )
    except ValueError:
        return GenericVisualDebugScreenshotDiagnostic(stage, False, error="invalid_png_path")
    except OSError:
        return GenericVisualDebugScreenshotDiagnostic(stage, False, error="screenshot_save_failed")
    saved = path.read_bytes()
    saved_digest = hashlib.sha256(saved).hexdigest()
    matches = (
        saved == encoded
        and len(saved) == fingerprint.encoded_image_byte_length
        and saved_digest == fingerprint.screenshot_sha256
    )
    if not matches:
        return GenericVisualDebugScreenshotDiagnostic(
            stage, False, str(path), saved_digest, len(saved), False,
            "saved_file_does_not_match_provider_input",
        )
    return GenericVisualDebugScreenshotDiagnostic(
        stage, True, str(path), saved_digest, len(saved), True,
    )


def _planner_result_payload(result) -> dict[str, object]:
    """Serialize only planner metadata; omit objective, targets, and literals."""
    plan = result.plan
    return {
        "success": result.success,
        "stop_reason": result.stop_reason,
        "message": "Planner run completed." if result.success else "Planner run stopped safely.",
        "steps_completed": result.steps_completed,
        "steps_attempted": result.steps_attempted,
        "total_steps": result.total_steps,
        "planner_replan_count": result.planner_replan_count,
        "max_planner_replans": result.max_planner_replans,
        "local_replan_count": result.local_replan_count,
        "planner_latency_ms": result.planner_latency_ms,
        "error_category": result.error_category,
        "provider_diagnostic": (
            result.provider_diagnostic.as_dict()
            if result.provider_diagnostic is not None else None
        ),
        "validation_diagnostic": (
            result.validation_diagnostic.as_dict()
            if result.validation_diagnostic is not None else None
        ),
        "plan": None if plan is None else [
            {
                "step_id": step.step_id,
                "kind": step.kind.value,
                "status": step.status.value,
                "attempts": step.attempts,
                "failure_reason": step.last_failure_reason.value if step.last_failure_reason else None,
            }
            for step in plan.steps
        ],
    }


def _run_planner_debug(args: argparse.Namespace) -> int:
    """Run the isolated typed-planner path; dry-run does not touch Windows/Jev."""
    from uuid import uuid4

    from agent.loop import AgentLimits
    from agent.planner import (
        MAX_PLAN_STEPS, OpenAIPlanner, PlannerCallError, PlannerConfigurationError, PlannerContext,
        PlanValidationError, validate_plan,
    )
    from agent.telemetry import AgentTelemetryEvent, utc_timestamp
    from agent.planner_agent import PlannerAgent

    def fail(
        reason: str, message: str,
        validation_diagnostic: dict[str, object] | None = None,
        provider_diagnostic: dict[str, object] | None = None,
        error_category: str | None = None,
    ) -> int:
        payload: dict[str, object] = {
            "success": False, "stop_reason": reason, "message": message,
        }
        if validation_diagnostic is not None:
            payload["validation_diagnostic"] = validation_diagnostic
        if provider_diagnostic is not None:
            payload["provider_diagnostic"] = provider_diagnostic
        if error_category in {
            "timeout", "connection_error", "authentication_error", "rate_limited",
            "bad_request", "model_not_found", "structured_output_error",
            "server_error", "unknown_provider_error",
        }:
            payload["error_category"] = error_category
        print(json.dumps(payload, indent=2), file=sys.stderr)
        return 1

    request = args.request
    if not isinstance(request, str) or not request.strip() or len(request) > 4_000:
        return fail("plan_validation_failed", "Task must contain 1 to 4000 characters.")
    max_steps = args.max_steps
    if max_steps is None:
        try:
            max_steps = int(os.environ.get("AGENT_MAX_STEPS", "8"))
        except ValueError:
            return fail("plan_validation_failed", "AGENT_MAX_STEPS must be an integer.")
    if type(max_steps) is not int or not 1 <= max_steps <= MAX_PLAN_STEPS:
        return fail("plan_validation_failed", "Planner max-steps must be between 1 and 8.")
    try:
        planner = OpenAIPlanner.from_environment()
    except PlannerConfigurationError:
        return fail("planner_configuration_error", "Planner configuration is unavailable.")
    except ImportError:
        return fail("planner_configuration_error", "Planner dependencies are unavailable.")

    run_id = uuid4().hex
    started = time.perf_counter()

    sink = JsonlTelemetrySink()

    def emit_planner_event(event: AgentTelemetryEvent) -> None:
        try:
            sink.emit(event)
        except Exception:
            pass

    try:
        candidate = planner.plan(request, PlannerContext(max_steps))
        plan = validate_plan(candidate, request, max_steps=max_steps)
    except KeyboardInterrupt:
        return 130
    except PlannerCallError as exc:
        elapsed = max(0, round((time.perf_counter() - started) * 1000))
        emit_planner_event(AgentTelemetryEvent(
            event_type="planner_called", run_id=run_id, planner_run_id=run_id,
            timestamp=utc_timestamp(), planner_called=True, planner_latency_ms=elapsed,
        ))
        return fail(
            "planner_error", "The planner did not return a usable structured plan.",
            provider_diagnostic=exc.diagnostic.as_dict(),
            error_category=exc.category,
        )
    except PlanValidationError as exc:
        elapsed = max(0, round((time.perf_counter() - started) * 1000))
        now = utc_timestamp()
        emit_planner_event(AgentTelemetryEvent(
            event_type="planner_called", run_id=run_id, planner_run_id=run_id,
            timestamp=now, planner_called=True, planner_latency_ms=elapsed,
        ))
        emit_planner_event(AgentTelemetryEvent(
            event_type="plan_validation", run_id=run_id, planner_run_id=run_id,
            timestamp=utc_timestamp(), plan_validation_result="rejected",
            validation_stage=exc.diagnostic.validation_stage,
            validation_code=exc.diagnostic.validation_code,
            validation_reason_category=exc.diagnostic.reason_category,
            validation_step_index=exc.diagnostic.step_index,
            validation_step_kind=exc.diagnostic.step_kind,
            validation_field_name=exc.diagnostic.field_name,
            validation_field_path=exc.diagnostic.field_path,
            validation_error_type=exc.diagnostic.error_type,
        ))
        return fail(
            "plan_validation_failed", "The planner output failed local validation.",
            exc.diagnostic.as_dict(),
        )
    except Exception:
        elapsed = max(0, round((time.perf_counter() - started) * 1000))
        emit_planner_event(AgentTelemetryEvent(
            event_type="planner_called", run_id=run_id, planner_run_id=run_id,
            timestamp=utc_timestamp(), planner_called=True, planner_latency_ms=elapsed,
        ))
        emit_planner_event(AgentTelemetryEvent(
            event_type="plan_validation", run_id=run_id, planner_run_id=run_id,
            timestamp=utc_timestamp(), plan_validation_result="rejected",
        ))
        return fail("planner_error", "The planner did not return a usable structured plan.")

    elapsed = max(0, round((time.perf_counter() - started) * 1000))
    emit_planner_event(AgentTelemetryEvent(
        event_type="planner_called", run_id=run_id, planner_run_id=run_id,
        timestamp=utc_timestamp(), planner_called=True, planner_latency_ms=elapsed,
        plan_step_count=len(plan.steps),
    ))
    emit_planner_event(AgentTelemetryEvent(
        event_type="plan_validation", run_id=run_id, planner_run_id=run_id,
        timestamp=utc_timestamp(), plan_validation_result="accepted",
        plan_step_count=len(plan.steps),
    ))
    preview = {
        "success": True,
        "stop_reason": "dry_run" if args.dry_run else "awaiting_confirmation",
        "message": "A valid plan was created; no task content is shown.",
        "plan": [
            {"step_id": step.step_id, "kind": step.kind.value}
            for step in plan.steps
        ],
    }
    print(json.dumps(preview, indent=2, ensure_ascii=True))
    if args.dry_run:
        return 0

    print(
        "EXPERIMENTAL PLANNER RUN: a validated plan may execute safety-checked Windows actions. Continue? [y/N]",
        file=sys.stderr,
    )
    try:
        answer = input().strip().casefold()
    except (EOFError, KeyboardInterrupt):
        return 130
    if answer not in {"y", "yes"}:
        print("Planner run cancelled.", file=sys.stderr)
        return 1

    try:
        catalog = WindowsApplicationCatalog()
        decision_maker = JevDecisionMaker.from_environment(catalog)
        visual_provider = visual_provider_from_environment()
        computer = WindowsComputer(
            ObservationOptions(), app_catalog=catalog,
            capture_service=WindowsWindowCapture() if visual_provider else None,
            visual_provider=visual_provider,
        )
        agent = PlannerAgent(
            planner, computer, decision_maker,
            policy=AutonomousActionPolicy(catalog),
            confirmation=_PlannerCliConfirmation(),
            limits=AgentLimits(
                max_steps=max_steps,
                confidence_threshold=float(getattr(decision_maker, "min_confidence", 0.8)),
            ),
            max_planner_replans=args.max_planner_replans,
            telemetry_collector=sink,
            app_catalog=catalog,
        )
    except (ConfigurationError, VisualProviderConfigurationError):
        return fail("planner_configuration_error", "Decision or visual provider configuration is unavailable.")
    except KeyboardInterrupt:
        return 130
    try:
        result = agent.run(request, initial_plan=plan)
    except KeyboardInterrupt:
        return 130
    print(json.dumps(_planner_result_payload(result), indent=2, ensure_ascii=True))
    return 0 if result.success else 1


def _run_planner_provider_check() -> int:
    """Exercise only the planner's Responses/Pydantic provider adapter."""

    from agent.planner import (
        OpenAIPlanner, PlannerCallError, PlannerConfigurationError, PlannerContext,
        _safe_model_name,
    )

    try:
        planner = OpenAIPlanner.from_environment()
    except (PlannerConfigurationError, ImportError):
        print(json.dumps({
            "success": False,
            "provider_stage": "configuration",
            "error_category": "configuration_error",
            "structured_output_error": False,
        }, indent=2))
        return 1
    except Exception:
        # Client construction failures can contain local configuration detail;
        # keep this diagnostic bounded and never print the exception string.
        print(json.dumps({
            "success": False,
            "provider_stage": "client_initialization",
            "error_category": "unknown_provider_error",
            "structured_output_error": False,
        }, indent=2))
        return 1

    model_name = _safe_model_name(planner.model) or "unavailable"
    try:
        # This fixed one-step request probes the same OpenAI client,
        # responses.parse call, and ProviderPlan schema as normal planning.
        planner.plan("Return a one-step plan that only finishes.", PlannerContext(1))
    except PlannerCallError as exc:
        diagnostic = exc.diagnostic
        result: dict[str, object] = {
            "success": False,
            "model_name": model_name,
            "provider_stage": diagnostic.provider_stage,
            "error_category": exc.category,
            "structured_output_error": diagnostic.structured_output_error,
        }
        if diagnostic.http_status is not None:
            result["http_status"] = diagnostic.http_status
        if diagnostic.provider_error_code is not None:
            result["provider_error_code"] = diagnostic.provider_error_code
        if diagnostic.request_id is not None:
            result["request_id"] = diagnostic.request_id
        print(json.dumps(result, indent=2))
        return 1
    except KeyboardInterrupt:
        raise
    except Exception:
        print(json.dumps({
            "success": False,
            "model_name": model_name,
            "provider_stage": "request",
            "error_category": "unknown_provider_error",
            "structured_output_error": False,
        }, indent=2))
        return 1

    print(json.dumps({
        "success": True,
        "model_name": model_name,
        "parsed_schema": True,
    }, indent=2))
    return 0


class _PlannerCliConfirmation:
    """A content-free terminal confirmation for consequential graph actions."""

    def confirm(self, action, reason: str) -> bool:
        del action, reason
        print(
            "This action requires separate safety confirmation. Continue? [y/N]",
            file=sys.stderr,
        )
        try:
            return input().strip().casefold() in {"y", "yes"}
        except EOFError:
            return False


def main(argv: Sequence[str] | None = None) -> int:
    # A project-local .env is convenient for development. Existing process
    # variables remain authoritative, and python-dotenv does not print values.
    load_dotenv(
        dotenv_path=Path(__file__).resolve().parent / ".env",
        override=False, verbose=False,
    )
    parser = argparse.ArgumentParser(description="voice-jev: supervised Windows action debugging")
    commands = parser.add_subparsers(dest="command")
    commands.add_parser(
        "voice-transcribe",
        help="Record and display one utterance; never executes computer actions",
    )
    commands.add_parser(
        "run-agent-voice-debug",
        help="Transcribe one utterance, display it, then use the confirmed generic debug path",
    )
    commands.add_parser(
        "check-visual-provider",
        help="Check configured visual-provider metadata without uploading a screenshot",
    )
    observe = commands.add_parser("observe", help="Print the foreground window as JSON")
    observe.add_argument("--max-depth", type=int, default=6)
    observe.add_argument("--max-controls", type=int, default=100)
    observe.add_argument("--max-nodes", type=int, default=500)
    observe.add_argument("--max-text-length", type=int, default=200)
    observe.add_argument("--max-observed-text-length", type=int, default=500)
    observe.add_argument("--delay", type=float, default=0, help="Seconds to switch to the target window")
    open_app = commands.add_parser("open-app", help="Find and launch a trusted installed application")
    open_app.add_argument("query", help="Installed application display name")
    press_key = commands.add_parser("press-key", help="Send one allowlisted key chord")
    press_key.add_argument("keys", help="enter, escape, tab, shift+tab, ctrl+a, ctrl+c")
    type_text = commands.add_parser("type-text", help="Replace the focused editable field's entire value")
    type_text.add_argument("text")
    debug_literal = commands.add_parser(
        "debug-type-literal",
        help="Explicitly type one literal into the manually focused foreground window",
    )
    debug_literal.add_argument("text")
    debug_literal.add_argument("--delay", type=float, default=5)
    inspect = commands.add_parser("inspect-and-click", help="Observe, choose an ID, and act on that snapshot")
    decide = commands.add_parser("decide", help="Observe and ask Jev for ONE action; never execute it")
    decide.add_argument("request")
    decide.add_argument("--max-controls", type=int, default=80, help="Bound the observed controls sent for a decision")
    decide.add_argument("--max-observed-text-length", type=int, default=500)
    decide.add_argument("--debug-context", action="store_true", help="Print the sanitized Jev request and selection counts to stderr")
    run_agent = commands.add_parser("run-agent", help="Run one bounded observe/decide/act request")
    run_agent.add_argument("request")
    run_agent.add_argument("--dry-run", action="store_true", help="Observe and propose one action without executing it")
    run_agent.add_argument("--debug", action="store_true", help="Print safe loop progress to stderr")
    run_agent.add_argument("--max-steps", type=int, default=None, help="Maximum actions/decisions (default: AGENT_MAX_STEPS or 8)")
    run_agent.add_argument("--max-controls", type=int, default=80, help="Bound controls sent to Jev")
    run_agent.add_argument("--max-observed-text-length", type=int, default=500)
    for command in (press_key, type_text, inspect, decide):
        command.add_argument("--delay", type=float, default=5, help="Seconds to switch to the target window (default: 5)")
    run_agent.add_argument("--delay", type=float, default=0, help="Seconds to switch to the target window")
    graph_agent = commands.add_parser(
        "run-agent-graph-debug",
        help="Experimentally run the bounded agent through explicit LangGraph nodes",
    )
    graph_agent.add_argument("request")
    graph_agent.add_argument("--dry-run", action="store_true", help="Propose one action without executing it")
    graph_agent.add_argument("--max-steps", type=int, default=None)
    graph_agent.add_argument("--max-replans", type=int, default=1)
    graph_agent.add_argument("--max-controls", type=int, default=80)
    graph_agent.add_argument("--max-observed-text-length", type=int, default=500)
    graph_agent.add_argument("--delay", type=float, default=0)
    eval_agent = commands.add_parser(
        "eval-agent", help="Run deterministic offline evaluations of the LangGraph runtime",
    )
    eval_agent.add_argument("--json", action="store_true", help="Print the full evaluation report as JSON")
    planner_agent = commands.add_parser(
        "run-agent-planner-debug",
        help="Experimentally plan a task, then execute typed subgoals through LangGraph and Jev",
    )
    planner_agent.add_argument("request")
    planner_agent.add_argument("--dry-run", action="store_true", help="Plan and validate without Windows observation or actions")
    planner_agent.add_argument("--max-steps", type=int, default=None)
    planner_agent.add_argument("--max-planner-replans", type=int, choices=range(4), default=1)
    commands.add_parser(
        "planner-provider-check",
        help="Safely test planner OpenAI Responses and structured-output connectivity",
    )
    generic_agent = commands.add_parser(
        "run-agent-generic-debug",
        help="Experimentally activate one generic target with bounded optional search",
    )
    generic_agent.add_argument("request")
    generic_agent.add_argument("--delay", type=float, default=0)
    generic_agent.add_argument("--max-controls", type=int, default=80)
    generic_agent.add_argument("--max-observed-text-length", type=int, default=500)
    generic_agent.add_argument(
        "--save-generic-visual-debug-screenshot", type=Path, default=None,
        help=("Explicitly save exact masked PNGs sent for generic visual grounding; "
              "additional stages receive deterministic filename suffixes"),
    )
    hybrid_agent = commands.add_parser(
        "run-agent-hybrid-debug",
        help="Experimentally let Jev request directed vision; visual clicks always stop",
    )
    hybrid_agent.add_argument("request")
    hybrid_agent.add_argument("--max-steps", type=int, default=None)
    hybrid_agent.add_argument("--max-controls", type=int, default=80)
    hybrid_agent.add_argument("--max-observed-text-length", type=int, default=500)
    hybrid_agent.add_argument("--delay", type=float, default=0)
    hybrid_agent.add_argument(
        "--save-visual-debug-screenshot", type=Path, default=None,
        help="Explicitly save the first exact masked PNG sent to the visual provider",
    )
    hybrid_click_agent = commands.add_parser(
        "run-agent-hybrid-click-debug",
        help="Experimentally execute at most one safe visual click, observe, then stop",
    )
    hybrid_click_agent.add_argument("request")
    hybrid_click_agent.add_argument("--max-steps", type=int, default=None)
    hybrid_click_agent.add_argument("--max-controls", type=int, default=80)
    hybrid_click_agent.add_argument("--max-observed-text-length", type=int, default=500)
    hybrid_click_agent.add_argument("--delay", type=float, default=0)
    hybrid_click_agent.add_argument(
        "--save-visual-debug-screenshot", type=Path, default=None,
        help="Explicitly save the first exact masked PNG sent to the visual provider",
    )
    hybrid_type_agent = commands.add_parser(
        "run-agent-hybrid-type-debug",
        help="Experimentally execute one safe visual click and one literal type, then stop",
    )
    hybrid_type_agent.add_argument("request")
    hybrid_type_agent.add_argument("--max-steps", type=int, default=None)
    hybrid_type_agent.add_argument("--max-controls", type=int, default=80)
    hybrid_type_agent.add_argument("--max-observed-text-length", type=int, default=500)
    hybrid_type_agent.add_argument("--delay", type=float, default=0)
    hybrid_type_agent.add_argument(
        "--save-visual-debug-screenshot", type=Path, default=None,
        help="Explicitly save the first exact masked PNG sent to the visual provider",
    )
    hybrid_result_agent = commands.add_parser(
        "run-agent-hybrid-result-debug",
        help="Experimentally select one validated search result after bounded click and type",
    )
    hybrid_result_agent.add_argument("request")
    hybrid_result_agent.add_argument("--max-steps", type=int, default=None)
    hybrid_result_agent.add_argument("--max-controls", type=int, default=80)
    hybrid_result_agent.add_argument("--max-observed-text-length", type=int, default=500)
    hybrid_result_agent.add_argument("--delay", type=float, default=0)
    hybrid_result_agent.add_argument("--save-visual-debug-screenshot", type=Path, default=None)
    hybrid_result_agent.add_argument(
        "--save-result-debug-screenshot", type=Path, default=None,
        help="Save the exact masked PNG sent only for Phase-3 result grounding",
    )
    list_apps = commands.add_parser("list-apps", help="List trusted locally discovered applications")
    packaged_activation = commands.add_parser(
        "debug-packaged-activation",
        help="Measure one trusted packaged-app launch mechanism and foreground result",
    )
    packaged_activation.add_argument("query", help="Packaged app name from the trusted catalog")
    packaged_activation.add_argument(
        "--mechanism", required=True, choices=("shell", "activation-manager"),
        help="Run exactly one mechanism in this invocation",
    )
    packaged_activation.add_argument("--timeout-seconds", type=float, default=8.0)
    packaged_activation.add_argument("--poll-interval-ms", type=int, default=200)
    find_app = commands.add_parser("find-app", help="Rank installed applications for a name or request")
    find_app.add_argument("query")
    find_app.add_argument("--limit", type=int, default=10)
    inspect_app = commands.add_parser("inspect-app", help="Print a sanitized UIA capability summary")
    inspect_app.add_argument("--delay", type=float, default=5)
    inspect_app.add_argument("--max-controls", type=int, default=100)
    inspect_app.add_argument("--max-observed-text-length", type=int, default=500)
    inspect_hybrid = commands.add_parser("inspect-hybrid", help="Validate UIA, fallback, and foreground capture safely")
    inspect_hybrid.add_argument("--delay", type=float, default=5)
    inspect_hybrid.add_argument("--max-controls", type=int, default=100)
    inspect_hybrid.add_argument("--max-observed-text-length", type=int, default=500)
    inspect_hybrid.add_argument(
        "--save-debug-screenshot", type=Path, default=None,
        help="Explicitly save a new masked PNG; refuses to overwrite",
    )
    inspect_hybrid.add_argument(
        "--save-debug-overlay", type=Path, default=None,
        help="Explicitly save a new masked PNG with vN boxes; refuses to overwrite",
    )
    inspect_directed = commands.add_parser(
        "inspect-hybrid-directed",
        help="Experimentally ground only UI relevant to a bounded objective; never act",
    )
    inspect_directed.add_argument("--objective", required=True)
    inspect_directed.add_argument("--delay", type=float, default=5)
    inspect_directed.add_argument("--max-controls", type=int, default=100)
    inspect_directed.add_argument("--max-observed-text-length", type=int, default=500)
    inspect_directed.add_argument(
        "--save-debug-screenshot", type=Path, default=None,
        help="Explicitly save a new masked PNG; refuses to overwrite",
    )
    inspect_directed.add_argument(
        "--save-debug-overlay", type=Path, default=None,
        help="Explicitly save a new masked PNG with vN boxes; refuses to overwrite",
    )
    visual_diagnostic = commands.add_parser(
        "diagnose-visual-grounding",
        help="Repeat directed visual observation over one in-memory screenshot; never act",
    )
    visual_diagnostic.add_argument(
        "--objective", action="append", dest="objectives",
        help="Bounded diagnostic objective; repeat for multiple objectives",
    )
    visual_diagnostic.add_argument("--runs", type=int, default=3)
    visual_diagnostic.add_argument("--delay", type=float, default=5)
    visual_diagnostic.add_argument("--max-controls", type=int, default=100)
    visual_diagnostic.add_argument("--max-observed-text-length", type=int, default=500)
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "eval-agent":
        report = run_evaluations()
        if args.json:
            print(json.dumps(asdict(report), indent=2, ensure_ascii=True))
        else:
            print(format_eval_report(report))
        return 0 if report.failed == 0 else 1
    if args.command == "run-agent-planner-debug":
        return _run_planner_debug(args)
    if args.command == "planner-provider-check":
        return _run_planner_provider_check()
    voice_diagnostics: dict[str, object] | None = None
    voice_ready_at: float | None = None
    if args.command in {"voice-transcribe", "run-agent-voice-debug"}:
        try:
            voice_result = voice_input_from_environment().capture_and_transcribe()
        except KeyboardInterrupt:
            return 130
        except VoiceInputError as exc:
            error_payload: dict[str, object] = {
                "success": False,
                "error": exc.category,
                "message": exc.safe_message,
            }
            if exc.provider_diagnostics is not None:
                error_payload.update(exc.provider_diagnostics.as_dict())
            print(json.dumps(error_payload, indent=2, ensure_ascii=True), file=sys.stderr)
            return 1
        voice_diagnostics = voice_result.diagnostics()
        print(f"Transcript: {voice_result.transcript.text}", flush=True)
        if args.command == "voice-transcribe":
            print(json.dumps({"voice_input": voice_diagnostics}, indent=2, ensure_ascii=True))
            return 0
        # Feed only the minimally normalized transcript to the existing generic
        # debug command. Its confirmation and safety path remain authoritative.
        voice_ready_at = time.perf_counter()
        args.command = "run-agent-generic-debug"
        args.request = voice_result.transcript.text
        args.delay = 0
        args.max_controls = 80
        args.max_observed_text_length = 500
        args.save_generic_visual_debug_screenshot = None
    try:
        options = ObservationOptions(
            max_depth=getattr(args, "max_depth", 6), max_controls=getattr(args, "max_controls", 100),
            max_nodes=getattr(args, "max_nodes", 500), max_text_length=getattr(args, "max_text_length", 200),
            max_observed_text_length=getattr(args, "max_observed_text_length", 500),
        )
        delay = getattr(args, "delay", 0)
        if not 0 <= delay <= 60:
            raise ValueError("delay must be between 0 and 60 seconds")
        result_screenshot = getattr(args, "save_result_debug_screenshot", None)
        if result_screenshot is not None:
            result_path = result_screenshot.resolve()
            if (result_path.suffix.casefold() != ".png" or result_path.exists()
                    or not result_path.parent.is_dir()):
                raise ValueError(
                    "result debug screenshot must be a new .png in an existing directory",
                )
        generic_screenshot = getattr(args, "save_generic_visual_debug_screenshot", None)
        if generic_screenshot is not None:
            generic_path = generic_screenshot.resolve()
            if (generic_path.suffix.casefold() != ".png" or generic_path.exists()
                    or not generic_path.parent.is_dir()):
                raise ValueError(
                    "generic visual debug screenshot must be a new .png in an existing directory",
                )
        if args.command in {
            "run-agent", "run-agent-hybrid-debug", "run-agent-hybrid-click-debug",
            "run-agent-hybrid-type-debug",
            "run-agent-hybrid-result-debug", "run-agent-graph-debug",
        }:
            configured_steps = args.max_steps
            if configured_steps is None:
                raw_steps = os.environ.get("AGENT_MAX_STEPS", "8")
                try:
                    configured_steps = int(raw_steps)
                except ValueError as exc:
                    raise ValueError("AGENT_MAX_STEPS must be an integer") from exc
            if configured_steps <= 0:
                raise ValueError("max steps must be positive")
        if (args.command == "run-agent-graph-debug"
                and not 0 <= args.max_replans <= 10):
            raise ValueError("max replans must be between 0 and 10")
        if args.command in {
            "run-agent-hybrid-debug", "run-agent-hybrid-click-debug",
            "run-agent-hybrid-type-debug",
            "run-agent-hybrid-result-debug",
            "run-agent-generic-debug",
        }:
            raw_activation_timeout = os.environ.get(
                "OPEN_APP_ACTIVATION_TIMEOUT_SECONDS", "3",
            )
            try:
                open_app_activation_timeout = float(raw_activation_timeout)
            except ValueError as exc:
                raise ValueError(
                    "OPEN_APP_ACTIVATION_TIMEOUT_SECONDS must be a number",
                ) from exc
            if not 0 <= open_app_activation_timeout <= 15:
                raise ValueError(
                    "OPEN_APP_ACTIVATION_TIMEOUT_SECONDS must be between 0 and 15",
                )
        if args.command in {
            "run-agent-hybrid-debug", "run-agent-hybrid-click-debug",
            "run-agent-hybrid-type-debug", "run-agent-hybrid-result-debug",
        }:
            try:
                visual_readiness_timeout = float(os.environ.get(
                    "VISUAL_READINESS_TIMEOUT_SECONDS", "3",
                ))
                visual_readiness_poll_ms = int(os.environ.get(
                    "VISUAL_READINESS_POLL_INTERVAL_MS", "200",
                ))
                visual_readiness_options = VisualReadinessOptions(
                    visual_readiness_timeout, visual_readiness_poll_ms / 1000,
                )
                result_readiness_options = ResultReadinessOptions(
                    float(os.environ.get("RESULT_READINESS_TIMEOUT_SECONDS", "7")),
                    int(os.environ.get("RESULT_READINESS_POLL_INTERVAL_MS", "200")) / 1000,
                    int(os.environ.get("RESULT_SETTLE_QUIET_MS", "1200")),
                    int(os.environ.get("RESULT_TRANSITION_GRACE_MS", "2500")),
                    float(os.environ.get("RESULT_SIMPLIFICATION_RATIO_THRESHOLD", "0.75")),
                    int(os.environ.get("RESULT_SIMPLIFICATION_MIN_SIGNALS", "2")),
                )
            except ValueError as exc:
                raise ValueError(
                    "Visual readiness timeout/poll settings are invalid",
                ) from exc
    except ValueError as exc:
        parser.error(str(exc))
    catalog = WindowsApplicationCatalog() if args.command in {
        "list-apps", "find-app", "inspect-app", "inspect-hybrid", "inspect-hybrid-directed",
        "diagnose-visual-grounding",
        "open-app", "debug-packaged-activation", "decide", "run-agent", "run-agent-hybrid-debug",
        "run-agent-graph-debug",
        "run-agent-hybrid-click-debug",
        "run-agent-hybrid-type-debug",
        "run-agent-hybrid-result-debug",
        "run-agent-generic-debug",
    } else None
    if args.command == "check-visual-provider":
        report = check_visual_provider_from_environment()
        print(json.dumps(report, indent=2, ensure_ascii=True))
        return 0 if report.get("connectivity") is True and report.get("model_available") is not False else 1
    visual_provider = None
    if args.command in {
        "inspect-hybrid", "inspect-hybrid-directed", "decide", "run-agent",
        "run-agent-graph-debug",
        "run-agent-hybrid-debug", "diagnose-visual-grounding",
        "run-agent-hybrid-click-debug",
        "run-agent-hybrid-type-debug",
        "run-agent-hybrid-result-debug",
        "run-agent-generic-debug",
    }:
        try:
            visual_provider = visual_provider_from_environment()
        except VisualProviderConfigurationError as exc:
            if args.command == "run-agent-generic-debug":
                visual_configuration = _generic_visual_configuration(
                    None,
                    configured_provider=os.environ.get("VISUAL_PROVIDER"),
                    computer=WindowsComputer,
                    configuration_error=True,
                )
                print(json.dumps({
                    "success": False,
                    "stop_reason": "visual_configuration_error",
                    "message": "Visual provider configuration is unavailable.",
                    "visual_configuration": visual_configuration,
                }, indent=2, ensure_ascii=True), file=sys.stderr)
                return 1
            print(json.dumps({"success": False, "error": str(exc)}, indent=2), file=sys.stderr)
            return 1
    if args.command == "list-apps":
        assert catalog is not None
        print(json.dumps([_candidate_json(candidate) for candidate in catalog.discover()], indent=2, ensure_ascii=True))
        return 0
    if args.command == "find-app":
        assert catalog is not None
        if args.limit < 1 or args.limit > 50:
            parser.error("limit must be between 1 and 50")
        matches = catalog.find(args.query, args.limit)
        print(json.dumps([_candidate_json(match.candidate, match.score) for match in matches], indent=2, ensure_ascii=True))
        return 0 if matches else 1
    decision_maker = None
    if args.command == "decide":
        try:
            decision_maker = JevDecisionMaker.from_environment(catalog)
        except ConfigurationError as exc:
            failure = DecisionResult("error", None, None, str(exc), error="configuration_error")
            print(json.dumps(asdict(failure), indent=2))
            return 1
    if args.command in {
        "run-agent-hybrid-debug", "run-agent-hybrid-click-debug",
        "run-agent-hybrid-type-debug",
        "run-agent-hybrid-result-debug",
    }:
        if args.command == "run-agent-hybrid-result-debug":
            print("EXPERIMENTAL HYBRID RESULT SELECTION", file=sys.stderr)
            print("AT MOST TWO VISUAL CLICKS, ONE LITERAL TYPE, AND ONE QUERY SUBMIT MAY EXECUTE", file=sys.stderr)
            print("QUERY SUBMIT IS LIMITED TO ENTER AFTER VERIFIED SEARCH INPUT", file=sys.stderr)
            print("NO FURTHER CONTINUATION", file=sys.stderr)
        elif args.command == "run-agent-hybrid-type-debug":
            print("EXPERIMENTAL HYBRID CLICK + TYPE EXECUTION", file=sys.stderr)
            print("AT MOST ONE VISUAL CLICK AND ONE LITERAL TYPE MAY EXECUTE", file=sys.stderr)
            print("NO RESULT SELECTION", file=sys.stderr)
            print("NO CONSEQUENTIAL VISUAL ACTIONS", file=sys.stderr)
        elif args.command == "run-agent-hybrid-click-debug":
            print("EXPERIMENTAL HYBRID CLICK EXECUTION", file=sys.stderr)
            print("AT MOST ONE VISUAL CLICK MAY EXECUTE", file=sys.stderr)
            print("NO VISUAL TYPING", file=sys.stderr)
            print("NO CONSEQUENTIAL VISUAL ACTIONS", file=sys.stderr)
        else:
            print("EXPERIMENTAL HYBRID DEBUG", file=sys.stderr)
            print("VISUAL EXECUTION: DISABLED", file=sys.stderr)
        if args.save_visual_debug_screenshot is not None:
            print(
                "WARNING: the explicitly saved masked screenshot may contain sensitive UI contents.",
                file=sys.stderr,
            )
    if args.command == "run-agent-generic-debug":
        print("EXPERIMENTAL GENERIC TARGET ACTIVATION", file=sys.stderr)
        print("CURRENT STATE DRIVES ACTION SELECTION", file=sys.stderr)
        print("BOUNDED SEARCH / BOUNDED ACTIVATION", file=sys.stderr)
        print("NO CONSEQUENTIAL ACTIONS", file=sys.stderr)
        print("STOPS AFTER TARGET ACTIVATION", file=sys.stderr)
    if args.command == "diagnose-visual-grounding":
        print("VISUAL DIAGNOSTIC ONLY / NO COMPUTER ACTIONS WILL EXECUTE", file=sys.stderr)
    if args.command == "debug-type-literal":
        print("EXPLICIT WINDOWS LITERAL INPUT DEBUG", file=sys.stderr)
        print("LITERAL TEXT ONLY / NO HOTKEY SYNTAX / NO CLIPBOARD / NO CONTINUATION", file=sys.stderr)
    if delay and args.command != "debug-type-literal":
        print(f"Switch to the target window within {delay:g} seconds.", file=sys.stderr)
        try:
            time.sleep(delay)
        except KeyboardInterrupt:
            return 130
    if args.command == "debug-type-literal":
        print(
            "WARNING: this will type the supplied literal into the current foreground window. Continue? [y/N]",
            file=sys.stderr,
        )
        try:
            answer = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            return 130
        if answer not in {"y", "yes"}:
            print("Literal input debug cancelled.", file=sys.stderr)
            return 1
        if delay:
            print(f"Focus the target window within {delay:g} seconds.", file=sys.stderr)
            try:
                time.sleep(delay)
            except KeyboardInterrupt:
                return 130
        try:
            diagnostic = debug_type_literal(args.text)
            print(json.dumps({"success": True, "diagnostic": asdict(diagnostic)}, indent=2))
            return 0
        except LiteralInputFailure as exc:
            print(json.dumps({
                "success": False, "error": "send_input_failed",
                "diagnostic": asdict(exc.diagnostic),
            }, indent=2))
            return 1
        except (ValueError, OSError, RuntimeError) as exc:
            print(json.dumps({"success": False, "error": type(exc).__name__}, indent=2))
            return 1
    if args.command in {
        "run-agent-hybrid-debug", "run-agent-hybrid-click-debug",
        "run-agent-hybrid-type-debug",
        "run-agent-hybrid-result-debug",
    }:
        if visual_provider is None:
            print(json.dumps({
                "success": False, "stop_reason": "configuration_error",
                "message": "Configure an observation-only visual provider for hybrid debug.",
            }, indent=2), file=sys.stderr)
            return 1
        assert catalog is not None
        try:
            directed_limit = directed_max_elements_from_environment()
            decision_maker = JevDecisionMaker.from_environment(
                catalog, grounding_max_candidates=directed_limit,
            )
        except (ConfigurationError, VisualProviderConfigurationError) as exc:
            print(json.dumps({
                "success": False, "stop_reason": "configuration_error", "message": str(exc),
            }, indent=2), file=sys.stderr)
            return 1
        computer = WindowsComputer(
            options, app_catalog=catalog, capture_service=WindowsWindowCapture(),
            visual_provider=visual_provider,
            retain_debug_capture=(args.save_visual_debug_screenshot is not None
                                  or getattr(args, "save_result_debug_screenshot", None) is not None),
            collect_provider_candidate_diagnostics=(
                args.command == "run-agent-hybrid-result-debug"
            ),
            visual_readiness_options=visual_readiness_options,
            result_readiness_options=result_readiness_options,
        )
        print((
            "WARNING: two visual clicks, one literal TypeAction, and one query-submit Enter may execute, followed by observation and stop. Continue? [y/N]"
            if args.command == "run-agent-hybrid-result-debug" else
            "WARNING: one visual click and one literal TypeAction may execute, followed by observation and stop. Continue? [y/N]"
            if args.command == "run-agent-hybrid-type-debug" else
            "WARNING: one safety-validated visual click may execute, followed by observation and stop. Continue? [y/N]"
            if args.command == "run-agent-hybrid-click-debug" else
            "WARNING: safe local/UIA actions may execute; visual clicks always stop. Continue? [y/N]"
        ), file=sys.stderr)
        try:
            answer = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            return 130
        if answer not in {"y", "yes"}:
            print("Hybrid debug run cancelled.", file=sys.stderr)
            return 1
        capture_saved = False

        def save_directed_capture(
            observation, capture,
        ) -> None:
            nonlocal capture_saved
            if capture_saved or args.save_visual_debug_screenshot is None:
                return
            fingerprint = observation.visual_request_fingerprint
            if fingerprint is None:
                save_debug_screenshot(capture, args.save_visual_debug_screenshot.resolve())
                print(
                    "VISUAL DEBUG SCREENSHOT SAVED: rejected readiness frame; NOT SENT TO PROVIDER.",
                    file=sys.stderr,
                )
                capture_saved = True
                return
            path = args.save_visual_debug_screenshot.resolve()
            save_debug_screenshot(capture, path)
            saved = path.read_bytes()
            import hashlib
            if (len(saved) != fingerprint.encoded_image_byte_length
                    or hashlib.sha256(saved).hexdigest() != fingerprint.screenshot_sha256):
                raise ValueError("Saved screenshot does not match the visual request fingerprint.")
            capture_saved = True
            print(
                "VISUAL DEBUG SCREENSHOT SAVED: exact masked provider input.",
                file=sys.stderr,
            )

        def save_result_capture(observation, capture) -> None:
            report = _save_exact_result_capture(
                observation, capture, args.save_result_debug_screenshot.resolve(),
            )
            print("RESULT GROUNDING SCREENSHOT: " + json.dumps(report, ensure_ascii=True),
                  file=sys.stderr)

        agent_type = (
            HybridResultDebugAgent
            if args.command == "run-agent-hybrid-result-debug" else
            HybridTypeDebugAgent
            if args.command == "run-agent-hybrid-type-debug" else
            HybridClickDebugAgent
            if args.command == "run-agent-hybrid-click-debug" else HybridDebugAgent
        )
        agent = agent_type(
            computer, decision_maker, policy=AutonomousActionPolicy(catalog),
            limits=AgentLimits(
                max_steps=configured_steps,
                confidence_threshold=float(decision_maker.min_confidence),
            ),
            open_app_activation_timeout_seconds=open_app_activation_timeout,
            directed_capture_callback=(
                save_directed_capture if args.save_visual_debug_screenshot is not None else None
            ),
            result_capture_callback=(
                save_result_capture
                if (args.command == "run-agent-hybrid-result-debug"
                    and args.save_result_debug_screenshot is not None) else None
            ),
        )
        try:
            result = agent.run(args.request)
        except KeyboardInterrupt:
            return 130
        print(json.dumps(asdict(result), indent=2, ensure_ascii=True))
        return 0 if result.success else 1
    if args.command == "run-agent-generic-debug":
        assert catalog is not None
        try:
            decision_maker = JevDecisionMaker.from_environment(catalog)
        except ConfigurationError as exc:
            print(json.dumps({
                "success": False, "stop_reason": "configuration_error", "message": str(exc),
            }, indent=2), file=sys.stderr)
            return 1
        computer = WindowsComputer(
            options, app_catalog=catalog,
            capture_service=WindowsWindowCapture() if visual_provider is not None else None,
            visual_provider=visual_provider,
            retain_debug_capture=(args.save_generic_visual_debug_screenshot is not None),
        )
        visual_configuration = _generic_visual_configuration(
            visual_provider,
            configured_provider=os.environ.get("VISUAL_PROVIDER"),
            computer=computer,
        )
        print(json.dumps({
            "visual_configuration": visual_configuration,
        }, indent=2, ensure_ascii=True), file=sys.stderr)
        capture_callback = None
        if args.save_generic_visual_debug_screenshot is not None:
            print(
                "WARNING: saved masked screenshots may contain sensitive UI contents; "
                "additional grounding stages use suffixed filenames.",
                file=sys.stderr,
            )
            output_base = args.save_generic_visual_debug_screenshot.resolve()
            stage_counts: Counter[str] = Counter()
            capture_call_count = 0

            def save_generic_capture(stage: str, observation):
                nonlocal capture_call_count
                capture = computer.take_debug_capture()
                if capture is None:
                    return GenericVisualDebugScreenshotDiagnostic(
                        stage, False, error="debug_capture_unavailable",
                    )
                try:
                    capture_call_count += 1
                    stage_name = {
                        "target-resolution": "target-resolution",
                        "target-resolution-after-type": "target-resolution-after-type",
                        "target-resolution-after-submit": "target-resolution-after-submit",
                        "query-field": "query-field",
                    }.get(stage, "visual-grounding")
                    stage_counts[stage_name] += 1
                    if capture_call_count == 1:
                        path = output_base
                    else:
                        occurrence = stage_counts[stage_name]
                        suffix = stage_name if occurrence == 1 else f"{stage_name}-{occurrence}"
                        path = output_base.with_name(
                            f"{output_base.stem}-{suffix}{output_base.suffix}",
                        )
                    report = _save_exact_generic_capture(
                        observation, capture, path, stage_name,
                    )
                    return report
                finally:
                    capture.discard()

        print(
            "WARNING: one trusted app launch, one query-field activation, one literal query, one query submit, "
            "and one final target activation may occur. "
            "The run stops after one post-action observation. Continue? [y/N]",
            file=sys.stderr,
        )
        try:
            answer = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            return 130
        if answer not in {"y", "yes"}:
            print("Generic target debug run cancelled.", file=sys.stderr)
            return 1
        agent = GenericTaskDebugAgent(
            computer, decision_maker, app_catalog=catalog,
            policy=GenericTargetActivationPolicy(catalog),
            budgets=GenericTaskBudgets(
                app_activation_timeout_seconds=open_app_activation_timeout,
            ),
            visual_debug_capture_callback=save_generic_capture if (
                args.save_generic_visual_debug_screenshot is not None
            ) else None,
        )
        try:
            handoff_to_agent_ms = (
                round((time.perf_counter() - voice_ready_at) * 1000)
                if voice_ready_at is not None else None
            )
            result = agent.run(args.request)
        except KeyboardInterrupt:
            return 130
        result_json = asdict(result)
        result_json["visual_configuration"] = visual_configuration
        if voice_diagnostics is not None:
            voice_diagnostics["handoff_to_agent_ms"] = handoff_to_agent_ms
            result_json["voice_input"] = voice_diagnostics
        print(json.dumps(result_json, indent=2, ensure_ascii=True))
        return 0 if result.success else 1
    if args.command == "run-agent-graph-debug":
        try:
            from agent.graph import GraphAgent
        except ImportError:
            print(json.dumps({
                "success": False,
                "stop_reason": "graph_dependency_unavailable",
                "message": "Install the project dependencies to use the experimental graph command.",
            }, indent=2), file=sys.stderr)
            return 1
        try:
            decision_maker = JevDecisionMaker.from_environment(catalog)
        except ConfigurationError:
            print(json.dumps({
                "success": False, "stop_reason": "decision_error",
                "message": "Decision provider configuration is unavailable.",
            }, indent=2), file=sys.stderr)
            return 1
        computer = WindowsComputer(
            options, app_catalog=catalog,
            capture_service=WindowsWindowCapture() if visual_provider else None,
            visual_provider=visual_provider,
        )
        if not args.dry_run:
            print(
                "EXPERIMENTAL LANGGRAPH RUN: safety-validated Windows actions may execute. Continue? [y/N]",
                file=sys.stderr,
            )
            try:
                answer = input().strip().lower()
            except (EOFError, KeyboardInterrupt):
                return 130
            if answer not in {"y", "yes"}:
                print("Graph run cancelled.", file=sys.stderr)
                return 1
        agent = GraphAgent(
            computer,
            decision_maker,
            policy=AutonomousActionPolicy(catalog),
            limits=AgentLimits(
                max_steps=configured_steps,
                confidence_threshold=float(getattr(decision_maker, "min_confidence", 0.8)),
            ),
            max_replans=args.max_replans,
            telemetry_collector=JsonlTelemetrySink(),
        )
        try:
            graph_result = agent.run(args.request, dry_run=args.dry_run)
        except KeyboardInterrupt:
            return 130
        print(json.dumps(_safe_graph_agent_result(graph_result), indent=2, ensure_ascii=True))
        return 0 if graph_result.result.success else 1
    if args.command == "run-agent":
        try:
            decision_maker = JevDecisionMaker.from_environment(catalog)
        except ConfigurationError as exc:
            failure = {"success": False, "stop_reason": "decision_error", "message": str(exc)}
            print(json.dumps(failure, indent=2))
            return 1

        computer = WindowsComputer(
            options, app_catalog=catalog,
            capture_service=WindowsWindowCapture() if visual_provider else None,
            visual_provider=visual_provider,
        )
        reporter = None
        if args.debug:
            def report(event: str, data: dict[str, object]) -> None:
                print(f"[agent] {event} {json.dumps(data, ensure_ascii=True)}", file=sys.stderr)

            reporter = report
        if not args.dry_run:
            print("WARNING: this runs safety-validated Windows actions from Jev. Continue? [y/N]", file=sys.stderr)
            try:
                answer = input().strip().lower()
            except (EOFError, KeyboardInterrupt):
                return 130
            if answer not in {"y", "yes"}:
                print("Agent run cancelled.", file=sys.stderr)
                return 1
        agent = Agent(
            computer,
            decision_maker,
            policy=AutonomousActionPolicy(catalog),
            limits=AgentLimits(
                max_steps=configured_steps,
                confidence_threshold=float(getattr(decision_maker, "min_confidence", 0.8)),
            ),
            reporter=reporter,
        )
        try:
            result = agent.run(args.request, dry_run=args.dry_run, debug=args.debug)
        except KeyboardInterrupt:
            return 130
        payload = _safe_agent_result(result) if args.debug else asdict(result)
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        return 0 if result.success else 1
    if args.command == "decide":
        # Use the read-only observer. This branch cannot instantiate an executor.
        assert decision_maker is not None
        observation = WindowsObserver(
            options, catalog,
            capture_service=WindowsWindowCapture() if visual_provider else None,
            visual_provider=visual_provider,
        ).observe()
        if args.debug_context:
            def show_context(context: dict[str, object]) -> None:
                print(json.dumps(context, indent=2, ensure_ascii=True), file=sys.stderr)

            decision = decision_maker.decide(args.request, observation, debug_context=show_context)
        else:
            decision = decision_maker.decide(args.request, observation)
        print(json.dumps(asdict(decision), indent=2, ensure_ascii=True))
        return 0 if decision.status == "ready" else 1
    if args.command == "observe":
        observation = WindowsObserver(options).observe()
        print(json.dumps(asdict(observation), indent=2, ensure_ascii=True))
        return 1 if observation.error else 0
    if args.command == "diagnose-visual-grounding":
        if visual_provider is None:
            print(json.dumps({"success": False, "error": "visual_provider_not_configured"}), file=sys.stderr)
            return 1
        if not 1 <= args.runs <= 10:
            parser.error("runs must be between 1 and 10")
        assert catalog is not None
        observer = WindowsObserver(
            options, catalog, capture_service=WindowsWindowCapture(),
            capture_without_provider=True, force_capture=True, retain_debug_capture=True,
        )
        observation = observer.observe()
        capture = observer.take_debug_capture()
        if capture is None or observation.screenshot is None:
            print(json.dumps({"success": False, "error": "screenshot_capture_failed"}), file=sys.stderr)
            return 1
        objectives = tuple(args.objectives or DEFAULT_DIAGNOSTIC_OBJECTIVES)
        try:
            report = diagnose_visual_grounding(
                visual_provider, capture, observation, objectives,
                runs=args.runs, max_elements=directed_max_elements_from_environment(),
            )
        except KeyboardInterrupt:
            return 130
        finally:
            capture.discard()
        print(json.dumps(report, indent=2, ensure_ascii=True))
        return 0
    if args.command == "inspect-app":
        assert catalog is not None
        observation = WindowsObserver(options, catalog).observe()
        redactor = Redactor()
        counts = Counter(control.control_type or "Unknown" for control in observation.elements)
        interactive = [control for control in observation.elements
                       if control.control_type in CLICK_TYPES and control.name and control.is_password is not True]
        candidate = catalog.resolve(observation.application_id) if observation.application_id else None
        focused = next((control for control in observation.elements if control.focused is True), None)
        summary = {
            "window": {
                "application_id": observation.application_id or None,
                "application_name": candidate.display_name if candidate else redactor.clean(observation.app_name),
                "process_id": observation.process_id,
                "package_identity_present": bool(observation.package_family_name),
                "title": redactor.clean(observation.window_title)[:160],
                "controls_observed": len(observation.elements),
            },
            "control_types": dict(sorted(counts.items())),
            "interactive_named_controls": [
                {"name": redactor.clean(control.name)[:120], "type": control.control_type,
                 "parent": redactor.clean(control.parent_name)[:120] or None}
                for control in interactive[:40]
            ],
            "focused_editable": bool(focused and focused.control_type in {"Edit", "Document"}),
            "focused_observed_text_present": bool(focused and focused.observed_text is not None),
            "focused_observed_text_length": len(focused.observed_text) if focused and focused.observed_text is not None else None,
            "truncated": observation.truncated,
            "inspection_errors": observation.inspection_errors,
            "error": observation.error,
        }
        print(json.dumps(summary, indent=2, ensure_ascii=True))
        return 1 if observation.error else 0
    if args.command in {"inspect-hybrid", "inspect-hybrid-directed"}:
        assert catalog is not None
        grounding = None
        if args.command == "inspect-hybrid-directed":
            try:
                grounding = bounded_grounding_request(
                    args.objective, directed_max_elements_from_environment(),
                )
            except ValueError as exc:
                parser.error(str(exc))
        retain = args.save_debug_screenshot is not None or args.save_debug_overlay is not None
        observer = WindowsObserver(
            options, catalog, capture_service=WindowsWindowCapture(),
            visual_provider=visual_provider, capture_without_provider=True,
            force_capture=True, retain_debug_capture=retain, visual_grounding=grounding,
        )
        observation = observer.observe()
        fallback = visual_fallback_policy(observation)
        capture = observer.take_debug_capture()
        screenshot_saved = False
        overlay_saved = False
        try:
            if args.save_debug_screenshot is not None:
                if capture is None:
                    raise ValueError("No screenshot was captured; debug file was not created.")
                save_debug_screenshot(capture, args.save_debug_screenshot.resolve())
                screenshot_saved = True
            if args.save_debug_overlay is not None:
                if capture is None:
                    raise ValueError("No screenshot was captured; debug overlay was not created.")
                save_debug_overlay(capture, observation.visual_elements, args.save_debug_overlay.resolve())
                overlay_saved = True
        except (ValueError, FileExistsError, OSError) as exc:
            print(json.dumps({"success": False, "error": str(exc)}, indent=2), file=sys.stderr)
            return 1
        finally:
            if capture is not None:
                capture.discard()
        metadata = observation.screenshot
        print(json.dumps({
            "window": {
                "application_id": observation.application_id or None,
                "process_id": observation.process_id,
                "title": Redactor().clean(observation.window_title)[:160],
                "bounds": asdict(metadata.window_bounds) if metadata else None,
            },
            "uia": {
                "controls": len(observation.elements),
                "useful_named_interactive_controls": fallback.useful_named_interactive_controls,
            },
            "vision_policy": {
                "fallback_required": fallback.required, "reason": fallback.reason,
            },
            "screenshot": {
                "captured": metadata is not None,
                "dimensions": None if metadata is None else [metadata.pixel_width, metadata.pixel_height],
                "capture_bounds": asdict(metadata.capture_bounds) if metadata else None,
                "dpi": None if metadata is None else [metadata.dpi_x, metadata.dpi_y],
                "scale": None if metadata is None else [metadata.scale_x, metadata.scale_y],
                "masked_regions": 0 if metadata is None else metadata.masked_regions,
                "saved": screenshot_saved,
                "overlay_saved": overlay_saved,
            },
            "visual_provider_available": visual_provider is not None,
            "visual_provider": observation.visual_provider or (
                getattr(visual_provider, "name", None) if visual_provider else None
            ),
            "visual_model": observation.visual_model or (
                getattr(visual_provider, "model", None) if visual_provider else None
            ),
            "provider_latency_ms": observation.visual_latency_ms,
            "visual_provider_call_count": observation.visual_provider_call_count,
            "visual_provider_attempts": [
                asdict(item) for item in observation.visual_provider_attempts
            ],
            "selected_visual_provider": observation.selected_visual_provider,
            "provider_failover_used": observation.provider_failover_used,
            "provider_failover_reason": observation.provider_failover_reason,
            "timing": {
                "screenshot_capture_ms": observation.screenshot_capture_ms,
                "request_build_ms": observation.visual_request_build_ms,
                "provider_latency_ms": observation.visual_latency_ms,
                "response_parse_ms": observation.visual_response_parse_ms,
                "total_visual_observation_ms": observation.visual_total_observation_ms,
            },
            "requested_max_elements": observation.visual_requested_max_elements,
            "returned_visual_elements": observation.visual_returned_elements,
            "directed_grounding": observation.visual_directed_grounding,
            "visual_grounding_status": observation.visual_grounding_status,
            "visual_pipeline": (
                asdict(observation.visual_pipeline)
                if observation.visual_pipeline is not None else None
            ),
            "visual_rejection_summary": dict(observation.visual_rejection_summary),
            "visual_request_fingerprint": (
                asdict(observation.visual_request_fingerprint)
                if observation.visual_request_fingerprint is not None else None
            ),
            "capture_diagnostics": (
                asdict(observation.capture_diagnostics)
                if observation.capture_diagnostics is not None else None
            ),
            "usage": dict(observation.visual_usage),
            "pricing_class": observation.visual_pricing_class or (
                value if visual_provider and isinstance(
                    value := getattr(visual_provider, "pricing_class", None), str,
                ) else None
            ),
            "visual_provider_error": (
                asdict(observation.visual_provider_error)
                if observation.visual_provider_error is not None else None
            ),
            "visual_controls": [
                {"id": item.id, "label": Redactor().clean(item.label)[:160],
                 "role": item.role, "confidence": item.confidence,
                 "clickable": item.clickable, "parent": Redactor().clean(item.parent)[:120] or None,
                 "rect": asdict(item.rectangle)}
                for item in observation.visual_elements
            ],
            "error": observation.error,
        }, indent=2, ensure_ascii=True))
        return 1 if observation.error else 0
    if args.command == "debug-packaged-activation":
        if not 0.1 <= args.timeout_seconds <= 15:
            parser.error("timeout-seconds must be between 0.1 and 15")
        if not 10 <= args.poll_interval_ms <= 1000:
            parser.error("poll-interval-ms must be between 10 and 1000")
        assert catalog is not None
        try:
            diagnostic = run_packaged_activation_diagnostic(
                args.query, args.mechanism, catalog,
                computer_factory=lambda **kwargs: WindowsComputer(options, **kwargs),
                timeout_seconds=args.timeout_seconds,
                poll_interval_seconds=args.poll_interval_ms / 1000,
            )
        except KeyboardInterrupt:
            return 130
        print(json.dumps(asdict(diagnostic), indent=2, ensure_ascii=True))
        return 0 if diagnostic.success else 1
    computer = WindowsComputer(options, app_catalog=catalog)
    if args.command == "open-app":
        assert catalog is not None
        matches = catalog.find(args.query, 5)
        if not matches or (len(matches) > 1 and matches[0].score == matches[1].score):
            print(json.dumps({"success": False, "error": "no_unique_application_match",
                              "candidates": [_candidate_json(match.candidate, match.score) for match in matches]}, indent=2))
            return 1
        result = computer.execute(OpenAppAction(matches[0].candidate.id))
    else:
        observation = computer.observe()
        if observation.error:
            print(json.dumps(asdict(observation), indent=2))
            return 1
        if args.command == "inspect-and-click":
            print(json.dumps(asdict(observation), indent=2, ensure_ascii=True))
            print("Return here and enter a control ID from THIS observation (blank cancels):", file=sys.stderr)
            try:
                control_id = input().strip()
            except (EOFError, KeyboardInterrupt):
                return 1
            if not control_id:
                return 1
            print(f"Switch back to the SAME target window within {delay:g} seconds.", file=sys.stderr)
            time.sleep(delay)
            result = computer.execute(ClickAction(control_id), observation)
        elif args.command == "type-text":
            result = computer.execute(TypeAction(args.text), observation)
        else:
            result = computer.execute(PressKeyAction(tuple(args.keys.split("+"))), observation)
    print(json.dumps(asdict(result), indent=2, ensure_ascii=True))
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
