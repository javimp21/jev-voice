"""Non-operational repeated visual-grounding diagnostics over one screenshot."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from computer.models import Observation, Rect, VisualGroundingStatus
from computer.visual import (
    ScreenshotCapture, VisualObserver, bounded_grounding_request,
    deduplicate_visual_elements, deduplicate_visual_elements_within,
    validate_visual_candidates_detailed, visual_request_fingerprint,
    VisualProviderFailure,
)
from computer.visual_providers.common import redact_secrets


DEFAULT_DIAGNOSTIC_OBJECTIVES = (
    "Find the control used to search for music.",
    "Find the visible control used to search for playable content.",
    "Find the search field.",
    "Find the control where the user can type a song name.",
)


def _normalized(rect: Rect, width: int, height: int) -> dict[str, int]:
    return {
        "left": round(rect.left * 1000 / width),
        "top": round(rect.top * 1000 / height),
        "right": round(rect.right * 1000 / width),
        "bottom": round(rect.bottom * 1000 / height),
    }


def diagnose_visual_grounding(
    provider: VisualObserver, capture: ScreenshotCapture, window: Observation,
    objectives: tuple[str, ...] = DEFAULT_DIAGNOSTIC_OBJECTIVES, *, runs: int = 3,
    max_elements: int = 5,
) -> dict[str, Any]:
    """Run independent observation-only calls; caller owns and discards capture."""
    if not 1 <= runs <= 10:
        raise ValueError("runs must be between 1 and 10")
    requests = tuple(bounded_grounding_request(item, max_elements) for item in objectives)
    results: list[dict[str, Any]] = []
    for objective_index, grounding in enumerate(requests, 1):
        fingerprint = visual_request_fingerprint(provider, capture, grounding)
        for run in range(1, runs + 1):
            base: dict[str, Any] = {
                "objective_id": f"objective_{objective_index}", "run": run,
                "provider": str(getattr(provider, "name", ""))[:80],
                "model": str(getattr(provider, "model", ""))[:100],
                "request_fingerprint": asdict(fingerprint) if fingerprint else None,
            }
            try:
                observed = provider.observe(capture, window, "", grounding)
                validated, rejection = validate_visual_candidates_detailed(
                    observed.candidates, capture.metadata,
                )
                within = deduplicate_visual_elements_within(validated)
                final = deduplicate_visual_elements(window.elements, within, capture.metadata)
                parsed = observed.parsed_element_count
                parsed = len(observed.candidates) if parsed is None else parsed
                raw = observed.raw_element_count
                raw = len(observed.candidates) if raw is None else raw
                status = (
                    VisualGroundingStatus.SUCCESS_WITH_CANDIDATES if final
                    else VisualGroundingStatus.SUCCESS_EMPTY if parsed == 0
                    else VisualGroundingStatus.VALIDATION_EMPTY if not validated
                    else VisualGroundingStatus.DEDUP_EMPTY
                )
                base.update({
                    "latency_ms": observed.latency_ms,
                    "status": status,
                    "raw_element_count": raw,
                    "parsed_element_count": parsed,
                    "validated_element_count": len(validated),
                    "deduplicated_element_count": len(final),
                    "final_element_count": len(final),
                    "usage": dict(observed.usage),
                    "rejection_summary": rejection,
                    "candidates": [{
                        "label": redact_secrets(item.label)[:160],
                        "role": item.role[:40],
                        "bbox_normalized": _normalized(
                            item.rectangle, capture.metadata.pixel_width,
                            capture.metadata.pixel_height,
                        ),
                    } for item in final[:max_elements]],
                    "provider_error": None,
                })
            except VisualProviderFailure as exc:
                base.update({
                    "latency_ms": None,
                    "status": (
                        VisualGroundingStatus.PARSE_ERROR
                        if exc.code in {"malformed_response", "invalid_response"}
                        else VisualGroundingStatus.PROVIDER_ERROR
                    ),
                    "raw_element_count": None, "parsed_element_count": None,
                    "validated_element_count": None,
                    "deduplicated_element_count": None, "final_element_count": 0,
                    "usage": {}, "rejection_summary": {}, "candidates": [],
                    "provider_error": asdict(exc.diagnostic),
                })
            except Exception:
                failure = VisualProviderFailure("unknown_api_error")
                base.update({
                    "latency_ms": None, "status": VisualGroundingStatus.PROVIDER_ERROR,
                    "raw_element_count": None, "parsed_element_count": None,
                    "validated_element_count": None,
                    "deduplicated_element_count": None, "final_element_count": 0,
                    "usage": {}, "rejection_summary": {}, "candidates": [],
                    "provider_error": asdict(failure.diagnostic),
                })
            results.append(base)
    return {
        "diagnostic_only": True, "computer_actions_enabled": False,
        "screenshot_capture_count": 1,
        "capture_diagnostics": (
            asdict(capture.diagnostics) if capture.diagnostics is not None else None
        ),
        "objective_count": len(requests), "runs_per_objective": runs,
        "provider_call_count": len(results), "results": results,
    }
