"""Direct HTTP integration with https://docs.typesafe.ai/api.

Uses the documented typed Choice primitive, not chat or generated JSON.
No redirects, retries, raw-response logging, or configurable credential host.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from http.client import HTTPSConnection
import json
import math
import os
import re
from typing import Any, Protocol


class ConfigurationError(ValueError):
    """Safe-to-display configuration failure (never includes a secret)."""


class APIError(RuntimeError):
    """Safe-to-display transport failure."""

    def __init__(
        self, message: str, *, category: str = "api_error", http_status: int | None = None,
        provider_error_code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.http_status = http_status
        self.provider_error_code = provider_error_code


class InvalidResponse(ValueError):
    """The API response does not match the requested decision contract."""

    def __init__(self, message: str, *, category: str = "invalid_response") -> None:
        super().__init__(message)
        self.category = category


@dataclass(frozen=True, slots=True)
class JevSettings:
    api_key: str = field(repr=False)
    model: str = "jev-latest"
    min_confidence: float = 0.8

    def __post_init__(self) -> None:
        if not self.api_key.strip():
            raise ConfigurationError("Set TYPESAFE_API_KEY in the environment before using decide.")
        if not self.api_key.isascii() or any(c.isspace() for c in self.api_key):
            raise ConfigurationError("TYPESAFE_API_KEY contains invalid whitespace or characters.")
        if not re.fullmatch(r"jev-(?:latest|\d+\.\d+\.\d+)", self.model):
            raise ConfigurationError("JEV_MODEL must be jev-latest or a version such as jev-1.13.0.")
        if not math.isfinite(self.min_confidence) or not 0 < self.min_confidence <= 1:
            raise ConfigurationError("JEV_MIN_CONFIDENCE must be greater than 0 and at most 1.")

    @classmethod
    def from_environment(cls) -> JevSettings:
        try:
            threshold = float(os.environ.get("JEV_MIN_CONFIDENCE", "0.8"))
        except ValueError:
            raise ConfigurationError("JEV_MIN_CONFIDENCE must be a finite number in (0, 1].") from None
        return cls(os.environ.get("TYPESAFE_API_KEY", "").strip(),
                   os.environ.get("JEV_MODEL", "jev-latest"), threshold)


class DecisionClient(Protocol):
    def evaluate(self, payload: dict[str, Any]) -> object: ...


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise InvalidResponse("Duplicate response fields.")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise InvalidResponse("Non-finite JSON number.")


class TypeSafeHTTPClient:
    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def evaluate(self, payload: dict[str, Any]) -> object:
        connection = HTTPSConnection("api.typesafe.ai", timeout=15)
        try:
            body = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
            connection.request("POST", "/v1/systemone", body=body, headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json", "Accept": "application/json",
            })
            response = connection.getresponse()
            if response.status != 200:
                # Never include the response body; validation errors may echo input.
                category = {
                    401: "authentication_error", 403: "permission_error",
                    422: "invalid_request", 429: "rate_limited", 529: "provider_unavailable",
                }.get(response.status, "server_error" if response.status >= 500 else "api_error")
                raise APIError(
                    f"TypeSafe HTTP {response.status}; check credentials, access, or service availability.",
                    category=category, http_status=int(response.status),
                    provider_error_code=f"http_{int(response.status)}",
                )
            data = response.read(262_145)
            if len(data) > 262_144:
                raise InvalidResponse("Response exceeded the size limit.", category="response_size_limit")
            if not data:
                raise InvalidResponse("TypeSafe returned an empty response.", category="empty_response")
            return json.loads(data, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
        except (APIError, InvalidResponse):
            raise
        except (ValueError, UnicodeError):
            raise InvalidResponse(
                "TypeSafe returned unreadable JSON.", category="malformed_response",
            ) from None
        except Exception:
            raise APIError(
                "TypeSafe request failed or timed out; no action was selected.",
                category="transport_error",
            ) from None
        finally:
            connection.close()
