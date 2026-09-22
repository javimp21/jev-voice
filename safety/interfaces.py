"""Policy decisions must precede any future action execution."""

from dataclasses import dataclass
from typing import Literal, Protocol

from computer.actions import Action
from computer.models import Observation


@dataclass(frozen=True, slots=True)
class SafetyDecision:
    disposition: Literal["allow", "deny", "confirm"]
    reason: str


class ActionPolicy(Protocol):
    def validate(self, action: Action, observation: Observation | None) -> SafetyDecision:
        """Validate the action against current UI state and confirmation rules."""
        ...


class Confirmation(Protocol):
    def confirm(self, action: Action, reason: str) -> bool:
        """Obtain explicit user consent; return False for refusal or cancellation."""
        ...
