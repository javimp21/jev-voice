"""Platform-independent decision contract; never executes the selected action."""

from collections.abc import Sequence
from typing import Protocol

from computer.models import Observation
from computer.results import ActionResult
from decision.models import DecisionResult


class DecisionMaker(Protocol):
    def decide(
        self, request: str, observation: Observation, history: Sequence[ActionResult] = (),
    ) -> DecisionResult:
        """Choose one next action or report completion from the current UI."""
        ...
