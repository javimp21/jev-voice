"""Platform-neutral observation and execution contracts."""

from typing import Protocol

from computer.actions import Action
from computer.models import Observation
from computer.results import ActionResult


class Observer(Protocol):
    """Read-only capability, independent of future action execution."""

    def observe(self) -> Observation:
        """Read the current UI and assign observation-scoped element IDs."""
        ...


class Computer(Observer, Protocol):
    def execute(self, action: Action, observation: Observation | None = None) -> ActionResult:
        """Execute once; UI actions require the exact current observation object."""
        ...
