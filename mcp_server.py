"""Local stdio MCP server for the existing Windows computer stack."""

from __future__ import annotations

from typing import Literal

from mcp.server import MCPServer
from mcp.types import ToolAnnotations

from computer.mcp_tools import (
    ActionSummary, ComputerToolService, ObservationSummary,
)
from computer.windows import ObservationOptions
from computer.windows_actions import WindowsComputer
from computer.windows_apps import WindowsApplicationCatalog
from safety.policy import AutonomousActionPolicy


def _default_service() -> ComputerToolService:
    """Create UIA-only dependencies; no screenshot or remote provider is wired."""

    catalog = WindowsApplicationCatalog()
    policy = AutonomousActionPolicy(catalog)
    computer = WindowsComputer(
        ObservationOptions(),
        policy=policy,
        app_catalog=catalog,
    )
    return ComputerToolService(computer, policy, catalog)


def create_mcp_server(service: ComputerToolService | None = None) -> MCPServer:
    tools = service or _default_service()
    mcp = MCPServer("voice-jev")

    @mcp.tool(
        description="Read the foreground window through the existing UI Automation observer.",
        annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
    )
    def observe() -> ObservationSummary:
        """Return a bounded, redacted summary and snapshot ID; never return geometry or values."""

        return tools.observe()

    @mcp.tool(
        description="Open a uniquely matched application from the trusted local app catalog.",
    )
    def open_app(app_name: str) -> ActionSummary:
        """Resolve an app locally, then use the existing OpenAppAction safety path."""

        return tools.open_app(app_name)

    @mcp.tool(
        description="Click a visible UIA control ID from the exact current observation.",
    )
    def click(observation_id: str, target_id: str) -> ActionSummary:
        """Use the existing ClickAction, policy, and snapshot-bound executor."""

        return tools.click(observation_id, target_id)

    @mcp.tool(
        description="Set literal text in the safely focused editable UIA control.",
    )
    def type_text(observation_id: str, text: str) -> ActionSummary:
        """Use the existing TypeAction; text is never parsed as key syntax."""

        return tools.type_text(observation_id, text)

    @mcp.tool(
        description="Press one locally allowlisted key or key combination.",
    )
    def press_key(
        observation_id: str,
        key_combo: Literal["enter", "escape", "tab", "shift+tab", "ctrl+a", "ctrl+c"],
    ) -> ActionSummary:
        """Use the existing PressKeyAction and key allowlist."""

        return tools.press_key(observation_id, key_combo)

    return mcp


server = create_mcp_server()


def main() -> None:
    # MCP stdio uses stdout for protocol messages; do not print banners here.
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
