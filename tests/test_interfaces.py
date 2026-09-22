"""Exercise the generic agent boundary without Windows dependencies."""

from unittest.mock import Mock

from agent.loop import Agent, AgentLimits


def test_empty_request_is_rejected_before_using_dependencies() -> None:
    computer = Mock()
    decision = Mock()
    result = Agent(computer, decision).run("  ")

    assert not result.success
    assert result.stop_reason == "invalid_request"
    computer.observe.assert_not_called()
    decision.decide.assert_not_called()


def test_agent_limits_reject_unbounded_configuration() -> None:
    try:
        AgentLimits(max_steps=0)
    except ValueError as exc:
        assert "max_steps" in str(exc)
    else:
        raise AssertionError("invalid limits were accepted")
