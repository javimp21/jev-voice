"""Offline regression checks for the deterministic agent evaluation harness."""

from dataclasses import replace

from agent.evals import DEFAULT_EVAL_SCENARIOS, format_eval_report, run_evaluations


def test_all_initial_eval_scenarios_pass_and_cover_required_cases() -> None:
    report = run_evaluations()

    assert report.total_scenarios == 10
    assert report.passed == 10
    assert report.failed == 0
    assert report.success_rate == 1.0
    assert {item.name for item in report.scenarios} == {
        "open_app_success", "safety_rejection", "executor_failure",
        "recoverable_incomplete_observation_then_success",
        "recoverable_observation_then_decision_failure", "replan_budget_exhausted",
        "duplicate_snapshot_failure", "finish_without_execution", "max_steps_reached",
        "invalid_decision",
    }
    assert report.stop_reason_distribution == {
        "decision_error": 2,
        "execution_failed": 1,
        "finished": 3,
        "max_steps": 1,
        "observation_failed": 2,
        "safety_rejected": 1,
    }
    assert report.average_steps == 1.5
    assert report.average_replans == 0.3


def test_eval_metrics_are_deterministic_and_transition_counts_are_reported() -> None:
    first = run_evaluations()
    second = run_evaluations()

    assert first == second
    assert first.transition_counts["OBSERVE->DECIDE"] == 10
    assert first.transition_counts["VERIFY_OR_CONTINUE->REPLAN"] == 3
    assert first.transition_counts["REPLAN->DECIDE"] == 3


def test_failed_eval_scenario_names_exact_mismatch() -> None:
    scenario = replace(DEFAULT_EVAL_SCENARIOS[0], expected_stop_reason="unexpected")
    report = run_evaluations((scenario,))

    assert report.failed == 1
    assert report.scenarios[0].name == "open_app_success"
    assert report.scenarios[0].passed is False
    assert report.scenarios[0].mismatches == ("stop_reason_mismatch",)
    rendered = format_eval_report(report)
    assert "FAIL open_app_success" in rendered
    assert "stop_reason_mismatch" in rendered


def test_eval_scenarios_are_data_driven_fakes_only() -> None:
    # The scenario inputs are typed snapshots and decisions; the harness owns
    # fake computer, policy, and decision implementations instead of adapters.
    assert all(scenario.fake_observations for scenario in DEFAULT_EVAL_SCENARIOS)
    assert all(scenario.max_steps > 0 for scenario in DEFAULT_EVAL_SCENARIOS)
    assert all(0 <= scenario.max_replans <= 10 for scenario in DEFAULT_EVAL_SCENARIOS)
    report = run_evaluations()
    assert report.passed == report.total_scenarios


def test_eval_cli_json_does_not_construct_live_runtime(monkeypatch, capsys) -> None:
    import main as cli

    def forbidden(*_args, **_kwargs):
        raise AssertionError("eval-agent must not construct a live Windows/provider runtime")

    monkeypatch.setattr(cli, "WindowsComputer", forbidden)
    monkeypatch.setattr(cli, "WindowsApplicationCatalog", forbidden)
    monkeypatch.setattr(cli, "visual_provider_from_environment", forbidden)
    monkeypatch.setattr(cli.JevDecisionMaker, "from_environment", forbidden)

    assert cli.main(["eval-agent", "--json"]) == 0
    report = __import__("json").loads(capsys.readouterr().out)
    assert report["total_scenarios"] == 10
    assert report["failed"] == 0
    assert "eval-only literal" not in str(report)
