"""Held-out execution stops at the native goal or common physical horizon."""

import pytest

from hitl_pmp.agentic_runtime.sandbox import SandboxSettings
from hitl_pmp.full_agentic.domains import Tossing3DFullAgentic
from hitl_pmp.full_agentic.runner import ScoreBridge


class FakeBridge:
    def __init__(self, *, goal_after):
        self.calls = 0
        self.goal_after = goal_after
        self.env = self

    def backend(self):
        return self

    def check_goals(self):
        return self.calls >= self.goal_after

    def step(self, *, action):
        self.calls += 1
        return {"calls": self.calls}


@pytest.mark.parametrize(("goal_after", "expected_solved"), [(2, True), (20, False)])
def test_no_physics_after_goal_or_horizon(*, goal_after, expected_solved):
    physical = FakeBridge(goal_after=goal_after)
    bridge = ScoreBridge(bridge=physical, budget=2, check_goal=physical.check_goals)
    for _ in range(2):
        bridge.step(action=[0] * 18)
    with pytest.raises(RuntimeError, match="episode_finished"):
        bridge.step(action=[0] * 18)
    assert physical.calls == bridge.clock.steps == 2
    assert bridge.success == expected_solved


def test_initial_files_supply_only_interface_and_observation(*, tmp_path):
    from hitl_pmp.full_agentic.runner import FullAgenticRunner

    class Bridge:
        def observe(self):
            return {"objects": []}

    FullAgenticRunner.initial_files(
        workspace=tmp_path, bridge=Bridge(), domain=Tossing3DFullAgentic
    )
    assert {p.name for p in tmp_path.iterdir()} == {
        "env_client.py",
        "robot_api.md",
        "initial_observation.json",
    }


def test_missing_controller_is_explicit_zero_step_evaluation(*, tmp_path):
    from hitl_pmp.full_agentic.runner import FullAgenticRunner

    records = FullAgenticRunner.evaluate(
        submission=tmp_path,
        output=tmp_path / "eval",
        count=2,
        settings=None,
        domain=Tossing3DFullAgentic,
    )
    assert len(records) == 2
    assert all(
        not r["solved"] and r["steps"] == 0 and r["error"] == "missing_controller" for r in records
    )


def test_prompt_reports_configured_prices_without_stale_unit_equations():
    from hitl_pmp.core.practice_costs import ChargeFunction, HumanCharge, PracticeCosts
    from hitl_pmp.full_agentic.runner import FullAgenticRunner

    costs = PracticeCosts(
        robot_step=ChargeFunction(value=2),
        human_skills={"reset_cube_far": HumanCharge(cost=ChargeFunction(value=7))},
    )
    prompt = FullAgenticRunner.task_prompt(costs=costs, domain=Tossing3DFullAgentic)
    assert '"value": 7.0' in prompt
    assert "both equal to 1 initially" not in prompt
    assert r"c_h(e) \equiv 1" not in prompt


@pytest.mark.parametrize("human_weight", [1, 20])
def test_rendered_prompt_requests_logged_decision_summaries(*, human_weight):
    from hitl_pmp.core.practice_costs import PracticeCosts
    from hitl_pmp.full_agentic.runner import FullAgenticRunner

    prompt = FullAgenticRunner.task_prompt(
        costs=PracticeCosts(human_weight=human_weight), domain=Tossing3DFullAgentic
    )
    summary = next(line for line in prompt.splitlines() if '"Decision summary:"' in line)
    assert "Before each practice batch" in summary
    assert "switch strategy or choose to finish" in summary
    assert "ordinary response" in summary
    assert "chosen practice side and next action" in summary
    assert "expected benefit" in summary
    assert "robot/human cost tradeoff" in summary
    assert "observations" in summary
    assert "reason for any switch or stop" in summary
    assert "uncertainty" in summary


class StopAtScoring(Exception):
    pass


@pytest.mark.parametrize("budget", [3000, 7])
def test_the_evaluation_bridge_allows_the_whole_evaluation_budget(*, tmp_path, monkeypatch, budget):
    """The bridge's own step limit must not end a held-out task before the scoring clock.

    The 2026-10-14 shelf rerun configured 3,000 evaluation steps, but `make_bridge` built
    the evaluation bridge at its 1,000-step default, so every unsolved task died at 1,000.
    """
    from hitl_pmp.full_agentic import runner
    from hitl_pmp.full_agentic.domains import FullAgenticDomain

    limits = []

    class Domain(FullAgenticDomain):
        make_environment = staticmethod(
            lambda *, options, human_seed: type("Env", (), {"close": lambda self: None})()
        )
        make_tasks = staticmethod(
            lambda *, env, seed: type("Tasks", (), {"sample_test_task": lambda self: None})()
        )
        goal_check = staticmethod(lambda *, env: lambda: False)

        @staticmethod
        def make_bridge(*, env, step_limit):
            limits.append(step_limit)
            return object()

    def score(**kwargs):
        raise StopAtScoring

    monkeypatch.setattr(runner, "ScoreBridge", score)
    (tmp_path / "approach.py").write_text("")
    with pytest.raises(StopAtScoring):
        runner.FullAgenticRunner.evaluate(
            submission=tmp_path,
            output=tmp_path / "eval",
            count=1,
            settings=SandboxSettings(image="i", robocode_checkout="/rc"),
            domain=Domain,
            budget=budget,
        )
    assert limits == [budget]


@pytest.mark.parametrize("budget", [500, 3000])
def test_the_evaluation_time_limit_scales_with_the_step_budget(*, budget):
    """Only the step budget may end a held-out task. The 60 s deadline the runner gives
    practice options (startup + execution) cut 22/37 unsolved shelf tasks short once the
    step limit was raised to 3,000, at 309-2,686 steps."""
    from hitl_pmp.full_agentic.runner import FullAgenticRunner

    practice = SandboxSettings(image="i", robocode_checkout="/rc", timeout_seconds=60)
    evaluation = FullAgenticRunner.evaluation_settings(settings=practice, budget=budget)
    seconds = FullAgenticRunner.EVALUATION_STARTUP_SECONDS
    seconds += FullAgenticRunner.EVALUATION_SECONDS_PER_STEP * budget
    assert evaluation.timeout_seconds == seconds
    assert evaluation.model_copy(update={"timeout_seconds": 60}) == practice


def test_evaluate_runs_under_the_budget_derived_time_limit(*, tmp_path, monkeypatch):
    from hitl_pmp.full_agentic import runner

    budgets = []

    def evaluation_settings(*, settings, budget):
        budgets.append(budget)
        raise StopAtScoring

    monkeypatch.setattr(
        runner.FullAgenticRunner, "evaluation_settings", staticmethod(evaluation_settings)
    )
    (tmp_path / "approach.py").write_text("")
    with pytest.raises(StopAtScoring):
        runner.FullAgenticRunner.evaluate(
            submission=tmp_path,
            output=tmp_path / "eval",
            count=1,
            settings=None,
            domain=Tossing3DFullAgentic,
            budget=3000,
        )
    assert budgets == [3000]
