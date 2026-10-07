"""Held-out execution stops at the native goal or common physical horizon."""

import pytest

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
    bridge = ScoreBridge(bridge=physical, budget=2)
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

    FullAgenticRunner.initial_files(workspace=tmp_path, bridge=Bridge())
    assert {p.name for p in tmp_path.iterdir()} == {
        "env_client.py",
        "robot_api.md",
        "initial_observation.json",
    }


def test_missing_controller_is_explicit_zero_step_evaluation(*, tmp_path):
    from hitl_pmp.full_agentic.runner import FullAgenticRunner

    records = FullAgenticRunner.evaluate(
        submission=tmp_path, output=tmp_path / "eval", count=2, settings=None
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
    prompt = FullAgenticRunner.task_prompt(costs=costs)
    assert '"value": 7.0' in prompt
    assert "both equal to 1 initially" not in prompt
    assert r"c_h(e) \equiv 1" not in prompt
