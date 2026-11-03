import pytest

from hitl_pmp.cli import Cli
from hitl_pmp.core.method.skill_provider import DomainContext
from hitl_pmp.environments.tossing3d.cli import Tossing3DCli
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DOracle, Tossing3DSkillProvider


@pytest.mark.parametrize(("unit", "schedule"), [("episodes", "once"), ("cost", "period")])
def test_dp_cli_wires_budget_schedule_and_practice_groundings(*, unit, schedule, monkeypatch):
    captured = []

    def capture(*, method_factory, **kwargs):
        env = Tossing3DEnvironment(scene_bg=False)
        captured.append(
            method_factory(
                DomainContext(
                    env=env,
                    skill_provider=Tossing3DSkillProvider(env=env),
                    oracle=Tossing3DOracle(env=env),
                )
            )
        )

    monkeypatch.setattr(Tossing3DCli, "run_method", staticmethod(capture))
    Cli.main(
        argv=[
            "--env",
            "tossing3d",
            "--method",
            "dp",
            "--dp-budget",
            "2000",
            "--dp-budget-unit",
            unit,
            "--dp-allocation-schedule",
            schedule,
            "--practice-groundings",
            "all",
        ]
    )
    method = captured[0]
    assert method.practice_groundings == "all"
    assert method.dp_budget_unit == unit
    assert method.dp_allocation_schedule == schedule
    assert method.score_human_skills is True
    assert not method.pursues_goal_in_period(period=0)
