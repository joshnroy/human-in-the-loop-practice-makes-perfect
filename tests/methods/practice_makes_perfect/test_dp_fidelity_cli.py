import pytest

from hitl_pmp.cli import Cli
from hitl_pmp.core.method.skill_provider import DomainContext
from hitl_pmp.environments.tossing3d.cli import Tossing3DCli
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DOracle, Tossing3DSkillProvider


@pytest.mark.parametrize(
    ("flags", "enabled"),
    [
        ([], False),
        (["--dp-allow-grounding-substitution"], True),
        (["--no-dp-allow-grounding-substitution"], False),
    ],
)
@pytest.mark.parametrize(("unit", "schedule"), [("episodes", "once"), ("cost", "period")])
def test_dp_cli_separates_statistical_sharing_from_target_substitution(
    *, flags, enabled, unit, schedule, monkeypatch
):
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
            "--competence-sharing",
            "parent",
            "--practice-groundings",
            "all",
            *flags,
        ]
    )
    method = captured[0]
    assert method.dp_allow_grounding_substitution is enabled
    assert method.competence_sharing == "parent"
    assert method.dp_budget_unit == unit
    assert method.dp_allocation_schedule == schedule
    assert method.score_human_skills is True
    assert not method.pursues_goal_in_period(period=0)
