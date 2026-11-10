import pytest

from hitl_pmp.core.method.skill_provider import DomainContext
from hitl_pmp.environments.tossing3d.cli import Tossing3DCli
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DOracle, Tossing3DSkillProvider
from tests.practice_cost_fixtures import CostFixtures


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
    CostFixtures.dispatch(
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
    assert not method.pursues_goal_in_period(period=0)


def test_dp_cli_parses_repeatable_initial_competences():
    base = ["--env", "tossing3d", "--method", "dp", "--dp-budget", "188"]
    args = CostFixtures.parse(
        argv=[
            *base,
            *("--dp-initial-competence", "PickCube=0.99"),
            *("--dp-initial-competence", "MoveToTossLocationAndToss=0.1"),
        ]
    )
    assert args.dp_initial_competence == {"PickCube": 0.99, "MoveToTossLocationAndToss": 0.1}
    assert CostFixtures.parse(argv=base).dp_initial_competence == {}
    for bad in (["PickCube"], ["PickCube=high"], ["PickCube=0.5", "PickCube=0.6"]):
        with pytest.raises(SystemExit):
            CostFixtures.parse(
                argv=[*base, *[t for value in bad for t in ("--dp-initial-competence", value)]]
            )
    with pytest.raises(SystemExit):
        CostFixtures.parse(
            argv=[
                *base,
                *("--dp-initial-competence", "PickCube=0.5"),
                *("--dp-prior-calibration", "priors.json"),
            ]
        )


def test_dp_cli_passes_initial_competences_to_the_method(*, monkeypatch):
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
    CostFixtures.dispatch(
        argv=[
            *("--env", "tossing3d", "--method", "dp", "--dp-budget", "188"),
            *("--dp-initial-competence", "MoveToTossLocationAndToss=0.1"),
            *("--dp-initial-learning-rate-for", "MoveToTossLocationAndToss=0.001"),
        ]
    )
    assert captured[0].dp_initial_competence == {"MoveToTossLocationAndToss": 0.1}
    assert captured[0].dp_initial_learning_rate_for == {"MoveToTossLocationAndToss": 0.001}


def test_dp_cli_parses_repeatable_per_skill_learning_rates():
    base = ["--env", "tossing3d", "--method", "dp", "--dp-budget", "188"]
    args = CostFixtures.parse(
        argv=[*base, *("--dp-initial-learning-rate-for", "MoveToTossLocationAndToss=0.001")]
    )
    assert args.dp_initial_learning_rate_for == {"MoveToTossLocationAndToss": 0.001}
    assert CostFixtures.parse(argv=base).dp_initial_learning_rate_for == {}
    for bad in (["PickCube"], ["PickCube=fast"], ["PickCube=0.1", "PickCube=0.2"]):
        with pytest.raises(SystemExit):
            CostFixtures.parse(
                argv=[*base, *[t for v in bad for t in ("--dp-initial-learning-rate-for", v)]]
            )
    with pytest.raises(SystemExit):
        CostFixtures.parse(
            argv=[
                *base,
                *("--dp-initial-learning-rate-for", "PickCube=0.1"),
                *("--dp-prior-calibration", "priors.json"),
            ]
        )
