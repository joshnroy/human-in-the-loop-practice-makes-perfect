from pathlib import Path

import numpy as np
import pytest

from hitl_pmp.cli import Cli
from hitl_pmp.core.method.skill_provider import DomainContext
from hitl_pmp.core.method.types import LabeledAction
from hitl_pmp.environments.tossing3d.cli import Tossing3DCli
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DOracle, Tossing3DSkillProvider
from hitl_pmp.environments.tossing3d.types import Tossing3DState
from hitl_pmp.methods.belief_space.tossing3d_method import Tossing3DPomdpMethod
from hitl_pmp.methods.practice_makes_perfect.ees_method import EesMethod, _EesEpisode

METHODS = ("ees", "expectimax", "curtis")


def build_method(*, name: str, horizon: int) -> EesMethod:
    env = Tossing3DEnvironment(scene_bg=False)
    kwargs = {
        "env": env,
        "skill_provider": Tossing3DSkillProvider(env=env),
        "seed": 0,
        "goal_pursuit_horizon": horizon,
        "reproduce_predicators_explore_target_only": True,
    }
    if name == "ees":
        return EesMethod(**kwargs)
    return Tossing3DPomdpMethod(
        **kwargs,
        pomdp_num_particles=16,
        pomdp_solver="expectimax" if name == "expectimax" else "determinized_astar",
        pomdp_determinized_variant="curtis" if name == "curtis" else "algorithm3",
    )


@pytest.mark.parametrize("name", METHODS)
@pytest.mark.parametrize("horizon", (0, 5))
def test_goal_phase_dispatches_exploitation_then_practice(
    *, name: str, horizon: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    method = build_method(name=name, horizon=horizon)
    groundings = method.all_practice_groundings()
    pick = next(ground for ground in groundings if ground.skill.name == "PickCube")
    toss = next(ground for ground in groundings if ground.skill.name == "MoveToTossLocationAndToss")
    goal = frozenset(atom for atom in toss.add_effects if atom.predicate.name == "InBin")
    state = Tossing3DState(
        data={obj: np.zeros(obj.type.dim) for obj in method.objects()},
        abstract_atoms=tuple(
            (atom.predicate.name, tuple(obj.name for obj in atom.objects))
            for atom in pick.preconditions
        ),
    )
    phases = []
    exploration = []

    def plan_to(**kwargs):
        assert kwargs["goal"] == goal
        phases.append("goal")
        return [pick]

    def practice_plan(**kwargs):
        phases.append("practice")
        return [pick]

    def execute(*, ground_skill, state, explore):
        assert ground_skill == pick
        exploration.append(explore)
        return LabeledAction(action=np.zeros(5), label="PickCube"), None

    monkeypatch.setattr(EesMethod, "plan_to", staticmethod(plan_to))
    monkeypatch.setattr(EesMethod, "execute_ground_skill", staticmethod(execute))
    monkeypatch.setattr(_EesEpisode, "_practice_plan", staticmethod(practice_plan))
    episode = _EesEpisode(method=method, goal=goal, practicing=True)
    for _ in range(horizon + 2):
        episode.step(state=state)
    assert phases == ["goal"] * horizon + ["practice", "practice"]
    assert exploration == [False] * horizon + [True, True]
    assert episode._goal_phase_done


@pytest.mark.parametrize("name", METHODS)
def test_evaluation_ignores_training_horizon_and_does_not_collect_outcomes(
    *, name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    method = build_method(name=name, horizon=0)
    groundings = method.all_practice_groundings()
    pick = next(ground for ground in groundings if ground.skill.name == "PickCube")
    toss = next(ground for ground in groundings if ground.skill.name == "MoveToTossLocationAndToss")
    goal = frozenset(atom for atom in toss.add_effects if atom.predicate.name == "InBin")
    state = Tossing3DState(
        data={obj: np.zeros(obj.type.dim) for obj in method.objects()},
        abstract_atoms=tuple(
            (atom.predicate.name, tuple(obj.name for obj in atom.objects))
            for atom in pick.preconditions
        ),
    )

    def plan_to(**kwargs):
        assert not kwargs["practicing"]
        return [pick]

    def execute(*, ground_skill, state, explore):
        assert not explore
        return LabeledAction(action=np.zeros(5), label="PickCube"), None

    monkeypatch.setattr(EesMethod, "plan_to", staticmethod(plan_to))
    monkeypatch.setattr(EesMethod, "execute_ground_skill", staticmethod(execute))
    episode = _EesEpisode(method=method, goal=goal, practicing=False)
    for _ in range(3):
        episode.step(state=state)
    assert not episode._goal_phase_done
    assert not method._all_attempt_outcomes


@pytest.mark.parametrize("score_humans", (True, False))
def test_human_scoring_flag_reaches_ees_method_factory(
    *, score_humans: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    built = []

    def capture(*, method_factory, **kwargs):
        env = Tossing3DEnvironment(scene_bg=False)
        built.append(
            method_factory(
                DomainContext(
                    env=env,
                    skill_provider=Tossing3DSkillProvider(env=env),
                    oracle=Tossing3DOracle(env=env),
                )
            )
        )

    monkeypatch.setattr(Tossing3DCli, "run_method", staticmethod(capture))
    argv = [
        "--env",
        "tossing3d",
        "--method",
        "ees",
        "--ees-score-human-skills" if score_humans else "--no-ees-score-human-skills",
    ]
    Cli.main(argv=argv)
    assert len(built) == 1
    assert built[0].score_human_skills is score_humans


def test_ees_decision_logging_factory_path(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    built = []

    def capture(*, method_factory, **kwargs):
        env = Tossing3DEnvironment(scene_bg=False)
        built.append(
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
            "ees",
            "--record-ees-decisions",
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert built[0].ees_decision_log == tmp_path / "ees_decisions.jsonl"
    assert not built[0].ees_decision_log.exists()


def test_ees_decision_logging_requires_output_dir() -> None:
    from hitl_pmp.methods.practice_makes_perfect.cli import EesCli

    args = Cli.parse_args(argv=["--env", "tossing3d", "--method", "ees", "--record-ees-decisions"])
    with pytest.raises(ValueError, match="requires --output-dir"):
        EesCli.method_kwargs(args=args, ctx=None, draw_recorder=None)


def test_pomdp_rejects_ees_decision_logging() -> None:
    with pytest.raises(ValueError, match="is an EES flag"):
        Cli.main(
            argv=[
                "--env",
                "tossing3d",
                "--method",
                "pomdp",
                "--record-ees-decisions",
                "--pomdp-observation-probability-weight",
                "0.0",
            ]
        )
