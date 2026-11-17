"""`--env racktoserving3d`: the 3D Serving scene behind exactly the chained toy's
symbolic interface (`tworoutes --both-resets --chained-route-b --robot-reset`). Only
whether a skill succeeded comes from physics; states, effects, costs and the learners'
reading of them are the toy's."""

import argparse
import importlib.util
from types import SimpleNamespace

import pytest

from hitl_pmp.cli import ENVIRONMENTS, STEP_PROTOCOL_ENVIRONMENTS
from hitl_pmp.core.method.types import Skill
from hitl_pmp.environments.racktoserving3d.cli import RackServingCli
from hitl_pmp.environments.racktoserving3d.environment import RackServingEnvironment
from hitl_pmp.environments.racktoserving3d.skill_provider import RackServingSkillProvider
from hitl_pmp.environments.racktoserving3d.step_protocol import RackServingStepProtocol
from hitl_pmp.environments.racktoserving3d.toy_environment import RackServingToyEnvironment
from hitl_pmp.environments.racktoserving3d.toy_interface import (
    GATHER,
    HUMAN_RESET,
    ROBOT_RESET,
    SWEEP3,
    SWEEP5,
    ServingToyTransitions,
)
from hitl_pmp.environments.tworoutes.costs import TwoRoutesCosts
from hitl_pmp.environments.tworoutes.environment import (
    HUMAN_RESET_B,
    ROUTE_A,
    ROUTE_B,
    ROUTE_B2,
    TwoRoutesEnvironment,
)
from hitl_pmp.environments.tworoutes.environment import (
    ROBOT_RESET as TOY_ROBOT_RESET,
)
from hitl_pmp.environments.tworoutes.skill_provider import TwoRoutesSkillProvider
from hitl_pmp.methods.belief_space.rewarded_routes_domain import RewardedRoutesPracticeDomain
from hitl_pmp.methods.practice_makes_perfect.cli import Tossing3DPomdpCli

SKILLS = {
    ROUTE_A: SWEEP3,
    ROUTE_B: GATHER,
    TOY_ROBOT_RESET: ROBOT_RESET,
    ROUTE_B2: SWEEP5,
    HUMAN_RESET_B: HUMAN_RESET,
}
PREDICATES = {
    "Ready": "FullRack",
    "Done": "Served",
    "ViaA": "ServedThree",
    "ViaB": "ServedFive",
    "AfterA": "AfterSweep3",
    "AfterB": "AfterGather",
    "After": "NeedsReset",
    "MidB": "Gathered",
    "AfterB2": "AfterSweep5",
}
# The toy environment's feature names, by predicate.
FLAGS = {
    "Ready": "ready",
    "Done": "done",
    "ViaA": "via_a",
    "ViaB": "via_b",
    "AfterA": "after_a",
    "AfterB": "after_b",
    "After": "after",
    "MidB": "mid_b",
    "AfterB2": "after_b2",
}


def _toy_provider() -> TwoRoutesSkillProvider:
    env = TwoRoutesEnvironment(
        both_resets=True, chained_route_b=True, route_a_reward=3.0, route_b_reward=4.0
    )
    return TwoRoutesSkillProvider(env=env, both_resets=True, robot_reset=True, chained_route_b=True)


def _provider(*, offer_human_reset: bool = True) -> RackServingSkillProvider:
    return RackServingSkillProvider(
        env=RackServingToyEnvironment(), offer_human_reset=offer_human_reset
    )


def _shape(*, skill: Skill, rename: dict[str, str]) -> tuple[frozenset[str], ...]:
    return tuple(
        frozenset(rename.get(a.predicate.name, a.predicate.name) for a in atoms)
        for atoms in (skill.preconditions, skill.add_effects, skill.delete_effects)
    )


def test_skill_names_map_one_to_one_in_the_toys_order():
    toy, ours = _toy_provider(), _provider()
    assert [SKILLS[s.name] for s in toy.skills()] == [s.name for s in ours.skills()]
    assert SKILLS[toy.human_skill().name] == ours.human_skill().name
    assert [p.name for p in ours.predicates()] == [PREDICATES[p.name] for p in toy.predicates()]


def test_preconditions_and_effects_are_the_toys_under_the_renaming():
    toy, ours = _toy_provider(), _provider()
    pairs = [
        *zip(toy.skills(), ours.skills(), strict=True),
        (toy.human_skill(), ours.human_skill()),
    ]
    for toy_skill, skill in pairs:
        assert _shape(skill=toy_skill, rename=PREDICATES) == _shape(skill=skill, rename={})
        assert len(skill.parameters) == 1


def test_failure_kept_predicates_and_rewards_are_the_toys():
    toy, ours = _toy_provider(), _provider()
    assert ours.failure_kept_predicates() == {
        SKILLS[name]: frozenset(PREDICATES[p] for p in kept)
        for name, kept in toy.failure_kept_predicates().items()
    }
    assert ours.goal_rewards() == {"ServedThree": 3.0, "ServedFive": 4.0}
    assert {PREDICATES[k]: v for k, v in toy.goal_rewards().items()} == ours.goal_rewards()


def test_from_provider_reads_the_toys_route_chain_and_learns_every_skill():
    toy = RewardedRoutesPracticeDomain.from_provider(provider=_toy_provider())
    ours = RewardedRoutesPracticeDomain.from_provider(provider=_provider())
    assert ours.route_rewards == tuple((SKILLS[n], r) for n, r in toy.route_rewards)
    assert ours.route_prerequisites == ((SWEEP5, (GATHER,)),)
    assert ours.route_prerequisites == tuple(
        (SKILLS[n], tuple(SKILLS[b] for b in before)) for n, before in toy.route_prerequisites
    )
    assert ours.belief_sources == tuple((SKILLS[n], s) for n, s in toy.belief_sources)
    assert {n for n, _ in ours.belief_sources} == set(SKILLS.values())
    assert ours.human_skill_names == frozenset({HUMAN_RESET})
    assert ours.failure_effect_skills == frozenset(SKILLS[n] for n in toy.failure_effect_skills)
    assert ours.failure_kept_predicates == tuple(
        sorted(
            (SKILLS[n], frozenset(PREDICATES[p] for p in kept))
            for n, kept in toy.failure_kept_predicates
        )
    )
    assert (ours.deployment_horizon, ours.competence_advances_every_try) == (
        toy.deployment_horizon,
        toy.competence_advances_every_try,
    )


def _toy_env(*, a: bool, b: bool, b2: bool, robot: bool, human: bool) -> TwoRoutesEnvironment:
    env = TwoRoutesEnvironment(
        both_resets=True,
        chained_route_b=True,
        competence_cap=1.0,
        route_a_start=float(a),
        route_a_rate=0.0,
        route_b_start=float(b),
        route_b_rate=0.0,
        route_b2_start=float(b2),
        route_b2_rate=0.0,
        robot_reset_competence=float(robot),
        human_reset_competence=float(human),
    )
    env.hard_reset()
    return env


def _toy_flags(*, env: TwoRoutesEnvironment) -> frozenset[str]:
    return frozenset(PREDICATES[p] for p, flag in FLAGS.items() if env.flag(name=flag))


SEQUENCES = [
    (ROUTE_A,),
    (ROUTE_B,),
    (ROUTE_B, ROUTE_B2),
    (ROUTE_A, TOY_ROBOT_RESET),
    (ROUTE_B, TOY_ROBOT_RESET),
    (ROUTE_B, ROUTE_B2, TOY_ROBOT_RESET),
    (ROUTE_A, HUMAN_RESET_B),
    (ROUTE_B, HUMAN_RESET_B),
    (ROUTE_B, ROUTE_B2, HUMAN_RESET_B),
    (ROUTE_B2,),
    (TOY_ROBOT_RESET,),
    (ROUTE_A, ROUTE_B),
]


@pytest.mark.parametrize("sequence", SEQUENCES)
@pytest.mark.parametrize("success", [True, False])
def test_every_transition_is_the_toys(*, sequence, success):
    env = _toy_env(a=success, b=success, b2=success, robot=success, human=success)
    flags = ServingToyTransitions.initial()
    assert flags == _toy_flags(env=env)
    for name in sequence:
        if name == HUMAN_RESET_B:
            env.reset_movables(destination=HUMAN_RESET_B)
        else:
            env.take_action(action=env.action_for(skill=name))
        skill = SKILLS[name]
        if ServingToyTransitions.applicable(skill=skill, flags=flags):
            flags = ServingToyTransitions.after(skill=skill, succeeded=success, flags=flags)
        assert flags == _toy_flags(env=env), (sequence, name)


def test_success_is_decided_by_the_measured_scene_alone():
    def ok(*, skill: str, **measured: object) -> bool:
        return ServingToyTransitions.succeeded(
            skill=skill,
            measured=SimpleNamespace(**{
                "plate_count": 0,
                "gathered": False,
                "full_rack": False,
                **measured,
            }),
        )

    assert [ok(skill=SWEEP3, plate_count=n) for n in range(6)] == [False] * 3 + [True] * 3
    assert [ok(skill=SWEEP5, plate_count=n) for n in range(6)] == [False] * 5 + [True]
    assert ok(skill=GATHER, gathered=True) and not ok(skill=GATHER, plate_count=5)
    for reset in (ROBOT_RESET, HUMAN_RESET):
        assert ok(skill=reset, full_rack=True) and not ok(skill=reset, plate_count=0)


class _Steps:
    def __init__(self) -> None:
        self.count = 0

    def control_steps(self) -> int:
        return self.count


@pytest.mark.parametrize("skill", [SWEEP3, GATHER, SWEEP5, ROBOT_RESET, None])
def test_only_the_robot_reset_is_priced_once_per_execution(*, skill):
    env = RackServingToyEnvironment(robot_reset_price=0.15)
    steps = _Steps()
    env._backend = steps  # type: ignore[assignment]
    env._toy_skill, env._toy_skill_start = skill, 7
    charged = []
    for count in range(7, 12):
        steps.count = count
        charged.append(TwoRoutesCosts.robot_step(context={"observation": env.cost_observation()}))
    assert charged == ([0.15, 0, 0, 0, 0] if skill == ROBOT_RESET else [0] * 5)


def test_racktoserving3d_is_the_toy_matching_environment():
    assert ENVIRONMENTS["racktoserving3d"] is RackServingCli
    assert STEP_PROTOCOL_ENVIRONMENTS["racktoserving3d"] is RackServingStepProtocol
    assert "racktoserving3d_toy" not in ENVIRONMENTS
    assert "racktoserving3d_toy" not in STEP_PROTOCOL_ENVIRONMENTS
    build = Tossing3DPomdpCli.DOMAINS["RackServingCli"]
    assert callable(build)
    assert isinstance(build(_provider()), RewardedRoutesPracticeDomain)
    assert RackServingStepProtocol.human_skill_names() == {HUMAN_RESET: HUMAN_RESET}
    assert _provider(offer_human_reset=False).human_cube_bin_reset_skills() == ()


def test_the_old_serving_interface_is_retired():
    assert importlib.util.find_spec("hitl_pmp.methods.belief_space.rack_serving_domain") is None
    for module in ("toy_cli", "toy_skill_provider", "toy_step_protocol"):
        assert importlib.util.find_spec(f"hitl_pmp.environments.racktoserving3d.{module}") is None
    names = {s.name for s in _provider().skills()} | {_provider().human_skill().name}
    assert names == {SWEEP3, GATHER, SWEEP5, ROBOT_RESET, HUMAN_RESET}
    parser = argparse.ArgumentParser()
    RackServingCli.add_arguments(parser=parser)
    flags = {action.dest for action in parser._actions}
    assert not flags & RackServingCli.PLANNER_FIELDS
    assert {"robot_reset_price", "sweep3_reward", "sweep5_reward"} <= flags


def test_the_cli_builds_the_toy_environment_on_a_slotted_start():
    parser = argparse.ArgumentParser()
    RackServingCli.add_arguments(parser=parser)
    args = parser.parse_args([])
    args.seed = 0
    problem = RackServingCli.build_problem(args=args, mode="training")
    assert isinstance(problem.env, RackServingToyEnvironment)
    assert problem.env.config.start_layout == "middle_three"
    assert problem.env.config.robot_batch_return
    assert isinstance(RackServingSkillProvider(env=problem.env).skills()[0], Skill)
    assert issubclass(RackServingToyEnvironment, RackServingEnvironment)
