import numpy as np

from hitl_pmp.environments.tworoutes.environment import TwoRoutesEnvironment
from hitl_pmp.environments.tworoutes.skill_provider import (
    TwoRoutesOracle,
    TwoRoutesSkillProvider,
)
from hitl_pmp.environments.tworoutes.tasks import TwoRoutesTasks
from hitl_pmp.planning.grounding import SkillGrounder


def provider(**kwargs: object) -> TwoRoutesSkillProvider:
    env = TwoRoutesEnvironment(route_a_start=1.0, competence_cap=1.0)
    env.hard_reset()
    return TwoRoutesSkillProvider(env=env, **kwargs)


def test_each_robot_skill_computes_the_action_the_environment_runs() -> None:
    skills = provider()
    env = skills.env
    for name in ("RouteA", "ResetA"):
        ground = skills.ground(name=name)
        params = skills.sample_params(ground_skill=ground, rng=np.random.default_rng(0))
        assert params.shape == (0,)
        env.take_action(
            action=skills.compute_action(
                ground_skill=ground, params=params, state=env.get_current_state()
            )
        )
        assert env.last_error is None
    assert env.flag(name="ready")


def test_the_human_reset_is_offered_unless_turned_off() -> None:
    offered = provider().human_cube_bin_reset_skill()
    assert offered is not None and offered.skill.name == "human_reset_b"
    assert provider().movables_reset_destination(ground_skill=offered) == "human_reset_b"
    assert provider().is_movables_reset_skill(ground_skill=offered)
    assert provider(offer_human_reset=False).human_cube_bin_reset_skill() is None


def test_the_symbolic_state_matches_the_environments() -> None:
    skills = provider()
    env = skills.env
    env.take_action(action=env.action_for(skill="RouteA"))
    atoms = SkillGrounder.abstract_state(
        state=env.get_current_state(), objects=skills.objects(), predicates=skills.predicates()
    )
    assert {a.predicate.name for a in atoms} == {"Done", "ViaA", "AfterA"}
    assert skills.ground(name="RouteA").add_effects <= atoms
    assert skills.types() == (TwoRoutesEnvironment.robot_type,)
    assert [s.name for s in skills.skills()] == ["RouteA", "RouteB", "ResetA"]


def test_the_oracle_takes_route_b_and_stops_once_done_or_spent() -> None:
    env = TwoRoutesEnvironment(route_b_start=1.0, competence_cap=1.0)
    env.hard_reset()
    oracle = TwoRoutesOracle(env=env)
    goal = TwoRoutesTasks(env=env).goal()
    first = oracle.get_labeled_action(state=env.get_current_state(), goal=goal)
    assert first.label == "RouteB"
    state = env.take_action(action=first.action)
    after = oracle.get_labeled_action(state=state, goal=goal)
    assert np.array_equal(after.action, env.noop_action())
