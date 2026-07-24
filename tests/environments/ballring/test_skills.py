import numpy as np

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.environments.ballring.environment import BallRingEnvironment
from hitl_pmp.environments.ballring.predicates import (
    BALL_IN_CUP,
    BALL_NOT_IN_CUP,
    BALL_ON_FLOOR,
    BALL_ON_TABLE,
    CUP_ON_FLOOR,
    CUP_ON_TABLE,
    HAND_EMPTY,
    HOLDING_BALL,
    HOLDING_CUP,
    IS_REACHABLE_BALL,
    IS_REACHABLE_CUP,
    IS_REACHABLE_SURFACE,
)
from hitl_pmp.environments.ballring.skills import BallRingSkills
from hitl_pmp.planning.grounding import SkillGrounder

E = BallRingEnvironment
_PREDS = (
    BALL_ON_TABLE,
    BALL_ON_FLOOR,
    CUP_ON_TABLE,
    CUP_ON_FLOOR,
    HOLDING_BALL,
    HOLDING_CUP,
    HAND_EMPTY,
    IS_REACHABLE_SURFACE,
    IS_REACHABLE_BALL,
    IS_REACHABLE_CUP,
    BALL_IN_CUP,
    BALL_NOT_IN_CUP,
)


def test_all_skills_are_the_sixteen_named_operators() -> None:
    names = [s.name for s in BallRingSkills.all_skills()]
    assert len(names) == 16
    assert len(set(names)) == 16
    assert "PlaceCupWithBallOnTable" not in names  # deliberately absent, as in predicators
    assert "PlaceBallOnTable" in names  # the "impossible" skill


def test_place_on_table_skills_have_two_continuous_params_others_zero() -> None:
    for skill in BallRingSkills.all_skills():
        expected = 2 if skill.name in ("PlaceBallOnTable", "PlaceCupWithoutBallOnTable") else 0
        assert skill.param_dim == expected


def test_sample_params_ranges() -> None:
    rng = np.random.default_rng(0)
    place = GroundSkill(
        skill=BallRingSkills.PLACE_CUP_WITHOUT_BALL_ON_TABLE,
        objects=(E.robot, E.ball, E.cup, E().target_table()),
    )
    params = BallRingSkills.sample_params(ground_skill=place, rng=rng)
    assert params.shape == (2,)
    assert 0.0 <= params[0] <= 1.0
    assert 0.0 <= params[1] <= 2 * np.pi

    nav = GroundSkill(skill=BallRingSkills.NAVIGATE_TO_CUP, objects=(E.robot, E.cup))
    assert BallRingSkills.sample_params(ground_skill=nav, rng=rng).shape == (0,)


def test_navigate_produces_a_reachable_collision_free_pose() -> None:
    env = E()
    state = env.sample_initial_state(rng=np.random.default_rng(3))
    target = env.target_table()
    nav = GroundSkill(skill=BallRingSkills.NAVIGATE_TO_TABLE, objects=(env.robot, target))
    action = BallRingSkills.compute_action(
        ground_skill=nav, params=np.zeros(0), state=state, env=env
    )
    assert action[0] == 0.0  # navigation mode
    env.set_state(state=state)
    next_state = env.take_action(action=action)
    assert env.is_reachable(state=next_state, robot=env.robot, other=target) is True
    assert env.exists_robot_collision(state=next_state) is False


def test_place_cup_on_normal_table_lands_it_on_the_table() -> None:
    env = E()
    state = env.sample_initial_state(rng=np.random.default_rng(1))
    # A normal (non-sticky) table never makes a cup fall, so any in-disk placement
    # keeps CupOnTable. Put the robot within reach and hold the cup.
    normal = env.get_tables(state=state)[0]
    state.set(obj=env.cup, feature_name="held", feature_val=1.0)
    nav = BallRingSkills.navigate_action(state=state, env=env, target=normal)
    state.set(obj=env.robot, feature_name="x", feature_val=float(nav[3]))
    state.set(obj=env.robot, feature_name="y", feature_val=float(nav[4]))
    env.set_state(state=state)
    place = GroundSkill(
        skill=BallRingSkills.PLACE_CUP_WITHOUT_BALL_ON_TABLE,
        objects=(env.robot, env.ball, env.cup, normal),
    )
    action = BallRingSkills.compute_action(
        ground_skill=place, params=np.array([0.0, 0.0]), state=state, env=env
    )
    assert action[0] == 1.0 and action[1] == 3.0  # pick/place onto a table
    next_state = env.take_action(action=action)
    assert env.on_table(state=next_state, obj=env.cup, table=normal) is True


def test_place_bare_ball_on_table_is_impossible_it_always_falls() -> None:
    env = E()
    state = env.sample_initial_state(rng=np.random.default_rng(1))
    normal = env.get_tables(state=state)[0]
    state.set(obj=env.ball, feature_name="held", feature_val=1.0)
    nav = BallRingSkills.navigate_action(state=state, env=env, target=normal)
    state.set(obj=env.robot, feature_name="x", feature_val=float(nav[3]))
    state.set(obj=env.robot, feature_name="y", feature_val=float(nav[4]))
    env.set_state(state=state)
    place = GroundSkill(
        skill=BallRingSkills.PLACE_BALL_ON_TABLE,
        objects=(env.robot, env.ball, env.cup, normal),
    )
    action = BallRingSkills.compute_action(
        ground_skill=place, params=np.array([0.3, 1.0]), state=state, env=env
    )
    next_state = env.take_action(action=action)
    # PlaceBallOnTable's add-effect (BallOnTable) is never actually achieved.
    assert env.on_table(state=next_state, obj=env.ball, table=normal) is False
    assert env.on_floor(state=next_state, obj=env.ball) is True


def test_navigate_skills_are_always_applicable_in_the_initial_state() -> None:
    env = E()
    state = env.sample_initial_state(rng=np.random.default_rng(5))
    objects = env.all_objects()
    true_atoms = SkillGrounder.abstract_state(state=state, objects=objects, predicates=_PREDS)
    applicable = SkillGrounder.applicable_ground_skills(
        skills=BallRingSkills.all_skills(), objects=objects, true_atoms=true_atoms
    )
    names = {gs.skill.name for gs in applicable}
    # Navigation has empty preconditions, so every NavigateTo* grounds.
    assert {"NavigateToTable", "NavigateToBall", "NavigateToCup"} <= names
