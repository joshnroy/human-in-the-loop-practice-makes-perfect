import numpy as np
import pytest

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.core.problem.environment.types import Object, State
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


def test_ignore_effects_match_predicators_nsrts_exactly() -> None:
    """Pins the whole table against
    ``ground_truth_models/ball_and_cup_sticky_table/nsrts.py``: only five of the
    sixteen operators declare ignore effects there, and every other one passes
    ``set()``. Getting this wrong is invisible in unit tests of individual skills but
    silently makes the symbolic model monotone, which lets the planner emit
    unexecutable plans (see test_ees_method.py's Ball-Ring plan-validity test)."""
    reachability = frozenset({IS_REACHABLE_SURFACE, IS_REACHABLE_BALL, IS_REACHABLE_CUP})
    expected = {
        "NavigateToTable": reachability,
        "NavigateToBall": reachability,
        "NavigateToCup": reachability,
        "PickBallFromTable": frozenset({BALL_IN_CUP}),
        "PlaceBallOnFloor": frozenset({BALL_IN_CUP, IS_REACHABLE_BALL}),
    }
    actual = {
        skill.name: skill.ignore_effects
        for skill in BallRingSkills.all_skills()
        if skill.ignore_effects
    }
    assert actual == expected


_PARAMETERIZED = frozenset({
    "PlaceBallOnTable",
    "PlaceCupWithoutBallOnTable",
    "PlaceBallOnFloor",
    "PlaceCupWithoutBallOnFloor",
    "PlaceCupWithBallOnFloor",
})


def test_only_the_five_placement_skills_have_continuous_params() -> None:
    """Every skill predicators samples a *placement point* for carries (u, theta)
    here: the two `place_on_table_sampler` skills and the three
    `place_on_floor_sampler` skills. Picks and navigation stay param_dim=0 -- see
    BallRingSkills' docstring for why that is sound under the deterministic config."""
    for skill in BallRingSkills.all_skills():
        assert skill.param_dim == (2 if skill.name in _PARAMETERIZED else 0)


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


def _hand_built_place_cup_state_and_skill() -> tuple[State, GroundSkill]:
    """A hand-constructed state with a single known table (specific radius / sticky /
    sticky-region offsets / center) and a cup of known radius, plus the ground
    PlaceCupWithoutBallOnTable skill over it. Concrete numbers chosen so the
    (u, theta) -> (x, y) conversion is exact (theta = 0)."""
    table = Object(name="sticky-table-0", type=E.table_type)
    # [x, y, radius, sticky, sticky_region_x_offset, sticky_region_y_offset,
    #  sticky_region_radius]
    table_features = np.array([0.5, 0.5, 0.2, 1.0, 0.03, -0.04, 0.07])
    # cup [x, y, radius, held]; radius 0.05 => size (diameter) 0.1.
    state = State(
        data={
            E.robot: np.array([0.0, 0.0]),
            E.ball: np.array([0.0, 0.0, 0.04, 0.0]),
            E.cup: np.array([0.0, 0.0, 0.05, 0.0]),
            table: table_features,
        }
    )
    place = GroundSkill(
        skill=BallRingSkills.PLACE_CUP_WITHOUT_BALL_ON_TABLE,
        objects=(E.robot, E.ball, E.cup, table),
    )
    return state, place


def test_oracle_sampler_input_pins_the_predicators_layout_and_converts_placement() -> None:
    """The exact oracle vector predicators emits for a PlaceCup*OnTable skill:
    [1.0, table_radius, sticky, sticky_region_x_offset, sticky_region_y_offset,
     sticky_region_radius, table_x, table_y, place_x, place_y]
    with place_x/place_y the *converted* (x, y) our (u, theta) params produce, not
    the raw (u, theta)."""
    state, place = _hand_built_place_cup_state_and_skill()
    # u = 0.5, theta = 0: dist = 0.5 * (table_radius 0.2 - size 0.1) = 0.05;
    # place_x = 0.5 + 0.05*cos(0) = 0.55, place_y = 0.5 + 0.05*sin(0) = 0.5.
    params = np.array([0.5, 0.0])
    row = BallRingSkills.oracle_sampler_input(ground_skill=place, state=state, params=params)
    assert row == pytest.approx([1.0, 0.2, 1.0, 0.03, -0.04, 0.07, 0.5, 0.5, 0.55, 0.5])


def test_oracle_place_coordinates_match_the_action_the_skill_commands() -> None:
    """The classifier is trained/scored on the placement coordinates, so they must be
    the same (x, y) the realized action actually commands."""
    state, place = _hand_built_place_cup_state_and_skill()
    params = np.array([0.37, 1.1])
    row = BallRingSkills.oracle_sampler_input(ground_skill=place, state=state, params=params)
    action = BallRingSkills._place_on_table_action(ground_skill=place, params=params, state=state)
    assert row is not None
    assert (row[-2], row[-1]) == pytest.approx((float(action[3]), float(action[4])))


def test_oracle_sampler_input_is_none_for_place_ball_on_table() -> None:
    """PlaceBallOnTable has 'Ball' not 'Cup' in its name -> "all" features, not oracle."""
    state, _ = _hand_built_place_cup_state_and_skill()
    table = Object(name="sticky-table-0", type=E.table_type)
    place_ball = GroundSkill(
        skill=BallRingSkills.PLACE_BALL_ON_TABLE, objects=(E.robot, E.ball, E.cup, table)
    )
    assert (
        BallRingSkills.oracle_sampler_input(
            ground_skill=place_ball, state=state, params=np.array([0.5, 0.0])
        )
        is None
    )


def test_oracle_sampler_input_is_none_for_a_pick_skill() -> None:
    table = Object(name="sticky-table-0", type=E.table_type)
    state, _ = _hand_built_place_cup_state_and_skill()
    pick = GroundSkill(
        skill=BallRingSkills.PICK_CUP_WITHOUT_BALL_FROM_TABLE,
        objects=(E.robot, E.cup, E.ball, table),
    )
    assert (
        BallRingSkills.oracle_sampler_input(ground_skill=pick, state=state, params=np.zeros(0))
        is None
    )


# --------------------------------------------------------- place-on-floor jitter


def _floor_place_cup_then_ball(
    *, env: BallRingEnvironment, state: State, rng: np.random.Generator
) -> State:
    """Floor-place the cup, then floor-place the ball, drawing each skill's
    (u, theta) from `rng` -- the sequence that used to put both objects at the
    *identical* room-center point. Drives the real
    compute_action -> env._simulate path, not a reimplementation of the sampler."""
    place_cup = GroundSkill(
        skill=BallRingSkills.PLACE_CUP_WITHOUT_BALL_ON_FLOOR,
        objects=(env.robot, env.ball, env.cup),
    )
    place_ball = GroundSkill(
        skill=BallRingSkills.PLACE_BALL_ON_FLOOR, objects=(env.robot, env.cup, env.ball)
    )
    current = state.model_copy(deep=True)
    for ground_skill, held in ((place_cup, env.cup), (place_ball, env.ball)):
        current.set(obj=held, feature_name="held", feature_val=1.0)
        env.set_state(state=current)
        action = BallRingSkills.compute_action(
            ground_skill=ground_skill,
            params=BallRingSkills.sample_params(ground_skill=ground_skill, rng=rng),
            state=current,
            env=env,
        )
        current = env.take_action(action=action)
    return current


def _predicted_ball_in_cup_rate(*, state: State, draws: int = 400_000) -> float:
    """The rate the *geometry alone* predicts, computed independently of the skill
    code: two points drawn as predicators' `place_on_floor_sampler` draws them
    (`dist ~ U(0, 2*radius)`, `theta ~ U(0, 2*pi)`, about the room center), counted
    as BallInCup when `dist(ball, cup) + ball_radius <= cup_radius`."""
    cup_r = state.get(obj=E.cup, feature_name="radius")
    ball_r = state.get(obj=E.ball, feature_name="radius")
    rng = np.random.default_rng(0)
    d_cup = rng.uniform(0.0, 2 * cup_r, draws)
    t_cup = rng.uniform(0.0, 2 * np.pi, draws)
    d_ball = rng.uniform(0.0, 2 * ball_r, draws)
    t_ball = rng.uniform(0.0, 2 * np.pi, draws)
    separation = np.hypot(
        d_ball * np.cos(t_ball) - d_cup * np.cos(t_cup),
        d_ball * np.sin(t_ball) - d_cup * np.sin(t_cup),
    )
    return float(np.mean(separation + ball_r <= cup_r))


def test_floor_placing_the_cup_then_the_ball_rarely_produces_ball_in_cup() -> None:
    """The gap this jitter closes. With both floor placements pinned to the exact
    room center, BallInCup held *every* time (the two centers coincide, and BallInCup
    only needs them within cup_radius - ball_radius ~ 0.0024). predicators scatters
    each placement in a disk of radius 2*radius(obj), which makes the coincidence
    rare -- the rate the actual geometry predicts, ~0.6%, not the ~100% we had."""
    env = E()
    state = env.sample_initial_state(rng=np.random.default_rng(11))
    predicted = _predicted_ball_in_cup_rate(state=state)
    assert predicted < 0.02  # the geometry itself says this is rare

    trials = 3000
    rng = np.random.default_rng(0)
    hits = sum(
        env.ball_in_cup(
            state=_floor_place_cup_then_ball(env=env, state=state, rng=rng),
            ball=env.ball,
            cup=env.cup,
        )
        for _ in range(trials)
    )
    measured = hits / trials
    assert measured < 0.05  # emphatically not the old 1.0
    # ...and it matches the independently-computed geometric prediction.
    assert measured == pytest.approx(predicted, abs=0.01)


def test_place_ball_on_floor_achieves_its_ball_not_in_cup_add_effect() -> None:
    """The corrupted competence signal: PlaceBallOnFloor's add-effects are
    BallNotInCup and BallOnFloor. Pinned to the room center after a floor-placed cup,
    BallNotInCup essentially never held, so the skill recorded a failure nearly every
    time. Checked through the Predicates, i.e. what the Method actually observes."""
    env = E()
    state = env.sample_initial_state(rng=np.random.default_rng(11))
    trials = 500
    rng = np.random.default_rng(1)
    achieved = 0
    for _ in range(trials):
        final = _floor_place_cup_then_ball(env=env, state=state, rng=rng)
        if BALL_NOT_IN_CUP.holds(final, (env.ball, env.cup)) and BALL_ON_FLOOR.holds(
            final, (env.ball,)
        ):
            achieved += 1
    assert achieved / trials > 0.95


def test_floor_placements_stay_on_the_floor_and_off_every_table() -> None:
    """The jitter disk is far smaller than the ring's clearance, so a floor place is
    still always a floor place -- CupOnFloor/BallOnFloor, never CupOnTable."""
    env = E()
    state = env.sample_initial_state(rng=np.random.default_rng(4))
    rng = np.random.default_rng(2)
    for _ in range(100):
        final = _floor_place_cup_then_ball(env=env, state=state, rng=rng)
        assert CUP_ON_FLOOR.holds(final, (env.cup,)) is True
        assert BALL_ON_FLOOR.holds(final, (env.ball,)) is True


def test_place_cup_with_ball_on_floor_keeps_ball_in_cup_under_jitter() -> None:
    """PlaceCupWithBallOnFloor's symbolic model declares no BallInCup delete *or*
    ignore effect, so the planner assumes the ball stays in the cup. The jitter must
    not break that: `_place_object` co-locates the contained ball with the cup, so one
    shared draw moves both and BallInCup survives."""
    env = E()
    state = env.sample_initial_state(rng=np.random.default_rng(8))
    place = GroundSkill(
        skill=BallRingSkills.PLACE_CUP_WITH_BALL_ON_FLOOR, objects=(env.robot, env.ball, env.cup)
    )
    rng = np.random.default_rng(3)
    for _ in range(100):
        current = state.model_copy(deep=True)
        # Hold the cup with the ball inside it (co-located, both held).
        for obj in (env.cup, env.ball):
            current.set(obj=obj, feature_name="held", feature_val=1.0)
        for feature in ("x", "y"):
            current.set(
                obj=env.ball,
                feature_name=feature,
                feature_val=current.get(obj=env.cup, feature_name=feature),
            )
        assert env.ball_in_cup(state=current, ball=env.ball, cup=env.cup) is True
        env.set_state(state=current)
        action = BallRingSkills.compute_action(
            ground_skill=place,
            params=BallRingSkills.sample_params(ground_skill=place, rng=rng),
            state=current,
            env=env,
        )
        final = env.take_action(action=action)
        assert BALL_IN_CUP.holds(final, (env.ball, env.cup)) is True
        assert CUP_ON_FLOOR.holds(final, (env.cup,)) is True
        assert BALL_ON_FLOOR.holds(final, (env.ball,)) is True


def test_place_on_floor_xy_reproduces_the_predicators_sampler_conversion() -> None:
    """(u, theta) -> room center + u*2*radius(objects[-1]) at angle theta, with
    objects[-1] the object being placed (predicators' `obj_to_place = objs[-1]`):
    the ball for PlaceBallOnFloor's [robot, cup, ball], the cup for
    PlaceCup*OnFloor's [robot, ball, cup]."""
    env = E()
    state = env.sample_initial_state(rng=np.random.default_rng(6))
    ball_r = state.get(obj=env.ball, feature_name="radius")
    cup_r = state.get(obj=env.cup, feature_name="radius")
    center = ((env.x_lb + env.x_ub) / 2, (env.y_lb + env.y_ub) / 2)

    place_ball = GroundSkill(
        skill=BallRingSkills.PLACE_BALL_ON_FLOOR, objects=(env.robot, env.cup, env.ball)
    )
    place_cup = GroundSkill(
        skill=BallRingSkills.PLACE_CUP_WITHOUT_BALL_ON_FLOOR,
        objects=(env.robot, env.ball, env.cup),
    )
    # u = 0 is the room center exactly (the old, unjittered behavior).
    assert BallRingSkills.place_on_floor_xy(
        ground_skill=place_ball, params=np.array([0.0, 1.3]), state=state, env=env
    ) == pytest.approx(center)
    # u = 1, theta = 0 is the disk's far edge, scaled by the *placed* object.
    assert BallRingSkills.place_on_floor_xy(
        ground_skill=place_ball, params=np.array([1.0, 0.0]), state=state, env=env
    ) == pytest.approx((center[0] + 2 * ball_r, center[1]))
    assert BallRingSkills.place_on_floor_xy(
        ground_skill=place_cup, params=np.array([1.0, 0.0]), state=state, env=env
    ) == pytest.approx((center[0] + 2 * cup_r, center[1]))


def test_place_ball_on_floor_commands_ball_only_but_place_cup_does_not() -> None:
    """predicators' `place_ball_on_floor_sampler` overrides `sample_arr[2] = 1.0` so
    the ball is placed *without* the cup it may be sitting in; the cup placements keep
    ball_only = 0 so a contained ball travels with the cup."""
    env = E()
    state = env.sample_initial_state(rng=np.random.default_rng(6))
    params = np.array([0.4, 2.0])
    ball_action = BallRingSkills.compute_action(
        ground_skill=GroundSkill(
            skill=BallRingSkills.PLACE_BALL_ON_FLOOR, objects=(env.robot, env.cup, env.ball)
        ),
        params=params,
        state=state,
        env=env,
    )
    cup_action = BallRingSkills.compute_action(
        ground_skill=GroundSkill(
            skill=BallRingSkills.PLACE_CUP_WITH_BALL_ON_FLOOR,
            objects=(env.robot, env.ball, env.cup),
        ),
        params=params,
        state=state,
        env=env,
    )
    assert (ball_action[0], ball_action[1], ball_action[2]) == (1.0, 0.0, 1.0)
    assert (cup_action[0], cup_action[1], cup_action[2]) == (1.0, 0.0, 0.0)


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
