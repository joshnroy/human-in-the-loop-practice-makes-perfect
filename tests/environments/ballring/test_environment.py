import numpy as np
from gymnasium.spaces import Box

from hitl_pmp.core.problem.environment.types import Object, State
from hitl_pmp.environments.ballring.environment import BallRingEnvironment

E = BallRingEnvironment


def _table(
    *,
    name: str,
    x: float,
    y: float,
    radius: float = 0.1,
    sticky: float = 0.0,
    sox: float = 0.0,
    soy: float = 0.0,
    srr: float = 0.03,
) -> tuple[Object, np.ndarray]:
    return Object(name=name, type=E.table_type), np.array([x, y, radius, sticky, sox, soy, srr])


def _state(
    *,
    tables: list[tuple[Object, np.ndarray]],
    robot: tuple[float, float] = (0.5, 0.5),
    ball: tuple[float, float, float, float] = (0.1, 0.1, 0.02, 0.0),
    cup: tuple[float, float, float, float] = (0.9, 0.9, 0.03, 0.0),
) -> State:
    data: dict[Object, np.ndarray] = {obj: feats for obj, feats in tables}
    data[E.robot] = np.array(robot)
    data[E.ball] = np.array(ball)
    data[E.cup] = np.array(cup)
    return State(data=data)


# --- action space ---


def test_action_space_is_five_dimensional_box_with_paper_bounds() -> None:
    assert isinstance(E.action_space, Box)
    assert E.action_space.shape == (5,)
    assert np.allclose(E.action_space.low, [0.0, 0.0, 0.0, 0.0, 0.0])
    assert np.allclose(E.action_space.high, [1.0, 3.0, 1.0, 1.0, 1.0])


def test_get_valid_actions_is_empty_for_continuous_space() -> None:
    assert E().get_valid_actions() == []


# --- geometry helpers ---


def test_circle_contains_point_matches_predicators_formula() -> None:
    assert E.circle_contains_point(cx=0.0, cy=0.0, radius=1.0, px=0.5, py=0.5) is True
    assert E.circle_contains_point(cx=0.0, cy=0.0, radius=1.0, px=1.0, py=1.0) is False


def test_circle_contains_circle_requires_full_containment() -> None:
    assert E.circle_contains_circle(cx=0.0, cy=0.0, radius=1.0, ox=0.0, oy=0.0, oradius=0.5) is True
    # Center inside but rim pokes out -> not contained.
    assert not E.circle_contains_circle(cx=0.0, cy=0.0, radius=1.0, ox=0.8, oy=0.0, oradius=0.5)


# --- navigation ---


def test_navigation_moves_robot_to_commanded_position() -> None:
    env = E()
    env.set_state(
        state=_state(tables=[_table(name="normal-table-0", x=0.5, y=0.5)], robot=(0.2, 0.2))
    )
    next_state = env.take_action(action=np.array([0.0, 0.0, 0.0, 0.3, 0.4]))
    assert next_state.get(obj=E.robot, feature_name="x") == 0.3
    assert next_state.get(obj=E.robot, feature_name="y") == 0.4


def test_navigation_blocked_by_collision_leaves_robot_in_place() -> None:
    env = E()
    env.set_state(
        state=_state(tables=[_table(name="normal-table-0", x=0.5, y=0.5)], robot=(0.2, 0.2))
    )
    # Target (0.5, 0.5) is inside the table circle -> collision -> no move.
    next_state = env.take_action(action=np.array([0.0, 0.0, 0.0, 0.5, 0.5]))
    assert next_state.get(obj=E.robot, feature_name="x") == 0.2
    assert next_state.get(obj=E.robot, feature_name="y") == 0.2


# --- picking ---


def test_pick_ball_when_action_grasps_it_sets_held() -> None:
    env = E()
    env.set_state(
        state=_state(
            tables=[_table(name="normal-table-0", x=0.5, y=0.5)],
            robot=(0.15, 0.15),
            ball=(0.15, 0.15, 0.02, 0.0),
        )
    )
    next_state = env.take_action(action=np.array([1.0, 1.0, 0.0, 0.15, 0.15]))
    assert env.holding(state=next_state, obj=E.ball) is True


def test_pick_ball_that_action_misses_is_a_noop() -> None:
    env = E()
    env.set_state(
        state=_state(
            tables=[_table(name="normal-table-0", x=0.5, y=0.5)],
            ball=(0.15, 0.15, 0.02, 0.0),
        )
    )
    # (0.9, 0.9) is nowhere near the ball at (0.15, 0.15).
    next_state = env.take_action(action=np.array([1.0, 1.0, 0.0, 0.9, 0.9]))
    assert env.holding(state=next_state, obj=E.ball) is False


def test_pick_cup_with_ball_in_it_lifts_both() -> None:
    env = E()
    # Ball sitting in the cup on the floor (same position, both unheld).
    state = _state(
        tables=[_table(name="normal-table-0", x=0.5, y=0.5)],
        robot=(0.25, 0.25),
        ball=(0.25, 0.25, 0.02, 0.0),
        cup=(0.25, 0.25, 0.03, 0.0),
    )
    assert env.ball_in_cup(state=state, ball=E.ball, cup=E.cup) is True
    env.set_state(state=state)
    next_state = env.take_action(action=np.array([1.0, 2.0, 0.0, 0.25, 0.25]))
    assert env.holding(state=next_state, obj=E.cup) is True
    assert env.holding(state=next_state, obj=E.ball) is True


def test_pick_fails_when_success_prob_is_zero() -> None:
    env = E(pick_success_prob=0.0)
    env.set_state(
        state=_state(
            tables=[_table(name="normal-table-0", x=0.5, y=0.5)],
            ball=(0.15, 0.15, 0.02, 0.0),
        )
    )
    next_state = env.take_action(action=np.array([1.0, 1.0, 0.0, 0.15, 0.15]))
    assert env.holding(state=next_state, obj=E.ball) is False


# --- placing: floor ---


def test_place_held_object_on_floor_drops_it_at_commanded_position() -> None:
    env = E()
    state = _state(
        tables=[_table(name="normal-table-0", x=0.5, y=0.5)],
        robot=(0.2, 0.2),
        ball=(0.2, 0.2, 0.02, 1.0),  # held
    )
    env.set_state(state=state)
    # (0.05, 0.05) is on no table -> floor placement.
    next_state = env.take_action(action=np.array([1.0, 1.0, 0.0, 0.05, 0.05]))
    assert env.holding(state=next_state, obj=E.ball) is False
    assert next_state.get(obj=E.ball, feature_name="x") == 0.05
    assert env.on_floor(state=next_state, obj=E.ball) is True


def test_place_held_ball_onto_cup_on_floor_creates_ball_in_cup() -> None:
    env = E()
    state = _state(
        tables=[_table(name="normal-table-0", x=0.5, y=0.5)],
        robot=(0.2, 0.2),
        ball=(0.2, 0.2, 0.02, 1.0),  # held ball
        cup=(0.3, 0.3, 0.03, 0.0),  # cup resting on the floor
    )
    env.set_state(state=state)
    next_state = env.take_action(action=np.array([1.0, 1.0, 0.0, 0.3, 0.3]))
    assert env.ball_in_cup(state=next_state, ball=E.ball, cup=E.cup) is True


# --- placing: tables ---


def test_place_cup_on_reachable_normal_table_stays_on_it() -> None:
    env = E()  # place_sticky_fall_prob defaults to 0.0
    state = _state(
        tables=[_table(name="normal-table-0", x=0.5, y=0.5, radius=0.1)],
        robot=(0.55, 0.5),  # within reachable_thresh (0.1) of the table center
        cup=(0.9, 0.9, 0.03, 1.0),  # held cup
    )
    env.set_state(state=state)
    (table,) = env.get_tables(state=state)
    next_state = env.take_action(action=np.array([1.0, 3.0, 0.0, 0.5, 0.5]))
    assert env.holding(state=next_state, obj=E.cup) is False
    assert env.on_table(state=next_state, obj=E.cup, table=table) is True


def test_place_cup_on_table_carries_the_contained_ball() -> None:
    env = E()
    state = _state(
        tables=[_table(name="normal-table-0", x=0.5, y=0.5, radius=0.1)],
        robot=(0.55, 0.5),
        ball=(0.9, 0.9, 0.02, 1.0),  # held, co-located with cup
        cup=(0.9, 0.9, 0.03, 1.0),  # held cup with ball inside
    )
    assert env.ball_in_cup(state=state, ball=E.ball, cup=E.cup) is True
    env.set_state(state=state)
    (table,) = env.get_tables(state=state)
    next_state = env.take_action(action=np.array([1.0, 3.0, 0.0, 0.5, 0.5]))
    assert env.on_table(state=next_state, obj=E.cup, table=table) is True
    assert env.on_table(state=next_state, obj=E.ball, table=table) is True


def test_place_bare_ball_on_table_always_falls_to_floor() -> None:
    env = E()  # place_ball_fall_prob defaults to 1.0
    state = _state(
        tables=[_table(name="normal-table-0", x=0.5, y=0.5, radius=0.1)],
        robot=(0.55, 0.5),
        ball=(0.9, 0.9, 0.02, 1.0),  # held bare ball
        cup=(0.1, 0.1, 0.03, 0.0),  # cup elsewhere, so no ball-in-cup
    )
    env.set_state(state=state)
    next_state = env.take_action(action=np.array([1.0, 3.0, 0.0, 0.5, 0.5]))
    assert env.holding(state=next_state, obj=E.ball) is False
    assert env.on_floor(state=next_state, obj=E.ball) is True


def test_place_cup_on_smooth_part_of_sticky_table_falls() -> None:
    env = E()  # place_smooth_fall_prob defaults to 1.0
    # Sticky region is a small circle offset to (0.5+0.03, 0.5); we place at the
    # table center (0.5, 0.5), which is OUTSIDE that region -> smooth -> falls.
    state = _state(
        tables=[
            _table(
                name="sticky-table-0",
                x=0.5,
                y=0.5,
                radius=0.1,
                sticky=1.0,
                sox=0.03,
                soy=0.0,
                srr=0.02,
            )
        ],
        robot=(0.55, 0.5),
        cup=(0.9, 0.9, 0.03, 1.0),
    )
    env.set_state(state=state)
    next_state = env.take_action(action=np.array([1.0, 3.0, 0.0, 0.5, 0.5]))
    assert env.on_floor(state=next_state, obj=E.cup) is True


def test_place_cup_on_safe_region_of_sticky_table_stays() -> None:
    env = E()
    # Place inside the sticky safe region (its center) -> stays (fall_prob=0.0).
    state = _state(
        tables=[
            _table(
                name="sticky-table-0",
                x=0.5,
                y=0.5,
                radius=0.1,
                sticky=1.0,
                sox=0.03,
                soy=0.0,
                srr=0.02,
            )
        ],
        robot=(0.55, 0.5),
        cup=(0.9, 0.9, 0.03, 1.0),
    )
    env.set_state(state=state)
    (table,) = env.get_tables(state=state)
    next_state = env.take_action(action=np.array([1.0, 3.0, 0.0, 0.53, 0.5]))
    assert env.on_table(state=next_state, obj=E.cup, table=table) is True


def test_place_on_unreachable_table_is_a_noop() -> None:
    env = E()
    state = _state(
        tables=[_table(name="normal-table-0", x=0.5, y=0.5, radius=0.1)],
        robot=(0.1, 0.1),  # far from the table -> not reachable
        cup=(0.9, 0.9, 0.03, 1.0),
    )
    env.set_state(state=state)
    next_state = env.take_action(action=np.array([1.0, 3.0, 0.0, 0.5, 0.5]))
    # Cup unchanged: still held, still at its old position.
    assert env.holding(state=next_state, obj=E.cup) is True
    assert next_state.get(obj=E.cup, feature_name="x") == 0.9


# --- take_action updates the tracked current_state ---


def test_take_action_updates_current_state() -> None:
    env = E()
    env.set_state(
        state=_state(tables=[_table(name="normal-table-0", x=0.5, y=0.5)], robot=(0.2, 0.2))
    )
    env.take_action(action=np.array([0.0, 0.0, 0.0, 0.3, 0.3]))
    assert env.get_current_state().get(obj=E.robot, feature_name="x") == 0.3


# --- hard_reset / sample_initial_state ---


def test_hard_reset_produces_a_valid_ring_state() -> None:
    env = E()
    env.hard_reset()
    state = env.get_current_state()
    tables = env.get_tables(state=state)
    assert len(tables) == env.num_tables
    # Ball starts on the first (normal) table, not the target (sticky) one.
    assert env.on_table(state=state, obj=E.ball, table=tables[0]) is True
    assert env.on_table(state=state, obj=E.ball, table=tables[-1]) is False
    # Cup starts on the floor; nothing is held initially.
    assert env.on_floor(state=state, obj=E.cup) is True
    assert env.hand_empty(state=state) is True


def test_sample_initial_state_is_deterministic_given_the_rng_seed() -> None:
    a = E().sample_initial_state(rng=np.random.default_rng(123))
    b = E().sample_initial_state(rng=np.random.default_rng(123))
    for obj in a.data:
        assert np.array_equal(a[obj], b[obj])


def test_sample_initial_state_names_one_sticky_table_last() -> None:
    state = E(num_tables=5, num_sticky_tables=1).sample_initial_state(rng=np.random.default_rng(0))
    tables = E.get_tables(state=state)
    assert tables[-1].name == "sticky-table-0"
    assert state.get(obj=tables[-1], feature_name="sticky") == 1.0
    assert all(state.get(obj=t, feature_name="sticky") == 0.0 for t in tables[:-1])


def test_sample_initial_state_robot_reachable_to_exactly_one_object() -> None:
    env = E()
    state = env.sample_initial_state(rng=np.random.default_rng(7))
    reachable = sum(
        env.is_reachable(state=state, robot=E.robot, other=obj)
        for obj in (*env.get_tables(state=state), E.ball, E.cup)
    )
    assert reachable == 1
    assert env.exists_robot_collision(state=state) is False
