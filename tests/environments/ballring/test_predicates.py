import numpy as np

from hitl_pmp.core.problem.environment.types import Object, State
from hitl_pmp.environments.ballring import predicates as P
from hitl_pmp.environments.ballring.environment import BallRingEnvironment

E = BallRingEnvironment


def _table(*, name: str, x: float, y: float, sticky: float = 0.0) -> tuple[Object, np.ndarray]:
    return Object(name=name, type=E.table_type), np.array([x, y, 0.1, sticky, 0.0, 0.0, 0.03])


def _state(
    *,
    robot: tuple[float, float] = (0.5, 0.5),
    ball: tuple[float, float, float, float] = (0.5, 0.5, 0.02, 0.0),
    cup: tuple[float, float, float, float] = (0.9, 0.9, 0.03, 0.0),
) -> State:
    obj, feats = _table(name="normal-table-0", x=0.5, y=0.5)
    return State(
        data={obj: feats, E.robot: np.array(robot), E.ball: np.array(ball), E.cup: np.array(cup)}
    )


def test_ball_on_table_is_the_only_goal_predicate() -> None:
    assert P.BALL_ON_TABLE.name == "BallOnTable"
    assert P.BALL_ON_TABLE.types == (E.ball_type, E.table_type)


def test_ball_on_table_holds_when_ball_is_within_table() -> None:
    state = _state(ball=(0.5, 0.5, 0.02, 0.0))
    (table,) = E.get_tables(state=state)
    assert P.BALL_ON_TABLE.holds(state, (E.ball, table)) is True


def test_ball_on_table_false_when_held() -> None:
    state = _state(ball=(0.5, 0.5, 0.02, 1.0))  # held
    (table,) = E.get_tables(state=state)
    assert P.BALL_ON_TABLE.holds(state, (E.ball, table)) is False


def test_ball_on_floor_true_when_off_every_table() -> None:
    state = _state(ball=(0.05, 0.05, 0.02, 0.0))
    assert P.BALL_ON_FLOOR.holds(state, (E.ball,)) is True


def test_holding_predicates_read_the_held_feature() -> None:
    state = _state(ball=(0.5, 0.5, 0.02, 1.0), cup=(0.9, 0.9, 0.03, 0.0))
    assert P.HOLDING_BALL.holds(state, (E.ball,)) is True
    assert P.HOLDING_CUP.holds(state, (E.cup,)) is False


def test_hand_empty_true_only_when_neither_ball_nor_cup_held() -> None:
    empty = _state(ball=(0.5, 0.5, 0.02, 0.0), cup=(0.9, 0.9, 0.03, 0.0))
    full = _state(ball=(0.5, 0.5, 0.02, 1.0), cup=(0.9, 0.9, 0.03, 0.0))
    assert P.HAND_EMPTY.holds(empty, ()) is True
    assert P.HAND_EMPTY.holds(full, ()) is False


def test_is_reachable_predicates_use_the_euclidean_threshold() -> None:
    near = _state(robot=(0.52, 0.5), ball=(0.5, 0.5, 0.02, 0.0))
    far = _state(robot=(0.9, 0.5), ball=(0.5, 0.5, 0.02, 0.0))
    assert P.IS_REACHABLE_BALL.holds(near, (E.robot, E.ball)) is True
    assert P.IS_REACHABLE_BALL.holds(far, (E.robot, E.ball)) is False


def test_ball_in_cup_and_its_negation_are_complementary() -> None:
    in_cup = _state(ball=(0.3, 0.3, 0.02, 0.0), cup=(0.3, 0.3, 0.03, 0.0))
    out = _state(ball=(0.1, 0.1, 0.02, 0.0), cup=(0.3, 0.3, 0.03, 0.0))
    assert P.BALL_IN_CUP.holds(in_cup, (E.ball, E.cup)) is True
    assert P.BALL_NOT_IN_CUP.holds(in_cup, (E.ball, E.cup)) is False
    assert P.BALL_IN_CUP.holds(out, (E.ball, E.cup)) is False
    assert P.BALL_NOT_IN_CUP.holds(out, (E.ball, E.cup)) is True
