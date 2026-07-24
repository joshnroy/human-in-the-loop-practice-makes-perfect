"""The symbolic layer for Ball-Ring: the twelve Predicates from predicators'
``BallAndCupStickyTableEnv`` (see ``predicates`` property there). Each is a thin
Predicate wrapping one of ``BallRingEnvironment``'s state classifiers via a lambda
adapter, exactly as ``lightswitch/predicates.py`` does -- Predicate.holds is a
positional ``(state, objects)`` callable (Goal.is_satisfied calls it that way), and
each lambda adapts that into the relevant keyword-only classifier call.

Only ``BALL_ON_TABLE`` is a goal predicate (matching predicators' ``goal_predicates``);
the rest are here because tasks.py/skills.py (the lifted operators, PR 2) need them
as preconditions/effects. They depend only on the environment, so they live in PR 1
alongside it -- a Goal cannot even be expressed without ``BALL_ON_TABLE``.
"""

from hitl_pmp.core.problem.tasks.types import Predicate

from .environment import BallRingEnvironment

_env = BallRingEnvironment

BALL_ON_TABLE = Predicate(
    name="BallOnTable",
    types=(_env.ball_type, _env.table_type),
    holds=lambda state, objects: _env.on_table(state=state, obj=objects[0], table=objects[1]),
)

CUP_ON_TABLE = Predicate(
    name="CupOnTable",
    types=(_env.cup_type, _env.table_type),
    holds=lambda state, objects: _env.on_table(state=state, obj=objects[0], table=objects[1]),
)

BALL_ON_FLOOR = Predicate(
    name="BallOnFloor",
    types=(_env.ball_type,),
    holds=lambda state, objects: _env.on_floor(state=state, obj=objects[0]),
)

CUP_ON_FLOOR = Predicate(
    name="CupOnFloor",
    types=(_env.cup_type,),
    holds=lambda state, objects: _env.on_floor(state=state, obj=objects[0]),
)

HOLDING_BALL = Predicate(
    name="HoldingBall",
    types=(_env.ball_type,),
    holds=lambda state, objects: _env.holding(state=state, obj=objects[0]),
)

HOLDING_CUP = Predicate(
    name="HoldingCup",
    types=(_env.cup_type,),
    holds=lambda state, objects: _env.holding(state=state, obj=objects[0]),
)

HAND_EMPTY = Predicate(
    name="HandEmpty",
    types=(),
    holds=lambda state, objects: _env.hand_empty(state=state),
)

IS_REACHABLE_SURFACE = Predicate(
    name="IsReachableSurface",
    types=(_env.robot_type, _env.table_type),
    holds=lambda state, objects: _env.is_reachable(state=state, robot=objects[0], other=objects[1]),
)

IS_REACHABLE_BALL = Predicate(
    name="IsReachableBall",
    types=(_env.robot_type, _env.ball_type),
    holds=lambda state, objects: _env.is_reachable(state=state, robot=objects[0], other=objects[1]),
)

IS_REACHABLE_CUP = Predicate(
    name="IsReachableCup",
    types=(_env.robot_type, _env.cup_type),
    holds=lambda state, objects: _env.is_reachable(state=state, robot=objects[0], other=objects[1]),
)

BALL_IN_CUP = Predicate(
    name="BallInCup",
    types=(_env.ball_type, _env.cup_type),
    holds=lambda state, objects: _env.ball_in_cup(state=state, ball=objects[0], cup=objects[1]),
)

BALL_NOT_IN_CUP = Predicate(
    name="BallNotInCup",
    types=(_env.ball_type, _env.cup_type),
    holds=lambda state, objects: not _env.ball_in_cup(state=state, ball=objects[0], cup=objects[1]),
)
