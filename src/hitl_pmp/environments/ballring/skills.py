from typing import ClassVar

import numpy as np

from hitl_pmp.core.method.types import GroundSkill, LiftedAtom, Skill, Variable
from hitl_pmp.core.problem.environment.types import Action, Object, State
from hitl_pmp.core.problem.tasks.types import Predicate

from .environment import BallRingEnvironment
from .predicates import (
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

_E = BallRingEnvironment

_robot = Variable(name="robot", type=_E.robot_type)
_ball = Variable(name="ball", type=_E.ball_type)
_cup = Variable(name="cup", type=_E.cup_type)
_table = Variable(name="table", type=_E.table_type)

# The robot has one pose, so it can be within reach of exactly one thing at a time:
# every NavigateTo* invalidates *all three* reachability predicates, not only the one
# it re-establishes. That is inexpressible as delete effects (which object would they
# name?), hence `ignore_effects` -- see `Skill`'s docstring in core/method/types.py.
# Ported from predicators' three NavigateTo* NSRTs.
_NAVIGATE_IGNORE_EFFECTS: frozenset[Predicate] = frozenset({
    IS_REACHABLE_SURFACE,
    IS_REACHABLE_BALL,
    IS_REACHABLE_CUP,
})


class BallRingSkills:
    """Lifted skill templates for Ball-Ring, ported from predicators'
    ``ground_truth_models/ball_and_cup_sticky_table/{options,nsrts}.py`` (16
    operators; predicators' ``PlaceCupWithBallOnTable`` is deliberately absent there,
    so it is absent here too). Preconditions/add/delete mirror the NSRTs
    symbol-for-symbol. A static-method container, never instantiated, same as every
    other business-logic class in this project.

    Continuous parameters: predicators' options are pass-through (the sampler emits
    the full action vector), with all geometry in the NSRT samplers. This port keeps
    the same split Light Switch uses -- ``sample_params`` draws the state-independent
    continuous degrees of freedom and ``compute_action`` realizes them against the
    state. The five **placement** skills (two place-on-table, three place-on-floor)
    carry continuous parameters ``param_dim=2`` (a fractional radius ``u`` in [0, 1]
    and an angle ``theta``): they are what a learning Method (EES) must tune.
    Everything else is ``param_dim=0``:

    - **Picks / place-in-cup** are deterministic (target the object's own center /
      the cup's center). predicators' ``pick_obj_sampler`` does draw a point, but
      *every* point it can draw succeeds: it offsets by ``(radius/2, radius/2)`` and
      jitters by up to ``radius/4``, so the worst case is
      ``radius*sqrt(2)/2 + radius/4 = 0.957 * radius`` from the object's center --
      inside ``object_contains_point``'s ``radius`` disk with 4.3% of the radius to
      spare -- and the paper config's ``pick_success_prob = 1.0`` then grasps
      unconditionally. A competence model over a skill that succeeds for every
      parameter value learns the same 1.0 either way, so there is nothing to learn.
    - **Navigation** deterministically scans angles across the
      ``(radius, reachable_thresh]`` annulus for a collision-free pose (see
      ``navigate_action``). This is sound only because that scan, like predicators'
      rejection sampler, always finds *some* valid pose: which one it finds is
      unobservable to the symbolic layer, since every ``NavigateTo*`` wipes all three
      reachability predicates via ``ignore_effects`` and placement only tests
      ``euclidean_reachable`` to the target. Scanning a single distance in the
      annulus did *not* always find one, which is what made the earlier version of
      this argument false.

    The place-on-floor skills' jitter is *not* cosmetic. predicators'
    ``place_on_floor_sampler`` scatters the placed object in a disk of radius
    ``2 * radius(obj)`` about the room center; targeting the exact center instead put
    a floor-placed cup and a floor-placed ball at the *identical* point, and at this
    domain's geometry ``BallInCup`` only needs the two centers within
    ``cup_radius - ball_radius`` (~0.0024 at ``num_tables=5``). That made ``BallInCup``
    hold with probability 1 after "place the cup on the floor, then place the ball on
    the floor" (~0.6% with the jitter restored) -- handing the agent a free
    ball-in-cup and recording a spurious *failure* for ``PlaceBallOnFloor``, whose
    ``BallNotInCup`` add-effect then never held.

    ``PlaceBallOnTable`` is an "impossible" skill under the deterministic config: a
    bare ball placed on any table always falls (``place_ball_fall_prob=1.0``), so its
    add-effect ``BallOnTable`` is never actually achieved -- the Ball-Ring analogue of
    Light Switch's ``JumpToLight``, something a competence-tracking Method must learn
    to stop practicing. The goal ``BallOnTable(ball, target)`` is reached instead via
    ``PlaceBallInCupOnTable`` (ball in cup, on a table).
    """

    # ------------------------------------------------------------------ navigate
    NAVIGATE_TO_TABLE: ClassVar[Skill] = Skill(
        name="NavigateToTable",
        parameters=(_robot, _table),
        preconditions=frozenset(),
        add_effects=frozenset({
            LiftedAtom(predicate=IS_REACHABLE_SURFACE, variables=(_robot, _table))
        }),
        delete_effects=frozenset(),
        ignore_effects=_NAVIGATE_IGNORE_EFFECTS,
        param_dim=0,
    )
    NAVIGATE_TO_BALL: ClassVar[Skill] = Skill(
        name="NavigateToBall",
        parameters=(_robot, _ball),
        preconditions=frozenset(),
        add_effects=frozenset({LiftedAtom(predicate=IS_REACHABLE_BALL, variables=(_robot, _ball))}),
        delete_effects=frozenset(),
        ignore_effects=_NAVIGATE_IGNORE_EFFECTS,
        param_dim=0,
    )
    NAVIGATE_TO_CUP: ClassVar[Skill] = Skill(
        name="NavigateToCup",
        parameters=(_robot, _cup),
        preconditions=frozenset(),
        add_effects=frozenset({LiftedAtom(predicate=IS_REACHABLE_CUP, variables=(_robot, _cup))}),
        delete_effects=frozenset(),
        ignore_effects=_NAVIGATE_IGNORE_EFFECTS,
        param_dim=0,
    )

    # ---------------------------------------------------------------- pick ball
    PICK_BALL_FROM_TABLE: ClassVar[Skill] = Skill(
        name="PickBallFromTable",
        parameters=(_robot, _ball, _cup, _table),
        preconditions=frozenset({
            LiftedAtom(predicate=BALL_NOT_IN_CUP, variables=(_ball, _cup)),
            LiftedAtom(predicate=IS_REACHABLE_SURFACE, variables=(_robot, _table)),
            LiftedAtom(predicate=BALL_ON_TABLE, variables=(_ball, _table)),
            LiftedAtom(predicate=HAND_EMPTY, variables=()),
        }),
        add_effects=frozenset({LiftedAtom(predicate=HOLDING_BALL, variables=(_ball,))}),
        delete_effects=frozenset({
            LiftedAtom(predicate=BALL_ON_TABLE, variables=(_ball, _table)),
            LiftedAtom(predicate=HAND_EMPTY, variables=()),
        }),
        # Lifting a ball off a table takes it out of whatever cup it was in --
        # including cups this skill's parameters never name.
        ignore_effects=frozenset({BALL_IN_CUP}),
        param_dim=0,
    )
    PICK_BALL_FROM_FLOOR: ClassVar[Skill] = Skill(
        name="PickBallFromFloor",
        parameters=(_robot, _ball, _cup),
        preconditions=frozenset({
            LiftedAtom(predicate=IS_REACHABLE_BALL, variables=(_robot, _ball)),
            LiftedAtom(predicate=BALL_ON_FLOOR, variables=(_ball,)),
            LiftedAtom(predicate=HAND_EMPTY, variables=()),
        }),
        add_effects=frozenset({
            LiftedAtom(predicate=HOLDING_BALL, variables=(_ball,)),
            LiftedAtom(predicate=BALL_NOT_IN_CUP, variables=(_ball, _cup)),
        }),
        delete_effects=frozenset({
            LiftedAtom(predicate=BALL_ON_FLOOR, variables=(_ball,)),
            LiftedAtom(predicate=HAND_EMPTY, variables=()),
            LiftedAtom(predicate=BALL_IN_CUP, variables=(_ball, _cup)),
        }),
        param_dim=0,
    )

    # ----------------------------------------------------------------- pick cup
    PICK_CUP_WITHOUT_BALL_FROM_TABLE: ClassVar[Skill] = Skill(
        name="PickCupWithoutBallFromTable",
        parameters=(_robot, _cup, _ball, _table),
        preconditions=frozenset({
            LiftedAtom(predicate=BALL_NOT_IN_CUP, variables=(_ball, _cup)),
            LiftedAtom(predicate=IS_REACHABLE_SURFACE, variables=(_robot, _table)),
            LiftedAtom(predicate=CUP_ON_TABLE, variables=(_cup, _table)),
            LiftedAtom(predicate=HAND_EMPTY, variables=()),
        }),
        add_effects=frozenset({LiftedAtom(predicate=HOLDING_CUP, variables=(_cup,))}),
        delete_effects=frozenset({
            LiftedAtom(predicate=CUP_ON_TABLE, variables=(_cup, _table)),
            LiftedAtom(predicate=HAND_EMPTY, variables=()),
        }),
        param_dim=0,
    )
    PICK_CUP_WITH_BALL_FROM_TABLE: ClassVar[Skill] = Skill(
        name="PickCupWithBallFromTable",
        parameters=(_robot, _cup, _ball, _table),
        preconditions=frozenset({
            LiftedAtom(predicate=BALL_IN_CUP, variables=(_ball, _cup)),
            LiftedAtom(predicate=IS_REACHABLE_SURFACE, variables=(_robot, _table)),
            LiftedAtom(predicate=CUP_ON_TABLE, variables=(_cup, _table)),
            LiftedAtom(predicate=HAND_EMPTY, variables=()),
            LiftedAtom(predicate=BALL_ON_TABLE, variables=(_ball, _table)),
        }),
        add_effects=frozenset({
            LiftedAtom(predicate=HOLDING_CUP, variables=(_cup,)),
            LiftedAtom(predicate=HOLDING_BALL, variables=(_ball,)),
        }),
        delete_effects=frozenset({
            LiftedAtom(predicate=CUP_ON_TABLE, variables=(_cup, _table)),
            LiftedAtom(predicate=HAND_EMPTY, variables=()),
            LiftedAtom(predicate=BALL_ON_TABLE, variables=(_ball, _table)),
        }),
        param_dim=0,
    )
    PICK_CUP_WITHOUT_BALL_FROM_FLOOR: ClassVar[Skill] = Skill(
        name="PickCupWithoutBallFromFloor",
        parameters=(_robot, _cup, _ball),
        preconditions=frozenset({
            LiftedAtom(predicate=BALL_NOT_IN_CUP, variables=(_ball, _cup)),
            LiftedAtom(predicate=IS_REACHABLE_CUP, variables=(_robot, _cup)),
            LiftedAtom(predicate=CUP_ON_FLOOR, variables=(_cup,)),
            LiftedAtom(predicate=HAND_EMPTY, variables=()),
        }),
        add_effects=frozenset({LiftedAtom(predicate=HOLDING_CUP, variables=(_cup,))}),
        delete_effects=frozenset({
            LiftedAtom(predicate=CUP_ON_FLOOR, variables=(_cup,)),
            LiftedAtom(predicate=HAND_EMPTY, variables=()),
        }),
        param_dim=0,
    )
    PICK_CUP_WITH_BALL_FROM_FLOOR: ClassVar[Skill] = Skill(
        name="PickCupWithBallFromFloor",
        parameters=(_robot, _cup, _ball),
        preconditions=frozenset({
            LiftedAtom(predicate=BALL_ON_FLOOR, variables=(_ball,)),
            LiftedAtom(predicate=BALL_IN_CUP, variables=(_ball, _cup)),
            LiftedAtom(predicate=IS_REACHABLE_BALL, variables=(_robot, _ball)),
            LiftedAtom(predicate=CUP_ON_FLOOR, variables=(_cup,)),
            LiftedAtom(predicate=HAND_EMPTY, variables=()),
        }),
        add_effects=frozenset({
            LiftedAtom(predicate=HOLDING_CUP, variables=(_cup,)),
            LiftedAtom(predicate=HOLDING_BALL, variables=(_ball,)),
        }),
        delete_effects=frozenset({
            LiftedAtom(predicate=CUP_ON_FLOOR, variables=(_cup,)),
            LiftedAtom(predicate=HAND_EMPTY, variables=()),
            LiftedAtom(predicate=BALL_ON_FLOOR, variables=(_ball,)),
        }),
        param_dim=0,
    )

    # ---------------------------------------------------------------- place ball
    PLACE_BALL_ON_TABLE: ClassVar[Skill] = Skill(
        name="PlaceBallOnTable",
        parameters=(_robot, _ball, _cup, _table),
        preconditions=frozenset({
            LiftedAtom(predicate=BALL_NOT_IN_CUP, variables=(_ball, _cup)),
            LiftedAtom(predicate=IS_REACHABLE_SURFACE, variables=(_robot, _table)),
            LiftedAtom(predicate=HOLDING_BALL, variables=(_ball,)),
        }),
        add_effects=frozenset({
            LiftedAtom(predicate=BALL_ON_TABLE, variables=(_ball, _table)),
            LiftedAtom(predicate=HAND_EMPTY, variables=()),
        }),
        delete_effects=frozenset({LiftedAtom(predicate=HOLDING_BALL, variables=(_ball,))}),
        param_dim=2,
    )
    PLACE_BALL_ON_FLOOR: ClassVar[Skill] = Skill(
        name="PlaceBallOnFloor",
        parameters=(_robot, _cup, _ball),
        preconditions=frozenset({LiftedAtom(predicate=HOLDING_BALL, variables=(_ball,))}),
        add_effects=frozenset({
            LiftedAtom(predicate=BALL_NOT_IN_CUP, variables=(_ball, _cup)),
            LiftedAtom(predicate=BALL_ON_FLOOR, variables=(_ball,)),
        }),
        delete_effects=frozenset({LiftedAtom(predicate=HOLDING_BALL, variables=(_ball,))}),
        # The ball lands somewhere in the middle of the room: out of every cup, and
        # no longer where the robot was standing when it was reachable.
        ignore_effects=frozenset({BALL_IN_CUP, IS_REACHABLE_BALL}),
        param_dim=2,
    )
    PLACE_BALL_IN_CUP_ON_FLOOR: ClassVar[Skill] = Skill(
        name="PlaceBallInCupOnFloor",
        parameters=(_robot, _ball, _cup),
        preconditions=frozenset({
            LiftedAtom(predicate=BALL_NOT_IN_CUP, variables=(_ball, _cup)),
            LiftedAtom(predicate=IS_REACHABLE_CUP, variables=(_robot, _cup)),
            LiftedAtom(predicate=CUP_ON_FLOOR, variables=(_cup,)),
            LiftedAtom(predicate=HOLDING_BALL, variables=(_ball,)),
        }),
        add_effects=frozenset({
            LiftedAtom(predicate=BALL_IN_CUP, variables=(_ball, _cup)),
            LiftedAtom(predicate=BALL_ON_FLOOR, variables=(_ball,)),
            LiftedAtom(predicate=HAND_EMPTY, variables=()),
        }),
        delete_effects=frozenset({
            LiftedAtom(predicate=HOLDING_BALL, variables=(_ball,)),
            LiftedAtom(predicate=BALL_NOT_IN_CUP, variables=(_ball, _cup)),
        }),
        param_dim=0,
    )
    PLACE_BALL_IN_CUP_ON_TABLE: ClassVar[Skill] = Skill(
        name="PlaceBallInCupOnTable",
        parameters=(_robot, _ball, _cup, _table),
        preconditions=frozenset({
            LiftedAtom(predicate=IS_REACHABLE_SURFACE, variables=(_robot, _table)),
            LiftedAtom(predicate=CUP_ON_TABLE, variables=(_cup, _table)),
            LiftedAtom(predicate=HOLDING_BALL, variables=(_ball,)),
            LiftedAtom(predicate=BALL_NOT_IN_CUP, variables=(_ball, _cup)),
        }),
        add_effects=frozenset({
            LiftedAtom(predicate=BALL_IN_CUP, variables=(_ball, _cup)),
            LiftedAtom(predicate=BALL_ON_TABLE, variables=(_ball, _table)),
            LiftedAtom(predicate=HAND_EMPTY, variables=()),
        }),
        delete_effects=frozenset({
            LiftedAtom(predicate=HOLDING_BALL, variables=(_ball,)),
            LiftedAtom(predicate=BALL_NOT_IN_CUP, variables=(_ball, _cup)),
        }),
        param_dim=0,
    )

    # ----------------------------------------------------------------- place cup
    PLACE_CUP_WITHOUT_BALL_ON_TABLE: ClassVar[Skill] = Skill(
        name="PlaceCupWithoutBallOnTable",
        parameters=(_robot, _ball, _cup, _table),
        preconditions=frozenset({
            LiftedAtom(predicate=IS_REACHABLE_SURFACE, variables=(_robot, _table)),
            LiftedAtom(predicate=HOLDING_CUP, variables=(_cup,)),
            LiftedAtom(predicate=BALL_NOT_IN_CUP, variables=(_ball, _cup)),
        }),
        add_effects=frozenset({
            LiftedAtom(predicate=CUP_ON_TABLE, variables=(_cup, _table)),
            LiftedAtom(predicate=HAND_EMPTY, variables=()),
        }),
        delete_effects=frozenset({LiftedAtom(predicate=HOLDING_CUP, variables=(_cup,))}),
        param_dim=2,
    )
    PLACE_CUP_WITHOUT_BALL_ON_FLOOR: ClassVar[Skill] = Skill(
        name="PlaceCupWithoutBallOnFloor",
        parameters=(_robot, _ball, _cup),
        preconditions=frozenset({
            LiftedAtom(predicate=HOLDING_CUP, variables=(_cup,)),
            LiftedAtom(predicate=BALL_NOT_IN_CUP, variables=(_ball, _cup)),
        }),
        add_effects=frozenset({
            LiftedAtom(predicate=CUP_ON_FLOOR, variables=(_cup,)),
            LiftedAtom(predicate=HAND_EMPTY, variables=()),
        }),
        delete_effects=frozenset({LiftedAtom(predicate=HOLDING_CUP, variables=(_cup,))}),
        param_dim=2,
    )
    PLACE_CUP_WITH_BALL_ON_FLOOR: ClassVar[Skill] = Skill(
        name="PlaceCupWithBallOnFloor",
        parameters=(_robot, _ball, _cup),
        preconditions=frozenset({
            LiftedAtom(predicate=HOLDING_CUP, variables=(_cup,)),
            LiftedAtom(predicate=BALL_IN_CUP, variables=(_ball, _cup)),
        }),
        add_effects=frozenset({
            LiftedAtom(predicate=CUP_ON_FLOOR, variables=(_cup,)),
            LiftedAtom(predicate=HAND_EMPTY, variables=()),
            LiftedAtom(predicate=BALL_ON_FLOOR, variables=(_ball,)),
        }),
        delete_effects=frozenset({
            LiftedAtom(predicate=HOLDING_CUP, variables=(_cup,)),
            LiftedAtom(predicate=HOLDING_BALL, variables=(_ball,)),
        }),
        param_dim=2,
    )

    @staticmethod
    def all_skills() -> tuple[Skill, ...]:
        s = BallRingSkills
        return (
            s.NAVIGATE_TO_TABLE,
            s.NAVIGATE_TO_BALL,
            s.NAVIGATE_TO_CUP,
            s.PICK_BALL_FROM_TABLE,
            s.PICK_BALL_FROM_FLOOR,
            s.PICK_CUP_WITHOUT_BALL_FROM_TABLE,
            s.PICK_CUP_WITH_BALL_FROM_TABLE,
            s.PICK_CUP_WITHOUT_BALL_FROM_FLOOR,
            s.PICK_CUP_WITH_BALL_FROM_FLOOR,
            s.PLACE_BALL_ON_TABLE,
            s.PLACE_BALL_ON_FLOOR,
            s.PLACE_BALL_IN_CUP_ON_FLOOR,
            s.PLACE_BALL_IN_CUP_ON_TABLE,
            s.PLACE_CUP_WITHOUT_BALL_ON_TABLE,
            s.PLACE_CUP_WITHOUT_BALL_ON_FLOOR,
            s.PLACE_CUP_WITH_BALL_ON_FLOOR,
        )

    _PICK_SKILL_NAMES: ClassVar[frozenset[str]] = frozenset({
        "PickBallFromTable",
        "PickBallFromFloor",
        "PickCupWithoutBallFromTable",
        "PickCupWithBallFromTable",
        "PickCupWithoutBallFromFloor",
        "PickCupWithBallFromFloor",
    })
    _PLACE_ON_TABLE_NAMES: ClassVar[frozenset[str]] = frozenset({
        "PlaceBallOnTable",
        "PlaceCupWithoutBallOnTable",
    })
    # predicators routes all three through one `place_on_floor_sampler`.
    _PLACE_ON_FLOOR_NAMES: ClassVar[frozenset[str]] = frozenset({
        "PlaceBallOnFloor",
        "PlaceCupWithoutBallOnFloor",
        "PlaceCupWithBallOnFloor",
    })
    _NAVIGATE_NAMES: ClassVar[frozenset[str]] = frozenset({
        "NavigateToTable",
        "NavigateToBall",
        "NavigateToCup",
    })

    @staticmethod
    def sample_params(*, ground_skill: GroundSkill, rng: np.random.Generator) -> np.ndarray:
        """State-independent draw of the continuous parameters. The five placement
        skills have any (param_dim=2): a fractional radius ``u`` in [0, 1) and an
        angle ``theta`` in [0, 2*pi). ``compute_action`` scales ``u`` by the
        state-dependent maximum distance, reproducing predicators' two samplers --
        ``dist = uniform(0, table_radius - size)`` for place-on-table
        (``place_on_table_sampler``) and ``dist = uniform(0, size)`` about the room
        center for place-on-floor (``place_on_floor_sampler``), both with
        ``theta = uniform(0, 2*pi)``."""
        if ground_skill.skill.param_dim == 0:
            return np.zeros(0)
        return np.array([rng.uniform(0.0, 1.0), rng.uniform(0.0, 2 * np.pi)])

    @staticmethod
    def compute_action(
        *, ground_skill: GroundSkill, params: np.ndarray, state: State, env: BallRingEnvironment
    ) -> Action:
        """Realize a chosen (ground skill, params) as a raw 5-D Ball-Ring action
        ``[move_or_pickplace, obj_type_id, ball_only, x, y]``. Takes the env instance
        explicitly (per the project convention) -- navigation needs it to reject
        colliding poses."""
        skills = BallRingSkills
        name = ground_skill.skill.name
        objects = ground_skill.objects

        if name in skills._NAVIGATE_NAMES:
            target = objects[1]
            return skills.navigate_action(state=state, env=env, target=target)

        if name in skills._PICK_SKILL_NAMES:
            obj = objects[1]  # the object being picked (ball or cup)
            obj_type_id = 1.0 if obj.type == env.ball_type else 2.0
            return np.array([
                1.0,
                obj_type_id,
                0.0,
                state.get(obj=obj, feature_name="x"),
                state.get(obj=obj, feature_name="y"),
            ])

        if name in skills._PLACE_ON_TABLE_NAMES:
            return skills._place_on_table_action(
                ground_skill=ground_skill, params=params, state=state
            )

        if name in skills._PLACE_ON_FLOOR_NAMES:
            x, y = skills.place_on_floor_xy(
                ground_skill=ground_skill, params=params, state=state, env=env
            )
            # ball_only=1 for PlaceBallOnFloor: place just the ball, not the cup
            # (predicators' `place_ball_on_floor_sampler` sets `sample_arr[2] = 1.0`).
            ball_only = 1.0 if name == "PlaceBallOnFloor" else 0.0
            return np.array([1.0, 0.0, ball_only, x, y])

        if name in ("PlaceBallInCupOnFloor", "PlaceBallInCupOnTable"):
            cup = objects[2]
            return np.array([
                1.0,
                2.0,  # "place in cup"
                0.0,
                state.get(obj=cup, feature_name="x"),
                state.get(obj=cup, feature_name="y"),
            ])

        raise ValueError(f"Unknown skill: {name}")

    @staticmethod
    def _room_center(*, env: BallRingEnvironment) -> tuple[float, float]:
        return (env.x_lb + env.x_ub) / 2, (env.y_lb + env.y_ub) / 2

    @staticmethod
    def place_on_floor_xy(
        *,
        ground_skill: GroundSkill,
        params: np.ndarray,
        state: State,
        env: BallRingEnvironment,
    ) -> tuple[float, float]:
        """The concrete floor point a place-on-floor skill's `(u, theta)` resolve to:
        a small random disk about the room center, of radius `2 * radius(obj)` where
        `obj` is the object being placed. Ported from predicators'
        `place_on_floor_sampler` (`size = radius * 2; dist = uniform(0, size);
        theta = uniform(0, 2*pi); x = x_c + dist*cos(theta)`).

        The placed object is `ground_skill.objects[-1]`, matching predicators'
        `obj_to_place = objs[-1]`: the ball for `PlaceBallOnFloor` (parameters
        `[robot, cup, ball]`) and the cup for both `PlaceCup*OnFloor` (parameters
        `[robot, ball, cup]`).

        The disk never overlaps a table (at `num_tables=5` the ring sits `~0.235`
        clear of the largest jitter), so a floor place still always lands on the
        floor -- the jitter only decouples two successive floor placements, which
        previously coincided exactly."""
        obj = ground_skill.objects[-1]
        u, theta = float(params[0]), float(params[1])
        size = state.get(obj=obj, feature_name="radius") * 2
        dist = u * size
        x_c, y_c = BallRingSkills._room_center(env=env)
        return float(x_c + dist * np.cos(theta)), float(y_c + dist * np.sin(theta))

    @staticmethod
    def place_on_table_xy(
        *, ground_skill: GroundSkill, params: np.ndarray, state: State
    ) -> tuple[float, float]:
        """The concrete placement point `(x, y)` a place-on-table skill's continuous
        parameters `(u, theta)` resolve to, given the state: a fractional radius `u`
        scaled by the usable table radius, at angle `theta` about the table center.
        Ported from predicators' `place_on_table_sampler`
        (`dist = uniform(0, table_radius - size); x = table_x + dist*cos(theta)`).

        Factored out so the raw action (`_place_on_table_action`) and the oracle
        sampler input (`oracle_sampler_input`) convert `(u, theta)` the *same* way --
        the oracle classifier is trained/scored on these placement coordinates, so
        they must match the coordinates the action actually commands.

        objects: `[robot, ball, cup, table]`; the placed object is `objects[-2]`, the
        table `objects[-1]` (predicators' `place_on_table_sampler` uses
        `objs[-2]`/`objs[-1]`)."""
        obj = ground_skill.objects[-2]
        table = ground_skill.objects[-1]
        u, theta = float(params[0]), float(params[1])
        table_x = state.get(obj=table, feature_name="x")
        table_y = state.get(obj=table, feature_name="y")
        table_radius = state.get(obj=table, feature_name="radius")
        size = state.get(obj=obj, feature_name="radius") * 2
        dist = u * max(table_radius - size, 0.0)
        x = table_x + dist * np.cos(theta)
        y = table_y + dist * np.sin(theta)
        return float(x), float(y)

    @staticmethod
    def _place_on_table_action(
        *, ground_skill: GroundSkill, params: np.ndarray, state: State
    ) -> Action:
        x, y = BallRingSkills.place_on_table_xy(
            ground_skill=ground_skill, params=params, state=state
        )
        return np.array([1.0, 3.0, 0.0, x, y])  # 3.0 = place onto table

    @staticmethod
    def oracle_sampler_input(
        *, ground_skill: GroundSkill, state: State, params: np.ndarray
    ) -> list[float] | None:
        """Ball-Ring's oracle feature selection for the learned sampler --
        predicators' `active_sampler_learning_feature_selection = "oracle"` branch for
        `ball_and_cup_sticky_table` in `utils.construct_active_sampler_input`.

        For a place-cup-*-on-table skill (option name contains both ``"PlaceCup"`` and
        ``"Table"``), returns the curated row -- bias, the target table's geometry and
        sticky-region description, and the placement coordinates -- so the
        cup-placement sampler sees the sticky-region features directly instead of
        having them buried under five tables' worth of concatenated ``state[obj]``
        clutter (the ``"all"`` layout). Every other skill returns ``None`` (fall back
        to ``"all"``): ``PlaceBallOnTable`` has ``"Ball"`` not ``"Cup"``, and this port
        deliberately has no ``PlaceCupWithBallOnTable`` (absent in predicators'
        ground-truth models too), so exactly one skill,
        ``PlaceCupWithoutBallOnTable``, takes this path.

        `place_x`/`place_y` are the converted `(x, y)` placement coordinates (via
        `place_on_table_xy`), matching predicators, whose place option params are the
        placement coordinates directly (`_, _, _, param_x, param_y = params`); ours
        sample `(u, theta)` and convert, so the conversion happens here."""
        name = ground_skill.skill.name
        if "PlaceCup" not in name or "Table" not in name:
            return None
        table = ground_skill.objects[-1]
        place_x, place_y = BallRingSkills.place_on_table_xy(
            ground_skill=ground_skill, params=params, state=state
        )
        return [
            1.0,  # bias
            state.get(obj=table, feature_name="radius"),
            state.get(obj=table, feature_name="sticky"),
            state.get(obj=table, feature_name="sticky_region_x_offset"),
            state.get(obj=table, feature_name="sticky_region_y_offset"),
            state.get(obj=table, feature_name="sticky_region_radius"),
            state.get(obj=table, feature_name="x"),
            state.get(obj=table, feature_name="y"),
            place_x,
            place_y,
        ]

    # Fixed set of candidate approach angles scanned for a collision-free pose --
    # deterministic stand-in for predicators' navigate rejection sampler.
    _NAV_ANGLES: ClassVar[tuple[float, ...]] = tuple(
        float(a) for a in np.linspace(0.0, 2 * np.pi, num=24, endpoint=False)
    )
    # Fractions of the way across the (radius, reachable_thresh] annulus, tried in
    # order. 0.5 first, so every navigation that already worked returns the exact
    # same pose it always did; the rest only ever run when 0.5 finds nothing.
    # Never 1.0: a pose at exactly reachable_thresh can round to just over it and
    # read as unreachable, so the outermost probe stops at 0.99.
    _NAV_ANNULUS_FRACTIONS: ClassVar[tuple[float, ...]] = (0.5, 0.7, 0.85, 0.95, 0.99)

    @staticmethod
    def navigate_action(*, state: State, env: BallRingEnvironment, target: Object) -> Action:
        """A collision-free robot pose within ``reachable_thresh`` of ``target`` --
        the deterministic realization of a NavigateTo* skill (also reused by the
        privileged oracle). Scans angles at a series of distances strictly inside the
        (target radius, ``reachable_thresh``] annulus and returns the first in-bounds,
        non-colliding pose.

        Scanning the annulus rather than one distance in it is what makes this a
        faithful stand-in for predicators' ``navigate_to_obj_sampler``, which
        rejection-samples ``dist ~ U(radius, reachable_thresh)`` until a pose is
        collision-free (and then *asserts* the reachability predicate, so it never
        gives up). At a single distance the scan genuinely fails whenever the target
        sits on a table: every angle at 0.5 of the annulus is still inside the
        table's own circle, so all 24 candidates collide -- measured on 300 initial
        states, that was 153 of 2100 target poses, all of them the ball on its start
        table. The old fallback then returned a colliding pose *unchecked*, which
        ``_simulate`` rejects, silently turning ``NavigateToBall`` into a no-op that
        records a failure predicators cannot produce. The extra fractions bring that
        to 0 of 2100."""
        tx = state.get(obj=target, feature_name="x")
        ty = state.get(obj=target, feature_name="y")
        radius = state.get(obj=target, feature_name="radius")
        span = env.reachable_thresh - radius
        # Distances strictly greater than the target's radius (no collision with the
        # target itself) but within reachable_thresh (reachable), matching the thin
        # annulus a large table leaves.
        fallback = (tx + radius + 0.5 * span, ty)
        for fraction in BallRingSkills._NAV_ANNULUS_FRACTIONS:
            dist = radius + fraction * span
            for theta in BallRingSkills._NAV_ANGLES:
                x = tx + dist * np.cos(theta)
                y = ty + dist * np.sin(theta)
                if not (env.x_lb <= x <= env.x_ub and env.y_lb <= y <= env.y_ub):
                    continue
                trial = state.model_copy(deep=True)
                trial.set(obj=env.robot, feature_name="x", feature_val=x)
                trial.set(obj=env.robot, feature_name="y", feature_val=y)
                if not env.exists_robot_collision(state=trial):
                    return np.array([0.0, 0.0, 0.0, x, y])
        return np.array([0.0, 0.0, 0.0, fallback[0], fallback[1]])
