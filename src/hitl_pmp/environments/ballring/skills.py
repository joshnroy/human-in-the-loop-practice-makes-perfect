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
    state. Only the two **place-on-table** skills carry continuous parameters
    ``param_dim=2`` (a fractional radius ``u`` in [0, 1] and an angle ``theta``): they
    are what a learning Method (EES) must tune. Everything else is ``param_dim=0``:

    - **Picks / place-in-cup** are deterministic (target the object's own center /
      the cup's center) -- under the paper's deterministic config a correctly aimed
      pick always grasps, so there is nothing to learn.
    - **Place-on-floor** targets the room center (always the floor, never a table),
      which always succeeds; matching predicators' small random disk there would add
      no learnable signal.
    - **Navigation** deterministically scans angles for a collision-free pose within
      ``reachable_thresh`` of the target -- the faithful *effect* of predicators'
      "sample a valid pose until one works" rejection sampler, without a failure mode
      the experiment doesn't care about.

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
        param_dim=0,
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
        param_dim=0,
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
        param_dim=0,
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
    _NAVIGATE_NAMES: ClassVar[frozenset[str]] = frozenset({
        "NavigateToTable",
        "NavigateToBall",
        "NavigateToCup",
    })

    @staticmethod
    def sample_params(*, ground_skill: GroundSkill, rng: np.random.Generator) -> np.ndarray:
        """State-independent draw of the continuous parameters. Only the two
        place-on-table skills have any (param_dim=2): a fractional radius ``u`` in
        [0, 1) and an angle ``theta`` in [0, 2*pi) -- ``compute_action`` scales ``u``
        by the (state-dependent) usable table radius, reproducing predicators'
        ``dist = uniform(0, table_radius - size)`` / ``theta = uniform(0, 2*pi)``."""
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

        if name == "PlaceBallOnFloor":
            x, y = skills._room_center(env=env)
            return np.array([1.0, 0.0, 1.0, x, y])  # ball_only=1: place just the ball

        if name in ("PlaceCupWithoutBallOnFloor", "PlaceCupWithBallOnFloor"):
            x, y = skills._room_center(env=env)
            return np.array([1.0, 0.0, 0.0, x, y])

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

    @staticmethod
    def navigate_action(*, state: State, env: BallRingEnvironment, target: Object) -> Action:
        """A collision-free robot pose within ``reachable_thresh`` of ``target`` --
        the deterministic realization of a NavigateTo* skill (also reused by the
        privileged oracle). Scans a fixed set of angles at the annulus distance and
        returns the first in-bounds, non-colliding pose."""
        tx = state.get(obj=target, feature_name="x")
        ty = state.get(obj=target, feature_name="y")
        radius = state.get(obj=target, feature_name="radius")
        # A distance strictly greater than the target's radius (no collision) but
        # within reachable_thresh (reachable), matching the thin annulus a large
        # table leaves. Scan angles for the first in-bounds, collision-free pose.
        dist = radius + 0.5 * (env.reachable_thresh - radius)
        best = (tx + dist, ty)
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
        return np.array([0.0, 0.0, 0.0, best[0], best[1]])
