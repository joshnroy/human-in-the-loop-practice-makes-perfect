from typing import ClassVar

import numpy as np
from gymnasium.spaces import Box
from pydantic import PrivateAttr

from hitl_pmp.core.problem.environment.environment import Environment
from hitl_pmp.core.problem.environment.types import Action, Object, State, Type


class BallRingEnvironment(Environment):
    """The simulated "Ball-Ring" environment from the Practice Makes Perfect paper --
    predicators' ``ball_and_cup_sticky_table`` (``BallAndCupStickyTableEnv`` in the
    sibling hitl-practice repo, ``predicators/envs/ball_and_cup_sticky_table.py``),
    ported symbol-for-symbol.

    A ball must be transported onto a target table arranged in a ring of tables. If
    the agent places a bare ball on any table it rolls off (falls to the floor); if
    the ball is first dropped into a cup, the ball+cup system stays on normal tables
    (and on the safe, non-smooth sub-region of the one sticky table). The target
    table is the sticky one, so the only reliable solution is: pick the ball, drop it
    into the cup, pick the cup (which lifts the contained ball too), navigate to the
    target table, and place the cup on its safe sub-region.

    Deterministic paper config. The reference env has genuine placement
    stochasticity (settings.py defaults), but the paper's simulated Ball-Ring uses
    the deterministic overrides from
    ``scripts/configs/active_sampler_learning.yaml``: every fall probability is 0.0
    or 1.0 and pick success is 1.0, so ``uniform() < prob`` is decided without
    reference to the draw. The single residual randomness is *where* a fallen object
    lands (``sample_floor_point_around_table``), seeded per-instance by ``noise_seed``
    so runs are reproducible; it never affects the happy path (a correctly placed
    cup never falls).

    Geometry/world constants (``x_lb``/``x_ub``/.../``reachable_thresh``/
    ``objs_scale``/``sticky_region_radius_scale``) and the four ``Type`` schemas stay
    ClassVar -- genuine structural constants identical for every instance, matching
    predicators' own ClassVars. ``robot``/``ball``/``cup`` are the domain's three
    singleton Objects (also ClassVar); tables are created per initial state (their
    count/names depend on ``num_tables``/``num_sticky_tables``). ``num_tables``/
    ``num_sticky_tables``/the fall+pick probabilities/``noise_seed`` are genuine
    per-run configuration, so they are real constructor fields (e.g.
    ``BallRingEnvironment(num_tables=5)``).
    """

    x_lb: ClassVar[float] = 0.0
    x_ub: ClassVar[float] = 1.0
    y_lb: ClassVar[float] = 0.0
    y_ub: ClassVar[float] = 1.0
    reachable_thresh: ClassVar[float] = 0.1
    objs_scale: ClassVar[float] = 0.25  # as a function of table radius
    sticky_region_radius_scale: ClassVar[float] = 0.35

    table_type: ClassVar[Type] = Type(
        name="table",
        feature_names=(
            "x",
            "y",
            "radius",
            "sticky",
            "sticky_region_x_offset",
            "sticky_region_y_offset",
            "sticky_region_radius",
        ),
    )
    robot_type: ClassVar[Type] = Type(name="robot", feature_names=("x", "y"))
    ball_type: ClassVar[Type] = Type(name="ball", feature_names=("x", "y", "radius", "held"))
    cup_type: ClassVar[Type] = Type(name="cup", feature_names=("x", "y", "radius", "held"))

    robot: ClassVar[Object] = Object(name="robot", type=robot_type)
    ball: ClassVar[Object] = Object(name="ball", type=ball_type)
    cup: ClassVar[Object] = Object(name="cup", type=cup_type)

    # Action space [move_or_pickplace, obj_type_id, ball_only, x, y] -- matches
    # predicators' 5D Box exactly. move_or_pickplace: 0 = navigate to (x, y), 1 =
    # pick/place the object of obj_type_id (1 = ball, 2 = cup, 3 = table) at (x, y).
    # ball_only handles placing only the ball while also holding the cup.
    action_space: ClassVar[Box] = Box(
        np.array([0.0, 0.0, 0.0, x_lb, y_lb], dtype=np.float32),
        np.array([1.0, 3.0, 1.0, x_ub, y_ub], dtype=np.float32),
    )

    # Deterministic paper config (active_sampler_learning.yaml's
    # ball_and_cup_sticky_table block), not settings.py's stochastic defaults.
    num_tables: int = 5
    num_sticky_tables: int = 1
    pick_success_prob: float = 1.0
    place_sticky_fall_prob: float = 0.0
    place_ball_fall_prob: float = 1.0
    place_smooth_fall_prob: float = 1.0
    noise_seed: int = 0

    # Seeded from noise_seed (predicators seeds its _noise_rng from CFG.seed too).
    # Private + rederived in model_post_init, never a constructor-settable field.
    _noise_rng: np.random.Generator = PrivateAttr()

    def model_post_init(self, __context: object) -> None:
        self._noise_rng = np.random.default_rng(self.noise_seed)

    # --- geometry helpers (static: pure functions of the numbers passed in) ---

    @staticmethod
    def circle_contains_point(*, cx: float, cy: float, radius: float, px: float, py: float) -> bool:
        """utils.Circle.contains_point: (px-cx)^2 + (py-cy)^2 <= radius^2."""
        return bool((px - cx) ** 2 + (py - cy) ** 2 <= radius**2)

    @staticmethod
    def circle_contains_circle(
        *, cx: float, cy: float, radius: float, ox: float, oy: float, oradius: float
    ) -> bool:
        """utils.Circle.contains_circle: dist_between_centers + oradius <= radius."""
        dist = float(np.sqrt((ox - cx) ** 2 + (oy - cy) ** 2))
        return bool(dist + oradius <= radius)

    @classmethod
    def object_contains_point(cls, *, state: State, obj: Object, px: float, py: float) -> bool:
        return cls.circle_contains_point(
            cx=state.get(obj=obj, feature_name="x"),
            cy=state.get(obj=obj, feature_name="y"),
            radius=state.get(obj=obj, feature_name="radius"),
            px=px,
            py=py,
        )

    @classmethod
    def euclidean_reachable(cls, *, x1: float, y1: float, x2: float, y2: float) -> bool:
        return bool(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) <= cls.reachable_thresh)

    # --- state classifiers (shared with predicates.py's Predicate singletons) ---

    @classmethod
    def get_tables(cls, *, state: State) -> tuple[Object, ...]:
        """The table Objects in a state, sorted by name -- matches predicators'
        ``sorted(state_dict)`` over the tables. With one sticky table this orders
        ``normal-table-1..k`` before ``sticky-table-0``, so ``get_tables(...)[-1]``
        is the target (sticky) table and ``[0]`` is the ball's start table."""
        tables = [obj for obj in state.data if obj.type == cls.table_type]
        return tuple(sorted(tables, key=lambda obj: obj.name))

    @staticmethod
    def holding(*, state: State, obj: Object) -> bool:
        return bool(state.get(obj=obj, feature_name="held") > 0.5)

    @classmethod
    def on_table(cls, *, state: State, obj: Object, table: Object) -> bool:
        if cls.holding(state=state, obj=obj):
            return False
        return cls.circle_contains_circle(
            cx=state.get(obj=table, feature_name="x"),
            cy=state.get(obj=table, feature_name="y"),
            radius=state.get(obj=table, feature_name="radius"),
            ox=state.get(obj=obj, feature_name="x"),
            oy=state.get(obj=obj, feature_name="y"),
            oradius=state.get(obj=obj, feature_name="radius"),
        )

    @classmethod
    def on_floor(cls, *, state: State, obj: Object) -> bool:
        if cls.holding(state=state, obj=obj):
            return False
        for table in cls.get_tables(state=state):
            if cls.on_table(state=state, obj=obj, table=table):
                return False
        return True

    @classmethod
    def hand_empty(cls, *, state: State) -> bool:
        return not (cls.holding(state=state, obj=cls.ball) or cls.holding(state=state, obj=cls.cup))

    @classmethod
    def is_reachable(cls, *, state: State, robot: Object, other: Object) -> bool:
        return cls.euclidean_reachable(
            x1=state.get(obj=robot, feature_name="x"),
            y1=state.get(obj=robot, feature_name="y"),
            x2=state.get(obj=other, feature_name="x"),
            y2=state.get(obj=other, feature_name="y"),
        )

    @classmethod
    def ball_in_cup(cls, *, state: State, ball: Object, cup: Object) -> bool:
        same_pos = cls.circle_contains_circle(
            cx=state.get(obj=cup, feature_name="x"),
            cy=state.get(obj=cup, feature_name="y"),
            radius=state.get(obj=cup, feature_name="radius"),
            ox=state.get(obj=ball, feature_name="x"),
            oy=state.get(obj=ball, feature_name="y"),
            oradius=state.get(obj=ball, feature_name="radius"),
        )
        holding_ball = cls.holding(state=state, obj=ball)
        holding_cup = cls.holding(state=state, obj=cup)
        return same_pos and (
            (holding_ball and holding_cup) or (not holding_ball and not holding_cup)
        )

    def table_is_sticky(self, *, state: State, table: Object) -> bool:
        return bool(state.get(obj=table, feature_name="sticky") > 0.5)

    def exists_robot_collision(self, *, state: State) -> bool:
        """True if the robot's point lies inside any table/cup/ball circle."""
        rx = state.get(obj=self.robot, feature_name="x")
        ry = state.get(obj=self.robot, feature_name="y")
        for obj in (*self.get_tables(state=state), self.cup, self.ball):
            if self.object_contains_point(state=state, obj=obj, px=rx, py=ry):
                return True
        return False

    # --- dynamics ---

    def take_action(self, *, action: Action) -> State:
        next_state = self._simulate(state=self.get_current_state(), action=action)
        self.set_state(state=next_state)
        return next_state

    def _simulate(self, *, state: State, action: Action) -> State:
        """Port of predicators' ``BallAndCupStickyTableEnv.simulate``. Defensive
        post-condition ``assert``s in the reference (checking a pick actually
        grasped, a place actually landed on table/floor) are dropped: they hold
        whenever the skills layer supplies valid coordinates, and dropping them
        makes an off-target action a no-op rather than a crash, without changing any
        successful transition."""
        move_or_pickplace = float(action[0])
        obj_type_id = float(action[1])
        ball_only = float(action[2])
        act_x = float(action[3])
        act_y = float(action[4])

        next_state = state.model_copy(deep=True)
        ball, cup, robot = self.ball, self.cup, self.robot
        hand_empty = self.hand_empty(state=state)
        ball_held = self.holding(state=state, obj=ball)
        cup_held = self.holding(state=state, obj=cup)
        ball_in_cup = self.ball_in_cup(state=state, ball=ball, cup=cup)

        obj_being_held: Object | None = None
        if (ball_held and not cup_held) or ball_only > 0.5:
            obj_being_held = ball
        elif cup_held:
            obj_being_held = cup

        if move_or_pickplace == 1.0:
            if hand_empty:
                self._handle_picking(
                    next_state=next_state,
                    state=state,
                    obj_type_id=obj_type_id,
                    act_x=act_x,
                    act_y=act_y,
                    ball_in_cup=ball_in_cup,
                )
            else:
                assert obj_being_held is not None
                self._handle_placing(
                    next_state=next_state,
                    state=state,
                    obj_type_id=obj_type_id,
                    ball_only=ball_only,
                    act_x=act_x,
                    act_y=act_y,
                    obj_being_held=obj_being_held,
                    ball_in_cup=ball_in_cup,
                )
        else:
            # Navigation: move to (act_x, act_y) unless it would collide.
            rx = state.get(obj=robot, feature_name="x")
            ry = state.get(obj=robot, feature_name="y")
            next_state.set(obj=robot, feature_name="x", feature_val=act_x)
            next_state.set(obj=robot, feature_name="y", feature_val=act_y)
            if self.exists_robot_collision(state=next_state):
                next_state.set(obj=robot, feature_name="x", feature_val=rx)
                next_state.set(obj=robot, feature_name="y", feature_val=ry)
        return next_state

    def _handle_picking(
        self,
        *,
        next_state: State,
        state: State,
        obj_type_id: float,
        act_x: float,
        act_y: float,
        ball_in_cup: bool,
    ) -> None:
        if self._noise_rng.uniform() >= self.pick_success_prob:
            return
        if obj_type_id == 1.0:
            if self.object_contains_point(state=state, obj=self.ball, px=act_x, py=act_y):
                next_state.set(obj=self.ball, feature_name="held", feature_val=1.0)
        else:
            assert obj_type_id == 2.0
            if self.object_contains_point(state=state, obj=self.cup, px=act_x, py=act_y):
                next_state.set(obj=self.cup, feature_name="held", feature_val=1.0)
                if ball_in_cup:
                    # Picking the cup lifts the contained ball too.
                    next_state.set(obj=self.ball, feature_name="held", feature_val=1.0)

    def _handle_placing(
        self,
        *,
        next_state: State,
        state: State,
        obj_type_id: float,
        ball_only: float,
        act_x: float,
        act_y: float,
        obj_being_held: Object,
        ball_in_cup: bool,
    ) -> None:
        table: Object | None = None
        for target in self.get_tables(state=state):
            if self.object_contains_point(state=state, obj=target, px=act_x, py=act_y):
                table = target
                break

        if table is None:
            # No reachability check for the floor: the robot may 'throw' onto it.
            self._place_object(
                next_state=next_state,
                act_x=act_x,
                act_y=act_y,
                obj_being_held=obj_being_held,
                ball_in_cup=ball_in_cup,
                ball_only=ball_only,
            )
            next_state.set(obj=obj_being_held, feature_name="held", feature_val=0.0)
            return

        table_x = state.get(obj=table, feature_name="x")
        table_y = state.get(obj=table, feature_name="y")
        if not self.euclidean_reachable(
            x1=state.get(obj=self.robot, feature_name="x"),
            y1=state.get(obj=self.robot, feature_name="y"),
            x2=table_x,
            y2=table_y,
        ):
            return

        next_state.set(obj=obj_being_held, feature_name="held", feature_val=0.0)
        if obj_type_id == 3.0:
            fall_prob = self.place_sticky_fall_prob
            if obj_being_held == self.ball:
                fall_prob = self.place_ball_fall_prob
            if self.table_is_sticky(state=state, table=table):
                sticky_region_x = table_x + state.get(
                    obj=table, feature_name="sticky_region_x_offset"
                )
                sticky_region_y = table_y + state.get(
                    obj=table, feature_name="sticky_region_y_offset"
                )
                sticky_region_radius = state.get(obj=table, feature_name="sticky_region_radius")
                on_safe_region = self.circle_contains_point(
                    cx=sticky_region_x,
                    cy=sticky_region_y,
                    radius=sticky_region_radius,
                    px=act_x,
                    py=act_y,
                )
                if not on_safe_region:
                    fall_prob = self.place_smooth_fall_prob if obj_being_held == self.cup else 1.0
            if self._noise_rng.uniform() < fall_prob:
                fall_x, fall_y = self._sample_floor_point_around_table(state=state, table=table)
                self._place_object(
                    next_state=next_state,
                    act_x=fall_x,
                    act_y=fall_y,
                    obj_being_held=obj_being_held,
                    ball_in_cup=ball_in_cup,
                    ball_only=ball_only,
                )
            else:
                self._place_object(
                    next_state=next_state,
                    act_x=act_x,
                    act_y=act_y,
                    obj_being_held=obj_being_held,
                    ball_in_cup=ball_in_cup,
                    ball_only=ball_only,
                )
        else:
            # obj_type_id == 2.0 while holding: drop the ball into the cup.
            assert obj_type_id == 2.0
            assert obj_being_held == self.ball
            next_state.set(obj=self.ball, feature_name="x", feature_val=act_x)
            next_state.set(obj=self.ball, feature_name="y", feature_val=act_y)
            next_state.set(obj=self.ball, feature_name="held", feature_val=0.0)

    def _place_object(
        self,
        *,
        next_state: State,
        act_x: float,
        act_y: float,
        obj_being_held: Object,
        ball_in_cup: bool,
        ball_only: float,
    ) -> None:
        """Port of ``_handle_placing_object``: set the held object down at (x, y),
        and carry the contained ball along when placing the cup."""
        next_state.set(obj=obj_being_held, feature_name="x", feature_val=act_x)
        next_state.set(obj=obj_being_held, feature_name="y", feature_val=act_y)
        next_state.set(obj=obj_being_held, feature_name="held", feature_val=0.0)
        if ball_in_cup and obj_being_held == self.cup and ball_only < 0.5:
            next_state.set(obj=self.ball, feature_name="x", feature_val=act_x)
            next_state.set(obj=self.ball, feature_name="y", feature_val=act_y)
            next_state.set(obj=self.ball, feature_name="held", feature_val=0.0)

    def _sample_floor_point_around_table(
        self, *, state: State, table: Object
    ) -> tuple[float, float]:
        x = state.get(obj=table, feature_name="x")
        y = state.get(obj=table, feature_name="y")
        radius = state.get(obj=table, feature_name="radius")
        dist_from_table = self.objs_scale * radius
        while True:
            dist = radius + self._noise_rng.uniform(
                radius + dist_from_table, radius + 1.15 * dist_from_table
            )
            theta = self._noise_rng.uniform(0, 2 * np.pi)
            sampled_x = x + dist * np.cos(theta)
            sampled_y = y + dist * np.sin(theta)
            if self.x_lb <= sampled_x <= self.x_ub and self.y_lb <= sampled_y <= self.y_ub:
                return float(sampled_x), float(sampled_y)

    def get_valid_actions(self) -> list[Action]:
        # Continuous 5D action space (matches predicators): no finite menu to
        # enumerate. A discrete skill layer lives in skills.py (see README), not here.
        return []

    def hard_reset(self) -> None:
        self.set_state(state=self.sample_initial_state(rng=np.random.default_rng(self.noise_seed)))

    # --- initial-state sampling (port of predicators' _get_tasks inner loop) ---

    def sample_initial_state(self, *, rng: np.random.Generator) -> State:
        assert self.num_tables >= 2
        origin_x = (self.x_ub - self.x_lb) / 2
        origin_y = (self.y_ub - self.y_lb) / 2
        d = min(self.x_ub - self.x_lb, self.y_ub - self.y_lb) / 3
        thetas = np.linspace(0, 2 * np.pi, num=self.num_tables, endpoint=False)
        angle_diff = thetas[1] - thetas[0]
        # Conservative radius so tables never overlap.
        radius = d * np.sin(angle_diff / 2) / 2
        size = radius * self.objs_scale
        sticky_region_radius = radius * self.sticky_region_radius_scale

        data: dict[Object, np.ndarray] = {}
        for i, theta in enumerate(thetas):
            x = d * np.cos(theta) + origin_x
            y = d * np.sin(theta) + origin_y
            sticky = 0.0 if i >= self.num_sticky_tables else 1.0
            prefix = "normal" if i >= self.num_sticky_tables else "sticky"
            table = Object(name=f"{prefix}-table-{i}", type=self.table_type)
            dist_from_center = rng.uniform(0.0, radius - sticky_region_radius)
            theta_from_center = rng.uniform(0.0, 2 * np.pi)
            data[table] = np.array([
                x,
                y,
                radius,
                sticky,
                dist_from_center * np.cos(theta_from_center),
                dist_from_center * np.sin(theta_from_center),
                sticky_region_radius,
            ])

        tables = tuple(sorted(data, key=lambda obj: obj.name))
        ball_table = tables[0]

        # Cup: somewhere on the floor.
        while True:
            data[self.cup] = np.array([
                rng.uniform(self.x_lb, self.x_ub),
                rng.uniform(self.y_lb, self.y_ub),
                size + 0.05 * size,  # cup must be bigger than ball
                0.0,
            ])
            state = State(data=dict(data))
            if self.on_floor(state=state, obj=self.cup):
                break

        # Ball: delicately balanced atop the ball table (intentionally hard to
        # recreate, per predicators' comment).
        table_x = float(data[ball_table][0])
        table_y = float(data[ball_table][1])
        while True:
            theta = rng.uniform(0, 2 * np.pi)
            dist = rng.uniform(0, radius)
            data[self.ball] = np.array([
                table_x + dist * np.cos(theta),
                table_y + dist * np.sin(theta),
                size - 0.05 * size,  # ball must be smaller than cup
                0.0,
            ])
            state = State(data=dict(data))
            if self.on_table(state=state, obj=self.ball, table=ball_table):
                break

        # Robot: reachable to exactly one object and not in collision (keeps the
        # domain reversible under predicators' NSRTs).
        while True:
            data[self.robot] = np.array([
                rng.uniform(self.x_lb, self.x_ub),
                rng.uniform(self.y_lb, self.y_ub),
            ])
            state = State(data=dict(data))
            if not self._invalid_robot_init_pos(state=state):
                break

        return State(data=dict(data))

    def _invalid_robot_init_pos(self, *, state: State) -> bool:
        num_reachable = 0
        for obj in (*self.get_tables(state=state), self.cup, self.ball):
            if self.is_reachable(state=state, robot=self.robot, other=obj):
                num_reachable += 1
            if num_reachable > 1:
                return True
        if num_reachable != 1:
            return True
        return self.exists_robot_collision(state=state)
