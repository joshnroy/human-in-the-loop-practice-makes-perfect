import numpy as np

from hitl_pmp.core.method.skill_provider import OraclePolicyProvider, SkillProvider
from hitl_pmp.core.method.types import GroundSkill, LabeledAction, Skill
from hitl_pmp.core.problem.environment.types import Action, Object, State, Type
from hitl_pmp.core.problem.tasks.types import Goal, Predicate

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
from .skills import BallRingSkills


class BallRingSkillProvider(SkillProvider):
    """The Ball-Ring domain's `SkillProvider`: exposes `BallRingSkills` and this
    domain's 12 predicates / 4 types / (robot, ball, cup + ring of tables) objects to
    a domain-agnostic Method (EesMethod/RandomSkillsMethod), and delegates
    `sample_params`/`compute_action` to `BallRingSkills` (passing the env instance
    that its navigation needs for collision checks)."""

    env: BallRingEnvironment

    def skills(self) -> tuple[Skill, ...]:
        return BallRingSkills.all_skills()

    def predicates(self) -> tuple[Predicate, ...]:
        return (
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

    def types(self) -> tuple[Type, ...]:
        return (
            BallRingEnvironment.robot_type,
            BallRingEnvironment.ball_type,
            BallRingEnvironment.cup_type,
            BallRingEnvironment.table_type,
        )

    def objects(self) -> tuple[Object, ...]:
        return self.env.all_objects()

    def sample_params(self, *, ground_skill: GroundSkill, rng: np.random.Generator) -> np.ndarray:
        return BallRingSkills.sample_params(ground_skill=ground_skill, rng=rng)

    def compute_action(
        self, *, ground_skill: GroundSkill, params: np.ndarray, state: State
    ) -> Action:
        return BallRingSkills.compute_action(
            ground_skill=ground_skill, params=params, state=state, env=self.env
        )

    def oracle_sampler_input(
        self, *, ground_skill: GroundSkill, state: State, params: np.ndarray
    ) -> list[float] | None:
        """Route the domain-agnostic Method's oracle-feature-selection hook to
        `BallRingSkills` -- a curated sampler input row for the cup-placement skill,
        `None` (fall back to `"all"`) for everything else. This is the seam that lets
        Ball-Ring reproduce the paper's `active_sampler_learning_feature_selection:
        oracle` without the Method importing this environment."""
        return BallRingSkills.oracle_sampler_input(
            ground_skill=ground_skill, state=state, params=params
        )


class BallRingOracle(OraclePolicyProvider):
    """Ball-Ring's privileged solver, driving `SkillOracleMethod` as the upper-bound
    baseline. Reads ground-truth state and, greedily each step, takes the next raw
    action toward `BallOnTable(ball, target)`. The only route (there is no
    place-cup-with-ball-on-table skill) is: put the *empty* cup on the sticky target's
    safe sub-region, hold the ball, then place the ball into that on-table cup. That
    is an 8-action plan -- exactly the paper's `H_eval`=8 horizon -- so the oracle
    solves at the horizon with no slack, which is itself informative about how tight
    the evaluation budget is on this domain."""

    env: BallRingEnvironment

    def get_labeled_action(self, *, state: State, goal: Goal) -> LabeledAction:
        del goal  # Ball-Ring always drives toward BallOnTable(ball, target) from state
        env = self.env
        target = env.target_table()
        ball, cup, robot = env.ball, env.cup, env.robot

        cup_on_target = env.on_table(state=state, obj=cup, table=target)
        holding_cup = env.holding(state=state, obj=cup)
        holding_ball = env.holding(state=state, obj=ball)
        reachable_target = env.is_reachable(state=state, robot=robot, other=target)

        # Phase 1: get the empty cup onto the sticky target table's safe region.
        if not cup_on_target:
            if holding_cup:
                if reachable_target:
                    return self._place_cup_on_safe_region(state=state, target=target)
                return self._navigate(state=state, target=target, label="NavigateToTable(target)")
            # Not holding the cup: pick it up (it starts on the floor, ball not in it).
            if env.is_reachable(state=state, robot=robot, other=cup):
                return self._pick(state=state, obj=cup, label="PickCupWithoutBallFromFloor")
            return self._navigate(state=state, target=cup, label="NavigateToCup")

        # Phase 2/3: cup is on the target; get the ball and drop it into that cup.
        if not holding_ball:
            ball_table = self._table_holding(state=state, obj=ball)
            if ball_table is not None:
                if env.is_reachable(state=state, robot=robot, other=ball_table):
                    return self._pick(state=state, obj=ball, label="PickBallFromTable")
                return self._navigate(state=state, target=ball_table, label="NavigateToTable(ball)")
            # Ball on the floor.
            if env.is_reachable(state=state, robot=robot, other=ball):
                return self._pick(state=state, obj=ball, label="PickBallFromFloor")
            return self._navigate(state=state, target=ball, label="NavigateToBall")

        if reachable_target:
            # Place the held ball into the on-target cup (obj_type_id 2 = place in cup).
            return LabeledAction(
                action=np.array([
                    1.0,
                    2.0,
                    0.0,
                    state.get(obj=cup, feature_name="x"),
                    state.get(obj=cup, feature_name="y"),
                ]),
                label="PlaceBallInCupOnTable",
            )
        return self._navigate(state=state, target=target, label="NavigateToTable(target)")

    def _table_holding(self, *, state: State, obj: Object) -> Object | None:
        for table in self.env.get_tables(state=state):
            if self.env.on_table(state=state, obj=obj, table=table):
                return table
        return None

    def _navigate(self, *, state: State, target: Object, label: str) -> LabeledAction:
        return LabeledAction(
            action=BallRingSkills.navigate_action(state=state, env=self.env, target=target),
            label=label,
        )

    def _pick(self, *, state: State, obj: Object, label: str) -> LabeledAction:
        obj_type_id = 1.0 if obj.type == self.env.ball_type else 2.0
        return LabeledAction(
            action=np.array([
                1.0,
                obj_type_id,
                0.0,
                state.get(obj=obj, feature_name="x"),
                state.get(obj=obj, feature_name="y"),
            ]),
            label=label,
        )

    def _place_cup_on_safe_region(self, *, state: State, target: Object) -> LabeledAction:
        """Place the cup at a point that is both inside the sticky safe sub-region
        (so it does not fall) and within the cup-fits-on-table disk (so CupOnTable
        holds). The safe-region *center* can lie beyond the placeable disk, so aim
        along the direction to it but cap the distance at the placeable maximum -- the
        capped point stays within the safe region's radius (verified by the geometry:
        the cap loss is at most ~0.18r < the safe radius 0.35r)."""
        env = self.env
        tx = state.get(obj=target, feature_name="x")
        ty = state.get(obj=target, feature_name="y")
        table_radius = state.get(obj=target, feature_name="radius")
        cup_radius = state.get(obj=env.cup, feature_name="radius")
        sox = state.get(obj=target, feature_name="sticky_region_x_offset")
        soy = state.get(obj=target, feature_name="sticky_region_y_offset")
        safe_dist = float(np.hypot(sox, soy))
        placeable_max = max(table_radius - 2 * cup_radius, 0.0)
        if safe_dist <= 1e-9:
            x, y = tx, ty
        else:
            dist = min(safe_dist, placeable_max)
            x = tx + dist * sox / safe_dist
            y = ty + dist * soy / safe_dist
        return LabeledAction(
            action=np.array([1.0, 3.0, 0.0, x, y]), label="PlaceCupWithoutBallOnTable(safe)"
        )
