from typing import ClassVar

import numpy as np
from gymnasium.spaces import Box

from hitl_pmp.core.problem.environment.environment import Environment
from hitl_pmp.core.problem.environment.types import Action, Object, State, Type


class LightSwitchEnvironment(Environment):
    """The "Light Switch" environment from the Practice Makes Perfect paper: a robot
    in a 1D grid must reach the light (in the last cell) and turn a dial until the
    light's level matches its target.

    Ported from the sibling hitl-practice repo's reference implementation,
    GridRowEnv (predicators/envs/grid_row.py) -- several numbers below come from that
    code rather than the paper's prose, which is imprecise/silent on them (the 0.1
    on-tolerance, the [0, 1] clamped dial range rather than the paper text's circular
    [0, 2*pi], grid_size=100). See the Notion page's "Details not in paper but in
    codebase" section for the full list.

    Cell objects (get_cells()) exist for skills.py's MoveRobot/JumpToLight skills and
    predicates.py's RobotInCell/LightInCell/Adjacent, matching GridRowEnv's own model
    -- cell membership itself is still just an x-position comparison (same_position),
    not a separate "in cell" mechanism the raw dynamics below need to know about. The
    raw action is [dx, dlight] (continuous, unbounded), matching GridRowEnv.action_space
    exactly -- the paper's MoveTo/ToggleLight/JumpToLight "skills" are a layer on top
    of this raw action space (predicators' ParameterizedOptions, ported in skills.py),
    not part of the environment itself.

    grid_size/canonical_light_target are real per-instance fields (constructor
    arguments, e.g. LightSwitchEnvironment(grid_size=10)) -- genuine per-run
    configuration, unlike robot_type/light_type/cell_type/robot/light/
    light_on_tolerance/same_position_tolerance/action_space below, which stay
    ClassVar. Those specifically stay ClassVar (not just for convenience) because
    predicates.py's module-level Predicate objects are built once at import time
    and their `holds` closures read light_on_tolerance/same_position_tolerance via
    a late-bound LightSwitchEnvironment class lookup -- Predicate.holds has a fixed
    (state, objects) signature (reused by Goal.is_satisfied and
    planning/grounding.py's SkillGrounder, both meant to stay environment-instance
    agnostic), so there's no per-call slot to pass an Environment instance through
    even if these became instance fields. robot_type/light_type/cell_type/robot/
    light/action_space are genuine structural constants regardless (same for every
    instance that will ever exist), so they'd stay ClassVar either way.
    """

    grid_size: int = 100  # predicators' settings.grid_row_num_cells default
    # hard_reset's non-random light_target. Chosen as the lower edge of Tasks' real
    # Uniform(0.5, 1.0) sampling range so hard_reset's canonical state (level=0.0) is
    # never trivially already "on".
    canonical_light_target: float = 0.5

    light_on_tolerance: ClassVar[float] = 0.1  # GridRowEnv._LightOn_holds
    same_position_tolerance: ClassVar[float] = 1e-3  # GridRowEnv._In_holds

    robot_type: ClassVar[Type] = Type(name="robot", feature_names=("x",))
    light_type: ClassVar[Type] = Type(name="light", feature_names=("level", "target", "x"))
    cell_type: ClassVar[Type] = Type(name="cell", feature_names=("x",))

    robot: ClassVar[Object] = Object(name="robot", type=robot_type)
    light: ClassVar[Object] = Object(name="light", type=light_type)

    action_space: ClassVar[Box] = Box(-np.inf, np.inf, (2,))

    def get_cells(self) -> tuple[Object, ...]:
        """One Object per grid position, x = i + 0.5 (matching GridRowEnv's
        _get_tasks). Built fresh on every call, not cached -- grid_size can differ
        between instances (CLI override, test overrides), so caching risks a stale
        value the same way LightSwitchProblem.max_episode_steps() already has to
        avoid. Object equality/hash are value-based (frozen pydantic), so rebuilding
        is correct, just not free -- negligible at this scale."""
        return tuple(Object(name=f"cell{i}", type=self.cell_type) for i in range(self.grid_size))

    def build_initial_state(self, *, light_level: float, light_target: float) -> State:
        """The robot always starts in cell 0; the light always sits in the last cell.
        Only light_level/light_target vary between callers -- hard_reset uses a
        canonical value, Tasks samples light_target per episode."""
        data: dict[Object, np.ndarray] = {
            self.robot: np.array([0.5]),
            self.light: np.array([light_level, light_target, self.grid_size - 0.5]),
        }
        for i, cell in enumerate(self.get_cells()):
            data[cell] = np.array([i + 0.5])
        return State(data=data)

    def take_action(self, *, action: Action) -> State:
        dx, dlight = float(action[0]), float(action[1])
        state = self.get_current_state()
        next_state = state.model_copy(deep=True)

        if LightSwitchEnvironment.same_position(state=state, obj1=self.robot, obj2=self.light):
            new_level = float(
                np.clip(state.get(obj=self.light, feature_name="level") + dlight, 0.0, 1.0)
            )
            next_state.set(obj=self.light, feature_name="level", feature_val=new_level)

        new_x = float(
            np.clip(state.get(obj=self.robot, feature_name="x") + dx, 0.0, self.grid_size)
        )
        next_state.set(obj=self.robot, feature_name="x", feature_val=new_x)

        self.set_state(state=next_state)
        return next_state

    def get_valid_actions(self) -> list[Action]:
        # The action space is continuous and unbounded (matches GridRowEnv) -- every
        # action in action_space is valid, so there is no finite/enumerable list to
        # return. Revisit once a Method needs a discrete action menu (e.g. a skill
        # layer on top, like GridRowEnv's MoveTo/ToggleLight/JumpToLight options).
        return []

    def hard_reset(self) -> None:
        self.set_state(
            state=self.build_initial_state(
                light_level=0.0, light_target=self.canonical_light_target
            )
        )

    @staticmethod
    def same_position(*, state: State, obj1: Object, obj2: Object) -> bool:
        """Whether obj1 and obj2 are at the same x position within tolerance --
        shared by take_action's light co-location check and predicates.py's
        RobotInCell/LightInCell (not just an internal detail of one method, hence
        public). Stays a staticmethod (not part of the Environment ABC, so free to
        differ in kind from take_action/hard_reset/etc.): it only reads
        same_position_tolerance, which stays ClassVar (see this class's own
        docstring), so there's no instance state for it to need self for."""
        x1 = state.get(obj=obj1, feature_name="x")
        x2 = state.get(obj=obj2, feature_name="x")
        return bool(abs(x1 - x2) < LightSwitchEnvironment.same_position_tolerance)
