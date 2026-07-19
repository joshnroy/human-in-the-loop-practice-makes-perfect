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

    Unlike GridRowEnv, this doesn't model grid cells as their own State Objects --
    GridRowEnv only needs them for planner-grounding predicates (RobotInCell,
    LightInCell, Adjacent), and this port's scope stops at LightOn (see
    predicates.py), so cell membership is just an x-position comparison
    (_same_position) rather than a symbolic Cell type. The raw action is
    [dx, dlight] (continuous, unbounded), matching GridRowEnv.action_space exactly --
    the paper's MoveTo/ToggleLight/JumpToLight "skills" are a layer on top of this raw
    action space (predicators' ParameterizedOptions), not part of the environment
    itself; they belong to a future Method, not here.
    """

    grid_size: ClassVar[int] = 100  # predicators' settings.grid_row_num_cells default
    light_on_tolerance: ClassVar[float] = 0.1  # GridRowEnv._LightOn_holds
    same_position_tolerance: ClassVar[float] = 1e-3  # GridRowEnv._In_holds
    # hard_reset's non-random light_target. Chosen as the lower edge of Tasks' real
    # Uniform(0.5, 1.0) sampling range so hard_reset's canonical state (level=0.0) is
    # never trivially already "on".
    canonical_light_target: ClassVar[float] = 0.5

    robot_type: ClassVar[Type] = Type(name="robot", feature_names=("x",))
    light_type: ClassVar[Type] = Type(name="light", feature_names=("level", "target", "x"))

    robot: ClassVar[Object] = Object(name="robot", type=robot_type)
    light: ClassVar[Object] = Object(name="light", type=light_type)

    action_space: ClassVar[Box] = Box(-np.inf, np.inf, (2,))

    @staticmethod
    def build_initial_state(*, light_level: float, light_target: float) -> State:
        """The robot always starts in cell 0; the light always sits in the last cell.
        Only light_level/light_target vary between callers -- hard_reset uses a
        canonical value, Tasks samples light_target per episode."""
        env = LightSwitchEnvironment
        return State(
            data={
                env.robot: np.array([0.5]),
                env.light: np.array([light_level, light_target, env.grid_size - 0.5]),
            }
        )

    @staticmethod
    def take_action(*, action: Action) -> State:
        env = LightSwitchEnvironment
        dx, dlight = float(action[0]), float(action[1])
        state = env.current_state
        next_state = state.model_copy(deep=True)

        if env._same_position(state=state, obj1=env.robot, obj2=env.light):
            new_level = float(
                np.clip(state.get(obj=env.light, feature_name="level") + dlight, 0.0, 1.0)
            )
            next_state.set(obj=env.light, feature_name="level", feature_val=new_level)

        new_x = float(np.clip(state.get(obj=env.robot, feature_name="x") + dx, 0.0, env.grid_size))
        next_state.set(obj=env.robot, feature_name="x", feature_val=new_x)

        env.set_state(state=next_state)
        return next_state

    @staticmethod
    def get_valid_actions() -> list[Action]:
        # The action space is continuous and unbounded (matches GridRowEnv) -- every
        # action in action_space is valid, so there is no finite/enumerable list to
        # return. Revisit once a Method needs a discrete action menu (e.g. a skill
        # layer on top, like GridRowEnv's MoveTo/ToggleLight/JumpToLight options).
        return []

    @staticmethod
    def hard_reset() -> None:
        LightSwitchEnvironment.set_state(
            state=LightSwitchEnvironment.build_initial_state(
                light_level=0.0, light_target=LightSwitchEnvironment.canonical_light_target
            )
        )

    @staticmethod
    def _same_position(*, state: State, obj1: Object, obj2: Object) -> bool:
        x1 = state.get(obj=obj1, feature_name="x")
        x2 = state.get(obj=obj2, feature_name="x")
        return bool(abs(x1 - x2) < LightSwitchEnvironment.same_position_tolerance)
