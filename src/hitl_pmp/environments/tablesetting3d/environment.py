"""Project Environment adapter; native KinDER owns all transitions and costs."""

from typing import Any, ClassVar

import numpy as np
from gymnasium.spaces import Box
from pydantic import Field, PrivateAttr

from hitl_pmp.core.problem.environment.environment import Environment
from hitl_pmp.core.problem.environment.types import Action, Object, State, Type

from .kinder_backend import KinderBackend


class TableSetting3DEnvironment(Environment):
    """An explicit costed skill interface, with no automatic practice resets.

    Action [0..3] sets a plate; [4..7] returns one plate; [8] asks the human.
    ``last_info`` reports realised seconds and observed human success/failure.
    Human failures are real state changes, never relabelled successful resets.
    """

    human_competence: float = Field(default=0.8, ge=0, le=1, allow_inf_nan=False)
    seed: int = Field(default=0, ge=0)
    action_space: ClassVar[Box] = Box(0, 9, shape=(1,), dtype=np.float32)
    robot_type: ClassVar[Type] = Type(name="robot", feature_names=("x", "y"))
    plate_type: ClassVar[Type] = Type(name="plate", feature_names=("x", "y", "z"))
    robot: ClassVar[Object] = Object(name="robot", type=robot_type)
    plates: ClassVar[tuple[Object, ...]] = tuple(
        Object(name=f"plate_{i}", type=Type(name="plate", feature_names=("x", "y", "z")))
        for i in range(4)
    )
    _backend: Any = PrivateAttr(default=None)
    _last_info: dict[str, Any] = PrivateAttr(default_factory=dict)

    @property
    def backend(self) -> Any:
        if self._backend is None:
            self._backend = KinderBackend.create(human_competence=self.human_competence)
            # A harness may restore a task before calling hard_reset. Seed that
            # first-use path too, without requiring a simulator render.
            self._backend.reset(seed=self.seed)
        return self._backend

    @property
    def last_info(self) -> dict[str, Any]:
        return self._last_info.copy()

    @classmethod
    def decode_state(cls, *, observation: np.ndarray) -> State:
        return State(
            data={
                cls.robot: observation[:2].copy(),
                **{
                    plate: observation[2 + 3 * i : 5 + 3 * i].copy()
                    for i, plate in enumerate(cls.plates)
                },
            }
        )

    @classmethod
    def encode_state(cls, *, state: State) -> np.ndarray:
        if set(state.data) != {cls.robot, *cls.plates}:
            raise ValueError("State must contain the robot and the four named plates")
        return np.concatenate([state[cls.robot], *(state[p] for p in cls.plates)]).astype(
            np.float32
        )

    def hard_reset(self) -> None:
        observation, self._last_info = self.backend.reset(seed=self.seed)
        self.current_state = self.decode_state(observation=observation)

    def set_state(self, *, state: State) -> None:
        self.backend.set_state(self.encode_state(state=state))
        self.current_state = state.model_copy(deep=True)
        # Restoring physical state is not another observation of the last action.
        self._last_info = {}

    def take_action(self, *, action: Action) -> State:
        action = np.asarray(action)
        if (
            action.shape != (1,)
            or not np.isfinite(action).all()
            or not 0 <= action[0] <= 9
            or action[0] != int(action[0])
        ):
            raise ValueError("Expected one integer skill ID in [0, 9]")
        observation, _, _, _, self._last_info = self.backend.step(int(action[0]))
        self.current_state = self.decode_state(observation=observation)
        return self.current_state

    def get_valid_actions(self) -> list[Action]:
        return [np.array([i], dtype=np.float32) for i in self.backend.valid_actions()]

    def noop_action(self) -> Action:
        return np.array([9], dtype=np.float32)

    def close(self) -> None:
        if self._backend is not None:
            self._backend.close()
