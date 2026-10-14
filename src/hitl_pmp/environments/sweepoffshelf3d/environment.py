"""`SweepOffShelf3DEnvironment` -- KINDER's `SweepOffShelf3D-o5` behind `core.Environment`.

A TidyBot sweeps five cubes off a cupboard shelf, over its open front edge, into a box on
the floor that is too heavy for it to lift. A human can pour the box back onto the shelf
(`reset_cubes`, which succeeds -- all five cubes -- with probability p and otherwise moves
nothing) and put the wiper back (`reset_wiper`, which always succeeds).

One `take_action` runs one whole skill -- thousands of MuJoCo control steps -- through
`controllers.py`. The action vector is ``[skill_id, cube_index, height, speed]``: the cube
slot is read only by the five single-cube skills, the last two only by `SweepShelf`.

As in Tossing3D, `set_state` can only restore an episode-initial state: MuJoCo's full
state does not fit in a flat feature vector, so a state with ``steps_taken > 0`` is
refused rather than silently replaced by its episode's start.
"""

import logging
from collections.abc import Callable
from typing import ClassVar

import numpy as np
from gymnasium.spaces import Box
from pydantic import ConfigDict, Field, PrivateAttr

from hitl_pmp.core.problem.environment.environment import Environment
from hitl_pmp.core.problem.environment.types import Action, Object, State, Type

from .backend import SweepOffShelf3DBackend
from .controllers import ControllerFailure, SweepOffShelf3DControllers
from .objects import SweepOffShelf3DObjects as O
from .skills import RESET_CUBES_NAME, RESET_WIPER_NAME
from .types import HumanEvent, SweepOffShelf3DState

logger = logging.getLogger(__name__)

SKILL_IDS: dict[str, int] = {
    "PickWiper": 0,
    "SweepShelf": 1,
    "ReturnWiper": 2,
    "PickCubeFromShelf": 3,
    "PickCubeFromFloor": 4,
    "PlaceCubeInBox": 5,
    "ReturnCubeToShelfFromBox": 6,
    "ReturnCubeToShelfFromFloor": 7,
}
NOOP_ID = -1


class SweepOffShelf3DEnvironment(Environment):
    """Concrete `core.Environment` over a live KINDER scene; constructing one imports
    nothing, so the symbolic layer stays testable without MuJoCo."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    robot_type: ClassVar[Type] = O.robot_type
    cube_type: ClassVar[Type] = O.cube_type
    wiper_type: ClassVar[Type] = O.wiper_type
    box_type: ClassVar[Type] = O.box_type
    scene_type: ClassVar[Type] = O.scene_type
    robot: ClassVar[Object] = O.robot
    cubes: ClassVar[tuple[Object, ...]] = O.cubes
    wiper: ClassVar[Object] = O.wiper
    box: ClassVar[Object] = O.box
    scene: ClassVar[Object] = O.scene

    action_space: ClassVar[Box] = Box(-np.inf, np.inf, (4,))

    scene_bg: bool = True
    canonical_seed: int = 0
    # p: the human's true competence at `reset_cubes`.
    reset_cubes_success_probability: float = Field(default=1.0, ge=0.0, le=1.0)
    human_seed: int = 0
    # Control steps one skill may take before it is cut off as a failure.
    skill_step_limit: int = 12000

    _backend: SweepOffShelf3DBackend | None = PrivateAttr(default=None)
    _controllers: SweepOffShelf3DControllers | None = PrivateAttr(default=None)
    _last_skill_error: str | None = PrivateAttr(default=None)
    _last_controller_steps: int = PrivateAttr(default=0)
    _human_events: list[HumanEvent] = PrivateAttr(default_factory=list)

    def backend(self) -> SweepOffShelf3DBackend:
        if self._backend is None:
            self._backend = SweepOffShelf3DBackend(
                scene_bg=self.scene_bg,
                reset_cubes_success_probability=self.reset_cubes_success_probability,
                human_seed=self.human_seed,
            )
            self._controllers = SweepOffShelf3DControllers(
                backend=self._backend, step_limit=self.skill_step_limit
            )
        return self._backend

    def controllers(self) -> SweepOffShelf3DControllers:
        self.backend()
        assert self._controllers is not None
        return self._controllers

    def last_skill_error(self) -> str | None:
        return self._last_skill_error

    def last_controller_steps(self) -> int:
        return self._last_controller_steps

    def human_events(self) -> tuple[HumanEvent, ...]:
        return tuple(self._human_events)

    # --- states -----------------------------------------------------------

    def _observed_state(self, *, seed: int, steps_taken: int) -> SweepOffShelf3DState:
        backend = self.backend()
        features = backend.features()
        data = {obj: features[obj.name] for obj in (O.robot, *O.cubes, O.wiper, O.box)}
        data[O.scene] = np.array([float(seed), float(steps_taken)])
        return SweepOffShelf3DState(data=data, abstract_atoms=backend.abstract_atoms())

    def _bookkeeping(self) -> tuple[int, int]:
        state = self.get_current_state()
        seed = int(round(state.get(obj=O.scene, feature_name="seed")))
        steps = int(round(state.get(obj=O.scene, feature_name="steps_taken")))
        return seed, steps

    # --- core.Environment --------------------------------------------------

    def take_action(self, *, action: Action) -> State:
        """Run one whole skill. Total over the action space: an unknown id or a skill
        whose controller gives up is a transition that still advances `steps_taken`;
        whatever the arm did before giving up stays done."""
        seed, steps_taken = self._bookkeeping()
        self._last_skill_error = None
        self._last_controller_steps = 0
        controllers = self.controllers()
        controllers.reset_counter()
        values = np.asarray(action, dtype=float)
        try:
            if values.shape != (4,) or not np.all(np.isfinite(values)):
                raise ControllerFailure(f"malformed action {action!r}")
            self._dispatch(skill_id=int(round(values[0])), values=values)
        except ControllerFailure as failure:
            self._last_skill_error = str(failure)
            logger.debug("skill failed: %s", failure)
        self._last_controller_steps = controllers.steps
        state = self._observed_state(seed=seed, steps_taken=steps_taken + 1)
        self.current_state = state
        return state

    def _dispatch(self, *, skill_id: int, values: np.ndarray) -> None:
        controllers = self.controllers()
        cube = O.cubes[int(np.clip(round(values[1]), 0, len(O.cubes) - 1))].name
        if skill_id == SKILL_IDS["PickWiper"]:
            controllers.pick_wiper()
        elif skill_id == SKILL_IDS["SweepShelf"]:
            controllers.sweep_shelf(height=float(values[2]), speed=float(values[3]))
        elif skill_id == SKILL_IDS["ReturnWiper"]:
            controllers.return_wiper()
        elif skill_id == SKILL_IDS["PickCubeFromShelf"]:
            controllers.pick_cube_from_shelf(cube=cube)
        elif skill_id == SKILL_IDS["PickCubeFromFloor"]:
            controllers.pick_cube_from_floor(cube=cube)
        elif skill_id == SKILL_IDS["PlaceCubeInBox"]:
            controllers.place_cube_in_box(cube=cube)
        elif skill_id in {
            SKILL_IDS["ReturnCubeToShelfFromBox"],
            SKILL_IDS["ReturnCubeToShelfFromFloor"],
        }:
            controllers.return_cube_to_shelf(cube=cube)
        elif skill_id != NOOP_ID:
            raise ControllerFailure(f"unknown skill id {skill_id}")

    def set_control_step_observers(
        self, *, before: Callable[[], None] | None, after: Callable[[], None] | None
    ) -> None:
        """Observe every KINDER control step a skill takes. A `ControlStepLimitReached`
        raised from `before` is a `BaseException`, so no skill's error handling catches
        it: it propagates out of `take_action` and the caller adopts the cut-off skill."""
        self.backend().set_step_observers(before=before, after=after)

    def adopt_interrupted_skill(self, *, action: Action, reason: str) -> State:
        """Account for a skill an external step budget stopped part-way through: record
        the interruption as its outcome and adopt the scene as the cut-off left it, the
        bookkeeping `take_action` never reached."""
        del action
        seed, steps_taken = self._bookkeeping()
        self._last_skill_error = f"interrupted: {reason}"
        controllers = self.controllers()
        self._last_controller_steps = controllers.steps
        controllers.on_objects_moved()
        state = self._observed_state(seed=seed, steps_taken=steps_taken + 1)
        self.current_state = state
        return state

    def get_valid_actions(self) -> list[Action]:
        return []

    def noop_action(self) -> Action:
        return np.array([float(NOOP_ID), 0.0, 0.0, 0.0])

    def set_state(self, *, state: State) -> None:
        steps_taken = int(round(state.get(obj=O.scene, feature_name="steps_taken")))
        if steps_taken != 0:
            raise ValueError(
                "SweepOffShelf3DEnvironment.set_state can only restore an episode-initial "
                f"state, but this one has steps_taken={steps_taken}"
            )
        self.reset_to_seed(seed=int(round(state.get(obj=O.scene, feature_name="seed"))))

    def hard_reset(self) -> None:
        self.reset_to_seed(seed=self.canonical_seed)

    def reset_to_seed(self, *, seed: int) -> State:
        self.backend().reset(seed=seed)
        self.controllers().on_scene_reset()
        state = self._observed_state(seed=seed, steps_taken=0)
        self.current_state = state
        return state

    def reset_movables(self, *, destination: str | None = None) -> bool:
        """The human's two skills, named by `destination`: `reset_cubes` or
        `reset_wiper`. The robot is untouched; `steps_taken` carries forward. True
        whenever the human attempted the skill, failed or not: a failure is an
        outcome the robot scores, not a refusal (`moved` is then empty)."""
        seed, steps_taken = self._bookkeeping()
        backend = self.backend()
        if destination == RESET_CUBES_NAME:
            moved = tuple(backend.reset_cubes())
        elif destination == RESET_WIPER_NAME:
            backend.reset_wiper()
            moved = (O.wiper.name,)
        else:
            raise ValueError(f"unknown human skill {destination!r}")
        self.controllers().on_objects_moved()
        self._human_events.append(
            HumanEvent(skill=destination, moved=moved, steps_taken=steps_taken)
        )
        self.current_state = self._observed_state(seed=seed, steps_taken=steps_taken)
        return True

    def close(self) -> None:
        if self._backend is not None:
            self._backend.close()
