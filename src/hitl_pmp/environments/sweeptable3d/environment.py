"""KINDER open-table sweep task and paid, in-place robot/human resets."""

from typing import ClassVar

import numpy as np
from pydantic import Field

from hitl_pmp.core.problem.environment.types import Object, State, Type
from hitl_pmp.environments.sweepoffshelf3d.controllers import ControllerFailure
from hitl_pmp.environments.sweepoffshelf3d.environment import SweepOffShelf3DEnvironment
from hitl_pmp.environments.sweepoffshelf3d.scripted_skills import SkillFailed
from hitl_pmp.environments.sweepoffshelf3d.types import SweepOffShelf3DState

from .backend import SweepTable3DBackend
from .controllers import SweepTable3DControllers
from .objects import SweepTable3DObjects as O
from .skills import NOOP_ID, SKILL_IDS


class SweepTable3DEnvironment(SweepOffShelf3DEnvironment):
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

    ball_type: ClassVar[Type] = O.ball_type
    balls: ClassVar[tuple[Object, ...]] = O.balls
    fixed_bin_offset: float = Field(default=-0.18, ge=-0.4, le=0.4)

    def backend(self) -> SweepTable3DBackend:
        if self._backend is None:
            self._backend = SweepTable3DBackend(
                scene_bg=self.scene_bg,
                reset_cubes_success_probability=self.reset_cubes_success_probability,
                human_seed=self.human_seed,
                fixed_bin_offset=self.fixed_bin_offset,
            )
            self._controllers = SweepTable3DControllers(
                backend=self._backend,
                step_limit=self.skill_step_limit,
            )
        assert isinstance(self._backend, SweepTable3DBackend)
        return self._backend

    def controllers(self) -> SweepTable3DControllers:
        self.backend()
        assert isinstance(self._controllers, SweepTable3DControllers)
        return self._controllers

    def _observed_state(self, *, seed: int, steps_taken: int) -> SweepOffShelf3DState:
        backend = self.backend()
        features = backend.features()
        data = {obj: features[obj.name] for obj in O.all_objects()}
        data[O.scene] = np.array([float(seed), float(steps_taken)])
        return SweepOffShelf3DState(data=data, abstract_atoms=backend.abstract_atoms())

    def _bookkeeping(self) -> tuple[int, int]:
        state = self.get_current_state()
        return (
            int(round(state.get(obj=O.scene, feature_name="seed"))),
            int(round(state.get(obj=O.scene, feature_name="steps_taken"))),
        )

    def set_state(self, *, state: State) -> None:
        if state.get(obj=O.scene, feature_name="steps_taken") != 0:
            raise ValueError("SweepTable3D can only restore episode-initial states")
        self.reset_to_seed(seed=int(round(state.get(obj=O.scene, feature_name="seed"))))

    def _begin_motion(self, *, skill_id: int) -> None:
        self._motion_steps = 0
        names = {value: name for name, value in SKILL_IDS.items()}
        if self._motion_sink is not None and skill_id in names:
            self._motion_sink.record_skill(
                name=names[skill_id],
                objects=(),
                params=(),
                controller={},
            )

    def _dispatch(self, *, skill_id: int, values: np.ndarray) -> None:
        controllers = self.controllers()
        cube = O.cubes[int(np.clip(round(values[1]), 0, len(O.cubes) - 1))].name
        if skill_id in {
            SKILL_IDS["ReturnCubeToTableFromBox"],
            SKILL_IDS["ReturnCubeToTableFromFloor"],
        }:
            controllers.return_cube_to_table(cube=cube)
        elif skill_id == SKILL_IDS["ReturnWiper"]:
            controllers.return_wiper()
        elif self.backend().task_failed():
            raise ControllerFailure(
                "task failed: restore all balls and return the tool before retrying"
            )
        elif skill_id == SKILL_IDS["PickWiper"]:
            controllers.pick_wiper()
        elif skill_id == SKILL_IDS["SweepTable"]:
            try:
                controllers.sweep_table(angle=float(values[2]), speed=float(values[3]))
            except SkillFailed as failure:
                raise ControllerFailure(str(failure)) from failure
        elif skill_id == SKILL_IDS["PickCubeFromTable"]:
            controllers.pick_cube_from_shelf(cube=cube)
        elif skill_id == SKILL_IDS["PickCubeFromFloor"]:
            controllers.pick_cube_from_floor(cube=cube)
        elif skill_id == SKILL_IDS["PlaceCubeInBox"]:
            controllers.place_cube_in_box(cube=cube)
        elif skill_id != NOOP_ID:
            raise ControllerFailure(f"unknown skill id {skill_id}")
