"""`RealTossing3DEnvironment`: the sim's Tossing3D domain, on the real robot.

A subclass of `Tossing3DEnvironment` with the same objects, predicates, skills and
operator model. It overrides only where the world comes from (`build_backend`), the
toss's parameter set (the real swing's), and what a human reset hands the backend (a
named spot instead of a sampled region).
"""

from typing import Any

from pydantic import ConfigDict

from hitl_pmp.core.problem.environment.types import State
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.layout import Tossing3DLayout
from hitl_pmp.environments.tossing3d.sides import Tossing3DSide
from hitl_pmp.environments.tossing3d.types import (
    KB_PICKUP_BLOCKED,
    Tossing3DBackend,
    Tossing3DState,
    TossParameterSet,
)
from hitl_pmp.environments.tossing3d.wide_long_range_proposal import REAL_TOSS_PARAMETERS

from .backend import PracticeScene, RealSkillStepCosts, RealTossingBackend
from .contract import TossingRobot


class RealTossing3DEnvironment(Tossing3DEnvironment):
    """`Tossing3DEnvironment` over a `RealTossingBackend` on `tossing_robot`, with `person`
    placing things. The barrier layout is the only one: the room has a tape."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tossing_robot: TossingRobot
    person: Any
    step_costs: RealSkillStepCosts = RealSkillStepCosts()
    toss_parameters: TossParameterSet | None = REAL_TOSS_PARAMETERS
    # Evaluation's episodes each open with `reset_to_task`, which asks for the scene, so
    # its `hard_reset` would ask for the same scene once more for nothing.
    evaluation: bool = False
    confirm_start: bool = False
    confirm_each_skill: bool = False

    def build_backend(self) -> Tossing3DBackend:
        if self.layout is not Tossing3DLayout.BARRIER:
            raise ValueError("the real room has only the barrier layout")
        return RealTossingBackend(
            robot=self.tossing_robot,
            person=self.person,
            parameters=self.toss_parameters or REAL_TOSS_PARAMETERS,
            step_costs=self.step_costs,
            confirm_start=self.confirm_start,
            confirm_each_skill=self.confirm_each_skill,
        )

    def hard_reset(self) -> None:
        if not self.evaluation:
            super().hard_reset()

    def reobserve(self, *, state: State) -> None:
        """Carry on from a resumed run's last practice `state`, in the room as it is now.

        Nothing about the world is restored or asked for: the scene is observed afresh,
        whatever was moved meanwhile, and the planner sees it as found. `state` supplies
        only what the room cannot show -- the episode bookkeeping (`seed`,
        `steps_taken`) and the observed pickup-refusal channel.
        """
        assert isinstance(state, Tossing3DState)
        self._pickup_blocked = (KB_PICKUP_BLOCKED, (self.cube.name,)) in (
            state.abstract_atoms or frozenset()
        )
        self._observe_afresh(state=state)

    def bin_reset_region(self, *, destination: str | None) -> dict[str, object] | None:
        """The side's one spot, by name: `RealTossingBackend.reset_cube_and_bin` asks the
        person to put the bin there."""
        if destination is None:
            return None
        return {"destination": Tossing3DSide(destination).value}

    def real_backend(self) -> RealTossingBackend:
        backend = self.backend()
        assert isinstance(backend, RealTossingBackend)
        return backend

    def restore_practice_scene(self, *, scene: PracticeScene) -> None:
        """Have the person put practice's `scene` back after an evaluation used the room.
        Not a skill or a reset the method chose: no step is charged, and the episode
        bookkeeping and the pickup-refusal channel carry over."""
        state = self.get_current_state()
        self.real_backend().restore_practice_scene(scene=scene)
        self._observe_afresh(state=state)

    def _observe_afresh(self, *, state: State) -> None:
        """Adopt the room as observed now, keeping `state`'s episode bookkeeping."""
        seed = int(round(state.get(obj=self.scene, feature_name="seed")))
        steps_taken = int(round(state.get(obj=self.scene, feature_name="steps_taken")))
        self._adopt(state=self._observed_state(seed=seed, steps_taken=steps_taken))
