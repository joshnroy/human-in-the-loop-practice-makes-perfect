"""`RealTossing3DEnvironment`: the sim's Tossing3D domain, on the real robot.

A subclass of `Tossing3DEnvironment` with the same objects, predicates, skills and
operator model. It overrides only where the world comes from (`build_backend`), the
toss's parameter set (the real swing's), and what a human reset hands the backend (a
named spot instead of a sampled region).
"""

from typing import Any

from pydantic import ConfigDict

from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.layout import Tossing3DLayout
from hitl_pmp.environments.tossing3d.sides import Tossing3DSide
from hitl_pmp.environments.tossing3d.types import Tossing3DBackend, TossParameterSet
from hitl_pmp.environments.tossing3d.wide_long_range_proposal import REAL_TOSS_PARAMETERS

from .backend import RealSkillStepCosts, RealTossingBackend
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

    def build_backend(self) -> Tossing3DBackend:
        if self.layout is not Tossing3DLayout.BARRIER:
            raise ValueError("the real room has only the barrier layout")
        return RealTossingBackend(
            robot=self.tossing_robot,
            person=self.person,
            parameters=self.toss_parameters or REAL_TOSS_PARAMETERS,
            step_costs=self.step_costs,
        )

    def hard_reset(self) -> None:
        if not self.evaluation:
            super().hard_reset()

    def bin_reset_region(self, *, destination: str | None) -> dict[str, object] | None:
        """The side's one spot, by name: `RealTossingBackend.reset_cube_and_bin` asks the
        person to put the bin there."""
        if destination is None:
            return None
        return {"destination": Tossing3DSide(destination).value}
