"""`RealTossing3DEnvironment`: the sim's Tossing3D domain, on the real robot.

Same objects, predicates, skills and operator model as `Tossing3DEnvironment` -- it is
a subclass, and overrides only where the world comes from (`build_backend`), the toss's
parameter set, and where a human reset puts the bin (the real room's regions instead of
the sim's).
"""

from pydantic import ConfigDict

from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.layout import Tossing3DLayout
from hitl_pmp.environments.tossing3d.sides import Tossing3DSide
from hitl_pmp.environments.tossing3d.types import Tossing3DBackend, TossParameterSet
from hitl_pmp.environments.tossing3d.wide_long_range_proposal import REAL_TOSS_PARAMETERS

from .backend import RealTossingBackend, RealWorld
from .types import RealSceneLayout


class RealTossing3DEnvironment(Tossing3DEnvironment):
    """`Tossing3DEnvironment` whose backend is a `RealTossingBackend` on `world`.

    `owner` names this environment to the shared `RealWorld` ("practice" or
    "evaluation"), which is how a scene one of them left is restored before the other
    acts. The barrier layout is the only one: the real room has a barrier and no
    same-side scene.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    world: RealWorld
    scene_layout: RealSceneLayout = RealSceneLayout()
    owner: str = "practice"
    toss_parameters: TossParameterSet | None = REAL_TOSS_PARAMETERS
    layout: Tossing3DLayout = Tossing3DLayout.BARRIER

    def build_backend(self) -> Tossing3DBackend:
        if self.layout is not Tossing3DLayout.BARRIER:
            raise ValueError("the real room has only the barrier layout")
        return RealTossingBackend(
            world=self.world,
            layout=self.scene_layout,
            parameters=self.toss_parameters or REAL_TOSS_PARAMETERS,
            owner=self.owner,
        )

    def hard_reset(self) -> None:
        """Practice: the person sets up the canonical scene. Evaluation: nothing --
        every evaluation episode opens with `reset_to_task`, which asks for its own
        scene, so a canonical one first would be one more set-up for nothing."""
        if self.owner == "evaluation":
            return
        super().hard_reset()

    def real_backend(self) -> RealTossingBackend:
        backend = self.backend()
        assert isinstance(backend, RealTossingBackend)
        return backend

    def reset_movables(self, *, destination: str | None = None) -> bool:
        """The human reset, into the real room's region for `destination`'s side.

        Mirrors `Tossing3DEnvironment.reset_movables` -- episode bookkeeping carries
        forward and the pickup-refusal channel heals -- with the sim's placement regions
        replaced by `scene_layout`'s.
        """
        state = self.get_current_state()
        seed = int(round(state.get(obj=self.scene, feature_name="seed")))
        steps_taken = int(round(state.get(obj=self.scene, feature_name="steps_taken")))
        backend = self.backend()
        bin_region = None
        if destination is not None:
            side = Tossing3DSide(destination)
            region = (
                self.scene_layout.robot_side_bin_region
                if side is Tossing3DSide.ROBOT
                else self.scene_layout.opposite_side_bin_region
            )
            bin_region = {**region.model_dump(mode="json"), "destination": destination}
        backend.reset_cube_and_bin(bin_region=bin_region)
        self._pickup_blocked = False
        next_state = self.build_state(
            observation=backend.observe(),
            seed=seed,
            steps_taken=steps_taken,
            object_centric=backend.snapshot(),
            abstract_atoms=backend.abstract_atoms(),
        )
        self._adopt(state=next_state)
        return True
