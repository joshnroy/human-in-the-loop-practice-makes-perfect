"""The kitchen as a Tossing3D scene: Tossing3D's state, skills and symbols, kitchen geometry.

`KitchenRecycling3DEnvironment` is a `Tossing3DEnvironment` whose backend loads the
kitchen (`backend.KitchenKinderBackend`). Its states carry the kitchen drive zone, which
Tossing3D's toss planning reads off the state, so every throw stand it proposes is one
the robot may drive to. The barrier is the mat edge: the robot's side is the kitchen
floor, the opposite side the mats, and an object on the robot's side only when its
whole footprint is off the mats.
"""

from typing import Any

from pydantic import Field

from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.types import AbstractAtom, FlatBarrier, Tossing3DState

from .backend import OPPOSITE_SIDE, KitchenKinderBackend
from .domain import KitchenConfig
from .drive_zone import KitchenDriveZone


class KitchenRecycling3DEnvironment(Tossing3DEnvironment):
    kitchen_config: KitchenConfig = Field(default_factory=KitchenConfig)
    canonical_seed: int = 0

    def build_backend(self) -> KitchenKinderBackend:
        config = self.kitchen_config
        if len(config.objects) != 1 or config.required_objects != (config.objects[0].name,):
            raise ValueError("The kitchen re-flavour of Tossing3D is single-object")
        return KitchenKinderBackend(kitchen_config=config, scene_bg=self.scene_bg)

    def drive_zone(self) -> tuple[float, float, float, float]:
        lower, upper = KitchenDriveZone.bounds(config=self.kitchen_config)
        return (float(lower[0]), float(lower[1]), float(upper[0]), float(upper[1]))

    def build_state(
        self,
        *,
        observation: Any,
        seed: int,
        steps_taken: int,
        object_centric: Any = None,
        abstract_atoms: frozenset[AbstractAtom] | None = None,
    ) -> Tossing3DState:
        state = super().build_state(
            observation=observation,
            seed=seed,
            steps_taken=steps_taken,
            object_centric=object_centric,
            abstract_atoms=abstract_atoms,
        )
        state.drive_zone = self.drive_zone()
        # The mat edge is flat: an object not wholly on the kitchen floor is on the mats.
        state.flat_barrier = FlatBarrier(bin_half_extent_x=self.kitchen_config.bin_outside[0] / 2)
        return state

    def bin_reset_region(self, *, destination: str | None) -> dict[str, object] | None:
        """The kitchen's two human placements stand in for Tossing3D's regions."""
        return {"kitchen_side": destination or OPPOSITE_SIDE}
