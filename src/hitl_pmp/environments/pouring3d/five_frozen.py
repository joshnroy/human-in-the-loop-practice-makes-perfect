"""Additive five-piece frozen proxy; preserves the published three-piece model.

The pitcher is the same approximate Winco PMCP-100 clear plastic vessel:
https://www.amazon.com/dp/B00C0N8098 . Frozen material assumptions are documented
in frozen_native. Human reset respawns separate pieces and lets gravity settle
them; the robot still uses the unchanged physical feedback pour controller.
"""

from typing import Any, cast

import mujoco
import numpy as np
from kinder.envs.dynamic3d.objects.primitive_objects import Bin
from kinder.envs.dynamic3d.pitcher_pour3d import ObjectCentricPitcherPour3DEnv
from scipy.spatial.transform import Rotation

from .feedback_controller import FeedbackPourController
from .frozen_native import FrozenShrimpEnv, FrozenShrimpScene


class FiveFrozenShrimpScene(FrozenShrimpScene):
    cube_names = tuple(f"shrimp_{i}" for i in range(5))

    def __init__(self, *args: Any, surface_mu: float = 0.1, **kwargs: Any) -> None:
        if not np.isfinite(surface_mu) or surface_mu < 0:
            raise ValueError("surface_mu must be finite and nonnegative")
        self.surface_mu = surface_mu
        kwargs.setdefault("requested_count", 5)
        kwargs["cube_material"] = None
        # Bypass only the three-piece constructor's count restriction. All
        # contact generation and shape-aware containment remain inherited.
        ObjectCentricPitcherPour3DEnv.__init__(self, *args, **kwargs)

    @staticmethod
    def reset_offsets() -> np.ndarray:
        angles = np.linspace(0, 2 * np.pi, 3, endpoint=False)
        bottom = np.column_stack((
            0.022 * np.cos(angles),
            0.022 * np.sin(angles),
            np.full(3, 0.0055),
        ))
        # Two additional pieces start above the bottom three with a clear gap.
        # This prevents intersecting colliders before native gravity settling.
        upper = np.array([[0, -0.018, 0.030], [0, 0.018, 0.030]])
        return np.vstack((bottom, upper))

    def respawn_payload_in_pitcher(self) -> None:
        pitcher = cast(Bin, self.get_object("pitcher"))
        origin = np.asarray(pitcher.get_position(), dtype=float)
        quat = np.asarray(pitcher.get_orientation(), dtype=float)
        rotation = Rotation.from_quat(quat[[1, 2, 3, 0]])
        for name, offset in zip(self.cube_names, self.reset_offsets(), strict=True):
            local = offset.copy()
            local[:2] += self.np_random.uniform(-0.0005, 0.0005, 2)
            local[2] += pitcher.wall_thickness
            shrimp = self.get_object(name)
            shrimp.set_pose(origin + rotation.apply(local), quat)
            shrimp.set_velocity([0.0] * 3, [0.0] * 3)
        model, data = self._model_and_data()
        mujoco.mj_forward(model, data)

    def _check_goals(self) -> bool:
        locations = self.cube_locations()
        return (
            locations.count("tray") == self.requested_count
            and locations.count("pitcher") == 5 - self.requested_count
            and self.cubes_settled()
        )

    def physical_snapshot(self) -> dict[str, Any]:
        snapshot = super().physical_snapshot()
        snapshot["payload_count"] = 5
        return snapshot


class FiveFrozenShrimpEnv(FrozenShrimpEnv):
    def _create_object_centric_env(self, *args: Any, **kwargs: Any) -> FiveFrozenShrimpScene:
        return FiveFrozenShrimpScene(*args, **kwargs)

    def _create_variant_specific_description(self) -> str:
        scene = cast(FiveFrozenShrimpScene, self._object_centric_env)
        return (
            f"Pour exactly {scene.requested_count} of five individually frozen shrimp "
            "proxies; retain the remainder, settle all pieces, and avoid spills."
        )


class FiveFrozenController(FeedbackPourController):
    def ready(self) -> bool:
        return (
            self.held is None
            and self.scene.cube_locations() == ["pitcher"] * 5
            and self.scene.cubes_settled()
            and self.vessel_home(name="pitcher")
            and self.vessel_home(name="tray")
        )
