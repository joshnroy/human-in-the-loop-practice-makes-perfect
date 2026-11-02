"""Replay exact kitchen poses and appliance observations, without rerunning policy."""

import numpy as np

from hitl_pmp.environments.pouring3d.practice.recording import FrozenPourReplayRenderer
from hitl_pmp.recording.run_record.types import PlainState

from .config import KitchenGeometry


class KitchenReplayRenderer(FrozenPourReplayRenderer):
    def show(self, *, state: PlainState) -> np.ndarray:
        import mujoco

        from .native import KitchenController

        if self._controller is None:
            self._controller = KitchenController.build(
                geometry=KitchenGeometry.model_validate(self.scene["geometry"]),
                deployment=bool(self.scene["deployment"]),
                requested=int(self.scene["requested_count"]),
                seed=0,
                surface_mu=float(self.scene["surface_mu"]),
                scene_bg=bool(self.scene.get("scene_bg", True))
                if self.scene_bg is None
                else self.scene_bg,
            )
            model = self._controller.model
            model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), self.width)
            model.vis.global_.offheight = max(int(model.vis.global_.offheight), self.height)
            self._renderer = mujoco.Renderer(model, height=self.height, width=self.width)
        self._controller.scene.set_cooking(enabled=bool(state["hob"][0]))
        return np.asarray(super().show(state=state), dtype=np.uint8)

    def current_state(self) -> PlainState:
        return {
            **super().current_state(),
            "hob": [float(self._controller.scene.cooking_observed())],
        }
