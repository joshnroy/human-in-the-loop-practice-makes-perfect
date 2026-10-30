"""Render recorded native joint states without rerunning physics or the policy."""

from typing import Any

import numpy as np
from pydantic import PrivateAttr

from hitl_pmp.recording.replay.renderer import ReplayRenderer
from hitl_pmp.recording.run_record.types import PlainState

from .backend import PourBackend


class FrozenPourReplayRenderer(ReplayRenderer):
    scene: dict[str, Any]
    scene_bg: bool | None = True
    width: int = 960
    height: int = 720
    _controller: Any = PrivateAttr(default=None)
    _renderer: Any = PrivateAttr(default=None)
    _request: float = PrivateAttr(default=2)

    def is_storyboard(self) -> bool:
        return not bool(self.scene.get("motion", False))

    def fps(self) -> int:
        return int(self.scene.get("fps", 10)) if not self.is_storyboard() else 2

    def show(self, *, state: PlainState) -> np.ndarray:
        import mujoco

        if self._controller is None:
            self._controller = PourBackend.create(
                count=int(self.scene["payload_count"]),
                requested=int(self.scene["requested_count"]),
                mu=float(self.scene["surface_mu"]),
                seed=0,
                payload_kind=str(self.scene.get("payload_kind", "shrimp")),
                layout=str(self.scene.get("layout", "floor")),
                receiving_vessel=str(self.scene.get("receiving_vessel", "tray")),
                scene_bg=bool(self.scene.get("scene_bg", True))
                if self.scene_bg is None
                else self.scene_bg,
            )
            model = self._controller.model
            model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), self.width)
            model.vis.global_.offheight = max(int(model.vis.global_.offheight), self.height)
            self._renderer = mujoco.Renderer(model, height=self.height, width=self.width)
        c = self._controller
        c.data.qpos[:] = state["qpos"]
        c.data.qvel[:] = state["qvel"]
        c.data.time = state["time"][0]
        self._request = state["request"][0]
        mujoco.mj_forward(c.model, c.data)
        self._renderer.update_scene(c.data, camera="task_view")
        return np.asarray(self._renderer.render(), dtype=np.uint8).copy()

    def current_state(self) -> PlainState:
        c = self._controller
        return {
            "qpos": c.data.qpos.tolist(),
            "qvel": c.data.qvel.tolist(),
            "time": [float(c.data.time)],
            "request": [self._request],
        }

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._controller is not None:
            self._controller.env.close()
            self._controller = None
