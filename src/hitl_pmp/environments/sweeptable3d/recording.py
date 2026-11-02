"""Replay the table scene using the shelf renderer's shared pose format."""

import numpy as np

from hitl_pmp.environments.sweepoffshelf3d.recording import SweepOffShelf3DReplayRenderer

from .backend import SweepTable3DBackend


class SweepTable3DReplayRenderer(SweepOffShelf3DReplayRenderer):
    """Object names and recorded poses are shared; geometry comes from the table task."""

    @staticmethod
    def motion_stride() -> int:
        return max(
            1, round(SweepTable3DBackend.CONTROL_FREQUENCY / SweepTable3DReplayRenderer.MOTION_FPS)
        )

    def _built(self) -> SweepTable3DBackend:
        if self._backend is None:
            scene_bg = self.scene.get("scene_bg", True) if self.scene_bg is None else self.scene_bg
            backend = SweepTable3DBackend(
                scene_bg=bool(scene_bg),
                fixed_bin_offset=float(self.scene.get("fixed_bin_offset", -0.18)),
            )
            backend.reset(seed=int(self.scene.get("canonical_seed", 0)))
            core = backend.core()
            for name in (*backend.cube_names, backend.wiper_name, backend.box_name):
                position, quaternion = core.get_object(name).get_pose()
                self._home[name] = (
                    np.asarray(position, dtype=float).copy(),
                    np.asarray(quaternion, dtype=float).copy(),
                )
            features = backend.features()
            x_min, y_min, x_max, y_max, _ = features[backend.box_name]
            self._home_box_center = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2])
            self._home_wiper_yaw = float(features[backend.wiper_name][3])
            self._backend = backend
        assert isinstance(self._backend, SweepTable3DBackend)
        return self._backend
