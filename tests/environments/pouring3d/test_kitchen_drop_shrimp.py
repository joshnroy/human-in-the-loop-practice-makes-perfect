"""DropShrimpIntoPitcher reports success only when the dumpling ends the skill in the jug."""

import mujoco
import numpy as np
import pytest

pytest.importorskip("kinder")

from hitl_pmp.environments.pouring3d.kitchen.environment import KitchenEnvironment
from hitl_pmp.environments.pouring3d.kitchen.native import KitchenController
from hitl_pmp.environments.pouring3d.practice.skills import SKILL_NAMES

pytestmark = pytest.mark.xdist_group("kitchen_drop_shrimp")

DROP = np.array([SKILL_NAMES.index("DropShrimpIntoPitcher"), 0, 1, 0, 0, 0], dtype=float)


class Landing:
    """A drop whose controller returns normally, the dumpling left at `where`."""

    @staticmethod
    def patched(*, where: str):
        def drop_shrimp(self, *, index: int) -> None:  # noqa: PLR0917 -- patched method
            name = self.scene.cube_names[index]
            if where == "desk":
                body = self.scene.get_object(name)
                pitcher, _ = self.body_pose(name="pitcher")
                body.set_pose(pitcher + [0.0, 0.25, 0.10], [1.0, 0.0, 0.0, 0.0])
                body.set_velocity([0.0] * 3, [0.0] * 3)
                mujoco.mj_forward(self.model, self.data)
            self.held = None
            self.pause(phase="release shrimp into pitcher", seconds=1, grip=0)

        return drop_shrimp


@pytest.mark.parametrize(("where", "expected"), [("desk", False), ("jug", True)])
def test_drop_success_is_whether_the_dumpling_ended_in_the_jug(*, monkeypatch, where, expected):
    monkeypatch.setattr(KitchenController, "drop_shrimp", Landing.patched(where=where))
    env = KitchenEnvironment(requested_count=1)
    env.take_action(action=DROP)
    in_jug = env.controller().scene.cube_in_vessel(env.controller().scene.cube_names[0], "pitcher")
    assert in_jug is expected
    assert env.last_info["success"] is expected, env.last_info
