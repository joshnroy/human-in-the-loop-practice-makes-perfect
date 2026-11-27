"""The pouring arm's IK accepts a pose within 10 mm (Josh, 2026-11-08).

Pick and place are assumed to work in this project. In the p = 0 Curtis run on agni
(2026-11-08) a drop, a skillet pick and a floor pick each failed on misses of
8.2-9.3 mm against the old 8 mm tolerance, and the failed drop let a dumpling fall to
the floor where nothing but the human could recover it.
"""

import importlib.util

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

pytestmark = [
    pytest.mark.xdist_group("pouring_ik_tolerance"),
    pytest.mark.skipif(importlib.util.find_spec("kinder") is None, reason="needs KinDER"),
]


def _miss_by(*, metres: float, monkeypatch) -> np.ndarray:
    from hitl_pmp.environments.pouring3d import native_controller, potsticker_controller
    from hitl_pmp.environments.pouring3d.kitchen.environment import KitchenEnvironment

    env = KitchenEnvironment(requested_count=1)
    try:
        c = env.controller()
        c.last_q = c.data.qpos[c.arm_q].copy()
        hand = c.data.site_xpos[c.site].copy()
        rotation = Rotation.from_matrix(c.data.site_xmat[c.site].reshape(3, 3))

        class _Stay:
            def __init__(self, *, x):
                self.x = np.array(x, dtype=float)

        # A solver that cannot move the hand: the residual is exactly the offset.
        for module in (native_controller, potsticker_controller):
            monkeypatch.setattr(module, "least_squares", lambda fun, x0, **_: _Stay(x=x0))
        return c.inverse_kinematics(position=hand + [metres, 0.0, 0.0], rotation=rotation)
    finally:
        env.close()


def test_a_9_mm_miss_is_accepted(*, monkeypatch):
    _miss_by(metres=0.009, monkeypatch=monkeypatch)


def test_an_11_mm_miss_is_still_infeasible(*, monkeypatch):
    with pytest.raises(ValueError, match="IK infeasible"):
        _miss_by(metres=0.011, monkeypatch=monkeypatch)
