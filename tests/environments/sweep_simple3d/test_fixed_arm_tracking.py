"""Arm gravity sag must not become the target for the next contact microstroke."""

from types import SimpleNamespace

import numpy as np

from hitl_pmp.environments.sweep_simple3d.physical.motion import Motion


def test_fixed_arm_target_corrects_observed_sag_and_default_preserves_pose() -> None:
    recorded = []
    session = SimpleNamespace(
        arm=lambda: np.full(7, 0.02),
        base=lambda: (0.0, 0.0, 0.0),
        step=lambda *, action: recorded.append(action.copy()),
    )
    motion = Motion.model_construct(session=session, scene=None)
    assert not motion.drive(path=[(0.1, 0.0, 0.0)], grip=1.0, max_ticks=1)
    np.testing.assert_allclose(recorded[-1][3:10], 0.0)
    desired = np.zeros(7)
    assert not motion.drive(path=[(0.1, 0.0, 0.0)], grip=1.0, max_ticks=1, arm=desired)
    np.testing.assert_allclose(recorded[-1][3:10], -0.02)
    np.testing.assert_array_equal(desired, np.zeros(7))
