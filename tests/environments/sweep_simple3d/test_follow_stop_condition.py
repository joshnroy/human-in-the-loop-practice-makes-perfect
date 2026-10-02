"""A physical feedback objective can stop a checked path without changing defaults."""

from types import SimpleNamespace

import numpy as np

from hitl_pmp.environments.sweep_simple3d.physical.motion import Motion


def test_optional_physical_stop_follows_guard_and_preserves_default() -> None:
    actions = []
    guards = []
    session = SimpleNamespace(
        arm=lambda: np.zeros(7), step=lambda *, action: actions.append(action)
    )
    motion = Motion.model_construct(session=session, scene=None)
    assert motion.follow(
        path=[np.ones(7)],
        grip=1.0,
        max_ticks=5,
        tick_guard=lambda: guards.append(len(actions)),
        stop_condition=lambda: len(guards) == 2,
    )
    assert len(actions) == 2
    assert guards == [1, 2]
    actions.clear()
    assert not motion.follow(path=[np.ones(7)], grip=1.0, max_ticks=3)
    assert len(actions) == 3
