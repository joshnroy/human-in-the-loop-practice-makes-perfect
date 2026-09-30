"""Recorded native joint-limit blockage admits checked raised leveling, not false success."""
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from hitl_pmp.environments.sweep_drawer3d.motion import Motion
from hitl_pmp.environments.sweep_simple3d.controllers import FloorPrimitives
from hitl_pmp.environments.sweep_simple3d.session import SweepSimpleSession


def test_recorded_unloaded_pose_uses_raised_checked_path_but_requires_physical_progress(
    *, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = json.loads(Path(__file__).with_name("raised_leveling_fixture.json").read_text())
    session = SweepSimpleSession(seed=0)
    primitive = None
    try:
        state = session.state.copy()
        for name, values in fixture["state"].items():
            obj = state.get_object_from_name(name)
            for key, value in zip(state.type_features[obj.type], values, strict=True):
                state.set(obj, key, value)
        session.env.unwrapped._object_centric_env.set_state(state)
        session._state = state
        session.mj_data.qpos[:] = fixture["qpos"]
        session.mj_data.qvel[:] = 0.
        mujoco.mj_forward(session.mj_model, session.mj_data)
        primitive = FloorPrimitives.create(session=session, distance=.7, heading_offset=0.)
        records, executions = [], []
        monkeypatch.setattr(SweepSimpleSession, "_write",
                            lambda self, *, record: records.append(record))

        def follow(self, **kwargs):  # noqa: PLR0917 -- instance-method replacement
            del self
            executions.append(kwargs)
            return False  # No physics executed: never manufacture a successful correction.

        monkeypatch.setattr(Motion, "follow", follow)
        qpos = session.mj_data.qpos.copy()
        assert not primitive.wiper_loaded_by_cube()
        assert not primitive.level_blade(bodies=primitive.scene.bodies())
        candidates = [r for r in records if r["kind"] == "blade_leveling_candidate"]
        assert [(r["turn"], r["lift"], r["path_found"]) for r in candidates] == [
            (.08, 0., False), (.04, 0., False), (.02, 0., False), (.02, .01, True)]
        assert len(executions) == 1
        execution = executions[0]
        assert execution["tol"] == execution["final_tol"] == .005
        assert execution["max_ticks"] == 180
        assert execution["tick_guard"] is not None
        end = np.asarray(execution["path"][-1])[:7]
        limits = primitive.scene._arm_limits
        assert np.min(np.minimum(end - limits[:, 0], limits[:, 1] - end)) > .02
        np.testing.assert_array_equal(session.mj_data.qpos, qpos)
        assert session.ticks == 0
        outcome = [r for r in records if r["kind"] == "blade_leveling_outcome"][-1]
        assert outcome["converged"] is False
        assert outcome["after_tilt"] == pytest.approx(outcome["before_tilt"])
    finally:
        if primitive is not None:
            primitive.scene._sim.close()
        session.close()
