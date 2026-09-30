"""Recorded native joint-limit blockage admits checked raised leveling, not false success."""
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from hitl_pmp.environments.sweep_drawer3d.motion import Motion
from hitl_pmp.environments.sweep_simple3d.controllers import FloorPlanningScene, FloorPrimitives
from hitl_pmp.environments.sweep_simple3d.session import SweepSimpleSession


@pytest.mark.parametrize("candidate_mode", ["native", "low_then_safe", "all_low"])
def test_recorded_unloaded_pose_uses_raised_checked_path_but_requires_physical_progress(
    *, monkeypatch: pytest.MonkeyPatch, candidate_mode: str,
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
        if candidate_mode != "native":
            calls = []

            def descent(self, **kwargs):  # noqa: PLR0917 -- instance-method replacement
                del self
                calls.append(kwargs)
                end = np.zeros(7)
                # v249 had only .001908 rad of native elbow reserve.
                end[1] = 2.238092 if candidate_mode == "all_low" or len(calls) < 4 else 2.20
                return [kwargs["start"], end]

            monkeypatch.setattr(FloorPlanningScene, "floor_descent", descent)
        qpos = session.mj_data.qpos.copy()
        assert not primitive.wiper_loaded_by_cube()
        assert not primitive.level_blade(bodies=primitive.scene.bodies())
        candidates = [r for r in records if r["kind"] == "blade_leveling_candidate"]
        if candidate_mode == "all_low":
            assert len(candidates) == 5
            assert not executions
            assert all(not r["joint_reserve_ok"] for r in candidates)
            np.testing.assert_array_equal(session.mj_data.qpos, qpos)
            return
        assert [(r["turn"], r["lift"]) for r in candidates] == [
            (.08, 0.), (.04, 0.), (.02, 0.), (.04, .02)]
        assert all(not r["joint_reserve_ok"] for r in candidates[:3])
        assert candidates[-1]["joint_reserve_ok"]
        if candidate_mode == "low_then_safe":
            assert all(r["path_found"] for r in candidates)
            assert candidates[0]["endpoint_native_joint_margin"] == pytest.approx(.001908)

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
