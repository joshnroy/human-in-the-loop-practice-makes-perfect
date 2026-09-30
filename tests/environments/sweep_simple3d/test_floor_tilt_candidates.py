"""Floor orientations retain native contact geometry and the controller tilt bound."""

from itertools import product
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from hitl_pmp.environments.sweep_simple3d.controllers import FloorPrimitives
from hitl_pmp.environments.sweep_simple3d.session import SweepSimpleSession


def test_existing_orientations_precede_bounded_nearby_fallbacks() -> None:
    primitive = SimpleNamespace(
        session=SimpleNamespace(quaternion=lambda **_: Rotation.from_euler("x", .894).as_quat()),
        scene=SimpleNamespace(max_tool_tilt=1.1),
    )
    candidates = FloorPrimitives.floor_orientation_candidates(primitive, tool_yaws=(0., np.pi))
    assert candidates[:4] == [(0., False, None), (np.pi, False, None),
                              (0., True, None), (np.pi, True, None)]
    assert [c[2] for c in candidates[4:12:2]] == pytest.approx([.914, .934, .874, .854])
    assert all(c[1] and 0 < c[2] <= .95 for c in candidates[4:])


def test_candidates_cannot_consume_existing_tilt_reserve() -> None:
    primitive = SimpleNamespace(
        session=SimpleNamespace(quaternion=lambda **_: Rotation.from_euler("x", 1.05).as_quat()),
        scene=SimpleNamespace(max_tool_tilt=1.1),
    )
    candidates = FloorPrimitives.floor_orientation_candidates(primitive, tool_yaws=(0.,))
    assert [c[2] for c in candidates[2:]] == [.2, .4, .6, .8, .95]
    assert all(c[2] <= primitive.scene.max_tool_tilt - .15 for c in candidates[2:])


@pytest.mark.parametrize("tilt", [.2, .4, .6, .8, .95, .914, .934])
@pytest.mark.parametrize("observed_tilt", [0.0, 1.659])
def test_native_floor_support_and_contact_yaw_are_preserved(
    *, tilt: float, observed_tilt: float
) -> None:
    session = SweepSimpleSession(seed=0)
    try:
        model, data = session.mj_model, session.mj_data
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wiper_0")
        address = model.jnt_qposadr[model.body_jntadr[body]]
        quaternion = Rotation.from_euler("x", observed_tilt).as_quat()
        data.qpos[address + 3:address + 7] = quaternion[[3, 0, 1, 2]]
        mujoco.mj_forward(model, data)
        primitive = SimpleNamespace(session=session, scene=SimpleNamespace(max_tool_tilt=1.1),
                                    floor_clearance=.005)
        xy = np.array([1.48, 1.035])
        pose = FloorPrimitives.floor_tool_pose(primitive, xy=xy, yaw=.034, tilt=tilt)
        rotation = Rotation.from_quat(pose.orientation).as_matrix()
        assert np.arccos(rotation[2, 2]) == pytest.approx(tilt)
        assert np.arctan2(rotation[1, 0], rotation[0, 0]) == pytest.approx(.034)
        np.testing.assert_array_equal(pose.position[:2], xy)
        model, data = session.mj_model, session.mj_data
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wiper_0")
        blade = max((g for g in range(model.ngeom) if model.geom_bodyid[g] == body),
                    key=lambda g: model.geom_size[g, 0])
        corners = np.array([
            data.geom_xpos[blade] + data.geom_xmat[blade].reshape(3, 3)
            @ (model.geom_size[blade] * signs)
            for signs in product((-1, 1), repeat=3)
        ])
        local = (corners - data.xpos[body]) @ data.xmat[body].reshape(3, 3)
        assert ((local @ rotation.T) + pose.position)[:, 2].min() == pytest.approx(.005)
        assert primitive.scene.max_tool_tilt == 1.1
    finally:
        session.close()


def test_overshooting_positive_steps_include_existing_ceiling_once_per_yaw() -> None:
    primitive = SimpleNamespace(
        session=SimpleNamespace(
            quaternion=lambda **_: Rotation.from_euler("x", .93465).as_quat()
        ),
        scene=SimpleNamespace(max_tool_tilt=1.1),
    )
    candidates = FloorPrimitives.floor_orientation_candidates(primitive, tool_yaws=(0., np.pi))
    nearby = candidates[4:]
    assert [c[2] for c in nearby[:6:2]] == pytest.approx([.95, .91465, .89465])
    assert sum(np.isclose(c[2], .95) for c in nearby) == 2
    assert all(c[2] <= .95 for c in nearby)
    assert primitive.scene.max_tool_tilt == 1.1


def test_tipped_grasp_gets_recorded_feasible_broad_fallback_without_duplicates() -> None:
    primitive = SimpleNamespace(
        session=SimpleNamespace(quaternion=lambda **_: Rotation.from_euler("x", 1.659).as_quat()),
        scene=SimpleNamespace(max_tool_tilt=1.1),
    )
    candidates = FloorPrimitives.floor_orientation_candidates(primitive, tool_yaws=(.0474,))
    assert candidates[:2] == [(.0474, False, None), (.0474, True, None)]
    assert [c[2] for c in candidates[2:]] == [.2, .4, .6, .8, .95]
    assert primitive.scene.max_tool_tilt == 1.1


def test_broad_candidates_deduplicate_observed_and_nearby_tilts() -> None:
    primitive = SimpleNamespace(
        session=SimpleNamespace(quaternion=lambda **_: Rotation.from_euler("x", .4).as_quat()),
        scene=SimpleNamespace(max_tool_tilt=.8),
    )
    candidates = FloorPrimitives.floor_orientation_candidates(primitive, tool_yaws=(0.,))
    values = [c[2] for c in candidates[2:]]
    assert values == pytest.approx([.42, .44, .38, .36, .2, .6])
    assert len(values) == len(set(values))
    assert all(v <= .65 for v in values)
