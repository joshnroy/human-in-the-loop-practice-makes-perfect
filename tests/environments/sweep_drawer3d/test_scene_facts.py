"""The scene constants the reset relies on, checked against the compiled KINDER model.

The scene moves with the reference/kindergarden pin; these tests make that coupling loud
rather than silent, as the Tossing3D geometry tests do.
"""

import importlib.util

import numpy as np
import pytest

needs_kinder = pytest.mark.skipif(
    importlib.util.find_spec("kinder") is None or importlib.util.find_spec("kinder_models") is None,
    reason="KINDER is an optional extra",
)


@pytest.fixture(scope="module")
def session():
    from hitl_pmp.environments.sweep_drawer3d.session import SweepDrawerSession

    s = SweepDrawerSession(seed=1)
    yield s
    s.close()


def _geom(*, session, body: str, suffix: str) -> tuple[np.ndarray, np.ndarray]:
    import mujoco

    m, d = session.mj_model, session.mj_data
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
    for g in range(m.ngeom):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if m.geom_bodyid[g] == b and name.endswith(suffix):
            return np.array(d.geom_xpos[g]), np.array(m.geom_size[g])
    raise KeyError(suffix)


@needs_kinder
def test_every_cube_starts_in_the_pile_region_on_the_counter(*, session) -> None:
    from hitl_pmp.environments.sweep_drawer3d.types import SweepDrawerScene

    assert session.drawer_pos() == pytest.approx(0.0, abs=1e-6)
    for c in SweepDrawerScene.CUBES:
        assert session.location(cube=c) == "counter"
        assert session.in_pile(cube=c)


@needs_kinder
def test_the_pile_region_is_the_task_jsons_blocks_init_region(*, session) -> None:
    import mujoco

    from hitl_pmp.environments.sweep_drawer3d.types import SweepDrawerScene

    m, d = session.mj_model, session.mj_data
    site = mujoco.mj_name2id(
        m, mujoco.mjtObj.mjOBJ_SITE, "kitchen_island_blocks_init_region_region_0"
    )
    center, half = d.site_xpos[site], m.site_size[site]
    # the site frame is the island's, rotated -90 deg: its x half-size runs along world y
    assert pytest.approx((center[0] - half[1], center[0] + half[1])) == SweepDrawerScene.PILE_X
    assert pytest.approx((center[1] - half[0], center[1] + half[0])) == SweepDrawerScene.PILE_Y


@needs_kinder
def test_drawer_geometry_matches_the_compiled_model(*, session) -> None:
    from hitl_pmp.environments.sweep_drawer3d.types import SweepDrawerScene as S

    front, fsize = _geom(session=session, body=S.DRAWER, suffix="_front")
    assert front[0] - fsize[0 if fsize[0] < fsize[1] else 1] == pytest.approx(
        S.DRAWER_FRONT_INNER_X, abs=2e-3
    )
    face, face_size = _geom(session=session, body=S.DRAWER, suffix="_face")
    assert face[2] + face_size[2] == pytest.approx(S.DRAWER_WALL_TOP, abs=2e-3)
    bottom, bsize = _geom(session=session, body=S.DRAWER, suffix="_bottom")
    assert bottom[2] + bsize[2] == pytest.approx(S.DRAWER_FLOOR, abs=1e-3)


@needs_kinder
def test_the_drawer_slides_far_enough_to_clear_the_countertop(*, session) -> None:
    """Opening it further than the stock OpenDrawer's ~0.11 m needs no task change: the
    joint allows 0.6 m."""
    import mujoco

    from hitl_pmp.environments.sweep_drawer3d.types import SweepDrawerScene as S

    m = session.mj_model
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, S.DRAWER + "_joint")
    assert m.jnt_range[j][1] >= 0.5


@needs_kinder
def test_kinder_models_planning_model_has_no_drawer_at_all(*, session) -> None:
    """The gap this module's PlanningScene fills: kinder-models' PyBulletSim is built from
    static colliders and cubes only, and the drawer is neither."""
    from kinder_models.dynamic3d.utils import PyBulletSim

    sim = PyBulletSim(session.state)
    assert not any("drawer" in name for name in sim._static_colliders)


@needs_kinder
def test_the_planning_scene_poses_the_drawer_from_the_live_simulator(*, session) -> None:
    import pybullet as p

    from hitl_pmp.environments.sweep_drawer3d.planning_scene import PlanningScene
    from hitl_pmp.environments.sweep_drawer3d.types import SweepDrawerScene as S

    scene = PlanningScene(session=session)
    assert len(scene.drawer_bodies) >= 6
    x0 = [
        p.getBasePositionAndOrientation(b, physicsClientId=scene.cid)[0][0]
        for b in scene.drawer_bodies
    ]
    state = session.state.copy()
    state.set(state.get_object_from_name(S.DRAWER), "pos", 0.2)
    session.restore(state=state)
    scene.sync()
    x1 = [
        p.getBasePositionAndOrientation(b, physicsClientId=scene.cid)[0][0]
        for b in scene.drawer_bodies
    ]
    assert np.allclose(np.array(x1) - np.array(x0), session.drawer_pos(), atol=5e-3)
    assert session.drawer_pos() == pytest.approx(0.2, abs=0.01)
