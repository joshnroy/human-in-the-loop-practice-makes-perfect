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
def test_what_the_session_reports_about_cubes_can_be_written_to_a_cycle_file(*, session) -> None:
    """The comparisons run on NumPy floats and give NumPy bools, which `json` refuses --
    after a seed has done all its work and only the record is left to write."""
    import json

    from hitl_pmp.environments.sweep_drawer3d.types import SweepDrawerScene

    piled = {c: session.in_pile(cube=c) for c in SweepDrawerScene.CUBES}
    assert all(type(v) is bool for v in piled.values())
    assert json.loads(json.dumps({"in_pile": piled, "locations": session.locations()}))


@needs_kinder
def test_the_replay_log_holds_every_joint_position_at_every_tick(*, tmp_path) -> None:
    """The state log records the gripper's command and one drawer; a replay drawn from it
    would show fingers where they were told to be and miss a drawer the robot bumped."""
    import json

    from hitl_pmp.environments.sweep_drawer3d.session import SweepDrawerSession

    path = tmp_path / "replay_log.jsonl"
    s = SweepDrawerSession(seed=1, replay_path=path)
    try:
        for _ in range(3):
            s.step(action=np.zeros(11))
        nq = int(s.mj_model.nq)
        last = [float(v) for v in s.mj_data.qpos]
    finally:
        s.close()
    records = [json.loads(line) for line in path.read_text().splitlines()]
    header, ticks = records[0], records[1:]
    assert header["kind"] == "header"
    assert header["seed"] == 1
    assert header["nq"] == nq
    assert sum(j["n"] for j in header["joints"]) == nq
    names = {j["name"] for j in header["joints"]}
    assert {"robot_left_driver_joint", "kitchen_island_drawer_s0c1_joint", "cube_0_joint"} <= names
    assert [t["t"] for t in ticks] == [0, 1, 2, 3]
    assert all(len(t["qpos"]) == nq for t in ticks)
    assert ticks[-1]["qpos"] == pytest.approx(last, abs=1e-5)


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


def _island_drawer_boxes(*, session) -> list[int]:
    import mujoco

    from hitl_pmp.environments.sweep_drawer3d.types import SweepDrawerScene as S

    m = session.mj_model
    out = []
    for g in range(m.ngeom):
        body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""
        if (
            body.startswith(S.ISLAND_DRAWERS)
            and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX
            and m.geom_contype[g] + m.geom_conaffinity[g] > 0
        ):
            out.append(g)
    return out


@needs_kinder
def test_the_island_has_six_drawers_and_the_task_opens_one_of_them(*, session) -> None:
    import mujoco

    from hitl_pmp.environments.sweep_drawer3d.types import SweepDrawerScene as S

    m = session.mj_model
    names = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "" for b in range(m.nbody)}
    drawers = {n for n in names if n.startswith(S.ISLAND_DRAWERS) and not n.endswith("_handle")}
    assert len(drawers) == 6
    assert S.DRAWER in drawers


@needs_kinder
def test_the_planning_scene_holds_every_island_drawer_not_only_the_tasks(*, session) -> None:
    """A floor cube by the island lies in front of a lower drawer's face, under its
    handle. With only the task's drawer modelled, arm plans there pass through both."""
    from hitl_pmp.environments.sweep_drawer3d.planning_scene import PlanningScene

    scene = PlanningScene(session=session)
    boxes = _island_drawer_boxes(session=session)
    assert len(boxes) == 54
    assert len(scene.drawer_bodies) + len(scene.other_drawer_bodies) == len(boxes)
    assert set(scene.other_drawer_bodies) <= scene.bodies()
    # leaving the task's drawer out of a check (to close the hand on its handle) must
    # not leave the others out with it
    assert set(scene.other_drawer_bodies) <= scene.bodies(without_drawer=True)
    assert not set(scene.drawer_bodies) & scene.bodies(without_drawer=True)


@needs_kinder
def test_a_lower_drawers_handle_is_an_obstacle_at_fingertip_height(*, session) -> None:
    import pybullet as p

    from hitl_pmp.environments.sweep_drawer3d.planning_scene import PlanningScene

    scene = PlanningScene(session=session)
    lows = [p.getAABB(b, physicsClientId=scene.cid) for b in scene.other_drawer_bodies]
    # the centre column's lower handle bar: x 0.929..0.935, |y| < 0.062, z 0.102..0.108
    assert any(
        lo[0] < 0.932 < hi[0] and lo[1] < 0.0 < hi[1] and lo[2] < 0.105 < hi[2] for lo, hi in lows
    )
