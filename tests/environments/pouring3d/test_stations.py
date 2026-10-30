"""Raised workstations require travel and keep spills on support surfaces observable."""

import numpy as np
import pytest

from hitl_pmp.environments.pouring3d.practice.environment import FrozenPourEnvironment


@pytest.mark.parametrize("count", [3, 5])
def test_stations_support_vessels_and_require_base_travel(*, count):
    env = FrozenPourEnvironment(layout="stations", payload_count=count, scene_bg=False)
    try:
        env.hard_reset()
        c = env.controller()
        assert c.home[2] == pytest.approx(0.74, abs=0.002)
        assert c.tray_home[2] == pytest.approx(0.84, abs=0.002)
        assert np.linalg.norm((c.home - c.tray_home)[:2]) > 2.0
        assert c.model.body("prep_table").jntnum == 0
        assert c.model.body("stovetop").jntnum == 0
        assert c.model.body("tray").jntnum == 1
        # The finite finger contact patch must resist twisting while carrying.
        for name in ("pitcher_handle_upper", "pitcher_handle_grip"):
            assert c.model.geom_condim[c.model.geom(name).id] == 4
        assert c.ready()
        # A ready payload is insufficient if the default arm intersects a worktop.
        for contact in c.data.contact:
            bodies = [c.model.body(int(c.model.geom_bodyid[g])).name or "" for g in contact.geom]
            if any(n.startswith("robot") for n in bodies) and any(
                n in {"prep_table", "stovetop"} for n in bodies
            ):
                assert contact.dist >= -0.001, bodies
        c.pause(phase="supported stability", seconds=2)
        assert c.ready()
    finally:
        env.close()


def test_tabletop_spill_is_observable_and_recoverable_symbolically():
    import mujoco

    from hitl_pmp.environments.pouring3d.practice.skill_provider import FrozenPourSkillProvider

    env = FrozenPourEnvironment(layout="stations", scene_bg=False)
    try:
        env.hard_reset()
        c = env.controller()
        shrimp = c.scene.get_object("shrimp_0")
        shrimp.set_pose([0.95, -1.30, 0.7455], [1, 0, 0, 0])
        shrimp.set_velocity([0, 0, 0], [0, 0, 0])
        mujoco.mj_forward(c.model, c.data)
        assert c.scene.cube_locations()[0] == "surface"
        state = env.observe()
        assert ("OnSurface", ("shrimp_0",)) in state.abstract_atoms
        assert ("NoSpills", ()) not in state.abstract_atoms
        provider = FrozenPourSkillProvider(env=env)
        ground = provider.ground(name="PickShrimpFromSurface", shrimp=0)
        assert all(
            atom.predicate.holds(state=state, objects=atom.objects) for atom in ground.preconditions
        )
        before = c.data.qpos.copy()
        time, steps = c.data.time, c.steps
        assert env.reset_movables(destination="reset_shrimp")
        assert c.ready()
        keep = np.ones(c.model.nq, dtype=bool)
        for name in c.scene.cube_names:
            joint = c.model.body_jntadr[c.model.body(name).id]
            q = c.model.jnt_qposadr[joint]
            keep[q : q + 7] = False
        np.testing.assert_array_equal(before[keep], c.data.qpos[keep])
        assert (c.data.time, c.steps) == (time, steps)
    finally:
        env.close()


@pytest.mark.parametrize(
    ("skill", "position", "yaw"),
    [
        ("PickShrimpFromTray", [0.75, 1.05, 0.8500], 0),
        ("PickShrimpFromSurface", [0.82, -1.25, 0.7455], 0),
        ("PickShrimpFromTray", [0.75, 1.05, 0.8500], 45),
        ("PickShrimpFromSurface", [0.82, -1.25, 0.7455], -45),
    ],
)
def test_raised_shrimp_pick_and_drop_clears_worktop_edges(*, skill, position, yaw):
    import mujoco

    from hitl_pmp.environments.pouring3d.practice.skill_provider import FrozenPourSkillProvider

    env = FrozenPourEnvironment(layout="stations", scene_bg=False)
    try:
        env.hard_reset()
        c = env.controller()
        shrimp = c.scene.get_object("shrimp_0")
        half_yaw = np.deg2rad(yaw) / 2
        shrimp.set_pose(position, [np.cos(half_yaw), 0, 0, np.sin(half_yaw)])
        shrimp.set_velocity([0, 0, 0], [0, 0, 0])
        mujoco.mj_forward(c.model, c.data)
        env.current_state = env.observe()
        provider = FrozenPourSkillProvider(env=env)
        body_names = [c.model.body(int(b)).name or "" for b in c.model.geom_bodyid]
        robot = np.array([n.startswith("robot") for n in body_names])
        obstacle = np.array([n in {"prep_table", "stovetop", "tray"} for n in body_names])
        original_step = c.scene._robot_env.sim.step
        forbidden_contacts = []

        def audit():
            original_step()
            pairs = c.data.contact.geom
            mask = (robot[pairs[:, 0]] & obstacle[pairs[:, 1]]) | (
                obstacle[pairs[:, 0]] & robot[pairs[:, 1]]
            )
            for index in np.flatnonzero(mask & (c.data.contact.dist < 0)):
                a, b = pairs[index]
                body = body_names[a if robot[a] else b]
                # Soft fingertip contact while pinching a supported object is
                # distinct from the wrist/arm catching below a worktop edge.
                if not (body.endswith("_pad") and c.phase in {"pinch shrimp", "retain shrimp"}):
                    forbidden_contacts.append((body, c.phase))

        c.scene._robot_env.sim.step = audit
        for name in (skill, "DropShrimpIntoPitcher"):
            ground = provider.ground(name=name, shrimp=0)
            env.take_action(
                action=provider.compute_action(
                    ground_skill=ground, params=np.zeros(0), state=env.get_current_state()
                )
            )
            assert env.last_info["success"], env.last_info
        assert c.ready()
        assert not forbidden_contacts, forbidden_contacts[:5]
    finally:
        env.close()


def test_failed_raised_grasp_physically_retreats_to_hover():
    import mujoco

    from hitl_pmp.environments.pouring3d.practice.skill_provider import FrozenPourSkillProvider

    env = FrozenPourEnvironment(layout="stations", scene_bg=False)
    try:
        env.hard_reset()
        c = env.controller()
        shrimp = c.scene.get_object("shrimp_0")
        shrimp.set_pose([0.75, 1.05, 0.8500], [1, 0, 0, 0])
        shrimp.set_velocity([0, 0, 0], [0, 0, 0])
        mujoco.mj_forward(c.model, c.data)
        env.current_state = env.observe()
        provider = FrozenPourSkillProvider(env=env)
        failed_at = []

        def fail_grasp():
            if c.phase == "pinch shrimp" and not failed_at:
                failed_at.append((c.data.time, c.hand_pose()[0].copy()))
                raise ValueError("injected grasp failure")

        c.before_step = fail_grasp
        ground = provider.ground(name="PickShrimpFromTray", shrimp=0)
        env.take_action(
            action=provider.compute_action(
                ground_skill=ground, params=np.zeros(0), state=env.get_current_state()
            )
        )
        assert failed_at
        assert not env.last_info["success"]
        assert env.last_info["error"] == "injected grasp failure"
        assert "recovery_error" not in env.last_info
        assert c.held is None
        assert c.data.time > failed_at[0][0] + 1.0
        assert c.hand_pose()[0][2] > failed_at[0][1][2] + 0.08
        assert c.scene.cube_locations() == ["tray", "pitcher", "pitcher"]
    finally:
        env.close()
