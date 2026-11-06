"""The kitchen's Settled atom uses the rest test its drop and ready() use.

EES seed 1 of pouring-long-20261105-de96f85 stranded after a successful drop. The
drop's settle wait returned on position rest, but an earlier dumpling was rocking
in place at 0.53 rad/s. That is above the KinDER speed rule's 0.25 rad/s, so
Settled was missing for it and PickPitcher was not applicable. Nothing else was
applicable either, because NeedsReset is false once every piece is in the jug.
"""

import importlib.util

import pytest

pytestmark = [
    pytest.mark.xdist_group("kitchen_settled_atom"),
    pytest.mark.skipif(importlib.util.find_spec("kinder") is None, reason="needs KinDER"),
]


def _spin(*, c, name: str, rad_s: float) -> None:
    import mujoco

    body = c.model.body(name).id
    dof = c.model.jnt_dofadr[c.model.body_jntadr[body]]
    c.data.qvel[dof + 3 : dof + 6] = [rad_s, 0.0, 0.0]
    mujoco.mj_forward(c.model, c.data)


def _pick_pitcher_applicable(*, env) -> bool:
    from hitl_pmp.environments.pouring3d.kitchen.skill_provider import KitchenSkillProvider

    state = env.observe()
    pick = KitchenSkillProvider(env=env).ground(name="PickPitcher")
    return all(
        atom.predicate.holds(state=state, objects=atom.objects) for atom in pick.preconditions
    )


def test_a_dumpling_rocking_in_place_is_settled_and_pick_pitcher_stays_applicable():
    from hitl_pmp.environments.pouring3d.kitchen.environment import KitchenEnvironment

    env = KitchenEnvironment(requested_count=1)
    try:
        c = env.controller()
        c.pause(phase="observe rest", seconds=2.1)
        assert c.scene.payload_at_rest()
        assert _pick_pitcher_applicable(env=env)
        # Rocking in place: the centre has not moved, the instantaneous spin fails
        # KinDER's 0.25 rad/s rule (the stranded state measured 0.53 rad/s).
        name = c.scene.cube_names[0]
        _spin(c=c, name=name, rad_s=0.53)
        assert not c.scene.cubes_settled()
        assert c.scene.payload_at_rest()
        assert ("Settled", (name,)) in env.observe().abstract_atoms
        assert _pick_pitcher_applicable(env=env)
    finally:
        env.close()


def test_a_dumpling_that_moved_and_still_spins_is_not_settled():
    import mujoco

    from hitl_pmp.environments.pouring3d.kitchen.environment import KitchenEnvironment

    env = KitchenEnvironment(requested_count=1)
    try:
        c = env.controller()
        c.pause(phase="observe rest", seconds=2.1)
        name = c.scene.cube_names[0]
        piece = c.scene.get_object(name)
        position = c.data.xpos[c.model.body(name).id].copy()
        # A displacement the rest window saw, so neither rest test holds.
        piece.set_pose(position + [0.0, 0.0, 0.01], [1, 0, 0, 0])
        mujoco.mj_forward(c.model, c.data)
        c.pause(phase="observe motion", seconds=0.1)
        _spin(c=c, name=name, rad_s=0.53)
        assert not c.scene.payload_at_rest()
        assert not c.scene.cubes_settled()
        assert ("Settled", (name,)) not in env.observe().abstract_atoms
        assert not _pick_pitcher_applicable(env=env)
    finally:
        env.close()
