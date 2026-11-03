"""The friction MuJoCo actually applies per kitchen contact, read from d.contact."""

import numpy as np
import pytest

pytest.importorskip("kinder")

from hitl_pmp.environments.pouring3d.kitchen.environment import KitchenEnvironment

pytestmark = pytest.mark.xdist_group("kitchen_friction")


def kind(*, model, geom: int) -> str:
    name = model.body(int(model.geom_bodyid[geom])).name
    if name.startswith("potsticker"):
        return "dumpling"
    if name.startswith("robot") and "_pad" in (model.geom(geom).name or ""):
        return "pad"
    return {"tray": "skillet", "pitcher": "jug", "stovetop": "desk"}.get(name, name)


def frictions(*, controller) -> dict[frozenset[str], set[float]]:
    m, d = controller.model, controller.data
    seen: dict[frozenset[str], set[float]] = {}
    for k in range(d.ncon):
        contact = d.contact[k]
        pair = frozenset({kind(model=m, geom=contact.geom1), kind(model=m, geom=contact.geom2)})
        if len(pair) == 1:
            pair = frozenset({next(iter(pair)) + "_self"})
        seen.setdefault(pair, set()).add(round(float(contact.friction[0]), 4))
    return seen


@pytest.fixture(scope="module")
def settled():
    """Contacts of the settled episode-initial stack in the jug, merged with those of
    a dumpling then settled in the skillet on the desk."""
    import mujoco

    env = KitchenEnvironment()
    try:
        c = env.controller()
        seen = frictions(controller=c)
        piece = c.scene.get_object(c.scene.cube_names[2])
        pan = c.data.xpos[c.model.body("tray").id].copy()
        piece.set_pose(pan + [0, 0, 0.012], [1, 0, 0, 0])
        piece.set_velocity([0.0] * 3, [0.0] * 3)
        mujoco.mj_forward(c.model, c.data)
        for _ in range(400):
            mujoco.mj_step(c.model, c.data)
        assert c.scene.cube_locations() == ["pitcher", "pitcher", "tray"]
        for pair, values in frictions(controller=c).items():
            seen.setdefault(pair, set()).update(values)
        yield c, seen
    finally:
        env.close()


def test_dumplings_grip_the_skillet_at_cast_iron_friction(*, settled):
    assert settled[1][frozenset({"dumpling", "skillet"})] == {0.4}


def test_dumplings_slide_on_the_jug_and_each_other_at_the_low_friction(*, settled):
    assert settled[1][frozenset({"dumpling", "jug"})] == {0.1}
    assert settled[1][frozenset({"dumpling_self"})] == {0.1}


def test_skillet_on_the_desk_takes_the_skillet_friction(*, settled):
    # Raising the skillet's priority makes its own coefficient govern every
    # contact it has; the desk contact was max(0.6, 1.0) = 1.0 before.
    assert settled[1][frozenset({"skillet", "desk"})] == {0.4}


def test_gripper_pads_keep_their_explicit_pair_friction(*, settled):
    import mujoco

    c, _ = settled
    saved = c.data.qpos.copy()
    try:
        pad = c.model.geom(f"{c.scene.robot_name}_left_pad1").id
        piece = c.scene.get_object(c.scene.cube_names[2])
        piece.set_pose(c.data.geom_xpos[pad].copy(), [1, 0, 0, 0])
        mujoco.mj_forward(c.model, c.data)
        assert frictions(controller=c)[frozenset({"dumpling", "pad"})] == {0.6}
    finally:
        c.data.qpos[:] = saved
        mujoco.mj_forward(c.model, c.data)


def test_kitchen_contact_priorities_order_skillet_over_dumplings_over_jug(*, settled):
    m = settled[0].model
    priority = {
        kind(model=m, geom=g): int(m.geom_priority[g])
        for g in range(m.ngeom)
        if kind(model=m, geom=g) in {"skillet", "dumpling", "jug"}
    }
    assert priority["skillet"] > priority["dumpling"] > priority["jug"]
    assert np.isclose(KitchenEnvironment().skillet_mu, 0.4)
    assert np.isclose(KitchenEnvironment().surface_mu, 0.1)
