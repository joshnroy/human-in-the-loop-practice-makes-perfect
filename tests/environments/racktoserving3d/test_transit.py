from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from hitl_pmp.environments.racktoserving3d.transit import (
    TRANSIT_MARGIN,
    ArmTransit,
    EmptyHandTransit,
)
from hitl_pmp.environments.sweepoffshelf3d.scripted_skills import SkillFailed


def make_tool():
    return SimpleNamespace(
        shelf=SimpleNamespace(
            model=SimpleNamespace(
                jnt_limited=np.ones(2, dtype=bool),
                jnt_range=np.array([[-np.pi, np.pi], [-np.pi, np.pi]]),
            ),
            joints=[0, 1],
            home_arm=np.zeros(2),
        ),
        check_configuration=Mock(),
        check_arm_segment=Mock(),
    )


def test_clear_transit_returns_checked_goal_without_mutating_start():
    tool = make_tool()
    planner = ArmTransit(tool=tool, base=np.zeros(3), holding=True)
    start, target = np.zeros(2), np.ones(2)
    route = planner.plan(start=start, target=target)
    assert route is not None
    np.testing.assert_allclose(route, [target])
    np.testing.assert_array_equal(start, np.zeros(2))
    assert planner.edge_checks == 1
    assert tool.check_configuration.call_count == 2
    assert tool.check_arm_segment.call_args.kwargs["holding"]


def test_failed_transit_respects_edge_budget():
    tool = make_tool()
    tool.check_arm_segment.side_effect = SkillFailed("blocked")
    planner = ArmTransit(tool=tool, base=np.zeros(3), holding=False, max_edge_checks=1)
    assert planner.plan(start=np.zeros(2), target=np.ones(2)) is None
    assert planner.edge_checks == 1
    tool.check_arm_segment.assert_called_once()


def test_transit_rejects_invalid_joint_goal():
    planner = ArmTransit(tool=make_tool(), base=np.zeros(3), holding=False)
    with pytest.raises(SkillFailed, match="goal violates joint limits"):
        planner.plan(start=np.zeros(2), target=np.array([4.0, 0]))


def test_wrapped_shortcut_cannot_cross_joint_stop():
    tool = make_tool()
    planner = ArmTransit(tool=tool, base=np.zeros(3), holding=False)
    planner.deadline = float("inf")
    assert not planner.edge_clear(start=np.array([3.0, 0]), target=np.array([-3.0, 0]))
    tool.check_arm_segment.assert_not_called()


# --- EmptyHandTransit against its per-pair reference, on real MuJoCo geometry ----

TRANSIT_SCENE = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="3 3 0.1"/>
    <geom name="counter" type="box" pos="0.9 0 0.45" size="0.3 0.5 0.45"/>
    <geom name="post" type="cylinder" pos="-0.4 0.6 0.5" size="0.05 0.5"/>
    <geom name="wire" type="capsule" pos="0.2 -0.7 0.9" size="0.01 0.3" euler="90 0 0"/>
    <geom name="ghost" type="box" pos="0 0.3 0.5" size="0.1 0.1 0.1" contype="0"
          conaffinity="0"/>
    <geom name="ball" type="sphere" pos="-0.6 -0.5 0.3" size="0.12"/>
    <body name="robot_base_link">
      <joint name="x" type="slide" axis="1 0 0"/>
      <joint name="y" type="slide" axis="0 1 0"/>
      <joint name="th" type="hinge" axis="0 0 1"/>
      <geom name="chassis" type="box" pos="0 0 0.2" size="0.25 0.2 0.15"/>
      <geom name="mast" type="capsule" pos="0.1 0 0.7" size="0.04 0.35"/>
      <geom name="hand" type="sphere" pos="0.4 0 0.9" size="0.05"/>
      <geom name="tucked" type="box" pos="-0.2 0 0.4" size="0.03 0.03 0.03" contype="0"
            conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""


def transit_shelf(*, start=(0.0, 0.0, 0.0)):
    import mujoco

    model = mujoco.MjModel.from_xml_string(TRANSIT_SCENE)
    data, scratch = mujoco.MjData(model), mujoco.MjData(model)
    data.qpos[:3] = start
    mujoco.mj_forward(model, data)
    body = model.body("robot_base_link").id
    robot = {g for g in range(model.ngeom) if model.geom_bodyid[g] == body}
    shelf = SimpleNamespace(
        model=model,
        data=data,
        ik_data=scratch,
        mujoco=mujoco,
        ba=np.arange(3),
        qa=np.arange(0),
        robot_geoms=robot,
        cupboard_geoms={model.geom("counter").id, model.geom("post").id},
        box_geoms={model.geom("ball").id},
        in_hand=None,
    )

    def site_at(*, base, arm):
        scratch.qpos[:] = data.qpos
        scratch.qpos[shelf.ba] = base
        mujoco.mj_kinematics(model, scratch)

    def collision_cost(*, d, holding):
        mujoco.mj_collision(model, d)
        return 0.0

    shelf.site_at = site_at
    shelf.collision_cost = collision_cost
    return shelf


def reference_clear(*, transit, base, arm):
    """`EmptyHandTransit.clear` as it was: every robot/obstacle pair, one at a time."""
    shelf = transit.shelf
    transit.refresh_geometry()
    model, scratch = shelf.model, shelf.ik_data
    shelf.site_at(base=base, arm=arm)
    if shelf.collision_cost(d=scratch, holding=False) > 0:
        return False
    for robot in transit.robot:
        for obstacle in transit.obstacles:
            separation = np.linalg.norm(scratch.geom_xpos[robot] - scratch.geom_xpos[obstacle])
            bound = model.geom_rbound[robot] + model.geom_rbound[obstacle] + TRANSIT_MARGIN
            if separation > bound:
                continue
            if (
                shelf.mujoco.mj_geomDistance(model, scratch, robot, obstacle, TRANSIT_MARGIN, None)
                < TRANSIT_MARGIN
            ):
                return False
    return True


def reference_geometry(*, shelf):
    model = shelf.model
    robot = {g for g in shelf.robot_geoms if model.geom_contype[g] or model.geom_conaffinity[g]}
    fixtures = {
        g
        for g in shelf.cupboard_geoms | shelf.box_geoms
        if model.geom_contype[g] or model.geom_conaffinity[g]
    }
    obstacles = {
        g
        for g in range(model.ngeom)
        if (model.geom_contype[g] or model.geom_conaffinity[g])
        and g not in robot
        and int(model.geom_type[g]) != 0
    }
    return robot, fixtures, obstacles


@pytest.mark.parametrize("seed", range(6))
def test_empty_hand_clear_matches_the_per_pair_reference(*, seed):
    shelf = transit_shelf()
    transit = EmptyHandTransit(shelf=shelf)
    rng = np.random.default_rng(seed)
    bases = np.column_stack([
        rng.uniform(-1.2, 1.6, 150),
        rng.uniform(-1.2, 1.2, 150),
        rng.uniform(-np.pi, np.pi, 150),
    ])
    outcomes = []
    for base in bases:
        expected = reference_clear(transit=transit, base=base, arm=np.zeros(0))
        assert transit.clear(base=base, arm=np.zeros(0)) == expected
        outcomes.append(expected)
    assert any(outcomes) and not all(outcomes)


def test_empty_hand_clear_agrees_on_the_margin_boundary():
    """Bases that put the hand within a hair of the counter's collision margin."""
    shelf = transit_shelf()
    transit = EmptyHandTransit(shelf=shelf)
    # The hand sphere's centre is 0.4 m ahead of the base; the counter's face is x=0.6.
    face, hand = 0.6, 0.4 + shelf.model.geom_size[shelf.model.geom("hand").id, 0]
    outcomes = []
    for offset in np.linspace(-3e-3, 3e-3, 61):
        base = np.array([face - hand - TRANSIT_MARGIN + offset, 0.0, 0.0])
        expected = reference_clear(transit=transit, base=base, arm=np.zeros(0))
        assert transit.clear(base=base, arm=np.zeros(0)) == expected
        outcomes.append(expected)
    assert any(outcomes) and not all(outcomes)


def test_empty_hand_geometry_matches_the_reference_sets():
    shelf = transit_shelf()
    transit = EmptyHandTransit(shelf=shelf)
    assert (transit.robot, transit.fixtures, transit.obstacles) == reference_geometry(shelf=shelf)
    post = shelf.model.geom("post").id
    shelf.model.geom_contype[post] = shelf.model.geom_conaffinity[post] = 0
    transit.refresh_geometry()
    assert post not in transit.obstacles
    assert (transit.robot, transit.fixtures, transit.obstacles) == reference_geometry(shelf=shelf)


@pytest.mark.parametrize("seed", range(4))
def test_empty_hand_plan_matches_the_reference_routes(*, seed, monkeypatch):
    rng = np.random.default_rng(100 + seed)
    for _ in range(5):
        start = (rng.uniform(-0.2, 0.2), rng.uniform(-0.3, 0.3), 0.0)
        target = np.array([rng.uniform(-0.5, 0.2), rng.uniform(-1.0, 1.0), rng.uniform(-1, 1)])
        fast = EmptyHandTransit(shelf=transit_shelf(start=start))
        route = fast.plan(target=target, arm=np.zeros(0))
        slow = EmptyHandTransit(shelf=transit_shelf(start=start))
        monkeypatch.setattr(
            slow,
            "clear",
            lambda *, base, arm, slow=slow: reference_clear(transit=slow, base=base, arm=arm),
        )
        expected = slow.plan(target=target, arm=np.zeros(0))
        assert (route is None) == (expected is None)
        if expected is not None:
            assert len(route) == len(expected)
            for got, want in zip(route, expected, strict=True):
                np.testing.assert_array_equal(got, want)
