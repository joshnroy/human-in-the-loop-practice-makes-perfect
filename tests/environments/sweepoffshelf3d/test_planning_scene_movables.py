"""The movable objects in KINDER's PyBullet planning scene (Josh, 2026-10-22).

Upstream's `PyBulletSim` already builds every cube, and `planning_sim` adds the box; the
wiper was the one movable it left out. At its edge home its handle overhangs the
cupboard's front, so planned approaches ran into it and the MuJoCo re-check refused
them: layout-31 (cube_1), layout-52 (cube_2) and layout-49 (cube_4) of the benchmark
failed for that reason alone. The resting wiper is now a static collider, inflated
`WIPER_OBSTACLE_MARGIN` (6 mm), except for the skill that picks it up and while the robot
carries it (`KinderSkills.wiper_on_robot`)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from hitl_pmp.environments.sweepoffshelf3d.backend import WIDE_BOX_TASK_ID
from hitl_pmp.environments.sweepoffshelf3d.environment import (
    SKILL_IDS,
    SweepOffShelf3DEnvironment,
)
from hitl_pmp.environments.sweepoffshelf3d.skill_benchmark.runner import CaseRunner
from hitl_pmp.environments.sweepoffshelf3d.skill_benchmark.states import StateCapture, StateSet
from hitl_pmp.environments.sweepoffshelf3d.skill_trace import SkillTraces

STATES = Path(__file__).resolve().parent / "data/wide_box_benchmark_states.jsonl"
WIPER = "wiper_0"
HELD = ("HoldingWiper", ("robot", WIPER))
HOME = ("WiperHome", (WIPER,))

# Module-scoped fixtures: keep them on one xdist worker (see tests/test_xdist_groups.py).
pytestmark = [
    pytest.mark.skipif(
        importlib.util.find_spec("kinder") is None
        or importlib.util.find_spec("kinder_models") is None,
        reason="needs KINDER",
    ),
    pytest.mark.xdist_group("sweepoffshelf3d_test_planning_scene_movables"),
]


@pytest.fixture(name="env", scope="module")
def _env():
    # The states were captured in the lab2 scene with the 50 cm box; a state restores only there.
    env = SweepOffShelf3DEnvironment(scene_bg=True, task_id=WIDE_BOX_TASK_ID)
    try:
        yield env
    finally:
        env.close()


@pytest.fixture(name="states", scope="module")
def _states():
    return {state.state_id: state for state in StateSet.load(path=STATES)}


def _kinder(*, env):
    kinder = env.controllers()._kinder  # noqa: SLF001
    kinder.sync()
    return kinder


def _wiper_bodies(*, sim):
    return {k: v for k, v in sim._static_colliders.items() if k.startswith(WIPER)}  # noqa: SLF001


def _aabb(*, sim, body):
    import pybullet as p

    low, high = p.getAABB(body, physicsClientId=sim.physics_client_id)
    return np.array(low), np.array(high)


def test_the_resting_wiper_is_a_collider_in_the_planning_scene_inflated_6_mm(*, env):
    env.reset_to_seed(seed=0)
    kinder = _kinder(env=env)
    shelf = kinder.shelf
    assert HOME in env.get_current_state().abstract_atoms
    sim = kinder.planning_sim()
    bodies = _wiper_bodies(sim=sim)
    assert len(bodies) == len(shelf.wiper_geoms)
    assert set(bodies.values()) <= sim.get_collision_bodies()
    for g, body in zip(shelf.wiper_geoms, bodies.values(), strict=True):
        assert int(shelf.model.geom_type[g]) == int(shelf.mujoco.mjtGeom.mjGEOM_BOX)
        rotation = shelf.data.geom_xmat[g].reshape(3, 3)
        half = np.abs(rotation) @ (shelf.model.geom_size[g] + 0.006)
        low, high = _aabb(sim=sim, body=body)
        centre = shelf.data.geom_xpos[g]
        # PyBullet pads every AABB by its own small collision margin.
        np.testing.assert_allclose(low, centre - half, atol=0.003)
        np.testing.assert_allclose(high, centre + half, atol=0.003)


def test_the_wiper_is_not_an_obstacle_to_the_skill_that_picks_it_up(*, env):
    env.reset_to_seed(seed=0)
    kinder = _kinder(env=env)
    assert _wiper_bodies(sim=kinder.planning_sim(target=WIPER)) == {}


def test_the_wiper_is_not_an_obstacle_while_the_robot_carries_it(*, env, monkeypatch):
    env.reset_to_seed(seed=0)
    kinder = _kinder(env=env)
    monkeypatch.setattr(kinder, "wiper_on_robot", lambda: True)
    assert _wiper_bodies(sim=kinder.planning_sim()) == {}


def _robot_touches_wiper_body(*, kinder, touch):
    """The wiper clause of #513's `robot_borne`, restated: a contact between a robot geom
    and a geom of the wiper's body closer than `touch`."""
    shelf = kinder.shelf
    m, d = shelf.model, shelf.data
    body = int(m.body(WIPER).id)
    for contact in d.contact[: d.ncon]:
        pair = (int(contact.geom1), int(contact.geom2))
        for mine, other in (pair, pair[::-1]):
            if (
                int(m.geom_bodyid[other]) == body
                and mine in shelf.robot_geoms
                and contact.dist < touch
            ):
                return True
    return False


def test_the_wiper_at_home_with_the_hand_away_is_not_on_the_robot(*, env):
    env.reset_to_seed(seed=0)
    kinder = _kinder(env=env)
    assert not kinder.held(name=WIPER)
    assert not _robot_touches_wiper_body(kinder=kinder, touch=0.001)
    assert not kinder.wiper_on_robot()


def test_a_wiper_resting_on_the_robot_unheld_is_on_the_robot_and_no_obstacle(*, env):
    """The case `held` alone would miss: the wiper lies against the robot (here, dropped
    through the chassis) but the pads do not both touch it. #513's `robot_borne` left it
    out of the planning scene, and so does this."""
    import mujoco

    env.reset_to_seed(seed=0)
    kinder = _kinder(env=env)
    shelf = kinder.shelf
    m, d = shelf.model, shelf.data
    joint = int(m.body_jntadr[m.body(WIPER).id])
    address = int(m.jnt_qposadr[joint])
    base = d.xpos[m.body("robot_base_link").id]
    d.qpos[address : address + 3] = base + np.array([0.0, 0.0, 0.15])
    mujoco.mj_forward(m, d)
    assert not kinder.held(name=WIPER)
    assert _robot_touches_wiper_body(kinder=kinder, touch=0.001)
    assert kinder.wiper_on_robot()
    assert _wiper_bodies(sim=kinder.planning_sim()) == {}


def _pick(*, env, states, state_id, cube, out):
    state = states[state_id]
    (case,) = state.cases
    assert (case.skill, case.cube) == ("PickCubeFromShelf", cube)
    result = CaseRunner.run(env=env, state=state, case=case, out=out, videos=False)
    trace = SkillTraces.last(env=env)
    assert trace is not None
    return result, [n.reason for n in trace.notes if "into the wiper" in n.reason]


@pytest.mark.parametrize(("state_id", "cube"), [("layout-31", "cube_1"), ("layout-52", "cube_2")])
def test_a_shelf_pick_the_resting_wiper_used_to_block_succeeds(
    *, env, states, state_id, cube, tmp_path: Path
):
    """Before, the re-check refused 5 (layout-31) and 6 (layout-52) approaches each as
    putting the hand into the wiper, and the pick failed."""
    result, into_wiper = _pick(env=env, states=states, state_id=state_id, cube=cube, out=tmp_path)
    assert into_wiper == []
    assert result.label == "success", (result.label, result.error)


def test_layout_49_plans_no_approach_into_the_wiper(*, env, states, tmp_path: Path):
    """layout-49 fails in the edge-home scene with or without the wiper in the planning
    scene, and never on the wiper: its approaches miss cube_4 in execution."""
    _, into_wiper = _pick(env=env, states=states, state_id="layout-49", cube="cube_4", out=tmp_path)
    assert into_wiper == []


def test_a_neighbouring_cube_is_an_obstacle_the_arm_planner_routes_around(*, env, states):
    """Regression for what upstream already does: every cube is in the planning scene at
    its current pose. A cube put where the straight arm path would pass is avoided."""
    import pybullet as p
    from pybullet_helpers.motion_planning import run_motion_planning

    from hitl_pmp.environments.sweepoffshelf3d.kinder_skills import HOME_ARM

    StateCapture.restore(env=env, state=states["layout-31"])
    kinder = _kinder(env=env)
    x = kinder.state().copy()
    sim = kinder.planning_sim()
    sim.set_state(x)
    fingers = list(sim.get_robot_joints())[7:]
    start = [*HOME_ARM.tolist(), *fingers]
    goal = [*(HOME_ARM + np.deg2rad([60.0, 30.0, 0, 30.0, 0, 0, 0])).tolist(), *fingers]
    sim.robot.set_joints([(a + b) / 2 for a, b in zip(start, goal, strict=True)])
    blocker = "cube_3"
    obj = x.get_object_from_name(blocker)
    # On the forearm, not at the end effector: a 2 cm cube fits between open fingers.
    forearm = p.getLinkState(sim.robot.robot_id, 4, physicsClientId=sim.physics_client_id)[0]
    for key, value in zip(("x", "y", "z"), forearm, strict=True):
        x.set(obj, key, float(value))
    sim.set_state(x)
    cube = sim._cubes[blocker]  # noqa: SLF001
    assert cube in sim.get_collision_bodies()

    def hits(*, conf):
        sim.robot.set_joints(list(conf))
        return bool(
            p.getClosestPoints(sim.robot.robot_id, cube, 0.0, physicsClientId=sim.physics_client_id)
        )

    straight = np.array(goal) - np.array(start)
    assert any(hits(conf=start + t * straight) for t in np.linspace(0, 1, 41))
    sim.robot.set_joints(start)
    plan = run_motion_planning(
        sim.robot,
        start,
        goal,
        collision_bodies=sim.get_collision_bodies(),
        seed=0,
        physics_client_id=sim.physics_client_id,
    )
    if plan is None:
        return  # refusing is the other correct answer
    confs = [np.array(q) for q in plan]
    for a, b in zip(confs, confs[1:], strict=False):
        assert not any(hits(conf=a + t * (b - a)) for t in np.linspace(0, 1, 11))


def test_the_wiper_skills_still_pick_sweep_and_return(*, env, monkeypatch):
    """PickWiper plans without the wiper; once it is held, nothing else plans with it."""
    env.reset_to_seed(seed=0)
    kinder = _kinder(env=env)
    built = []
    original = kinder.planning_sim

    def recording(**kwargs):
        sim = original(**kwargs)
        touching = _robot_touches_wiper_body(kinder=kinder, touch=0.001)
        on_robot = kinder.held(name=WIPER) or touching
        assert kinder.wiper_on_robot() == on_robot
        built.append((on_robot, kwargs.get("target"), bool(_wiper_bodies(sim=sim))))
        return sim

    monkeypatch.setattr(kinder, "planning_sim", recording)

    def act(*, skill):
        env.take_action(action=np.array([float(SKILL_IDS[skill]), 0.0, -0.06, 0.0]))
        return env.get_current_state().abstract_atoms

    atoms = act(skill="PickWiper")
    assert HELD in atoms, env.last_skill_error()
    assert any(target == WIPER for _, target, _ in built)
    atoms = act(skill="SweepShelf")
    assert HELD in atoms, env.last_skill_error()
    atoms = act(skill="ReturnWiper")
    assert HOME in atoms, env.last_skill_error()
    assert not any(wiper for held, target, wiper in built if held or target == WIPER)


def _return_from_box(*, env, states, state_id, cube):
    StateCapture.restore(env=env, state=states[state_id])
    _kinder(env=env)
    index = float(cube.split("_")[1])
    skill = float(SKILL_IDS["ReturnCubeToShelfFromBox"])
    env.take_action(action=np.array([skill, index, 0.0, 0.0]))
    return env.get_current_state().abstract_atoms


def test_a_box_return_beside_a_neighbour_keeps_its_off_centre_grasp(*, env, states):
    """cube_2 rests against cube_1's side in the box, and only a grasp shifted away from
    cube_2 returns cube_1 alone; its approach passes under the wiper's handle, which
    overhangs the cupboard's front above the box."""
    atoms = _return_from_box(
        env=env, states=states, state_id="harvest-ReturnCubeToShelfFromBox-19", cube="cube_1"
    )
    assert ("OnShelf", ("cube_1",)) in atoms, env.last_skill_error()
    assert ("InBox", ("cube_2", "box_0")) in atoms


def test_a_one_centimetre_margin_took_that_off_centre_grasp_away(*, env, states, monkeypatch):
    """Why the margin is 6 mm, not 1 cm (Josh, 2026-10-22: "sure 6mm good"): at 1 cm the
    grown handle refused the one grasp above, and the return planned none."""
    import hitl_pmp.environments.sweepoffshelf3d.kinder_skills as kinder_skills

    monkeypatch.setattr(kinder_skills, "WIPER_OBSTACLE_MARGIN", 0.01)
    atoms = _return_from_box(
        env=env, states=states, state_id="harvest-ReturnCubeToShelfFromBox-19", cube="cube_1"
    )
    assert ("InBox", ("cube_1", "box_0")) in atoms
    assert "no grasp on cube_1 planned" in (env.last_skill_error() or "")
