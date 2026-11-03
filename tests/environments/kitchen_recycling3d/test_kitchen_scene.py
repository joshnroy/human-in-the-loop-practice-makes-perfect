"""Tossing3D's skills driving the live kitchen scene."""

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.environments.kitchen_recycling3d.bin_geometry import TaperedBin
from hitl_pmp.environments.kitchen_recycling3d.controllers import RecyclingPickController
from hitl_pmp.environments.kitchen_recycling3d.domain import KitchenConfig
from hitl_pmp.environments.kitchen_recycling3d.environment import KitchenRecycling3DEnvironment
from hitl_pmp.environments.kitchen_recycling3d.placement import NEAR_BIN_POSITION
from hitl_pmp.environments.tossing3d.predicates import HOLDING, IN_BIN, PICK_PLANNABLE
from hitl_pmp.environments.tossing3d.skills import Tossing3DSkills
from hitl_pmp.environments.tossing3d.toss import Tossing3DToss

pytestmark = [
    pytest.mark.skipif(
        importlib.util.find_spec("kinder") is None
        or importlib.util.find_spec("kinder_models") is None,
        reason="needs KINDER",
    ),
    pytest.mark.xdist_group("kitchen_recycling3d_scene"),
]

CONFIG = Path(__file__).resolve().parents[3] / "scripts/kitchen_recycling/experiment-config.json"
SEED = 3


@pytest.fixture(scope="module")
def env():
    environment = KitchenRecycling3DEnvironment(
        kitchen_config=KitchenConfig.model_validate_json(CONFIG.read_text()), scene_bg=False
    )
    yield environment
    environment.close()


def toss_skill(*, env: KitchenRecycling3DEnvironment) -> GroundSkill:
    return GroundSkill(
        skill=Tossing3DSkills.MOVE_TO_TOSS_LOCATION_AND_TOSS,
        objects=(env.robot, env.bin, env.cube, env.barrier),
    )


def test_the_far_bins_standoff_band_starts_at_the_drive_zone_edge(*, env):
    state = env.reset_to_seed(seed=SEED)
    assert state.drive_zone == env.drive_zone()
    low, high = Tossing3DToss.standoff_bounds(ground_skill=toss_skill(env=env), state=state)
    assert low == pytest.approx(2.05 - 0.71, abs=1e-4)
    assert high == 2.6


def test_pick_plannable_is_the_kitchen_pick_controllers_verdict(*, env):
    state = env.reset_to_seed(seed=SEED)
    backend = env.kinder_backend()
    controller = backend._ground_controller(
        module="tossing",
        key="pick_cube",
        object_names=(backend.robot_name, backend.cube_name, backend.barrier_name),
        state=state.object_centric,
    )
    assert isinstance(controller, RecyclingPickController)
    assert PICK_PLANNABLE.holds(state, (env.robot, env.cube))


@pytest.mark.parametrize(
    ("destination", "params"),
    # Measured against the 0.384 m tapered bin; the old 0.165 m bin's known throw,
    # (1.6, 180, 650), no longer clears the rim.
    [(None, (1.34, 160.0, 750.0)), ("robot_side", (1.25, 260.0, 650.0))],
)
def test_a_known_throw_lands_in_the_bin_by_kinders_classifier(*, env, destination, params):
    env.reset_to_seed(seed=SEED)
    if destination is not None:
        env.reset_movables(destination=destination)
    held = env.take_action(action=np.array([Tossing3DEnvironmentIds.pick, 0, 0, 0, 0.0]))
    assert env.last_skill_error() is None
    params = np.array(params)
    landed = env.take_action(action=Tossing3DToss.compute_action(params=params, state=held))
    assert IN_BIN.holds(landed, (env.cube, env.bin))
    assert env.is_solved()


def test_same_side_reset_brings_the_bin_beside_the_robot(*, env):
    env.reset_to_seed(seed=SEED)
    env.reset_movables(destination="robot_side")
    state = env.get_current_state()
    assert np.allclose(state.data[env.bin][:2], NEAR_BIN_POSITION[:2], atol=0.02)
    assert state.data[env.cube][0] < 1.15


def test_opposite_side_reset_restores_the_bin_target_on_the_mats(*, env):
    env.reset_to_seed(seed=SEED)
    env.reset_movables(destination="robot_side")
    env.reset_movables(destination="opposite_side")
    state = env.get_current_state()
    assert np.allclose(state.data[env.bin][:2], (2.05, 0.0), atol=0.02)
    assert state.data[env.cube][0] < 1.15


RECORDED = json.loads((Path(__file__).parent / "fixtures/curtis0_double_reset.json").read_text())
RECORDED_CUBE = (RECORDED["cube_0"]["x"], RECORDED["cube_0"]["y"])


def place(*, env: KitchenRecycling3DEnvironment, objects: dict[str, dict[str, float]]) -> None:
    """Write recorded or chosen features into the live scene and let it settle."""
    backend = env.kinder_backend()
    state = backend.live_state().copy()
    for name, features in objects.items():
        obj = state.get_object_from_name(name)
        for feature in state.type_features[obj.type]:
            if feature in features:
                state.set(obj, feature, float(features[feature]))
            elif feature in ("vx", "vy", "vz", "wx", "wy", "wz"):
                state.set(obj, feature, 0.0)
    backend._object_centric().set_state(state)
    backend._state = state
    backend._settle(seconds=backend.PLACEMENT_SETTLE_S)


def cube_at(*, x: float, y: float, z: float = 0.03) -> dict[str, dict[str, float]]:
    quaternion = {"qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0}
    return {"cube_0": {"x": x, "y": y, "z": z, **quaternion}}


def test_cubes_anywhere_inside_the_tapered_bin_score(*, env):
    env.reset_to_seed(seed=SEED)
    backend = env.kinder_backend()
    bin_x, bin_y = 2.05, 0.0
    # Centre, 12 cm off-centre (outside the old +-0.10 m box), and against each sloped
    # wall and corner at the floor.
    for dx, dy in ((0, 0), (0.12, 0), (-0.12, 0), (0.115, 0.065), (0, 0.068), (-0.11, -0.065)):
        place(env=env, objects=cube_at(x=bin_x + dx, y=bin_y + dy, z=0.04))
        state = env._observed_state(seed=SEED, steps_taken=0)
        cube = state.data[env.cube]
        assert backend.check_goals(), (dx, dy, cube[:3])
        assert IN_BIN.holds(state, (env.cube, env.bin)), (dx, dy, cube[:3])


def test_a_cube_beside_the_bin_under_its_rim_never_scores(*, env):
    env.reset_to_seed(seed=SEED)
    backend = env.kinder_backend()
    # Against the outside of the narrow base: inside the rim's footprint, outside the bin.
    for dx, dy in ((0.175, 0.0), (-0.175, 0.0), (0.0, 0.125), (0.0, -0.125)):
        place(env=env, objects=cube_at(x=2.05 + dx, y=dy, z=0.026))
        state = env._observed_state(seed=SEED, steps_taken=0)
        assert not backend.check_goals(), (dx, dy, state.data[env.cube][:3])
        assert not IN_BIN.holds(state, (env.cube, env.bin))


def test_the_recorded_double_reset_spot_is_refused_by_the_pick_dry_run(*, env):
    env.reset_to_seed(seed=SEED)
    place(env=env, objects={"robot": RECORDED["robot"], "bin_0": RECORDED["bin_0"]})
    place(env=env, objects=cube_at(x=RECORDED_CUBE[0], y=RECORDED_CUBE[1], z=0.026))
    assert not env.kinder_backend()._pick_plannable_here()


def test_a_reset_never_leaves_the_cube_where_the_pick_dry_run_refuses(*, env, monkeypatch):
    """Curtis seed 0, step 49,948: the first seeded spot is refused, so the next is
    taken, and the reset ends with PickPlannable true."""
    env.reset_to_seed(seed=SEED)
    backend = env.kinder_backend()
    place(env=env, objects={"robot": RECORDED["robot"]})
    seeded = type(backend)._recovery_positions

    def recorded_first(*args, **kwargs):
        yield RECORDED_CUBE
        yield from seeded(*args, **kwargs)

    monkeypatch.setattr(type(backend), "_recovery_positions", recorded_first)
    env.reset_movables(destination="robot_side")
    state = env.get_current_state()
    assert backend.reset_rejections[-1] >= 1
    assert not np.allclose(state.data[env.cube][:2], RECORDED_CUBE, atol=0.01)
    assert PICK_PLANNABLE.holds(state, (env.robot, env.cube))


@pytest.mark.parametrize("seed", [0, 1, 2, 5])
def test_every_reset_leaves_the_cube_pickable(*, env, seed):
    env.reset_to_seed(seed=seed)
    for destination in ("robot_side", "opposite_side", "robot_side"):
        env.reset_movables(destination=destination)
        state = env.get_current_state()
        assert PICK_PLANNABLE.holds(state, (env.robot, env.cube)), (seed, destination)


class Tossing3DEnvironmentIds:
    pick = float(KitchenRecycling3DEnvironment.pick_cube_id)


HOVER_ABORT = json.loads((Path(__file__).parent / "fixtures/curtis2_hover_abort.json").read_text())


def test_the_pick_planning_scenes_bin_carries_every_panel(*, env):
    import pybullet as p

    env.reset_to_seed(seed=SEED)
    backend = env.kinder_backend()
    sim = backend._build_pick_collision_sim(state=backend.live_state())
    try:
        panels = TaperedBin.of(config=env.kitchen_config).panels()
        bodies = {
            "bin": (sim.named_collision_bodies()[env.bin.name], len(panels)),
            "walls": (sim._bin_walls[env.bin.name], sum(panel.wall for panel in panels)),
        }
        client = sim.physics_client_id
        for name, (body, expected) in bodies.items():
            links = range(-1, p.getNumJoints(body, physicsClientId=client))
            shapes = sum(
                len(p.getCollisionShapeData(body, link, physicsClientId=client)) for link in links
            )
            assert shapes == expected, name
    finally:
        sim.close()


def test_the_recorded_hover_abort_beside_the_kitchen_facing_wall_now_lifts_the_cube(*, env):
    """Curtis seed 2, step 14,087 (Della v6): the cube 6 cm outside the bin's -y wall.
    The planning bin had lost that wall's top strip, so the hover plan ran the arm into
    the rim and MuJoCo held it there until the step limit."""
    env.reset_to_seed(seed=SEED)
    env.restore_plain_snapshot(plain=HOVER_ABORT, dtype=np.float64)
    after = env.take_action(action=np.array([env.pick_cube_id, 0, 0, 0, 0]))
    assert env.last_skill_error() is None, env.last_skill_error()
    assert HOLDING.holds(after, (env.robot, env.cube))


class BraceletMesh:
    """The bracelet link's collision mesh in a planning sim, by file and by size."""

    VISION = "bracelet_with_vision_link.STL"
    NO_VISION = "bracelet_no_vision_link.STL"

    @staticmethod
    def of(*, sim: Any) -> tuple[str, float]:
        import pybullet as p

        client, robot = sim.physics_client_id, sim.robot.robot_id
        link = sim.robot.link_from_name("bracelet_link")
        (shape,) = p.getCollisionShapeData(robot, link, physicsClientId=client)
        low, high = p.getAABB(robot, link, physicsClientId=client)
        return Path(shape[4].decode()).name, float(np.prod(np.subtract(high, low)))


def test_the_kitchen_planning_robot_wears_the_vision_bracelet(*, env):
    """KINDER's MuJoCo TidyBot and the real one carry the camera bracelet, which reaches
    ~3 cm further than upstream PyBullet's no-vision one."""
    from kinder_models.dynamic3d.utils import PyBulletSim

    env.reset_to_seed(seed=SEED)
    backend = env.kinder_backend()
    state = backend.live_state()
    kitchen, upstream = backend._build_pick_collision_sim(state=state), PyBulletSim(state)
    try:
        for sim in (kitchen, upstream):
            sim.set_state(state)
        (vision, vision_volume), (plain, plain_volume) = (
            BraceletMesh.of(sim=kitchen),
            BraceletMesh.of(sim=upstream),
        )
        assert (vision, plain) == (BraceletMesh.VISION, BraceletMesh.NO_VISION)
        assert vision_volume > plain_volume
        # Only the mesh changes: the links the controllers address keep their indices.
        frames = ("finger_tip", "tool_frame", "end_effector_link", "robotiq_arg2f_base_link")
        for frame in (*frames, "bracelet_link"):
            assert kitchen.robot.link_from_name(frame) == upstream.robot.link_from_name(frame)
        assert kitchen.robot.arm_joint_names == upstream.robot.arm_joint_names
        np.testing.assert_allclose(
            kitchen.get_ee_pose().position, upstream.get_ee_pose().position, atol=1e-9
        )
    finally:
        kitchen.close()
        upstream.close()


BRACELET_ABORT = json.loads(
    (Path(__file__).parent / "fixtures/ees2_bracelet_hover_abort.json").read_text()
)


def test_the_recorded_bracelet_rim_abort_now_lifts_the_cube_without_touching_the_bin(
    *, env, monkeypatch
):
    """EES seed 2, step 10,283 (Della v7): the cube just outside the bin's robot-facing
    wall. PyBullet's no-vision bracelet cleared the rim by 2.8 mm on the hover path;
    MuJoCo's vision bracelet ran 3-4 mm into it and the pick aborted on contact."""
    import mujoco

    env.reset_to_seed(seed=SEED)
    env.restore_plain_snapshot(plain=BRACELET_ABORT, dtype=np.float64)
    backend = env.kinder_backend()
    sim = backend._object_centric()._robot_env.sim
    model, data = sim.model.mj_model, sim.data.mj_data

    def body(*, geom: int) -> str:
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[geom]) or ""

    touches: list[str] = []
    observe = RecyclingPickController.observe

    def watched(*args: Any) -> None:
        (self, *_) = args
        for contact in data.contact[: data.ncon]:
            pair = sorted((body(geom=contact.geom1), body(geom=contact.geom2)))
            if any("robot" in b for b in pair) and any(b.startswith("bin") for b in pair):
                touches.append(f"{self.current_phase.name}: {pair}")
        observe(*args)

    monkeypatch.setattr(RecyclingPickController, "observe", watched)
    after = env.take_action(action=np.array([env.pick_cube_id, 0, 0, 0, 0]))
    assert env.last_skill_error() is None, env.last_skill_error()
    assert HOLDING.holds(after, (env.robot, env.cube))
    assert not touches, touches[:5]
