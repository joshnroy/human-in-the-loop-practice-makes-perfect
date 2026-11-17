"""Welded food handling (`food_handling="welded"`, the default): the food skills drive and
reach for real against the fixtures, ignore food, and take or set down the dumpling by a
grasp weld and a snapped, upright pose.

`refused_pick_*.json` are PickFood starts the 80k-step practice runs of 2026-11-06 refused
at 0 steps under physical handling ("No collision-free food grasp"), restored from
`hitl-pmp-results/rack-counter-probe-20261106/states/`: a dumpling tipped on the plate
(EES seed 0, call 21), one in the plate-rack gap (seed 0, call 40), one in a gathered
column on the rack (seed 0, call 85) and one at the plate's rim (seed 2, call 90).
"""

import json
from pathlib import Path

import numpy as np
import pytest

FIXTURES = Path(__file__).parent / "fixtures"
REFUSED = sorted(FIXTURES.glob("refused_pick_*.json"))
# Starts welded PickFood itself failed from on 8adcf4a55 (replay of the 80 overnight
# refusals): a pad grazing the counter or a wire while the base drives (ees0 107, ees1
# 42), the lift after the weld (ees0 18, ees2 32), the scraper's handle resting inside
# the robot's base (ees1 11), and a refusal after the blocked-start escape (ees1 111).
WELDED_FAILURES = sorted(FIXTURES.glob("welded_failure_*.json"))
STILL_REFUSED = (
    "ees1 seq 111: the robot stands beside the counter's +y side with its arm over the "
    "counter; the tuck to the carry pose and every base path from there are refused. "
    "Not yet fixed (follow-up)."
)
STARTS = [
    pytest.param(
        path,
        id=path.stem,
        marks=[pytest.mark.xfail(reason=STILL_REFUSED, strict=True)]
        if "refused_after_escape" in path.stem
        else [],
    )
    for path in REFUSED + WELDED_FAILURES
]
PICK, PLACE, RETURN = 3.0, 4.0, 5.0


def restored(*, fixture: Path, food_handling: str = "welded"):
    """The environment at a recorded PickFood start, under `food_handling`."""
    import mujoco

    from hitl_pmp.environments.racktoserving3d.config import RackServingConfig
    from hitl_pmp.environments.racktoserving3d.environment import RackServingEnvironment
    from hitl_pmp.environments.racktoserving3d.tasks import RackServingTasks

    plain = json.loads(fixture.read_text())
    config = RackServingConfig.model_validate({**plain["config"], "food_handling": food_handling})
    env = RackServingEnvironment(config=config)
    RackServingTasks(env=env, seed=0).sample_train_task()
    backend = env.backend()
    model, data = backend.model(), backend.data()
    for joint, value in plain["robot_joints"].items():
        data.qpos[model.jnt_qposadr[model.joint(joint).id]] = value
    for name, pose in plain["free_bodies"].items():
        if name in ("rack_0", "box_0"):
            backend.core().get_object(name).set_pose(pose[:3], pose[3:])
            ik_data = env.rack_controllers().shelf.ik_data
            ik_data.mocap_pos[:], ik_data.mocap_quat[:] = data.mocap_pos, data.mocap_quat
            continue
        address = model.jnt_qposadr[model.joint(f"{name}_joint").id]
        data.qpos[address : address + 7] = pose
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    shelf = env.rack_controllers().shelf
    shelf.grip = backend.grip = float(plain["grip"])
    shelf.base_target = data.qpos[shelf.ba].copy()
    # The arm is commanded where it was recorded, as the run's last skill left it; a
    # stale reference (the reset's home) would drag it there on the first hold.
    shelf.arm_reference = data.qpos[shelf.qa].copy()
    return env, plain["target"]


def fresh(*, seed: int = 0):
    from hitl_pmp.environments.racktoserving3d.environment import RackServingEnvironment

    env = RackServingEnvironment()
    env.reset_to_seed(seed=seed)
    return env


def index(*, name: str) -> float:
    return float(name.rsplit("_", 1)[1])


def act(*, env, skill: float, name: str) -> tuple[int, str | None]:
    env.take_action(action=np.array([skill, index(name=name), 0, 0, 0]))
    return env._last_controller_steps, env.last_skill_error()


def upright(*, env, name: str) -> float:
    data, model = env.backend().data(), env.backend().model()
    return float(data.xmat[model.body(name).id].reshape(3, 3)[2, 2])


@pytest.mark.parametrize("fixture", STARTS)
def test_a_pick_refused_under_physical_handling_executes_and_lifts(*, fixture):
    env, target = restored(fixture=fixture)
    try:
        backend = env.backend()
        start_z = float(backend.data().xpos[backend.model().body(target).id][2])
        steps, error = act(env=env, skill=PICK, name=target)
        assert error is None, error
        assert steps > 0
        assert backend.held(name=target)
        assert backend.location(name=target) == "HoldingFood"
        lifted = float(backend.data().xpos[backend.model().body(target).id][2])
        assert lifted > start_z + 0.03
    finally:
        env.close()


def test_held_means_the_weld_is_engaged_not_pad_contact():
    from hitl_pmp.environments.racktoserving3d.welded_food import WeldedFood

    env = fresh()
    try:
        backend = env.backend()
        assert not any(backend.held(name=name) for name in backend.cube_names)
        WeldedFood.snap_into_hand(
            backend=backend, shelf=env.rack_controllers().shelf, name="cube_1"
        )
        assert backend.held(name="cube_1")
        assert not backend.held(name="cube_0")
        WeldedFood.release(backend=backend)
        assert not backend.held(name="cube_1")
    finally:
        env.close()


def test_place_on_a_plate_holding_four_sets_the_fifth_down_upright_and_settled():
    from hitl_pmp.environments.racktoserving3d.welded_food import WeldedFood

    env = fresh()
    try:
        backend = env.backend()
        plate = backend.support(name="box_0")
        cx, cy = plate.center_x, plate.center_y
        for offset, name in zip(
            (-0.045, -0.015, 0.015, 0.045), backend.cube_names[:4], strict=True
        ):
            WeldedFood.set_down(
                backend=backend, name=name, xy=np.array([cx + offset, cy - 0.04]), top_z=plate.top_z
            )
        env.rack_controllers().shelf.hold(count=40, label="test_settle")
        assert all(backend.settled_on_tray(name=name) for name in backend.cube_names[:4])
        steps, error = act(env=env, skill=PICK, name="cube_4")
        assert error is None, error
        steps, error = act(env=env, skill=PLACE, name="cube_4")
        assert error is None, error
        assert steps > 0
        assert not backend.held(name="cube_4")
        assert backend.settled_on_tray(name="cube_4")
        assert upright(env=env, name="cube_4") > 0.98
        assert all(backend.location(name=name) == "OnTray" for name in backend.cube_names)
    finally:
        env.close()


def test_return_food_to_rack_puts_it_on_the_rack():
    env = fresh()
    try:
        backend = env.backend()
        steps, error = act(env=env, skill=PICK, name="cube_2")
        assert error is None, error
        steps, error = act(env=env, skill=RETURN, name="cube_2")
        assert error is None, error
        assert steps > 0
        assert backend.location(name="cube_2") == "OnRack"
        assert upright(env=env, name="cube_2") > 0.98
    finally:
        env.close()


def test_the_base_drives_off_a_dumpling_on_the_floor_beneath_it():
    """A dumpling on the floor under the chassis neither blocks the plan nor the drive:
    robot bodies and food never collide, and planning ignores food."""
    from hitl_pmp.environments.racktoserving3d.welded_food import WeldedFood

    env = fresh()
    try:
        backend = env.backend()
        shelf = env.rack_controllers().shelf
        base = shelf.data.qpos[shelf.ba].copy()
        WeldedFood.set_down(backend=backend, name="cube_4", xy=base[:2], top_z=0.0)
        shelf.hold(count=20, label="test_settle")
        assert backend.location(name="cube_4") == "SpilledFloor"
        steps, error = act(env=env, skill=PICK, name="cube_0")
        assert error is None, error
        moved = float(np.linalg.norm(shelf.data.qpos[shelf.ba][:2] - base[:2]))
        assert moved > 0.1
        assert backend.held(name="cube_0")
    finally:
        env.close()


def test_physical_handling_builds_the_scene_it_always_did():
    """No grasp welds, no robot-food exclusions, KINDER's own scene builder; and food
    is held by both pads' contact."""
    from hitl_pmp.environments.racktoserving3d.config import RackServingConfig
    from hitl_pmp.environments.racktoserving3d.environment import RackServingEnvironment

    physical = RackServingEnvironment(config=RackServingConfig(food_handling="physical"))
    welded = fresh()
    try:
        physical.reset_to_seed(seed=0)
        core = physical.backend().core()
        assert "_create_scene_xml" not in vars(core)
        model, other = physical.backend().model(), welded.backend().model()
        names = physical.backend().cube_names
        assert other.neq == model.neq + len(names)
        assert model.nbody == other.nbody and model.ngeom == other.ngeom
        assert np.array_equal(model.geom_size, other.geom_size)
        assert np.array_equal(model.body_mass, other.body_mass)
        assert all("grasp_weld" not in (model.equality(i).name or "") for i in range(model.neq))
        robot = [
            geom
            for geom in range(model.ngeom)
            if model.body(model.geom_bodyid[geom]).name.startswith("robot_")
        ]
        food = sorted(set().union(*(physical.backend().body_geoms(name=n) for n in names)))
        rest = sorted(set(range(model.ngeom)) - set(robot) - set(food))

        def collides(*, m, a: int, b: int) -> bool:
            return bool(
                (m.geom_contype[a] & m.geom_conaffinity[b])
                or (m.geom_contype[b] & m.geom_conaffinity[a])
            )

        # Physical: KINDER's own filter, untouched; robot pads do touch food.
        assert any(collides(m=model, a=r, b=f) for r in robot for f in food)
        assert not any(collides(m=other, a=r, b=f) for r in robot for f in food)
        # Welded changes no other pair: food against the world, other food, the robot
        # against everything else.
        for a, b in [(f, g) for f in food for g in rest + food] + [
            (r, g) for r in robot for g in rest
        ]:
            assert collides(m=model, a=a, b=b) == collides(m=other, a=a, b=b), (a, b)
    finally:
        physical.close()
        welded.close()
