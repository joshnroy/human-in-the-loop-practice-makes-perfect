"""States the 12db26a launch got stuck in: a skill retried from them must not move a few
steps and fail again. It either does its job or refuses at 0 steps, so the step
protocol's zero-step guard can end the loop.

`stuck_ees0_return_scraper_behind_counter.json`: EES seed 0, recording tick 12261, the
start of the 23rd of 24 consecutive ReturnScraper failures (3 to 76 steps each, in the
scripted `return_clear_rack` line: "IK failed at [1.724 0.012 1.269]"). A GatherAndSweep
had left the base behind the counter (x 2.05, heading 2.84 rad) holding the scraper; each
failed attempt crept the arm further, to joint 6 at its 2.09 rad limit.
"""

from pathlib import Path

import numpy as np
import pytest

from tests.environments.racktoserving3d.test_planned_food import restored

FIXTURES = Path(__file__).parent / "fixtures"
EES0_RETURN = FIXTURES / "stuck_ees0_return_scraper_behind_counter.json"


def restored_holding(*, fixture: Path):
    env = restored(fixture=fixture)
    controllers = env.rack_controllers()
    if env.backend().held(name="wiper_0"):
        shelf = controllers.shelf
        shelf.in_hand = np.linalg.inv(shelf.site_pose()) @ controllers.tool.wiper_pose()
    return env


def done_or_refused(*, env, action: np.ndarray) -> tuple[int, str | None]:
    env.take_action(action=action)
    return env._last_controller_steps, env.last_skill_error()


def test_return_scraper_from_behind_the_counter_returns_or_refuses():
    env = restored_holding(fixture=EES0_RETURN)
    try:
        assert env.backend().held(name="wiper_0")
        steps, error = done_or_refused(env=env, action=np.array([2.0, 0, 0, 0, 0]))
        if steps > 0:
            assert error is None, f"{steps} steps, then: {error}"
            assert env.rack_controllers().tool.at_home()
    finally:
        env.close()


def test_a_transit_starting_just_past_a_joint_stop_recovers():
    """Curtis seed 2 (12db26a) failed GatherAndSweep 17 times, 4 steps each, with
    "squeeze_plus_y_arm_clearance: Arm transit start violates joint limits". That run
    recorded no arm joints and a local replay diverged, so this state is synthetic: from
    reset, joint 6 set 0.005 rad past its stop (MuJoCo's limits are soft, and a loaded
    arm can rest just beyond one), with the hand empty so the jump drops nothing."""
    import mujoco

    from hitl_pmp.environments.racktoserving3d.environment import RackServingEnvironment

    env = RackServingEnvironment()
    try:
        env.reset_to_seed(seed=0)
        controllers = env.rack_controllers()
        shelf, tool = controllers.shelf, controllers.tool
        model, data = shelf.model, shelf.data
        joint = shelf.joints[5]
        data.qpos[shelf.qa[5]] = model.jnt_range[joint, 1] + 0.005
        mujoco.mj_forward(model, data)
        here = shelf.site_pose()
        with controllers.feedback():
            tool.transit(
                position=here[:3, 3] + [0.0, 0.0, 0.03],
                rotation=here[:3, :3],
                label="recover_past_stop",
                holding=False,
            )
        assert data.qpos[shelf.qa[5]] <= model.jnt_range[joint, 1]
    finally:
        env.close()


def test_return_scraper_after_a_sweep_that_failed_mid_stroke_escapes_and_returns():
    """Seed 1, PickScraper then GatherAndSweep(0.07, 0.030, 0.10), which stopped at
    "sweep_level: measured scraper pose failed to converge" with the blade over the rack.
    The scripted clear-and-circulate prefix returned from here; the planner alone could
    not start (the blade inside its margin) and refused, 0 steps."""
    env = restored_holding(fixture=FIXTURES / "after_failed_sweep_level_seed1.json")
    try:
        assert env.backend().held(name="wiper_0")
        steps, error = done_or_refused(env=env, action=np.array([2.0, 0, 0, 0, 0]))
        assert error is None, f"{steps} steps, then: {error}"
        assert env.rack_controllers().tool.at_home()
    finally:
        env.close()


CURTIS0_RIM = FIXTURES / "stuck_curtis0_blade_on_plate_rim.json"


@pytest.mark.parametrize(
    "action",
    [
        np.array([1.0, 0.0, 0.05390138822446226, 0.034378602500964084, 0.10454009583688345]),
        np.array([6.0, 0.0, 0.0, 0.0, 0.11124104374405863]),
    ],
    ids=["GatherAndSweep", "Sweep"],
)
def test_a_stroke_from_a_blade_resting_on_the_plate_rim_escapes_before_raising(*, action):
    """Curtis seed 0 (c6ed1a6), from practice step 17013: 76 consecutive calls refused
    at 0 steps, "squeeze_plus_y_raise" / "sweep_raise: Cartesian fraction 0.250:
    axis-clipped joint segment at 0.075 rad: Scraper blade or handle would collide". A
    Sweep that failed at the plate left the blade resting on the plate's rim, its edge
    centre 6 mm beyond the plate's well, so `exit_surface` (rack or well only) let the
    stroke raise straight up from the rim, with the arm nearly straight from the rear
    stance. These are the run's own two actions. The strokes after the escape may still
    fail (that is the skill's competence); leaving the rim and raising must not."""
    env = restored_holding(fixture=CURTIS0_RIM)
    try:
        assert env.backend().held(name="wiper_0")
        steps, error = done_or_refused(env=env, action=action)
        assert steps > 0, error
        for phase in ("_exit_retreat", "_lift_off", "_raise"):
            assert phase not in (error or ""), f"{steps} steps, then: {error}"
    finally:
        env.close()


EES0_PICK_FOOD = FIXTURES / "stuck_ees0_pick_food_drops.json"
EES0_FOOD_IN_FLIGHT = FIXTURES / "stuck_ees0_food_in_flight.json"


def food_in_motion_or_hand(*, env) -> dict[str, tuple[float, bool]]:
    """Each food that is still moving or touches the robot: (speed m/s, touching)."""
    backend = env.backend()
    model, data = backend.model(), backend.data()
    robot = env.rack_controllers().shelf.robot_geoms
    found = {}
    for name in backend.cube_names:
        dof = model.jnt_dofadr[model.joint(f"{name}_joint").id]
        speed = float(np.linalg.norm(data.qvel[dof : dof + 3]))
        touching = any(
            g in robot and model.body(model.geom_bodyid[o]).name.startswith(name)
            for c in data.contact[: data.ncon]
            for g, o in ((c.geom1, c.geom2), (c.geom2, c.geom1))
        )
        if speed > 0.02 or touching:
            found[name] = (round(speed, 3), touching)
    return found


def test_a_pick_food_that_loses_its_food_leaves_it_at_rest_and_the_scraper_pickable():
    """EES seed 0 (c6ed1a6), practice step 5272: PickFood(cube_0) lifted, then failed
    "Food cube_0 is not held by bilateral pad contact" with the hand still closed and
    the dumpling falling beside it (z 1.07 m, 15 cm above the counter). Physics only
    advances inside a skill, so every later plan saw it hanging there: 10 PickScraper
    calls in a row refused at 0 steps, "no stance reaches the scraper (No
    collision-free base path ...: 239, ...)", about 20 min each on Della. Replayed
    locally from the PickFood's own start, the same failure and the same refusal,
    count for count, follow."""
    env = restored_holding(fixture=EES0_PICK_FOOD)
    try:
        steps, error = done_or_refused(env=env, action=np.array([3.0, 0, 0, 0, 0]))
        assert steps > 0
        assert error is not None and "not held" in error, error
        assert food_in_motion_or_hand(env=env) == {}
        steps, error = done_or_refused(env=env, action=np.array([0.0, 0, 0, 0, 0]))
        assert steps > 0, error
        assert error is None, f"{steps} steps, then: {error}"
    finally:
        env.close()


def restored_frozen(*, fixture: Path):
    """The recorded motion state exactly, with no settling step: as a zero-step call
    after the recorded tick saw it."""
    import json

    import mujoco

    from hitl_pmp.environments.racktoserving3d.config import RackServingConfig
    from hitl_pmp.environments.racktoserving3d.environment import RackServingEnvironment
    from hitl_pmp.environments.racktoserving3d.tasks import RackServingTasks

    plain = json.loads(fixture.read_text())
    config = RackServingConfig.model_validate({
        k: v for k, v in plain["config"].items() if k in RackServingConfig.model_fields
    })
    env = RackServingEnvironment(config=config)
    RackServingTasks(env=env, seed=0).sample_train_task()
    backend = env.backend()
    model, data = backend.model(), backend.data()
    for joint, value in plain["robot_joints"].items():
        data.qpos[model.jnt_qposadr[model.joint(joint).id]] = value
    for name, pose in plain["free_bodies"].items():
        if name in ("rack_0", "box_0"):
            # Welded since 2026-11-06; the fixture predates that and is restored
            # exactly, including a plate a push had already moved. A welded pose is
            # otherwise fixed from reset on, so the controllers' long-lived IK data
            # is told of the move here.
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
    return env


def test_a_food_lost_from_the_closed_hand_is_waited_for_until_it_rests():
    """The exact recorded state the EES loop froze in: cube_0 against the closed
    hand's right pad, 15 cm above the counter, not held. Losing a food waits, the arm
    still and the hand as it is, so nothing is left in flight for the next plan."""
    from hitl_pmp.environments.racktoserving3d.food import RackFoodController
    from hitl_pmp.environments.sweepoffshelf3d.scripted_skills import SkillFailed

    env = restored_frozen(fixture=EES0_FOOD_IN_FLIGHT)
    try:
        assert "cube_0" in food_in_motion_or_hand(env=env)
        food = RackFoodController(controllers=env.rack_controllers())
        with pytest.raises(SkillFailed, match="not held"):
            food._require_held(name="cube_0")
        assert food_in_motion_or_hand(env=env) == {}
        assert env.rack_controllers().shelf.grip == 1.0
    finally:
        env.close()


EES0_PICK_SCRAPER_HAND_OVER_RACK = FIXTURES / "stuck_ees0_pickscraper_hand_over_rack.json"


def test_pick_scraper_from_a_hand_left_over_the_rack_leaves_it_and_picks():
    """rack-80k-32db025 EES seed 0, practice step 11736: a PickFood(cube_0) failed
    ("not held by bilateral pad contact") and left the closed, empty hand 3 cm above
    the rack, reaching 0.77 m over the counter from the base's side stance, 11 mm from
    cube_2 in PyBullet: inside the planner's 15 mm margin, though MuJoCo's guard is
    clear. Every PickScraper then refused at 0 steps, all 385 stances, 237 of them "No
    collision-free base path": no base path can start from a configuration the body
    check rejects. PickFood and ReturnScraper already leave such a start first."""
    env = restored_frozen(fixture=EES0_PICK_SCRAPER_HAND_OVER_RACK)
    try:
        assert not env.backend().held(name="wiper_0")
        steps, error = done_or_refused(env=env, action=np.array([0.0, 0, 0, 0, 0]))
        assert steps > 0, error
        assert error is None, f"{steps} steps, then: {error}"
        assert env.backend().held(name="wiper_0")
    finally:
        env.close()


EES1_HELD_FOOD_ON_RACK = FIXTURES / "stuck_ees1_held_food_place_start_blocked.json"


@pytest.mark.parametrize(
    "action",
    [np.array([4.0, 4.0, 0.0, 0.0, 0.0]), np.array([5.0, 4.0, 0.0, 0.0, 0.0])],
    ids=["PlaceFoodOnTray", "ReturnFoodToRack"],
)
def test_a_food_held_resting_on_the_rack_leaves_the_blocked_start(*, action):
    """rack-80k-32db025 EES seed 1, practice step 18,044: a PickFood(cube_4) ended
    "not held by bilateral pad contact", yet both closed pads still touch cube_4, so it
    counts as held, resting on the rack's wires (MuJoCo: 0.09 to 0.24 mm wire
    contacts, the execution guard clear). Carried rigidly in PyBullet it overlaps
    three wires by 1.2 mm and sits 5 mm from cube_3, inside the 15 mm planning margin,
    so no plan could start: every PlaceFoodOnTray and ReturnFoodToRack refused at 0
    steps, 12 in a row. A blocked start is now left along a checked escape first.

    The pads only touch the food; they do not squeeze it (20 steps of closed-hand
    settling lose it too), so the escape's first step lifts the hand off it and the
    skill fails "not held", the hand still closed and the food at rest on the rack.
    That ends the loop: the food is no longer held, so the next call differs."""
    env = restored_frozen(fixture=EES1_HELD_FOOD_ON_RACK)
    try:
        assert env.backend().held(name="cube_4")
        steps, error = done_or_refused(env=env, action=action)
        assert steps > 0, error
        if error is not None:
            assert "not held" in error, f"{steps} steps, then: {error}"
            assert env.rack_controllers().shelf.grip == 1.0, "a failed food skill opened"
            assert food_in_motion_or_hand(env=env) == {}
            assert env.backend().location(name="cube_4") == "OnRack"
    finally:
        env.close()
