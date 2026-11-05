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
