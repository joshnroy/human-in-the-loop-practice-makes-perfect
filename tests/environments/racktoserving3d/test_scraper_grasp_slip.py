"""The held scraper keeps its grasp while the base carries it round the counter.

Della dry run rack-dry-ca2b306-20261104 lost the scraper out of the pads in 13/15
failed evaluation episodes, during ReturnScraper's or GatherAndSweep's base routes.
Measured locally (seed 861111389): holding still or translating the base moves the
scraper in the hand by under 1 um per control step, but each in-place turn of pi
slides it 18 mm and turns it 8 deg in the grasp, so the pads walk off the end of the
handle after a few turns. The base position actuator receives one step target per
control period and snaps to it in milliseconds; the yaw jerk is what slips the grasp.
"""

import importlib.util

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("kinder") is None, reason="KINDER simulator"
)


def test_turning_the_base_in_place_does_not_slide_the_scraper_in_the_grasp():
    from hitl_pmp.environments.racktoserving3d.environment import RackServingEnvironment
    from hitl_pmp.environments.racktoserving3d.tasks import RackServingTasks

    env = RackServingEnvironment()
    try:
        RackServingTasks(env=env, seed=0).sample_train_task()
        env.reset_to_seed(seed=861111389)
        env.take_action(action=np.array([0.0, 0.0, 0.0, 0.0, 0.0]))
        assert env._last_skill_error is None
        controllers = env.rack_controllers()
        shelf, tool = controllers.shelf, controllers.tool
        config = env.backend().config
        rear_x = config.counter_x_max + 0.60
        outside_y = config.counter_y_min - 0.80
        with controllers.feedback():
            tool.move_base(target=np.array([tool.base[0], outside_y, 0.0]), label="out")
            tool.move_base(target=np.array([rear_x, outside_y, 0.0]), label="along")
            before = np.linalg.inv(shelf.site_pose()) @ shelf.wiper_pose()
            tool.move_base(target=np.array([rear_x, outside_y, np.pi]), label="turn")
            after = np.linalg.inv(shelf.site_pose()) @ shelf.wiper_pose()
        drift = float(np.linalg.norm(after[:3, 3] - before[:3, 3]))
        turn = Rotation.from_matrix(after[:3, :3] @ before[:3, :3].T).magnitude()
        assert tool.pads_hold_grip()
        assert drift < 0.002, f"scraper slid {drift * 1000:.1f} mm in the grasp"
        assert np.degrees(turn) < 1.0, f"scraper turned {np.degrees(turn):.1f} deg in the grasp"
    finally:
        env.close()
