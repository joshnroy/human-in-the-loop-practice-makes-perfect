"""A coffee-table-height counter: the scraper skills must stand where the arm can reach."""

import numpy as np

from hitl_pmp.environments.racktoserving3d.config import RackServingConfig
from hitl_pmp.environments.racktoserving3d.environment import RackServingEnvironment

COFFEE_TABLE = RackServingConfig(counter_height=0.45, counter_width=0.8, tool_rest_y=-0.255)


def test_pick_scraper_holds_the_scraper_from_a_coffee_table():
    env = RackServingEnvironment(config=COFFEE_TABLE)
    try:
        env.reset_to_seed(seed=0)
        env.take_action(action=np.array([0.0, 0, 0, 0, 0]))
        assert env.last_skill_error() is None
        assert env.backend().held(name="wiper_0")
    finally:
        env.close()
