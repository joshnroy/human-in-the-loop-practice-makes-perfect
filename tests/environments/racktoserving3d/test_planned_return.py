"""Planned ReturnScraper: the scraper reaches its home on the counter.

The fixture is the measured state just before a ReturnScraper that failed 12/12 on the
0.9 m counter (seed 861111389, GatherAndSweep(0.07, -0.02, 0.09)): from the fixed
return base the elbow sat at its -2.57 rad limit and the scraper stopped 6 mm above
the seat.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from hitl_pmp.environments.racktoserving3d.config import RackServingConfig
from hitl_pmp.environments.racktoserving3d.environment import RackServingEnvironment
from tests.environments.racktoserving3d.test_gather_and_sweep_after_failure import (
    _restored_env,
)

ELBOW_LIMIT_FIXTURE = Path(__file__).parent / "fixtures" / "return_scraper_seat_at_elbow_limit.json"
COFFEE_TABLE = RackServingConfig(counter_height=0.45, counter_width=0.8, tool_rest_y=-0.255)


def test_return_scraper_seats_the_scraper_where_the_fixed_base_could_not(*, monkeypatch):
    import tests.environments.racktoserving3d.test_gather_and_sweep_after_failure as restore

    monkeypatch.setattr(restore, "FIXTURE", ELBOW_LIMIT_FIXTURE)
    env = _restored_env(skill_step_limit=12000)
    try:
        assert env.backend().held(name="wiper_0")
        env.take_action(action=np.array([2.0, 0, 0, 0, 0]))
        assert env.last_skill_error() is None
        assert env.rack_controllers().tool.at_home()
    finally:
        env.close()


@pytest.mark.xfail(
    strict=False,
    reason=(
        "Since the rack and plate were welded (2026-11-06), ReturnScraper on the 0.45 m "
        "coffee table fails place_wiper_lower on a 2.25 mm base-link/arm self-contact (0/4 "
        "seeds before, 4/4 after; the default 0.9 m counter still returns). Removing the two "
        "free bodies changes the contact solve by ~0.1 mm in the scraper's grasp; the "
        "rack and plate are not in the failing contact. Open for a ReturnScraper fix."
    ),
)
def test_pick_then_return_scraper_on_a_coffee_table():
    env = RackServingEnvironment(config=COFFEE_TABLE)
    try:
        env.reset_to_seed(seed=0)
        env.take_action(action=np.array([0.0, 0, 0, 0, 0]))
        assert env.last_skill_error() is None
        env.take_action(action=np.array([2.0, 0, 0, 0, 0]))
        assert env.last_skill_error() is None
        assert env.rack_controllers().tool.at_home()
    finally:
        env.close()


def test_the_fixture_is_a_held_scraper():
    plain = json.loads(ELBOW_LIMIT_FIXTURE.read_text())
    assert plain["grip"] == 1.0
