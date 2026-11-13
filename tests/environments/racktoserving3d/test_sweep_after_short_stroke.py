"""A Sweep that follows a short stroke finishes its release instead of being refused.

Learning-curve session 10 (2026-11-12), 8 seeds of setup A: 29/531 Sweep tries ended
"sweep_level: Cartesian fraction ...: axis-clipped joint segment at 0.025 rad: Scraper
blade or handle would collide with fixture geometry". All 29 had both their own and
the previous Sweep's d below 0.08 (29/125); none of the 125 short tries after a long
stroke and none of the 273 long tries was refused.

The refusal came at the end of the try's own stroke, on the first move of
`release_swept_food`. The arm has seven joints, and each try's approach starts from
where the previous one left the arm, so a short previous stroke hands on a different
arm posture for the same blade pose. In that posture one 1 cm waypoint of the levelling
move asks more than 0.025 rad of some joints, and the axis-clipped joint path (which the
servo does not follow) dips one blade end 3.13 mm under the rack top, 0.13 mm past the
3 mm support allowance. The straight path the servo does follow was clear.

This is seed 0's first refused try, replayed from the episode start: PickScraper, three
Sweeps with the person's batch return after each, then Sweep(0.021).
"""

import importlib.util

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("kinder") is None, reason="KINDER simulator"
)

SWEEPS_BEFORE = (0.08813802861464752, 0.11717547863232534, 0.03906547393208453)
REFUSED = 0.02104916999820973


def test_a_short_sweep_after_a_short_sweep_releases_and_delivers_nothing():
    from hitl_pmp.environments.racktoserving3d.config import RackServingConfig
    from hitl_pmp.environments.racktoserving3d.environment import RackServingEnvironment
    from hitl_pmp.environments.racktoserving3d.skills import SKILL_IDS

    env = RackServingEnvironment(
        config=RackServingConfig(
            scraper_attachment="grasp_weld",
            bulk_transfer_only=True,
            two_route_rewards=True,
            start_layout="middle_three",
            plate_tuck=0.035,
        ),
        scene_bg=False,
    )
    try:
        env.reset_to_seed(seed=0)
        env.take_action(action=np.array([SKILL_IDS["PickScraper"], 0, 0, 0, 0.0]))
        assert env.last_skill_error() is None
        for d in SWEEPS_BEFORE:
            env.take_action(action=np.array([SKILL_IDS["Sweep"], 0, 0, 0, d]))
            assert env.last_skill_error() is None
            assert env.backend().plate_count() == 3
            env.reset_movables(destination="human_batch_return")
        env.take_action(action=np.array([SKILL_IDS["Sweep"], 0, 0, 0, REFUSED]))
        # d = 0.021 stops the blade 1.4 cm short of the rack's front edge: a bad
        # parameter, which fails by delivering nothing and not by a refusal.
        assert env.last_skill_error() is None
        assert env.backend().plate_count() == 0
        assert env.backend().held(name="wiper_0")
    finally:
        env.close()
