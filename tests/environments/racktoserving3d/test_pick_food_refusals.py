"""PickFood states the c6ed1a6 launch refused at 0 steps with "No collision-free food
grasp: No planned reach to the food pre-grasp", each the recorded tick just before the
refused call in Curtis seed 1's evaluation at 64,000 practice steps (0/10 solved, all
10 ending on this refusal).

`pick_food_after_retreat_touching_served_food.json` (task 2, step 4079) and
`pick_food_pad_resting_on_served_food.json` (task 6, step 2862): PlaceFoodOnTray set its
dumpling down beside one already served, then failed its retreat at the first step
("Food motion would collide with ungrasped food"): the open hand rests touching the
served dumpling. No plan starts there, so every reach was refused.

`pick_food_hand_low_over_plate.json` (task 3, step 6887): the same failed retreat, the
hand touching no food but within the planner's margin of the plate.
"""

from pathlib import Path

import numpy as np

from tests.environments.racktoserving3d.test_planned_food import restored

FIXTURES = Path(__file__).parent / "fixtures"
RETREAT_TOUCHING = FIXTURES / "pick_food_after_retreat_touching_served_food.json"
PAD_ON_SERVED = FIXTURES / "pick_food_pad_resting_on_served_food.json"
HAND_OVER_PLATE = FIXTURES / "pick_food_hand_low_over_plate.json"


def picked(*, fixture: Path, food: int) -> None:
    env = restored(fixture=fixture)
    try:
        env.take_action(action=np.array([3.0, float(food), 0, 0, 0]))
        steps, error = env._last_controller_steps, env.last_skill_error()
        assert steps > 0, error
        assert error is None, f"{steps} steps, then: {error}"
        assert env.backend().held(name=f"cube_{food}")
    finally:
        env.close()


def test_pick_food_after_a_retreat_left_the_hand_touching_served_food_picks():
    picked(fixture=RETREAT_TOUCHING, food=2)


def test_pick_food_with_the_hand_left_low_over_the_plate_picks():
    picked(fixture=HAND_OVER_PLATE, food=4)


def test_a_pad_resting_on_served_food_may_leave_it_but_not_press_in():
    """Contact with an ungrasped dumpling present when a departing scope begins may
    persist, as a pad-fixture contact may; it may not deepen."""
    from hitl_pmp.environments.racktoserving3d.food import RackFoodController
    from hitl_pmp.environments.sweepoffshelf3d.scripted_skills import SkillFailed

    env = restored(fixture=PAD_ON_SERVED)
    try:
        food = RackFoodController(controllers=env.rack_controllers())
        food.target_geoms = food.backend.body_geoms(name="cube_2")
        shelf = food.shelf
        base, arm = shelf.data.qpos[shelf.ba].copy(), shelf.data.qpos[shelf.qa].copy()
        try:
            food.motion.check_configuration(base=base, arm=arm, holding=False)
        except SkillFailed as exc:
            assert "ungrasped food" in str(exc)
        else:
            raise AssertionError("the fixture's pad no longer touches the served dumpling")
        here = shelf.site_pose()
        with food.motion.departing_from_contacts():
            food.motion.check_configuration(base=base, arm=arm, holding=False)
            up = shelf.ik(
                position=here[:3, 3] + [0.0, 0.0, 0.02],
                rotation=here[:3, :3],
                base=base,
                holding=False,
                seed=arm,
            )[0]
            food.motion.check_configuration(base=base, arm=up, holding=False)
            down = shelf.ik(
                position=here[:3, 3] - [0.0, 0.0, 0.01],
                rotation=here[:3, :3],
                base=base,
                holding=False,
                seed=arm,
            )[0]
            try:
                food.motion.check_configuration(base=base, arm=down, holding=False)
            except SkillFailed:
                pass
            else:
                raise AssertionError("pressing 1 cm into the plate was allowed")
    finally:
        env.close()


def test_a_placement_retreat_starting_against_a_served_dumpling_leaves_it():
    """The state PlaceFoodOnTray(cube_1) left behind: cube_1 released beside cube_0,
    a pad on cube_0. Its retreat, the straight line back to the pre-placement, failed
    at the first step; the departing scope `place` now wraps it lets that contact go."""
    from hitl_pmp.environments.racktoserving3d.food import RackFoodController

    env = restored(fixture=PAD_ON_SERVED)
    try:
        food = RackFoodController(controllers=env.rack_controllers())
        food.target_geoms = food.backend.body_geoms(name="cube_1")
        shelf = food.shelf
        here = shelf.site_pose()
        pre = here[:3, 3] - 0.10 * here[:3, 2] + [0.0, 0.0, 0.08]
        with food.controllers.feedback(), food.motion.departing_from_contacts():
            food.motion.line(
                position=pre,
                rotation=here[:3, :3],
                label="food_retreat",
                holding=False,
                step=0.005,
            )
        assert np.linalg.norm(shelf.site_pose()[:3, 3] - pre) < 0.01
    finally:
        env.close()
