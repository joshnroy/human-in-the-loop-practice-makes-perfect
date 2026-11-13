"""RobotBatchReturn: the robot puts every dumpling back on its start slot itself.

The robot counterpart of the person's batch return (Josh, 2026-11-13), opt-in by
`RackServingConfig.robot_batch_return`. Setup A is the two-route Serving setup the
budget sweeps run: three dumplings in the middle of the rack and one out on each side.
"""

import importlib.util

import numpy as np
import pytest
from pydantic import ValidationError

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.environments.racktoserving3d.config import RackServingConfig
from hitl_pmp.environments.racktoserving3d.objects import RackServing3DObjects as Objects
from hitl_pmp.environments.racktoserving3d.skill_provider import RackServingActions
from hitl_pmp.environments.racktoserving3d.skills import SKILL_IDS
from hitl_pmp.environments.sweepoffshelf3d.scripted_skills import SkillFailed

kinder = pytest.mark.skipif(importlib.util.find_spec("kinder") is None, reason="KINDER simulator")
SETUP_A = {
    "start_layout": "middle_three",
    "side_offset": 0.13,
    "plate_tuck": 0.035,
    "surface_kind": "wire_solid",
    "scraper_attachment": "grasp_weld",
    "two_route_rewards": True,
    "bulk_transfer_only": True,
}
WITH_ROBOT_RETURN = RackServingConfig(**SETUP_A, robot_batch_return=True)
SWEEP_D = 0.08
GATHER_AND_SWEEP = (0.07, 0.03, 0.10)
MIDDLE = ("cube_1", "cube_2", "cube_3")


def _provider(*, config):
    from hitl_pmp.environments.racktoserving3d.environment import RackServingEnvironment
    from hitl_pmp.environments.racktoserving3d.skill_provider import RackServingSkillProvider

    return RackServingSkillProvider(env=RackServingEnvironment(config=config))


def _act(*, provider, name, params=()):
    """Run the provider's skill `name` on its only grounding; return the skill error."""
    skill = next(skill for skill in provider.skills() if skill.name == name)
    objects = {
        "PickScraper": (Objects.robot, Objects.scraper),
        "ReturnScraper": (Objects.robot, Objects.scraper),
        "Sweep": (Objects.robot, Objects.scraper, Objects.rack, Objects.tray, *Objects.food),
        "GatherAndSweep": (
            Objects.robot,
            Objects.scraper,
            Objects.rack,
            Objects.tray,
            *Objects.food,
        ),
        "RobotBatchReturn": (Objects.robot, Objects.rack, Objects.tray, *Objects.food),
    }[name]
    ground = GroundSkill(skill=skill, objects=objects)
    provider.env.take_action(
        action=RackServingActions.encode(ground_skill=ground, params=np.array(params))
    )
    return provider.env.last_skill_error()


def _atoms(*, env):
    return env.get_current_state().abstract_atoms


def _on_plate(*, env):
    backend = env.backend()
    return {name for name in backend.cube_names if backend.settled_on_tray(name=name)}


def _assert_the_start_layout_is_back(*, env):
    """Every dumpling rests on the rack within the layout's own jitter of its slot."""
    backend, config = env.backend(), env.config
    atoms = _atoms(env=env)
    assert ("AnyFoodOffRack", ()) not in atoms
    assert not {name for name, _ in atoms} & {"Served", "ServedThree", "ServedFive"}
    offsets = (
        -config.side_offset,
        -config.spaced_pitch,
        0.0,
        config.spaced_pitch,
        config.side_offset,
    )
    for name, offset in zip(backend.cube_names, offsets, strict=True):
        assert ("OnRack", (name, "rack_0")) in atoms
        assert ("FoodAvailable", (name,)) in atoms
        x, y = backend.food_centre(name=name)
        assert abs(x - config.rack_center_x) <= 0.0071, name
        assert abs(y - offset) <= 0.0071, name
        assert abs(backend.food_yaw(name=name) - np.pi / 2) <= 0.171, name


def test_the_robot_batch_return_is_off_by_default_and_needs_slots_and_welded_food():
    assert RackServingConfig().robot_batch_return is False
    assert RackServingConfig(**SETUP_A).robot_batch_return is False
    assert WITH_ROBOT_RETURN.robot_batch_return_yaw == "snap"
    with pytest.raises(ValidationError, match="robot_batch_return"):
        RackServingConfig(robot_batch_return=True)
    with pytest.raises(ValidationError, match="robot_batch_return"):
        RackServingConfig(**SETUP_A, robot_batch_return=True, food_handling="physical")


def test_the_provider_offers_the_robot_batch_return_only_under_the_flag_and_in_training():
    plain = _provider(config=RackServingConfig(**SETUP_A))
    offered = _provider(config=WITH_ROBOT_RETURN)
    deployed = _provider(config=WITH_ROBOT_RETURN.model_copy(update={"mode": "deployment"}))
    names = [skill.name for skill in plain.skills()]
    assert "RobotBatchReturn" not in names
    assert [skill.name for skill in offered.skills()] == [*names, "RobotBatchReturn"]
    assert deployed.skills() == plain.skills()
    # Still the robot's own action, never one of the person's.
    assert "RobotBatchReturn" not in {
        ground.skill.name for ground in offered.human_cube_bin_reset_skills()
    }
    assert SKILL_IDS["RobotBatchReturn"] == 7
    ground = GroundSkill(
        skill=offered.skills()[-1],
        objects=(Objects.robot, Objects.rack, Objects.tray, *Objects.food),
    )
    action = RackServingActions.encode(ground_skill=ground, params=np.zeros(0))
    assert action.tolist() == [7.0, 0.0, 0.0, 0.0, 0.0]
    assert offered.sample_params(ground_skill=ground, rng=np.random.default_rng(0)).size == 0


@kinder
def test_the_robot_batch_return_is_refused_unless_enabled():
    provider = _provider(config=RackServingConfig(**SETUP_A))
    env = provider.env
    try:
        env.reset_to_seed(seed=0)
        env.take_action(action=np.array([SKILL_IDS["RobotBatchReturn"], 0, 0, 0, 0.0]))
        assert "not enabled" in (env.last_skill_error() or "")
    finally:
        env.close()


@kinder
def test_after_a_sweep_the_robot_return_restores_the_start_and_a_sweep_lands_three_again():
    provider = _provider(config=WITH_ROBOT_RETURN)
    env = provider.env
    try:
        env.reset_to_seed(seed=0)
        assert _act(provider=provider, name="PickScraper") is None
        assert _act(provider=provider, name="Sweep", params=(SWEEP_D,)) is None
        assert _on_plate(env=env) == set(MIDDLE)
        assert ("Served", ()) in _atoms(env=env)
        human_cost = env.backend().ledger.human_cost

        assert _act(provider=provider, name="RobotBatchReturn") is None

        _assert_the_start_layout_is_back(env=env)
        # One action from the state bulk practice is in: the scraper is held again.
        assert ("HoldingScraper", ("robot", "wiper_0")) in _atoms(env=env)
        assert env.backend().ledger.human_cost == human_cost
        report = env.rack_controllers().metrics["robot_batch_return"]
        # The blade is as wide as the outer two are far apart, so it can nudge them.
        assert set(MIDDLE) <= set(report["moved"])
        # The one abstraction (`robot_batch_return_yaw` "snap"): the hand brought each
        # dumpling to its slot, and the set-down wrote the slot's yaw.
        assert report["placed"].keys() == set(report["moved"])
        for placed in report["placed"].values():
            assert placed["yaw_written_by_set_down"] is True
            assert placed["position_error_m"] <= 0.002
            assert placed["yaw_error_rad"] <= 0.02
        assert report["scraper_carried"] is True
        # As after the person's return: the same Sweep lands the same three.
        assert _act(provider=provider, name="Sweep", params=(SWEEP_D,)) is None
        assert _on_plate(env=env) == set(MIDDLE)
    finally:
        env.close()


@kinder
def test_after_a_gather_and_sweep_the_robot_return_restores_the_start_and_it_runs_again():
    provider = _provider(config=WITH_ROBOT_RETURN)
    env = provider.env
    try:
        env.reset_to_seed(seed=0)
        assert _act(provider=provider, name="PickScraper") is None
        assert _act(provider=provider, name="GatherAndSweep", params=GATHER_AND_SWEEP) is None
        first = _on_plate(env=env)
        assert len(first) == 5

        assert _act(provider=provider, name="RobotBatchReturn") is None

        _assert_the_start_layout_is_back(env=env)
        assert ("HoldingScraper", ("robot", "wiper_0")) in _atoms(env=env)
        assert _act(provider=provider, name="GatherAndSweep", params=GATHER_AND_SWEEP) is None
        landed = _on_plate(env=env)
        assert len(landed) >= 4
        if landed != first:
            # The dumplings are back exactly where they were; the scraper is not. The
            # return puts it home and picks it up again, from wherever its last
            # placement left the robot, and a GatherAndSweep at these parameters lands
            # 4 or 5 depending on that grasp. A person's return leaves the scraper in
            # the hand. Measured on setup A (2026-11-13): 5 landed on 16/18 seeds after
            # a full robot return before the motion fixes, 12/19 after them; paired
            # over 17 seeds, 4 went from 5 to 4 and none the other way (exact McNemar
            # p = 0.125). Seed 0 is one of the four. Open for Josh's decision.
            pytest.xfail("a GatherAndSweep after a robot return depends on its scraper re-grasp")
    finally:
        env.close()


@kinder
def test_a_robot_return_that_fails_part_way_leaves_the_scene_as_it_is(*, monkeypatch):
    from hitl_pmp.environments.racktoserving3d.robot_batch_return import RackRobotBatchReturn

    provider = _provider(config=WITH_ROBOT_RETURN)
    env = provider.env
    try:
        env.reset_to_seed(seed=0)
        assert _act(provider=provider, name="PickScraper") is None
        assert _act(provider=provider, name="Sweep", params=(SWEEP_D,)) is None
        put_back = RackRobotBatchReturn.put_back
        calls = []

        def refused_on_the_third(self, *, name, slot):  # noqa: PLR0917 -- a method's self
            calls.append(name)
            if len(calls) == 3:
                raise SkillFailed("No safe food placement: refused for this test")
            put_back(self, name=name, slot=slot)

        monkeypatch.setattr(RackRobotBatchReturn, "put_back", refused_on_the_third)
        error = _act(provider=provider, name="RobotBatchReturn")
        # All five were out of place: the blade nudges the outer two as it passes.
        assert "returned 2 of 5" in error
        assert "refused for this test" in error
        backend = env.backend()
        atoms = _atoms(env=env)
        for name in calls[:2]:
            assert ("OnRack", (name, "rack_0")) in atoms
        assert backend.location(name=calls[2]) == "OnTray"
        # As measured: the scraper was put home and the hand is empty. Both returns
        # are still applicable; no bulk skill is.
        assert ("AnyFoodOffRack", ()) in atoms
        assert ("ScraperHome", ("wiper_0",)) in atoms
        assert ("HandEmpty", ("robot",)) in atoms
        assert env.rack_controllers().metrics["robot_batch_return"]["moved"] == calls[:2]
        # The plate no longer holds three, so the Sweep's verdict is withdrawn.
        assert ("Served", ()) not in atoms

        monkeypatch.setattr(RackRobotBatchReturn, "put_back", put_back)
        assert _act(provider=provider, name="RobotBatchReturn") is None
        _assert_the_start_layout_is_back(env=env)
        # It started with an empty hand, so it ends with one.
        assert ("HandEmpty", ("robot",)) in _atoms(env=env)
    finally:
        env.close()
