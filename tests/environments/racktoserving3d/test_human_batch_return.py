"""The human batch return leaves every dumpling resting on the rack, as measured.

Della EES dry run: `human_batch_return` placed each dumpling 2 mm above the rack
and never stepped physics, so the measured state had no OnRack or FoodAvailable
and EES requested the return 200 times in a row.
"""

import importlib.util

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("kinder") is None, reason="KINDER simulator"
)


def _env_with_food_on_the_counter():
    from hitl_pmp.environments.racktoserving3d.environment import RackServingEnvironment
    from hitl_pmp.environments.racktoserving3d.tasks import RackServingTasks

    env = RackServingEnvironment()
    RackServingTasks(env=env, seed=0).sample_train_task()
    backend = env.backend()
    height = backend.config.counter_height + backend.config.dumpling_height / 2 + 0.002
    placements = {
        name: (np.array([1.02 + 0.12 * index, 0.3, height]), np.array([1.0, 0, 0, 0]))
        for index, name in enumerate(backend.cube_names)
    }
    backend._commit(placements=placements)
    env.rack_controllers().shelf.hold(count=40, label="food_onto_counter")
    env.current_state = env._observed_state(seed=0, steps_taken=1)
    return env


def test_batch_return_leaves_every_dumpling_measured_on_the_rack():
    env = _env_with_food_on_the_counter()
    try:
        before = env.get_current_state().abstract_atoms
        assert ("AnyFoodOffRack", ()) in before
        assert not any(atom[0] == "OnRack" for atom in before)
        env.reset_movables(destination="human_batch_return")
        atoms = env.get_current_state().abstract_atoms
        for name in env.backend().cube_names:
            assert ("OnRack", (name, "rack_0")) in atoms
            assert ("FoodAvailable", (name,)) in atoms
        assert ("AnyFoodOffRack", ()) not in atoms
    finally:
        env.close()


def test_batch_return_physics_is_not_a_robot_practice_step():
    from hitl_pmp.step_protocol import PracticeClock

    env = _env_with_food_on_the_counter()
    try:
        clock = PracticeClock(budget=5, interval=100, on_measure=lambda: None)
        env.set_control_step_observers(before=clock.before_robot_step, after=clock.after_robot_step)
        charge = clock.before_human(skill="human_batch_return")
        env.reset_movables(destination="human_batch_return")
        clock.human_invoked(charge=charge)
        assert clock.robot_steps == 0
        assert clock.steps == clock.human_invocations == 1
        assert env.backend().ledger.human_cost == env.backend().config.human_interruption_cost
        assert env.backend().ledger.human_interventions == 1
    finally:
        env.close()
