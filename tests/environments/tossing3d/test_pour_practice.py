"""A practice policy must choose and observe the physical recovery itself."""

import importlib.util
import json
from pathlib import Path

import pytest

from hitl_pmp.core.practice_costs import PracticeAccounting, PracticeCosts
from hitl_pmp.core.problem.tasks.types import Goal, GroundAtom, Task
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.predicates import HOLDING
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
from hitl_pmp.methods.belief_space.tossing3d_method import Tossing3DPomdpMethod
from hitl_pmp.methods.practice_makes_perfect.ees_method import EesMethod
from hitl_pmp.step_protocol import PracticeClock

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("kinder") is None, reason="KINDER unavailable"
)


@pytest.mark.parametrize("solver", ["ees", "expectimax", "curtis"])
def test_practice_policy_recovers_and_charges_real_ticks(*, solver):
    env = Tossing3DEnvironment(scene_bg=False)
    try:
        env.hard_reset()
        plain = json.loads(
            (Path(__file__).parent / "fixtures/pick_plannable/exp22_F.json").read_text()
        )
        state = env.restore_plain_snapshot(plain=plain)
        provider = Tossing3DSkillProvider(env=env, offer_human_reset=False)
        config = dict(env=env, skill_provider=provider, seed=0, goal_pursuit_horizon=0)
        method = (
            EesMethod(**config)
            if solver == "ees"
            else Tossing3DPomdpMethod(
                **config,
                pomdp_solver="expectimax" if solver == "expectimax" else "determinized_astar",
                pomdp_determinized_variant="algorithm3" if solver == "expectimax" else "curtis",
                pomdp_search_depth=6,
                pomdp_max_search_iterations=100,
                pomdp_num_samples=100,
                pomdp_num_particles=1024,
                pomdp_inference_engine="grid",
                pomdp_observation_probability_weight=0 if solver == "expectimax" else 0.01,
            )
        )
        costs = PracticeCosts.load(path="src/hitl_pmp/presets/tossing3d_tick_priced_costs.json")
        accounting = PracticeAccounting(costs=costs)
        method.configure_practice_accounting(accounting=accounting)
        clock = PracticeClock(
            budget=2000,
            interval=2000,
            human_steps=1,
            on_measure=lambda: None,
            on_progress=lambda: None,
            accounting=accounting,
        )
        env.set_control_step_observers(before=clock.before_robot_step, after=clock.after_robot_step)
        task = Task(
            initial_state=state,
            goal=Goal(
                atoms=frozenset({GroundAtom(predicate=HOLDING, objects=(env.robot, env.cube))})
            ),
        )
        policy = method.get_practice_policy(task=task)
        for index, expected_id in enumerate((env.pour_bin_id, env.pick_cube_id)):
            method.observe_practice_action_budget(remaining_actions=20 - index)
            action = policy(state)
            assert int(action.action[0]) == expected_id, action.label
            before_steps, before_cost = clock.steps, accounting.total_cost
            clock.robot_active = True
            state = env.take_action(action=action.action)
            clock.robot_active = False
            ticks = sum(env.last_controller_steps())
            method.observe_execution_cost(
                cost=accounting.total_cost - before_cost, steps=ticks, complete=True
            )
            assert env.last_skill_error() is None, env.last_skill_error()
            assert clock.steps - before_steps == ticks
            assert accounting.total_cost - before_cost == pytest.approx(ticks * 0.01)
            method.observe_environment_reset(state=state)
        assert HOLDING.holds(state, (env.robot, env.cube))
        outcomes = method.practice_outcomes()
        assert outcomes["PourCubeOutOfBin"].num_successes >= 1
        if solver != "ees":
            assert method.pomdp_state.pending_examples["PourCubeOutOfBin"] >= 1
            assert method.pomdp_state.accumulated_cost == pytest.approx(accounting.total_cost)
    finally:
        env.close()
