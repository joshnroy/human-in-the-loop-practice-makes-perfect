import math

import numpy as np
import pytest

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
from hitl_pmp.environments.lightswitch.skills import LightSwitchSkills
from hitl_pmp.environments.lightswitch.tasks import LightSwitchTasks
from hitl_pmp.methods.practice_makes_perfect.ees_method import EesMethod


def _build(*, grid_size: int = 4, seed: int = 0) -> tuple[EesMethod, LightSwitchEnvironment]:
    env = LightSwitchEnvironment(grid_size=grid_size)
    return EesMethod(env=env, seed=seed), env


def _turn_on_light(*, env: LightSwitchEnvironment) -> GroundSkill:
    cells = env.get_cells()
    return GroundSkill(
        skill=LightSwitchSkills.TURN_ON_LIGHT,
        objects=(env.robot, cells[-1], env.light),
    )


def _move_robot_backwards(*, env: LightSwitchEnvironment) -> GroundSkill:
    """A legal MoveRobot that no optimal plan from cell0 to the light ever uses --
    it walks the wrong way."""
    cells = env.get_cells()
    return GroundSkill(skill=LightSwitchSkills.MOVE_ROBOT, objects=(env.robot, cells[2], cells[1]))


def _record_one_seen_task(*, method: EesMethod, env: LightSwitchEnvironment) -> None:
    tasks = LightSwitchTasks(env=env, seed=0)
    task = tasks.sample_train_task()
    method.record_seen_task(
        init_atoms=method.abstract_state(state=task.initial_state), goal=task.goal.atoms
    )


def test_skill_costs_are_negative_log_competence() -> None:
    """The load-bearing EES identity: minimizing summed -log(competence) over a plan
    maximizes the product of competences, i.e. the probability the plan succeeds
    without replanning (the paper's J_task objective)."""
    method, env = _build()
    skill = _turn_on_light(env=env)
    method.observe_outcome(ground_skill=skill, success=True)

    costs = method.skill_costs()
    competence = method.competence_model(ground_skill=skill).get_current_competence()
    assert costs[skill] == pytest.approx(-math.log(competence))


def test_default_cost_is_the_beta_prior_mean() -> None:
    """Ground skills never executed get predicators' default cost, -log of the
    Beta(10, 1) prior mean -- not an arbitrary constant."""
    method, _ = _build()
    assert method.default_cost() == pytest.approx(-math.log(10.0 / 11.0))


def test_observe_outcome_records_into_that_skills_competence_model() -> None:
    method, env = _build()
    skill = _turn_on_light(env=env)
    before = method.competence_model(ground_skill=skill).get_current_competence()
    for _ in range(5):
        method.observe_outcome(ground_skill=skill, success=False)
    after = method.competence_model(ground_skill=skill).get_current_competence()
    assert after < before


def test_score_prefers_a_skill_whose_improvement_helps_the_seen_tasks() -> None:
    """Planning progress: a skill that appears in the cached plans for seen tasks
    should score strictly better than one that appears nowhere, because only the
    former reduces those plans' total cost when its competence is extrapolated up."""
    method, env = _build()
    _record_one_seen_task(method=method, env=env)
    used = _turn_on_light(env=env)
    unused = _move_robot_backwards(env=env)
    # Both have been tried (so both are candidates), with identical histories so the
    # only thing distinguishing them is whether they appear in the cached plans.
    for skill in (used, unused):
        method.observe_outcome(ground_skill=skill, success=True)
        method.observe_outcome(ground_skill=skill, success=False)
    method.advance_competence_cycle()
    for skill in (used, unused):
        method.observe_outcome(ground_skill=skill, success=True)

    assert method.score_ground_skill(ground_skill=used) > method.score_ground_skill(
        ground_skill=unused
    )


def test_score_skips_a_perfect_skill() -> None:
    """predicators' active_sampler_explorer_skip_perfect: a skill already at 100%
    measured success is not worth practicing, so it scores -inf."""
    method, env = _build()
    skill = _turn_on_light(env=env)
    _record_one_seen_task(method=method, env=env)
    for _ in range(10):
        method.observe_outcome(ground_skill=skill, success=True)
    assert method.score_ground_skill(ground_skill=skill) == -math.inf


def test_end_cycle_advances_every_competence_model() -> None:
    method, env = _build()
    skill = _turn_on_light(env=env)
    method.observe_outcome(ground_skill=skill, success=True)
    model = method.competence_model(ground_skill=skill)
    assert len(model.cycle_observations) == 1
    method.end_cycle()
    assert len(model.cycle_observations) == 2


def test_evaluation_policy_records_no_training_data() -> None:
    """get_task_policy runs on held-out test tasks -- learning from it would be
    training on the test set. Pinning that it doesn't."""
    method, env = _build()
    tasks = LightSwitchTasks(env=env, seed=0)
    task = tasks.sample_train_task()
    env.set_state(state=task.initial_state)

    policy = method.get_task_policy(task=task)
    state = env.get_current_state()
    for _ in range(6):
        state = env.take_action(action=policy(state).action)

    assert method.total_observations() == 0


def test_practice_policy_records_training_data() -> None:
    method, env = _build()
    tasks = LightSwitchTasks(env=env, seed=0)
    task = tasks.sample_train_task()
    env.set_state(state=task.initial_state)

    policy = method.get_practice_policy(task=task)
    state = env.get_current_state()
    for _ in range(8):
        state = env.take_action(action=policy(state).action)

    assert method.total_observations() > 0


def test_practice_policy_eventually_tries_the_impossible_skill_then_deprioritizes_it() -> None:
    """The Light Switch trap: JumpToLight can be planned for but never achieves its
    claimed effect. EES must be able to observe that and drive its competence down,
    which is what makes -log(competence) blow up and steer plans away from it."""
    method, env = _build()
    jump_cells = env.get_cells()
    jump = GroundSkill(
        skill=LightSwitchSkills.JUMP_TO_LIGHT,
        objects=(env.robot, jump_cells[0], jump_cells[1], jump_cells[-1], env.light),
    )
    for _ in range(20):
        method.observe_outcome(ground_skill=jump, success=False)
    competence = method.competence_model(ground_skill=jump).get_current_competence()
    assert competence < 0.5
    assert method.skill_costs()[jump] > method.default_cost()


def test_labels_report_the_ground_skill_that_produced_the_action() -> None:
    method, env = _build()
    tasks = LightSwitchTasks(env=env, seed=0)
    task = tasks.sample_train_task()
    env.set_state(state=task.initial_state)
    policy = method.get_task_policy(task=task)
    labeled = policy(env.get_current_state())
    assert any(
        labeled.label.startswith(name)
        for name in ("MoveRobot", "TurnOnLight", "TurnOffLight", "JumpToLight")
    )


def test_two_methods_with_the_same_seed_behave_identically() -> None:
    method_a, env_a = _build(seed=7)
    method_b, env_b = _build(seed=7)
    tasks_a = LightSwitchTasks(env=env_a, seed=0)
    tasks_b = LightSwitchTasks(env=env_b, seed=0)
    task_a = tasks_a.sample_train_task()
    task_b = tasks_b.sample_train_task()
    env_a.set_state(state=task_a.initial_state)
    env_b.set_state(state=task_b.initial_state)

    policy_a = method_a.get_practice_policy(task=task_a)
    policy_b = method_b.get_practice_policy(task=task_b)
    state_a, state_b = env_a.get_current_state(), env_b.get_current_state()
    for _ in range(5):
        labeled_a = policy_a(state_a)
        labeled_b = policy_b(state_b)
        assert labeled_a.label == labeled_b.label
        assert np.allclose(labeled_a.action, labeled_b.action)
        state_a = env_a.take_action(action=labeled_a.action)
        state_b = env_b.take_action(action=labeled_b.action)


def test_ees_learns_to_solve_light_switch_over_practice_cycles() -> None:
    """The headline claim, asserted rather than assumed: EES's final evaluation
    strictly beats its own first (pre-practice) evaluation on held-out test tasks.
    This is the whole point of the port -- the sampler for TurnOnLight starts as a
    uniform prior over dlight and has to be specialized by practice."""
    import argparse

    from hitl_pmp.environments.lightswitch.problem import LightSwitchProblem
    from hitl_pmp.method_runner import MethodRunner

    env = LightSwitchEnvironment(grid_size=5)
    problem = LightSwitchProblem(env=env, tasks=LightSwitchTasks(env=env, seed=0))
    metrics = MethodRunner.run(
        args=argparse.Namespace(num_test_tasks=5, output_dir=None),
        method=EesMethod(env=env, seed=0, sampler_max_train_iters=300),
        problem=problem,
        num_cycles=6,
        max_steps_per_interaction=40,
        renderer=None,
        render_fps=2,
    )
    curve = metrics.task_training_curve()
    assert len(curve) == 7  # initial evaluation + one per cycle
    assert curve[-1][1] > curve[0][1]


def test_random_exploration_attempts_are_kept_out_of_competence_but_kept_as_sampler_data() -> None:
    """predicators suppresses the competence update when the epsilon-greedy random
    branch fires. Competence has to mean "how good is this skill when the robot
    actually tries", not "how often does a coin flip work" -- at epsilon=0.5 the
    latter roughly halves the apparent competence of a mastered skill, corrupting
    the plan costs and practice scores computed from it. The sampler still keeps
    the attempt: a deliberately random parameter that failed is exactly the
    negative example the classifier needs.

    Note the epsilon branch only exists once a sampler has been *fitted* -- before
    that there is nothing to be greedy about, so the first cycle's attempts do
    update competence (predicators behaves the same way, using the unwrapped base
    sampler until the first learning cycle). Hence the warm-up cycle below."""
    env = LightSwitchEnvironment(grid_size=4)
    # epsilon=1.0 => once fitted, every parameterized attempt takes the random branch.
    method = EesMethod(env=env, seed=0, exploration_epsilon=1.0, sampler_max_train_iters=50)
    tasks = LightSwitchTasks(env=env, seed=0)

    def _practice(*, steps: int) -> None:
        task = tasks.sample_train_task()
        env.set_state(state=task.initial_state)
        policy = method.get_practice_policy(task=task)
        state = env.get_current_state()
        for _ in range(steps):
            state = env.take_action(action=policy(state).action)

    _practice(steps=12)
    method.end_cycle()  # fits the samplers, so the epsilon branch now exists

    def _parameterized_competence_observations() -> int:
        return sum(
            method.competence_model(ground_skill=ground_skill).num_observations
            for ground_skill in method._competence_models
            if ground_skill.skill.param_dim > 0
        )

    competence_before = _parameterized_competence_observations()
    sampler_before = method.sampler(skill_name="TurnOnLight", param_dim=1).num_observations

    _practice(steps=12)

    assert _parameterized_competence_observations() == competence_before
    assert method.sampler(skill_name="TurnOnLight", param_dim=1).num_observations > sampler_before
    # Param-free skills (MoveRobot) have no sampler and so no epsilon branch --
    # their competence keeps being tracked normally.
    assert method.total_observations() > competence_before
