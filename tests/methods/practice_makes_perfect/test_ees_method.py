import math

import numpy as np
import pytest

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.core.problem.tasks.types import Goal, Task
from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
from hitl_pmp.environments.lightswitch.predicates import ADJACENT, LIGHT_ON
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


def test_reset_environment_directly_sets_state_and_returns_true() -> None:
    method, env = _build()
    start_state = env.build_initial_state(light_level=0.3, light_target=0.8)
    assert method.reset_environment(start_state=start_state) is True
    assert env.get_current_state() is start_state


def test_measured_success_rate_is_zero_for_a_never_executed_skill() -> None:
    """Zero rather than the prior mean, so `skip_perfect` can't fire on a skill
    with no evidence at all."""
    method, env = _build()
    assert method.measured_success_rate(ground_skill=_turn_on_light(env=env)) == 0.0


def test_random_choice_is_reproducible_for_a_given_seed() -> None:
    method_a, env_a = _build(seed=3)
    method_b, env_b = _build(seed=3)
    options_a = list(env_a.get_cells())
    skills_a = [
        GroundSkill(skill=LightSwitchSkills.MOVE_ROBOT, objects=(env_a.robot, c, c))
        for c in options_a
    ]
    skills_b = [
        GroundSkill(skill=LightSwitchSkills.MOVE_ROBOT, objects=(env_b.robot, c, c))
        for c in env_b.get_cells()
    ]
    assert method_a.random_choice(ground_skills=skills_a) == method_b.random_choice(
        ground_skills=skills_b
    )


def test_score_is_pure_exploration_bonus_when_no_tasks_have_been_seen() -> None:
    """With nothing to situate against, planning progress is undefined -- the score
    falls back to the UCB bonus alone rather than erroring or ranking arbitrarily."""
    method, env = _build()
    skill = _turn_on_light(env=env)
    method.observe_outcome(ground_skill=skill, success=False)
    score = method.score_ground_skill(ground_skill=skill)
    assert score >= 0.0
    assert math.isfinite(score)


def test_evaluation_policy_emits_a_no_op_once_the_goal_is_already_satisfied() -> None:
    """run_task_episode stops the moment the goal holds, so this path is normally
    unreachable -- but the policy must degrade to a no-op rather than crash if it
    is stepped anyway."""
    method, env = _build()
    initial_state = env.build_initial_state(light_level=0.5, light_target=0.5)
    light_on = LIGHT_ON(state=initial_state, objects=(env.light,))
    task = Task(initial_state=initial_state, goal=Goal(atoms=frozenset({light_on})))
    env.set_state(state=initial_state)

    policy = method.get_task_policy(task=task)
    labeled = policy(env.get_current_state())
    assert labeled.label.startswith("no-op")
    assert labeled.action.tolist() == [0.0, 0.0]


def test_refresh_planning_progress_plans_skips_tasks_it_cannot_plan_for() -> None:
    """An unreachable goal raises PlanningFailure inside the refresh loop; that task
    is dropped rather than taking the whole scoring pass down with it."""
    method, env = _build()
    unreachable = frozenset({
        ADJACENT(
            state=env.build_initial_state(light_level=0.0, light_target=0.5),
            objects=(env.get_cells()[0], env.get_cells()[0]),
        )
    })
    method.record_seen_task(
        init_atoms=method.abstract_state(
            state=env.build_initial_state(light_level=0.0, light_target=0.5)
        ),
        goal=unreachable,
    )
    method.refresh_planning_progress_plans()
    assert method.planning_progress_plans() == []


def test_the_four_unreachable_method_hooks_raise() -> None:
    method, _ = _build()
    with pytest.raises(NotImplementedError):
        method.generate_train_task(tbd_inputs=None)
    with pytest.raises(NotImplementedError):
        method.execute_setup_command(setup_command=None)  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError):
        method.execute_skill(skill=None)  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError):
        method.improve_skill_parameters(skill=None, rollout=None)  # type: ignore[arg-type]


def test_practice_bootstraps_from_a_random_applicable_skill_with_no_candidates_yet() -> None:
    """At the very start of practice the candidate set is empty (no ground skill has
    been executed, so nothing has a competence model). If the assigned goal is
    already satisfied there is no goal-pursuit phase to populate it either, so EES
    must fall back to a uniformly random applicable skill -- that bootstrap is what
    creates the first candidates."""
    method, env = _build()
    initial_state = env.build_initial_state(light_level=0.5, light_target=0.5)
    light_on = LIGHT_ON(state=initial_state, objects=(env.light,))
    task = Task(initial_state=initial_state, goal=Goal(atoms=frozenset({light_on})))
    env.set_state(state=initial_state)

    policy = method.get_practice_policy(task=task)
    state = env.get_current_state()
    for _ in range(4):
        labeled = policy(state)
        assert not labeled.label.startswith("no-op")
        state = env.take_action(action=labeled.action)

    assert method.total_observations() > 0


def test_practice_signals_completion_when_nothing_is_applicable() -> None:
    """With no applicable skill and no reachable candidate, EES ends the
    interaction period rather than burning the remaining budget on no-ops -- which
    is what makes PracticeLoop's transition count data-driven."""
    from hitl_pmp.core.method.method import InteractionComplete

    method, env = _build()
    # Robot parked off-cell: no ground skill's preconditions hold anywhere, so
    # there is nothing to bootstrap from (same construction as
    # test_random_skills_policy's no-applicable-skill case).
    stranded = env.build_initial_state(light_level=0.0, light_target=0.7)
    stranded.set(obj=LightSwitchEnvironment.robot, feature_name="x", feature_val=1.23)
    task = Task(initial_state=stranded, goal=Goal(atoms=frozenset()))
    env.set_state(state=stranded)

    policy = method.get_practice_policy(task=task)
    with pytest.raises(InteractionComplete):
        policy(env.get_current_state())


def test_evaluation_still_degrades_to_a_no_op_rather_than_ending_the_episode() -> None:
    """Evaluation must NOT raise: run_task_episode owns termination there (goal
    check plus horizon), so the policy degrades to a no-op instead."""
    method, env = _build()
    initial_state = env.build_initial_state(light_level=0.5, light_target=0.5)
    light_on = LIGHT_ON(state=initial_state, objects=(env.light,))
    task = Task(initial_state=initial_state, goal=Goal(atoms=frozenset({light_on})))
    env.set_state(state=initial_state)

    labeled = method.get_task_policy(task=task)(env.get_current_state())
    assert labeled.label.startswith("no-op")


def test_random_exploration_attempts_do_not_touch_competence_by_default() -> None:
    """The core suppression this port implements: at the paper's epsilon = 0.5,
    half of all practice attempts are coin flips by construction, so counting them
    would make "competence" measure how often a coin flip works rather than how good
    the skill is when the robot actually tries."""
    method, env = _build()
    skill = _turn_on_light(env=env)
    for _ in range(10):
        method.observe_outcome(ground_skill=skill, success=False, was_random_exploration=True)
    assert method.competence_model(ground_skill=skill).num_observations == 0


def test_double_observe_flag_replicates_predicators_observe_counts() -> None:
    """predicators calls observe() unconditionally (active_sampler_explorer.py:407)
    and then again under `if not exploration_indicator` (:442-443), so a greedy
    attempt lands twice and a random one lands once -- the suppression its own
    comment describes never actually takes effect. The flag exists to measure what
    that bug costs, since the paper's published curve contains it."""
    method, env = _build(seed=1)
    skill = _turn_on_light(env=env)

    method.reproduce_predicators_double_observe = True
    method.observe_outcome(ground_skill=skill, success=True, was_random_exploration=False)
    assert method.competence_model(ground_skill=skill).num_observations == 2

    method.observe_outcome(ground_skill=skill, success=True, was_random_exploration=True)
    assert method.competence_model(ground_skill=skill).num_observations == 3


def test_double_observe_flag_defaults_off_so_the_headline_result_is_the_fixed_one() -> None:
    method, _ = _build()
    assert method.reproduce_predicators_double_observe is False


def test_double_observe_caps_a_mastered_skills_competence_below_one() -> None:
    """Why the bug slows learning: with random attempts counted at half the weight
    of greedy ones, a skill the robot has actually mastered still reads as mediocre
    (its random attempts keep failing), so `skip_perfect` never fires and EES keeps
    spending transitions re-practicing it."""
    buggy, env = _build()
    buggy.reproduce_predicators_double_observe = True
    fixed, _ = _build()
    skill = _turn_on_light(env=env)
    # A mastered skill at epsilon = 0.5: every greedy attempt succeeds, every
    # random one fails (the toggle tolerance covers ~10% of the parameter range).
    for _ in range(20):
        for method in (buggy, fixed):
            method.observe_outcome(ground_skill=skill, success=True, was_random_exploration=False)
            method.observe_outcome(ground_skill=skill, success=False, was_random_exploration=True)

    assert fixed.measured_success_rate(ground_skill=skill) == 1.0
    assert buggy.measured_success_rate(ground_skill=skill) < 0.75


def test_practice_target_history_flag_defaults_off() -> None:
    """The all-attempts practice-target bookkeeping is opt-in: by default the port
    keeps its own random-excluding behavior, so the headline result is unchanged."""
    method, _ = _build()
    assert method.reproduce_predicators_practice_target_history is False


def _feed_mastered_at_epsilon_half(*, method: EesMethod, skill: GroundSkill, reps: int) -> None:
    """A mastered skill under epsilon = 0.5: every greedy attempt succeeds, every
    random one fails (the toggle tolerance covers only a slice of the param range)."""
    for _ in range(reps):
        method.observe_outcome(ground_skill=skill, success=True, was_random_exploration=False)
        method.observe_outcome(ground_skill=skill, success=False, was_random_exploration=True)


def test_flag_on_counts_random_attempts_in_measured_success_rate() -> None:
    """predicators reads `_ground_op_hist`, appended on *every* execution including
    epsilon-random ones. With the flag ON, `measured_success_rate` matches that: a
    mastered skill whose random attempts keep failing does NOT read as perfect."""
    method, env = _build()
    method.reproduce_predicators_practice_target_history = True
    skill = _turn_on_light(env=env)
    _feed_mastered_at_epsilon_half(method=method, skill=skill, reps=20)
    # 20 greedy successes + 20 random failures = 20/40.
    assert method.measured_success_rate(ground_skill=skill) == pytest.approx(0.5)


def test_flag_on_stops_skip_perfect_from_firing_on_a_greedy_only_perfect_skill() -> None:
    """The mechanism the flag targets: OFF, a mastered skill's random failures are
    invisible so its measured rate is 1.0 and `skip_perfect` scores it -inf; ON, the
    random failures count, the rate is below 1.0, and the skill stays a candidate."""
    off, env = _build()
    on, _ = _build()
    on.reproduce_predicators_practice_target_history = True
    skill = _turn_on_light(env=env)
    for method in (off, on):
        _feed_mastered_at_epsilon_half(method=method, skill=skill, reps=20)

    # OFF (current behavior): only greedy successes are visible -> perfect -> skipped.
    assert off.measured_success_rate(ground_skill=skill) == 1.0
    assert off.score_ground_skill(ground_skill=skill) == -math.inf
    # ON (predicators): random failures count -> not perfect -> still scored finitely.
    assert on.score_ground_skill(ground_skill=skill) != -math.inf
    assert math.isfinite(on.score_ground_skill(ground_skill=skill))


def test_flag_on_counts_random_attempts_in_the_ucb_denominator() -> None:
    """The UCB bonus is `c * sqrt(log(total) / num_tries)`. With no seen tasks the
    score is that bonus alone. ON counts the random attempts toward `num_tries`, so
    for a single skill (where total == num_tries) the larger denominator makes the
    ON bonus strictly smaller than the OFF one."""
    off, env = _build()
    on, _ = _build()
    on.reproduce_predicators_practice_target_history = True
    skill = _turn_on_light(env=env)
    # 2 greedy attempts (one each way, so neither arm reads as perfect) + 8 random
    # failures: OFF's num_tries sees 2 attempts, ON's sees 10.
    for method in (off, on):
        method.observe_outcome(ground_skill=skill, success=True, was_random_exploration=False)
        method.observe_outcome(ground_skill=skill, success=False, was_random_exploration=False)
        for _ in range(8):
            method.observe_outcome(ground_skill=skill, success=False, was_random_exploration=True)

    off_score = off.score_ground_skill(ground_skill=skill)
    on_score = on.score_ground_skill(ground_skill=skill)
    assert off_score > 0.0 and on_score > 0.0  # pure UCB bonus, no tasks seen
    assert on_score < off_score


def test_flag_does_not_change_competence_in_either_state() -> None:
    """The two decisions the port accidentally coupled are separated by the flag:
    competence (the planner's edge costs / J_task) must EXCLUDE random attempts
    regardless of the flag. Only the practice-target bookkeeping moves."""
    off, env = _build()
    on, _ = _build()
    on.reproduce_predicators_practice_target_history = True
    skill = _turn_on_light(env=env)
    for method in (off, on):
        _feed_mastered_at_epsilon_half(method=method, skill=skill, reps=20)

    off_model = off.competence_model(ground_skill=skill)
    on_model = on.competence_model(ground_skill=skill)
    # Random attempts excluded from competence in BOTH states: only the 20 greedy
    # successes land, so competence and its observation count are identical.
    assert off_model.num_observations == 20
    assert on_model.num_observations == 20
    assert off_model.get_current_competence() == on_model.get_current_competence()
