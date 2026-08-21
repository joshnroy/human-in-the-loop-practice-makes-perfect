import math

import numpy as np
import pytest

from hitl_pmp.core.method.method import (
    HumanCubeBinResetRequested,
    HumanHelpRequested,
    HumanRandomTaskResetRequested,
    InteractionComplete,
)
from hitl_pmp.core.method.skill_provider import ASK_FOR_RESET_CUBE_BIN_ONLY_NAME
from hitl_pmp.core.method.types import (
    GroundSkill,
    LiftedAtom,
    SamplerConsultation,
    Skill,
    Variable,
)
from hitl_pmp.core.problem.environment.types import Object, State
from hitl_pmp.core.problem.tasks.types import Goal, GroundAtom, Task
from hitl_pmp.environments.ballring.environment import BallRingEnvironment
from hitl_pmp.environments.ballring.predicates import (
    BALL_IN_CUP,
    IS_REACHABLE_BALL,
    IS_REACHABLE_CUP,
    IS_REACHABLE_SURFACE,
)
from hitl_pmp.environments.ballring.skill_provider import BallRingSkillProvider
from hitl_pmp.environments.ballring.tasks import BallRingTasks
from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
from hitl_pmp.environments.lightswitch.predicates import ADJACENT, LIGHT_ON
from hitl_pmp.environments.lightswitch.skill_provider import LightSwitchSkillProvider
from hitl_pmp.environments.lightswitch.skills import LightSwitchSkills
from hitl_pmp.environments.lightswitch.tasks import LightSwitchTasks
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.predicates import HAND_EMPTY, ON_GROUND, REACHABLE
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
from hitl_pmp.environments.tossing3d.skills import Tossing3DSkills
from hitl_pmp.methods.practice_makes_perfect.ees_method import (
    EesMethod,
    _EesEpisode,
    _SkillAttempt,
)
from hitl_pmp.methods.practice_makes_perfect.human_reset_skill import (
    ASK_FOR_RESET_RANDOM_TASK_NAME,
    ASK_FOR_RESET_TASK_INITIAL_NAME,
    HumanResetSkillBuilder,
)
from hitl_pmp.planning.fast_downward import PlanningFailure


def _build(*, grid_size: int = 4, seed: int = 0) -> tuple[EesMethod, LightSwitchEnvironment]:
    env = LightSwitchEnvironment(grid_size=grid_size)
    return EesMethod(env=env, skill_provider=LightSwitchSkillProvider(env=env), seed=seed), env


def _a_state(*, env: LightSwitchEnvironment) -> State:
    """Any concrete state to hand `observe_pending`, which now takes the post-action
    state as well as its abstraction -- see that method. These tests drive the tally,
    not the state, so the initial one is as good as any."""
    return env.hard_reset()


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


class _CountingEesMethod(EesMethod):
    """EesMethod that counts how many skill outcomes it was actually asked to
    score. observe_outcome fires once per observed skill regardless of whether the
    epsilon-greedy branch fired, so this counts observations even where
    total_observations() (competence only, random attempts excluded) would not."""

    observe_outcome_calls: int = 0

    def observe_outcome(
        self, *, ground_skill: GroundSkill, success: bool, was_random_exploration: bool = False
    ) -> None:
        self.observe_outcome_calls += 1
        super().observe_outcome(
            ground_skill=ground_skill,
            success=success,
            was_random_exploration=was_random_exploration,
        )


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
        args=argparse.Namespace(num_test_tasks=10, output_dir=None),
        # 1000, not the 300 this used to run: below ~1000 gradient steps the
        # classifier is not converged, so its argmax moves between refits and the
        # curve oscillates instead of climbing (measured on this exact config:
        # 300 -> [0, .2, 0, 1, 1, .1, .1, .1, .2] but 1000 -> [0, .3, .7, then 1
        # for the rest]). This assertion is about whether EES learns, so it must
        # not be run at a sampler budget where it demonstrably cannot.
        method=EesMethod(
            env=env,
            skill_provider=LightSwitchSkillProvider(env=env),
            seed=0,
            sampler_max_train_iters=1000,
        ),
        problem=problem,
        num_cycles=6,
        max_steps_per_interaction=40,
        renderer=None,
        render_fps=2,
    )
    curve = metrics.task_training_curve()
    assert len(curve) == 7  # initial evaluation + one per cycle
    assert curve[-1][1] > curve[0][1]
    # The evaluation set is fixed for the whole run, so this is a real like-for-like
    # comparison rather than two different draws from the task distribution.
    assert curve[-1][1] == 1.0


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
    method = EesMethod(
        env=env,
        skill_provider=LightSwitchSkillProvider(env=env),
        seed=0,
        exploration_epsilon=1.0,
        sampler_max_train_iters=50,
        # This test isolates competence's random-exclusion, so it needs every
        # parameterized attempt to be random; target-only exploration would make a
        # prefix TurnOnLight greedy and counted, so pin it off here.
        reproduce_predicators_explore_target_only=False,
    )
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


def test_reset_environment_reports_failure_and_leaves_the_environment_alone() -> None:
    """EES has no way to self-navigate, so it reports failure -- and does not reach for
    the privileged `set_state` that would fake one."""
    method, env = _build()
    stranded = env.build_initial_state(light_level=0.1, light_target=0.9)
    env.set_state(state=stranded)
    start_state = env.build_initial_state(light_level=0.3, light_target=0.8)
    assert method.reset_environment(start_state=start_state) is False
    assert env.get_current_state() is stranded


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


# Deliberately NOT read off `Skill.ignore_effects`: this is the *reference* model,
# transcribed from predicators' ground_truth_models/ball_and_cup_sticky_table/nsrts.py
# (lines 84, 285, 457, 514, 526). A simulator that reads the port's own declarations
# would validate a plan against the very model the plan was produced from, and so
# would pass even with the declarations missing entirely.
_REFERENCE_BALLRING_IGNORE_EFFECTS: dict[str, frozenset] = {
    "NavigateToTable": frozenset({IS_REACHABLE_SURFACE, IS_REACHABLE_BALL, IS_REACHABLE_CUP}),
    "NavigateToBall": frozenset({IS_REACHABLE_SURFACE, IS_REACHABLE_BALL, IS_REACHABLE_CUP}),
    "NavigateToCup": frozenset({IS_REACHABLE_SURFACE, IS_REACHABLE_BALL, IS_REACHABLE_CUP}),
    "PickBallFromTable": frozenset({BALL_IN_CUP}),
    "PlaceBallOnFloor": frozenset({BALL_IN_CUP, IS_REACHABLE_BALL}),
}


def _apply_ground_skill(
    *, atoms: frozenset[GroundAtom], ground_skill: GroundSkill
) -> frozenset[GroundAtom]:
    """The symbolic successor, ported from predicators' `utils.apply_operator`.
    Ignore effects are dropped FIRST, so a predicate that is both ignored and added
    (every NavigateTo*) ends up true, not false."""
    ignored = _REFERENCE_BALLRING_IGNORE_EFFECTS.get(ground_skill.skill.name, frozenset())
    survivors = {atom for atom in atoms if atom.predicate not in ignored}
    return frozenset((survivors - ground_skill.delete_effects) | ground_skill.add_effects)


def test_integration_ballring_plans_are_executable_under_the_correct_symbolic_model() -> None:
    """INTEGRATION (shells out to a real Fast Downward). REGRESSION: before
    `ignore_effects` existed, the Ball-Ring operator model was monotone -- navigating
    never revoked reachability -- so FD returned plans like

        NavigateToTable(robot, normal-table-1)
        NavigateToTable(robot, sticky-table-0)   <- navigated away
        PickBallFromTable(robot, ball, cup, normal-table-1)   <- picks where it isn't
        PlaceBallOnTable(robot, ball, cup, sticky-table-0)

    which the real environment cannot execute at step 2. Simulating the returned plan
    under the correct model (ignore effects included) is what catches this: it is
    strictly stronger than pattern-matching the specific bad shape above."""
    env = BallRingEnvironment()
    method = EesMethod(env=env, skill_provider=BallRingSkillProvider(env=env), seed=0)
    task = BallRingTasks(env=env, seed=0).sample_test_task()
    atoms = method.abstract_state(state=task.initial_state)

    plan = method.plan_to(init_atoms=atoms, goal=task.goal.atoms, costs=method.skill_costs())
    assert plan, "expected a non-empty plan for a reachable Ball-Ring goal"
    for step, ground_skill in enumerate(plan):
        unmet = ground_skill.preconditions - atoms
        assert not unmet, (
            f"step {step} ({ground_skill.skill.name}) has unmet preconditions {unmet}; "
            f"plan={[(g.skill.name, [o.name for o in g.objects]) for g in plan]}"
        )
        atoms = _apply_ground_skill(atoms=atoms, ground_skill=ground_skill)
    assert task.goal.atoms <= atoms


def test_integration_ballring_plan_never_picks_from_a_table_it_navigated_away_from() -> None:
    """The specific defect, spelled out: once a NavigateTo* intervenes, an earlier
    NavigateToTable's IsReachableSurface no longer licenses a pick from that table."""
    env = BallRingEnvironment()
    method = EesMethod(env=env, skill_provider=BallRingSkillProvider(env=env), seed=0)
    task = BallRingTasks(env=env, seed=0).sample_test_task()

    plan = method.plan_to(
        init_atoms=method.abstract_state(state=task.initial_state),
        goal=task.goal.atoms,
        costs=method.skill_costs(),
    )
    reachable_table: Object | None = None
    for ground_skill in plan:
        name = ground_skill.skill.name
        if name.startswith("NavigateTo"):
            reachable_table = ground_skill.objects[1] if name == "NavigateToTable" else None
        elif name.endswith("FromTable") or name.endswith("OnTable"):
            assert ground_skill.objects[-1] == reachable_table, (
                f"{name} acts on {ground_skill.objects[-1].name} but the robot is at "
                f"{reachable_table.name if reachable_table else 'no table'}"
            )


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


def test_double_observe_caps_a_mastered_skills_competence_below_one() -> None:
    """Why the bug slows learning: with random attempts counted at half the weight
    of greedy ones, a skill the robot has actually mastered still reads as mediocre
    (its random attempts keep failing), so `skip_perfect` never fires and EES keeps
    spending transitions re-practicing it."""
    buggy, env = _build()
    buggy.reproduce_predicators_double_observe = True
    fixed, _ = _build()
    # Isolate double-observe's effect on competence: read measured_success_rate from
    # the competence history (not the all-attempts one), so pin practice-target-history
    # off on both -- the only difference between them is then double-observe.
    buggy.reproduce_predicators_practice_target_history = False
    fixed.reproduce_predicators_practice_target_history = False
    skill = _turn_on_light(env=env)
    # A mastered skill at epsilon = 0.5: every greedy attempt succeeds, every
    # random one fails (the toggle tolerance covers ~10% of the parameter range).
    for _ in range(20):
        for method in (buggy, fixed):
            method.observe_outcome(ground_skill=skill, success=True, was_random_exploration=False)
            method.observe_outcome(ground_skill=skill, success=False, was_random_exploration=True)

    assert fixed.measured_success_rate(ground_skill=skill) == 1.0
    assert buggy.measured_success_rate(ground_skill=skill) < 0.75


def test_predicators_matching_flag_defaults() -> None:
    """The port defaults toward matching predicators, with two documented exceptions.
    practice_target_history is ON (a clean match). double_observe stays OFF (it is null
    on the success curve but corrupts competence). explore_target_only stays OFF
    because it is coupled to a horizon cap this port lacks -- ON alone starves
    goal-directed learning (see its field comment)."""
    method, _ = _build()
    assert method.reproduce_predicators_practice_target_history is True
    assert method.reproduce_predicators_explore_target_only is False
    assert method.reproduce_predicators_double_observe is False
    # predicators sets its horizon per environment rather than globally, so there is
    # no faithful single default; None keeps this port's original uncapped goal phase.
    assert method.goal_pursuit_horizon is None


def _practice_episode(*, method: EesMethod) -> _EesEpisode:
    """A practice episode whose goal is already trivially satisfied is not usable here
    -- the point is to tick the horizon while the goal phase is still running -- so
    this uses an empty goal set and drives the countdown directly."""
    return _EesEpisode(method=method, goal=frozenset(), practicing=True)


def test_goal_pursuit_horizon_ends_the_goal_phase_once_its_budget_runs_out() -> None:
    """predicators' `assigned_task_horizon` (active_sampler_explorer.py:191-198):
    spend at most this many skills pursuing the assigned train-task goal, then give up
    and practice for the rest of the period. Exhausting it also drops the in-flight
    plan, matching predicators clearing `current_policy` so it replans -- those queued
    skills were chosen to reach a goal we just stopped pursuing."""
    env = LightSwitchEnvironment(grid_size=4)
    method = EesMethod(
        env=env,
        skill_provider=LightSwitchSkillProvider(env=env),
        seed=0,
        goal_pursuit_horizon=2,
    )
    episode = _practice_episode(method=method)
    episode._plan = [_turn_on_light(env=env)]

    for _ in range(2):
        episode._tick_goal_pursuit_horizon()
        assert episode._goal_phase_done is False

    episode._tick_goal_pursuit_horizon()
    assert episode._goal_phase_done is True
    assert episode._plan == []


def test_goal_pursuit_is_uncapped_by_default() -> None:
    """The port's original behavior, kept as the default: pursue the assigned goal
    until it is achieved or planning fails, however many skills that takes."""
    method, _ = _build()
    episode = _practice_episode(method=method)
    for _ in range(50):
        episode._tick_goal_pursuit_horizon()
    assert episode._goal_phase_done is False


def test_goal_pursuit_horizon_does_not_apply_to_evaluation_episodes() -> None:
    """The cap lives in predicators' *explorer*, so it governs practice only.
    run_task_episode owns when an evaluation episode ends; cutting its goal pursuit
    short here would just make it emit no-ops on tasks it could still have solved."""
    env = LightSwitchEnvironment(grid_size=4)
    method = EesMethod(
        env=env,
        skill_provider=LightSwitchSkillProvider(env=env),
        seed=0,
        goal_pursuit_horizon=0,
    )
    episode = _EesEpisode(method=method, goal=frozenset(), practicing=False)
    for _ in range(10):
        episode._tick_goal_pursuit_horizon()
    assert episode._goal_phase_done is False


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
    off.reproduce_predicators_practice_target_history = False
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
    off.reproduce_predicators_practice_target_history = False
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
    off.reproduce_predicators_practice_target_history = False
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


def test_observe_environment_reset_scores_the_pending_skill_against_the_true_outcome() -> None:
    """The state handed in is the one the skill actually produced, so a skill that
    worked is recorded as a success.

    The paired test below shows the alternative -- scoring against the state the
    harness is about to reset *to* -- records the same successful skill as a
    failure. That is the whole reason the hook exists."""
    method, env = _build()
    tasks = LightSwitchTasks(env=env, seed=0)
    task = tasks.sample_train_task()
    env.set_state(state=task.initial_state)
    policy = method.get_practice_policy(task=task)
    executed_state = env.take_action(action=policy(env.get_current_state()).action)

    method.observe_environment_reset(state=executed_state)

    assert method.total_observations() == 1


def test_scoring_a_pending_skill_against_a_reset_state_would_record_a_false_failure() -> None:
    """Pins the failure mode Method.observe_environment_reset prevents, so the hook
    cannot be quietly removed as redundant: judged against a freshly reset
    environment, the very same successful skill comes out a failure, and a
    reset-frequency sweep would inject one such mislabel per reset."""
    method, env = _build()
    tasks = LightSwitchTasks(env=env, seed=0)
    task = tasks.sample_train_task()
    env.set_state(state=task.initial_state)
    policy = method.get_practice_policy(task=task)
    executed_state = env.take_action(action=policy(env.get_current_state()).action)
    executed_skill = method._practice_episode._pending
    assert executed_skill is not None

    method.observe_environment_reset(state=executed_state)
    truthful_rate = method.competence_model(ground_skill=executed_skill).get_current_competence()

    rewound, rewound_env = _build()
    rewound_env.set_state(state=task.initial_state)
    rewound_policy = rewound.get_practice_policy(task=task)
    rewound_env.take_action(action=rewound_policy(rewound_env.get_current_state()).action)
    rewound.observe_environment_reset(state=task.initial_state)
    mislabelled_rate = rewound.competence_model(
        ground_skill=executed_skill
    ).get_current_competence()

    assert mislabelled_rate < truthful_rate


def test_observe_environment_reset_is_a_no_op_before_any_practice_period() -> None:
    method, env = _build()
    tasks = LightSwitchTasks(env=env, seed=0)
    method.observe_environment_reset(state=tasks.sample_train_task().initial_state)
    assert method.total_observations() == 0


def test_the_number_of_observed_outcomes_does_not_depend_on_the_reset_interval() -> None:
    """The invariant that makes a reset-interval sweep interpretable: every arm
    must learn from the same number of skill outcomes, so the arms differ only in
    how often the robot is rescued.

    A period of n steps yields n - 1 observations at every interval -- the last
    skill of a period is still never observed (this port's deviation 2), and every
    mid-period reset settles its in-flight skill instead of dropping or
    mislabelling it. If this count moved with the interval, the manipulation would
    be confounded with how much training data each arm collected."""
    import argparse

    from hitl_pmp.environments.lightswitch.problem import LightSwitchProblem
    from hitl_pmp.method_runner import MethodRunner

    observed: dict[int | None, int] = {}
    for interval in (None, 2, 4, 8):
        env = LightSwitchEnvironment(grid_size=4)
        problem = LightSwitchProblem(env=env, tasks=LightSwitchTasks(env=env, seed=0))
        method = _CountingEesMethod(
            env=env,
            skill_provider=LightSwitchSkillProvider(env=env),
            seed=0,
            sampler_max_train_iters=10,
        )
        metrics = MethodRunner.run(
            args=argparse.Namespace(
                num_test_tasks=1, output_dir=None, practice_reset_interval=interval
            ),
            method=method,
            problem=problem,
            num_cycles=2,
            max_steps_per_interaction=8,
            renderer=None,
            render_fps=2,
        )
        # Guards the comparison itself: an arm that ended a period early would have
        # fewer steps to observe for reasons unrelated to the reset interval.
        assert metrics.evaluations[-1][0] == 16
        observed[interval] = method.observe_outcome_calls
    assert set(observed.values()) == {2 * (8 - 1)}, observed


def test_the_no_plan_no_op_really_leaves_the_environment_alone() -> None:
    """The defect: this branch used to emit `np.zeros(action_space.shape)`, which is
    a real action on nearly every domain here. On Ball-Ring zeros decodes to
    "navigate to (0, 0)", so a method that could not plan silently walked the robot
    across the table and the run reported whatever that caused.

    Ball-Ring rather than Light Switch precisely because Light Switch is the one
    domain where zeros happens to be inert -- it is why this went unnoticed. The goal
    is empty, which is satisfied by every state, so the goal phase ends immediately
    and the episode reaches the no-plan branch without invoking a planner at all."""
    env = BallRingEnvironment()
    env.hard_reset()
    method = EesMethod(env=env, skill_provider=BallRingSkillProvider(env=env), seed=0)
    episode = _EesEpisode(method=method, goal=frozenset(), practicing=False)

    state = env.get_current_state()
    before = {obj.name: tuple(features) for obj, features in state.data.items()}
    labeled = episode.step(state=state)
    assert labeled.label == "no-op (no plan)"

    env.take_action(action=labeled.action)
    after = env.get_current_state()
    assert {obj.name: tuple(features) for obj, features in after.data.items()} == before


def test_a_failed_plan_is_counted_where_nothing_recorded_it_before() -> None:
    """The `??robot` defect cost an hour of silent nonsense: EES caught
    PlanningFailure every single step, degraded to a no-op, and the run exited 0 with
    a full stats.json reporting 0/5. Nothing anywhere recorded that planning had
    failed, so "the method scored zero" could not be told from "the method never
    planned" without re-running it by hand.

    Counted inside plan_to rather than at any catch site, so the three existing
    `except PlanningFailure:` handlers -- and any future one -- cannot diverge."""
    method, env = _build()
    unreachable = frozenset({
        ADJACENT(
            state=env.build_initial_state(light_level=0.0, light_target=0.5),
            objects=(env.get_cells()[0], env.get_cells()[0]),
        )
    })
    assert method.planning_outcomes() == (0, 0)

    with pytest.raises(PlanningFailure):
        method.plan_to(init_atoms=frozenset(), goal=unreachable, costs=method.skill_costs())

    assert method.planning_outcomes() == (1, 1)


def test_a_successful_plan_is_not_counted_as_a_failure() -> None:
    """Guards the opposite error: a counter that ticks on every call would report a
    healthy run as a catastrophic one, which is worse than reporting nothing."""
    method, env = _build()
    tasks = LightSwitchTasks(env=env, seed=0)
    task = tasks.sample_train_task()

    plan = method.plan_to(
        init_atoms=method.abstract_state(state=task.initial_state),
        goal=task.goal.atoms,
        costs=method.skill_costs(),
    )

    assert plan
    assert method.planning_outcomes() == (0, 1)


def test_practice_outcomes_start_empty() -> None:
    """Empty rather than one zeroed entry per lifted skill: "never asked" is exactly
    the state a missing key means, and it is one half of the discrimination these
    counters exist for."""
    method, _env = _build()
    assert method.practice_outcomes() == {}


def test_a_practice_attempt_is_tallied_against_its_lifted_skill() -> None:
    """Keyed by the *lifted* skill name, not the grounding: one learned sampler is
    fitted per skill name (predicators'
    `active_sampler_learning_object_specific_samplers = False`), so "was this
    sampler starved?" is a question about the lifted skill."""
    method, env = _build()
    turn_on = _turn_on_light(env=env)
    episode = _EesEpisode(method=method, goal=frozenset(), practicing=True)
    episode._pending = turn_on

    episode.observe_pending(true_atoms=turn_on.add_effects, state=_a_state(env=env))

    tally = method.practice_outcomes()["TurnOnLight"]
    assert (tally.num_successes, tally.num_attempts) == (1, 1)


def test_an_evaluation_episode_tallies_nothing() -> None:
    """Evaluation runs on held-out test tasks and observes no outcomes at all, so a
    practice counter that ticked there would be counting the test set."""
    method, env = _build()
    turn_on = _turn_on_light(env=env)
    episode = _EesEpisode(method=method, goal=frozenset(), practicing=False)
    episode._pending = turn_on

    episode.observe_pending(true_atoms=turn_on.add_effects, state=_a_state(env=env))

    assert method.practice_outcomes() == {}


def test_a_failed_attempt_counts_toward_attempts_but_not_successes() -> None:
    """`0/17` and `0/0` are the two readings this instrument exists to tell apart."""
    method, env = _build()
    turn_on = _turn_on_light(env=env)
    episode = _EesEpisode(method=method, goal=frozenset(), practicing=True)
    episode._pending = turn_on

    episode.observe_pending(true_atoms=frozenset(), state=_a_state(env=env))

    tally = method.practice_outcomes()["TurnOnLight"]
    assert (tally.num_successes, tally.num_attempts) == (0, 1)


def test_an_epsilon_random_attempt_is_recorded_in_its_own_pool() -> None:
    """A coin flip says nothing about what the sampler learned, so pooling it with
    the greedy draws is what made a previous greedy-versus-random split provisional
    (see `SamplerChoice`). The total still counts it: it really was an attempt."""
    method, env = _build()
    turn_on = _turn_on_light(env=env)
    episode = _EesEpisode(method=method, goal=frozenset(), practicing=True)
    episode._pending = turn_on
    episode._pending_sampler_record = _SkillAttempt(
        skill_name="TurnOnLight",
        param_dim=1,
        params=[0.0],
        sampler_input=[1.0, 0.0],
        was_random_exploration=True,
        was_informed_choice=False,
        consultation=SamplerConsultation.EPSILON_RANDOM,
        records_training_row=True,
    )

    episode.observe_pending(true_atoms=turn_on.add_effects, state=_a_state(env=env))

    tally = method.practice_outcomes()["TurnOnLight"]
    assert (tally.num_successes, tally.num_attempts) == (1, 1)
    assert (tally.num_random_successes, tally.num_random_attempts) == (1, 1)
    assert (tally.num_informed_successes, tally.num_informed_attempts) == (0, 0)


def test_an_informed_attempt_is_recorded_in_its_own_pool() -> None:
    """The discriminating quantity between "the sampler was never really asked" and
    "asked and missed": only an informed draw is one whose classifier actually ranked
    the candidates."""
    method, env = _build()
    turn_on = _turn_on_light(env=env)
    episode = _EesEpisode(method=method, goal=frozenset(), practicing=True)
    episode._pending = turn_on
    episode._pending_sampler_record = _SkillAttempt(
        skill_name="TurnOnLight",
        param_dim=1,
        params=[0.0],
        sampler_input=[1.0, 0.0],
        was_random_exploration=False,
        was_informed_choice=True,
        consultation=SamplerConsultation.INFORMED,
        records_training_row=True,
    )

    episode.observe_pending(true_atoms=frozenset(), state=_a_state(env=env))

    tally = method.practice_outcomes()["TurnOnLight"]
    assert (tally.num_informed_successes, tally.num_informed_attempts) == (0, 1)
    assert (tally.num_random_successes, tally.num_random_attempts) == (0, 0)


def test_a_uniform_fallback_draw_is_neither_random_nor_informed() -> None:
    """`LearnedSkillSampler.sample`'s deviation-6 branch: a fitted classifier whose
    scores could not discriminate. It is an attempt and it is not evidence of
    learning, so it must fall in neither pool -- the remainder is recoverable as
    attempts minus the two."""
    method, env = _build()
    turn_on = _turn_on_light(env=env)
    episode = _EesEpisode(method=method, goal=frozenset(), practicing=True)
    episode._pending = turn_on
    episode._pending_sampler_record = _SkillAttempt(
        skill_name="TurnOnLight",
        param_dim=1,
        params=[0.0],
        sampler_input=[1.0, 0.0],
        was_random_exploration=False,
        was_informed_choice=False,
        consultation=SamplerConsultation.UNINFORMATIVE,
        records_training_row=True,
    )

    episode.observe_pending(true_atoms=turn_on.add_effects, state=_a_state(env=env))

    tally = method.practice_outcomes()["TurnOnLight"]
    assert (tally.num_successes, tally.num_attempts) == (1, 1)
    assert tally.num_random_attempts == 0
    assert tally.num_informed_attempts == 0
    assert tally.num_fallback_attempts() == 1


def test_a_parameter_free_skill_tallies_apart_from_an_uninformative_sampler() -> None:
    """The Tossing3D flaw, reduced to its smallest reproduction.

    `Toss` (`param_dim = 0`, no sampler ever constructed) and `MoveToThrowPose`
    (`param_dim = 1`, a sampler consulted on every execution that could never
    discriminate) both rendered as pure fallback, so `stats.json` could not tell them
    apart and only reading `execute_ground_skill` could. Their remedies are opposite:
    the first means the domain is decomposed wrong and the parameter must move, the
    second means the success predicate is uninformative and must be tightened.

    Light Switch carries the same pair -- `MoveRobot` is `param_dim = 0`, `TurnOnLight`
    is `param_dim = 1` -- so the discrimination is testable without a simulator. Both
    executions here take `LearnedSkillSampler.sample`'s uniform draw or no draw at all;
    the tallies must still differ."""
    method, env = _build()
    tasks = LightSwitchTasks(env=env, seed=0)
    state = tasks.sample_train_task().initial_state
    move = _move_robot_backwards(env=env)
    turn_on = _turn_on_light(env=env)

    # Fit TurnOnLight's sampler on a single class. `MlpBinaryClassifier` takes its
    # single-class shortcut, every candidate scores identically, and `sample` therefore
    # takes deviation 6's uniform fallback on every draw -- never the epsilon branch,
    # which is only reached once the scores discriminate.
    for _ in range(4):
        method.observe_sampler_outcome(
            skill_name="TurnOnLight",
            param_dim=1,
            sampler_input=method.sampler_input_row(
                ground_skill=turn_on, state=state, params=np.zeros(1)
            ),
            success=True,
        )
    method.fit_samplers()

    episode = _EesEpisode(method=method, goal=frozenset(), practicing=True)
    for ground_skill in (move, turn_on):
        _labeled, record = method.execute_ground_skill(
            ground_skill=ground_skill, state=state, explore=True
        )
        episode._pending = ground_skill
        episode._pending_sampler_record = record
        episode.observe_pending(true_atoms=ground_skill.add_effects, state=state)

    unparameterized = method.practice_outcomes()["MoveRobot"]
    uninformative = method.practice_outcomes()["TurnOnLight"]

    # Both are one attempt that no classifier informed, so every pool #111 stored
    # agrees -- which is exactly why the two were indistinguishable.
    assert unparameterized.num_attempts == uninformative.num_attempts == 1
    assert unparameterized.num_fallback_attempts() == uninformative.num_fallback_attempts() == 1
    # ... and this is the discrimination that has to exist on top of that.
    assert unparameterized.num_unparameterized_attempts == 1
    assert unparameterized.num_uninformative_attempts() == 0
    assert uninformative.num_unparameterized_attempts == 0
    assert uninformative.num_uninformative_attempts() == 1


def test_practice_outcomes_accumulate_across_a_whole_run() -> None:
    """Cumulative over the run, like `planning_outcomes`: the Method keeps monotonic
    counters and method_runner.py owns the per-window differencing, so the two can
    never get out of step with the loop's cadence."""
    method, env = _build()
    turn_on = _turn_on_light(env=env)
    episode = _EesEpisode(method=method, goal=frozenset(), practicing=True)

    for hit in (True, False, True):
        episode._pending = turn_on
        episode.observe_pending(
            true_atoms=turn_on.add_effects if hit else frozenset(), state=_a_state(env=env)
        )

    tally = method.practice_outcomes()["TurnOnLight"]
    assert (tally.num_successes, tally.num_attempts) == (2, 3)


# ------------------------------------------- practice-target selection (which skills
# EES actually chooses, and which it silently declines)


def test_practice_target_outcomes_is_empty_before_anything_is_scored() -> None:
    method, _env = _build()
    assert method.practice_target_outcomes() == {}


def test_a_perfect_skill_is_recorded_as_declined_not_merely_absent() -> None:
    """The Tossing3D blind spot in miniature. A grounding at a measured rate of 1.0
    scores -inf under skip_perfect and choose_practice_target drops it, so it never
    appears as a practice target -- indistinguishable, from the outside, from a
    grounding that was never a candidate at all. This is the counter that tells them
    apart."""
    method, env = _build()
    perfect = _turn_on_light(env=env)
    for _ in range(5):
        method.observe_outcome(ground_skill=perfect, success=True)
    assert method.measured_success_rate(ground_skill=perfect) == 1.0

    method.choose_practice_target()

    tally = method.practice_target_outcomes()["TurnOnLight"]
    assert tally.num_declined_perfect == 1
    assert tally.num_scored == 0


def test_an_imperfect_skill_is_recorded_as_scored() -> None:
    method, env = _build()
    imperfect = _turn_on_light(env=env)
    method.observe_outcome(ground_skill=imperfect, success=True)
    method.observe_outcome(ground_skill=imperfect, success=False)

    method.choose_practice_target()

    tally = method.practice_target_outcomes()["TurnOnLight"]
    assert tally.num_scored == 1
    assert tally.num_declined_perfect == 0


def test_declining_is_counted_once_per_grounding_not_once_per_lifted_skill() -> None:
    """score_ground_skill is keyed by GROUND skill while the tally is keyed by the
    lifted name, so two perfect groundings of one skill must contribute 2, not 1."""
    method, env = _build()
    cells = env.get_cells()
    groundings = [
        GroundSkill(skill=LightSwitchSkills.MOVE_ROBOT, objects=(env.robot, cells[0], cells[1])),
        GroundSkill(skill=LightSwitchSkills.MOVE_ROBOT, objects=(env.robot, cells[1], cells[2])),
    ]
    for grounding in groundings:
        method.observe_outcome(ground_skill=grounding, success=True)

    method.choose_practice_target()

    assert method.practice_target_outcomes()["MoveRobot"].num_declined_perfect == 2


def test_skip_perfect_off_scores_a_perfect_skill_instead_of_declining_it() -> None:
    method, env = _build()
    method.skip_perfect = False
    perfect = _turn_on_light(env=env)
    method.observe_outcome(ground_skill=perfect, success=True)

    method.choose_practice_target()

    tally = method.practice_target_outcomes()["TurnOnLight"]
    assert tally.num_declined_perfect == 0
    assert tally.num_scored == 1


class _SilentTargetEesMethod(EesMethod):
    """EES with the practice-target recorder removed, so a test can compare a run that
    records against an otherwise identical one that does not."""

    def record_practice_target(self, *, name: str, field: str) -> None:
        return None


def test_recording_practice_targets_does_not_change_what_ees_does() -> None:
    """The claim the whole record rests on: this is an audit of EES, not a change to
    it. Two methods built from one seed, one with the recorder replaced by a no-op,
    must produce the identical ranked candidate list AND leave their RNGs in the
    identical state -- an extra draw here would silently re-roll every later tiebreak
    and every sampled parameter in the run."""
    recording, env = _build(seed=3)
    silent = _SilentTargetEesMethod(
        env=env, skill_provider=LightSwitchSkillProvider(env=env), seed=3
    )

    cells = env.get_cells()
    groundings = [
        _turn_on_light(env=env),
        GroundSkill(skill=LightSwitchSkills.MOVE_ROBOT, objects=(env.robot, cells[0], cells[1])),
    ]
    for method in (recording, silent):
        _record_one_seen_task(method=method, env=env)
        method.observe_outcome(ground_skill=groundings[0], success=True)
        method.observe_outcome(ground_skill=groundings[0], success=False)
        # Perfect, so skip_perfect drops it -- the branch that records the most.
        method.observe_outcome(ground_skill=groundings[1], success=True)

    assert recording.choose_practice_target() == silent.choose_practice_target()
    assert recording._rng.bit_generator.state == silent._rng.bit_generator.state
    # And the recorder really was doing something in the arm that kept it.
    assert recording.practice_target_outcomes()["MoveRobot"].num_declined_perfect == 1
    assert silent.practice_target_outcomes() == {}


def test_the_committed_practice_target_is_recorded_as_selected() -> None:
    """_practice_plan takes the first candidate whose preconditions it can reach.
    That commitment is what "EES practiced this skill on purpose" means, and it is
    not what the execution tally counts -- an en-route prefix step is executed too."""
    method, env = _build()
    tasks = LightSwitchTasks(env=env, seed=0)
    task = tasks.sample_train_task()
    _record_one_seen_task(method=method, env=env)
    target = _turn_on_light(env=env)
    method.observe_outcome(ground_skill=target, success=False)

    episode = _EesEpisode(method=method, goal=task.goal.atoms, practicing=True)
    true_atoms = method.abstract_state(state=task.initial_state)
    plan = episode._practice_plan(true_atoms=true_atoms)

    assert plan, "expected _practice_plan to commit to some candidate"
    selected = plan[-1].skill.name
    assert method.practice_target_outcomes()[selected].num_selected == 1


def _reset_build(
    *,
    ask_for_reset_task_initial_cost: float | None = None,
    ask_for_reset_random_task_cost: float | None = None,
    grid_size: int = 3,
    seed: int = 0,
) -> tuple[EesMethod, LightSwitchEnvironment]:
    env = LightSwitchEnvironment(grid_size=grid_size)
    method = EesMethod(
        env=env,
        skill_provider=LightSwitchSkillProvider(env=env),
        seed=seed,
        ask_for_reset_task_initial_cost=ask_for_reset_task_initial_cost,
        ask_for_reset_random_task_cost=ask_for_reset_random_task_cost,
    )
    return method, env


def _moved_to_cell2_atoms(*, method: EesMethod, task: Task) -> frozenset[GroundAtom]:
    """This task's own initial state, but with the robot relocated to cell2 -- two
    MoveRobot steps from where init_atoms has it (cell0), on grid_size=3."""
    moved = task.initial_state.model_copy(deep=True)
    moved.set(obj=LightSwitchEnvironment.robot, feature_name="x", feature_val=2.5)
    return method.abstract_state(state=moved)


def test_ees_declares_it_cannot_ask_for_help_unless_a_reset_skill_is_configured() -> None:
    """The predicate practice_loop.py validates against, so it has to be False for the
    incumbent construction -- otherwise every existing EES run would start demanding a
    HumanOracle it never had. True if EITHER reset skill is configured."""
    method, _env = _build()
    assert method.may_request_human_help() is False
    task_initial_only, _env = _reset_build(ask_for_reset_task_initial_cost=0.1)
    assert task_initial_only.may_request_human_help() is True
    random_only, _env = _reset_build(ask_for_reset_random_task_cost=0.1)
    assert random_only.may_request_human_help() is True


def test_an_unconfigured_ees_practice_policy_is_completely_unwrapped() -> None:
    """Not merely "a wrapper that never fires": there is no wrapper at all. get_practice_
    policy returns the episode's own step method directly, structurally identical to
    get_task_policy -- the structural claim behind an unconfigured run staying
    byte-identical to before either reset skill existed."""
    method, env = _build()
    task = LightSwitchTasks(env=env, seed=0).sample_train_task()
    policy = method.get_practice_policy(task=task)
    assert policy.__qualname__.startswith("EesMethod.get_practice_policy")


def test_plan_to_prefers_a_cheap_reset_over_an_expensive_walk_back() -> None:
    """The core claim: EES's own optimal planner, not a harness heuristic, decides to
    ask for help when it is the cheaper way to reach a symbolic target. Goal here is
    literally task_initial_atoms (satisfiable exactly two ways: walk two MoveRobot
    steps back, at ~2x the default per-skill cost, or the one configured-cheap reset
    step) -- seq-opt-lmcut is optimal, so a strictly cheaper reset must win."""
    method, env = _reset_build(ask_for_reset_task_initial_cost=0.01)
    task = LightSwitchTasks(env=env, seed=0).sample_train_task()
    task_initial_atoms = method.abstract_state(state=task.initial_state)
    current_atoms = _moved_to_cell2_atoms(method=method, task=task)

    plan = method.plan_to(
        init_atoms=current_atoms,
        goal=task_initial_atoms,
        costs=method.skill_costs(),
        task_initial_atoms=task_initial_atoms,
    )
    assert [ground.skill.name for ground in plan] == [ASK_FOR_RESET_TASK_INITIAL_NAME]


def test_plan_to_does_not_offer_the_reset_skill_without_task_initial_atoms() -> None:
    """Configured cost alone is not enough -- plan_to also needs task_initial_atoms,
    which only a practice-context caller supplies (see _EesEpisode). Without it, the
    only route back is the ordinary (more expensive) walk."""
    method, env = _reset_build(ask_for_reset_task_initial_cost=0.01)
    task = LightSwitchTasks(env=env, seed=0).sample_train_task()
    task_initial_atoms = method.abstract_state(state=task.initial_state)
    current_atoms = _moved_to_cell2_atoms(method=method, task=task)

    plan = method.plan_to(
        init_atoms=current_atoms, goal=task_initial_atoms, costs=method.skill_costs()
    )
    assert ASK_FOR_RESET_TASK_INITIAL_NAME not in [ground.skill.name for ground in plan]
    assert plan, "the ordinary walk-back route must still exist"


def test_plan_to_never_offers_the_reset_skill_when_its_cost_is_unconfigured() -> None:
    """`None` (the default) means the skill is not offered to the planner at all, even
    when a caller supplies task_initial_atoms -- the other half of the two-gate check
    plan_to's own docstring describes."""
    method, env = _reset_build()
    task = LightSwitchTasks(env=env, seed=0).sample_train_task()
    task_initial_atoms = method.abstract_state(state=task.initial_state)
    current_atoms = _moved_to_cell2_atoms(method=method, task=task)

    plan = method.plan_to(
        init_atoms=current_atoms,
        goal=task_initial_atoms,
        costs=method.skill_costs(),
        task_initial_atoms=task_initial_atoms,
    )
    assert ASK_FOR_RESET_TASK_INITIAL_NAME not in [ground.skill.name for ground in plan]
    assert plan


def test_the_reset_skill_resets_to_the_tasks_own_initial_atoms_not_wherever_planning_started() -> (
    None
):
    """The bug this guards against: task_initial_atoms must be the PERIOD's fixed
    initial state, never confused with `init_atoms` -- wherever a particular plan_to
    call happens to be planning FROM, which moves every step. Conflating the two would
    build an operator whose effect is "reset to wherever you already are"."""
    method, env = _reset_build(ask_for_reset_task_initial_cost=0.01)
    task = LightSwitchTasks(env=env, seed=0).sample_train_task()
    task_initial_atoms = method.abstract_state(state=task.initial_state)
    current_atoms = _moved_to_cell2_atoms(method=method, task=task)
    assert current_atoms != task_initial_atoms, "the fixture must actually move the robot"

    plan = method.plan_to(
        init_atoms=current_atoms,
        goal=task_initial_atoms,
        costs=method.skill_costs(),
        task_initial_atoms=task_initial_atoms,
    )
    reset_ground_skill = plan[0]
    assert reset_ground_skill.add_effects == task_initial_atoms


def test_step_dispatches_a_selected_reset_skill_as_a_human_help_request() -> None:
    """Once the plan's next step is ask_for_reset_task_initial, step() must intercept
    it before execute_ground_skill: no controller call, no competence model created for
    it (that is what "bypasses the competence model entirely" means structurally), and
    the banked cost is exactly the configured flag, not anything read off `costs`."""
    method, env = _reset_build(ask_for_reset_task_initial_cost=0.37)
    task = LightSwitchTasks(env=env, seed=0).sample_train_task()
    task_initial_atoms = method.abstract_state(state=task.initial_state)
    reset_ground_skill = HumanResetSkillBuilder.build_ask_for_reset_task_initial(
        objects=method.objects(), predicates=method.predicates(), init_atoms=task_initial_atoms
    )
    episode = _EesEpisode(
        method=method,
        goal=task.goal.atoms,
        practicing=True,
        task_initial_atoms=task_initial_atoms,
    )
    episode._plan = [reset_ground_skill]

    with pytest.raises(HumanHelpRequested) as excinfo:
        episode.step(state=task.initial_state)
    assert excinfo.value.cost == 0.37
    assert method._competence_models == {}


def test_ees_evaluation_never_offers_or_selects_either_reset_skill() -> None:
    """No evaluation episode is ever rescued -- measurement would be measuring the
    human. get_task_policy supplies no task_initial_atoms at all, so plan_to can never
    offer the reset skill there regardless of either cost flag."""
    method, env = _reset_build(
        ask_for_reset_task_initial_cost=0.01, ask_for_reset_random_task_cost=0.01
    )
    task = LightSwitchTasks(env=env, seed=0).sample_train_task()
    policy = method.get_task_policy(task=task)
    state = task.initial_state
    for _ in range(5):
        policy(state)


# --------------------------------------------------------------------------------
# ask_for_reset_random_task as a real mid-plan ground skill, not just a stuck fallback.
#
# Mirrors the ask_for_reset_task_initial tests above wherever the two skills' shapes
# actually agree: both are built from THIS episode's own task_initial_atoms (never a
# freshly sampled task's atoms, which are not known at plan time), so a plan_to call
# offering both is exercising two ground skills with identical add/delete effects and
# different names/costs. See HumanResetSkillBuilder's own docstring for why that
# construction is sound for Tossing3D specifically rather than in general.


def test_plan_to_prefers_a_cheap_random_task_reset_over_an_expensive_walk_back() -> None:
    """The mid-plan counterpart of test_plan_to_prefers_a_cheap_reset_over_an_expensive_
    walk_back for ask_for_reset_task_initial: FastDownward must be free to select
    ask_for_reset_random_task inside an ordinary plan_to call, not only as the
    step()-level substitute for InteractionComplete."""
    method, env = _reset_build(ask_for_reset_random_task_cost=0.01)
    task = LightSwitchTasks(env=env, seed=0).sample_train_task()
    task_initial_atoms = method.abstract_state(state=task.initial_state)
    current_atoms = _moved_to_cell2_atoms(method=method, task=task)

    plan = method.plan_to(
        init_atoms=current_atoms,
        goal=task_initial_atoms,
        costs=method.skill_costs(),
        task_initial_atoms=task_initial_atoms,
    )
    assert [ground.skill.name for ground in plan] == [ASK_FOR_RESET_RANDOM_TASK_NAME]


def test_plan_to_does_not_offer_the_random_task_reset_skill_without_task_initial_atoms() -> None:
    method, env = _reset_build(ask_for_reset_random_task_cost=0.01)
    task = LightSwitchTasks(env=env, seed=0).sample_train_task()
    task_initial_atoms = method.abstract_state(state=task.initial_state)
    current_atoms = _moved_to_cell2_atoms(method=method, task=task)

    plan = method.plan_to(
        init_atoms=current_atoms, goal=task_initial_atoms, costs=method.skill_costs()
    )
    assert ASK_FOR_RESET_RANDOM_TASK_NAME not in [ground.skill.name for ground in plan]
    assert plan, "the ordinary walk-back route must still exist"


def test_plan_to_never_offers_the_random_task_reset_skill_when_its_cost_is_unconfigured() -> None:
    method, env = _reset_build()
    task = LightSwitchTasks(env=env, seed=0).sample_train_task()
    task_initial_atoms = method.abstract_state(state=task.initial_state)
    current_atoms = _moved_to_cell2_atoms(method=method, task=task)

    plan = method.plan_to(
        init_atoms=current_atoms,
        goal=task_initial_atoms,
        costs=method.skill_costs(),
        task_initial_atoms=task_initial_atoms,
    )
    assert ASK_FOR_RESET_RANDOM_TASK_NAME not in [ground.skill.name for ground in plan]
    assert plan


def test_plan_to_offers_and_independently_costs_both_reset_skills_when_both_configured() -> None:
    """Regression test for a composition bug: injecting the second reset skill's cost
    into `ground_skill_costs` by rebuilding from the bare `costs` argument (rather than
    from `ground_skill_costs`, which already carries the first skill's injected entry)
    would silently drop the first skill's cost from the dict it is offered under, even
    though its Skill object is still present in `skills` -- so whichever of the two is
    cheaper must be selected correctly in BOTH orderings, not just one."""
    method, env = _reset_build(
        ask_for_reset_task_initial_cost=0.02, ask_for_reset_random_task_cost=0.01
    )
    task = LightSwitchTasks(env=env, seed=0).sample_train_task()
    task_initial_atoms = method.abstract_state(state=task.initial_state)
    current_atoms = _moved_to_cell2_atoms(method=method, task=task)
    plan = method.plan_to(
        init_atoms=current_atoms,
        goal=task_initial_atoms,
        costs=method.skill_costs(),
        task_initial_atoms=task_initial_atoms,
    )
    assert [ground.skill.name for ground in plan] == [ASK_FOR_RESET_RANDOM_TASK_NAME]

    method2, env2 = _reset_build(
        ask_for_reset_task_initial_cost=0.01, ask_for_reset_random_task_cost=0.02
    )
    task2 = LightSwitchTasks(env=env2, seed=0).sample_train_task()
    task_initial_atoms2 = method2.abstract_state(state=task2.initial_state)
    current_atoms2 = _moved_to_cell2_atoms(method=method2, task=task2)
    plan2 = method2.plan_to(
        init_atoms=current_atoms2,
        goal=task_initial_atoms2,
        costs=method2.skill_costs(),
        task_initial_atoms=task_initial_atoms2,
    )
    assert [ground.skill.name for ground in plan2] == [ASK_FOR_RESET_TASK_INITIAL_NAME]


def test_step_dispatches_random_task_reset_as_a_human_random_task_reset_request() -> None:
    """The mid-plan counterpart of test_step_dispatches_a_selected_reset_skill_as_a_
    human_help_request: once the plan's next step is ask_for_reset_random_task, step()
    must intercept it and raise HumanRandomTaskResetRequested (not HumanHelpRequested)
    before execute_ground_skill, banking exactly the configured flag."""
    method, env = _reset_build(ask_for_reset_random_task_cost=0.19)
    task = LightSwitchTasks(env=env, seed=0).sample_train_task()
    task_initial_atoms = method.abstract_state(state=task.initial_state)
    reset_ground_skill = HumanResetSkillBuilder.build_ask_for_reset_random_task(
        objects=method.objects(), predicates=method.predicates(), init_atoms=task_initial_atoms
    )
    episode = _EesEpisode(
        method=method,
        goal=task.goal.atoms,
        practicing=True,
        task_initial_atoms=task_initial_atoms,
    )
    episode._plan = [reset_ground_skill]

    with pytest.raises(HumanRandomTaskResetRequested) as excinfo:
        episode.step(state=task.initial_state)
    assert excinfo.value.cost == 0.19
    assert method._competence_models == {}


def test_nothing_left_to_practice_raises_the_free_interaction_complete_even_when_configured() -> (
    None
):
    """The step()-level bootstrap fallback that used to substitute a priced
    HumanRandomTaskResetRequested here is REMOVED (Josh's correction): it had no
    analogue on ask_for_reset_task_initial and, unlike the mid-plan ground skill, fired
    regardless of the configured cost's VALUE -- a bare `is not None` check, not a real
    comparison. `_practice_plan`'s bootstrap path (no candidate has ever been scored,
    so `choose_practice_target` returns nothing to iterate) never calls `plan_to` at
    all, so the mid-plan ground skill was never offered a chance to be selected here
    either -- meaning this path had NO cost-sensitive route to a reset, only the
    unconditional one. A genuinely-never-scored, nothing-bootstrap-applicable state must
    now degrade to the free InteractionComplete, exactly like the `never` baseline with
    no reset skill configured at all (test_practice_signals_completion_when_nothing_is_
    applicable) -- configuring ask_for_reset_random_task_cost no longer changes this
    specific outcome."""
    method, env = _reset_build(ask_for_reset_random_task_cost=0.42)
    stranded = env.build_initial_state(light_level=0.0, light_target=0.7)
    stranded.set(obj=LightSwitchEnvironment.robot, feature_name="x", feature_val=1.23)
    task = Task(initial_state=stranded, goal=Goal(atoms=frozenset()))
    env.set_state(state=stranded)
    assert method._competence_models == {}, "the bootstrap path requires no scored candidate yet"

    policy = method.get_practice_policy(task=task)
    with pytest.raises(InteractionComplete):
        policy(env.get_current_state())


# --------------------------------------------------------------------------------
# The post-hoc acceptance ceiling: "no plan" is not a competing finite-cost
# alternative FastDownward's own search can weigh against ANY finite reset cost, so a
# reset ground skill -- once offered -- is unconditionally accepted at any magnitude
# by a pure cost-minimizing search. Confirmed empirically (not just asserted) against
# costs 0.001 through 1e6 during this investigation, against Tossing3D's real one-way
# door: MOVE_TO_TOSS_LOCATION_AND_TOSS deletes Reachable(cube, barrier)
# unconditionally (hit or miss) and no skill ever re-adds it, so a state with
# Reachable=False is a genuine dead end for both PickCube and the toss skill -- no
# live simulator needed, since objects/predicates/skills are static declarations.
#
# plan_to must decline a reset whose cost exceeds a real, non-arbitrary ceiling
# derived from costs already in scope (the per-ground-skill -log(competence) costs
# this same call already receives) rather than accepting it unconditionally.


def _tossing3d_dead_end() -> tuple[EesMethod, Tossing3DEnvironment, GroundAtom, GroundAtom]:
    """A Tossing3D EesMethod plus the one-way-door dead-end atoms: `stuck_atoms` (past
    the barrier, Reachable=False, unreachable to any real skill) and
    `task_initial_atoms` (the goal these tests plan toward, and what a reset would
    produce -- Reachable=True, matching this domain's actual task-initial shape,
    verified empirically to be scene-seed-invariant -- see HumanResetSkillBuilder's
    own docstring)."""
    env = Tossing3DEnvironment()
    provider = Tossing3DSkillProvider(env=env)
    method = EesMethod(env=env, skill_provider=provider, seed=0)
    cube, barrier, robot = env.cube, env.barrier, env.robot
    task_initial_atoms = frozenset({
        HAND_EMPTY(state=None, objects=(robot,)),
        ON_GROUND(state=None, objects=(cube,)),
        REACHABLE(state=None, objects=(cube, barrier)),
    })
    stuck_atoms = frozenset({
        HAND_EMPTY(state=None, objects=(robot,)),
        ON_GROUND(state=None, objects=(cube,)),
    })
    return method, env, stuck_atoms, task_initial_atoms


def test_plan_to_confirms_the_dead_end_is_genuinely_unreachable_without_any_reset() -> None:
    """Fixture sanity: without a reset offered at all, this state has no walk back --
    the premise every test below depends on."""
    method, _env, stuck_atoms, task_initial_atoms = _tossing3d_dead_end()
    with pytest.raises(PlanningFailure):
        method.plan_to(init_atoms=stuck_atoms, goal=task_initial_atoms, costs=method.skill_costs())


def test_plan_to_declines_a_reset_whose_cost_exceeds_the_ceiling_and_stays_stuck() -> None:
    """With no ground skill ever scored (costs == {}), the ceiling falls back to
    default_cost() (~0.095 at the prior's own defaults) -- a reset costed at 1000.0
    clears nothing, and since the dead end has no cheaper real alternative either
    (test above), plan_to must raise PlanningFailure rather than returning a plan that
    pays for it."""
    method, _env, stuck_atoms, task_initial_atoms = _tossing3d_dead_end()
    method.ask_for_reset_random_task_cost = 1000.0
    assert method.skill_costs() == {}, "a fresh Method has no scored competence models yet"

    with pytest.raises(PlanningFailure):
        method.plan_to(
            init_atoms=stuck_atoms,
            goal=task_initial_atoms,
            costs=method.skill_costs(),
            task_initial_atoms=task_initial_atoms,
        )


def test_plan_to_accepts_a_reset_whose_cost_clears_the_ceiling() -> None:
    """The converse: a reset costed BELOW default_cost() (the ceiling when nothing has
    been scored yet) is still accepted -- the ceiling is a real bar, not a blanket
    decline."""
    method, _env, stuck_atoms, task_initial_atoms = _tossing3d_dead_end()
    method.ask_for_reset_random_task_cost = 0.01
    assert method.default_cost() > 0.01, "the fixture must actually clear the ceiling"

    plan = method.plan_to(
        init_atoms=stuck_atoms,
        goal=task_initial_atoms,
        costs=method.skill_costs(),
        task_initial_atoms=task_initial_atoms,
    )
    assert [g.skill.name for g in plan] == [ASK_FOR_RESET_RANDOM_TASK_NAME]


def test_plan_to_falls_back_to_a_pricier_real_plan_rather_than_an_unaffordable_reset() -> None:
    """When a real (reset-free) alternative exists at all, declining an over-ceiling
    reset must fall back to it rather than giving up outright -- "give up" only wins
    over a genuine real option when no real option exists, never merely because a
    reset was rejected. Reuses LightSwitch's own cheap-reset-vs-walk-back fixture
    (test_plan_to_prefers_a_cheap_random_task_reset_over_an_expensive_walk_back), but
    inverted: the reset is now priced ABOVE the ceiling (5.0, versus a fresh Method's
    default_cost() of ~0.095) while the ordinary 2-step MoveRobot walk back to cell0
    still exists -- plan_to must return that real, pricier walk, not PlanningFailure
    and not the unaffordable reset."""
    method, env = _reset_build(ask_for_reset_random_task_cost=5.0)
    task = LightSwitchTasks(env=env, seed=0).sample_train_task()
    task_initial_atoms = method.abstract_state(state=task.initial_state)
    current_atoms = _moved_to_cell2_atoms(method=method, task=task)
    assert method.skill_costs() == {}
    assert method.default_cost() < 5.0, "the fixture must actually exceed the ceiling"

    plan = method.plan_to(
        init_atoms=current_atoms,
        goal=task_initial_atoms,
        costs=method.skill_costs(),
        task_initial_atoms=task_initial_atoms,
    )
    assert ASK_FOR_RESET_RANDOM_TASK_NAME not in [g.skill.name for g in plan]
    assert plan, "a real (pricier) walk-back must still be returned"


def test_the_ceiling_rises_with_a_skills_own_observed_unreliability() -> None:
    """The ceiling is derived from costs already in scope (the same per-ground-skill
    -log(competence) dict this plan_to call already receives), not an invented
    constant -- so it moves as competence estimates evolve, exactly the "same order of
    magnitude as the reset costs Josh is sweeping" argument. Driving PickCube's own
    competence down (repeated observed failures) raises its -log(competence) past both
    swept values (0.5, 1.0), at which point a reset costed at either is accepted where
    it was declined before any failures were observed."""
    method, env, stuck_atoms, task_initial_atoms = _tossing3d_dead_end()
    pick_cube = GroundSkill(
        skill=Tossing3DSkills.PICK_CUBE, objects=(env.robot, env.cube, env.barrier)
    )
    for _ in range(30):
        method.observe_outcome(ground_skill=pick_cube, success=False)
    pick_cube_cost = method.skill_costs()[pick_cube]
    assert pick_cube_cost > 1.0, "the fixture must actually drive the ceiling past both swept costs"

    for swept_cost in (0.5, 1.0):
        method.ask_for_reset_random_task_cost = swept_cost
        plan = method.plan_to(
            init_atoms=stuck_atoms,
            goal=task_initial_atoms,
            costs=method.skill_costs(),
            task_initial_atoms=task_initial_atoms,
        )
        assert [g.skill.name for g in plan] == [ASK_FOR_RESET_RANDOM_TASK_NAME]


def test_a_never_policy_run_can_stay_stuck_at_both_swept_costs_when_nothing_is_scored() -> None:
    """The empirical check Josh asked for, at the two costs actually swept (0.5, 1.0):
    from the SAME one-way-door dead end, with nothing yet scored (a fresh Method,
    matching the very start of a `never`-policy run before any competence has been
    observed), the robot ends up stuck-and-unrescued at both swept costs -- not
    deterministically reset -- because the ceiling (default_cost() at this point,
    ~0.095) is well below either. This is the "reports a real finding" half of the
    ask: at this specific fresh-competence point, BOTH swept costs decline, they do
    not split; test_the_ceiling_rises_with_a_skills_own_observed_unreliability above
    is what shows the SAME two costs later being ACCEPTED once competence evolves, so
    across a run's full lifetime the mix does happen, just not from a static snapshot
    at the very first step alone."""
    for swept_cost in (0.5, 1.0):
        method, _env, stuck_atoms, task_initial_atoms = _tossing3d_dead_end()
        method.ask_for_reset_random_task_cost = swept_cost
        with pytest.raises(PlanningFailure):
            method.plan_to(
                init_atoms=stuck_atoms,
                goal=task_initial_atoms,
                costs=method.skill_costs(),
                task_initial_atoms=task_initial_atoms,
            )


class _CubeBinCapableSkillProvider(LightSwitchSkillProvider):
    """A LightSwitch-backed stand-in for a domain whose SkillProvider offers
    `ask_for_reset_cube_bin_only` -- exercises EesMethod's own wiring (plan_to's
    injection, _EesEpisode.step's interception) without importing Tossing3D or
    touching a simulator, the same reason `_AskingMethod` in test_practice_loop.py
    avoids importing a real Method.

    The returned operator adds `Adjacent(cell0, cell0)` -- a reflexive atom no real
    LightSwitch skill's effects ever touch (`Adjacent` is only ever a precondition in
    skills.py, never an add/delete effect), so a goal of exactly that atom is
    achievable if and only if this ground skill was actually offered to the planner.
    That makes plan_to's own injection directly observable through a real Fast
    Downward plan, rather than only through step()'s interception (covered
    separately below)."""

    def human_cube_bin_reset_skill(self) -> GroundSkill:
        cell_var = Variable(name="cell", type=LightSwitchEnvironment.cell_type)
        skill = Skill(
            name=ASK_FOR_RESET_CUBE_BIN_ONLY_NAME,
            parameters=(cell_var,),
            preconditions=frozenset(),
            add_effects=frozenset({LiftedAtom(predicate=ADJACENT, variables=(cell_var, cell_var))}),
            delete_effects=frozenset(),
            param_dim=0,
        )
        env_cell0 = self.env.get_cells()[0]
        return GroundSkill(skill=skill, objects=(env_cell0,))


def _cube_bin_reset_build(
    *, ask_for_reset_cube_bin_cost: float | None = None, grid_size: int = 3, seed: int = 0
) -> tuple[EesMethod, LightSwitchEnvironment]:
    env = LightSwitchEnvironment(grid_size=grid_size)
    method = EesMethod(
        env=env,
        skill_provider=_CubeBinCapableSkillProvider(env=env),
        seed=seed,
        ask_for_reset_cube_bin_cost=ask_for_reset_cube_bin_cost,
    )
    return method, env


def test_ees_declares_it_can_ask_for_help_when_the_cube_bin_reset_is_configured() -> None:
    method, _env = _cube_bin_reset_build(ask_for_reset_cube_bin_cost=0.2)
    assert method.may_request_human_help() is True


def test_plan_to_offers_the_cube_bin_reset_skill_when_configured_and_the_domain_supports_it() -> (
    None
):
    """A real Fast Downward plan, not just a constructed operator: the goal
    (`Adjacent(cell0, cell0)`, reflexive and false in every real init state) is
    achievable if and only if plan_to actually added the injected ground skill to
    the planner's candidate set -- see _CubeBinCapableSkillProvider's own docstring
    for why no ordinary LightSwitch skill could ever produce it.

    0.01, not 0.1: this must clear plan_to's own shared cost ceiling (default_cost()
    when nothing has been observed yet, ~0.095) to test injection rather than
    affordability -- see the dedicated affordability tests below for that."""
    method, env = _cube_bin_reset_build(ask_for_reset_cube_bin_cost=0.01)
    task = LightSwitchTasks(env=env, seed=0).sample_train_task()
    init_atoms = method.abstract_state(state=task.initial_state)
    cell0 = env.get_cells()[0]
    goal = frozenset({GroundAtom(predicate=ADJACENT, objects=(cell0, cell0))})
    assert not goal <= init_atoms, "the fixture must actually start with the goal false"

    plan = method.plan_to(init_atoms=init_atoms, goal=goal, costs=method.skill_costs())
    assert [ground.skill.name for ground in plan] == [ASK_FOR_RESET_CUBE_BIN_ONLY_NAME]


def test_an_unaffordable_cube_bin_reset_is_rejected_even_as_the_only_route_to_the_goal() -> None:
    """The bug this guards against, reproduced and closed: classical cost-optimal
    planning has no price for "give up", so PlanningFailure is not a competing
    finite-cost alternative -- offering ask_for_reset_cube_bin_only unconditionally
    means ANY configured cost is preferred over reporting no plan, however large.
    Verified against a probe sweeping 0.001 through 1,000,000 before this fix
    existed: every one of them produced a plan through the reset. This is the same
    goal as test_plan_to_offers_the_cube_bin_reset_skill_..., the ONLY route to
    which is the reset -- but priced far above any of this run's own competence-
    derived costs, so plan_to's shared ceiling (see plan_to's own docstring --
    ask_for_reset_cube_bin_only goes through the exact same reset_costs_by_name
    mechanism as the other two reset skills, not a copy of its own) declines it and
    re-requests with no reset offered at all. The goal is genuinely unreachable
    without it, so PlanningFailure propagates from Fast Downward itself -- there is
    no cube-bin-specific message to match anymore, since there is no cube-bin-
    specific code path left to produce one."""
    method, env = _cube_bin_reset_build(ask_for_reset_cube_bin_cost=1_000_000.0)
    task = LightSwitchTasks(env=env, seed=0).sample_train_task()
    init_atoms = method.abstract_state(state=task.initial_state)
    cell0 = env.get_cells()[0]
    goal = frozenset({GroundAtom(predicate=ADJACENT, objects=(cell0, cell0))})

    with pytest.raises(PlanningFailure):
        method.plan_to(init_atoms=init_atoms, goal=goal, costs=method.skill_costs())


def test_an_affordable_cube_bin_reset_still_clears_the_threshold() -> None:
    """The companion to the test above: the fix must not make the skill unusable --
    a genuinely cheap reset (well under default_cost(), the fallback threshold
    before anything has been observed) still gets used when it is the only route to
    the goal."""
    method, env = _cube_bin_reset_build(ask_for_reset_cube_bin_cost=0.001)
    assert method.default_cost() > 0.001, "the fixture must actually be cheap"
    task = LightSwitchTasks(env=env, seed=0).sample_train_task()
    init_atoms = method.abstract_state(state=task.initial_state)
    cell0 = env.get_cells()[0]
    goal = frozenset({GroundAtom(predicate=ADJACENT, objects=(cell0, cell0))})

    plan = method.plan_to(init_atoms=init_atoms, goal=goal, costs=method.skill_costs())
    assert [ground.skill.name for ground in plan] == [ASK_FOR_RESET_CUBE_BIN_ONLY_NAME]


def test_the_affordability_threshold_is_derived_from_this_runs_own_costs_not_a_fixed_constant() -> (
    None
):
    """Self-calibrating, not a second flag to tune: the same configured reset cost
    is rejected when nothing has depressed any skill's competence yet, and accepted
    once some OTHER ground skill's own competence-derived cost has risen above it --
    "the same order of magnitude as EES's own competence-derived skill costs",
    read directly off method.skill_costs() rather than a constant invented here."""
    reset_cost = 0.3
    method, env = _cube_bin_reset_build(ask_for_reset_cube_bin_cost=reset_cost)
    task = LightSwitchTasks(env=env, seed=0).sample_train_task()
    init_atoms = method.abstract_state(state=task.initial_state)
    cell0 = env.get_cells()[0]
    goal = frozenset({GroundAtom(predicate=ADJACENT, objects=(cell0, cell0))})
    assert reset_cost > method.default_cost(), "the fixture must start unaffordable"

    with pytest.raises(PlanningFailure):
        method.plan_to(init_atoms=init_atoms, goal=goal, costs=method.skill_costs())

    # Depress TURN_ON_LIGHT's competence (repeated failures) until its own
    # competence-derived cost clears reset_cost -- nothing about this reset itself
    # changed, only what this run now believes about a DIFFERENT skill.
    turn_on = _turn_on_light(env=env)
    for _ in range(20):
        method.observe_outcome(ground_skill=turn_on, success=False)
    costs_now = method.skill_costs()
    assert max(costs_now.values()) > reset_cost, "the fixture must depress competence enough"

    plan = method.plan_to(init_atoms=init_atoms, goal=goal, costs=costs_now)
    assert [ground.skill.name for ground in plan] == [ASK_FOR_RESET_CUBE_BIN_ONLY_NAME]


def test_plan_to_raises_when_configured_against_a_domain_with_no_cube_bin_reset_skill() -> None:
    """LightSwitchSkillProvider (used directly, not the fake above) inherits the base
    SkillProvider's None default -- exactly every domain but Tossing3D today. Setting
    the cost flag against it is a misconfiguration plan_to must report, not silently
    ignore."""
    method, env = _reset_build(seed=0)
    method.ask_for_reset_cube_bin_cost = 0.1
    task = LightSwitchTasks(env=env, seed=0).sample_train_task()
    init_atoms = method.abstract_state(state=task.initial_state)
    with pytest.raises(ValueError, match="human_cube_bin_reset_skill"):
        method.plan_to(init_atoms=init_atoms, goal=frozenset(), costs=method.skill_costs())


def test_step_dispatches_a_selected_cube_bin_reset_skill_as_a_human_cube_bin_reset() -> None:
    """Once the plan's next step is ask_for_reset_cube_bin_only, step() must intercept
    it before execute_ground_skill -- no controller call, no competence model created
    for it -- and the banked cost is exactly the configured flag."""
    method, env = _cube_bin_reset_build(ask_for_reset_cube_bin_cost=0.37)
    task = LightSwitchTasks(env=env, seed=0).sample_train_task()
    reset_ground_skill = _CubeBinCapableSkillProvider(env=env).human_cube_bin_reset_skill()
    episode = _EesEpisode(method=method, goal=task.goal.atoms, practicing=True)
    episode._plan = [reset_ground_skill]

    with pytest.raises(HumanCubeBinResetRequested) as excinfo:
        episode.step(state=task.initial_state)
    assert excinfo.value.cost == 0.37
    assert method._competence_models == {}


def test_ees_evaluation_never_offers_the_cube_bin_reset_skill_either() -> None:
    """Same invariant as the other two reset skills: no evaluation episode is ever
    rescued. Unlike ask_for_reset_task_initial, this is not gated on
    task_initial_atoms (its effects don't depend on per-episode init state at all --
    see human_cube_bin_reset_skill), so this checks get_task_policy runs cleanly
    end-to-end rather than that plan_to declined to offer the skill."""
    method, env = _cube_bin_reset_build(ask_for_reset_cube_bin_cost=0.01)
    task = LightSwitchTasks(env=env, seed=0).sample_train_task()
    policy = method.get_task_policy(task=task)
    state = task.initial_state
    for _ in range(5):
        policy(state)
