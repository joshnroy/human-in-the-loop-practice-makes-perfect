import pytest

from hitl_pmp.core.method.method import InteractionComplete
from hitl_pmp.core.problem.environment.types import State
from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
from hitl_pmp.environments.lightswitch.skill_provider import LightSwitchSkillProvider
from hitl_pmp.environments.lightswitch.tasks import LightSwitchTasks
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
from hitl_pmp.methods.practice_makes_perfect.random_skills_method import RandomSkillsMethod
from tests.environments.tossing3d.observations import state as tossing3d_state


def _method(*, env: LightSwitchEnvironment, seed: int = 0) -> RandomSkillsMethod:
    return RandomSkillsMethod(env=env, skill_provider=LightSwitchSkillProvider(env=env), seed=seed)


def test_get_labeled_action_returns_an_applicable_skill_from_the_provider() -> None:
    """Fully domain-agnostic now: the method grounds/executes via the injected
    SkillProvider, so the emitted label names one of Light Switch's own skills."""
    env = LightSwitchEnvironment()
    method = _method(env=env)
    state = env.build_initial_state(light_level=0.0, light_target=0.7)
    label = method.get_labeled_action(state=state).label
    assert any(
        label.startswith(name)
        for name in ("MoveRobot", "TurnOnLight", "TurnOffLight", "JumpToLight")
    )


def test_solves_a_sampled_task_eventually_given_enough_steps() -> None:
    """Not a guaranteed-two-action solve like the oracle -- a genuine uniform random
    walk over applicable ground skills takes many more steps than the paper's own
    per-episode horizon to reliably reach the goal. Uses a small grid and a large
    step budget purely to confirm the wired-up policy is genuinely making progress."""
    env = LightSwitchEnvironment(grid_size=5)
    tasks = LightSwitchTasks(env=env, seed=0)
    task = tasks.sample_train_task()
    env.set_state(state=task.initial_state)
    method = _method(env=env, seed=0)
    policy = method.get_task_policy(task=task)

    state = env.get_current_state()
    for _ in range(2000):
        if task.goal.is_satisfied(state=state):
            break
        state = env.take_action(action=policy(state).action)
    assert task.goal.is_satisfied(state=state) is True


def test_reset_environment_reports_failure_and_leaves_the_environment_alone() -> None:
    """This baseline has no way to self-navigate, so it reports failure -- and does not
    reach for the privileged `set_state` that would fake one."""
    env = LightSwitchEnvironment()
    method = _method(env=env)
    stranded = env.build_initial_state(light_level=0.1, light_target=0.9)
    env.set_state(state=stranded)
    start_state = env.build_initial_state(light_level=0.3, light_target=0.8)
    assert method.reset_environment(start_state=start_state) is False
    assert env.get_current_state() is stranded


def test_generate_train_task_is_unreachable() -> None:
    with pytest.raises(NotImplementedError):
        _method(env=LightSwitchEnvironment()).generate_train_task(tbd_inputs=None)


def test_execute_setup_command_is_unreachable() -> None:
    with pytest.raises(NotImplementedError):
        _method(env=LightSwitchEnvironment()).execute_setup_command(setup_command=None)  # type: ignore[arg-type]


def test_execute_skill_is_unreachable() -> None:
    with pytest.raises(NotImplementedError):
        _method(env=LightSwitchEnvironment()).execute_skill(skill=None)  # type: ignore[arg-type]


def test_improve_skill_parameters_is_unreachable() -> None:
    with pytest.raises(NotImplementedError):
        _method(env=LightSwitchEnvironment()).improve_skill_parameters(skill=None, rollout=None)  # type: ignore[arg-type]


def test_same_seed_produces_identical_action_sequences() -> None:
    env_a = LightSwitchEnvironment()
    env_b = LightSwitchEnvironment()
    state = env_a.build_initial_state(light_level=0.0, light_target=0.7)
    method_a = _method(env=env_a, seed=42)
    method_b = _method(env=env_b, seed=42)

    for _ in range(10):
        labeled_a = method_a.get_labeled_action(state=state)
        labeled_b = method_b.get_labeled_action(state=state)
        assert labeled_a.label == labeled_b.label
        assert labeled_a.action.tolist() == labeled_b.action.tolist()


def test_different_seeds_can_produce_different_action_sequences() -> None:
    env_a = LightSwitchEnvironment()
    env_b = LightSwitchEnvironment()
    light_x = float(env_a.grid_size - 0.5)
    state = env_a.build_initial_state(light_level=0.0, light_target=0.7)
    state.set(obj=LightSwitchEnvironment.robot, feature_name="x", feature_val=light_x)
    method_a = _method(env=env_a, seed=1)
    method_b = _method(env=env_b, seed=2)

    labels_a = [method_a.get_labeled_action(state=state).label for _ in range(20)]
    labels_b = [method_b.get_labeled_action(state=state).label for _ in range(20)]
    assert labels_a != labels_b


def _missed_toss_dead_end() -> tuple[Tossing3DEnvironment, RandomSkillsMethod, State]:
    """A real Tossing3D dead end, offline: the state after a toss that missed.

    Not a contrivance. `Toss` unconditionally deletes `Reachable(cube, barrier)` --
    the barrier is one-way -- so once the cube is past it `Pick` is inapplicable, and
    with an empty hand neither `MoveToThrowPose` nor `Toss` is either. Every episode
    of this domain ends here, which is why the old assert crashed 10/10 runs.

    The numbers are the recorded ones, not invented: cube at x = 2.86 is a *missed*
    toss (the goal region is x in [1.85, 2.15], so the cube is outside it and the
    task is unsolved -- had it landed in, `run_task_episode`'s goal check would end
    the episode before ever calling the policy), and the base at x = 0.65 is a real
    post-toss pose, so `RobotAtSuccessfulThrowPose` genuinely holds and the dead end is
    not an artifact of the robot standing somewhere it could never be.
    """
    env = Tossing3DEnvironment()
    method = RandomSkillsMethod(env=env, skill_provider=Tossing3DSkillProvider(env=env), seed=0)
    state = tossing3d_state(env=env, cube_x=2.86, base_x=0.65, steps_taken=3)
    assert method.applicable_ground_skills(state=state) == []
    return env, method, state


def test_a_dead_end_evaluation_step_degrades_to_a_no_op_rather_than_asserting() -> None:
    """The evaluation half, matching EesMethod: `run_task_episode` owns termination,
    so the policy hands back an inert action instead of raising."""
    env, method, dead_end = _missed_toss_dead_end()

    labeled = method.get_task_policy(task=None)(dead_end)  # type: ignore[arg-type]

    assert labeled.label == "no-op (no applicable skills)"
    assert labeled.action.tolist() == env.noop_action().tolist()


def test_a_dead_end_practice_step_ends_the_period_instead_of_burning_it() -> None:
    """The practice half, also matching EesMethod. Without this the two arms are not
    comparable: practice_loop.py charges a transition per step with no goal check, so
    a dead-ended period would spend its whole remaining budget on no-ops while EES
    stops -- and `--method random-skills --num-cycles 10` over EES's budget is exactly
    how the two get plotted on one transition axis."""
    _env, method, dead_end = _missed_toss_dead_end()

    with pytest.raises(InteractionComplete):
        method.get_practice_policy(task=None)(dead_end)  # type: ignore[arg-type]


def test_the_two_phases_agree_wherever_a_skill_is_applicable() -> None:
    """The dead end is the *only* place the phases differ: this baseline explores and
    exploits identically, so a practice policy that diverged anywhere else would be a
    second behaviour to keep in sync rather than one override."""
    env = LightSwitchEnvironment()
    state = env.build_initial_state(light_level=0.0, light_target=0.7)
    practice = _method(env=env, seed=3).get_practice_policy(task=None)  # type: ignore[arg-type]
    evaluation = _method(env=env, seed=3).get_task_policy(task=None)  # type: ignore[arg-type]

    for _ in range(10):
        from_practice, from_evaluation = practice(state), evaluation(state)
        assert from_practice.label == from_evaluation.label
        assert from_practice.action.tolist() == from_evaluation.action.tolist()
