"""A practice session's last completed try is observed when the session ends.

A method reads a try's outcome on its next policy call, and a session's last action has
none, so that try used to reach neither the sampler's training rows nor the competence
counts: one action in every session. On Serving's 10-action sessions that was 345 of
2,683 Sweep and GatherAndSweep tries in the 2026-11-12 budget sweep."""

import argparse

from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
from hitl_pmp.environments.lightswitch.problem import LightSwitchProblem
from hitl_pmp.environments.lightswitch.skill_provider import LightSwitchSkillProvider
from hitl_pmp.environments.lightswitch.tasks import LightSwitchTasks
from hitl_pmp.method_runner import MethodRunner
from hitl_pmp.methods.practice_makes_perfect.ees_method import EesMethod
from hitl_pmp.methods.practice_makes_perfect.random_skills_method import RandomSkillsMethod
from hitl_pmp.step_protocol import StepPracticeRunner
from tests.test_step_protocol import (
    FakeEvaluations,
    FakeProblem,
    FakeStepProtocol,
    ScriptedMethod,
    _step_args,
)


def _ees():
    env = LightSwitchEnvironment(grid_size=4)
    method = EesMethod(env=env, skill_provider=LightSwitchSkillProvider(env=env), seed=0)
    return method, env


def _one_practice_action(*, method, env):
    task = LightSwitchTasks(env=env, seed=0).sample_train_task()
    env.set_state(state=task.initial_state)
    policy = method.get_practice_policy(task=task)
    return env.take_action(action=policy(env.get_current_state()).action)


def test_the_session_end_scores_the_try_in_flight_against_the_state_it_produced():
    method, env = _ees()
    after = _one_practice_action(method=method, env=env)
    assert method.total_observations() == 0
    method.observe_session_end(state=after)
    assert method.total_observations() == 1
    assert method._practice_episode._pending is None
    # Nothing is left in flight, so a second call observes nothing twice.
    method.observe_session_end(state=after)
    assert method.total_observations() == 1


def test_a_session_end_before_any_practice_is_a_no_op():
    method, env = _ees()
    start = LightSwitchTasks(env=env, seed=0).sample_train_task().initial_state
    method.observe_session_end(state=start)
    assert method.total_observations() == 0


def test_the_random_skills_baseline_tallies_its_sessions_last_try_too():
    env = LightSwitchEnvironment(grid_size=3)
    method = RandomSkillsMethod(env=env, skill_provider=LightSwitchSkillProvider(env=env), seed=0)
    state = env.build_initial_state(light_level=0.0, light_target=0.7)
    env.set_state(state=state)
    labeled = method.get_practice_policy(task=None)(state)
    after = env.take_action(action=labeled.action)
    method.observe_session_end(state=after)
    assert sum(t.num_attempts for t in method.practice_outcomes().values()) == 1


class _CountsOutcomes(EesMethod):
    outcomes: int = 0

    def observe_outcome(self, *, ground_skill, success, was_random_exploration=False):
        self.outcomes += 1
        super().observe_outcome(
            ground_skill=ground_skill,
            success=success,
            was_random_exploration=was_random_exploration,
        )


def test_the_cycle_loop_observes_every_action_of_every_session():
    env = LightSwitchEnvironment(grid_size=4)
    problem = LightSwitchProblem(env=env, tasks=LightSwitchTasks(env=env, seed=0))
    method = _CountsOutcomes(
        env=env,
        skill_provider=LightSwitchSkillProvider(env=env),
        seed=0,
        sampler_max_train_iters=10,
    )
    metrics = MethodRunner.run(
        args=argparse.Namespace(num_test_tasks=1, output_dir=None, practice_reset_interval=None),
        method=method,
        problem=problem,
        num_cycles=2,
        max_steps_per_interaction=8,
        renderer=None,
        render_fps=2,
    )
    assert metrics.evaluations[-1][0] == 16
    assert method.outcomes == 16


class _RecordsSessionEnds(ScriptedMethod):
    """The order of session-end observations and learning updates, with the state each
    session end was handed."""

    def model_post_init(self, context, /) -> None:  # noqa: PLR0917
        super().model_post_init(context)
        self._calls: list = []

    def observe_session_end(self, *, state) -> None:
        self._calls.append(("session_end", state))

    def end_cycle(self) -> None:
        self._calls.append(("end_cycle",))


def _run(*, tmp_path, script, budget, max_actions):
    env = LightSwitchEnvironment(grid_size=4)
    method = _RecordsSessionEnds(env=env, skill_provider=LightSwitchSkillProvider(env=env), seed=0)
    method._script = list(script)
    StepPracticeRunner.run(
        environment=FakeStepProtocol,
        args=_step_args(
            output_dir=tmp_path / "run",
            budget=budget,
            interval=100,
            max_steps_per_interaction=max_actions,
        ),
        method=method,
        problem=FakeProblem(),
        evaluations=FakeEvaluations(),
    )
    return method._calls


def test_the_step_protocol_observes_each_sessions_last_try_before_it_learns(*, tmp_path):
    # Two full sessions of three one-step skills, then one the budget ends after one.
    calls = _run(tmp_path=tmp_path, script=[], budget=7, max_actions=3)
    assert calls == [("session_end", ("after", 1.0)), ("end_cycle",)] * 3


def test_a_try_the_step_budget_cut_is_not_observed_at_the_session_end(*, tmp_path):
    # A two-step skill completes; a five-step one is cut by the budget of four.
    calls = _run(tmp_path=tmp_path, script=[2, 5], budget=4, max_actions=20)
    assert calls == [("end_cycle",)]
