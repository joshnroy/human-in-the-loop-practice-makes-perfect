from collections.abc import Iterator

import pytest

from hitl_pmp.core.problem.problem import Problem
from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
from hitl_pmp.environments.lightswitch.metrics import LightSwitchMetrics
from hitl_pmp.environments.lightswitch.predicates import (
    ADJACENT,
    LIGHT_IN_CELL,
    LIGHT_OFF,
    LIGHT_ON,
    ROBOT_IN_CELL,
)
from hitl_pmp.environments.lightswitch.problem import LightSwitchProblem
from hitl_pmp.environments.lightswitch.skills import LightSwitchSkills
from hitl_pmp.environments.lightswitch.tasks import LightSwitchTasks
from hitl_pmp.methods.practice_makes_perfect.practice_loop import PracticeLoop
from hitl_pmp.methods.practice_makes_perfect.random_skills_method import RandomSkillsMethod


@pytest.fixture(autouse=True)
def _wire_random_skills_on_light_switch() -> Iterator[None]:
    env = LightSwitchEnvironment
    original_grid_size = env.grid_size
    # PracticeLoop drives Problem's own inherited facade methods
    # (hard_reset, sample_train_task, ...), which hardcode references to
    # Problem.env/Problem.tasks specifically -- LightSwitchProblem.env being
    # set in its own class body isn't picked up by those inherited methods, so
    # this wiring step is genuinely required here, matching
    # tests/core/problem/test_problem.py's own _wire_problem() pattern.
    original_problem_env = getattr(Problem, "env", None)
    original_problem_tasks = getattr(Problem, "tasks", None)
    try:
        env.grid_size = 5  # small: RandomSkillsMethod never plans, so keep episodes cheap
        Problem.env = LightSwitchEnvironment
        Problem.tasks = LightSwitchTasks
        RandomSkillsMethod.env = env
        RandomSkillsMethod.tasks = LightSwitchTasks
        RandomSkillsMethod.predicates = (
            ADJACENT,
            LIGHT_IN_CELL,
            LIGHT_OFF,
            LIGHT_ON,
            ROBOT_IN_CELL,
        )
        RandomSkillsMethod.skills = (
            LightSwitchSkills.MOVE_ROBOT,
            LightSwitchSkills.TURN_ON_LIGHT,
            LightSwitchSkills.TURN_OFF_LIGHT,
            LightSwitchSkills.JUMP_TO_LIGHT,
        )
        RandomSkillsMethod.objects = (env.robot, env.light, *env.get_cells())
        RandomSkillsMethod.compute_action = LightSwitchSkills.compute_action
        RandomSkillsMethod.sample_params = LightSwitchSkills.sample_params
        RandomSkillsMethod.reset_state(seed=0)
        LightSwitchMetrics.reset()
        yield
    finally:
        env.grid_size = original_grid_size
        if original_problem_env is not None:
            Problem.env = original_problem_env
        if original_problem_tasks is not None:
            Problem.tasks = original_problem_tasks
        LightSwitchMetrics.reset()


def test_practice_loop_produces_a_training_curve_for_random_skills() -> None:
    PracticeLoop.run(
        problem=LightSwitchProblem,
        method=RandomSkillsMethod,
        metrics=LightSwitchMetrics,
        num_cycles=2,
        max_steps_per_interaction=3,
        num_test_tasks=2,
    )
    curve = LightSwitchMetrics.task_training_curve()
    # 1 initial checkpoint + 1 per cycle.
    assert len(curve) == 3
    transitions = [t for t, _ in curve]
    assert transitions == [0, 3, 6]
    # Every percentage is a valid fraction of num_test_tasks=2.
    assert all(pct in (0.0, 0.5, 1.0) for _, pct in curve)


def test_practice_loop_rarely_solves_with_random_skills() -> None:
    """Random Skills never plans toward a goal at all -- across enough test
    tasks, at grid_size=5, it should solve at most a small fraction by sheer
    luck (matching the paper's own near-0% curve), not routinely succeed.
    Uses a threshold rather than asserting exactly 0.0: a purely-random walk
    genuinely can land on the light and sample a lucky dlight often enough
    that "exactly zero out of many draws" isn't a safe assertion for any
    given seed (empirically confirmed: 3 of 20 tried seeds solved exactly 1
    of 15 test tasks by chance) -- what should hold regardless of seed is
    that it stays rare."""
    PracticeLoop.run(
        problem=LightSwitchProblem,
        method=RandomSkillsMethod,
        metrics=LightSwitchMetrics,
        num_cycles=0,
        max_steps_per_interaction=1,
        num_test_tasks=15,
    )
    _, percent_solved = LightSwitchMetrics.task_training_curve()[0]
    assert percent_solved <= 0.2
