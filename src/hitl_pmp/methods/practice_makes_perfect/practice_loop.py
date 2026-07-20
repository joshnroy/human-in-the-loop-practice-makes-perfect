from collections.abc import Callable

from hitl_pmp.core.method.method import Method
from hitl_pmp.core.metrics.metrics import Metrics
from hitl_pmp.core.problem.problem import Problem


class PracticeLoop:
    """Drives PMP-style online learning: one initial evaluation, then num_cycles
    rounds of (one interaction period, an optional per-cycle retraining hook,
    an evaluation sweep over sampled test tasks) -- mirrors predicators'
    main.py:_run_pipeline. hard_reset() is called exactly once, before the very
    first evaluation, and never again -- but this does NOT mean state is reset-free
    end to end: run_task_episode (called once per test task inside _evaluate) resets
    the environment itself via env.set_state(state=task.initial_state), since it has
    to start each evaluation episode from that task's own initial state. So state is
    only continuous *within* one interaction period (a Method's own get_task_policy
    is expected to decide internally, e.g. by checking task.goal.is_satisfied
    against the current state on every call, when to keep pursuing the sampled
    task's goal versus switch to self-directed practice for the rest of the period);
    each new interaction period actually resumes from wherever the immediately
    preceding evaluation sweep's last episode left off, not from where the
    interaction period itself ended.

    Domain- and Method-agnostic (any core.Problem/core.Method/core.Metrics
    triple), though scoped under methods/practice_makes_perfect/ for now
    since its cycle/interaction-period structure is shaped by this specific
    reproduction's needs. A static-method container, never instantiated, same
    as every other business-logic class in this project.

    Caller must wire Problem.env/Problem.tasks (ClassVar assignment on the base
    Problem class, e.g. `Problem.env = LightSwitchEnvironment; Problem.tasks =
    LightSwitchTasks`) before calling run() -- run() only ever calls the `problem`
    argument's inherited facade methods (hard_reset, sample_train_task, etc.),
    which read Problem.env/Problem.tasks off the base class by name, not off
    whichever subclass was passed in (see core/README.md's Problem facade section).
    A concrete Problem subclass that sets env/tasks as its own class attributes
    (rather than assigning them onto the shared Problem base) will NOT satisfy
    this -- LightSwitchProblem, for instance, only works here because nothing in
    this class calls its inherited facade methods without that wiring already
    having been done by the caller."""

    @staticmethod
    def run(
        *,
        problem: type[Problem],
        method: type[Method],
        metrics: type[Metrics],
        num_cycles: int,
        max_steps_per_interaction: int,
        num_test_tasks: int,
        on_cycle_end: Callable[[], None] | None = None,
    ) -> None:
        problem.hard_reset()
        num_online_transitions = 0
        PracticeLoop._evaluate(
            problem=problem,
            method=method,
            metrics=metrics,
            num_test_tasks=num_test_tasks,
            num_online_transitions=num_online_transitions,
        )
        for _ in range(num_cycles):
            task = problem.sample_train_task()
            policy = method.get_task_policy(task=task)
            state = problem.get_current_state()
            for _ in range(max_steps_per_interaction):
                labeled_action = policy(state)
                state = problem.take_action(action=labeled_action.action)
                num_online_transitions += 1
            if on_cycle_end is not None:
                on_cycle_end()
            PracticeLoop._evaluate(
                problem=problem,
                method=method,
                metrics=metrics,
                num_test_tasks=num_test_tasks,
                num_online_transitions=num_online_transitions,
            )

    @staticmethod
    def _evaluate(
        *,
        problem: type[Problem],
        method: type[Method],
        metrics: type[Metrics],
        num_test_tasks: int,
        num_online_transitions: int,
    ) -> None:
        num_solved = 0
        for _ in range(num_test_tasks):
            task = problem.sample_test_task()
            solved, _frames = problem.run_task_episode(
                task=task, policy=method.get_task_policy(task=task)
            )
            num_solved += int(solved)
        metrics.record_evaluation(
            num_online_transitions=num_online_transitions,
            num_solved=num_solved,
            num_total=num_test_tasks,
        )
