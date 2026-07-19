"""Runs the Light Switch oracle policy over sampled test tasks and reports a success
rate -- the "Step 3: implement an oracle" step from the Problem Setting recipe,
establishing an upper bound on the metrics before any learning method exists.
"""

from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
from hitl_pmp.environments.lightswitch.oracle_policy import ORACLE_POLICY
from hitl_pmp.environments.lightswitch.problem import LightSwitchProblem
from hitl_pmp.environments.lightswitch.tasks import LightSwitchTasks

NUM_TEST_TASKS = 20


def main() -> None:
    LightSwitchProblem.wire()
    LightSwitchEnvironment.hard_reset()

    num_solved = 0
    for i in range(NUM_TEST_TASKS):
        task = LightSwitchTasks.sample_test_task()
        solved = LightSwitchProblem.run_task_episode(task=task, policy=ORACLE_POLICY)
        num_solved += int(solved)
        print(f"task {i + 1}/{NUM_TEST_TASKS}: {'solved' if solved else 'FAILED'}")

    print(f"success rate: {num_solved}/{NUM_TEST_TASKS} ({num_solved / NUM_TEST_TASKS:.0%})")


if __name__ == "__main__":
    main()
