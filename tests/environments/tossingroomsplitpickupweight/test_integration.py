import pytest

from hitl_pmp.core.metrics.metrics import Metrics
from hitl_pmp.environments.tossingroomsplitpickupweight.environment import (
    TossingRoomSplitPickupWeightEnvironment,
)
from hitl_pmp.environments.tossingroomsplitpickupweight.problem import (
    TossingRoomSplitPickupWeightProblem,
)
from hitl_pmp.environments.tossingroomsplitpickupweight.skill_provider import (
    TossingRoomSplitPickupWeightOracle,
    TossingRoomSplitPickupWeightSkillProvider,
)
from hitl_pmp.environments.tossingroomsplitpickupweight.tasks import (
    TossingRoomSplitPickupWeightGoalType,
    TossingRoomSplitPickupWeightTasks,
)
from hitl_pmp.methods.oracle.skill_oracle_method import SkillOracleMethod
from hitl_pmp.methods.practice_makes_perfect.ees_method import EesMethod
from hitl_pmp.practice_loop import PracticeLoop


@pytest.mark.parametrize("goal_type", list(TossingRoomSplitPickupWeightGoalType))
def test_oracle_solves_every_goal_type_on_train_and_test_tasks(
    *, goal_type: TossingRoomSplitPickupWeightGoalType
) -> None:
    env = TossingRoomSplitPickupWeightEnvironment()
    tasks = TossingRoomSplitPickupWeightTasks(env=env, seed=0, forced_goal_type=goal_type)
    problem = TossingRoomSplitPickupWeightProblem(env=env, tasks=tasks)
    method = SkillOracleMethod(env=env, oracle=TossingRoomSplitPickupWeightOracle(env=env))

    for sample in (tasks.sample_train_task, tasks.sample_test_task):
        for _ in range(10):
            task = sample()
            solved, _ = problem.run_task_episode(
                task=task, policy=method.get_task_policy(task=task)
            )
            assert solved is True


def test_oracle_solves_the_default_mixed_distribution() -> None:
    env = TossingRoomSplitPickupWeightEnvironment()
    tasks = TossingRoomSplitPickupWeightTasks(env=env, seed=0)
    problem = TossingRoomSplitPickupWeightProblem(env=env, tasks=tasks)
    method = SkillOracleMethod(env=env, oracle=TossingRoomSplitPickupWeightOracle(env=env))
    solved = 0
    for _ in range(30):
        task = tasks.sample_test_task()
        ok, _ = problem.run_task_episode(task=task, policy=method.get_task_policy(task=task))
        solved += int(ok)
    assert solved == 30


def test_oracle_presses_both_buttons_in_the_only_feasible_order() -> None:
    """EMPTY is now an ordering task: the trash button (room 6) must be pressed BEFORE
    crossing the one-way ledge down to the recycling one (room 1), because nothing to the
    right of the ledge is reachable again afterwards. Asserted on the realised action
    sequence, not just on the solved flag, so a solve that happened to work by another
    route would still be visible."""
    env = TossingRoomSplitPickupWeightEnvironment()
    tasks = TossingRoomSplitPickupWeightTasks(
        env=env, seed=0, forced_goal_type=TossingRoomSplitPickupWeightGoalType.EMPTY
    )
    problem = TossingRoomSplitPickupWeightProblem(env=env, tasks=tasks)
    method = SkillOracleMethod(env=env, oracle=TossingRoomSplitPickupWeightOracle(env=env))
    for _ in range(10):
        task = tasks.sample_test_task()
        state = problem.reset_to_task(task=task)
        policy = method.get_task_policy(task=task)
        pressed: list[int] = []
        for _ in range(problem.max_episode_steps()):
            if task.goal.is_satisfied(state=state):
                break
            action = policy(state).action
            if int(round(action[0])) == TossingRoomSplitPickupWeightEnvironment.SKILL_PRESS:
                pressed.append(int(round(action[1])))
            state = env.take_action(action=action)
        assert task.goal.is_satisfied(state=state) is True
        assert pressed == [
            TossingRoomSplitPickupWeightEnvironment.TRASH_KIND,
            TossingRoomSplitPickupWeightEnvironment.RECYCLING_KIND,
        ]


def test_oracle_never_issues_the_blocked_rightward_ledge_step() -> None:
    """The oracle is forward-only: it should solve without ever attempting the single
    irreversible-blocked move (rightward across the ledge), so it never needs help."""
    env = TossingRoomSplitPickupWeightEnvironment()
    tasks = TossingRoomSplitPickupWeightTasks(env=env, seed=0)
    problem = TossingRoomSplitPickupWeightProblem(env=env, tasks=tasks)
    method = SkillOracleMethod(env=env, oracle=TossingRoomSplitPickupWeightOracle(env=env))
    for _ in range(30):
        task = tasks.sample_test_task()
        state = problem.reset_to_task(task=task)
        policy = method.get_task_policy(task=task)
        for _ in range(problem.max_episode_steps()):
            if task.goal.is_satisfied(state=state):
                break
            action = policy(state).action
            robot_room = int(round(state.get(obj=env.robot, feature_name="room")))
            if int(round(action[0])) == TossingRoomSplitPickupWeightEnvironment.SKILL_MOVE_ROOM:
                to_room = int(round(action[1]))
                assert not (
                    robot_room == env.blocked_right_from and to_room == env.blocked_right_from + 1
                )
            state = env.take_action(action=action)


class TestEesRunsOnThisDomainAndTrainsBothThrowsSeparately:
    """The premise the PR-2 experiment rests on, checked against a real run rather than
    against the representation alone: EES plans over the split symbolic layer with real
    Fast Downward, practices both throws, and accumulates their training data into two
    different samplers.

    Deliberately short (4 cycles x 60 steps) -- this is a wiring check, not a
    measurement. It was 3 until the throw-representation change moved the training stream
    (`build_task` now draws four uniforms per task, not two): seed 0's first three
    practice tasks are now trash/empty/empty, so `ThrowRecycling` was never reached and
    the non-vacuity assertion below went quiet.

    The counts are asserted as inequalities for exactly that reason: how
    *many* attempts each throw gets is what the experiment measures, and pinning it here
    would either duplicate that result or make this test fail whenever the measurement
    legitimately moves.
    """

    @staticmethod
    def _run() -> EesMethod:
        env = TossingRoomSplitPickupWeightEnvironment()
        tasks = TossingRoomSplitPickupWeightTasks(env=env, seed=0, num_test_tasks=4)
        problem = TossingRoomSplitPickupWeightProblem(env=env, tasks=tasks)
        method = EesMethod(
            env=env,
            skill_provider=TossingRoomSplitPickupWeightSkillProvider(env=env),
            seed=0,
            sampler_max_train_iters=100,
        )
        PracticeLoop.run(
            problem=problem,
            method=method,
            metrics=Metrics(),
            num_cycles=4,
            max_steps_per_interaction=60,
            num_test_tasks=4,
        )
        return method

    @staticmethod
    def test_both_throw_samplers_receive_data_and_they_are_different_objects() -> None:
        method = TestEesRunsOnThisDomainAndTrainsBothThrowsSeparately._run()
        trash = method.sampler(skill_name="ThrowTrash", param_dim=1)
        recycling = method.sampler(skill_name="ThrowRecycling", param_dim=1)
        assert trash is not recycling
        # Non-vacuity: a run in which neither throw was ever practiced would satisfy any
        # "they are separate" assertion trivially.
        assert trash.num_observations > 0
        assert recycling.num_observations > 0
        # And no row is in both -- the separation claim, on data a real run produced.
        trash_rows = {tuple(row) for row in trash.observed_inputs()}
        recycling_rows = {tuple(row) for row in recycling.observed_inputs()}
        assert not (trash_rows & recycling_rows)

    @staticmethod
    def test_no_sampler_is_created_for_the_unsplit_name() -> None:
        """A `Throw` sampler appearing would mean something still emits the old name."""
        method = TestEesRunsOnThisDomainAndTrainsBothThrowsSeparately._run()
        assert method.sampler(skill_name="Throw", param_dim=1).num_observations == 0
