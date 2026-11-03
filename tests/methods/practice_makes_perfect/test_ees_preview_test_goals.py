"""EES told the test-task goals up front scores every Pour skill from the start."""

import pytest

from hitl_pmp.cli import Cli
from hitl_pmp.environments.pouring3d.kitchen.cli import KitchenCli
from hitl_pmp.environments.pouring3d.practice.environment import FrozenPourEnvironment
from hitl_pmp.environments.pouring3d.practice.skill_provider import FrozenPourSkillProvider
from hitl_pmp.environments.pouring3d.practice.tasks import FrozenPourTasks
from hitl_pmp.methods.practice_makes_perfect.ees_method import EesMethod

pytestmark = pytest.mark.xdist_group("ees_preview_test_goals")


@pytest.fixture(scope="module")
def world():
    env = FrozenPourEnvironment(scene_bg=False, payload_count=3, requested_count=1)
    tasks = FrozenPourTasks(env=env, requested_counts=(1, 2, 3))
    yield env, tasks, FrozenPourSkillProvider(env=env)
    env.close()


def _requests(*, goals):
    return [next(iter(goal)).objects[0].name for goal in goals]


def _pours(*, method):
    return sorted(
        ground.skill.name
        for ground in method.practice_candidates()
        if ground.skill.name.startswith("Pour")
    )


def test_test_task_goals_follow_the_schedule_without_drawing_a_test_task(*, world):
    env, tasks, _ = world
    index = tasks._test_index
    state = tasks._test.bit_generator.state
    goals = tasks.test_task_goals()
    assert _requests(goals=[goal.atoms for goal in goals]) == [
        "request_1",
        "request_2",
        "request_3",
    ]
    assert tasks._test_index == index
    assert tasks._test.bit_generator.state == state
    assert env.requested_count == 1


def test_preview_registers_all_three_goals_and_pours_before_the_first_decision(*, world):
    env, tasks, provider = world
    method = EesMethod(
        env=env,
        skill_provider=provider,
        preview_test_goals=tuple(goal.atoms for goal in tasks.test_task_goals()),
    )
    method.get_practice_policy(task=tasks.sample_train_task_in_place())
    goals = [goal for _init, goal in method.planning_progress_tasks()]
    assert _requests(goals=goals[:3]) == ["request_1", "request_2", "request_3"]
    assert _pours(method=method) == ["Pour1", "Pour2", "Pour3"]
    planned = {step.skill.name for plan in method.planning_progress_plans() for step in plan}
    assert {"Pour1", "Pour2", "Pour3"} <= planned
    pour2 = next(g for g in method.practice_candidates() if g.skill.name == "Pour2")
    wait = provider.ground(name="Wait")
    assert method.score_ground_skill(ground_skill=pour2) > method.score_ground_skill(
        ground_skill=wait
    )


def test_preview_tasks_survive_recency_eviction(*, world):
    env, tasks, provider = world
    method = EesMethod(
        env=env,
        skill_provider=provider,
        planning_progress_max_tasks=2,
        preview_test_goals=tuple(goal.atoms for goal in tasks.test_task_goals()),
    )
    task = tasks.sample_train_task_in_place()
    method.get_practice_policy(task=task)
    init = method.abstract_state(state=task.initial_state)
    for _ in range(5):
        method.record_seen_task(init_atoms=init, goal=task.goal.atoms)
    goals = [goal for _init, goal in method.planning_progress_tasks()]
    assert _requests(goals=goals) == [
        "request_1",
        "request_2",
        "request_3",
        "request_1",
        "request_1",
    ]


def test_default_registers_nothing_up_front(*, world):
    env, tasks, provider = world
    method = EesMethod(env=env, skill_provider=provider)
    method.get_practice_policy(task=tasks.sample_train_task_in_place())
    assert method.practice_candidates() == []
    assert method.planning_progress_tasks() == []


def _kitchen_args(*, extra):
    return Cli.parse_args(
        argv=[
            "--env",
            "pouring3d_kitchen",
            "--method",
            "ees",
            "--payload-count",
            "3",
            "--requested-counts",
            "1",
            "2",
            "3",
            "--num-test-tasks",
            "3",
            "--practice-step-budget",
            "100",
            *extra,
        ]  # fmt: skip
    )


@pytest.mark.parametrize("flag,expected", [([], 0), (["--ees-preview-test-goals"], 3)])
def test_cli_flag_passes_the_kitchen_test_goals_to_ees(*, flag, expected, monkeypatch):
    from hitl_pmp.cli import METHODS

    built = []

    def capture(*, args, method_factory, **kwargs):
        problem = KitchenCli.build_problem(args=args)
        built.append(method_factory(KitchenCli.context(args=args, problem=problem)))

    monkeypatch.setattr(KitchenCli, "run_method", staticmethod(capture))
    args = _kitchen_args(extra=flag)
    METHODS["ees"].run(args=args, env_cli=KitchenCli)
    (method,) = built
    assert len(method.preview_test_goals) == expected
    if expected:
        assert _requests(goals=method.preview_test_goals) == [
            "request_1",
            "request_2",
            "request_3",
        ]
