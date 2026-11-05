"""EES told the test-task goals up front scores every Pour skill from the start."""

import pytest

from hitl_pmp.cli import Cli
from hitl_pmp.environments.pouring3d.kitchen.cli import KitchenCli
from hitl_pmp.environments.pouring3d.practice.environment import FrozenPourEnvironment
from hitl_pmp.environments.pouring3d.practice.skill_provider import FrozenPourSkillProvider
from hitl_pmp.environments.pouring3d.practice.tasks import FrozenPourTasks
from hitl_pmp.methods.practice_makes_perfect.ees_method import EesMethod
from tests.practice_cost_fixtures import CostFixtures

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
        preview_test_goals=tuple(
            EesMethod.named_goal(goal=goal.atoms) for goal in tasks.test_task_goals()
        ),
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
        preview_test_goals=tuple(
            EesMethod.named_goal(goal=goal.atoms) for goal in tasks.test_task_goals()
        ),
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
            *CostFixtures.argv(env="pouring3d_kitchen"),
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
        assert [goal[0][1][0] for goal in method.preview_test_goals] == [
            "request_1",
            "request_2",
            "request_3",
        ]


def test_preview_goals_survive_the_deployment_snapshot(*, world):
    from hitl_pmp.step_protocol import DeploymentSnapshot

    env, tasks, provider = world
    goals = tuple(EesMethod.named_goal(goal=goal.atoms) for goal in tasks.test_task_goals())
    method = EesMethod(env=env, skill_provider=provider, preview_test_goals=goals)
    method.get_practice_policy(task=tasks.sample_train_task_in_place())
    raw = DeploymentSnapshot.encode(method=method)
    restored = DeploymentSnapshot.restore(raw=raw, env=env, provider=provider)
    assert restored.preview_test_goals == goals


def _scored(*, method, monkeypatch, ground_skill):
    """The task ids planning progress scores against, and the score's denominator."""
    records = []
    monkeypatch.setattr(
        EesMethod, "_write_ees_decision", lambda self, *, record: records.append(record)
    )
    method.score_ground_skill(ground_skill=ground_skill)
    (score,) = [r for r in records if r["event"] == "score"]
    return sorted(method._task_plan_cache, key=str), score["normalization_denominator"]


def _with_seen_deliver_one(*, method, tasks, preview):
    task = tasks.sample_train_task_in_place()
    method.get_practice_policy(task=task)
    init = method.abstract_state(state=task.initial_state)
    for _ in range(4):
        method.record_seen_task(init_atoms=init, goal=task.goal.atoms)
    assert len(method._seen_tasks) >= 4
    assert len(method._preview_tasks) == (3 if preview else 0)


def test_with_the_preview_on_scoring_ignores_the_seen_training_tasks(*, world, monkeypatch):
    env, tasks, provider = world
    method = EesMethod(
        env=env,
        skill_provider=provider,
        preview_test_goals=tuple(
            EesMethod.named_goal(goal=goal.atoms) for goal in tasks.test_task_goals()
        ),
    )
    _with_seen_deliver_one(method=method, tasks=tasks, preview=True)
    pour2 = next(g for g in method.practice_candidates() if g.skill.name == "Pour2")
    ids, denominator = _scored(method=method, monkeypatch=monkeypatch, ground_skill=pour2)
    assert all(kind in {"preview", "replan"} for kind, _ in ids), ids
    replans = len(method._replanning_tasks)
    assert denominator == 3 + replans
    if not replans:
        # Each test goal weighs once: Pour1, Pour2 and Pour3 each in one plan.
        plans = method.planning_progress_plans()
        pours = [s.skill.name for plan in plans for s in plan if s.skill.name.startswith("Pour")]
        assert sorted(pours) == ["Pour1", "Pour2", "Pour3"]


def test_with_the_preview_off_scoring_still_uses_the_seen_training_tasks(*, world, monkeypatch):
    env, tasks, provider = world
    method = EesMethod(env=env, skill_provider=provider)
    _with_seen_deliver_one(method=method, tasks=tasks, preview=False)
    ids, denominator = _scored(
        method=method, monkeypatch=monkeypatch, ground_skill=provider.ground(name="Wait")
    )
    assert "train" in {kind for kind, _ in ids}
    assert denominator == len(method._seen_tasks)
