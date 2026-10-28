"""Practice selects skill groundings; evaluation supplies balanced fixed goals."""

import numpy as np
import pytest

from hitl_pmp.environments.pouring3d.practice.environment import FrozenPourEnvironment
from hitl_pmp.environments.pouring3d.practice.problem import FrozenPourProblem
from hitl_pmp.environments.pouring3d.practice.skill_provider import FrozenPourSkillProvider
from hitl_pmp.environments.pouring3d.practice.tasks import FrozenPourTasks
from hitl_pmp.methods.practice_makes_perfect.ees_method import EesMethod
from hitl_pmp.practice_loop import PracticeLoop, PracticeResetPolicy


def test_practice_sampling_does_not_assign_a_count_or_change_physics():
    env = FrozenPourEnvironment(scene_bg=False)
    tasks = FrozenPourTasks(env=env, requested_counts=(1, 2, 3))
    problem = FrozenPourProblem(env=env, tasks=tasks)
    provider = FrozenPourSkillProvider(env=env)
    method = EesMethod(env=env, skill_provider=provider)
    try:
        state = env.get_current_state()
        c = env.controller()
        before = (c.data.qpos.copy(), c.data.qvel.copy(), c.data.time, c.steps)
        for _ in range(6):
            task = PracticeLoop._sample_practice_task(
                problem=problem, practice_reset_policy=PracticeResetPolicy.NEVER
            )
            assert task.initial_state is state
            assert env.requested_count == c.scene.requested_count == 2
        rows = []
        for request in (3, 1, 2):
            ground = provider.ground(name="Pour", request=request)
            params = np.array([90.5, 0.008, 30.0])
            action = provider.compute_action(ground_skill=ground, state=state, params=params)
            assert action[2] == request
            rows.append(method.sampler_input_row(ground_skill=ground, state=state, params=params))
        # Bias + robot(4) + pitcher(13) + tray(13): the next feature is count.
        assert [row[31] for row in rows] == [3, 1, 2]
        assert method.sampler(skill_name="Pour", param_dim=3) is method.sampler(
            skill_name="Pour", param_dim=3
        )
        np.testing.assert_array_equal(c.data.qpos, before[0])
        np.testing.assert_array_equal(c.data.qvel, before[1])
        assert (c.data.time, c.steps) == before[2:]
    finally:
        env.close()


def test_evaluation_counts_are_balanced_and_reset_sampling_preserves_request():
    env = FrozenPourEnvironment(scene_bg=False)
    tasks = FrozenPourTasks(env=env, seed=2, requested_counts=(1, 2, 3))
    try:
        test_tasks = [tasks.sample_test_task() for _ in range(6)]
        assert [next(iter(t.goal.atoms)).objects[0].name for t in test_tasks] == [
            "request_1",
            "request_2",
            "request_3",
            "request_1",
            "request_2",
            "request_3",
        ]
        env.reset_to_seed(seed=3, requested=2)
        tasks.sample_reset_start_task()
        assert env.requested_count == 2
        assert next(iter(tasks.sample_test_task().goal.atoms)).objects[0].name == "request_1"
    finally:
        env.close()


@pytest.mark.parametrize("counts", [(), (0,), (4,), (1, 1), (1, 2.5)])
def test_invalid_request_schedule_is_rejected(*, counts):
    with pytest.raises(ValueError):
        FrozenPourTasks(env=FrozenPourEnvironment(), requested_counts=counts)


@pytest.mark.parametrize("count", [3, 5])
def test_practice_reset_packing_has_no_initial_intersections(*, count):
    env = FrozenPourEnvironment(payload_count=count, scene_bg=False)
    try:
        env.hard_reset()
        c = env.controller()
        for _ in range(10):
            c.scene.respawn_payload_in_pitcher()
            assert c.scene.cube_locations() == ["pitcher"] * count
            distances = [
                float(contact.dist)
                for contact in c.data.contact
                if any(
                    (c.model.geom(int(g)).name or "").startswith("shrimp_") for g in contact.geom
                )
            ]
            assert min(distances, default=0) >= -1e-6
    finally:
        env.close()


@pytest.mark.parametrize(
    "method,preset", [("ees", "pouring3d-balanced-ees"), ("pomdp", "pouring3d-balanced")]
)
def test_presets_leave_practice_choice_to_robot_and_balance_evaluation(*, method, preset):
    from hitl_pmp.cli import Cli
    from hitl_pmp.environments.pouring3d.practice.cli import FrozenPourCli

    argv = [
        "--env",
        "pouring3d",
        "--method",
        method,
        "--preset",
        preset,
        "--practice-step-budget",
        "1000",
    ]
    args = Cli.parse_args(argv=argv)
    assert args.goal_pursuit_horizon == 0
    problem = FrozenPourCli.build_problem(args=args)
    assert problem.tasks.requested_counts == (1, 2, 3)
    assert args.num_test_tasks == 9
    fixed = FrozenPourCli.build_problem(args=Cli.parse_args(argv=[*argv, "--requested-count", "2"]))
    assert fixed.tasks.requested_counts == (2,)
    with pytest.raises(ValueError, match="multiple"):
        FrozenPourCli.build_problem(args=Cli.parse_args(argv=[*argv, "--num-test-tasks", "5"]))
