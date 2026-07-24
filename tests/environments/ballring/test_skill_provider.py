import numpy as np

from hitl_pmp.environments.ballring.environment import BallRingEnvironment
from hitl_pmp.environments.ballring.problem import BallRingProblem
from hitl_pmp.environments.ballring.skill_provider import BallRingOracle, BallRingSkillProvider
from hitl_pmp.environments.ballring.tasks import BallRingTasks

E = BallRingEnvironment


def test_provider_exposes_the_full_symbolic_layer() -> None:
    provider = BallRingSkillProvider(env=E())
    assert len(provider.skills()) == 16
    assert len(provider.predicates()) == 12
    assert len(provider.types()) == 4
    # robot, ball, cup + the 5-table ring.
    names = [obj.name for obj in provider.objects()]
    assert set(names) == {"robot", "ball", "cup"} | {
        f"{'sticky' if i == 0 else 'normal'}-table-{i}" for i in range(5)
    }


def test_provider_objects_match_the_objects_in_a_sampled_state() -> None:
    """objects() is reconstructed from config (no state); it must value-equal the
    objects a sampled State actually contains, or grounding would silently miss them."""
    env = E()
    state = env.sample_initial_state(rng=np.random.default_rng(0))
    assert set(BallRingSkillProvider(env=env).objects()) == set(state.data)


def test_oracle_solves_a_sampled_test_task_within_the_horizon() -> None:
    env = E()
    problem = BallRingProblem(env=env, tasks=BallRingTasks(env=env, seed=0))
    oracle = BallRingOracle(env=env)
    task = problem.tasks.sample_test_task()
    solved, _ = problem.run_task_episode(
        task=task, policy=lambda state: oracle.get_labeled_action(state=state)
    )
    assert solved is True


def test_oracle_solves_multiple_seeds() -> None:
    env = E()
    problem = BallRingProblem(env=env, tasks=BallRingTasks(env=env, seed=7))
    oracle = BallRingOracle(env=env)
    solved = 0
    for _ in range(8):
        task = problem.tasks.sample_test_task()
        ok, _ = problem.run_task_episode(
            task=task, policy=lambda state: oracle.get_labeled_action(state=state)
        )
        solved += int(ok)
    assert solved == 8
