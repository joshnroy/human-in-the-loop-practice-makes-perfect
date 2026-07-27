import numpy as np

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.environments.ballring.environment import BallRingEnvironment
from hitl_pmp.environments.ballring.problem import BallRingProblem
from hitl_pmp.environments.ballring.skill_provider import BallRingOracle, BallRingSkillProvider
from hitl_pmp.environments.ballring.skills import BallRingSkills
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


def test_provider_routes_oracle_sampler_input_to_the_skills() -> None:
    """The domain-agnostic Method reaches oracle feature selection only through the
    provider hook, so the provider must delegate to BallRingSkills (and return None
    for a non-cup-place skill, i.e. fall back to "all")."""
    env = E()
    provider = BallRingSkillProvider(env=env)
    state = env.sample_initial_state(rng=np.random.default_rng(0))
    table = env.target_table()
    place_cup = GroundSkill(
        skill=BallRingSkills.PLACE_CUP_WITHOUT_BALL_ON_TABLE,
        objects=(env.robot, env.ball, env.cup, table),
    )
    params = np.array([0.4, 0.7])
    assert provider.oracle_sampler_input(
        ground_skill=place_cup, state=state, params=params
    ) == BallRingSkills.oracle_sampler_input(ground_skill=place_cup, state=state, params=params)

    nav = GroundSkill(skill=BallRingSkills.NAVIGATE_TO_TABLE, objects=(env.robot, table))
    assert provider.oracle_sampler_input(ground_skill=nav, state=state, params=np.zeros(0)) is None


def test_oracle_solves_a_sampled_test_task_within_the_horizon() -> None:
    env = E()
    problem = BallRingProblem(env=env, tasks=BallRingTasks(env=env, seed=0))
    oracle = BallRingOracle(env=env)
    task = problem.tasks.sample_test_task()
    solved, _ = problem.run_task_episode(
        task=task, policy=lambda state: oracle.get_labeled_action(state=state, goal=task.goal)
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
            task=task,
            policy=lambda state, goal=task.goal: oracle.get_labeled_action(state=state, goal=goal),
        )
        solved += int(ok)
    assert solved == 8
