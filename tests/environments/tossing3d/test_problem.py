from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.problem import Tossing3DProblem
from hitl_pmp.environments.tossing3d.tasks import Tossing3DTasks


def _problem() -> Tossing3DProblem:
    """Constructing the whole composition root opens no simulator: the environment's
    backend is lazy and Tasks only derives its RNG streams."""
    env = Tossing3DEnvironment()
    return Tossing3DProblem(env=env, tasks=Tossing3DTasks(env=env))


def test_horizon_is_the_shortest_solve_plus_two_spare_actions() -> None:
    """The paper's H_eval convention, the same one Light Switch and Tossing Room cite."""
    problem = _problem()
    assert problem.shortest_solve() == 3
    assert problem.max_episode_steps() == problem.shortest_solve() + 2


def test_no_human_oracle_is_wired() -> None:
    """Tossing3D genuinely needs one -- a tossed cube is unrecoverable -- and none has
    been built. `human` staying None is the honest record of that, and the reason
    `Metrics.num_human_interventions()` reports zero for this domain."""
    assert _problem().human is None


def test_train_and_test_seeds_come_from_disjoint_blocks() -> None:
    """A test episode must never be one the Method already practiced on."""
    tasks = Tossing3DTasks(env=Tossing3DEnvironment(), seed=0)
    train = [int(tasks.train_rng.integers(0, tasks.seed_space)) for _ in range(500)]
    test = [
        int(tasks.test_rng.integers(tasks.seed_space, 2 * tasks.seed_space)) for _ in range(500)
    ]
    assert max(train) < tasks.seed_space <= min(test)


def test_set_seed_rederives_both_streams_together() -> None:
    tasks = Tossing3DTasks(env=Tossing3DEnvironment(), seed=0)
    first = int(tasks.train_rng.integers(0, tasks.seed_space))
    tasks.set_seed(seed=0)
    assert int(tasks.train_rng.integers(0, tasks.seed_space)) == first
    tasks.set_seed(seed=1)
    assert int(tasks.train_rng.integers(0, tasks.seed_space)) != first
