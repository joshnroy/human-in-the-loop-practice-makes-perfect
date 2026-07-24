import numpy as np
from pydantic import PrivateAttr

from hitl_pmp.core.problem.tasks.tasks import Tasks
from hitl_pmp.core.problem.tasks.types import Goal, Task

from .environment import BallRingEnvironment
from .predicates import BALL_ON_TABLE


class BallRingTasks(Tasks):
    """Task/goal generation for Ball-Ring. Train and test draw from the same
    distribution (``env.sample_initial_state``, the port of predicators'
    ``_get_tasks`` inner loop); only the RNG stream differs -- the test stream is
    derived from ``seed + test_env_seed_offset`` (``CFG.test_env_seed_offset=10000``
    in hitl-practice/predicators/settings.py), exactly as ``LightSwitchTasks`` does.

    The goal is always ``BallOnTable(ball, target_table)`` where ``target_table`` is
    the sticky table (``env.get_tables(state)[-1]``, matching predicators'
    ``tables[-1]``) -- the initial state deliberately starts the ball balanced on a
    *different* (normal) table, so the goal is never already satisfied and the agent
    must actually transport the ball via the cup.

    env must be the same ``BallRingEnvironment`` instance the surrounding Problem
    drives: ``sample_initial_state`` reads that instance's own
    ``num_tables``/``num_sticky_tables`` to place the ring of tables.
    """

    env: BallRingEnvironment
    seed: int = 0
    test_env_seed_offset: int = 10000

    _train_rng: np.random.Generator = PrivateAttr()
    _test_rng: np.random.Generator = PrivateAttr()

    def model_post_init(self, __context: object) -> None:
        self.set_seed(seed=self.seed)

    @property
    def train_rng(self) -> np.random.Generator:
        return self._train_rng

    @property
    def test_rng(self) -> np.random.Generator:
        return self._test_rng

    def set_seed(self, *, seed: int) -> None:
        """Reset seed and rederive both RNG streams together -- mirrors predicators'
        BaseEnv._set_seed (train_rng from seed, test_rng from
        seed + test_env_seed_offset). The single entry point for reseeding."""
        self.seed = seed
        self._train_rng = np.random.default_rng(seed)
        self._test_rng = np.random.default_rng(seed + self.test_env_seed_offset)

    def sample_train_task(self) -> Task:
        return self._sample_task(rng=self.train_rng)

    def sample_test_task(self) -> Task:
        """A randomly sampled test task, not known to the agent ahead of time."""
        return self._sample_task(rng=self.test_rng)

    def _sample_task(self, *, rng: np.random.Generator) -> Task:
        initial_state = self.env.sample_initial_state(rng=rng)
        target_table = self.env.get_tables(state=initial_state)[-1]
        goal_atom = BALL_ON_TABLE(state=initial_state, objects=(self.env.ball, target_table))
        return Task(initial_state=initial_state, goal=Goal(atoms=frozenset({goal_atom})))
