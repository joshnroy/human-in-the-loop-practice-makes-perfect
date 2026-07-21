import numpy as np
from pydantic import PrivateAttr

from hitl_pmp.core.problem.tasks.tasks import Tasks
from hitl_pmp.core.problem.tasks.types import Goal, Task

from .environment import LightSwitchEnvironment
from .predicates import LIGHT_ON


class LightSwitchTasks(Tasks):
    """Task/goal generation for Light Switch. The robot always starts in cell 0 and
    the light always starts at level=0.0 -- only the light's target varies per
    episode, sampled Uniform(0.5, 1.0) (GridRowEnv._get_tasks in the sibling
    hitl-practice repo). Train and test use the same sampling distribution, only the
    RNG stream differs -- matches predicators' train/test seed-offset convention
    (CFG.test_env_seed_offset=10000 in hitl-practice/predicators/settings.py).

    env must be the same LightSwitchEnvironment instance the surrounding Problem
    drives (set_seed's own seed defaults to constructing this Tasks with a fresh
    Uniform(0.5, 1.0) RNG stream already wired) -- sample_train_task/
    sample_test_task call self.env.build_initial_state, which needs that specific
    instance's grid_size to place the light/cells correctly."""

    env: LightSwitchEnvironment
    seed: int = 0
    test_env_seed_offset: int = 10000
    target_low: float = 0.5
    target_high: float = 1.0

    # Leading underscore + PrivateAttr, not plain fields: derived from seed/
    # test_env_seed_offset (set_seed rederives both together), not independently
    # constructor-settable -- a caller should never pass train_rng/test_rng
    # directly, only seed.
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
        seed + test_env_seed_offset). The single entry point for reseeding; nothing
        else should assign train_rng/test_rng directly."""
        self.seed = seed
        self._train_rng = self._make_rng(offset=0)
        self._test_rng = self._make_rng(offset=self.test_env_seed_offset)

    def _make_rng(self, *, offset: int) -> np.random.Generator:
        return np.random.default_rng(self.seed + offset)

    def sample_train_task(self) -> Task:
        return self._sample_task(rng=self.train_rng)

    def sample_test_task(self) -> Task:
        return self._sample_task(rng=self.test_rng)

    def _sample_task(self, *, rng: np.random.Generator) -> Task:
        target = float(rng.uniform(self.target_low, self.target_high))
        initial_state = self.env.build_initial_state(light_level=0.0, light_target=target)
        light_on = LIGHT_ON(state=initial_state, objects=(self.env.light,))
        goal = Goal(atoms=frozenset({light_on}))
        return Task(initial_state=initial_state, goal=goal)
