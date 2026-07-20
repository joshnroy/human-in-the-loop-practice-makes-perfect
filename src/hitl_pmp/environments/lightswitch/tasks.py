from typing import ClassVar

import numpy as np

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
    """

    seed: ClassVar[int] = 0
    test_env_seed_offset: ClassVar[int] = 10000
    target_low: ClassVar[float] = 0.5
    target_high: ClassVar[float] = 1.0

    train_rng: ClassVar[np.random.Generator]
    test_rng: ClassVar[np.random.Generator]

    @staticmethod
    def set_seed(*, seed: int) -> None:
        """Reset seed and rederive both RNG streams together -- mirrors predicators'
        BaseEnv._set_seed (train_rng from seed, test_rng from
        seed + test_env_seed_offset). The single entry point for reseeding; nothing
        else should assign train_rng/test_rng directly."""
        LightSwitchTasks.seed = seed
        LightSwitchTasks.train_rng = LightSwitchTasks._make_rng(offset=0)
        LightSwitchTasks.test_rng = LightSwitchTasks._make_rng(
            offset=LightSwitchTasks.test_env_seed_offset
        )

    @staticmethod
    def _make_rng(*, offset: int) -> np.random.Generator:
        return np.random.default_rng(LightSwitchTasks.seed + offset)

    @staticmethod
    def sample_train_task() -> Task:
        return LightSwitchTasks._sample_task(rng=LightSwitchTasks.train_rng)

    @staticmethod
    def sample_test_task() -> Task:
        return LightSwitchTasks._sample_task(rng=LightSwitchTasks.test_rng)

    @staticmethod
    def _sample_task(*, rng: np.random.Generator) -> Task:
        target = float(rng.uniform(LightSwitchTasks.target_low, LightSwitchTasks.target_high))
        initial_state = LightSwitchEnvironment.build_initial_state(
            light_level=0.0, light_target=target
        )
        light_on = LIGHT_ON(state=initial_state, objects=(LightSwitchEnvironment.light,))
        goal = Goal(atoms=frozenset({light_on}))
        return Task(initial_state=initial_state, goal=goal)


LightSwitchTasks.set_seed(seed=LightSwitchTasks.seed)
