import numpy as np
from pydantic import PrivateAttr

from hitl_pmp.core.problem.tasks.tasks import Tasks
from hitl_pmp.core.problem.tasks.types import Goal, Task

from .environment import Tossing3DEnvironment
from .predicates import IN_GOAL_REGION


class Tossing3DTasks(Tasks):
    """Task generation for Tossing3D. A task IS a KINDER episode seed: KINDER's own
    task JSON already defines the initial-state distribution (the cube is placed
    uniformly in `blocks_init_region` with a random yaw, and the robot base is jittered
    around the origin), so sampling a task means drawing a seed and letting KINDER's
    sampler do the placing, rather than inventing a second distribution on top of it.

    The initial state is *read back* from a real reset rather than constructed. That
    matters: `Problem.reset_to_task` reinstalls a task by re-running its seed, so an
    analytically-built `initial_state` would differ from the state the episode actually
    starts in and every step-0 goal check would be measuring the wrong thing.

    Train and test share the sampling distribution, differing only in RNG stream
    (predicators' test_env_seed_offset convention, mirrored from Light Switch and
    Tossing Room). Both streams draw KINDER seeds from disjoint integer ranges so a
    test episode can never be a train episode the Method has already practiced on.

    Every task carries the same goal -- `InGoalRegion(cube_0, blocks_goal_region)`,
    KINDER's own `goal_state` for this variant. There is deliberately no goal-family
    mix here the way Tossing Room has one: the benchmark defines exactly one goal.

    env must be the same Tossing3DEnvironment instance the surrounding Problem drives:
    sampling a task drives that instance's simulator through a real reset.
    """

    env: Tossing3DEnvironment
    seed: int = 0
    test_env_seed_offset: int = 10000
    # KINDER reset seeds are drawn uniformly from [0, seed_space). Disjoint train/test
    # blocks: train draws from [0, seed_space), test from [seed_space, 2 * seed_space).
    seed_space: int = 1_000_000

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
        """Reset seed and rederive both RNG streams together -- the single entry point
        for reseeding, mirroring Light Switch's and Tossing Room's Tasks."""
        self.seed = seed
        self._train_rng = np.random.default_rng(self.seed)
        self._test_rng = np.random.default_rng(self.seed + self.test_env_seed_offset)

    def sample_train_task(self) -> Task:
        return self.build_task(kinder_seed=int(self.train_rng.integers(0, self.seed_space)))

    def sample_test_task(self) -> Task:
        return self.build_task(
            kinder_seed=int(self.test_rng.integers(self.seed_space, 2 * self.seed_space))
        )

    def build_task(self, *, kinder_seed: int) -> Task:
        initial_state = self.env.reset_to_seed(seed=kinder_seed)
        atom = IN_GOAL_REGION(state=initial_state, objects=(self.env.cube, self.env.goal_region))
        return Task(initial_state=initial_state, goal=Goal(atoms=frozenset({atom})))
