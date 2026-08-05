"""Task generation for Tossing3D.

One goal family, and it is upstream's: `["on", "cube_0", "blocks_goal_region"]`, which
`_check_goals()` evaluates as containment in a ground region. `predicates.IN_GOAL_REGION`
reproduces that check exactly (see its docstring), so a `Task` here asks for precisely
what KINDER scores.

## A task is a scene seed, not an arithmetic construction

Every other domain in this repo builds an initial `State` by writing numbers into a
feature vector. This one cannot: an initial state is whatever MuJoCo settles into after
`env.reset(seed=...)` places the cube, bin, barrier and robot from their sampling regions
and lets the physics settle. So `sample_train_task`/`sample_test_task` draw a *seed*, run
a real reset, and read the resulting state back.

Two consequences worth stating plainly, because neither is true of the other domains:

- **Sampling a task is expensive and has side effects.** It rebuilds the scene, so it
  leaves the live simulator sitting at the sampled task. Callers that sample several
  tasks up front get the last one's scene, and must `reset_to_task` before running any of
  them -- which `PracticeLoop` and `Problem.run_task_episode` both already do.
- **The train/test split is a split of scene seeds**, drawn from two RNG streams derived
  from `seed` exactly as Light Switch and Tossing Room derive theirs. A test task is a
  scene the method never practiced on, not a different goal.
"""

import numpy as np
from pydantic import PrivateAttr

from hitl_pmp.core.problem.tasks.tasks import Tasks
from hitl_pmp.core.problem.tasks.types import Goal, Task

from .environment import Tossing3DEnvironment
from .predicates import IN_GOAL_REGION

# Scene seeds are drawn from [0, SCENE_SEED_LIMIT). Gymnasium seeds are non-negative
# ints; the bound is 2**31 - 1 rather than 2**63 purely so a seed printed in a log or a
# filename stays readable.
SCENE_SEED_LIMIT = 2**31 - 1


class Tossing3DTasks(Tasks):
    """Draws Tossing3D scenes, split into a train and a test stream by seed."""

    env: Tossing3DEnvironment
    seed: int = 0
    # Same convention and same default as Light Switch and Tossing Room: the test stream
    # is `seed + offset`, so two runs at different `--seed` share no scenes and the split
    # is reproducible from the seed alone.
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
        self.seed = seed
        self._train_rng = self._make_rng(offset=0)
        self._test_rng = self._make_rng(offset=self.test_env_seed_offset)

    def sample_train_task(self) -> Task:
        return self.build_task(scene_seed=self.draw_scene_seed(rng=self._train_rng))

    def sample_test_task(self) -> Task:
        """A held-out scene the method never practiced on."""
        return self.build_task(scene_seed=self.draw_scene_seed(rng=self._test_rng))

    def build_task(self, *, scene_seed: int) -> Task:
        """The task for one specific scene seed.

        Public and seed-addressed so a demo or a fidelity test can pin the exact scene
        every measured number in this domain's docs was taken at
        (`Tossing3DEnvironment.canonical_seed`, upstream's own 125) rather than
        rediscovering it through an RNG stream.
        """
        initial_state = self.env.reset_to_seed(seed=scene_seed)
        return Task(
            initial_state=initial_state,
            goal=Goal(
                atoms=frozenset({
                    IN_GOAL_REGION(
                        state=initial_state,
                        objects=(self.env.cube, self.env.goal_region),
                    )
                })
            ),
        )

    @staticmethod
    def draw_scene_seed(*, rng: np.random.Generator) -> int:
        return int(rng.integers(0, SCENE_SEED_LIMIT))

    def _make_rng(self, *, offset: int) -> np.random.Generator:
        return np.random.default_rng(self.seed + offset)
