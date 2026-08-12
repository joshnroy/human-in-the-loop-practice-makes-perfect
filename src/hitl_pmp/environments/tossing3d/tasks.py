"""Task generation for Tossing3D.

One goal family, and it is upstream's: `["on", "cube_0", "blocks_goal_region"]`, which
`_check_goals()` evaluates as containment in a ground region. `predicates.IN_BIN`
reproduces that check exactly (see its docstring), so a `Task` here asks for precisely
what KINDER scores.

The goal atom names the **bin**, not the region, because this domain assumes the bin's
interior *is* that region -- the box `IN_BIN` tests against is still the live
`blocks_goal_region` bbox, carried in the `State` on the bin object. See `predicates.py`'s
module docstring for the assumption and for the task config under which it is false.

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
  them -- which `Problem.run_task_episode` and `PracticeLoop`'s scheduled arm both do.

  **A caller that does *not* reset afterwards must not call it at all.** That was never
  a property of `sample_train_task`; it held only because every caller happened to reset
  next, until `practice_reset_policy=never` stopped doing so and got a reset-free arm
  that rebuilt the scene every cycle while reporting `num_practice_resets == 0`.
  `sample_train_task_in_place` below is the path for those callers, and
  `PracticeLoop._sample_practice_task` checks the guarantee rather than trusting it.
- **The train/test split is a split of scene seeds**, drawn from two RNG streams derived
  from `seed` exactly as Light Switch and Tossing Room derive theirs. A test task is a
  scene the method never practiced on, not a different goal.
"""

import numpy as np
from pydantic import PrivateAttr

from hitl_pmp.core.problem.tasks.tasks import Tasks
from hitl_pmp.core.problem.tasks.types import Goal, Task

from .environment import Tossing3DEnvironment
from .predicates import IN_BIN

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

    def sample_train_task_in_place(self) -> Task:
        """The training task for the scene the robot is already standing in.

        No simulator operation at all, which is the entire point: this is what
        `PracticeLoop` calls under `practice_reset_policy=never`, where nothing is
        allowed to move the practice environment. `sample_train_task` cannot serve
        there, because on this domain the only way to obtain an initial `State` is to
        rebuild the scene -- so it re-seeded the live simulator every cycle while the
        loop, having declined to install the resulting state, recorded zero resets. See
        `Tasks.sample_train_task_in_place` for the measurement that exposed it.

        Two facts make the substitution sound rather than a convenient approximation.
        `Predicate.__call__` (`core/problem/tasks/types.py`) discards the `state` it is
        handed and returns a `GroundAtom` over the objects alone, and this domain has a
        single goal family over `ClassVar` objects -- so the goal below is exactly the
        goal `build_task` would have produced for any scene, and a `Task` needs a
        `State` only to satisfy the type.

        The consequence is real and intended: a reset-free run practices in **one
        scene** for its whole length, whatever `hard_reset` left behind
        (`canonical_seed`). On this domain handing the robot a new scene and resetting
        it are the same physical act, so a reset-free arm cannot have scene variety --
        which mirrors a real robot, standing in one room, rather than a limitation to
        engineer around. The train seed stream is therefore not drawn from at all under
        this policy.
        """
        state = self.env.get_current_state()
        return Task(
            initial_state=state,
            goal=Goal(
                atoms=frozenset({IN_BIN(state=state, objects=(self.env.cube, self.env.bin))})
            ),
        )

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
                    IN_BIN(state=initial_state, objects=(self.env.cube, self.env.bin))
                })
            ),
        )

    @staticmethod
    def draw_scene_seed(*, rng: np.random.Generator) -> int:
        return int(rng.integers(0, SCENE_SEED_LIMIT))

    def _make_rng(self, *, offset: int) -> np.random.Generator:
        return np.random.default_rng(self.seed + offset)
