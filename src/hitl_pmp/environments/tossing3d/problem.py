"""`Tossing3DProblem` -- the `core.Problem` facade over this domain."""

import numpy as np

from hitl_pmp.core.method.types import Policy
from hitl_pmp.core.problem.problem import Problem
from hitl_pmp.core.problem.tasks.types import Task
from hitl_pmp.core.renderer.renderer import Renderer

from .environment import Tossing3DEnvironment
from .tasks import Tossing3DTasks


class Tossing3DProblem(Problem):
    """Composition root for Tossing3D. `human` stays unset for now.

    That last point deserves a note rather than silence: this is the domain in this repo
    that most *wants* a `HumanOracle`, because a tossed cube is genuinely unretrievable
    and a real robot would need someone to walk over and pick it up. None is wired,
    so `Metrics.num_human_interventions()` reports `(0.0, 0)` -- not because no
    intervention was needed, but because none is representable yet.
    """

    env: Tossing3DEnvironment
    tasks: Tossing3DTasks

    def max_episode_steps(self) -> int:
        """The shortest solve plus two.

        The shortest solve is exactly three skills -- `Pick`, `MoveToThrowPose`, `Toss` --
        and there is no shorter route, since `Toss` requires `Holding` and `NearBin` and
        nothing else grants either. The `+ 2` is this repo's standing convention and is
        deliberately small here: a generous horizon in a domain whose skills are
        stochastic quietly becomes a retry dial, and in *this* domain a retry is not even
        available -- after a toss the cube is past the barrier, `Reachable` is false, and
        no further skill applies. So the extra budget buys one recovery from a failed
        grasp and nothing more, which is the honest amount.
        """
        return 3 + 2

    def run_task_episode(
        self, *, task: Task, policy: Policy, renderer: type[Renderer] | None = None
    ) -> tuple[bool, list[np.ndarray]]:
        state = self.reset_to_task(task=task)
        frames = [renderer.render_frame(state=state, env=self.env)] if renderer is not None else []
        for _ in range(self.max_episode_steps()):
            if task.goal.is_satisfied(state=state):
                return True, frames
            labeled_action = policy(state)
            state = self.env.take_action(action=labeled_action.action)
            if renderer is not None:
                frames.append(
                    renderer.render_frame(state=state, env=self.env, label=labeled_action.label)
                )
        return task.goal.is_satisfied(state=state), frames
