import numpy as np

from hitl_pmp.core.method.types import Policy
from hitl_pmp.core.problem.problem import Problem
from hitl_pmp.core.problem.tasks.types import Task
from hitl_pmp.core.renderer.renderer import Renderer

from .environment import Tossing3DEnvironment
from .tasks import Tossing3DTasks


class Tossing3DProblem(Problem):
    """No HumanOracle is set (`Problem.human` stays None), the same as Tossing Room and
    Light Switch. That is a statement about what has been *built*, not about the
    domain: unlike those two, Tossing3D genuinely needs one, because a tossed cube is
    unrecoverable and only a human could fetch it back. Until such a HumanOracle exists,
    the only thing that undoes a toss is `PracticeLoop`'s free per-period reset -- which
    is exactly the confound the experiment log for this domain spells out."""

    env: Tossing3DEnvironment
    tasks: Tossing3DTasks

    def max_episode_steps(self) -> int:
        """The paper's H_eval convention (Appendix F), the same one Light Switch and
        Tossing Room cite: the longest shortest solve this domain admits, plus exactly
        two spare actions.

        The shortest solve is three skills -- Pick, MoveToThrowPose, Toss -- so the
        horizon is 5. The two spare actions matter more here than the count does: Toss
        is the domain's stochastic skill, and a *missed* toss puts the cube beyond the
        barrier, so unlike Tossing Room a spare step buys no free retry at all. It only
        buys recovery from a failed *Pick*, whose inverse-kinematics failures are the
        other thing a learned sampler can fix.
        """
        return self.shortest_solve() + 2

    def shortest_solve(self) -> int:
        """Pick, MoveToThrowPose, Toss. A named method rather than a literal so the
        horizon and any future skill-set change stay in step."""
        return 3

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
