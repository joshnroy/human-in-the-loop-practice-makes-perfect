"""`Tossing3DProblem` -- the `core.Problem` facade over this domain."""

import numpy as np

from hitl_pmp.core.method.types import EpisodeTrace, LabeledAction, Policy
from hitl_pmp.core.problem.environment.types import State
from hitl_pmp.core.problem.problem import Problem
from hitl_pmp.core.problem.tasks.types import Task
from hitl_pmp.core.renderer.renderer import Renderer

from .environment import Tossing3DEnvironment
from .renderer import Tossing3DRenderer
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

        The shortest solve is exactly two skills -- `PickCube`, then
        `MoveToTossLocationAndToss` -- and there is no shorter route, since the toss
        requires `Holding` and nothing but the pick grants it. The `+ 2` is this repo's
        standing convention and is deliberately small here: a generous horizon in a domain
        whose skills are stochastic quietly becomes a retry dial, and in *this* domain a
        retry is not even available -- after a toss the cube is past the barrier,
        `Reachable` is false, and no further skill applies. So the extra budget buys one
        recovery from a failed grasp and nothing more, which is the honest amount.

        **This was `3 + 2` under the three-skill decomposition and is `2 + 2` now.** The
        shortest solve lost a step when upstream composed the base move into the toss, so
        leaving the literal at 5 would have kept a horizon one step more generous than the
        convention it claims -- and more generous than the horizon every published
        Tossing3D baseline was measured under, which would confound a comparison against
        them in the new domain's favour.
        """
        return 2 + 2

    def run_task_episode(
        self, *, task: Task, policy: Policy, renderer: type[Renderer] | None = None
    ) -> tuple[bool, list[np.ndarray], EpisodeTrace]:
        """Run one episode, optionally recording it at physics rate rather than per skill.

        The loop is the standard one. What is domain-specific is *how many frames a
        transition is worth*: one `take_action` here runs a whole KINDER controller, so
        rendering only at transition boundaries produced a three-frame `episode.mp4` of a
        domain whose entire point is a throw. When a renderer is given, the backend
        collects every physics tick (`gymnasium.wrappers.RenderCollection`; see
        `kinder_backend.py`) and each skill contributes that whole burst instead.

        **Recording is scoped to this call, and only when a renderer was passed.** A
        renderer is passed exactly on the `--output-dir` / `--num-render-checkpoints` /
        `--record-full-loop` demo paths, so a plain training run never pays for hundreds
        of MuJoCo renders per skill -- and the `finally` means an episode that raised
        cannot leave the next one recording.
        """
        backend = self.env.backend()
        backend.set_substep_recording(enabled=renderer is not None)
        try:
            state = self.reset_to_task(task=task)
            frames = (
                [renderer.render_frame(state=state, env=self.env)] if renderer is not None else []
            )
            # The reset collected the initial scene as well; the captioned frame above
            # already covers it, so start the first skill from an empty buffer.
            backend.drain_substep_frames()
            states = [state]
            actions: list[LabeledAction] = []
            for _ in range(self.max_episode_steps()):
                if task.goal.is_satisfied(state=state):
                    return True, frames, EpisodeTrace(states=states, actions=actions)
                labeled_action = policy(state)
                state = self.env.take_action(action=labeled_action.action)
                actions.append(labeled_action)
                states.append(state)
                if renderer is not None:
                    frames.extend(
                        self._skill_frames(
                            renderer=renderer, state=state, label=labeled_action.label
                        )
                    )
            return (
                task.goal.is_satisfied(state=state),
                frames,
                EpisodeTrace(states=states, actions=actions),
            )
        finally:
            backend.set_substep_recording(enabled=False)

    def _skill_frames(
        self, *, renderer: type[Renderer], state: State, label: str
    ) -> list[np.ndarray]:
        """The frames one skill contributes: its physics ticks, or one boundary frame.

        The fallback is not defensive padding -- a controller whose motion planning fails
        never steps the simulator, so there is genuinely nothing to collect, and the
        storyboard frame is what keeps that skill visible in the clip at all.
        """
        assert issubclass(renderer, Tossing3DRenderer), (
            f"Tossing3DProblem records through Tossing3DRenderer, got {renderer.__name__}"
        )
        substeps = self.env.backend().drain_substep_frames()
        if not substeps:
            return [renderer.render_frame(state=state, env=self.env, label=label)]
        return renderer.render_substep_frames(
            frames=substeps, state=state, env=self.env, label=label
        )
