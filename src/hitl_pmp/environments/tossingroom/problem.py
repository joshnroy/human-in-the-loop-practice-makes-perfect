import numpy as np

from hitl_pmp.core.method.types import Policy
from hitl_pmp.core.problem.problem import Problem
from hitl_pmp.core.problem.tasks.types import Task
from hitl_pmp.core.renderer.renderer import Renderer

from .environment import TossingRoomEnvironment
from .tasks import TossingRoomTasks


class TossingRoomProblem(Problem):
    """No HumanOracle is set (Problem.human stays None): the irreversible ledge exists
    in the dynamics, but the oracle solves every task forward-only and never needs to
    be lifted back up it, so run_task_episode never needs Problem.execute_human_command
    -- mirrors LightSwitchProblem. env/tasks are required constructor fields, narrowed
    to this domain's own concrete types."""

    env: TossingRoomEnvironment
    tasks: TossingRoomTasks

    def max_episode_steps(self) -> int:
        """The paper's H_eval convention (Appendix F), the same one
        LightSwitchProblem cites: the **longest shortest solve this layout admits,
        plus exactly two spare actions**. Computed fresh each call so overridden
        layout config is respected.

        This was `2 * num_rooms + 2` -- self-described as "a generous bound". The
        generosity is not free, because `Throw` is the domain's one *stochastic*
        skill: a failed throw leaves the robot holding the item in the bin room, so
        the very next step replans to `Throw` again. Every spare step is therefore
        another free draw at the ~0.19-probability window a uniformly random force
        lands in, and the horizon silently sets how many draws the evaluation grants.
        At `2 * num_rooms + 2 = 16` an *unpracticed* EES scored 94.7% purely by
        retrying, versus 62.7% at 7 (measured over 300 episodes, 10 seeds x 30 test
        tasks: 1133 Throw actions against 648, up to 13 in a single episode, with the
        non-stochastic skill counts identical at both horizons). 94.7% leaves a
        learned sampler no headroom at all to demonstrate anything.

        Those five figures are the *original* motivating measurement and are quoted as
        history, not as a current result: they predate both the change that makes a
        missed Throw release the item and the fixed 14/14/2 test-set composition, so
        neither reproduces against today's code. Re-measured against it, an unpracticed
        EES scores 24.0% and the horizon is *flat* from 5 through 12 -- which is the
        stronger form of the same argument, since the retry channel the generous bound
        was feeding is now closed outright. See
        docs/experiment-logs/2026-08-02-tossingroom-ees-bringup.md.
        Two spare actions is what Light Switch's `grid_size + 2` grants -- its solve
        is `grid_size - 1` moves plus one toggle -- so this ports the *spare budget*,
        which is the load-bearing quantity, rather than the coincidental room count.

        Note this bounds only the EVALUATION episode. An interaction period's length
        is `--max-steps-per-interaction`, untouched by this, so tightening the horizon
        cannot change how much practice experience any Method receives.
        """
        return self.longest_shortest_solve() + 2

    def longest_shortest_solve(self) -> int:
        """Skills in the longest of this layout's three goal families' shortest
        solves: a throw goal is Pickup + walk to that item's bin room + Throw, and the
        empty goal is walk to the button room + Press. Targets the one-way ledge makes
        unreachable are skipped -- they are unsolvable at *any* horizon, so letting
        them set the budget would only hand the solvable goals extra retries."""
        lengths: list[int] = []
        for bin_room in (self.env.recycling_bin_room, self.env.trash_bin_room):
            distance = self.rooms_to_walk(room=bin_room)
            if distance is not None:
                lengths.append(1 + distance + 1)
        distance = self.rooms_to_walk(room=self.env.button_room)
        if distance is not None:
            lengths.append(distance + 1)
        # default=1 only for a degenerate layout with nothing reachable at all; the
        # horizon still has to be positive for run_task_episode's goal check to run.
        return max(lengths, default=1)

    def rooms_to_walk(self, *, room: int) -> int | None:
        """MoveRoom steps from the start room to `room`, or None if the one-way ledge
        blocks it. Closed form rather than a graph search: the rooms are a 1-D hallway
        whose single blocked edge is the rightward step out of `blocked_right_from`,
        so leftward is always free and rightward is free unless the walk crosses that
        edge."""
        start = self.env.start_room
        if room <= start:
            return start - room
        if start <= self.env.blocked_right_from < room:
            return None
        return room - start

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
