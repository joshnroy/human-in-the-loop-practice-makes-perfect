import numpy as np

from hitl_pmp.core.method.types import Policy
from hitl_pmp.core.problem.problem import Problem
from hitl_pmp.core.problem.tasks.types import Task
from hitl_pmp.core.renderer.renderer import Renderer

from .environment import TossingRoomEnvironment
from .tasks import TossingRoomTasks


class TossingRoomProblem(Problem):
    """`TossingRoomProblem` verbatim, retyped to this domain: no HumanOracle is set
    (`Problem.human` stays None), since the irreversible ledge exists in the dynamics
    but the oracle solves every task forward-only and never needs to be lifted back up
    it. env/tasks are required constructor fields, narrowed to this domain's own
    concrete types."""

    env: TossingRoomEnvironment
    tasks: TossingRoomTasks

    def max_episode_steps(self) -> int:
        """The paper's H_eval convention (Appendix F): the **longest shortest solve
        this layout admits, plus exactly two spare actions**. Computed fresh each call
        so overridden layout config is respected.

        Two spare actions, not a generous bound, because a throw is the domain's one
        *stochastic* skill and every spare step used to be another free retry of it --
        which silently turned the evaluation horizon into a "number of attempts" dial.
        That channel is closed anyway now that a missed throw releases the item, but the
        horizon stays tight so it cannot reopen. Ported from `TossingRoomProblem`, whose
        docstring carries the full measurement history.

        The default layout's horizon moved 7 -> 12 when each bin got its own button:
        EMPTY became the longest solve (see `longest_shortest_solve`). That does NOT
        reopen the retry channel the tight bound exists to close -- a TRASH retry needs
        step 13 and RECYCLING can never retry at all.

        Note this bounds only the EVALUATION episode. An interaction period's length is
        `--max-steps-per-interaction`, untouched by this, so tightening the horizon
        cannot change how much practice experience any Method receives."""
        return self.longest_shortest_solve() + 2

    def longest_shortest_solve(self) -> int:
        """Skills in the longest of this layout's three goal families' shortest solves: a
        throw goal is a pickup + walk to that item's bin room + a throw, and the empty
        goal is a walk to each bin's own button plus a press at each. Targets the one-way
        ledge makes unreachable are skipped -- they are unsolvable at *any* horizon, so
        letting them set the budget would only hand the solvable goals extra retries.

        On the default layout EMPTY is now the longest (10: 3 moves, a press, 5 moves, a
        press), where it used to be 4 with one button emptying both bins -- so the horizon
        is 12 rather than 7. TRASH's own solve is unchanged at 5 and still admits no
        second attempt: a retry costs the eight-step round trip to the pile, landing at
        step 13. RECYCLING can never be retried at any horizon, since the ledge severs the
        bin room from the pile.

        **Under `--two-way-ledge` EMPTY's solve is 9, not 10, and the horizon is 11.**
        The reverse order (recycling first: 2 moves, a press, 5 moves, a press) becomes
        feasible and is cheaper, so `empty_both_bins_solve` returns it. That is a real
        change in the domain's difficulty and must be stated wherever a two-way number is
        put beside a one-way one -- it is not a method effect. It does not reopen the
        retry channel the tight bound exists to close: a TRASH retry still lands at step
        13, past 11 as it was past 12. The two throw families' own solves (4 and 5) are
        unchanged either way, since neither walk crosses the ledge rightward."""
        lengths: list[int] = []
        for bin_room in (self.env.recycling_bin_room, self.env.trash_bin_room):
            distance = self.rooms_to_walk(room=bin_room)
            if distance is not None:
                lengths.append(1 + distance + 1)
        empty = self.empty_both_bins_solve()
        if empty is not None:
            lengths.append(empty)
        # default=1 only for a degenerate layout with nothing reachable at all; the
        # horizon still has to be positive for run_task_episode's goal check to run.
        return max(lengths, default=1)

    def empty_both_bins_solve(self) -> int | None:
        """MoveRooms + presses to empty BOTH bins, or None when no order of the two
        works. Each bin has its own button beside it, so the robot must visit both rooms,
        and the one-way ledge means the order matters: on the default layout only
        trash-then-recycling is feasible. Both orders are costed and the cheaper feasible
        one wins, rather than hardcoding "rightmost first" -- the ledge's direction is
        layout config.

        Costing both orders is what makes `--two-way-ledge` correct here for free: with
        no blocked edge BOTH orders are feasible, and the cheaper (recycling first, 9 on
        the default layout) wins. A hardcoded "rightmost first" would have silently
        reported 10 for a world in which 9 is achievable, over-sizing the horizon."""
        candidates: list[int] = []
        for first, second in (
            (self.env.trash_bin_room, self.env.recycling_bin_room),
            (self.env.recycling_bin_room, self.env.trash_bin_room),
        ):
            to_first = self.rooms_to_walk(room=first)
            between = self.rooms_to_walk_between(from_room=first, to_room=second)
            if to_first is not None and between is not None:
                candidates.append(to_first + 1 + between + 1)
        return min(candidates, default=None)

    def rooms_to_walk(self, *, room: int) -> int | None:
        """MoveRoom steps from the start room to `room`, or None if the one-way ledge
        blocks it."""
        return self.rooms_to_walk_between(from_room=self.env.start_room, to_room=room)

    def rooms_to_walk_between(self, *, from_room: int, to_room: int) -> int | None:
        """MoveRoom steps between two rooms, or None if the one-way ledge blocks the
        walk. Closed form rather than a graph search: the rooms are a 1-D hallway whose
        single blocked edge is the rightward step out of `blocked_right_from`, so leftward
        is always free and rightward is free unless the walk crosses that edge.

        Under `--two-way-ledge` there is no blocked edge at all, so this never returns
        None and the hallway is an ordinary corridor. The feasibility question is asked
        of `Environment.ledge_blocks_rightward` rather than re-derived from
        `blocked_right_from` here, so the dynamics, the symbolic model and this horizon
        calculation cannot disagree about which edges exist."""
        if to_room <= from_room:
            return from_room - to_room
        if any(
            self.env.ledge_blocks_rightward(from_room=room) for room in range(from_room, to_room)
        ):
            return None
        return to_room - from_room

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
