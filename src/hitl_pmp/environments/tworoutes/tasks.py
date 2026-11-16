from hitl_pmp.core.problem.tasks.tasks import Tasks
from hitl_pmp.core.problem.tasks.types import Goal, GroundAtom, Task

from .environment import TwoRoutesEnvironment
from .predicates import TwoRoutesPredicates


class TwoRoutesTasks(Tasks):
    """One task: start ready, be done. The goal names no route, so either satisfies
    it; which one did is in the state (`ViaA`/`ViaB`)."""

    env: TwoRoutesEnvironment

    def goal(self) -> Goal:
        return Goal(
            atoms=frozenset({
                GroundAtom(
                    predicate=TwoRoutesPredicates.get(name="Done"), objects=(self.env.robot,)
                )
            })
        )

    def build_task(self) -> Task:
        return Task(initial_state=self.env.build_state(ready=True), goal=self.goal())

    def sample_train_task(self) -> Task:
        return self.build_task()

    def sample_test_task(self) -> Task:
        return self.build_task()

    def sample_reset_start_task(self) -> Task:
        return self.build_task()

    def sample_train_task_in_place(self) -> Task:
        return Task(initial_state=self.env.get_current_state(), goal=self.goal())
