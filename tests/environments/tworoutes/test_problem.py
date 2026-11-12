import numpy as np

from hitl_pmp.core.method.types import LabeledAction
from hitl_pmp.environments.tworoutes.environment import TwoRoutesEnvironment
from hitl_pmp.environments.tworoutes.problem import TwoRoutesProblem
from hitl_pmp.environments.tworoutes.tasks import TwoRoutesTasks
from hitl_pmp.humans.oracle import UnconditionalHumanOracle


def problem(**kwargs: float) -> TwoRoutesProblem:
    env = TwoRoutesEnvironment(competence_cap=1.0, **kwargs)
    return TwoRoutesProblem(env=env, tasks=TwoRoutesTasks(env=env), human=UnconditionalHumanOracle)


def test_an_episode_is_one_attempt() -> None:
    failing = problem(route_a_start=0.0, route_a_rate=0.0)
    calls: list[int] = []

    def policy(state: object) -> LabeledAction:  # noqa: PLR0917 -- the Policy signature
        calls.append(1)
        return LabeledAction(action=failing.env.action_for(skill="RouteA"), label="RouteA")

    solved, frames, trace = failing.run_task_episode(task=failing.sample_test_task(), policy=policy)
    assert (solved, frames, len(calls), len(trace.actions)) == (False, [], 1, 1)

    sure = problem(route_b_start=1.0)
    solved, _, trace = sure.run_task_episode(
        task=sure.sample_test_task(),
        policy=lambda state: LabeledAction(
            action=sure.env.action_for(skill="RouteB"), label="RouteB"
        ),
    )
    assert solved and np.array_equal(trace.actions[0].action, sure.env.action_for(skill="RouteB"))


def test_the_human_reset_goes_through_the_problem() -> None:
    scene = problem(route_b_start=1.0)
    scene.hard_reset()
    scene.take_action(action=scene.env.action_for(skill="RouteB"))
    scene.execute_movables_reset(destination="human_reset_b")
    assert scene.env.flag(name="ready")
