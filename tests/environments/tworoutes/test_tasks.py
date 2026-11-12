import pytest

from hitl_pmp.environments.tworoutes.environment import TwoRoutesEnvironment
from hitl_pmp.environments.tworoutes.tasks import TwoRoutesTasks


def test_every_task_starts_ready_and_asks_only_to_be_done() -> None:
    env = TwoRoutesEnvironment()
    tasks = TwoRoutesTasks(env=env)
    for task in (
        tasks.sample_train_task(),
        tasks.sample_test_task(),
        tasks.sample_reset_start_task(),
    ):
        assert task.goal.describe() == "Done(robot)"
        assert task.initial_state.get(obj=env.robot, feature_name="ready") == 1.0
        assert not task.goal.is_satisfied(state=task.initial_state)


@pytest.mark.parametrize("via", ["via_a", "via_b"])
def test_either_route_satisfies_the_goal(*, via: str) -> None:
    env = TwoRoutesEnvironment()
    goal = TwoRoutesTasks(env=env).goal()
    assert goal.is_satisfied(state=env.build_state(done=True, **{via: True}))


def test_an_in_place_task_starts_from_the_scene_as_it_is() -> None:
    env = TwoRoutesEnvironment()
    env.set_state(state=env.build_state(after_b=True))
    task = TwoRoutesTasks(env=env).sample_train_task_in_place()
    assert task.initial_state.get(obj=env.robot, feature_name="after_b") == 1.0
