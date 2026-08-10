import math

import numpy as np
import pytest

from hitl_pmp.core.problem.environment.environment import Environment
from hitl_pmp.core.problem.environment.types import Action, Object, State, Type
from hitl_pmp.core.problem.human.types import CommandGoalDescription, CommandStartStateDescription
from hitl_pmp.core.problem.tasks.types import Goal, GroundAtom, Predicate
from hitl_pmp.humans.oracle import UnconditionalHumanOracle

# A schema no domain in this repo uses, deliberately: the oracle is domain-agnostic, so
# the test should not be able to lean on any real domain's layout.
_WIDGET = Type(name="widget", feature_names=("a", "b", "c"))
_OBJ = Object(name="widget1", type=_WIDGET)


def _state(*, a: float) -> State:
    return State(data={_OBJ: np.array([a, 0.0, 0.0])})


class _Env(Environment):
    action_space = None  # type: ignore[assignment]

    def take_action(self, *, action: Action) -> State:
        del action
        return self.get_current_state()

    def get_valid_actions(self) -> list[Action]:
        return []

    def noop_action(self) -> Action:
        return np.zeros(1)

    def hard_reset(self) -> None:
        self.set_state(state=_state(a=0.0))


def _exploding_goal() -> Goal:
    """A Goal whose classifier raises if anything ever evaluates it.

    This is how "unconditional" is asserted rather than described: a v0 human performs no
    feasibility check at all, so neither method may consult the goal."""

    def holds(state: State, objects: tuple[Object, ...]) -> bool:  # noqa: PLR0917 (Predicate.holds is positional)
        del state, objects
        raise AssertionError("the v0 oracle must not evaluate the goal")

    predicate = Predicate(name="Boom", types=(_WIDGET,), holds=holds)
    return Goal(atoms=frozenset({GroundAtom(predicate=predicate, objects=(_OBJ,))}))


def _command(*, target_state: State | None) -> CommandGoalDescription:
    return CommandGoalDescription(goal=_exploding_goal(), target_state=target_state)


def test_cost_is_the_unit_cost_when_a_target_state_is_given() -> None:
    cost = UnconditionalHumanOracle.calculate_cost_for_human_command(
        command_start_state_description=CommandStartStateDescription(state=_state(a=1.0)),
        command_goal_description=_command(target_state=_state(a=2.0)),
    )
    assert cost == UnconditionalHumanOracle.intervention_cost


def test_cost_is_infinite_without_a_target_state() -> None:
    """`Cost` is documented as inf when infeasible, and a v0 human cannot bring about a
    symbolic goal -- it can only put the world where it is told."""
    cost = UnconditionalHumanOracle.calculate_cost_for_human_command(
        command_start_state_description=CommandStartStateDescription(state=_state(a=1.0)),
        command_goal_description=_command(target_state=None),
    )
    assert math.isinf(cost)


def test_cost_does_not_depend_on_the_start_state() -> None:
    """Unconditional: every command costs the same however far the world has drifted."""
    costs = {
        UnconditionalHumanOracle.calculate_cost_for_human_command(
            command_start_state_description=CommandStartStateDescription(state=_state(a=a)),
            command_goal_description=_command(target_state=_state(a=99.0)),
        )
        for a in (0.0, 1.0, 1000.0)
    }
    assert costs == {UnconditionalHumanOracle.intervention_cost}


def test_execute_installs_the_target_state() -> None:
    env = _Env()
    env.hard_reset()
    UnconditionalHumanOracle.execute_human_command(
        command_start_state_description=CommandStartStateDescription(state=env.get_current_state()),
        command_goal_description=_command(target_state=_state(a=42.0)),
        env=env,
    )
    assert env.get_current_state()[_OBJ].tolist() == [42.0, 0.0, 0.0]


def test_execute_raises_without_a_target_state() -> None:
    """Loud rather than a silent no-op: the caller has already been charged a cost by
    the time this runs, so doing nothing would bill an intervention that never happened."""
    env = _Env()
    env.hard_reset()
    with pytest.raises(ValueError, match="target_state"):
        UnconditionalHumanOracle.execute_human_command(
            command_start_state_description=CommandStartStateDescription(
                state=env.get_current_state()
            ),
            command_goal_description=_command(target_state=None),
            env=env,
        )


def test_execute_does_not_alias_the_target_state() -> None:
    """The target is usually a `Task.initial_state`, and the fixed test set replays the
    same `Task` objects at every checkpoint -- handing the environment the very array a
    Task holds would let one intervention rewrite that task for the rest of the run."""
    env = _Env()
    env.hard_reset()
    target = _state(a=5.0)
    UnconditionalHumanOracle.execute_human_command(
        command_start_state_description=CommandStartStateDescription(state=env.get_current_state()),
        command_goal_description=_command(target_state=target),
        env=env,
    )
    env.get_current_state().set(obj=_OBJ, feature_name="a", feature_val=6.0)
    assert target[_OBJ].tolist() == [5.0, 0.0, 0.0]


def test_execute_returns_nothing() -> None:
    env = _Env()
    env.hard_reset()
    result = UnconditionalHumanOracle.execute_human_command(
        command_start_state_description=CommandStartStateDescription(state=env.get_current_state()),
        command_goal_description=_command(target_state=_state(a=1.0)),
        env=env,
    )
    assert result is None
