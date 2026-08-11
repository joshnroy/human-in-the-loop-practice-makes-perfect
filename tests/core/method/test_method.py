from typing import Any

import numpy as np

from hitl_pmp.core.method.method import HumanHelpRequested, InteractionComplete, Method
from hitl_pmp.core.method.types import (
    GroundSkill,
    LabeledAction,
    Policy,
    Rollout,
    SetupCommand,
)
from hitl_pmp.core.problem.environment.environment import Environment
from hitl_pmp.core.problem.environment.types import Action, Object, State, Type
from hitl_pmp.core.problem.tasks.types import Goal, Task

_BLOCK = Type(name="block", feature_names=("x",))
_OBJ = Object(name="block1", type=_BLOCK)


class _Env(Environment):
    def take_action(self, *, action: Action) -> State:
        raise NotImplementedError

    def get_valid_actions(self) -> list[Action]:
        raise NotImplementedError

    def noop_action(self) -> Action:
        raise NotImplementedError

    def hard_reset(self) -> None:
        raise NotImplementedError


class _MinimalMethod(Method):
    """Implements only the six genuinely abstract methods -- deliberately does NOT
    override get_practice_policy/end_cycle, so this pins their concrete defaults."""

    def reset_environment(self, *, start_state: State) -> bool:
        raise NotImplementedError

    def get_task_policy(self, *, task: Task) -> Policy:
        del task
        return lambda state: LabeledAction(action=np.array([0.0]), label="minimal")

    def generate_train_task(self, *, tbd_inputs: Any) -> Task:
        raise NotImplementedError

    def execute_setup_command(self, *, setup_command: SetupCommand) -> None:
        raise NotImplementedError

    def execute_skill(self, *, skill: GroundSkill) -> Rollout:
        raise NotImplementedError

    def improve_skill_parameters(self, *, skill: GroundSkill, rollout: Rollout) -> None:
        raise NotImplementedError


def _task() -> Task:
    return Task(initial_state=State(data={_OBJ: np.array([0.0])}), goal=Goal(atoms=frozenset()))


def test_method_declares_expected_abstract_methods() -> None:
    """get_practice_policy/end_cycle are deliberately NOT in this set: they're
    concrete defaults, so a Method that doesn't practice (every baseline built so
    far) needs no boilerplate to satisfy them."""
    assert Method.__abstractmethods__ == frozenset({
        "reset_environment",
        "get_task_policy",
        "generate_train_task",
        "execute_setup_command",
        "execute_skill",
        "improve_skill_parameters",
    })


def test_get_practice_policy_defaults_to_get_task_policy() -> None:
    method = _MinimalMethod(env=_Env())
    task = _task()
    state = task.initial_state
    practice = method.get_practice_policy(task=task)
    evaluation = method.get_task_policy(task=task)
    assert practice(state).label == evaluation(state).label


def test_end_cycle_defaults_to_a_no_op() -> None:
    """A non-learning Method has nothing to retrain, so the default must be safe to
    call unconditionally from PracticeLoop -- it just has to not raise."""
    method = _MinimalMethod(env=_Env())
    method.end_cycle()


def test_a_method_can_override_practice_policy_independently_of_task_policy() -> None:
    """The whole point of the split: a learning Method explores during practice but
    exploits during evaluation, without the two codepaths being conflated."""

    class _SplitMethod(_MinimalMethod):
        def get_practice_policy(self, *, task: Task) -> Policy:
            del task
            return lambda state: LabeledAction(action=np.array([1.0]), label="practicing")

    method = _SplitMethod(env=_Env())
    task = _task()
    state = task.initial_state
    assert method.get_practice_policy(task=task)(state).label == "practicing"
    assert method.get_task_policy(task=task)(state).label == "minimal"


def test_planning_outcomes_default_to_no_failures_out_of_no_attempts() -> None:
    """Concrete, not abstract: a Method with no planner has nothing to report, and
    every non-planning baseline should need no boilerplate to say so. A pair rather
    than a bare count, so a failure number can never be reported without the
    denominator that makes it readable."""
    assert _MinimalMethod(env=_Env()).planning_outcomes() == (0, 0)


def test_practice_outcomes_default_to_nothing_recorded() -> None:
    """Empty, not "one all-zero entry per skill": a Method that never scores its own
    skill executions (every non-learning baseline) has no attempts to report, and an
    all-zero entry would claim the skill was practiced and never succeeded. `{}` says
    "this Method does not measure that"; a present entry says "it did, and this is
    what happened"."""
    assert _MinimalMethod(env=_Env()).practice_outcomes() == {}


def test_current_competences_default_to_nothing_tracked() -> None:
    """`{}`, not an all-zero entry per ground skill it might one day see: a Method
    that tracks no competence model (every non-learning baseline) has nothing to
    report, and the default must need no boilerplate to say so -- the same contract
    practice_outcomes/planning_outcomes already hold."""
    assert _MinimalMethod(env=_Env()).current_competences() == {}


def test_human_help_requested_is_not_an_interaction_complete() -> None:
    """The two signals must stay tellable apart by `except`, in both directions.

    `InteractionComplete` means "no ground skill is applicable, so the period ends";
    `HumanHelpRequested` means "I am still able to act and getting nowhere, so please
    reposition me, and the period carries on". Making either a subclass of the other --
    or reusing one for both -- would silently give every arm that catches one the
    behaviour of the other, and `InteractionComplete`'s meaning is EES-wide with
    already-merged results resting on it."""
    assert not issubclass(HumanHelpRequested, InteractionComplete)
    assert not issubclass(InteractionComplete, HumanHelpRequested)
    assert issubclass(HumanHelpRequested, Exception)


def test_a_method_declares_it_cannot_ask_for_human_help_by_default() -> None:
    """The harness reads this exactly once, up front, to decide whether a missing
    `Problem.human` is a fatal misconfiguration. Defaulting to False is what keeps every
    existing Method needing no boilerplate."""
    assert _MinimalMethod(env=_Env()).may_request_human_help() is False


def test_observe_help_granted_defaults_to_a_no_op() -> None:
    """A Method that never asks is never told, but the hook has to be safe to call
    unconditionally from the loop -- the same contract `end_cycle` has."""
    _MinimalMethod(env=_Env()).observe_help_granted(state=_task().initial_state)
