import numpy as np
import pytest

from hitl_pmp.core.method.method import HumanHelpRequested, InteractionComplete
from hitl_pmp.core.method.skill_provider import SkillProvider
from hitl_pmp.core.method.types import (
    GroundSkill,
    LabeledAction,
    LiftedAtom,
    Policy,
    Skill,
    Variable,
)
from hitl_pmp.core.problem.environment.types import Action, Object, State, Type
from hitl_pmp.core.problem.tasks.types import Predicate
from hitl_pmp.methods.help_seeking import HelpSeekingPolicy, HelpSeekingTrigger

_BLOCK = Type(name="block", feature_names=("x",))
_OBJ = Object(name="thing", type=_BLOCK)


def _state(*, x: float) -> State:
    return State(data={_OBJ: np.array([x])})


def _acting_policy(*, label: str = "acted") -> Policy:
    return lambda state: LabeledAction(action=np.array([0.0]), label=label)


def _on_stuck(*, patience: int = 3, **kwargs) -> HelpSeekingPolicy:
    return HelpSeekingPolicy(trigger=HelpSeekingTrigger.ON_STUCK, stuck_patience=patience, **kwargs)


def _at_random(*, mean_steps: int = 4, seed: int = 0) -> HelpSeekingPolicy:
    return HelpSeekingPolicy(
        trigger=HelpSeekingTrigger.AT_RANDOM,
        mean_steps_between_requests=mean_steps,
        seed=seed,
    )


# --- A minimal SkillProvider double for ON_NO_APPLICABLE_SKILL, independent of any
# real domain: one skill, gated on one feature of one object, so a test can flip
# "applicable" on and off just by choosing x. `Flagged` holds iff x >= 1.0.
_FLAG_VAR = Variable(name="obj", type=_BLOCK)
_FLAGGED = Predicate(
    name="Flagged",
    types=(_BLOCK,),
    holds=lambda state, objects: bool(state.get(obj=objects[0], feature_name="x") >= 1.0),
)
_TOGGLE = Skill(
    name="Toggle",
    parameters=(_FLAG_VAR,),
    preconditions=frozenset({LiftedAtom(predicate=_FLAGGED, variables=(_FLAG_VAR,))}),
    add_effects=frozenset(),
    delete_effects=frozenset(),
    param_dim=0,
)


class _FakeSkillProvider(SkillProvider):
    def skills(self) -> tuple[Skill, ...]:
        return (_TOGGLE,)

    def predicates(self) -> tuple[Predicate, ...]:
        return (_FLAGGED,)

    def types(self) -> tuple[Type, ...]:
        return (_BLOCK,)

    def objects(self) -> tuple[Object, ...]:
        return (_OBJ,)

    def sample_params(self, *, ground_skill: GroundSkill, rng: np.random.Generator) -> np.ndarray:
        del ground_skill, rng
        return np.zeros(0)

    def compute_action(
        self, *, ground_skill: GroundSkill, params: np.ndarray, state: State
    ) -> Action:
        del ground_skill, params, state
        return np.zeros(1)


def _on_no_applicable_skill(*, seed: int = 0) -> HelpSeekingPolicy:
    return HelpSeekingPolicy(
        trigger=HelpSeekingTrigger.ON_NO_APPLICABLE_SKILL,
        skill_provider=_FakeSkillProvider(),
        seed=seed,
    )


def test_the_trigger_enum_renders_as_its_wire_words() -> None:
    """So argparse prints `on-stuck` in --help and in its error message for a bad
    value, and so the chosen value lands in config_snapshot.json as a readable word --
    the same reason PracticeResetPolicy does this."""
    assert str(HelpSeekingTrigger.NEVER) == "never"
    assert str(HelpSeekingTrigger.ON_STUCK) == "on-stuck"
    assert str(HelpSeekingTrigger.AT_RANDOM) == "at-random"
    assert str(HelpSeekingTrigger.ON_NO_APPLICABLE_SKILL) == "on-no-applicable-skill"


def test_a_wrapped_policy_delegates_while_the_robot_is_getting_somewhere() -> None:
    policy = _on_stuck(patience=2).wrap(inner_policy=_acting_policy())
    for x in range(10):
        assert policy(_state(x=float(x))).label == "acted"


def test_a_wrapped_policy_asks_for_help_once_it_stops_reaching_anywhere_new() -> None:
    policy = _on_stuck(patience=2).wrap(inner_policy=_acting_policy())
    assert policy(_state(x=1.0)).label == "acted"  # novel
    assert policy(_state(x=1.0)).label == "acted"  # first repeat
    with pytest.raises(HumanHelpRequested):
        policy(_state(x=1.0))  # second repeat reaches patience


def test_the_ask_happens_before_the_inner_policy_is_consulted() -> None:
    """Order is load-bearing twice over. It is what lets InteractionComplete propagate
    out of the inner policy exactly as it does with no wrapper at all -- the wrapper
    never swallows it and never turns it into a help request -- and it is what makes a
    request cost the step the inner policy would otherwise have spent."""
    calls = []

    def _inner(state: State) -> LabeledAction:  # noqa: PLR0917 (core.Policy is positional)
        calls.append(state)
        raise InteractionComplete

    policy = _on_stuck(patience=1).wrap(inner_policy=_inner)
    with pytest.raises(InteractionComplete):
        policy(_state(x=1.0))  # novel, so the wrapper delegates and the signal escapes
    assert len(calls) == 1
    with pytest.raises(HumanHelpRequested):
        policy(_state(x=1.0))  # a repeat, so the wrapper asks and never delegates
    assert len(calls) == 1


def test_an_unwrapped_inner_policy_still_raises_interaction_complete_through() -> None:
    """The signal EES-wide behaviour rests on is untouched by wrapping: a help-seeking
    arm still ends its period on InteractionComplete, it just may be repositioned
    first."""

    def _inner(state: State) -> LabeledAction:  # noqa: PLR0917 (core.Policy is positional)
        del state
        raise InteractionComplete

    policy = _at_random(mean_steps=10**9).wrap(inner_policy=_inner)
    with pytest.raises(InteractionComplete):
        policy(_state(x=1.0))


def test_being_told_help_was_granted_stops_the_request_repeating_forever() -> None:
    """Without this a rescued robot asks again on the very next call forever: the human
    by construction puts it back somewhere it has already been, so every state is
    instantly non-novel until the visited set is cleared."""
    seeking = _on_stuck(patience=1)
    policy = seeking.wrap(inner_policy=_acting_policy())
    policy(_state(x=1.0))
    with pytest.raises(HumanHelpRequested):
        policy(_state(x=1.0))
    seeking.note_help_granted()
    assert policy(_state(x=1.0)).label == "acted"


def test_beginning_a_period_starts_a_fresh_stretch() -> None:
    """The detector's visited set is per practice period: whatever the previous period
    left behind, the robot has not yet failed to make progress in this one."""
    seeking = _on_stuck(patience=1)
    policy = seeking.wrap(inner_policy=_acting_policy())
    policy(_state(x=1.0))
    with pytest.raises(HumanHelpRequested):
        policy(_state(x=1.0))
    seeking.begin_period()
    assert policy(_state(x=1.0)).label == "acted"


def test_a_random_arm_asks_whether_or_not_the_robot_is_getting_anywhere() -> None:
    """The two modes are genuinely independent: this one asks on a schedule of its own,
    whether or not the robot needed it. Every state here is novel, so an on-stuck arm
    would never ask at all."""
    seeking = _at_random(mean_steps=2, seed=0)
    policy = seeking.wrap(inner_policy=_acting_policy())
    asks = 0
    for x in range(200):
        try:
            policy(_state(x=float(x)))
        except HumanHelpRequested:
            asks += 1
    assert 0 < asks < 200


def test_a_random_arm_asks_at_about_its_configured_rate() -> None:
    seeking = _at_random(mean_steps=10, seed=0)
    policy = seeking.wrap(inner_policy=_acting_policy())
    asks = 0
    for x in range(10000):
        try:
            policy(_state(x=float(x)))
        except HumanHelpRequested:
            asks += 1
    # Bernoulli(1/10) over 10000 draws: sd is ~30, so 850-1150 is ~5 sd wide and cannot
    # flake on a fixed seed while still failing an off-by-an-order-of-magnitude rate.
    assert 850 < asks < 1150


def _ask_pattern(*, seed: int, steps: int = 200) -> tuple[bool, ...]:
    policy = _at_random(seed=seed).wrap(inner_policy=_acting_policy())
    pattern = []
    for x in range(steps):
        try:
            policy(_state(x=float(x)))
            pattern.append(False)
        except HumanHelpRequested:
            pattern.append(True)
    return tuple(pattern)


def test_a_random_arm_is_fully_determined_by_its_seed() -> None:
    assert _ask_pattern(seed=7) == _ask_pattern(seed=7)


def test_two_seeds_give_different_request_timing() -> None:
    """A sweep's seeds must not all ask on identical steps -- the BallRing
    --noise-seed trap, where a constant default made every arm identical."""
    assert len({_ask_pattern(seed=seed) for seed in range(6)}) == 6


def test_an_on_stuck_arm_consumes_no_randomness() -> None:
    """So that changing --stuck-patience cannot shift a random arm's stream, and so an
    on-stuck arm is bit-identical whatever --seed it is handed."""
    first = _on_stuck(patience=3, seed=0).wrap(inner_policy=_acting_policy())
    second = _on_stuck(patience=3, seed=999).wrap(inner_policy=_acting_policy())
    for policy in (first, second):
        # Three calls take the repeat counter to 2, one short of patience, so both arms
        # are still delegating -- and they are one call from asking together.
        for _ in range(3):
            assert policy(_state(x=1.0)).label == "acted"
    for policy in (first, second):
        with pytest.raises(HumanHelpRequested):
            policy(_state(x=1.0))


def test_a_random_arm_consumes_exactly_one_draw_per_policy_call() -> None:
    """Pinned directly against a reference stream rather than inferred from the fire
    pattern, because it is the invariant that makes an arm's schedule readable from its
    seed alone: N calls consume the first N draws and no others."""
    seeking = _at_random(mean_steps=3, seed=11)
    policy = seeking.wrap(inner_policy=_acting_policy())
    expected = np.random.default_rng(11).random(50) < 1.0 / 3
    for index, should_ask in enumerate(expected):
        try:
            policy(_state(x=float(index)))
            assert not should_ask
        except HumanHelpRequested:
            assert should_ask


def test_the_never_trigger_asks_for_nothing_and_draws_nothing() -> None:
    """A defensive default rather than the path a `never` run takes: a Method with no
    help-seeking configured holds no policy at all, which is what keeps such a run
    byte-identical. This pins that a policy built with NEVER anyway is inert."""
    seeking = HelpSeekingPolicy(trigger=HelpSeekingTrigger.NEVER, stuck_patience=1)
    policy = seeking.wrap(inner_policy=_acting_policy())
    for _ in range(20):
        assert policy(_state(x=1.0)).label == "acted"


def test_stuck_patience_must_be_at_least_one() -> None:
    with pytest.raises(ValueError):
        _on_stuck(patience=0)


def test_the_mean_step_gap_must_be_at_least_one() -> None:
    with pytest.raises(ValueError):
        _at_random(mean_steps=0)


def test_two_policies_do_not_share_a_detector() -> None:
    first = _on_stuck(patience=1).wrap(inner_policy=_acting_policy())
    second = _on_stuck(patience=1).wrap(inner_policy=_acting_policy())
    first(_state(x=1.0))
    second(_state(x=1.0))
    with pytest.raises(HumanHelpRequested):
        first(_state(x=1.0))
    assert second(_state(x=2.0)).label == "acted"


# --- ON_NO_APPLICABLE_SKILL: fires exactly when zero ground skills are applicable,
# checked fresh every call, no memory, no randomness.


def test_on_no_applicable_skill_requires_a_skill_provider() -> None:
    """Every other per-trigger field (stuck_patience, mean_steps_between_requests) has
    a default that makes it harmless when unused; skill_provider cannot, since there is
    no default SkillProvider to fall back to. Constructing this trigger without one is
    a misconfiguration worth refusing up front, the same way practice_loop.py refuses a
    Method that may ask paired with a Problem that has no HumanOracle."""
    with pytest.raises(ValueError, match="skill_provider"):
        HelpSeekingPolicy(trigger=HelpSeekingTrigger.ON_NO_APPLICABLE_SKILL)


def test_a_wrapped_policy_asks_when_no_ground_skill_is_applicable() -> None:
    policy = _on_no_applicable_skill().wrap(inner_policy=_acting_policy())
    with pytest.raises(HumanHelpRequested):
        policy(_state(x=0.0))  # Flagged is false, so Toggle has no applicable grounding


def test_a_wrapped_policy_delegates_when_a_ground_skill_is_applicable() -> None:
    policy = _on_no_applicable_skill().wrap(inner_policy=_acting_policy())
    assert policy(_state(x=1.0)).label == "acted"  # Flagged holds, so Toggle applies


def test_on_no_applicable_skill_reacts_to_the_state_it_is_given_each_call() -> None:
    """No memory beyond the current call -- a direct boolean check, not a detector.
    Toggling applicability back and forth toggles the ask/delegate outcome right back,
    which a stateful (patience-counter or visited-set) implementation would not do."""
    seeking = _on_no_applicable_skill()
    policy = seeking.wrap(inner_policy=_acting_policy())
    assert policy(_state(x=1.0)).label == "acted"
    with pytest.raises(HumanHelpRequested):
        policy(_state(x=0.0))
    assert policy(_state(x=1.0)).label == "acted"


def test_on_no_applicable_skill_consumes_no_randomness() -> None:
    """So that this trigger is bit-identical whatever --seed it is handed, matching the
    same guarantee ON_STUCK gives -- see test_an_on_stuck_arm_consumes_no_randomness."""
    first = _on_no_applicable_skill(seed=0).wrap(inner_policy=_acting_policy())
    second = _on_no_applicable_skill(seed=999).wrap(inner_policy=_acting_policy())
    for policy in (first, second):
        assert policy(_state(x=1.0)).label == "acted"
    for policy in (first, second):
        with pytest.raises(HumanHelpRequested):
            policy(_state(x=0.0))
