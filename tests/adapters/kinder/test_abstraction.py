"""KINDER's one-shot state abstractor behind `core`'s per-predicate `Predicate.holds`.

The abstractor computes *every* atom in one call, and does a `PyBulletSim.set_state` to
do it. `core.Predicate.holds` is asked one predicate at a time, and `Goal.is_satisfied`
asks for all of them in a row -- so the naive wrapper pays for a full forward-kinematics
pass per predicate per state. Hence the cache, and hence these tests.

The abstractor here is a stand-in rather than the real one. That is deliberate: the
property under test is *when the wrapper calls through and when it does not*, which a
real simulator would hide behind three seconds of MuJoCo. The real abstractor is
exercised against this wrapper in `tests/environments/tossing3d/`.
"""

import importlib.util

import numpy as np
import pytest

from hitl_pmp.adapters.kinder.abstraction import KinderAbstraction
from hitl_pmp.adapters.kinder.state_translation import KinderStateTranslator

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("relational_structs") is None,
    reason="relational_structs is part of the optional tossing3d extra",
)


def _kinder_state():
    from relational_structs import Object as KinderObject
    from relational_structs import ObjectCentricState
    from relational_structs import Type as KinderType

    robot_type = KinderType("robot")
    block_type = KinderType("block")
    type_features = {robot_type: ["grip"], block_type: ["x"]}
    data = {
        KinderObject("robot", robot_type): np.array([0.0], dtype=np.float64),
        KinderObject("block_0", block_type): np.array([1.5], dtype=np.float64),
    }
    return ObjectCentricState(data, type_features)


def _kinder_predicates():
    from relational_structs import Predicate as KinderPredicate
    from relational_structs import Type as KinderType

    return (
        KinderPredicate("HandEmpty", [KinderType("robot")]),
        KinderPredicate("Far", [KinderType("block")]),
    )


class _CountingAbstractor:
    """A stand-in abstractor: reports `Far(block_0)` iff x exceeds a threshold that lives
    *outside* the state, standing in for the live simulator's goal region, and counts how
    many times it was actually called."""

    def __init__(self) -> None:
        self.calls = 0
        self.threshold = 1.0

    def __call__(self, kinder_state):  # noqa: PLR0917  (mirrors KINDER's positional call)
        from relational_structs import GroundAtom

        self.calls += 1
        hand_empty, far = _kinder_predicates()
        robot = kinder_state.get_object_from_name("robot")
        block = kinder_state.get_object_from_name("block_0")
        atoms = {GroundAtom(hand_empty, [robot])}
        if kinder_state.get(block, "x") > self.threshold:
            atoms.add(GroundAtom(far, [block]))
        return _AbstractState(atoms)


class _AbstractState:
    def __init__(self, atoms) -> None:  # noqa: PLR0917  (mirrors RelationalAbstractState)
        self.atoms = atoms


def _build():
    kinder_state = _kinder_state()
    translator = KinderStateTranslator.from_kinder_state(kinder_state=kinder_state)
    abstractor = _CountingAbstractor()
    abstraction = KinderAbstraction.build(
        translator=translator,
        state_abstractor=abstractor,
        kinder_predicates=_kinder_predicates(),
    )
    return abstraction, abstractor, translator.to_core_state(kinder_state=kinder_state)


def test_a_predicate_reports_membership_in_the_abstractors_own_atom_set() -> None:
    """Not a reimplementation of the classifier -- a lookup in what upstream computed."""
    abstraction, _, state = _build()
    hand_empty = abstraction.predicate(name="HandEmpty")
    far = abstraction.predicate(name="Far")
    robot = abstraction.translator.core_objects["robot"]
    block = abstraction.translator.core_objects["block_0"]

    assert hand_empty.holds(state, (robot,)) is True
    assert far.holds(state, (block,)) is True


def test_asking_every_predicate_about_one_state_costs_one_abstractor_call() -> None:
    """The whole reason the cache exists. Five predicates over one state is one
    `PyBulletSim.set_state`, not five."""
    abstraction, abstractor, state = _build()
    robot = abstraction.translator.core_objects["robot"]
    block = abstraction.translator.core_objects["block_0"]

    for _ in range(3):
        abstraction.predicate(name="HandEmpty").holds(state, (robot,))
        abstraction.predicate(name="Far").holds(state, (block,))

    assert abstractor.calls == 1


def test_a_different_state_is_a_different_cache_entry() -> None:
    abstraction, abstractor, state = _build()
    block = abstraction.translator.core_objects["block_0"]
    far = abstraction.predicate(name="Far")

    assert far.holds(state, (block,)) is True
    nearer = state.model_copy(deep=True)
    nearer.set(obj=block, feature_name="x", feature_val=0.5)

    assert far.holds(nearer, (block,)) is False
    assert abstractor.calls == 2


def test_the_cache_survives_a_change_the_state_cannot_see_until_it_is_invalidated() -> None:
    """**The invalidation contract, and why the key cannot be the state alone.**

    `Tossing3DStateAbstractor.state_abstractor` says so in its own docstring: "poses come
    from `state`, the goal region from the live simulator". So one `core.State` maps to
    different atom sets across a scene rebuild, and a cache keyed on the state would keep
    serving the pre-rebuild answer forever.

    The first assertion below is the bug this test exists to pin -- it asserts the *stale*
    answer, deliberately, because that is what a correct cache does until it is told the
    world moved. The owner of the simulator is the only thing that knows a rebuild
    happened, so `invalidate()` is its responsibility to call.
    """
    abstraction, abstractor, state = _build()
    block = abstraction.translator.core_objects["block_0"]
    far = abstraction.predicate(name="Far")

    assert far.holds(state, (block,)) is True

    # The simulator moved under us; the state did not change at all.
    abstractor.threshold = 2.0
    assert far.holds(state, (block,)) is True, "a cache keyed on the state must not re-ask"
    assert abstractor.calls == 1

    abstraction.invalidate()

    assert far.holds(state, (block,)) is False
    assert abstractor.calls == 2


def test_an_atom_over_a_predicate_nobody_declared_is_loud() -> None:
    """A pin bump that adds a predicate must not be absorbed in silence. Dropping the
    unknown atom would leave a domain's operator model quietly incomplete, which is the
    exact class of drift consuming upstream's abstractor is meant to remove."""
    kinder_state = _kinder_state()
    translator = KinderStateTranslator.from_kinder_state(kinder_state=kinder_state)
    hand_empty, _ = _kinder_predicates()
    abstraction = KinderAbstraction.build(
        translator=translator,
        state_abstractor=_CountingAbstractor(),
        kinder_predicates=(hand_empty,),
    )
    state = translator.to_core_state(kinder_state=kinder_state)
    robot = translator.core_objects["robot"]

    with pytest.raises(ValueError, match="Far"):
        abstraction.predicate(name="HandEmpty").holds(state, (robot,))


def test_predicates_are_returned_in_the_order_they_were_declared() -> None:
    abstraction, _, _ = _build()

    assert [p.name for p in abstraction.predicates()] == ["HandEmpty", "Far"]


def test_a_core_predicates_types_are_kinders_own() -> None:
    abstraction, _, _ = _build()

    assert [t.name for t in abstraction.predicate(name="HandEmpty").types] == ["robot"]
