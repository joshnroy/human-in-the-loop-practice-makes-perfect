"""The two-way translation between KINDER's `ObjectCentricState` and `core.State`.

Offline: `relational_structs` is pure Python and imports without MuJoCo (verified), so
these run wherever the gate runs rather than being simulator-gated. The states below are
built by hand rather than by resetting a scene, which is the point -- a translation bug
should surface here, in milliseconds, not inside a rollout.
"""

import importlib.util

import numpy as np
import pytest

from hitl_pmp.adapters.kinder.state_translation import KinderStateTranslator
from hitl_pmp.core.problem.environment.types import Object, State, Type

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("relational_structs") is None,
    reason="relational_structs is part of the optional tossing3d extra",
)


def _kinder_state():
    """A hand-built two-object KINDER state, with no simulator anywhere near it."""
    from relational_structs import Object as KinderObject
    from relational_structs import ObjectCentricState
    from relational_structs import Type as KinderType

    robot_type = KinderType("robot")
    block_type = KinderType("block")
    type_features = {robot_type: ["x", "grip"], block_type: ["x", "y", "z"]}
    robot = KinderObject("robot", robot_type)
    block = KinderObject("block_0", block_type)
    data = {
        robot: np.array([0.25, 1.0], dtype=np.float64),
        block: np.array([1.5, -0.5, 0.125], dtype=np.float64),
    }
    return ObjectCentricState(data, type_features)


def test_a_core_type_carries_kinders_whole_feature_schema_not_a_subset() -> None:
    """The schema is copied wholesale, in order, because the reverse translation has to
    be lossless: KINDER's own classifiers run forward kinematics off the arm joints, so a
    `core.State` that kept only the features hitl's own arithmetic once read could not
    rebuild a state the abstractor would accept."""
    translator = KinderStateTranslator.from_kinder_state(kinder_state=_kinder_state())

    assert translator.core_types["robot"].feature_names == ("x", "grip")
    assert translator.core_types["block"].feature_names == ("x", "y", "z")


def test_translation_round_trips_every_feature_bit_for_bit() -> None:
    """Lossless in both directions. Not `approx`: the translation is a copy, not a
    computation, so anything but equality means a feature was reordered or truncated."""
    kinder_state = _kinder_state()
    translator = KinderStateTranslator.from_kinder_state(kinder_state=kinder_state)

    core_state = translator.to_core_state(kinder_state=kinder_state)
    back = translator.to_kinder_state(state=core_state)

    assert set(back.get_object_names()) == set(kinder_state.get_object_names())
    for obj in kinder_state:
        np.testing.assert_array_equal(np.asarray(back[obj]), np.asarray(kinder_state[obj]))


def test_a_core_object_keeps_its_kinder_name_and_type() -> None:
    translator = KinderStateTranslator.from_kinder_state(kinder_state=_kinder_state())

    assert translator.core_objects["block_0"].type.name == "block"
    assert translator.core_objects["robot"].type.name == "robot"


def test_objects_hitl_added_are_ignored_rather_than_rejected() -> None:
    """A domain may carry pseudo-objects KINDER knows nothing about -- Tossing3D's
    `scene`, which holds the seed a rebuild needs and the step count `set_state` refuses
    on. They are dropped on the way back rather than raising, because the KINDER state
    being rebuilt is only ever handed to KINDER, which has no place to put them."""
    kinder_state = _kinder_state()
    translator = KinderStateTranslator.from_kinder_state(kinder_state=kinder_state)
    core_state = translator.to_core_state(kinder_state=kinder_state)

    scene_type = Type(name="scene", feature_names=("seed",))
    scene = Object(name="scene", type=scene_type)
    widened = State(data={**core_state.data, scene: np.array([125.0])})

    back = translator.to_kinder_state(state=widened)

    assert set(back.get_object_names()) == {"robot", "block_0"}


def test_a_missing_kinder_object_is_a_loud_error_not_a_silent_gap() -> None:
    """The opposite direction of the rule above. An object KINDER *expects* and the
    `core.State` does not carry cannot be defaulted -- the abstractor would read zeros as
    a real pose -- so it raises."""
    kinder_state = _kinder_state()
    translator = KinderStateTranslator.from_kinder_state(kinder_state=kinder_state)
    core_state = translator.to_core_state(kinder_state=kinder_state)

    without_block = State(
        data={obj: vec for obj, vec in core_state.data.items() if obj.name != "block_0"}
    )

    with pytest.raises(KeyError, match="block_0"):
        translator.to_kinder_state(state=without_block)
