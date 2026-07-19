import numpy as np
import pytest
from pydantic import ValidationError

from hitl_pmp.core.problem.environment.types import Object, State, Type


def test_type_dim_matches_feature_names() -> None:
    block = Type(name="block", feature_names=("x", "y", "z"))
    assert block.dim == 3


def test_type_dim_zero_for_no_features() -> None:
    marker = Type(name="marker", feature_names=())
    assert marker.dim == 0


def test_object_equal_content_is_equal_and_hashable() -> None:
    block = Type(name="block", feature_names=("x",))
    obj1 = Object(name="block1", type=block)
    obj2 = Object(name="block1", type=block)
    assert obj1 == obj2
    assert hash(obj1) == hash(obj2)


def test_object_is_frozen() -> None:
    block = Type(name="block", feature_names=("x",))
    obj = Object(name="block1", type=block)
    with pytest.raises(ValidationError):
        obj.name = "renamed"  # type: ignore[misc]


def test_state_accepts_matching_feature_dims() -> None:
    block = Type(name="block", feature_names=("x", "y"))
    obj = Object(name="block1", type=block)
    state = State(data={obj: np.array([1.0, 2.0])})
    assert state[obj].tolist() == [1.0, 2.0]


def test_state_rejects_mismatched_feature_dims() -> None:
    block = Type(name="block", feature_names=("x", "y"))
    obj = Object(name="block1", type=block)
    with pytest.raises(ValidationError):
        State(data={obj: np.array([1.0])})


def test_state_get_and_set_round_trip_by_feature_name() -> None:
    block = Type(name="block", feature_names=("x", "y"))
    obj = Object(name="block1", type=block)
    state = State(data={obj: np.array([1.0, 2.0])})
    assert state.get(obj=obj, feature_name="y") == 2.0
    state.set(obj=obj, feature_name="y", feature_val=5.0)
    assert state.get(obj=obj, feature_name="y") == 5.0
