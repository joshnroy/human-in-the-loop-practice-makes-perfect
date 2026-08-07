import numpy as np
import pytest

from hitl_pmp.core.problem.environment.types import Object, State, Type
from hitl_pmp.methods.stuck_detector import StuckDetector

robot_type = Type(name="robot", feature_names=("room", "holding"))
pile_type = Type(name="pile", feature_names=("room", "holding"))
robot = Object(name="robot", type=robot_type)
# Same NAME as `robot`, different Type -- the tossingroom --unsplit-skills pattern,
# where `trash` exists under both `trash_type` and the shared `item_type`.
pile_named_robot = Object(name="robot", type=pile_type)


def state(*, room: float, holding: float = 0.0) -> State:
    return State(data={robot: np.array([room, holding])})


def test_a_fresh_detector_is_not_stuck():
    assert not StuckDetector(patience=3).is_stuck()


def test_patience_must_be_at_least_one():
    with pytest.raises(ValueError):
        StuckDetector(patience=0)


def test_a_run_of_distinct_states_never_becomes_stuck():
    detector = StuckDetector(patience=3)
    for room in range(20):
        detector.observe(state=state(room=float(room)))
        assert not detector.is_stuck()


def test_repeating_one_state_becomes_stuck_only_at_patience():
    detector = StuckDetector(patience=3)
    detector.observe(state=state(room=1.0))
    assert detector.steps_since_novel == 0
    for expected in (1, 2):
        detector.observe(state=state(room=1.0))
        assert detector.steps_since_novel == expected
        assert not detector.is_stuck()
    detector.observe(state=state(room=1.0))
    assert detector.steps_since_novel == 3
    assert detector.is_stuck()


def test_a_novel_state_clears_the_counter():
    detector = StuckDetector(patience=2)
    detector.observe(state=state(room=1.0))
    detector.observe(state=state(room=1.0))
    detector.observe(state=state(room=1.0))
    assert detector.is_stuck()
    detector.observe(state=state(room=7.0))
    assert detector.steps_since_novel == 0
    assert not detector.is_stuck()


def test_cycling_between_already_visited_states_counts_as_no_progress():
    """The load-bearing case, and why a bare `state != previous_state` check is not
    enough: a robot stranded behind Tossing Room's one-way ledge still walks, so every
    step changes the state while none of them reaches anywhere new."""
    detector = StuckDetector(patience=4)
    for room in (0.0, 1.0, 2.0):
        detector.observe(state=state(room=room))
    assert not detector.is_stuck()
    for room in (1.0, 0.0, 1.0, 2.0):
        detector.observe(state=state(room=room))
    assert detector.is_stuck()


def test_restart_forgets_the_visited_set():
    detector = StuckDetector(patience=2)
    for _ in range(4):
        detector.observe(state=state(room=1.0))
    assert detector.is_stuck()
    detector.restart()
    assert detector.steps_since_novel == 0
    assert not detector.is_stuck()
    # The same state is novel again, so the stretch really did start over rather than
    # only zeroing the counter.
    detector.observe(state=state(room=1.0))
    assert detector.steps_since_novel == 0


def test_one_changed_feature_makes_a_state_novel():
    detector = StuckDetector(patience=1)
    detector.observe(state=state(room=1.0, holding=0.0))
    detector.observe(state=state(room=1.0, holding=0.0))
    assert detector.is_stuck()
    detector.observe(state=state(room=1.0, holding=2.0))
    assert not detector.is_stuck()


def test_objects_sharing_a_name_under_different_types_are_distinct():
    detector = StuckDetector(patience=1)
    detector.observe(state=State(data={robot: np.array([1.0, 0.0])}))
    detector.observe(state=State(data={pile_named_robot: np.array([1.0, 0.0])}))
    assert detector.steps_since_novel == 0


def test_the_key_does_not_depend_on_dict_insertion_order():
    other = Object(name="other", type=robot_type)
    detector = StuckDetector(patience=1)
    detector.observe(state=State(data={robot: np.array([1.0, 0.0]), other: np.array([2.0, 0.0])}))
    detector.observe(state=State(data={other: np.array([2.0, 0.0]), robot: np.array([1.0, 0.0])}))
    assert detector.steps_since_novel == 1


def test_two_detectors_do_not_share_a_visited_set():
    """PrivateAttr default_factory, not a mutable class attribute: two arms of one
    sweep run in one process in the tests, and a shared set would make the second
    detector inherit the first's history."""
    first = StuckDetector(patience=1)
    second = StuckDetector(patience=1)
    first.observe(state=state(room=1.0))
    second.observe(state=state(room=1.0))
    assert second.steps_since_novel == 0
