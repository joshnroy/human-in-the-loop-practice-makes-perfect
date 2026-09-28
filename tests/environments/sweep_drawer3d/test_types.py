"""A reset counts as successful only if all four of the task's reset criteria hold."""

from hitl_pmp.environments.sweep_drawer3d.types import ResetOutcome


def _outcome(**overrides: object) -> ResetOutcome:
    fields: dict[str, object] = {
        "locations": {f"cube_{i}": "counter" for i in range(5)},
        "in_pile": {f"cube_{i}": True for i in range(5)},
        "drawer_pos": 0.001,
        "wiper_xy_error": 0.004,
        "wiper_yaw_error_deg": 0.1,
        "wiper_on_counter": True,
        "robot_actions": 13,
        "ticks": 5000,
        "wall_s": 120.0,
    }
    fields.update(overrides)
    return ResetOutcome.model_validate(fields)


def test_everything_back_is_a_success() -> None:
    assert _outcome().success


def test_one_cube_short_of_the_pile_is_a_failure() -> None:
    o = _outcome(in_pile={f"cube_{i}": i != 3 for i in range(5)})
    assert o.n_in_pile == 4
    assert not o.success


def test_a_drawer_left_open_is_a_failure() -> None:
    assert not _outcome(drawer_pos=0.05).success


def test_a_wiper_off_its_spot_is_a_failure() -> None:
    """OpenDrawer and Sweep are anchored on the wiper's pose, so 'somewhere on the
    counter' is not enough."""
    assert not _outcome(wiper_xy_error=0.05).success
    assert not _outcome(wiper_on_counter=False).success


def _retrieval(**overrides: object):
    from hitl_pmp.environments.sweep_drawer3d.types import Retrieval

    fields: dict[str, object] = {
        "origin": "drawer",
        "blocked": True,
        "assists": (),
        "grasp": "single",
    }
    fields.update(overrides)
    return Retrieval.model_validate(fields)


def test_a_blocked_cube_is_rescued_by_every_move_that_shifted_it() -> None:
    r = _retrieval(assists=("wiggle", "nudge"))
    assert r.rescued_by == ("wiggle", "nudge")
    assert r.pathway == "wiggle > nudge > pick"


def test_a_cube_that_was_graspable_where_it_lay_was_rescued_by_nothing() -> None:
    """The wiggle shifts every cube in the drawer, including ones a grasp already fitted;
    shifting a cube is not rescuing it."""
    r = _retrieval(blocked=False, assists=("wiggle",))
    assert r.rescued_by == ()
    assert r.pathway == "pick"


def test_a_row_grasp_is_named_in_the_pathway() -> None:
    assert _retrieval(assists=("wiggle",), grasp="row").pathway == "wiggle > row grasp"


def test_a_blocked_cube_freed_without_being_moved_says_so() -> None:
    """Its neighbours were taken away, or a second pick attempt worked."""
    r = _retrieval(assists=())
    assert r.rescued_by == ()
    assert r.pathway == "neighbours removed > pick"


def test_an_outcome_without_retrievals_still_validates() -> None:
    """Cycle files written before repositioning existed carry no retrieval record."""
    assert _outcome().retrievals == {}


def test_a_squeeze_grasp_is_named_in_the_pathway() -> None:
    """It is not a repositioning move -- nothing is shifted before the pick -- but it is
    the reason a cube turned against a wall is picked at all."""
    r = _retrieval(blocked=False, grasp="squeeze")
    assert r.pathway == "squeeze grasp"
    assert _retrieval(assists=("wiggle",), grasp="squeeze").pathway == "wiggle > squeeze grasp"


def test_mechanisms_that_can_be_switched_off_are_named_once() -> None:
    from hitl_pmp.environments.sweep_drawer3d.types import Mechanisms

    assert set(Mechanisms.ALL) == {"wiggle", "nudge", "squeeze", "shift", "drawers", "board"}
    assert Mechanisms.check(names=("wiggle", "squeeze")) == frozenset({"wiggle", "squeeze"})


def test_a_misspelt_mechanism_is_refused_rather_than_ignored() -> None:
    """An ablation that silently switched nothing off would be reported as a null result."""
    import pytest

    from hitl_pmp.environments.sweep_drawer3d.types import Mechanisms

    with pytest.raises(ValueError, match="wigle"):
        Mechanisms.check(names=("wigle",))
