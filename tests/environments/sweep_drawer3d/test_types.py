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
