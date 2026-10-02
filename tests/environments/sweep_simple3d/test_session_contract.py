"""Native scene selection must not drift back to the drawer task."""

from hitl_pmp.environments.sweep_simple3d.session import SweepSimpleSession


def test_simple_session_selects_native_floor_task() -> None:
    assert SweepSimpleSession.ENV_ID == (
        "kinder/SweepSimple3D-o5-sweep_the_blocks_to_the_left_side_of_the_kitchen_island-v0"
    )


def test_physical_session_defaults_to_the_surviving_simple_scene() -> None:
    from hitl_pmp.environments.sweep_simple3d.physical.session import SweepPhysicalSession

    assert SweepPhysicalSession.ENV_ID == SweepSimpleSession.ENV_ID
