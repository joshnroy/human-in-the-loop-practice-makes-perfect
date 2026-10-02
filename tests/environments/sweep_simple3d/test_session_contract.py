"""Native scene selection must not drift back to the drawer task."""

from hitl_pmp.environments.sweep_simple3d.session import SweepSimpleSession


def test_simple_session_selects_native_floor_task() -> None:
    assert SweepSimpleSession.ENV_ID == (
        "kinder/SweepSimple3D-o5-sweep_the_blocks_to_the_left_side_of_the_kitchen_island-v0"
    )


def test_drawer_session_keeps_original_scene() -> None:
    from hitl_pmp.environments.sweep_drawer3d.session import SweepDrawerSession

    assert SweepDrawerSession.ENV_ID == "kinder/SweepIntoDrawer3D-o5-v0"
