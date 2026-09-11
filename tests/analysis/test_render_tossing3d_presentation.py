from analysis.render_tossing3d_presentation import reset_destination


def test_reset_destination_reads_the_bound_side_object() -> None:
    decision = {
        "action": {
            "skill": {"name": "ask_for_reset_cube_bin_only"},
            "objects": [{"name": "robot"}, {"name": "opposite_side"}],
        }
    }
    assert reset_destination(decision) == "OPPOSITE SIDE"


def test_reset_destination_ignores_non_reset_actions() -> None:
    decision = {
        "action": {
            "skill": {"name": "PickCube"},
            "objects": [{"name": "robot_side"}],
        }
    }
    assert reset_destination(decision) is None
