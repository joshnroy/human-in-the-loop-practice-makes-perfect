from analysis.render_tossing3d_presentation import (
    _cycle_end_reason,
    _decision_values,
    _is_applicable,
    reset_destination,
)


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


def test_decision_values_preserve_all_logged_root_actions() -> None:
    decision = {
        "action": {"skill": {"name": "PickCube"}},
        "value": 0.8,
        "action_values": {"PickCube": 0.8, "OpenGripper": 0.6, "STOP": 0.5},
    }
    assert _decision_values(decision) == {
        "PickCube": 0.8,
        "OpenGripper": 0.6,
        "STOP": 0.5,
    }


def test_cycle_end_reason_reports_stop() -> None:
    assert _cycle_end_reason({"action": "STOP"}) == "PRACTICE CYCLE ENDED — STOP SELECTED"


def test_same_side_open_gripper_requires_closed_empty() -> None:
    assert not _is_applicable({"atoms": ["name='HandEmpty'"]}, "OpenGripper")
    assert _is_applicable({"atoms": ["name='ClosedEmpty'"]}, "OpenGripper")
