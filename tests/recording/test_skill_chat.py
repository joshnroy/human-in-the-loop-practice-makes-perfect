import numpy as np

from hitl_pmp.recording.skill_chat import SkillChatOverlay


def test_competence_panel_preserves_scene_and_uses_separate_space() -> None:
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    rendered = SkillChatOverlay.compose(
        frame=frame,
        history=[],
        values={"STOP": 0.4},
        competences={"PickCube (fixed)": 0.5, "Toss (belief mean)": 0.6},
    )
    assert rendered.shape == (640, 1280, 3)
    np.testing.assert_array_equal(rendered[:, :640], frame)


def test_chat_preserves_scene_and_has_constant_size() -> None:
    frame = np.full((640, 640, 3), 120, dtype=np.uint8)
    empty = SkillChatOverlay.compose(frame=frame, history=[])
    populated = SkillChatOverlay.compose(
        frame=frame, history=["01 PickCube", "02 HUMAN RESET cube + bin", "03 PickCube"]
    )
    assert empty.shape == populated.shape == (640, 960, 3)
    np.testing.assert_array_equal(populated[:, :640], frame)
    assert not np.array_equal(empty[:, 640:], populated[:, 640:])


def test_chat_drops_oldest_entries() -> None:
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    history = [f"{index:02d} PickCube" for index in range(20)]
    np.testing.assert_array_equal(
        SkillChatOverlay.compose(frame=frame, history=history),
        SkillChatOverlay.compose(frame=frame, history=history[-8:]),
    )


def test_value_chart_updates_without_changing_scene_or_frame_size() -> None:
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    first = SkillChatOverlay.compose(
        frame=frame, history=[], values={"STOP": 0.5, "PickCube": -0.1}
    )
    second = SkillChatOverlay.compose(
        frame=frame, history=[], values={"STOP": 0.4, "PickCube": 0.6}
    )
    assert first.shape == second.shape == (640, 960, 3)
    np.testing.assert_array_equal(first[:, :640], frame)
    assert not np.array_equal(first[:, 640:], second[:, 640:])
