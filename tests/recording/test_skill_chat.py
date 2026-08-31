import numpy as np

from hitl_pmp.recording.skill_chat import SkillChatOverlay


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
