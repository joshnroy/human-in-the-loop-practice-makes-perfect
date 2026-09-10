"""Canonical Tossing3D skill names used by the practice model."""

PICK_SKILL = "PickCube"
PICK_SKILLS = frozenset({
    PICK_SKILL,
    "PickCubeFromFloor",
    "PickCubeFromBin",
    "PickCubeFromRim",
})
TOSS_SKILL = "MoveToTossLocationAndToss"
OPEN_GRIPPER_SKILL = "OpenGripper"
RESET_SKILL = "ask_for_reset_cube_bin_only"
RESET_SKILLS = frozenset({RESET_SKILL})
PRACTICE_BUDGET = 20.0
LEARNING_RATE_PROCESS_NOISE_STD = 0.05
