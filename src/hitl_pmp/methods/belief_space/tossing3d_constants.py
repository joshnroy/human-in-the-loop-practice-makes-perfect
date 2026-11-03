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
POUR_BIN_SKILL = "PourCubeOutOfBin"
RESET_SAME_SIDE_SKILL = "ResetCubeBinSameSide"
RESET_OPPOSITE_SIDE_SKILL = "ResetCubeBinOppositeSide"
HUMAN_RESET_SKILLS = frozenset({RESET_SAME_SIDE_SKILL, RESET_OPPOSITE_SIDE_SKILL})
NON_HUMAN_RESET_SKILL = "non_human_reset_cube_bin_only"
RESET_SKILLS = HUMAN_RESET_SKILLS | {NON_HUMAN_RESET_SKILL}
# Predicates that together name one functional value: an object is on at most one
# side, so re-asserting either member of a group replaces the whole group. Before the
# sides were split into robot-relative predicates this was one ternary predicate,
# replaced as a unit.
FUNCTIONAL_PREDICATE_GROUPS = (
    frozenset({"CubeOnRobotSide", "CubeOnOppositeSide"}),
    frozenset({"BinOnRobotSide", "BinOnOppositeSide"}),
)
PRACTICE_BUDGET = 20.0
LEARNING_RATE_PROCESS_NOISE_STD = 0.005
