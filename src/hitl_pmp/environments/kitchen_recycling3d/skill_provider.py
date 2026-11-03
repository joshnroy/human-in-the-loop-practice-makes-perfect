"""Tossing3D's provider, charging the two human resets under the kitchen's cost-file names.

Only the charge label differs: `skills`, the reset groundings and their costs are
Tossing3D's, which `test_skill_table.py` diffs field by field.
"""

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider

from .backend import OPPOSITE_SIDE, ROBOT_SIDE

SAME_SIDE_RESET = "ResetCubeBinSameSide"
OPPOSITE_SIDE_RESET = "ResetCubeBinOppositeSide"
RESET_NAMES = {ROBOT_SIDE: SAME_SIDE_RESET, OPPOSITE_SIDE: OPPOSITE_SIDE_RESET}


class KitchenRecyclingSkillProvider(Tossing3DSkillProvider):
    def human_skill_name(self, *, ground_skill: GroundSkill) -> str | None:
        destination = self.movables_reset_destination(ground_skill=ground_skill)
        return None if destination is None else RESET_NAMES[destination]
