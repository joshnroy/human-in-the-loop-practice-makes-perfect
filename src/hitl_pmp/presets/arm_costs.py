"""Which step-accounting cost file each final Tossing3D arm runs with.

Every arm is charged 5 per human reset (Josh, 2026-10-23, shared with the shelf). For
EES that is its plan cost as is (EXP-22c's ees-lamcost5; its robot skills cost
-log(competence)). For the planner arms it is an exact rescale of the old pricing --
reset 500, robot step 1, lambda 3e-6 -- to reset 5, robot step 0.01, lambda 3e-4, so
lambda * cost, the only way the planner reads a cost, is unchanged. The planner and EES
files now hold the same prices; they stay separate files because an always-ask arm
also needs the sampled-start reset priced, or the step protocol refuses to start it.
"""

from pathlib import Path
from typing import ClassVar

from .preset import PRESET_DIRECTORY


class Tossing3DArmCosts:
    PLANNER_HUMAN_RESET_COST: ClassVar[float] = 5
    EES_HUMAN_RESET_COST: ClassVar[float] = 5

    FILES: ClassVar[dict[tuple[str, bool], str]] = {
        ("pomdp", False): "tossing3d_tick_priced_costs.json",
        ("pomdp", True): "tossing3d_always_ask_tick_priced_costs.json",
        ("ees", False): "tossing3d_ees_tick_priced_costs.json",
        ("ees", True): "tossing3d_ees_always_ask_tick_priced_costs.json",
    }

    @staticmethod
    def cost_config(*, method: str, always_ask: bool) -> Path:
        name = Tossing3DArmCosts.FILES.get((method, always_ask))
        if name is None:
            raise ValueError(
                f"no Tossing3D cost file for --method {method}; known: "
                f"{sorted({known for known, _ in Tossing3DArmCosts.FILES})}"
            )
        return (PRESET_DIRECTORY / name).resolve()
