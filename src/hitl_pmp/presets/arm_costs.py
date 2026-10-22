"""Which step-accounting cost file each final Tossing3D arm runs with.

One rule, two units (Josh, 2026-10-21, shared with the shelf): a human reset costs 500
for the planner arms, priced in control ticks like everything they plan with, and 5 for
EES, whose plan costs are -log(competence) (EXP-22c's ees-lamcost5). Under the step
protocol `EesMethod.practice_action_cost` plans with the cost file's human price as is,
so EES needs its own file; with the planners' it would price a reset at 500 and
essentially never plan one. An always-ask arm also needs the sampled-start reset
priced, or the step protocol refuses to start it.
"""

from pathlib import Path
from typing import ClassVar

from .preset import PRESET_DIRECTORY


class Tossing3DArmCosts:
    PLANNER_HUMAN_RESET_COST: ClassVar[float] = 500
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
