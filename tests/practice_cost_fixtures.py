"""Costs for tests. A domain that offers a human reset is priced only by a practice-cost
file, so a test that runs one, or builds a method that plans with one, needs a file (by
CLI) or an accounting (built directly)."""

from collections.abc import Mapping
from pathlib import Path
from typing import ClassVar

from hitl_pmp.core.practice_costs import (
    ChargeFunction,
    HumanCharge,
    PracticeAccounting,
    PracticeCosts,
)

ROOT = Path(__file__).parents[1]

# Why a test of a cycle-mode Tossing3D script is an expected failure for now.
RETIRED_CYCLE_MODE = (
    "cycle-mode Tossing3D is retired (it runs only under the step protocol with a costs "
    "file); this script's port-or-delete decision is pending with Josh"
)


class CostFixtures:
    # Unit prices: a robot step and every human skill the test domains name cost 1.
    UNIT: ClassVar[Path] = Path(__file__).parent / "unit_practice_costs.json"
    # A committed costs file that prices every human skill of each step-protocol domain.
    BY_ENV: ClassVar[dict[str, Path]] = {
        "tossing3d": ROOT / "src/hitl_pmp/presets/tossing3d_tick_priced_costs.json",
        "tossing3d_real": ROOT / "src/hitl_pmp/presets/tossing3d_tick_priced_costs.json",
        "kitchen_recycling3d": ROOT / "scripts/kitchen_recycling/costs.json",
        "pouring3d": ROOT / "src/hitl_pmp/presets/pouring3d_tick_priced_costs.json",
        "pouring3d_kitchen": ROOT / "configs/kitchen_pilot_costs.json",
        "racktoserving3d": ROOT / "src/hitl_pmp/presets/racktoserving3d_tick_priced_costs.json",
        "sweepoffshelf3d": ROOT / "configs/sweepoffshelf3d_practice_costs.json",
    }

    @staticmethod
    def argv(*, env: str) -> list[str]:
        return ["--practice-cost-config", str(CostFixtures.BY_ENV[env])]

    @staticmethod
    def step_argv(*, env: str, budget: int = 10) -> list[str]:
        """What a step-protocol domain needs to parse at all: a budget and a costs file."""
        return ["--practice-step-budget", str(budget), *CostFixtures.argv(env=env)]

    @staticmethod
    def costs(
        *, human: Mapping[str, float], robot_step: float = 1.0, human_weight: float = 1.0
    ) -> PracticeCosts:
        return PracticeCosts(
            robot_step=ChargeFunction(value=robot_step),
            human_skills={
                name: HumanCharge(cost=ChargeFunction(value=cost)) for name, cost in human.items()
            },
            human_weight=human_weight,
        )

    @staticmethod
    def accounting(
        *, human: Mapping[str, float], robot_step: float = 1.0, human_weight: float = 1.0
    ) -> PracticeAccounting:
        return PracticeAccounting(
            costs=CostFixtures.costs(human=human, robot_step=robot_step, human_weight=human_weight)
        )

    @staticmethod
    def unit_with_human_duration(*, path: Path, steps: int) -> Path:
        """The unit costs, with every human skill taking `steps` counted steps."""
        costs = PracticeCosts.model_validate_json(CostFixtures.UNIT.read_text())
        slow = HumanCharge(duration=ChargeFunction(value=steps))
        path.write_text(
            costs.model_copy(
                update={"human_skills": dict.fromkeys(costs.human_skills, slow)}
            ).model_dump_json()
        )
        return path

    @staticmethod
    def dispatch(*, argv: list[str]) -> None:
        """Parse a step-protocol domain's run (which needs a budget and a costs file to
        parse at all) and hand it to its method CLI, so a test that patches the env CLI's
        `run_method` captures the method factory without running the protocol."""
        from hitl_pmp.cli import ENVIRONMENTS, METHODS, Cli

        env = argv[argv.index("--env") + 1]
        args = Cli.parse_args(argv=[*argv, *CostFixtures.step_argv(env=env)])
        Cli.require_explicit_observation_weight(argv=argv, args=args)
        METHODS[args.method].run(args=args, env_cli=ENVIRONMENTS[args.env])
