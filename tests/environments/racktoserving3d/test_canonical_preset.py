"""The rack's canonical preset mirrors tossing3d-canonical and prices the rack's humans."""

from pathlib import Path

from hitl_pmp.cli import Cli
from hitl_pmp.core.practice_costs import PracticeCosts
from hitl_pmp.environments.racktoserving3d.skills import RackServing3DSkills as Skills
from hitl_pmp.presets.preset import CliPreset

ARGV = [
    "--env",
    "racktoserving3d",
    "--method",
    "pomdp",
    "--preset",
    "racktoserving3d-canonical",
    "--practice-step-budget",
    "1000",
]


def test_the_rack_preset_parses_and_shares_tossing3d_canonicals_values():
    args = vars(Cli.parse_args(argv=ARGV))
    assert args["preset"] == "racktoserving3d-canonical"
    assert args["max_steps_per_interaction"] == 20
    tossing = CliPreset.load(name="tossing3d-canonical").shared
    rack = CliPreset.load(name="racktoserving3d-canonical").shared
    assert {k: v for k, v in rack.items() if k != "practice_cost_config"} == {
        k: v for k, v in tossing.items() if k != "practice_cost_config"
    }
    for key, value in rack.items():
        if key != "practice_cost_config":
            assert args[key] == value or str(args[key]) == str(value), key


def test_the_rack_cost_file_prices_every_rack_human_skill_like_tossing3d():
    args = Cli.parse_args(argv=ARGV)
    path = Path(args.practice_cost_config)
    assert path.is_absolute() and path.is_file()
    assert path.name == "racktoserving3d_tick_priced_costs.json"
    costs = PracticeCosts.load(path=path)
    assert set(costs.human_skills) == {skill.name for skill in Skills.human_skills()}
    for charge in costs.human_skills.values():
        assert (charge.cost.value, charge.duration.value) == (5, 1)
    assert costs.robot_step.value == 0.01 and costs.robot_step.function is None
    assert costs.human_weight == 1
    assert costs.objective_lambda == args.pomdp_linear_cost_lambda == 3e-4
