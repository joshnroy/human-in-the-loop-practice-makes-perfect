"""Serving's canonical preset is the toy-interface planner arm, priced as the toy is."""

from pathlib import Path

from hitl_pmp.cli import Cli
from hitl_pmp.core.practice_costs import PracticeAccounting, PracticeCosts
from hitl_pmp.environments.racktoserving3d.step_protocol import RackServingStepProtocol
from hitl_pmp.environments.racktoserving3d.toy_environment import RackServingToyEnvironment
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


def test_the_rack_preset_parses_as_the_first_decision_checks_ours_arm():
    args = vars(Cli.parse_args(argv=ARGV))
    assert args["preset"] == "racktoserving3d-canonical"
    assert args["max_steps_per_interaction"] == 20
    rack = CliPreset.load(name="racktoserving3d-canonical").shared
    assert rack["pomdp_solver"] == "determinized_astar"
    assert rack["pomdp_determinized_variant"] == "curtis"
    assert rack["pomdp_max_search_iterations"] == 1000
    assert rack["pomdp_observation_probability_weight"] == 0.3
    assert rack["pomdp_plan_beyond_session"] and rack["pomdp_learn_skill_costs"]
    for key, value in rack.items():
        if key != "practice_cost_config":
            assert args[key] == value or str(args[key]) == str(value), key


def test_the_rack_cost_file_prices_only_the_two_resets_as_the_toy_does():
    args = Cli.parse_args(argv=ARGV)
    path = Path(args.practice_cost_config)
    assert path.is_absolute() and path.is_file()
    assert path.name == "racktoserving3d_tick_priced_costs.json"
    costs = PracticeCosts.load(path=path)
    assert set(costs.human_skills) == set(RackServingStepProtocol.human_skill_names())
    assert (
        costs.human_skills["human_reset"].cost.value,
        costs.human_skills["human_reset"].duration.value,
    ) == (0.3, 1)
    assert costs.robot_step.value == 0.0
    assert costs.robot_step.function == "hitl_pmp.environments.tworoutes.costs:robot_step"
    assert (costs.human_weight, costs.objective_lambda) == (1, 1.0)
    accounting = PracticeAccounting(costs=costs)
    for resetting, price in ((True, 0.15), (False, 0.0)):
        observation = {"resetting": resetting, "robot_reset_price": 0.15}
        assert accounting.robot_charge(observation=observation).cost == price
    assert RackServingToyEnvironment.model_fields["robot_reset_price"].default == 0.15
