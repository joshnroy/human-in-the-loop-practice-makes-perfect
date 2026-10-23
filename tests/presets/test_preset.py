import argparse
import json
from pathlib import Path

import pytest

from hitl_pmp.cli import Cli
from hitl_pmp.config_snapshot import ConfigSnapshot
from hitl_pmp.core.practice_costs import PracticeCosts
from hitl_pmp.environments.tossing3d.layout import Tossing3DLayout
from hitl_pmp.practice_loop import PracticeResetPolicy
from hitl_pmp.presets.preset import CliPreset

PRESET_ONLY = ["--env", "tossing3d", "--method", "pomdp", "--preset", "tossing3d-canonical"]
# The preset is defined only under the step protocol, which its cost file prices.
SIM_ARGV = [*PRESET_ONLY, "--practice-step-budget", "1000"]

# Written out by hand on purpose: this is the canonical arm, and a change to any of
# these values changes every result run through the preset.
CANONICAL_SHARED = {
    "pomdp_observation_probability_weight": 0.0,
    "pomdp_linear_cost_lambda": 3e-4,
    "pomdp_search_depth": 6,
    "pomdp_inference_engine": "grid",
    "pomdp_num_particles": 1024,
    "ees_reset_gate": False,
    "goal_pursuit_horizon": 0,
    "goal_pursuit_interval": 1,
    "practice_reset_policy": PracticeResetPolicy.NEVER,
    "defer_rendering": True,
}
CANONICAL_TOSSING3D = {
    "layout": Tossing3DLayout.BARRIER.value,
    "evaluation_layout": Tossing3DLayout.BARRIER.value,
    "max_steps_per_interaction": 20,
}


def parse(*, argv: list[str]) -> argparse.Namespace:
    return Cli.parse_args(argv=argv)


def test_the_canonical_preset_resolves_to_exactly_these_values() -> None:
    args = vars(parse(argv=SIM_ARGV))
    for key, value in {**CANONICAL_SHARED, **CANONICAL_TOSSING3D}.items():
        assert args[key] == value, key
    assert args["preset"] == "tossing3d-canonical"


def test_the_canonical_preset_sets_no_value_beyond_those_listed() -> None:
    values = CliPreset.load(name="tossing3d-canonical").values(env="tossing3d", method="pomdp")
    assert set(values) == set(CANONICAL_SHARED) | set(CANONICAL_TOSSING3D) | {
        "practice_cost_config"
    }


def test_the_cost_config_is_one_cost_per_control_tick_at_the_presets_lambda() -> None:
    args = parse(argv=SIM_ARGV)
    path = Path(args.practice_cost_config)
    assert path.is_absolute() and path.is_file()
    costs = PracticeCosts.load(path=path)
    # Tick-priced: a skill costs exactly the control ticks it is charged, so no table
    # of per-skill tick counts lives here -- the sim's ticks come from MuJoCo.
    assert costs.robot_step.value == 0.01 and costs.robot_step.function is None
    assert costs.objective_lambda == args.pomdp_linear_cost_lambda
    assert set(costs.human_skills) == {"reset_cube_far", "reset_cube_and_bin_near"}
    for charge in costs.human_skills.values():
        assert (charge.cost.value, charge.duration.value) == (5, 1)
    assert costs.human_weight == 1


def test_the_preset_refuses_to_run_outside_the_step_protocol(
    *, capsys: pytest.CaptureFixture[str]
) -> None:
    # In cycle mode the cost file is never read, so the arm would silently not be
    # the canonical one. Cycle-mode reproductions take explicit flags instead.
    with pytest.raises(SystemExit):
        parse(argv=PRESET_ONLY)
    assert "--practice-step-budget" in capsys.readouterr().err


def test_without_the_preset_the_observation_weight_default_stays_point_one() -> None:
    args = parse(argv=["--env", "tossing3d", "--method", "pomdp"])
    assert args.pomdp_observation_probability_weight == 0.1
    assert args.preset is None
    assert args.practice_cost_config is None


@pytest.mark.parametrize(
    ("flags", "key", "expected"),
    [
        (
            ["--pomdp-observation-probability-weight", "0.1"],
            "pomdp_observation_probability_weight",
            0.1,
        ),
        (["--ees-reset-gate"], "ees_reset_gate", True),
        (["--pomdp-num-particles", "20000"], "pomdp_num_particles", 20000),
        (["--max-steps-per-interaction", "4"], "max_steps_per_interaction", 4),
        (["--practice-cost-config", "mine.json"], "practice_cost_config", Path("mine.json")),
        (
            ["--practice-reset-policy", "scheduled"],
            "practice_reset_policy",
            PracticeResetPolicy.SCHEDULED,
        ),
    ],
)
def test_an_explicit_flag_overrides_the_preset_wherever_it_appears(
    *, flags: list[str], key: str, expected: object
) -> None:
    assert getattr(parse(argv=SIM_ARGV + flags), key) == expected
    assert getattr(parse(argv=flags + SIM_ARGV), key) == expected


def test_every_resolved_preset_value_lands_in_the_config_snapshot() -> None:
    args = parse(argv=SIM_ARGV)
    snapshot = ConfigSnapshot.collect(args=args, fd_exec_path=None).args
    assert snapshot["preset"] == "tossing3d-canonical"
    assert snapshot["pomdp_observation_probability_weight"] == "0.0"
    assert snapshot["pomdp_linear_cost_lambda"] == "0.0003"
    assert snapshot["practice_cost_config"] == str(args.practice_cost_config)


def test_a_preset_refuses_an_environment_it_does_not_name() -> None:
    with pytest.raises(ValueError, match="lightswitch"):
        parse(argv=["--env", "lightswitch", "--method", "pomdp", "--preset", "tossing3d-canonical"])


def test_a_preset_refuses_a_method_it_does_not_name() -> None:
    with pytest.raises(ValueError, match="ees"):
        parse(argv=["--env", "tossing3d", "--method", "ees", "--preset", "tossing3d-canonical"])


def test_a_preset_key_that_names_no_flag_is_an_error(*, tmp_path: Path) -> None:
    (tmp_path / "typo.json").write_text(
        json.dumps({
            "description": "a typo",
            "method": "pomdp",
            "shared": {"pomdp_observation_probabilty_weight": 0.0},
            "environments": {"tossing3d": {}},
        })
    )
    preset = CliPreset.load(name="typo", directory=tmp_path)
    parser = argparse.ArgumentParser()
    parser.add_argument("--pomdp-observation-probability-weight", type=float, default=0.1)
    with pytest.raises(ValueError, match="pomdp_observation_probabilty_weight"):
        preset.apply(parser=parser, env="tossing3d", method="pomdp")


def test_an_unknown_preset_name_lists_the_known_ones() -> None:
    with pytest.raises(SystemExit):
        parse(argv=["--env", "tossing3d", "--method", "pomdp", "--preset", "nope"])
    assert "tossing3d-canonical" in CliPreset.names()
