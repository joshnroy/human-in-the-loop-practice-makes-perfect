import shlex
from pathlib import Path

import pytest

from hitl_pmp.cli import Cli
from hitl_pmp.core.practice_costs import PracticeCosts
from hitl_pmp.presets.arm_costs import Tossing3DArmCosts
from scripts.tossing3d_launch import DELLA, FULL, SEEDS, Tossing3DLaunch

ARMS = [
    "ees",
    "expectimax",
    "determinized",
    "ees-always-ask",
    "expectimax-always-ask",
    "determinized-always-ask",
]


def _parse(*, name: str, seed: int = 0) -> object:
    arm = Tossing3DLaunch.arms()[name]
    return Cli.parse_args(
        argv=Tossing3DLaunch.cli_argv(arm=arm, size=FULL, seed=seed, output_dir=Path("/out"))
    )


def test_the_plan_is_six_arms_over_seeds_zero_to_five() -> None:
    # Josh, 2026-10-22: six seeds per classical arm, so a paired sign test can reach 0.031.
    assert list(Tossing3DLaunch.arms()) == ARMS
    assert SEEDS == 6


@pytest.mark.parametrize("name", ARMS)
def test_every_arm_runs_the_canonical_budget_under_the_step_protocol(*, name: str) -> None:
    args = _parse(name=name)
    assert args.env == "tossing3d"
    assert args.practice_step_budget == 160000
    assert args.measurement_interval_steps == 3200
    assert args.num_test_tasks == 10
    assert args.evaluation_control_steps == 500
    assert args.layout == "barrier" and args.evaluation_layout == "barrier"
    assert args.max_steps_per_interaction == 20


@pytest.mark.parametrize("name", [n for n in ARMS if not n.startswith("ees")])
def test_the_planner_arms_are_the_canonical_preset(*, name: str) -> None:
    args = _parse(name=name)
    assert args.method == "pomdp"
    assert args.preset == "tossing3d-canonical"
    assert args.pomdp_observation_probability_weight == 0.0
    solver = "determinized_astar" if name.startswith("determinized") else "expectimax"
    assert args.pomdp_solver == solver


@pytest.mark.parametrize("name", ARMS)
def test_each_arm_prices_human_resets_with_its_own_cost_file(*, name: str) -> None:
    args = _parse(name=name)
    method = "ees" if name.startswith("ees") else "pomdp"
    always_ask = name.endswith("always-ask")
    assert args.always_ask_reset is always_ask
    expected = Tossing3DArmCosts.cost_config(method=method, always_ask=always_ask)
    assert Path(args.practice_cost_config).resolve() == expected
    PracticeCosts.load(path=args.practice_cost_config)


def test_ees_runs_without_skip_perfect_and_with_its_default_prior() -> None:
    # The shelf's labelled deviation (Josh, 2026-10-21), applied here too; EXP-22c's
    # prior, since only the shelf chose Beta(1, 3).
    args = _parse(name="ees")
    assert args.method == "ees"
    assert args.reproduce_predicators_skip_perfect is False
    assert "--ees-prior-alpha" not in Tossing3DLaunch.cli_argv(
        arm=Tossing3DLaunch.arms()["ees"], size=FULL, seed=0, output_dir=Path("/out")
    )


@pytest.mark.parametrize("name", ARMS)
def test_no_arm_stops_early(*, name: str) -> None:
    # Every arm spends the whole budget, so the curves stay comparable to the end.
    argv = Tossing3DLaunch.cli_argv(
        arm=Tossing3DLaunch.arms()[name], size=FULL, seed=0, output_dir=Path("/out")
    )
    assert "--stop-after-perfect-evaluations" not in argv


def test_the_della_manifest_has_one_task_per_arm_and_seed() -> None:
    arms = list(Tossing3DLaunch.arms().values())
    lines = Tossing3DLaunch.della_manifest(arms=arms, size=FULL, sha="abc123", tag="t3")
    assert len(lines) == 36
    seen = set()
    for line in lines:
        name, sha, arm, seed, out, argv = line.split("\t")
        assert sha == "abc123"
        assert out == f"{DELLA}/results/t3/{arm}/{Tossing3DLaunch.arms()[arm].method}/{seed}"
        parsed = shlex.split(argv)
        assert parsed[parsed.index("--seed") + 1] == seed
        assert parsed[parsed.index("--output-dir") + 1] == out
        seen.add((arm, int(seed)))
    assert seen == {(arm, seed) for arm in ARMS for seed in range(6)}


def test_the_della_time_limits_are_twice_the_dry_run_projection() -> None:
    # Josh, 2026-10-22: 2x the dry run's projection per group.
    from scripts.tossing3d_launch import DELLA_GROUPS

    assert {group: limits.time for group, limits in DELLA_GROUPS.items()} == {
        "ees": "08:00:00",
        "determinized": "16:00:00",
        "expectimax": "48:00:00",
    }
