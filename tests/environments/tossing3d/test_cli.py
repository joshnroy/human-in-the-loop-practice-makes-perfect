"""`Tossing3DCli`'s flags and its registration in the global CLI.

**This file is still entirely offline, and that is a property rather than a leftover.**
Constructing a `Tossing3DEnvironment` builds no simulator (`backend()` is lazy), and
`draw_scene_seed` is a pure function of an RNG, so the strongest assertions here need no
KINDER at all. `run_method` drives a real simulator and lives in `test_kinder_fidelity.py`.
"""

import argparse
import sys
from pathlib import Path

import pytest

from hitl_pmp.cli import ENVIRONMENTS, Cli
from hitl_pmp.environments.tossing3d.cli import Tossing3DCli
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.skill_oracle_policy import ORACLE_PARAMETER_SEED
from hitl_pmp.environments.tossing3d.tasks import Tossing3DTasks


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-test-tasks", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=None)
    Tossing3DCli.add_arguments(parser=parser)
    return parser


def test_the_environment_is_registered_under_tossing3d() -> None:
    assert ENVIRONMENTS["tossing3d"] is Tossing3DCli


def test_registering_it_does_not_pull_a_simulator_into_the_global_cli() -> None:
    """`hitl_pmp.cli` imports every registered environment's CLI. If this domain's import
    chain reached MuJoCo, `--env lightswitch` would stop working on a machine without the
    optional extra."""
    Cli.parse_args(argv=["--env", "lightswitch", "--method", "skill-oracle"])
    assert "mujoco" not in sys.modules


def test_the_defaults_are_read_off_the_models_rather_than_re_literalled() -> None:
    args = _build_parser().parse_args([])
    fields = Tossing3DEnvironment.model_fields
    assert args.variant == fields["variant"].default
    assert args.canonical_seed == fields["canonical_seed"].default
    assert args.scene_bg is True
    assert args.test_env_seed_offset == Tossing3DTasks.model_fields["test_env_seed_offset"].default
    assert args.oracle_parameter_seed == ORACLE_PARAMETER_SEED


def test_there_is_no_scene_selection_flag() -> None:
    """**The retired choice, pinned as an absence.** `--task-config` offered `stock`
    against `coincident`; once upstream's bin fix landed on `Tossing3D-o1.json` itself
    the two loaded the same scene, so the flag claimed a distinction that no longer
    existed. Asserted rather than merely deleted, because a flag that silently came back
    would bring the fork back with it -- and argparse rejecting the string is what stops
    an old command line from being read as a scene choice that was honoured."""
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["--task-config", "stock"])


def test_there_is_no_throw_standoff_flag() -> None:
    """**The second retired choice, pinned the same way and for a sharper reason.**
    `--oracle-throw-standoff` named the distance a separate `MoveToThrowPose` stopped at
    before a separate `Toss` fired. Upstream fuses the base move and the throw into one
    controller, which draws standoff, rotation, speed and release millisecond as a single
    four-vector from its own sampler -- so there is no longer a moment at which a standoff
    could be imposed, and a hand-picked value cannot be substituted into the draw
    piecewise.

    This matters more than an unused flag usually would: the value it carried (1.35) is
    still quoted throughout `docs/`, so an old command line would look like it was being
    honoured while the oracle drew something else entirely. argparse rejecting it is what
    makes that impossible rather than merely unlikely."""
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["--oracle-throw-standoff", "1.35"])


def test_the_oracle_parameter_seed_is_overridable() -> None:
    """What is left to choose is *which draw* the oracle takes, not what it draws. Fixed
    by default rather than threaded from `--seed`, because the oracle is a reference arm
    and should behave identically across the runs it is being compared against."""
    args = _build_parser().parse_args(["--oracle-parameter-seed", "77"])
    assert args.oracle_parameter_seed == 77


def test_the_global_cli_registers_this_domains_flags_when_env_is_tossing3d() -> None:
    args = Cli.parse_args(
        argv=["--env", "tossing3d", "--method", "skill-oracle", "--canonical-seed", "7"]
    )
    assert args.canonical_seed == 7
    assert args.num_test_tasks == 10


def test_a_run_that_writes_no_video_resolves_a_playback_rate_without_a_simulator() -> None:
    """`render_fps` is only ever read when a clip is written, and a run without
    `--output-dir` gets no renderer. Building a MuJoCo scene just to fill in a number
    nothing plays at would make every headless run pay for the one that does not."""
    assert Tossing3DCli.resolve_render_fps(env=Tossing3DEnvironment(), renderer=None) == (
        Tossing3DCli.unrendered_render_fps
    )
    assert "mujoco" not in sys.modules


# --- the separate evaluation Problem, and the reset-free arm it unlocks -----------
#
# Every evaluation episode opens with `reset_to_task`, a privileged state-write, so a
# shared Problem hands the practice environment `--num-test-tasks` free resets per
# sweep. That is invisible under the per-period reset and fatal to a reset-free arm:
# `--practice-reset-policy never` would be a label rather than a condition. These pin
# the split that makes it real.


def test_build_problem_returns_wholly_independent_objects() -> None:
    """Two calls must share nothing. A shared Environment is the exact defect this
    split exists to remove; a shared Tasks would re-use one test-task stream and make
    the second sweep's task set depend on the first's."""
    args = _build_parser().parse_args([])
    first = Tossing3DCli.build_problem(args=args)
    second = Tossing3DCli.build_problem(args=args)

    assert first is not second
    assert first.env is not second.env
    assert first.tasks is not second.tasks


def test_the_two_problems_are_configured_identically() -> None:
    """Independent but not different: the evaluation environment must be the same
    world, or the test set measures a different domain than practice trains on."""
    args = _build_parser().parse_args([])
    first = Tossing3DCli.build_problem(args=args)
    second = Tossing3DCli.build_problem(args=args)

    assert first.env.model_dump() == second.env.model_dump()
    assert first.tasks.seed == second.tasks.seed
    assert first.tasks.test_env_seed_offset == second.tasks.test_env_seed_offset


def test_both_problems_draw_the_same_test_scene_seeds() -> None:
    """The load-bearing one, and it needs no simulator: a task here is fully
    determined by its scene seed, so equal seed streams mean equal test sets. If these
    diverged, the split would silently change which tasks a run is scored on and every
    number would stop being comparable to one taken before it."""
    args = _build_parser().parse_args([])
    first = Tossing3DCli.build_problem(args=args)
    second = Tossing3DCli.build_problem(args=args)

    drawn = [Tossing3DTasks.draw_scene_seed(rng=first.tasks.test_rng) for _ in range(8)]
    redrawn = [Tossing3DTasks.draw_scene_seed(rng=second.tasks.test_rng) for _ in range(8)]
    assert drawn == redrawn
    # Not a constant stream -- otherwise the equality above would hold vacuously.
    assert len(set(drawn)) > 1


def test_building_a_problem_still_builds_no_simulator() -> None:
    """The property every offline test in this file rests on, asserted once directly
    rather than only relied upon."""
    args = _build_parser().parse_args([])
    Tossing3DCli.build_problem(args=args)
    assert "mujoco" not in sys.modules
