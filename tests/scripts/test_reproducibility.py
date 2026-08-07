"""Locks in the guarantee every sweep depends on: one `--seed` integer fully
determines a run's results.

This is asserted end-to-end through the real CLI rather than by unit-testing each
RNG, because the guarantee is a property of the *whole* pipeline -- task sampling,
skill/parameter sampling, and torch training all draw randomness, and any one of
them reaching for an unseeded global would break reproducibility without breaking
any narrower test. A sweep whose numbers cannot be regenerated months later is not
a research result, so this is worth pinning explicitly.
"""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

# The DISTRIBUTION is `kindergarden`; the IMPORT package is `kinder`. Keying on the
# distribution name here would skip everything, always. Matches how every other
# simulator-backed test in this repo gates (see tests/environments/tossing3d/).
needs_kinder = pytest.mark.skipif(
    importlib.util.find_spec("kinder") is None or importlib.util.find_spec("kinder_models") is None,
    reason="KINDER is an optional extra (`kindergarden` + `kinder_models`); CI never installs it",
)

# grid_size=1 puts the robot in the light's own cell from the start, so TurnOnLight is
# immediately applicable and this baseline's solve rate genuinely varies with the seed
# (5/8, 3/8, 1/8 for seeds 1/2/3). At larger grids it scores 0 for every seed, which
# would make the different-seeds assertion vacuous.
LIGHTSWITCH_ARGS = ("--grid-size", "1", "--num-test-tasks", "8")

# Deliberately the smallest run that still executes real controllers: one evaluation
# sweep over two scenes. Each task is a full pick/move/toss attempt in MuJoCo, so this
# is minutes rather than the milliseconds Light Switch costs -- which is why it is
# gated off CI rather than folded into the default gate.
TOSSING3D_ARGS = ("--num-test-tasks", "2", "--num-cycles", "0")

# This baseline solves 0/8 here at every seed, so the different-seeds assertion rests on
# the rest of stats.json -- the per-skill practice outcomes and the per-window planning
# counts, which do vary. Two short cycles are enough to produce them.
TOSSINGROOM_ARGS = (
    "--num-test-tasks",
    "8",
    "--num-cycles",
    "2",
    "--max-steps-per-interaction",
    "20",
)


class ReproducibilityHarness:
    """A static-method container, never instantiated, same as every other
    business-logic class in this project."""

    @staticmethod
    def run(
        *,
        output_dir: Path,
        seed: int,
        threads: int,
        env_name: str = "lightswitch",
        env_args: tuple[str, ...] = LIGHTSWITCH_ARGS,
    ) -> dict[str, object]:
        """One short real CLI run; returns its parsed stats.json."""
        output_dir.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(  # noqa: S603
            [
                sys.executable,
                "-m",
                "hitl_pmp.cli",
                "--env",
                env_name,
                "--method",
                "random-skills",
                *env_args,
                "--seed",
                str(seed),
                "--output-dir",
                str(output_dir),
            ],
            capture_output=True,
            text=True,
            env={**os.environ, "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads)},
            check=True,
        )
        assert "success rate" in completed.stdout
        parsed: dict[str, object] = json.loads((output_dir / "stats.json").read_text())
        return parsed

    @staticmethod
    def run_tossing3d(*, output_dir: Path, seed: int, threads: int) -> dict[str, object]:
        return ReproducibilityHarness.run(
            output_dir=output_dir,
            seed=seed,
            threads=threads,
            env_name="tossing3d",
            env_args=TOSSING3D_ARGS,
        )

    @staticmethod
    def run_pickup_weight(*, output_dir: Path, seed: int, threads: int) -> dict[str, object]:
        return ReproducibilityHarness.run(
            output_dir=output_dir,
            seed=seed,
            threads=threads,
            env_name="tossingroom",
            env_args=TOSSINGROOM_ARGS,
        )


def test_the_same_seed_produces_identical_results(*, tmp_path: Path) -> None:
    first = ReproducibilityHarness.run(output_dir=tmp_path / "a", seed=7, threads=1)
    second = ReproducibilityHarness.run(output_dir=tmp_path / "b", seed=7, threads=1)
    assert first == second


def test_results_do_not_depend_on_the_math_thread_count(*, tmp_path: Path) -> None:
    """scripts/run_sweep.py pins children to one math thread so parallel workers
    don't oversubscribe. That is only a safe optimization if it cannot change the
    numbers -- multi-threaded float reductions can otherwise reassociate."""
    single = ReproducibilityHarness.run(output_dir=tmp_path / "t1", seed=7, threads=1)
    multi = ReproducibilityHarness.run(output_dir=tmp_path / "t4", seed=7, threads=4)
    assert single == multi


def test_different_seeds_produce_different_results(*, tmp_path: Path) -> None:
    """Guards the opposite failure: a run that ignores --seed entirely would pass
    both tests above while making a multi-seed sweep meaningless."""
    first = ReproducibilityHarness.run(output_dir=tmp_path / "s1", seed=1, threads=1)
    second = ReproducibilityHarness.run(output_dir=tmp_path / "s2", seed=2, threads=1)
    assert first != second


# --- Tossing3D -------------------------------------------------------------------
#
# Everything above runs Light Switch, whose whole dynamics is a few lines of numpy.
# Tossing3D is the opposite case and the one actually at risk: MuJoCo physics, PyBullet
# motion planning and four upstream controllers, none of which this project seeds
# directly -- `--seed` reaches them only as the scene seed handed to `env.reset(seed=)`.
# A domain whose runs cannot be regenerated is not a research result, so the guarantee
# is pinned here per-domain rather than assumed to transfer from Light Switch.


@needs_kinder
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_tossing3d_the_same_seed_produces_identical_results(*, tmp_path: Path, seed: int) -> None:
    """Three seeds, not one: a single seed can be reproducible by luck (e.g. every
    task fails identically), which would make the check vacuous."""
    first = ReproducibilityHarness.run_tossing3d(output_dir=tmp_path / "a", seed=seed, threads=1)
    second = ReproducibilityHarness.run_tossing3d(output_dir=tmp_path / "b", seed=seed, threads=1)
    assert first == second


@needs_kinder
def test_tossing3d_results_do_not_depend_on_the_math_thread_count(*, tmp_path: Path) -> None:
    """The Light Switch analogue of this test guards torch; here it guards the whole
    simulator stack, whose BLAS-backed float reductions can reassociate with the thread
    count. run_sweep.py pins children to one math thread, so a sweep and a bare CLI run
    would otherwise be two different experiments."""
    single = ReproducibilityHarness.run_tossing3d(output_dir=tmp_path / "t1", seed=0, threads=1)
    multi = ReproducibilityHarness.run_tossing3d(output_dir=tmp_path / "t4", seed=0, threads=4)
    assert single == multi


# --- Tossing Room (split throws, weight drawn at pickup) --------------------------
#
# The third domain pinned here, and the only one whose ENVIRONMENT itself pre-samples
# anything: `tossingroom` draws each task's pickup weights up front from
# a seed that task carries in its own State. Nothing Light Switch exercises would notice
# if that seed stopped depending on `--seed`, and the failure mode -- every seed of a
# sweep practising on an identical weight sequence -- is invisible in a run's own output,
# so a sweep would look fine and measure one condition ten times. `BallRingEnvironment`'s
# `--noise-seed`, which defaults to 0, is exactly that shape.
#
# Unlike Tossing3D this needs no gate: it is pure numpy and torch, runs on CI, and costs
# about 5 s per invocation. Its two Tossing Room siblings draw only inside `Tasks`, which
# is the same machinery Light Switch already covers, so they are deliberately left out
# rather than tripling this file's runtime to re-assert it.


def test_pickup_weight_the_same_seed_produces_identical_results(*, tmp_path: Path) -> None:
    first = ReproducibilityHarness.run_pickup_weight(output_dir=tmp_path / "a", seed=7, threads=1)
    second = ReproducibilityHarness.run_pickup_weight(output_dir=tmp_path / "b", seed=7, threads=1)
    assert first == second


def test_pickup_weight_results_do_not_depend_on_the_math_thread_count(*, tmp_path: Path) -> None:
    single = ReproducibilityHarness.run_pickup_weight(output_dir=tmp_path / "t1", seed=7, threads=1)
    multi = ReproducibilityHarness.run_pickup_weight(output_dir=tmp_path / "t4", seed=7, threads=4)
    assert single == multi


def test_pickup_weight_different_seeds_produce_different_results(*, tmp_path: Path) -> None:
    """The one that would catch a pre-sampled weight schedule wired to a constant rather
    than to `--seed`."""
    first = ReproducibilityHarness.run_pickup_weight(output_dir=tmp_path / "s1", seed=1, threads=1)
    second = ReproducibilityHarness.run_pickup_weight(output_dir=tmp_path / "s2", seed=2, threads=1)
    assert first != second
