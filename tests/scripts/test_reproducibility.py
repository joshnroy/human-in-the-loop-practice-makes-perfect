"""Locks in the guarantee every sweep depends on: one `--seed` integer fully
determines a run's results.

This is asserted end-to-end through the real CLI rather than by unit-testing each
RNG, because the guarantee is a property of the *whole* pipeline -- task sampling,
skill/parameter sampling, and torch training all draw randomness, and any one of
them reaching for an unseeded global would break reproducibility without breaking
any narrower test. A sweep whose numbers cannot be regenerated months later is not
a research result, so this is worth pinning explicitly.
"""

import json
import os
import subprocess
import sys
from pathlib import Path


class ReproducibilityHarness:
    """A static-method container, never instantiated, same as every other
    business-logic class in this project."""

    @staticmethod
    def run(*, output_dir: Path, seed: int, threads: int) -> dict[str, object]:
        """One short real CLI run; returns its parsed stats.json."""
        output_dir.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(  # noqa: S603
            [
                sys.executable,
                "-m",
                "hitl_pmp.cli",
                "--env",
                "lightswitch",
                "--method",
                "random-skills",
                # grid_size=1 puts the robot in the light's own cell from the
                # start, so TurnOnLight is immediately applicable and this
                # baseline's solve rate genuinely varies with the seed (5/8, 3/8,
                # 1/8 for seeds 1/2/3). At larger grids it scores 0 for every
                # seed, which would make the different-seeds assertion vacuous.
                "--grid-size",
                "1",
                "--num-test-tasks",
                "8",
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
