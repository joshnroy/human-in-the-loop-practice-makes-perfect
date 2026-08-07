"""Covers the per-draw sampler record and, above all, the property that makes it
safe to turn on: it is a pure *observer*.

The byte-identity test below is the load-bearing one. `stats.json`'s byte-stability
is what this repo uses to prove a change did not alter results -- PR #146 leaned on
exactly that property to show the Tossing Room rename was behaviour-neutral -- so
instrumentation that wrote into `stats.json` would destroy the very check that makes
instrumentation trustworthy. Hence a sibling file, the same reasoning `timing.json`
and `config_snapshot.json` already follow, and hence a test that asserts the bytes
rather than the parsed dict: a reordered key or a changed float repr is exactly the
kind of drift a dict comparison hides.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

# Two cycles of real EES on Tossing Room: pure numpy + torch, ~5 s per invocation, and
# it runs on CI. Its `ThrowTrash`/`ThrowRecycling` skills are `param_dim=1`, so a run
# this short still consults a sampler many times -- which is what these tests need.
# Tossing3D would exercise the same code path through a MuJoCo simulator at minutes per
# invocation, and adds nothing this file asserts.
TOSSINGROOM_ARGS = (
    "--num-test-tasks",
    "4",
    "--num-cycles",
    "2",
    "--max-steps-per-interaction",
    "20",
)


class DrawHarness:
    """A static-method container, never instantiated, same as every other
    business-logic class in this project."""

    @staticmethod
    def run(*, output_dir: Path, seed: int = 7, record: bool) -> Path:
        """One short real EES run through the actual CLI; returns its output dir.

        Through the CLI rather than by calling `MethodRunner` directly, because the
        claim under test is about the whole pipeline: a recorder that perturbed the
        RNG would still look inert to a unit test that never ran a policy.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(  # noqa: S603
            [
                sys.executable,
                "-m",
                "hitl_pmp.cli",
                "--env",
                "tossingroom",
                "--method",
                "ees",
                *TOSSINGROOM_ARGS,
                "--seed",
                str(seed),
                "--output-dir",
                str(output_dir),
                *(("--record-sampler-draws",) if record else ()),
            ],
            capture_output=True,
            text=True,
            env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
            check=True,
        )
        return output_dir

    @staticmethod
    def draws(*, output_dir: Path) -> list[dict[str, object]]:
        text = (output_dir / "sampler_draws.jsonl").read_text()
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def test_recording_leaves_stats_json_byte_identical(*, tmp_path: Path) -> None:
    """The observer guarantee, asserted the same way the repo proves any change is
    behaviour-neutral. If this fails, the instrumentation is a confound and every
    number measured with it on is suspect."""
    off = DrawHarness.run(output_dir=tmp_path / "off", record=False)
    on = DrawHarness.run(output_dir=tmp_path / "on", record=True)
    assert (off / "stats.json").read_bytes() == (on / "stats.json").read_bytes()


def test_nothing_is_written_unless_asked(*, tmp_path: Path) -> None:
    """Off by default, so every archived run and every open PR's results are
    untouched by this landing."""
    off = DrawHarness.run(output_dir=tmp_path / "off", record=False)
    assert not (off / "sampler_draws.jsonl").exists()


def test_every_draw_is_one_json_object_per_line(*, tmp_path: Path) -> None:
    """JSONL rather than one JSON document, so a run killed mid-flight still leaves
    a readable file up to its last flushed draw -- a 100-cycle run is hours, and a
    record only readable after a clean exit would be useless exactly when it matters."""
    on = DrawHarness.run(output_dir=tmp_path / "on", record=True)
    draws = DrawHarness.draws(output_dir=on)
    assert draws, "a 2-cycle EES run consults a param_dim=1 sampler at least once"
    for draw in draws:
        assert set(draw) == {
            "cycle",
            "skill",
            "consultation",
            "success",
            "params",
            "achieved",
        }


def test_draws_carry_the_chosen_parameters_and_the_post_action_state(*, tmp_path: Path) -> None:
    """The two things #133 could not answer from integer counters: which parameter
    the sampler actually chose, and what the environment did with it."""
    on = DrawHarness.run(output_dir=tmp_path / "on", record=True)
    draws = DrawHarness.draws(output_dir=on)
    throws = [d for d in draws if str(d["skill"]).startswith("Throw")]
    assert throws, "Tossing Room's throw skills are the param_dim=1 ones"
    for draw in throws:
        params = draw["params"]
        assert isinstance(params, list) and len(params) == 1
        assert all(isinstance(p, float) for p in params)
        achieved = draw["achieved"]
        # Keyed "<object>.<feature>" over the ground skill's own objects, so a
        # domain-agnostic reader can recover any feature the skill touched.
        assert isinstance(achieved, dict) and achieved
        assert all("." in key for key in achieved)


def test_consultation_pools_match_the_stats_json_tally(*, tmp_path: Path) -> None:
    """The per-draw file and `stats.json`'s counters are two views of the same events,
    so they must agree. This is what stops the new file drifting into a parallel,
    silently-different account of the run -- the failure that would make a trajectory
    figure disagree with the success rates published beside it."""
    on = DrawHarness.run(output_dir=tmp_path / "on", record=True)
    draws = DrawHarness.draws(output_dir=on)
    stats = json.loads((on / "stats.json").read_text())

    tallied: dict[str, int] = {}
    for window in stats["practice_outcomes_per_cycle"]:
        for skill_name, tally in window.items():
            # Attempts on which a sampler was actually consulted: every pool except
            # NO_SAMPLER, which is `param_dim == 0` and so never reaches a sampler at
            # all. That is exactly the set `execute_ground_skill` builds a record for.
            counted = tally["num_attempts"] - tally["num_unparameterized_attempts"]
            tallied[skill_name] = tallied.get(skill_name, 0) + counted

    recorded: dict[str, int] = {}
    for draw in draws:
        name = str(draw["skill"])
        recorded[name] = recorded.get(name, 0) + 1

    assert recorded == {name: count for name, count in tallied.items() if count}
