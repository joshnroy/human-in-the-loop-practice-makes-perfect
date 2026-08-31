"""Covers the per-checkpoint skill-competence record and, above all, the property
that makes it safe to turn on: it is a pure *observer*.

The byte-identity test below is the load-bearing one, for the same reason
`tests/test_sampler_draws.py`'s is: `stats.json`'s byte-stability is what this repo
uses to prove a change did not alter results, so instrumentation that wrote into it
would destroy the very check that makes instrumentation trustworthy. Hence a sibling
file (`hitl_pmp/competence_log.py`'s `CompetenceLogRecorder`), the same shape as
`SamplerDrawRecorder`.

**Why the runs are module-scoped fixtures.** Every test here asserts a different
property of the *same* three CLI runs, and `--seed` fully determines a run, so a
per-test invocation re-derived byte-identical output six more times. Sharing them
costs no assertion: what each test checks is unchanged, only how often the subprocess
that produces the evidence is paid for. The contract that makes it safe is that these
directories are **read-only** to every test below -- a test that wrote into one would
leak into its siblings, so anything needing a mutable directory must take `tmp_path`
and run its own.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

# Two cycles of real EES on Tossing Room: pure numpy + torch, ~5 s per invocation, and
# it runs on CI. See tests/test_sampler_draws.py for why this domain/size was chosen.
TOSSINGROOM_ARGS = (
    "--num-test-tasks",
    "4",
    "--num-cycles",
    "2",
    "--max-steps-per-interaction",
    "20",
)


class CompetenceHarness:
    """A static-method container, never instantiated, same as every other
    business-logic class in this project."""

    @staticmethod
    def run(*, output_dir: Path, seed: int = 7, record: bool, method: str = "ees") -> Path:
        """One short real run through the actual CLI; returns its output dir.

        Through the CLI rather than by calling `MethodRunner` directly, because the
        claim under test is about the whole pipeline: a recorder that perturbed the
        RNG would still look inert to a unit test that never ran a policy."""
        output_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(  # noqa: S603
            [
                sys.executable,
                "-m",
                "hitl_pmp.cli",
                "--env",
                "tossingroom",
                "--method",
                method,
                *TOSSINGROOM_ARGS,
                "--seed",
                str(seed),
                "--output-dir",
                str(output_dir),
                *(("--record-skill-competence",) if record else ()),
            ],
            capture_output=True,
            text=True,
            env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
            check=True,
        )
        return output_dir

    @staticmethod
    def records(*, output_dir: Path) -> list[dict[str, object]]:
        text = (output_dir / "competence_log.jsonl").read_text()
        return [json.loads(line) for line in text.splitlines() if line.strip()]


@pytest.fixture(scope="module")
def recording_off(*, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """EES with --record-skill-competence absent: the control arm."""
    return CompetenceHarness.run(output_dir=tmp_path_factory.mktemp("off"), record=False)


@pytest.fixture(scope="module")
def recording_on(*, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """EES with the recorder on, same seed -- so the only difference from
    `recording_off` is the flag."""
    return CompetenceHarness.run(output_dir=tmp_path_factory.mktemp("on"), record=True)


@pytest.fixture(scope="module")
def random_skills_recording_on(*, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A method that tracks no competence model at all, with the flag on."""
    return CompetenceHarness.run(
        output_dir=tmp_path_factory.mktemp("random_skills_on"),
        record=True,
        method="random-skills",
    )


def test_recording_leaves_stats_json_byte_identical(
    *, recording_off: Path, recording_on: Path
) -> None:
    """The observer guarantee, asserted the same way the repo proves any change is
    behaviour-neutral. If this fails, the instrumentation is a confound and every
    number measured with it on is suspect."""
    assert (recording_off / "stats.json").read_bytes() == (recording_on / "stats.json").read_bytes()


def test_nothing_is_written_unless_asked(*, recording_off: Path) -> None:
    """Off by default, so every archived run and every open PR's results are
    untouched by this landing."""
    assert not (recording_off / "competence_log.jsonl").exists()


def test_a_non_tracking_method_writes_nothing_even_when_asked(
    *, random_skills_recording_on: Path
) -> None:
    """random-skills tracks no competence model (Method.current_competences defaults
    to {}), so --record-skill-competence has nothing to write and must not leave a
    stray empty file behind -- the same lazy-open contract sampler_draws.py holds."""
    assert not (random_skills_recording_on / "competence_log.jsonl").exists()


def test_every_record_is_one_json_object_per_line(*, recording_on: Path) -> None:
    """JSONL rather than one JSON document, so a run killed mid-flight still leaves
    a readable file up to its last flushed record."""
    records = CompetenceHarness.records(output_dir=recording_on)
    assert records, "a 2-cycle EES run on Tossing Room practices at least one skill"
    for record in records:
        assert set(record) == {
            "timestamp",
            "elapsed_seconds",
            "checkpoint",
            "num_online_transitions",
            "skill",
            "objects",
            "competence",
        }
        assert isinstance(record["competence"], float)
        assert 0.0 <= record["competence"] <= 1.0


def test_checkpoints_and_transitions_match_stats_json_evaluations(*, recording_on: Path) -> None:
    """The whole point: the sidecar's checkpoints line up with stats.json's own
    evaluation sweeps, so a reader can join the two without guessing an offset."""
    records = CompetenceHarness.records(output_dir=recording_on)
    stats = json.loads((recording_on / "stats.json").read_text())
    evaluations = stats["evaluations"]

    seen_checkpoints = {int(record["checkpoint"]) for record in records}
    # num_cycles=2 means 3 evaluation sweeps (0, 1, 2); the first has practiced
    # nothing yet, so it may legitimately be absent -- but no checkpoint beyond the
    # last sweep, and none before it, may appear.
    assert seen_checkpoints <= set(range(len(evaluations)))

    transitions_by_checkpoint = {i: entry[0] for i, entry in enumerate(evaluations)}
    for record in records:
        checkpoint = int(record["checkpoint"])
        assert record["num_online_transitions"] == transitions_by_checkpoint[checkpoint]


def test_a_skill_never_practiced_by_the_first_checkpoint_is_simply_absent(
    *, recording_on: Path
) -> None:
    """Checkpoint 0 runs before any practice, so EesMethod's competence-model dict is
    still empty -- consistent with competence_model()'s own lazy creation, nothing
    invented here. That means checkpoint 0 contributes no rows at all, not rows with
    a made-up default value."""
    records = CompetenceHarness.records(output_dir=recording_on)
    assert all(int(record["checkpoint"]) != 0 for record in records)
