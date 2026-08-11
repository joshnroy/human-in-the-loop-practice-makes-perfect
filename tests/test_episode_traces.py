"""Covers the per-step episode-trace record and, above all, the property that makes
it safe to turn on: it is a pure *observer*.

The byte-identity test below is the load-bearing one, for the same reason
`tests/test_sampler_draws.py`'s is: `stats.json`'s byte-stability is what this repo
uses to prove a change did not alter results, so instrumentation that wrote into it
would destroy the very check that makes instrumentation trustworthy. Hence a sibling
file (`hitl_pmp/episode_traces.py`'s `EpisodeTraceRecorder`), the same shape as
`SamplerDrawRecorder`/`CompetenceLogRecorder`.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from hitl_pmp.core.method.types import EpisodeTrace
from hitl_pmp.core.problem.environment.types import Object, State, Type
from hitl_pmp.episode_traces import EpisodeTraceRecorder

# Light Switch + skill-oracle: one evaluation sweep, no practice, and the oracle
# always solves -- so every test task's episode is guaranteed to take at least one
# real step (a grid_size of 3 puts the robot at least one move away from the light).
# Pure numpy, ms-level, and it runs on CI.
LIGHTSWITCH_ORACLE_ARGS = ("--grid-size", "3", "--num-test-tasks", "4")


class TraceHarness:
    """A static-method container, never instantiated, same as every other
    business-logic class in this project."""

    @staticmethod
    def run(
        *,
        output_dir: Path,
        seed: int = 7,
        record: bool,
        method: str = "skill-oracle",
        env_args: tuple[str, ...] = LIGHTSWITCH_ORACLE_ARGS,
    ) -> Path:
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
                "lightswitch",
                "--method",
                method,
                *env_args,
                "--seed",
                str(seed),
                "--output-dir",
                str(output_dir),
                *(("--record-episode-traces",) if record else ()),
            ],
            capture_output=True,
            text=True,
            env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
            check=True,
        )
        return output_dir

    @staticmethod
    def steps(*, output_dir: Path) -> list[dict[str, object]]:
        text = (output_dir / "episode_traces.jsonl").read_text()
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def test_recording_leaves_stats_json_byte_identical(*, tmp_path: Path) -> None:
    """The observer guarantee, asserted the same way the repo proves any change is
    behaviour-neutral. If this fails, the instrumentation is a confound and every
    number measured with it on is suspect."""
    off = TraceHarness.run(output_dir=tmp_path / "off", record=False)
    on = TraceHarness.run(output_dir=tmp_path / "on", record=True)
    assert (off / "stats.json").read_bytes() == (on / "stats.json").read_bytes()


def test_nothing_is_written_unless_asked(*, tmp_path: Path) -> None:
    """Off by default, so every archived run and every open PR's results are
    untouched by this landing."""
    off = TraceHarness.run(output_dir=tmp_path / "off", record=False)
    assert not (off / "episode_traces.jsonl").exists()


def test_every_step_is_one_json_object_per_line(*, tmp_path: Path) -> None:
    """JSONL rather than one JSON document, so a run killed mid-flight still leaves
    a readable file up to its last flushed episode."""
    on = TraceHarness.run(output_dir=tmp_path / "on", record=True)
    steps = TraceHarness.steps(output_dir=on)
    assert steps, "skill-oracle on a grid_size=3 board takes at least one real step"
    for step in steps:
        assert set(step) == {
            "checkpoint",
            "num_online_transitions",
            "task_index",
            "goal",
            "step_index",
            "solved",
            "action_label",
            "action",
            "state",
        }
        assert isinstance(step["action"], list)
        assert all(isinstance(value, float) for value in step["action"])
        assert isinstance(step["state"], dict) and step["state"]
        assert all(isinstance(value, float) for value in step["state"].values())


def test_step_indices_are_zero_based_and_contiguous_per_episode(*, tmp_path: Path) -> None:
    """Each (checkpoint, task_index) episode's steps must read 0, 1, 2, ... with no
    gaps -- a reader counts steps-to-goal by taking max(step_index) + 1."""
    on = TraceHarness.run(output_dir=tmp_path / "on", record=True)
    steps = TraceHarness.steps(output_dir=on)
    by_episode: dict[tuple[int, int], list[int]] = {}
    for step in steps:
        key = (int(step["checkpoint"]), int(step["task_index"]))
        by_episode.setdefault(key, []).append(int(step["step_index"]))
    for indices in by_episode.values():
        assert sorted(indices) == list(range(len(indices)))


def test_solved_is_constant_across_every_step_of_one_episode(*, tmp_path: Path) -> None:
    """`solved` is the whole episode's outcome, not a per-step goal check -- every
    line of one episode must agree, matching TaskOutcome's own per-task granularity."""
    on = TraceHarness.run(output_dir=tmp_path / "on", record=True)
    steps = TraceHarness.steps(output_dir=on)
    by_episode: dict[tuple[int, int], set[bool]] = {}
    for step in steps:
        key = (int(step["checkpoint"]), int(step["task_index"]))
        by_episode.setdefault(key, set()).add(bool(step["solved"]))
    for solved_values in by_episode.values():
        assert len(solved_values) == 1


def test_a_zero_step_episode_contributes_no_rows(*, tmp_path: Path) -> None:
    """A task whose initial state already satisfies its goal takes no action at
    all, so its EpisodeTrace.actions is empty -- record_episode must not write a
    stray row, nor leave a stray empty file behind, matching
    SamplerDrawRecorder/CompetenceLogRecorder's own lazy-open contract.

    A direct unit test against the recorder rather than a full CLI run: no
    Light Switch configuration reliably produces an already-satisfied test task
    (the oracle always takes at least one action), so this exercises the same
    "empty actions" path record_episode itself branches on."""
    block = Type(name="block", feature_names=("x",))
    obj = Object(name="block1", type=block)
    state = State(data={obj: np.array([0.0])})
    trace = EpisodeTrace(states=[state], actions=[])

    recorder = EpisodeTraceRecorder(output_path=tmp_path / "episode_traces.jsonl")
    recorder.record_episode(
        checkpoint=0,
        num_online_transitions=0,
        task_index=0,
        goal="TurnOnLight()",
        solved=True,
        trace=trace,
    )
    assert not (tmp_path / "episode_traces.jsonl").exists()


def test_checkpoints_match_stats_json_evaluations(*, tmp_path: Path) -> None:
    """The whole point: the sidecar's checkpoints line up with stats.json's own
    evaluation sweeps, so a reader can join the two without guessing an offset."""
    on = TraceHarness.run(output_dir=tmp_path / "on", record=True)
    steps = TraceHarness.steps(output_dir=on)
    stats = json.loads((on / "stats.json").read_text())
    evaluations = stats["evaluations"]

    seen_checkpoints = {int(step["checkpoint"]) for step in steps}
    assert seen_checkpoints <= set(range(len(evaluations)))

    transitions_by_checkpoint = {i: entry[0] for i, entry in enumerate(evaluations)}
    for step in steps:
        checkpoint = int(step["checkpoint"])
        assert step["num_online_transitions"] == transitions_by_checkpoint[checkpoint]


def test_solved_episodes_agree_with_task_outcomes(*, tmp_path: Path) -> None:
    """A second, independent cross-check against stats.json (like
    test_sampler_draws.py's consultation-pool check): the sidecar's own `solved`
    per (checkpoint, task_index) must match TaskOutcome's, or a trace-length figure
    built from this file could disagree with the success rates published beside it."""
    on = TraceHarness.run(output_dir=tmp_path / "on", record=True)
    steps = TraceHarness.steps(output_dir=on)
    stats = json.loads((on / "stats.json").read_text())

    solved_by_episode: dict[tuple[int, int], bool] = {}
    for step in steps:
        key = (int(step["checkpoint"]), int(step["task_index"]))
        solved_by_episode[key] = bool(step["solved"])

    breakdowns = stats.get("breakdowns")
    assert breakdowns, "skill-oracle records outcomes for every evaluated task"
    for checkpoint, breakdown in enumerate(breakdowns):
        for outcome in breakdown["outcomes"]:
            key = (checkpoint, int(outcome["task_index"]))
            if key in solved_by_episode:
                assert solved_by_episode[key] == outcome["solved"]
