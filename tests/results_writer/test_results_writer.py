"""Covers the `ResultsWriter` interface and the static registry the harness iterates.

The properties under test are the ones that make adding a writer to `RESULTS_WRITERS`
a safe act rather than a risky one: every registered writer declines unless its own
flag was passed, the ABC's hooks are no-ops so a writer implements only the boundary
it cares about, and `MethodRunner` really does drive the list rather than naming any
concrete writer. The last of those is why the spy below is monkeypatched *into* the
registry: a test that called the hooks directly would pass even if the harness never
iterated at all.
"""

import argparse
from pathlib import Path
from typing import ClassVar

import pytest

from hitl_pmp.core.metrics.metrics import Metrics
from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
from hitl_pmp.environments.lightswitch.problem import LightSwitchProblem
from hitl_pmp.environments.lightswitch.skill_provider import LightSwitchOracle
from hitl_pmp.environments.lightswitch.tasks import LightSwitchTasks
from hitl_pmp.method_runner import MethodRunner
from hitl_pmp.methods.oracle.skill_oracle_method import SkillOracleMethod
from hitl_pmp.results_writer.registry import RESULTS_WRITERS
from hitl_pmp.results_writer.results_writer import ResultsWriter
from hitl_pmp.results_writer.types import CheckpointScalars, RunSummaryScalars


class SpyResultsWriter(ResultsWriter):
    """Records which hooks the harness fired, and with what.

    A real subclass rather than a mock, so it also exercises the ABC's own
    construction path -- a writer that could not be instantiated as a pydantic model
    would fail here rather than in a later, less obvious place."""

    # ClassVars, not fields: a pydantic field default is deep-copied per instance, so
    # the harness's own instance would append somewhere the test cannot see.
    checkpoints: ClassVar[list[CheckpointScalars]] = []
    summaries: ClassVar[list[RunSummaryScalars]] = []

    @staticmethod
    def open_if_requested(
        *, args: argparse.Namespace, num_cycles: int
    ) -> "SpyResultsWriter | None":
        del num_cycles
        if not getattr(args, "spy", False):
            return None
        return SpyResultsWriter()

    def record_checkpoint(self, *, metrics: Metrics) -> None:
        scalars = CheckpointScalars.from_metrics(metrics=metrics)
        assert scalars is not None
        self.checkpoints.append(scalars)

    def close(self, *, metrics: Metrics) -> None:
        self.summaries.append(RunSummaryScalars.from_metrics(metrics=metrics))


def _args(*, spy: bool = True, output_dir: Path | None = None) -> argparse.Namespace:
    return argparse.Namespace(num_test_tasks=3, output_dir=output_dir, spy=spy)


def _build_problem() -> LightSwitchProblem:
    env = LightSwitchEnvironment()
    return LightSwitchProblem(env=env, tasks=LightSwitchTasks(env=env))


def _run(*, args: argparse.Namespace, num_cycles: int = 2) -> Metrics:
    problem = _build_problem()
    return MethodRunner.run(
        args=args,
        method=SkillOracleMethod(env=problem.env, oracle=LightSwitchOracle(env=problem.env)),
        problem=problem,
        num_cycles=num_cycles,
        max_steps_per_interaction=2,
        renderer=None,
        render_fps=2,
    )


@pytest.fixture
def spy_registry(*, monkeypatch: pytest.MonkeyPatch) -> type[SpyResultsWriter]:
    """Replace the registry the harness reads, and hand back a class whose class-level
    lists start empty -- so one test's hook calls cannot leak into the next."""
    monkeypatch.setattr(SpyResultsWriter, "checkpoints", [])
    monkeypatch.setattr(SpyResultsWriter, "summaries", [])
    monkeypatch.setattr("hitl_pmp.method_runner.RESULTS_WRITERS", (SpyResultsWriter,))
    return SpyResultsWriter


def test_every_registered_writer_declines_when_a_run_gives_it_nothing_to_work_with() -> None:
    """The property that makes the static list safe to grow: a run with no
    `--output-dir` and no flags opens nothing, so adding a writer cannot change any
    existing run. Asserted over the real registry rather than one writer, because the
    guarantee has to hold for whatever the list contains today.

    Deliberately *not* "declines unless its own flag was passed": `RunProgressWriter`
    is always on and has no flag, so `--output-dir` is the condition that has to be
    absent here. `test_run_progress.py` covers the other side of that."""
    bare = argparse.Namespace(output_dir=None)
    assert [
        writer for writer in RESULTS_WRITERS if writer.open_if_requested(args=bare, num_cycles=2)
    ] == []


def test_the_registry_holds_only_results_writer_subclasses() -> None:
    """The harness calls `open_if_requested`/`record_checkpoint`/`close` on whatever is
    listed, so an entry that is not a ResultsWriter would fail at run time, in a run,
    rather than here."""
    assert RESULTS_WRITERS, "the registry is the whole mechanism; an empty one wires nothing"
    for writer in RESULTS_WRITERS:
        assert issubclass(writer, ResultsWriter)


def test_the_hooks_are_no_ops_by_default() -> None:
    """A writer that cares about only one boundary implements only that one. Same
    reason Method.get_practice_policy has a concrete default: a non-learning baseline
    should need no boilerplate."""

    class MinimalWriter(ResultsWriter):
        @staticmethod
        def open_if_requested(
            *, args: argparse.Namespace, num_cycles: int
        ) -> "MinimalWriter | None":
            return None

    writer = MinimalWriter()
    writer.record_checkpoint(metrics=Metrics())
    writer.close(metrics=Metrics())


def test_the_harness_fires_one_checkpoint_hook_per_evaluation_sweep(
    *, spy_registry: type[SpyResultsWriter]
) -> None:
    """num_cycles=2 is three evaluation sweeps -- one before any practice, then one per
    cycle -- so the hook count is the sweep count, and it matches stats.json's own
    `evaluations` length exactly. That equality is what lets a reader join a writer's
    output back to stats.json without guessing an offset."""
    metrics = _run(args=_args(), num_cycles=2)
    assert len(spy_registry.checkpoints) == len(metrics.evaluations) == 3
    assert [scalars.checkpoint for scalars in spy_registry.checkpoints] == [0, 1, 2]


def test_each_checkpoint_hook_reads_the_sweep_that_just_finished(
    *, spy_registry: type[SpyResultsWriter]
) -> None:
    """Fired after `_evaluate` has appended, so the transitions and solved/total a
    writer sees are the ones that sweep was actually recorded under."""
    metrics = _run(args=_args(), num_cycles=2)
    observed = [
        (scalars.num_online_transitions, scalars.num_solved, scalars.num_total)
        for scalars in spy_registry.checkpoints
    ]
    assert observed == [tuple(entry) for entry in metrics.evaluations]


def test_close_is_fired_once_with_the_finished_run(*, spy_registry: type[SpyResultsWriter]) -> None:
    metrics = _run(args=_args(), num_cycles=1)
    assert len(spy_registry.summaries) == 1
    summary = spy_registry.summaries[0]
    assert summary.num_checkpoints == len(metrics.evaluations)
    assert (summary.num_solved, summary.num_total) == metrics.evaluations[-1][1:]


def test_close_is_fired_even_when_the_run_raises(
    *, spy_registry: type[SpyResultsWriter], monkeypatch: pytest.MonkeyPatch
) -> None:
    """In a `finally`, matching LoopRecorder's own placement: a crashed run is exactly
    when someone wants whatever was recorded up to the crash, and an unfinished W&B run
    would otherwise never be flushed."""

    def explode(**_kwargs: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr("hitl_pmp.method_runner.PracticeLoop.run", explode)
    with pytest.raises(RuntimeError, match="boom"):
        _run(args=_args())
    assert len(spy_registry.summaries) == 1


def test_a_writer_that_declines_is_never_driven(*, spy_registry: type[SpyResultsWriter]) -> None:
    """Off by default is the whole safety argument: with the flag absent the spy
    returns None from open_if_requested and no hook fires."""
    _run(args=_args(spy=False))
    assert spy_registry.checkpoints == []
    assert spy_registry.summaries == []


def test_checkpoint_scalars_are_none_before_any_sweep() -> None:
    """`Metrics.evaluations` is empty until the first sweep is recorded, so there is no
    checkpoint to describe -- reported as None rather than as a row of zeros, which a
    reader could not tell apart from a genuine 0/0 sweep."""
    assert CheckpointScalars.from_metrics(metrics=Metrics()) is None
