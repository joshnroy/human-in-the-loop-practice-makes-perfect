"""Covers `WandbResultsWriter` and, above all, the property that makes it safe to turn
on: it is a pure *observer*.

The byte-identity test below is the load-bearing one, for the same reason
`tests/test_competence_log.py`'s is: `stats.json`'s byte-stability is what this repo
uses to prove a change did not alter results, so instrumentation that perturbed it
would destroy the very check that makes instrumentation trustworthy.

**Everything that needs `wandb` skips without it**, gating on
`importlib.util.find_spec("wandb")` exactly as the KINDER-backed tests do -- `wandb` is
an optional extra and CI does not install it. The tests that do *not* need it (off by
default, and the up-front error when the flag is unusable) stay ungated, so CI still
checks that this landing left every unrecorded run alone.

Every run here is forced to `WANDB_MODE=offline`, so nothing touches the network or
needs a credential; W&B's own on-disk offline directory is the artifact under test.
"""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from hitl_pmp.core.metrics.metrics import Metrics
from hitl_pmp.results_writer.run_collision import MissingVariationAxisError
from hitl_pmp.results_writer.types import ExistingRun
from hitl_pmp.results_writer.wandb_writer import WandbResultsWriter

# Two cycles of the oracle on Light Switch: pure numpy, about a second per invocation,
# and it needs no optional dependency of its own. The claim under test is about the
# harness, not about any particular domain's dynamics.
LIGHTSWITCH_ARGS = ("--env", "lightswitch", "--method", "skill-oracle", "--num-test-tasks", "3")

wandb_installed = pytest.mark.skipif(
    importlib.util.find_spec("wandb") is None,
    reason="wandb is an optional extra; CI does not install it",
)


class Args:
    """Resolved namespaces for the writer's own entry point. A static-method container,
    never instantiated, same as every other business-logic class in this project."""

    @staticmethod
    def lightswitch(*, output_dir: Path) -> argparse.Namespace:
        """A complete namespace, not the two attributes a given assertion reads: the
        namer raises on a namespace missing a field it names, which is the point of it,
        so a half-built one would exercise that error path by accident."""
        return argparse.Namespace(
            record_wandb=True,
            output_dir=output_dir,
            env="lightswitch",
            method="skill-oracle",
            seed=0,
            practice_reset_policy="scheduled",
            re_run=False,
        )


class WandbHarness:
    """A static-method container, never instantiated, same as every other
    business-logic class in this project."""

    @staticmethod
    def run(*, output_dir: Path, seed: int = 7, record: bool) -> Path:
        """One short real run through the actual CLI; returns its output dir.

        Through the CLI rather than by calling `MethodRunner` directly, because the
        claim under test is about the whole pipeline: a writer that perturbed the RNG
        would still look inert to a unit test that never ran a policy."""
        output_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(  # noqa: S603
            [
                sys.executable,
                "-m",
                "hitl_pmp.cli",
                *LIGHTSWITCH_ARGS,
                "--seed",
                str(seed),
                "--output-dir",
                str(output_dir),
                *(("--record-wandb",) if record else ()),
            ],
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                # Belt and braces: the writer already defaults to offline, and this
                # makes the test independent of an ambient WANDB_MODE on the machine
                # that runs it.
                "WANDB_MODE": "offline",
                "WANDB_SILENT": "true",
            },
            check=True,
        )
        return output_dir

    @staticmethod
    def offline_run_dir(*, output_dir: Path) -> Path:
        """The single `wandb/offline-run-*` directory W&B wrote inside this run's own
        `--output-dir`, which is where `dir=` puts it -- deliberately not the repo root,
        so a sweep's W&B data lands beside the `stats.json` it describes."""
        candidates = sorted((output_dir / "wandb").glob("offline-run-*"))
        assert len(candidates) == 1, f"expected exactly one offline run, got {candidates}"
        return candidates[0]

    @staticmethod
    def metrics_with_one_sweep(*, num_solved: int = 2, num_total: int = 3) -> Metrics:
        metrics = Metrics()
        metrics.record_evaluation(
            num_online_transitions=17, num_solved=num_solved, num_total=num_total
        )
        return metrics


@pytest.fixture(scope="module")
def recording_off(*, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The oracle with --record-wandb absent: the control arm. Ungated, because a run
    without the flag must work on a machine with no `wandb` at all."""
    return WandbHarness.run(output_dir=tmp_path_factory.mktemp("off"), record=False)


@pytest.fixture(scope="module")
def recording_on(*, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The same run with the writer on, same seed -- so the only difference from
    `recording_off` is the flag."""
    return WandbHarness.run(output_dir=tmp_path_factory.mktemp("on"), record=True)


def test_nothing_is_written_unless_asked(*, recording_off: Path) -> None:
    """Off by default, so every archived run and every open PR's results are untouched
    by this landing -- including the absence of a `wandb/` directory."""
    assert not (recording_off / "wandb").exists()


def test_the_flag_needs_an_output_dir() -> None:
    """Raises up front rather than running for hours and writing nothing, matching
    every other recorder's `open_if_requested`."""
    args = argparse.Namespace(record_wandb=True, output_dir=None)
    with pytest.raises(ValueError, match="--record-wandb needs --output-dir"):
        WandbResultsWriter.open_if_requested(args=args, num_cycles=2)


@pytest.mark.skipif(
    importlib.util.find_spec("wandb") is not None,
    reason="this is the error path for a machine WITHOUT the optional extra",
)
def test_the_flag_fails_loudly_when_wandb_is_not_installed(*, tmp_path: Path) -> None:
    """`config_snapshot.py`'s never-raises policy is right for provenance and wrong
    here: a run that looks logged and is not is the expensive failure. Checked at
    open time, before the run starts, not on the first checkpoint hours in."""
    args = argparse.Namespace(record_wandb=True, output_dir=tmp_path)
    with pytest.raises(ValueError, match="pip install"):
        WandbResultsWriter.open_if_requested(args=args, num_cycles=2)


def test_declines_without_the_flag(*, tmp_path: Path) -> None:
    assert (
        WandbResultsWriter.open_if_requested(
            args=argparse.Namespace(output_dir=tmp_path), num_cycles=2
        )
        is None
    )


def test_the_mode_defaults_to_offline(*, monkeypatch: pytest.MonkeyPatch) -> None:
    """The module docstring's central promise: no credential, no network, no blocking,
    unless the environment explicitly asks otherwise."""
    monkeypatch.delenv("WANDB_MODE", raising=False)
    assert WandbResultsWriter.resolve_mode() == "offline"


def test_an_explicit_mode_wins(*, monkeypatch: pytest.MonkeyPatch) -> None:
    """Watching one long run live stays a launch-time choice rather than a code change."""
    monkeypatch.setenv("WANDB_MODE", "online")
    assert WandbResultsWriter.resolve_mode() == "online"


def test_an_unrecognised_mode_is_rejected_rather_than_silently_downgraded(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A typo'd `WANDB_MODE=onlien` must not fall back to offline: that is exactly the
    "looks logged and is not" failure `open_if_requested` already refuses for a missing
    dependency, discovered only when the data is wanted."""
    monkeypatch.setenv("WANDB_MODE", "onlien")
    with pytest.raises(ValueError, match="WANDB_MODE"):
        WandbResultsWriter.resolve_mode()


@wandb_installed
def test_the_resolved_mode_is_settled_before_the_run_starts(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resolved at open time and carried on the writer, so a bad value fails before the
    run rather than at the first checkpoint hours in -- and so the mode a run used is a
    readable property of the writer rather than a re-read of the environment."""
    monkeypatch.setenv("WANDB_MODE", "disabled")
    writer = WandbResultsWriter.open_if_requested(
        args=Args.lightswitch(output_dir=tmp_path), num_cycles=2
    )
    assert writer is not None
    assert writer.mode == "disabled"


@wandb_installed
def test_recording_leaves_stats_json_byte_identical(
    *, recording_off: Path, recording_on: Path
) -> None:
    """The observer guarantee, asserted the same way the repo proves any change is
    behaviour-neutral. If this fails, the instrumentation is a confound and every
    number measured with it on is suspect."""
    assert (recording_off / "stats.json").read_bytes() == (recording_on / "stats.json").read_bytes()


@wandb_installed
def test_the_other_sidecars_are_untouched_too(*, recording_off: Path, recording_on: Path) -> None:
    """`config_snapshot.json` records the resolved argparse namespace, so it does
    legitimately differ by the flag; nothing else may. `progress.jsonl` carries
    timestamps and is excluded from any byte comparison by design."""
    assert (recording_off / "episode.mp4").exists() == (recording_on / "episode.mp4").exists()
    off_config = json.loads((recording_off / "config_snapshot.json").read_text())
    on_config = json.loads((recording_on / "config_snapshot.json").read_text())
    # ConfigSnapshot stringifies the namespace, so these are "False"/"True".
    assert off_config["args"]["record_wandb"] == "False"
    assert on_config["args"]["record_wandb"] == "True"


@wandb_installed
def test_the_offline_run_lands_inside_the_output_dir(*, recording_on: Path) -> None:
    """`dir=<output-dir>`, so a sweep's W&B data sits beside the `stats.json` it
    describes and inside the gitignored results tree -- rather than in W&B's default
    `./wandb/`, which is the current working directory and would be the repo root."""
    assert WandbHarness.offline_run_dir(output_dir=recording_on).is_dir()


@wandb_installed
def test_the_config_and_summary_reach_wandb(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What is actually handed to W&B, read back off the live `Run` object.

    In-process rather than from the offline directory on disk, because W&B's offline
    mode writes only a binary transaction log (`run-*.wandb`) -- `config.yaml` and
    `wandb-summary.json` are materialized by `wandb sync`, which needs a credential.
    `wandb.run` is the public handle on the current run, so this asserts through
    supported API rather than by decoding that log with W&B internals, which would
    break on any upgrade.

    The `Path`-valued flag is the point of `_as_scalar`: a resolved namespace holds
    `Path`s and enums, and an unstringified one is where a config silently fails to
    serialize."""
    # Gated above; deliberately not imported at module scope, so this file still
    # collects on a machine with no wandb.
    import wandb

    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_SILENT", "true")
    args = argparse.Namespace(
        record_wandb=True,
        output_dir=tmp_path,
        env="lightswitch",
        method="skill-oracle",
        seed=3,
        practice_reset_policy="scheduled",
        re_run=False,
        num_render_checkpoints=1,
        record_full_loop=Path("loop.mp4"),
    )
    writer = WandbResultsWriter.open_if_requested(args=args, num_cycles=2)
    assert writer is not None
    metrics = WandbHarness.metrics_with_one_sweep()
    writer.record_checkpoint(metrics=metrics)

    run = wandb.run
    assert run is not None
    # The *resolved* namespace, so defaulted flags land too -- the same reason
    # config_snapshot.py records vars(args) rather than sys.argv.
    assert run.config["env"] == "lightswitch"
    assert run.config["num_render_checkpoints"] == "1"
    assert run.config["record_full_loop"] == "loop.mp4"
    # Built by the shared namer: the environment, the arm and the seed, so the project's
    # run list distinguishes runs without anyone opening them.
    assert run.name == "lightswitch-skill-oracle-scheduled-seed3"
    assert set(run.tags) == {"lightswitch", "skill-oracle"}

    # The checkpoint's own scalars, which `log` also mirrors into the summary. Read
    # before `close`, because W&B's summary is not readable through public API once the
    # run is finished -- so the post-finish state is checked by the global going None
    # rather than by reading values back out of a closed run.
    assert run.summary["checkpoint"] == 0
    assert run.summary["num_online_transitions"] == 17
    # x/y kept as a pair, never a rate, all the way to the dashboard.
    assert (run.summary["num_solved"], run.summary["num_total"]) == (2, 3)

    writer.close(metrics=metrics)
    assert wandb.run is None, "close() must finish the run, not leave a live handle"


@wandb_installed
def test_no_wandb_run_is_started_before_the_first_checkpoint(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Lazy open, matching every existing recorder: `wandb.init()` spawns a background
    service process, and a run that crashes before its first sweep should not leave an
    empty W&B run behind. `close` on an unopened writer is a no-op."""
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_SILENT", "true")
    writer = WandbResultsWriter.open_if_requested(
        args=Args.lightswitch(output_dir=tmp_path), num_cycles=2
    )
    assert writer is not None
    writer.close(metrics=Metrics())
    assert not (tmp_path / "wandb").exists()


@wandb_installed
def test_offline_never_reaches_for_the_network(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The collision check needs an API to ask, and offline mode has none. Firing one
    anyway would break what offline mode is for: a sweep of ~22 concurrent runs that
    opens no sockets and needs no credential."""
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setattr(
        WandbResultsWriter,
        "_existing_runs",
        staticmethod(lambda **_: pytest.fail("offline must not query the W&B API")),
    )
    assert WandbResultsWriter.open_if_requested(
        args=Args.lightswitch(output_dir=tmp_path), num_cycles=2
    )


@wandb_installed
def test_online_refuses_a_name_that_is_already_another_experiment(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The wiring, with the fetch stubbed: online, the writer asks W&B what it already
    holds under this name and hands it to the checker. The checker's own cases are
    covered in `test_run_collision.py`; what is asserted here is that setup fails
    *before* a writer is returned, so no work happens first."""
    monkeypatch.setenv("WANDB_MODE", "online")
    monkeypatch.setattr(
        WandbResultsWriter,
        "_existing_runs",
        staticmethod(
            lambda *, run_name: (
                ExistingRun(
                    name=run_name,
                    identifier="ezy6q16y",
                    url="https://wandb.ai/josh-princeton/hitl-pmp/runs/ezy6q16y",
                    # A different seed: the same name, a genuinely different experiment.
                    config={"env": "lightswitch", "method": "skill-oracle", "seed": "999"},
                ),
            )
        ),
    )
    with pytest.raises(MissingVariationAxisError, match="seed"):
        WandbResultsWriter.open_if_requested(
            args=Args.lightswitch(output_dir=tmp_path), num_cycles=2
        )


@wandb_installed
def test_online_reports_an_unreachable_api_rather_than_skipping(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A check that silently passes when it could not run is worse than no check. An
    online run with no reachable API is going to fail in `wandb.init` moments later
    anyway, so failing here costs nothing and says something useful."""
    # Gated above; deliberately not imported at module scope, so this file still
    # collects on a machine with no wandb.
    import wandb

    def unreachable(**_: object) -> object:
        raise OSError("network is unreachable")

    monkeypatch.setenv("WANDB_MODE", "online")
    monkeypatch.setattr(wandb, "Api", unreachable)
    with pytest.raises(ValueError, match="could not check"):
        WandbResultsWriter.open_if_requested(
            args=Args.lightswitch(output_dir=tmp_path), num_cycles=2
        )
