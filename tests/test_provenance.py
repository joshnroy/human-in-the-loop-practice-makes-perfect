import argparse
from pathlib import Path

import numpy as np
import torch

from hitl_pmp.provenance import RunProvenance, resolve_reproducibility_scope


def _args(**kwargs: object) -> argparse.Namespace:
    return argparse.Namespace(**kwargs)


def test_collect_records_the_numerical_stack() -> None:
    """The versions that were varying between machines are exactly the ones a
    reader needs in order to rule the environment in or out."""
    provenance = RunProvenance.collect(args=_args(seed=0), fd_exec_path=None)
    assert provenance.torch_version == torch.__version__
    assert provenance.numpy_version == np.__version__
    assert provenance.python_version
    assert provenance.platform_machine


def test_collect_records_resolved_args_including_defaulted_ones() -> None:
    """The motivating gap: --sampler-max-train-iters had its default changed
    mid-project, so a run that never passed it explicitly could not have its
    iteration count recovered afterwards."""
    provenance = RunProvenance.collect(
        args=_args(seed=3, sampler_max_train_iters=10000), fd_exec_path=None
    )
    assert provenance.args["seed"] == "3"
    assert provenance.args["sampler_max_train_iters"] == "10000"


def test_collect_marks_fast_downward_unset_when_it_is() -> None:
    provenance = RunProvenance.collect(args=_args(), fd_exec_path=None)
    assert provenance.fast_downward_commit == "unset"


def test_git_degrades_to_unknown_outside_a_repository(*, tmp_path: Path) -> None:
    """Provenance is diagnostic metadata -- a missing .git must never be the
    reason a run that already burned hours of compute fails at the write step."""
    assert RunProvenance._git(repo=tmp_path, args=["rev-parse", "HEAD"]) == "unknown"


def test_collect_still_succeeds_when_fast_downward_path_is_not_a_repository(
    *, tmp_path: Path
) -> None:
    provenance = RunProvenance.collect(args=_args(), fd_exec_path=str(tmp_path))
    assert provenance.fast_downward_commit == "unknown"


def test_provenance_round_trips_through_json() -> None:
    """It is written to disk and read back by whoever is diagnosing a mismatch."""
    original = RunProvenance.collect(args=_args(seed=1), fd_exec_path=None)
    assert RunProvenance.model_validate_json(original.model_dump_json()) == original


def test_reproducibility_scope_does_not_overclaim() -> None:
    """The claim this project used to make -- that a single --seed "fully
    determines a run" -- is true only per-machine. Ten Tossing Room seeds re-run
    from the same commit with the same flags on a second machine disagreed on
    four of them. The wording must not overclaim a cause either."""
    scope = resolve_reproducibility_scope()
    assert "same machine" in scope
    assert "NOT known to be portable" in scope
