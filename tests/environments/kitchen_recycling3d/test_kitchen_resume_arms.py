"""Pause and resume on the kitchen tossing simulator, for every practice arm.

The Della proof (docs/experiment-logs/2026-11-11-tossing-resume-validation.md) covered
Curtis alone. Each arm here goes through the same three real CLI processes on
`kitchen_recycling3d` and the same comparison:

1. fresh, `--checkpoint-cycles --stop-after-cycles 1`: eval 0, practice 1, eval 1, exit;
2. `--resume`, one more cycle: practice 2, eval 2, exit;
3. reference: fresh, two cycles, never interrupted.

Process 2 must end exactly where process 3 did: method state, samplers, RNGs, simulator
state, accounting, actions and evaluations (`tests/resume_equivalence.py`).

Every arm's argv is the launch's own (`scripts/kitchen_recycling/prepare.py`, which the
v13 manifest was built from), cut down only in size: a few action slots per cycle, one
held-out task, one evaluation process. Each case owns its run directories, so the cases
are independent and share no fixture.
"""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, ClassVar

import pytest

from scripts.kitchen_recycling import prepare
from tests.resume_equivalence import ResumeEquivalence

pytestmark = [
    pytest.mark.skipif(
        importlib.util.find_spec("kinder") is None
        or importlib.util.find_spec("kinder_models") is None,
        reason="needs KINDER",
    ),
]

REPO = Path(__file__).resolve().parents[3]
FORCED_RESET = "reset_to_sampled_start"


class Arms:
    """The arms as launched, by the name the paper gives each."""

    # v13's DP rows (manifest `dp` block): 600 episodes, the toss prior Josh set.
    DP: ClassVar[dict[str, Any]] = {
        "arm": "dp",
        "repeated_stop_limit": 100,
        "dp_episode_budget": 600,
        "dp_initial_competence": ("MoveToTossLocationAndToss=0.1",),
        "dp_initial_learning_rate_for": ("MoveToTossLocationAndToss=0.001",),
    }
    # v13c's DP-period: DP re-planning every period, with #789's rate floor.
    DP_PERIOD: ClassVar[tuple[str, ...]] = (
        "--dp-allocation-schedule",
        "period",
        "--dp-min-learning-rate",
        "0.0001",
    )
    LAUNCHED: ClassVar[dict[str, tuple[dict[str, Any], tuple[str, ...]]]] = {
        "curtis": ({"arm": "curtis"}, ()),
        "ees": ({"arm": "ees"}, ()),
        "dp-once": (DP, ()),
        "dp-period": (DP, DP_PERIOD),
        "always-ask-curtis": ({"arm": "curtis"}, ("--always-ask-reset",)),
        "always-ask-ees": ({"arm": "ees"}, ("--always-ask-reset",)),
        "no-human-curtis": ({"arm": "curtis"}, ("--no-human-reset",)),
        "no-human-ees": ({"arm": "ees"}, ("--no-human-reset",)),
        "expectimax": ({"arm": "expectimax"}, ()),
    }
    # The Della proof's size: budget and planner settings stay the launch's own.
    SMOKE: ClassVar[tuple[str, ...]] = (
        "--checkpoint-cycles",
        "--measure-after-learning-updates",
        "1",
        "2",
        "--measurement-interval-steps",
        "200000",
        "--num-test-tasks",
        "1",
        "--evaluation-workers",
        "1",
        "--evaluation-task-rng",
        "per-task",
        "--evaluation-policy",
        "method",
    )

    @staticmethod
    def cost_file(*, name: str, directory: Path) -> Path:
        """The launch's cost file; an always-ask arm also needs its forced reset priced,
        at the same 5 and one tick as the kitchen's two resets."""
        costs = json.loads((REPO / "scripts/kitchen_recycling/costs.json").read_text())
        if name.startswith("always-ask"):
            costs["human_skills"][FORCED_RESET] = {"cost": {"value": 5}, "duration": {"value": 1}}
        path = directory / "costs.json"
        path.write_text(json.dumps(costs, indent=2) + "\n")
        return path

    @staticmethod
    def argv(*, name: str, directory: Path, action_slots: int) -> list[str]:
        launch, extra = Arms.LAUNCHED[name]
        return [
            *prepare.argv(
                seed=0,
                cost_file=str(Arms.cost_file(name=name, directory=directory)),
                kitchen_config=str(REPO / "scripts/kitchen_recycling/experiment-config.json"),
                practice_step_budget=200000,
                **launch,
            ),
            *extra,
            *Arms.SMOKE,
            "--max-steps-per-interaction",
            str(action_slots),
        ]


class Stages:
    @staticmethod
    def run(*, argv: list[str], output: Path, cycles: int, resume: bool, log: Path) -> None:
        """One real CLI process, ending in a clean pause after `cycles` more cycles."""
        command = [
            sys.executable,
            "-m",
            "hitl_pmp.cli",
            *argv,
            "--output-dir",
            str(output),
            "--stop-after-cycles",
            str(cycles),
            *(("--resume", str(output)) if resume else ()),
        ]
        threads = (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "BLIS_NUM_THREADS",
        )
        # The proof's process environment: one math thread, and a fixed hash seed.
        environment = {**os.environ, **dict.fromkeys(threads, "1"), "PYTHONHASHSEED": "0"}
        with log.open("w") as stream:
            result = subprocess.run(  # noqa: S603
                command,
                cwd=REPO,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=1500,
            )
        assert result.returncode == 0, f"{log.name}:\n{log.read_text()[-4000:]}"
        status = json.loads((output / "status.json").read_text())
        assert status["phase"] == "paused", status

    @staticmethod
    def forced_resets(*, output: Path) -> dict[int, int]:
        """How many forced resets each practice cycle saw."""
        counts: dict[int, int] = {}
        for row in ResumeEquivalence.events(output=output):
            if row["event"] == "human" and row["skill"] == FORCED_RESET:
                counts[row["cycle"]] = counts.get(row["cycle"], 0) + 1
        return counts


CASES = [
    pytest.param("curtis", 2, id="curtis"),
    pytest.param("ees", 2, id="ees"),
    pytest.param("dp-once", 2, id="dp-once"),
    pytest.param("dp-period", 2, id="dp-period"),
    # A forced reset follows the toss, the second action: four slots put one reset
    # before the pause and one after it.
    pytest.param("always-ask-curtis", 4, id="always-ask-curtis"),
    pytest.param("always-ask-ees", 4, id="always-ask-ees"),
    pytest.param("no-human-curtis", 2, id="no-human-curtis"),
    pytest.param("no-human-ees", 2, id="no-human-ees"),
    pytest.param("expectimax", 2, id="expectimax"),
]


@pytest.mark.parametrize(("name", "action_slots"), CASES)
def test_a_resumed_run_ends_where_an_uninterrupted_one_does(
    *, tmp_path: Path, name: str, action_slots: int
) -> None:
    if name.startswith("dp") and importlib.util.find_spec("gurobipy") is None:
        pytest.skip("the DP arm is launched with --dp-solver gurobi; needs gurobipy")
    argv = Arms.argv(name=name, directory=tmp_path, action_slots=action_slots)
    resumed, reference, boundary = (tmp_path / d for d in ("resumed", "reference", "boundary"))
    boundary.mkdir()

    Stages.run(argv=argv, output=resumed, cycles=1, resume=False, log=tmp_path / "cycle1.log")
    for filename in ("checkpoint.pickle", *ResumeEquivalence.RECORDER_FILES):
        if (resumed / filename).exists():
            shutil.copy2(resumed / filename, boundary / filename)
    before_pause = Stages.forced_resets(output=resumed)
    Stages.run(argv=argv, output=resumed, cycles=1, resume=True, log=tmp_path / "cycle2.log")
    Stages.run(argv=argv, output=reference, cycles=2, resume=False, log=tmp_path / "reference.log")

    report = ResumeEquivalence.compare(
        resumed=resumed, reference=reference, first_boundary=boundary
    )
    (tmp_path / "comparison.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    assert report["insensitive"] == []
    assert report["failed_structural"] == []
    assert report["mismatched"] == []
    if name.startswith("always-ask"):
        # A reset count that carries over needs a reset on each side of the pause.
        resets = Stages.forced_resets(output=resumed)
        assert before_pause.get(0, 0) >= 1, before_pause
        assert resets.get(1, 0) >= 1, resets
        assert resets == Stages.forced_resets(output=reference)
