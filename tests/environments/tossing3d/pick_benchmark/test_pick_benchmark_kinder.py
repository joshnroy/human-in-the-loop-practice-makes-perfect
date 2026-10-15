"""The pick benchmark on the live scene: a state restores exactly, a run keeps its dry
run, ticks and (for a failure) a video, and another pick implementation runs on the
same states."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.pick_benchmark.runner import (
    CurrentPickController,
    PickCaseRunner,
    PickOutcome,
)
from hitl_pmp.environments.tossing3d.pick_benchmark.states import (
    PickCapture,
    PickFamilies,
    PickSpec,
)

pytestmark = [
    pytest.mark.skipif(
        importlib.util.find_spec("kinder") is None
        or importlib.util.find_spec("kinder_models") is None,
        reason="needs KINDER",
    ),
    pytest.mark.xdist_group("tossing3d_pick_benchmark"),
]


@pytest.fixture(name="env", scope="module")
def _env():
    env = Tossing3DEnvironment(scene_bg=True)
    try:
        yield env
    finally:
        env.close()


@pytest.fixture(name="start", scope="module")
def _start(*, env):
    state, note = PickFamilies.build(
        env=env, spec=PickSpec(state_id="start-test", family="start", seed=0, provenance="t")
    )
    assert state is not None, note
    return state


def test_a_captured_state_restores_with_the_same_classification(*, env, start):
    assert start.skill == "PickCube" and start.in_target
    env.reset_to_seed(seed=7)
    PickCapture.restore(env=env, state=start)
    assert PickCapture.atoms(env=env) == start.atoms
    other = start.model_copy(update={"scene_fingerprint": "elsewhere"})
    with pytest.raises(ValueError, match="regenerate the state set"):
        PickCapture.restore(env=env, state=other)


def test_the_current_pick_succeeds_from_the_start_and_logs_every_tick(
    *, env, start, tmp_path: Path
):
    result = PickCaseRunner.run(
        env=env, state=start, implementation=CurrentPickController(), out=tmp_path
    )
    assert result.label == "success", result
    assert result.dry_run_plannable and result.pybullet == "planned"
    run = tmp_path / "runs" / result.case_id
    ticks = np.load(run / "ticks.npz")
    assert ticks["cube_0"].shape[0] == result.controller_steps
    assert json.loads((run / "run.json").read_text())["dry_run"]["plannable"] is True
    assert result.video is None


class RefusesEverything:
    """A stand-in for a new pick controller: never plans, never moves."""

    name = "refuses everything"

    def plan_failure(self, *, env: Tossing3DEnvironment) -> str | None:
        del env
        return "no plan by construction"

    def pick(self, *, env: Tossing3DEnvironment) -> PickOutcome:
        del env
        return PickOutcome(steps=0, terminated=False, error="no plan by construction")


def test_another_implementation_runs_on_the_same_state_and_its_failure_is_filmed(
    *, env, start, tmp_path: Path
):
    result = PickCaseRunner.run(
        env=env, state=start, implementation=RefusesEverything(), out=tmp_path
    )
    assert (result.label, result.dry_run_plannable) == ("no_plan", False)
    assert result.video is not None and (tmp_path / result.video).stat().st_size > 0
    run = json.loads((tmp_path / "runs" / result.case_id / "run.json").read_text())
    assert run["implementation"] == "refuses everything"
