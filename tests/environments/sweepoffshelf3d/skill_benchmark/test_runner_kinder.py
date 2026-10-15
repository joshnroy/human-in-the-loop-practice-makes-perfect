"""Benchmark states and runs on the live scene: a state restores exactly (a cube in the
hand stays in it), a refused scene is refused, and a run leaves its trace, per-step log
and -- for a failure -- a video."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from hitl_pmp.environments.sweepoffshelf3d.environment import SweepOffShelf3DEnvironment
from hitl_pmp.environments.sweepoffshelf3d.scripted_skills import SkillFailed
from hitl_pmp.environments.sweepoffshelf3d.skill_benchmark.runner import CaseRunner
from hitl_pmp.environments.sweepoffshelf3d.skill_benchmark.states import (
    Setups,
    StateCapture,
    StateFamilies,
    StateSpec,
)
from hitl_pmp.environments.sweepoffshelf3d.skill_benchmark.types import BenchmarkCase

pytestmark = [
    pytest.mark.skipif(
        importlib.util.find_spec("kinder") is None
        or importlib.util.find_spec("kinder_models") is None,
        reason="needs KINDER",
    ),
    pytest.mark.xdist_group("sweepoffshelf3d_skill_benchmark_runner"),
]


@pytest.fixture(name="env", scope="module")
def _env():
    env = SweepOffShelf3DEnvironment(scene_bg=True)
    try:
        yield env
    finally:
        env.close()


@pytest.fixture(name="cube_in_hand", scope="module")
def _cube_in_hand(*, env):
    pick = np.array([4.0, 2.0, 0.0, 0.0])  # PickCubeFromFloor, cube_2
    spec = StateSpec(
        state_id="cube-in-hand-test",
        family="cube_in_hand",
        seed=0,
        cases=[BenchmarkCase(skill="PlaceCubeInBox", cube="cube_2")],
        provenance="test",
        setup=Setups.chain(
            Setups.cube_at(x=0.95, y=0.45, z=0.0105, cube="cube_2"), Setups.skill(action=pick)
        ),
    )
    state, note = StateFamilies.build(env=env, spec=spec)
    assert state is not None, note
    return state


def test_a_captured_state_restores_with_the_cube_still_in_the_hand(*, env, cube_in_hand):
    assert "HoldingCube(robot,cube_2)" in cube_in_hand.atoms
    StateCapture.restore(env=env, state=cube_in_hand)
    assert np.allclose(env.backend().data().qpos, cube_in_hand.qpos)
    Setups.settle(env=env, steps=50)
    assert env.backend().held(name="cube_2")


def test_a_state_from_another_scene_is_refused(*, env, cube_in_hand):
    other = cube_in_hand.model_copy(update={"scene_fingerprint": "kinder/Other|nq=1"})
    with pytest.raises(ValueError, match="regenerate the state set"):
        StateCapture.restore(env=env, state=other)


def test_a_run_leaves_its_trace_and_step_log(*, env, cube_in_hand, tmp_path: Path):
    case = cube_in_hand.cases[0]
    result = CaseRunner.run(env=env, state=cube_in_hand, case=case, out=tmp_path)
    assert result.label == "success", result
    assert result.pybullet == "planned"
    run = tmp_path / "runs" / result.case_id
    steps = np.load(run / "steps.npz")
    assert steps["qpos"].shape == (result.controller_steps, len(cube_in_hand.qpos))
    assert "place_in_box" in set(steps["label_names"].tolist())
    assert (run / "trace.json").exists() and not (run / "video.mp4").exists()
    assert result.video is None


def test_a_failure_gets_a_video(*, env, cube_in_hand, tmp_path: Path, monkeypatch):
    def fails(self, *, cube):  # noqa: PLR0917 -- replaces a bound method
        for _ in range(40):
            self.env_step(action=np.zeros(11), label="place_in_box")
        raise SkillFailed("place_cube_in_box: forced for the test")

    from hitl_pmp.environments.sweepoffshelf3d.kinder_skills import KinderSkills

    monkeypatch.setattr(KinderSkills, "place_cube_in_box", fails)
    result = CaseRunner.run(env=env, state=cube_in_hand, case=cube_in_hand.cases[0], out=tmp_path)
    assert result.label == "execution_failed"
    assert result.video is not None
    assert (tmp_path / result.video).stat().st_size > 0


def test_an_inapplicable_case_is_invalid_not_a_failure(*, env, cube_in_hand, tmp_path: Path):
    result = CaseRunner.run(
        env=env, state=cube_in_hand, case=BenchmarkCase(skill="PickWiper"), out=tmp_path
    )
    assert result.label == "invalid"


def test_a_failure_that_never_moved_still_gets_its_log_and_a_video(
    *, env, cube_in_hand, tmp_path: Path, monkeypatch
):
    def refuses(self, *, cube):  # noqa: PLR0917 -- replaces a bound method
        raise SkillFailed("place_cube_in_box: no placement planned")

    from hitl_pmp.environments.sweepoffshelf3d.kinder_skills import KinderSkills

    monkeypatch.setattr(KinderSkills, "place_cube_in_box", refuses)
    result = CaseRunner.run(env=env, state=cube_in_hand, case=cube_in_hand.cases[0], out=tmp_path)
    assert (result.label, result.controller_steps) == ("no_plan", 0)
    steps = np.load(tmp_path / "runs" / result.case_id / "steps.npz")
    assert steps["qpos"].shape == (0, len(cube_in_hand.qpos))
    assert result.video is not None and (tmp_path / result.video).stat().st_size > 0
