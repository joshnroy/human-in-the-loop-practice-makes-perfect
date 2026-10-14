"""SweepOffShelf3D runs the step protocol: counted control steps, the budget cutting a
skill off mid-execution, its two human skills, KINDER's own goal verdict and a run
recording."""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from hitl_pmp.cli import STEP_PROTOCOL_ENVIRONMENTS
from hitl_pmp.core.control_steps import ControlStepLimitReached
from hitl_pmp.environments.sweepoffshelf3d.cli import SweepOffShelf3DCli
from hitl_pmp.environments.sweepoffshelf3d.environment import SKILL_IDS
from hitl_pmp.environments.sweepoffshelf3d.step_protocol import SweepOffShelf3DStepProtocol
from hitl_pmp.recording.run_record.types import Phase

_NEEDS_KINDER = pytest.mark.skipif(
    importlib.util.find_spec("kinder") is None, reason="needs KINDER"
)


def _args(*, tmp_path: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    SweepOffShelf3DCli.add_arguments(parser=parser)
    args = parser.parse_args([])
    args.seed = 0
    args.method = "ees"
    args.output_dir = str(tmp_path)
    return args


def test_the_shelf_is_registered_for_the_step_protocol() -> None:
    assert STEP_PROTOCOL_ENVIRONMENTS["sweepoffshelf3d"] is SweepOffShelf3DStepProtocol


def test_its_human_skills_are_charged_and_logged_by_their_own_names() -> None:
    assert SweepOffShelf3DStepProtocol.human_skill_names() == {
        "reset_cubes": "reset_cubes",
        "reset_wiper": "reset_wiper",
    }


def test_evaluation_offers_no_human_skill(*, tmp_path: Path) -> None:
    problem = SweepOffShelf3DStepProtocol.build_evaluation_problem(args=_args(tmp_path=tmp_path))
    provider = SweepOffShelf3DStepProtocol.evaluation_skill_provider(problem=problem)
    assert provider.human_cube_bin_reset_skills() == ()


@_NEEDS_KINDER
def test_every_control_step_of_a_skill_reaches_the_observers(*, tmp_path: Path) -> None:
    problem = SweepOffShelf3DStepProtocol.build_practice_problem(args=_args(tmp_path=tmp_path))
    try:
        problem.hard_reset()
        calls: list[str] = []
        problem.env.set_control_step_observers(
            before=lambda: calls.append("before"), after=lambda: calls.append("after")
        )
        problem.take_action(action=np.array([float(SKILL_IDS["PickWiper"]), 0.0, 0.0, 0.0]))
        steps = problem.env.last_controller_steps()
        assert steps > 0
        assert calls == ["before", "after"] * steps
        problem.env.set_control_step_observers(before=None, after=None)
        calls.clear()
        problem.take_action(action=np.array([float(SKILL_IDS["ReturnWiper"]), 0.0, 0.0, 0.0]))
        assert calls == []
    finally:
        problem.env.close()


@_NEEDS_KINDER
def test_the_budget_cuts_a_skill_off_and_the_state_is_adopted(*, tmp_path: Path) -> None:
    problem = SweepOffShelf3DStepProtocol.build_practice_problem(args=_args(tmp_path=tmp_path))
    try:
        problem.hard_reset()
        ticks = []

        def before() -> None:
            if len(ticks) >= 5:
                raise ControlStepLimitReached()
            ticks.append(1)

        problem.env.set_control_step_observers(before=before, after=None)
        action = np.array([float(SKILL_IDS["PickWiper"]), 0.0, 0.0, 0.0])
        with pytest.raises(ControlStepLimitReached):
            problem.take_action(action=action)
        problem.env.set_control_step_observers(before=None, after=None)
        state = SweepOffShelf3DStepProtocol.adopt_interrupted_skill(
            problem=problem, action=action, reason="practice_step_budget"
        )
        assert state.get(obj=problem.env.scene, feature_name="steps_taken") == 1
        assert SweepOffShelf3DStepProtocol.last_skill_error(problem=problem) == (
            "interrupted: practice_step_budget"
        )
    finally:
        problem.env.close()


@_NEEDS_KINDER
def test_the_goal_check_is_kinders_own_verdict(*, tmp_path: Path) -> None:
    problem = SweepOffShelf3DStepProtocol.build_evaluation_problem(args=_args(tmp_path=tmp_path))
    try:
        task = problem.sample_test_task()
        problem.reset_to_task(task=task)
        check = SweepOffShelf3DStepProtocol.goal_check(problem=problem, task=task)
        verdict = bool(problem.env.backend().core()._check_goals())  # noqa: SLF001
        assert check() is verdict is False
        assert SweepOffShelf3DStepProtocol.task_seed(task=task) >= 0
    finally:
        problem.env.close()


@_NEEDS_KINDER
def test_a_recording_opens_on_the_live_scene(*, tmp_path: Path) -> None:
    args = _args(tmp_path=tmp_path)
    problem = SweepOffShelf3DStepProtocol.build_practice_problem(args=args)
    try:
        path = tmp_path / "recording.jsonl"
        writer = SweepOffShelf3DStepProtocol.attach_recording(
            args=args, problem=problem, output_path=path
        )
        problem.hard_reset()
        writer.begin_segment(phase=Phase.PRACTICE, scene_seed=0)
        writer.close()
        header, segment = (json.loads(line) for line in path.read_text().splitlines()[:2])
        assert header["environment"] == "sweepoffshelf3d"
        assert segment["kind"] == "segment"
    finally:
        problem.env.close()
