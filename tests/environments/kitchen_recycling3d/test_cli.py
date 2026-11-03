"""Kitchen runner integration, with simulator imports deferred to opt-in tests."""

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


def _args(*, argv: tuple[str, ...] = ()) -> argparse.Namespace:
    from hitl_pmp.environments.kitchen_recycling3d.cli import KitchenRecyclingCli

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--method", default="ees")
    KitchenRecyclingCli.add_arguments(parser=parser)
    return parser.parse_args(("--no-scene-bg", *argv))


def test_registered_for_both_runners() -> None:
    from hitl_pmp.cli import ENVIRONMENTS, STEP_PROTOCOL_ENVIRONMENTS
    from hitl_pmp.environments.kitchen_recycling3d.cli import KitchenRecyclingCli
    from hitl_pmp.environments.kitchen_recycling3d.step_protocol import (
        KitchenRecyclingStepProtocol,
    )

    assert ENVIRONMENTS["kitchen_recycling3d"] is KitchenRecyclingCli
    assert STEP_PROTOCOL_ENVIRONMENTS["kitchen_recycling3d"] is KitchenRecyclingStepProtocol


def test_cli_import_does_not_load_optional_simulators() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import hitl_pmp.cli; "
            "assert not {'kinder', 'mujoco', 'pybullet'} & sys.modules.keys()",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_preview_config_is_frozen_for_evaluation_workers(*, tmp_path: Path) -> None:
    from hitl_pmp.environments.kitchen_recycling3d.cli import KitchenRecyclingCli

    preview = Path(__file__).resolve().parents[3] / "configs/kitchen_recycling3d_preview.json"
    path = tmp_path / "kitchen.json"
    path.write_text(preview.read_text())
    args = _args(argv=("--kitchen-config", str(path), "--no-human-reset"))
    config = KitchenRecyclingCli.config(args=args)
    assert config.calibrations
    assert not config.scatter_objects
    assert not args.human_reset
    path.unlink()
    worker_args = argparse.Namespace(**vars(args))
    assert KitchenRecyclingCli.config(args=worker_args) == config
    assert json.loads(args.kitchen_config_resolved) == config.model_dump(mode="json")


def test_full_run_defaults_scatter_objects() -> None:
    from hitl_pmp.environments.kitchen_recycling3d.cli import KitchenRecyclingCli

    assert KitchenRecyclingCli.config(args=_args()).scatter_objects


@pytest.mark.parametrize("name", ["cube_control", "gatorade_20oz", "food_can"])
def test_single_object_config_is_identical_in_practice_and_deployment(*, name, tmp_path):
    from hitl_pmp.environments.kitchen_recycling3d.domain import KitchenConfig
    from hitl_pmp.environments.kitchen_recycling3d.step_protocol import (
        KitchenRecyclingStepProtocol as Adapter,
    )

    config = KitchenConfig()
    only = config.object(name=name)
    config = KitchenConfig(objects=(only,), required_objects=(name,))
    path = tmp_path / "single.json"
    path.write_text(config.model_dump_json())
    args = _args(argv=("--kitchen-config", str(path)))
    training = Adapter.build_practice_problem(args=args)
    path.unlink()
    deployment = Adapter.build_evaluation_problem(args=argparse.Namespace(**vars(args)))
    try:
        assert training.env.config == deployment.env.config == config
        assert tuple(obj.name for obj in training.env.items) == (name,)
        assert tuple(obj.name for obj in deployment.env.items) == (name,)
        provider = Adapter.practice_context(args=args, problem=training).skill_provider
        assert {obj.name for obj in provider.objects() if obj.type.name == "item"} == {name}
        excluded = {"cube_control", "gatorade_20oz", "food_can"} - {name}
        assert not any(other in skill.name for other in excluded for skill in provider.skills())
        assert {key for key in Adapter.human_skill_names() if key.startswith("recover:")} == {
            f"recover:{name}"
        }
        assert not Adapter.evaluation_skill_provider(problem=deployment).movables_reset_skills()
    finally:
        training.env.close()
        deployment.env.close()


def test_invalid_config_fails_before_building_simulation(*, tmp_path: Path) -> None:
    from hitl_pmp.environments.kitchen_recycling3d.cli import KitchenRecyclingCli

    path = tmp_path / "invalid.json"
    path.write_text('{"robot_radius": -1}')
    with pytest.raises(ValueError):
        KitchenRecyclingCli.build_problem(
            args=_args(argv=("--kitchen-config", str(path))), phase="training"
        )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_requested_ees_flags_parse_without_launching(*, seed: int) -> None:
    from hitl_pmp.cli import Cli

    args = Cli.parse_args(
        argv=[
            "--env",
            "kitchen_recycling3d",
            "--method",
            "ees",
            "--seed",
            str(seed),
            "--practice-step-budget",
            "160000",
            "--practice-groundings",
            "all",
            "--no-reproduce-predicators-skip-perfect",
            "--goal-pursuit-horizon",
            "0",
            "--stop-after-perfect-evaluations",
            "0",
        ]
    )
    assert args.practice_step_budget == 160000
    assert args.practice_groundings == "all"
    assert not args.reproduce_predicators_skip_perfect
    assert args.goal_pursuit_horizon == 0
    assert args.stop_after_perfect_evaluations == 0
    assert not args.stop_on_no_practice_action


@pytest.mark.parametrize(
    "solver,variant", [("expectimax", "algorithm3"), ("determinized_astar", "curtis")]
)
def test_belief_space_solver_flags_parse_without_launching(*, solver: str, variant: str) -> None:
    from hitl_pmp.cli import Cli

    args = Cli.parse_args(
        argv=[
            "--env",
            "kitchen_recycling3d",
            "--method",
            "pomdp",
            "--pomdp-solver",
            solver,
            "--pomdp-determinized-variant",
            variant,
            "--pomdp-observation-probability-weight",
            "0.0",
            "--practice-step-budget",
            "160000",
        ]
    )
    assert args.pomdp_solver == solver
    assert args.pomdp_determinized_variant == variant


def test_cycle_runner_uses_separate_environments_and_closes_them(*, monkeypatch) -> None:
    from hitl_pmp.environments.kitchen_recycling3d.cli import KitchenRecyclingCli
    from hitl_pmp.environments.kitchen_recycling3d.environment import (
        KitchenRecyclingEnvironment as Environment,
    )
    from hitl_pmp.method_runner import MethodRunner

    closed = []
    calls = []
    monkeypatch.setattr(Environment, "close", lambda self: closed.append(self))
    monkeypatch.setattr(MethodRunner, "run", lambda **kwargs: calls.append(kwargs))
    method = object()
    KitchenRecyclingCli.run_method(
        args=_args(),
        method_factory=lambda context: method,
        num_cycles=0,
        max_steps_per_interaction=1,
    )
    call = calls[0]
    practice, evaluation = call["problem"], call["evaluation_problem"]
    assert practice.env is not evaluation.env
    assert practice.env.phase == "training"
    assert evaluation.env.phase == "deployment"
    assert call["method"] is method
    assert call["renderer"] is None
    assert len(closed) == 2
    assert closed[0] is evaluation.env
    assert closed[1] is practice.env


def test_step_evaluation_disables_human_skills() -> None:
    from hitl_pmp.environments.kitchen_recycling3d.step_protocol import (
        KitchenRecyclingStepProtocol as Adapter,
    )

    problem = Adapter.build_evaluation_problem(args=_args())
    try:
        provider = Adapter.evaluation_skill_provider(problem=problem)
        assert problem.env.phase == "deployment"
        assert provider.human_cube_bin_reset_skills() == ()
        assert Adapter.evaluation_pool() is None
    finally:
        problem.env.close()


def test_training_environment_cannot_be_used_as_deployment() -> None:
    from hitl_pmp.environments.kitchen_recycling3d.step_protocol import (
        KitchenRecyclingStepProtocol as Adapter,
    )

    problem = Adapter.build_practice_problem(args=_args())
    try:
        with pytest.raises(ValueError, match="separate deployment"):
            Adapter.evaluation_skill_provider(problem=problem)
    finally:
        problem.env.close()


def test_runner_closes_both_environments_when_method_construction_fails(*, monkeypatch) -> None:
    from hitl_pmp.environments.kitchen_recycling3d.cli import KitchenRecyclingCli
    from hitl_pmp.environments.kitchen_recycling3d.environment import (
        KitchenRecyclingEnvironment as Environment,
    )

    closed = []
    monkeypatch.setattr(Environment, "close", lambda self: closed.append(self.phase))

    def fail(*, context):
        raise RuntimeError("method construction failed")

    with pytest.raises(RuntimeError, match="method construction failed"):
        KitchenRecyclingCli.run_method(
            args=_args(),
            method_factory=lambda context: fail(context=context),
            num_cycles=0,
            max_steps_per_interaction=1,
        )
    assert closed == ["deployment", "training"]


def test_human_destination_accounting_matches_provider() -> None:
    from hitl_pmp.environments.kitchen_recycling3d.step_protocol import (
        KitchenRecyclingStepProtocol as Adapter,
    )

    args = _args()
    problem = Adapter.build_practice_problem(args=args)
    try:
        provider = Adapter.practice_context(args=args, problem=problem).skill_provider
        destinations = Adapter.human_skill_names()
        for skill in provider.human_cube_bin_reset_skills():
            destination = provider.movables_reset_destination(ground_skill=skill)
            assert destinations[destination] == provider.human_skill_name(ground_skill=skill)
        assert "recover:gatorade_20oz" in destinations
        assert "recover:food_can" in destinations
        assert "relocate_bin_near" in destinations
        assert "restore_bin_target" in destinations
    finally:
        problem.env.close()


def test_structured_recording_without_simulator(*, tmp_path: Path, monkeypatch) -> None:
    import numpy as np

    from hitl_pmp.core.problem.environment.types import State
    from hitl_pmp.environments.kitchen_recycling3d.step_protocol import (
        KitchenRecyclingStepProtocol as Adapter,
    )
    from hitl_pmp.recording.run_record.types import Phase

    args = _args()
    problem = Adapter.build_practice_problem(args=args)
    problem.env.current_state = State(
        data={obj: np.zeros(obj.type.dim) for obj in problem.env.objects.all_objects()}
    )
    monkeypatch.setattr(type(problem.env), "get_current_state", lambda self: self.current_state)
    path = tmp_path / "recording.jsonl"
    writer = Adapter.attach_recording(args=args, problem=problem, output_path=path)
    try:
        writer.begin_segment(phase=Phase.PRACTICE, scene_seed=0)
        writer.robot_action(
            label="Pick_gatorade_20oz_far(robot, gatorade_20oz, bin)",
            start_step=0,
            end_step=3,
            success=False,
            outcome="controller failure",
        )
        writer.human_action(skill="recover:gatorade_20oz", start_step=3, end_step=4, success=True)
        writer.end_segment(success=False, steps=4)
    finally:
        writer.close()
        problem.env.close()
    entries = [json.loads(line) for line in path.read_text().splitlines()]
    assert entries[0]["scene"]["perception"] == "simulator_ground_truth"
    assert entries[0]["scene"]["video"] is False
    assert set(entries[0]["feature_names"]) == set(entries[2]["state"])
    actions = [entry for entry in entries if entry["kind"] == "action"]
    assert [entry["actor"] for entry in actions] == ["robot", "human"]
    assert actions[0]["outcome"] == "controller failure"
    assert actions[1]["skill"] == "recover:gatorade_20oz"
    assert entries[-1]["kind"] == "segment_end"


@pytest.mark.skipif(importlib.util.find_spec("kinder") is None, reason="needs KINDER")
def test_structured_recording_and_budget_interruption(*, tmp_path: Path) -> None:
    from hitl_pmp.core.control_steps import ControlStepLimitReached
    from hitl_pmp.environments.kitchen_recycling3d.step_protocol import (
        KitchenRecyclingStepProtocol as Adapter,
    )
    from hitl_pmp.recording.run_record.types import Phase

    args = _args()
    problem = Adapter.build_practice_problem(args=args)
    writer = Adapter.attach_recording(
        args=args, problem=problem, output_path=tmp_path / "recording.jsonl"
    )
    try:
        problem.hard_reset()
        writer.begin_segment(phase=Phase.PRACTICE, scene_seed=args.seed)
        task = problem.tasks.sample_train_task_in_place()
        assert Adapter.task_seed(task=task) >= 0
        context = Adapter.practice_context(args=args, problem=problem)
        action = context.oracle.get_labeled_action(
            state=problem.get_current_state(), goal=task.goal
        )
        ticks = []

        def before() -> None:
            if len(ticks) >= 2:
                raise ControlStepLimitReached()

        problem.env.set_control_step_observers(before=before, after=lambda: ticks.append(1))
        with pytest.raises(ControlStepLimitReached):
            problem.take_action(action=action.action)
        problem.env.set_control_step_observers(before=None, after=None)
        Adapter.adopt_interrupted_skill(
            problem=problem, action=action.action, reason="practice_step_budget"
        )
        assert len(ticks) == 2
        assert "practice_step_budget" in Adapter.last_skill_error(problem=problem)
        writer.robot_action(
            label=action.label,
            start_step=0,
            end_step=2,
            success=False,
            outcome="practice_step_budget",
        )
        writer.end_segment(success=False, steps=2)
    finally:
        writer.close()
        problem.env.close()
    entries = [json.loads(line) for line in (tmp_path / "recording.jsonl").read_text().splitlines()]
    assert entries[0]["environment"] == "kitchen_recycling3d"
    assert entries[0]["scene"]["perception"] == "simulator_ground_truth"
    actions = [entry for entry in entries if entry["kind"] == "action"]
    assert actions[0]["end_step"] == 2
    assert actions[0]["outcome"] == "practice_step_budget"
    assert entries[-1]["kind"] == "segment_end"
