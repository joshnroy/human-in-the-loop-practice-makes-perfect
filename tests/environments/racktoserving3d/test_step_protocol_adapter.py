"""Rack step-runner wiring without a native simulator or experiment launch."""

import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from hitl_pmp.cli import ENVIRONMENTS, METHODS, STEP_PROTOCOL_ENVIRONMENTS, Cli
from hitl_pmp.core.problem.environment.types import State
from hitl_pmp.core.problem.tasks.types import Task
from hitl_pmp.environments.racktoserving3d.cli import RackServingCli
from hitl_pmp.environments.racktoserving3d.environment import RackServingEnvironment
from hitl_pmp.environments.racktoserving3d.objects import RackServing3DObjects as Objects
from hitl_pmp.environments.racktoserving3d.skill_provider import RackServingOracle
from hitl_pmp.environments.racktoserving3d.step_protocol import RackServingStepProtocol
from hitl_pmp.environments.sweepoffshelf3d.step_protocol import SweepOffShelf3DStepProtocol
from hitl_pmp.launch_gate import LaunchGate
from hitl_pmp.recording.run_record.types import Phase
from hitl_pmp.step_protocol import StepPracticeRunner, StepProtocolEntry, StepProtocolValidation
from tests.practice_cost_fixtures import CostFixtures


def _args(*, tmp_path: Path, extra: tuple[str, ...] = ()):
    return Cli.parse_args(
        argv=[
            "--env",
            "racktoserving3d",
            "--method",
            "ees",
            "--practice-step-budget",
            "5",
            *CostFixtures.argv(env="racktoserving3d"),
            "--practice-reset-policy",
            "never",
            "--defer-rendering",
            "--output-dir",
            str(tmp_path),
            "--seed",
            "17",
            *extra,
        ]
    )


def _task(*, problem, seed: int = 29) -> Task:
    data = {obj: np.zeros(obj.type.dim) for obj in Objects.all_objects()}
    data[Objects.scene] = np.array([seed, 0, 0, 0], dtype=float)
    return Task(initial_state=State(data=data), goal=problem.tasks.goal())


def test_registration_preserves_the_shelf_adapter():
    assert ENVIRONMENTS["racktoserving3d"] is RackServingCli
    assert STEP_PROTOCOL_ENVIRONMENTS["racktoserving3d"] is RackServingStepProtocol
    assert STEP_PROTOCOL_ENVIRONMENTS["sweepoffshelf3d"] is SweepOffShelf3DStepProtocol


def test_step_arguments_satisfy_runner_validation(*, tmp_path):
    args = _args(tmp_path=tmp_path)
    StepProtocolValidation.check(args=args)
    assert args.defer_rendering
    assert RackServingStepProtocol.evaluation_pool() is None


def test_problem_builders_separate_modes_without_building_native_scenes(*, tmp_path):
    args = _args(tmp_path=tmp_path)
    practice = RackServingStepProtocol.build_practice_problem(args=args)
    evaluation = RackServingStepProtocol.build_evaluation_problem(args=args)
    repeated = RackServingStepProtocol.build_evaluation_problem(args=args)
    assert practice.env is not evaluation.env
    assert practice.env.config.mode == "training"
    assert evaluation.env.config.mode == "deployment"
    assert practice.human is not None
    assert evaluation.human is None
    assert practice.env._backend is evaluation.env._backend is None
    assert practice.tasks.seed == evaluation.tasks.seed == repeated.tasks.seed == args.seed
    assert evaluation.tasks.test_env_seed_offset == args.test_env_seed_offset
    assert args.seed == 17


def test_evaluations_repeat_the_same_task_stream_without_practice_interference(
    *, tmp_path, monkeypatch
):
    args = _args(tmp_path=tmp_path)
    practice = RackServingStepProtocol.build_practice_problem(args=args)
    first = RackServingStepProtocol.build_evaluation_problem(args=args)
    second = RackServingStepProtocol.build_evaluation_problem(args=args)

    def reset(*, seed):
        return _task(problem=practice, seed=seed).initial_state

    monkeypatch.setattr(RackServingEnvironment, "reset_to_seed", staticmethod(reset))
    practice.sample_train_task()
    practice.tasks.sample_reset_start_task()
    practice.sample_test_task()
    first_seeds = [
        RackServingStepProtocol.task_seed(task=first.sample_test_task()) for _ in range(3)
    ]
    second_seeds = [
        RackServingStepProtocol.task_seed(task=second.sample_test_task()) for _ in range(3)
    ]
    assert first_seeds == second_seeds
    assert len(set(first_seeds)) == 3


@pytest.mark.parametrize("enabled", [True, False])
def test_practice_context_honors_human_configuration(*, tmp_path, enabled):
    args = _args(tmp_path=tmp_path, extra=() if enabled else ("--no-human-reset",))
    problem = RackServingStepProtocol.build_practice_problem(args=args)
    context = RackServingStepProtocol.practice_context(args=args, problem=problem)
    assert context.env is context.skill_provider.env is problem.env
    assert isinstance(context.oracle, RackServingOracle)
    assert context.oracle.env is problem.env
    names = {ground.skill.name for ground in context.skill_provider.human_cube_bin_reset_skills()}
    assert names == (set(RackServingStepProtocol.human_skill_names()) if enabled else set())


def test_evaluation_never_offers_human_skills_even_for_a_training_problem(*, tmp_path):
    args = _args(tmp_path=tmp_path)
    for builder in (
        RackServingStepProtocol.build_practice_problem,
        RackServingStepProtocol.build_evaluation_problem,
    ):
        problem = builder(args=args)
        provider = RackServingStepProtocol.evaluation_skill_provider(problem=problem)
        assert provider.env is problem.env
        assert provider.human_cube_bin_reset_skills() == ()


def test_human_cost_and_log_names_match_rack_destinations():
    assert RackServingStepProtocol.human_skill_names() == {
        "human_batch_return": "human_batch_return",
        "human_tool_recovery": "human_tool_recovery",
    }


def test_goal_check_reads_live_rack_verdict_not_shelf_goal(*, tmp_path, monkeypatch):
    problem = RackServingStepProtocol.build_evaluation_problem(args=_args(tmp_path=tmp_path))
    backend = Mock()
    backend.task_succeeded.side_effect = [False, True, False]
    monkeypatch.setattr(RackServingEnvironment, "backend", lambda self: backend)
    task = _task(problem=problem)
    check = RackServingStepProtocol.goal_check(problem=problem, task=task)
    assert [check(), check(), check()] == [False, True, False]
    backend.core.assert_not_called()
    assert RackServingStepProtocol.task_seed(task=task) == 29


@pytest.mark.parametrize("mode", ["training", "deployment"])
def test_recording_keeps_live_state_and_full_variant_config(*, tmp_path, monkeypatch, mode):
    args = _args(tmp_path=tmp_path, extra=("--gap", "0.02"))
    problem = RackServingCli.build_problem(args=args, mode=mode)
    backend = Mock()
    backend.features.return_value = {"rack_0": np.array([1.25, 2.5])}
    monkeypatch.setattr(RackServingEnvironment, "backend", lambda self: backend)
    path = tmp_path / "recording.jsonl"
    writer = RackServingStepProtocol.attach_recording(args=args, problem=problem, output_path=path)
    try:
        backend.features.assert_not_called()
        writer.begin_segment(phase=Phase.PRACTICE, scene_seed=29)
        backend.features.return_value = {"rack_0": np.array([3.0, 4.0])}
        writer.robot_action(label="PickScraper", start_step=0, end_step=5, success=None)
        writer.begin_cycle(cycle=1, task=_task(problem=problem))
    finally:
        writer.close()
    entries = [json.loads(line) for line in path.read_text().splitlines()]
    header = entries[0]
    assert header["environment"] == "racktoserving3d"
    assert header["method"] == "ees"
    assert header["run_seed"] == 17
    for name, value in problem.env.config.model_dump(mode="json").items():
        assert header["scene"][name] == value
    assert header["scene"]["canonical_seed"] == args.canonical_seed
    assert header["scene"]["skill_step_limit"] == args.skill_step_limit
    assert header["scene"]["test_env_seed_offset"] == args.test_env_seed_offset
    ticks = [entry for entry in entries if entry["kind"] == "tick"]
    assert ticks[0]["state"] == {"rack_0": [1.25, 2.5]}
    assert ticks[-1]["state"] == {"rack_0": [3.0, 4.0]}
    assert entries[-1]["task_seed"] == 29


def test_interruption_adoption_delegates_to_rack_environment(*, tmp_path, monkeypatch):
    problem = RackServingStepProtocol.build_practice_problem(args=_args(tmp_path=tmp_path))
    state = _task(problem=problem).initial_state
    adopted = Mock(return_value=state)
    monkeypatch.setattr(RackServingEnvironment, "adopt_interrupted_skill", adopted)
    action = np.array([0.0, 0.0, 0.0, 0.0])
    assert (
        RackServingStepProtocol.adopt_interrupted_skill(
            problem=problem, action=action, reason="practice_step_budget"
        )
        is state
    )
    adopted.assert_called_once_with(action=action, reason="practice_step_budget")
    problem.env._last_skill_error = "interrupted: practice_step_budget"
    assert RackServingStepProtocol.last_skill_error(problem=problem) == (
        "interrupted: practice_step_budget"
    )


def test_no_shelf_trace_or_native_setup_during_evaluation_cleanup(*, tmp_path):
    problem = RackServingStepProtocol.build_evaluation_problem(args=_args(tmp_path=tmp_path))
    RackServingStepProtocol.interrupted(problem=problem, action=np.zeros(4))
    assert RackServingStepProtocol.last_skill_trace(problem=problem) is None
    assert problem.env._backend is None


def test_global_cli_dispatches_to_adapter_without_launching(*, tmp_path, monkeypatch):
    args = _args(tmp_path=tmp_path)
    entry = Mock()
    gate = Mock()
    monkeypatch.setattr(Cli, "parse_args", lambda **kwargs: args)
    monkeypatch.setattr(StepProtocolEntry, "run", entry)
    monkeypatch.setattr(LaunchGate, "enforce", gate)
    Cli.main()
    entry.assert_called_once_with(
        args=args,
        method_cli=METHODS["ees"],
        env_cli=RackServingCli,
        environment=RackServingStepProtocol,
    )
    gate.assert_called_once_with(domain="racktoserving3d", override=False)


@pytest.mark.parametrize("fails", [False, True])
def test_real_entry_builds_ees_with_rack_context_and_closes_resources(
    *, tmp_path, monkeypatch, fails
):
    args = _args(tmp_path=tmp_path)
    runner = Mock(side_effect=RuntimeError("runner failed") if fails else None)
    closed = []
    monkeypatch.setattr(StepPracticeRunner, "run", runner)
    monkeypatch.setattr(RackServingEnvironment, "close", lambda self: closed.append(self))

    def enter():
        StepProtocolEntry.run(
            args=args,
            method_cli=METHODS["ees"],
            env_cli=RackServingCli,
            environment=RackServingStepProtocol,
        )

    if fails:
        with pytest.raises(RuntimeError, match="runner failed"):
            enter()
    else:
        enter()
    runner.assert_called_once()
    call = runner.call_args.kwargs
    assert call["problem"].env.config.mode == "training"
    assert call["environment"] is RackServingStepProtocol
    assert call["evaluations"] is None
    assert call["recorder"].header.environment == "racktoserving3d"
    assert closed == [call["problem"].env]
    assert call["problem"].env._backend is None


@pytest.mark.parametrize("bulk_only", [True, False])
def test_a_bulk_transfer_only_task_ends_with_its_first_sweep(*, tmp_path, bulk_only):
    from hitl_pmp.environments.racktoserving3d.skills import SKILL_IDS

    extra = ("--bulk-transfer-only", "1") if bulk_only else ()
    problem = RackServingStepProtocol.build_evaluation_problem(
        args=_args(tmp_path=tmp_path, extra=extra)
    )
    for name, skill in SKILL_IDS.items():
        reason = RackServingStepProtocol.evaluation_terminal_reason(
            problem=problem, action=np.array([skill, 0, 0, 0, 0], dtype=float)
        )
        ends = bulk_only and name in {"Sweep", "GatherAndSweep"}
        assert reason == ("bulk_attempt_finished" if ends else None), name
