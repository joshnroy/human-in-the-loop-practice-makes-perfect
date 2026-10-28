"""Table CLI composition, failure scoring and recording without simulator execution."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from hitl_pmp.cli import ENVIRONMENTS, STEP_PROTOCOL_ENVIRONMENTS, Cli
from hitl_pmp.core.method.types import LabeledAction
from hitl_pmp.core.problem.environment.types import Action, State
from hitl_pmp.core.problem.tasks.types import Task
from hitl_pmp.environments.sweepoffshelf3d.types import SweepOffShelf3DState
from hitl_pmp.environments.sweeptable3d.backend import SweepTable3DBackend
from hitl_pmp.environments.sweeptable3d.cli import SweepTable3DCli
from hitl_pmp.environments.sweeptable3d.environment import SweepTable3DEnvironment
from hitl_pmp.environments.sweeptable3d.objects import SweepTable3DObjects as O
from hitl_pmp.environments.sweeptable3d.recording import SweepTable3DReplayRenderer
from hitl_pmp.environments.sweeptable3d.skill_trace import SkillTraces
from hitl_pmp.environments.sweeptable3d.skills import SKILL_IDS
from hitl_pmp.environments.sweeptable3d.skills import SweepTable3DSkills as S
from hitl_pmp.environments.sweeptable3d.step_protocol import SweepTable3DStepProtocol
from hitl_pmp.environments.sweeptable3d.tasks import SweepTable3DTasks
from hitl_pmp.method_runner import MethodRunner
from hitl_pmp.recording.run_record.types import Phase
from hitl_pmp.step_protocol import StepProtocolEntry


def _args(*argv: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--method", default="ees")
    SweepTable3DCli.add_arguments(parser=parser)
    return parser.parse_args(list(argv))


def _state(*, seed: int = 0, solved: bool = False) -> SweepOffShelf3DState:
    data = {obj: np.zeros(len(obj.type.feature_names)) for obj in O.all_objects()}
    data[O.scene] = np.array([float(seed), 0.0])
    return SweepOffShelf3DState(
        data=data,
        abstract_atoms=frozenset({
            ("SweepSucceeded", ()),
            *(("InBox", (cube.name, O.box.name)) for cube in O.cubes),
        })
        if solved
        else frozenset(),
    )


def test_table_is_registered_and_parsed_by_the_global_cli() -> None:
    assert ENVIRONMENTS["sweeptable3d"] is SweepTable3DCli
    assert STEP_PROTOCOL_ENVIRONMENTS["sweeptable3d"] is SweepTable3DStepProtocol
    args = Cli.parse_args(
        argv=["--env", "sweeptable3d", "--method", "ees", "--fixed-bin-offset", "0.2"]
    )
    assert args.fixed_bin_offset == 0.2 and args.env == "sweeptable3d"
    assert _args().defer_rendering


def test_global_cli_builds_the_table_method_and_both_problems_without_a_simulator(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs: list[dict[str, Any]] = []
    monkeypatch.setattr(MethodRunner, "run", staticmethod(lambda **kwargs: runs.append(kwargs)))
    Cli.main(
        argv=[
            "--env",
            "sweeptable3d",
            "--method",
            "ees",
            "--num-cycles",
            "0",
            "--fixed-bin-offset",
            "0.0",
        ]
    )
    (run,) = runs
    practice, evaluation = run["problem"], run["evaluation_problem"]
    assert run["method"].env is practice.env
    assert practice.env is not evaluation.env
    assert practice.env.fixed_bin_offset == evaluation.env.fixed_bin_offset == 0.0
    assert practice.env._backend is None and evaluation.env._backend is None
    assert run["renderer"] is None


def test_global_cli_dispatches_table_evaluation_to_the_step_protocol(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs: list[dict[str, Any]] = []
    monkeypatch.setattr(
        StepProtocolEntry, "run", staticmethod(lambda **kwargs: runs.append(kwargs))
    )
    Cli.main(
        argv=[
            "--env",
            "sweeptable3d",
            "--method",
            "ees",
            "--evaluate-only",
            "--skip-skill-benchmark-gate",
            "--fixed-bin-offset",
            "0.0",
        ]
    )
    (run,) = runs
    assert run["environment"] is SweepTable3DStepProtocol
    assert run["env_cli"] is SweepTable3DCli
    assert run["args"].fixed_bin_offset == 0.0


@pytest.mark.parametrize("offset", [None, -0.4, 0.0, 0.4])
def test_table_configuration_is_carried_into_a_lazy_environment(*, offset: float | None) -> None:
    flags = [] if offset is None else ["--fixed-bin-offset", str(offset)]
    args = _args(*flags, "--reset-cubes-success-probability", "0.25", "--no-scene-bg")
    problem = SweepTable3DCli.build_problem(args=args, human_seed=19)
    assert problem.env.fixed_bin_offset == (-0.18 if offset is None else offset)
    assert problem.env.reset_cubes_success_probability == 0.25
    assert problem.env.human_seed == 19 and not problem.env.scene_bg
    assert problem.env._backend is None


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--fixed-bin-offset", "-0.41"),
        ("--fixed-bin-offset", "0.41"),
        ("--fixed-bin-offset", "nan"),
        ("--fixed-bin-offset", "inf"),
        ("--reset-cubes-success-probability", "-0.1"),
        ("--reset-cubes-success-probability", "1.1"),
    ],
)
def test_invalid_domain_flags_are_rejected(*, flag: str, value: str) -> None:
    with pytest.raises(SystemExit):
        _args(f"{flag}={value}")


@pytest.mark.parametrize(
    "scope,practice_motion,evaluation_motion",
    [
        ("off", False, False),
        ("evaluation", False, True),
        ("all", True, True),
    ],
)
def test_step_protocol_separates_scenes_humans_and_motion_scopes(
    *, scope: str, practice_motion: bool, evaluation_motion: bool
) -> None:
    args = _args("--record-motion", scope, "--no-human-reset")
    practice = SweepTable3DStepProtocol.build_practice_problem(args=args)
    evaluation = SweepTable3DStepProtocol.build_evaluation_problem(args=args)
    assert practice.env is not evaluation.env
    assert practice.env.human_seed == args.seed
    assert evaluation.env.human_seed == args.seed + 1
    assert practice.tasks.seed == evaluation.tasks.seed == args.seed
    assert practice.env.motion_recording is practice_motion
    assert evaluation.env.motion_recording is evaluation_motion
    context = SweepTable3DStepProtocol.practice_context(args=args, problem=practice)
    assert context.env is practice.env
    assert context.skill_provider.human_cube_bin_reset_skills() == ()
    assert (
        SweepTable3DStepProtocol.evaluation_skill_provider(
            problem=evaluation
        ).human_cube_bin_reset_skills()
        == ()
    )
    assert SweepTable3DStepProtocol.human_skill_names() == {
        "reset_cubes": "reset_cubes",
        "reset_wiper": "reset_wiper",
    }


def test_task_streams_are_reproducible_independent_and_support_in_place_sampling(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    resets: list[int] = []

    def reset(*, env: SweepTable3DEnvironment, seed: int) -> State:
        resets.append(seed)
        env.current_state = _state(seed=seed)
        return env.current_state

    monkeypatch.setattr(
        SweepTable3DEnvironment, "reset_to_seed", lambda self, **kwargs: reset(env=self, **kwargs)
    )
    tasks = SweepTable3DTasks(env=SweepTable3DEnvironment(), seed=13)
    reference = SweepTable3DTasks(env=SweepTable3DEnvironment(), seed=13)
    first = tasks.sample_train_task()
    assert SweepTable3DStepProtocol.task_seed(task=first) == SweepTable3DStepProtocol.task_seed(
        task=reference.sample_train_task()
    )
    reset_task = tasks.sample_reset_start_task()
    assert SweepTable3DStepProtocol.task_seed(
        task=reset_task
    ) == SweepTable3DStepProtocol.task_seed(task=reference.sample_reset_start_task())
    tasks.sample_reset_start_task()
    for sample_name in ("sample_train_task", "sample_test_task"):
        sample = getattr(tasks, sample_name)()
        expected = getattr(reference, sample_name)()
        assert SweepTable3DStepProtocol.task_seed(task=sample) == (
            SweepTable3DStepProtocol.task_seed(task=expected)
        )
        assert sample.goal.atoms == SweepTable3DTasks.goal().atoms
    before = len(resets)
    in_place = tasks.sample_train_task_in_place()
    assert len(resets) == before
    assert in_place.initial_state is tasks.env.get_current_state()


@pytest.mark.parametrize(
    "initial_failed,fail_after_action,solved,actions",
    [
        (True, False, False, 0),
        (False, True, False, 1),
        (False, False, True, 1),
    ],
)
def test_legacy_evaluation_stops_at_failure_and_never_credits_a_simultaneous_goal(
    *,
    monkeypatch: pytest.MonkeyPatch,
    initial_failed: bool,
    fail_after_action: bool,
    solved: bool,
    actions: int,
) -> None:
    problem = SweepTable3DCli.build_problem(args=_args(), human_seed=0)
    signal = {"failed": initial_failed}
    backend = SimpleNamespace(task_failed=lambda: signal["failed"])
    monkeypatch.setattr(SweepTable3DEnvironment, "backend", lambda self: backend)

    def set_state(*, env: SweepTable3DEnvironment, state: State) -> None:
        env.current_state = state

    def take_action(*, env: SweepTable3DEnvironment, action: Action) -> State:
        del action
        signal["failed"] = fail_after_action
        env.current_state = _state(solved=True)
        return env.current_state

    monkeypatch.setattr(
        SweepTable3DEnvironment, "set_state", lambda self, **kwargs: set_state(env=self, **kwargs)
    )
    monkeypatch.setattr(
        SweepTable3DEnvironment,
        "take_action",
        lambda self, **kwargs: take_action(env=self, **kwargs),
    )
    task = Task(initial_state=_state(solved=initial_failed), goal=SweepTable3DTasks.goal())
    result, frames, trace = problem.run_task_episode(
        task=task, policy=lambda state: LabeledAction(action=np.zeros(4), label="test")
    )
    assert result is solved and frames == []
    assert len(trace.actions) == actions and len(trace.states) == actions + 1


def test_step_protocol_goal_and_failure_checks_read_the_live_backend(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    signal = {"failed": False, "goal": False}
    backend = SimpleNamespace(
        task_failed=lambda: signal["failed"],
        task_succeeded=lambda: signal["goal"],
    )
    monkeypatch.setattr(SweepTable3DEnvironment, "backend", lambda self: backend)
    problem = SweepTable3DCli.build_problem(args=_args(), human_seed=0)
    task = Task(initial_state=_state(), goal=SweepTable3DTasks.goal())
    check_goal = SweepTable3DStepProtocol.goal_check(problem=problem, task=task)
    check_failure = SweepTable3DStepProtocol.failure_check(problem=problem, task=task)
    assert not check_goal() and not check_failure()
    signal.update(failed=True, goal=True)
    assert not check_goal() and check_failure()


@pytest.mark.parametrize("skill", S.robot_skills(), ids=lambda skill: skill.name)
def test_trace_grounding_uses_table_types_and_all_five_sweep_targets(*, skill: Any) -> None:
    ground = SkillTraces.ground(action=np.array([SKILL_IDS[skill.name], 2.0, 45.0, 0.1]))
    assert ground is not None and ground.skill is skill
    assert all(obj in O.all_objects() for obj in ground.objects)
    cubes = tuple(obj for obj in ground.objects if obj.type == O.cube_type)
    if skill is S.SWEEP_TABLE:
        assert cubes == O.cubes
    elif skill in S.RETURN_CUBE_SKILLS:
        assert cubes == (O.balls[2], *O.balls)
    elif cubes:
        assert cubes == (O.cubes[2],)


def test_recorded_sweep_trace_preserves_table_parameters_and_missing_cube_effects(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    action = np.array([SKILL_IDS[S.SWEEP_TABLE.name], 0.0, -24.0, 0.13])
    ground = SkillTraces.ground(action=action)
    assert ground is not None
    before = frozenset(
        (atom.predicate.name, tuple(obj.name for obj in atom.objects))
        for atom in ground.preconditions
    )
    after = frozenset(
        (atom.predicate.name, tuple(obj.name for obj in atom.objects))
        for atom in ground.add_effects
    )
    after -= {("InBox", (O.cubes[-1].name, O.box.name))}
    problem = SweepTable3DCli.build_problem(args=_args(), human_seed=0)
    monkeypatch.setattr(
        SweepTable3DEnvironment, "last_skill_attempt", lambda self: (action, before, after, [])
    )
    monkeypatch.setattr(SweepTable3DEnvironment, "last_controller_steps", lambda self: 779)
    trace = SweepTable3DStepProtocol.last_skill_trace(problem=problem)
    assert trace is not None
    assert trace["skill"] == "SweepTable" and trace["params"] == [-24.0, 0.13]
    assert trace["objects"] == [O.robot.name, O.wiper.name, O.box.name, *(c.name for c in O.cubes)]
    assert trace["outcome"] == "effects_missing"
    assert trace["missing_effects"] == [f"InBox({O.cubes[-1].name},{O.box.name})"]


@pytest.mark.parametrize("motion", [False, True])
def test_recording_uses_table_scene_metadata_and_the_configured_tick_source(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, motion: bool
) -> None:
    args = _args("--fixed-bin-offset", "0.2")
    problem = SweepTable3DCli.build_problem(args=args, human_seed=0, motion_recording=motion)
    boundary = {
        obj.name: values.tolist() for obj, values in _state().data.items() if obj != O.scene
    }
    moving = {**boundary, "robot_joints": [0.0] * len(SweepTable3DBackend.ROBOT_JOINTS)}
    backend = SimpleNamespace(features=lambda: boundary, motion_state=lambda: moving)
    sinks: list[int] = []
    monkeypatch.setattr(SweepTable3DEnvironment, "backend", lambda self: backend)
    monkeypatch.setattr(
        SweepTable3DEnvironment,
        "attach_motion_sink",
        lambda self, **kwargs: sinks.append(kwargs["stride"]),
    )
    path = tmp_path / "recording.jsonl"
    writer = SweepTable3DStepProtocol.attach_recording(args=args, problem=problem, output_path=path)
    writer.begin_segment(phase=Phase.EVALUATION, task_seed=42)
    writer.close()
    entries = [json.loads(line) for line in path.read_text().splitlines()]
    header = entries[0]
    assert header["environment"] == "sweeptable3d"
    assert header["scene"]["fixed_bin_offset"] == 0.2
    assert ("motion" in header["scene"]) is motion
    assert len(sinks) == int(motion)
    tick = next(entry for entry in entries if entry["kind"] == "tick")
    assert tick["state"] == (moving if motion else boundary)


def test_replay_builds_a_table_backend_with_the_recorded_scene_configuration(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    resets: list[tuple[int, float | None]] = []
    monkeypatch.setattr(
        SweepTable3DBackend,
        "reset",
        lambda self, **kwargs: resets.append((kwargs["seed"], self.fixed_bin_offset)),
    )
    monkeypatch.setattr(
        SweepTable3DBackend,
        "core",
        lambda self: SimpleNamespace(
            get_object=lambda name: SimpleNamespace(
                get_pose=lambda: (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
            )
        ),
    )
    monkeypatch.setattr(
        SweepTable3DBackend,
        "features",
        lambda self: {O.box.name: np.array([1.0, 2.0, 3.0, 4.0, 0.1]), O.wiper.name: np.zeros(4)},
    )
    renderer = SweepTable3DReplayRenderer(
        scene={"scene_bg": False, "canonical_seed": 42, "fixed_bin_offset": -0.2}, scene_bg=None
    )
    backend = renderer._built()
    assert isinstance(backend, SweepTable3DBackend) and not backend.scene_bg
    assert renderer._built() is backend and resets == [(42, -0.2)]
    assert set(renderer._home) == {*(cube.name for cube in O.cubes), O.wiper.name, O.box.name}
    np.testing.assert_array_equal(renderer._home_box_center, [2.0, 3.0])
