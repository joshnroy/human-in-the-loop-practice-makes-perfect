"""`Tossing3DStepProtocol` reproduces exactly what the step protocol hard-wired before
it was made domain-generic: the same human-skill names and their order,
state-log files and headers, skill providers, and control-step wiring.

None of this builds a simulator.
"""

import argparse
import json
from pathlib import Path

from hitl_pmp.cli import STEP_PROTOCOL_ENVIRONMENTS, Cli
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
from hitl_pmp.environments.tossing3d.state_log import StateLogHeader
from hitl_pmp.environments.tossing3d.step_protocol import Tossing3DStepProtocol


def _args(*, tmp_path: Path, extra: tuple[str, ...] | list[str] = ()) -> argparse.Namespace:
    return Cli.parse_args(
        argv=[
            "--env",
            "tossing3d",
            "--method",
            "ees",
            "--practice-step-budget",
            "2500",
            "--output-dir",
            str(tmp_path),
            *extra,
        ]
    )


def test_tossing3d_is_the_registered_step_protocol_domain():
    assert STEP_PROTOCOL_ENVIRONMENTS["tossing3d"] is Tossing3DStepProtocol


def test_the_pre_generalisation_constants_are_unchanged():
    # Order matters: it is the key order of `human_resets_by_destination` in every
    # status/measurement record, which existing analyses read.
    assert list(Tossing3DStepProtocol.human_skill_names().items()) == [
        ("robot_side", "reset_cube_and_bin_near"),
        ("opposite_side", "reset_cube_far"),
    ]


def test_the_cli_defaults_for_the_step_schedule_are_unchanged(*, tmp_path: Path) -> None:
    args = _args(tmp_path=tmp_path)
    assert args.measurement_interval_steps == 1700
    assert args.evaluation_control_steps == 500
    shelf_schedule = _args(
        tmp_path=tmp_path,
        extra=["--measurement-interval-steps", "15000", "--evaluation-control-steps", "3000"],
    )
    assert shelf_schedule.measurement_interval_steps == 15000
    assert shelf_schedule.evaluation_control_steps == 3000


def test_practice_context_matches_the_old_inline_wiring(*, tmp_path: Path) -> None:
    args = _args(tmp_path=tmp_path, extra=["--human-reset-practice-cost", "7"])
    problem = Tossing3DStepProtocol.build_practice_problem(args=args)
    context = Tossing3DStepProtocol.practice_context(args=args, problem=problem)
    assert context.env is problem.env
    assert isinstance(context.skill_provider, Tossing3DSkillProvider)
    assert context.skill_provider.human_reset_practice_cost == 7
    assert context.skill_provider.offer_human_reset == args.human_reset
    assert context.oracle.throw_standoff == args.oracle_throw_standoff
    evaluation = Tossing3DStepProtocol.build_evaluation_problem(args=args)
    provider = Tossing3DStepProtocol.evaluation_skill_provider(problem=evaluation)
    assert provider.env is evaluation.env
    assert provider.offer_human_reset is False


def test_state_logs_keep_their_old_names_and_headers(*, tmp_path: Path) -> None:
    args = _args(tmp_path=tmp_path)
    problem = Tossing3DStepProtocol.build_practice_problem(args=args)
    close = Tossing3DStepProtocol.attach_state_log(
        args=args, problem=problem, directory=tmp_path, evaluation=False
    )
    assert problem.env._state_log_writer is not None
    close()
    evaluation_directory = tmp_path / "evaluations" / "0000"
    evaluation_directory.mkdir(parents=True)
    evaluation = Tossing3DStepProtocol.build_evaluation_problem(args=args)
    Tossing3DStepProtocol.attach_state_log(
        args=args, problem=evaluation, directory=evaluation_directory, evaluation=True
    )()
    expected = StateLogHeader(
        layout=problem.env.layout,
        variant=args.variant,
        scene_bg=args.scene_bg,
        canonical_seed=args.canonical_seed,
        seed=args.seed,
        test_env_seed_offset=args.test_env_seed_offset,
    ).model_dump(mode="json")
    for path in (tmp_path / "tossing3d_state_log.jsonl", evaluation_directory / "states.jsonl"):
        header = json.loads(path.read_text().splitlines()[0])
        assert {key: header[key] for key in expected} == expected


def test_control_step_observers_reach_the_kinder_backend(*, tmp_path: Path) -> None:
    problem = Tossing3DStepProtocol.build_practice_problem(args=_args(tmp_path=tmp_path))
    calls = []
    problem.env.set_control_step_observers(
        before=lambda: calls.append("before"), after=lambda: calls.append("after")
    )
    backend = problem.env.backend()
    backend._before_control_step()
    backend._after_control_step()
    assert calls == ["before", "after"]
    problem.env.set_control_step_observers(before=None, after=None)
    assert backend._before_control_step is None and backend._after_control_step is None
