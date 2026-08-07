import argparse
import json
from pathlib import Path

import pytest

from hitl_pmp.cli import METHODS, Cli
from hitl_pmp.methods.pure_agent.cli import PureAgentCli
from hitl_pmp.methods.pure_agent.prompts import PromptArm
from hitl_pmp.methods.pure_agent.transcript_store import TranscriptStore
from hitl_pmp.methods.pure_agent.types import AuthoringRound, AuthoringTranscript

FIRST_SKILL = """
def policy(observation):
    skill = observation["skills"][0]
    return {"skill_index": 0, "params": [0.0] * skill["param_dim"]}
"""


def _transcript(*, num_rounds: int) -> AuthoringTranscript:
    return AuthoringTranscript(
        rounds=[
            AuthoringRound(round_index=index, prompt=f"prompt {index}", policy_source=FIRST_SKILL)
            for index in range(num_rounds)
        ],
        num_decisions=1234,
    )


def test_the_method_is_registered_under_pure_agent() -> None:
    assert METHODS["pure-agent"] is PureAgentCli


def test_the_transcript_round_trips_through_the_store(*, tmp_path: Path) -> None:
    written = TranscriptStore.write(transcript=_transcript(num_rounds=3), directory=tmp_path)
    assert written == tmp_path / "transcript.json"
    loaded = TranscriptStore.read(path=written)
    assert loaded.policy_sources() == (FIRST_SKILL,) * 3
    assert loaded.num_decisions == 1234
    # The directory works as well as the file, so a caller holding --output-dir does not
    # have to remember the filename.
    assert TranscriptStore.read(path=tmp_path).policy_sources() == (FIRST_SKILL,) * 3


def test_each_round_is_also_written_as_readable_python(*, tmp_path: Path) -> None:
    """The single most informative output of this baseline is what the agent wrote, and
    nobody reads that out of a JSON string with escaped newlines."""
    TranscriptStore.write(transcript=_transcript(num_rounds=2), directory=tmp_path)
    assert (tmp_path / "round_000_policy.py").read_text() == FIRST_SKILL
    assert (tmp_path / "round_001_policy.py").read_text() == FIRST_SKILL


def test_reading_a_missing_transcript_says_what_is_missing(*, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no pure-agent transcript"):
        TranscriptStore.read(path=tmp_path / "nope.json")


def test_a_transcript_whose_rounds_do_not_cover_the_cycles_is_a_replay_error(
    *,
    tmp_path: Path,
) -> None:
    """The failure that would otherwise be invisible: too few recorded rounds would hold
    the last policy for every remaining checkpoint and flatten the learning curve."""
    TranscriptStore.write(transcript=_transcript(num_rounds=1), directory=tmp_path)
    args = _parse(
        argv=[
            "--env",
            "lightswitch",
            "--method",
            "pure-agent",
            "--pure-agent-replay",
            str(tmp_path),
            "--num-cycles",
            "2",
            "--max-steps-per-interaction",
            "3",
            "--num-test-tasks",
            "1",
            "--grid-size",
            "3",
        ]
    )
    with pytest.raises(RuntimeError, match="no recorded source"):
        PureAgentCli.run(args=args, env_cli=_env_cli(args=args))


def test_a_replay_run_completes_end_to_end_and_writes_stats(*, tmp_path: Path) -> None:
    """The whole replay path through the real global CLI: no agent, no network, no
    Docker, and a stats.json at the end of it."""
    transcript_dir = tmp_path / "transcript"
    TranscriptStore.write(transcript=_transcript(num_rounds=3), directory=transcript_dir)
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    Cli.main(
        argv=[
            "--env",
            "lightswitch",
            "--method",
            "pure-agent",
            "--pure-agent-replay",
            str(transcript_dir),
            "--num-cycles",
            "2",
            "--max-steps-per-interaction",
            "3",
            "--num-test-tasks",
            "2",
            "--grid-size",
            "3",
            "--output-dir",
            str(output_dir),
        ]
    )
    stats = json.loads((output_dir / "stats.json").read_text())
    assert len(stats["evaluations"]) == 3  # the initial sweep plus one per cycle


def test_two_replays_of_one_transcript_produce_a_byte_identical_stats_json(
    *,
    tmp_path: Path,
) -> None:
    """The property the whole record-then-replay design exists for. Authoring is
    nondeterministic; a measured run must not be."""
    transcript_dir = tmp_path / "transcript"
    TranscriptStore.write(transcript=_transcript(num_rounds=2), directory=transcript_dir)

    def run(*, name: str) -> bytes:
        output_dir = tmp_path / name
        output_dir.mkdir()
        Cli.main(
            argv=[
                "--env",
                "lightswitch",
                "--method",
                "pure-agent",
                "--pure-agent-replay",
                str(transcript_dir),
                "--num-cycles",
                "1",
                "--max-steps-per-interaction",
                "4",
                "--num-test-tasks",
                "2",
                "--grid-size",
                "3",
                "--seed",
                "7",
                "--output-dir",
                str(output_dir),
            ]
        )
        return (output_dir / "stats.json").read_bytes()

    assert run(name="a") == run(name="b")


def test_the_prompt_arm_flag_is_recorded_but_does_not_drive_a_replay() -> None:
    """A replay builds no prompts, so the arm is provenance rather than configuration --
    and `described` must not demand a description file a replay has no use for."""
    args = _parse(
        argv=[
            "--env",
            "lightswitch",
            "--method",
            "pure-agent",
            "--pure-agent-replay",
            "unused",
            "--pure-agent-prompt-arm",
            "described",
        ]
    )
    assert args.pure_agent_prompt_arm is PromptArm.DESCRIBED


def test_replay_is_required_because_this_cli_can_never_author() -> None:
    with pytest.raises(SystemExit):
        _parse(argv=["--env", "lightswitch", "--method", "pure-agent"])


def _parse(*, argv: list[str]) -> argparse.Namespace:
    return Cli.parse_args(argv=argv)


def _env_cli(*, args: argparse.Namespace) -> type:
    from hitl_pmp.cli import ENVIRONMENTS

    return ENVIRONMENTS[args.env]
