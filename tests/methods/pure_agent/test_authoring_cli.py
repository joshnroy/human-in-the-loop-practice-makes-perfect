import argparse
import importlib.util
from pathlib import Path

import pytest

from hitl_pmp.cli import METHODS, Cli
from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
from hitl_pmp.environments.lightswitch.skill_provider import LightSwitchSkillProvider
from hitl_pmp.methods.pure_agent.agent_backend import ScriptedAgentBackend
from hitl_pmp.methods.pure_agent.authoring_cli import PureAgentAuthoringCli
from hitl_pmp.methods.pure_agent.claude_code_backend import ClaudeCodeAgentBackend
from hitl_pmp.methods.pure_agent.prompts import PromptArm
from hitl_pmp.methods.pure_agent.pure_agent_method import PureAgentMethod
from hitl_pmp.methods.pure_agent.transcript_store import TranscriptStore

# Nothing in this file makes an API call or starts a container. The authoring CLI's
# argument handling, its transcript writing and the real backend's filesystem half are
# all exercisable for free; the one thing that costs money -- `query` -- is the
# experiment, not a test.
HAS_PRPL_AGENT_UTILS = importlib.util.find_spec("prpl_agent_utils") is not None

FIRST_SKILL = """
def policy(observation):
    return {"skill_index": 0, "params": [0.0] * observation["skills"][0]["param_dim"]}
"""


def test_authoring_is_a_separate_method_from_replay() -> None:
    """Two names, so a measured run has no code path that could reach a backend."""
    assert METHODS["pure-agent-author"] is PureAgentAuthoringCli
    assert METHODS["pure-agent"] is not PureAgentAuthoringCli


def test_authoring_without_an_output_dir_is_refused_before_any_query() -> None:
    """The one failure worth refusing rather than warning about: paid queries discarded."""
    args = Cli.parse_args(
        argv=[
            "--env",
            "lightswitch",
            "--method",
            "pure-agent-author",
            "--pure-agent-sandbox-dir",
            "unused",
        ]
    )
    with pytest.raises(ValueError, match="requires --output-dir"):
        PureAgentAuthoringCli.run(args=args, env_cli=_env_cli())


def test_the_described_arm_needs_a_description_file(*, tmp_path: Path) -> None:
    args = _args(sandbox=tmp_path, arm="described", description=None)
    with pytest.raises(ValueError, match="requires --pure-agent-domain-description"):
        PureAgentAuthoringCli.read_description(args=args)


def test_the_described_arm_reads_the_description_from_disk(*, tmp_path: Path) -> None:
    path = tmp_path / "domain.md"
    path.write_text("A robot walks a row of cells.")
    args = _args(sandbox=tmp_path, arm="described", description=path)
    assert PureAgentAuthoringCli.read_description(args=args) == "A robot walks a row of cells."


def test_the_minimal_arm_ignores_a_description_it_was_given(*, tmp_path: Path) -> None:
    """So that a single sweep script can pass the same flags to both arms and have the
    arm alone decide what the agent sees."""
    path = tmp_path / "domain.md"
    path.write_text("A robot walks a row of cells.")
    args = _args(sandbox=tmp_path, arm="minimal", description=path)
    assert PureAgentAuthoringCli.read_description(args=args) == ""


def test_save_writes_the_transcript_and_the_readable_policies(*, tmp_path: Path) -> None:
    env = LightSwitchEnvironment(grid_size=3)
    method = PureAgentMethod(
        env=env,
        skill_provider=LightSwitchSkillProvider(env=env),
        backend=ScriptedAgentBackend(sources=(FIRST_SKILL, FIRST_SKILL)),
    )
    # end_cycle authors round 0 lazily and then the revision, so this is two rounds --
    # the same N+1 shape a --num-cycles N run produces.
    method.end_cycle()
    PureAgentAuthoringCli.save(authored=[method], output_dir=tmp_path)
    assert TranscriptStore.read(path=tmp_path).policy_sources() == (FIRST_SKILL, FIRST_SKILL)
    assert (tmp_path / "round_000_policy.py").read_text() == FIRST_SKILL
    assert (tmp_path / "round_001_policy.py").read_text() == FIRST_SKILL


def test_save_with_no_method_built_writes_nothing(*, tmp_path: Path) -> None:
    """Reached when the run raised before the method factory ever fired. There is nothing
    to record, and writing an empty transcript would leave something replayable that
    represents no authoring at all."""
    PureAgentAuthoringCli.save(authored=[], output_dir=tmp_path)
    assert not (tmp_path / "transcript.json").exists()


def test_the_backend_reads_policy_py_out_of_the_sandbox(*, tmp_path: Path) -> None:
    backend = ClaudeCodeAgentBackend(sandbox_dir=tmp_path)
    assert backend.policy_source() is None
    (tmp_path / "policy.py").write_text(FIRST_SKILL)
    assert backend.policy_source() == FIRST_SKILL


def test_the_backend_describes_itself_by_model_and_isolation(*, tmp_path: Path) -> None:
    """Recorded into the transcript, because a transcript authored on the host and one
    authored in the container are different evidence."""
    assert ClaudeCodeAgentBackend(sandbox_dir=tmp_path).describe() == "claude-code-sonnet-docker"
    assert (
        ClaudeCodeAgentBackend(sandbox_dir=tmp_path, use_docker=False).describe()
        == "claude-code-sonnet-host"
    )


def test_docker_is_on_by_default(*, tmp_path: Path) -> None:
    """The package default and its safe mode: the only writable host path is the sandbox
    and an in-container firewall restricts the network. A silent downgrade to host mode
    would be a real change in what the agent can reach."""
    assert ClaudeCodeAgentBackend(sandbox_dir=tmp_path).use_docker is True


def test_a_missing_dependency_says_how_to_fix_it(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CI never installs `prpl_agent_utils`, so this is the message a CI-shaped machine
    gets. Importing this module must NOT fail there, which is why the import is lazy."""
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    with pytest.raises(RuntimeError, match="pip install"):
        ClaudeCodeAgentBackend(sandbox_dir=tmp_path).get_agent()


@pytest.mark.skipif(not HAS_PRPL_AGENT_UTILS, reason="prpl_agent_utils is not installed")
def test_the_underlying_agent_is_constructed_once_and_reused(*, tmp_path: Path) -> None:
    """Construction touches no network and starts no container -- but the SAME instance
    has to come back every time, because the sandbox and the conversation persisting
    across rounds is the whole mechanism."""
    backend = ClaudeCodeAgentBackend(sandbox_dir=tmp_path)
    assert backend.get_agent() is backend.get_agent()


def test_a_budget_capped_query_keeps_its_file_and_its_cost(*, tmp_path: Path) -> None:
    """The upstream shape that cost a round on this backend's first smoke run: the CLI
    stops on its own --max-budget-usd, writes `policy.py`, reports the exact cost, and
    exits in a way `prpl_agent_utils` raises on. The work is paid for either way, so it is
    salvaged rather than discarded -- and the round stays marked as cut off."""
    log = tmp_path / ".agent_logs"
    log.mkdir()
    (log / "stream.jsonl").write_text(
        '{"type": "assistant"}\n'
        '{"type": "result", "subtype": "error_max_budget_usd", "is_error": true, '
        '"num_turns": 2, "total_cost_usd": 0.886295}\n'
    )
    recovered = ClaudeCodeAgentBackend(sandbox_dir=tmp_path).recover_metadata()
    assert recovered["total_cost_usd"] == pytest.approx(0.886295)
    assert recovered["num_turns"] == 2
    assert recovered["stop_reason"] == "error_max_budget_usd"


def test_an_unreadable_stream_log_leaves_the_cost_honestly_unknown(*, tmp_path: Path) -> None:
    """`{}` rather than zeros: a round whose cost could not be recovered must be counted
    by num_rounds_missing_cost, not silently reported as a free query."""
    backend = ClaudeCodeAgentBackend(sandbox_dir=tmp_path)
    assert backend.recover_metadata() == {}
    log = tmp_path / ".agent_logs"
    log.mkdir()
    (log / "stream.jsonl").write_text("not json\n{}\n")
    assert backend.recover_metadata() == {}


def _args(*, sandbox: Path, arm: str, description: Path | None) -> argparse.Namespace:
    argv = [
        "--env",
        "lightswitch",
        "--method",
        "pure-agent-author",
        "--pure-agent-sandbox-dir",
        str(sandbox),
        "--pure-agent-prompt-arm",
        arm,
    ]
    if description is not None:
        argv += ["--pure-agent-domain-description", str(description)]
    return Cli.parse_args(argv=argv)


def _env_cli() -> type:
    from hitl_pmp.cli import ENVIRONMENTS

    return ENVIRONMENTS["lightswitch"]


def test_the_prompt_arm_parses_to_the_enum(*, tmp_path: Path) -> None:
    assert (
        _args(sandbox=tmp_path, arm="described", description=None).pure_agent_prompt_arm
        is PromptArm.DESCRIBED
    )
