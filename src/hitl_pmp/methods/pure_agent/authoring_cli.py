import argparse
from pathlib import Path

from hitl_pmp.cli_protocols import EnvironmentCli
from hitl_pmp.core.method.method import Method
from hitl_pmp.core.method.skill_provider import DomainContext
from hitl_pmp.methods.practice_makes_perfect.cli import PracticeCycleCli
from hitl_pmp.methods.pure_agent.claude_code_backend import ClaudeCodeAgentBackend
from hitl_pmp.methods.pure_agent.prompts import (
    DEFAULT_MAX_DUMPED_TRANSITIONS,
    FeedbackArm,
    PromptArm,
)
from hitl_pmp.methods.pure_agent.pure_agent_method import PureAgentMethod
from hitl_pmp.methods.pure_agent.transcript_store import TranscriptStore


class PureAgentAuthoringCli:
    """`--method pure-agent-author`: the run that queries a real agent and records what it
    wrote. A static-method container, never instantiated, same as every other
    business-logic class in this project.

    **A separate `--method` from `pure-agent`, not a flag on it.** Authoring spends real
    money and cannot be reproduced; measurement must do neither. Two names means the two
    cannot be confused by a mistyped flag or a stale shell-history line, and it means the
    replay CLI has no code path that could reach a backend at all.

    **The numbers this run reports are not the result.** It takes the same actions a
    replay of its own transcript takes, so its `stats.json` is the same -- but the run
    that gets cited should be the replay, because that is the one anyone else can
    reproduce. What this run is *for* is `transcript.json`."""

    @staticmethod
    def add_arguments(*, parser: argparse.ArgumentParser) -> None:
        # 2 cycles = 3 authoring rounds (round 0 is authored before the first evaluation
        # sweep), which is the notebook's own round count and what its "few tens of cents"
        # cost figure is quoted at. Deliberately lower than EES's 10: every extra cycle
        # here is another paid query, and the replay -- which is what gets charted -- can
        # be re-run at any cycle count the transcript covers.
        PracticeCycleCli.add_arguments(parser=parser, default_num_cycles=2, default_max_steps=150)
        parser.add_argument(
            "--pure-agent-sandbox-dir",
            type=Path,
            required=True,
            help="The agent's persistent working directory. Its `policy.py` and its "
            "conversation both live here and both persist across rounds, which is what "
            "lets round k+1 revise round k's file rather than start over. Use a FRESH "
            "directory per authoring run: reusing one resumes the previous run's "
            "conversation and silently makes round 0 something other than a "
            "zero-feedback policy.",
        )
        parser.add_argument(
            "--pure-agent-prompt-arm",
            type=PromptArm,
            choices=list(PromptArm),
            default=PromptArm.MINIMAL,
            help="minimal: the symbolic layer alone (lifted skills, predicates, types, "
            "objects) -- exactly what a domain-agnostic planning Method knows. described: "
            "the same plus --pure-agent-domain-description, the analogue of the "
            "reference notebook naming its environment, which flags itself as a large "
            "hint.",
        )
        parser.add_argument(
            "--pure-agent-feedback",
            type=FeedbackArm,
            choices=list(FeedbackArm),
            default=FeedbackArm.ZERO_SHOT,
            help="zero-shot: between-round feedback is the per-lifted-skill x/y tallies "
            "alone, which is what this method shipped with. in-context: the same tallies "
            "PLUS a JSON dump of the individual parameterized practice executions -- what "
            "the policy saw, the parameters it chose, and whether they worked. Both arms "
            "get the tallies; the axis is the worked examples. 'Zero-shot' does not mean "
            "'no feedback'.",
        )
        parser.add_argument(
            "--pure-agent-max-dumped-transitions",
            type=int,
            default=DEFAULT_MAX_DUMPED_TRANSITIONS,
            help="Cap on how many practice transitions the in-context arm dumps. The most "
            "recent are kept and the prompt states x/y whenever it truncated. Unused on "
            "the zero-shot arm.",
        )
        parser.add_argument(
            "--pure-agent-domain-description",
            type=Path,
            default=None,
            help="File holding a natural-language account of the domain. Required by "
            "--pure-agent-prompt-arm described and unused by minimal.",
        )
        parser.add_argument(
            "--pure-agent-model",
            type=str,
            default=ClaudeCodeAgentBackend.model_fields["model"].default,
            help="Model the sandboxed Claude Code CLI runs.",
        )
        parser.add_argument(
            "--pure-agent-max-budget-usd",
            type=float,
            default=ClaudeCodeAgentBackend.model_fields["max_budget_usd_per_query"].default,
            help="Per-query spend cap. The whole run's worst case is this times "
            "(--num-cycles + 1).",
        )
        parser.add_argument(
            "--pure-agent-docker",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Run the agent in the prpl-agent-sandbox container (default on, and the "
            "safe mode: the only writable host path is the sandbox directory and an "
            "in-container firewall restricts network access). --no-pure-agent-docker "
            "runs the CLI on the host, where a hook still blocks writes outside the "
            "sandbox but shell commands can read the host filesystem.",
        )

    @staticmethod
    def run(*, args: argparse.Namespace, env_cli: type[EnvironmentCli]) -> None:
        if args.output_dir is None:
            # Up front, before a single paid query: an authoring run with nowhere to write
            # its transcript has spent money and kept nothing, which is the one failure
            # mode worth refusing rather than warning about.
            raise ValueError(
                "--method pure-agent-author requires --output-dir: the transcript is the "
                "whole point of the run, and without somewhere to write it the queries "
                "are paid for and discarded."
            )
        description = PureAgentAuthoringCli.read_description(args=args)
        # Captured out of the factory closure, because `run_method` returns None and the
        # transcript can only be read off the Method instance the harness actually drove.
        authored: list[PureAgentMethod] = []

        # noqa PLR0917: `method_factory` is `Callable[[DomainContext], Method]`, so the
        # parameter is positional by interface. A named function rather than the lambda
        # every other method-CLI uses, because this one has to keep the instance it built.
        def build(context: DomainContext) -> Method:  # noqa: PLR0917
            method = PureAgentMethod(
                env=context.env,
                skill_provider=context.skill_provider,
                prompt_arm=args.pure_agent_prompt_arm,
                domain_description=description,
                feedback_arm=args.pure_agent_feedback,
                max_dumped_transitions=args.pure_agent_max_dumped_transitions,
                backend=ClaudeCodeAgentBackend(
                    sandbox_dir=args.pure_agent_sandbox_dir,
                    model=args.pure_agent_model,
                    use_docker=args.pure_agent_docker,
                    max_budget_usd_per_query=args.pure_agent_max_budget_usd,
                ),
            )
            authored.append(method)
            return method

        try:
            env_cli.run_method(
                args=args,
                method_factory=build,
                num_cycles=args.num_cycles,
                max_steps_per_interaction=args.max_steps_per_interaction,
            )
        finally:
            # In a `finally` so a run that crashes half way still keeps every round it
            # already paid for. A partial transcript replays fine at a smaller
            # --num-cycles, and losing paid queries to an unrelated crash would be the
            # most expensive possible way to fail.
            PureAgentAuthoringCli.save(authored=authored, output_dir=args.output_dir)

    @staticmethod
    def save(*, authored: list[PureAgentMethod], output_dir: Path) -> None:
        if not authored:
            return
        transcript = authored[0].authoring_transcript()
        path = TranscriptStore.write(transcript=transcript, directory=output_dir)
        print(f"pure-agent transcript: {path} ({len(transcript.rounds)} rounds)")
        # Printed as counts, and the cost as a LOWER BOUND rather than a total: a round
        # whose CLI never emitted a cost contributes 0.0, and reporting that as the spend
        # would understate it silently.
        print(
            f"pure-agent authoring: {transcript.num_failed_rounds()}/"
            f"{len(transcript.rounds)} rounds produced no usable policy; "
            f"{transcript.num_malformed_decisions}/{transcript.num_decisions} decisions "
            "were malformed"
        )
        print(
            f"pure-agent spend: ${transcript.total_cost_usd():.4f} over "
            f"{len(transcript.rounds)} rounds "
            f"({transcript.num_rounds_missing_cost()}/{len(transcript.rounds)} rounds "
            "reported no cost, so this is a lower bound)"
        )
        print(
            f"be-the-policy price: {transcript.num_decisions} decisions this run, i.e. "
            "the API calls a variant querying inside the policy would have made"
        )

    @staticmethod
    def read_description(*, args: argparse.Namespace) -> str:
        """The DESCRIBED arm's domain text, or `""` on the MINIMAL arm.

        Read here rather than inside the Method so that a missing file fails before any
        paid query, and so the Method stays a plain string away from the filesystem."""
        path = args.pure_agent_domain_description
        if args.pure_agent_prompt_arm is not PromptArm.DESCRIBED:
            return ""
        if path is None:
            raise ValueError(
                "--pure-agent-prompt-arm described requires "
                "--pure-agent-domain-description. Without one this arm is byte-for-byte "
                "the minimal arm, and the two would be pooled by every reader downstream."
            )
        return Path(path).read_text()
