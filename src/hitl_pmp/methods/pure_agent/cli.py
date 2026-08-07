import argparse
from pathlib import Path

from hitl_pmp.cli_protocols import EnvironmentCli
from hitl_pmp.methods.practice_makes_perfect.cli import PracticeCycleCli
from hitl_pmp.methods.pure_agent.prompts import FeedbackArm, PromptArm
from hitl_pmp.methods.pure_agent.pure_agent_method import PureAgentMethod
from hitl_pmp.methods.pure_agent.transcript_store import TranscriptStore


class PureAgentCli:
    """Plugs `PureAgentMethod` into the global CLI under `--method pure-agent`. A
    static-method container, never instantiated, same as every other business-logic class
    in this project.

    **Replay only, today.** Every run driven from here reads a recorded authoring
    transcript and makes no API call at all, which is what makes it deterministic,
    seed-pinned and safe to fan out across a sweep. Authoring -- which queries a real
    agent, spends money and cannot be reproduced -- is a separate entrypoint, added with
    the backend that does it. That is the record-then-replay split, and putting the two
    behind one flag would make it possible to spend money by forgetting one.

    It borrows `PracticeCycleCli` from `methods/practice_makes_perfect/` rather than
    declaring its own `--num-cycles`/`--max-steps-per-interaction`. That class exists so
    every online-learning arm describes the same protocol knobs identically, and this arm
    has to sit on the same online-transitions x-axis as EES and Random Skills or the
    three cannot go on one chart. Two copies of the two flags would be two places for
    their help text and defaults to drift."""

    @staticmethod
    def add_arguments(*, parser: argparse.ArgumentParser) -> None:
        # The same 10/150 as EES, for the axis-sharing reason above. A replay's cycle
        # count must not exceed the authored transcript's rounds minus one -- the method
        # raises rather than reusing the last round if it does.
        PracticeCycleCli.add_arguments(parser=parser, default_num_cycles=10, default_max_steps=150)
        parser.add_argument(
            "--pure-agent-replay",
            type=Path,
            required=True,
            help="Transcript to replay: either the transcript.json an authoring run "
            "wrote, or the directory holding it. The policy authored for round k is used "
            "for cycle k, so a --num-cycles N replay needs N+1 recorded rounds (round 0 "
            "is authored before the first evaluation sweep). REQUIRED: a run driven from "
            "this CLI never queries an agent, so there is nothing to fall back to.",
        )
        parser.add_argument(
            "--pure-agent-prompt-arm",
            type=PromptArm,
            choices=list(PromptArm),
            default=PromptArm.MINIMAL,
            help="Which prompt arm the transcript being replayed was authored under. "
            "Recorded into config_snapshot.json so a replayed run says which arm it "
            "belongs to; it does not change what a replay does, since a replay builds no "
            "prompts.",
        )
        parser.add_argument(
            "--pure-agent-feedback",
            type=FeedbackArm,
            choices=list(FeedbackArm),
            default=FeedbackArm.ZERO_SHOT,
            help="Which feedback arm the transcript being replayed was authored under. "
            "Recorded into config_snapshot.json for the same reason as the prompt arm, "
            "and with the same non-effect: a replay builds no prompts, so this cannot "
            "change what it does. It is how a results directory says which arm it is, "
            "which matters when the two arms differ only in a transcript.",
        )

    @staticmethod
    def run(*, args: argparse.Namespace, env_cli: type[EnvironmentCli]) -> None:
        transcript = TranscriptStore.read(path=args.pure_agent_replay)
        env_cli.run_method(
            args=args,
            method_factory=lambda ctx: PureAgentMethod(
                env=ctx.env,
                skill_provider=ctx.skill_provider,
                replay_sources=transcript.policy_sources(),
                # Never DESCRIBED on a replay path: the validator would demand a
                # description this run has no use for (it builds no prompts), and the arm
                # a replay belongs to is already recorded by config_snapshot.json off the
                # flag itself.
                prompt_arm=PromptArm.MINIMAL,
            ),
            num_cycles=args.num_cycles,
            max_steps_per_interaction=args.max_steps_per_interaction,
        )
