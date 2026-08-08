import argparse
from pathlib import Path

from hitl_pmp.cli_protocols import EnvironmentCli
from hitl_pmp.methods.practice_makes_perfect.cli import PracticeCycleCli
from hitl_pmp.methods.pure_agent.claude_code_backend import ClaudeCodeAgentBackend
from hitl_pmp.methods.pure_agent.prompts import PromptArm
from hitl_pmp.methods.pure_agent.pure_agent_method import PureAgentMethod


class PureAgentCli:
    """Plugs `PureAgentMethod` into the global CLI under `--method pure-agent`. A
    static-method container, never instantiated, same as every other business-logic class
    in this project.

    **Every run driven from here queries a real agent and spends real subscription
    allowance**, once per environment step. That is the baseline, not an option: there is
    no cheap mode behind this flag, and a run's size is set entirely by `--num-cycles`,
    `--max-steps-per-interaction` and `--num-test-tasks`. Multiply them before launching --
    the decision count is `num_cycles * max_steps + (num_cycles + 1) * num_test_tasks *
    horizon`, and every one of those is a network call.

    It borrows `PracticeCycleCli` from `methods/practice_makes_perfect/` rather than
    declaring its own `--num-cycles`/`--max-steps-per-interaction`. That class exists so
    every online-learning arm describes the same protocol knobs identically, and this arm
    has to sit on the same online-transitions x-axis as EES and Random Skills or the three
    cannot go on one chart."""

    @staticmethod
    def add_arguments(*, parser: argparse.ArgumentParser) -> None:
        # The same 10/150 as EES, for the axis-sharing reason above. They are the DEFAULTS
        # rather than a recommendation here: at 10 cycles a Tossing Room run is ~5,500
        # agent calls, so most runs will want fewer.
        PracticeCycleCli.add_arguments(parser=parser, default_num_cycles=10, default_max_steps=150)
        parser.add_argument(
            "--pure-agent-sandbox-dir",
            type=Path,
            required=True,
            help="Directory the agent's two sandboxes live under: <dir>/practice and "
            "<dir>/evaluation. They are separate on purpose -- the split between them is "
            "the practice/evaluation firewall. Sessions and the CLI's own stream logs "
            "persist here, so point it somewhere with room and keep it out of the repo.",
        )
        parser.add_argument(
            "--pure-agent-model",
            type=str,
            default=ClaudeCodeAgentBackend.model_fields["model"].default,
            help="Model the Claude Code CLI runs, for BOTH backends. Opus by default: "
            "this baseline is the one that is supposed to be good at the reasoning, and "
            "a per-step call is one short turn, so the model is not where the cost is.",
        )
        parser.add_argument(
            "--pure-agent-max-total-cost-usd",
            type=float,
            required=True,
            help="Hard ceiling on this run's subscription allowance, in API-equivalent "
            "USD. REQUIRED, and required rather than defaulted on purpose: a run makes "
            "one agent call per environment step (thousands of them), the weekly "
            "allowance it draws on has no overflow and cannot be topped up, and at 100%% "
            "every agent on this machine stops until the window resets. Once the ceiling "
            "is reached the run makes no further calls and finishes on no-ops -- its "
            "results from that point are not a measurement of the method, and it says so "
            "on stderr. Pass a negative number to disable it, which you should not do "
            "without having done the arithmetic.",
        )
        parser.add_argument(
            "--pure-agent-max-cost-usd-per-query",
            type=float,
            default=ClaudeCodeAgentBackend.model_fields["max_budget_usd_per_query"].default,
            help="Per-call cap handed to the Claude Code CLI. Bounds one long-tailed "
            "call; --pure-agent-max-total-cost-usd is what bounds the run. A capped call "
            "that had already answered still yields its answer (the CLI's stop discards "
            "the result, so it is recovered from the stream log).",
        )
        parser.add_argument(
            "--pure-agent-prompt-arm",
            type=PromptArm,
            choices=list(PromptArm),
            default=PromptArm.MINIMAL,
            help="minimal: the symbolic layer only, exactly what a planning Method sees. "
            "described: additionally a natural-language account of the domain from "
            "--pure-agent-domain-description, which is the hint the prpl-agent-utils "
            "notebook flags in its own prompt.",
        )
        parser.add_argument(
            "--pure-agent-domain-description",
            type=str,
            default="",
            help="The natural-language account of the domain used by "
            "--pure-agent-prompt-arm described. Ignored on the minimal arm.",
        )
        parser.add_argument(
            "--pure-agent-use-docker",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Run the CLI inside the prpl-agent-sandbox container. Defaults OFF, "
            "which is a deviation from prpl-agent-utils' own default and is stated as "
            "one: this agent is given no tools, so the only thing it can emit is text, "
            "and this machine's Docker socket is not accessible to this user anyway.",
        )

    @staticmethod
    def run(*, args: argparse.Namespace, env_cli: type[EnvironmentCli]) -> None:
        sandbox_root: Path = args.pure_agent_sandbox_dir
        if args.pure_agent_prompt_arm is PromptArm.DESCRIBED and not (
            args.pure_agent_domain_description.strip()
        ):
            # Up front, before anything is spent. The described arm IS the description;
            # running it with an empty one would produce a run labelled `described` in
            # config_snapshot.json whose prompts are byte-identical to the minimal arm,
            # and the two would then be compared as if they differed.
            raise ValueError(
                "--pure-agent-prompt-arm described needs a non-empty "
                "--pure-agent-domain-description: without one the arm is the minimal "
                "arm wearing the other arm's name."
            )
        env_cli.run_method(
            args=args,
            method_factory=lambda ctx: PureAgentMethod(
                env=ctx.env,
                skill_provider=ctx.skill_provider,
                practice_backend=PureAgentCli.backend(args=args, sandbox=sandbox_root / "practice"),
                evaluation_backend=PureAgentCli.backend(
                    args=args, sandbox=sandbox_root / "evaluation"
                ),
                prompt_arm=args.pure_agent_prompt_arm,
                domain_description=args.pure_agent_domain_description,
                ledger_path=(
                    args.output_dir / "agent_calls.jsonl" if args.output_dir is not None else None
                ),
                # A negative value is how the flag says "no ceiling", so it maps to the
                # field's own `None`. Zero is NOT that: it means "spend nothing", which is
                # a legitimate thing to ask for and a useful dry run.
                max_total_cost_usd=(
                    None
                    if args.pure_agent_max_total_cost_usd < 0
                    else args.pure_agent_max_total_cost_usd
                ),
            ),
            num_cycles=args.num_cycles,
            max_steps_per_interaction=args.max_steps_per_interaction,
        )

    @staticmethod
    def backend(*, args: argparse.Namespace, sandbox: Path) -> ClaudeCodeAgentBackend:
        """One of the run's two backends. Built by a helper rather than inline twice so
        the two cannot drift in model or transport -- a practice agent on a different
        model from the evaluation agent would make the arm meaningless and would be
        invisible in the output."""
        return ClaudeCodeAgentBackend(
            sandbox_dir=sandbox,
            model=args.pure_agent_model,
            use_docker=args.pure_agent_use_docker,
            max_budget_usd_per_query=args.pure_agent_max_cost_usd_per_query,
        )
