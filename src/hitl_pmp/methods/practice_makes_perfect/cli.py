import argparse

from hitl_pmp.cli_protocols import EnvironmentCli
from hitl_pmp.core.method.skill_provider import SkillProvider
from hitl_pmp.methods.help_seeking import HelpSeekingPolicy, HelpSeekingTrigger
from hitl_pmp.sampler_draws import SamplerDrawRecorder

from .ees_method import EesMethod
from .random_skills_method import RandomSkillsMethod


class HelpSeekingCli:
    """`--ask-for-help` and the two knobs that size it.

    **These are method flags, not harness flags, and the import layering makes that
    unarguable rather than a matter of taste.** `hitl_pmp.methods` sits *above*
    `hitl_pmp.method_runner`, so `MethodRunner` cannot import `help_seeking` and could
    not build the policy even if someone wanted it to. Deciding when to ask for help is
    part of the method; the harness only answers. What the human then *does* stays on
    the global CLI as `--human-reset-target`, because that is a property of the human.

    A separate container from `PracticeCycleCli` beside it, for the same reason that one
    exists: a second help-seeking Method should get these three flags described
    identically rather than paraphrased. A static-method container, never instantiated,
    same as every other business-logic class in this project."""

    @staticmethod
    def add_arguments(*, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--ask-for-help",
            type=HelpSeekingTrigger,
            choices=list(HelpSeekingTrigger),
            default=HelpSeekingTrigger.NEVER,
            help="When this method asks a human to reposition the robot during a "
            "practice period. 'never' (the default) is the incumbent behaviour: the "
            "method builds no help-seeking policy at all, so the run takes exactly the "
            "code path it took before this flag existed and needs no HumanOracle. "
            "'on-stuck' asks once --stuck-patience consecutive practice steps have all "
            "landed in states already visited this period -- an absorbing region, which "
            "is what Tossing Room's one-way ledge produces. 'at-random' asks on a "
            "schedule of its own instead, at --mean-steps-between-help-requests, "
            "whether or not the robot was getting anywhere; it is the control for "
            "'on-stuck', paying the same cost at the same rate with the timing carrying "
            "no information. 'on-no-applicable-skill' asks exactly when zero ground "
            "skills are applicable in the current state -- the method's own "
            "InteractionComplete condition, checked before it fires rather than after. "
            "Deliberately naive (no novelty tracking, no schedule): on a domain where "
            "some skill is nearly always applicable (Tossing Room's MoveRoom) it asks "
            "almost never, which is the point -- it is the baseline 'on-stuck' is "
            "motivated by. The period CONTINUES after a rescue rather than ending -- "
            "that is the difference from the method's own InteractionComplete, which is "
            "untouched by this flag. What the human does on arrival is "
            "--human-reset-target.",
        )
        parser.add_argument(
            "--stuck-patience",
            type=lambda value: HelpSeekingCli.parse_positive_int(value=value),
            default=20,
            help="How many consecutive already-visited states count as stuck. Only read "
            "under --ask-for-help on-stuck. The default is sized for Tossing Room, "
            "where a productive robot's pickup counter is monotone (so it reaches a "
            "novel state on essentially every step) while a robot stranded behind the "
            "ledge has only a handful of reachable states.",
        )
        parser.add_argument(
            "--mean-steps-between-help-requests",
            type=lambda value: HelpSeekingCli.parse_positive_int(value=value),
            default=150,
            help="Mean gap in policy calls between requests under --ask-for-help "
            "at-random: each call asks with probability 1/N. The default equals the "
            "usual --max-steps-per-interaction, so the arm gets about one rescue per "
            "period -- the rate a scheduled per-period reset would have given it free.",
        )

    @staticmethod
    def parse_positive_int(*, value: str) -> int:
        """argparse `type` for a flag that must be >= 1.

        Deliberately a second copy of `cli.py`'s `Cli.parse_positive_int` rather than a
        shared import: `hitl_pmp.cli` is the TOP layer and `hitl_pmp.methods` sits below
        it, so importing it from here would invert the dependency the import-linter
        contract exists to enforce. Six lines of duplication is the cheaper of the two
        costs."""
        parsed = int(value)
        if parsed < 1:
            raise argparse.ArgumentTypeError(f"must be >= 1, got {parsed}")
        return parsed

    @staticmethod
    def build_policy(
        *, args: argparse.Namespace, skill_provider: SkillProvider | None = None
    ) -> HelpSeekingPolicy | None:
        """This run's help-seeking policy, or None when the method should never ask.

        `None` for `--ask-for-help never`, which is the default and is what keeps every
        existing run byte-identical: the Method then returns its practice policy
        completely unwrapped, holds no detector, and draws no randomness.

        Seeded from the global `--seed`, so an arm's whole request schedule is fixed by
        the same number that fixes its task stream. That matters most for 'at-random': a
        constant seed here would have every seed of a sweep ask on identical steps,
        which is the BallRing `--noise-seed` trap. It costs nothing under 'on-stuck',
        which consumes no randomness at all.

        `skill_provider` is only read by the resulting policy under
        'on-no-applicable-skill'; passed through unconditionally (like
        `stuck_patience`/`mean_steps_between_requests`) rather than only when that
        trigger is selected, so a caller that switches `--ask-for-help` at the command
        line needs no code change here. `None` by default because most callers
        (`NEVER`/`ON_STUCK`/`AT_RANDOM` arms, and every existing test) have no provider
        to hand it and do not need one.

        `getattr` throughout because a hand-built Namespace -- a test, a one-off script
        -- predates every one of these flags."""
        trigger = getattr(args, "ask_for_help", HelpSeekingTrigger.NEVER)
        if trigger is HelpSeekingTrigger.NEVER:
            return None
        return HelpSeekingPolicy(
            trigger=trigger,
            stuck_patience=getattr(args, "stuck_patience", 20),
            mean_steps_between_requests=getattr(args, "mean_steps_between_help_requests", 150),
            skill_provider=skill_provider,
            seed=getattr(args, "seed", 0),
        )


class PracticeCycleCli:
    """The two flags every online-learning method-CLI in this subpackage needs, in
    one place so `--method ees` and `--method random-skills` describe the same
    protocol knobs identically (they have to be directly comparable: the paper
    plots every approach against the same online-transitions x-axis, which is
    num_cycles * max_steps_per_interaction). A static-method container, never
    instantiated, same as every other business-logic class in this project."""

    @staticmethod
    def add_arguments(
        *, parser: argparse.ArgumentParser, default_num_cycles: int, default_max_steps: int
    ) -> None:
        parser.add_argument(
            "--num-cycles",
            type=int,
            default=default_num_cycles,
            help="Number of online-learning cycles (the paper's 'free periods'). "
            "Each is one interaction period followed by one evaluation sweep.",
        )
        parser.add_argument(
            "--max-steps-per-interaction",
            type=int,
            default=default_max_steps,
            help="Environment steps per interaction period. The paper uses 150 for Light Switch.",
        )


class EesCli:
    """Plugs EesMethod (the paper's own method) into the global CLI under
    --method ees. A static-method container, never instantiated, same as every
    other business-logic class in this project."""

    @staticmethod
    def add_arguments(*, parser: argparse.ArgumentParser) -> None:
        # 10 cycles = predicators' num_online_learning_cycles default; 150 steps =
        # the paper's stated Light Switch free-period length.
        PracticeCycleCli.add_arguments(parser=parser, default_num_cycles=10, default_max_steps=150)
        # EES is the one Method that can ask for a human today -- it composes
        # HelpSeekingMixin. RandomSkillsCli deliberately does not register these.
        HelpSeekingCli.add_arguments(parser=parser)
        parser.add_argument(
            "--exploration-epsilon",
            type=float,
            default=EesMethod.model_fields["exploration_epsilon"].default,
            help="Epsilon-greedy exploration rate for the practice-time sampler "
            "(the paper uses 0.5).",
        )
        parser.add_argument(
            "--sampler-max-train-iters",
            type=int,
            default=EesMethod.model_fields["sampler_max_train_iters"].default,
            help="Gradient steps per sampler refit. The default 10000 matches "
            "predicators' settings.py default; the paper's launch configs override it "
            "to 100000, which trains the classifier to interpolation on Ball-Ring.",
        )
        # These three default TRUE to match predicators (which reproduces the paper);
        # pass --no-... to ablate a single deviation.
        parser.add_argument(
            "--reproduce-predicators-double-observe",
            action=argparse.BooleanOptionalAction,
            default=EesMethod.model_fields["reproduce_predicators_double_observe"].default,
            help="Match predicators' double-observe() (default on). --no-... counts a "
            "practice outcome once instead of twice.",
        )
        parser.add_argument(
            "--reproduce-predicators-practice-target-history",
            action=argparse.BooleanOptionalAction,
            default=EesMethod.model_fields["reproduce_predicators_practice_target_history"].default,
            help="Compute skip_perfect and the UCB num_tries/total from an all-attempts "
            "history (greedy + random), matching predicators' _ground_op_hist (default "
            "on). --no-... reads the random-excluding competence history instead.",
        )
        parser.add_argument(
            "--reproduce-predicators-explore-target-only",
            action=argparse.BooleanOptionalAction,
            default=EesMethod.model_fields["reproduce_predicators_explore_target_only"].default,
            help="Explore (epsilon-greedy) only on the practice-target skill, greedy "
            "for the prefix that reaches it, matching predicators (default on). "
            "--no-... explores every skill during practice.",
        )
        parser.add_argument(
            "--goal-pursuit-horizon",
            type=int,
            default=EesMethod.model_fields["goal_pursuit_horizon"].default,
            help="Skills spent pursuing the assigned train-task goal before switching "
            "to practice for the rest of the period (predicators' per-env CFG.horizon: "
            "8 for Ball-Ring, num_cells+2 for Light Switch). Omit for uncapped.",
        )
        parser.add_argument(
            "--planning-timeout",
            type=float,
            default=EesMethod.model_fields["planning_timeout"].default,
            help="Per-call Fast Downward timeout, in seconds.",
        )
        parser.add_argument(
            "--competence-window-size",
            type=int,
            default=EesMethod.model_fields["competence_window_size"].default,
            help="Optimistic competence model window (predicators' default 5; the "
            "paper overrides to 2 for the simulated Ball-Ring).",
        )
        parser.add_argument(
            "--competence-recency-size",
            type=int,
            default=EesMethod.model_fields["competence_recency_size"].default,
            help="Optimistic competence model recency window (predicators' default 5; "
            "the paper overrides to 2 for the simulated Ball-Ring).",
        )

    @staticmethod
    def run(*, args: argparse.Namespace, env_cli: type[EnvironmentCli]) -> None:
        # Built here rather than inside the factory lambda so a missing --output-dir is
        # rejected before the environment is constructed, and so it is built exactly
        # once even if a caller ever invokes the factory more than once (two recorders
        # on one path would truncate each other's file).
        draw_recorder = SamplerDrawRecorder.open_if_requested(args=args)
        env_cli.run_method(
            args=args,
            method_factory=lambda ctx: EesMethod(
                env=ctx.env,
                skill_provider=ctx.skill_provider,
                seed=args.seed,
                draw_recorder=draw_recorder,
                help_seeking=HelpSeekingCli.build_policy(
                    args=args, skill_provider=ctx.skill_provider
                ),
                exploration_epsilon=args.exploration_epsilon,
                sampler_max_train_iters=args.sampler_max_train_iters,
                goal_pursuit_horizon=args.goal_pursuit_horizon,
                planning_timeout=args.planning_timeout,
                competence_window_size=args.competence_window_size,
                competence_recency_size=args.competence_recency_size,
                reproduce_predicators_double_observe=args.reproduce_predicators_double_observe,
                reproduce_predicators_practice_target_history=(
                    args.reproduce_predicators_practice_target_history
                ),
                reproduce_predicators_explore_target_only=(
                    args.reproduce_predicators_explore_target_only
                ),
            ),
            num_cycles=args.num_cycles,
            max_steps_per_interaction=args.max_steps_per_interaction,
        )


class RandomSkillsCli:
    """Plugs RandomSkillsMethod into the global CLI under --method random-skills.
    Lives alongside RandomSkillsMethod under methods/practice_makes_perfect/ (not
    environments/lightswitch/), matching methods/oracle/cli.py's SkillOracleCli
    precedent: a method-CLI is method-specific glue, not environment-specific. A
    static-method container, never instantiated, same as every other
    business-logic class in this project."""

    @staticmethod
    def add_arguments(*, parser: argparse.ArgumentParser) -> None:
        """RandomSkillsMethod's own RNG reuses the global --seed (already
        registered by hitl_pmp/cli.py), so there's no separate seed flag here.

        --num-cycles defaults to 0 because this baseline never learns, so a single
        evaluation sweep tells you everything -- but it still accepts the flag, so
        it can be run over the *same* transition budget as --method ees when the
        two need to appear on one chart (this baseline collects transitions
        without improving, which is exactly the paper's point about it)."""
        PracticeCycleCli.add_arguments(parser=parser, default_num_cycles=0, default_max_steps=150)

    @staticmethod
    def run(*, args: argparse.Namespace, env_cli: type[EnvironmentCli]) -> None:
        env_cli.run_method(
            args=args,
            method_factory=lambda ctx: RandomSkillsMethod(
                env=ctx.env, skill_provider=ctx.skill_provider, seed=args.seed
            ),
            num_cycles=args.num_cycles,
            max_steps_per_interaction=args.max_steps_per_interaction,
        )
