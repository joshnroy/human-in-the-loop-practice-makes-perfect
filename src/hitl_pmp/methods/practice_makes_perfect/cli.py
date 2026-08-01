import argparse

from hitl_pmp.cli_protocols import EnvironmentCli

from .ees_method import EesMethod
from .random_skills_method import RandomSkillsMethod


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
            help="Gradient steps per sampler refit. predicators' own config uses "
            "100000; the default here is far lower so a run finishes in minutes.",
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
        # The three planning-progress-scoring deviations, each separately measurable. All
        # three default OFF (this port's current behavior): each one moves scores on
        # Ball-Ring, the domain being measured, so no default flips without its own run.
        # See the matching EesMethod fields for the evidence behind each.
        parser.add_argument(
            "--reproduce-predicators-planning-progress-task-prefix",
            action=argparse.BooleanOptionalAction,
            default=EesMethod.model_fields[
                "reproduce_predicators_planning_progress_task_prefix"
            ].default,
            help="Price planning progress against a fixed *prefix* of the seen train "
            "tasks (predicators' sorted(seen_idxs)[:max_tasks]), so the objective stops "
            "moving between cycles (default off: off prices the most recent ones, which "
            "is what this port did originally).",
        )
        parser.add_argument(
            "--reproduce-predicators-planning-progress-normalizer",
            action=argparse.BooleanOptionalAction,
            default=EesMethod.model_fields[
                "reproduce_predicators_planning_progress_normalizer"
            ].default,
            help="Divide the summed plan cost by the number of train tasks ever seen "
            "(predicators' forever-growing denominator, which steadily raises the UCB "
            "bonus's relative weight) rather than by the number of plans priced, which "
            "saturates (default off).",
        )
        parser.add_argument(
            "--reproduce-predicators-replanning-tasks",
            action=argparse.BooleanOptionalAction,
            default=EesMethod.model_fields["reproduce_predicators_replanning_tasks"].default,
            help="Also price planning progress against up to 5 fictitious tasks created "
            "by mid-period re-planning (predicators' _replanning_tasks deque, reset each "
            "cycle), which makes the score sensitive to where the robot actually is "
            "(default off).",
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
        env_cli.run_method(
            args=args,
            method_factory=lambda ctx: EesMethod(
                env=ctx.env,
                skill_provider=ctx.skill_provider,
                seed=args.seed,
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
                reproduce_predicators_planning_progress_task_prefix=(
                    args.reproduce_predicators_planning_progress_task_prefix
                ),
                reproduce_predicators_planning_progress_normalizer=(
                    args.reproduce_predicators_planning_progress_normalizer
                ),
                reproduce_predicators_replanning_tasks=(
                    args.reproduce_predicators_replanning_tasks
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
