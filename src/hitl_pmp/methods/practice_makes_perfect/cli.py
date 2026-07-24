import argparse

from hitl_pmp.environments.lightswitch.cli import LightSwitchCli

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
        parser.add_argument(
            "--reproduce-predicators-double-observe",
            action="store_true",
            help="Ablation: restore predicators' double-observe() bug, which counts "
            "a greedy practice outcome twice and a random one once. The paper's "
            "published curve contains it, so this is the comparable setting.",
        )
        parser.add_argument(
            "--reproduce-predicators-practice-target-history",
            action="store_true",
            help="Ablation: compute skip_perfect and the UCB num_tries/total from an "
            "all-attempts history (greedy + random), matching predicators' "
            "_ground_op_hist. Off by default, which reads the random-excluding "
            "competence history instead; competence itself is unaffected either way.",
        )
        parser.add_argument(
            "--planning-timeout",
            type=float,
            default=EesMethod.model_fields["planning_timeout"].default,
            help="Per-call Fast Downward timeout, in seconds.",
        )

    @staticmethod
    def run(*, args: argparse.Namespace) -> None:
        LightSwitchCli.run_method(
            args=args,
            method_factory=lambda env: EesMethod(
                env=env,
                seed=args.seed,
                exploration_epsilon=args.exploration_epsilon,
                sampler_max_train_iters=args.sampler_max_train_iters,
                planning_timeout=args.planning_timeout,
                reproduce_predicators_double_observe=args.reproduce_predicators_double_observe,
                reproduce_predicators_practice_target_history=(
                    args.reproduce_predicators_practice_target_history
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
    def run(*, args: argparse.Namespace) -> None:
        LightSwitchCli.run_method(
            args=args,
            method_factory=lambda env: RandomSkillsMethod(env=env, seed=args.seed),
            num_cycles=args.num_cycles,
            max_steps_per_interaction=args.max_steps_per_interaction,
        )
