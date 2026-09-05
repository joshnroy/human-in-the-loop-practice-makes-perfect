import argparse
import sys

from hitl_pmp.cli_protocols import EnvironmentCli
from hitl_pmp.methods.belief_space.tossing3d_method import Tossing3DPomdpMethod
from hitl_pmp.sampler_draws import SamplerDrawRecorder

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
        # EES can ask for a human via a real, planner-priced ground skill.
        # RandomSkillsCli has no planner, so it doesn't register this.
        parser.add_argument(
            "--ask-for-reset-cube-bin-cost",
            type=float,
            default=EesMethod.model_fields["ask_for_reset_cube_bin_cost"].default,
            help="Cost of the ask_for_reset_cube_bin_only ground skill: a mid-plan "
            "step, priced against a competence-based ceiling (see EesMethod.plan_to), "
            "built by the domain's own SkillProvider -- its effect (reposition "
            "whichever objects this domain calls 'movable, not the robot' to a "
            "freshly sampled ground pose) can only be written in terms of that "
            "domain's own predicates. Configuring this against a domain whose "
            "SkillProvider.human_cube_bin_reset_skill() has nothing to offer (every "
            "domain but Tossing3D today) is a misconfiguration plan_to reports rather "
            "than silently ignores. Omitted (the default, None) means the skill is not "
            "offered to the planner at all, so a run takes exactly the code path it "
            "took before this skill existed.",
        )
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
            help="Compute the UCB num_tries/total from an all-attempts history (greedy "
            "+ random), matching predicators' _ground_op_hist (default on). --no-... "
            "reads the random-excluding competence history instead.",
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
                ask_for_reset_cube_bin_cost=args.ask_for_reset_cube_bin_cost,
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


class Tossing3DPomdpCli(EesCli):
    """Exact belief-space practice selection for Tossing3D."""

    @staticmethod
    def add_arguments(*, parser: argparse.ArgumentParser) -> None:
        EesCli.add_arguments(parser=parser)
        parser.set_defaults(goal_pursuit_horizon=0)
        parser.add_argument(
            "--pomdp-num-samples",
            type=int,
            default=100,
            help="Theta samples per unique expectimax state.",
        )
        parser.add_argument(
            "--pomdp-search-depth",
            type=int,
            default=Tossing3DPomdpMethod.model_fields["pomdp_search_depth"].default,
            help="Exact belief-space expectimax depth in future skill executions.",
        )
        parser.add_argument(
            "--pomdp-num-particles",
            type=int,
            default=Tossing3DPomdpMethod.model_fields["pomdp_num_particles"].default,
            help="Particles per robot skill.",
        )

    @staticmethod
    def run(*, args: argparse.Namespace, env_cli: type[EnvironmentCli]) -> None:
        if env_cli.__name__ != "Tossing3DCli":
            raise ValueError("--method pomdp currently supports only --env tossing3d")
        # Preserve replayable simulator states, but do not render or encode frames in
        # the training process. POMDP runs reconstruct their diagnostic videos from
        # the state and decision logs after the experiment finishes.
        args.defer_rendering = True
        # Cached recursion adds interpreter frames per depth; this is not a search cutoff.
        sys.setrecursionlimit(max(sys.getrecursionlimit(), 10 * args.pomdp_search_depth + 1000))
        draw_recorder = SamplerDrawRecorder.open_if_requested(args=args)
        env_cli.run_method(
            args=args,
            method_factory=lambda ctx: Tossing3DPomdpMethod(
                env=ctx.env,
                skill_provider=ctx.skill_provider,
                seed=args.seed,
                draw_recorder=draw_recorder,
                ask_for_reset_cube_bin_cost=args.ask_for_reset_cube_bin_cost,
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
                pomdp_search_depth=args.pomdp_search_depth,
                pomdp_num_samples=args.pomdp_num_samples,
                pomdp_num_particles=args.pomdp_num_particles,
                decision_log=(
                    args.output_dir / "pomdp_decisions.jsonl"
                    if args.output_dir is not None
                    else None
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
