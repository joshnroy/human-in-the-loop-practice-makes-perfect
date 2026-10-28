import argparse
import sys
from pathlib import Path
from typing import Any, get_args

from hitl_pmp.cli_protocols import EnvironmentCli
from hitl_pmp.core.practice_costs import SkillStepCosts
from hitl_pmp.methods.belief_space.pouring_domain import PouringPracticeDomain
from hitl_pmp.methods.belief_space.practice_domain import (
    SWEEP_OFF_SHELF_DOMAIN,
    TOSSING3D_DOMAIN,
    PracticeDomain,
)
from hitl_pmp.methods.belief_space.tossing3d_method import Tossing3DPomdpMethod
from hitl_pmp.methods.belief_space.types.competence_evidence import CompetenceEvidence
from hitl_pmp.methods.belief_space.types.determinized import DeterminizedVariant, DrawScope
from hitl_pmp.sampler_draws import SamplerDrawRecorder

from .always_ask_reset import AlwaysAskReset
from .dp_method import DpMethod
from .dp_priors import DpPriorCalibration
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
    def positive_float(*, value: str) -> float:
        parsed = float(value)
        if not parsed > 0:
            raise argparse.ArgumentTypeError(f"must be > 0, got {value}")
        return parsed

    @staticmethod
    def always_ask_reset(*, args: argparse.Namespace, ctx: Any) -> AlwaysAskReset | None:
        if not args.always_ask_reset:
            return None
        return AlwaysAskReset.for_provider(
            skill_provider=ctx.skill_provider, cost=args.always_ask_reset_cost
        )

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
            help="Gradient steps per sampler refit. The default 10000 matches "
            "predicators' settings.py default; the paper's launch configs override it "
            "to 100000, which trains the classifier to interpolation on Ball-Ring.",
        )
        # Each default is the EesMethod field's; see that field for why it is what it
        # is. Only --reproduce-predicators-practice-target-history defaults on.
        parser.add_argument(
            "--reproduce-predicators-double-observe",
            action=argparse.BooleanOptionalAction,
            default=EesMethod.model_fields["reproduce_predicators_double_observe"].default,
            help="Observe each practice outcome an extra time (default off). Despite the "
            "name this is NOT predicators' behaviour: predicators observes once, only "
            "for non-exploration attempts, as the default does. On is a deviation.",
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
            "for the prefix that reaches it, matching predicators (default off). "
            "--no-... explores every skill during practice.",
        )
        parser.add_argument(
            "--reproduce-predicators-seen-task-order",
            action=argparse.BooleanOptionalAction,
            default=EesMethod.model_fields["reproduce_predicators_seen_task_order"].default,
            help="Score practice candidates against the first 10 seen tasks, as "
            "predicators' sorted(seen_train_task_idxs)[:10] does (default off: the 10 "
            "most recent).",
        )
        parser.add_argument(
            "--reproduce-predicators-skip-perfect",
            action=argparse.BooleanOptionalAction,
            default=EesMethod.model_fields["reproduce_predicators_skip_perfect"].default,
            help="Rank a candidate whose measured success rate is exactly 1.0 last, "
            "scoring it -inf, as predicators' active_sampler_explorer_skip_perfect does "
            "(default off).",
        )
        parser.add_argument(
            "--reproduce-predicators-random-when-stranded",
            action=argparse.BooleanOptionalAction,
            default=EesMethod.model_fields["reproduce_predicators_random_when_stranded"].default,
            help="When no practice candidate is reachable, act randomly (uniform "
            "initiable skill, one proposal draw, not flagged as exploration) for the rest "
            "of the period, as predicators' explorer does (default off: one random "
            "applicable skill, then back to scoring).",
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
            "--goal-pursuit-init-cycles",
            type=int,
            default=EesMethod.model_fields["goal_pursuit_init_cycles"].default,
            help="Pursue the task goal in the first N practice periods regardless of "
            "--goal-pursuit-interval (predicators' "
            "active_sampler_learning_init_cycles_to_pursue_goal, default 1).",
        )
        parser.add_argument(
            "--goal-pursuit-interval",
            type=int,
            default=EesMethod.model_fields["goal_pursuit_interval"].default,
            help="Pursue the task goal in periods where period %% N == 0 (predicators' "
            "active_sampler_learning_explore_pursue_goal_interval: 5 in settings.py, 1 in "
            "the paper's yaml). Default 1: every period, as before.",
        )
        parser.add_argument(
            "--planning-timeout",
            type=float,
            default=EesMethod.model_fields["planning_timeout"].default,
            help="Per-call Fast Downward timeout, in seconds.",
        )
        parser.add_argument(
            "--ees-reset-gate",
            action=argparse.BooleanOptionalAction,
            default=EesMethod.model_fields["reset_cost_gate"].default,
            help="Decline a practice plan whose human reset costs more than the "
            "priciest ordinary skill (default on). --no-... accepts any reset-using "
            "plan, so Fast Downward's cost minimisation alone chooses between a reset "
            "and a reset-free route. Evaluation never offers the reset either way.",
        )
        for name, field in (("alpha", "prior_alpha"), ("beta", "prior_beta")):
            parser.add_argument(
                f"--ees-prior-{name}",
                type=lambda value: EesCli.positive_float(value=value),
                default=EesMethod.model_fields[field].default,
                help=f"{name} of EES's Beta competence prior (default Beta(10, 1), the "
                "paper's). The SweepOffShelf3D rerun uses Beta(1, 3), as every other "
                "method there does: a labelled deviation from the paper.",
            )
        parser.add_argument(
            "--ees-score-human-skills",
            action=argparse.BooleanOptionalAction,
            # None, not the method's default, so --method pomdp can tell a flag it
            # would ignore from no flag at all.
            default=None,
            help="Score each human skill's outcome against its effects, track its "
            "competence and price it at its human cost plus -log(competence), like a "
            "robot skill (default on: human skills are first-class, a deviation from "
            "the paper). --no-ees-score-human-skills restores the paper's EES, whose "
            "human rescue always succeeds, as every EES run before 2026-10-25 used "
            "unless it passed this flag.",
        )
        parser.add_argument(
            "--always-ask-reset",
            action="store_true",
            help="Wrap the planner (EES, or --method pomdp with either solver) so the "
            "robot always asks: a human full reset to a freshly sampled start state after "
            "every skill with continuous parameters, success or failure, and whenever the "
            "goal is unreachable without a human. The planner still chooses every robot "
            "action. Off by default.",
        )
        parser.add_argument(
            "--always-ask-reset-cost",
            type=float,
            default=None,
            help="Price of one --always-ask-reset reset when no --practice-cost-config "
            "prices reset_to_sampled_start (default: the domain's own human-reset price, "
            "--human-reset-practice-cost).",
        )
        parser.add_argument(
            "--defer-rendering",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Build no in-run renderer, so no episode.mp4 or period videos are "
            "encoded during the run (default off: render as before). Read only by "
            "domains that render in-run (Tossing3D); --method pomdp forces it on.",
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
        # Resolved in place, so the config snapshot records the value the run used.
        if args.ees_score_human_skills is None:
            args.ees_score_human_skills = EesMethod.model_fields["score_human_skills"].default
        # Built here rather than inside the factory lambda so a missing --output-dir is
        # rejected before the environment is constructed, and so it is built exactly
        # once even if a caller ever invokes the factory more than once (two recorders
        # on one path would truncate each other's file).
        draw_recorder = SamplerDrawRecorder.open_if_requested(args=args)
        env_cli.run_method(
            args=args,
            method_factory=lambda ctx: EesMethod(
                **EesCli.method_kwargs(args=args, ctx=ctx, draw_recorder=draw_recorder)
            ),
            num_cycles=args.num_cycles,
            max_steps_per_interaction=args.max_steps_per_interaction,
        )

    @staticmethod
    def method_kwargs(
        *, args: argparse.Namespace, ctx: Any, draw_recorder: SamplerDrawRecorder | None
    ) -> dict[str, Any]:
        """EesMethod's constructor arguments from EES's flags, shared by every method
        that is EES with a different practice-target choice (`DpCli`)."""
        return dict(
            env=ctx.env,
            skill_provider=ctx.skill_provider,
            seed=args.seed,
            draw_recorder=draw_recorder,
            exploration_epsilon=args.exploration_epsilon,
            sampler_max_train_iters=args.sampler_max_train_iters,
            goal_pursuit_horizon=args.goal_pursuit_horizon,
            goal_pursuit_init_cycles=args.goal_pursuit_init_cycles,
            goal_pursuit_interval=args.goal_pursuit_interval,
            planning_timeout=args.planning_timeout,
            competence_window_size=args.competence_window_size,
            competence_recency_size=args.competence_recency_size,
            reset_cost_gate=args.ees_reset_gate,
            always_ask_reset=EesCli.always_ask_reset(args=args, ctx=ctx),
            score_human_skills=args.ees_score_human_skills,
            prior_alpha=args.ees_prior_alpha,
            prior_beta=args.ees_prior_beta,
            reproduce_predicators_double_observe=args.reproduce_predicators_double_observe,
            reproduce_predicators_practice_target_history=(
                args.reproduce_predicators_practice_target_history
            ),
            reproduce_predicators_explore_target_only=(
                args.reproduce_predicators_explore_target_only
            ),
            reproduce_predicators_seen_task_order=args.reproduce_predicators_seen_task_order,
            reproduce_predicators_skip_perfect=args.reproduce_predicators_skip_perfect,
            reproduce_predicators_random_when_stranded=(
                args.reproduce_predicators_random_when_stranded
            ),
        )


class Tossing3DPomdpCli(EesCli):
    """Exact belief-space practice selection for Tossing3D and SweepOffShelf3D."""

    # The practice domain for each supported environment CLI, by class name: a
    # method-CLI never imports an environment.
    DOMAINS: dict[str, PracticeDomain] = {
        "Tossing3DCli": TOSSING3D_DOMAIN,
        "Tossing3DRealCli": TOSSING3D_DOMAIN,
        "SweepOffShelf3DCli": SWEEP_OFF_SHELF_DOMAIN,
    }

    @staticmethod
    def num_rollouts(text: str, /) -> int | None:  # noqa: PLR0917 -- argparse's `type`
        if text == "all":
            return None
        value = int(text)
        if value < 1:
            raise argparse.ArgumentTypeError("num rollouts must be 'all' or at least 1")
        return value

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
            default=None,
            help="Exact belief-space expectimax depth in future skill executions "
            "(default: the domain's -- 3 for Tossing3D, 7 for SweepOffShelf3D).",
        )
        parser.add_argument(
            "--pomdp-solver",
            choices=("expectimax", "determinized_astar"),
            default=Tossing3DPomdpMethod.model_fields["pomdp_solver"].default,
            help="Belief-space traversal: exact expectimax or sampled best-first search.",
        )
        parser.add_argument(
            "--pomdp-max-search-iterations",
            type=int,
            default=Tossing3DPomdpMethod.model_fields["pomdp_max_search_iterations"].default,
            help="Priority-queue iterations per determinized A* search (Algorithm 3's H).",
        )
        parser.add_argument(
            "--pomdp-observation-probability-weight",
            type=float,
            default=Tossing3DPomdpMethod.model_fields[
                "pomdp_observation_probability_weight"
            ].default,
            help=(
                "Weight on negative log observation probability: averaged over outcomes "
                "in expectimax, or accumulated on the determinized sampled path."
            ),
        )
        parser.add_argument(
            "--pomdp-draw-scope",
            choices=get_args(DrawScope),
            default=Tossing3DPomdpMethod.model_fields["pomdp_draw_scope"].default,
            help="Determinized A* only: which search nodes share one sampled outcome per "
            "action. node: none (each node draws afresh, so a cheap reset re-rolls); "
            "state: nodes with the same atoms and acting-skill belief; skill: nodes with "
            "the same acting-skill belief, wherever it is attempted.",
        )
        parser.add_argument(
            "--pomdp-determinized-variant",
            choices=get_args(DeterminizedVariant),
            default=Tossing3DPomdpMethod.model_fields["pomdp_determinized_variant"].default,
            help="Determinized A* only. algorithm3: one sampled outcome per (node, action), "
            "nodes keyed with their cost. curtis: Curtis et al.'s PO_DAStar -- one child per "
            "distinct outcome, each charged -w log p, nodes keyed by the belief alone.",
        )
        parser.add_argument(
            "--pomdp-num-rollouts",
            type=Tossing3DPomdpCli.num_rollouts,
            default=Tossing3DPomdpMethod.model_fields["pomdp_num_rollouts"].default,
            help="Curtis variant only: outcome draws per (node, action), branching on their "
            "empirical frequencies; 'all' (the default) enumerates every outcome.",
        )
        parser.add_argument(
            "--pomdp-log-full-search-tree",
            action=argparse.BooleanOptionalAction,
            default=Tossing3DPomdpMethod.model_fields["pomdp_log_full_search_tree"].default,
            help="Log every determinized A* node, queue pop, and generated edge.",
        )
        parser.add_argument(
            "--pomdp-compress-decision-log",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Write pomdp_decisions.jsonl.zst, one zstd frame per record, instead of "
            "plain pomdp_decisions.jsonl. Belief dumps make the plain log reach TB per run.",
        )
        parser.add_argument(
            "--pomdp-num-particles",
            type=int,
            default=Tossing3DPomdpMethod.model_fields["pomdp_num_particles"].default,
            help="Particles per robot skill.",
        )
        parser.add_argument(
            "--pomdp-competence-model",
            choices=("global_curve", "local_trend"),
            default="local_trend",
            help="Competence dynamics: notebook Model A (phi curve) or Model B (local trend).",
        )
        parser.add_argument(
            "--pomdp-inference-engine",
            choices=("particle", "grid"),
            default="particle",
            help="Bayesian filtering representation; costs retain their shared particle model.",
        )
        parser.add_argument(
            "--pomdp-competence-evidence",
            choices=[evidence.value for evidence in CompetenceEvidence],
            default=Tossing3DPomdpMethod.model_fields["pomdp_competence_evidence"].default.value,
            help="Which toss practice outcomes condition competence: all attempts, "
            "all but epsilon-greedy random picks (default), or only the classifier's "
            "informed argmax picks. Every attempt still trains the sampler and "
            "advances the training clock; skills without a sampler are always "
            "conditioned.",
        )
        parser.add_argument("--pomdp-grid-competence-bins", type=int, default=25)
        parser.add_argument("--pomdp-grid-learning-rate-bins", type=int, default=16)
        parser.add_argument("--pomdp-competence-process-noise-std", type=float, default=0.03)
        parser.add_argument("--pomdp-learning-rate-decay", type=float, default=0.9)
        parser.add_argument("--pomdp-learning-rate-max", type=float, default=0.15)
        parser.add_argument(
            "--pomdp-learning-rate-process-noise-std",
            type=float,
            default=Tossing3DPomdpMethod.model_fields[
                "pomdp_learning_rate_process_noise_std"
            ].default,
            help="Model B learning-rate transition noise; "
            "eta is inferred from success/failure only.",
        )
        parser.add_argument(
            "--pomdp-learning-time-scale",
            type=float,
            default=Tossing3DPomdpMethod.model_fields["pomdp_learning_time_scale"].default,
            help="Positive multiplier for the forecast learning time scale; above one "
            "slows Model A's phi rates or Model B's rate magnitudes and decay. "
            "Actual training-example counts and competence noise stay unchanged.",
        )
        parser.add_argument(
            "--pomdp-linear-cost-lambda",
            type=float,
            default=Tossing3DPomdpMethod.model_fields["pomdp_linear_cost_lambda"].default,
            help="Select linear G(C, theta) = g(theta) - lambda * C with this "
            "coefficient. Omit to use hard-budget G with B=20 instead.",
        )
        parser.add_argument(
            "--pomdp-skill-step-costs",
            type=Path,
            default=None,
            help="A SkillStepCosts JSON (scripts/skill_step_costs.py): price each robot "
            "skill at this fixed number of control steps instead of the learned "
            "per-grounding average. The practice budget is still charged real steps.",
        )

    @staticmethod
    def run(*, args: argparse.Namespace, env_cli: type[EnvironmentCli]) -> None:
        if args.ees_score_human_skills is not None:
            raise ValueError(
                "--ees-score-human-skills is an EES flag: --method pomdp already scores "
                "every human skill its practice domain does not treat as deterministic"
            )
        defaults = (
            EesMethod.model_fields["prior_alpha"].default,
            EesMethod.model_fields["prior_beta"].default,
        )
        if (args.ees_prior_alpha, args.ees_prior_beta) != defaults:
            raise ValueError(
                "--ees-prior-alpha/--ees-prior-beta are EES flags: --method pomdp takes its "
                "competence priors from its practice domain"
            )
        domain = Tossing3DPomdpCli.DOMAINS.get(env_cli.__name__)
        if env_cli.__name__ == "FrozenPourCli":
            counts = tuple(args.requested_counts or (2,))
            if args.requested_count is not None:
                counts = (args.requested_count,)
            domain = PouringPracticeDomain.create(
                payload_count=args.payload_count, evaluation_counts=counts
            )
        if domain is None:
            raise ValueError("--method pomdp supports tossing3d, sweepoffshelf3d, and pouring3d")
        if args.pomdp_search_depth is None:
            # Resolved here so config_snapshot.json records the depth actually searched.
            args.pomdp_search_depth = domain.default_search_depth
        # Preserve replayable simulator states, but do not render or encode frames in
        # the training process. POMDP runs reconstruct their diagnostic videos from
        # the state and decision logs after the experiment finishes.
        args.defer_rendering = True
        # Cached recursion adds interpreter frames per depth; this is not a search cutoff.
        if args.pomdp_solver == "expectimax":
            sys.setrecursionlimit(max(sys.getrecursionlimit(), 10 * args.pomdp_search_depth + 1000))
        draw_recorder = SamplerDrawRecorder.open_if_requested(args=args)
        skill_step_costs = (
            None
            if args.pomdp_skill_step_costs is None
            else dict(SkillStepCosts.load(path=args.pomdp_skill_step_costs).skill_steps)
        )
        env_cli.run_method(
            args=args,
            method_factory=lambda ctx: Tossing3DPomdpMethod(
                env=ctx.env,
                skill_provider=ctx.skill_provider,
                seed=args.seed,
                draw_recorder=draw_recorder,
                exploration_epsilon=args.exploration_epsilon,
                sampler_max_train_iters=args.sampler_max_train_iters,
                goal_pursuit_horizon=args.goal_pursuit_horizon,
                goal_pursuit_init_cycles=args.goal_pursuit_init_cycles,
                goal_pursuit_interval=args.goal_pursuit_interval,
                planning_timeout=args.planning_timeout,
                competence_window_size=args.competence_window_size,
                competence_recency_size=args.competence_recency_size,
                reset_cost_gate=args.ees_reset_gate,
                always_ask_reset=EesCli.always_ask_reset(args=args, ctx=ctx),
                reproduce_predicators_double_observe=args.reproduce_predicators_double_observe,
                reproduce_predicators_practice_target_history=(
                    args.reproduce_predicators_practice_target_history
                ),
                reproduce_predicators_explore_target_only=(
                    args.reproduce_predicators_explore_target_only
                ),
                reproduce_predicators_seen_task_order=args.reproduce_predicators_seen_task_order,
                reproduce_predicators_skip_perfect=args.reproduce_predicators_skip_perfect,
                reproduce_predicators_random_when_stranded=(
                    args.reproduce_predicators_random_when_stranded
                ),
                pomdp_search_depth=args.pomdp_search_depth,
                pomdp_solver=args.pomdp_solver,
                pomdp_max_search_iterations=args.pomdp_max_search_iterations,
                pomdp_observation_probability_weight=args.pomdp_observation_probability_weight,
                pomdp_log_full_search_tree=args.pomdp_log_full_search_tree,
                pomdp_draw_scope=args.pomdp_draw_scope,
                pomdp_determinized_variant=args.pomdp_determinized_variant,
                pomdp_num_rollouts=args.pomdp_num_rollouts,
                pomdp_num_samples=args.pomdp_num_samples,
                pomdp_num_particles=args.pomdp_num_particles,
                pomdp_competence_model=args.pomdp_competence_model,
                pomdp_inference_engine=args.pomdp_inference_engine,
                pomdp_competence_evidence=CompetenceEvidence(args.pomdp_competence_evidence),
                pomdp_grid_competence_bins=args.pomdp_grid_competence_bins,
                pomdp_grid_learning_rate_bins=args.pomdp_grid_learning_rate_bins,
                pomdp_competence_process_noise_std=args.pomdp_competence_process_noise_std,
                pomdp_learning_rate_decay=args.pomdp_learning_rate_decay,
                pomdp_learning_rate_max=args.pomdp_learning_rate_max,
                pomdp_learning_rate_process_noise_std=(args.pomdp_learning_rate_process_noise_std),
                pomdp_learning_time_scale=args.pomdp_learning_time_scale,
                pomdp_linear_cost_lambda=args.pomdp_linear_cost_lambda,
                pomdp_skill_step_costs=skill_step_costs,
                pomdp_domain=domain,
                decision_log=(
                    args.output_dir
                    / (
                        "pomdp_decisions.jsonl.zst"
                        if args.pomdp_compress_decision_log
                        else "pomdp_decisions.jsonl"
                    )
                    if args.output_dir is not None
                    else None
                ),
            ),
            num_cycles=args.num_cycles,
            max_steps_per_interaction=args.max_steps_per_interaction,
        )


class DpCli(EesCli):
    """Plugs DpMethod, the Deliberate Practice baseline, into the global CLI under
    --method dp. It is EES with a different practice-target choice, so it takes every
    EES flag (its priors seed DP's competence models too) plus its own. Human skills
    are always scored: DpMethod requires it, whatever --ees-score-human-skills says."""

    @staticmethod
    def unit_interval(*, value: str) -> float:
        parsed = float(value)
        if not 0.0 <= parsed <= 1.0:
            raise argparse.ArgumentTypeError(f"must be in [0, 1], got {value}")
        return parsed

    @staticmethod
    def positive_int(*, value: str) -> int:
        parsed = int(value)
        if parsed < 1:
            raise argparse.ArgumentTypeError(f"must be >= 1, got {value}")
        return parsed

    @staticmethod
    def add_arguments(*, parser: argparse.ArgumentParser) -> None:
        EesCli.add_arguments(parser=parser)
        fields = DpMethod.model_fields
        parser.add_argument(
            "--dp-budget",
            type=lambda value: EesCli.positive_float(value=value),
            required=True,
            help="DP budget B in the explicit --dp-budget-unit: cost or target-skill episodes.",
        )
        parser.add_argument(
            "--dp-budget-unit",
            choices=["cost", "episodes"],
            default=fields["dp_budget_unit"].default,
            help="Cost includes robot steps and human resets; episodes count completed target "
            "rollouts, successful or failed, excluding setup executions and human resets.",
        )
        parser.add_argument(
            "--dp-allocation-schedule",
            choices=["once", "period"],
            default=fields["dp_allocation_schedule"].default,
            help="Allocate once before the curriculum, or reallocate every practice period.",
        )
        parser.add_argument(
            "--dp-solver",
            choices=["gurobi", "enumeration"],
            default=fields["dp_solver"].default,
            help="Gurobi globally solves the bilinear joint allocation/policy model. "
            "Enumeration is an explicit small-problem reference, never an automatic fallback.",
        )
        parser.add_argument(
            "--dp-solver-time-limit",
            type=lambda value: EesCli.positive_float(value=value),
            default=fields["dp_solver_time_limit"].default,
            help="Total Gurobi seconds per allocation, including least-cost tie breaking; "
            "failure to certify optimality raises instead of silently proceeding.",
        )
        parser.add_argument(
            "--dp-prior-calibration",
            type=Path,
            default=None,
            help="Validated held-out per-skill DP priors. Overrides robot competence and "
            "learning-rate priors without changing EES or human scoring. Source drift raises.",
        )
        parser.add_argument(
            "--dp-initial-learning-rate",
            type=float,
            default=fields["dp_initial_learning_rate"].default,
            help="Every skill's prior Delta, its predicted competence gain per practice "
            "attempt (default 0.1, the paper's illustrative value; it gives no "
            "experimental one).",
        )
        parser.add_argument(
            "--dp-smoothing",
            type=lambda value: DpCli.unit_interval(value=value),
            default=fields["dp_smoothing"].default,
            help="The smoothing factor eps of DP's running-average Delta update, in "
            "[0, 1] (default 0.5; the paper gives no experimental value).",
        )
        parser.add_argument(
            "--dp-max-skills",
            type=lambda value: DpCli.positive_int(value=value),
            default=fields["dp_max_skills"].default,
            help="Raise, rather than approximate, when more skills than this would be "
            "enumerated by the exact allocator.",
        )
        parser.add_argument(
            "--dp-max-allocations",
            type=lambda value: DpCli.positive_int(value=value),
            default=fields["dp_max_allocations"].default,
            help="Raise, rather than approximate, when the exact allocator could "
            "enumerate more allocations than this.",
        )
        parser.add_argument(
            "--dp-max-states",
            type=lambda value: DpCli.positive_int(value=value),
            default=fields["dp_max_states"].default,
            help="Raise rather than approximate when the reachable deployment MDP "
            "exceeds this many symbolic states.",
        )

    @staticmethod
    def run(*, args: argparse.Namespace, env_cli: type[EnvironmentCli]) -> None:
        if args.dp_prior_calibration is not None:
            calibration = DpPriorCalibration.load(path=args.dp_prior_calibration)
            if (calibration.environment, calibration.layout, calibration.variant) != (
                args.env,
                getattr(args, "layout", None),
                getattr(args, "variant", None),
            ):
                raise ValueError("DP prior calibration does not match this environment")
        draw_recorder = SamplerDrawRecorder.open_if_requested(args=args)
        env_cli.run_method(
            args=args,
            method_factory=lambda ctx: DpMethod(
                **{
                    **EesCli.method_kwargs(args=args, ctx=ctx, draw_recorder=draw_recorder),
                    "score_human_skills": True,
                },
                dp_budget=args.dp_budget,
                dp_budget_unit=args.dp_budget_unit,
                dp_allocation_schedule=args.dp_allocation_schedule,
                dp_solver=args.dp_solver,
                dp_solver_time_limit=args.dp_solver_time_limit,
                dp_prior_calibration=args.dp_prior_calibration,
                dp_solver_log=(
                    args.output_dir / "dp_allocations.jsonl" if args.output_dir else None
                ),
                dp_initial_learning_rate=args.dp_initial_learning_rate,
                dp_smoothing=args.dp_smoothing,
                dp_max_skills=args.dp_max_skills,
                dp_max_allocations=args.dp_max_allocations,
                dp_max_states=args.dp_max_states,
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
