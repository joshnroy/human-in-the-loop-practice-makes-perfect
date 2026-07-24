import argparse
from collections.abc import Callable

from hitl_pmp.core.method.method import Method
from hitl_pmp.core.renderer.renderer import Renderer
from hitl_pmp.method_runner import MethodRunner

from .environment import BallRingEnvironment
from .problem import BallRingProblem
from .tasks import BallRingTasks


class BallRingCli:
    """Plugs Ball-Ring into the generic runner (see hitl_pmp/cli.py): exposes its
    configurable values as argparse flags, then ``run_method`` is this domain's
    composition root -- the one place that constructs the actual
    BallRingEnvironment/BallRingTasks/BallRingProblem instances from those flags
    before driving a method through PracticeLoop (via method_runner.py's
    MethodRunner). Mirrors ``LightSwitchCli`` exactly; a static-method container,
    never instantiated.

    No renderer is passed yet: PR 1 ships the env/tasks/problem/facade only, and a
    Ball-Ring ``renderer.py`` (like ``lightswitch/renderer.py``) is a later addition
    -- until then ``--output-dir`` writes ``stats.json`` but no ``episode.mp4``.
    Likewise, no Ball-Ring ``Method`` exists yet (the skill-oracle/random-skills/EES
    method CLIs are still hardcoded to Light Switch, a pre-existing ``TODO(scale)``),
    so ``run_method`` is driven by a caller-supplied ``method_factory`` -- the same
    shape ``LightSwitchCli.run_method`` uses.
    """

    @staticmethod
    def add_arguments(*, parser: argparse.ArgumentParser) -> None:
        """--env/--seed/--num-test-tasks/--method/--output-dir are global flags added
        by hitl_pmp/cli.py, not here -- everything below is specific to this domain."""
        parser.add_argument(
            "--num-tables",
            type=int,
            default=BallRingEnvironment.model_fields["num_tables"].default,
            help="Number of tables in the ring (must be >= 2).",
        )
        parser.add_argument(
            "--num-sticky-tables",
            type=int,
            default=BallRingEnvironment.model_fields["num_sticky_tables"].default,
            help="How many of the tables are sticky (the last such is the target).",
        )
        parser.add_argument(
            "--pick-success-prob",
            type=float,
            default=BallRingEnvironment.model_fields["pick_success_prob"].default,
            help="P(a pick succeeds). 1.0 in the deterministic paper config.",
        )
        parser.add_argument(
            "--place-sticky-fall-prob",
            type=float,
            default=BallRingEnvironment.model_fields["place_sticky_fall_prob"].default,
            help="P(a cup placed on a normal table / safe sticky region falls). 0.0 in paper.",
        )
        parser.add_argument(
            "--place-ball-fall-prob",
            type=float,
            default=BallRingEnvironment.model_fields["place_ball_fall_prob"].default,
            help="P(a bare ball placed on a table falls). 1.0 in paper.",
        )
        parser.add_argument(
            "--place-smooth-fall-prob",
            type=float,
            default=BallRingEnvironment.model_fields["place_smooth_fall_prob"].default,
            help="P(a cup placed on the smooth part of the sticky table falls). 1.0 in paper.",
        )
        parser.add_argument(
            "--noise-seed",
            type=int,
            default=BallRingEnvironment.model_fields["noise_seed"].default,
            help="Seed for the env's residual placement-noise RNG (fall landing points).",
        )
        parser.add_argument(
            "--test-env-seed-offset",
            type=int,
            default=BallRingTasks.model_fields["test_env_seed_offset"].default,
            help="Offset added to --seed to derive the test RNG stream.",
        )

    @staticmethod
    def run_method(
        *,
        args: argparse.Namespace,
        method_factory: Callable[[BallRingEnvironment], Method],
        num_cycles: int,
        max_steps_per_interaction: int,
    ) -> None:
        """This domain's composition root: builds the actual BallRingEnvironment/
        BallRingTasks/BallRingProblem instances from args, calls ``method_factory(env)``
        to build the Method, then delegates the domain-agnostic rest (PracticeLoop,
        printing, stats.json) to method_runner.py's MethodRunner. Mirrors
        ``LightSwitchCli.run_method``."""
        env = BallRingEnvironment(
            num_tables=args.num_tables,
            num_sticky_tables=args.num_sticky_tables,
            pick_success_prob=args.pick_success_prob,
            place_sticky_fall_prob=args.place_sticky_fall_prob,
            place_ball_fall_prob=args.place_ball_fall_prob,
            place_smooth_fall_prob=args.place_smooth_fall_prob,
            noise_seed=args.noise_seed,
        )
        tasks = BallRingTasks(
            env=env, seed=args.seed, test_env_seed_offset=args.test_env_seed_offset
        )
        problem = BallRingProblem(env=env, tasks=tasks)

        renderer: type[Renderer] | None = None  # no Ball-Ring renderer yet (see class docstring)
        MethodRunner.run(
            args=args,
            method=method_factory(env),
            problem=problem,
            num_cycles=num_cycles,
            max_steps_per_interaction=max_steps_per_interaction,
            renderer=renderer,
            render_fps=2,
            num_render_checkpoints=getattr(args, "num_render_checkpoints", 1),
        )
