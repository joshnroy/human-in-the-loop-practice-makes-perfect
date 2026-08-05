import argparse
from collections.abc import Callable
from typing import ClassVar

from hitl_pmp.core.method.method import Method
from hitl_pmp.core.method.skill_provider import DomainContext
from hitl_pmp.core.renderer.renderer import Renderer
from hitl_pmp.method_runner import MethodRunner

from .environment import LightSwitchEnvironment
from .problem import LightSwitchProblem
from .renderer import LightSwitchRenderer
from .skill_provider import LightSwitchOracle, LightSwitchSkillProvider
from .tasks import LightSwitchTasks


class LightSwitchCli:
    """Plugs Light Switch into the generic runner (see hitl_pmp/cli.py): exposes
    its configurable values as argparse flags, then run_method (below) is this
    domain's composition root -- the one place that actually constructs
    LightSwitchEnvironment/LightSwitchTasks/LightSwitchProblem instances from
    those flags, before driving a method through PracticeLoop (via
    method_runner.py's MethodRunner). A static-method container, never
    instantiated, same as every other business-logic class in this project."""

    render_fps: ClassVar[int] = 2  # slow -- episodes are only a few actions long

    @staticmethod
    def add_arguments(*, parser: argparse.ArgumentParser) -> None:
        """--env/--seed/--num-test-tasks/--method/--output-dir are global flags
        added by hitl_pmp/cli.py, not here -- everything below is specific to
        this domain."""
        parser.add_argument(
            "--grid-size",
            type=int,
            default=LightSwitchEnvironment.model_fields["grid_size"].default,
            help="Number of cells in the Light Switch grid.",
        )
        parser.add_argument(
            "--light-on-tolerance",
            type=float,
            default=LightSwitchEnvironment.light_on_tolerance,
            help="Max |level - target| for the LightOn predicate to hold.",
        )
        parser.add_argument(
            "--same-position-tolerance",
            type=float,
            default=LightSwitchEnvironment.same_position_tolerance,
            help='Max |robot.x - light.x| to count as "at the light".',
        )
        parser.add_argument(
            "--canonical-light-target",
            type=float,
            default=LightSwitchEnvironment.model_fields["canonical_light_target"].default,
            help="light_target used by Environment.hard_reset (not task sampling).",
        )
        parser.add_argument(
            "--test-env-seed-offset",
            type=int,
            default=LightSwitchTasks.model_fields["test_env_seed_offset"].default,
            help="Offset added to --seed to derive the test RNG stream.",
        )
        parser.add_argument(
            "--target-low",
            type=float,
            default=LightSwitchTasks.model_fields["target_low"].default,
            help="Lower bound of the light's sampled target (Uniform[target_low, target_high)).",
        )
        parser.add_argument(
            "--target-high",
            type=float,
            default=LightSwitchTasks.model_fields["target_high"].default,
            help="Upper bound of the light's sampled target.",
        )

    @staticmethod
    def apply_config(*, args: argparse.Namespace) -> None:
        """light_on_tolerance/same_position_tolerance are the only two values that
        stay ClassVar on LightSwitchEnvironment rather than becoming constructor
        arguments (see that class's own docstring for why: predicates.py's
        module-level Predicate objects are singletons built once at import time,
        and their `holds` closures read these two via a late-bound class lookup --
        there's no per-instance slot for them to read instead without a much wider
        change to Predicate.holds' signature). Everything else this domain exposes
        as a CLI flag flows through constructor arguments in run_method below, not
        through class-level mutation."""
        LightSwitchEnvironment.light_on_tolerance = args.light_on_tolerance
        LightSwitchEnvironment.same_position_tolerance = args.same_position_tolerance

    @staticmethod
    def run_method(
        *,
        args: argparse.Namespace,
        method_factory: Callable[[DomainContext], Method],
        num_cycles: int,
        max_steps_per_interaction: int,
    ) -> None:
        """Shared by every method-CLI driving this domain: this domain's composition
        root -- builds the actual LightSwitchEnvironment/LightSwitchTasks/
        LightSwitchProblem instances plus this domain's SkillProvider/
        OraclePolicyProvider (bundled into a DomainContext), calls
        `method_factory(context)` to build the Method instance, then delegates the
        domain-agnostic rest (PracticeLoop, printing, video-writing) to MethodRunner.
        method_factory takes the whole DomainContext (not just env) so a method-CLI
        picks exactly what its Method needs from it -- e.g. EesCli/RandomSkillsCli
        pass `lambda ctx: RandomSkillsMethod(skill_provider=ctx.skill_provider,
        env=ctx.env, seed=args.seed)`, SkillOracleCli passes
        `lambda ctx: SkillOracleMethod(env=ctx.env, oracle=ctx.oracle)` -- and the
        method side never imports this environment. num_cycles/
        max_steps_per_interaction come from the *caller* (an oracle passes 0/0),
        since that's a property of which method is being driven, not of Light
        Switch."""
        LightSwitchCli.apply_config(args=args)
        env = LightSwitchEnvironment(
            grid_size=args.grid_size, canonical_light_target=args.canonical_light_target
        )
        tasks = LightSwitchTasks(
            env=env,
            seed=args.seed,
            test_env_seed_offset=args.test_env_seed_offset,
            target_low=args.target_low,
            target_high=args.target_high,
        )
        problem = LightSwitchProblem(env=env, tasks=tasks)
        context = DomainContext(
            env=env,
            skill_provider=LightSwitchSkillProvider(env=env),
            oracle=LightSwitchOracle(env=env),
        )

        renderer: type[Renderer] | None = (
            LightSwitchRenderer if MethodRunner.rendering_needed(args=args) else None
        )
        MethodRunner.run(
            args=args,
            method=method_factory(context),
            problem=problem,
            num_cycles=num_cycles,
            max_steps_per_interaction=max_steps_per_interaction,
            renderer=renderer,
            render_fps=LightSwitchCli.render_fps,
            num_render_checkpoints=getattr(args, "num_render_checkpoints", 1),
        )
