import argparse
from collections.abc import Callable
from typing import ClassVar

from hitl_pmp.core.method.method import Method
from hitl_pmp.core.method.skill_provider import DomainContext
from hitl_pmp.core.renderer.renderer import Renderer
from hitl_pmp.method_runner import MethodRunner

from .environment import Tossing3DEnvironment
from .problem import Tossing3DProblem
from .renderer import Tossing3DRenderer
from .skill_provider import Tossing3DOracle, Tossing3DSkillProvider
from .tasks import Tossing3DTasks


class Tossing3DCli:
    """Plugs KINDER's Tossing3D into the generic runner (see hitl_pmp/cli.py): exposes
    its configurable values as argparse flags, then run_method is this domain's
    composition root. Mirrors TossingRoomCli.

    Note what is NOT a flag: `throw_standoff` and `throw_pose_tolerance` are
    `Tossing3DEnvironment` ClassVars, because a module-level `Predicate` reads them and
    cannot see per-instance config -- the same rule Light Switch follows.
    """

    render_fps: ClassVar[int] = 1  # a storyboard, one frame per skill -- see the renderer

    @staticmethod
    def add_arguments(*, parser: argparse.ArgumentParser) -> None:
        """--env/--seed/--num-test-tasks/--method/--output-dir are global flags added
        by hitl_pmp/cli.py, not here -- everything below is specific to this domain."""
        fields = Tossing3DEnvironment.model_fields
        parser.add_argument(
            "--variant",
            choices=["o1", "o2"],
            default=fields["variant"].default,
            help="Which KINDER Tossing3D variant. o1 has one cube; o2 has two and is "
            "NOT supported by this port's single-cube symbolic layer.",
        )
        parser.add_argument(
            "--swing-low",
            type=float,
            default=fields["swing_low"].default,
            help="Lower bound of the uniform prior over Toss's swing dial.",
        )
        parser.add_argument(
            "--swing-high",
            type=float,
            default=fields["swing_high"].default,
            help="Upper bound of the uniform prior over Toss's swing dial. 1.0 is "
            "KINDER's own demo toss, which overshoots the goal region into the bin.",
        )
        parser.add_argument(
            "--pick-distance-low",
            type=float,
            default=fields["pick_distance_low"].default,
            help="Lower bound of the uniform prior over Pick's base standoff distance.",
        )
        parser.add_argument(
            "--pick-distance-high",
            type=float,
            default=fields["pick_distance_high"].default,
            help="Upper bound of the uniform prior over Pick's base standoff distance.",
        )
        parser.add_argument(
            "--canonical-seed",
            type=int,
            default=fields["canonical_seed"].default,
            help="KINDER episode seed used by Environment.hard_reset (not task sampling).",
        )
        task_fields = Tossing3DTasks.model_fields
        parser.add_argument(
            "--test-env-seed-offset",
            type=int,
            default=task_fields["test_env_seed_offset"].default,
            help="Offset added to --seed to derive the test RNG stream.",
        )

    @staticmethod
    def run_method(
        *,
        args: argparse.Namespace,
        method_factory: Callable[[DomainContext], Method],
        num_cycles: int,
        max_steps_per_interaction: int,
    ) -> None:
        """This domain's composition root -- builds Tossing3DEnvironment/
        Tossing3DTasks/Tossing3DProblem from args, bundles this domain's
        SkillProvider/OraclePolicyProvider into a DomainContext, then delegates the
        domain-agnostic rest to method_runner.py's MethodRunner."""
        env = Tossing3DEnvironment(
            variant=args.variant,
            swing_low=args.swing_low,
            swing_high=args.swing_high,
            pick_distance_low=args.pick_distance_low,
            pick_distance_high=args.pick_distance_high,
            canonical_seed=args.canonical_seed,
        )
        tasks = Tossing3DTasks(
            env=env, seed=args.seed, test_env_seed_offset=args.test_env_seed_offset
        )
        problem = Tossing3DProblem(env=env, tasks=tasks)
        context = DomainContext(
            env=env,
            skill_provider=Tossing3DSkillProvider(env=env),
            oracle=Tossing3DOracle(env=env),
        )

        renderer: type[Renderer] | None = Tossing3DRenderer if args.output_dir is not None else None
        MethodRunner.run(
            args=args,
            method=method_factory(context),
            problem=problem,
            num_cycles=num_cycles,
            max_steps_per_interaction=max_steps_per_interaction,
            renderer=renderer,
            render_fps=Tossing3DCli.render_fps,
            num_render_checkpoints=getattr(args, "num_render_checkpoints", 1),
        )
