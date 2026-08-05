"""Plugs Tossing3D into the global runner (`hitl_pmp/cli.py`) as `--env tossing3d`.

    python -m hitl_pmp.cli --env tossing3d --method skill-oracle \\
        --num-test-tasks 5 --output-dir /tmp/tossing3d

Run it under the KINDER venv, not `hitl-pmp` -- KINDER pulls MuJoCo, PyBullet and OpenCV
and caps `requires-python` at `<3.13`, so it lives in its own virtualenv (see CLAUDE.md).
"""

import argparse
from collections.abc import Callable
from typing import ClassVar

from hitl_pmp.core.method.method import Method
from hitl_pmp.core.method.skill_provider import DomainContext
from hitl_pmp.core.renderer.renderer import Renderer
from hitl_pmp.method_runner import MethodRunner

from .environment import Tossing3DEnvironment, Tossing3DTaskConfig
from .problem import Tossing3DProblem
from .renderer import Tossing3DRenderer
from .skill_oracle_policy import ORACLE_THROW_STANDOFF
from .skill_provider import Tossing3DOracle, Tossing3DSkillProvider
from .tasks import Tossing3DTasks


class Tossing3DCli:
    """This domain's argparse flags and composition root. A static-method container,
    never instantiated, same as every other business-logic class in this project."""

    # Only reached when no video is being written at all, so nothing plays at it: a run
    # without --output-dir gets no renderer, and MethodRunner then never opens a writer.
    # A recorded run reads KINDER's own metadata instead -- see resolve_render_fps.
    unrendered_render_fps: ClassVar[int] = 1

    @staticmethod
    def resolve_render_fps(*, env: Tossing3DEnvironment, renderer: type[Renderer] | None) -> int:
        """The clip's playback rate, taken from the simulator rather than chosen here.

        A recorded episode is now one frame per physics tick, so the right rate is the
        environment's own `metadata["render_fps"]` (20 on Tossing3D; several other KINDER
        scenes report 10, which is exactly why this is read and not written down). It
        used to be a hardcoded 1, which was correct while an episode was a four-frame
        storyboard and would now stretch a throw across two minutes.

        Reading metadata needs a live scene, hence the `hard_reset()`. That is not extra
        work: sampling a task in this domain is itself a simulator rebuild, so the run
        builds this scene either way, and it only happens on the recorded path.
        """
        if renderer is None:
            return Tossing3DCli.unrendered_render_fps
        env.hard_reset()
        return env.backend().render_fps()

    @staticmethod
    def add_arguments(*, parser: argparse.ArgumentParser) -> None:
        """--env/--seed/--num-test-tasks/--method/--output-dir are global flags added by
        hitl_pmp/cli.py, not here -- everything below is specific to this domain."""
        fields = Tossing3DEnvironment.model_fields
        parser.add_argument(
            "--task-config",
            choices=[config.value for config in Tossing3DTaskConfig],
            default=fields["task_config"].default.value,
            help="Which scene JSON to load. 'coincident' (the default) is upstream's o1 "
            "with the bin put back onto the goal region, so landing IN the bin is what "
            "scores. 'stock' is upstream's own o1, where a cube landing in the bin is a "
            "scored FAILURE. Never compare a number taken under one against the other.",
        )
        parser.add_argument(
            "--variant",
            default=fields["variant"].default,
            help="KINDER Tossing3D variant. Only o1 is supported: o2 needs two cubes in "
            "the goal region and this domain's symbolic layer is single-cube.",
        )
        parser.add_argument(
            "--no-scene-bg",
            dest="scene_bg",
            action="store_false",
            help="Render the bare 'simple' scene instead of the MimicLabs lab. Fast, "
            "needs no ~1 GB asset download, and looks wrong -- a smoke test only. "
            "Physics is unaffected.",
        )
        parser.add_argument(
            "--canonical-seed",
            type=int,
            default=fields["canonical_seed"].default,
            help="Scene seed used by Environment.hard_reset (not by task sampling). "
            "Upstream's own test seed, and the one every measured number in this "
            "domain's docs was taken at.",
        )
        task_fields = Tossing3DTasks.model_fields
        parser.add_argument(
            "--test-env-seed-offset",
            type=int,
            default=task_fields["test_env_seed_offset"].default,
            help="Offset added to --seed to derive the test scene-seed stream.",
        )
        parser.add_argument(
            "--oracle-throw-standoff",
            type=float,
            default=ORACLE_THROW_STANDOFF,
            help="How far from the bin the oracle stops before throwing, in metres. The "
            "default is upstream's own test value and solves the coincident scene; the "
            "stock scene needs a larger one (1.55, measured) because its bin sits 23 cm "
            "further out.",
        )
        parser.set_defaults(scene_bg=True)

    @staticmethod
    def run_method(
        *,
        args: argparse.Namespace,
        method_factory: Callable[[DomainContext], Method],
        num_cycles: int,
        max_steps_per_interaction: int,
    ) -> None:
        """This domain's composition root. Mirrors `TossingRoomCli.run_method`.

        The environment is closed in a `finally`: it owns a live MuJoCo context and,
        through each controller, PyBullet clients, and leaving those behind in a process
        that goes on to do something else is exactly how this domain's memory problems
        started.
        """
        env = Tossing3DEnvironment(
            task_config=Tossing3DTaskConfig(args.task_config),
            variant=args.variant,
            scene_bg=args.scene_bg,
            canonical_seed=args.canonical_seed,
        )
        tasks = Tossing3DTasks(
            env=env, seed=args.seed, test_env_seed_offset=args.test_env_seed_offset
        )
        problem = Tossing3DProblem(env=env, tasks=tasks)
        context = DomainContext(
            env=env,
            skill_provider=Tossing3DSkillProvider(env=env),
            oracle=Tossing3DOracle(env=env, throw_standoff=args.oracle_throw_standoff),
        )
        renderer: type[Renderer] | None = Tossing3DRenderer if args.output_dir is not None else None
        try:
            MethodRunner.run(
                args=args,
                method=method_factory(context),
                problem=problem,
                num_cycles=num_cycles,
                max_steps_per_interaction=max_steps_per_interaction,
                renderer=renderer,
                render_fps=Tossing3DCli.resolve_render_fps(env=env, renderer=renderer),
                num_render_checkpoints=getattr(args, "num_render_checkpoints", 1),
            )
        finally:
            env.close()
