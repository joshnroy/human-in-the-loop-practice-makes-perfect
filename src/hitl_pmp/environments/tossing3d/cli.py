"""Plugs Tossing3D into the global runner (`hitl_pmp/cli.py`) as `--env tossing3d`.

    python -m hitl_pmp.cli --env tossing3d --method skill-oracle \\
        --num-test-tasks 5 --output-dir /tmp/tossing3d

KINDER installs into `hitl-pmp` itself, as the `tossing3d` extra; see CLAUDE.md.
"""

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

from hitl_pmp.core.method.method import Method
from hitl_pmp.core.method.skill_provider import DomainContext
from hitl_pmp.core.renderer.renderer import Renderer
from hitl_pmp.humans.oracle import UnconditionalHumanOracle
from hitl_pmp.method_runner import MethodRunner

from .environment import Tossing3DEnvironment
from .layout import Tossing3DLayout
from .problem import Tossing3DProblem
from .renderer import Tossing3DRenderer
from .skill_oracle_policy import ORACLE_THROW_STANDOFF
from .skill_provider import Tossing3DOracle, Tossing3DSkillProvider
from .state_log import StateLogHeader, StateLogWriter
from .tasks import Tossing3DTasks

# The state log's own filename under --output-dir, named after its content like
# stats.json/config_snapshot.json/episode_traces.jsonl are -- see state_log.py.
STATE_LOG_FILENAME = "tossing3d_state_log.jsonl"


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
        parser.add_argument(
            "--layout",
            choices=[layout.value for layout in Tossing3DLayout],
            default=Tossing3DLayout.BARRIER.value,
            help="Scene layout: original barrier benchmark or same-side retrieval scene.",
        )
        fields = Tossing3DEnvironment.model_fields
        parser.add_argument(
            "--evaluation-layout",
            choices=[layout.value for layout in Tossing3DLayout],
            default=None,
            help="Evaluation scene layout; omitted means the practice layout. "
            "Use barrier to test same-side practice on the original far-side benchmark.",
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
            "default is upstream's own test value and solves the shipped scene. Which "
            "standoffs solve is a property of the scene's geometry, not a constant of "
            "this domain, so it stays overridable.",
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
        """This domain's composition root. Mirrors
        `TossingRoomCli.run_method`.

        Both environments are closed in a `finally`: each owns a live MuJoCo context and,
        through each controller, PyBullet clients, and leaving those behind in a process
        that goes on to do something else is exactly how this domain's memory problems
        started. `close()` is guarded on the backend existing and is idempotent, so
        closing an evaluation environment that never woke up costs nothing.

        **It builds that triple twice**, one for practice and one wholly separate for
        evaluation, matching `TossingRoomCli.run_method`. Every evaluation episode opens
        with `reset_to_task`, a privileged state-write, so sharing the triple hands the
        practice environment `--num-test-tasks` free resets per sweep. Under the default
        per-period reset that is invisible; for `--practice-reset-policy never` it is
        fatal, because the arm would be reset every sweep while `num_practice_resets`
        still reported 0. See PracticeLoop's "separate evaluation environment" section.

        **Why this domain can be split, when `ballring` cannot.** The split is only
        sound if the second instance cannot shift what practice draws.
        `Tossing3DEnvironment` holds no RNG field at all -- the only randomness in the
        domain lives in `Tossing3DTasks`' train/test streams, derived independently from
        the same seed -- and the simulator is re-seeded on every `reset` from the scene
        seed, so no history carries across. `ballring`'s `_noise_rng` is consumed by
        evaluation and therefore does shift the practice stream, which is why it stays
        excluded and would need a re-baseline rather than a wiring change.

        The cost this domain pays that Tossing Room does not is a **second live MuJoCo
        scene** for the length of the run. That is real and is the reason this was
        deferred; it is not free, and a sweep's memory cap has to be sized for it.
        """
        practice_problem = Tossing3DCli.build_problem(args=args)
        # The same seed stream, independent objects, and an optional explicit
        # geometry override. Changing layout must not resample the test tasks.
        evaluation_problem = Tossing3DCli.build_evaluation_problem(args=args)
        # Matching layouts retain the shared chronological log. Different layouts
        # need separate headers/files so every recorded state can be replayed in
        # its own geometry. State capture is gated on output-dir as before.
        if args.output_dir is not None:
            state_log_writer = StateLogWriter(
                output_path=Path(args.output_dir) / STATE_LOG_FILENAME,
                header=StateLogHeader(
                    layout=getattr(args, "layout", Tossing3DLayout.BARRIER),
                    variant=args.variant,
                    scene_bg=args.scene_bg,
                    canonical_seed=args.canonical_seed,
                    seed=args.seed,
                    test_env_seed_offset=args.test_env_seed_offset,
                ),
            )
            practice_problem.env.attach_state_log_writer(writer=state_log_writer)
            if evaluation_problem.env.layout == practice_problem.env.layout:
                evaluation_state_log_writer = state_log_writer
            else:
                # A replay header describes one scene. Never mix far-side ticks
                # into a same-side log, which would silently replay wrong geometry.
                evaluation_state_log_writer = StateLogWriter(
                    output_path=Path(args.output_dir) / "tossing3d_evaluation_state_log.jsonl",
                    header=state_log_writer.header.model_copy(
                        update={"layout": evaluation_problem.env.layout}
                    ),
                )
            evaluation_problem.env.attach_state_log_writer(writer=evaluation_state_log_writer)
        else:
            state_log_writer = None
            evaluation_state_log_writer = None
        # The Method is wired to the *practice* environment deliberately: its env
        # reference is structural config (skills, predicates, object handles, all of
        # which are ClassVars here). Evaluation may change geometry, but uses the
        # same learned samplers and observed state features, without refitting on tests.
        context = DomainContext(
            env=practice_problem.env,
            skill_provider=Tossing3DSkillProvider(env=practice_problem.env),
            oracle=Tossing3DOracle(
                env=practice_problem.env, throw_standoff=args.oracle_throw_standoff
            ),
        )
        renderer: type[Renderer] | None = Tossing3DRenderer if args.output_dir is not None else None
        try:
            MethodRunner.run(
                args=args,
                method=method_factory(context),
                problem=practice_problem,
                evaluation_problem=evaluation_problem,
                num_cycles=num_cycles,
                max_steps_per_interaction=max_steps_per_interaction,
                renderer=renderer,
                # The EVALUATION environment, not the practice one. resolve_render_fps
                # hard_resets whatever it is handed in order to read live metadata, and
                # rendering happens on the evaluation problem anyway -- so pointing it
                # at practice would deal the reset-free arm a privileged state-write
                # from the CLI before the run even starts.
                render_fps=Tossing3DCli.resolve_render_fps(
                    env=evaluation_problem.env, renderer=renderer
                ),
                num_render_checkpoints=getattr(args, "num_render_checkpoints", 1),
            )
        finally:
            practice_problem.env.close()
            evaluation_problem.env.close()
            if state_log_writer is not None:
                state_log_writer.close()
            if evaluation_state_log_writer is not None:
                evaluation_state_log_writer.close()

    @staticmethod
    def build_evaluation_problem(*, args: argparse.Namespace) -> Tossing3DProblem:
        """Keep test-task seeds fixed while optionally changing only scene layout."""
        evaluation_args = argparse.Namespace(**vars(args))
        evaluation_args.layout = getattr(args, "evaluation_layout", None) or getattr(
            args, "layout", Tossing3DLayout.BARRIER
        )
        return Tossing3DCli.build_problem(args=evaluation_args)

    @staticmethod
    def build_problem(*, args: argparse.Namespace) -> Tossing3DProblem:
        """One fully-independent Environment + Tasks + Problem triple from args.

        Called twice by `run_method` (practice and evaluation) -- factored out so the
        two cannot drift apart in configuration, which would silently make the
        evaluation set a different set of tasks than the one being trained against.

        Constructing this builds **no simulator**: `Tossing3DEnvironment.backend()` is
        lazy, so the MuJoCo scene appears on first reset/step rather than here. That is
        what lets the tests above run on CI without the optional KINDER extra.
        """
        env = Tossing3DEnvironment(
            layout=getattr(args, "layout", Tossing3DLayout.BARRIER),
            variant=args.variant,
            scene_bg=args.scene_bg,
            canonical_seed=args.canonical_seed,
        )
        tasks = Tossing3DTasks(
            env=env, seed=args.seed, test_env_seed_offset=args.test_env_seed_offset
        )
        # Wired for the same reason tossingroom's own build_problem wires it:
        # --method ees's ask_for_reset_cube_bin_only ground skill needs a real
        # HumanOracle on the practice Problem, or PracticeLoop.run refuses up front the
        # moment its cost flag is configured. This function builds both the practice
        # and evaluation Problem (called twice, see above), so both get one -- harmless
        # on the evaluation side, since no evaluation policy can ever raise a
        # human-help exception in the first place, and harmless for every other
        # Method/config, since a HumanOracle that is never asked costs nothing and
        # changes no existing run's behaviour.
        return Tossing3DProblem(env=env, tasks=tasks, human=UnconditionalHumanOracle)
