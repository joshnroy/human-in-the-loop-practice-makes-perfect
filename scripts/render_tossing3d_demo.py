"""Render a smooth Tossing3D clip the way KINDER renders its own, and write a GIF with
KINDER's own optimizer.

Why this exists rather than `--output-dir`'s `episode.mp4`: a `hitl_pmp` transition is a
whole *skill* -- several hundred MuJoCo control ticks -- so `core.Renderer`, which is one
frame per transition by construction, can only ever produce a 4-frame storyboard. That is
the right thing for a learning-curve checkpoint and the wrong thing for a clip meant to
show a reviewer that the integration works. KINDER renders one frame per `env.step()`,
which is why the clips on the benchmark's site are smooth:

    # kindergarden/scripts/generate_demo_video.py
    env = kinder.make(env_id, render_mode="rgb_array", scene_bg=True)
    fps = env.metadata.get("render_fps", 10)
    frames = [env.render()]
    for action in actions:
        env.step(action)
        frames.append(env.render())
    iio.mimsave(output_path, frames, fps=fps, loop=loop)
    optimize_gif(output_path)

This follows that exactly, with `KinderBackend.capture_frames_into` standing in for the
per-step `env.render()` (the tick loop is inside a KINDER controller here, not in this
file), and `scene_bg=True` because their own docs generator passes it for every Dynamic3D
env -- it is where the textured floor comes from. Nothing is hand-rolled: the pixels are
`env.render()`, the frame rate is KINDER's own `render_fps` metadata, and the GIF is
written by `imageio` and squeezed by `kinder.gif_utils.optimize_gif` (gifsicle
`--optimize=3 --colors=256 --lossy=80`), not by a palette recipe invented here.

One deliberate departure from their script: every frame carries a caption strip, drawn
from `Tossing3DRenderer.caption` -- the same formatter the storyboard renderer uses --
reporting the active skill, the cube's live x, the scored goal region's x edges and the
bin's x. Without it this clip is a 3-D projection a reader has to judge by eye, and this
particular scene is the one where that goes wrong: KINDER's goal region stops short of
the bin, so the honest, *solved* ending has the cube resting on the floor with the bin
untouched. The numbers are what make that readable as success rather than as a near miss.
The per-tick cube position comes from `KinderBackend.capture_features_into`, so the
caption tracks the flight rather than freezing at the transition boundary.

`optimize_gif` is a `gifsicle` wrapper -- a size reducer, not a quality renderer -- and it
would be the wrong tool on the 4-frame storyboard clips, which are already ~55 KB and
would only get worse. It is the right tool here for one measured reason: a 171-frame
640x512 GIF is **13 MB** unoptimised, which is not a committable artifact. See
`draw_captions` and `write_gif` for why the frames are octree-quantised before it runs
and why its own lossy pass is then switched off: together they land at 1.2 MB, smaller
*and* closer to the raw render than the 1.9 MB this clip previously shipped at.

`scene_bg` is a *render-time* choice and stays off everywhere else: it costs 4.18 ms per
`render()` against 1.36 ms for the plain scene, and every number already measured on this
domain was measured without it. It is verified purely cosmetic -- `--check-scene-bg` runs
the same rollout both ways and asserts the cube trajectory is bit-identical.

Imports `hitl_pmp` directly, unlike `run_sweep.py` which only shells out to the CLI: it
needs an in-process handle on the live simulator to install the frame sink, which no
command line can express. `scripts/` is outside `lint-imports`' `hitl_pmp` root package,
so this cannot affect the layering contract either way.

Needs the `tossing3d` extra (see `src/hitl_pmp/environments/tossing3d/README.md`), so run
it from the KINDER virtualenv:

    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python -m scripts.render_tossing3d_demo \\
        --output docs/tossing3d_skill_oracle_demo.gif
"""

import argparse
from pathlib import Path
from typing import ClassVar

import numpy as np

from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.problem import Tossing3DProblem
from hitl_pmp.environments.tossing3d.renderer import Tossing3DRenderer
from hitl_pmp.environments.tossing3d.skill_oracle_policy import SkillOraclePolicy
from hitl_pmp.environments.tossing3d.tasks import Tossing3DTasks


class Tossing3DDemoRenderer:
    """A static-method container, never instantiated, same as every other business-logic
    class in this project."""

    # Height in pixels of the caption strip appended *below* KINDER's 640x480 frame, and
    # the text size in it. A strip rather than an overlay: nothing in the rendered scene
    # is occluded, which matters here because the two things a viewer is being asked to
    # compare -- the cube and the bin -- can both sit low in the frame. 640x512 keeps
    # both dimensions divisible by 16.
    caption_strip_height: ClassVar[int] = 32
    caption_font_size: ClassVar[int] = 15

    @staticmethod
    def add_arguments(*, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--output", type=Path, required=True, help="Where to write the GIF.")
        parser.add_argument(
            "--seed", type=int, default=0, help="Task-sampling seed (not the KINDER seed)."
        )
        parser.add_argument(
            "--no-scene-bg",
            action="store_true",
            help="Render KINDER's bare 'simple' scene instead of its MimicLabs room.",
        )
        parser.add_argument(
            "--tail-hold",
            type=int,
            default=20,
            help="How many frames of the final still to keep (20 = 1s at KINDER's fps).",
        )
        parser.add_argument(
            "--check-scene-bg",
            action="store_true",
            help="Also run the rollout with the plain scene and assert the cube "
            "trajectory is bit-identical, i.e. that scene_bg is purely cosmetic.",
        )

    @staticmethod
    def rollout(
        *, seed: int, scene_bg: bool
    ) -> tuple[list[np.ndarray], list[str], list[float], bool, int]:
        """One oracle episode. Returns its per-tick frames, the per-tick caption for
        each of them, the cube's x at each transition, whether the goal was satisfied,
        and KINDER's own render fps."""
        env = Tossing3DEnvironment(scene_bg=scene_bg)
        tasks = Tossing3DTasks(env=env, seed=seed)
        problem = Tossing3DProblem(env=env, tasks=tasks)
        task = problem.sample_test_task()

        frames: list[np.ndarray] = []
        # One entry per frame, filled by the backend's sinks in lockstep with `frames`,
        # plus the pre-rollout entry appended by hand below alongside its frame.
        features: list[dict[str, tuple[float, ...]]] = []
        labels: list[str | None] = []
        cube_xs: list[float] = []
        state = problem.reset_to_task(task=task)
        backend = env.backend()
        frames.append(backend.render())
        features.append(backend.read_features())
        labels.append(None)  # `caption` renders a None label as "start"
        cube_xs.append(state.get(obj=env.cube, feature_name="x"))

        backend.capture_frames_into(sink=frames)
        backend.capture_features_into(sink=features)
        for _ in range(problem.max_episode_steps()):
            if task.goal.is_satisfied(state=state):
                break
            labeled_action = SkillOraclePolicy.get_labeled_action(state=state, env=env)
            # The only per-tick skill identity available: a skill spans exactly the
            # frames its own `take_action` appended, so bracket the call and label them.
            before = len(frames)
            state = env.take_action(action=labeled_action.action)
            labels.extend([labeled_action.label] * (len(frames) - before))
            cube_xs.append(state.get(obj=env.cube, feature_name="x"))
        backend.capture_frames_into(sink=None)
        backend.capture_features_into(sink=None)
        assert len(frames) == len(features) == len(labels), "sinks fell out of lockstep"

        # The claim the clip is evidence for, read off the State rather than the pixels.
        print("goal atoms: " + ", ".join(str(atom) for atom in task.goal.atoms))
        cube = tuple(state.get(obj=env.cube, feature_name=a) for a in ("x", "y", "z"))
        print(f"final cube (x, y, z) = {cube}")
        print(f"goal region bounds = {env.goal_region_bounds()}")
        print(f"bin centre x = {state.get(obj=env.bin_object, feature_name='x'):.4f}")
        print(f"Goal.is_satisfied = {task.goal.is_satisfied(state=state)}")

        captions = Tossing3DDemoRenderer.captions(
            env=env,
            features=features,
            labels=labels,
            kinder_seed=int(round(state.get(obj=env.scene, feature_name="seed"))),
        )
        return (
            frames,
            captions,
            cube_xs,
            task.goal.is_satisfied(state=state),
            backend.render_fps(),
        )

    @staticmethod
    def captions(
        *,
        env: Tossing3DEnvironment,
        features: list[dict[str, tuple[float, ...]]],
        labels: list[str | None],
        kinder_seed: int,
    ) -> list[str]:
        """One caption per frame, through `Tossing3DRenderer.caption` -- the same
        formatter the storyboard uses, so the two cannot drift.

        The features come from `capture_features_into`, i.e. the state at the tick that
        produced the frame, not the state at the surrounding transition boundary. That
        distinction is the whole point during the toss: the cube's x moves ~1.3 m over
        the flight, and a boundary-sampled caption would sit frozen through all of it.
        """
        region = env.goal_region_bounds()
        return [
            Tossing3DRenderer.caption(
                state=env.build_state(features=per_tick, seed=kinder_seed, region=region),
                env=env,
                label=label,
            )
            for per_tick, label in zip(features, labels, strict=True)
        ]

    @staticmethod
    def trim_static_tail(*, frames: list[np.ndarray], hold: int) -> int:
        """How many leading frames to keep, dropping the tail after the scene stops
        changing but for `hold` of them. Returns a count rather than a slice because the
        caption list has to be cut at exactly the same index.

        `KinderBackend.SETTLE_STEPS` runs 150 zero-action ticks after a toss so the cube
        is on the ground before any goal test reads it. The cube is airborne for maybe a
        quarter of those; the rest render an identical still. Measured on the oracle
        rollout: 276 captured ticks, 150 of which differ from their predecessor -- so a
        third of the clip's running time is a freeze-frame. This drops only frames
        byte-identical to the last one, so nothing that moves is ever cut. It is computed
        on the raw KINDER frames, before any caption is drawn, so "the scene stopped
        changing" stays a statement about the scene.
        """
        end = len(frames)
        while end > 1 and np.array_equal(frames[end - 1], frames[end - 2]):
            end -= 1
        return min(len(frames), end + hold)

    @staticmethod
    def draw_captions(*, frames: list[np.ndarray], captions: list[str]) -> list[np.ndarray]:
        """Append a caption strip below each frame and write `captions[i]` into it.

        Deliberately not `Tossing3DRenderer.render_frame`: that one re-renders the *live*
        MuJoCo scene, so it cannot caption a frame that was captured hundreds of ticks
        ago, and it round-trips every frame through a matplotlib figure. Only the text is
        shared. The font is matplotlib's own bundled DejaVu Sans, located through
        `font_manager` -- matplotlib is already a hard dependency of this repo, so that is
        a font guaranteed present rather than a system font that may not be.

        **Each frame is octree-quantised here, and that is load-bearing, not tidying.**
        A GIF holds 256 colours, and the green cube is about 400 of a frame's 327,680
        pixels -- 0.06%. Median cut, which is what `imageio`'s GIF writer reaches for by
        default, allocates palette entries by pixel population, and at that share the
        cube's entry is marginal: measured, adding the caption strip is enough to tip it
        over, and the cube comes out grey-brown instead of green. Octree allocates by
        occupancy in colour space instead, and reproduces the cube's box mean to within
        a value of 1 -- closer than the uncaptioned clip managed. Quantising to 256
        colours first makes the writer's own pass a no-op, so the caption can no longer
        cost the scene its colours.
        """
        from matplotlib import font_manager  # noqa: PLC0415
        from PIL import Image, ImageDraw, ImageFont  # noqa: PLC0415

        font = ImageFont.truetype(
            font_manager.findfont("DejaVu Sans"), Tossing3DDemoRenderer.caption_font_size
        )
        strip = Tossing3DDemoRenderer.caption_strip_height
        captioned: list[np.ndarray] = []
        for frame, caption in zip(frames, captions, strict=True):
            height, width = frame.shape[0], frame.shape[1]
            canvas = Image.new("RGB", (width, height + strip), (0, 0, 0))
            canvas.paste(Image.fromarray(frame), (0, 0))
            ImageDraw.Draw(canvas).text(
                (10, height + strip // 2), caption, font=font, fill=(255, 255, 255), anchor="lm"
            )
            quantized = canvas.quantize(
                colors=256, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE
            )
            captioned.append(np.asarray(quantized.convert("RGB"), dtype=np.uint8))
        return captioned

    @staticmethod
    def write_gif(*, frames: list[np.ndarray], fps: int, output: Path) -> None:
        """KINDER's own write path, imported lazily so this module still imports
        without the optional dependency."""
        import imageio.v2 as iio  # noqa: PLC0415
        from kinder.gif_utils import optimize_gif  # noqa: PLC0415

        output.parent.mkdir(parents=True, exist_ok=True)
        iio.mimsave(output, frames, fps=fps, loop=0)
        # KINDER's own optimizer, at their own default colour count and with lossy
        # compression off. Lossy was needed when the frames reached this point with a
        # median-cut palette and a 13 MB intermediate; `draw_captions` now hands over
        # octree-quantised 256-colour frames, so the intermediate is 1.2 MB and gifsicle
        # only has to pack it. Measured on the same clip, over the cube's own pixels
        # against the raw render (mean absolute channel error, and file size):
        #
        #     no pre-quantisation, 64 colours, lossy 120   13.7   2.02 MB
        #     octree,              64 colours, lossy 120    8.9   0.79 MB
        #     octree,             256 colours, lossy   0    2.9   1.18 MB   <- this
        #
        # i.e. both better-looking and smaller than the settings this clip shipped with.
        optimize_gif(output, colors=256, lossy=0)

    @staticmethod
    def main(*, argv: list[str] | None = None) -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        Tossing3DDemoRenderer.add_arguments(parser=parser)
        args = parser.parse_args(argv)

        frames, captions, cube_xs, solved, fps = Tossing3DDemoRenderer.rollout(
            seed=args.seed, scene_bg=not args.no_scene_bg
        )
        distinct = 1 + sum(
            int(not np.array_equal(a, b)) for a, b in zip(frames, frames[1:], strict=False)
        )
        print(f"frames={len(frames)} distinct={distinct} fps={fps} solved={solved}")
        print("cube x per transition: " + ", ".join(f"{x:.4f}" for x in cube_xs))
        print(f"first caption: {captions[0]}")
        print(f"last caption:  {captions[-1]}")

        if args.check_scene_bg:
            _, _, plain_xs, plain_solved, _ = Tossing3DDemoRenderer.rollout(
                seed=args.seed, scene_bg=False
            )
            assert plain_xs == cube_xs, f"scene_bg changed the trajectory: {cube_xs} vs {plain_xs}"
            assert plain_solved == solved
            print("scene_bg is purely cosmetic: identical cube trajectory both ways")

        keep = Tossing3DDemoRenderer.trim_static_tail(frames=frames, hold=args.tail_hold)
        print(f"trimmed to {keep} frames ({keep / fps:.1f}s at {fps} fps)")
        Tossing3DDemoRenderer.write_gif(
            frames=Tossing3DDemoRenderer.draw_captions(
                frames=frames[:keep], captions=captions[:keep]
            ),
            fps=fps,
            output=args.output,
        )


if __name__ == "__main__":
    Tossing3DDemoRenderer.main()
