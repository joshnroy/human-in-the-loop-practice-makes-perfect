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

`optimize_gif` is a `gifsicle --lossy` wrapper -- a size reducer, not a quality renderer
-- and it would be the wrong tool on the 4-frame storyboard clips, which are already
~55 KB and would only get worse. It is the right tool here for one measured reason: a
171-frame 640x480 GIF is **11.6 MB** unoptimised, which is not a committable artifact.
Their defaults take it to 3.5 MB and 64 colours / lossy 120 takes it to 1.9 MB, the same
size as their own `docs/envs/assets/group_gifs/Tossing3D.gif`.

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

import numpy as np

from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.problem import Tossing3DProblem
from hitl_pmp.environments.tossing3d.skill_oracle_policy import SkillOraclePolicy
from hitl_pmp.environments.tossing3d.tasks import Tossing3DTasks


class Tossing3DDemoRenderer:
    """A static-method container, never instantiated, same as every other business-logic
    class in this project."""

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
    def rollout(*, seed: int, scene_bg: bool) -> tuple[list[np.ndarray], list[float], bool, int]:
        """One oracle episode. Returns its per-tick frames, the cube's x at each
        transition, whether the goal was satisfied, and KINDER's own render fps."""
        env = Tossing3DEnvironment(scene_bg=scene_bg)
        tasks = Tossing3DTasks(env=env, seed=seed)
        problem = Tossing3DProblem(env=env, tasks=tasks)
        task = problem.sample_test_task()

        frames: list[np.ndarray] = []
        cube_xs: list[float] = []
        state = problem.reset_to_task(task=task)
        backend = env.backend()
        frames.append(backend.render())
        cube_xs.append(state.get(obj=env.cube, feature_name="x"))

        backend.capture_frames_into(sink=frames)
        for _ in range(problem.max_episode_steps()):
            if task.goal.is_satisfied(state=state):
                break
            labeled_action = SkillOraclePolicy.get_labeled_action(state=state, env=env)
            state = env.take_action(action=labeled_action.action)
            cube_xs.append(state.get(obj=env.cube, feature_name="x"))
        backend.capture_frames_into(sink=None)

        # The claim the clip is evidence for, read off the State rather than the pixels.
        print("goal atoms: " + ", ".join(str(atom) for atom in task.goal.atoms))
        cube = tuple(state.get(obj=env.cube, feature_name=a) for a in ("x", "y", "z"))
        print(f"final cube (x, y, z) = {cube}")
        # NOTE: these are the raw task-JSON bounds; KINDER scores against a version
        # inflated by 0.05 m per axis -- see KinderBackend.goal_region_bounds.
        print(f"goal region bounds (raw JSON) = {env.goal_region_bounds()}")
        print(f"bin centre x = {state.get(obj=env.bin_object, feature_name='x'):.4f}")
        print(f"Goal.is_satisfied = {task.goal.is_satisfied(state=state)}")

        return frames, cube_xs, task.goal.is_satisfied(state=state), backend.render_fps()

    @staticmethod
    def trim_static_tail(*, frames: list[np.ndarray], hold: int) -> list[np.ndarray]:
        """Drop the frames after the scene stops changing, keeping `hold` of them.

        `KinderBackend.SETTLE_STEPS` runs 150 zero-action ticks after a toss so the cube
        is on the ground before any goal test reads it. The cube is airborne for maybe a
        quarter of those; the rest render an identical still. Measured on the oracle
        rollout: 276 captured ticks, 150 of which differ from their predecessor -- so a
        third of the clip's running time is a freeze-frame. This drops only frames
        byte-identical to the last one, so nothing that moves is ever cut.
        """
        end = len(frames)
        while end > 1 and np.array_equal(frames[end - 1], frames[end - 2]):
            end -= 1
        return frames[: min(len(frames), end + hold)]

    @staticmethod
    def write_gif(*, frames: list[np.ndarray], fps: int, output: Path) -> None:
        """KINDER's own write path, imported lazily so this module still imports
        without the optional dependency."""
        import imageio.v2 as iio  # noqa: PLC0415
        from kinder.gif_utils import optimize_gif  # noqa: PLC0415

        output.parent.mkdir(parents=True, exist_ok=True)
        iio.mimsave(output, frames, fps=fps, loop=0)
        # KINDER's own optimizer, through its own parameters. Their defaults (256
        # colours, lossy 80) leave this at 3.5 MB because it is ~10x the length of the
        # clips those defaults were chosen for; 64/120 brings it to the same ~1.9 MB as
        # their own `docs/envs/assets/group_gifs/Tossing3D.gif` with no visible loss on
        # a scene whose palette is floor, robot and one green cube.
        optimize_gif(output, colors=64, lossy=120)

    @staticmethod
    def main(*, argv: list[str] | None = None) -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        Tossing3DDemoRenderer.add_arguments(parser=parser)
        args = parser.parse_args(argv)

        frames, cube_xs, solved, fps = Tossing3DDemoRenderer.rollout(
            seed=args.seed, scene_bg=not args.no_scene_bg
        )
        distinct = 1 + sum(
            int(not np.array_equal(a, b)) for a, b in zip(frames, frames[1:], strict=False)
        )
        print(f"frames={len(frames)} distinct={distinct} fps={fps} solved={solved}")
        print("cube x per transition: " + ", ".join(f"{x:.4f}" for x in cube_xs))

        if args.check_scene_bg:
            _, plain_xs, plain_solved, _ = Tossing3DDemoRenderer.rollout(
                seed=args.seed, scene_bg=False
            )
            assert plain_xs == cube_xs, f"scene_bg changed the trajectory: {cube_xs} vs {plain_xs}"
            assert plain_solved == solved
            print("scene_bg is purely cosmetic: identical cube trajectory both ways")

        trimmed = Tossing3DDemoRenderer.trim_static_tail(frames=frames, hold=args.tail_hold)
        print(f"trimmed to {len(trimmed)} frames ({len(trimmed) / fps:.1f}s at {fps} fps)")
        Tossing3DDemoRenderer.write_gif(frames=trimmed, fps=fps, output=args.output)


if __name__ == "__main__":
    Tossing3DDemoRenderer.main()
