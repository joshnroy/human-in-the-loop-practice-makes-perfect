"""Render the SweepIntoHole3D scene so it can be judged by eye before any skill exists.

Four PNGs per scene from the task's own cameras: the `task_view` Dustpan3D inherits,
a closer `hole_view` framing the pit, its rim, the shovel and the cube strip, that
same view after two cubes have been dropped into the pit so containment is visible,
and a `top_view` with the whole layout and the pit opening. The `mimiclabs-lab2`
scene is the primary set (the one Dustpan3D runs were rendered in); the `simple`
scene is a secondary set with a `simple_` infix in the filename.

The drop is the environment's own physics (`set_pose` above the opening, then
settling), and the script refuses to write the third image unless `cube_in_hole`
agrees both cubes are in -- a picture of cubes resting on the rim would be a
misleading deliverable.

Every KINDER import lives inside `import_kinder`, so the module imports and its
argument parsing tests on a machine without MuJoCo; the render itself is gated on
`kinder` being installed, and the lab2 set on its assets being present.
"""

import argparse
import os
from pathlib import Path
from typing import Any

import imageio.v3 as iio
from pydantic import BaseModel, ConfigDict

from hitl_pmp.environments.tossing3d.kinder_backend import KinderBackend

ENV_ID = "kinder/SweepIntoHole3D-o5-v0"

# Scene name as the task JSON declares it -> filename infix. The lab scene is the
# primary deliverable, so it carries no infix.
SCENES: dict[str, str] = {"mimiclabs-lab2": "", "simple": "simple_"}


class View(BaseModel):
    model_config = ConfigDict(frozen=True)

    stem: str
    camera: str
    after_drop: bool

    def filename(self, *, scene: str) -> str:
        return f"sweep_into_hole3d_{SCENES[scene]}{self.stem}.png"


VIEWS: tuple[View, ...] = (
    View(stem="task_view", camera="task_view", after_drop=False),
    View(stem="hole_view", camera="hole_view", after_drop=False),
    View(stem="hole_view_two_cubes_in_pit", camera="hole_view", after_drop=True),
    View(stem="top_view", camera="top_view", after_drop=False),
)


class RenderPlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    output_dir: Path
    seed: int = 0
    scenes: tuple[str, ...] = tuple(SCENES)
    # 2000 steps at the model's 0.5 ms timestep is one second: a 12 cm drop has
    # stopped bouncing well within that (measured: the cube's height is unchanged
    # between 1500 and 4000 steps).
    settle_steps: int = 2000
    # Two drop points inside the opening, apart from each other and from the walls.
    pit_drops: dict[str, tuple[float, float, float]] = {
        "cube_0": (0.95, -0.08, 0.05),
        "cube_1": (1.12, 0.07, 0.05),
    }


class SweepIntoHole3DRenderer:
    @staticmethod
    def import_kinder() -> Any:
        """Import KINDER in the order its registration requires, keeping the GL backend.

        `register_all_environments()` rewrites `MUJOCO_GL`/`PYOPENGL_PLATFORM` when
        `DISPLAY` is unset, and the Dynamic3D envs register only if a dynamic3d
        *module* was imported first; `KinderBackend.configure_headless_rendering`
        re-asserts the requested backend on either side of the call.
        """
        os.environ.setdefault("DISABLE_AUTO_DYNAMIC3D_SCENES_DOWNLOAD", "1")
        KinderBackend.configure_headless_rendering()
        import kinder
        import kinder.envs.dynamic3d.envs  # noqa: F401  (the MODULE, not the package)

        kinder.register_all_environments()
        KinderBackend.configure_headless_rendering()
        return kinder

    @staticmethod
    def mimiclabs_assets_present() -> bool:
        """Whether the lab scenes' meshes have been downloaded into the installed KINDER."""
        kinder = SweepIntoHole3DRenderer.import_kinder()
        root = Path(kinder.__path__[0]) / "envs" / "dynamic3d" / "models" / "assets"
        return (root / "mimiclabs_scenes" / "meshes").is_dir()

    @staticmethod
    def render(*, plan: RenderPlan) -> dict[str, Path]:
        kinder = SweepIntoHole3DRenderer.import_kinder()
        plan.output_dir.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}
        for scene in plan.scenes:
            env = kinder.make(ENV_ID, render_mode="rgb_array", scene_bg=scene)
            try:
                env.reset(seed=plan.seed)
                # `.unwrapped` is the KinDER facade; the cameras, objects and goal
                # predicate live on the object-centric env one level further in.
                live = env.unwrapped._object_centric_env  # noqa: SLF001
                for view in VIEWS:
                    if not view.after_drop:
                        written[view.filename(scene=scene)] = SweepIntoHole3DRenderer._write(
                            live=live, view=view, scene=scene, output_dir=plan.output_dir
                        )
                SweepIntoHole3DRenderer._drop_into_pit(live=live, plan=plan)
                for view in VIEWS:
                    if view.after_drop:
                        written[view.filename(scene=scene)] = SweepIntoHole3DRenderer._write(
                            live=live, view=view, scene=scene, output_dir=plan.output_dir
                        )
            finally:
                env.close()
        return written

    @staticmethod
    def _drop_into_pit(*, live: Any, plan: RenderPlan) -> None:
        for name, position in plan.pit_drops.items():
            cube = live.get_object(name)
            cube.set_pose(list(position), [1.0, 0.0, 0.0, 0.0])
            cube.set_velocity([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        sim = live._robot_env.sim  # noqa: SLF001
        for _ in range(plan.settle_steps):
            sim.step()
        not_in = [name for name in plan.pit_drops if not live.cube_in_hole(name)]
        if not_in:
            raise RuntimeError(f"dropped cubes did not settle in the pit: {not_in}")

    @staticmethod
    def _write(*, live: Any, view: View, scene: str, output_dir: Path) -> Path:
        live.set_render_camera(view.camera)
        path = output_dir / view.filename(scene=scene)
        iio.imwrite(path, live.render())
        return path

    @staticmethod
    def parse_args(*, argv: list[str] | None = None) -> RenderPlan:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--output-dir", type=Path, required=True)
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--scenes", nargs="+", choices=sorted(SCENES), default=list(SCENES))
        args = parser.parse_args(argv)
        return RenderPlan(output_dir=args.output_dir, seed=args.seed, scenes=tuple(args.scenes))

    @staticmethod
    def main() -> None:
        plan = SweepIntoHole3DRenderer.parse_args()
        for filename, path in SweepIntoHole3DRenderer.render(plan=plan).items():
            print(f"{filename}: {path}")


if __name__ == "__main__":
    SweepIntoHole3DRenderer.main()
