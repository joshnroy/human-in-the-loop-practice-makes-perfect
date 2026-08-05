"""Render KINDER's Tossing3D oracle rollout to a GIF, once per toss standoff.

The point of the two default standoffs is the **contrast between them**, which is why
the default run renders both:

| standoff | cube comes to rest | `_check_goals()` |
| --- | --- | --- |
| 1.35 | *in the bin* (z sits above its start height) | `False` |
| 1.55 | on bare floor, short of the bin | `True` |

Same skill sequence, same seed, same parameters -- only the `move_to_target` standoff
differs. The throw that lands in the bin scores nothing; the one that misses the bin
scores. That is not a bug in this script: KINDER's goal predicate for `Tossing3D-o1`
is `["on", "cube_0", "blocks_goal_region"]`, a ground region that the bin merely sits
near. Each clip is captioned with its own measured rest position and
`_check_goals()` value so a reader who opens only one of them still sees the point.

Nothing here is a reimplementation. The rollout drives upstream's own controllers
(`pick_shelf` from `kinder_models.dynamic3d.shelf`; `move_to_target`,
`move_arm_to_conf` and `toss` from `kinder_models.dynamic3d.tossing`) with upstream's
own parameters, copied from
`kinder-models/tests/dynamic3d/tossing/test_tossing_parameterized_skills.py::test_pick_ground_toss`
so this depends on no branch of ours. It lives in `scripts/` rather than `analysis/`
because it *drives* a simulator, which `analysis/` may never do (CLAUDE.md).

Usage (KINDER is an optional extra; it is not in the `hitl-pmp` conda env, so this
runs under the KINDER venv, which needs `pydantic` and wants `gifsicle` on PATH):

    /path/to/kinder-venv/bin/python scripts/tossing3d_oracle_demo.py \\
        --output-dir docs

Run it under a memory cap. KINDER leaks roughly one PyBullet client and ~150 MB per
skill execution, and a kernel OOM on this box takes the whole login session with it:

    systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 -p OOMPolicy=continue \\
        /path/to/kinder-venv/bin/python scripts/tossing3d_oracle_demo.py

## Rendering settings, and the one that was got wrong twice

`--scene-bg` (default on) is what makes this the MimicLabs `lab2` scene the task JSON
names, and is what the ~1 GB asset download is for; `--no-scene-bg` maps to a scene
literally called `simple`, a bare ground plane, and is only useful as a fast smoke
test. Upstream's demo path (`reference/kindergarden/scripts/generate_demo_video.py`)
also offers `realistic_bg=True` for non-Dynamic3D 3D envs; it has never been tried
here and is deliberately not wired up.

`--camera` defaults to **`task_view`**, the camera Tossing3D's own task config
defines. It is deliberately *not* `agentview_1`, despite that being what upstream's
demo script sets, because upstream sets it only `if "TidyBot" in env_id` -- and this
env id is `kinder/Tossing3D-o1-v0`, which does not contain `TidyBot`. `agentview_1`
is not in this scene's `camera_names` at all (`frontview`, `birdview`, `agentview`,
`sideview`, `task_view`, `robot_base`, `robot_wrist`), `set_render_camera` does not
validate the name, and the resulting clip is a near-static shot of a wall: 6/32
sampled frames were unique under `agentview`, against 32/32 under `task_view`. The
whole point of the clip is to see the dynamics, so it uses the camera that shows
them.

fps is read from `env.metadata["render_fps"]` (20 for this env), never hardcoded --
several other KINDER envs report 10, and a hardcoded value would silently render
those at the wrong speed.

## The goal region is shaded in, using KINDER's own mechanism

The default clips draw `blocks_goal_region` as a translucent blue box, because
"the cube lands in the bin and this scores a **failure**" is only legible once you
can see where the goal actually is. `--no-goal-region` renders clean.

This is not an overlay drawn on top of the image. KINDER creates a box **site** for
every region at construction and calls `visualize_regions()` unconditionally
(`envs.py:455` for ground regions); those sites are invisible only because their
alpha is `0` -- the `[1, 0, 0, 0]` default in `objects/base.py:370`/`:900`, or, for
this scene's goal region, a `[0.2, 0.6, 0.2, 0.0]` set in the task JSON. So the
first-class configuration route is an `"rgba"` key on the region's JSON entry, and
that file is under `reference/`, a third-party checkout this repo never modifies.
The same value is therefore written into the compiled `MjModel` after `env.reset()`
instead. A site is a massless, collision-free marker, so this is a colour and
nothing else: the rest positions and `_check_goals()` values are unchanged with the
overlay on, which is checked by re-running both standoffs.

The site is found by reading the model's site names back and matching, never by a
hardcoded string: the name is assembled as `{fixture}_{region}_region_{index}` deep
inside upstream's XML builder. On this scene it resolves to
`ground_blocks_goal_region_region_0`, whose x-extent is `[1.8500, 2.1500]` -- and
the caption reports that measured bracket, so the shaded box and the printed numbers
cannot drift apart.
"""

import argparse
import functools
import os
import sys
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, ConfigDict, Field

# Upstream's own test parameters -- see the module docstring.
DEFAULT_ENV_ID = "kinder/Tossing3D-o1-v0"
DEFAULT_STANDOFFS = (1.35, 1.55)
DEFAULT_SEED = 125
DEFAULT_CAMERA = "task_view"
DEFAULT_OUTPUT_DIR = Path("docs")
PICK_PARAM_SEED = 123
WINDUP_CONF_DEG = (0, 50, 180, -110, 0, -100, 90)
FULL_TOSS_CONF_DEG = (0, 20, 180, -35, 0, 25, 90)
FILENAME_PREFIX = "tossing3d_oracle_standoff"

# These three exist because the clips are **committed**, so their size is a review
# concern rather than a preference. The rollout is 128 simulator steps of a
# wood-textured lab floor, so essentially every pixel changes every frame and GIF's
# inter-frame compression buys nothing: at every=1 with upstream's own 256/80 the
# optimised clip is 3.9 MB, against a whole-repo .git of 26 MB. Keeping every 2nd
# frame (played back at half fps, so the clip runs at real speed) is the dominant
# saving; the palette does the rest. Restore upstream's exact settings with
# `--every 1 --colors 256 --lossy 80`.
DEFAULT_EVERY = 2
DEFAULT_COLORS = 128
DEFAULT_LOSSY = 100

# A DISPLAY only has to *exist*; nothing is drawn to it. See configure_headless_rendering.
FALLBACK_DISPLAY = ":0"

CONTRAST_LINE = "the toss that lands IN the bin does not score; the one that misses it does"

# The goal region is what the clip is *about*, so it is drawn. KINDER already creates a
# box site for every region and already calls `visualize_regions()` unconditionally
# (`envs.py:455` for ground regions); the sites are invisible purely because their alpha
# is 0 -- either the `[1, 0, 0, 0]` default in `objects/base.py:370`/`:900`, or, for this
# scene's goal region, a `[0.2, 0.6, 0.2, 0.0]` written into the task JSON. Setting an
# `"rgba"` in that JSON is therefore the first-class configuration route, and it is
# unavailable here: the task JSON lives under `reference/`, which is a third-party
# checkout this repo reads and never modifies. So the same value is written into the
# compiled model at runtime instead -- purely a colour, after the scene is built.
GOAL_REGION_RGBA = (0.15, 0.55, 1.00, 0.33)

# Matched against the site names read back off the compiled model rather than hardcoded,
# because the name is assembled as `{fixture}_{region}_region_{index}` deep inside
# upstream's XML builder. Five of this scene's six sites are regions, so the marker has
# to discriminate: `_region_` or `ground_` alone would match four of them.
GOAL_REGION_SITE_MARKER = "goal_region"

# The cube starts on the floor, so its own start height is its half-extent. Comparing
# the rest height against that -- rather than against a hardcoded number, or against
# bin geometry we would have to look up -- is what makes "on the floor" a measured
# claim. Both observed cases miss this 2 mm band by an order of magnitude
# (+0.0195 m vs +0.0000 m), so nothing here rides on the exact tolerance.
RESTING_TOLERANCE_M = 0.002


class RestingPlace(Enum):
    """Where the cube ended up, phrased for the caption."""

    IN_THE_BIN = "sitting on top of the bin, not the floor"
    ON_BARE_FLOOR = "sitting on bare floor, not in the bin"


class KinderApi(BaseModel):
    """Handles to everything this script needs from KINDER, imported in one place so
    that the module itself imports on a machine with no MuJoCo (CI never installs
    KINDER) and so that the EGL environment is guaranteed to be set first."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    kinder: ModuleType
    robot_type: Any
    tossing_controllers: Any
    shelf_controllers: Any
    optimize_gif: Any


class GoalRegion(BaseModel):
    """The goal-region site that was made visible, and where it actually is.

    `x_min`/`x_max` are read off the site's own `pos`/`size` rather than restated from
    a doc, so the caption's numbers and the shaded box in the frame cannot disagree.
    """

    name: str
    x_min: float
    x_max: float


class ClipResult(BaseModel):
    """What one rendered standoff produced. Reported as a table at the end."""

    standoff: float
    path: Path
    rest: tuple[float, float, float]
    bin_x: float
    solved: bool
    num_frames: int
    bytes_before: int
    bytes_after: int
    steps: dict[str, int]
    goal_region: GoalRegion | None


def configure_headless_rendering(
    *, environ: MutableMapping[str, str] | None = None
) -> dict[str, str]:
    """Point MuJoCo at EGL, and make sure a DISPLAY exists, before KINDER is imported.

    This is the trap that costs an hour. `kinder.register_all_environments()` rewrites
    `MUJOCO_GL` to `osmesa` when `DISPLAY` is unset; `import mujoco` then raises;
    `_check_deps` **swallows** that exception; and every Dynamic3D environment
    silently vanishes into a `NameNotFound` from `kinder.make`. Nothing reports the
    real cause. Setting `DISPLAY` skips the rewrite entirely, and forcing
    `MUJOCO_GL`/`PYOPENGL_PLATFORM` to `egl` overrides an inherited `osmesa` -- the
    one value known to break the import -- rather than respecting it.
    """
    target = os.environ if environ is None else environ
    target.setdefault("DISPLAY", FALLBACK_DISPLAY)
    target["MUJOCO_GL"] = "egl"
    target["PYOPENGL_PLATFORM"] = "egl"
    return {key: target[key] for key in ("DISPLAY", "MUJOCO_GL", "PYOPENGL_PLATFORM")}


def import_kinder() -> KinderApi:
    """Import KINDER, after configuring the environment, and register its envs.

    Registration happens here too, because `register_all_environments()` is the call
    that rewrites `MUJOCO_GL`, so the environment is re-asserted immediately after it
    rather than left to a caller who might forget.

    Importing the *package* `kinder.envs.dynamic3d` is not enough to register the
    Dynamic3D envs -- a module inside it, such as `kinder.envs.dynamic3d.envs`, has to
    be imported. The distribution is `kindergarden`; the import package is `kinder`.
    """
    resolved = configure_headless_rendering()
    print(f"headless rendering: {resolved}", flush=True)

    import kinder
    import kinder.envs.dynamic3d.envs  # noqa: F401  (the MODULE, not the package)
    from kinder.envs.dynamic3d.object_types import MujocoTidyBotRobotObjectType
    from kinder.gif_utils import optimize_gif
    from kinder_models.dynamic3d.shelf.parameterized_skills import (
        create_lifted_controllers as shelf_create_lifted_controllers,
    )
    from kinder_models.dynamic3d.tossing.parameterized_skills import (
        create_lifted_controllers as tossing_create_lifted_controllers,
    )

    kinder.register_all_environments()
    configure_headless_rendering()

    return KinderApi(
        kinder=kinder,
        robot_type=MujocoTidyBotRobotObjectType,
        tossing_controllers=tossing_create_lifted_controllers,
        shelf_controllers=shelf_create_lifted_controllers,
        optimize_gif=optimize_gif,
    )


def site_names(*, model: Any) -> list[str]:
    """Every site name in a compiled `mujoco.MjModel`, in model order.

    `model` is typed loosely on purpose: taking the real `mujoco.MjModel` as an
    annotation would put a KINDER-only import at module scope, which is exactly what
    lets this module import and test on a machine with no MuJoCo.
    """
    return [model.site(index).name for index in range(model.nsite)]


def find_goal_region_site(*, names: Sequence[str]) -> str:
    """The one site that is the goal region, or an error naming every candidate.

    Failing loudly matters more than usual here: if upstream renames the region, a
    lenient match would render a clip with no overlay -- or with the *wrong* region
    shaded -- and nothing in the output would say so.
    """
    matches = [name for name in names if GOAL_REGION_SITE_MARKER in name]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one site containing {GOAL_REGION_SITE_MARKER!r}, "
            f"found {len(matches)}: {matches} (all sites: {list(names)})"
        )
    return matches[0]


def reveal_goal_region(*, model: Any, rgba: Sequence[float]) -> GoalRegion:
    """Give the goal-region site a visible alpha, in place, on the compiled model.

    `model.site(name).rgba` is a writable *view* into the model, and the offscreen
    render context holds the same `MjModel` object, so the write shows up in the next
    frame with no re-compile. It has to happen **after** `env.reset()`, which rebuilds
    the scene XML and therefore the model.

    Nothing but a colour changes: a site is a massless, collision-free marker, so this
    cannot touch the physics, any skill parameter, or the goal predicate.
    """
    name = find_goal_region_site(names=site_names(model=model))
    site = model.site(name)
    site.rgba[:] = rgba
    center_x, half_x = float(site.pos[0]), float(site.size[0])
    return GoalRegion(name=name, x_min=center_x - half_x, x_max=center_x + half_x)


def gif_filename(*, standoff: float) -> str:
    """`1.35` -> `tossing3d_oracle_standoff_1p35.gif`.

    The decimal point becomes `p` so the stem carries exactly one dot: these files get
    linked by raw URL from a PR body, and a second dot is the kind of thing that
    quietly breaks a link or a downstream tool.
    """
    return f"{FILENAME_PREFIX}_{standoff:.2f}".replace(".", "p") + ".gif"


def fps_from_metadata(*, metadata: Mapping[str, Any], every: int) -> float:
    """Playback rate for a clip that kept every `every`-th simulator frame.

    Read from the environment rather than hardcoded: Tossing3D reports 20, while
    several other KINDER envs report 10. Missing metadata raises instead of falling
    back to upstream's default of 10, which would render this domain's clips at half
    speed with nothing to say so.
    """
    if "render_fps" not in metadata:
        raise ValueError(f"environment metadata has no render_fps: {dict(metadata)}")
    return float(metadata["render_fps"]) / every


def should_keep_frame(*, index: int, every: int) -> bool:
    """Whether the `index`-th simulator step is recorded, counting from the first."""
    return index % every == 0


def caption_lines(
    *,
    standoff: float,
    rest: tuple[float, float, float],
    start_z: float,
    bin_x: float,
    solved: bool,
    goal_region: GoalRegion | None = None,
    tolerance: float = RESTING_TOLERANCE_M,
) -> list[str]:
    """The caption burned into every frame of one clip.

    Every number in it is measured in the run that produced the clip; the only fixed
    text is `CONTRAST_LINE`, which is what makes a single clip self-explanatory.

    The legend line is appended only when the region is actually drawn (`--no-goal-
    region` renders clean), so the caption never describes something absent from the
    frame. Its bracket is what turns the shaded box into a readable claim: the cube
    rests outside it while visibly inside the bin.
    """
    x, y, z = rest
    lift = z - start_z
    place = RestingPlace.ON_BARE_FLOOR if abs(lift) <= tolerance else RestingPlace.IN_THE_BIN
    if place is RestingPlace.ON_BARE_FLOOR:
        height = f"z matches its start height ({start_z:.4f} m): {place.value}"
    else:
        height = f"z is {lift:+.4f} m off its start height: {place.value}"
    lines = [
        f"Tossing3D-o1 skill oracle | toss standoff {standoff:g} | _check_goals() = {solved}",
        f"cube at rest x={x:.4f} y={y:.4f} z={z:.4f}   (bin_0 at x={bin_x:.4f})",
        height,
        CONTRAST_LINE,
    ]
    if goal_region is not None:
        lines.append(
            "shaded blue = the goal region blocks_goal_region, "
            f"x in [{goal_region.x_min:.4f}, {goal_region.x_max:.4f}]"
        )
    return lines


def _load_font(*, size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    """DejaVu ships with matplotlib, which is already a hard dependency of this repo,
    so no system font has to be found. `load_default()` is a last resort that keeps a
    caption legible-ish rather than crashing the render at the very end."""
    import matplotlib

    path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(str(path), size)
    except OSError:
        return ImageFont.load_default()


def _fit_font(
    *, width: int, lines: Sequence[str], padding: int
) -> tuple[ImageFont.ImageFont | ImageFont.FreeTypeFont, int]:
    """Largest font size in [8, 15] whose widest line fits, else the smallest."""
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    font = _load_font(size=8)
    size = 8
    for candidate in range(15, 7, -1):
        font = _load_font(size=candidate)
        size = candidate
        if max(probe.textlength(line, font=font) for line in lines) <= width - 2 * padding:
            break
    return font, size


def annotate(*, frames: Sequence[np.ndarray], lines: Sequence[str]) -> list[np.ndarray]:
    """Stack a caption bar **below** every frame, rather than over it.

    Overlaying would sit on top of the floor tiles the cube has to be seen against.
    The bar is identical in every frame, so it is rendered once.
    """
    if not frames:
        return []
    padding = 6
    width = frames[0].shape[1]
    font, size = _fit_font(width=width, lines=lines, padding=padding)
    line_height = size + 5
    bar_image = Image.new("RGB", (width, line_height * len(lines) + 2 * padding), (16, 16, 16))
    draw = ImageDraw.Draw(bar_image)
    for index, line in enumerate(lines):
        draw.text((padding, padding + index * line_height), line, font=font, fill=(238, 238, 238))
    bar = np.asarray(bar_image, dtype=np.uint8)
    return [np.vstack([np.asarray(frame, dtype=np.uint8), bar]) for frame in frames]


def write_gif(
    *,
    frames: Sequence[np.ndarray],
    path: Path,
    fps: float,
    optimize: Callable[[Path], Any] | None,
) -> tuple[int, int]:
    """Write the clip and optionally shrink it, returning (bytes before, bytes after).

    Both numbers are returned because the artefact is committed, so its size is a
    review concern: the raw clip is several MB and `kinder.gif_utils.optimize_gif`
    (a `gifsicle --lossy` wrapper) typically takes a large fraction off. `optimize`
    is injected rather than imported so this stays testable without KINDER, and
    `None` is a legitimate value -- gifsicle is not always installed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, list(frames), fps=fps, loop=0)
    before = path.stat().st_size
    if optimize is None:
        return before, before
    optimize(path)
    return before, path.stat().st_size


class Rollout(BaseModel):
    """One episode in progress: the env, the live state, and the frames kept so far."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    env: Any
    every: int
    state: Any = None
    frames: list[np.ndarray] = Field(default_factory=list)
    index: int = 0

    def execute(
        self,
        *,
        controller: Any,
        params: np.ndarray,
        limit: int,
        label: str,
        disable_collision_objects: Sequence[str] | None = None,
    ) -> int:
        """Drive one grounded controller to termination, recording frames.

        Every skill in this rollout terminates on its own; a controller that runs out
        of steps means something changed upstream, so that raises rather than being
        quietly truncated into a misleading clip.
        """
        if disable_collision_objects is None:
            controller.reset(self.state, params)
        else:
            controller.reset(
                self.state, params, disable_collision_objects=list(disable_collision_objects)
            )
        for step in range(limit):
            obs, _, _, _, _ = self.env.step(controller.step())
            self.state = self.env.observation_space.devectorize(obs)
            controller.observe(self.state)
            if should_keep_frame(index=self.index, every=self.every):
                self.frames.append(self.env.render())
            self.index += 1
            if controller.terminated():
                print(f"  {label}: terminated after {step + 1} steps", flush=True)
                return step + 1
        raise RuntimeError(f"{label} did not terminate within {limit} steps")


def render_clip(
    *,
    api: KinderApi,
    standoff: float,
    env_id: str,
    seed: int,
    camera: str,
    scene_bg: bool,
    every: int,
    colors: int,
    lossy: int,
    goal_region: bool,
    output_dir: Path,
) -> ClipResult:
    """Run the oracle skill sequence once at `standoff` and write its captioned GIF."""
    print(f"standoff {standoff}", flush=True)
    env = api.kinder.make(env_id, render_mode="rgb_array", scene_bg=scene_bg)
    object_centric = env.unwrapped._object_centric_env  # noqa: SLF001
    available = list(getattr(object_centric, "camera_names", []))
    if available and camera not in available:
        # set_render_camera stores the name without validating it, so an unknown
        # camera renders *something* -- silently, and not the task.
        raise ValueError(f"camera {camera!r} is not in this scene: {available}")
    object_centric.set_render_camera(camera)

    observation, _ = env.reset(seed=seed)
    # After reset, never before: reset rebuilds the scene XML and recompiles the model,
    # so an rgba written earlier would be thrown away along with the model it was on.
    region: GoalRegion | None = None
    if goal_region:
        region = reveal_goal_region(
            model=object_centric._robot_env.sim.model.mj_model,  # noqa: SLF001
            rgba=GOAL_REGION_RGBA,
        )
        print(f"  goal region {region.name}: x in [{region.x_min:.4f}, {region.x_max:.4f}]")
    rollout = Rollout(env=env, every=every, state=env.observation_space.devectorize(observation))
    cube = rollout.state.get_object_from_name("cube_0")
    bin_object = rollout.state.get_object_from_name("bin_0")
    start_z = float(rollout.state.get(cube, "z"))
    bin_x = float(rollout.state.get(bin_object, "x"))
    robot = list(rollout.state.get_objects(api.robot_type))[0]

    shelf = api.shelf_controllers(env.action_space)
    tossing = api.tossing_controllers(env.action_space)
    steps: dict[str, int] = {}

    pick = shelf["pick_shelf"].ground((robot, cube))
    steps["pick_shelf"] = rollout.execute(
        controller=pick,
        params=pick.sample_parameters(rollout.state, np.random.default_rng(PICK_PARAM_SEED)),
        limit=400,
        label="pick_shelf",
    )
    move = tossing["move_to_target"].ground((robot, bin_object))
    steps["move_to_target"] = rollout.execute(
        controller=move,
        params=np.array([standoff, 0.0]),
        limit=200,
        label="move_to_target",
        disable_collision_objects=["cube_0"],
    )
    windup = tossing["move_arm_to_conf"].ground((robot,))
    steps["move_arm_to_conf"] = rollout.execute(
        controller=windup,
        params=np.deg2rad(WINDUP_CONF_DEG),
        limit=200,
        label="move_arm_to_conf",
    )
    throw = tossing["toss"].ground((robot,))
    steps["toss"] = rollout.execute(
        controller=throw,
        params=np.deg2rad(FULL_TOSS_CONF_DEG),
        limit=200,
        label="toss",
    )

    rest = tuple(float(rollout.state.get(cube, key)) for key in ("x", "y", "z"))
    solved = bool(object_centric._check_goals())  # noqa: SLF001
    print(f"  cube at rest: x={rest[0]:.4f} y={rest[1]:.4f} z={rest[2]:.4f}", flush=True)
    print(f"  _check_goals() = {solved}", flush=True)

    lines = caption_lines(
        standoff=standoff,
        rest=rest,
        start_z=start_z,
        bin_x=bin_x,
        solved=solved,
        goal_region=region,
    )
    path = output_dir / gif_filename(standoff=standoff)
    before, after = write_gif(
        frames=annotate(frames=rollout.frames, lines=lines),
        path=path,
        fps=fps_from_metadata(metadata=env.metadata, every=every),
        optimize=functools.partial(api.optimize_gif, colors=colors, lossy=lossy),
    )
    print(f"  wrote {path}: {len(rollout.frames)} frames, {before} -> {after} bytes", flush=True)
    env.close()
    return ClipResult(
        standoff=standoff,
        path=path,
        rest=rest,
        bin_x=bin_x,
        solved=solved,
        num_frames=len(rollout.frames),
        bytes_before=before,
        bytes_after=after,
        steps=steps,
        goal_region=region,
    )


def _positive_int(value: str) -> int:  # noqa: PLR0917 (argparse type: called positionally)
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {value}")
    return parsed


def parse_args(*, argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--standoffs",
        type=float,
        nargs="+",
        default=list(DEFAULT_STANDOFFS),
        help="move_to_target standoffs to render, one clip each (default: 1.35 1.55 -- "
        "the pair whose contrast is the finding)",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--env-id", default=DEFAULT_ENV_ID)
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="fixed, never drawn -- the same convention as scripts/run_sweep.py",
    )
    parser.add_argument(
        "--camera",
        default=DEFAULT_CAMERA,
        help="scene camera to render from; see the module docstring for why this is "
        "task_view and not agentview_1",
    )
    parser.add_argument(
        "--no-scene-bg",
        dest="scene_bg",
        action="store_false",
        help="render the bare 'simple' scene instead of the lab. Fast, needs no asset "
        "download, and looks wrong -- a smoke test only. Physics is unaffected.",
    )
    parser.add_argument(
        "--no-goal-region",
        dest="goal_region",
        action="store_false",
        help="render clean, without the goal region shaded in. On by default, because "
        "the committed clips exist to show that the cube lands in the bin and *outside* "
        "the goal region, which is unreadable when the region is invisible.",
    )
    parser.add_argument(
        "--every",
        type=_positive_int,
        default=DEFAULT_EVERY,
        help="keep every Nth simulator frame; playback fps is divided to match, so the "
        "clip still runs at real speed (default: %(default)s -- see the size note above)",
    )
    parser.add_argument(
        "--colors",
        type=_positive_int,
        default=DEFAULT_COLORS,
        help="gifsicle palette size (upstream's default is 256)",
    )
    parser.add_argument(
        "--lossy",
        type=_positive_int,
        default=DEFAULT_LOSSY,
        help="gifsicle lossy level, higher is smaller (upstream's default is 80)",
    )
    parser.set_defaults(scene_bg=True, goal_region=True)
    return parser.parse_args(list(argv))


def main(*, argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv=sys.argv[1:] if argv is None else argv)
    api = import_kinder()
    results = [
        render_clip(
            api=api,
            standoff=standoff,
            env_id=args.env_id,
            seed=args.seed,
            camera=args.camera,
            scene_bg=args.scene_bg,
            every=args.every,
            colors=args.colors,
            lossy=args.lossy,
            goal_region=args.goal_region,
            output_dir=args.output_dir,
        )
        for standoff in args.standoffs
    ]
    print("\nstandoff  rest x     rest z     _check_goals()  bytes before -> after  file")
    for result in results:
        print(
            f"{result.standoff:<9.2f} {result.rest[0]:<10.4f} {result.rest[2]:<10.4f} "
            f"{str(result.solved):<15} {result.bytes_before} -> {result.bytes_after}  "
            f"{result.path}"
        )


if __name__ == "__main__":
    main()
