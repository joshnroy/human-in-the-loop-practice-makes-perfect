"""Everything in `scripts/tossing3d_oracle_demo.py` that can be pinned without a
simulator: argument parsing, the standoff -> filename mapping, the headless-rendering
environment, frame subsampling, fps derivation, the caption text, and GIF writing.

**No test here imports KINDER, and that is the point rather than a gap.** CI never
installs it (it is an optional extra pulled from two git SHAs, and it wants a ~1 GB
asset download plus a GPU-backed EGL context), so a KINDER-gated test would skip on
CI *and* skip locally -- the `hitl-pmp` conda env has no `kinder` either; the renders
run in a separate venv. A test that never executes anywhere pins nothing. So the
module is instead written so that every KINDER import lives inside one function, and
`test_every_kinder_import_lives_inside_import_kinder` pins that structurally: the
module imports, typechecks and tests on a machine with no MuJoCo, which is the
property CI actually needs. The simulator-side numbers are recorded in the PR body
and reproduced by running the script, not by pytest.
"""

import ast
import inspect
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest

from scripts.tossing3d_oracle_demo import (
    CONTRAST_LINE,
    DEFAULT_STANDOFFS,
    RestingPlace,
    annotate,
    caption_lines,
    configure_headless_rendering,
    fps_from_metadata,
    gif_filename,
    import_kinder,
    parse_args,
    should_keep_frame,
    write_gif,
)

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "tossing3d_oracle_demo.py"


def _frames(*, count: int = 3, width: int = 8, height: int = 6) -> list[np.ndarray]:
    return [
        np.full((height, width, 3), fill_value=index * 10, dtype=np.uint8) for index in range(count)
    ]


# --- the headless-rendering environment ------------------------------------------
#
# This is the trap that costs an hour: register_all_environments() rewrites MUJOCO_GL
# to osmesa when DISPLAY is unset, mujoco then fails to import, _check_deps swallows
# the error, and every Dynamic3D env silently vanishes into NameNotFound. Nothing
# raises -- the env id just stops existing.


def test_configure_sets_a_display_when_there_is_none() -> None:
    environ: dict[str, str] = {}
    configure_headless_rendering(environ=environ)
    assert environ["DISPLAY"]


def test_configure_keeps_a_display_that_is_already_set() -> None:
    """A real X display must not be stomped on -- the point is only that *some*
    DISPLAY exists, so upstream skips its osmesa rewrite."""
    environ = {"DISPLAY": ":7"}
    configure_headless_rendering(environ=environ)
    assert environ["DISPLAY"] == ":7"


def test_configure_forces_egl_over_a_preset_osmesa() -> None:
    """osmesa is the value that breaks the import, so an inherited one is overridden
    rather than respected."""
    environ = {"MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa"}
    configure_headless_rendering(environ=environ)
    assert environ["MUJOCO_GL"] == "egl"
    assert environ["PYOPENGL_PLATFORM"] == "egl"


def test_configure_reports_what_it_resolved() -> None:
    resolved = configure_headless_rendering(environ={})
    assert set(resolved) == {"DISPLAY", "MUJOCO_GL", "PYOPENGL_PLATFORM"}


def test_every_kinder_import_lives_inside_import_kinder() -> None:
    """Why the module imports on a machine with no MuJoCo: nothing at module scope
    reaches for KINDER. A stray top-level `import kinder` would break CI collection
    and would also import KINDER *before* the EGL environment is set."""
    tree = ast.parse(SCRIPT_PATH.read_text())
    inside = {
        node
        for function in ast.walk(tree)
        if isinstance(function, ast.FunctionDef) and function.name == "import_kinder"
        for node in ast.walk(function)
    }
    offenders = [
        ast.unparse(node)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and node not in inside
        and any(
            (alias.name if isinstance(node, ast.Import) else (node.module or "")).startswith((
                "kinder",
                "kinder_models",
            ))
            for alias in node.names
        )
    ]
    assert offenders == []


def test_import_kinder_configures_the_environment_before_importing() -> None:
    """Ordering, not mere presence: setting MUJOCO_GL after `import kinder` is a
    no-op, because upstream reads it at import time."""
    source = inspect.getsource(import_kinder)
    body = ast.parse(source.lstrip()).body[0]
    assert isinstance(body, ast.FunctionDef)
    statements = [node for node in body.body if not isinstance(node, ast.Expr | ast.Pass)]
    first_call = next(node for node in ast.walk(statements[0]) if isinstance(node, ast.Call))
    assert isinstance(first_call.func, ast.Name)
    assert first_call.func.id == "configure_headless_rendering"


# --- standoff -> filename ---------------------------------------------------------


def test_gif_filename_is_derived_from_the_standoff() -> None:
    assert gif_filename(standoff=1.35) == "tossing3d_oracle_standoff_1p35.gif"
    assert gif_filename(standoff=1.55) == "tossing3d_oracle_standoff_1p55.gif"


def test_gif_filename_carries_no_dot_but_the_extension() -> None:
    """A committed artefact gets linked by raw URL from a PR body; a second dot in
    the stem is the kind of thing that quietly breaks a link or a downstream tool."""
    assert gif_filename(standoff=1.5).count(".") == 1


def test_gif_filename_pads_to_two_decimals() -> None:
    assert gif_filename(standoff=1.5) == "tossing3d_oracle_standoff_1p50.gif"


def test_distinct_standoffs_get_distinct_filenames() -> None:
    names = {gif_filename(standoff=standoff) for standoff in DEFAULT_STANDOFFS}
    assert len(names) == len(DEFAULT_STANDOFFS)


# --- fps and subsampling ----------------------------------------------------------


def test_fps_comes_from_the_environment_metadata() -> None:
    """Not a hardcoded constant: upstream's own demo reads render_fps, and Tossing3D
    reports 20 while several other KINDER envs report 10."""
    assert fps_from_metadata(metadata={"render_fps": 20}, every=1) == 20.0


def test_fps_is_divided_by_the_subsampling_stride() -> None:
    """Keeping every 2nd frame at the full rate would play the clip at 2x speed."""
    assert fps_from_metadata(metadata={"render_fps": 20}, every=2) == 10.0


def test_a_missing_render_fps_is_an_error_rather_than_a_guess() -> None:
    """Upstream defaults to 10 here. A silent default would render this domain's
    clips at half speed and nothing would say so, so this raises instead."""
    with pytest.raises(ValueError, match="render_fps"):
        fps_from_metadata(metadata={}, every=1)


def test_should_keep_frame_keeps_everything_at_stride_one() -> None:
    assert [index for index in range(5) if should_keep_frame(index=index, every=1)] == [
        0,
        1,
        2,
        3,
        4,
    ]


def test_should_keep_frame_keeps_every_nth_starting_at_the_first() -> None:
    assert [index for index in range(10) if should_keep_frame(index=index, every=3)] == [0, 3, 6, 9]


# --- argument parsing -------------------------------------------------------------


def test_the_default_run_is_the_contrasting_pair() -> None:
    """The two clips exist because the contrast is the finding; rendering only one
    of them by default would lose it."""
    args = parse_args(argv=[])
    assert args.standoffs == [1.35, 1.55]


def test_the_scene_background_is_on_by_default() -> None:
    """`scene_bg=False` maps to a scene literally named `simple` -- a bare ground
    plane. It is what upstream's *unit test* uses, for speed, and it looks wrong."""
    assert parse_args(argv=[]).scene_bg is True


def test_the_default_camera_is_the_one_the_task_config_defines() -> None:
    """Deliberately not `agentview_1`. Upstream's demo script sets that only
    `if "TidyBot" in env_id`, and `kinder/Tossing3D-o1-v0` does not contain
    `TidyBot`; the name is not in this scene's `camera_names` at all, and
    `set_render_camera` does not validate it, so it silently renders a wall."""
    assert parse_args(argv=[]).camera == "task_view"


def test_standoffs_and_output_dir_are_overridable() -> None:
    args = parse_args(argv=["--standoffs", "1.4", "--output-dir", "/tmp/x"])
    assert args.standoffs == [1.4]
    assert args.output_dir == Path("/tmp/x")


def test_a_non_positive_stride_is_rejected() -> None:
    with pytest.raises(SystemExit):
        parse_args(argv=["--every", "0"])


def test_the_defaults_trade_size_for_a_committed_artefact() -> None:
    """These clips live in git, so their size is a review concern rather than a
    preference: at every=1 with upstream's 256/80 palette the optimised clip is
    3.9 MB against a 26 MB whole-repo .git. Every 2nd frame at half the fps runs at
    the same real-time speed."""
    args = parse_args(argv=[])
    assert args.every == 2
    assert (args.colors, args.lossy) == (128, 100)


def test_upstreams_own_optimisation_settings_are_still_reachable() -> None:
    args = parse_args(argv=["--every", "1", "--colors", "256", "--lossy", "80"])
    assert (args.every, args.colors, args.lossy) == (1, 256, 80)


def test_the_seed_is_fixed_rather_than_drawn() -> None:
    """Same convention as scripts/run_sweep.py: a demo has to regenerate the same
    clip months later."""
    assert parse_args(argv=[]).seed == 125


# --- the caption ------------------------------------------------------------------


def test_the_caption_reports_the_standoff_and_the_rest_position() -> None:
    text = " ".join(
        caption_lines(
            standoff=1.35,
            rest=(2.2197, 0.0103, 0.0444),
            start_z=0.0249,
            bin_x=2.25,
            solved=False,
        )
    )
    assert "1.35" in text
    assert "2.2197" in text
    assert "0.0444" in text


def test_a_cube_above_its_start_height_is_reported_as_resting_on_the_bin() -> None:
    """The cube starts on the floor, so its own start height *is* its half-extent --
    a measured reference rather than a hardcoded tolerance about bin geometry."""
    lines = caption_lines(
        standoff=1.35, rest=(2.2197, 0.0103, 0.0444), start_z=0.0249, bin_x=2.25, solved=False
    )
    assert RestingPlace.IN_THE_BIN.value in " ".join(lines)


def test_a_cube_level_with_its_start_height_is_reported_as_resting_on_bare_floor() -> None:
    lines = caption_lines(
        standoff=1.55, rest=(2.0268, 0.0105, 0.0249), start_z=0.0249, bin_x=2.25, solved=True
    )
    assert RestingPlace.ON_BARE_FLOOR.value in " ".join(lines)


def test_the_caption_reports_check_goals_verbatim() -> None:
    solved = " ".join(
        caption_lines(
            standoff=1.55, rest=(2.0, 0.0, 0.0249), start_z=0.0249, bin_x=2.25, solved=True
        )
    )
    missed = " ".join(
        caption_lines(
            standoff=1.35, rest=(2.2, 0.0, 0.0444), start_z=0.0249, bin_x=2.25, solved=False
        )
    )
    assert "_check_goals() = True" in solved
    assert "_check_goals() = False" in missed


def test_both_clips_carry_the_same_line_stating_the_contrast() -> None:
    """A reader who opens only one of the two clips still has to be able to tell
    that the throw which scores is the one that misses the bin."""
    solved = caption_lines(
        standoff=1.55, rest=(2.0268, 0.0105, 0.0249), start_z=0.0249, bin_x=2.25, solved=True
    )
    missed = caption_lines(
        standoff=1.35, rest=(2.2197, 0.0103, 0.0444), start_z=0.0249, bin_x=2.25, solved=False
    )
    assert CONTRAST_LINE in solved
    assert CONTRAST_LINE in missed


# --- annotation and GIF writing ---------------------------------------------------


def test_annotate_keeps_every_frame_and_its_width() -> None:
    frames = _frames(count=4, width=64, height=48)
    annotated = annotate(frames=frames, lines=["hello", "world"])
    assert len(annotated) == 4
    assert all(frame.shape[1] == 64 for frame in annotated)
    assert all(frame.dtype == np.uint8 for frame in annotated)


def test_annotate_adds_a_bar_below_the_frame_rather_than_covering_it() -> None:
    frames = _frames(count=1, width=64, height=48)
    annotated = annotate(frames=frames, lines=["hello"])
    assert annotated[0].shape[0] > 48
    assert np.array_equal(annotated[0][:48], frames[0])


def test_annotate_actually_draws_the_text() -> None:
    """A caption bar that came out uniformly black would look like a deliberate
    letterbox and nobody would notice the text was missing."""
    frames = _frames(count=1, width=200, height=48)
    annotated = annotate(frames=frames, lines=["a caption"])
    bar = annotated[0][48:]
    assert len(np.unique(bar)) > 1


def test_write_gif_reports_the_size_before_and_after_optimisation(*, tmp_path: Path) -> None:
    """Both numbers are reported because the committed artefact's size is a review
    concern -- the raw agentview_1 clip is several MB."""
    path = tmp_path / "clip.gif"

    def shrink(target: Path) -> Path:  # noqa: PLR0917 (injected callback: positional)
        target.write_bytes(b"GIF89a" + b"\x00" * 8)
        return target

    before, after = write_gif(frames=_frames(count=4), path=path, fps=20.0, optimize=shrink)
    assert before > after
    assert after == path.stat().st_size


def test_write_gif_without_an_optimizer_reports_the_same_size_twice(*, tmp_path: Path) -> None:
    """gifsicle is not always installed; the run must still produce a clip and say
    plainly that nothing was optimised."""
    path = tmp_path / "clip.gif"
    before, after = write_gif(frames=_frames(count=4), path=path, fps=20.0, optimize=None)
    assert before == after == path.stat().st_size


def test_write_gif_writes_a_readable_gif(*, tmp_path: Path) -> None:
    path = tmp_path / "clip.gif"
    write_gif(frames=_frames(count=4), path=path, fps=20.0, optimize=None)
    assert len(imageio.mimread(path)) == 4
