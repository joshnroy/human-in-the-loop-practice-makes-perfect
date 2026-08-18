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
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest

from scripts.tossing3d_oracle_demo import (
    CONTAINMENT_TOLERANCE_M,
    CONTRAST_LINE,
    DEFAULT_STANDOFFS,
    GOAL_REGION_RGBA,
    Footprint,
    GoalRegion,
    RestingPlace,
    ScoringGeometry,
    StandoffSeedResult,
    annotate,
    bin_footprint,
    caption_lines,
    configure_headless_rendering,
    find_goal_region_site,
    footprint_from_extents,
    fps_from_metadata,
    gif_filename,
    goal_region_footprint,
    import_kinder,
    parse_args,
    print_and_write_grid,
    reveal_goal_region,
    should_keep_frame,
    site_names,
    verify_containment,
    write_gif,
)

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "tossing3d_oracle_demo.py"

# Measured off the compiled model at the pins this branch carries, not restated from a
# doc. **Both numbers moved, and one of them stopped varying.**
#
# `LIVE_GOAL_X` was `(1.8500, 2.1500)`, which was exactly the bin's own footprint --
# upstream `270fdb6` narrowed `blocks_goal_region` to `[2.00, 2.05]`, inflating to
# `[1.95, 2.10]`. So the scored box is now strictly *inside* the bin: 100 mm of inset at
# the near edge and 50 mm at the far one.
#
# `LIVE_BIN_X` was a per-seed dict, because `bin_init_region` used to be a 1 mm-wide
# sampling range and the bin's footprint shifted by a fraction of a millimetre per seed.
# It is zero-width now (`[[2.0, 0.0, 2.0, 0.0]]`), so the footprint is identical at every
# seed -- measured at seeds 0 and 125 and found bit-identical, which is why one constant
# is now honest where two were needed before.
#
# `PRE_FIX_BIN_X` is the geometry `kindergarden` PR #126 removed -- the bin sitting 23 cm
# off the region that scores. It is kept, under its own name and with its own per-seed
# jitter, because it is the non-vacuity check for `verify_containment`: a verifier that
# accepted it would be asserting nothing.
LIVE_GOAL_X = (1.9500, 2.1000)
PRE_FIX_BIN_X = {0: (2.080813, 2.380813), 125: (2.080101, 2.380101)}
LIVE_BIN_X = (1.8500, 2.1500)


def _frames(*, count: int = 3, width: int = 8, height: int = 6) -> list[np.ndarray]:
    return [
        np.full((height, width, 3), fill_value=index * 10, dtype=np.uint8) for index in range(count)
    ]


class _FakeSite:
    """One row of a compiled MuJoCo model's site tables.

    `rgba`/`pos`/`size` are numpy arrays because that is what `mujoco.MjModel.site()`
    hands back: writable *views* into the model, which is the whole reason an in-place
    write to one of them shows up in the next render.
    """

    def __init__(
        self,
        *,
        name: str,
        pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
        size: tuple[float, float, float] = (0.1, 0.1, 0.1),
        rgba: tuple[float, float, float, float] = (0.2, 0.6, 0.2, 0.0),
    ) -> None:
        self.name = name
        self.pos = np.array(pos, dtype=float)
        self.size = np.array(size, dtype=float)
        self.rgba = np.array(rgba, dtype=float)


class _FakeModel:
    """The two-signature `model.site(...)` accessor: by index, and by name."""

    def __init__(self, *sites: _FakeSite) -> None:
        self._sites = list(sites)
        self.nsite = len(self._sites)

    def site(self, key: int | str) -> _FakeSite:  # noqa: PLR0917 (mirrors mujoco's API)
        if isinstance(key, int):
            return self._sites[key]
        return next(site for site in self._sites if site.name == key)


def _tossing3d_model() -> _FakeModel:
    """The six sites this scene actually compiles to, measured off the real model."""
    return _FakeModel(
        _FakeSite(name="ground_bin_init_region_region_0", pos=(2.2305, 0.0, 0.05)),
        _FakeSite(name="ground_blocks_init_region_region_0", pos=(0.625, 0.0, 0.05)),
        _FakeSite(
            name="ground_blocks_goal_region_region_0",
            pos=(2.0, 0.0, 0.075),
            size=(0.15, 0.15, 0.075),
        ),
        _FakeSite(name="ground_barrier_init_region_region_0", pos=(1.3005, 0.0005, 0.05)),
        _FakeSite(name="ground_robot_init_region_region_0", pos=(0.0, 0.0, 0.05)),
        _FakeSite(name="robot_pinch_site", rgba=(0.5, 0.5, 0.5, 0.3)),
    )


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


def test_configure_undoes_kinders_rewrite_but_keeps_a_requested_backend() -> None:
    """**This inverts a decision that stood until 2026-08-14.** The function used to force
    `egl` unconditionally, on the premise that "osmesa is the value that breaks the
    import". That premise is a property of this workstation, not of osmesa: with
    `libosmesa6-dev` installed it is the backend that *works*, and a headless CI runner
    has no EGL driver at all. So the two cases are now told apart by *when* the value
    arrived -- see `KinderBackend.configure_headless_rendering`, which this name is an
    alias of -- and only the rewrite is undone."""
    from_kinders_rewrite = {"MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa"}
    configure_headless_rendering(environ=from_kinders_rewrite, backend="egl")
    assert from_kinders_rewrite["MUJOCO_GL"] == "egl"
    assert from_kinders_rewrite["PYOPENGL_PLATFORM"] == "egl"

    requested = {"MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": ""}
    configure_headless_rendering(environ=requested, backend="osmesa")
    assert requested["MUJOCO_GL"] == "osmesa"
    assert requested["PYOPENGL_PLATFORM"] == "osmesa"


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


def test_the_default_standoffs_are_the_inherited_pair() -> None:
    """Pinned as-is, deliberately. This pair was chosen on the pre-fix scene, where
    1.35 landed the cube in the bin and scored `False` while 1.55 missed the bin and
    scored `True` -- a contrast the fixed scene no longer produces. Retiring the
    `--task-config` flag did not re-measure them, so the defaults are inherited rather
    than justified and this test says so instead of asserting a reason that is gone."""
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


# --- the standoff x seed grid (--seeds / --results-json) --------------------------
#
# `run_standoff_seed_grid` itself needs KINDER (it calls `render_clip`), so it is not
# tested here -- same reasoning as the rest of this file. `print_and_write_grid` is
# pure aggregation over already-computed results and needs no simulator, so it is.


def test_seeds_defaults_to_none_so_the_single_seed_path_is_unchanged() -> None:
    assert parse_args(argv=[]).seeds is None
    assert parse_args(argv=[]).results_json is None


def test_seeds_and_results_json_are_parsed() -> None:
    args = parse_args(argv=["--seeds", "0", "1", "2", "--results-json", "/tmp/grid.json"])
    assert args.seeds == [0, 1, 2]
    assert args.results_json == Path("/tmp/grid.json")


def test_print_and_write_grid_reports_solved_counts_per_standoff(
    *, capsys: pytest.CaptureFixture[str]
) -> None:
    """`x/y`, never a bare percentage: the fraction alone would hide that 1.125 was
    tested on the same 5 seeds as 1.15, which is the only thing that says a 2/5 and a
    5/5 are comparable."""
    results = [
        StandoffSeedResult(standoff=1.15, seed=seed, solved=solved, rest=(0.0, 0.0, 0.0), steps={})
        for seed, solved in ((0, True), (1, True), (2, False))
    ] + [
        StandoffSeedResult(standoff=1.40, seed=seed, solved=False, rest=(0.0, 0.0, 0.0), steps={})
        for seed in (0, 1, 2)
    ]
    print_and_write_grid(results=results, results_json=None)
    out = capsys.readouterr().out
    assert "1.150" in out
    assert "2/3" in out
    assert "1.400" in out
    assert "0/3" in out
    assert "2/6" in out  # the total line


def test_print_and_write_grid_writes_the_full_grid_as_json(*, tmp_path: Path) -> None:
    results = [
        StandoffSeedResult(
            standoff=1.15, seed=0, solved=True, rest=(2.0, 0.0, 0.05), steps={"toss": 12}
        )
    ]
    out_path = tmp_path / "grid.json"
    print_and_write_grid(results=results, results_json=out_path)
    written = json.loads(out_path.read_text())
    assert written == [
        {
            "standoff": 1.15,
            "seed": 0,
            "solved": True,
            "rest": [2.0, 0.0, 0.05],
            "steps": {"toss": 12},
        }
    ]


# --- the goal-region overlay ------------------------------------------------------
#
# KINDER already creates a box site for every region and already calls
# visualize_regions() unconditionally; the sites are invisible only because the task
# JSON gives them alpha 0. So this is upstream's own mechanism, turned on at runtime
# rather than by editing a file under reference/ (which is never modified).


def test_site_names_are_read_off_the_compiled_model() -> None:
    """Read back, never assumed: the site name is generated as
    `{fixture}_{region}_region_{index}` deep inside upstream's XML builder."""
    assert site_names(model=_tossing3d_model()) == [
        "ground_bin_init_region_region_0",
        "ground_blocks_init_region_region_0",
        "ground_blocks_goal_region_region_0",
        "ground_barrier_init_region_region_0",
        "ground_robot_init_region_region_0",
        "robot_pinch_site",
    ]


def test_the_goal_region_site_is_found_by_what_it_is_not_by_a_hardcoded_name() -> None:
    """Five of this scene's six region sites are *not* the goal, so matching has to
    discriminate -- `_region_0` or `ground_` alone would hit four of them."""
    assert (
        find_goal_region_site(names=site_names(model=_tossing3d_model()))
        == "ground_blocks_goal_region_region_0"
    )


def test_a_scene_with_no_goal_region_site_lists_what_it_did_find() -> None:
    """Renaming this site upstream must fail loudly with the candidates in hand,
    not render a clip with no overlay and nothing to say so."""
    with pytest.raises(ValueError, match="robot_pinch_site"):
        find_goal_region_site(names=["robot_pinch_site"])


def test_an_ambiguous_match_is_an_error_rather_than_a_guess() -> None:
    with pytest.raises(ValueError, match="ground_b_goal_region_region_0"):
        find_goal_region_site(
            names=["ground_a_goal_region_region_0", "ground_b_goal_region_region_0"]
        )


def test_the_overlay_colour_is_actually_visible() -> None:
    """The entire defect being fixed is an alpha of 0.0, so a zero alpha here would
    reintroduce it silently."""
    assert GOAL_REGION_RGBA[3] > 0.0


def test_the_overlay_is_translucent_so_the_cube_stays_visible_through_it() -> None:
    """The cube comes to rest near the region's edge; an opaque box would hide the
    very thing the clip exists to show."""
    assert GOAL_REGION_RGBA[3] < 1.0


def test_revealing_the_region_writes_the_new_alpha_into_the_model() -> None:
    model = _tossing3d_model()
    reveal_goal_region(model=model, rgba=GOAL_REGION_RGBA)
    assert tuple(model.site("ground_blocks_goal_region_region_0").rgba) == GOAL_REGION_RGBA


def test_revealing_the_region_touches_no_other_site() -> None:
    """Presentation only: the bin/blocks/robot init regions stay invisible, and the
    robot's own pinch site keeps the alpha upstream gave it."""
    model = _tossing3d_model()
    reveal_goal_region(model=model, rgba=GOAL_REGION_RGBA)
    others = [
        model.site(index).rgba[3]
        for index in range(model.nsite)
        if model.site(index).name != "ground_blocks_goal_region_region_0"
    ]
    assert others == [0.0, 0.0, 0.0, 0.0, 0.3]


def test_revealing_the_region_reports_its_measured_x_extent() -> None:
    """`x in [1.85, 2.15]` is the number that makes the clip legible: it is where the
    bin now sits, so a cube seen coming to rest inside the bin is a cube seen scoring."""
    region = reveal_goal_region(model=_tossing3d_model(), rgba=GOAL_REGION_RGBA)
    assert (round(region.x_min, 4), round(region.x_max, 4)) == (1.85, 2.15)


def test_revealing_the_region_reports_the_site_it_changed() -> None:
    region = reveal_goal_region(model=_tossing3d_model(), rgba=GOAL_REGION_RGBA)
    assert region.name == "ground_blocks_goal_region_region_0"


# --- the overlay flag -------------------------------------------------------------


def test_the_goal_region_overlay_is_on_by_default() -> None:
    """The clips carry it: where the cube came to rest only means anything once the
    box that scores is visible in the same frame."""
    assert parse_args(argv=[]).goal_region is True


def test_the_overlay_can_be_turned_off_for_a_clean_render() -> None:
    assert parse_args(argv=["--no-goal-region"]).goal_region is False


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


def test_every_clip_carries_the_one_line_stating_the_mechanism() -> None:
    """There is one scene, so there is one such line, and it holds whichever way the
    verdict went: a reader who opens a single clip still learns what scoring means
    here, rather than having to infer it from one rollout."""
    solved = caption_lines(
        standoff=1.55, rest=(2.0268, 0.0105, 0.0249), start_z=0.0249, bin_x=2.25, solved=True
    )
    missed = caption_lines(
        standoff=1.35, rest=(2.2197, 0.0103, 0.0444), start_z=0.0249, bin_x=2.25, solved=False
    )
    assert CONTRAST_LINE in solved
    assert CONTRAST_LINE in missed


def test_the_caption_names_the_shaded_box_when_the_overlay_is_on() -> None:
    """A translucent blue box with nothing saying what it is would be scenery."""
    lines = caption_lines(
        standoff=1.35,
        rest=(2.2197, 0.0103, 0.0444),
        start_z=0.0249,
        bin_x=2.2301,
        solved=False,
        goal_region=GoalRegion(name="ground_blocks_goal_region_region_0", x_min=1.85, x_max=2.15),
    )
    text = " ".join(lines)
    assert "1.8500" in text
    assert "2.1500" in text


def test_the_caption_says_nothing_about_a_region_that_is_not_drawn() -> None:
    """`--no-goal-region` renders clean, so a legend for an absent box would be a
    caption describing something that is not in the frame."""
    with_overlay = caption_lines(
        standoff=1.35,
        rest=(2.2197, 0.0103, 0.0444),
        start_z=0.0249,
        bin_x=2.2301,
        solved=False,
        goal_region=GoalRegion(name="ground_blocks_goal_region_region_0", x_min=1.85, x_max=2.15),
    )
    without = caption_lines(
        standoff=1.35, rest=(2.2197, 0.0103, 0.0444), start_z=0.0249, bin_x=2.2301, solved=False
    )
    assert len(with_overlay) == len(without) + 1
    assert without == with_overlay[: len(without)]


# --- there is one scene, so there is nothing to select ----------------------------
#
# This script used to carry a `--task-config` flag choosing between the KINDER-registered
# `Tossing3D-o1.json` and a copy of it, committed here, that moved `bin_init_region` back
# onto the region that scores. Upstream fixed the defect on the stock file itself
# (`kindergarden` PR #126, carried by the `reference/kindergarden` pin), the committed
# copy was retired, and the flag went with it. What is worth pinning now is that the
# retirement is complete: no scene selection at all, and an old command line that still
# passes the flag fails loudly rather than silently rendering something else.


def test_the_demo_no_longer_selects_a_scene() -> None:
    """It renders whatever the installed KINDER ships, so there is no config to name."""
    assert not hasattr(parse_args(argv=[]), "task_config")


def test_the_retired_flag_is_rejected_rather_than_quietly_ignored() -> None:
    """A command line copied out of an old experiment log must fail at parse time. The
    alternative -- accepting and ignoring it -- would render the shipped scene under a
    caption and a filename claiming a config that no longer exists."""
    with pytest.raises(SystemExit):
        parse_args(argv=["--task-config", "coincident-bin-goal"])


def test_the_filename_is_a_function_of_the_standoff_and_nothing_else() -> None:
    """The config token is gone from the name, which is what keeps the clips already
    committed under these names regenerable by this script."""
    assert list(inspect.signature(gif_filename).parameters) == ["standoff"]


# --- live geometry: is the bin actually where the goal region is? -----------------
#
# Everything below is asserted against numbers read out of the *compiled* model, never
# out of the JSON. The JSON range is inflated by `ground_placement_threshold = 0.05`
# per side before it becomes a region, and the bin's placement is sampled, so the JSON
# literals and the geometry that actually scores are two different things.


def test_a_footprint_spans_its_extremes() -> None:
    footprint = footprint_from_extents(centers_x=[2.0, 1.9, 2.1], half_widths_x=[0.15, 0.01, 0.01])
    assert (round(footprint.x_min, 4), round(footprint.x_max, 4)) == (1.85, 2.15)


def test_a_footprint_knows_its_own_width() -> None:
    assert footprint_from_extents(centers_x=[2.0], half_widths_x=[0.15]).width == pytest.approx(0.3)


def test_a_footprint_needs_at_least_one_geom() -> None:
    with pytest.raises(ValueError, match="no geoms"):
        footprint_from_extents(centers_x=[], half_widths_x=[])


class _FakeGeomModel:
    """The slice of a compiled `MjModel`/`MjData` pair that the bin footprint reads.

    The bin's five geoms are **unnamed** in the compiled model -- upstream's `Bin`
    builder never sets a `name` on them -- so they can only be found through their
    body. That is the part worth faking: a name-based match would silently find
    nothing and report an empty footprint.
    """

    def __init__(self, *, bodies: list[str], centers: list[float], halves: list[float]) -> None:
        self._bodies = bodies
        self.ngeom = len(bodies)
        self.geom_bodyid = list(range(len(bodies)))
        self.geom_size = [(half, half, half) for half in halves]
        self.geom_xpos = [(center, 0.0, 0.0) for center in centers]

    def body(self, index: int) -> _FakeSite:  # noqa: PLR0917 (mirrors mujoco's API)
        """`_FakeSite` stands in for a body row too -- both are looked up by index and
        read for nothing but `.name`."""
        return _FakeSite(name=self._bodies[index])


def _bin_model() -> _FakeGeomModel:
    """The real bin: a 0.3 m base plus four walls, centred on 2.0.

    Not centred on 2.000813 any more. `bin_init_region` was a 1 mm-wide sampling range,
    so the bin used to land a fraction of a millimetre off 2.0 differently per seed; it
    is zero-width at this pin, and the footprint measures identical at seeds 0 and 125.
    """
    return _FakeGeomModel(
        bodies=["floor", "bin_0", "bin_0", "bin_0", "bin_0", "bin_0", "cube_0"],
        centers=[0.0, 2.0, 2.0, 2.0, 1.86, 2.14, 0.625],
        halves=[5.0, 0.15, 0.15, 0.15, 0.01, 0.01, 0.025],
    )


def test_the_bin_footprint_is_read_off_the_geoms_of_the_bin_body() -> None:
    """The floor geom is 10 m wide and the cube is right there in the model; matching
    on the body is what keeps them out of the answer."""
    model = _bin_model()
    footprint = bin_footprint(model=model, data=model, body_name="bin_0")
    assert (round(footprint.x_min, 6), round(footprint.x_max, 6)) == LIVE_BIN_X


def test_a_missing_bin_body_raises_rather_than_reporting_an_empty_footprint() -> None:
    """An empty match would sail through the containment check as a zero-width box."""
    model = _bin_model()
    with pytest.raises(ValueError, match="bin_1"):
        bin_footprint(model=model, data=model, body_name="bin_1")


class _FakeRegion:
    """A `Region`, whose `bbox` is only sim-backed once `env` is set."""

    def __init__(self, *, bbox: list[float]) -> None:
        self._bbox = bbox
        self.env = None

    @property
    def bbox(self) -> list[float]:
        if self.env is None:
            raise AssertionError("bbox read without the sim env attached")
        return self._bbox


class _FakeGroundFixture:
    def __init__(self, **regions: list[_FakeRegion]) -> None:
        self.region_objects = regions


def test_the_goal_region_footprint_comes_from_the_live_sim_backed_bbox() -> None:
    """`Region.bbox` silently falls back to the XML/parent-frame value when `env` is
    unset, so the env has to be attached before it is read."""
    fixture = _FakeGroundFixture(
        blocks_goal_region=[_FakeRegion(bbox=[1.95, -0.12, 0.0, 2.10, 0.12, 0.15])]
    )
    footprint = goal_region_footprint(ground_fixture=fixture, robot_env=object())
    assert (footprint.x_min, footprint.x_max) == LIVE_GOAL_X


def test_reading_the_goal_region_leaves_the_region_as_it_found_it() -> None:
    """It is a measurement, so it must not leave a sim reference behind on a region
    that upstream deliberately constructs with `env=None`."""
    region = _FakeRegion(bbox=[1.85, -0.15, 0.0, 2.15, 0.15, 0.15])
    fixture = _FakeGroundFixture(blocks_goal_region=[region])
    goal_region_footprint(ground_fixture=fixture, robot_env=object())
    assert region.env is None


def test_an_absent_goal_region_raises_with_what_was_there_instead() -> None:
    fixture = _FakeGroundFixture(bin_init_region=[_FakeRegion(bbox=[0.0] * 6)])
    with pytest.raises(ValueError, match="bin_init_region"):
        goal_region_footprint(ground_fixture=fixture, robot_env=object())


# --- the containment check --------------------------------------------------------


def _geometry(*, goal: tuple[float, float], bin_x: tuple[float, float]) -> ScoringGeometry:
    return ScoringGeometry(
        goal=Footprint(x_min=goal[0], x_max=goal[1]),
        bin_footprint=Footprint(x_min=bin_x[0], x_max=bin_x[1]),
    )


def test_the_pre_fix_geometry_overlapped_only_partly_and_that_was_the_defect() -> None:
    """1.9 cm of a 30 cm box, at seed 0, against the narrowed scored box. The two were
    not disjoint -- they just had no sliver a cube could be in that was both in the bin
    and in the goal."""
    assert _geometry(goal=LIVE_GOAL_X, bin_x=PRE_FIX_BIN_X[0]).overlap == pytest.approx(
        0.019187, abs=1e-6
    )


def test_the_pre_fix_geometry_is_rejected_by_the_containment_check() -> None:
    """The non-vacuity check for everything below it: a verifier that passed on the
    geometry upstream's fix removed would be asserting nothing at all."""
    for seed, bin_x in PRE_FIX_BIN_X.items():
        with pytest.raises(ValueError, match="not inside the bin"):
            verify_containment(geometry=_geometry(goal=LIVE_GOAL_X, bin_x=bin_x))
        assert seed in PRE_FIX_BIN_X


def test_the_shipped_geometry_passes() -> None:
    verify_containment(geometry=_geometry(goal=LIVE_GOAL_X, bin_x=LIVE_BIN_X))


def test_the_scored_box_is_inset_from_the_bin_rather_than_merely_not_spilling() -> None:
    """The tolerance is not doing the work, and the direction is the interesting part.
    The scored box is inset by 100 mm at the near edge and 50 mm at the far one, so
    `overhang` is 50 mm *negative* against a 1e-9 bound -- eight orders of margin. The
    pre-fix bin spills by more than 100 mm, on the other side of zero."""
    live = _geometry(goal=LIVE_GOAL_X, bin_x=LIVE_BIN_X)
    assert live.overhang == pytest.approx(-0.05, abs=1e-6)
    assert live.overhang < -CONTAINMENT_TOLERANCE_M
    assert min(_geometry(goal=LIVE_GOAL_X, bin_x=b).overhang for b in PRE_FIX_BIN_X.values()) > 0.1


def test_a_coincident_scored_box_is_still_accepted() -> None:
    """Containment, not containment-with-margin. Before the region was narrowed the two
    boxes were exactly equal, which is a correct scene with zero clearance on every edge;
    a check demanding positive clearance would call it a failure. This is why the
    tolerance absorbs float noise only."""
    verify_containment(geometry=_geometry(goal=(1.85, 2.15), bin_x=(1.85, 2.15)))


def test_the_containment_error_names_both_boxes_it_compared() -> None:
    """The message has to be enough to diagnose from, since this fires mid-run."""
    with pytest.raises(ValueError, match=r"2\.3808"):
        verify_containment(geometry=_geometry(goal=LIVE_GOAL_X, bin_x=PRE_FIX_BIN_X[0]))


def test_a_bin_shifted_off_the_scored_box_at_all_is_rejected() -> None:
    """A near miss, not just the stock 23 cm one -- the check is a real bound, and one
    with no slack, so a single millimetre of spill is caught."""
    with pytest.raises(ValueError, match="not inside the bin"):
        verify_containment(geometry=_geometry(goal=(1.95, 2.10), bin_x=(1.951, 2.10)))


# --- the caption's fixed line -----------------------------------------------------


def test_the_caption_reports_the_bin_footprint_beside_the_goal_box() -> None:
    """Putting both bracketed ranges on one line is what lets a viewer read the
    containment off the frame rather than take it on trust -- and, now that the two boxes
    differ, see that they differ."""
    lines = caption_lines(
        standoff=1.35,
        rest=(2.0, 0.0, 0.0444),
        start_z=0.0249,
        bin_x=2.0008,
        solved=True,
        goal_region=GoalRegion(name="ground_blocks_goal_region_region_0", x_min=1.85, x_max=2.15),
        bin_x_range=Footprint(x_min=1.850813, x_max=2.150813),
    )
    text = " ".join(lines)
    assert "1.8500" in text
    assert "2.1500" in text
    assert "1.8508" in text


# --- sweeping the standoff --------------------------------------------------------


def test_sweeping_is_off_by_default_so_the_normal_run_still_writes_clips() -> None:
    assert parse_args(argv=[]).sweep is False


def test_sweeping_can_be_asked_for() -> None:
    """Finding the standoff that solves the fixed scene means running the physics many
    times, and rendering every one of them would dominate the cost for no benefit."""
    assert parse_args(argv=["--sweep"]).sweep is True


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
